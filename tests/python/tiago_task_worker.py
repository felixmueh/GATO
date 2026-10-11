"""Fresh-process TIAGo task runner; recording wrappers never change control inputs."""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
CASES = Path(__file__).with_name("tiago_task_cases")
MODULE = "bsqp.bsqpN32_tiago_right"


def write_json(path, value):
    # Failed numerical outputs must still produce valid, reviewable JSON.
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if hasattr(item, "tolist"):
            return clean(item.tolist())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_execution(case, model, x, u, t, plans, stats, execution_error=None,
                       reference=None):
    """Independently check recorded executed states, not native convergence flags."""
    import numpy as np
    import pinocchio as pin

    x, u, t, plans = map(np.asarray, (x, u, t, plans))
    packed_size = case["N"] * (14 + 7) - 7
    shapes_valid = bool(
        x.ndim == 2 and x.shape[1] == 14 and len(x) >= 2
        and u.shape == (len(x) - 1, 7) and t.shape == (len(x),)
        and plans.ndim == 3 and plans.shape[1:] == (1, packed_size) and len(plans)
    )
    if not shapes_valid:
        return {"case": case["name"], "passed": False,
                "gates": {"recording_shapes": False},
                "metrics": {"shapes": {"x": list(x.shape), "u": list(u.shape),
                                       "t": list(t.shape), "plans": list(plans.shape)}},
                "execution_error": execution_error}

    data = model.createData()
    frame = model.getFrameId("arm_right_tool_link")

    def pose(q):
        pin.framesForwardKinematics(model, data, q.astype(np.float64))
        placement = data.oMf[frame]
        return placement.translation.copy(), placement.rotation.copy()

    start_position, _ = pose(np.asarray(case["initial_state"][:7]))
    finite = bool(np.isfinite(x).all() and np.isfinite(u).all())
    positions = np.asarray([pose(q)[0] for q in x[:, :7]]) if finite else np.full((len(x), 3), np.nan)
    limits = {
        "position_violation_rad": float(max(0, np.max(model.lowerPositionLimit - x[:, :7]), np.max(x[:, :7] - model.upperPositionLimit))) if finite else float("inf"),
        "velocity_violation_rad_s": float(max(0, np.max(np.abs(x[:, 7:]) - model.velocityLimit))) if finite else float("inf"),
        "torque_violation_Nm": float(max(0, np.max(np.abs(u) - model.effortLimit))) if finite and len(u) else float("inf"),
    }
    metrics = {
        "final_L1_speed_rad_s": float(np.linalg.norm(x[-1, 7:], ord=1)),
        "executed_displacement_m": float(np.max(np.linalg.norm(positions - start_position, axis=1))),
        "executed_duration_s": float(t[-1]), "plant_steps": len(u), "solves": len(plans),
        "physical_limits": limits,
        "public_goal_outcomes": stats.get("goal_outcomes"),
        "public_goal_reached_times": stats.get("goal_reached_times"),
    }
    gates = {
        "completed_without_exception": execution_error is None,
        "recording_shapes": shapes_valid,
        "initial_state_matches": bool(np.allclose(x[0], case["initial_state"], atol=1e-6, rtol=0)),
        "finite_execution": finite,
        "finite_plans": bool(len(plans) and np.isfinite(np.asarray(plans)).all()),
        "executed_control": bool(len(u) and len(plans)),
        "nontrivial_motion": metrics["executed_displacement_m"] >= case["minimum_executed_displacement_m"],
        "position_limits": limits["position_violation_rad"] <= 1e-4,
        "velocity_limits": limits["velocity_violation_rad_s"] <= 1e-3,
        "effort_limits": limits["torque_violation_Nm"] <= 1e-3,
    }
    increments = np.diff(t)
    gates["recording_clock"] = bool(np.isfinite(t).all() and t[0] == 0
                                     and np.all(increments > 0)
                                     and np.all(increments <= case["sim_dt_s"] + 1e-9))
    if case["kind"] == "reaching":
        goal = np.asarray(case["goal"], dtype=np.float32)
        target_rotation = pin.rpy.rpyToMatrix(*map(float, goal[3:]))
        final_position, final_rotation = pose(x[-1, :7]) if finite else (np.full(3, np.nan), np.full((3, 3), np.nan))
        final_axis = float(np.rad2deg(np.arccos(np.clip(final_rotation[:, 2] @ target_rotation[:, 2], -1, 1))))
        metrics.update(
            initial_distance_m=float(np.linalg.norm(start_position - goal[:3])),
            final_distance_m=float(np.linalg.norm(final_position - goal[:3])),
            final_axis_error_deg=final_axis,
            final_full_SO3_error_deg=float(np.rad2deg(np.linalg.norm(pin.log3(target_rotation.T @ final_rotation)))) if finite else float("nan"),
        )
        reached_times = stats.get("goal_reached_times", [])
        gates.update(
            initial_motion_required=metrics["initial_distance_m"] >= case["minimum_initial_distance_m"],
            bounded_positive_duration=0 < t[-1] <= case["timeout_s"] + case["dt_s"],
            public_reaching_success=stats.get("goal_outcomes") == ["reached"],
            position_reached=metrics["final_distance_m"] < case["position_threshold_m"],
            axis_reached=final_axis < case["axis_threshold_deg"],
            speed_reached=metrics["final_L1_speed_rad_s"] < case["speed_L1_threshold_rad_s"],
            completion_clock_matches_execution=bool(len(reached_times) == 1 and reached_times[0] is not None
                                                     and abs(reached_times[0] - t[-1]) < 1e-8),
        )
    else:
        # Match every plant observation to its own time. The controller's
        # goal_distances measures the next knot and cannot certify this gate.
        samples = np.asarray(reference).reshape(-1, 6)
        gates["reference_covers_preview"] = bool(
            np.isfinite(samples).all()
            and (len(samples) - 1) * case["dt_s"] >= t[-1] + (case["N"] - 1) * case["dt_s"]
        )
        coordinates = np.clip(t / case["dt_s"], 0, len(samples) - 1)
        lower = np.floor(coordinates).astype(int)
        upper = np.minimum(lower + 1, len(samples) - 1)
        alpha = (coordinates - lower)[:, None]
        desired = (1 - alpha) * samples[lower, :3] + alpha * samples[upper, :3]
        errors = np.linalg.norm(positions - desired, axis=1)
        rms = float(np.sqrt(np.trapz(errors ** 2, t) / t[-1])) if t[-1] > 0 else float("inf")
        metrics.update(tracking_RMS_error_m=rms, tracking_max_error_m=float(np.max(errors)),
                       reference_displacement_m=float(np.max(np.linalg.norm(desired - desired[0], axis=1))))
        gates.update(
            completed_tracking_duration=case["duration_s"] - 1e-8 <= t[-1] <= case["duration_s"] + case["dt_s"] + 1e-8,
            tracking_RMS_error=rms <= case["maximum_RMS_error_m"],
            tracking_max_error=metrics["tracking_max_error_m"] <= case["maximum_error_m"],
            moving_reference=metrics["reference_displacement_m"] >= case["minimum_executed_displacement_m"],
        )
    summary = {"case": case["name"], "passed": all(gates.values()),
               "gates": gates, "metrics": metrics, "execution_error": execution_error}
    return summary


def run_case(case, output):
    # Import dependencies only in the requested worker. Missing CUDA, extension,
    # or Pinocchio is a failure, never an opt-in skip.
    import numpy as np
    import pinocchio as pin
    import torch

    sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tiago_src")]
    native = importlib.import_module(MODULE)
    from gato_tiago import tiago_mpc_controller as production

    urdf = ROOT / "gato/dynamics/tiago_right/tiago_right_arm.urdf"
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    frame = model.getFrameId("arm_right_tool_link")
    initial = np.asarray(case["initial_state"], dtype=np.float32)

    def pose(q):
        pin.framesForwardKinematics(model, data, q.astype(np.float64))
        placement = data.oMf[frame]
        return placement.translation.copy(), placement.rotation.copy()

    start_position, _ = pose(initial[:7])
    reference = None
    if case["kind"] == "reaching":
        goal = np.asarray(case["goal"], dtype=np.float32)
        initial_distance = float(np.linalg.norm(start_position - goal[:3]))
        if initial_distance < case["minimum_initial_distance_m"]:
            raise RuntimeError(f"Task starts too close to goal: {initial_distance} m")
    else:
        from gato_tiago.references import tiago_horizontal_figure8
        reference, metadata = tiago_horizontal_figure8(case["dt_s"], case["reference_anchor_m"],
                                                      cycles=case["reference_cycles"])
        if hashlib.sha256(reference.tobytes()).hexdigest() != case["reference_sha256"]:
            raise RuntimeError("Public tracking reference differs from frozen protocol")
        np.savez_compressed(output / "reference.npz", reference=reference)

    controller = production.MPC_GATO(
        model, str(urdf), N=case["N"], dt=case["dt_s"], batch_size=1,
        plant_type="tiago_right", solver_params=case["solver_params"],
        track_full_stats=True, final_approach=False,
    )
    if controller.solver.lib is not native:
        raise RuntimeError("Production wrapper loaded an unexpected native module")

    tracked_sources = ["tiago_src/gato_tiago/tiago_mpc_controller.py",
                       "tiago_src/gato_tiago/config.py", "python/bsqp/interface.py",
                       "python/bsqp/common.py", "tiago_src/gato_tiago/references.py",
                       str(urdf.relative_to(ROOT))]
    receipt = {
        "case": case, "case_sha256": sha(CASES / f'{case["name"]}.json'),
        "worker_sha256": sha(__file__), "source_root": str(ROOT),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                 cwd=ROOT, text=True).strip(),
        "source_sha256": {name: sha(ROOT / name) for name in tracked_sources},
        "module_path": native.__file__, "module_sha256": sha(native.__file__),
        "python": sys.version, "numpy": np.__version__, "pinocchio": pin.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "controller": str(Path(production.__file__).resolve()),
        "timing": "fixed simulated controller clock; wall solve time recorded separately",
        "seed": "public constant-current-state, zero-control seed and public warm starts",
        "limits": "public soft penalties; independent sampled executed q/v/u validation",
        "collision_scope": "obstacle-free arm-only model; collision clearance unassessed",
        "final_approach": False,
    }
    write_json(output / "receipt.json", receipt)

    states, controls, times = [initial.copy()], [], [0.0]
    plans, solve_times = [], []
    original_step, original_solve = production.rk4, controller.solver.solve

    def recorded_step(model, data, q, v, u, dt, forces):
        next_q, next_v = original_step(model, data, q, v, u, dt, forces)
        state = np.concatenate([next_q, next_v])
        states.append(state.copy())
        controls.append(np.asarray(u).copy())
        times.append(times[-1] + dt)
        if not np.isfinite(state).all() or np.max(np.abs(state[7:])) > 50:
            raise RuntimeError("Nonfinite or runaway executed state")
        return next_q, next_v

    def recorded_solve(*args, **kwargs):
        start = time.perf_counter()
        result = original_solve(*args, **kwargs)
        plans.append(np.asarray(result[0]).copy())
        solve_times.append(time.perf_counter() - start)
        if not np.isfinite(plans[-1]).all():
            raise RuntimeError("Nonfinite returned solver plan")
        return result

    stats, execution_error = {}, None
    production.rk4, controller.solver.solve = recorded_step, recorded_solve
    try:
        if case["kind"] == "reaching":
            _, stats = controller.run_mpc_goals(
                initial.copy(), [goal], sim_dt=case["sim_dt_s"],
                goal_timeout=case["timeout_s"], goal_threshold=case["position_threshold_m"],
                goal_axis_threshold=np.deg2rad(case["axis_threshold_deg"]),
                velocity_threshold=case["speed_L1_threshold_rad_s"], goal_dwell_time=0.0,
            )
        else:
            _, stats = controller.run_mpc_fig8(
                initial.copy(), reference, sim_dt=case["sim_dt_s"],
                sim_time=case["duration_s"], offline_timing="controller_dt",
            )
    except Exception:
        execution_error = traceback.format_exc()
    finally:
        production.rk4, controller.solver.solve = original_step, original_solve

    x, u, t = np.asarray(states), np.asarray(controls).reshape(-1, 7), np.asarray(times)
    # Save observations before metric evaluation, including partially failed runs.
    np.savez_compressed(output / "execution.npz", t=t, x=x, u=u,
                        plans=np.asarray(plans), solve_wall_s=np.asarray(solve_times))
    write_json(output / "public_stats.json", stats)
    summary = evaluate_execution(case, model, x, u, t, plans, stats, execution_error, reference)
    summary["metrics"]["solve_wall_total_s"] = float(sum(solve_times))
    write_json(output / "summary.json", summary)
    print((output / "summary.json").read_text(encoding="utf-8"), end="")
    return 0 if summary["passed"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["short_reach", "long_reach", "figure8_tracking"], required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    case = json.loads((CASES / f"{args.case}.json").read_text(encoding="utf-8"))
    try:
        return run_case(case, args.artifact_dir)
    except Exception:
        summary = {"case": args.case, "passed": False, "error": traceback.format_exc()}
        write_json(args.artifact_dir / "summary.json", summary)
        print((args.artifact_dir / "summary.json").read_text(encoding="utf-8"), end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
