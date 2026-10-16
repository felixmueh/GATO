#!/usr/bin/env python3
"""One frozen N256 / terminal10 / 10ms reaching case in a fresh B1 session."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASELINE_HERE = HERE.parent / "random_reaching_baseline"
BASE = "85ef2ea5cf5b96726be6f4b5441f4a7c0ca041b4"
BASELINE_PROTOCOL_SHA256 = "511b99b2a0ed817f0bb881a84ca978e0de5d74d5e046321451db6231fe2c6b8f"
MODEL = "gato/dynamics/tiago_right/tiago_right_arm.urdf"
HISTORICAL_WORKER = "84bad9ad3d88b1b50d01f4d7aa3721bbc31ea210"
N = 256
FIXED = dict(timing_mode="A", offline_timing="controller_dt", N=N,
             knot_dt_s=.03, feedback_s=.01, sim_dt_s=.003, timeout_s=30.,
             position_tolerance_m=.05, tool_axis_tolerance_deg=30.,
             velocity_l1_tolerance_rad_s=1., goal_dwell_s=0., final_approach=False, batch=1,
             reset_rho_each_solve=False, joint_cost_mode=1, canonical_timeout_s=8.,
             arrival_orientation_gate=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if hasattr(value, "tolist"):
        return safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def prepare(args):
    """Stdlib validation, with no native import or device query."""
    protocol = json.loads(args.protocol.read_text())
    baseline = json.loads((BASELINE_HERE / "protocols/OFF_TORQUE-orientation.json").read_text())
    if sha(BASELINE_HERE / "protocols/OFF_TORQUE-orientation.json") != BASELINE_PROTOCOL_SHA256:
        raise ValueError("Frozen historical cost reference changed")
    expected = dict(baseline["solver_params"], max_sqp_iters=1000, N_cost=800.)
    if protocol["solver_params"] != expected or protocol["limit_penalty"] != baseline["limit_penalty"]:
        raise ValueError("Frozen costs or interior bounds differ")
    for key, value in FIXED.items():
        if protocol.get(key) != value:
            raise ValueError("Frozen task/timing policy differs: " + key)
    if sha(args.tasks) != protocol["tasks_sha256"]:
        raise ValueError("Frozen30 tasks changed")
    tasks = json.loads(args.tasks.read_text())
    if len(tasks) != 30 or len({r["case_id"] for r in tasks}) != 30:
        raise ValueError("Expected30 unique cases")
    for row in tasks:
        if len(row["initial_state"]) != 14 or len(row["goal_reference_single"]) != 6 or row["saved_goal_witness_used"]:
            raise ValueError("Invalid full state, reference or witness policy")
    if args.preflight_input:
        if sha(args.preflight_input) != protocol["canonical_sha256"]:
            raise ValueError("Frozen canonical geometry changed")
        cases = ["canonical"]
    else:
        if args.case_id not in {r["case_id"] for r in tasks}:
            raise ValueError("Unknown frozen case")
        cases = [args.case_id]
    if args.output.exists():
        raise ValueError("Fresh output directory required")
    for name, expected_sha in protocol["source_sha256"].items():
        if sha(args.source / name) != expected_sha:
            raise ValueError("Reviewed source changed: " + name)
    return protocol, dict(protocol=protocol, source=str(args.source), source_commit=git(args.source,"rev-parse","HEAD"),
        source_sha256=protocol["source_sha256"], module=str(args.module),
        module_sha256=sha(args.module) if args.module.is_file() else None,
        tasks=str(args.tasks), tasks_sha256=sha(args.tasks), cases=cases, wrapper_sha256=sha(__file__),
        preflight_input=str(args.preflight_input) if args.preflight_input else None,
        recording="Dense full q/v and applied torque, initial full-state hold/zero-control seed, alltyped native work, feedback metrics and physicalchecks. Routine horizons omitted; invalidplan retained.",
        execution="ModeA/controller_dt; N256/30ms knots/10ms feedback/3+3+3+1ms local RK4; solve time unpaid. Fresh session/reset dual+rho once thenretain; no posture bias/witness.")


def load_runtime(source, module):
    """Bind the explicit fresh module before constructing production MPC_GATO."""
    sys.path[:0] = [str(source / "python"), str(source / "tiago_src")]
    import numpy as np
    import pinocchio as pin
    importlib.import_module("bsqp")
    name = "bsqp.bsqpN256_tiago_right"
    spec = importlib.util.spec_from_file_location(name, module)
    native = importlib.util.module_from_spec(spec)
    sys.modules[name] = native
    spec.loader.exec_module(native)
    controller_module = importlib.import_module("gato_tiago.tiago_mpc_controller")
    common = importlib.import_module("bsqp.common")
    for loaded, relative in ((controller_module, "tiago_src/gato_tiago/tiago_mpc_controller.py"), (common, "python/bsqp/common.py")):
        if Path(loaded.__file__).resolve() != source / relative:
            raise ValueError("Imported Python implementation from a different source")
    return np, pin, controller_module.MPC_GATO, common, native


def unpack(np, packed):
    body = np.asarray(packed)[:(N-1) * 21].reshape(N-1, 21)
    return np.vstack((body[:, :14], np.asarray(packed)[-14:])), body[:, 14:]


def pose(pin, model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return (data.oMf[model.getFrameId("torso_lift_link")].inverse()
            * data.oMf[model.getFrameId("arm_right_tool_link")]).copy()


def metrics(np, pin, model, data, x, goal):
    if not np.isfinite(x).all():
        return dict(position_error_m=float("nan"), tool_axis_error_deg=float("nan"),
                    orientation_error_deg=float("nan"), velocity_l1_rad_s=float("nan"))
    actual = pose(pin, model, data, np.asarray(x[:7], dtype=float))
    return dict(position_error_m=float(np.linalg.norm(actual.translation - goal.translation)),
                tool_axis_error_deg=float(np.rad2deg(np.arccos(np.clip(actual.rotation[:, 2] @ goal.rotation[:, 2], -1., 1.)))),
                orientation_error_deg=float(np.rad2deg(np.linalg.norm(pin.log3(goal.rotation.T @ actual.rotation)))),
                velocity_l1_rad_s=float(np.abs(x[7:]).sum()))


def arrival(observation, protocol):
    return (observation['position_error_m'] <= protocol['position_tolerance_m'] and
            observation['velocity_l1_rad_s'] <= protocol['velocity_l1_tolerance_rad_s'])


def limit_checks(np, model, states, controls, state_times, control_times):
    states, controls = np.asarray(states).reshape(-1, 14), np.asarray(controls).reshape(-1, 7)
    finite = bool(np.isfinite(states).all() and np.isfinite(controls).all())
    result = dict(finite=finite, physical_pass=finite)
    q = states[:, :7]
    excesses = dict(joint=np.maximum(np.maximum(model.lowerPositionLimit-q, q-model.upperPositionLimit), 0.),
                    velocity=np.maximum(np.abs(states[:, 7:])-model.velocityLimit, 0.),
                    effort=np.maximum(np.abs(controls)-model.effortLimit, 0.))
    for name, array in excesses.items():
        times = np.asarray(control_times if name == "effort" else state_times)
        tolerance = 1e-6 if name == "joint" else 1e-6 * (model.velocityLimit if name == "velocity" else model.effortLimit)
        valid = bool(np.isfinite(array).all() and np.all(array <= tolerance))
        result["physical_pass"] &= valid
        violations = np.argwhere(array > tolerance)
        first = None
        if len(violations):
            index, joint = map(int, violations[0])
            first = dict(sample_index=index, time_s=float(times[index]), joint_index=joint,
                         joint_name=str(model.names[joint+1]), excess=float(array[index, joint]))
        maximum = float(np.max(array)) if array.size and np.isfinite(array).all() else (0. if not array.size else float("nan"))
        result[name] = dict(pass_check=valid, maximum_excess=maximum, first_violation=first)
    result["maximum_velocity_ratio"] = float(np.max(np.abs(states[:, 7:])/model.velocityLimit))
    result["maximum_effort_ratio"] = float(np.max(np.abs(controls)/model.effortLimit)) if controls.size else 0.
    firsts = []
    for kind, array in excesses.items():
        raw = np.argwhere(array > 0.)
        first = None
        if len(raw):
            index, joint = map(int, raw[0])
            first = dict(kind=kind, time_s=float((control_times if kind == "effort" else state_times)[index]),
                         sample_index=index, joint_index=joint, joint_name=str(model.names[joint+1]),
                         excess=float(array[index,joint]))
            firsts.append(first)
        result[kind]["first_raw_violation"] = first
        result[kind]["raw_breach_samples"] = int(np.count_nonzero(np.any(array > 0.,axis=1)))
    result["first_raw_violation"] = min(firsts,key=lambda r:(r["time_s"], ["joint","velocity","effort"].index(r["kind"]),r["joint_index"])) if firsts else None
    result["raw_violation_kinds"] = [kind for kind,array in excesses.items() if np.any(array > 0.)]
    return result


def work_summary(np, stats, cap):
    pcg = np.asarray(stats["pcg_iters"]).reshape(-1)
    sqp = int(np.asarray(stats["sqp_iters"]).reshape(-1)[0])
    ls = int(stats["ls_num_iters"])
    zero_exit = bool(pcg.size and pcg[-1] == 0 and ls < sqp)
    # Existing KKT flags are recorded but are never a study acceptance gate.
    reason = "zero_pcg_outer_exit" if zero_exit else "existing_residual_exit" if ls < sqp else "budget_exhausted" if sqp >= cap else "unclassified_existing_exit"
    return dict(actual_sqp=sqp, pcg_iteration_sum=int(pcg.sum()), pcg_count_entries=int(pcg.size),
                last_pcg_iterations=int(pcg[-1]) if pcg.size else None, line_search_count=ls,
                accepted_steps=int(np.count_nonzero(np.asarray(stats["step_size"]) > 0)),
                zero_pcg_outer_exit=zero_exit, stop_reason=reason,
                existing_kkt_flag=int(np.asarray(stats["kkt_converged"]).reshape(-1)[0]),
                initial_merit=float(np.asarray(stats["initial_merit"]).reshape(-1)[0]),
                final_merit=float(np.asarray(stats["final_merit"]).reshape(-1)[0]))


def task(np, pin, model, data, args, case_id):
    if args.preflight_input:
        old = json.loads(args.preflight_input.read_text())
        initial = np.asarray(old["initial_state"], dtype=float)
        target = np.asarray(old["goal"], dtype=float)
        goal = pin.SE3(pin.rpy.rpyToMatrix(target[3:]), target[:3])
        indices = dict(case_id="canonical", pair_id=-1, start_type="canonical", source_base_time_s=None)
    else:
        row = next(r for r in json.loads(args.tasks.read_text()) if r["case_id"] == case_id)
        initial = np.asarray(row["initial_state"], dtype=float)
        goal = pin.SE3(np.asarray(row["target_rotation"], dtype=float), np.asarray(row["target_xyz"], dtype=float))
        target = np.asarray(row["goal_reference_single"], dtype=np.float32)
        indices = dict(case_id=case_id, pair_id=row["pair_id"], start_type=row["start_type"],
                       source_base_time_s=row.get("source_base_time_s", row.get("restart_time_s")))
    reference = np.tile(target, N).astype(np.float32)[None]
    return initial, np.zeros(7), goal, reference, indices


def run_episode(runtime, args, protocol, pair):
    np, pin, MPC, common, native = runtime
    out = args.output / ("preflight" if pair == "canonical" else str(pair))
    out.mkdir()
    cfg = protocol["solver_params"]
    model = pin.buildModelFromUrdf(str(args.source / MODEL))
    model.gravity.linear = np.array([0., 0., -9.81])
    data = model.createData()
    x, ignored_reference, goal, reference, indices = task(np, pin, model, data, args, pair)
    states, controls, times, control_times = [x.copy()], [], [0.], []
    actual_inputs = []
    rows, observations = [], [metrics(np, pin, model, data, x, goal)]
    horizon_stats = []
    timeout = protocol["canonical_timeout_s"] if pair == "canonical" else protocol["timeout_s"]
    reached = None
    position_touch = 0. if observations[0]["position_error_m"] <= .05 else None
    pose_touch = 0. if position_touch == 0. and observations[0]["tool_axis_error_deg"] <= 30. else None
    failure = None
    failure_phase = None
    interrupted = False
    phase = "controller_setup"
    started = time.perf_counter()
    warm = common.initialize_warm_start(x, N, 14, 7)[None].astype(np.float32)
    np.savez_compressed(out / "inputs.npz", initial_state=x, target_xyz=goal.translation, target_rotation=goal.rotation,
                        reference=reference, first_seed=warm)
    write(out / "preparation.json", dict(**indices, protocol_id=protocol["id"],
          selected_input_sha256=sha(out / "inputs.npz"), initial_state_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
          goal_pose_sha256=hashlib.sha256(np.r_[goal.translation, goal.rotation.ravel()].tobytes()).hexdigest(),
          initial=observations[0], solver_params=cfg))
    try:
        # Module/context-global configuration is set once, before a fresh solver.
        # Flat mode ignores this zero reference; no goal witness is loaded.
        native.set_tiago_joint_cost_mode(protocol["joint_cost_mode"], ignored_reference.tolist())
        effective_joint_cost = native.get_tiago_joint_cost_mode()
        if abs(effective_joint_cost["joint_margin_rad"] - protocol["limit_penalty"]["cost_bounds"]["joint_margin_rad"]) > 1e-12:
            raise ValueError("Loaded compiled joint margin differs from declared arm")
        if effective_joint_cost["mode"] != protocol["joint_cost_mode"] or not np.array_equal(
                np.asarray(effective_joint_cost["q_reference"]), ignored_reference):
            raise ValueError("Native mode/reference setter did not retain the supplied configuration")
        write(out / "joint_cost_config.json", dict(requested_mode=protocol["joint_cost_mode"],
              q_reference=ignored_reference, effective_native=effective_joint_cost,
              native_scalar_reference=ignored_reference.astype(np.float32),
              reference_scope="zero placeholder ignored in flat mode1; no goal witness or posture bias"))
        controller = MPC(model, str(args.source / MODEL), N=N, dt=.03, batch_size=1,
                         plant_type="tiago_right", solver_params=cfg, track_full_stats=True)
        controller.force_estimator = None
        controller.solver.set_f_ext_B(np.zeros((1, 6), dtype=np.float32))
        controller.solver.reset_dual()
        controller.solver.reset_rho()
        phase = "episode"
        step = 0
        with (out / "solves.jsonl").open("w") as log:
            while times[-1] < timeout - 1e-10:
                if arrival(observations[-1], protocol):
                    reached = times[-1]
                    break
                now, duration = times[-1], min(protocol["feedback_s"], timeout - times[-1])
                actual_inputs.append(x.copy())
                prefix_index = len(states)-1
                solve_started = time.perf_counter()
                optimized, native_us = controller.solver.solve(x[None], reference, warm.copy())
                wall = time.perf_counter() - solve_started
                stats = controller.solver.get_stats()
                arrays = {name: np.asarray(stats[name]) for name in
                          ("pcg_iters", "sqp_iters", "step_size", "min_merit", "kkt_converged", "initial_merit", "final_merit")}
                horizon_stats.append({name: value.copy() for name, value in arrays.items()})
                predicted_x, predicted_u = unpack(np, optimized[0])
                row = dict(step=step, feedback_time_s=now, executed_duration_s=duration,
                           native_solve_us=float(native_us), solve_wall_s=wall,
                           work=work_summary(np, stats, cfg["max_sqp_iters"]))
                if not np.isfinite(optimized).all() or not math.isfinite(row["work"]["final_merit"]):
                    np.savez_compressed(out / "invalid_optimized.npz", optimized=optimized, actual_state=x, seed=warm)
                    rows.append(row); log.write(json.dumps(safe(row), allow_nan=False) + "\n"); log.flush()
                    raise RuntimeError("No finite optimized plan and merit; invalid output retained")
                u = predicted_u[0].astype(float)
                elapsed = 0.
                while elapsed < duration - 1e-12:
                    h = min(.003, duration-elapsed)
                    q, v = common.rk4(model, controller.data, x[:7], x[7:], u, h, controller.actual_f_ext)
                    x = np.r_[q, v]
                    control_times.append(times[-1]); controls.append(u.copy())
                    times.append(times[-1]+h); states.append(x.copy())
                    observations.append(metrics(np, pin, model, data, x, goal))
                    observation = observations[-1]
                    position = observation["position_error_m"] <= .05
                    touching = position and observation["tool_axis_error_deg"] <= 30.
                    if position and position_touch is None:
                        position_touch = times[-1]
                    if touching and pose_touch is None:
                        pose_touch = times[-1]
                    elapsed += h
                    if not np.isfinite(x).all():
                        break
                row["executed_prefix_limits"] = limit_checks(np, model, states[prefix_index:], [u], times[prefix_index:], [now])
                row["final"] = observations[-1]
                rows.append(row); log.write(json.dumps(safe(row), allow_nan=False) + "\n"); log.flush()
                if not np.isfinite(x).all():
                    raise RuntimeError("Nonfinite executed plant state retained")
                m = observations[-1]
                position = m["position_error_m"] <= .05
                touching = position and m["tool_axis_error_deg"] <= 30.
                if arrival(m, protocol):
                    reached = times[-1]
                    break
                warm = common.shift_packed_trajectory_warm_start(optimized[0], x, 14, 7, N, duration, .03)[None]
                step += 1
    except BaseException as error:
        failure = traceback.format_exc()
        failure_phase = phase
        interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
        write(out / "execution_failure.json", dict(error=failure, failure_phase=phase, interrupted=interrupted,
              simulated_seconds=times[-1], completed_solves=len(rows)))
    finally:
        # Retain complete typed work counters, not routine B1 solver horizons.
        # Full executed motion, applied controls, timestamps and initial seed remain.
        records = dict(actual_state=np.asarray(actual_inputs))
        for name in (horizon_stats[0] if horizon_stats else []):
            values = [row[name] for row in horizon_stats]
            records[name] = np.concatenate([value.ravel() for value in values])
            records[name + "_offsets"] = np.cumsum([0] + [value.size for value in values])
            records[name + "_shapes"] = np.asarray([value.shape for value in values], dtype=np.int64)
        np.savez_compressed(out / "work_stats.npz", **records)
        obs = {key: np.asarray([row[key] for row in observations]) for key in observations[0]}
        np.savez_compressed(out / "executed.npz", states=np.asarray(states), controls=np.asarray(controls).reshape(-1, 7),
                            times=np.asarray(times), control_times=np.asarray(control_times), **obs)
        checks = limit_checks(np, model, states, controls, times, control_times)
        indices_last = np.flatnonzero(np.asarray(times) >= max(0., times[-1]-1.))
        last_q = np.asarray(states)[indices_last, :7]
        errors = obs["position_error_m"][indices_last]
        hover = dict(observation_seconds=float(times[-1]-times[indices_last[0]]),
                     position_error_min_m=float(np.min(errors)), position_error_max_m=float(np.max(errors)),
                     net_position_error_reduction_m=float(errors[0]-errors[-1]),
                     joint_total_variation_L1_rad=float(np.abs(np.diff(last_q, axis=0)).sum()),
                     maximum_joint_speed_rad_s=float(np.max(np.abs(np.asarray(states)[indices_last, 7:]))),
                     maximum_speed_L1_rad_s=float(np.max(obs["velocity_l1_rad_s"][indices_last])),
                     limitation="Observation ends at rough arrival or timeout; no post-arrival settling inference")
        initial_xyz = pose(pin, model, data, np.asarray(states[0][:7], dtype=float)).translation
        final_xyz = pose(pin, model, data, np.asarray(states[-1][:7], dtype=float)).translation if np.isfinite(states[-1]).all() else np.full(3, np.nan)
        result = dict(**indices, arm=protocol["arm"], N=N, prediction_duration_s=(N-1)*.03,
                      outcome="execution_failed" if failure else "reached" if reached is not None else "timeout",
                      complete=True, task_success=reached is not None and failure is None,
                      physical_pass=bool(checks["physical_pass"] and failure is None and not interrupted), execution_error=failure,
                      failure_phase=failure_phase, shared_setup_failure=bool(failure and failure_phase == "controller_setup"),
                      interrupted=interrupted,
                      simulated_seconds=times[-1], reached_seconds=reached,
                      effective_timeout_s=timeout, joint_cost_mode=protocol["joint_cost_mode"],
                      arrival_policy=protocol['arrival_policy'], arrival_orientation_gate=False,
                      orientation_required_for_arrival=False,
                      tool_axis_tolerance_role='30degrees used only for descriptive first_pose_touch_seconds',
                      first_position_touch_seconds=position_touch, first_pose_touch_seconds=pose_touch,
                      touch_sampling="first dense3ms observation; rough arrival checked at configured feedback boundary",
                      initial=observations[0], final=observations[-1], physical=checks, last_second=hover,
                      executed_endpoint_displacement_m=float(np.linalg.norm(final_xyz-initial_xyz)),
                      feedback_solves=len(rows), actual_sqp_sum=sum(r["work"]["actual_sqp"] for r in rows),
                      actual_pcg_iteration_sum=sum(r["work"]["pcg_iteration_sum"] for r in rows),
                      zero_pcg_exit_solves=sum(r["work"]["zero_pcg_outer_exit"] for r in rows),
                      solve_wall_s=sum(r["solve_wall_s"] for r in rows), native_solve_us=sum(r["native_solve_us"] for r in rows),
                      episode_wall_s=time.perf_counter()-started, timing_mode="A", offline_timing="controller_dt")
        if pair == "canonical":
            old = json.loads(args.preflight_input.read_text())
            result["canonical_geometry_check"] = observations[0]["position_error_m"] >= old.get("minimum_initial_distance_m", .1)
            result["canonical_motion_check"] = result["executed_endpoint_displacement_m"] >= old.get("minimum_executed_displacement_m", .05)
            result["preflight_pass"] = bool(result["task_success"] and result["physical_pass"] and result["canonical_geometry_check"] and result["canonical_motion_check"])
        write(out / "metrics.json", result)
    print(json.dumps(safe(result), allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case-id")
    group.add_argument("--preflight-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate source/configuration/input hashes; no native imports, GPU calls or output writes")
    args = parser.parse_args()
    for key in ("source", "module", "tasks", "protocol", "output", "preflight_input"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    protocol, info = prepare(args)
    if args.dry_run:
        print(json.dumps(safe(info), indent=2, allow_nan=False))
        return 0
    if not args.module.is_file():
        parser.error("Native module missing: " + str(args.module))
    args.output.mkdir(parents=True, exist_ok=False)
    info["created_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "provenance.json", info)
    write(args.output / "protocol.json", protocol)
    if args.preflight_input:
        old = json.loads(args.preflight_input.read_text())
        write(args.output / "preflight_effective.json", dict(protocol, initial_state=old["initial_state"],
              goal=old["goal"], endpoint_delta_rad=old["endpoint_delta_rad"],
              reset_rho_each_solve=False, name=protocol["id"] + "-canonical",
              geometry_source_sha256=sha(args.preflight_input), status="new run; original file used for geometry only"))
    try:
        runtime = load_runtime(args.source, args.module)
    except BaseException:
        write(args.output / "setup_failure.json", dict(error=traceback.format_exc()))
        raise
    results = []
    for pair in info["cases"]:
        result = run_episode(runtime, args, protocol, pair)
        results.append(result)
        write(args.output / "summary.json", results)
        # Constructor/setup failures are shared failures, holding dependent cases.
        if result["shared_setup_failure"] or result["interrupted"]:
            break
    return int(any(r["execution_error"] or (args.preflight_input and not r["preflight_pass"]) for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
