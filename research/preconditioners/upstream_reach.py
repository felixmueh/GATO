#!/usr/bin/env python3
"""One public-upstream Indy7 whole-reaching episode, with retained raw records."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_native(path, plant, knots):
    target = f"gato.bsqpN{knots}_{plant}"
    if path is None:
        # Import directly: a missing built module must not trigger auto-building.
        return importlib.import_module(target)
    name = path.name.split(".")[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load native extension {path}")
    native = importlib.util.module_from_spec(spec)
    sys.modules[name] = native
    spec.loader.exec_module(native)
    sys.modules[target] = native
    return native


def physical_limits(model, q, v, u):
    finite = bool(np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(u).all())
    out = {"finite": finite, "sampling": "initial state and every 1 ms plant substep, including terminal state",
           "collision_check": "not performed; simulated numerical benchmark"}
    if not finite:
        return {**out, "model_limits_pass": False}
    qp = float(np.maximum(np.maximum(model.lowerPositionLimit - q, q - model.upperPositionLimit), 0).max())
    vp = float(np.maximum(np.abs(v) - model.velocityLimit, 0).max())
    up = float(np.maximum(np.abs(u) - model.effortLimit, 0).max()) if len(u) else 0.0
    return {**out, "position_max_violation_rad": qp, "speed_max_violation_rad_s": vp,
            "effort_max_violation_nm": up, "comparison_tolerance": 1e-5,
            "model_limits_pass": qp <= 1e-5 and vp <= 1e-5 and up <= 1e-5}


def transfer_total(metrics):
    """Only an explicit cumulative ALL-copy counter establishes compute-only wall."""
    if not isinstance(metrics, dict):
        return None
    totals = metrics.get("totals", metrics)
    if metrics.get("transfer_measurement_available", totals.get("transfer_measurement_available", True)) is False:
        return None
    return float(totals["transfer_wall_us"]) if "transfer_wall_us" in totals else None


def resolve_algorithm(method, linsys=None):
    if method == "gpu_bdsv":
        if linsys not in (None, "bdsv"):
            raise ValueError("gpu_bdsv requires --linsys bdsv or no explicit override")
        return "normal", "bdsv", "GPU direct BDSV solve; iteration fields are bookkeeping, not CG work"
    linsys = linsys or "pcg"
    if linsys == "bdsv":
        if method not in ("normal", "native", "normal_raw", "normal_shared"):
            raise ValueError("A preconditioner method cannot be combined with direct BDSV")
        return method, linsys, "GPU direct BDSV solve; iteration fields are bookkeeping, not CG work"
    if method.startswith("block_ldlt"):
        return method, linsys, "CPU exact block LDLT factors + GPU triangular solves inside PCG"
    return method, linsys, "GPU native damped-stair PCG" if method in ("normal", "normal_shared", "normal_raw", "native") else method


def main(argv=None):
    worker_entry = time.perf_counter_ns()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modulepath", type=Path, help="experimental .so; omit to load the checkout's standard module")
    ap.add_argument("--checkout", type=Path, default=ROOT, help="public source/native checkout to evaluate")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("upstream_reach_config.json"))
    ap.add_argument("--method", default="normal")
    ap.add_argument("--linsys", choices=("pcg", "bdsv"))
    ap.add_argument("--primary-timing", choices=("wall", "compute_excluding_transfers"), default="compute_excluding_transfers")
    ap.add_argument("--task-file", type=Path, help="pre-frozen public pickplace/fig8 task; no per-run IK generation")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout / "python"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    native_method, linsys, algorithm = resolve_algorithm(args.method, args.linsys)
    config.update(linsys=linsys, native_method=native_method, algorithm=algorithm,
                  iteration_count_semantics="actual PCG iterations" if linsys == "pcg" else "native direct-solve bookkeeping, not CG iterations")
    task = json.loads(args.task_file.read_text()) if args.task_file else None
    task_kind = task["task"] if task else "pickplace"
    config.update(method=args.method, primary_timing=args.primary_timing, argv=sys.argv,
                  checkout=str(checkout),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
                  dirty_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True),
                  driver_sha256=sha256(__file__), run_config_sha256=sha256(args.config),
                  harness_source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  harness_dirty_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
                  task_kind=task_kind,
                  wall_boundary="production run_mpc_goals entry/exit, including reset, initial plan, all feedback solves, Pinocchio simulation and shared recording overhead",
                  timing_scope=("actual whole-task wall; no transfer subtraction; construction and file writes excluded"
                    if args.primary_timing == "wall" else "task wall minus all instrumented H2D/D2H copy wall; construction and file writes excluded"),
                  scenario_source="examples/12_constraint_mechanisms.py: seed 1, scenario 0, ramps=(0.0,)")
    if task:
        config.update(task_id=task["case_id"], scenario_seed=task["seed"],
                      frozen_task_file=str(args.task_file.resolve()), frozen_task_sha256=sha256(args.task_file),
                      scenario_source="frozen public gato.tasks generator; see frozen_task.json",
                      wall_boundary="production run_mpc_goals/run_mpc_fig8 entry/exit, including reset, initial plan, all feedback solves, Pinocchio simulation and shared recording overhead")
    write_json(output / "config.json", config)
    (output / "upstream_reach.py").write_text(Path(__file__).read_text())
    (output / "upstream_reach_config.json").write_bytes(args.config.read_bytes())
    if task:
        (output / "frozen_task.json").write_bytes(args.task_file.read_bytes())
    records, currents, references, seeds, solutions = [], [], [], [], []
    plant_q, plant_v, plant_u, plant_t = [], [], [], []
    stats, metrics, error = {}, {}, None
    native = model = mpc = None
    measured_start = measured_end = wall_us = None
    setup_wall_us = None
    reference_traj = None
    transcript = io.StringIO()
    controller_walls = []
    try:
        import pinocchio as pin
        import gato
        from gato import MPC_GATO, tasks
        from gato.config import SolverParams
        from gato import _grid

        plant, knots = config["plant"], config["knots"]
        native = load_native(args.modulepath.resolve() if args.modulepath else None, plant, knots)
        if hasattr(native, "set_method"):
            native.set_method(native_method)
        elif native_method not in ("native", "normal"):
            raise RuntimeError("requested method requires native set_method()")
        if hasattr(native, "set_end_to_end_mode"):
            native.set_end_to_end_mode(args.primary_timing == "wall")
        if hasattr(native, "prepare"):
            native.prepare()
        if hasattr(native, "shared_memory_info"):
            config["shared_memory_info_after_prepare"] = native.shared_memory_info()
        urdf = Path(gato.robot_urdf(plant))
        model = pin.buildModelFromUrdf(str(urdf))
        scenario = task["scenario"] if task else tasks.pickplace_scenarios(model, plant, 1, seed=config["scenario_seed"],
                                             n_goals=config["n_goals"], ramps=(config["goal_ramp_s"],))[0]
        initial = np.r_[scenario["q0"], np.zeros(model.nv)]
        goals = np.asarray(scenario["goals"]) if task_kind == "pickplace" else np.empty((0, 3))
        reference_traj = None
        if task_kind == "fig8":
            reference_path = args.task_file.parent / task["reference_file"]
            if sha256(reference_path) != task["reference_sha256"]:
                raise RuntimeError("Frozen tracking reference hash changed")
            with np.load(reference_path) as saved:
                reference_traj = saved["reference"].copy()
            config.update(requested_tracking_duration_s=task["sim_time_s"],
                          frozen_reference_sha256=task["reference_sha256"])
        params = SolverParams.from_mapping(config["solver_params"])
        mpc = MPC_GATO(model, model_path=str(urdf), N=knots, dt=config["dt_s"],
                       batch_size=config["batch_size"], track_full_stats=True, plant_type=plant,
                       linsys=config["linsys"], limits=config["limits"], params=params)
        if hasattr(native, "shared_memory_info"):
            config["shared_memory_info_after_construction"] = native.shared_memory_info()
            allocated = config["shared_memory_info_after_construction"]
            if native_method.endswith('_shared') and not (allocated["last_s_allocation_managed"] and allocated["s_allocation_count"] >= 1):
                raise RuntimeError("Requested managed S allocation was not recorded; no fallback")
        config.update(scenario=scenario, initial_state=initial, goals=goals,
                      initial_ee_position_m=mpc.solver.ee_pos(initial[:model.nq]),
                      initial_goal_distances_m=np.linalg.norm(goals - mpc.solver.ee_pos(initial[:model.nq]), axis=1),
                      native_module=str(native.__file__), native_sha256=sha256(native.__file__),
                      native_source_id=getattr(native, "NATIVE_SOURCE_ID", None),
                      urdf=str(urdf), urdf_sha256=sha256(urdf), pinocchio_version=pin.__version__,
                      grid_rbd_version=_grid.installed_version(), glass_commit=_grid.glass_commit(),
                      effective_solver_params=mpc.solver.params.asdict(),
                      effective_linsys=mpc.controller.linsys, effective_limits=mpc.controller.limits)
        write_json(output / "config.json", config)
        task_arrays = dict(initial_state=initial, goals=goals)
        if reference_traj is not None:
            task_arrays["full_tracking_reference"] = reference_traj
        np.savez_compressed(output / "task.npz", **task_arrays)
        original_solve, original_step = mpc.solver.solve, mpc.world.step
        original_controller_step, original_warmup = mpc.controller.step, mpc.controller.warmup

        def observed_solve(xcur, reference, seed=None, **kw):
            call = len(records)
            record = {"call": call, "phase": "initial_plan" if call == 0 else "feedback"}
            records.append(record)
            currents.append(np.asarray(xcur).copy())
            references.append(np.asarray(reference).copy())
            seeds.append(np.asarray(seed).copy() if seed is not None else np.empty(0))
            begin = time.perf_counter_ns()
            try:
                result = original_solve(xcur, reference, seed, **kw)
                record["wrapper_solve_wall_us"] = (time.perf_counter_ns() - begin) / 1000
                record["native_solve_wall_us"] = result.solve_time_us
                record["stats"] = {field.name: (getattr(result.stats, field.name).copy()
                                   if isinstance(getattr(result.stats, field.name), np.ndarray)
                                   else getattr(result.stats, field.name))
                                   for field in dataclasses.fields(result.stats)}
                solutions.append(np.asarray(result.xu).copy())
                record["output_finite"] = bool(np.isfinite(result.xu).all())
                return result
            except BaseException as exc:
                record["wrapper_solve_wall_us"] = (time.perf_counter_ns() - begin) / 1000
                record["error"] = repr(exc)
                raise

        def observed_step(q, v, u, dt):
            result = original_step(q, v, u, dt)
            plant_q.append(np.asarray(result[0]).copy())
            plant_v.append(np.asarray(result[1]).copy())
            plant_u.append(np.asarray(u).copy())
            plant_t.append(plant_t[-1] + float(dt))
            return result

        def controller_call(original, phase, *values, **kw):
            begin = time.perf_counter_ns()
            try:
                return original(*values, **kw)
            finally:
                controller_walls.append({"phase": phase, "wall_us": (time.perf_counter_ns() - begin) / 1000})

        mpc.solver.solve, mpc.world.step = observed_solve, observed_step
        mpc.controller.step = lambda *a, **kw: controller_call(original_controller_step, "feedback", *a, **kw)
        mpc.controller.warmup = lambda *a, **kw: controller_call(original_warmup, "initial_plan", *a, **kw)
        plant_q.append(initial[:model.nq].copy())
        plant_v.append(initial[model.nq:].copy())
        plant_t.append(0.0)
        if hasattr(native, "reset_metrics"):
            native.reset_metrics()
        setup_wall_us = (time.perf_counter_ns() - worker_entry) / 1000
        with contextlib.redirect_stdout(transcript):
            measured_start = time.time_ns()
            begin = time.perf_counter_ns()
            try:
                if task_kind == "fig8":
                    _, stats = mpc.run_mpc_fig8(initial, reference_traj, sim_dt=config["sim_dt_s"],
                        sim_time=task["sim_time_s"], pace_by_solve_time=config["pace_by_solve_time"])
                else:
                    _, stats = mpc.run_mpc_goals(initial, list(goals), sim_dt=config["sim_dt_s"],
                        goal_timeout=config["goal_timeout_s"], goal_threshold=config["goal_threshold_m"],
                        velocity_threshold=config["velocity_threshold_rad_s"], velocity_norm=config["velocity_norm"],
                        settle_time=config["settle_time_s"], goal_ramp=config["goal_ramp_s"],
                        pace_by_solve_time=config["pace_by_solve_time"])
            finally:
                measured_end = time.time_ns()
                wall_us = (time.perf_counter_ns() - begin) / 1000
    except BaseException:
        error = traceback.format_exc()
    finally:
        # All serialization and native telemetry export happen after timing, also on failure.
        if native is not None and hasattr(native, "metrics"):
            try:
                metrics = native.metrics()
            except BaseException:
                metrics = {"export_error": traceback.format_exc()}
        if native is not None and hasattr(native, "shared_memory_info"):
            try:
                config["shared_memory_info_after_task"] = native.shared_memory_info()
            except BaseException:
                config["shared_memory_info_export_error"] = traceback.format_exc()
            write_json(output / "config.json", config)
        write_json(output / "metrics.json", metrics)
        write_json(output / "solve_records.json", records)
        write_json(output / "task_stats.json", stats)
        write_json(output / "controller_records.json", controller_walls)
        (output / "task_stdout.txt").write_text(transcript.getvalue())
        arrays = {"plant_q": np.asarray(plant_q), "plant_v": np.asarray(plant_v),
                  "plant_u": np.asarray(plant_u), "plant_time_s": np.asarray(plant_t)}
        for name, values in (("input_state", currents), ("reference", references),
                             ("warm_seed", seeds), ("solution", solutions)):
            for i, value in enumerate(values):
                arrays[f"{name}_{i:04d}"] = value
        np.savez_compressed(output / "raw.npz", **arrays)
        end_to_end_mode = bool(config.get("shared_memory_info_after_prepare", {}).get("end_to_end_mode", False))
        transfer_us = transfer_total(metrics) if args.method not in ("native", "normal_raw") and not end_to_end_mode else None
        total_pcg = sum(int(np.asarray(r["stats"]["pcg_iters"]).sum()) for r in records if "stats" in r)
        successful = [r for r in records if "stats" in r]
        if native_method == "block_ldlt_shared" and error is None and successful:
            if not config.get("shared_memory_info_after_task", {}).get("shared_s_pointer_verified", False):
                error = "Completed shared LDLT solve did not report managed S pointer verification"
        native_times = np.asarray([r["native_solve_wall_us"] for r in successful], dtype=float)
        feedback_calls = sum(r["phase"] == "feedback" for r in records)
        adjusted_us = wall_us - transfer_us if wall_us is not None and transfer_us is not None else None
        realized_s = plant_t[-1] if plant_t else None
        terminal_ee = None
        if mpc is not None and plant_q:
            try:
                terminal_ee = mpc.solver.ee_pos(plant_q[-1])
            except BaseException:
                config["terminal_fk_export_error"] = traceback.format_exc()
                write_json(output / "config.json", config)
        tracking = None
        if task_kind == "fig8":
            errors = np.asarray(stats.get("goal_distances_knot0", []), dtype=float)
            endpoint_index = int(realized_s / config["dt_s"]) if realized_s is not None else None
            endpoint = None
            if endpoint_index is not None and reference_traj is not None and terminal_ee is not None:
                endpoint_goal = reference_traj.reshape(-1, 6)[endpoint_index, :3]
                endpoint = float(np.linalg.norm(terminal_ee - endpoint_goal))
            tracking = dict(requested_duration_s=task["sim_time_s"], realized_duration_s=realized_s,
                completed_requested_duration=bool(error is None and realized_s is not None and realized_s + 1e-9 >= task["sim_time_s"]),
                error_convention="Euclidean position error at production feedback timestamps, native reference knot 0",
                error_samples=len(errors), rms_m=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                p95_m=float(np.percentile(errors, 95)) if len(errors) else None,
                max_m=float(errors.max()) if len(errors) else None,
                endpoint_error_m=endpoint, last_feedback_error_m=float(errors[-1]) if len(errors) else None,
                endpoint_reference_index=endpoint_index)
        summary = dict(error=error, measured_start_unix_ns=measured_start, measured_end_unix_ns=measured_end,
            whole_task_wall_us=wall_us, all_transfer_wall_us=transfer_us,
            whole_task_compute_us=adjusted_us, task_kind=task_kind, tracking=tracking,
            primary_timing=args.primary_timing,
            primary_task_time_us=wall_us if args.primary_timing == "wall" else adjusted_us,
            algorithm=algorithm, native_method=native_method, linsys=linsys,
            whole_episode_wall_ms=wall_us / 1000 if wall_us is not None else None,
            solver_calls_including_initial=len(records), initial_plan_calls=sum(r["phase"] == "initial_plan" for r in records),
            feedback_calls=feedback_calls,
            episode_wall_per_solver_call_ms=wall_us / len(records) / 1000 if wall_us is not None and records else None,
            episode_wall_per_feedback_call_ms=wall_us / feedback_calls / 1000 if wall_us is not None and feedback_calls else None,
            native_solve_wall_median_ms=float(np.median(native_times)) / 1000 if len(native_times) else None,
            native_solve_wall_p95_ms=float(np.percentile(native_times, 95)) / 1000 if len(native_times) else None,
            episode_setup_wall_us=setup_wall_us,
            worker_elapsed_before_summary_export_us=(time.perf_counter_ns() - worker_entry) / 1000,
            realized_physical_duration_s=realized_s,
            adjusted_cost_per_completed_solve_us=adjusted_us / len(successful) if adjusted_us is not None and successful else None,
            adjusted_cost_per_physical_sim_second_us=adjusted_us / realized_s if adjusted_us is not None and realized_s else None,
            transfer_measurement_available=transfer_us is not None,
            transfer_scope="all instrumented H2D/D2H copies" if transfer_us is not None else "unmeasuredtransfer",
            solve_calls_attempted=len(records), solve_calls_completed=len(successful), pcg_iterations_total=total_pcg,
            actual_pcg_iterations_total=total_pcg if linsys == "pcg" else None,
            linear_solver_iteration_bookkeeping_total=total_pcg,
            native_solve_wall_us_total=sum(r["native_solve_wall_us"] for r in successful),
            wrapper_solve_wall_us_total=sum(r.get("wrapper_solve_wall_us", 0) for r in records),
            controller_wall_us_total=sum(r["wall_us"] for r in controller_walls),
            all_completed_solver_outputs_finite=bool(successful) and all(r["output_finite"] for r in successful),
            task_outcomes=stats.get("goal_outcomes"), time_to_all_reached_s=stats.get("time_to_all_reached"),
            all_goals_reached=stats.get("goal_outcomes") == ["reached"] * config["n_goals"] if task_kind == "pickplace" else None,
            all_saved_merits_finite=bool(successful) and all(np.isfinite(np.asarray(r["stats"][key], dtype=float)).all()
                for r in successful for key in ("initial_merit", "final_merit", "min_merit")),
            terminal_ee_position_m=terminal_ee,
            physical_limits=physical_limits(model, np.asarray(plant_q), np.asarray(plant_v), np.asarray(plant_u))
                if model is not None and plant_q else {"checked": False},
            merit_description="native initial/final/min merit retained per call; includes penalties, not a separately evaluated trajectory objective")
        write_json(output / "summary.json", summary)
        def shown(value):
            return f"{value:.3f}" if value is not None else "unavailable"
        print(f"WHOLE EPISODE wall {shown(summary['whole_episode_wall_ms'])} ms | "
              f"{len(records)} solver calls ({summary['initial_plan_calls']} initial + {feedback_calls} feedback) | "
              f"wall/call {shown(summary['episode_wall_per_solver_call_ms'])} ms | sim {shown(realized_s)} s", flush=True)
        print(f"Native solve wall median/p95 {shown(summary['native_solve_wall_median_ms'])}/"
              f"{shown(summary['native_solve_wall_p95_ms'])} ms | {algorithm}", flush=True)
        print(json.dumps(jsonable(summary), allow_nan=False), flush=True)
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
