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
    return float(totals["transfer_wall_us"]) if "transfer_wall_us" in totals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modulepath", type=Path, help="experimental .so; omit to load the checkout's standard module")
    ap.add_argument("--checkout", type=Path, default=ROOT, help="public source/native checkout to evaluate")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("upstream_reach_config.json"))
    ap.add_argument("--method", default="normal")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout / "python"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    config.update(method=args.method, argv=sys.argv,
                  checkout=str(checkout),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
                  dirty_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True),
                  driver_sha256=sha256(__file__), run_config_sha256=sha256(args.config),
                  wall_boundary="production run_mpc_goals entry/exit, including reset, initial plan, all feedback solves, Pinocchio simulation and shared recording overhead",
                  timing_scope="task wall minus all instrumented H2D/D2H copy wall; construction and file writes excluded",
                  scenario_source="examples/12_constraint_mechanisms.py: seed 1, scenario 0, ramps=(0.0,)")
    write_json(output / "config.json", config)
    (output / "upstream_reach.py").write_text(Path(__file__).read_text())
    (output / "upstream_reach_config.json").write_bytes(args.config.read_bytes())
    records, currents, references, seeds, solutions = [], [], [], [], []
    plant_q, plant_v, plant_u, plant_t = [], [], [], []
    stats, metrics, error = {}, {}, None
    native = model = mpc = None
    measured_start = measured_end = wall_us = None
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
            native.set_method(args.method)
        elif args.method not in ("native", "normal"):
            raise RuntimeError("requested method requires native set_method()")
        if hasattr(native, "prepare"):
            native.prepare()
        urdf = Path(gato.robot_urdf(plant))
        model = pin.buildModelFromUrdf(str(urdf))
        scenario = tasks.pickplace_scenarios(model, plant, 1, seed=config["scenario_seed"],
                                             n_goals=config["n_goals"], ramps=(config["goal_ramp_s"],))[0]
        initial = np.r_[scenario["q0"], np.zeros(model.nv)]
        goals = np.asarray(scenario["goals"])
        params = SolverParams.from_mapping(config["solver_params"])
        mpc = MPC_GATO(model, model_path=str(urdf), N=knots, dt=config["dt_s"],
                       batch_size=config["batch_size"], track_full_stats=True, plant_type=plant,
                       linsys=config["linsys"], limits=config["limits"], params=params)
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
        np.savez_compressed(output / "task.npz", initial_state=initial, goals=goals)
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
        with contextlib.redirect_stdout(transcript):
            measured_start = time.time_ns()
            begin = time.perf_counter_ns()
            try:
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
        transfer_us = transfer_total(metrics) if args.method not in ("native", "normal_raw") else None
        total_pcg = sum(int(np.asarray(r["stats"]["pcg_iters"]).sum()) for r in records if "stats" in r)
        successful = [r for r in records if "stats" in r]
        summary = dict(error=error, measured_start_unix_ns=measured_start, measured_end_unix_ns=measured_end,
            whole_task_wall_us=wall_us, all_transfer_wall_us=transfer_us,
            whole_task_compute_us=wall_us - transfer_us if wall_us is not None and transfer_us is not None else None,
            transfer_measurement_available=transfer_us is not None,
            transfer_scope="all instrumented H2D/D2H copies" if transfer_us is not None else "unmeasuredtransfer",
            solve_calls_attempted=len(records), solve_calls_completed=len(successful), pcg_iterations_total=total_pcg,
            native_solve_wall_us_total=sum(r["native_solve_wall_us"] for r in successful),
            wrapper_solve_wall_us_total=sum(r.get("wrapper_solve_wall_us", 0) for r in records),
            controller_wall_us_total=sum(r["wall_us"] for r in controller_walls),
            all_completed_solver_outputs_finite=bool(successful) and all(r["output_finite"] for r in successful),
            task_outcomes=stats.get("goal_outcomes"), time_to_all_reached_s=stats.get("time_to_all_reached"),
            all_goals_reached=stats.get("goal_outcomes") == ["reached"] * config["n_goals"],
            terminal_ee_position_m=mpc.solver.ee_pos(plant_q[-1]) if mpc is not None and plant_q else None,
            physical_limits=physical_limits(model, np.asarray(plant_q), np.asarray(plant_v), np.asarray(plant_u))
                if model is not None and plant_q else {"checked": False},
            merit_description="native initial/final/min merit retained per call; includes penalties, not a separately evaluated trajectory objective")
        write_json(output / "summary.json", summary)
        print(json.dumps(jsonable(summary), allow_nan=False), flush=True)
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
