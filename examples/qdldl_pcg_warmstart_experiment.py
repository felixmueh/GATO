#!/usr/bin/env python3
"""Capture QDLDL Tiago solves and replay them as PCG warm starts."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "python"))
sys.path.append(str(REPO_ROOT / "examples"))

from bsqp.common import initialize_warm_start, rk4
from bsqp.interface import BSQP
from large_easy_tracking import (
    BASE_SOLVER_PARAMS,
    DT,
    N,
    OUTPUT_ROOT as LARGE_EASY_OUTPUT_ROOT,
    PLANTS,
    SEGMENT_TIME,
    SIM_DT,
    end_effector_position,
    generate_reference,
    gravity_compensation,
    reference_window,
)


OUTPUT_ROOT = REPO_ROOT / "test-artifacts" / "qdldl_warmstart"


def solver_params_for(plant):
    return {**BASE_SOLVER_PARAMS, **PLANTS[plant]["solver"]}


def make_reference_context(plant, dt, segment_time):
    config = PLANTS[plant]
    model_path = config["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model for {plant}: {model_path}")

    model = pin.buildModelFromUrdf(str(model_path))
    data = model.createData()
    reference, joint_reference, waypoint_xyz = generate_reference(
        model,
        data,
        config["waypoints"],
        config["frame"],
        dt,
        segment_time,
    )
    return config, model, data, reference, joint_reference, waypoint_xyz


def stat_array(stats, key, dtype):
    return np.asarray(stats.get(key, np.array([])), dtype=dtype).reshape(-1)


def first_or_nan(values):
    values = np.asarray(values).reshape(-1)
    return float(values[0]) if values.size else float("nan")


def collect_solve_record(index, solver, solution, qdldl_solution=None):
    pcg_iters = stat_array(solver.stats, "pcg_iters", np.int32)
    step_sizes = stat_array(solver.stats, "step_size", np.float32)
    final_merit = stat_array(solver.stats, "final_merit", np.float32)
    initial_merit = stat_array(solver.stats, "initial_merit", np.float32)
    kkt_converged = stat_array(solver.stats, "kkt_converged", np.int32)
    sqp_iters = stat_array(solver.stats, "sqp_iters", np.int32)
    pcg_times_us = stat_array(solver.stats, "pcg_times_us", np.float32)

    record = {
        "tick": int(index),
        "sqp_time_us": float(solver.stats["sqp_time_us"]),
        "sqp_iters": int(sqp_iters[0]) if sqp_iters.size else -1,
        "kkt_converged": int(kkt_converged[0]) if kkt_converged.size else 0,
        "initial_merit": first_or_nan(initial_merit),
        "final_merit": first_or_nan(final_merit),
        "accepted_step_count": int(np.sum(step_sizes > 0.0)),
        "line_search_fail_count": int(np.sum(step_sizes < 0.0)),
        "max_pcg_iters_observed": int(np.max(pcg_iters)) if pcg_iters.size else -1,
        "mean_pcg_iters_observed": float(np.mean(pcg_iters)) if pcg_iters.size else float("nan"),
        "total_linear_solve_time_us": float(np.sum(pcg_times_us)) if pcg_times_us.size else float("nan"),
        "nonfinite_output": int(not np.isfinite(solution).all()),
        "pcg_iters_raw": [int(v) for v in pcg_iters],
    }
    if qdldl_solution is not None:
        diff = solution.astype(np.float64) - qdldl_solution.astype(np.float64)
        record["solution_l2_vs_qdldl"] = float(np.linalg.norm(diff))
        record["solution_linf_vs_qdldl"] = float(np.max(np.abs(diff)))
    return record


def summarize_records(records, max_pcg_iters):
    if not records:
        return {}

    max_iters = np.asarray([r["max_pcg_iters_observed"] for r in records], dtype=np.float64)
    raw_iters = [np.asarray(r.get("pcg_iters_raw", []), dtype=np.float64).reshape(-1) for r in records]
    raw_iters = np.concatenate([v for v in raw_iters if v.size]) if any(v.size for v in raw_iters) else max_iters
    sqp_times = np.asarray([r["sqp_time_us"] for r in records], dtype=np.float64)
    final_merit = np.asarray([r["final_merit"] for r in records], dtype=np.float64)
    accepted = np.asarray([r["accepted_step_count"] > 0 for r in records], dtype=bool)
    kkt = np.asarray([r["kkt_converged"] for r in records], dtype=np.int32)
    nonfinite = np.asarray([r["nonfinite_output"] for r in records], dtype=np.int32)

    summary = {
        "subproblems": int(len(records)),
        "linear_solves": int(raw_iters.size),
        "kkt_converged_fraction": float(np.mean(kkt > 0)),
        "accepted_step_fraction": float(np.mean(accepted)),
        "mean_pcg_iters": float(np.mean(raw_iters)),
        "p95_pcg_iters": float(np.quantile(raw_iters, 0.95)),
        "max_pcg_iters": int(np.max(raw_iters)),
        "pcg_cap_hit_fraction": float(np.mean(raw_iters >= max_pcg_iters)),
        "mean_subproblem_max_pcg_iters": float(np.mean(max_iters)),
        "subproblem_pcg_cap_hit_fraction": float(np.mean(max_iters >= max_pcg_iters)),
        "mean_sqp_time_ms": float(np.mean(sqp_times) / 1000.0),
        "p95_sqp_time_ms": float(np.quantile(sqp_times, 0.95) / 1000.0),
        "max_sqp_time_ms": float(np.max(sqp_times) / 1000.0),
        "mean_final_merit": float(np.mean(final_merit)),
        "nonfinite_output_count": int(np.sum(nonfinite)),
    }

    if "solution_l2_vs_qdldl" in records[0]:
        l2 = np.asarray([r["solution_l2_vs_qdldl"] for r in records], dtype=np.float64)
        linf = np.asarray([r["solution_linf_vs_qdldl"] for r in records], dtype=np.float64)
        summary.update(
            {
                "mean_solution_l2_vs_qdldl": float(np.mean(l2)),
                "p95_solution_l2_vs_qdldl": float(np.quantile(l2, 0.95)),
                "max_solution_l2_vs_qdldl": float(np.max(l2)),
                "mean_solution_linf_vs_qdldl": float(np.mean(linf)),
                "max_solution_linf_vs_qdldl": float(np.max(linf)),
            }
        )
    return summary


def write_records_csv(path, records):
    if not records:
        return

    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def capture(args):
    config, model, data, reference, joint_reference, waypoint_xyz = make_reference_context(
        args.plant,
        args.dt,
        args.segment_time,
    )
    solver_params = solver_params_for(args.plant)
    solver = BSQP(
        str(config["model_path"]),
        1,
        args.N,
        args.dt,
        plant_type=args.plant,
        adapt_rho=True,
        **solver_params,
    )
    if solver.linsys_solver != "QDLDL":
        raise RuntimeError(f"capture requires QDLDL module, active LINSYS_SOLVER={solver.linsys_solver}")

    q = config["waypoints"][0].copy()
    qd = np.zeros(model.nv, dtype=np.float32)
    x = np.concatenate([q, qd]).astype(np.float32)
    substeps = int(round(args.dt / args.sim_dt))
    num_ticks = reference.shape[0] if args.max_ticks is None else min(args.max_ticks, reference.shape[0])

    x_before = []
    reference_windows = []
    warm_start_in = []
    qdldl_solutions = []
    records = []
    rejected_solves = 0

    for index in range(num_ticks):
        solver.reset_dual()
        solver.reset_rho()
        warm_start = initialize_warm_start(x, args.N, solver.nx, solver.nu).reshape(1, -1).astype(np.float32)
        refs = reference_window(reference, index, args.N).astype(np.float32)
        solution, _ = solver.solve(x.reshape(1, -1), refs, warm_start)

        x_before.append(x.copy())
        reference_windows.append(refs.reshape(-1).copy())
        warm_start_in.append(warm_start.reshape(-1).copy())
        qdldl_solutions.append(solution.reshape(-1).copy())
        records.append(collect_solve_record(index, solver, solution))

        step_sizes = stat_array(solver.stats, "step_size", np.float32)
        if np.any(step_sizes > 0.0):
            u = solution[0, solver.nx : solver.nx + solver.nu].astype(np.float32)
        else:
            rejected_solves += 1
            u = gravity_compensation(model, data, q, qd)

        for _ in range(substeps):
            q, qd = rk4(model, data, q.astype(float), qd.astype(float), u.astype(float), args.sim_dt)
        x = np.concatenate([q, qd]).astype(np.float32)
        if not np.isfinite(x).all():
            raise RuntimeError(f"{args.plant} QDLDL capture diverged at step {index}")

    output_dir = OUTPUT_ROOT / args.plant
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = output_dir / "qdldl_capture.npz"
    np.savez_compressed(
        capture_path,
        x_before=np.asarray(x_before, dtype=np.float32),
        reference_window=np.asarray(reference_windows, dtype=np.float32),
        warm_start_in=np.asarray(warm_start_in, dtype=np.float32),
        qdldl_solution=np.asarray(qdldl_solutions, dtype=np.float32),
        reference=reference.astype(np.float32),
        joint_reference=joint_reference.astype(np.float32),
        waypoint_xyz=waypoint_xyz.astype(np.float32),
        plant=np.asarray(args.plant),
        N=np.asarray(args.N, dtype=np.int32),
        dt=np.asarray(args.dt, dtype=np.float32),
        sim_dt=np.asarray(args.sim_dt, dtype=np.float32),
        segment_time=np.asarray(args.segment_time, dtype=np.float32),
        model_path=np.asarray(str(config["model_path"])),
        solver_params=np.asarray(json.dumps(solver_params, sort_keys=True)),
    )

    summary = summarize_records(records, solver_params["max_pcg_iters"])
    summary["rejected_solves"] = int(rejected_solves)
    summary_payload = {
        "mode": "capture",
        "plant": args.plant,
        "linsys_solver": solver.linsys_solver,
        "N": args.N,
        "dt": args.dt,
        "sim_dt": args.sim_dt,
        "segment_time": args.segment_time,
        "solver_params": solver_params,
        "summary": summary,
        "capture_path": str(capture_path),
        "large_easy_output_root": str(LARGE_EASY_OUTPUT_ROOT),
    }
    write_records_csv(output_dir / "qdldl_capture_per_tick.csv", records)
    write_summary(output_dir / "qdldl_capture_summary.json", summary_payload)

    print_report(summary_payload)


def perturb_warm_start(warm_start, sigma, nx, seed):
    if sigma == 0.0:
        return warm_start.copy()

    rng = np.random.default_rng(seed)
    perturbed = warm_start.copy()
    mask = np.isfinite(perturbed)
    mask[:nx] = False
    noise = rng.normal(loc=0.0, scale=sigma, size=perturbed.shape).astype(np.float32)
    perturbed[mask] = (perturbed[mask] + noise[mask]).astype(np.float32)
    return perturbed


def replay(args):
    capture_path = Path(args.capture)
    if not capture_path.exists():
        raise FileNotFoundError(f"Missing capture file: {capture_path}")

    capture_data = np.load(capture_path, allow_pickle=False)
    plant = str(capture_data["plant"])
    N_capture = int(capture_data["N"])
    if args.plant and args.plant != plant:
        raise RuntimeError(f"capture plant is {plant}, requested {args.plant}")
    if args.N != N_capture:
        raise RuntimeError(f"capture N is {N_capture}, requested {args.N}")

    config = PLANTS[plant]
    solver_params = solver_params_for(plant)
    solver = BSQP(
        str(config["model_path"]),
        1,
        args.N,
        float(capture_data["dt"]),
        plant_type=plant,
        adapt_rho=True,
        **solver_params,
    )
    if solver.linsys_solver != "PCG":
        raise RuntimeError(f"replay requires PCG module, active LINSYS_SOLVER={solver.linsys_solver}")

    x_before = np.asarray(capture_data["x_before"], dtype=np.float32)
    reference_windows = np.asarray(capture_data["reference_window"], dtype=np.float32)
    qdldl_solutions = np.asarray(capture_data["qdldl_solution"], dtype=np.float32)
    num_ticks = x_before.shape[0] if args.max_ticks is None else min(args.max_ticks, x_before.shape[0])

    records = []
    pcg_solutions = []
    for index in range(num_ticks):
        solver.reset_dual()
        solver.reset_rho()
        warm_start = perturb_warm_start(
            qdldl_solutions[index],
            args.sigma,
            solver.nx,
            args.seed + index,
        ).reshape(1, -1)
        solution, _ = solver.solve(
            x_before[index].reshape(1, -1),
            reference_windows[index].reshape(1, -1),
            warm_start,
        )
        pcg_solutions.append(solution.reshape(-1).copy())
        records.append(collect_solve_record(index, solver, solution, qdldl_solutions[index].reshape(1, -1)))

    output_dir = OUTPUT_ROOT / plant / f"pcg_replay_sigma_{args.sigma:g}".replace("-", "m").replace(".", "p")
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "pcg_replay.npz"
    np.savez_compressed(
        replay_path,
        pcg_solution=np.asarray(pcg_solutions, dtype=np.float32),
        qdldl_solution=qdldl_solutions[:num_ticks],
        x_before=x_before[:num_ticks],
        reference_window=reference_windows[:num_ticks],
        sigma=np.asarray(args.sigma, dtype=np.float32),
        seed=np.asarray(args.seed, dtype=np.int32),
    )

    summary = summarize_records(records, solver_params["max_pcg_iters"])
    summary_payload = {
        "mode": "replay",
        "plant": plant,
        "linsys_solver": solver.linsys_solver,
        "N": args.N,
        "dt": float(capture_data["dt"]),
        "sigma": args.sigma,
        "seed": args.seed,
        "solver_params": solver_params,
        "summary": summary,
        "capture_path": str(capture_path),
        "replay_path": str(replay_path),
    }
    write_records_csv(output_dir / "pcg_replay_per_tick.csv", records)
    write_summary(output_dir / "pcg_replay_summary.json", summary_payload)

    print_report(summary_payload)


def print_report(payload):
    summary = payload["summary"]
    print(f"mode: {payload['mode']}")
    print(f"plant: {payload['plant']}")
    print(f"linear_solver: {payload['linsys_solver']}")
    print(f"N: {payload['N']}")
    if payload["mode"] == "replay":
        print(f"sigma: {payload['sigma']}")
    print(f"subproblems: {summary.get('subproblems', 0)}")
    print(f"kkt_converged_fraction: {summary.get('kkt_converged_fraction', float('nan')):.3f}")
    print(f"accepted_step_fraction: {summary.get('accepted_step_fraction', float('nan')):.3f}")
    print(f"mean_pcg_iters: {summary.get('mean_pcg_iters', float('nan')):.2f}")
    print(f"p95_pcg_iters: {summary.get('p95_pcg_iters', float('nan')):.2f}")
    print(f"max_pcg_iters: {summary.get('max_pcg_iters', -1)}")
    print(f"pcg_cap_hit_fraction: {summary.get('pcg_cap_hit_fraction', float('nan')):.3f}")
    print(f"mean_sqp_time: {summary.get('mean_sqp_time_ms', float('nan')):.3f}ms")
    print(f"p95_sqp_time: {summary.get('p95_sqp_time_ms', float('nan')):.3f}ms")
    print(f"max_sqp_time: {summary.get('max_sqp_time_ms', float('nan')):.3f}ms")
    print(f"mean_final_merit: {summary.get('mean_final_merit', float('nan')):.6g}")
    print(f"nonfinite_output_count: {summary.get('nonfinite_output_count', -1)}")
    if payload["mode"] == "replay":
        print(f"mean_solution_l2_vs_qdldl: {summary.get('mean_solution_l2_vs_qdldl', float('nan')):.6g}")
        print(f"max_solution_linf_vs_qdldl: {summary.get('max_solution_linf_vs_qdldl', float('nan')):.6g}")
        print(f"replay_path: {payload['replay_path']}")
    else:
        print(f"rejected_solves: {summary.get('rejected_solves', -1)}")
        print(f"capture_path: {payload['capture_path']}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    capture_parser = subparsers.add_parser("capture", help="Run QDLDL closed loop and save warm-start handoff data.")
    capture_parser.add_argument("--plant", choices=sorted(PLANTS), default="tiago_right")
    capture_parser.add_argument("--N", type=int, default=N)
    capture_parser.add_argument("--dt", type=float, default=DT)
    capture_parser.add_argument("--sim-dt", type=float, default=SIM_DT)
    capture_parser.add_argument("--segment-time", type=float, default=SEGMENT_TIME)
    capture_parser.add_argument("--max-ticks", type=int, default=None)
    capture_parser.set_defaults(func=capture)

    replay_parser = subparsers.add_parser("replay", help="Replay captured QDLDL subproblems with PCG.")
    replay_parser.add_argument("--plant", choices=sorted(PLANTS), default=None)
    replay_parser.add_argument("--N", type=int, default=N)
    replay_parser.add_argument("--capture", default=str(OUTPUT_ROOT / "tiago_right" / "qdldl_capture.npz"))
    replay_parser.add_argument("--sigma", type=float, default=0.0)
    replay_parser.add_argument("--seed", type=int, default=1)
    replay_parser.add_argument("--max-ticks", type=int, default=None)
    replay_parser.set_defaults(func=replay)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
