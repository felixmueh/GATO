#!/usr/bin/env python3
"""One-shot Tiago trajectory-optimization smoke case."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "python"))

from bsqp.config import TIAGO_RIGHT_START_CONFIGS
from bsqp.interface import BSQP


MODEL_PATH = REPO_ROOT / "gato" / "dynamics" / "tiago_right" / "tiago_right_arm.urdf"
DT = 0.03
N = 32
SOLVER_PARAMS = {
    "max_sqp_iters": 20,
    "kkt_tol": 1e-4,
    "max_pcg_iters": 1000,
    "pcg_tol": 1e-3,
    "solve_ratio": 1.0,
    "mu": 1.0,
    "q_cost": 160.0,
    "qd_cost": 2e-2,
    "u_cost": 1e-6,
    "N_cost": 800.0,
    "q_lim_cost": 0.01,
    "vel_lim_cost": 0.0,
    "ctrl_lim_cost": 0.0,
    "rho": 0.02,
}


TARGET_OFFSETS = {
    "hold": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "small-x": np.array([0.01, 0.0, 0.0], dtype=np.float32),
    "small-y": np.array([0.0, 0.01, 0.0], dtype=np.float32),
    "small-z": np.array([0.0, 0.0, 0.01], dtype=np.float32),
}


def tool_position(model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    torso_id = model.getFrameId("torso_lift_link")
    tool_id = model.getFrameId("arm_right_tool_link")
    return (data.oMf[torso_id].inverse() * data.oMf[tool_id]).translation.copy()


def gravity_compensation(model, data, q, qd):
    return pin.rnea(model, data, q.astype(float), qd.astype(float), np.zeros(model.nv)).astype(np.float32)


def static_warm_start(x, u, horizon, nx, nu):
    warm = np.zeros(horizon * (nx + nu) - nu, dtype=np.float32)
    for knot in range(horizon):
        base = knot * (nx + nu)
        warm[base : base + nx] = x
        if knot < horizon - 1:
            warm[base + nx : base + nx + nu] = u
    return warm.reshape(1, -1)


def reference_window(target, horizon):
    refs = np.zeros((horizon, 6), dtype=np.float32)
    refs[:, :3] = target.astype(np.float32)
    return refs.reshape(1, -1)


def stat_array(stats, key, dtype):
    return np.asarray(stats.get(key, np.array([])), dtype=dtype).reshape(-1)


def summarize_solver_stats(solver, solution, max_pcg_iters):
    pcg_iters = stat_array(solver.stats, "pcg_iters", np.int32)
    step_sizes = stat_array(solver.stats, "step_size", np.float32)
    final_merit = stat_array(solver.stats, "final_merit", np.float32)
    initial_merit = stat_array(solver.stats, "initial_merit", np.float32)
    sqp_iters = stat_array(solver.stats, "sqp_iters", np.int32)
    kkt_converged = stat_array(solver.stats, "kkt_converged", np.int32)

    return {
        "sqp_time_ms": float(solver.stats["sqp_time_us"]) / 1000.0,
        "sqp_iters": int(sqp_iters[0]) if sqp_iters.size else -1,
        "kkt_converged": int(kkt_converged[0]) if kkt_converged.size else 0,
        "initial_merit": float(initial_merit[0]) if initial_merit.size else float("nan"),
        "final_merit": float(final_merit[0]) if final_merit.size else float("nan"),
        "accepted_step_count": int(np.sum(step_sizes > 0.0)),
        "line_search_fail_count": int(np.sum(step_sizes < 0.0)),
        "linear_solves": int(pcg_iters.size),
        "mean_pcg_iters": float(np.mean(pcg_iters)) if pcg_iters.size else float("nan"),
        "p95_pcg_iters": float(np.quantile(pcg_iters, 0.95)) if pcg_iters.size else float("nan"),
        "max_pcg_iters": int(np.max(pcg_iters)) if pcg_iters.size else -1,
        "pcg_cap_hit_fraction": float(np.mean(pcg_iters >= max_pcg_iters)) if pcg_iters.size else float("nan"),
        "nonfinite_output": int(not np.isfinite(solution).all()),
        "pcg_iters": [int(v) for v in pcg_iters],
        "step_sizes": [float(v) for v in step_sizes],
    }


def run_case(args):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing Tiago model: {MODEL_PATH}")
    if args.case not in TARGET_OFFSETS:
        raise ValueError(f"Unknown case {args.case!r}")

    model = pin.buildModelFromUrdf(str(MODEL_PATH))
    data = model.createData()
    q = TIAGO_RIGHT_START_CONFIGS["comfortable"].astype(np.float32).copy()
    qd = np.zeros(model.nv, dtype=np.float32)
    x = np.concatenate([q, qd]).astype(np.float32)
    u_static = gravity_compensation(model, data, q, qd)

    current_tool = tool_position(model, data, q.astype(float)).astype(np.float32)
    target = current_tool + TARGET_OFFSETS[args.case]

    solver_params = {**SOLVER_PARAMS}
    for key in ("u_cost", "qd_cost", "q_lim_cost", "rho", "pcg_tol"):
        value = getattr(args, key)
        if value is not None:
            solver_params[key] = value
    if args.max_pcg_iters is not None:
        solver_params["max_pcg_iters"] = args.max_pcg_iters
    if args.max_sqp_iters is not None:
        solver_params["max_sqp_iters"] = args.max_sqp_iters
    solver = BSQP(
        str(MODEL_PATH),
        1,
        args.N,
        args.dt,
        plant_type="tiago_right",
        adapt_rho=True,
        **solver_params,
    )
    warm = static_warm_start(x, u_static, args.N, solver.nx, solver.nu)
    refs = reference_window(target, args.N)

    solver.reset_dual()
    solver.reset_rho()
    solution, _ = solver.solve(x.reshape(1, -1), refs, warm)
    summary = summarize_solver_stats(solver, solution, solver_params["max_pcg_iters"])

    first_u = solution[0, solver.nx : solver.nx + solver.nu].astype(np.float32)
    predicted_first_q = solution[0, : solver.nq].astype(np.float32)
    predicted_final_base = (args.N - 1) * (solver.nx + solver.nu)
    predicted_final_q = solution[0, predicted_final_base : predicted_final_base + solver.nq].astype(np.float32)
    final_tool = tool_position(model, data, predicted_final_q.astype(float)).astype(np.float32)

    report = {
        "case": args.case,
        "linear_solver": solver.linsys_solver,
        "N": args.N,
        "dt": args.dt,
        "model_path": str(MODEL_PATH),
        "solver_params": solver_params,
        "current_tool": [float(v) for v in current_tool],
        "target_tool": [float(v) for v in target],
        "initial_tool_error_m": float(np.linalg.norm(current_tool - target)),
        "final_predicted_tool": [float(v) for v in final_tool],
        "final_predicted_tool_error_m": float(np.linalg.norm(final_tool - target)),
        "gravity_comp_control_norm": float(np.linalg.norm(u_static)),
        "gravity_comp_control_linf": float(np.max(np.abs(u_static))),
        "first_control_norm": float(np.linalg.norm(first_u)),
        "first_control_linf": float(np.max(np.abs(first_u))),
        "first_state_delta_norm": float(np.linalg.norm(predicted_first_q - q)),
        "summary": summary,
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(TARGET_OFFSETS), default="hold")
    parser.add_argument("--N", type=int, default=N)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--u-cost", type=float, default=None)
    parser.add_argument("--qd-cost", type=float, default=None)
    parser.add_argument("--q-lim-cost", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--pcg-tol", type=float, default=None)
    parser.add_argument("--max-pcg-iters", type=int, default=None)
    parser.add_argument("--max-sqp-iters", type=int, default=None)
    return parser.parse_args()


def main():
    run_case(parse_args())


if __name__ == "__main__":
    main()
