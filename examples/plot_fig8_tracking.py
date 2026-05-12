import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "python"))

from bsqp.common import initialize_warm_start, rk4
from bsqp.config import IIWA14_START_CONFIGS, TIAGO_RIGHT_START_CONFIGS
from bsqp.interface import BSQP


OUTPUT_DIR = REPO_ROOT / "test-artifacts" / "fig8_tracking"
N = 16
DT = 0.03
SIM_DT = 0.003
SIM_TIME = 5.0
FIG8_PERIOD = 4.0
REJECT_FALLBACK_DAMPING = 5.0
PLANT_CONFIGS = {
    "iiwa14": {
        "model_path": REPO_ROOT / "examples" / "iiwa_description" / "iiwa14.urdf",
        "q_start": IIWA14_START_CONFIGS["home"].astype(np.float32),
        "amplitude": np.array([0.045, 0.0, 0.035], dtype=np.float32),
        "fresh_warm_start": False,
        "reset_dual_each_solve": False,
        "solver": {
            "max_sqp_iters": 5,
            "kkt_tol": 1e-4,
            "max_pcg_iters": 160,
            "pcg_tol": 1e-4,
            "solve_ratio": 1.0,
            "mu": 1.0,
            "q_cost": 20.0,
            "qd_cost": 5e-2,
            "u_cost": 1e-6,
            "N_cost": 400.0,
            "q_lim_cost": 0.01,
            "vel_lim_cost": 0.0,
            "ctrl_lim_cost": 0.0,
            "rho": 0.02,
        },
    },
    "tiago_right": {
        "model_path": REPO_ROOT / "gato" / "dynamics" / "tiago_right" / "tiago_right_arm.urdf",
        "q_start": TIAGO_RIGHT_START_CONFIGS["comfortable"].astype(np.float32),
        "amplitude": np.array([0.040, -0.055, 0.025], dtype=np.float32),
        "fresh_warm_start": True,
        "reset_dual_each_solve": True,
        "solver": {
            "max_sqp_iters": 5,
            "kkt_tol": 1e-4,
            "max_pcg_iters": 160,
            "pcg_tol": 1e-4,
            "solve_ratio": 1.0,
            "mu": 1.0,
            "q_cost": 80.0,
            "qd_cost": 1e-1,
            "u_cost": 1e-6,
            "N_cost": 400.0,
            "q_lim_cost": 0.01,
            "vel_lim_cost": 0.0,
            "ctrl_lim_cost": 0.0,
            "rho": 0.02,
        },
    },
}


def tool_position(model, data, q, plant_type):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    if plant_type == "tiago_right":
        torso_id = model.getFrameId("torso_lift_link")
        tool_id = model.getFrameId("arm_right_tool_link")
        return (data.oMf[torso_id].inverse() * data.oMf[tool_id]).translation.copy()
    return data.oMi[model.njoints - 1].translation.copy()


def figure8_position(center, amplitude, t):
    phase = 2.0 * np.pi * t / FIG8_PERIOD
    return center + np.array(
        [
            amplitude[0] * np.sin(phase),
            amplitude[1] * np.sin(phase),
            amplitude[2] * np.sin(2.0 * phase),
        ],
        dtype=np.float32,
    )


def reference_window(center, amplitude, start_time):
    points = np.zeros((N, 6), dtype=np.float32)
    for knot in range(N):
        points[knot, :3] = figure8_position(center, amplitude, start_time + (knot + 1) * DT)
    return points.reshape(1, -1)


def shift_warm_start(solution, x_current, nx, nu):
    shifted = np.zeros_like(solution)
    flat = solution[0]
    for knot in range(N):
        dst = knot * (nx + nu)
        src = min(knot + 1, N - 1) * (nx + nu)
        shifted[0, dst:dst + nx] = flat[src:src + nx]
        if knot < N - 1:
            u_src = min(knot + 1, N - 2) * (nx + nu) + nx
            shifted[0, dst + nx:dst + nx + nu] = flat[u_src:u_src + nu]
    shifted[0, :nx] = x_current
    return shifted.astype(np.float32)


def gravity_compensation(model, data, q, qd):
    qdd_desired = np.clip(-REJECT_FALLBACK_DAMPING * qd.astype(float), -20.0, 20.0)
    return pin.rnea(model, data, q.astype(float), qd.astype(float), qdd_desired).astype(np.float32)


def stack_solver_stat_rows(rows, fill_value, dtype):
    if not rows:
        return np.empty((0, 0), dtype=dtype)

    width = max(row.size for row in rows)
    stacked = np.full((len(rows), width), fill_value, dtype=dtype)
    for row_index, row in enumerate(rows):
        stacked[row_index, :row.size] = row
    return stacked


def run_tracking(plant_type):
    cfg = PLANT_CONFIGS[plant_type]
    model = pin.buildModelFromUrdf(str(cfg["model_path"]))
    data = model.createData()
    q = cfg["q_start"].copy()
    qd = np.zeros(model.nv, dtype=np.float32)
    x = np.concatenate([q, qd]).astype(np.float32)
    center = tool_position(model, data, q, plant_type).astype(np.float32)

    solver = BSQP(
        str(cfg["model_path"]),
        1,
        N,
        DT,
        plant_type=plant_type,
        adapt_rho=True,
        **cfg["solver"],
    )
    warm_start = initialize_warm_start(x, N, solver.nx, solver.nu).reshape(1, -1).astype(np.float32)

    actual = []
    desired = []
    timestamps = []
    solve_times_us = []
    pcg_iters = []
    rejected_solves = 0
    total_time = 0.0

    while total_time < SIM_TIME:
        current_ref = reference_window(center, cfg["amplitude"], total_time)
        desired_now = figure8_position(center, cfg["amplitude"], total_time)
        actual_now = tool_position(model, data, q, plant_type)
        actual.append(actual_now)
        desired.append(desired_now)
        timestamps.append(total_time)

        solver.reset_rho()
        if cfg["reset_dual_each_solve"]:
            solver.reset_dual()
        if cfg["fresh_warm_start"]:
            warm_start = initialize_warm_start(x, N, solver.nx, solver.nu).reshape(1, -1).astype(np.float32)
        solve_start = time.perf_counter()
        solution, _ = solver.solve(x.reshape(1, -1), current_ref, warm_start)
        solve_times_us.append(float(solver.stats["sqp_time_us"]))
        pcg_iters.append(np.asarray(solver.stats["pcg_iters"], dtype=np.int32).reshape(-1))
        step_sizes = np.asarray(solver.stats["step_size"], dtype=np.float32).reshape(-1)
        if np.any(step_sizes > 0.0):
            u = solution[0, solver.nx:solver.nx + solver.nu].astype(np.float32)
        else:
            rejected_solves += 1
            u = gravity_compensation(model, data, q, qd)
        _ = time.perf_counter() - solve_start

        for _ in range(int(DT / SIM_DT)):
            q, qd = rk4(model, data, q.astype(float), qd.astype(float), u.astype(float), SIM_DT)
        x = np.concatenate([q, qd]).astype(np.float32)
        if not np.isfinite(x).all():
            raise RuntimeError(f"{plant_type} figure-8 tracking diverged")

        warm_start = shift_warm_start(solution, x, solver.nx, solver.nu)
        total_time += DT

    actual = np.asarray(actual, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    errors = np.linalg.norm(actual - desired, axis=1)
    return {
        "plant_type": plant_type,
        "actual": actual,
        "desired": desired,
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "errors": errors,
        "solve_times_us": np.asarray(solve_times_us, dtype=np.float64),
        "pcg_iters": stack_solver_stat_rows(pcg_iters, -1, np.int32),
        "rejected_solves": rejected_solves,
    }


def save_plot(stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plant = stats["plant_type"]
    actual = stats["actual"]
    desired = stats["desired"]
    errors = stats["errors"]
    timestamps = stats["timestamps"]

    fig = plt.figure(figsize=(11.0, 8.0))
    grid = fig.add_gridspec(2, 2)
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xz = fig.add_subplot(grid[0, 1])
    ax_err = fig.add_subplot(grid[1, 1])

    ax_3d.plot(desired[:, 0], desired[:, 1], desired[:, 2], linestyle=":", color="#747474", linewidth=1.6, label="desired")
    ax_3d.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="#00693E", linewidth=1.8, label="actual")
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("z [m]")
    ax_3d.set_title("3D tool path")
    ax_3d.legend(loc="best")

    ax_xz.plot(desired[:, 0], desired[:, 2], linestyle=":", color="#747474", linewidth=1.6, label="desired")
    ax_xz.plot(actual[:, 0], actual[:, 2], color="#00693E", linewidth=1.8, label="actual")
    ax_xz.set_xlabel("x [m]")
    ax_xz.set_ylabel("z [m]")
    ax_xz.set_title("XZ projection")
    ax_xz.set_aspect("equal", adjustable="box")
    ax_xz.grid(True, alpha=0.3)
    ax_xz.legend(loc="best")

    ax_err.plot(timestamps, errors, color="#C90016", linewidth=1.5)
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("tracking error [m]")
    ax_err.set_title("Tracking error")
    ax_err.grid(True, alpha=0.3)

    fig.suptitle(f"{plant} figure-8 tracking | mean error {np.mean(errors):.4f} m")
    fig.tight_layout()
    out = OUTPUT_DIR / f"{plant}_fig8_tracking.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    fig_ts, axes = plt.subplots(4, 1, figsize=(10.5, 8.0), sharex=True)
    axis_names = ["x", "y", "z"]
    for axis_index, axis_name in enumerate(axis_names):
        axes[axis_index].plot(timestamps, desired[:, axis_index], linestyle=":", color="#747474", linewidth=1.4, label="desired")
        axes[axis_index].plot(timestamps, actual[:, axis_index], color="#00693E", linewidth=1.5, label="actual")
        axes[axis_index].set_ylabel(f"{axis_name} [m]")
        axes[axis_index].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[3].plot(timestamps, errors, color="#C90016", linewidth=1.5)
    axes[3].set_ylabel("error [m]")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(True, alpha=0.3)
    fig_ts.suptitle(f"{plant} figure-8 tracking coordinates")
    fig_ts.tight_layout()
    ts_out = OUTPUT_DIR / f"{plant}_fig8_tracking_timeseries.png"
    fig_ts.savefig(ts_out, dpi=160)
    plt.close(fig_ts)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run and plot time-indexed figure-8 tracking.")
    parser.add_argument("--plant", choices=sorted(PLANT_CONFIGS), required=True)
    args = parser.parse_args()

    stats = run_tracking(args.plant)
    plot_path = save_plot(stats)
    valid_pcg = stats["pcg_iters"].reshape(-1)
    print(f"{args.plant} figure-8 tracking")
    print(f"samples: {stats['timestamps'].size}")
    print(f"mean_error: {np.mean(stats['errors']):.6f}m")
    print(f"max_error: {np.max(stats['errors']):.6f}m")
    print(f"final_error: {stats['errors'][-1]:.6f}m")
    print(f"mean_sqp_time: {np.mean(stats['solve_times_us']) / 1000.0:.3f}ms")
    print(f"rejected_solves: {stats['rejected_solves']}")
    print(f"mean_pcg_iters: {np.mean(valid_pcg):.2f}")
    print(f"pcg_max_iter_fraction: {np.mean(valid_pcg >= PLANT_CONFIGS[args.plant]['solver']['max_pcg_iters']):.3f}")
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
