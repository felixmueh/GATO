import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "python"))

from bsqp.common import initialize_warm_start, rk4
from bsqp.config import TIAGO_RIGHT_START_CONFIGS
from bsqp.interface import BSQP


MODEL_PATH = REPO_ROOT / "gato" / "dynamics" / "tiago_right" / "tiago_right_arm.urdf"
OUTPUT_DIR = REPO_ROOT / "test-artifacts" / "tiago_large_tracking"
N = 16
DT = 0.03
SIM_DT = 0.003
SEGMENT_TIME = 5.0
REJECT_FALLBACK_DAMPING = 5.0
WORKSPACE_JOINT_WAYPOINTS = np.array(
    [
        TIAGO_RIGHT_START_CONFIGS["comfortable"],
        [-3.54884509, -0.69177985, 1.23300691, -0.12537238, 0.56022805, 0.80190422, -2.15310950],
        [-4.46710114, -2.18088622, 0.65142492, -0.52816790, 2.55711817, 0.66662889, 2.09391246],
        [-3.18236653, -2.11662276, -1.50345407, -0.07159287, 1.67217184, 0.75557566, -1.87291204],
        [-0.68219534, -2.29215443, 2.36679395, 0.01532510, 1.75894834, 1.02114498, -1.81019345],
        [-1.82603380, -1.45393528, -0.32920454, -0.40065100, 1.28555480, 0.84114169, 1.03900327],
        TIAGO_RIGHT_START_CONFIGS["comfortable"],
    ],
    dtype=np.float32,
)
ARM_LINK_NAMES = [
    "torso_lift_link",
    "arm_right_1_link",
    "arm_right_2_link",
    "arm_right_3_link",
    "arm_right_4_link",
    "arm_right_5_link",
    "arm_right_6_link",
    "arm_right_7_link",
    "arm_right_tool_link",
]
SOLVER_PARAMS = {
    "max_sqp_iters": 5,
    "kkt_tol": 1e-4,
    "max_pcg_iters": 160,
    "pcg_tol": 1e-3,
    "solve_ratio": 1.0,
    "mu": 1.0,
    "q_cost": 320.0,
    "qd_cost": 2e-2,
    "u_cost": 1e-6,
    "N_cost": 1600.0,
    "q_lim_cost": 0.01,
    "vel_lim_cost": 0.0,
    "ctrl_lim_cost": 0.0,
    "rho": 0.02,
}


def tool_position(model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    torso_id = model.getFrameId("torso_lift_link")
    tool_id = model.getFrameId("arm_right_tool_link")
    return (data.oMf[torso_id].inverse() * data.oMf[tool_id]).translation.copy()


def arm_link_positions(model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    torso_id = model.getFrameId("torso_lift_link")
    torso_inv = data.oMf[torso_id].inverse()
    points = []
    for link_name in ARM_LINK_NAMES:
        frame_id = model.getFrameId(link_name)
        points.append((torso_inv * data.oMf[frame_id]).translation.copy())
    return np.asarray(points, dtype=np.float64)


def smoothstep(alpha):
    return alpha * alpha * (3.0 - 2.0 * alpha)


def generate_reference_trajectory(model, data):
    waypoints = np.asarray(
        [tool_position(model, data, q.astype(float)) for q in WORKSPACE_JOINT_WAYPOINTS],
        dtype=np.float32,
    )
    samples_per_segment = int(SEGMENT_TIME / DT)
    samples = []
    for q_start, q_end in zip(WORKSPACE_JOINT_WAYPOINTS[:-1], WORKSPACE_JOINT_WAYPOINTS[1:]):
        for sample_idx in range(samples_per_segment):
            alpha = (sample_idx + 1) / samples_per_segment
            q = (1.0 - smoothstep(alpha)) * q_start + smoothstep(alpha) * q_end
            samples.append(tool_position(model, data, q.astype(float)))
    return np.asarray(samples, dtype=np.float32), waypoints


def reference_window(reference, index):
    points = np.zeros((N, 6), dtype=np.float32)
    for knot in range(N):
        points[knot, :3] = reference[min(index + knot + 1, reference.shape[0] - 1)]
    return points.reshape(1, -1)


def solve_tool_ik(model, data, q_start, target, max_iters=180):
    q = q_start.astype(float).copy()
    tool_id = model.getFrameId("arm_right_tool_link")

    for _ in range(max_iters):
        current = tool_position(model, data, q)
        error = target - current
        if np.linalg.norm(error) < 1e-3:
            break

        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        jacobian = pin.computeFrameJacobian(model, data, q, tool_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 5e-5 * np.eye(3), error)
        step_norm = np.linalg.norm(dq)
        if step_norm > 0.15:
            dq *= 0.15 / step_norm
        q = pin.integrate(model, q, dq)
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    return q


def validate_reference_with_ik(model, data, q_start, reference):
    sample_indices = np.unique(np.linspace(0, reference.shape[0] - 1, 80).astype(np.int64))
    q = q_start.astype(float).copy()
    errors = []
    for index in sample_indices:
        q = solve_tool_ik(model, data, q, reference[index])
        errors.append(float(np.linalg.norm(tool_position(model, data, q) - reference[index])))
    errors = np.asarray(errors, dtype=np.float64)
    if np.max(errors) > 0.01:
        raise RuntimeError(f"Reference is not reliably Tiago-reachable: sampled IK max error {np.max(errors):.6f}m")
    return {"ik_mean_error": float(np.mean(errors)), "ik_max_error": float(np.max(errors))}


def gravity_compensation(model, data, q, qd):
    qdd_desired = np.clip(-REJECT_FALLBACK_DAMPING * qd.astype(float), -20.0, 20.0)
    return pin.rnea(model, data, q.astype(float), qd.astype(float), qdd_desired).astype(np.float32)


def stack_solver_stat_rows(rows, fill_value, dtype):
    if not rows:
        return np.empty((0, 0), dtype=dtype)

    width = max(row.size for row in rows)
    stacked = np.full((len(rows), width, ), fill_value, dtype=dtype)
    for row_index, row in enumerate(rows):
        stacked[row_index, :row.size] = row
    return stacked


def run_tracking():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing generated Tiago arm URDF: {MODEL_PATH}")

    model = pin.buildModelFromUrdf(str(MODEL_PATH))
    data = model.createData()
    q = WORKSPACE_JOINT_WAYPOINTS[0].copy()
    qd = np.zeros(model.nv, dtype=np.float32)
    x = np.concatenate([q, qd]).astype(np.float32)
    reference, waypoints = generate_reference_trajectory(model, data)
    ik_check = validate_reference_with_ik(model, data, q.astype(float), reference)

    solver = BSQP(str(MODEL_PATH), 1, N, DT, plant_type="tiago_right", adapt_rho=True, **SOLVER_PARAMS)

    actual = []
    arm_points = []
    desired = []
    timestamps = []
    solve_times_us = []
    pcg_iters = []
    rejected_solves = 0
    total_time = 0.0

    for index, desired_now in enumerate(reference):
        actual_now = tool_position(model, data, q)
        actual.append(actual_now)
        arm_points.append(arm_link_positions(model, data, q.astype(float)))
        desired.append(desired_now)
        timestamps.append(total_time)

        # The Python wrapper can reset solver state, but does not expose a way
        # to shift dual variables with the horizon. Carrying only the primal
        # warm start makes this Tiago path reject almost every step, so this
        # visual example uses independent MPC solves at each tick.
        solver.reset_dual()
        solver.reset_rho()
        warm_start = initialize_warm_start(x, N, solver.nx, solver.nu).reshape(1, -1).astype(np.float32)
        solution, _ = solver.solve(x.reshape(1, -1), reference_window(reference, index), warm_start)
        solve_times_us.append(float(solver.stats["sqp_time_us"]))
        pcg_iters.append(np.asarray(solver.stats["pcg_iters"], dtype=np.int32).reshape(-1))
        step_sizes = np.asarray(solver.stats["step_size"], dtype=np.float32).reshape(-1)
        if np.any(step_sizes > 0.0):
            u = solution[0, solver.nx:solver.nx + solver.nu].astype(np.float32)
        else:
            rejected_solves += 1
            u = gravity_compensation(model, data, q, qd)

        for _ in range(int(DT / SIM_DT)):
            q, qd = rk4(model, data, q.astype(float), qd.astype(float), u.astype(float), SIM_DT)
        x = np.concatenate([q, qd]).astype(np.float32)
        if not np.isfinite(x).all():
            raise RuntimeError("Tiago large trajectory tracking diverged")

        total_time += DT

    actual = np.asarray(actual, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    return {
        "actual": actual,
        "arm_points": np.asarray(arm_points, dtype=np.float64),
        "desired": desired,
        "waypoints": waypoints.astype(np.float64),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "errors": np.linalg.norm(actual - desired, axis=1),
        "solve_times_us": np.asarray(solve_times_us, dtype=np.float64),
        "pcg_iters": stack_solver_stat_rows(pcg_iters, -1, np.int32),
        "rejected_solves": rejected_solves,
        **ik_check,
    }


def save_artifacts(stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    actual = stats["actual"]
    desired = stats["desired"]
    waypoints = stats["waypoints"]
    timestamps = stats["timestamps"]
    errors = stats["errors"]
    arm_points = stats["arm_points"]

    np.savetxt(OUTPUT_DIR / "tiago_large_reference.traj", np.pad(desired, ((0, 0), (0, 3))), delimiter=",")
    np.savetxt(OUTPUT_DIR / "tiago_large_actual.csv", actual, delimiter=",")

    fig = plt.figure(figsize=(11.0, 8.0))
    grid = fig.add_gridspec(2, 2)
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xz = fig.add_subplot(grid[0, 1])
    ax_err = fig.add_subplot(grid[1, 1])

    ax_3d.plot(desired[:, 0], desired[:, 1], desired[:, 2], linestyle=":", color="#747474", linewidth=1.8, label="reference")
    ax_3d.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="#00693E", linewidth=1.8, label="actual")
    ax_3d.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], marker="*", s=100, color="#C90016", label="waypoints")
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("z [m]")
    ax_3d.set_title("Tiago workspace reach trajectory")
    ax_3d.legend(loc="best")

    ax_xz.plot(desired[:, 0], desired[:, 2], linestyle=":", color="#747474", linewidth=1.8, label="reference")
    ax_xz.plot(actual[:, 0], actual[:, 2], color="#00693E", linewidth=1.8, label="actual")
    ax_xz.scatter(waypoints[:, 0], waypoints[:, 2], marker="*", s=90, color="#C90016", label="waypoints")
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

    fig.suptitle(f"Tiago workspace reach tracking | mean error {np.mean(errors):.4f} m")
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "tiago_large_tracking.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    finite_ids = np.arange(actual.shape[0])
    frame_count = min(360, finite_ids.size)
    frame_ids = finite_ids[np.linspace(0, finite_ids.size - 1, frame_count).astype(np.int64)]
    fig_gif = plt.figure(figsize=(8.2, 7.0))
    ax = fig_gif.add_subplot(111, projection="3d")
    ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], linestyle=":", color="#747474", linewidth=1.5)
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], marker="*", s=90, color="#C90016", alpha=0.55)
    arm_line, = ax.plot([], [], [], "o-", color="#00693E", linewidth=3.0, markersize=4)
    trail_line, = ax.plot([], [], [], color="#003192", linewidth=1.5, alpha=0.8)
    target_marker, = ax.plot([], [], [], "x", color="#C90016", markersize=8)

    all_points = np.concatenate([arm_points.reshape(-1, 3), desired, waypoints], axis=0)
    center = np.mean(all_points, axis=0)
    span = float(np.max(np.ptp(all_points, axis=0)))
    radius = max(0.28, span * 0.60)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=24, azim=-54)
    fig_gif.tight_layout()

    def update(frame_index):
        idx = frame_ids[frame_index]
        points = arm_points[idx]
        arm_line.set_data(points[:, 0], points[:, 1])
        arm_line.set_3d_properties(points[:, 2])
        trail_line.set_data(actual[:idx + 1, 0], actual[:idx + 1, 1])
        trail_line.set_3d_properties(actual[:idx + 1, 2])
        target_marker.set_data([desired[idx, 0]], [desired[idx, 1]])
        target_marker.set_3d_properties([desired[idx, 2]])
        ax.set_title(f"Tiago workspace reach | t={timestamps[idx]:.2f}s | err={errors[idx]:.3f}m")
        return arm_line, trail_line, target_marker

    anim = animation.FuncAnimation(fig_gif, update, frames=frame_count, interval=50, blit=False)
    gif_path = OUTPUT_DIR / "tiago_large_tracking.gif"
    anim.save(gif_path, writer=animation.PillowWriter(fps=20))
    plt.close(fig_gif)
    return plot_path, gif_path


def main():
    stats = run_tracking()
    plot_path, gif_path = save_artifacts(stats)
    valid_pcg = stats["pcg_iters"][stats["pcg_iters"] >= 0]
    print("Tiago workspace reach trajectory tracking")
    print(f"samples: {stats['timestamps'].size}")
    print(f"sim_time: {stats['timestamps'][-1] + DT:.3f}s")
    print(f"ik_mean_error: {stats['ik_mean_error']:.6f}m")
    print(f"ik_max_error: {stats['ik_max_error']:.6f}m")
    print(f"mean_error: {np.mean(stats['errors']):.6f}m")
    print(f"p95_error: {np.quantile(stats['errors'], 0.95):.6f}m")
    print(f"max_error: {np.max(stats['errors']):.6f}m")
    print(f"final_error: {stats['errors'][-1]:.6f}m")
    print(f"mean_sqp_time: {np.mean(stats['solve_times_us']) / 1000.0:.3f}ms")
    print(f"rejected_solves: {stats['rejected_solves']}")
    print(f"mean_pcg_iters: {np.mean(valid_pcg):.2f}")
    print(f"pcg_max_iter_fraction: {np.mean(valid_pcg >= SOLVER_PARAMS['max_pcg_iters']):.3f}")
    print(f"plot: {plot_path}")
    print(f"gif: {gif_path}")


if __name__ == "__main__":
    main()
