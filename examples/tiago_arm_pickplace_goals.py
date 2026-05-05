import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "python"))

from bsqp.common import initialize_warm_start, rk4
from bsqp.config import TIAGO_RIGHT_START_CONFIGS
from bsqp.interface import BSQP


MODEL_PATH = REPO_ROOT / "gato" / "dynamics" / "tiago_right" / "tiago_right_arm.urdf"
OUTPUT_DIR = REPO_ROOT / "test-artifacts" / "tiago_pickplace"
N = 16
DT = 0.03
SIM_DT = 0.003
GOAL_THRESHOLD = 0.01
GOAL_DWELL_TIME = 0.3
GOAL_TIMEOUT = 20.0
GOAL_DISTANCE = 0.25
REFERENCE_MAX_STEP = GOAL_DISTANCE
REJECT_FALLBACK_DAMPING = 5.0
SENSIBLE_Q_LIMIT_CLEARANCE = 0.20
MAX_SQP_ITERS = 5
MAX_PCG_ITERS = 160
PCG_TOL = 1e-4
KKT_TOL = 1e-4
Q_COST = 20.0
QD_COST = 1e-1
U_COST = 1e-6
N_COST = 400.0
Q_LIM_COST = 0.01
RHO = 0.02
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


def pickplace_goals(ee0):
    # Use reachable directions from a moderate right-arm workspace loop, but
    # normalize each consecutive Cartesian segment to GOAL_DISTANCE.
    path_offsets = np.array(
        [
            [0.150271, -0.200214, 0.045349],
            [-0.039511, 0.033506, -0.025988],
            [0.149905, -0.155695, 0.036798],
            [-0.118381, 0.063711, -0.021699],
        ],
        dtype=np.float32,
    )

    offsets = []
    previous = np.zeros(3, dtype=np.float32)
    current = np.zeros(3, dtype=np.float32)
    for target in path_offsets:
        delta = target - previous
        current = current + delta / np.linalg.norm(delta) * GOAL_DISTANCE
        offsets.append(current.copy())
        previous = target
    offsets = np.asarray(offsets, dtype=np.float32)
    return ee0 + offsets


def segment_endpoint(current, goal, max_step=REFERENCE_MAX_STEP):
    delta = goal - current
    distance = float(np.linalg.norm(delta))
    if distance <= max_step:
        return goal
    return current + delta / distance * max_step


def segment_reference(current, goal, max_step=REFERENCE_MAX_STEP):
    segment_end = segment_endpoint(current, goal, max_step)

    points = np.zeros((N, 6), dtype=np.float32)
    for knot in range(N):
        alpha = (knot + 1) / N
        points[knot, :3] = current + alpha * (segment_end - current)
    return points.reshape(1, -1)


def solve_tool_ik(model, data, q_start, target, max_iters=100):
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


def validate_pickplace_circuit(model, data, q_start, goals):
    q = q_start.astype(float).copy()
    waypoints = [q.copy()]
    start_clearance = np.minimum(q - model.lowerPositionLimit, model.upperPositionLimit - q)
    worst_joint = int(np.argmin(start_clearance))
    min_clearance = float(start_clearance[worst_joint])

    for goal in goals:
        q = solve_tool_ik(model, data, q, goal, max_iters=200)
        error = float(np.linalg.norm(tool_position(model, data, q) - goal))
        if error > 3e-3:
            raise RuntimeError(f"Pick/place goal is not reliably reachable before rollout: IK error {error:.6f}m")
        clearance_to_lower = q - model.lowerPositionLimit
        clearance_to_upper = model.upperPositionLimit - q
        clearance = np.minimum(clearance_to_lower, clearance_to_upper)
        joint_idx = int(np.argmin(clearance))
        if float(clearance[joint_idx]) < min_clearance:
            min_clearance = float(clearance[joint_idx])
            worst_joint = joint_idx
        waypoints.append(q.copy())

    if min_clearance < SENSIBLE_Q_LIMIT_CLEARANCE:
        joint_name = model.names[worst_joint + 1] if worst_joint is not None else "unknown"
        raise RuntimeError(
            "Pick/place circuit is too close to Tiago joint limits before rollout: "
            f"minimum clearance {min_clearance:.3f} rad at {joint_name}; "
            f"required {SENSIBLE_Q_LIMIT_CLEARANCE:.3f} rad"
        )

    segment_lengths = [
        float(np.linalg.norm(pin.difference(model, waypoints[idx], waypoints[idx + 1])))
        for idx in range(len(waypoints) - 1)
    ]
    return {
        "min_q_limit_clearance": min_clearance,
        "max_joint_space_segment": max(segment_lengths) if segment_lengths else 0.0,
    }


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


def run_pickplace():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing generated Tiago arm URDF: {MODEL_PATH}")

    model = pin.buildModelFromUrdf(str(MODEL_PATH))
    data = model.createData()
    q = TIAGO_RIGHT_START_CONFIGS["comfortable"].astype(np.float32)
    qd = np.zeros(model.nv, dtype=np.float32)
    x = np.concatenate([q, qd]).astype(np.float32)
    ee0 = tool_position(model, data, q)
    goals = np.asarray(pickplace_goals(ee0), dtype=np.float32)
    circuit_check = validate_pickplace_circuit(model, data, q.astype(float), goals)

    solver_params = {
        "max_sqp_iters": MAX_SQP_ITERS,
        "kkt_tol": KKT_TOL,
        "max_pcg_iters": MAX_PCG_ITERS,
        "pcg_tol": PCG_TOL,
        "solve_ratio": 1.0,
        "mu": 1.0,
        "q_cost": Q_COST,
        "qd_cost": QD_COST,
        "u_cost": U_COST,
        "N_cost": N_COST,
        "q_lim_cost": Q_LIM_COST,
        "vel_lim_cost": 0.0,
        "ctrl_lim_cost": 0.0,
        "rho": RHO,
    }

    solver = BSQP(str(MODEL_PATH), 1, N, DT, plant_type="tiago_right", adapt_rho=True, **solver_params)

    actual = []
    arm_points = []
    distances = []
    timestamps = []
    active_goal_indices = []
    control_durations = []
    sqp_times_us = []
    sqp_iters = []
    pcg_iters = []
    rejected_solves = 0
    outcomes = ["not_reached"] * len(goals)
    total_time = 0.0

    for goal_index, goal in enumerate(goals):
        goal_start_time = total_time
        dwell_time = 0.0

        while total_time - goal_start_time < GOAL_TIMEOUT:
            current_ee = tool_position(model, data, q)
            distance = float(np.linalg.norm(current_ee - goal))
            velocity = float(np.linalg.norm(qd, ord=1))

            actual.append(current_ee)
            arm_points.append(arm_link_positions(model, data, q.astype(float)))
            distances.append(distance)
            timestamps.append(total_time)
            active_goal_indices.append(goal_index)

            if distance < GOAL_THRESHOLD and velocity < 20.0:
                dwell_time += DT
            else:
                dwell_time = 0.0

            if dwell_time >= GOAL_DWELL_TIME:
                outcomes[goal_index] = "reached"
                break

            solve_start = time.perf_counter()
            solver.reset_dual()
            solver.reset_rho()
            warm_start = initialize_warm_start(x, N, solver.nx, solver.nu).reshape(1, -1).astype(np.float32)
            solution, _ = solver.solve(x.reshape(1, -1), segment_reference(current_ee, goal), warm_start)
            control_duration = time.perf_counter() - solve_start
            control_durations.append(control_duration)
            sqp_times_us.append(float(solver.stats["sqp_time_us"]))
            sqp_iters.append(int(np.asarray(solver.stats["sqp_iters"]).reshape(-1)[0]))
            pcg_iters.append(np.asarray(solver.stats["pcg_iters"], dtype=np.int32).reshape(-1))
            u = solution[0, solver.nx:solver.nx + solver.nu].astype(np.float32)
            step_sizes = np.asarray(solver.stats["step_size"], dtype=np.float32).reshape(-1)
            if not np.any(step_sizes > 0.0):
                rejected_solves += 1
                u = gravity_compensation(model, data, q, qd)

            for _ in range(int(DT / SIM_DT)):
                q, qd = rk4(model, data, q.astype(float), qd.astype(float), u.astype(float), SIM_DT)
            x = np.concatenate([q, qd]).astype(np.float32)
            if not np.isfinite(x).all():
                outcomes[goal_index] = "diverged"
                break

            total_time += DT
        if outcomes[goal_index] == "not_reached":
            outcomes[goal_index] = "timeout"
        if outcomes[goal_index] == "diverged":
            break

    stats = {
        "actual": np.asarray(actual, dtype=np.float64),
        "arm_points": np.asarray(arm_points, dtype=np.float64),
        "distances": np.asarray(distances, dtype=np.float64),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "sim_time": float(total_time),
        "active_goal_indices": np.asarray(active_goal_indices, dtype=np.int32),
        "control_durations": np.asarray(control_durations, dtype=np.float64),
        "sqp_times_us": np.asarray(sqp_times_us, dtype=np.float64),
        "sqp_iters": np.asarray(sqp_iters, dtype=np.int32),
        "pcg_iters": stack_solver_stat_rows(pcg_iters, -1, np.int32),
        "rejected_solves": rejected_solves,
        "outcomes": outcomes,
        "circuit_check": circuit_check,
    }
    return ee0, goals, stats


def summarize(goals, stats):
    distances = stats["distances"]
    finite_distances = distances[np.isfinite(distances)]
    if finite_distances.size == 0:
        finite_distances = np.array([float("nan")])
    return {
        "goals_reached": sum(outcome == "reached" for outcome in stats["outcomes"]),
        "samples": int(distances.size),
        "sim_time": float(stats.get("sim_time", stats["timestamps"][-1] if stats["timestamps"].size else 0.0)),
        "mean_error": float(np.mean(finite_distances)),
        "max_error": float(np.max(finite_distances)),
        "final_error": float(distances[-1]),
        "outcomes": stats["outcomes"],
        "goals": goals,
        **stats,
    }


def assert_success(summary):
    if summary["goals_reached"] != len(summary["goals"]):
        raise RuntimeError(f"Only reached {summary['goals_reached']}/{len(summary['goals'])}: {summary['outcomes']}")
    if summary["final_error"] > GOAL_THRESHOLD:
        raise RuntimeError(f"Final error above threshold: {summary['final_error']:.6f}m")


def save_artifacts(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    actual = summary["actual"]
    arm_points = summary["arm_points"]
    goals = summary["goals"]
    distances = summary["distances"]
    timestamps = summary["timestamps"]
    active = summary["active_goal_indices"]
    finite = np.isfinite(actual).all(axis=1) & np.isfinite(distances) & np.isfinite(timestamps)
    if not finite.any():
        return
    actual_finite = actual[finite]

    fig = plt.figure(figsize=(11.0, 8.0))
    grid = fig.add_gridspec(2, 2)
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xz = fig.add_subplot(grid[0, 1])
    ax_err = fig.add_subplot(grid[1, 1])

    ax_3d.plot(actual_finite[:, 0], actual_finite[:, 1], actual_finite[:, 2], color="#00693E", linewidth=1.8)
    ax_3d.scatter(goals[:, 0], goals[:, 1], goals[:, 2], marker="*", s=100, color="#C90016")
    for idx, goal in enumerate(goals):
        ax_3d.text(goal[0], goal[1], goal[2], str(idx + 1))
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("z [m]")
    ax_3d.set_title("3D goal path")

    ax_xz.plot(actual_finite[:, 0], actual_finite[:, 2], color="#00693E", linewidth=1.8, label="actual")
    ax_xz.scatter(goals[:, 0], goals[:, 2], marker="*", s=100, color="#C90016", label="goals")
    ax_xz.set_xlabel("x [m]")
    ax_xz.set_ylabel("z [m]")
    ax_xz.set_title("XZ projection")
    ax_xz.set_aspect("equal", adjustable="box")
    ax_xz.grid(True, alpha=0.3)
    ax_xz.legend(loc="best")

    ax_err.plot(timestamps[finite], distances[finite], color="#C90016", linewidth=1.5)
    ax_err.axhline(GOAL_THRESHOLD, color="#747474", linestyle=":", linewidth=1.0, label="threshold")
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("distance to active goal [m]")
    ax_err.set_title("Goal tracking error")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")

    fig.suptitle(f"Tiago pick/place goals | reached {summary['goals_reached']}/{len(goals)}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tiago_pickplace_goals.png", dpi=160)
    plt.close(fig)

    finite_ids = np.flatnonzero(finite)
    if finite_ids.size < 2:
        return
    frame_count = min(300, finite_ids.size)
    frame_ids = finite_ids[np.linspace(0, finite_ids.size - 1, frame_count).astype(np.int64)]
    fig_gif, (ax_path, ax_dist) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    ax_path.scatter(goals[:, 0], goals[:, 2], marker="*", s=100, color="#C90016", label="goals")
    for idx, goal in enumerate(goals):
        ax_path.text(goal[0], goal[2], str(idx + 1))
    path_line, = ax_path.plot([], [], color="#00693E", linewidth=2.0)
    path_point, = ax_path.plot([], [], "o", color="#00693E", markersize=5)
    target_point, = ax_path.plot([], [], "x", color="#003192", markersize=7)
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("z [m]")
    ax_path.set_title("XZ path")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True, alpha=0.3)
    all_x = np.concatenate([actual_finite[:, 0], goals[:, 0]])
    all_z = np.concatenate([actual_finite[:, 2], goals[:, 2]])
    ax_path.set_xlim(np.min(all_x) - 0.05, np.max(all_x) + 0.05)
    ax_path.set_ylim(np.min(all_z) - 0.05, np.max(all_z) + 0.05)

    ax_dist.plot(timestamps[finite], distances[finite], color="#747474", linewidth=1.0)
    ax_dist.axhline(GOAL_THRESHOLD, color="#C90016", linestyle=":", linewidth=1.0)
    dist_point, = ax_dist.plot([], [], "o", color="#C90016", markersize=5)
    ax_dist.set_xlabel("time [s]")
    ax_dist.set_ylabel("distance [m]")
    ax_dist.set_title("Distance to active goal")
    ax_dist.set_xlim(timestamps[finite_ids[0]], timestamps[finite_ids[-1]])
    ax_dist.set_ylim(0.0, max(0.02, float(np.max(distances[finite])) * 1.1))
    ax_dist.grid(True, alpha=0.3)
    fig_gif.tight_layout()

    def update(frame_index):
        idx = frame_ids[frame_index]
        goal = goals[active[idx]]
        path_line.set_data(actual[:idx + 1, 0], actual[:idx + 1, 2])
        path_point.set_data([actual[idx, 0]], [actual[idx, 2]])
        target_point.set_data([goal[0]], [goal[2]])
        dist_point.set_data([timestamps[idx]], [distances[idx]])
        return path_line, path_point, target_point, dist_point

    anim = animation.FuncAnimation(fig_gif, update, frames=frame_count, interval=50, blit=True)
    anim.save(OUTPUT_DIR / "tiago_pickplace_goals.gif", writer=animation.PillowWriter(fps=20))
    plt.close(fig_gif)

    fig_arm = plt.figure(figsize=(7.5, 6.5))
    ax_arm = fig_arm.add_subplot(111, projection="3d")
    arm_line, = ax_arm.plot([], [], [], "o-", color="#00693E", linewidth=3.0, markersize=4)
    trail_line, = ax_arm.plot([], [], [], color="#003192", linewidth=1.5, alpha=0.8)
    active_goal_marker, = ax_arm.plot([], [], [], "*", color="#C90016", markersize=12)
    ax_arm.scatter(goals[:, 0], goals[:, 1], goals[:, 2], marker="*", s=90, color="#C90016", alpha=0.45)
    for idx, goal in enumerate(goals):
        ax_arm.text(goal[0], goal[1], goal[2], str(idx + 1))

    all_points = np.concatenate([arm_points[finite].reshape(-1, 3), goals], axis=0)
    center = np.mean(all_points, axis=0)
    span = float(np.max(np.ptp(all_points, axis=0)))
    radius = max(0.25, span * 0.6)
    ax_arm.set_xlim(center[0] - radius, center[0] + radius)
    ax_arm.set_ylim(center[1] - radius, center[1] + radius)
    ax_arm.set_zlim(center[2] - radius, center[2] + radius)
    ax_arm.set_xlabel("x [m]")
    ax_arm.set_ylabel("y [m]")
    ax_arm.set_zlabel("z [m]")
    ax_arm.set_title("Tiago right arm pick/place circuit")
    ax_arm.view_init(elev=24, azim=-54)
    fig_arm.tight_layout()

    def update_arm(frame_index):
        idx = frame_ids[frame_index]
        points = arm_points[idx]
        goal = goals[active[idx]]
        arm_line.set_data(points[:, 0], points[:, 1])
        arm_line.set_3d_properties(points[:, 2])
        trail_line.set_data(actual[:idx + 1, 0], actual[:idx + 1, 1])
        trail_line.set_3d_properties(actual[:idx + 1, 2])
        active_goal_marker.set_data([goal[0]], [goal[1]])
        active_goal_marker.set_3d_properties([goal[2]])
        ax_arm.set_title(f"Tiago right arm pick/place circuit | t={timestamps[idx]:.2f}s")
        return arm_line, trail_line, active_goal_marker

    arm_anim = animation.FuncAnimation(fig_arm, update_arm, frames=frame_count, interval=50, blit=False)
    arm_anim.save(OUTPUT_DIR / "tiago_pickplace_arm3d.gif", writer=animation.PillowWriter(fps=20))
    plt.close(fig_arm)


def main():
    ee0, goals, stats = run_pickplace()
    summary = summarize(goals, stats)
    save_artifacts(summary)

    print("Tiago pick/place goals")
    print("ee_start: ", np.array2string(ee0, precision=4))
    print(f"goals_reached: {summary['goals_reached']}/{len(goals)}")
    print(f"outcomes: {summary['outcomes']}")
    print(f"min_q_limit_clearance: {summary['circuit_check']['min_q_limit_clearance']:.3f}rad")
    print(f"max_joint_space_segment: {summary['circuit_check']['max_joint_space_segment']:.3f}rad")
    print(f"samples: {summary['samples']}")
    print(f"sim_time: {summary['sim_time']:.3f}s")
    if summary["control_durations"].size:
        mean_control_dt = float(np.mean(summary["control_durations"]))
        print(f"mean_control_period: {mean_control_dt * 1000.0:.3f}ms")
        print(f"mean_control_frequency: {1.0 / mean_control_dt:.3f}Hz")
        print(f"mean_sqp_time: {np.mean(summary['sqp_times_us']) / 1000.0:.3f}ms")
        print(f"mean_sqp_iters: {np.mean(summary['sqp_iters']):.2f}")
        print(f"rejected_solves: {summary['rejected_solves']}")
        valid_pcg_iters = summary["pcg_iters"][summary["pcg_iters"] >= 0]
        if valid_pcg_iters.size:
            print(f"mean_pcg_iters: {np.mean(valid_pcg_iters):.2f}")
            print(f"pcg_max_iter_fraction: {np.mean(valid_pcg_iters >= MAX_PCG_ITERS):.3f}")
    print(f"mean_error: {summary['mean_error']:.6f}m")
    print(f"max_error: {summary['max_error']:.6f}m")
    print(f"final_error: {summary['final_error']:.6f}m")
    print(f"plot: {OUTPUT_DIR / 'tiago_pickplace_goals.png'}")
    print(f"gif: {OUTPUT_DIR / 'tiago_pickplace_goals.gif'}")
    print(f"arm_3d_gif: {OUTPUT_DIR / 'tiago_pickplace_arm3d.gif'}")
    assert_success(summary)


if __name__ == "__main__":
    main()
