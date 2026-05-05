import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "python"))

from bsqp.common import initialize_warm_start, rk4
from bsqp.config import IIWA14_START_CONFIGS
from bsqp.interface import BSQP


MODEL_PATH = REPO_ROOT / "examples" / "iiwa_description" / "iiwa14.urdf"
OUTPUT_DIR = REPO_ROOT / "test-artifacts" / "iiwa_pickplace"
N = 16
DT = 0.03
SIM_DT = 0.003
GOAL_THRESHOLD = 0.01
GOAL_DWELL_TIME = 0.3
GOAL_TIMEOUT = 20.0


def tool_position(model, data, q):
    pin.forwardKinematics(model, data, q)
    return data.oMi[model.njoints - 1].translation.copy()


def pickplace_goals(ee0):
    offsets = np.array(
        [
            [0.20, 0.00, -0.10],
            [0.02, 0.24, -0.18],
            [-0.22, 0.04, -0.08],
            [0.00, -0.22, -0.16],
        ],
        dtype=np.float32,
    )
    return ee0 + offsets


def segment_reference(current, goal, max_step=0.05):
    delta = goal - current
    distance = float(np.linalg.norm(delta))
    segment_end = goal if distance <= max_step else current + delta / distance * max_step

    points = np.zeros((N, 6), dtype=np.float32)
    for knot in range(N):
        alpha = (knot + 1) / N
        points[knot, :3] = current + alpha * (segment_end - current)
    return points.reshape(1, -1)


def shift_warm_start(xu_solution, x_current, nx, nu):
    shifted = np.zeros_like(xu_solution)
    flat = xu_solution[0]
    for knot in range(N):
        dst = knot * (nx + nu)
        src = min(knot + 1, N - 1) * (nx + nu)
        shifted[0, dst:dst + nx] = flat[src:src + nx]
        if knot < N - 1:
            u_src = min(knot + 1, N - 2) * (nx + nu) + nx
            shifted[0, dst + nx:dst + nx + nu] = flat[u_src:u_src + nu]
    shifted[0, :nx] = x_current
    return shifted.astype(np.float32)


def run_pickplace():
    model = pin.buildModelFromUrdf(str(MODEL_PATH))
    data = model.createData()
    q = IIWA14_START_CONFIGS["home"].astype(np.float32)
    qd = np.zeros(model.nv, dtype=np.float32)
    x = np.concatenate([q, qd]).astype(np.float32)
    ee0 = tool_position(model, data, q)
    goals = pickplace_goals(ee0)

    solver = BSQP(
        model_path=str(MODEL_PATH),
        batch_size=1,
        N=N,
        dt=DT,
        plant_type="iiwa14",
        max_sqp_iters=5,
        max_pcg_iters=160,
        pcg_tol=1e-4,
        q_cost=20.0,
        qd_cost=5e-2,
        u_cost=1e-6,
        N_cost=400.0,
        q_lim_cost=0.01,
        rho=0.02,
    )

    warm_start = initialize_warm_start(x, N, solver.nx, solver.nu).reshape(1, -1).astype(np.float32)
    actual = []
    distances = []
    timestamps = []
    active_goal_indices = []
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

            reference = segment_reference(current_ee, goal)
            solution, _ = solver.solve(x.reshape(1, -1), reference, warm_start)
            u = solution[0, solver.nx:solver.nx + solver.nu].astype(np.float32)

            for _ in range(int(DT / SIM_DT)):
                q, qd = rk4(model, data, q.astype(float), qd.astype(float), u.astype(float), SIM_DT)
            x = np.concatenate([q, qd]).astype(np.float32)
            if not np.isfinite(x).all():
                outcomes[goal_index] = "diverged"
                break

            total_time += DT
            warm_start = shift_warm_start(solution, x, solver.nx, solver.nu)

        if outcomes[goal_index] == "not_reached":
            outcomes[goal_index] = "timeout"
            break
        if outcomes[goal_index] == "diverged":
            break

    stats = {
        "actual": np.asarray(actual, dtype=np.float64),
        "distances": np.asarray(distances, dtype=np.float64),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "active_goal_indices": np.asarray(active_goal_indices, dtype=np.int32),
        "outcomes": outcomes,
    }
    return ee0, goals, stats


def summarize(goals, stats):
    distances = stats["distances"]
    finite_distances = distances[np.isfinite(distances)]
    return {
        "goals_reached": sum(outcome == "reached" for outcome in stats["outcomes"]),
        "samples": int(distances.size),
        "sim_time": float(stats["timestamps"][-1]) if stats["timestamps"].size else 0.0,
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
    goals = summary["goals"]
    distances = summary["distances"]
    timestamps = summary["timestamps"]
    active = summary["active_goal_indices"]

    fig = plt.figure(figsize=(11.0, 8.0))
    grid = fig.add_gridspec(2, 2)
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xz = fig.add_subplot(grid[0, 1])
    ax_err = fig.add_subplot(grid[1, 1])

    ax_3d.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="#00693E", linewidth=1.8)
    ax_3d.scatter(goals[:, 0], goals[:, 1], goals[:, 2], marker="*", s=100, color="#C90016")
    for idx, goal in enumerate(goals):
        ax_3d.text(goal[0], goal[1], goal[2], str(idx + 1))
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("z [m]")
    ax_3d.set_title("3D goal path")

    ax_xz.plot(actual[:, 0], actual[:, 2], color="#00693E", linewidth=1.8, label="actual")
    ax_xz.scatter(goals[:, 0], goals[:, 2], marker="*", s=100, color="#C90016", label="goals")
    ax_xz.set_xlabel("x [m]")
    ax_xz.set_ylabel("z [m]")
    ax_xz.set_title("XZ projection")
    ax_xz.set_aspect("equal", adjustable="box")
    ax_xz.grid(True, alpha=0.3)
    ax_xz.legend(loc="best")

    ax_err.plot(timestamps, distances, color="#C90016", linewidth=1.5)
    ax_err.axhline(GOAL_THRESHOLD, color="#747474", linestyle=":", linewidth=1.0, label="threshold")
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("distance to active goal [m]")
    ax_err.set_title("Goal tracking error")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")

    fig.suptitle(f"IIWA14 pick/place goals | reached {summary['goals_reached']}/{len(goals)}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "iiwa14_pickplace_goals.png", dpi=160)
    plt.close(fig)

    frame_count = min(300, actual.shape[0])
    frame_ids = np.linspace(0, actual.shape[0] - 1, frame_count).astype(np.int64)
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
    all_x = np.concatenate([actual[:, 0], goals[:, 0]])
    all_z = np.concatenate([actual[:, 2], goals[:, 2]])
    ax_path.set_xlim(np.min(all_x) - 0.05, np.max(all_x) + 0.05)
    ax_path.set_ylim(np.min(all_z) - 0.05, np.max(all_z) + 0.05)

    ax_dist.plot(timestamps, distances, color="#747474", linewidth=1.0)
    ax_dist.axhline(GOAL_THRESHOLD, color="#C90016", linestyle=":", linewidth=1.0)
    dist_point, = ax_dist.plot([], [], "o", color="#C90016", markersize=5)
    ax_dist.set_xlabel("time [s]")
    ax_dist.set_ylabel("distance [m]")
    ax_dist.set_title("Distance to active goal")
    ax_dist.set_xlim(timestamps[0], timestamps[-1])
    ax_dist.set_ylim(0.0, max(0.02, float(np.max(distances)) * 1.1))
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
    anim.save(OUTPUT_DIR / "iiwa14_pickplace_goals.gif", writer=animation.PillowWriter(fps=20))
    plt.close(fig_gif)


def main():
    ee0, goals, stats = run_pickplace()
    summary = summarize(goals, stats)
    save_artifacts(summary)
    assert_success(summary)

    print("IIWA14 pick/place goals")
    print("ee_start: ", np.array2string(ee0, precision=4))
    print(f"goals_reached: {summary['goals_reached']}/{len(goals)}")
    print(f"samples: {summary['samples']}")
    print(f"sim_time: {summary['sim_time']:.3f}s")
    print(f"mean_error: {summary['mean_error']:.6f}m")
    print(f"max_error: {summary['max_error']:.6f}m")
    print(f"final_error: {summary['final_error']:.6f}m")
    print(f"plot: {OUTPUT_DIR / 'iiwa14_pickplace_goals.png'}")
    print(f"gif: {OUTPUT_DIR / 'iiwa14_pickplace_goals.gif'}")


if __name__ == "__main__":
    main()
