#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "python"))

from bsqp.common import initialize_warm_start, rk4
from bsqp.config import INDY7_START_CONFIGS, TIAGO_RIGHT_START_CONFIGS
from bsqp.interface import BSQP


DT = 0.03
SIM_DT = 0.003
N = 32
SEGMENT_TIME = 3.0
REJECT_FALLBACK_DAMPING = 5.0
OUTPUT_ROOT = REPO_ROOT / "test-artifacts" / "large_easy_tracking"


BASE_SOLVER_PARAMS = {
    "max_sqp_iters": 20,
    "kkt_tol": 1e-4,
    "max_pcg_iters": 1000,
    "pcg_tol": 1e-3,
    "solve_ratio": 1.0,
    "mu": 1.0,
    "u_cost": 1e-6,
    "q_lim_cost": 0.01,
    "vel_lim_cost": 0.0,
    "ctrl_lim_cost": 0.0,
    "rho": 0.02,
}


PLANTS = {
    "tiago_right": {
        "model_path": REPO_ROOT / "gato" / "dynamics" / "tiago_right" / "tiago_right_arm.urdf",
        "frame": ("torso_lift_link", "arm_right_tool_link"),
        "waypoints": np.asarray(
            [
                TIAGO_RIGHT_START_CONFIGS["comfortable"],
                [0.35, -1.05, 1.55, 0.35, 0.75, 0.80, 1.10],
                [-1.45, -1.25, 0.20, 0.85, -0.85, 1.20, -1.00],
                [-2.35, -0.55, 1.20, -0.10, 1.15, 0.65, 1.80],
                [-0.80, -1.75, 0.65, 0.95, -1.10, 1.35, -1.60],
                TIAGO_RIGHT_START_CONFIGS["comfortable"],
            ],
            dtype=np.float32,
        ),
        "solver": {"q_cost": 160.0, "qd_cost": 2e-2, "N_cost": 800.0},
    },
    "iiwa14": {
        "model_path": REPO_ROOT / "examples" / "iiwa_description" / "iiwa14.urdf",
        "frame": None,
        "waypoints": np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.75, 0.55, -0.65, -0.85, 0.80, 0.65, -0.55],
                [-0.85, 0.45, 0.75, -0.65, -0.95, 0.55, 0.70],
                [0.55, -0.70, 0.45, 0.90, -0.55, -0.65, 0.85],
                [-0.65, -0.45, -0.75, 0.65, 0.85, -0.50, -0.75],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "solver": {"q_cost": 5.0, "qd_cost": 5e-2, "N_cost": 100.0},
    },
    "indy7": {
        "model_path": REPO_ROOT / "examples" / "indy7_description" / "indy7.urdf",
        "frame": None,
        "waypoints": np.asarray(
            [
                INDY7_START_CONFIGS["ready"],
                [-0.35, -0.75, 1.25, -0.80, 0.85, 0.70],
                [-1.75, 0.45, 0.55, 0.95, -0.75, -0.65],
                [-0.65, -0.95, 1.45, 0.70, 0.55, -0.90],
                [-1.25, 0.20, 0.75, -1.05, -0.65, 0.95],
                INDY7_START_CONFIGS["ready"],
            ],
            dtype=np.float32,
        ),
        "solver": {"q_cost": 5.0, "qd_cost": 5e-2, "N_cost": 100.0},
    },
}


def smoothstep(alpha):
    return alpha * alpha * (3.0 - 2.0 * alpha)


def end_effector_position(model, data, q, frame_spec):
    pin.forwardKinematics(model, data, q)
    if frame_spec is None:
        return data.oMi[model.njoints - 1].translation.copy()

    pin.updateFramePlacements(model, data)
    base_frame, tool_frame = frame_spec
    base_id = model.getFrameId(base_frame)
    tool_id = model.getFrameId(tool_frame)
    return (data.oMf[base_id].inverse() * data.oMf[tool_id]).translation.copy()


def arm_link_positions(model, data, q, frame_spec):
    pin.forwardKinematics(model, data, q)
    if frame_spec is None:
        return np.asarray([data.oMi[joint_id].translation.copy() for joint_id in range(1, model.njoints)], dtype=np.float64)

    pin.updateFramePlacements(model, data)
    base_frame, _ = frame_spec
    base_id = model.getFrameId(base_frame)
    base_inv = data.oMf[base_id].inverse()
    link_names = [
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
    return np.asarray(
        [(base_inv * data.oMf[model.getFrameId(link_name)]).translation.copy() for link_name in link_names],
        dtype=np.float64,
    )


def generate_reference(model, data, waypoints, frame_spec, dt, segment_time):
    samples_per_segment = int(round(segment_time / dt))
    if samples_per_segment < 2:
        raise ValueError("segment_time must contain at least two control ticks")

    reference = []
    joint_reference = []
    for q_start, q_end in zip(waypoints[:-1], waypoints[1:]):
        for sample_idx in range(samples_per_segment):
            alpha = (sample_idx + 1) / samples_per_segment
            q = (1.0 - smoothstep(alpha)) * q_start + smoothstep(alpha) * q_end
            joint_reference.append(q.copy())
            reference.append(end_effector_position(model, data, q.astype(float), frame_spec))

    waypoint_xyz = np.asarray(
        [end_effector_position(model, data, q.astype(float), frame_spec) for q in waypoints],
        dtype=np.float64,
    )
    return np.asarray(reference, dtype=np.float32), np.asarray(joint_reference, dtype=np.float32), waypoint_xyz


def reference_window(reference, index, horizon):
    points = np.zeros((horizon, 6), dtype=np.float32)
    for knot in range(horizon):
        points[knot, :3] = reference[min(index + knot + 1, reference.shape[0] - 1)]
    return points.reshape(1, -1)


def gravity_compensation(model, data, q, qd):
    qdd_desired = np.clip(-REJECT_FALLBACK_DAMPING * qd.astype(float), -20.0, 20.0)
    return pin.rnea(model, data, q.astype(float), qd.astype(float), qdd_desired).astype(np.float32)


def stack_solver_stat_rows(rows, fill_value, dtype):
    if not rows:
        return np.empty((0, 0), dtype=dtype)

    width = max(row.size for row in rows)
    stacked = np.full((len(rows), width), fill_value, dtype=dtype)
    for row_index, row in enumerate(rows):
        stacked[row_index, : row.size] = row
    return stacked


def summarize(
    reference,
    actual,
    errors,
    solve_times_us,
    linear_solve_times_us,
    pcg_iters,
    rejected_solves,
    solver_params,
    joint_positions,
    joint_velocities,
    planned_controls,
    applied_controls,
    model,
    dt,
):
    valid_pcg = pcg_iters[pcg_iters >= 0]
    valid_linear_times = linear_solve_times_us[linear_solve_times_us >= 0.0]
    footprint = np.ptp(reference, axis=0)
    arc_length = float(np.sum(np.linalg.norm(np.diff(reference, axis=0), axis=1)))
    joint_min = np.min(joint_positions, axis=0)
    joint_max = np.max(joint_positions, axis=0)
    lower = model.lowerPositionLimit.astype(np.float64)
    upper = model.upperPositionLimit.astype(np.float64)
    velocity_limit = model.velocityLimit.astype(np.float64)
    effort_limit = model.effortLimit.astype(np.float64)
    max_abs_velocity = np.max(np.abs(joint_velocities), axis=0)
    max_abs_planned_control = np.max(np.abs(planned_controls), axis=0)
    max_abs_applied_control = np.max(np.abs(applied_controls), axis=0)
    if joint_velocities.shape[0] > 1:
        max_abs_acceleration = np.max(np.abs(np.diff(joint_velocities, axis=0)) / dt, axis=0)
    else:
        max_abs_acceleration = np.full(model.nv, float("nan"))
    return {
        "footprint_m": [float(v) for v in footprint],
        "arc_length_m": arc_length,
        "mean_error_m": float(np.mean(errors)),
        "p95_error_m": float(np.quantile(errors, 0.95)),
        "max_error_m": float(np.max(errors)),
        "final_error_m": float(errors[-1]),
        "rejected_solves": int(rejected_solves),
        "mean_pcg_iters": float(np.mean(valid_pcg)) if valid_pcg.size else float("nan"),
        "pcg_cap_hit_fraction": float(np.mean(valid_pcg >= solver_params["max_pcg_iters"])) if valid_pcg.size else float("nan"),
        "mean_sqp_time_ms": float(np.mean(solve_times_us) / 1000.0),
        "p95_sqp_time_ms": float(np.quantile(solve_times_us, 0.95) / 1000.0),
        "max_sqp_time_ms": float(np.max(solve_times_us) / 1000.0),
        "mean_linear_solve_time_ms": float(np.mean(valid_linear_times) / 1000.0) if valid_linear_times.size else float("nan"),
        "p95_linear_solve_time_ms": float(np.quantile(valid_linear_times, 0.95) / 1000.0) if valid_linear_times.size else float("nan"),
        "max_linear_solve_time_ms": float(np.max(valid_linear_times) / 1000.0) if valid_linear_times.size else float("nan"),
        "joint_position_min_rad": [float(v) for v in joint_min],
        "joint_position_max_rad": [float(v) for v in joint_max],
        "joint_position_lower_limit_rad": [float(v) for v in lower],
        "joint_position_upper_limit_rad": [float(v) for v in upper],
        "max_joint_position_violation_rad": float(
            np.max(np.maximum(np.maximum(lower - joint_min, 0.0), np.maximum(joint_max - upper, 0.0)))
        ),
        "max_abs_joint_velocity_rad_s": [float(v) for v in max_abs_velocity],
        "joint_velocity_limit_rad_s": [float(v) for v in velocity_limit],
        "max_joint_velocity_violation_rad_s": float(np.max(np.maximum(max_abs_velocity - velocity_limit, 0.0))),
        "max_abs_joint_acceleration_rad_s2_finite_difference": [float(v) for v in max_abs_acceleration],
        "max_abs_planned_torque_nm": [float(v) for v in max_abs_planned_control],
        "max_abs_applied_torque_nm": [float(v) for v in max_abs_applied_control],
        "applied_torque_limit_nm": [float(v) for v in effort_limit],
        "max_planned_torque_violation_nm": float(np.max(np.maximum(max_abs_planned_control - effort_limit, 0.0))),
        "max_applied_torque_violation_nm": float(np.max(np.maximum(max_abs_applied_control - effort_limit, 0.0))),
    }


def run_tracking(
    plant,
    horizon,
    dt,
    sim_dt,
    segment_time,
    solver_overrides=None,
    output_label=None,
    reset_dual_each_tick=True,
    warm_start_policy="repeat_current",
    reject_fallback=True,
    saturate_controls=False,
):
    config = PLANTS[plant]
    model_path = config["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model for {plant}: {model_path}")

    model = pin.buildModelFromUrdf(str(model_path))
    data = model.createData()
    waypoints = config["waypoints"]
    if waypoints.shape[1] != model.nq:
        raise RuntimeError(f"{plant} waypoint dimension {waypoints.shape[1]} does not match model.nq={model.nq}")
    if np.any(waypoints < model.lowerPositionLimit - 1e-6) or np.any(waypoints > model.upperPositionLimit + 1e-6):
        raise RuntimeError(f"{plant} waypoints exceed model position limits")

    solver_params = {**BASE_SOLVER_PARAMS, **config["solver"]}
    if solver_overrides:
        solver_params.update(solver_overrides)
    reference, joint_reference, waypoint_xyz = generate_reference(model, data, waypoints, config["frame"], dt, segment_time)
    solver = BSQP(str(model_path), 1, horizon, dt, plant_type=plant, adapt_rho=True, **solver_params)

    q = waypoints[0].copy()
    qd = np.zeros(model.nv, dtype=np.float32)
    x = np.concatenate([q, qd]).astype(np.float32)

    actual = []
    arm_points = []
    desired = []
    joint_positions = []
    joint_velocities = []
    planned_controls = []
    applied_controls = []
    timestamps = []
    solve_times_us = []
    linear_solve_times_us = []
    pcg_iters = []
    rejected_solves = 0
    total_time = 0.0
    substeps = int(round(dt / sim_dt))
    previous_solution = None

    for index, desired_now in enumerate(reference):
        actual_now = end_effector_position(model, data, q.astype(float), config["frame"])
        actual.append(actual_now)
        arm_points.append(arm_link_positions(model, data, q.astype(float), config["frame"]))
        desired.append(desired_now)
        joint_positions.append(q.copy())
        joint_velocities.append(qd.copy())
        timestamps.append(total_time)

        if reset_dual_each_tick:
            solver.reset_dual()
        solver.reset_rho()
        if warm_start_policy == "repeat_current" or previous_solution is None:
            warm_start = initialize_warm_start(x, horizon, solver.nx, solver.nu).reshape(1, -1).astype(np.float32)
        elif warm_start_policy == "previous_solution":
            warm_start = previous_solution.reshape(1, -1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported warm_start_policy={warm_start_policy!r}")
        solution, _ = solver.solve(x.reshape(1, -1), reference_window(reference, index, horizon), warm_start)
        previous_solution = solution.copy()
        solve_times_us.append(float(solver.stats["sqp_time_us"]))
        linear_solve_times_us.append(np.asarray(solver.stats["pcg_times_us"], dtype=np.float32).reshape(-1))
        pcg_iters.append(np.asarray(solver.stats["pcg_iters"], dtype=np.int32).reshape(-1))
        step_sizes = np.asarray(solver.stats["step_size"], dtype=np.float32).reshape(-1)
        if np.any(step_sizes > 0.0) or not reject_fallback:
            planned_u = solution[0, solver.nx : solver.nx + solver.nu].astype(np.float32)
        else:
            rejected_solves += 1
            planned_u = gravity_compensation(model, data, q, qd)

        u = planned_u
        if saturate_controls:
            u = np.clip(u, -model.effortLimit.astype(np.float32), model.effortLimit.astype(np.float32))
        planned_controls.append(planned_u.copy())
        applied_controls.append(u.copy())

        for _ in range(substeps):
            q, qd = rk4(model, data, q.astype(float), qd.astype(float), u.astype(float), sim_dt)
        x = np.concatenate([q, qd]).astype(np.float32)
        if not np.isfinite(x).all():
            raise RuntimeError(f"{plant} tracking diverged at step {index}")

        total_time += dt

    actual = np.asarray(actual, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    errors = np.linalg.norm(actual - desired, axis=1)
    solve_times_us = np.asarray(solve_times_us, dtype=np.float64)
    linear_solve_times_us = stack_solver_stat_rows(linear_solve_times_us, -1.0, np.float32)
    pcg_iters = stack_solver_stat_rows(pcg_iters, -1, np.int32)
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    joint_velocities = np.asarray(joint_velocities, dtype=np.float64)
    planned_controls = np.asarray(planned_controls, dtype=np.float64)
    applied_controls = np.asarray(applied_controls, dtype=np.float64)
    summary = summarize(
        desired,
        actual,
        errors,
        solve_times_us,
        linear_solve_times_us,
        pcg_iters,
        rejected_solves,
        solver_params,
        joint_positions,
        joint_velocities,
        planned_controls,
        applied_controls,
        model,
        dt,
    )

    return {
        "plant": plant,
        "linsys_solver": solver.linsys_solver,
        "model_path": str(model_path),
        "horizon": horizon,
        "dt": dt,
        "sim_dt": sim_dt,
        "segment_time": segment_time,
        "output_label": output_label,
        "reset_dual_each_tick": reset_dual_each_tick,
        "warm_start_policy": warm_start_policy,
        "reject_fallback": reject_fallback,
        "saturate_controls": saturate_controls,
        "solver_params": solver_params,
        "reference": desired,
        "joint_reference": joint_reference.astype(np.float64),
        "actual": actual,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "planned_controls": planned_controls,
        "applied_controls": applied_controls,
        "arm_points": np.asarray(arm_points, dtype=np.float64),
        "waypoints_xyz": waypoint_xyz,
        "timestamps": timestamps,
        "errors": errors,
        "solve_times_us": solve_times_us,
        "linear_solve_times_us": linear_solve_times_us,
        "pcg_iters": pcg_iters,
        "summary": summary,
    }


def save_artifacts(stats, save_media=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    plant = stats["plant"]
    linsys_solver = stats["linsys_solver"].lower()
    output_name = linsys_solver
    if stats.get("output_label"):
        output_name = f"{output_name}_{stats['output_label']}"
    output_dir = OUTPUT_ROOT / plant / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = stats["timestamps"]
    reference = stats["reference"]
    actual = stats["actual"]
    errors = stats["errors"]
    waypoints = stats["waypoints_xyz"]
    arm_points = stats["arm_points"]

    np.savetxt(
        output_dir / "reference.csv",
        np.column_stack([timestamps, reference]),
        delimiter=",",
        header="t,x,y,z",
        comments="",
    )
    np.savetxt(
        output_dir / "actual.csv",
        np.column_stack([timestamps, actual, errors]),
        delimiter=",",
        header="t,x,y,z,error",
        comments="",
    )
    np.savetxt(output_dir / "joint_reference.csv", stats["joint_reference"], delimiter=",")
    np.savetxt(
        output_dir / "joints.csv",
        np.column_stack([timestamps, stats["joint_positions"], stats["joint_velocities"]]),
        delimiter=",",
        header="t," + ",".join([f"q{i}" for i in range(stats["joint_positions"].shape[1])])
        + ","
        + ",".join([f"qd{i}" for i in range(stats["joint_velocities"].shape[1])]),
        comments="",
    )
    np.savetxt(
        output_dir / "planned_controls.csv",
        np.column_stack([timestamps, stats["planned_controls"]]),
        delimiter=",",
        header="t," + ",".join([f"u{i}" for i in range(stats["planned_controls"].shape[1])]),
        comments="",
    )
    np.savetxt(
        output_dir / "applied_controls.csv",
        np.column_stack([timestamps, stats["applied_controls"]]),
        delimiter=",",
        header="t," + ",".join([f"u{i}" for i in range(stats["applied_controls"].shape[1])]),
        comments="",
    )
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "plant": plant,
                "linsys_solver": stats["linsys_solver"],
                "model_path": stats["model_path"],
                "horizon": stats["horizon"],
                "dt": stats["dt"],
                "sim_dt": stats["sim_dt"],
                "segment_time": stats["segment_time"],
                "reset_dual_each_tick": stats["reset_dual_each_tick"],
                "warm_start_policy": stats["warm_start_policy"],
                "reject_fallback": stats["reject_fallback"],
                "saturate_controls": stats["saturate_controls"],
                "solver_params": stats["solver_params"],
                "summary": stats["summary"],
            },
            f,
            indent=2,
        )

    if not save_media:
        return output_dir, None, None

    fig = plt.figure(figsize=(11.0, 7.5))
    grid = fig.add_gridspec(2, 2)
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xy = fig.add_subplot(grid[0, 1])
    ax_err = fig.add_subplot(grid[1, 1])

    ax_3d.plot(reference[:, 0], reference[:, 1], reference[:, 2], linestyle=":", color="#747474", linewidth=1.8, label="reference")
    ax_3d.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="#00693E", linewidth=1.8, label="actual")
    ax_3d.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], marker="*", s=90, color="#C90016", label="joint waypoints FK")
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("z [m]")
    ax_3d.legend(loc="best")

    ax_xy.plot(reference[:, 0], reference[:, 1], linestyle=":", color="#747474", linewidth=1.8, label="reference")
    ax_xy.plot(actual[:, 0], actual[:, 1], color="#00693E", linewidth=1.8, label="actual")
    ax_xy.scatter(waypoints[:, 0], waypoints[:, 1], marker="*", s=80, color="#C90016")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("XY projection")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best")

    ax_err.plot(timestamps, errors, color="#C90016", linewidth=1.5)
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("tracking error [m]")
    ax_err.set_title("Tracking error")
    ax_err.grid(True, alpha=0.3)

    summary = stats["summary"]
    fig.suptitle(
        f"{plant} {stats['linsys_solver']} large easy tracking | mean {summary['mean_error_m']:.4f} m | "
        f"PCG cap {summary['pcg_cap_hit_fraction']:.3f}"
    )
    fig.tight_layout()
    plot_path = output_dir / "tracking.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    frame_ids = np.linspace(0, actual.shape[0] - 1, min(360, actual.shape[0])).astype(np.int64)
    fig_gif = plt.figure(figsize=(8.2, 7.0))
    ax = fig_gif.add_subplot(111, projection="3d")
    ax.plot(reference[:, 0], reference[:, 1], reference[:, 2], linestyle=":", color="#747474", linewidth=1.5)
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], marker="*", s=80, color="#C90016", alpha=0.55)
    arm_line, = ax.plot([], [], [], "o-", color="#00693E", linewidth=3.0, markersize=4)
    trail_line, = ax.plot([], [], [], color="#003192", linewidth=1.5, alpha=0.8)
    target_marker, = ax.plot([], [], [], "x", color="#C90016", markersize=8)

    all_points = np.concatenate([arm_points.reshape(-1, 3), reference, actual, waypoints], axis=0)
    center = np.mean(all_points, axis=0)
    radius = max(0.5, float(np.max(np.ptp(all_points, axis=0))) * 0.60)
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
        trail_line.set_data(actual[: idx + 1, 0], actual[: idx + 1, 1])
        trail_line.set_3d_properties(actual[: idx + 1, 2])
        target_marker.set_data([reference[idx, 0]], [reference[idx, 1]])
        target_marker.set_3d_properties([reference[idx, 2]])
        ax.set_title(f"{plant} {stats['linsys_solver']} | t={timestamps[idx]:.2f}s | err={errors[idx]:.3f}m")
        return arm_line, trail_line, target_marker

    anim = animation.FuncAnimation(fig_gif, update, frames=frame_ids.size, interval=50, blit=False)
    gif_path = output_dir / "tracking.gif"
    anim.save(gif_path, writer=animation.PillowWriter(fps=20))
    plt.close(fig_gif)
    return output_dir, plot_path, gif_path


def print_report(stats, output_dir, plot_path, gif_path):
    summary = stats["summary"]
    footprint = summary["footprint_m"]
    print(f"Large easy tracking: {stats['plant']}")
    print(f"linear_solver: {stats['linsys_solver']}")
    print(f"model_path: {stats['model_path']}")
    print(f"N: {stats['horizon']}")
    print(f"DT: {stats['dt']:.3f}s")
    print(f"sim_time: {stats['timestamps'][-1] + stats['dt']:.3f}s")
    print(f"solver_params: {json.dumps(stats['solver_params'], sort_keys=True)}")
    print(f"footprint: {footprint[0]:.3f}m x {footprint[1]:.3f}m x {footprint[2]:.3f}m")
    print(f"arc_length: {summary['arc_length_m']:.3f}m")
    print(f"mean_error: {summary['mean_error_m']:.6f}m")
    print(f"p95_error: {summary['p95_error_m']:.6f}m")
    print(f"max_error: {summary['max_error_m']:.6f}m")
    print(f"final_error: {summary['final_error_m']:.6f}m")
    print(f"rejected_solves: {summary['rejected_solves']}")
    print(f"mean_pcg_iters: {summary['mean_pcg_iters']:.2f}")
    print(f"pcg_cap_hit_fraction: {summary['pcg_cap_hit_fraction']:.3f}")
    print(f"mean_sqp_time: {summary['mean_sqp_time_ms']:.3f}ms")
    print(f"p95_sqp_time: {summary['p95_sqp_time_ms']:.3f}ms")
    print(f"max_sqp_time: {summary['max_sqp_time_ms']:.3f}ms")
    print(f"mean_linear_solve_time: {summary['mean_linear_solve_time_ms']:.3f}ms")
    print(f"p95_linear_solve_time: {summary['p95_linear_solve_time_ms']:.3f}ms")
    print(f"max_linear_solve_time: {summary['max_linear_solve_time_ms']:.3f}ms")
    print(f"max_joint_position_violation: {summary['max_joint_position_violation_rad']:.6f}rad")
    print(f"max_joint_velocity_violation: {summary['max_joint_velocity_violation_rad_s']:.6f}rad/s")
    print(f"max_planned_torque_violation: {summary['max_planned_torque_violation_nm']:.6f}Nm")
    print(f"max_applied_torque_violation: {summary['max_applied_torque_violation_nm']:.6f}Nm")
    print(f"output_dir: {output_dir}")
    if plot_path is not None:
        print(f"plot: {plot_path}")
    if gif_path is not None:
        print(f"gif: {gif_path}")


def compact_float_label(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(description="Track a large but smooth FK-generated end-effector path.")
    parser.add_argument("--plant", choices=sorted(PLANTS), default="tiago_right")
    parser.add_argument("--N", type=int, default=N)
    parser.add_argument("--DT", type=float, default=DT)
    parser.add_argument("--sim-dt", type=float, default=SIM_DT)
    parser.add_argument("--segment-time", type=float, default=SEGMENT_TIME)
    parser.add_argument("--u-cost", type=float, default=None)
    parser.add_argument("--q-cost", type=float, default=None)
    parser.add_argument("--qd-cost", type=float, default=None)
    parser.add_argument("--N-cost", type=float, default=None)
    parser.add_argument("--vel-lim-cost", type=float, default=None)
    parser.add_argument("--ctrl-lim-cost", type=float, default=None)
    parser.add_argument("--kkt-tol", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--pcg-tol", type=float, default=None)
    parser.add_argument("--max-pcg-iters", type=int, default=None)
    parser.add_argument("--max-sqp-iters", type=int, default=None)
    parser.add_argument("--no-reset-dual", action="store_true")
    parser.add_argument("--warm-start-policy", choices=["repeat_current", "previous_solution"], default="repeat_current")
    parser.add_argument("--no-reject-fallback", action="store_true")
    parser.add_argument("--saturate-controls", action="store_true")
    parser.add_argument("--output-label", default=None)
    parser.add_argument("--no-media", action="store_true", help="Save CSV/JSON only; skip PNG/GIF rendering.")
    return parser.parse_args()


def main():
    args = parse_args()
    solver_overrides = {}
    for key, value in (
        ("u_cost", args.u_cost),
        ("q_cost", args.q_cost),
        ("qd_cost", args.qd_cost),
        ("N_cost", args.N_cost),
        ("vel_lim_cost", args.vel_lim_cost),
        ("ctrl_lim_cost", args.ctrl_lim_cost),
        ("kkt_tol", args.kkt_tol),
        ("rho", args.rho),
        ("mu", args.mu),
        ("pcg_tol", args.pcg_tol),
        ("max_pcg_iters", args.max_pcg_iters),
        ("max_sqp_iters", args.max_sqp_iters),
    ):
        if value is not None:
            solver_overrides[key] = value

    output_label = args.output_label
    if output_label is None and args.u_cost is not None:
        output_label = f"ucost_{compact_float_label(args.u_cost)}"

    stats = run_tracking(
        args.plant,
        args.N,
        args.DT,
        args.sim_dt,
        args.segment_time,
        solver_overrides,
        output_label,
        reset_dual_each_tick=not args.no_reset_dual,
        warm_start_policy=args.warm_start_policy,
        reject_fallback=not args.no_reject_fallback,
        saturate_controls=args.saturate_controls,
    )
    output_dir, plot_path, gif_path = save_artifacts(stats, save_media=not args.no_media)
    print_report(stats, output_dir, plot_path, gif_path)


if __name__ == "__main__":
    main()
