#!/usr/bin/env python3
"""Run one non-notebook GATO figure-8 tracking example and save artifacts."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "python"))
sys.path.append(str(REPO_ROOT / "tiago_src"))

from bsqp.common import figure8
from bsqp.config import (
    BATCH_COLORS,
    DEFAULT_SOLVER_PARAMS,
    FIG8_DEFAULT_PARAMS,
    IIWA14_START_CONFIGS,
    INDY7_START_CONFIGS,
)
from gato_tiago.config import (
    TIAGO_RIGHT_DEFAULT_START_CONFIG,
    TIAGO_RIGHT_START_CONFIGS,
    TIAGO_TRACKING_SOLVER_PARAMS,
)
from gato_tiago.tiago_mpc_controller import MPC_GATO


OUTPUT_ROOT = REPO_ROOT / "example_artifacts" / "gato_fig8_tracking"
TIAGO_FIRST_REFERENCE = np.array([0.557576, -0.672736, -0.276456], dtype=np.float64)
TIAGO_HORIZONTAL_FIG8_PARAMS = {
    "x_span": 0.32,
    "y_amplitude": 0.08,
    "edge_clearance": 0.02,
    "period": 24.0,
    "cycles": FIG8_DEFAULT_PARAMS["cycles"],
}


def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    return pin.buildModelFromUrdf(str(model_path))


def ee_position(model, q, plant):
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    if plant == "tiago_right":
        pin.updateFramePlacements(model, data)
        torso_id = model.getFrameId("torso_lift_link")
        tool_id = model.getFrameId("arm_right_tool_link")
        return (data.oMf[torso_id].inverse() * data.oMf[tool_id]).translation.copy()
    return data.oMi[model.njoints - 1].translation.copy()


def ee_pose_rpy(model, q, plant):
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    if plant == "tiago_right":
        pin.updateFramePlacements(model, data)
        torso_id = model.getFrameId("torso_lift_link")
        tool_id = model.getFrameId("arm_right_tool_link")
        placement = data.oMf[torso_id].inverse() * data.oMf[tool_id]
    else:
        placement = data.oMi[model.njoints - 1]
    return placement.translation.copy(), pin.rpy.matrixToRpy(placement.rotation).copy()


def rotation_from_vectors(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.dot(source, target))

    if np.linalg.norm(cross) < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        axis = np.zeros(3)
        axis[np.argmin(np.abs(source))] = 1.0
        cross = np.cross(source, axis)
        cross /= np.linalg.norm(cross)
        return -np.eye(3) + 2.0 * np.outer(cross, cross)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / float(np.dot(cross, cross)))


def transformed_indy7_figure8(dt, tiago_start):
    source = figure8(dt, **FIG8_DEFAULT_PARAMS).reshape(-1, 6)
    source_points = source[:, :3]

    indy_model = load_model(REPO_ROOT / "examples" / "indy7_description" / "indy7.urdf")
    indy_start = ee_position(indy_model, INDY7_START_CONFIGS["zero"].astype(np.float64), "indy7")
    indy_delta = source_points[0] - indy_start
    tiago_delta = TIAGO_FIRST_REFERENCE - tiago_start

    rotation = rotation_from_vectors(indy_delta, tiago_delta)
    transformed = source.copy()
    transformed[:, :3] = (rotation @ (source_points - source_points[0]).T).T + TIAGO_FIRST_REFERENCE
    return transformed.reshape(-1), {
        "reference_mode": "transformed_indy7",
        "tiago_first_reference": TIAGO_FIRST_REFERENCE.tolist(),
        "indy_start_delta": indy_delta.tolist(),
        "tiago_start_delta": tiago_delta.tolist(),
        "start_delta_norm_m": float(np.linalg.norm(tiago_delta)),
        "transform_rotation": rotation.tolist(),
    }


def tiago_horizontal_figure8(dt, start_ee):
    params = TIAGO_HORIZONTAL_FIG8_PARAMS
    steps_per_cycle = int(params["period"] / dt)
    phase = np.linspace(0.0, 2.0 * np.pi, steps_per_cycle, endpoint=False)
    left_edge = np.asarray(start_ee, dtype=np.float64).copy()
    left_edge[0] += params["edge_clearance"]

    cycle = np.zeros((steps_per_cycle, 6), dtype=np.float64)
    cycle[:, 0] = left_edge[0] + 0.5 * params["x_span"] * (1.0 - np.cos(phase))
    cycle[:, 1] = left_edge[1] + params["y_amplitude"] * np.sin(2.0 * phase)
    cycle[:, 2] = left_edge[2]
    reference = np.tile(cycle, (int(params["cycles"]), 1))
    return reference.reshape(-1), {
        "reference_mode": "tiago_horizontal",
        "start_edge_clearance_m": float(params["edge_clearance"]),
        "x_span_m": float(params["x_span"]),
        "y_amplitude_m": float(params["y_amplitude"]),
        "period_sec": float(params["period"]),
        "cycles": int(params["cycles"]),
        "left_edge": left_edge.tolist(),
    }


def plant_config(
    plant,
    dt,
    *,
    tiago_start_config=TIAGO_RIGHT_DEFAULT_START_CONFIG,
    track_start_orientation=False,
    orientation_rpy=None,
):
    if plant == "indy7":
        model_path = REPO_ROOT / "examples" / "indy7_description" / "indy7.urdf"
        return {
            "model_path": model_path,
            "start_q": INDY7_START_CONFIGS["zero"].astype(np.float32),
            "batch_sizes": [1],
            "f_ext": np.zeros(6, dtype=np.float32),
            "reference": figure8(dt, **FIG8_DEFAULT_PARAMS),
            "reference_metadata": {"reference_mode": "default"},
        }
    if plant == "iiwa14":
        model_path = REPO_ROOT / "examples" / "iiwa_description" / "iiwa14.urdf"
        return {
            "model_path": model_path,
            "start_q": IIWA14_START_CONFIGS["zero"].astype(np.float32),
            "batch_sizes": [1],
            "f_ext": np.zeros(6, dtype=np.float32),
            "reference": figure8(dt, **FIG8_DEFAULT_PARAMS),
            "reference_metadata": {"reference_mode": "default"},
        }
    if plant == "tiago_right":
        model_path = REPO_ROOT / "gato" / "dynamics" / "tiago_right" / "tiago_right_arm.urdf"
        model = load_model(model_path)
        start_q = TIAGO_RIGHT_START_CONFIGS[tiago_start_config].astype(np.float32)
        start_ee, start_rpy = ee_pose_rpy(model, start_q.astype(np.float64), plant)
        reference, metadata = tiago_horizontal_figure8(dt, start_ee)
        orientation_target = None
        if orientation_rpy is not None:
            orientation_target = np.asarray(orientation_rpy, dtype=np.float64)
        elif track_start_orientation:
            orientation_target = start_rpy
        if orientation_target is not None:
            if orientation_target.shape != (3,):
                raise ValueError("orientation_rpy must contain exactly 3 values")
            reference = reference.reshape(-1, 6)
            reference[:, 3:6] = orientation_target
            reference = reference.reshape(-1)
        return {
            "model_path": model_path,
            "start_q": start_q,
            "batch_sizes": [1],
            "f_ext": np.zeros(6, dtype=np.float32),
            "reference": reference,
            "reference_metadata": {
                **metadata,
                "tiago_start_config": tiago_start_config,
                "tiago_start_ee": start_ee.tolist(),
                "orientation_tracking": orientation_target is not None,
                "orientation_rpy": (
                    orientation_target.tolist() if orientation_target is not None else None
                ),
            },
        }
    raise ValueError(f"Unsupported plant: {plant}")


def reference_at_timestamps(reference, timestamps, dt):
    points = reference.reshape(-1, 6)[:, :3]
    indices = np.minimum((timestamps / dt).astype(np.int64) + 1, points.shape[0] - 1)
    return points[indices], indices


def stack_stat_rows(rows, fill_value, dtype):
    rows = [np.asarray(row, dtype=dtype).reshape(-1) for row in rows]
    if not rows:
        return np.empty((0, 0), dtype=dtype)
    width = max(row.size for row in rows)
    out = np.full((len(rows), width), fill_value, dtype=dtype)
    for idx, row in enumerate(rows):
        out[idx, : row.size] = row
    return out


def summarize(stats, model, solver_params):
    errors = np.asarray(stats["goal_distances"], dtype=np.float64)
    solve_times = np.asarray(stats["solve_times"], dtype=np.float64)
    joints = np.asarray(stats["joint_positions"], dtype=np.float64)
    velocities = np.asarray(stats["joint_velocities"], dtype=np.float64)
    applied_controls = np.asarray(stats["applied_controls"], dtype=np.float64)
    planned_controls = np.asarray(stats["planned_controls"], dtype=np.float64)
    sqp_iters = np.asarray(stats.get("sqp_iters", []), dtype=np.float64)
    pcg_iters = stack_stat_rows(stats.get("pcg_iters", []), -1, np.int32)
    pcg_times_us = stack_stat_rows(stats.get("pcg_times_us", []), -1.0, np.float32)
    kkt_converged = np.asarray(stats.get("kkt_converged", []), dtype=np.int32)

    joint_min = np.min(joints, axis=0)
    joint_max = np.max(joints, axis=0)
    max_abs_velocity = np.max(np.abs(velocities), axis=0)
    lower = model.lowerPositionLimit.astype(np.float64)
    upper = model.upperPositionLimit.astype(np.float64)
    velocity_limit = model.velocityLimit.astype(np.float64)
    effort_limit = model.effortLimit.astype(np.float64)
    max_abs_applied_control = np.max(np.abs(applied_controls), axis=0)
    max_abs_planned_control = np.max(np.abs(planned_controls), axis=0)
    valid_pcg_iters = pcg_iters[pcg_iters >= 0]
    valid_pcg_times_us = pcg_times_us[pcg_times_us >= 0.0]

    return {
        "iterations": int(errors.size),
        "mean_error_m": float(np.mean(errors)),
        "p95_error_m": float(np.quantile(errors, 0.95)),
        "max_error_m": float(np.max(errors)),
        "final_error_m": float(errors[-1]),
        "mean_solve_time_ms": float(np.mean(solve_times)),
        "p95_solve_time_ms": float(np.quantile(solve_times, 0.95)),
        "mean_sqp_iters": float(np.mean(sqp_iters)) if sqp_iters.size else None,
        "kkt_converged_fraction": float(np.mean(kkt_converged > 0)) if kkt_converged.size else None,
        "linear_solves": int(valid_pcg_iters.size),
        "mean_pcg_iters": float(np.mean(valid_pcg_iters)) if valid_pcg_iters.size else None,
        "p95_pcg_iters": float(np.quantile(valid_pcg_iters, 0.95)) if valid_pcg_iters.size else None,
        "max_pcg_iters_observed": int(np.max(valid_pcg_iters)) if valid_pcg_iters.size else None,
        "pcg_cap_hit_fraction": float(np.mean(valid_pcg_iters >= solver_params["max_pcg_iters"])) if valid_pcg_iters.size else None,
        "mean_pcg_time_ms": float(np.mean(valid_pcg_times_us) / 1000.0) if valid_pcg_times_us.size else None,
        "p95_pcg_time_ms": float(np.quantile(valid_pcg_times_us, 0.95) / 1000.0) if valid_pcg_times_us.size else None,
        "max_joint_position_violation_rad": float(
            np.max(np.maximum(np.maximum(lower - joint_min, 0.0), np.maximum(joint_max - upper, 0.0)))
        ),
        "max_abs_joint_velocity_rad_s": [float(v) for v in max_abs_velocity],
        "joint_velocity_limit_rad_s": [float(v) for v in velocity_limit],
        "max_joint_velocity_violation_rad_s": float(np.max(np.maximum(max_abs_velocity - velocity_limit, 0.0))),
        "max_abs_applied_torque_nm": [float(v) for v in max_abs_applied_control],
        "max_abs_planned_torque_nm": [float(v) for v in max_abs_planned_control],
        "torque_limit_nm": [float(v) for v in effort_limit],
        "max_applied_torque_violation_nm": float(np.max(np.maximum(max_abs_applied_control - effort_limit, 0.0))),
        "max_planned_torque_violation_nm": float(np.max(np.maximum(max_abs_planned_control - effort_limit, 0.0))),
    }


def save_csvs(output_dir, batch_size, stats, reference_points, reference_indices):
    timestamps = np.asarray(stats["timestamps"], dtype=np.float64)
    actual = np.asarray(stats["ee_actual"], dtype=np.float64)[:, :3]
    errors = np.asarray(stats["goal_distances"], dtype=np.float64)
    solve_times = np.asarray(stats["solve_times"], dtype=np.float64)
    np.savetxt(
        output_dir / f"batch_{batch_size}_actual.csv",
        np.column_stack([timestamps, reference_indices, reference_points, actual, errors, solve_times]),
        delimiter=",",
        header="t,reference_index,ref_x,ref_y,ref_z,x,y,z,error,solve_time_ms",
        comments="",
    )

    joints = np.asarray(stats["joint_positions"], dtype=np.float64)
    velocities = np.asarray(stats["joint_velocities"], dtype=np.float64)
    header = ["t"] + [f"q{i}" for i in range(joints.shape[1])] + [f"qd{i}" for i in range(velocities.shape[1])]
    np.savetxt(
        output_dir / f"batch_{batch_size}_joints.csv",
        np.column_stack([timestamps, joints, velocities]),
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    applied_controls = np.asarray(stats["applied_controls"], dtype=np.float64)
    planned_controls = np.asarray(stats["planned_controls"], dtype=np.float64)
    header = ["t"] + [f"u{i}" for i in range(applied_controls.shape[1])]
    np.savetxt(
        output_dir / f"batch_{batch_size}_applied_controls.csv",
        np.column_stack([timestamps, applied_controls]),
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    np.savetxt(
        output_dir / f"batch_{batch_size}_planned_controls.csv",
        np.column_stack([timestamps, planned_controls]),
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    pcg_iters = stack_stat_rows(stats.get("pcg_iters", []), -1, np.int32)
    if pcg_iters.size:
        np.savetxt(
            output_dir / f"batch_{batch_size}_pcg_iters.csv",
            np.column_stack([timestamps, pcg_iters]),
            delimiter=",",
            fmt=["%.9g"] + ["%d"] * pcg_iters.shape[1],
            header="t," + ",".join([f"linear_solve_{i}" for i in range(pcg_iters.shape[1])]),
            comments="",
        )

    pcg_times_us = stack_stat_rows(stats.get("pcg_times_us", []), -1.0, np.float32)
    if pcg_times_us.size:
        np.savetxt(
            output_dir / f"batch_{batch_size}_pcg_times_us.csv",
            np.column_stack([timestamps, pcg_times_us]),
            delimiter=",",
            header="t," + ",".join([f"linear_solve_{i}" for i in range(pcg_times_us.shape[1])]),
            comments="",
        )


def save_renderings(output_dir, plant, reference, results, make_gif):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    ref = reference.reshape(-1, 6)[:, :3]
    batch_sizes = sorted(results)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, dims, title in ((axes[0], (0, 2), "XZ"), (axes[1], (0, 1), "XY")):
        ax.plot(ref[:, dims[0]], ref[:, dims[1]], ":", color="#747474", linewidth=1.3, label="reference")
        for batch_size in batch_sizes:
            actual = np.asarray(results[batch_size]["stats"]["ee_actual"], dtype=np.float64)[:, :3]
            ax.plot(
                actual[:, dims[0]],
                actual[:, dims[1]],
                color=BATCH_COLORS.get(batch_size, "#000000"),
                linewidth=1.5,
                label=f"batch {batch_size}",
            )
        ax.set_xlabel(f"{'xyz'[dims[0]]} [m]")
        ax.set_ylabel(f"{'xyz'[dims[1]]} [m]")
        ax.set_title(title)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle(f"{plant} figure-8 tracking")
    fig.tight_layout()
    fig.savefig(output_dir / "tracking_projections.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], ":", color="#747474", linewidth=1.3, label="reference")
    for batch_size in batch_sizes:
        actual = np.asarray(results[batch_size]["stats"]["ee_actual"], dtype=np.float64)[:, :3]
        color = BATCH_COLORS.get(batch_size, "#000000")
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], color=color, linewidth=1.5, label=f"batch {batch_size}")
        ax.scatter(actual[0, 0], actual[0, 1], actual[0, 2], color=color, marker="o", s=40)
    ax.scatter(ref[0, 0], ref[0, 1], ref[0, 2], color="#C90016", marker="x", s=70, label="reference start")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"{plant} figure-8 tracking")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "tracking_3d.png", dpi=160)
    plt.close(fig)

    if not make_gif or not batch_sizes:
        return

    batch_size = batch_sizes[0]
    stats = results[batch_size]["stats"]
    actual = np.asarray(stats["ee_actual"], dtype=np.float64)[:, :3]
    timestamps = np.asarray(stats["timestamps"], dtype=np.float64)
    targets = results[batch_size]["reference_points"]
    frame_ids = np.linspace(0, actual.shape[0] - 1, min(300, actual.shape[0])).astype(np.int64)

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], ":", color="#747474", linewidth=1.2)
    trail, = ax.plot([], [], [], color="#003192", linewidth=1.7)
    actual_point, = ax.plot([], [], [], "o", color="#003192", markersize=5)
    target_point, = ax.plot([], [], [], "x", color="#C90016", markersize=7)

    all_points = np.concatenate([ref, actual], axis=0)
    center = np.mean(all_points, axis=0)
    radius = max(0.2, float(np.max(np.ptp(all_points, axis=0))) * 0.60)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=24, azim=-54)
    fig.tight_layout()

    def update(frame_index):
        idx = frame_ids[frame_index]
        trail.set_data(actual[: idx + 1, 0], actual[: idx + 1, 1])
        trail.set_3d_properties(actual[: idx + 1, 2])
        actual_point.set_data([actual[idx, 0]], [actual[idx, 1]])
        actual_point.set_3d_properties([actual[idx, 2]])
        target_point.set_data([targets[idx, 0]], [targets[idx, 1]])
        target_point.set_3d_properties([targets[idx, 2]])
        ax.set_title(f"{plant} batch {batch_size} | t={timestamps[idx]:.2f}s")
        return trail, actual_point, target_point

    anim = animation.FuncAnimation(fig, update, frames=frame_ids.size, interval=50, blit=False)
    anim.save(output_dir / f"batch_{batch_size}_tracking.gif", writer=animation.PillowWriter(fps=20))
    plt.close(fig)


def run(args):
    cfg = plant_config(
        args.plant,
        args.dt,
        tiago_start_config=args.tiago_start_config,
        track_start_orientation=args.track_start_orientation,
        orientation_rpy=args.ee_orientation_rpy,
    )
    output_dir = args.output_root / args.plant
    if args.output_label:
        output_dir = output_dir / args.output_label
    ros_controller = None
    if args.ros_tiago:
        if args.plant != "tiago_right":
            raise ValueError("--ros-tiago requires --plant tiago_right")
        from gato_tiago.tiago_controller_process import TiagoControllerOrchestrator

        ros_controller = TiagoControllerOrchestrator(
            target_hz=args.ros_target_hz,
            reset_q=cfg["start_q"],
            reset_duration_sec=args.ros_reset_duration,
            stale_timeout_sec=args.ros_stale_timeout,
            max_abs_torque=args.ros_max_abs_torque,
            clamp_torque=args.ros_clamp_torque,
            collision_safety_enabled=not args.ros_disable_collision_safety,
            collision_min_distance_m=args.ros_collision_min_distance,
            collision_check_timeout_sec=args.ros_collision_check_timeout,
            collision_max_monitored_geometry_speed_m_s=args.ros_collision_max_monitored_geometry_speed,
            collision_blacklist_path=args.ros_collision_blacklist,
            joint_position_margin_rad=args.ros_joint_position_margin_rad,
            joint_velocity_scale=args.ros_joint_velocity_scale,
        )

    if args.batch_sizes is not None:
        cfg["batch_sizes"] = args.batch_sizes
    solver_params = (
        dict(TIAGO_TRACKING_SOLVER_PARAMS)
        if args.plant == "tiago_right"
        else dict(DEFAULT_SOLVER_PARAMS)
    )
    if args.vel_lim_cost is not None:
        solver_params["vel_lim_cost"] = args.vel_lim_cost
    if args.u_cost is not None:
        solver_params["u_cost"] = args.u_cost
    if args.q_lim_cost is not None:
        solver_params["q_lim_cost"] = args.q_lim_cost
    if args.ctrl_lim_cost is not None:
        solver_params["ctrl_lim_cost"] = args.ctrl_lim_cost
    if args.track_start_orientation or args.ee_orientation_rpy is not None:
        solver_params["ee_orient_cost"] = args.ee_orient_cost
        solver_params["ee_orient_N_cost"] = args.ee_orient_N_cost

    model = load_model(cfg["model_path"])
    reference = cfg["reference"].astype(np.float32)
    x_start = np.concatenate([cfg["start_q"], np.zeros(model.nv, dtype=np.float32)]).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        output_dir / "reference.csv",
        reference.reshape(-1, 6)[:, :3],
        delimiter=",",
        header="x,y,z",
        comments="",
    )

    results = {}
    summaries = {}
    controller_state_summary = None
    try:
        for batch_size in cfg["batch_sizes"]:
            mpc = MPC_GATO(
                model=model,
                model_path=str(cfg["model_path"]),
                N=args.N,
                dt=args.dt,
                batch_size=batch_size,
                constant_f_ext=cfg["f_ext"],
                track_full_stats=True,
                plant_type=args.plant,
                solver_params=solver_params,
            )
            mpc.force_estimator = None
            _, stats = mpc.run_mpc_fig8(
                x_start=x_start,
                fig8_traj=reference,
                sim_dt=args.sim_dt,
                sim_time=args.sim_time,
                controller=ros_controller,
                controller_timeout=args.ros_controller_timeout,
            )
            timestamps = np.asarray(stats["timestamps"], dtype=np.float64)
            ref_points, ref_indices = reference_at_timestamps(reference, timestamps, args.dt)
            save_csvs(output_dir, batch_size, stats, ref_points, ref_indices)
            summaries[str(batch_size)] = summarize(stats, model, solver_params)
            results[batch_size] = {"stats": stats, "reference_points": ref_points}
    finally:
        if ros_controller is not None:
            ros_controller.close(timeout_sec=args.ros_controller_timeout)
            controller_state_summary = ros_controller.write_state_history_csv(
                output_dir / "controller_state_history.csv"
            )
            ros_controller.write_full_state_history_jsonl(
                output_dir / "full_joint_state_history.jsonl"
            )

    save_renderings(output_dir, args.plant, reference, results, make_gif=not args.no_gif)

    payload = {
        "plant": args.plant,
        "model_path": str(cfg["model_path"]),
        "N": args.N,
        "dt": args.dt,
        "sim_dt": args.sim_dt,
        "sim_time": args.sim_time,
        "batch_sizes": cfg["batch_sizes"],
        "constant_f_ext": [float(v) for v in cfg["f_ext"]],
        "solver_params": solver_params,
        "ros_tiago": bool(args.ros_tiago),
        "ros_clamp_torque": bool(args.ros_clamp_torque),
        "timestamp_source": (
            "joint_state_header_stamp_elapsed" if args.ros_tiago else "offline_simulation_time"
        ),
        "reference_extent_m": [float(v) for v in np.ptp(reference.reshape(-1, 6)[:, :3], axis=0)],
        "reference_metadata": cfg["reference_metadata"],
        "summary_by_batch": summaries,
        "controller_state_history": controller_state_summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"plant: {args.plant}")
    print(f"output_dir: {output_dir}")
    for batch_size, summary in summaries.items():
        print(
            f"batch {batch_size}: mean_error={summary['mean_error_m']:.6f}m "
            f"max_error={summary['max_error_m']:.6f}m "
            f"mean_solve={summary['mean_solve_time_ms']:.3f}ms"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant", choices=("indy7", "iiwa14", "tiago_right"), required=True)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.008)
    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--sim-time", type=float, default=16.0)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument(
        "--tiago-start-config",
        choices=tuple(TIAGO_RIGHT_START_CONFIGS),
        default=TIAGO_RIGHT_DEFAULT_START_CONFIG,
        help="Named Tiago right-arm start pose used when --plant tiago_right.",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-label", default=None)
    parser.add_argument("--vel-lim-cost", type=float, default=None)
    parser.add_argument("--u-cost", type=float, default=None)
    parser.add_argument("--q-lim-cost", type=float, default=None)
    parser.add_argument("--ctrl-lim-cost", type=float, default=None)
    parser.add_argument(
        "--track-start-orientation",
        action="store_true",
        help="Track the Tiago start-pose tool RPY as a constant figure-8 orientation target.",
    )
    parser.add_argument(
        "--ee-orientation-rpy",
        nargs=3,
        type=float,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Constant figure-8 EE orientation target. For local tool z-axis up, use 0 0 0.",
    )
    parser.add_argument("--ee-orient-cost", type=float, default=0.2)
    parser.add_argument("--ee-orient-N-cost", type=float, default=4.0)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--ros-tiago", action="store_true")
    parser.add_argument("--ros-target-hz", type=float, default=125.0)
    parser.add_argument("--ros-reset-duration", type=float, default=2.0)
    parser.add_argument("--ros-stale-timeout", type=float, default=0.1)
    parser.add_argument("--ros-max-abs-torque", type=float, default=30.0)
    parser.add_argument("--ros-clamp-torque", action="store_true")
    parser.add_argument("--ros-disable-collision-safety", action="store_true")
    parser.add_argument("--ros-collision-min-distance", type=float, default=0.04)
    parser.add_argument("--ros-collision-check-timeout", type=float, default=0.05)
    parser.add_argument("--ros-collision-max-monitored-geometry-speed", type=float, default=1.0)
    parser.add_argument("--ros-collision-blacklist", type=Path, default=None)
    parser.add_argument("--ros-joint-position-margin-rad", type=float, default=0.0)
    parser.add_argument("--ros-joint-velocity-scale", type=float, default=1.0)
    parser.add_argument("--ros-controller-timeout", type=float, default=8.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
