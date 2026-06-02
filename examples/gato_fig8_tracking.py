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

from bsqp.common import figure8
from bsqp.config import (
    BATCH_COLORS,
    DEFAULT_SOLVER_PARAMS,
    FIG8_DEFAULT_PARAMS,
    IIWA14_START_CONFIGS,
    INDY7_START_CONFIGS,
    TIAGO_RIGHT_START_CONFIGS,
)
from bsqp.mpc_controller import MPC_GATO


OUTPUT_ROOT = REPO_ROOT / "example_artifacts" / "gato_fig8_tracking"
TIAGO_FIRST_REFERENCE = np.array([0.557576, -0.672736, -0.276456], dtype=np.float64)


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


def plant_config(plant, dt):
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
        start_q = TIAGO_RIGHT_START_CONFIGS["comfortable"].astype(np.float32)
        start_ee = ee_position(model, start_q.astype(np.float64), plant)
        reference, metadata = transformed_indy7_figure8(dt, start_ee)
        return {
            "model_path": model_path,
            "start_q": start_q,
            "batch_sizes": [1],
            "f_ext": np.zeros(6, dtype=np.float32),
            "reference": reference,
            "reference_metadata": {**metadata, "tiago_start_ee": start_ee.tolist()},
        }
    raise ValueError(f"Unsupported plant: {plant}")


def reference_at_timestamps(reference, timestamps, dt):
    points = reference.reshape(-1, 6)[:, :3]
    indices = np.minimum((timestamps / dt).astype(np.int64) + 1, points.shape[0] - 1)
    return points[indices], indices


def summarize(stats, model):
    errors = np.asarray(stats["goal_distances"], dtype=np.float64)
    solve_times = np.asarray(stats["solve_times"], dtype=np.float64)
    joints = np.asarray(stats["joint_positions"], dtype=np.float64)
    velocities = np.asarray(stats["joint_velocities"], dtype=np.float64)
    sqp_iters = np.asarray(stats.get("sqp_iters", []), dtype=np.float64)

    joint_min = np.min(joints, axis=0)
    joint_max = np.max(joints, axis=0)
    max_abs_velocity = np.max(np.abs(velocities), axis=0)
    lower = model.lowerPositionLimit.astype(np.float64)
    upper = model.upperPositionLimit.astype(np.float64)
    velocity_limit = model.velocityLimit.astype(np.float64)

    return {
        "iterations": int(errors.size),
        "mean_error_m": float(np.mean(errors)),
        "p95_error_m": float(np.quantile(errors, 0.95)),
        "max_error_m": float(np.max(errors)),
        "final_error_m": float(errors[-1]),
        "mean_solve_time_ms": float(np.mean(solve_times)),
        "p95_solve_time_ms": float(np.quantile(solve_times, 0.95)),
        "mean_sqp_iters": float(np.mean(sqp_iters)) if sqp_iters.size else None,
        "max_joint_position_violation_rad": float(
            np.max(np.maximum(np.maximum(lower - joint_min, 0.0), np.maximum(joint_max - upper, 0.0)))
        ),
        "max_abs_joint_velocity_rad_s": [float(v) for v in max_abs_velocity],
        "joint_velocity_limit_rad_s": [float(v) for v in velocity_limit],
        "max_joint_velocity_violation_rad_s": float(np.max(np.maximum(max_abs_velocity - velocity_limit, 0.0))),
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
    cfg = plant_config(args.plant, args.dt)
    if args.batch_sizes is not None:
        cfg["batch_sizes"] = args.batch_sizes

    model = load_model(cfg["model_path"])
    reference = cfg["reference"].astype(np.float32)
    x_start = np.concatenate([cfg["start_q"], np.zeros(model.nv, dtype=np.float32)]).astype(np.float32)
    output_dir = args.output_root / args.plant
    if args.output_label:
        output_dir = output_dir / args.output_label
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
        )
        mpc.force_estimator = None
        _, stats = mpc.run_mpc_fig8(x_start=x_start, fig8_traj=reference, sim_dt=args.sim_dt, sim_time=args.sim_time)
        timestamps = np.asarray(stats["timestamps"], dtype=np.float64)
        ref_points, ref_indices = reference_at_timestamps(reference, timestamps, args.dt)
        save_csvs(output_dir, batch_size, stats, ref_points, ref_indices)
        summaries[str(batch_size)] = summarize(stats, model)
        results[batch_size] = {"stats": stats, "reference_points": ref_points}

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
        "solver_params": dict(DEFAULT_SOLVER_PARAMS),
        "reference_extent_m": [float(v) for v in np.ptp(reference.reshape(-1, 6)[:, :3], axis=0)],
        "reference_metadata": cfg["reference_metadata"],
        "summary_by_batch": summaries,
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
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--sim-time", type=float, default=16.0)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-label", default=None)
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
