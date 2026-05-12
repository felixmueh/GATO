#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "python"))

from bsqp.common import figure8
from bsqp.config import BATCH_COLORS, FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS, TIAGO_RIGHT_START_CONFIGS
from bsqp.mpc_controller import MPC_GATO


OUTPUT_ROOT = REPO_ROOT / "test-artifacts" / "gato_fig8_tracking"

TIAGO_SOLVER_PARAMS = {
    "max_sqp_iters": 20,
    "kkt_tol": 1e-3,
    "max_pcg_iters": 120,
    "pcg_tol": 1e-3,
    "solve_ratio": 1.0,
    "mu": 1.0,
    "q_cost": 160.0,
    "qd_cost": 1e-2,
    "u_cost": 1e-5,
    "N_cost": 50.0,
    "q_lim_cost": 0.01,
    "vel_lim_cost": 0.0,
    "ctrl_lim_cost": 0.0,
    "rho": 0.01,
}


def rotation_from_vectors(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.dot(source, target))
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        axis = np.zeros(3)
        axis[np.argmin(np.abs(source))] = 1.0
        cross = np.cross(source, axis)
        cross = cross / np.linalg.norm(cross)
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


def tiago_tool_position(model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    torso_id = model.getFrameId("torso_lift_link")
    tool_id = model.getFrameId("arm_right_tool_link")
    return (data.oMf[torso_id].inverse() * data.oMf[tool_id]).translation.copy()


def indy7_start_offset():
    model = pin.buildModelFromUrdf(str(REPO_ROOT / "examples" / "indy7_description" / "indy7.urdf"))
    data = model.createData()
    q_start = INDY7_START_CONFIGS["zero"].astype(np.float64)
    pin.forwardKinematics(model, data, q_start)
    actual_start = data.oMi[model.njoints - 1].translation.copy()
    indy_ref = figure8(0.01, **FIG8_DEFAULT_PARAMS).reshape(-1, 6)[:, :3]
    return indy_ref[0] - actual_start


def transformed_indy7_figure8(dt, tiago_start, tiago_first_reference):
    source = figure8(dt, **FIG8_DEFAULT_PARAMS).reshape(-1, 6)
    source_points = source[:, :3]
    source_first = source_points[0].copy()
    source_delta = indy7_start_offset()
    target_delta = np.asarray(tiago_first_reference, dtype=np.float64) - np.asarray(tiago_start, dtype=np.float64)
    if not np.isclose(np.linalg.norm(source_delta), np.linalg.norm(target_delta), rtol=0.0, atol=1e-5):
        raise ValueError(
            "tiago_first_reference must preserve the Indy7 start offset length "
            f"({np.linalg.norm(source_delta):.6f} m)"
        )
    rotation = rotation_from_vectors(source_delta, target_delta)
    transformed = source.copy()
    transformed[:, :3] = (rotation @ (source_points - source_first).T).T + tiago_first_reference
    return transformed.reshape(-1), rotation, source_delta, target_delta


def experiment_config(plant, tiago_reference_offset=None, tiago_first_reference=None, solver_overrides=None):
    common = {
        "N": 32,
        "dt": 0.01,
        "sim_time": 16.0,
        "sim_dt": 0.001,
    }
    if plant == "indy7":
        return {
            **common,
            "plant_type": "indy7",
            "batch_sizes": [1, 32, 128],
            "model_path": REPO_ROOT / "examples" / "indy7_description" / "indy7.urdf",
            "model_dir": REPO_ROOT / "examples" / "indy7_description",
            "start_q": INDY7_START_CONFIGS["zero"].astype(np.float32),
            "f_ext": np.array([0.0, 0.0, -60.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "fig8_params": dict(FIG8_DEFAULT_PARAMS),
            "solver_params": None,
            "reset_dual_each_tick": False,
            "warm_start_policy": "previous_solution",
        }
    if plant == "tiago_right":
        solver_params = dict(TIAGO_SOLVER_PARAMS)
        if solver_overrides:
            solver_params.update(solver_overrides)
        model_path = REPO_ROOT / "gato" / "dynamics" / "tiago_right" / "tiago_right_arm.urdf"
        model = pin.buildModelFromUrdf(str(model_path))
        data = model.createData()
        start_q = TIAGO_RIGHT_START_CONFIGS["comfortable"].astype(np.float32)
        center = tiago_tool_position(model, data, start_q)
        amplitude_z = 0.04
        custom_fig8_traj = None
        transform_rotation = None
        indy_start_delta = None
        tiago_start_delta = None
        reference_mode = "local_small"
        fig8_params = {
            **FIG8_DEFAULT_PARAMS,
            "A_x": 0.04,
            "A_z": amplitude_z,
            "offset": [float(center[0]), float(center[1]), float(center[2] - amplitude_z / 2.0)],
            "theta": 0.0,
        }
        if tiago_reference_offset is not None:
            center = center + np.asarray(tiago_reference_offset, dtype=np.float64)
            fig8_params["offset"] = [float(center[0]), float(center[1]), float(center[2] - amplitude_z / 2.0)]
            reference_mode = "local_small_offset"
        if tiago_first_reference is not None:
            custom_fig8_traj, transform_rotation, indy_start_delta, tiago_start_delta = transformed_indy7_figure8(
                common["dt"],
                center,
                np.asarray(tiago_first_reference, dtype=np.float64),
            )
            fig8_params = dict(FIG8_DEFAULT_PARAMS)
            reference_mode = "transformed_indy7"
        return {
            **common,
            "plant_type": "tiago_right",
            "batch_sizes": [1],
            "model_path": model_path,
            "model_dir": model_path.parent,
            "start_q": start_q,
            "f_ext": np.zeros(6, dtype=np.float32),
            "fig8_params": fig8_params,
            "custom_fig8_traj": custom_fig8_traj,
            "reference_mode": reference_mode,
            "transform_rotation": transform_rotation.tolist() if transform_rotation is not None else None,
            "indy_start_delta": indy_start_delta.tolist() if indy_start_delta is not None else None,
            "tiago_start_delta": tiago_start_delta.tolist() if tiago_start_delta is not None else None,
            "solver_params": solver_params,
            "reset_dual_each_tick": True,
            "warm_start_policy": "repeat_current",
            "tiago_reference_offset": (
                [float(v) for v in tiago_reference_offset]
                if tiago_reference_offset is not None
                else [0.0, 0.0, 0.0]
            ),
        }
    raise ValueError(f"unsupported plant: {plant}")


def run_plant(
    plant,
    no_media=False,
    output_label=None,
    tiago_reference_offset=None,
    tiago_first_reference=None,
    solver_overrides=None,
    control_timestep_policy="solve_time",
    saturate_controls=False,
):
    cfg = experiment_config(
        plant,
        tiago_reference_offset=tiago_reference_offset,
        tiago_first_reference=tiago_first_reference,
        solver_overrides=solver_overrides,
    )
    if plant == "tiago_right":
        model = pin.buildModelFromUrdf(str(cfg["model_path"]))
    else:
        model, _, _ = pin.buildModelsFromUrdf(str(cfg["model_path"]), str(cfg["model_dir"]))
    cfg["position_lower"] = model.lowerPositionLimit.astype(float)
    cfg["position_upper"] = model.upperPositionLimit.astype(float)
    cfg["velocity_limit"] = np.full(model.nv, 2.5, dtype=np.float64) if plant == "tiago_right" else np.full(model.nv, np.inf)
    cfg["control_limit"] = (
        np.array([43.0, 43.0, 26.0, 26.0, 26.0, 26.0, 26.0], dtype=np.float64)
        if plant == "tiago_right"
        else np.full(model.nv, np.inf)
    )
    fig8_traj = cfg.get("custom_fig8_traj")
    if fig8_traj is None:
        fig8_traj = figure8(cfg["dt"], **cfg["fig8_params"])
    ref_points = fig8_traj.reshape(-1, 6)[:, :3]
    x_start = np.hstack((cfg["start_q"], np.zeros(model.nv, dtype=np.float32)))
    output_dir = OUTPUT_ROOT / plant
    if output_label:
        output_dir = output_dir / output_label
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    summaries = {}
    for batch_size in cfg["batch_sizes"]:
        mpc = MPC_GATO(
            model=model,
            model_path=str(cfg["model_path"]),
            N=cfg["N"],
            dt=cfg["dt"],
            batch_size=batch_size,
            constant_f_ext=cfg["f_ext"],
            track_full_stats=True,
            plant_type=cfg["plant_type"],
            solver_params=cfg["solver_params"],
            reset_dual_each_tick=cfg["reset_dual_each_tick"],
            warm_start_policy=cfg["warm_start_policy"],
            control_timestep_policy=control_timestep_policy,
            saturate_controls=saturate_controls,
        )
        _, stats = mpc.run_mpc_fig8(
            x_start=x_start,
            fig8_traj=fig8_traj,
            sim_dt=cfg["sim_dt"],
            sim_time=cfg["sim_time"],
        )
        stats["linsys_solver"] = mpc.solver.linsys_solver
        results[batch_size] = stats
        summaries[str(batch_size)] = summarize(stats, cfg)
        save_batch_csv(output_dir, batch_size, stats)
        save_joint_csv(output_dir, batch_size, stats)
        save_control_csv(output_dir, batch_size, stats, "applied_controls")
        save_control_csv(output_dir, batch_size, stats, "planned_controls")

    np.savetxt(
        output_dir / "reference.csv",
        ref_points,
        delimiter=",",
        header="x,y,z",
        comments="",
    )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "plant": plant,
                "config": {
                    "N": cfg["N"],
                    "dt": cfg["dt"],
                    "sim_time": cfg["sim_time"],
                    "sim_dt": cfg["sim_dt"],
                    "batch_sizes": cfg["batch_sizes"],
                    "model_path": str(cfg["model_path"]),
                    "fig8_params": cfg["fig8_params"],
                    "reference_mode": cfg.get("reference_mode"),
                    "transform_rotation": cfg.get("transform_rotation"),
                    "indy_start_delta": cfg.get("indy_start_delta"),
                    "tiago_start_delta": cfg.get("tiago_start_delta"),
                    "reference_extent_m": [float(v) for v in np.ptp(ref_points, axis=0)],
                    "solver_params": cfg["solver_params"],
                    "reset_dual_each_tick": cfg["reset_dual_each_tick"],
                    "warm_start_policy": cfg["warm_start_policy"],
                    "control_timestep_policy": control_timestep_policy,
                    "saturate_controls": saturate_controls,
                    "tiago_reference_offset": cfg.get("tiago_reference_offset"),
                },
                "summary_by_batch": summaries,
            },
            f,
            indent=2,
        )

    plot_path = None
    if not no_media:
        plot_path = save_plot(output_dir, plant, ref_points, results)

    return output_dir, plot_path, summaries


def summarize(stats, cfg):
    errors = np.asarray(stats["goal_distances"], dtype=np.float64)
    solve_times = np.asarray(stats["solve_times"], dtype=np.float64)
    sqp_iters = np.asarray(stats.get("sqp_iters", []), dtype=np.float64)
    joint_positions = np.asarray(stats["joint_positions"], dtype=np.float64)
    joint_velocities = np.asarray(stats["joint_velocities"], dtype=np.float64)
    controls = np.asarray(stats["applied_controls"], dtype=np.float64)
    planned_controls = np.asarray(stats["planned_controls"], dtype=np.float64)
    joint_min = np.min(joint_positions, axis=0)
    joint_max = np.max(joint_positions, axis=0)
    max_abs_velocity = np.max(np.abs(joint_velocities), axis=0)
    max_abs_control = np.max(np.abs(controls), axis=0)
    max_abs_planned_control = np.max(np.abs(planned_controls), axis=0)
    lower = np.asarray(cfg["position_lower"], dtype=np.float64)
    upper = np.asarray(cfg["position_upper"], dtype=np.float64)
    velocity_limit = np.asarray(cfg["velocity_limit"], dtype=np.float64)
    control_limit = np.asarray(cfg["control_limit"], dtype=np.float64)
    lower_violation = np.maximum(lower - joint_min, 0.0)
    upper_violation = np.maximum(joint_max - upper, 0.0)
    max_position_violation = float(np.max(np.maximum(lower_violation, upper_violation)))
    max_velocity_violation = float(np.max(np.maximum(max_abs_velocity - velocity_limit, 0.0)))
    max_control_violation = float(np.max(np.maximum(max_abs_control - control_limit, 0.0)))
    max_planned_control_violation = float(np.max(np.maximum(max_abs_planned_control - control_limit, 0.0)))

    timestamps = np.asarray(stats["timestamps"], dtype=np.float64)
    if timestamps.size > 1:
        dt = np.diff(timestamps)
        accelerations = np.diff(joint_velocities, axis=0) / dt[:, None]
        max_abs_acceleration = np.max(np.abs(accelerations), axis=0)
    else:
        max_abs_acceleration = np.full(joint_velocities.shape[1], float("nan"))

    return {
        "iterations": int(len(stats["timestamps"])),
        "mean_error_m": float(np.mean(errors)),
        "p95_error_m": float(np.quantile(errors, 0.95)),
        "max_error_m": float(np.max(errors)),
        "final_error_m": float(errors[-1]),
        "mean_solve_time_ms": float(np.mean(solve_times)),
        "p95_solve_time_ms": float(np.quantile(solve_times, 0.95)),
        "mean_sqp_iters": float(np.mean(sqp_iters)) if sqp_iters.size else None,
        "linsys_solver": stats.get("linsys_solver"),
        "joint_position_min_rad": [float(v) for v in joint_min],
        "joint_position_max_rad": [float(v) for v in joint_max],
        "joint_position_lower_limit_rad": [float(v) for v in lower],
        "joint_position_upper_limit_rad": [float(v) for v in upper],
        "max_joint_position_violation_rad": max_position_violation,
        "max_abs_joint_velocity_rad_s": [float(v) for v in max_abs_velocity],
        "joint_velocity_limit_rad_s": [float(v) for v in velocity_limit],
        "max_joint_velocity_violation_rad_s": max_velocity_violation,
        "max_abs_joint_acceleration_rad_s2_finite_difference": [float(v) for v in max_abs_acceleration],
        "max_abs_applied_torque_nm": [float(v) for v in max_abs_control],
        "max_abs_planned_torque_nm": [float(v) for v in max_abs_planned_control],
        "applied_torque_limit_nm": [float(v) for v in control_limit],
        "max_applied_torque_violation_nm": max_control_violation,
        "max_planned_torque_violation_nm": max_planned_control_violation,
    }


def save_batch_csv(output_dir, batch_size, stats):
    timestamps = np.asarray(stats["timestamps"], dtype=np.float64)
    actual = np.asarray(stats["ee_actual"], dtype=np.float64)[:, :3]
    errors = np.asarray(stats["goal_distances"], dtype=np.float64)
    solve_times = np.asarray(stats["solve_times"], dtype=np.float64)
    out = np.column_stack([timestamps, actual, errors, solve_times])
    np.savetxt(
        output_dir / f"batch_{batch_size}_actual.csv",
        out,
        delimiter=",",
        header="t,x,y,z,error,solve_time_ms",
        comments="",
    )


def save_joint_csv(output_dir, batch_size, stats):
    timestamps = np.asarray(stats["timestamps"], dtype=np.float64)
    joint_positions = np.asarray(stats["joint_positions"], dtype=np.float64)
    joint_velocities = np.asarray(stats["joint_velocities"], dtype=np.float64)
    out = np.column_stack([timestamps, joint_positions, joint_velocities])
    nq = joint_positions.shape[1]
    nv = joint_velocities.shape[1]
    header = ["t"] + [f"q{i}" for i in range(nq)] + [f"qd{i}" for i in range(nv)]
    np.savetxt(
        output_dir / f"batch_{batch_size}_joints.csv",
        out,
        delimiter=",",
        header=",".join(header),
        comments="",
    )


def save_control_csv(output_dir, batch_size, stats, key):
    timestamps = np.asarray(stats["timestamps"], dtype=np.float64)
    controls = np.asarray(stats[key], dtype=np.float64)
    out = np.column_stack([timestamps, controls])
    header = ["t"] + [f"u{i}" for i in range(controls.shape[1])]
    stem = "applied_controls" if key == "applied_controls" else "planned_controls"
    np.savetxt(
        output_dir / f"batch_{batch_size}_{stem}.csv",
        out,
        delimiter=",",
        header=",".join(header),
        comments="",
    )


def save_plot(output_dir, plant, ref_points, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batch_sizes = sorted(results.keys())
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(4.0 * len(batch_sizes), 4.2), squeeze=False)
    axes = axes[0]
    for idx, batch_size in enumerate(batch_sizes):
        ax = axes[idx]
        ax.plot(ref_points[:, 0], ref_points[:, 2], ":", linewidth=1.0, alpha=0.5, label="Reference")
        ee_actual = np.asarray(results[batch_size]["ee_actual"], dtype=np.float64)
        color = BATCH_COLORS.get(batch_size, "#000000")
        ax.plot(
            ee_actual[:, 0],
            ee_actual[:, 2],
            color=color,
            linewidth=1.5,
            label=f"Batch Size = {batch_size}",
            alpha=0.8,
        )
        ax.set_xlabel("X [m]")
        if idx == 0:
            ax.set_ylabel("Z [m]")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(loc="best")
    fig.suptitle(f"{plant} GATO figure-8 tracking")
    fig.tight_layout()
    plot_path = output_dir / "tracking_xz.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    fig_3d = plt.figure(figsize=(7.0, 6.0))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], ":", linewidth=1.4, alpha=0.6, label="Reference")
    for batch_size in batch_sizes:
        ee_actual = np.asarray(results[batch_size]["ee_actual"], dtype=np.float64)
        color = BATCH_COLORS.get(batch_size, "#000000")
        ax_3d.plot(ee_actual[:, 0], ee_actual[:, 1], ee_actual[:, 2], color=color, linewidth=1.5, label=f"Batch Size = {batch_size}")
        ax_3d.scatter(ee_actual[0, 0], ee_actual[0, 1], ee_actual[0, 2], color=color, marker="o", s=45)
    ax_3d.scatter(ref_points[0, 0], ref_points[0, 1], ref_points[0, 2], color="#C90016", marker="x", s=70, label="Reference start")
    ax_3d.set_xlabel("X [m]")
    ax_3d.set_ylabel("Y [m]")
    ax_3d.set_zlabel("Z [m]")
    ax_3d.set_title(f"{plant} GATO figure-8 tracking")
    ax_3d.legend(loc="best")
    fig_3d.tight_layout()
    fig_3d.savefig(output_dir / "tracking_3d.png", dpi=160)
    plt.close(fig_3d)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Run the GATO figure-8 notebook experiment and save review artifacts.")
    parser.add_argument("--plant", choices=["indy7", "tiago_right"], required=True)
    parser.add_argument("--output-label", default=None)
    parser.add_argument(
        "--tiago-reference-offset",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        default=None,
        help="Shift Tiago's figure-8 start/reference center away from the current tool pose.",
    )
    parser.add_argument(
        "--tiago-first-reference",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Use the exact Indy7 figure-8 shape transformed so Tiago's first reference is this point.",
    )
    parser.add_argument("--u-cost", type=float, default=None)
    parser.add_argument("--q-cost", type=float, default=None)
    parser.add_argument("--N-cost", type=float, default=None)
    parser.add_argument("--qd-cost", type=float, default=None)
    parser.add_argument("--vel-lim-cost", type=float, default=None)
    parser.add_argument("--ctrl-lim-cost", type=float, default=None)
    parser.add_argument("--control-timestep-policy", choices=["solve_time", "fixed_dt"], default="solve_time")
    parser.add_argument("--saturate-controls", action="store_true")
    parser.add_argument("--no-media", action="store_true")
    args = parser.parse_args()

    solver_overrides = {}
    for key, value in (
        ("u_cost", args.u_cost),
        ("q_cost", args.q_cost),
        ("N_cost", args.N_cost),
        ("qd_cost", args.qd_cost),
        ("vel_lim_cost", args.vel_lim_cost),
        ("ctrl_lim_cost", args.ctrl_lim_cost),
    ):
        if value is not None:
            solver_overrides[key] = value

    output_dir, plot_path, summaries = run_plant(
        args.plant,
        no_media=args.no_media,
        output_label=args.output_label,
        tiago_reference_offset=args.tiago_reference_offset,
        tiago_first_reference=args.tiago_first_reference,
        solver_overrides=solver_overrides,
        control_timestep_policy=args.control_timestep_policy,
        saturate_controls=args.saturate_controls,
    )
    print(f"plant: {args.plant}")
    for batch_size, summary in summaries.items():
        print(
            f"batch {batch_size}: mean_error={summary['mean_error_m']:.6f}m "
            f"max_error={summary['max_error_m']:.6f}m "
            f"mean_solve={summary['mean_solve_time_ms']:.3f}ms"
        )
    print(f"output_dir: {output_dir}")
    if plot_path is not None:
        print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
