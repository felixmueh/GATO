#!/usr/bin/env python3
"""Run and plot a simple Tiago right-arm MPC reach-to-target experiment."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

PROJECT_ROOT = Path(".")
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "tiago_src"))


MODEL_PATH = Path("gato") / "dynamics" / "tiago_right" / "tiago_right_arm.urdf"
OUTPUT_ROOT = Path("example_artifacts") / "tiago_reach_target"
DEFAULT_TARGET_OFFSET = np.array([0.11, -0.085, 0.035], dtype=np.float64)
DEFAULT_LIMIT_CLEARANCE = 0.08
DEFAULT_RUN_GOAL_COUNT = 6
DEFAULT_RUN_GOAL_RADIUS = 0.22
DEFAULT_RUN_GOAL_SEED = 13
DEFAULT_GOAL_COUNT = 5
DEFAULT_GOAL_CANDIDATES = 240
DEFAULT_GOAL_SAMPLE_RADIUS = 0.30
DEFAULT_GOAL_SEED = 7
DEFAULT_GOAL_JOINT_MAX_OFFSET = 1.2
DEFAULT_CLOSE_GOAL_OFFSETS = np.array(
    [
        [0.11, -0.085, 0.035],
        [0.22, -0.170, 0.070],
    ],
    dtype=np.float64,
)
PICK_PLACE_RECTANGLE = {
    "frame": "torso_lift_link",
    "corner": np.array([0.6077361, -0.60, -0.38947671], dtype=np.float64),
    "depth_vector": np.array([0.20, 0.0, 0.0], dtype=np.float64),
    "width_vector": np.array([0.0, 0.30, 0.0], dtype=np.float64),
    "width_points": 4,
    "depth_points": 3,
    "tool_down_axis": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    "orientation_cone_half_angle_deg": 45.0,
}
RECTANGLE_GRID_SOLVER_OVERRIDES = {
    "vel_lim_cost": 0.05,
    "ctrl_lim_cost": 0.05,
}
FIRST_TUNABLES_IF_UNSTABLE = ["u_cost", "qd_cost", "N_cost", "q_cost", "rho"]

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
ARM_JOINT_NAMES = [
    "arm_right_1_joint",
    "arm_right_2_joint",
    "arm_right_3_joint",
    "arm_right_4_joint",
    "arm_right_5_joint",
    "arm_right_6_joint",
    "arm_right_7_joint",
]

def load_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    return pin.buildModelFromUrdf(str(model_path))


def tool_position(model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    torso_id = model.getFrameId("torso_lift_link")
    tool_id = model.getFrameId("arm_right_tool_link")
    return (data.oMf[torso_id].inverse() * data.oMf[tool_id]).translation.copy()


def tool_pose(model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    torso_id = model.getFrameId("torso_lift_link")
    tool_id = model.getFrameId("arm_right_tool_link")
    return data.oMf[torso_id].inverse() * data.oMf[tool_id]


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


def _orthonormal_basis(axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def _angle_points(origin, u, v, angles, radius):
    return origin + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)


def _joint_limit_arcs(origin, u, v, q_min, q_max, *, radius, points_per_turn=96):
    span = float(q_max - q_min)
    full = 2.0 * np.pi
    if span >= full - 1e-5:
        allowed_angles = np.linspace(0.0, full, points_per_turn)
        return [_angle_points(origin, u, v, allowed_angles, radius)], []

    allowed_count = max(8, int(points_per_turn * max(span, 0.0) / full))
    blocked_span = max(full - span, 0.0)
    blocked_count = max(8, int(points_per_turn * blocked_span / full))
    allowed_angles = np.linspace(q_min, q_max, allowed_count)
    blocked_angles = np.linspace(q_max, q_min + full, blocked_count)
    return (
        [_angle_points(origin, u, v, allowed_angles, radius)],
        [_angle_points(origin, u, v, blocked_angles, radius)],
    )


def joint_axis_overlays(model, data, q, *, arrow_length=0.075, ring_radius=0.055):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    torso_id = model.getFrameId("torso_lift_link")
    torso_inv = data.oMf[torso_id].inverse()

    origins = []
    axis_segments = []
    allowed_arcs = []
    blocked_arcs = []
    lower = model.lowerPositionLimit.astype(np.float64)
    upper = model.upperPositionLimit.astype(np.float64)
    for joint_index, joint_name in enumerate(ARM_JOINT_NAMES):
        joint_id = model.getJointId(joint_name)
        placement = torso_inv * data.oMi[joint_id]
        origin = placement.translation.copy()
        axis = placement.rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        u, v = _orthonormal_basis(axis)
        allowed, blocked = _joint_limit_arcs(origin, u, v, lower[joint_index], upper[joint_index], radius=ring_radius)
        origins.append(origin)
        axis_segments.append(np.vstack([origin, origin + arrow_length * axis]))
        allowed_arcs.append(allowed)
        blocked_arcs.append(blocked)

    return {
        "origins": np.asarray(origins, dtype=np.float64),
        "axis_segments": np.asarray(axis_segments, dtype=np.float64),
        "allowed_arcs": allowed_arcs,
        "blocked_arcs": blocked_arcs,
    }


def solve_tool_ik(model, q_start, target, *, max_iters=200, tolerance=1e-3):
    data = model.createData()
    q = q_start.astype(np.float64).copy()
    tool_id = model.getFrameId("arm_right_tool_link")

    for _ in range(max_iters):
        current = tool_position(model, data, q)
        error = target - current
        if np.linalg.norm(error) < tolerance:
            break

        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        jacobian = pin.computeFrameJacobian(model, data, q, tool_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        damping = 5e-5 * np.eye(3)
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping, error)
        step_norm = np.linalg.norm(dq)
        if step_norm > 0.15:
            dq *= 0.15 / step_norm
        q = pin.integrate(model, q, dq)
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    final_error = float(np.linalg.norm(tool_position(model, data, q) - target))
    return q, final_error


def solve_tool_axis_ik(
    model,
    q_start,
    target,
    *,
    target_axis=PICK_PLACE_RECTANGLE["tool_down_axis"],
    seed_qs=(),
    max_evals=160,
):
    data = model.createData()
    lower = model.lowerPositionLimit.astype(np.float64)
    upper = model.upperPositionLimit.astype(np.float64)
    target = np.asarray(target, dtype=np.float64)
    target_axis = np.asarray(target_axis, dtype=np.float64)
    target_axis /= np.linalg.norm(target_axis)

    def residual(q):
        pose = tool_pose(model, data, q)
        actual_axis = pose.rotation[:, 2]
        return np.concatenate(
            [
                40.0 * (pose.translation - target),
                6.0 * np.cross(actual_axis, target_axis),
                [3.0 * (1.0 - float(np.dot(actual_axis, target_axis)))],
            ]
        )

    best = None
    for seed_q in (q_start, *seed_qs):
        result = least_squares(
            residual,
            np.clip(np.asarray(seed_q, dtype=np.float64), lower, upper),
            bounds=(lower, upper),
            max_nfev=max_evals,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
            x_scale="jac",
        )
        pose = tool_pose(model, data, result.x)
        pos_error = float(np.linalg.norm(pose.translation - target))
        axis_dot = float(np.clip(np.dot(pose.rotation[:, 2], target_axis), -1.0, 1.0))
        axis_error = float(np.arccos(axis_dot))
        clearance = joint_limit_summary(model, result.x)["min_clearance_rad"]
        score = pos_error + 0.05 * axis_error - 0.0005 * clearance
        candidate = (score, result.x, pos_error, axis_error, clearance)
        if best is None or candidate < best:
            best = candidate

    _, q, pos_error, axis_error, _ = best
    return q, pos_error, axis_error


def tool_axis_ik_seed_qs(model, q_start, *, count=10):
    lower = model.lowerPositionLimit.astype(np.float64)
    upper = model.upperPositionLimit.astype(np.float64)
    rng = np.random.default_rng(20260616)
    seeds = []
    for scale in (0.35, 0.7, 1.0):
        seeds.append(np.clip(q_start + scale * np.array([0.0, 0.7, 0.4, -0.4, -0.5, 0.8, 0.4]), lower, upper))
        seeds.append(np.clip(q_start + scale * np.array([-0.6, 0.5, 0.7, 0.2, -0.8, 0.5, 0.6]), lower, upper))
        seeds.append(np.clip(q_start + scale * np.array([0.6, 0.4, -0.5, -0.2, 0.7, 0.6, -0.5]), lower, upper))
    while len(seeds) < count:
        seeds.append(rng.uniform(lower + 0.03, upper - 0.03))
    return seeds[:count]


def joint_limit_summary(model, q_values):
    q_arr = np.atleast_2d(np.asarray(q_values, dtype=np.float64))
    lower = model.lowerPositionLimit.astype(np.float64)
    upper = model.upperPositionLimit.astype(np.float64)
    lower_violation = np.maximum(lower - q_arr, 0.0)
    upper_violation = np.maximum(q_arr - upper, 0.0)
    clearance = np.minimum(q_arr - lower, upper - q_arr)
    return {
        "min_clearance_rad": float(np.min(clearance)),
        "max_violation_rad": float(np.max(np.maximum(lower_violation, upper_violation))),
    }


def velocity_limit_summary(model, qd_values):
    qd_arr = np.atleast_2d(np.asarray(qd_values, dtype=np.float64))
    velocity_limit = model.velocityLimit.astype(np.float64)
    max_abs_velocity = np.max(np.abs(qd_arr), axis=0)
    return {
        "max_abs_velocity_rad_s": [float(v) for v in max_abs_velocity],
        "velocity_limit_rad_s": [float(v) for v in velocity_limit],
        "max_violation_rad_s": float(np.max(np.maximum(max_abs_velocity - velocity_limit, 0.0))),
    }


def preflight(
    model,
    start_q,
    goals,
    *,
    required_clearance,
    require_tool_down=False,
    tool_axis_tolerance_deg=5.0,
):
    start_limits = joint_limit_summary(model, start_q)
    if start_limits["min_clearance_rad"] < required_clearance:
        raise RuntimeError(
            "Start configuration is too close to a joint limit: "
            f"minimum clearance {start_limits['min_clearance_rad']:.3f} rad, "
            f"required {required_clearance:.3f} rad"
        )

    goal_checks = []
    seed_qs = tool_axis_ik_seed_qs(model, start_q) if require_tool_down else []
    for goal_index, target in enumerate(np.asarray(goals, dtype=np.float64)):
        if require_tool_down:
            ik_q, ik_error, axis_error = solve_tool_axis_ik(model, start_q, target, seed_qs=seed_qs)
            seed_qs.insert(0, ik_q)
        else:
            ik_q, ik_error = solve_tool_ik(model, start_q, target)
            axis_error = None
        ik_limits = joint_limit_summary(model, ik_q)
        if ik_error > 3e-3:
            raise RuntimeError(f"Goal {goal_index} is not reliably reachable before rollout: IK error {ik_error:.6f}m")
        if axis_error is not None and axis_error > np.deg2rad(tool_axis_tolerance_deg):
            raise RuntimeError(
                f"Goal {goal_index} IK tool axis is not down-facing enough: "
                f"axis error {np.rad2deg(axis_error):.3f}deg, "
                f"allowed {tool_axis_tolerance_deg:.3f}deg"
            )
        if ik_limits["min_clearance_rad"] < required_clearance:
            raise RuntimeError(
                f"Goal {goal_index} IK configuration is too close to a joint limit: "
                f"minimum clearance {ik_limits['min_clearance_rad']:.3f} rad, "
                f"required {required_clearance:.3f} rad"
            )
        goal_checks.append(
            {
                "goal_index": int(goal_index),
                "target": [float(v) for v in target],
                "ik_error_m": ik_error,
                "tool_axis_error_deg": None if axis_error is None else float(np.rad2deg(axis_error)),
                "ik_q": [float(v) for v in ik_q],
                "ik_joint_limits": ik_limits,
            }
        )

    return {
        "start_joint_limits": start_limits,
        "require_tool_down": bool(require_tool_down),
        "tool_axis_tolerance_deg": float(tool_axis_tolerance_deg),
        "goals": goal_checks,
    }


def rectangle_grid_points(rectangle=PICK_PLACE_RECTANGLE, *, width_points=None, depth_points=None):
    width_points = rectangle["width_points"] if width_points is None else int(width_points)
    depth_points = rectangle["depth_points"] if depth_points is None else int(depth_points)
    if width_points < 1 or depth_points < 1:
        raise ValueError("rectangle grid dimensions must be positive")

    corner = np.asarray(rectangle["corner"], dtype=np.float64)
    width_vector = np.asarray(rectangle["width_vector"], dtype=np.float64)
    depth_vector = np.asarray(rectangle["depth_vector"], dtype=np.float64)
    width_fractions = np.linspace(0.0, 1.0, width_points)
    depth_fractions = np.linspace(0.0, 1.0, depth_points)

    points = []
    indices = []
    for depth_index, depth_fraction in enumerate(depth_fractions):
        for width_index, width_fraction in enumerate(width_fractions):
            points.append(corner + depth_fraction * depth_vector + width_fraction * width_vector)
            indices.append((width_index, depth_index))
    return np.asarray(points, dtype=np.float64), indices


def rotation_with_local_z(local_z, yaw):
    local_z = np.asarray(local_z, dtype=np.float64)
    local_z /= np.linalg.norm(local_z)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(local_z, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    local_x = helper - float(np.dot(helper, local_z)) * local_z
    local_x /= np.linalg.norm(local_x)
    local_y = np.cross(local_z, local_x)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    local_x_yawed = cos_yaw * local_x + sin_yaw * local_y
    local_y_yawed = np.cross(local_z, local_x_yawed)
    return np.column_stack([local_x_yawed, local_y_yawed, local_z])


def sample_down_cone_orientations(rng, count, *, half_angle_deg):
    down = PICK_PLACE_RECTANGLE["tool_down_axis"]
    u, v = _orthonormal_basis(down)
    half_angle = np.deg2rad(float(half_angle_deg))
    orientations = []
    for _ in range(count):
        cos_theta = rng.uniform(np.cos(half_angle), 1.0)
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi = rng.uniform(0.0, 2.0 * np.pi)
        yaw = rng.uniform(-np.pi, np.pi)
        local_z = cos_theta * down + sin_theta * (np.cos(phi) * u + np.sin(phi) * v)
        orientations.append(pin.rpy.matrixToRpy(rotation_with_local_z(local_z, yaw)))
    return np.asarray(orientations, dtype=np.float64)


def sample_rectangle_grid_goals(
    *,
    count,
    seed,
    width_points=PICK_PLACE_RECTANGLE["width_points"],
    depth_points=PICK_PLACE_RECTANGLE["depth_points"],
    orientation_cone_half_angle_deg=PICK_PLACE_RECTANGLE["orientation_cone_half_angle_deg"],
):
    rng = np.random.default_rng(seed)
    grid, grid_indices = rectangle_grid_points(width_points=width_points, depth_points=depth_points)
    replace = count > grid.shape[0]
    selected = rng.choice(grid.shape[0], size=count, replace=replace)
    orientations = sample_down_cone_orientations(
        rng,
        count,
        half_angle_deg=orientation_cone_half_angle_deg,
    )
    goals = grid[selected]
    return goals, {
        "mode": "rectangle_grid",
        "frame": PICK_PLACE_RECTANGLE["frame"],
        "seed": int(seed),
        "requested_goal_count": int(count),
        "grid_width_points": int(width_points),
        "grid_depth_points": int(depth_points),
        "rectangle_corner": [float(v) for v in PICK_PLACE_RECTANGLE["corner"]],
        "rectangle_depth_vector": [float(v) for v in PICK_PLACE_RECTANGLE["depth_vector"]],
        "rectangle_width_vector": [float(v) for v in PICK_PLACE_RECTANGLE["width_vector"]],
        "all_grid_points": [[float(v) for v in point] for point in grid],
        "all_grid_indices": [[int(i), int(j)] for i, j in grid_indices],
        "selected_grid_indices": [[int(grid_indices[i][0]), int(grid_indices[i][1])] for i in selected],
        "selected_grid_flat_indices": [int(i) for i in selected],
        "selected_goals": [[float(v) for v in goal] for goal in goals],
        "tool_axis_requirement": {
            "local_axis": "arm_right_tool_link +z",
            "target_axis": [float(v) for v in PICK_PLACE_RECTANGLE["tool_down_axis"]],
        },
        "orientation_cone_half_angle_deg": float(orientation_cone_half_angle_deg),
        "sampled_orientation_rpy": [[float(v) for v in rpy] for rpy in orientations],
        "orientation_note": "Orientations are recorded for future pose-goal circuits; the current MPC reach rollout still tracks 3D positions.",
    }


def sample_reachable_goals(model, start_q, start_ee, *, count, candidates, radius, joint_max_offset, required_clearance, seed):
    rng = np.random.default_rng(seed)
    start_limits = joint_limit_summary(model, start_q)
    if start_limits["min_clearance_rad"] < required_clearance:
        raise RuntimeError(
            "Start configuration is too close to a joint limit: "
            f"minimum clearance {start_limits['min_clearance_rad']:.3f} rad, "
            f"required {required_clearance:.3f} rad"
        )

    lower = model.lowerPositionLimit.astype(np.float64)
    upper = model.upperPositionLimit.astype(np.float64)
    data = model.createData()
    selected = []

    for goal_index in range(count):
        offset_limit = float(joint_max_offset) * float(goal_index + 1) / float(max(count, 1))
        valid = []
        for _ in range(candidates):
            delta = rng.uniform(-offset_limit, offset_limit, size=start_q.shape[0])
            q = np.clip(start_q + delta, lower + required_clearance, upper - required_clearance).astype(np.float64)
            target = tool_position(model, data, q)
            distance = float(np.linalg.norm(target - start_ee))
            if distance <= 1e-9:
                continue
            if radius > 0.0 and distance > radius:
                continue
            ik_limits = joint_limit_summary(model, q)
            if ik_limits["min_clearance_rad"] >= required_clearance:
                valid.append(
                    {
                        "target": target,
                        "distance_from_start": distance,
                        "sample_q": q,
                        "joint_offset_limit_rad": offset_limit,
                        "ik_joint_limits": ik_limits,
                    }
                )
        if not valid:
            raise RuntimeError(
                f"No reachable sampled goals found for goal {goal_index} with joint offset limit {offset_limit:.3f} rad. "
                "Try larger --goal-candidates, smaller --required-limit-clearance, or larger --goal-joint-max-offset."
            )
        selected.append(valid[int(rng.integers(0, len(valid)))])

    selected_goals = np.asarray([candidate["target"] for candidate in selected], dtype=np.float64)
    return selected_goals, {
        "mode": "joint_offset_sampled_fk",
        "seed": int(seed),
        "requested_goal_count": int(count),
        "candidates_per_goal": int(candidates),
        "sample_radius_m": float(radius),
        "joint_max_offset_rad": float(joint_max_offset),
        "source": "random_joint_angles_forward_kinematics",
        "selected_distances_from_start_m": [float(np.linalg.norm(goal - start_ee)) for goal in selected_goals],
        "selected_joint_offset_limits_rad": [float(candidate["joint_offset_limit_rad"]) for candidate in selected],
        "selected_goals": [[float(v) for v in goal] for goal in selected_goals],
    }


def sample_away_hemisphere_goals(
    model,
    start_q,
    start_ee,
    *,
    count,
    radius,
    candidates,
    required_clearance,
    seed,
):
    rng = np.random.default_rng(seed)
    start_ee = np.asarray(start_ee, dtype=np.float64)
    goals = []
    goal_checks = []
    attempts = 0
    max_attempts = max(candidates * max(count, 1), count)
    min_radius = min(0.08, radius)

    while len(goals) < count and attempts < max_attempts:
        attempts += 1
        direction = rng.normal(size=3)
        direction[0] = abs(direction[0])
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            continue
        direction /= norm
        if direction[0] < 0.35:
            continue

        distance = rng.uniform(min_radius, radius)
        target = start_ee + distance * direction
        try:
            check = preflight(
                model,
                start_q,
                np.asarray([target], dtype=np.float64),
                required_clearance=required_clearance,
            )["goals"][0]
        except RuntimeError:
            continue
        goals.append(target)
        goal_checks.append(check)

    if len(goals) < count:
        raise RuntimeError(
            f"Only sampled {len(goals)}/{count} away-hemisphere goals. "
            "Try larger --goal-candidates, smaller --required-limit-clearance, "
            "or smaller --run-goal-radius."
        )

    goals = np.asarray(goals, dtype=np.float64)
    return goals, {
        "mode": "away_hemisphere",
        "frame": "torso_lift_link",
        "away_axis": "+x",
        "seed": int(seed),
        "requested_goal_count": int(count),
        "candidate_budget": int(max_attempts),
        "radius_m": float(radius),
        "selected_distances_from_start_m": [float(np.linalg.norm(goal - start_ee)) for goal in goals],
        "selected_goals": [[float(v) for v in goal] for goal in goals],
        "preflight_goals": goal_checks,
    }


def write_goals_csv(path, goals):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(goals, dtype=np.float64), delimiter=",", header="x,y,z", comments="")


def read_goals_csv(path):
    goals = np.loadtxt(path, delimiter=",", skiprows=1)
    goals = np.atleast_2d(goals).astype(np.float64)
    if goals.shape[1] != 3:
        raise ValueError(f"Expected goals CSV with 3 columns x,y,z, got shape {goals.shape}: {path}")
    return goals


def active_targets_for_timestamps(timestamps, goals, goal_reached_times):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    target_rows = np.zeros((timestamps.size, 3), dtype=np.float64)
    for row_index, timestamp in enumerate(timestamps):
        goal_index = 0
        for reached_time in goal_reached_times[:-1]:
            if reached_time is not None and timestamp > float(reached_time):
                goal_index += 1
        goal_index = min(goal_index, goals.shape[0] - 1)
        target_rows[row_index] = goals[goal_index]
    return target_rows


def summarize(
    model,
    timestamps,
    goals,
    target_rows,
    ee_actual,
    distances,
    joints,
    velocities,
    solve_times,
    goal_outcomes,
    goal_reached_times,
    time_to_all_reached,
    applied_controls=None,
    planned_controls=None,
):
    if timestamps.size == 0:
        return {
            "iterations": 0,
            "goal_outcomes": goal_outcomes,
            "goals": [[float(v) for v in goal] for goal in goals],
        }

    effort_limit = model.effortLimit.astype(np.float64)
    torque_limits = {
        "checked": False,
        "reason": "MPC_GATO.run_mpc_goals did not expose applied or planned controls in its stats.",
    }
    if applied_controls is not None and planned_controls is not None:
        applied_controls = np.asarray(applied_controls, dtype=np.float64)
        planned_controls = np.asarray(planned_controls, dtype=np.float64)
        if applied_controls.size and planned_controls.size:
            max_abs_applied = np.max(np.abs(applied_controls), axis=0)
            max_abs_planned = np.max(np.abs(planned_controls), axis=0)
            torque_limits = {
                "checked": True,
                "max_abs_applied_torque_nm": [float(v) for v in max_abs_applied],
                "max_abs_planned_torque_nm": [float(v) for v in max_abs_planned],
                "torque_limit_nm": [float(v) for v in effort_limit],
                "max_applied_violation_nm": float(np.max(np.maximum(max_abs_applied - effort_limit, 0.0))),
                "max_planned_violation_nm": float(np.max(np.maximum(max_abs_planned - effort_limit, 0.0))),
            }

    return {
        "iterations": int(timestamps.size),
        "goal_outcomes": goal_outcomes,
        "goal_reached_times": goal_reached_times,
        "time_to_all_reached": time_to_all_reached,
        "goals": [[float(v) for v in goal] for goal in goals],
        "final_target": [float(v) for v in target_rows[-1]],
        "final_ee": [float(v) for v in ee_actual[-1]],
        "mean_error_m": float(np.mean(distances)),
        "max_error_m": float(np.max(distances)),
        "final_error_m": float(distances[-1]),
        "final_joint_velocity_l1_rad_s": float(np.linalg.norm(velocities[-1], ord=1)),
        "final_joint_velocity_l2_rad_s": float(np.linalg.norm(velocities[-1])),
        "mean_solve_time_ms": float(np.mean(solve_times)),
        "p95_solve_time_ms": float(np.quantile(solve_times, 0.95)),
        "joint_limits": joint_limit_summary(model, joints),
        "velocity_limits": velocity_limit_summary(model, velocities),
        "torque_limits": torque_limits,
    }


def expr_dir_from_args(args):
    return args.expr_dir if args.expr_dir is not None else args.output_root / args.output_label


def json_path(path):
    return str(Path(path).as_posix())


def jsonable_args(args):
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = json_path(value)
        else:
            result[key] = value
    return result


def save_run_data(
    expr_dir,
    metadata,
    timestamps,
    target_rows,
    ee_actual,
    distances,
    joints,
    velocities,
    solve_times,
    applied_controls=None,
    planned_controls=None,
):
    data_dir = expr_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    with (data_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    if timestamps.size == 0:
        return

    np.savetxt(
        data_dir / "trajectory.csv",
        np.column_stack([timestamps, target_rows, ee_actual, distances, solve_times]),
        delimiter=",",
        header="t,target_x,target_y,target_z,x,y,z,error,solve_time_ms",
        comments="",
    )

    header = ["t"] + [f"q{i}" for i in range(joints.shape[1])] + [f"qd{i}" for i in range(velocities.shape[1])]
    np.savetxt(
        data_dir / "joints.csv",
        np.column_stack([timestamps, joints, velocities]),
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    if applied_controls is not None and planned_controls is not None:
        applied_controls = np.asarray(applied_controls, dtype=np.float64)
        planned_controls = np.asarray(planned_controls, dtype=np.float64)
        if applied_controls.size and planned_controls.size:
            header = ["t"] + [f"u{i}" for i in range(applied_controls.shape[1])]
            np.savetxt(
                data_dir / "applied_controls.csv",
                np.column_stack([timestamps, applied_controls]),
                delimiter=",",
                header=",".join(header),
                comments="",
            )
            np.savetxt(
                data_dir / "planned_controls.csv",
                np.column_stack([timestamps, planned_controls]),
                delimiter=",",
                header=",".join(header),
                comments="",
            )


def save_trial_data(data_dir, trial_index, goal, timestamps, target_rows, ee_actual, distances, joints, velocities, solve_times):
    if timestamps.size == 0:
        return
    prefix = data_dir / f"trial_{trial_index:02d}"
    np.savetxt(
        prefix.with_name(prefix.name + "_trajectory.csv"),
        np.column_stack([timestamps, target_rows, ee_actual, distances, solve_times]),
        delimiter=",",
        header="t,target_x,target_y,target_z,x,y,z,error,solve_time_ms",
        comments="",
    )
    header = ["t"] + [f"q{i}" for i in range(joints.shape[1])] + [f"qd{i}" for i in range(velocities.shape[1])]
    np.savetxt(
        prefix.with_name(prefix.name + "_joints.csv"),
        np.column_stack([timestamps, joints, velocities]),
        delimiter=",",
        header=",".join(header),
        comments="",
    )


def trial_renderable(model, joints):
    if joints.size == 0 or not np.isfinite(joints).all():
        return False
    data = model.createData()
    try:
        for q in joints:
            arm_points = arm_link_positions(model, data, q)
            overlay = joint_axis_overlays(model, data, q)
            if not np.isfinite(arm_points).all() or not np.isfinite(overlay["axis_segments"]).all():
                return False
            for arcs in (overlay["allowed_arcs"], overlay["blocked_arcs"]):
                for joint_arcs in arcs:
                    for arc in joint_arcs:
                        if not np.isfinite(arc).all():
                            return False
    except Exception:
        return False
    return True


def load_run_data(expr_dir):
    data_dir = expr_dir / "data"
    with (data_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if "trials" in metadata:
        timestamps = []
        target_rows = []
        ee_actual = []
        distances = []
        solve_times = []
        joints = []
        velocities = []
        time_offset = 0.0
        nq = None
        for trial in metadata["trials"]:
            trial_index = int(trial["trial_index"])
            trajectory_path = data_dir / f"trial_{trial_index:02d}_trajectory.csv"
            joints_path = data_dir / f"trial_{trial_index:02d}_joints.csv"
            if not trajectory_path.exists() or not joints_path.exists():
                continue
            trajectory = np.atleast_2d(np.loadtxt(trajectory_path, delimiter=",", skiprows=1))
            joints_raw = np.atleast_2d(np.loadtxt(joints_path, delimiter=",", skiprows=1))
            if nq is None:
                nq = (joints_raw.shape[1] - 1) // 2
            finite_rows = (
                np.isfinite(trajectory).all(axis=1)
                & np.isfinite(joints_raw).all(axis=1)
            )
            trajectory = trajectory[finite_rows]
            joints_raw = joints_raw[finite_rows]
            if trajectory.size == 0:
                continue
            t = trajectory[:, 0] - trajectory[0, 0] + time_offset
            timestamps.append(t)
            target_rows.append(trajectory[:, 1:4])
            ee_actual.append(trajectory[:, 4:7])
            distances.append(trajectory[:, 7])
            solve_times.append(trajectory[:, 8])
            joints.append(joints_raw[:, 1 : 1 + nq])
            velocities.append(joints_raw[:, 1 + nq :])
            time_offset = float(t[-1]) + 0.5

        if not timestamps:
            return metadata, None

        return metadata, {
            "timestamps": np.concatenate(timestamps),
            "target_rows": np.vstack(target_rows),
            "goals": np.asarray(metadata.get("goals", []), dtype=np.float64),
            "ee_actual": np.vstack(ee_actual),
            "distances": np.concatenate(distances),
            "solve_times": np.concatenate(solve_times),
            "joints": np.vstack(joints),
            "velocities": np.vstack(velocities),
        }

    trajectory_path = data_dir / "trajectory.csv"
    joints_path = data_dir / "joints.csv"
    if not trajectory_path.exists() and "plot_trial" in metadata:
        trial_index = int(metadata["plot_trial"])
        trajectory_path = data_dir / f"trial_{trial_index:02d}_trajectory.csv"
        joints_path = data_dir / f"trial_{trial_index:02d}_joints.csv"
    if not trajectory_path.exists() or not joints_path.exists():
        return metadata, None

    trajectory = np.loadtxt(trajectory_path, delimiter=",", skiprows=1)
    joints_raw = np.loadtxt(joints_path, delimiter=",", skiprows=1)
    trajectory = np.atleast_2d(trajectory)
    joints_raw = np.atleast_2d(joints_raw)

    nq = (joints_raw.shape[1] - 1) // 2
    return metadata, {
        "timestamps": trajectory[:, 0],
        "target_rows": trajectory[:, 1:4],
        "goals": np.asarray(metadata.get("goals", [trajectory[0, 1:4]]), dtype=np.float64),
        "ee_actual": trajectory[:, 4:7],
        "distances": trajectory[:, 7],
        "solve_times": trajectory[:, 8],
        "joints": joints_raw[:, 1 : 1 + nq],
        "velocities": joints_raw[:, 1 + nq :],
    }


def save_error_plot(expr_dir, timestamps, distances, *, interactive=False):
    import matplotlib

    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(timestamps, distances, color="#003192", linewidth=1.7)
    ax.set_xlabel("simulation time [s]")
    ax.set_ylabel("target error [m]")
    ax.set_title("Tiago reach target")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(expr_dir / "target_error.png", dpi=160)
    if interactive:
        plt.show()
    else:
        plt.close(fig)


def animation_interval_ms(timestamps, playback_speed):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.size < 2:
        return 50.0
    dt = float(np.median(np.diff(timestamps)))
    speed = max(float(playback_speed), 1e-6)
    return max(1.0, 1000.0 * dt / speed)


def save_gif(expr_dir, timestamps, ee_actual, target_rows, goals, arm_points, joint_axes, *, interactive=False, write_gif=True, playback_speed=1.0):
    import matplotlib

    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    actual = np.asarray(ee_actual, dtype=np.float64)[:, :3]
    target_rows = np.asarray(target_rows, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    frame_ids = np.linspace(0, actual.shape[0] - 1, min(220, actual.shape[0])).astype(np.int64)

    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    arm_line, = ax.plot([], [], [], "-o", color="#2F4F2F", linewidth=2.4, markersize=3)
    axis_lines = [ax.plot([], [], [], color="#D97817", linewidth=1.4)[0] for _ in ARM_JOINT_NAMES]
    allowed_ring_lines = [ax.plot([], [], [], color="#2E8B57", linewidth=1.8, alpha=0.85)[0] for _ in ARM_JOINT_NAMES]
    blocked_ring_lines = [ax.plot([], [], [], color="#6E1414", linewidth=1.6, alpha=0.80)[0] for _ in ARM_JOINT_NAMES]
    trail, = ax.plot([], [], [], color="#003192", linewidth=1.8)
    actual_point, = ax.plot([], [], [], "o", color="#003192", markersize=6)
    goal_points, = ax.plot(goals[:, 0], goals[:, 1], goals[:, 2], "x", color="#C90016", markersize=8)
    active_target_point, = ax.plot([], [], [], "o", color="#C90016", markersize=4)

    ring_clouds = [
        arc
        for overlay in joint_axes
        for arcs in (overlay["allowed_arcs"], overlay["blocked_arcs"])
        for joint_arcs in arcs
        for arc in joint_arcs
        if len(arc)
    ]
    all_rings = np.vstack(ring_clouds) if ring_clouds else np.empty((0, 3), dtype=np.float64)
    all_axes = np.asarray([overlay["axis_segments"] for overlay in joint_axes], dtype=np.float64)
    all_points = np.vstack([actual, target_rows, goals, arm_points.reshape(-1, 3), all_rings.reshape(-1, 3), all_axes.reshape(-1, 3)])
    all_points = all_points[np.isfinite(all_points).all(axis=1)]
    if all_points.size == 0:
        raise RuntimeError("No finite points available for 3D plot limits")
    center = np.mean(all_points, axis=0)
    radius = max(0.08, float(np.max(np.ptp(all_points, axis=0))) * 0.75)
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
        arm_line.set_data(arm_points[idx, :, 0], arm_points[idx, :, 1])
        arm_line.set_3d_properties(arm_points[idx, :, 2])
        overlay = joint_axes[idx]
        for joint_idx, line in enumerate(axis_lines):
            segment = overlay["axis_segments"][joint_idx]
            line.set_data(segment[:, 0], segment[:, 1])
            line.set_3d_properties(segment[:, 2])
        for joint_idx, line in enumerate(allowed_ring_lines):
            arcs = overlay["allowed_arcs"][joint_idx]
            ring = arcs[0] if arcs else np.empty((0, 3), dtype=np.float64)
            line.set_data(ring[:, 0], ring[:, 1])
            line.set_3d_properties(ring[:, 2])
        for joint_idx, line in enumerate(blocked_ring_lines):
            arcs = overlay["blocked_arcs"][joint_idx]
            ring = arcs[0] if arcs else np.empty((0, 3), dtype=np.float64)
            line.set_data(ring[:, 0], ring[:, 1])
            line.set_3d_properties(ring[:, 2])
        actual_point.set_data([actual[idx, 0]], [actual[idx, 1]])
        actual_point.set_3d_properties([actual[idx, 2]])
        active_target = target_rows[idx]
        active_target_point.set_data([active_target[0]], [active_target[1]])
        active_target_point.set_3d_properties([active_target[2]])
        ax.set_title(f"Tiago reach target | t={timestamps[idx]:.2f}s")
        return (*axis_lines, *allowed_ring_lines, *blocked_ring_lines, arm_line, trail, actual_point, goal_points, active_target_point)

    interval_ms = animation_interval_ms(timestamps[frame_ids], playback_speed) if interactive else 50.0
    update(frame_ids.size - 1)
    fig.savefig(expr_dir / "reach_target_3d.png", dpi=160)
    if write_gif or interactive:
        anim = animation.FuncAnimation(fig, update, frames=frame_ids.size, interval=interval_ms, blit=False)
    if write_gif:
        anim.save(expr_dir / "reach_target.gif", writer=animation.PillowWriter(fps=20))
    if interactive:
        plt.show()
    else:
        plt.close(fig)


def run_experiment(args):
    from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS, TIAGO_TRACKING_SOLVER_PARAMS
    from gato_tiago.tiago_mpc_controller import MPC_GATO

    model = load_model()
    start_q = TIAGO_RIGHT_START_CONFIGS["comfortable"].astype(np.float32)
    start_data = model.createData()
    start_ee = tool_position(model, start_data, start_q.astype(np.float64))
    target_offset = np.asarray(args.target_offset, dtype=np.float64)
    if args.goals_file is not None:
        goals = read_goals_csv(args.goals_file)
        goal_selection = {"mode": "goals_file", "goals_file": json_path(args.goals_file)}
        if not args.ros_tiago:
            run_goal_sweep(args, model, start_q, start_ee, goals, goal_selection)
            return
    elif args.goal_mode == "rectangle-grid":
        goals, goal_selection = sample_rectangle_grid_goals(
            count=args.run_goal_count,
            seed=args.goal_seed,
            width_points=args.grid_width_points,
            depth_points=args.grid_depth_points,
            orientation_cone_half_angle_deg=args.orientation_cone_half_angle_deg,
        )
    elif args.goal_mode == "offset":
        goals = np.vstack([start_ee + target_offset, start_ee + 2.0 * target_offset])
        goal_selection = {
            "mode": "offset",
            "target_offset": [float(v) for v in target_offset],
        }
    else:
        goals, goal_selection = sample_away_hemisphere_goals(
            model,
            start_q.astype(np.float64),
            start_ee,
            count=args.run_goal_count,
            radius=args.run_goal_radius,
            candidates=args.goal_candidates,
            required_clearance=args.required_limit_clearance,
            seed=args.goal_seed,
        )

    preflight_summary = preflight(
        model,
        start_q.astype(np.float64),
        goals,
        required_clearance=args.required_limit_clearance,
        require_tool_down=goal_selection.get("mode") == "rectangle_grid",
        tool_axis_tolerance_deg=args.tool_axis_tolerance_deg,
    )

    solver_params = dict(TIAGO_TRACKING_SOLVER_PARAMS)
    if goal_selection.get("mode") == "rectangle_grid":
        solver_params.update(RECTANGLE_GRID_SOLVER_OVERRIDES)
    if args.vel_lim_cost is not None:
        solver_params["vel_lim_cost"] = args.vel_lim_cost
    if args.ctrl_lim_cost is not None:
        solver_params["ctrl_lim_cost"] = args.ctrl_lim_cost

    x_start = np.concatenate([start_q, np.zeros(model.nv, dtype=np.float32)]).astype(np.float32)
    controller = MPC_GATO(
        model=model,
        model_path=str(MODEL_PATH),
        N=args.N,
        dt=args.dt,
        batch_size=1,
        constant_f_ext=np.zeros(6, dtype=np.float32),
        track_full_stats=True,
        plant_type="tiago_right",
        solver_params=solver_params,
    )
    controller.force_estimator = None

    ros_controller = None
    if args.ros_tiago:
        from gato_tiago.tiago_controller_process import TiagoControllerOrchestrator

        ros_controller = TiagoControllerOrchestrator(
            target_hz=args.ros_target_hz,
            reset_q=start_q,
            reset_duration_sec=args.ros_reset_duration,
            stale_timeout_sec=args.ros_stale_timeout,
            max_abs_torque=args.ros_max_abs_torque,
            clamp_torque=args.ros_clamp_torque,
        )

    expr_dir = expr_dir_from_args(args)
    controller_state_summary = None
    try:
        _, stats = controller.run_mpc_goals(
            x_start=x_start,
            goals=[goal.astype(np.float32) for goal in goals],
            sim_dt=args.sim_dt,
            goal_timeout=args.goal_timeout,
            goal_threshold=args.goal_threshold,
            velocity_threshold=args.velocity_threshold,
            goal_dwell_time=args.goal_dwell_time,
            controller=ros_controller,
            controller_timeout=args.ros_controller_timeout,
        )
    finally:
        if ros_controller is not None:
            ros_controller.close(timeout_sec=args.ros_controller_timeout)
            controller_state_summary = ros_controller.write_state_history_csv(
                expr_dir / "data" / "controller_state_history.csv"
            )

    timestamps = np.asarray(stats.get("timestamps", []), dtype=np.float64)
    ee_actual = np.asarray(stats.get("ee_actual", []), dtype=np.float64)
    distances = np.asarray(stats.get("goal_distances", []), dtype=np.float64)
    joints = np.asarray(stats.get("joint_positions", []), dtype=np.float64)
    velocities = np.asarray(stats.get("joint_velocities", []), dtype=np.float64)
    solve_times = np.asarray(stats.get("solve_times", []), dtype=np.float64)
    applied_controls = np.asarray(stats.get("applied_controls", []), dtype=np.float64)
    planned_controls = np.asarray(stats.get("planned_controls", []), dtype=np.float64)
    goal_outcomes = stats.get("goal_outcomes", [])
    goal_reached_times = stats.get("goal_reached_times", [])
    time_to_all_reached = stats.get("time_to_all_reached")
    target_rows = active_targets_for_timestamps(timestamps, goals, goal_reached_times)
    summary = summarize(
        model,
        timestamps,
        goals,
        target_rows,
        ee_actual,
        distances,
        joints,
        velocities,
        solve_times,
        goal_outcomes,
        goal_reached_times,
        time_to_all_reached,
        applied_controls,
        planned_controls,
    )

    metadata = {
        "args": {**jsonable_args(args), "output_root": json_path(args.output_root), "expr_dir": json_path(expr_dir) if args.expr_dir is not None else None},
        "model_path": json_path(MODEL_PATH),
        "start_ee": [float(v) for v in start_ee],
        "goals": [[float(v) for v in goal] for goal in goals],
        "goal_selection": goal_selection,
        "preflight": preflight_summary,
        "solver_params": {
            key: float(value) if isinstance(value, (int, float, np.floating)) else value
            for key, value in solver_params.items()
        },
        "ros_tiago": bool(args.ros_tiago),
        "ros_clamp_torque": bool(args.ros_clamp_torque),
        "timestamp_source": (
            "joint_state_header_stamp_elapsed" if args.ros_tiago else "offline_simulation_time"
        ),
        "controller_state_history": controller_state_summary,
        "first_tunables_if_unstable": FIRST_TUNABLES_IF_UNSTABLE,
        "summary": summary,
    }
    save_run_data(
        expr_dir,
        metadata,
        timestamps,
        target_rows,
        ee_actual,
        distances,
        joints,
        velocities,
        solve_times,
        applied_controls,
        planned_controls,
    )

    print(f"expr_dir: {expr_dir}")
    print(f"data_dir: {expr_dir / 'data'}")
    print(f"start_ee: {start_ee}")
    print(f"goals: {goals}")
    print(f"outcome: {summary.get('goal_outcomes')}")
    if summary.get("iterations", 0):
        print(
            f"final_error={summary['final_error_m']:.6f}m "
            f"final_velocity_l1={summary['final_joint_velocity_l1_rad_s']:.6f}rad/s "
            f"mean_solve={summary['mean_solve_time_ms']:.3f}ms"
        )


def run_single_goal_trial(model, start_q, goal, args, solver_params):
    from gato_tiago.tiago_mpc_controller import MPC_GATO

    controller = MPC_GATO(
        model=model,
        model_path=str(MODEL_PATH),
        N=args.N,
        dt=args.dt,
        batch_size=1,
        constant_f_ext=np.zeros(6, dtype=np.float32),
        track_full_stats=True,
        plant_type="tiago_right",
        solver_params=solver_params,
    )
    controller.force_estimator = None
    x_start = np.concatenate([start_q, np.zeros(model.nv, dtype=np.float32)]).astype(np.float32)
    _, stats = controller.run_mpc_goals(
        x_start=x_start,
        goals=[goal.astype(np.float32)],
        sim_dt=args.sim_dt,
        goal_timeout=args.goal_timeout,
        goal_threshold=args.goal_threshold,
        velocity_threshold=args.velocity_threshold,
        goal_dwell_time=args.goal_dwell_time,
    )
    timestamps = np.asarray(stats.get("timestamps", []), dtype=np.float64)
    ee_actual = np.asarray(stats.get("ee_actual", []), dtype=np.float64)
    distances = np.asarray(stats.get("goal_distances", []), dtype=np.float64)
    joints = np.asarray(stats.get("joint_positions", []), dtype=np.float64)
    velocities = np.asarray(stats.get("joint_velocities", []), dtype=np.float64)
    solve_times = np.asarray(stats.get("solve_times", []), dtype=np.float64)
    applied_controls = np.asarray(stats.get("applied_controls", []), dtype=np.float64)
    planned_controls = np.asarray(stats.get("planned_controls", []), dtype=np.float64)
    target_rows = np.tile(goal, (timestamps.size, 1))
    summary = summarize(
        model,
        timestamps,
        np.asarray([goal], dtype=np.float64),
        target_rows,
        ee_actual,
        distances,
        joints,
        velocities,
        solve_times,
        stats.get("goal_outcomes", []),
        stats.get("goal_reached_times", []),
        stats.get("time_to_all_reached"),
        applied_controls,
        planned_controls,
    )
    return {
        "stats": stats,
        "timestamps": timestamps,
        "ee_actual": ee_actual,
        "distances": distances,
        "joints": joints,
        "velocities": velocities,
        "solve_times": solve_times,
        "target_rows": target_rows,
        "summary": summary,
    }


def run_goal_sweep(args, model, start_q, start_ee, goals, goal_selection):
    from gato_tiago.config import TIAGO_TRACKING_SOLVER_PARAMS

    solver_params = dict(TIAGO_TRACKING_SOLVER_PARAMS)
    if args.vel_lim_cost is not None:
        solver_params["vel_lim_cost"] = args.vel_lim_cost
    if args.ctrl_lim_cost is not None:
        solver_params["ctrl_lim_cost"] = args.ctrl_lim_cost

    expr_dir = expr_dir_from_args(args)
    data_dir = expr_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    trial_summaries = []
    plot_trial = None
    for trial_index, goal in enumerate(goals):
        trial = run_single_goal_trial(model, start_q, goal, args, solver_params)
        save_trial_data(data_dir, trial_index, goal, trial["timestamps"], trial["target_rows"], trial["ee_actual"], trial["distances"], trial["joints"], trial["velocities"], trial["solve_times"])
        reached = all(outcome == "reached" for outcome in trial["summary"].get("goal_outcomes", []))
        finite = bool(
            np.isfinite(trial["distances"]).all()
            and np.isfinite(trial["joints"]).all()
            and np.isfinite(trial["velocities"]).all()
            and np.isfinite(trial["ee_actual"]).all()
            and np.isfinite(trial["target_rows"]).all()
            and np.isfinite(trial["solve_times"]).all()
        )
        renderable = trial_renderable(model, trial["joints"]) if finite else False
        trial_summary = {
            "trial_index": int(trial_index),
            "goal": [float(v) for v in goal],
            "distance_from_start_m": float(np.linalg.norm(goal - start_ee)),
            "reached": bool(reached),
            "finite": finite,
            "renderable": renderable,
            "summary": trial["summary"],
        }
        trial_summaries.append(trial_summary)
        if finite and renderable:
            plot_trial = trial_index
        print(
            f"trial {trial_index}: distance={trial_summary['distance_from_start_m']:.3f}m "
            f"reached={reached} finite={finite} renderable={renderable} final_error={trial['summary'].get('final_error_m')}"
        )

    successful = [trial for trial in trial_summaries if trial["reached"] and trial["finite"] and trial["renderable"]]
    if successful:
        plot_trial = max(successful, key=lambda trial: trial["distance_from_start_m"])["trial_index"]

    metadata = {
        "args": {**jsonable_args(args), "output_root": json_path(args.output_root), "expr_dir": json_path(expr_dir) if args.expr_dir is not None else None},
        "mode": "sampled_goal_sweep",
        "model_path": json_path(MODEL_PATH),
        "start_ee": [float(v) for v in start_ee],
        "goals": [[float(v) for v in goal] for goal in goals],
        "goal_selection": goal_selection,
        "solver_params": {
            key: float(value) if isinstance(value, (int, float, np.floating)) else value
            for key, value in solver_params.items()
        },
        "first_tunables_if_unstable": FIRST_TUNABLES_IF_UNSTABLE,
        "plot_trial": int(plot_trial) if plot_trial is not None else 0,
        "trials": trial_summaries,
    }
    with (data_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"expr_dir: {expr_dir}")
    print(f"data_dir: {data_dir}")
    print(f"sweep trials: {len(trial_summaries)}")
    print(f"plot_trial: {metadata['plot_trial']}")


def close_seed_goals(model, start_q, start_ee, *, required_clearance, count):
    goals = []
    for offset in DEFAULT_CLOSE_GOAL_OFFSETS[:count]:
        target = start_ee + offset
        preflight(model, start_q, np.asarray([target], dtype=np.float64), required_clearance=required_clearance)
        goals.append(target)
    return np.asarray(goals, dtype=np.float64)


def sample_goals_command(args):
    from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS

    model = load_model()
    start_q = TIAGO_RIGHT_START_CONFIGS["comfortable"].astype(np.float64)
    start_data = model.createData()
    start_ee = tool_position(model, start_data, start_q)
    if args.goal_mode == "rectangle-grid":
        goals, goal_selection = sample_rectangle_grid_goals(
            count=args.goal_count,
            seed=args.goal_seed,
            width_points=args.grid_width_points,
            depth_points=args.grid_depth_points,
            orientation_cone_half_angle_deg=args.orientation_cone_half_angle_deg,
        )
        preflight(
            model,
            start_q,
            goals,
            required_clearance=args.required_limit_clearance,
            require_tool_down=True,
            tool_axis_tolerance_deg=args.tool_axis_tolerance_deg,
        )
        metadata = goal_selection
    else:
        close_goals = close_seed_goals(
            model,
            start_q,
            start_ee,
            required_clearance=args.required_limit_clearance,
            count=args.include_close_goals,
        )
        sampled_count = max(args.goal_count - close_goals.shape[0], 0)
        sampled_goals, goal_selection = sample_reachable_goals(
            model,
            start_q,
            start_ee,
            count=sampled_count,
            candidates=args.goal_candidates,
            radius=args.goal_sample_radius,
            joint_max_offset=args.goal_joint_max_offset,
            required_clearance=args.required_limit_clearance,
            seed=args.goal_seed,
        ) if sampled_count else (np.empty((0, 3), dtype=np.float64), {})
        goals = np.vstack([close_goals, sampled_goals]) if close_goals.size else sampled_goals
        metadata = {"mode": "joint_offset_sampled", "sampled": goal_selection}
    expr_dir = expr_dir_from_args(args)
    goals_path = expr_dir / "data" / "goals.csv"
    write_goals_csv(goals_path, goals)
    with (expr_dir / "data" / "goals_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"wrote: {goals_path}")
    print(f"goals:\n{goals}")


def create_plots(args):
    expr_dir = args.expr_dir
    metadata, data = load_run_data(expr_dir)
    if data is None:
        raise RuntimeError(f"Missing trajectory/joint data under {expr_dir / 'data'}")

    model = load_model(Path(metadata.get("model_path", MODEL_PATH)))
    save_error_plot(expr_dir, data["timestamps"], data["distances"], interactive=args.interactive)

    arm_data = model.createData()
    arm_points = np.asarray([arm_link_positions(model, arm_data, q) for q in data["joints"]], dtype=np.float64)
    joint_axes = [joint_axis_overlays(model, arm_data, q) for q in data["joints"]]
    save_gif(
        expr_dir,
        data["timestamps"],
        data["ee_actual"],
        data["target_rows"],
        data["goals"],
        arm_points,
        joint_axes,
        interactive=args.interactive,
        write_gif=not args.no_gif,
        playback_speed=args.playback_speed,
    )

    print(f"expr_dir: {expr_dir}")
    print(f"wrote: {expr_dir / 'target_error.png'}")
    if not args.no_gif:
        print(f"wrote: {expr_dir / 'reach_target.gif'}")
    print(f"wrote: {expr_dir / 'reach_target_3d.png'}")


def add_run_args(parser):
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--sim-dt", type=float, default=0.003)
    parser.add_argument("--goal-timeout", type=float, default=8.0)
    parser.add_argument("--goal-threshold", type=float, default=0.04)
    parser.add_argument("--velocity-threshold", type=float, default=1.0)
    parser.add_argument("--goal-dwell-time", type=float, default=0.0)
    parser.add_argument("--goal-mode", choices=("rectangle-grid", "away-hemisphere", "offset"), default="rectangle-grid")
    parser.add_argument("--target-offset", nargs=3, type=float, default=DEFAULT_TARGET_OFFSET.tolist())
    parser.add_argument("--run-goal-count", type=int, default=DEFAULT_RUN_GOAL_COUNT)
    parser.add_argument("--run-goal-radius", type=float, default=DEFAULT_RUN_GOAL_RADIUS)
    parser.add_argument("--goal-candidates", type=int, default=DEFAULT_GOAL_CANDIDATES)
    parser.add_argument("--goal-seed", type=int, default=DEFAULT_RUN_GOAL_SEED)
    parser.add_argument("--grid-width-points", type=int, default=PICK_PLACE_RECTANGLE["width_points"])
    parser.add_argument("--grid-depth-points", type=int, default=PICK_PLACE_RECTANGLE["depth_points"])
    parser.add_argument("--orientation-cone-half-angle-deg", type=float, default=PICK_PLACE_RECTANGLE["orientation_cone_half_angle_deg"])
    parser.add_argument("--tool-axis-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--goals-file", type=Path, default=None)
    parser.add_argument("--required-limit-clearance", type=float, default=DEFAULT_LIMIT_CLEARANCE)
    parser.add_argument("--vel-lim-cost", type=float, default=None)
    parser.add_argument("--ctrl-lim-cost", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-label", default="default")
    parser.add_argument("--expr-dir", type=Path, default=None)
    parser.add_argument("--ros-tiago", action="store_true")
    parser.add_argument("--ros-target-hz", type=float, default=100.0)
    parser.add_argument("--ros-reset-duration", type=float, default=2.0)
    parser.add_argument("--ros-stale-timeout", type=float, default=0.25)
    parser.add_argument("--ros-max-abs-torque", type=float, default=30.0)
    parser.add_argument("--ros-clamp-torque", action="store_true")
    parser.add_argument("--ros-controller-timeout", type=float, default=8.0)


def add_sample_goal_args(parser):
    parser.add_argument("--goal-mode", choices=("rectangle-grid", "joint-offset"), default="rectangle-grid")
    parser.add_argument("--goal-count", type=int, default=DEFAULT_GOAL_COUNT)
    parser.add_argument("--goal-candidates", type=int, default=DEFAULT_GOAL_CANDIDATES)
    parser.add_argument("--goal-sample-radius", type=float, default=DEFAULT_GOAL_SAMPLE_RADIUS)
    parser.add_argument("--goal-seed", type=int, default=DEFAULT_GOAL_SEED)
    parser.add_argument("--goal-joint-max-offset", type=float, default=DEFAULT_GOAL_JOINT_MAX_OFFSET)
    parser.add_argument("--include-close-goals", type=int, default=0)
    parser.add_argument("--required-limit-clearance", type=float, default=DEFAULT_LIMIT_CLEARANCE)
    parser.add_argument("--grid-width-points", type=int, default=PICK_PLACE_RECTANGLE["width_points"])
    parser.add_argument("--grid-depth-points", type=int, default=PICK_PLACE_RECTANGLE["depth_points"])
    parser.add_argument("--orientation-cone-half-angle-deg", type=float, default=PICK_PLACE_RECTANGLE["orientation_cone_half_angle_deg"])
    parser.add_argument("--tool-axis-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-label", default="sampled5")
    parser.add_argument("--expr-dir", type=Path, default=None)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="run the MPC experiment and write data")
    add_run_args(run_parser)

    sample_parser = subparsers.add_parser("sample-goals", help="sample reachable goals once and write data/goals.csv")
    add_sample_goal_args(sample_parser)

    plot_parser = subparsers.add_parser("plot", help="load data and write plots")
    plot_parser.add_argument("expr_dir", type=Path)
    plot_parser.add_argument("-i", "--interactive", action="store_true")
    plot_parser.add_argument("--playback-speed", type=float, default=1.0)
    plot_parser.add_argument("--no-gif", action="store_true")

    args = parser.parse_args()
    if args.command is None:
        add_run_args(parser)
        args = parser.parse_args(["run", *sys.argv[1:]])
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.command == "plot":
        create_plots(parsed)
    elif parsed.command == "sample-goals":
        sample_goals_command(parsed)
    else:
        run_experiment(parsed)
