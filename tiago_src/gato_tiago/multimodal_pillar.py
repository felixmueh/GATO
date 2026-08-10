"""Parameterized Tiago pillar-routing problems and certification helpers.

PRIMARY BENCHMARK PROPERTY: all candidates share the same robot initial state,
reference, and objective; only the full trajectory initialization changes.
Poor initial trajectories can fail, select a worse winding class, or retain a
much worse objective.  In the reference hard case, SQP returns every accepted
route-informed initialization unchanged.  This is intentionally a benchmark
of initialization sensitivity and multimodal proposal coverage, not evidence
of solver-converged local or global optima.

The CUDA plant variant interprets each six-value reference row as
``[goal_xyz, pillar_xy, clearance_radius]``.  All solver candidates share the
same reference and differ only in their trajectory initialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pinocchio as pin

from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS


MODEL_PATH = Path("gato/dynamics/tiago_right/tiago_right_arm.urdf")
TORSO_FRAME = "torso_lift_link"
TOOL_FRAME = "arm_right_tool_link"
ARM_FRAMES = (
    TORSO_FRAME,
    "arm_right_1_link",
    "arm_right_2_link",
    "arm_right_3_link",
    "arm_right_4_link",
    "arm_right_5_link",
    "arm_right_6_link",
    "arm_right_7_link",
    TOOL_FRAME,
)


@dataclass(frozen=True)
class Difficulty:
    name: str
    lane: str
    knots: int
    dt: float
    travel_m: float
    radius_m: float
    route_margin_m: float
    asymmetry_m: float
    max_sqp_iters: int
    max_pcg_iters: int
    pcg_tol: float
    q_cost: float
    terminal_cost: float
    qd_cost: float
    u_cost: float
    pillar_cost: float
    terminal_tolerance_m: float
    minimum_clearance_margin_m: float


DIFFICULTIES = {
    "easy": Difficulty(
        name="easy",
        lane="outer_multimodal",
        knots=32,
        dt=0.06,
        travel_m=0.28,
        radius_m=0.075,
        route_margin_m=0.045,
        asymmetry_m=0.04,
        max_sqp_iters=36,
        max_pcg_iters=400,
        pcg_tol=1e-3,
        q_cost=1.5,
        terminal_cost=180.0,
        qd_cost=0.20,
        u_cost=1e-4,
        pillar_cost=500.0,
        terminal_tolerance_m=0.005,
        minimum_clearance_margin_m=0.005,
    ),
    "medium": Difficulty(
        name="medium",
        lane="outer_multimodal",
        knots=64,
        dt=0.045,
        travel_m=0.34,
        radius_m=0.105,
        route_margin_m=0.04,
        asymmetry_m=0.05,
        max_sqp_iters=36,
        max_pcg_iters=500,
        pcg_tol=8e-4,
        q_cost=2.0,
        terminal_cost=260.0,
        qd_cost=0.24,
        u_cost=7.5e-5,
        pillar_cost=800.0,
        terminal_tolerance_m=0.005,
        minimum_clearance_margin_m=0.005,
    ),
    "hard": Difficulty(
        name="hard",
        lane="long_horizon_sqp_pcg",
        knots=128,
        dt=0.03,
        travel_m=0.40,
        radius_m=0.13,
        route_margin_m=0.035,
        asymmetry_m=0.06,
        max_sqp_iters=40,
        max_pcg_iters=600,
        pcg_tol=7.5e-4,
        q_cost=2.5,
        terminal_cost=360.0,
        qd_cost=0.28,
        u_cost=2.5e-5,
        pillar_cost=1200.0,
        terminal_tolerance_m=0.005,
        minimum_clearance_margin_m=0.005,
    ),
}


@dataclass(frozen=True)
class PillarInstance:
    seed: int
    difficulty: str
    start_q: tuple[float, ...]
    start_position: tuple[float, float, float]
    goal_position: tuple[float, float, float]
    pillar_xy: tuple[float, float]
    clearance_radius_m: float

    def reference_row(self) -> np.ndarray:
        row = np.asarray(
            [*self.goal_position, *self.pillar_xy, self.clearance_radius_m],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(row)) or row[5] <= 0.0:
            raise ValueError("Pillar reference must be finite with a positive radius")
        return row

    def metadata(self) -> dict:
        return asdict(self)

    @classmethod
    def from_metadata(cls, metadata: dict) -> "PillarInstance":
        required = {
            "seed", "difficulty", "start_q", "start_position", "goal_position",
            "pillar_xy", "clearance_radius_m",
        }
        missing = required.difference(metadata)
        if missing:
            raise ValueError(f"instance metadata is missing: {sorted(missing)}")
        return cls(
            seed=int(metadata["seed"]),
            difficulty=str(metadata["difficulty"]),
            start_q=tuple(float(value) for value in metadata["start_q"]),
            start_position=tuple(float(value) for value in metadata["start_position"]),
            goal_position=tuple(float(value) for value in metadata["goal_position"]),
            pillar_xy=tuple(float(value) for value in metadata["pillar_xy"]),
            clearance_radius_m=float(metadata["clearance_radius_m"]),
        )


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    route_side: int | None
    route_scale: float = 1.0


def load_model(model_path: Path | str = MODEL_PATH):
    return pin.buildModelFromUrdf(str(model_path))


def tool_position(model, data, q) -> np.ndarray:
    pin.forwardKinematics(model, data, np.asarray(q, dtype=np.float64))
    pin.updateFramePlacements(model, data)
    torso_id = model.getFrameId(TORSO_FRAME)
    tool_id = model.getFrameId(TOOL_FRAME)
    return (data.oMf[torso_id].inverse() * data.oMf[tool_id]).translation.copy()


def arm_link_positions(model, data, q) -> np.ndarray:
    pin.forwardKinematics(model, data, np.asarray(q, dtype=np.float64))
    pin.updateFramePlacements(model, data)
    torso_inv = data.oMf[model.getFrameId(TORSO_FRAME)].inverse()
    return np.asarray(
        [(torso_inv * data.oMf[model.getFrameId(name)]).translation for name in ARM_FRAMES],
        dtype=np.float64,
    )


def generate_instance(
    seed: int,
    difficulty: Difficulty,
    *,
    model=None,
    start_q: Iterable[float] | None = None,
) -> PillarInstance:
    """Generate one reproducible, asymmetric start/pillar/goal geometry.

    Generation establishes the necessary geometry.  A problem is only called
    multimodal after :func:`certify_results` finds both feasible winding modes.
    """

    if model is None:
        model = load_model()
    if start_q is None:
        start_q = TIAGO_RIGHT_START_CONFIGS["comfortable_high_clearance"]
    start_q = np.asarray(start_q, dtype=np.float64).copy()
    data = model.createData()
    rng = np.random.default_rng(seed)

    # Vary the robot posture as well as the workspace geometry.  This prevents
    # the family from reducing to a fixed-start lookup table while retaining a
    # comfortable margin to the arm joint limits.
    start_q += rng.uniform(-0.025, 0.025, size=start_q.shape)
    start_q = np.clip(
        start_q,
        model.lowerPositionLimit.astype(np.float64) + 0.08,
        model.upperPositionLimit.astype(np.float64) - 0.08,
    )
    start = tool_position(model, data, start_q)

    travel = difficulty.travel_m + rng.uniform(-0.018, 0.018)
    travel_angle = rng.uniform(-0.16, 0.16)
    travel_direction = np.array(
        [np.cos(travel_angle), np.sin(travel_angle)], dtype=np.float64
    )
    goal = start + np.array(
        [
            travel * travel_direction[0],
            travel * travel_direction[1],
            rng.uniform(-0.025, 0.025),
        ],
        dtype=np.float64,
    )
    fraction = rng.uniform(0.46, 0.54)
    pillar = start[:2] + fraction * (goal[:2] - start[:2])
    path_normal = np.array([-travel_direction[1], travel_direction[0]])
    lateral_sign = rng.choice((-1.0, 1.0))
    lateral_offset = lateral_sign * rng.uniform(0.35, 1.0) * difficulty.asymmetry_m
    pillar += lateral_offset * path_normal
    radius = difficulty.radius_m + rng.uniform(-0.006, 0.006)

    endpoint_clearances = np.linalg.norm(
        np.vstack([start[:2], goal[:2]]) - pillar[None, :], axis=1
    ) - radius
    if float(np.min(endpoint_clearances)) < 0.035:
        raise RuntimeError("Generated pillar is too close to an endpoint")

    return PillarInstance(
        seed=int(seed),
        difficulty=difficulty.name,
        start_q=tuple(float(v) for v in start_q),
        start_position=tuple(float(v) for v in start),
        goal_position=tuple(float(v) for v in goal),
        pillar_xy=tuple(float(v) for v in pillar),
        clearance_radius_m=float(radius),
    )


def reference_batch(instance: PillarInstance, knots: int, batch_size: int) -> np.ndarray:
    reference = np.tile(instance.reference_row(), knots)
    return np.tile(reference, (batch_size, 1)).astype(np.float32)


def pillar_residual_and_gradient(position_xy, pillar_xy, radius):
    position_xy = np.asarray(position_xy, dtype=np.float64)
    pillar_xy = np.asarray(pillar_xy, dtype=np.float64)
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    delta = position_xy - pillar_xy
    residual = 1.0 - float(delta @ delta) / (radius * radius)
    if residual <= 0.0:
        return 0.0, np.zeros(2, dtype=np.float64)
    return residual, -2.0 * delta / (radius * radius)


def candidate_specs(budget: int) -> list[CandidateSpec]:
    if budget not in (1, 2, 4, 8):
        raise ValueError("candidate budget must be one of 1, 2, 4, or 8")
    # INITIALIZATION IS THE EXPERIMENTAL VARIABLE. Every candidate receives the
    # same x0/reference/objective. Prefixes form nested candidate-budget
    # experiments: the single-candidate baseline is deliberately cold, while
    # larger budgets add distinct route-informed full-trajectory warm starts.
    master = [
        CandidateSpec("cold", None),
        CandidateSpec("clockwise_nominal", 1, 1.2),
        CandidateSpec("counterclockwise_nominal", -1, 1.4),
        CandidateSpec("clockwise_wide", 1, 1.4),
        CandidateSpec("counterclockwise_wide", -1, 1.8),
        CandidateSpec("clockwise_intermediate", 1, 1.3),
        CandidateSpec("counterclockwise_wider", -1, 2.2),
        CandidateSpec("counterclockwise_tighter", -1, 1.1),
    ]
    return master[:budget]


def minimum_jerk_progress(knots: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, knots, dtype=np.float64)
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def task_seed_path(
    instance: PillarInstance,
    difficulty: Difficulty,
    route_side: int,
    route_scale: float = 1.0,
) -> np.ndarray:
    """Construct a visual task-space seed; it is not the solver reference."""

    start = np.asarray(instance.start_position, dtype=np.float64)
    goal = np.asarray(instance.goal_position, dtype=np.float64)
    progress = minimum_jerk_progress(difficulty.knots)
    points = start[None, :] + progress[:, None] * (goal - start)[None, :]
    if route_side:
        direction = goal[:2] - start[:2]
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        desired_mid = np.asarray(instance.pillar_xy) + route_side * route_scale * (
            instance.clearance_radius_m + difficulty.route_margin_m
        ) * normal
        line_mid = 0.5 * (start[:2] + goal[:2])
        points[:, :2] += (
            np.sin(np.pi * progress)[:, None] * (desired_mid - line_mid)[None, :]
        )
    return points


def solve_position_path_ik(
    model,
    q_start: np.ndarray,
    targets: np.ndarray,
    *,
    tolerance: float = 8e-4,
    max_iterations: int = 70,
) -> tuple[np.ndarray, float]:
    data = model.createData()
    tool_id = model.getFrameId(TOOL_FRAME)
    lower = model.lowerPositionLimit.astype(np.float64)
    upper = model.upperPositionLimit.astype(np.float64)
    q = np.asarray(q_start, dtype=np.float64).copy()
    q_path = []
    max_error = 0.0

    for target in np.asarray(targets, dtype=np.float64):
        posture = q.copy()
        for _ in range(max_iterations):
            current = tool_position(model, data, q)
            error = target - current
            if np.linalg.norm(error) <= tolerance:
                break
            pin.computeJointJacobians(model, data, q)
            pin.updateFramePlacements(model, data)
            jacobian = pin.computeFrameJacobian(
                model, data, q, tool_id, pin.LOCAL_WORLD_ALIGNED
            )[:3, :]
            gram = jacobian @ jacobian.T + 2e-4 * np.eye(3)
            jacobian_pinv = jacobian.T @ np.linalg.solve(gram, np.eye(3))
            null_projector = np.eye(model.nv) - jacobian_pinv @ jacobian
            dq = jacobian_pinv @ error + 0.04 * null_projector @ (posture - q)
            step = np.linalg.norm(dq)
            if step > 0.10:
                dq *= 0.10 / step
            q = np.clip(pin.integrate(model, q, dq), lower + 0.01, upper - 0.01)
        error = float(np.linalg.norm(tool_position(model, data, q) - target))
        max_error = max(max_error, error)
        q_path.append(q.copy())
    return np.asarray(q_path, dtype=np.float64), max_error


def _discrete_state_derivatives(q_path: np.ndarray, dt: float):
    """Choose qd/qdd that exactly reproduce q under GATO's trapezoidal step."""

    qd = np.empty_like(q_path)
    qdd = np.empty((q_path.shape[0] - 1, q_path.shape[1]), dtype=np.float64)
    qd[0] = 0.0
    for knot in range(q_path.shape[0] - 1):
        qd[knot + 1] = 2.0 * (q_path[knot + 1] - q_path[knot]) / dt - qd[knot]
        qdd[knot] = (qd[knot + 1] - qd[knot]) / dt
    return qd, qdd


def pack_warm_start(
    model,
    instance: PillarInstance,
    difficulty: Difficulty,
    spec: CandidateSpec,
) -> tuple[np.ndarray, dict]:
    nq, nv = model.nq, model.nv
    nx, nu = nq + nv, nv
    start_q = np.asarray(instance.start_q, dtype=np.float64)
    x0 = np.hstack([start_q, np.zeros(nv, dtype=np.float64)])
    ik_error = 0.0

    if spec.route_side is None:
        q_path = np.tile(start_q, (difficulty.knots, 1))
    else:
        task_path = task_seed_path(
            instance, difficulty, spec.route_side, route_scale=spec.route_scale
        )
        q_path, ik_error = solve_position_path_ik(model, start_q, task_path)

    qd_path, qdd_path = _discrete_state_derivatives(q_path, difficulty.dt)
    controls = np.empty((difficulty.knots - 1, nu), dtype=np.float64)
    data = model.createData()
    effort = model.effortLimit.astype(np.float64)
    for knot in range(difficulty.knots - 1):
        controls[knot] = pin.rnea(
            model, data, q_path[knot], qd_path[knot], qdd_path[knot]
        )

    packed = np.zeros(difficulty.knots * (nx + nu) - nu, dtype=np.float32)
    for knot in range(difficulty.knots):
        offset = knot * (nx + nu)
        packed[offset : offset + nq] = q_path[knot]
        packed[offset + nq : offset + nx] = qd_path[knot]
        if knot < difficulty.knots - 1:
            packed[offset + nx : offset + nx + nu] = controls[knot]
    packed[:nx] = x0
    return packed, {
        "seed_ik_max_error_m": float(ik_error),
        "seed_max_velocity_ratio": float(
            np.max(np.abs(qd_path) / model.velocityLimit.astype(np.float64))
        ),
        "seed_max_torque_ratio": float(
            np.max(np.abs(controls) / effort)
        ),
    }


def unpack_trajectory(trajectory, model, knots: int):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    nq, nv = model.nq, model.nv
    nx, nu = nq + nv, nv
    q = np.empty((knots, nq), dtype=np.float64)
    qd = np.empty((knots, nv), dtype=np.float64)
    u = np.empty((knots - 1, nu), dtype=np.float64)
    for knot in range(knots):
        offset = knot * (nx + nu)
        q[knot] = trajectory[offset : offset + nq]
        qd[knot] = trajectory[offset + nq : offset + nx]
        if knot < knots - 1:
            u[knot] = trajectory[offset + nx : offset + nx + nu]
    return q, qd, u


def pack_trajectory(q, qd, u, model, knots: int) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    qd = np.asarray(qd, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    nq, nv = model.nq, model.nv
    nx = nq + nv
    packed = np.zeros(knots * (nx + nv) - nv, dtype=np.float32)
    for knot in range(knots):
        offset = knot * (nx + nv)
        packed[offset : offset + nq] = q[knot]
        packed[offset + nq : offset + nx] = qd[knot]
        if knot < knots - 1:
            packed[offset + nx : offset + nx + nv] = u[knot]
    return packed


def dense_joint_path(q, qd, dt: float, samples_per_interval: int = 16):
    rows = []
    with np.errstate(invalid="ignore", over="ignore"):
        for knot in range(q.shape[0] - 1):
            for alpha in np.linspace(0.0, 1.0, samples_per_interval, endpoint=False):
                a2, a3 = alpha * alpha, alpha * alpha * alpha
                h00 = 2.0 * a3 - 3.0 * a2 + 1.0
                h10 = a3 - 2.0 * a2 + alpha
                h01 = -2.0 * a3 + 3.0 * a2
                h11 = a3 - a2
                rows.append(
                    h00 * q[knot]
                    + h10 * dt * qd[knot]
                    + h01 * q[knot + 1]
                    + h11 * dt * qd[knot + 1]
                )
    rows.append(q[-1].copy())
    return np.asarray(rows, dtype=np.float64)


def tool_path(model, q_path) -> np.ndarray:
    data = model.createData()
    return np.asarray([tool_position(model, data, q) for q in q_path], dtype=np.float64)


def winding_signature(tool_positions, pillar_xy, *, threshold_rad: float = 2.0):
    delta = np.asarray(tool_positions, dtype=np.float64)[:, :2] - np.asarray(
        pillar_xy, dtype=np.float64
    )[None, :]
    angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    winding_angle = float(angles[-1] - angles[0])
    if winding_angle > threshold_rad:
        label = "counterclockwise"
    elif winding_angle < -threshold_rad:
        label = "clockwise"
    else:
        label = "no_winding"
    return label, winding_angle


def _barrier(values, lower, upper):
    values = np.asarray(values, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return -np.log(np.maximum(values - lower, 1e-10)) - np.log(
        np.maximum(upper - values, 1e-10)
    )


def trajectory_costs(model, instance, difficulty, q, qd, u):
    positions = tool_path(model, q)
    goal = np.asarray(instance.goal_position, dtype=np.float64)
    pillar = np.asarray(instance.pillar_xy, dtype=np.float64)
    radius = instance.clearance_radius_m
    components = {
        "target": 0.0,
        "velocity": 0.0,
        "control": 0.0,
        "pillar": 0.0,
        "joint_barrier": 0.0,
        "velocity_barrier": 0.0,
        "control_barrier": 0.0,
    }
    q_lower, q_upper = model.lowerPositionLimit, model.upperPositionLimit
    vel = model.velocityLimit
    effort = model.effortLimit
    for knot in range(difficulty.knots):
        target_weight = (
            difficulty.terminal_cost if knot == difficulty.knots - 1 else difficulty.q_cost
        )
        error = positions[knot] - goal
        components["target"] += 0.5 * target_weight * float(error @ error)
        components["velocity"] += 0.5 * difficulty.qd_cost * float(qd[knot] @ qd[knot])
        components["joint_barrier"] += 0.01 * float(
            np.sum(_barrier(q[knot], q_lower, q_upper))
        )
        components["velocity_barrier"] += 0.001 * float(
            np.sum(_barrier(qd[knot], -vel, vel))
        )
        residual, _ = pillar_residual_and_gradient(positions[knot, :2], pillar, radius)
        components["pillar"] += 0.5 * difficulty.pillar_cost * residual * residual
        if knot < difficulty.knots - 1:
            components["control"] += 0.5 * difficulty.u_cost * float(u[knot] @ u[knot])
            components["control_barrier"] += 0.003 * float(
                np.sum(_barrier(u[knot], -effort, effort))
            )
    components = {key: float(value) for key, value in components.items()}
    components["nonnegative_task_motion"] = sum(
        components[key] for key in ("target", "velocity", "control", "pillar")
    )
    components["solver_objective"] = sum(
        value
        for key, value in components.items()
        if key not in ("nonnegative_task_motion", "solver_objective")
    )
    return components


def dynamics_defect(model, q, qd, u, dt: float):
    data = model.createData()
    defects = []
    for knot in range(u.shape[0]):
        qdd = pin.aba(model, data, q[knot], qd[knot], u[knot])
        q_next = q[knot] + dt * qd[knot] + 0.5 * dt * dt * qdd
        qd_next = qd[knot] + dt * qdd
        defects.append(
            np.linalg.norm(np.hstack([q_next - q[knot + 1], qd_next - qd[knot + 1]]))
        )
    return float(np.max(defects, initial=0.0))


def evaluate_trajectory(model, instance, difficulty, trajectory):
    q, qd, u = unpack_trajectory(trajectory, model, difficulty.knots)
    dense_q = dense_joint_path(q, qd, difficulty.dt)
    dense_tool = tool_path(model, dense_q)
    radial = np.linalg.norm(
        dense_tool[:, :2] - np.asarray(instance.pillar_xy)[None, :], axis=1
    )
    min_clearance = float(np.min(radial) - instance.clearance_radius_m)
    label, winding_angle = winding_signature(dense_tool, instance.pillar_xy)
    terminal_error = float(
        np.linalg.norm(dense_tool[-1] - np.asarray(instance.goal_position))
    )
    joint_violation = float(
        max(
            np.max(model.lowerPositionLimit - dense_q, initial=0.0),
            np.max(dense_q - model.upperPositionLimit, initial=0.0),
            0.0,
        )
    )
    velocity_ratio = float(np.max(np.abs(qd) / model.velocityLimit))
    torque_ratio = float(np.max(np.abs(u) / model.effortLimit))
    defect = dynamics_defect(model, q, qd, u, difficulty.dt)
    costs = trajectory_costs(model, instance, difficulty, q, qd, u)
    physical_feasible = bool(
        np.isfinite(costs["solver_objective"])
        and terminal_error <= difficulty.terminal_tolerance_m
        and min_clearance >= difficulty.minimum_clearance_margin_m
        and joint_violation <= 0.0
        and velocity_ratio <= 1.0
        and torque_ratio <= 1.0
        and label != "no_winding"
    )
    return {
        "mode": label,
        "winding_angle_rad": winding_angle,
        "objective": costs["solver_objective"],
        "nonnegative_task_motion_cost": costs["nonnegative_task_motion"],
        "cost_components": costs,
        "terminal_error_m": terminal_error,
        "min_dense_clearance_m": min_clearance,
        "max_joint_violation_rad": joint_violation,
        "max_velocity_ratio": velocity_ratio,
        "max_torque_ratio": torque_ratio,
        "max_dynamics_defect": defect,
        "physical_feasible": physical_feasible,
        "tool_path": dense_tool,
        "q": q,
        "qd": qd,
        "u": u,
        "pillar_xy": np.asarray(instance.pillar_xy, dtype=np.float64),
        "clearance_radius_m": float(instance.clearance_radius_m),
    }


def certify_results(
    results: list[dict],
    *,
    min_support_per_mode: int = 2,
    minimum_mode_gap_fraction: float = 0.02,
    minimum_unique_path_rms_separation_m: float = 0.005,
):
    modes = ("clockwise", "counterclockwise")
    raw_by_mode = {
        mode: [result for result in results if result["certifiable"] and result["mode"] == mode]
        for mode in modes
    }
    supported_by_mode = {}
    for mode, rows in raw_by_mode.items():
        unique = []
        for row in sorted(rows, key=lambda candidate: candidate["objective"]):
            path = np.asarray(row["tool_path"], dtype=np.float64)
            duplicate = any(
                path.shape == np.asarray(other["tool_path"]).shape
                and float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (
                                    path
                                    - np.asarray(other["tool_path"], dtype=np.float64)
                                )
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                )
                < minimum_unique_path_rms_separation_m
                for other in unique
            )
            if not duplicate:
                unique.append(row)
        supported_by_mode[mode] = unique
    support = {mode: len(rows) for mode, rows in supported_by_mode.items()}
    best_rows_by_mode = {
        mode: (min(rows, key=lambda row: row["objective"]) if rows else None)
        for mode, rows in supported_by_mode.items()
    }
    best_objective_by_mode = {
        mode: (row["objective"] if row else None)
        for mode, row in best_rows_by_mode.items()
    }
    best_task_cost_by_mode = {
        mode: (row["nonnegative_task_motion_cost"] if row else None)
        for mode, row in best_rows_by_mode.items()
    }
    two_mode_recovery = all(support[mode] >= min_support_per_mode for mode in modes)
    certifiable = [row for row in results if row["certifiable"]]
    best_row = (
        min(certifiable, key=lambda row: row["objective"])
        if certifiable
        else None
    )
    best_known_objective = best_row["objective"] if best_row else None
    best_known_task_cost = (
        best_row["nonnegative_task_motion_cost"] if best_row else None
    )
    ranking_gap = None
    absolute_gap = None
    gap_scale = None
    best_mode = best_row["mode"] if best_row else None
    if all(best_rows_by_mode[mode] is not None for mode in modes):
        values = sorted(float(best_objective_by_mode[mode]) for mode in modes)
        absolute_gap = values[1] - values[0]
        # Differences in the full objective are invariant to additive barrier
        # constants.  Normalize by a separate positive physical-cost scale.
        gap_scale = max(
            1.0,
            0.5
            * sum(float(best_task_cost_by_mode[mode]) for mode in modes),
        )
        ranking_gap = absolute_gap / gap_scale

    cross_mode_clearance = None
    if two_mode_recovery:
        first = min(
            supported_by_mode[modes[0]], key=lambda row: row["objective"]
        )["tool_path"]
        second = min(
            supported_by_mode[modes[1]], key=lambda row: row["objective"]
        )["tool_path"]
        width = min(first.shape[0], second.shape[0])
        midpoint_path = 0.5 * (first[:width, :2] + second[:width, :2])
        pillar_xy = np.asarray(supported_by_mode[modes[0]][0]["pillar_xy"])
        radius = float(supported_by_mode[modes[0]][0]["clearance_radius_m"])
        cross_mode_clearance = float(
            np.min(np.linalg.norm(midpoint_path - pillar_xy[None, :], axis=1)) - radius
        )

    return {
        "finite_budget_two_mode_coverage": bool(two_mode_recovery),
        "ranked_finite_budget_benchmark": bool(
            two_mode_recovery
            and ranking_gap is not None
            and ranking_gap >= minimum_mode_gap_fraction
        ),
        "minimum_support_per_mode": int(min_support_per_mode),
        "minimum_unique_path_rms_separation_m": float(
            minimum_unique_path_rms_separation_m
        ),
        "minimum_mode_gap_fraction": float(minimum_mode_gap_fraction),
        "raw_certifiable_candidates": {
            mode: len(rows) for mode, rows in raw_by_mode.items()
        },
        "certifiable_support": support,
        "best_solver_objective_by_mode": best_objective_by_mode,
        "task_motion_cost_at_mode_best": best_task_cost_by_mode,
        "best_known_solver_objective": best_known_objective,
        "task_motion_cost_at_best_known": best_known_task_cost,
        "best_known_mode": best_mode,
        "mode_objective_gap_absolute": absolute_gap,
        "mode_objective_gap_scale": gap_scale,
        "mode_objective_gap_fraction": ranking_gap,
        "cross_mode_midpoint_min_clearance_m": cross_mode_clearance,
    }
