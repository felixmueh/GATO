"""Quarantined reachable-task witness for Tiago multimodal-toll Stage V0/V1.

This module is a one-way consumer of the public schema.  Benchmark and
initializer code must never import it or receive :class:`ConstructionWitness`.
Task execution remains disabled in this static checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS
from gato_tiago.multimodal_pillar import TOOL_FRAME, tool_position
from gato_tiago.multimodal_toll import (
    ACTUAL_TRAVEL_RANGE_M,
    CLEARANCE_MARGIN_M,
    CYLINDER_OFFSET_M,
    DAMPING,
    DEVELOPMENT_TASK_SEEDS,
    GOAL_JOINT_MARGIN_RAD,
    HELDOUT_TASK_SEEDS,
    MAX_VERTICAL_TRAVEL_M,
    MIN_ENDPOINT_PHYSICAL_CLEARANCE_M,
    PHYSICAL_RADIUS_M,
    PLANAR_ANGLE_RAD,
    Q0_JITTER_RAD,
    REQUESTED_TRAVEL_M,
    TOLL_OFFSET_FROM_CYLINDER_M,
    TOLL_SIGMA_M,
    TollReference,
    TollTask,
    _sha256_array,
    physical_clearance,
)


# Narrow Stage V1 task-construction capability. This does not authorize any
# route oracle, optimizer, benchmark initializer, or SQP execution.
TASK_CONSTRUCTION_AUTHORIZATION = object()


@dataclass(frozen=True)
class ConstructionWitness:
    """Oracle-only arrays proving how the reachable public task was built."""

    q_goal: tuple[float, ...]
    dq: tuple[float, ...]
    requested_delta_xyz: tuple[float, float, float]
    q0_jitter: tuple[float, ...]
    phi: float
    offset_sign: int
    start_xyz: tuple[float, float, float]
    actual_travel_m: float
    planar_travel_m: float
    endpoint_physical_clearance_m: float
    chord_min_distance_to_cylinder_m: float
    chord_intersects_physical_cylinder: bool
    chord_intersects_optimizer_keepout: bool
    oracle_only: bool = True
    benchmark_seed_eligible: bool = False


def construct_task_from_draws(model, q0_jitter, phi, offset_sign, *, task_seed=-1):
    """Construct public task plus quarantined witness from frozen-order draws."""

    jitter = np.asarray(q0_jitter, dtype=np.float64)
    if jitter.shape != (7,) or np.any(np.abs(jitter) > Q0_JITTER_RAD):
        raise ValueError("q0 jitter must be seven values in the frozen range")
    if not -PLANAR_ANGLE_RAD <= float(phi) <= PLANAR_ANGLE_RAD:
        raise ValueError("phi is outside the frozen range")
    if offset_sign not in (-1, 1):
        raise ValueError("offset_sign must be -1 or +1")

    q0 = np.asarray(
        TIAGO_RIGHT_START_CONFIGS["comfortable_high_clearance"], dtype=np.float64
    ) + jitter
    lower = np.asarray(model.lowerPositionLimit, dtype=np.float64)
    upper = np.asarray(model.upperPositionLimit, dtype=np.float64)
    if np.any(q0 - lower < GOAL_JOINT_MARGIN_RAD) or np.any(
        upper - q0 < GOAL_JOINT_MARGIN_RAD
    ):
        raise RuntimeError("q0 violates the frozen joint-margin assertion")

    data = model.createData()
    start = tool_position(model, data, q0)
    tool_id = model.getFrameId(TOOL_FRAME)
    pin.computeJointJacobians(model, data, q0)
    pin.updateFramePlacements(model, data)
    jacobian = pin.getFrameJacobian(
        model, data, tool_id, pin.LOCAL_WORLD_ALIGNED
    )[:3, :]
    delta = REQUESTED_TRAVEL_M * np.asarray(
        [np.cos(phi), np.sin(phi), 0.0], dtype=np.float64
    )
    dq = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + DAMPING**2 * np.eye(3), delta
    )
    q_goal = q0 + dq
    if np.any(q_goal - lower < GOAL_JOINT_MARGIN_RAD) or np.any(
        upper - q_goal < GOAL_JOINT_MARGIN_RAD
    ):
        raise RuntimeError("q_goal violates the frozen joint-margin assertion")
    goal = tool_position(model, data, q_goal)
    actual_travel = float(np.linalg.norm(goal - start))
    planar_travel = float(np.linalg.norm(goal[:2] - start[:2]))
    if not ACTUAL_TRAVEL_RANGE_M[0] <= actual_travel <= ACTUAL_TRAVEL_RANGE_M[1]:
        raise RuntimeError("actual tool travel violates the frozen range")
    if abs(float(goal[2] - start[2])) > MAX_VERTICAL_TRAVEL_M:
        raise RuntimeError("actual vertical travel violates the frozen range")

    if planar_travel <= 0:
        raise RuntimeError("planar chord is degenerate")
    direction = (goal[:2] - start[:2]) / planar_travel
    normal = np.asarray([-direction[1], direction[0]])
    cylinder = (
        0.5 * (start[:2] + goal[:2])
        + offset_sign * CYLINDER_OFFSET_M * normal
    )
    default_side = -int(offset_sign)
    toll = cylinder + default_side * TOLL_OFFSET_FROM_CYLINDER_M * normal
    reference = TollReference(
        goal_xyz=tuple(goal),
        cylinder_xy=tuple(cylinder),
        physical_radius_m=PHYSICAL_RADIUS_M,
        toll_xy=tuple(toll),
        toll_sigma_m=TOLL_SIGMA_M,
        clearance_margin_m=CLEARANCE_MARGIN_M,
    )
    ref32 = reference.as_float32()
    endpoint_clearance = min(
        physical_clearance(start[:2], ref32),
        physical_clearance(goal[:2], ref32),
    )
    if endpoint_clearance < MIN_ENDPOINT_PHYSICAL_CLEARANCE_M:
        raise RuntimeError("endpoint physical clearance violates the frozen assertion")
    cylinder_delta = cylinder - start[:2]
    chord_min_distance = float(
        abs(direction[0] * cylinder_delta[1] - direction[1] * cylinder_delta[0])
    )
    chord_intersects_physical = chord_min_distance < PHYSICAL_RADIUS_M
    chord_intersects_keepout = chord_min_distance < (
        PHYSICAL_RADIUS_M + CLEARANCE_MARGIN_M
    )
    if not chord_intersects_physical or not chord_intersects_keepout:
        raise RuntimeError("direct chord does not intersect the frozen cylinder")

    x0 = np.concatenate([q0, np.zeros(model.nv, dtype=np.float64)]).astype(
        np.float32
    )
    public_task = TollTask(
        task_seed=int(task_seed),
        x0=tuple(float(v) for v in x0),
        reference=TollReference.from_solver_bytes(ref32),
        default_side=default_side,
        solver_x0_sha256=_sha256_array(x0),
        solver_reference_sha256=_sha256_array(ref32),
    )
    witness = ConstructionWitness(
        q_goal=tuple(float(value) for value in q_goal),
        dq=tuple(float(value) for value in dq),
        requested_delta_xyz=tuple(float(value) for value in delta),
        q0_jitter=tuple(float(value) for value in jitter),
        phi=float(phi),
        offset_sign=int(offset_sign),
        start_xyz=tuple(float(value) for value in start),
        actual_travel_m=actual_travel,
        planar_travel_m=planar_travel,
        endpoint_physical_clearance_m=endpoint_clearance,
        chord_min_distance_to_cylinder_m=chord_min_distance,
        chord_intersects_physical_cylinder=chord_intersects_physical,
        chord_intersects_optimizer_keepout=chord_intersects_keepout,
    )
    return public_task, witness


def generate_task(task_seed, model, *, authorization=None):
    allowed = frozenset(DEVELOPMENT_TASK_SEEDS + HELDOUT_TASK_SEEDS)
    if int(task_seed) not in allowed:
        raise ValueError("task_seed is not in the frozen development/held-out sets")
    if (
        TASK_CONSTRUCTION_AUTHORIZATION is None
        or authorization is not TASK_CONSTRUCTION_AUTHORIZATION
    ):
        raise RuntimeError("task instantiation is blocked pending verifier authorization")
    rng = np.random.default_rng(int(task_seed))
    jitter = rng.uniform(-Q0_JITTER_RAD, Q0_JITTER_RAD, size=7)
    phi = float(rng.uniform(-PLANAR_ANGLE_RAD, PLANAR_ANGLE_RAD))
    sign = int(rng.choice((-1, 1)))
    return construct_task_from_draws(
        model, jitter, phi, sign, task_seed=task_seed
    )
