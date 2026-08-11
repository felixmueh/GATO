"""Quarantined exact-eight-DLS reachable-task constructor for V3.

This module is oracle-only.  Benchmark and initializer modules must never
import it or receive :class:`V3ConstructionWitness`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

import numpy as np
import pinocchio as pin

from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS
from gato_tiago.multimodal_pillar import TOOL_FRAME, tool_position
from gato_tiago.multimodal_toll import (
    ACTUAL_TRAVEL_RANGE_M,
    CLEARANCE_MARGIN_M,
    CYLINDER_OFFSET_M,
    GOAL_JOINT_MARGIN_RAD,
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
from gato_tiago.multimodal_toll_v3 import (
    DLS_DAMPING,
    DLS_ITERATIONS,
    DLS_UNIT_STEP,
    validate_v3_task_seed,
)


TASK_CONSTRUCTION_AUTHORIZATION = None


def _history_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(
        f"{value.dtype.str}|{value.shape}|".encode() + value.tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class DLSHistory:
    q_float64: np.ndarray
    tool_position_float64: np.ndarray
    residual_float64: np.ndarray
    jacobian_float64: np.ndarray
    dq_float64: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "q_float64": self.q_float64,
            "tool_position_float64": self.tool_position_float64,
            "residual_float64": self.residual_float64,
            "jacobian_float64": self.jacobian_float64,
            "dq_float64": self.dq_float64,
        }

    def hashes(self) -> dict[str, str]:
        return {name: _history_hash(value) for name, value in self.arrays().items()}


@dataclass(frozen=True)
class V3ConstructionWitness:
    q0_jitter: tuple[float, ...]
    phi: float
    offset_sign: int
    requested_target_xyz: tuple[float, float, float]
    actual_travel_m: float
    planar_travel_m: float
    endpoint_physical_clearance_m: float
    chord_min_distance_to_cylinder_m: float
    chord_intersects_physical_cylinder: bool
    chord_intersects_optimizer_keepout: bool
    history: DLSHistory
    history_hashes: dict[str, str]
    oracle_only: bool = True
    benchmark_seed_eligible: bool = False


def exact_eight_dls(
    q0: np.ndarray,
    requested_target_xyz: np.ndarray,
    position_and_jacobian: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    final_position: Callable[[np.ndarray], np.ndarray],
) -> DLSHistory:
    """Run the frozen eight unit-step full-residual DLS recurrences."""

    q = np.asarray(q0)
    target = np.asarray(requested_target_xyz)
    if q.shape != (7,) or q.dtype != np.float64 or not np.all(np.isfinite(q)):
        raise ValueError("q0 must be one finite float64 seven-vector")
    if target.shape != (3,) or target.dtype != np.float64 or not np.all(np.isfinite(target)):
        raise ValueError("target must be one finite float64 xyz vector")

    q_rows = [q.copy()]
    p_rows = []
    residual_rows = []
    jacobian_rows = []
    dq_rows = []
    for _ in range(DLS_ITERATIONS):
        position, jacobian = position_and_jacobian(q)
        position = np.asarray(position)
        jacobian = np.asarray(jacobian)
        if position.shape != (3,) or position.dtype != np.float64:
            raise ValueError("tool position callback must return float64 shape (3,)")
        if jacobian.shape != (3, 7) or jacobian.dtype != np.float64:
            raise ValueError("tool Jacobian callback must return float64 shape (3,7)")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(jacobian)):
            raise ValueError("DLS callback outputs must be finite")
        residual = target - position
        dq = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + DLS_DAMPING**2 * np.eye(3), residual
        )
        p_rows.append(position.copy())
        residual_rows.append(residual.copy())
        jacobian_rows.append(jacobian.copy())
        dq_rows.append(dq.copy())
        q = q + DLS_UNIT_STEP * dq
        q_rows.append(q.copy())

    final_value = final_position(q)
    final_value = np.asarray(final_value)
    if final_value.shape != (3,) or final_value.dtype != np.float64:
        raise ValueError("final tool position must be float64 shape (3,)")
    if not np.all(np.isfinite(final_value)):
        raise ValueError("final tool position must be finite")
    p_rows.append(final_value.copy())
    residual_rows.append((target - final_value).copy())
    return DLSHistory(
        q_float64=np.stack(q_rows),
        tool_position_float64=np.stack(p_rows),
        residual_float64=np.stack(residual_rows),
        jacobian_float64=np.stack(jacobian_rows),
        dq_float64=np.stack(dq_rows),
    )


def certify_history_regeneration(left: DLSHistory, right: DLSHistory) -> dict:
    expected_shapes = {
        "q_float64": (9, 7),
        "tool_position_float64": (9, 3),
        "residual_float64": (9, 3),
        "jacobian_float64": (8, 3, 7),
        "dq_float64": (8, 7),
    }
    exact = {}
    for name, shape in expected_shapes.items():
        a = getattr(left, name)
        b = getattr(right, name)
        exact[name] = bool(
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.shape == b.shape == shape
            and a.dtype == b.dtype == np.float64
            and np.all(np.isfinite(a))
            and np.all(np.isfinite(b))
            and np.array_equal(a, b)
            and _history_hash(a) == _history_hash(b)
        )
    return {
        "arrays_exact": exact,
        "all_history_regeneration_gates_pass": all(exact.values()),
    }


def certify_witness_regeneration(
    witness: V3ConstructionWitness, regenerated: DLSHistory
) -> dict:
    history_gate = certify_history_regeneration(witness.history, regenerated)
    stored_hashes_exact = bool(
        witness.history_hashes == witness.history.hashes() == regenerated.hashes()
    )
    return {
        **history_gate,
        "stored_history_hashes_exact": stored_hashes_exact,
        "all_witness_regeneration_gates_pass": bool(
            history_gate["all_history_regeneration_gates_pass"]
            and stored_hashes_exact
        ),
    }


def q_history_gate(history: DLSHistory, lower, upper) -> dict:
    q = history.q_float64
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    exact_shapes = (
        isinstance(q, np.ndarray)
        and q.shape == (DLS_ITERATIONS + 1, 7)
        and q.dtype == np.float64
        and lower.shape == upper.shape == (7,)
        and lower.dtype == upper.dtype == np.float64
    )
    finite = bool(exact_shapes and np.all(np.isfinite(q)))
    if exact_shapes:
        hard_signed = np.minimum(q - lower, upper - q)
        reserve_signed = hard_signed - GOAL_JOINT_MARGIN_RAD
        intermediate_hard = hard_signed[1:8]
        intermediate_reserve = reserve_signed[1:8]
        hard_flat_index = int(np.argmin(intermediate_hard))
        reserve_flat_index = int(np.argmin(intermediate_reserve))
        hard_local = np.unravel_index(hard_flat_index, intermediate_hard.shape)
        reserve_local = np.unravel_index(
            reserve_flat_index, intermediate_reserve.shape
        )
        hard_worst_iterate_per_joint = np.argmin(
            intermediate_hard, axis=0
        ) + 1
        reserve_worst_iterate_per_joint = np.argmin(
            intermediate_reserve, axis=0
        ) + 1
    else:
        hard_signed = np.full((9, 7), np.nan)
        reserve_signed = np.full((9, 7), np.nan)
        intermediate_hard = hard_signed[1:8]
        intermediate_reserve = reserve_signed[1:8]
        hard_local = reserve_local = (0, 0)
        hard_worst_iterate_per_joint = np.zeros(7, dtype=np.int64)
        reserve_worst_iterate_per_joint = np.zeros(7, dtype=np.int64)
    q0_margin = bool(finite and np.all(reserve_signed[0] >= 0.0))
    q8_margin = bool(finite and np.all(reserve_signed[8] >= 0.0))
    return {
        "all_nine_q_iterates_exact_shape_dtype": exact_shapes,
        "all_nine_q_iterates_finite": finite,
        "q0_inside_frozen_reserve": q0_margin,
        "q8_inside_frozen_reserve": q8_margin,
        "intermediate_numerical_ik_iterates_report_only": True,
        "intermediate_iterates_never_executed": True,
        "intermediate_iterates_never_public": True,
        "intermediate_iterates_never_exposed_to_public_task_or_benchmark": True,
        "intermediate_iterates_never_used_by_initializer": True,
        "intermediate_iterates_never_used_by_sqp": True,
        "intermediate_hard_signed_margin_float64": intermediate_hard.tolist(),
        "intermediate_reserve_signed_margin_float64": intermediate_reserve.tolist(),
        "intermediate_hard_min_per_joint": np.min(
            intermediate_hard, axis=0
        ).tolist(),
        "intermediate_reserve_min_per_joint": np.min(
            intermediate_reserve, axis=0
        ).tolist(),
        "intermediate_min_hard_signed_margin": float(np.min(intermediate_hard)),
        "intermediate_min_reserve_signed_margin": float(
            np.min(intermediate_reserve)
        ),
        "intermediate_worst_hard_iterate_index": int(hard_local[0] + 1),
        "intermediate_worst_hard_joint_index": int(hard_local[1]),
        "intermediate_worst_reserve_iterate_index": int(reserve_local[0] + 1),
        "intermediate_worst_reserve_joint_index": int(reserve_local[1]),
        "intermediate_worst_hard_iterate_per_joint": (
            hard_worst_iterate_per_joint.tolist()
        ),
        "intermediate_worst_reserve_iterate_per_joint": (
            reserve_worst_iterate_per_joint.tolist()
        ),
        "intermediate_all_inside_hard_limits_report_only": bool(
            finite and np.all(intermediate_hard >= 0.0)
        ),
        "intermediate_all_inside_reserve_report_only": bool(
            finite and np.all(intermediate_reserve >= 0.0)
        ),
        "all_q_history_gates_pass": bool(
            exact_shapes and finite and q0_margin and q8_margin
        ),
    }


def _pin_position_and_jacobian(model, data):
    tool_id = model.getFrameId(TOOL_FRAME)

    def evaluate(q: np.ndarray):
        position = tool_position(model, data, q).astype(np.float64, copy=False)
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        jacobian = pin.getFrameJacobian(
            model, data, tool_id, pin.LOCAL_WORLD_ALIGNED
        )[:3, :].astype(np.float64, copy=True)
        return position, jacobian

    return evaluate


def regenerate_witness_history(model, witness: V3ConstructionWitness) -> DLSHistory:
    """Independently replay the frozen construction recurrence for auditing."""

    data = model.createData()
    evaluator = _pin_position_and_jacobian(model, data)
    position_only = lambda q: tool_position(model, data, q).astype(
        np.float64, copy=False
    )
    return exact_eight_dls(
        witness.history.q_float64[0].copy(),
        np.asarray(witness.requested_target_xyz, dtype=np.float64),
        evaluator,
        final_position=position_only,
    )


def construct_task_from_draws(model, q0_jitter, phi, offset_sign, *, task_seed=-1):
    """Construct one public task and its quarantined exact-history witness."""

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
    data = model.createData()
    evaluator = _pin_position_and_jacobian(model, data)
    position_only = lambda q: tool_position(model, data, q).astype(
        np.float64, copy=False
    )
    start = position_only(q0)
    delta = REQUESTED_TRAVEL_M * np.asarray(
        [np.cos(phi), np.sin(phi), 0.0], dtype=np.float64
    )
    requested_target = start + delta
    history = exact_eight_dls(
        q0, requested_target, evaluator, final_position=position_only
    )
    if not q_history_gate(history, lower, upper)["all_q_history_gates_pass"]:
        raise RuntimeError("a V3 physical endpoint violates the frozen joint gate")
    goal = history.tool_position_float64[-1]
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
    cylinder = 0.5 * (start[:2] + goal[:2]) + offset_sign * CYLINDER_OFFSET_M * normal
    default_side = -int(offset_sign)
    toll_xy = cylinder + default_side * TOLL_OFFSET_FROM_CYLINDER_M * normal
    reference = TollReference(
        goal_xyz=tuple(goal),
        cylinder_xy=tuple(cylinder),
        physical_radius_m=PHYSICAL_RADIUS_M,
        toll_xy=tuple(toll_xy),
        toll_sigma_m=TOLL_SIGMA_M,
        clearance_margin_m=CLEARANCE_MARGIN_M,
    )
    ref32 = reference.as_float32()
    endpoint_clearance = min(
        physical_clearance(start[:2], ref32), physical_clearance(goal[:2], ref32)
    )
    if endpoint_clearance < MIN_ENDPOINT_PHYSICAL_CLEARANCE_M:
        raise RuntimeError("endpoint physical clearance violates the frozen assertion")
    cylinder_delta = cylinder - start[:2]
    chord_distance = float(
        abs(direction[0] * cylinder_delta[1] - direction[1] * cylinder_delta[0])
    )
    chord_intersects_physical = chord_distance < PHYSICAL_RADIUS_M
    chord_intersects_keepout = chord_distance < PHYSICAL_RADIUS_M + CLEARANCE_MARGIN_M
    if not chord_intersects_physical or not chord_intersects_keepout:
        raise RuntimeError("direct chord does not intersect the frozen cylinder")

    x0 = np.concatenate([q0, np.zeros(model.nv, dtype=np.float64)]).astype(np.float32)
    task = TollTask(
        task_seed=int(task_seed),
        x0=tuple(float(value) for value in x0),
        reference=TollReference.from_solver_bytes(ref32),
        default_side=default_side,
        solver_x0_sha256=_sha256_array(x0),
        solver_reference_sha256=_sha256_array(ref32),
    )
    witness = V3ConstructionWitness(
        q0_jitter=tuple(float(value) for value in jitter),
        phi=float(phi),
        offset_sign=int(offset_sign),
        requested_target_xyz=tuple(float(value) for value in requested_target),
        actual_travel_m=actual_travel,
        planar_travel_m=planar_travel,
        endpoint_physical_clearance_m=endpoint_clearance,
        chord_min_distance_to_cylinder_m=chord_distance,
        chord_intersects_physical_cylinder=chord_intersects_physical,
        chord_intersects_optimizer_keepout=chord_intersects_keepout,
        history=history,
        history_hashes=history.hashes(),
    )
    return task, witness


def generate_task(task_seed, model, *, authorization=None):
    validate_v3_task_seed(task_seed)
    if (
        TASK_CONSTRUCTION_AUTHORIZATION is None
        or authorization is not TASK_CONSTRUCTION_AUTHORIZATION
    ):
        raise RuntimeError("V3 task instantiation is blocked pending verifier authorization")
    rng = np.random.default_rng(int(task_seed))
    jitter = rng.uniform(-Q0_JITTER_RAD, Q0_JITTER_RAD, size=7)
    phi = float(rng.uniform(-PLANAR_ANGLE_RAD, PLANAR_ANGLE_RAD))
    sign = int(rng.choice((-1, 1)))
    return construct_task_from_draws(model, jitter, phi, sign, task_seed=task_seed)
