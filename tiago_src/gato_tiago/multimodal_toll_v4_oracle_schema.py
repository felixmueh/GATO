"""Quarantined deterministic box-constrained task constructor for V4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
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
from gato_tiago.multimodal_toll_v4 import (
    DLS_DAMPING,
    DLS_ITERATIONS,
    DLS_UNIT_STEP,
    FACE_COUNT,
    FACE_FEASIBILITY_TOLERANCE,
    FACE_STATUS_ORDER,
    FREE_NORMAL_RESIDUAL_TOLERANCE,
    OBJECTIVE_TIE_ABSOLUTE_TOLERANCE,
    SELECTED_KKT_TOLERANCE,
    VERIFY_DQ_MAXABS_TOLERANCE,
    VERIFY_GLOBAL_DOMINANCE_TOLERANCE,
    VERIFY_OBJECTIVE_TOLERANCE,
    validate_v4_task_seed,
)


TASK_CONSTRUCTION_AUTHORIZATION = object()


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
    increment_lower_float64: np.ndarray
    increment_upper_float64: np.ndarray
    face_status_int8: np.ndarray
    face_dq_float64: np.ndarray
    face_objective_float64: np.ndarray
    face_primal_violation_float64: np.ndarray
    face_free_residual_float64: np.ndarray
    face_feasible_bool: np.ndarray
    selected_face_index_int64: np.ndarray
    selected_status_int8: np.ndarray
    selected_objective_float64: np.ndarray
    selected_gradient_float64: np.ndarray
    selected_lower_slack_float64: np.ndarray
    selected_upper_slack_float64: np.ndarray
    selected_lower_dual_float64: np.ndarray
    selected_upper_dual_float64: np.ndarray
    selected_primal_violation_float64: np.ndarray
    selected_free_residual_float64: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    def hashes(self) -> dict[str, str]:
        return {name: _history_hash(value) for name, value in self.arrays().items()}


@dataclass(frozen=True)
class V4ConstructionWitness:
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


def _objective(jacobian, residual, dq) -> float:
    defect = jacobian @ dq - residual
    return float(
        0.5 * np.dot(defect, defect)
        + 0.5 * DLS_DAMPING**2 * np.dot(dq, dq)
    )


def _constructor_status_table() -> np.ndarray:
    return np.asarray(
        list(itertools.product(FACE_STATUS_ORDER, repeat=7)), dtype=np.int8
    )


def _enumerate_constructor_faces(jacobian, residual, lower, upper) -> dict:
    """Exhaust all faces in frozen lexicographic order; never prune."""

    hessian = jacobian.T @ jacobian + DLS_DAMPING**2 * np.eye(7)
    rhs = jacobian.T @ residual
    statuses = _constructor_status_table()
    face_dq = np.empty((FACE_COUNT, 7), dtype=np.float64)
    objectives = np.empty(FACE_COUNT, dtype=np.float64)
    primal = np.empty(FACE_COUNT, dtype=np.float64)
    free_residual = np.empty(FACE_COUNT, dtype=np.float64)
    feasible = np.zeros(FACE_COUNT, dtype=np.bool_)

    for face_index, status in enumerate(statuses):
        dq = np.empty(7, dtype=np.float64)
        lower_active = status == -1
        free = status == 0
        upper_active = status == 1
        dq[lower_active] = lower[lower_active]
        dq[upper_active] = upper[upper_active]
        if np.any(free):
            active = ~free
            reduced_rhs = rhs[free] - hessian[np.ix_(free, active)] @ dq[active]
            dq[free] = np.linalg.solve(hessian[np.ix_(free, free)], reduced_rhs)
        gradient = hessian @ dq - rhs
        bound_violation = float(
            max(
                0.0,
                float(np.max(lower - dq)),
                float(np.max(dq - upper)),
            )
        )
        normal_residual = (
            float(np.max(np.abs(gradient[free]))) if np.any(free) else 0.0
        )
        face_dq[face_index] = dq
        objectives[face_index] = _objective(jacobian, residual, dq)
        primal[face_index] = bound_violation
        free_residual[face_index] = normal_residual
        feasible[face_index] = bool(
            np.all(np.isfinite(dq))
            and np.isfinite(objectives[face_index])
            and bound_violation <= FACE_FEASIBILITY_TOLERANCE
            and normal_residual <= FREE_NORMAL_RESIDUAL_TOLERANCE
        )

    feasible_indices = np.flatnonzero(feasible)
    if feasible_indices.size == 0:
        raise RuntimeError("box-DLS exhaustive enumeration found no feasible face")
    minimum = float(np.min(objectives[feasible_indices]))
    selected_candidates = feasible_indices[
        objectives[feasible_indices]
        <= minimum + OBJECTIVE_TIE_ABSOLUTE_TOLERANCE
    ]
    selected = int(selected_candidates[0])
    return {
        "status": statuses,
        "dq": face_dq,
        "objective": objectives,
        "primal": primal,
        "free_residual": free_residual,
        "feasible": feasible,
        "selected": selected,
        "hessian": hessian,
        "rhs": rhs,
    }


def _independent_status_from_code(code: int) -> np.ndarray:
    values = np.empty(7, dtype=np.int8)
    remaining = int(code)
    for joint in range(7):
        divisor = 3 ** (6 - joint)
        digit = remaining // divisor
        remaining %= divisor
        values[joint] = (-1, 0, 1)[digit]
    return values


def _enumerate_verifier_faces(jacobian, residual, lower, upper) -> dict:
    """Independent re-enumerator; it does not call the constructor enumerator."""

    hessian = np.matmul(jacobian.T, jacobian)
    hessian = hessian + np.eye(7, dtype=np.float64) * (DLS_DAMPING**2)
    rhs = np.matmul(jacobian.T, residual)
    statuses = np.empty((FACE_COUNT, 7), dtype=np.int8)
    face_dq = np.empty((FACE_COUNT, 7), dtype=np.float64)
    objectives = np.empty(FACE_COUNT, dtype=np.float64)
    primal = np.empty(FACE_COUNT, dtype=np.float64)
    free_residual = np.empty(FACE_COUNT, dtype=np.float64)
    feasible = np.zeros(FACE_COUNT, dtype=np.bool_)

    for face_index in range(FACE_COUNT):
        status = _independent_status_from_code(face_index)
        statuses[face_index] = status
        dq = np.empty(7, dtype=np.float64)
        free_indices = np.flatnonzero(status == 0)
        fixed_indices = np.flatnonzero(status != 0)
        for joint in fixed_indices:
            dq[joint] = lower[joint] if status[joint] == -1 else upper[joint]
        if free_indices.size:
            free_hessian = hessian[np.ix_(free_indices, free_indices)]
            coupling = hessian[np.ix_(free_indices, fixed_indices)]
            right = rhs[free_indices] - coupling.dot(dq[fixed_indices])
            dq[free_indices] = np.linalg.solve(free_hessian, right)
        gradient = hessian.dot(dq) - rhs
        lower_violation = float(np.max(lower - dq))
        upper_violation = float(np.max(dq - upper))
        violation = max(0.0, lower_violation, upper_violation)
        residual_norm = (
            float(np.max(np.abs(gradient[free_indices])))
            if free_indices.size
            else 0.0
        )
        defect = jacobian.dot(dq) - residual
        objective = float(
            0.5 * defect.dot(defect)
            + 0.5 * DLS_DAMPING**2 * dq.dot(dq)
        )
        face_dq[face_index] = dq
        objectives[face_index] = objective
        primal[face_index] = violation
        free_residual[face_index] = residual_norm
        feasible[face_index] = bool(
            np.isfinite(dq).all()
            and np.isfinite(objective)
            and violation <= FACE_FEASIBILITY_TOLERANCE
            and residual_norm <= FREE_NORMAL_RESIDUAL_TOLERANCE
        )

    feasible_indices = np.flatnonzero(feasible)
    if feasible_indices.size == 0:
        raise RuntimeError("independent box-DLS enumeration found no feasible face")
    minimum = float(np.min(objectives[feasible_indices]))
    candidates = feasible_indices[
        objectives[feasible_indices]
        <= minimum + OBJECTIVE_TIE_ABSOLUTE_TOLERANCE
    ]
    return {
        "status": statuses,
        "dq": face_dq,
        "objective": objectives,
        "primal": primal,
        "free_residual": free_residual,
        "feasible": feasible,
        "selected": int(candidates[0]),
        "hessian": hessian,
        "rhs": rhs,
    }


def _selected_diagnostics(table: dict, lower, upper) -> dict:
    index = table["selected"]
    status = table["status"][index]
    dq = table["dq"][index]
    gradient = table["hessian"] @ dq - table["rhs"]
    lower_dual = np.where(status == -1, gradient, 0.0)
    upper_dual = np.where(status == 1, -gradient, 0.0)
    return {
        "index": index,
        "status": status.copy(),
        "dq": dq.copy(),
        "objective": float(table["objective"][index]),
        "gradient": gradient,
        "lower_slack": dq - lower,
        "upper_slack": upper - dq,
        "lower_dual": lower_dual,
        "upper_dual": upper_dual,
        "primal": float(table["primal"][index]),
        "free_residual": float(table["free_residual"][index]),
    }


def _build_history(
    q0,
    target,
    position_and_jacobian,
    lower_joint,
    upper_joint,
    final_position,
    *,
    enumerate_faces,
) -> DLSHistory:
    q = np.asarray(q0)
    target = np.asarray(target)
    lower_joint = np.asarray(lower_joint)
    upper_joint = np.asarray(upper_joint)
    if q.shape != (7,) or q.dtype != np.float64 or not np.isfinite(q).all():
        raise ValueError("q0 must be one finite float64 seven-vector")
    if target.shape != (3,) or target.dtype != np.float64 or not np.isfinite(target).all():
        raise ValueError("target must be one finite float64 xyz vector")
    if (
        lower_joint.shape != (7,)
        or upper_joint.shape != (7,)
        or lower_joint.dtype != np.float64
        or upper_joint.dtype != np.float64
    ):
        raise ValueError("joint bounds must be float64 seven-vectors")

    rows = {
        "q": [q.copy()],
        "p": [],
        "residual": [],
        "jacobian": [],
        "dq": [],
        "increment_lower": [],
        "increment_upper": [],
        "face_status": [],
        "face_dq": [],
        "face_objective": [],
        "face_primal": [],
        "face_free_residual": [],
        "face_feasible": [],
        "selected_index": [],
        "selected_status": [],
        "selected_objective": [],
        "selected_gradient": [],
        "selected_lower_slack": [],
        "selected_upper_slack": [],
        "selected_lower_dual": [],
        "selected_upper_dual": [],
        "selected_primal": [],
        "selected_free_residual": [],
    }
    for _ in range(DLS_ITERATIONS):
        position, jacobian = position_and_jacobian(q)
        position = np.asarray(position)
        jacobian = np.asarray(jacobian)
        if position.shape != (3,) or position.dtype != np.float64:
            raise ValueError("tool position callback must return float64 shape (3,)")
        if jacobian.shape != (3, 7) or jacobian.dtype != np.float64:
            raise ValueError("tool Jacobian callback must return float64 shape (3,7)")
        if not np.isfinite(position).all() or not np.isfinite(jacobian).all():
            raise ValueError("DLS callback outputs must be finite")
        residual = target - position
        increment_lower = lower_joint + GOAL_JOINT_MARGIN_RAD - q
        increment_upper = upper_joint - GOAL_JOINT_MARGIN_RAD - q
        if np.any(increment_lower > increment_upper):
            raise RuntimeError("box-DLS increment bounds are empty")
        table = enumerate_faces(
            jacobian, residual, increment_lower, increment_upper
        )
        selected = _selected_diagnostics(
            table, increment_lower, increment_upper
        )
        rows["p"].append(position.copy())
        rows["residual"].append(residual.copy())
        rows["jacobian"].append(jacobian.copy())
        rows["dq"].append(selected["dq"])
        rows["increment_lower"].append(increment_lower.copy())
        rows["increment_upper"].append(increment_upper.copy())
        rows["face_status"].append(table["status"].copy())
        rows["face_dq"].append(table["dq"].copy())
        rows["face_objective"].append(table["objective"].copy())
        rows["face_primal"].append(table["primal"].copy())
        rows["face_free_residual"].append(table["free_residual"].copy())
        rows["face_feasible"].append(table["feasible"].copy())
        rows["selected_index"].append(selected["index"])
        rows["selected_status"].append(selected["status"])
        rows["selected_objective"].append(selected["objective"])
        rows["selected_gradient"].append(selected["gradient"])
        rows["selected_lower_slack"].append(selected["lower_slack"])
        rows["selected_upper_slack"].append(selected["upper_slack"])
        rows["selected_lower_dual"].append(selected["lower_dual"])
        rows["selected_upper_dual"].append(selected["upper_dual"])
        rows["selected_primal"].append(selected["primal"])
        rows["selected_free_residual"].append(selected["free_residual"])
        q = q + DLS_UNIT_STEP * selected["dq"]
        rows["q"].append(q.copy())

    final_value = np.asarray(final_position(q))
    if final_value.shape != (3,) or final_value.dtype != np.float64:
        raise ValueError("final tool position must be float64 shape (3,)")
    if not np.isfinite(final_value).all():
        raise ValueError("final tool position must be finite")
    rows["p"].append(final_value.copy())
    rows["residual"].append((target - final_value).copy())
    return DLSHistory(
        q_float64=np.stack(rows["q"]),
        tool_position_float64=np.stack(rows["p"]),
        residual_float64=np.stack(rows["residual"]),
        jacobian_float64=np.stack(rows["jacobian"]),
        dq_float64=np.stack(rows["dq"]),
        increment_lower_float64=np.stack(rows["increment_lower"]),
        increment_upper_float64=np.stack(rows["increment_upper"]),
        face_status_int8=np.stack(rows["face_status"]),
        face_dq_float64=np.stack(rows["face_dq"]),
        face_objective_float64=np.stack(rows["face_objective"]),
        face_primal_violation_float64=np.stack(rows["face_primal"]),
        face_free_residual_float64=np.stack(rows["face_free_residual"]),
        face_feasible_bool=np.stack(rows["face_feasible"]),
        selected_face_index_int64=np.asarray(rows["selected_index"], dtype=np.int64),
        selected_status_int8=np.stack(rows["selected_status"]),
        selected_objective_float64=np.asarray(rows["selected_objective"], dtype=np.float64),
        selected_gradient_float64=np.stack(rows["selected_gradient"]),
        selected_lower_slack_float64=np.stack(rows["selected_lower_slack"]),
        selected_upper_slack_float64=np.stack(rows["selected_upper_slack"]),
        selected_lower_dual_float64=np.stack(rows["selected_lower_dual"]),
        selected_upper_dual_float64=np.stack(rows["selected_upper_dual"]),
        selected_primal_violation_float64=np.asarray(rows["selected_primal"], dtype=np.float64),
        selected_free_residual_float64=np.asarray(rows["selected_free_residual"], dtype=np.float64),
    )


def exact_eight_box_dls(
    q0,
    requested_target_xyz,
    position_and_jacobian,
    lower_joint,
    upper_joint,
    *,
    final_position,
) -> DLSHistory:
    return _build_history(
        q0,
        requested_target_xyz,
        position_and_jacobian,
        lower_joint,
        upper_joint,
        final_position,
        enumerate_faces=_enumerate_constructor_faces,
    )


def independently_reenumerate_eight_box_dls(
    q0,
    requested_target_xyz,
    position_and_jacobian,
    lower_joint,
    upper_joint,
    *,
    final_position,
) -> DLSHistory:
    return _build_history(
        q0,
        requested_target_xyz,
        position_and_jacobian,
        lower_joint,
        upper_joint,
        final_position,
        enumerate_faces=_enumerate_verifier_faces,
    )


def _selected_kkt(history: DLSHistory) -> dict:
    gradients = []
    lower_slacks = []
    upper_slacks = []
    lower_duals = []
    upper_duals = []
    for step in range(DLS_ITERATIONS):
        jacobian = history.jacobian_float64[step]
        residual = history.residual_float64[step]
        dq = history.dq_float64[step]
        gradient = (
            jacobian.T @ jacobian + DLS_DAMPING**2 * np.eye(7)
        ) @ dq - jacobian.T @ residual
        status = history.selected_status_int8[step]
        gradients.append(gradient)
        lower_slacks.append(dq - history.increment_lower_float64[step])
        upper_slacks.append(history.increment_upper_float64[step] - dq)
        lower_duals.append(np.where(status == -1, gradient, 0.0))
        upper_duals.append(np.where(status == 1, -gradient, 0.0))
    gradient = np.stack(gradients)
    lower_slack = np.stack(lower_slacks)
    upper_slack = np.stack(upper_slacks)
    lower_dual = np.stack(lower_duals)
    upper_dual = np.stack(upper_duals)
    primal = float(max(0.0, np.max(-lower_slack), np.max(-upper_slack)))
    free = np.where(history.selected_status_int8 == 0, np.abs(gradient), 0.0)
    lower_sign = np.where(
        history.selected_status_int8 == -1,
        np.maximum(-gradient, 0.0),
        0.0,
    )
    upper_sign = np.where(
        history.selected_status_int8 == 1,
        np.maximum(gradient, 0.0),
        0.0,
    )
    free_max = float(np.max(free))
    lower_max = float(np.max(lower_sign))
    upper_max = float(np.max(upper_sign))
    stationarity = (
        gradient - lower_dual + upper_dual
    )
    stationarity_max = float(np.max(np.abs(stationarity)))
    lower_complementarity = float(
        np.max(
            np.abs(
                lower_dual * lower_slack
            )
        )
    )
    upper_complementarity = float(
        np.max(
            np.abs(
                upper_dual * upper_slack
            )
        )
    )
    return {
        "primal_max": primal,
        "free_gradient_maxabs": free_max,
        "lower_dual_sign_violation": lower_max,
        "upper_dual_sign_violation": upper_max,
        "stationarity_maxabs": stationarity_max,
        "lower_complementarity_maxabs": lower_complementarity,
        "upper_complementarity_maxabs": upper_complementarity,
        "retained_gradient_maxabs_error": float(
            np.max(np.abs(history.selected_gradient_float64 - gradient))
        ),
        "retained_lower_slack_maxabs_error": float(
            np.max(np.abs(history.selected_lower_slack_float64 - lower_slack))
        ),
        "retained_upper_slack_maxabs_error": float(
            np.max(np.abs(history.selected_upper_slack_float64 - upper_slack))
        ),
        "retained_lower_dual_maxabs_error": float(
            np.max(np.abs(history.selected_lower_dual_float64 - lower_dual))
        ),
        "retained_upper_dual_maxabs_error": float(
            np.max(np.abs(history.selected_upper_dual_float64 - upper_dual))
        ),
        "retained_primal_maxabs_error": float(
            np.max(
                np.abs(
                    history.selected_primal_violation_float64
                    - np.asarray(
                        [
                            max(
                                0.0,
                                float(np.max(-lower_slack[k])),
                                float(np.max(-upper_slack[k])),
                            )
                            for k in range(DLS_ITERATIONS)
                        ],
                        dtype=np.float64,
                    )
                )
            )
        ),
        "retained_free_residual_maxabs_error": float(
            np.max(
                np.abs(
                    history.selected_free_residual_float64
                    - np.max(free, axis=1)
                )
            )
        ),
        "selected_kkt_max": max(
            primal,
            free_max,
            lower_max,
            upper_max,
            stationarity_max,
            lower_complementarity,
            upper_complementarity,
            float(np.max(np.abs(history.selected_gradient_float64 - gradient))),
            float(np.max(np.abs(history.selected_lower_slack_float64 - lower_slack))),
            float(np.max(np.abs(history.selected_upper_slack_float64 - upper_slack))),
            float(np.max(np.abs(history.selected_lower_dual_float64 - lower_dual))),
            float(np.max(np.abs(history.selected_upper_dual_float64 - upper_dual))),
            float(
                np.max(
                    np.abs(
                        history.selected_primal_violation_float64
                        - np.asarray(
                            [
                                max(
                                    0.0,
                                    float(np.max(-lower_slack[k])),
                                    float(np.max(-upper_slack[k])),
                                )
                                for k in range(DLS_ITERATIONS)
                            ],
                            dtype=np.float64,
                        )
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        history.selected_free_residual_float64
                        - np.max(free, axis=1)
                    )
                )
            ),
        ),
    }


def _history_schema_gate(history: DLSHistory) -> bool:
    shapes_and_dtypes = {
        "q_float64": ((9, 7), np.float64),
        "tool_position_float64": ((9, 3), np.float64),
        "residual_float64": ((9, 3), np.float64),
        "jacobian_float64": ((8, 3, 7), np.float64),
        "dq_float64": ((8, 7), np.float64),
        "increment_lower_float64": ((8, 7), np.float64),
        "increment_upper_float64": ((8, 7), np.float64),
        "face_status_int8": ((8, FACE_COUNT, 7), np.int8),
        "face_dq_float64": ((8, FACE_COUNT, 7), np.float64),
        "face_objective_float64": ((8, FACE_COUNT), np.float64),
        "face_primal_violation_float64": ((8, FACE_COUNT), np.float64),
        "face_free_residual_float64": ((8, FACE_COUNT), np.float64),
        "face_feasible_bool": ((8, FACE_COUNT), np.bool_),
        "selected_face_index_int64": ((8,), np.int64),
        "selected_status_int8": ((8, 7), np.int8),
        "selected_objective_float64": ((8,), np.float64),
        "selected_gradient_float64": ((8, 7), np.float64),
        "selected_lower_slack_float64": ((8, 7), np.float64),
        "selected_upper_slack_float64": ((8, 7), np.float64),
        "selected_lower_dual_float64": ((8, 7), np.float64),
        "selected_upper_dual_float64": ((8, 7), np.float64),
        "selected_primal_violation_float64": ((8,), np.float64),
        "selected_free_residual_float64": ((8,), np.float64),
    }
    arrays = history.arrays()
    if set(arrays) != set(shapes_and_dtypes):
        return False
    expected_status = _constructor_status_table()
    return bool(
        all(
            isinstance(arrays[name], np.ndarray)
            and arrays[name].shape == shape
            and arrays[name].dtype == dtype
            for name, (shape, dtype) in shapes_and_dtypes.items()
        )
        and np.all((history.face_status_int8 >= -1) & (history.face_status_int8 <= 1))
        and all(
            np.array_equal(history.face_status_int8[k], expected_status)
            for k in range(DLS_ITERATIONS)
        )
        and np.all(
            (history.selected_face_index_int64 >= 0)
            & (history.selected_face_index_int64 < FACE_COUNT)
        )
        and all(
            np.all(np.isfinite(value))
            for value in arrays.values()
            if value.dtype != np.bool_
        )
        and np.all(np.any(history.face_feasible_bool, axis=1))
    )


def certify_history_regeneration(left: DLSHistory, right: DLSHistory) -> dict:
    left_arrays = left.arrays()
    right_arrays = right.arrays()
    exact_schema = _history_schema_gate(left) and _history_schema_gate(right)
    if not exact_schema:
        return {
            "exact_array_schema": False,
            "all_arrays_finite": False,
            "status_and_feasibility_tables_exact": False,
            "selected_dq_agrees": False,
            "selected_objective_agrees": False,
            "all_face_dq_agree": False,
            "all_face_objectives_agree": False,
            "all_face_primal_agrees": False,
            "all_face_free_residual_agrees": False,
            "global_dominance_pass": False,
            "retained_global_dominance_pass": False,
            "selected_face_table_binding_exact": False,
            "independent_selected_identity_exact": False,
            "selected_kkt_pass": False,
            "independent_verifier_calls_constructor_enumerator": False,
            "all_history_regeneration_gates_pass": False,
        }
    status_exact = bool(
        exact_schema
        and np.array_equal(left.face_status_int8, right.face_status_int8)
        and np.array_equal(left.face_feasible_bool, right.face_feasible_bool)
    )
    selected_dq_maxabs = float(
        np.max(np.abs(left.dq_float64 - right.dq_float64))
    )
    core_selected_maxabs = max(
        float(np.max(np.abs(left.q_float64 - right.q_float64))),
        float(
            np.max(
                np.abs(
                    left.tool_position_float64
                    - right.tool_position_float64
                )
            )
        ),
        float(np.max(np.abs(left.residual_float64 - right.residual_float64))),
        float(np.max(np.abs(left.jacobian_float64 - right.jacobian_float64))),
        float(
            np.max(
                np.abs(
                    left.increment_lower_float64
                    - right.increment_lower_float64
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    left.increment_upper_float64
                    - right.increment_upper_float64
                )
            )
        ),
    )
    selected_objective_maxabs = float(
        np.max(
            np.abs(
                left.selected_objective_float64
                - right.selected_objective_float64
            )
        )
    )
    face_dq_maxabs = float(
        np.max(np.abs(left.face_dq_float64 - right.face_dq_float64))
    )
    face_objective_maxabs = float(
        np.max(
            np.abs(
                left.face_objective_float64 - right.face_objective_float64
            )
        )
    )
    face_primal_maxabs = float(
        np.max(
            np.abs(
                left.face_primal_violation_float64
                - right.face_primal_violation_float64
            )
        )
    )
    face_free_residual_maxabs = float(
        np.max(
            np.abs(
                left.face_free_residual_float64
                - right.face_free_residual_float64
            )
        )
    )
    independent_minimum = np.asarray(
        [
            np.min(right.face_objective_float64[k, right.face_feasible_bool[k]])
            for k in range(DLS_ITERATIONS)
        ],
        dtype=np.float64,
    )
    global_dominance = float(
        np.max(left.selected_objective_float64 - independent_minimum)
    )
    constructor_minimum = np.asarray(
        [
            np.min(left.face_objective_float64[k, left.face_feasible_bool[k]])
            for k in range(DLS_ITERATIONS)
        ],
        dtype=np.float64,
    )
    retained_global_dominance = float(
        np.max(left.selected_objective_float64 - constructor_minimum)
    )
    selected_table_binding = bool(
        all(
            np.array_equal(
                left.selected_status_int8[k],
                left.face_status_int8[k, left.selected_face_index_int64[k]],
            )
            and np.array_equal(
                left.dq_float64[k],
                left.face_dq_float64[k, left.selected_face_index_int64[k]],
            )
            and left.selected_objective_float64[k]
            == left.face_objective_float64[
                k, left.selected_face_index_int64[k]
            ]
            for k in range(DLS_ITERATIONS)
        )
    )
    independent_selected_identity = bool(
        np.array_equal(
            left.selected_face_index_int64, right.selected_face_index_int64
        )
        and np.array_equal(left.selected_status_int8, right.selected_status_int8)
    )
    kkt = _selected_kkt(left)
    all_finite = bool(
        exact_schema
        and all(
            np.all(np.isfinite(value))
            for value in [*left_arrays.values(), *right_arrays.values()]
            if value.dtype != np.bool_
        )
    )
    return {
        "exact_array_schema": exact_schema,
        "all_arrays_finite": all_finite,
        "status_and_feasibility_tables_exact": status_exact,
        "selected_dq_maxabs": selected_dq_maxabs,
        "selected_dq_agrees": selected_dq_maxabs <= VERIFY_DQ_MAXABS_TOLERANCE,
        "selected_q_p_r_j_bounds_maxabs": core_selected_maxabs,
        "selected_q_p_r_j_bounds_agree": core_selected_maxabs
        <= VERIFY_DQ_MAXABS_TOLERANCE,
        "selected_objective_maxabs": selected_objective_maxabs,
        "selected_objective_agrees": selected_objective_maxabs
        <= VERIFY_OBJECTIVE_TOLERANCE,
        "all_face_dq_maxabs": face_dq_maxabs,
        "all_face_dq_agree": face_dq_maxabs <= VERIFY_DQ_MAXABS_TOLERANCE,
        "all_face_objective_maxabs": face_objective_maxabs,
        "all_face_objectives_agree": face_objective_maxabs
        <= VERIFY_OBJECTIVE_TOLERANCE,
        "all_face_primal_maxabs": face_primal_maxabs,
        "all_face_primal_agrees": face_primal_maxabs
        <= FACE_FEASIBILITY_TOLERANCE,
        "all_face_free_residual_maxabs": face_free_residual_maxabs,
        "all_face_free_residual_agrees": face_free_residual_maxabs
        <= FREE_NORMAL_RESIDUAL_TOLERANCE,
        "global_dominance_max": global_dominance,
        "global_dominance_pass": global_dominance
        <= VERIFY_GLOBAL_DOMINANCE_TOLERANCE,
        "retained_global_dominance_max": retained_global_dominance,
        "retained_global_dominance_pass": retained_global_dominance
        <= VERIFY_GLOBAL_DOMINANCE_TOLERANCE,
        "selected_face_table_binding_exact": selected_table_binding,
        "independent_selected_identity_exact": independent_selected_identity,
        **kkt,
        "selected_kkt_pass": kkt["selected_kkt_max"]
        <= SELECTED_KKT_TOLERANCE,
        "independent_verifier_calls_constructor_enumerator": False,
        "all_history_regeneration_gates_pass": bool(
            exact_schema
            and all_finite
            and status_exact
            and selected_dq_maxabs <= VERIFY_DQ_MAXABS_TOLERANCE
            and core_selected_maxabs <= VERIFY_DQ_MAXABS_TOLERANCE
            and selected_objective_maxabs <= VERIFY_OBJECTIVE_TOLERANCE
            and face_dq_maxabs <= VERIFY_DQ_MAXABS_TOLERANCE
            and face_objective_maxabs <= VERIFY_OBJECTIVE_TOLERANCE
            and face_primal_maxabs <= FACE_FEASIBILITY_TOLERANCE
            and face_free_residual_maxabs <= FREE_NORMAL_RESIDUAL_TOLERANCE
            and global_dominance <= VERIFY_GLOBAL_DOMINANCE_TOLERANCE
            and retained_global_dominance <= VERIFY_GLOBAL_DOMINANCE_TOLERANCE
            and selected_table_binding
            and independent_selected_identity
            and kkt["selected_kkt_max"] <= SELECTED_KKT_TOLERANCE
        ),
    }


def certify_witness_regeneration(
    witness: V4ConstructionWitness, regenerated: DLSHistory
) -> dict:
    history_gate = certify_history_regeneration(witness.history, regenerated)
    stored_hashes_exact = witness.history_hashes == witness.history.hashes()
    return {
        **history_gate,
        "stored_constructor_history_hashes_exact": bool(stored_hashes_exact),
        "all_witness_regeneration_gates_pass": bool(
            history_gate["all_history_regeneration_gates_pass"]
            and stored_hashes_exact
        ),
    }


def q_history_gate(history: DLSHistory, lower, upper) -> dict:
    q = history.q_float64
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    exact = bool(
        q.shape == (9, 7)
        and q.dtype == np.float64
        and history.dq_float64.shape == (8, 7)
        and history.dq_float64.dtype == np.float64
        and lower.shape == upper.shape == (7,)
        and lower.dtype == upper.dtype == np.float64
    )
    finite = bool(exact and np.isfinite(q).all() and np.isfinite(history.dq_float64).all())
    recurrence = bool(
        finite
        and np.array_equal(
            q[1:], q[:-1] + DLS_UNIT_STEP * history.dq_float64
        )
    )
    hard_signed = (
        np.minimum(q - lower, upper - q)
        if exact
        else np.full((9, 7), np.nan)
    )
    reserve_signed = hard_signed - GOAL_JOINT_MARGIN_RAD
    all_reserve = bool(finite and np.all(reserve_signed >= 0.0))
    return {
        "all_q_and_dq_exact_shape_dtype": exact,
        "all_q0_through_q8_finite": finite,
        "exact_recurrence": recurrence,
        "all_q0_through_q8_inside_hard_limits": bool(
            finite and np.all(hard_signed >= 0.0)
        ),
        "all_q0_through_q8_inside_frozen_reserve": all_reserve,
        "hard_signed_margin_float64": hard_signed.tolist(),
        "reserve_signed_margin_float64": reserve_signed.tolist(),
        "minimum_hard_signed_margin": float(np.min(hard_signed)),
        "minimum_reserve_signed_margin": float(np.min(reserve_signed)),
        "worst_hard_iterate_index": int(np.unravel_index(np.argmin(hard_signed), hard_signed.shape)[0]),
        "worst_hard_joint_index": int(np.unravel_index(np.argmin(hard_signed), hard_signed.shape)[1]),
        "worst_reserve_iterate_index": int(np.unravel_index(np.argmin(reserve_signed), reserve_signed.shape)[0]),
        "worst_reserve_joint_index": int(np.unravel_index(np.argmin(reserve_signed), reserve_signed.shape)[1]),
        "numerical_ik_history_oracle_only": True,
        "numerical_ik_history_never_public_or_benchmark": True,
        "numerical_ik_history_never_initializer_or_sqp": True,
        "all_q_history_gates_pass": bool(exact and finite and recurrence and all_reserve),
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


def regenerate_witness_history(model, witness: V4ConstructionWitness) -> DLSHistory:
    data = model.createData()
    evaluator = _pin_position_and_jacobian(model, data)
    position_only = lambda q: tool_position(model, data, q).astype(np.float64, copy=False)
    return independently_reenumerate_eight_box_dls(
        witness.history.q_float64[0].copy(),
        np.asarray(witness.requested_target_xyz, dtype=np.float64),
        evaluator,
        np.asarray(model.lowerPositionLimit, dtype=np.float64),
        np.asarray(model.upperPositionLimit, dtype=np.float64),
        final_position=position_only,
    )


def construct_task_from_draws(model, q0_jitter, phi, offset_sign, *, task_seed=-1):
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
    position_only = lambda q: tool_position(model, data, q).astype(np.float64, copy=False)
    start = position_only(q0)
    requested_target = start + REQUESTED_TRAVEL_M * np.asarray(
        [np.cos(phi), np.sin(phi), 0.0], dtype=np.float64
    )
    history = exact_eight_box_dls(
        q0,
        requested_target,
        evaluator,
        lower,
        upper,
        final_position=position_only,
    )
    if not q_history_gate(history, lower, upper)["all_q_history_gates_pass"]:
        raise RuntimeError("a V4 iterate violates the frozen reserve or recurrence gate")
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
        physical_clearance(start[:2], ref32),
        physical_clearance(goal[:2], ref32),
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
    witness = V4ConstructionWitness(
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
    validate_v4_task_seed(task_seed)
    if (
        TASK_CONSTRUCTION_AUTHORIZATION is None
        or authorization is not TASK_CONSTRUCTION_AUTHORIZATION
    ):
        raise RuntimeError("V4 task instantiation is blocked pending verifier authorization")
    rng = np.random.default_rng(int(task_seed))
    jitter = rng.uniform(-Q0_JITTER_RAD, Q0_JITTER_RAD, size=7)
    phi = float(rng.uniform(-PLANAR_ANGLE_RAD, PLANAR_ANGLE_RAD))
    sign = int(rng.choice((-1, 1)))
    return construct_task_from_draws(model, jitter, phi, sign, task_seed=task_seed)
