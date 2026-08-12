"""Static P1 acceleration-shooting schema and pure certificates.

Production callbacks are deliberately absent.  A later audited runner may
bind exact Pinocchio FK/J/RNEA derivatives and CUDA replay to these equations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Callable, Mapping

import numpy as np

from gato_tiago.circular_portfolio import (
    ABA_RNEA_TOLERANCE,
    ACCELERATION_SCALE_RAD_S2,
    ACCELERATION_WEIGHT,
    BACKEND,
    BACKEND_HESSIAN,
    BACKEND_OPTIONS,
    CAMPAIGN_WALL_LIMIT_S,
    CLOSED_WINDING_RESIDUAL,
    CONSTRUCTION_CLEARANCE_M,
    CONSTRUCTION_LIMIT_RATIO,
    CONSTRUCTION_Q_RESERVE_RAD,
    CONTROL_WEIGHT,
    DENSE_SAMPLES,
    DENSE_SUBSTEPS,
    DLS_DAMPING,
    DLS_STEPS,
    DT,
    ENDPOINT_TOLERANCE,
    EXPECTED_LEDGER,
    FINAL_TOOL_SPEED_TOLERANCE_MPS,
    INTERVALS,
    JERK_WEIGHT,
    KNOTS,
    LINEAR_PROXY_REGULARIZATION,
    MODEL_TOOL_TOLERANCE_M,
    PATH_MAX_TOLERANCE_M,
    PATH_RMS_TOLERANCE_M,
    PATH_SCALE_M,
    PHYSICAL_RADIUS_M,
    PROFILE_WALL_LIMIT_S,
    RECURRENCE_TOLERANCE,
    SHOOTING_INEQUALITIES,
    SHOOTING_VARIABLES,
    TERMINAL_CUDA_TOLERANCE_M,
    TERMINAL_PIN_TOLERANCE_M,
    TASK_IDENTITIES,
    TOLL_WEIGHT,
    CYLINDER_WEIGHT,
    Q_COST,
    N_COST,
    QD_COST,
    U_COST,
    Q_LIMIT_COST,
    VELOCITY_LIMIT_COST,
    CONTROL_LIMIT_COST,
    smoothstep_toll_residual_gradient,
    validate_reference_row,
)


RUNNER_EXECUTION_AUTHORIZATION = object()

PROFILE_ARRAY_SPECS = {
    "proxy_q_before_float64": ((94, DLS_STEPS, 7), np.dtype(np.float64)),
    "proxy_position_float64": ((94, DLS_STEPS, 3), np.dtype(np.float64)),
    "proxy_residual_float64": ((94, DLS_STEPS, 3), np.dtype(np.float64)),
    "proxy_jacobian_float64": ((94, DLS_STEPS, 3, 7), np.dtype(np.float64)),
    "proxy_dq_float64": ((94, DLS_STEPS, 7), np.dtype(np.float64)),
    "proxy_q_float64": ((KNOTS, 7), np.dtype(np.float64)),
    "proxy_kkt_rhs_float64": ((SHOOTING_VARIABLES + 14,), np.dtype(np.float64)),
    "proxy_kkt_solution_float64": ((SHOOTING_VARIABLES + 14,), np.dtype(np.float64)),
    "proxy_kkt_endpoint_residual_float64": ((14,), np.dtype(np.float64)),
    "proxy_kkt_solve_residual_float64": ((), np.dtype(np.float64)),
    "initial_acceleration_float64": ((INTERVALS, 7), np.dtype(np.float64)),
    "acceleration_float64": ((INTERVALS, 7), np.dtype(np.float64)),
    "q_float64": ((KNOTS, 7), np.dtype(np.float64)),
    "qd_float64": ((KNOTS, 7), np.dtype(np.float64)),
    "controls_float64": ((INTERVALS, 7), np.dtype(np.float64)),
    # Exact solver/model replay controls.  These are the float32 CUDA boundary
    # bytes; Pin promotes these same values to float64 for common-input replay.
    "applied_controls_float32": ((INTERVALS, 7), np.dtype(np.float32)),
    "positions_float64": ((KNOTS, 3), np.dtype(np.float64)),
    "endpoint_residual_float64": ((14,), np.dtype(np.float64)),
    "inequalities_float64": ((SHOOTING_INEQUALITIES,), np.dtype(np.float64)),
    "pin_dense_state_float64": ((DENSE_SAMPLES, 14), np.dtype(np.float64)),
    "pin_dense_tool_float64": ((DENSE_SAMPLES, 3), np.dtype(np.float64)),
    "cuda_dense_state_float32": ((DENSE_SAMPLES, 14), np.dtype(np.float32)),
    "cuda_dense_tool_float32": ((DENSE_SAMPLES, 3), np.dtype(np.float32)),
    "pin_full_cost_float64": ((), np.dtype(np.float64)),
    "pin_base_cost_float64": ((), np.dtype(np.float64)),
    "cuda_full_cost_float64": ((), np.dtype(np.float64)),
    "cuda_base_cost_float64": ((), np.dtype(np.float64)),
    "pin_toll_residual_float64": ((KNOTS,), np.dtype(np.float64)),
    "cuda_toll_residual_float64": ((KNOTS,), np.dtype(np.float64)),
    "shooting_objective_float64": ((), np.dtype(np.float64)),
    "optimizer_success_bool": ((), np.dtype(np.bool_)),
    "optimizer_status_int64": ((), np.dtype(np.int64)),
    "optimizer_iterations_int64": ((), np.dtype(np.int64)),
    "optimizer_wall_float64": ((), np.dtype(np.float64)),
}

DERIVATIVE_PROBE_INDICES = (0, 8, 191)
MAX_INEQUALITY_CSR_NNZ = SHOOTING_INEQUALITIES * SHOOTING_VARIABLES
DERIVATIVE_ARRAY_SPECS = {
    "derivative_probe_indices_int64": ((3,), np.dtype(np.int64)),
    "derivative_probe_q_float64": ((3, 7), np.dtype(np.float64)),
    "derivative_probe_qd_float64": ((3, 7), np.dtype(np.float64)),
    "derivative_probe_qdd_float64": ((3, 7), np.dtype(np.float64)),
    "derivative_rnea_dq_float64": ((3, 7, 7), np.dtype(np.float64)),
    "derivative_rnea_dv_float64": ((3, 7, 7), np.dtype(np.float64)),
    "derivative_rnea_mass_float64": ((3, 7, 7), np.dtype(np.float64)),
    "derivative_rnea_mass_fd_float64": ((3, 7, 7), np.dtype(np.float64)),
    "derivative_objective_point_float64": ((3, SHOOTING_VARIABLES), np.dtype(np.float64)),
    "derivative_objective_direction_float64": ((3, SHOOTING_VARIABLES), np.dtype(np.float64)),
    "derivative_objective_gradient_float64": ((3, SHOOTING_VARIABLES), np.dtype(np.float64)),
    "derivative_endpoint_jacobian_float64": ((14, SHOOTING_VARIABLES), np.dtype(np.float64)),
    "derivative_probe0_inequality_dense_float64": ((SHOOTING_INEQUALITIES, SHOOTING_VARIABLES), np.dtype(np.float64)),
}
GLOBAL_ARRAY_SPECS = {
    "proxy_kkt_matrix_float64": ((SHOOTING_VARIABLES + 14, SHOOTING_VARIABLES + 14), np.dtype(np.float64)),
    "proxy_kkt_condition_float64": ((), np.dtype(np.float64)),
}
CSR_FIELDS = ("data_float64", "indices_int32", "indptr_int32", "shape_int64", "nnz_int64")


def exact_profile_array_schema(arrays: Mapping):
    if set(arrays) != set(PROFILE_ARRAY_SPECS):
        return False
    return all(
        np.asarray(arrays[name]).shape == shape
        and np.asarray(arrays[name]).dtype == dtype
        and np.isfinite(np.asarray(arrays[name])).all()
        for name, (shape, dtype) in PROFILE_ARRAY_SPECS.items()
    )


def array_hash(value) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        f"{array.dtype.str}|{array.shape}|".encode() + array.tobytes()
    ).hexdigest()


def pack_solver_seed(q, qd, controls):
    """Canonical GATO XU layout: each knot [q,qd,u], final knot [q,qd]."""
    q = np.asarray(q, dtype=np.float32)
    qd = np.asarray(qd, dtype=np.float32)
    controls = np.asarray(controls, dtype=np.float32)
    if q.shape != (KNOTS, 7) or qd.shape != (KNOTS, 7) or controls.shape != (INTERVALS, 7):
        raise ValueError("solver seed arrays have invalid shapes")
    seed = np.empty(KNOTS * (14 + 7) - 7, dtype=np.float32)
    for knot in range(KNOTS):
        offset = knot * 21
        seed[offset:offset + 7] = q[knot]
        seed[offset + 7:offset + 14] = qd[knot]
        if knot < INTERVALS:
            seed[offset + 14:offset + 21] = controls[knot]
    return seed


def unpack_solver_seed(seed):
    value = np.asarray(seed)
    if value.shape != (KNOTS * 21 - 7,) or value.dtype != np.float32:
        raise ValueError("solver seed must be one exact float32 XU row")
    q = np.empty((KNOTS, 7), np.float32)
    qd = np.empty((KNOTS, 7), np.float32)
    controls = np.empty((INTERVALS, 7), np.float32)
    for knot in range(KNOTS):
        offset = knot * 21
        q[knot] = value[offset:offset + 7]
        qd[knot] = value[offset + 7:offset + 14]
        if knot < INTERVALS:
            controls[knot] = value[offset + 14:offset + 21]
    return q, qd, controls


def unpack_acceleration(flat):
    value = np.asarray(flat, dtype=np.float64)
    if value.shape != (SHOOTING_VARIABLES,) or not np.isfinite(value).all():
        raise ValueError("shooting acceleration must be 665 finite float64 values")
    return value.reshape(INTERVALS, 7)


def integrate_acceleration(q0, qd0, acceleration):
    q0 = np.asarray(q0, dtype=np.float64)
    qd0 = np.asarray(qd0, dtype=np.float64)
    qdd = np.asarray(acceleration, dtype=np.float64)
    if q0.shape != (7,) or qd0.shape != (7,) or qdd.shape != (INTERVALS, 7):
        raise ValueError("shooting recurrence shapes invalid")
    q = np.empty((KNOTS, 7), dtype=np.float64)
    qd = np.empty((KNOTS, 7), dtype=np.float64)
    q[0], qd[0] = q0, qd0
    for knot in range(INTERVALS):
        q[knot + 1] = q[knot] + DT * qd[knot] + 0.5 * DT**2 * qdd[knot]
        qd[knot + 1] = qd[knot] + DT * qdd[knot]
    return q, qd


def recurrence_sensitivities():
    dq = np.zeros((KNOTS, 7, INTERVALS, 7), dtype=np.float64)
    dv = np.zeros_like(dq)
    eye = np.eye(7)
    for knot in range(INTERVALS):
        dq[knot + 1] = dq[knot] + DT * dv[knot]
        dv[knot + 1] = dv[knot]
        dq[knot + 1, :, knot, :] += 0.5 * DT**2 * eye
        dv[knot + 1, :, knot, :] += DT * eye
    return dq.reshape(KNOTS, 7, SHOOTING_VARIABLES), dv.reshape(
        KNOTS, 7, SHOOTING_VARIABLES
    )


def proxy_joint_path(q0, q_goal, tool_reference, kinematics: Callable):
    """Exactly eight unbounded DLS steps only at knots 1..94."""

    q0 = np.asarray(q0, dtype=np.float64)
    q_goal = np.asarray(q_goal, dtype=np.float64)
    target = np.asarray(tool_reference, dtype=np.float64)
    if q0.shape != (7,) or q_goal.shape != (7,) or target.shape != (KNOTS, 3):
        raise ValueError("proxy inputs have invalid shape")
    proxy = np.empty((KNOTS, 7), dtype=np.float64)
    q_before = np.empty((94, DLS_STEPS, 7), dtype=np.float64)
    position_history = np.empty((94, DLS_STEPS, 3), dtype=np.float64)
    residual_history = np.empty_like(position_history)
    jacobian_history = np.empty((94, DLS_STEPS, 3, 7), dtype=np.float64)
    dq_history = np.empty((94, DLS_STEPS, 7), dtype=np.float64)
    proxy[0] = q0
    for knot in range(1, KNOTS - 1):
        q = proxy[knot - 1].copy()
        for step in range(DLS_STEPS):
            position, jacobian = kinematics(q)
            residual = target[knot] - np.asarray(position, dtype=np.float64)
            J = np.asarray(jacobian, dtype=np.float64)
            dq_step = J.T @ np.linalg.solve(
                J @ J.T + DLS_DAMPING**2 * np.eye(3), residual
            )
            q_before[knot - 1, step] = q
            position_history[knot - 1, step] = position
            residual_history[knot - 1, step] = residual
            jacobian_history[knot - 1, step] = J
            dq_history[knot - 1, step] = dq_step
            q = q + dq_step
        proxy[knot] = q
    proxy[-1] = q_goal
    return {
        "proxy_q_before_float64": q_before,
        "proxy_position_float64": position_history,
        "proxy_residual_float64": residual_history,
        "proxy_jacobian_float64": jacobian_history,
        "proxy_dq_float64": dq_history,
        "proxy_q_float64": proxy,
    }


def proxy_kkt_matrix():
    dq, dv = recurrence_sensitivities()
    endpoint = np.vstack((dq[-1], dv[-1]))
    H = np.zeros((SHOOTING_VARIABLES, SHOOTING_VARIABLES))
    for knot in range(KNOTS):
        H += dq[knot].T @ dq[knot]
    H += LINEAR_PROXY_REGULARIZATION * np.eye(SHOOTING_VARIABLES)
    matrix = np.block(
        [[H, endpoint.T], [endpoint, np.zeros((14, 14), dtype=np.float64)]]
    )
    return matrix, np.asarray(np.linalg.cond(matrix), np.float64)


def linear_proxy_initial_acceleration(q0, q_goal, proxy):
    """One equality-constrained linear solve; no inverse or pseudoinverse."""

    q0 = np.asarray(q0, dtype=np.float64)
    q_goal = np.asarray(q_goal, dtype=np.float64)
    proxy = np.asarray(proxy, dtype=np.float64)
    if q0.shape != q_goal.shape or q0.shape != (7,) or proxy.shape != (KNOTS, 7):
        raise ValueError("linear proxy inputs invalid")
    dq, dv = recurrence_sensitivities()
    position_map = dq[:, :, :]
    endpoint = np.vstack((dq[-1], dv[-1]))
    base = np.repeat(q0[None, :], KNOTS, axis=0)
    rhs = np.zeros(SHOOTING_VARIABLES)
    for knot in range(KNOTS):
        J = position_map[knot]
        defect = proxy[knot] - base[knot]
        rhs += J.T @ defect
    target = np.r_[q_goal - q0, np.zeros(7)]
    kkt, _condition = proxy_kkt_matrix()
    kkt_rhs = np.r_[rhs, target]
    full_solution = np.linalg.solve(kkt, kkt_rhs)
    solve_residual = float(np.max(np.abs(kkt @ full_solution - kkt_rhs)))
    endpoint_residual = endpoint @ full_solution[:SHOOTING_VARIABLES] - target
    return {
        "initial_acceleration_float64": full_solution[:SHOOTING_VARIABLES].reshape(INTERVALS, 7),
        "proxy_kkt_rhs_float64": kkt_rhs,
        "proxy_kkt_solution_float64": full_solution,
        "proxy_kkt_endpoint_residual_float64": endpoint_residual,
        "proxy_kkt_solve_residual_float64": np.asarray(solve_residual, dtype=np.float64),
    }


def certify_proxy_construction(history: Mapping, q0, q_goal, tool_reference,
                               kinematics: Callable):
    regenerated = proxy_joint_path(q0, q_goal, tool_reference, kinematics)
    if set(history) != set(regenerated):
        return {"passes": False}
    gates = {
        name: np.array_equal(np.asarray(history[name]), value)
        for name, value in regenerated.items()
    }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def certify_proxy_kkt(retained: Mapping, q0, q_goal, proxy, global_arrays: Mapping):
    try:
        regenerated = linear_proxy_initial_acceleration(q0, q_goal, proxy)
    except (ValueError, np.linalg.LinAlgError):
        return {"passes": False}
    if set(retained) != set(regenerated) or set(global_arrays) != set(GLOBAL_ARRAY_SPECS):
        return {"passes": False}
    matrix, condition = proxy_kkt_matrix()
    exact = all(np.array_equal(np.asarray(retained[name]), value)
                for name, value in regenerated.items())
    residual = float(regenerated["proxy_kkt_solve_residual_float64"])
    endpoint = np.asarray(regenerated["proxy_kkt_endpoint_residual_float64"])
    condition_value = float(condition)
    gates = {
        "exact_regeneration": exact,
        "global_matrix": np.array_equal(global_arrays["proxy_kkt_matrix_float64"], matrix),
        "global_condition": np.array_equal(global_arrays["proxy_kkt_condition_float64"], condition),
        "finite_condition": np.isfinite(condition_value) and condition_value > 0,
        "solve_residual": residual <= 1e-8,
        "endpoint_residual": np.max(np.abs(endpoint), initial=0.0) <= 1e-9,
    }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def reconstruct_controls(q, qd, qdd, rnea: Callable):
    controls = np.stack(
        [rnea(q[k], qd[k], qdd[k]) for k in range(INTERVALS)]
    ).astype(np.float64)
    if controls.shape != (INTERVALS, 7) or not np.isfinite(controls).all():
        raise ValueError("RNEA controls invalid")
    return controls


def position_and_control_sensitivities(
    q, qd, qdd, kinematics: Callable, rnea_derivatives: Callable
):
    """Exact shooting first derivatives from Pin FK/J and RNEA derivatives."""

    dq, dv = recurrence_sensitivities()
    positions, dp = [], []
    for knot in range(KNOTS):
        position, jacobian = kinematics(q[knot])
        positions.append(np.asarray(position, dtype=np.float64))
        dp.append(np.asarray(jacobian, dtype=np.float64) @ dq[knot])
    du = np.empty((INTERVALS, 7, SHOOTING_VARIABLES), dtype=np.float64)
    for knot in range(INTERVALS):
        derivative_q, derivative_v, derivative_a_upper = rnea_derivatives(
            q[knot], qd[knot], qdd[knot]
        )
        derivative_a = symmetric_mass_from_pin_upper(derivative_a_upper)
        selector = np.zeros((7, SHOOTING_VARIABLES), dtype=np.float64)
        selector[:, knot * 7:(knot + 1) * 7] = np.eye(7)
        du[knot] = (
            np.asarray(derivative_q) @ dq[knot]
            + np.asarray(derivative_v) @ dv[knot]
            + np.asarray(derivative_a) @ selector
        )
    return np.stack(positions), np.stack(dp), du


def symmetric_mass_from_pin_upper(upper_triangle):
    """Reconstruct Pinocchio's symmetric mass matrix from its valid upper half."""
    value = np.asarray(upper_triangle, dtype=np.float64)
    if value.shape != (7, 7) or not np.isfinite(value).all():
        raise ValueError("Pin mass derivative must be finite shape (7,7)")
    upper = np.triu(value)
    return upper + np.triu(value, 1).T


def certify_rnea_acceleration_derivative(q, qd, qdd, rnea: Callable,
                                         rnea_derivatives: Callable):
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    qdd = np.asarray(qdd, dtype=np.float64)
    if q.shape != qd.shape or q.shape != qdd.shape or q.shape != (7,):
        return {"passes": False}
    _dq, _dv, upper = rnea_derivatives(q, qd, qdd)
    mass = symmetric_mass_from_pin_upper(upper)
    step = 1e-6
    fd = np.column_stack([
        (np.asarray(rnea(q, qd, qdd + step * np.eye(7)[column]))
         - np.asarray(rnea(q, qd, qdd - step * np.eye(7)[column]))) / (2 * step)
        for column in range(7)
    ])
    maximum = float(np.max(np.abs(mass - fd), initial=0.0))
    gates = {
        "finite": np.isfinite(mass).all() and np.isfinite(fd).all(),
        "symmetric": np.array_equal(mass, mass.T),
        "mass_matches_rnea_acceleration_fd": maximum <= 1e-7,
    }
    return {
        "mass_float64": mass, "mass_fd_float64": fd,
        "maxabs": maximum, "gates": gates, "passes": bool(all(gates.values())),
    }


def certify_full_rnea_derivatives(retained: Mapping, q, qd, qdd,
                                  rnea: Callable, rnea_derivatives: Callable):
    exact = {"dq_float64", "dv_float64", "mass_float64", "mass_fd_float64"}
    if set(retained) != exact:
        return {"passes": False}
    q = np.asarray(q, np.float64); qd = np.asarray(qd, np.float64); qdd = np.asarray(qdd, np.float64)
    dq, dv, _upper = rnea_derivatives(q, qd, qdd)
    mass = certify_rnea_acceleration_derivative(q, qd, qdd, rnea, rnea_derivatives)
    step = 1e-6
    fd_q = np.column_stack([(rnea(q + step*np.eye(7)[j], qd, qdd)
                             - rnea(q - step*np.eye(7)[j], qd, qdd))/(2*step) for j in range(7)])
    fd_v = np.column_stack([(rnea(q, qd + step*np.eye(7)[j], qdd)
                             - rnea(q, qd - step*np.eye(7)[j], qdd))/(2*step) for j in range(7)])
    gates = {
        "dq_retained": np.array_equal(np.asarray(retained["dq_float64"]), dq),
        "dv_retained": np.array_equal(np.asarray(retained["dv_float64"]), dv),
        "mass_retained": mass["passes"]
        and np.array_equal(np.asarray(retained["mass_float64"]), mass["mass_float64"])
        and np.array_equal(np.asarray(retained["mass_fd_float64"]), mass["mass_fd_float64"]),
        "dq_fd": np.max(np.abs(np.asarray(dq)-fd_q), initial=0.0) <= 1e-5,
        "dv_fd": np.max(np.abs(np.asarray(dv)-fd_v), initial=0.0) <= 1e-5,
    }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def certify_retained_mass_diagnostics(retained: Mapping, q, qd, qdd,
                                      rnea: Callable,
                                      rnea_derivatives: Callable):
    exact = {"rnea_mass_float64", "rnea_mass_fd_float64", "rnea_mass_maxabs_float64"}
    if set(retained) != exact:
        return {"passes": False}
    details = [
        certify_rnea_acceleration_derivative(q[k], qd[k], qdd[k], rnea, rnea_derivatives)
        for k in range(INTERVALS)
    ]
    if not all(row.get("passes") is True for row in details):
        return {"passes": False}
    mass = np.stack([row["mass_float64"] for row in details])
    fd = np.stack([row["mass_fd_float64"] for row in details])
    maximum = np.asarray(max(row["maxabs"] for row in details), dtype=np.float64)
    gates = {
        "mass_exact": np.array_equal(np.asarray(retained["rnea_mass_float64"]), mass),
        "fd_exact": np.array_equal(np.asarray(retained["rnea_mass_fd_float64"]), fd),
        "maximum_exact": np.array_equal(
            np.asarray(retained["rnea_mass_maxabs_float64"]), maximum
        ),
    }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def shooting_objective_gradient(
    qdd, controls, positions, tool_reference, effort,
    position_sensitivity, control_sensitivity,
):
    qdd = np.asarray(qdd, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    position_error = np.asarray(positions) - np.asarray(tool_reference)
    gradient = np.einsum(
        "ki,kij->j", position_error / PATH_SCALE_M**2,
        np.asarray(position_sensitivity),
    )
    gradient += (
        ACCELERATION_WEIGHT / ACCELERATION_SCALE_RAD_S2**2 * qdd.ravel()
    )
    difference = np.diff(qdd, axis=0) / ACCELERATION_SCALE_RAD_S2**2
    jerk_gradient = np.zeros_like(qdd)
    jerk_gradient[:-1] -= JERK_WEIGHT * difference
    jerk_gradient[1:] += JERK_WEIGHT * difference
    gradient += jerk_gradient.ravel()
    scaled_control_gradient = controls / np.asarray(effort)[None, :] ** 2
    gradient += CONTROL_WEIGHT * np.einsum(
        "ki,kij->j", scaled_control_gradient, np.asarray(control_sensitivity)
    )
    return gradient


def shooting_endpoint_jacobian():
    dq, dv = recurrence_sensitivities()
    return np.vstack((dq[-1], dv[-1]))


def shooting_inequality_jacobian(
    q, qd, controls, positions, pillar,
    position_sensitivity, control_sensitivity,
):
    dq, dv = recurrence_sensitivities()
    dp = np.asarray(position_sensitivity)
    du = np.asarray(control_sensitivity)
    delta = np.asarray(positions)[:, :2] - np.asarray(pillar)[None, :]
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance <= 0):
        raise ValueError("clearance derivative undefined at pillar center")
    clearance = np.einsum("ki,kij->kj", delta / distance[:, None], dp[:, :2, :])
    result = np.vstack(
        (
            dq.reshape(KNOTS * 7, SHOOTING_VARIABLES),
            -dq.reshape(KNOTS * 7, SHOOTING_VARIABLES),
            -dv.reshape(KNOTS * 7, SHOOTING_VARIABLES),
            dv.reshape(KNOTS * 7, SHOOTING_VARIABLES),
            -du.reshape(INTERVALS * 7, SHOOTING_VARIABLES),
            du.reshape(INTERVALS * 7, SHOOTING_VARIABLES),
            clearance,
        )
    )
    assert result.shape == (SHOOTING_INEQUALITIES, SHOOTING_VARIABLES)
    return result


def directional_derivative_gate(value, analytic, point, direction):
    point = np.asarray(point, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    step = 1e-6 / max(1.0, float(np.linalg.norm(direction)))
    finite_difference = (value(point + step * direction) - value(point - step * direction)) / (2 * step)
    predicted = analytic(point) @ direction
    difference = np.max(np.abs(np.asarray(finite_difference) - np.asarray(predicted)), initial=0.0)
    scale = max(
        np.max(np.abs(np.asarray(finite_difference)), initial=0.0),
        np.max(np.abs(np.asarray(predicted)), initial=0.0),
    )
    passes = difference <= 1e-7 if scale <= 1e-7 else difference / scale <= 1e-5
    return {
        "step": step, "absolute_error": float(difference),
        "scale": float(scale), "passes": bool(passes),
    }


def certify_sparse_derivative_diagnostic(dense, csr, retained: Mapping):
    """Bind deterministic CSR ordering and dense equivalence for audit probes."""
    required = {"data_float64", "indices_int32", "indptr_int32", "shape_int64"}
    if set(retained) != required:
        return {"passes": False}
    dense = np.asarray(dense, dtype=np.float64)
    data = np.asarray(retained["data_float64"])
    indices = np.asarray(retained["indices_int32"])
    indptr = np.asarray(retained["indptr_int32"])
    shape = np.asarray(retained["shape_int64"])
    try:
        reconstructed = np.asarray(csr.toarray(), dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return {"passes": False}
    sorted_rows = all(
        np.all(np.diff(indices[indptr[row]:indptr[row + 1]]) > 0)
        for row in range(dense.shape[0])
    )
    gates = {
        "exact_dtypes": data.dtype == np.float64 and indices.dtype == np.int32
        and indptr.dtype == np.int32 and shape.dtype == np.int64,
        "exact_shape": np.array_equal(shape, np.asarray(dense.shape, np.int64)),
        "csr_arrays": np.array_equal(data, csr.data)
        and np.array_equal(indices, csr.indices)
        and np.array_equal(indptr, csr.indptr),
        "canonical_sorted_indices": bool(sorted_rows),
        "sparse_dense_exact": np.array_equal(reconstructed, dense),
        "finite": np.isfinite(dense).all() and np.isfinite(data).all(),
    }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def certify_production_derivative_evidence(arrays: Mapping, evaluations):
    """Recompute all three frozen real-model probes, including full 4114 CSR."""
    fixed_names = set(DERIVATIVE_ARRAY_SPECS)
    dynamic_names = {
        f"derivative_probe{probe}_inequality_csr_{field}"
        for probe in range(3) for field in CSR_FIELDS
    }
    if set(arrays) != fixed_names | dynamic_names or len(evaluations) != 3:
        return {"passes": False}
    if not all(
        np.asarray(arrays[name]).shape == shape
        and np.asarray(arrays[name]).dtype == dtype
        and np.isfinite(np.asarray(arrays[name])).all()
        for name, (shape, dtype) in DERIVATIVE_ARRAY_SPECS.items()
    ):
        return {"passes": False}
    gates = {
        "indices": np.array_equal(
            arrays["derivative_probe_indices_int64"],
            np.asarray(DERIVATIVE_PROBE_INDICES, np.int64),
        ),
        "endpoint": np.array_equal(
            arrays["derivative_endpoint_jacobian_float64"],
            shooting_endpoint_jacobian(),
        ),
    }
    step = 1e-6
    for probe, evaluation in enumerate(evaluations):
        q = np.asarray(evaluation["q"], np.float64)
        qd = np.asarray(evaluation["qd"], np.float64)
        qdd = np.asarray(evaluation["qdd"], np.float64)
        rnea = evaluation["rnea"]
        rnea_derivatives = evaluation["rnea_derivatives"]
        derivative_q, derivative_v, derivative_a = rnea_derivatives(q, qd, qdd)
        rnea_detail = certify_full_rnea_derivatives({
            "dq_float64": arrays[f"derivative_rnea_dq_float64"][probe],
            "dv_float64": arrays[f"derivative_rnea_dv_float64"][probe],
            "mass_float64": arrays[f"derivative_rnea_mass_float64"][probe],
            "mass_fd_float64": arrays[f"derivative_rnea_mass_fd_float64"][probe],
        }, q, qd, qdd, rnea, rnea_derivatives)
        prefix = "derivative_"
        gates[f"probe_{probe}_state"] = all(
            np.array_equal(arrays[f"{prefix}probe_{name}_float64"][probe], value)
            for name, value in (("q", q), ("qd", qd), ("qdd", qdd))
        )
        gates[f"probe_{probe}_full_rnea_derivatives"] = rnea_detail["passes"]
        point = arrays[f"{prefix}objective_point_float64"][probe]
        direction = arrays[f"{prefix}objective_direction_float64"][probe]
        gradient = np.asarray(evaluation["objective_gradient"](point))
        inequality = np.asarray(evaluation["inequality_jacobian"](point))
        gates[f"probe_{probe}_objective"] = (
            np.array_equal(arrays[f"{prefix}objective_gradient_float64"][probe], gradient)
            and directional_derivative_gate(
                evaluation["objective"], evaluation["objective_gradient"], point, direction
            )["passes"]
        )
        gates[f"probe_{probe}_inequality_direction"] = directional_derivative_gate(
            evaluation["inequality"], evaluation["inequality_jacobian"], point, direction
        )["passes"]
        retained_dense = arrays["derivative_probe0_inequality_dense_float64"] if probe == 0 else None
        base = f"derivative_probe{probe}_inequality_csr_"
        nnz_value = np.asarray(arrays[base + "nnz_int64"])
        data = np.asarray(arrays[base + "data_float64"])
        indices = np.asarray(arrays[base + "indices_int32"])
        indptr = np.asarray(arrays[base + "indptr_int32"])
        shape = np.asarray(arrays[base + "shape_int64"])
        nnz = int(nnz_value.item()) if nnz_value.shape == () else -1
        try:
            from scipy.sparse import csr_matrix
            csr = csr_matrix((data, indices, indptr), shape=tuple(shape))
            csr_ok = (
                shape.dtype == np.int64 and np.array_equal(shape, inequality.shape)
                and data.dtype == np.float64 and indices.dtype == np.int32
                and indptr.dtype == np.int32 and indptr.shape == (SHOOTING_INEQUALITIES + 1,)
                and nnz == len(data) == len(indices) == int(indptr[-1])
                and np.array_equal(csr.toarray(), inequality)
                and all(np.all(np.diff(indices[indptr[row]:indptr[row + 1]]) > 0)
                        for row in range(SHOOTING_INEQUALITIES))
            )
        except (ValueError, IndexError):
            csr_ok = False
        gates[f"probe_{probe}_full_dense_csr"] = bool(
            (probe != 0 or np.array_equal(retained_dense, inequality)) and csr_ok
        )
    return {"gates": gates, "passes": bool(all(gates.values()))}


def shooting_objective(q, qdd, controls, positions, tool_reference, effort):
    path = (np.asarray(positions) - np.asarray(tool_reference)) / PATH_SCALE_M
    scaled_acceleration = np.asarray(qdd) / ACCELERATION_SCALE_RAD_S2
    difference = np.diff(np.asarray(qdd), axis=0) / ACCELERATION_SCALE_RAD_S2
    scaled_control = np.asarray(controls) / np.asarray(effort)[None, :]
    return float(
        0.5 * np.sum(path * path)
        + 0.5 * ACCELERATION_WEIGHT * np.sum(scaled_acceleration**2)
        + 0.5 * JERK_WEIGHT * np.sum(difference**2)
        + 0.5 * CONTROL_WEIGHT * np.sum(scaled_control**2)
    )


def shooting_endpoint(q, qd, q_goal):
    return np.r_[np.asarray(q)[-1] - np.asarray(q_goal), np.asarray(qd)[-1]]


def shooting_inequalities(q, qd, controls, positions, lower, upper, velocity, effort, pillar):
    q = np.asarray(q); qd = np.asarray(qd); controls = np.asarray(controls)
    lower = np.asarray(lower); upper = np.asarray(upper)
    velocity = np.asarray(velocity); effort = np.asarray(effort)
    distance = np.linalg.norm(np.asarray(positions)[:, :2] - np.asarray(pillar)[None, :], axis=1)
    values = np.concatenate(
        (
            (q - (lower + CONSTRUCTION_Q_RESERVE_RAD)).ravel(),
            ((upper - CONSTRUCTION_Q_RESERVE_RAD) - q).ravel(),
            (CONSTRUCTION_LIMIT_RATIO * velocity - qd).ravel(),
            (CONSTRUCTION_LIMIT_RATIO * velocity + qd).ravel(),
            (CONSTRUCTION_LIMIT_RATIO * effort - controls).ravel(),
            (CONSTRUCTION_LIMIT_RATIO * effort + controls).ravel(),
            distance - (PHYSICAL_RADIUS_M + CONSTRUCTION_CLEARANCE_M),
        )
    )
    assert values.shape == (SHOOTING_INEQUALITIES,)
    return values


@dataclass(frozen=True)
class ShootingResult:
    identity: tuple
    acceleration_float64: np.ndarray
    q_float64: np.ndarray
    qd_float64: np.ndarray
    controls_float64: np.ndarray
    positions_float64: np.ndarray
    objective: float
    endpoint_residual_float64: np.ndarray
    inequalities_float64: np.ndarray
    optimizer_success: bool
    optimizer_status: int
    optimizer_iterations: int
    optimizer_wall_s: float


def certify_shooting_result(
    result: ShootingResult, q0, q_goal, tool_reference,
    lower, upper, velocity, effort, pillar,
    kinematics: Callable, rnea: Callable, aba: Callable,
):
    """Independently reconstruct every retained shooting field and gate."""

    q, qd = integrate_acceleration(q0, np.zeros(7), result.acceleration_float64)
    controls = reconstruct_controls(q, qd, result.acceleration_float64, rnea)
    positions = np.stack([kinematics(value)[0] for value in q]).astype(np.float64)
    aba_acceleration = np.stack(
        [aba(q[k], qd[k], controls[k]) for k in range(INTERVALS)]
    ).astype(np.float64)
    inequalities = shooting_inequalities(
        q, qd, controls, positions, lower, upper, velocity, effort, pillar
    )
    objective = shooting_objective(
        q, result.acceleration_float64, controls, positions,
        tool_reference, effort,
    )
    path_error = positions - np.asarray(tool_reference)
    endpoint = shooting_endpoint(q, qd, q_goal)
    recurrence_exact = bool(
        np.max(np.abs(q - result.q_float64), initial=0.0) <= RECURRENCE_TOLERANCE
        and np.max(np.abs(qd - result.qd_float64), initial=0.0) <= RECURRENCE_TOLERANCE
    )
    gates = {
        "identity": tuple(result.identity) in EXPECTED_LEDGER,
        "one_attempt_success": result.optimizer_success is True,
        "iteration_cap": 0 <= result.optimizer_iterations <= BACKEND_OPTIONS["maxiter"],
        "wall_cap": 0.0 <= result.optimizer_wall_s <= PROFILE_WALL_LIMIT_S,
        "finite": all(
            np.isfinite(value).all()
            for value in (
                result.acceleration_float64, result.q_float64,
                result.qd_float64, result.controls_float64,
                result.positions_float64, result.endpoint_residual_float64,
                result.inequalities_float64,
            )
        ),
        "recurrence": recurrence_exact,
        "controls_reconstructed": np.array_equal(controls, result.controls_float64),
        "positions_reconstructed": np.array_equal(positions, result.positions_float64),
        "aba_rnea": np.max(
            np.abs(aba_acceleration - result.acceleration_float64), initial=0.0
        ) <= ABA_RNEA_TOLERANCE,
        "endpoint": np.max(np.abs(endpoint), initial=0.0) <= ENDPOINT_TOLERANCE,
        "endpoint_retained": np.array_equal(endpoint, result.endpoint_residual_float64),
        "inequalities_reconstructed": np.array_equal(
            inequalities, result.inequalities_float64
        ),
        "hard_inequalities": np.min(inequalities, initial=0.0) >= 0.0,
        "objective_reconstructed": objective == result.objective,
        "path_rms": float(np.sqrt(np.mean(np.sum(path_error * path_error, axis=1)))) <= PATH_RMS_TOLERANCE_M,
        "path_max": np.max(np.linalg.norm(path_error, axis=1), initial=0.0) <= PATH_MAX_TOLERANCE_M,
    }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def _open_turn(path, center):
    delta = np.asarray(path, dtype=np.float64)[:, :2] - np.asarray(center)[None, :]
    angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    return float(angles[-1] - angles[0])


def _dense_reference(reference_path):
    reference = np.asarray(reference_path, dtype=np.float64)
    if reference.shape != (KNOTS, 3):
        raise ValueError("solver reference path shape invalid")
    fraction = np.arange(DENSE_SAMPLES, dtype=np.float64) / DENSE_SUBSTEPS
    base = np.arange(KNOTS, dtype=np.float64)
    return np.column_stack([np.interp(fraction, base, reference[:, axis]) for axis in range(3)])


def certify_independent_replay(pin: Mapping, cuda: Mapping, reference_path, route: str,
                               canonical_reference, tool_velocity: Callable):
    """Pure P2 seed certificate; every reported scalar is recomputed."""

    required = {
        "state", "control", "tool", "joint_lower",
        "joint_upper", "velocity_limit", "effort_limit", "pillar_xy",
        "physical_radius_m", "x0", "reference_float32",
    }
    if set(pin) != required or set(cuda) != required or route not in ("short", "long"):
        return {"passes": False}
    ps = np.asarray(pin["state"]); cs = np.asarray(cuda["state"])
    pt = np.asarray(pin["tool"]); ct = np.asarray(cuda["tool"])
    pu = np.asarray(pin["control"]); cu = np.asarray(cuda["control"])
    reference = np.asarray(reference_path)
    canonical = np.asarray(canonical_reference)
    finite = all(np.isfinite(value).all() for value in (ps, cs, pt, ct, pu, cu))
    shape = bool(
        ps.shape == cs.shape == (DENSE_SAMPLES, 14)
        and pt.shape == ct.shape == (DENSE_SAMPLES, 3)
        and pu.shape == cu.shape == (INTERVALS, 7)
        and reference.shape == (KNOTS, 3) and canonical.shape == (KNOTS * 10,)
        and canonical.dtype == np.float32 and np.isfinite(canonical).all()
    )
    if not finite or not shape:
        return {"finite": finite, "shape": shape, "passes": False}
    common_inputs = all(
        np.array_equal(np.asarray(pin[name]), np.asarray(cuda[name]))
        for name in (
            "joint_lower", "joint_upper", "velocity_limit", "effort_limit",
            "pillar_xy", "x0", "reference_float32",
        )
    ) and float(pin["physical_radius_m"]) == float(cuda["physical_radius_m"])
    lower = np.asarray(pin["joint_lower"]); upper = np.asarray(pin["joint_upper"])
    velocity = np.asarray(pin["velocity_limit"]); effort = np.asarray(pin["effort_limit"])
    pillar = np.asarray(pin["pillar_xy"]); radius = float(pin["physical_radius_m"])
    pin_clearance = np.linalg.norm(pt[:, :2] - pillar[None, :], axis=1) - radius
    cuda_clearance = np.linalg.norm(ct[:, :2] - pillar[None, :], axis=1) - radius
    reference_turn = _open_turn(reference, pillar)
    if reference_turn == 0:
        return {"passes": False}
    # The actual path must have the same sign as its route reference and a
    # material open turn.  The explicit expression avoids trusted labels.
    route_sign = int(np.sign(reference_turn))
    dense_reference = _dense_reference(reference)
    pin_circular_error = np.linalg.norm(pt - dense_reference, axis=1)
    cuda_circular_error = np.linalg.norm(ct - dense_reference, axis=1)
    try:
        pin_speed = float(np.linalg.norm(tool_velocity("pin", ps[-1, :7], ps[-1, 7:])))
        cuda_speed = float(np.linalg.norm(tool_velocity("cuda", cs[-1, :7], cs[-1, 7:])))
    except (TypeError, ValueError):
        return {"passes": False}
    turn_threshold = 2.4 if route == "short" else 3.0
    gates = {
        "finite": finite,
        "shape": shape,
        "common_initial_state": np.array_equal(ps[0], cs[0]),
        "common_controls": np.array_equal(pu, cu),
        "common_inputs": common_inputs,
        "canonical_reference": np.array_equal(
            np.asarray(pin["reference_float32"]), canonical
        ) and np.array_equal(np.asarray(cuda["reference_float32"]), canonical),
        "exact_initial_state": np.array_equal(ps[0], np.asarray(pin["x0"]))
        and np.array_equal(cs[0], np.asarray(cuda["x0"])),
        "pin_q_limits": bool(np.all(ps[:, :7] >= lower) and np.all(ps[:, :7] <= upper)),
        "cuda_q_limits": bool(np.all(cs[:, :7] >= lower) and np.all(cs[:, :7] <= upper)),
        "pin_v_limits": bool(np.all(np.abs(ps[:, 7:]) <= velocity)),
        "cuda_v_limits": bool(np.all(np.abs(cs[:, 7:]) <= velocity)),
        "pin_u_limits": bool(np.all(np.abs(pu) <= effort)),
        "cuda_u_limits": bool(np.all(np.abs(cu) <= effort)),
        "pin_clearance": np.min(pin_clearance) >= 0.005,
        "cuda_clearance": np.min(cuda_clearance) >= 0.005,
        "model_tool_agreement": np.max(np.linalg.norm(pt - ct, axis=1)) <= MODEL_TOOL_TOLERANCE_M,
        "pin_terminal": np.linalg.norm(pt[-1] - reference[-1]) <= TERMINAL_PIN_TOLERANCE_M,
        "cuda_terminal": np.linalg.norm(ct[-1] - reference[-1]) <= TERMINAL_CUDA_TOLERANCE_M,
        "pin_speed_jqd": pin_speed <= FINAL_TOOL_SPEED_TOLERANCE_MPS,
        "cuda_speed_jqd": cuda_speed <= FINAL_TOOL_SPEED_TOLERANCE_MPS,
        "pin_circular_rms": float(np.sqrt(np.mean(pin_circular_error**2))) <= PATH_RMS_TOLERANCE_M,
        "cuda_circular_rms": float(np.sqrt(np.mean(cuda_circular_error**2))) <= PATH_RMS_TOLERANCE_M,
        "pin_circular_max": np.max(pin_circular_error, initial=0.0) <= PATH_MAX_TOLERANCE_M,
        "cuda_circular_max": np.max(cuda_circular_error, initial=0.0) <= PATH_MAX_TOLERANCE_M,
        "pin_route_turn": abs(_open_turn(pt, pillar)) >= turn_threshold
        and int(np.sign(_open_turn(pt, pillar))) == route_sign,
        "cuda_route_turn": abs(_open_turn(ct, pillar)) >= turn_threshold
        and int(np.sign(_open_turn(ct, pillar))) == route_sign,
    }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def _closed_winding(left, right, center):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("topology paths must have one common shape")
    closed = np.vstack((left, right[::-1], left[:1]))
    delta = closed[:, :2] - np.asarray(center)[None, :]
    angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    winding = float((angles[-1] - angles[0]) / (2 * np.pi))
    nearest = int(np.rint(winding))
    return winding, nearest, abs(winding - nearest)


def certify_portfolio_topology(rows):
    """Require every one of 16 lanes and every pair before any deduplication."""

    if len(rows) != 16 or [row.get("lane") for row in rows] != list(range(16)):
        return {"passes": False}
    expected_routes = ["short"] * 8 + ["long"] * 8
    if [row.get("route") for row in rows] != expected_routes:
        return {"passes": False}
    pair_gates = []
    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            center = np.asarray(left["pillar_xy"])
            common = np.array_equal(center, np.asarray(right["pillar_xy"]))
            try:
                cuda = _closed_winding(left["cuda_tool"], right["cuda_tool"], center)
                pin = _closed_winding(left["pin_tool"], right["pin_tool"], center)
            except ValueError:
                pair_gates.append(False)
                continue
            expected = 0 if left["route"] == right["route"] else 1
            pair_gates.append(bool(
                common
                and abs(cuda[1]) == expected and abs(pin[1]) == expected
                and cuda[1] == pin[1]
                and cuda[2] <= CLOSED_WINDING_RESIDUAL
                and pin[2] <= CLOSED_WINDING_RESIDUAL
            ))
    return {
        "all_16_rows": True,
        "all_120_pairs": len(pair_gates) == 120,
        "all_pair_classes_pass": bool(all(pair_gates)),
        "passes": bool(len(pair_gates) == 120 and all(pair_gates)),
    }


def certify_b1_b16_binding(b1: Mapping, b16: Mapping):
    required_b1 = {"x0_float32", "reference_float32", "seed_xu_float32"}
    required_b16 = required_b1
    if set(b1) != required_b1 or set(b16) != required_b16:
        return {"passes": False}
    x1 = np.asarray(b1["x0_float32"]); r1 = np.asarray(b1["reference_float32"])
    z1 = np.asarray(b1["seed_xu_float32"])
    x16 = np.asarray(b16["x0_float32"]); r16 = np.asarray(b16["reference_float32"])
    z16 = np.asarray(b16["seed_xu_float32"])
    gates = {
        "b1_shapes": x1.shape == (1, 14) and r1.shape == (1, 960) and z1.shape == (1, 2009),
        "b16_shapes": x16.shape == (16, 14) and r16.shape == (16, 960) and z16.shape == (16, 2009),
        "exact_float32": all(value.dtype == np.float32 for value in (x1, r1, z1, x16, r16, z16)),
        "all_finite": all(np.isfinite(value).all() for value in (x1, r1, z1, x16, r16, z16)),
        "common_x0": x16.shape == (16, 14) and np.all(x16 == x16[:1]),
        "common_reference": r16.shape == (16, 960) and np.all(r16 == r16[:1]),
        "b1_equals_lane0_x0": x16.shape == (16, 14) and np.array_equal(x1[0], x16[0]),
        "b1_equals_lane0_reference": r16.shape == (16, 960) and np.array_equal(r1[0], r16[0]),
        "b1_equals_lane0_seed": z16.shape == (16, 2009) and np.array_equal(z1[0], z16[0]),
        "all_lanes_unique": z16.shape == (16, 2009)
        and len({np.ascontiguousarray(row).tobytes() for row in z16}) == 16,
        "all_seed_roundtrips": z16.shape == (16, 2009) and all(
            np.array_equal(pack_solver_seed(*unpack_solver_seed(row)), row)
            for row in z16
        ),
    }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def aggregate_task_profiles(rows, topology, reversal):
    identities = [(row.get("route"), row.get("profile")) for row in rows]
    expected = [(route, profile) for route in ("short", "long") for profile in range(8)]
    gates = {
        "exact_ordered_16": identities == expected,
        "all_attempted_once": all(
            row.get("attempt_count") == 1 and row.get("retained") is True
            for row in rows
        ),
        "all_profile_certificates": all(row.get("passes") is True for row in rows),
        "topology": topology.get("passes") is True,
        "both_model_reversal": reversal.get("passes") is True,
        "no_filter_retry_replacement": all(
            row.get("filtered") is False and row.get("replacement") is None
            for row in rows
        ),
    }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def aggregate_campaign(tasks):
    gates = {
        "exact_12_tasks": [tuple(row.get("identity", ())) for row in tasks]
        == list(TASK_IDENTITIES),
        "all_tasks_pass": all(row.get("passes") is True for row in tasks),
        "all_192_profiles_retained": sum(row.get("profile_count", 0) for row in tasks) == 192,
    }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def path_length(path):
    value = np.asarray(path, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError("tool path must be finite with shape (samples,3)")
    return float(np.sum(np.linalg.norm(np.diff(value, axis=0), axis=1)))


def _joint_barrier(value, lower, upper):
    left = np.maximum(np.asarray(value) - np.asarray(lower), 1e-10)
    right = np.maximum(np.asarray(upper) - np.asarray(value), 1e-10)
    return -np.log(left) - np.log(right)


def reconstruct_portfolio_cost(state, control, tool, reference, lower, upper,
                               velocity, effort, *, include_toll):
    state = np.asarray(state, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    tool = np.asarray(tool, dtype=np.float64)
    reference = np.asarray(reference)
    if (state.shape != (KNOTS, 14) or control.shape != (INTERVALS, 7)
            or tool.shape != (KNOTS, 3)
            or reference.shape != (KNOTS * 10,) or reference.dtype != np.float32):
        raise ValueError("external objective arrays have invalid shape/dtype")
    ref = reference.reshape(KNOTS, 10)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    effort = np.asarray(effort, dtype=np.float64)
    if any(value.shape != (7,) for value in (lower, upper, velocity, effort)):
        raise ValueError("external objective limits invalid")
    total = 0.0
    toll_contribution = 0.0
    residuals = np.zeros(KNOTS, dtype=np.float64)
    for knot in range(KNOTS):
        row = validate_reference_row(ref[knot])
        error = tool[knot] - row[:3]
        total += 0.5 * (N_COST if knot == KNOTS - 1 else Q_COST) * float(error @ error)
        delta = tool[knot, :2] - row[3:5]
        keepout = float(row[5] + row[9])
        cylinder = max(1.0 - float(delta @ delta) / keepout**2, 0.0)
        total += 0.5 * CYLINDER_WEIGHT * cylinder**2
        if knot < KNOTS - 1:
            residuals[knot] = smoothstep_toll_residual_gradient(tool[knot, :2], row)[0]
            toll = 0.5 * TOLL_WEIGHT * residuals[knot] ** 2
            if include_toll:
                total += toll
            toll_contribution += toll
        total += 0.5 * QD_COST * float(state[knot, 7:] @ state[knot, 7:])
        total += Q_LIMIT_COST * float(np.sum(_joint_barrier(state[knot, :7], lower, upper)))
        total += VELOCITY_LIMIT_COST * float(np.sum(_joint_barrier(state[knot, 7:], -velocity, velocity)))
        if knot < INTERVALS:
            total += 0.5 * U_COST * float(control[knot] @ control[knot])
            total += CONTROL_LIMIT_COST * float(np.sum(_joint_barrier(control[knot], -effort, effort)))
    return {"cost": float(total), "toll_contribution": float(toll_contribution),
            "toll_residual_float64": residuals}


def certify_seed_reversal(short: Mapping, long: Mapping):
    """Reconstruct both-model base/full costs from exact retained trajectories."""

    exact = {"pin_path", "cuda_path", "pin_state", "cuda_state", "controls",
             "reference_float32", "joint_lower", "joint_upper",
             "velocity_limit", "effort_limit", "pin_base_cost", "cuda_base_cost",
             "pin_full_cost", "cuda_full_cost", "pin_toll_residual",
             "cuda_toll_residual"}
    if set(short) != exact or set(long) != exact:
        return {"passes": False}
    values = [short, long]
    arrays_ok = all(
        np.asarray(row[f"{model}_path"]).shape == (DENSE_SAMPLES, 3)
        and np.asarray(row[f"{model}_state"]).shape == (DENSE_SAMPLES, 14)
        and np.asarray(row["controls"]).shape == (INTERVALS, 7)
        and np.asarray(row["reference_float32"]).shape == (KNOTS * 10,)
        and np.asarray(row["reference_float32"]).dtype == np.float32
        and all(np.isfinite(np.asarray(value)).all() for value in row.values())
        for row in values for model in ("pin", "cuda")
    )
    if not arrays_ok:
        return {"arrays": arrays_ok, "passes": False}
    gates = {"arrays": True}
    gates["paired_canonical_problem"] = all(
        np.array_equal(np.asarray(short[name]), np.asarray(long[name]))
        for name in ("reference_float32", "joint_lower", "joint_upper",
                     "velocity_limit", "effort_limit")
    )
    for model in ("pin", "cuda"):
        short_length = path_length(short[f"{model}_path"])
        long_length = path_length(long[f"{model}_path"])
        def costs(row, include_toll):
            return reconstruct_portfolio_cost(
                np.asarray(row[f"{model}_state"])[::DENSE_SUBSTEPS],
                row["controls"], np.asarray(row[f"{model}_path"])[::DENSE_SUBSTEPS],
                row["reference_float32"], row["joint_lower"], row["joint_upper"],
                row["velocity_limit"], row["effort_limit"], include_toll=include_toll,
            )
        short_base_detail = costs(short, False); short_full_detail = costs(short, True)
        long_base_detail = costs(long, False); long_full_detail = costs(long, True)
        short_base = short_base_detail["cost"]
        long_base = long_base_detail["cost"]
        short_full = short_full_detail["cost"]
        long_full = long_full_detail["cost"]
        gates[f"{model}_shorter_by_5mm"] = long_length - short_length >= 0.005
        gates[f"{model}_base_advantage"] = long_base - short_base >= max(
            0.01, 0.02 * abs(long_base)
        )
        gates[f"{model}_full_short_worse_absolute"] = short_full - long_full >= 0.01
        gates[f"{model}_full_short_worse_relative"] = (
            short_full >= 1.05 * long_full
        )
        short_toll = short_full_detail["toll_residual_float64"]
        long_toll = long_full_detail["toll_residual_float64"]
        gates[f"{model}_retained_base_costs"] = (
            float(short[f"{model}_base_cost"]) == short_base
            and float(long[f"{model}_base_cost"]) == long_base
        )
        gates[f"{model}_retained_full_costs"] = (
            float(short[f"{model}_full_cost"]) == short_full
            and float(long[f"{model}_full_cost"]) == long_full
        )
        gates[f"{model}_retained_toll_residuals"] = (
            np.array_equal(np.asarray(short[f"{model}_toll_residual"]), short_toll)
            and np.array_equal(np.asarray(long[f"{model}_toll_residual"]), long_toll)
        )
        gates[f"{model}_short_saturated_at_least_8"] = int(
            np.count_nonzero(short_toll[1:-1] >= 0.95)
        ) >= 8
        gates[f"{model}_short_toll_material"] = (
            short_full_detail["toll_contribution"] >= 0.01
        )
        gates[f"{model}_long_toll_at_most_one_percent"] = (
            long_full_detail["toll_contribution"]
            <= 0.01 * short_full_detail["toll_contribution"]
        )
    gates["pin_cuda_reversal_agreement"] = all(
        gates[f"{model}_full_short_worse_absolute"]
        and gates[f"{model}_full_short_worse_relative"]
        for model in ("pin", "cuda")
    )
    return {"gates": gates, "passes": bool(all(gates.values()))}


def watchdog(elapsed_s: float, completed: int):
    if completed < 0 or completed > len(EXPECTED_LEDGER) or elapsed_s < 0:
        raise ValueError("watchdog inputs invalid")
    projected = np.inf if completed == 0 else elapsed_s / completed * len(EXPECTED_LEDGER)
    rejected = elapsed_s >= CAMPAIGN_WALL_LIMIT_S or (
        completed > 0 and projected > CAMPAIGN_WALL_LIMIT_S
    )
    return {
        "elapsed_s": float(elapsed_s), "completed": completed,
        "projected_s": float(projected), "threshold_s": CAMPAIGN_WALL_LIMIT_S,
        "runtime_watchdog_rejected": bool(rejected),
        "permanent_rejection": bool(rejected),
        "oracle_evidence": False, "benchmark_evidence": False,
    }


def backend_declaration():
    return {
        "backend": BACKEND,
        "options": dict(BACKEND_OPTIONS),
        "hess_argument": BACKEND_HESSIAN,
        "attempts_per_profile": 1,
        "profile_wall_limit_s": PROFILE_WALL_LIMIT_S,
        "campaign_wall_limit_s": CAMPAIGN_WALL_LIMIT_S,
        "exact_first_derivatives": True,
        "optimizer_hessian": "BFGS_not_evidence",
        "fallback": None,
        "threads": {
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        },
    }


def execute_constructor(*args, authorization=None, **kwargs):
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("circular portfolio P1 execution is blocked")
    return run_production_constructor(
        *args, authorization=authorization, **kwargs
    )


def run_production_constructor(
    identity, q0, q_goal, tool_reference, lower, upper, velocity, effort,
    pillar, kinematics, rnea, rnea_derivatives, aba, *, campaign_deadline,
    monotonic=time.monotonic, authorization=None,
):  # pragma: no cover - separately authorized Pin/SciPy boundary
    """One exact trust-constr attempt; no fallback, retry, or replacement."""
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("circular portfolio P1 execution is blocked")
    from scipy.optimize import BFGS, NonlinearConstraint, minimize
    from scipy.sparse import csr_matrix
    started=monotonic(); deadline=min(float(campaign_deadline),started+PROFILE_WALL_LIMIT_S)
    def guard():
        if monotonic() >= deadline:
            raise TimeoutError("circular portfolio profile wall limit")
    def guarded_kinematics(q):
        guard(); value = kinematics(q); guard(); return value
    def guarded_rnea(q, qd, qdd):
        guard(); value = rnea(q, qd, qdd); guard(); return value
    def guarded_rnea_derivatives(q, qd, qdd):
        guard(); value = rnea_derivatives(q, qd, qdd); guard(); return value
    def guarded_aba(q, qd, u):
        guard(); value = aba(q, qd, u); guard(); return value
    q0=np.asarray(q0,np.float64); q_goal=np.asarray(q_goal,np.float64)
    tool_reference=np.asarray(tool_reference,np.float64)
    guard()
    proxy=proxy_joint_path(q0,q_goal,tool_reference,guarded_kinematics)
    guard()
    initial=linear_proxy_initial_acceleration(q0,q_goal,proxy["proxy_q_float64"])
    guard()
    endpoint_jac=csr_matrix(shooting_endpoint_jacobian())
    def evaluate(flat):
        guard()
        qdd=unpack_acceleration(flat); q,qd=integrate_acceleration(q0,np.zeros(7),qdd)
        controls=reconstruct_controls(q,qd,qdd,guarded_rnea)
        positions,dp,du=position_and_control_sensitivities(
            q,qd,qdd,guarded_kinematics,guarded_rnea_derivatives
        )
        guard()
        return qdd,q,qd,controls,positions,dp,du
    def objective(flat):
        guard()
        qdd,q,_qd,u,p,_dp,_du=evaluate(flat)
        value=shooting_objective(q,qdd,u,p,tool_reference,effort); guard(); return value
    def gradient(flat):
        guard()
        qdd,_q,_qd,u,p,dp,du=evaluate(flat)
        value=shooting_objective_gradient(qdd,u,p,tool_reference,effort,dp,du); guard(); return value
    def equality(flat):
        guard()
        qdd=unpack_acceleration(flat); q,qd=integrate_acceleration(q0,np.zeros(7),qdd)
        value=shooting_endpoint(q,qd,q_goal); guard(); return value
    def inequality(flat):
        guard()
        _qdd,q,qd,u,p,_dp,_du=evaluate(flat)
        value=shooting_inequalities(q,qd,u,p,lower,upper,velocity,effort,pillar); guard(); return value
    def inequality_jac(flat):
        guard()
        _qdd,q,qd,u,p,dp,du=evaluate(flat)
        value=csr_matrix(shooting_inequality_jacobian(q,qd,u,p,pillar,dp,du)); guard(); return value
    def callback(_x,_state=None):
        guard()
        return False
    guard()
    result=minimize(
        objective,initial["initial_acceleration_float64"].ravel(),method="trust-constr",
        jac=gradient,hess=BFGS(),constraints=(
            NonlinearConstraint(equality,0.,0.,jac=lambda _z:endpoint_jac),
            NonlinearConstraint(inequality,0.,np.inf,jac=inequality_jac),
        ),callback=callback,options=dict(BACKEND_OPTIONS),
    )
    guard(); qdd,q,qd,u,p,_dp,_du=evaluate(result.x); guard()
    endpoint_value=equality(result.x); inequality_value=inequality(result.x)
    finished=monotonic(); guard()
    shooting=ShootingResult(tuple(identity),qdd,q,qd,u,p,float(result.fun),endpoint_value,
                            inequality_value,bool(result.success),int(result.status),
                            int(result.niter),float(finished-started))
    certificate=certify_shooting_result(shooting,q0,q_goal,tool_reference,lower,upper,
                                        velocity,effort,pillar,guarded_kinematics,
                                        guarded_rnea,guarded_aba)
    finished=monotonic(); guard()
    certificate["gates"]["wall_cap"] = 0.0 <= finished-started <= PROFILE_WALL_LIMIT_S
    certificate["passes"] = bool(all(certificate["gates"].values()))
    retained={**proxy,**initial,"acceleration_float64":qdd,"q_float64":q,
              "qd_float64":qd,"controls_float64":u,
              "applied_controls_float32":u.astype(np.float32),"positions_float64":p,
              "endpoint_residual_float64":endpoint_value,
              "inequalities_float64":inequality_value,
              "shooting_objective_float64":np.asarray(result.fun,np.float64),
              "optimizer_success_bool":np.asarray(result.success,np.bool_),
              "optimizer_status_int64":np.asarray(result.status,np.int64),
              "optimizer_iterations_int64":np.asarray(result.niter,np.int64),
              "optimizer_wall_float64":np.asarray(finished-started,np.float64)}
    return {"identity":list(identity),"retained":retained,
            "certificate":certificate,"passes":certificate["passes"]}
