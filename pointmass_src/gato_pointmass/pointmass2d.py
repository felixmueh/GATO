"""Frozen analytic point-mass task, dynamics, objective, and cheap seeds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np


BUDGET = 16
DEVELOPMENT_TASK_SEEDS = (7000, 7001, 7002)
DEVELOPMENT_METHOD_SEEDS = (8000, 8001, 8002, 8003, 8004)
HELDOUT_TASK_SEEDS = (7100, 7101, 7102, 7103, 7104)
HELDOUT_METHOD_SEEDS = tuple(range(8100, 8120))
CENTRAL_HELDOUT_TASK_SEED = 7102
LOW_FREQUENCY_INDICES = (1, 2, 3)


@dataclass(frozen=True)
class PointMassProtocol:
    knots: int = 32
    dt: float = 0.05
    budget: int = BUDGET
    position_limit: float = 0.60
    velocity_limit: float = 2.0
    acceleration_limit: float = 4.0
    start_goal_half_distance: float = 0.35
    clearance_margin: float = 0.02
    disk_offset_range: tuple[float, float] = (0.025, 0.045)
    disk_radius_range: tuple[float, float] = (0.09, 0.11)
    running_position_weight: float = 0.1
    running_velocity_weight: float = 0.05
    running_control_weight: float = 0.01
    running_obstacle_weight: float = 100.0
    terminal_position_weight: float = 200.0
    terminal_velocity_weight: float = 20.0
    terminal_obstacle_weight: float = 100.0
    seed_endpoint_fraction: float = 0.60
    seed_perturbation_fraction_of_acceleration_limit: float = 0.10
    dense_substeps: int = 16


PROTOCOL = PointMassProtocol()


def sha256_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def solver_task_identity_hash(x0, reference):
    """Hash the exact float32 tensors that define a solver-bound task."""
    digest = hashlib.sha256()
    for name, value, shape in (
        ("x0", x0, (4,)),
        ("reference", reference, (6,)),
    ):
        array = np.asarray(value)
        if array.dtype != np.float32 or array.shape != shape:
            raise ValueError(f"{name} must be float32 with shape {shape}")
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def generate_task(task_seed, protocol=PROTOCOL):
    """Generate a task whose canonical identity is its solver float32 bytes."""
    rng = np.random.default_rng(int(task_seed))
    drawn_theta = float(rng.uniform(-np.pi, np.pi))
    drawn_direction = np.asarray(
        [np.cos(drawn_theta), np.sin(drawn_theta)], dtype=np.float64
    )
    drawn_normal = np.asarray(
        [-drawn_direction[1], drawn_direction[0]], dtype=np.float64
    )
    drawn_offset_sign = float(rng.choice((-1.0, 1.0)))
    drawn_offset = float(rng.uniform(*protocol.disk_offset_range))
    drawn_radius = float(rng.uniform(*protocol.disk_radius_range))
    drawn_start = -protocol.start_goal_half_distance * drawn_direction
    drawn_goal = protocol.start_goal_half_distance * drawn_direction
    drawn_disk = drawn_offset_sign * drawn_offset * drawn_normal

    # GATO consumes these exact float32 arrays. Every float64 oracle field below
    # is promoted from these bytes, never from the higher-precision RNG draws.
    x0_solver = np.asarray([*drawn_start, 0.0, 0.0], dtype=np.float32)
    reference_solver = np.asarray(
        [*drawn_goal, *drawn_disk, drawn_radius, 0.0], dtype=np.float32
    )
    start = x0_solver[:2].astype(np.float64)
    goal = reference_solver[:2].astype(np.float64)
    disk = reference_solver[2:4].astype(np.float64)
    radius = float(reference_solver[4])
    travel = goal - start
    direction = travel / np.linalg.norm(travel)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    signed_offset = float(disk @ normal)
    offset_sign = 1.0 if signed_offset >= 0.0 else -1.0
    offset = abs(signed_offset)
    theta = float(np.arctan2(direction[1], direction[0]))
    endpoint_clearance = (
        np.linalg.norm(np.vstack([start, goal]) - disk[None, :], axis=1)
        - radius
    )
    return {
        "task_seed": int(task_seed),
        "theta_rad": theta,
        "direction": direction,
        "normal": normal,
        "offset_sign": offset_sign,
        "disk_offset_m": offset,
        "disk_radius_m": radius,
        "required_distance_m": radius + protocol.clearance_margin,
        "start_position_m": start,
        "goal_position_m": goal,
        "disk_center_m": disk,
        "x0": x0_solver,
        "reference": reference_solver,
        "solver_x0_sha256": sha256_array(x0_solver),
        "solver_reference_sha256": sha256_array(reference_solver),
        "solver_task_identity_sha256": solver_task_identity_hash(
            x0_solver, reference_solver
        ),
        "canonical_numeric_source": "exact_solver_bound_float32_bytes",
        "drawn_theta_rad_before_float32": drawn_theta,
        "drawn_offset_sign_before_float32": drawn_offset_sign,
        "drawn_disk_offset_m_before_float32": drawn_offset,
        "drawn_disk_radius_m_before_float32": drawn_radius,
        "disk_longitudinal_offset_m": float(disk @ direction),
        "endpoint_physical_clearance_m": endpoint_clearance,
        "predicted_worse_mode": (
            "clockwise" if offset_sign > 0.0 else "counterclockwise"
        ),
        "rng": "numpy.default_rng",
        "draw_order": ("theta", "offset_sign", "disk_offset", "disk_radius"),
        "rejection_or_retry_count": 0,
        "protocol": asdict(protocol),
    }


def affine_state_map(task, protocol=PROTOCOL, substeps=1):
    """Return exact affine dense state map x=b+B@vec(U), time-major U."""
    intervals = protocol.knots - 1
    step = protocol.dt / int(substeps)
    sample_count = intervals * int(substeps) + 1
    control_width = 2 * intervals
    offset = np.zeros((sample_count, 4), dtype=np.float64)
    mapping = np.zeros((sample_count, 4, control_width), dtype=np.float64)
    x0 = np.asarray(task["x0"], dtype=np.float64)
    offset[0] = x0
    for dense_index in range(sample_count - 1):
        coarse = dense_index // int(substeps)
        offset[dense_index + 1, :2] = (
            offset[dense_index, :2] + step * offset[dense_index, 2:]
        )
        offset[dense_index + 1, 2:] = offset[dense_index, 2:]
        mapping[dense_index + 1, :2] = (
            mapping[dense_index, :2] + step * mapping[dense_index, 2:]
        )
        mapping[dense_index + 1, 2:] = mapping[dense_index, 2:]
        for axis in range(2):
            column = 2 * coarse + axis
            mapping[dense_index + 1, axis, column] += 0.5 * step * step
            mapping[dense_index + 1, 2 + axis, column] += step
    return offset, mapping


def rollout_controls(task, controls, protocol=PROTOCOL, substeps=1):
    controls = np.asarray(controls, dtype=np.float64)
    single = controls.ndim == 2
    if single:
        controls = controls[None, ...]
    expected = (protocol.knots - 1, 2)
    if controls.shape[1:] != expected:
        raise ValueError(f"controls must have trailing shape {expected}")
    offset, mapping = affine_state_map(task, protocol, substeps)
    flat = controls.reshape(len(controls), -1)
    states = offset[None, ...] + np.einsum("sij,bj->bsi", mapping, flat)
    return states[0] if single else states


def terminal_control_map(task, protocol=PROTOCOL):
    offset, mapping = affine_state_map(task, protocol, substeps=1)
    return offset[-1], mapping[-1]


def terminal_target(task, endpoint_fraction=1.0):
    start = np.asarray(task["start_position_m"], dtype=np.float64)
    goal = np.asarray(task["goal_position_m"], dtype=np.float64)
    position = start + float(endpoint_fraction) * (goal - start)
    return np.hstack([position, np.zeros(2)])


def terminal_nullspace(task, protocol=PROTOCOL):
    _, terminal_map = terminal_control_map(task, protocol)
    _, singular, vt = np.linalg.svd(terminal_map, full_matrices=True)
    rank = int(np.sum(singular > 1e-12))
    nullspace = vt[rank:].T
    return terminal_map, nullspace, singular


def minimum_energy_controls(task, endpoint_fraction, protocol=PROTOCOL):
    offset, terminal_map = terminal_control_map(task, protocol)
    rhs = terminal_target(task, endpoint_fraction) - offset
    flat = terminal_map.T @ np.linalg.solve(terminal_map @ terminal_map.T, rhs)
    return flat.reshape(protocol.knots - 1, 2)


def low_frequency_control_draws(method_seed, protocol=PROTOCOL):
    intervals = protocol.knots - 1
    time = (np.arange(intervals, dtype=np.float64) + 0.5) / intervals
    basis = np.asarray(
        [np.cos(np.pi * frequency * time) for frequency in LOW_FREQUENCY_INDICES]
    )
    rng = np.random.default_rng(int(method_seed))
    coefficients = rng.normal(size=(8, 2, len(LOW_FREQUENCY_INDICES)))
    draws = np.einsum("baf,ft->bta", coefficients, basis)
    return draws, coefficients, basis


def generate_obstacle_and_mode_neutral_seeds(
    task, method_seed, protocol=PROTOCOL
):
    """Create 16 exact-dynamic, deliberately 60%-endpoint candidates."""
    nominal = minimum_energy_controls(
        task, protocol.seed_endpoint_fraction, protocol
    )
    terminal_map, nullspace, singular = terminal_nullspace(task, protocol)
    draws, coefficients, basis = low_frequency_control_draws(method_seed, protocol)
    projected = np.asarray(
        [
            (nullspace @ (nullspace.T @ draw.reshape(-1))).reshape(
                protocol.knots - 1, 2
            )
            for draw in draws
        ]
    )
    amplitude = (
        protocol.seed_perturbation_fraction_of_acceleration_limit
        * protocol.acceleration_limit
    )
    projected *= amplitude / np.max(np.abs(projected), axis=(1, 2))[:, None, None]
    controls = [nominal]
    labels = ["nominal_60_percent_endpoint"]
    for pair in range(7):
        controls.extend([nominal + projected[pair], nominal - projected[pair]])
        labels.extend(
            [f"neutral_{pair}_plus", f"neutral_{pair}_minus"]
        )
    controls.append(nominal + projected[7])
    labels.append("neutral_7_independent")
    controls = np.asarray(controls, dtype=np.float64)
    states = rollout_controls(task, controls, protocol)
    return states, controls, {
        "method_seed": int(method_seed),
        "initializer_family": "obstacle_and_mode_neutral",
        "candidate_labels": labels,
        "candidate_count": len(controls),
        "nominal_endpoint_fraction": protocol.seed_endpoint_fraction,
        "perturbation_infinity_amplitude": amplitude,
        "low_frequency_indices": LOW_FREQUENCY_INDICES,
        "gaussian_coefficients": coefficients,
        "basis": basis,
        "terminal_map_singular_values": singular,
        "disk_or_mode_inputs": 0,
        "oracle_inputs_or_artifacts": 0,
        "collision_checks": 0,
        "retries_or_rejections": 0,
        "calibration_or_feedback_iterations": 0,
        "single_exact_rollout": True,
    }


def obstacle_residual(position, task, protocol=PROTOCOL):
    position = np.asarray(position, dtype=np.float64)
    delta = position - np.asarray(task["disk_center_m"], dtype=np.float64)
    required = float(task["disk_radius_m"] + protocol.clearance_margin)
    value = 1.0 - float(delta @ delta) / (required * required)
    if value <= 0.0:
        return 0.0, np.zeros(2), np.zeros((2, 2))
    gradient = -2.0 * delta / (required * required)
    hessian = -2.0 * np.eye(2) / (required * required)
    return value, gradient, hessian


def objective_control_derivatives(task, controls, protocol=PROTOCOL):
    """Exact independent objective, gradient, Hessian w.r.t. flattened U."""
    controls = np.asarray(controls, dtype=np.float64)
    flat = controls.reshape(-1)
    offset, mapping = affine_state_map(task, protocol, substeps=1)
    states = offset + np.einsum("sij,j->si", mapping, flat)
    width = len(flat)
    gradient = np.zeros(width)
    hessian = np.zeros((width, width))
    total = 0.0
    goal = np.asarray(task["goal_position_m"], dtype=np.float64)
    for knot, state in enumerate(states):
        terminal = knot + 1 == protocol.knots
        scale = 1.0 if terminal else protocol.dt
        position_weight = (
            protocol.terminal_position_weight
            if terminal
            else protocol.running_position_weight
        )
        velocity_weight = (
            protocol.terminal_velocity_weight
            if terminal
            else protocol.running_velocity_weight
        )
        obstacle_weight = (
            protocol.terminal_obstacle_weight
            if terminal
            else protocol.running_obstacle_weight
        )
        position_error = state[:2] - goal
        velocity = state[2:]
        residual_value, residual_grad, residual_hess = obstacle_residual(
            state[:2], task, protocol
        )
        total += scale * 0.5 * (
            position_weight * float(position_error @ position_error)
            + velocity_weight * float(velocity @ velocity)
            + obstacle_weight * residual_value * residual_value
        )
        state_gradient = np.zeros(4)
        state_gradient[:2] = (
            position_weight * position_error
            + obstacle_weight * residual_value * residual_grad
        )
        state_gradient[2:] = velocity_weight * velocity
        state_hessian = np.zeros((4, 4))
        state_hessian[:2, :2] = (
            position_weight * np.eye(2)
            + obstacle_weight
            * (
                np.outer(residual_grad, residual_grad)
                + residual_value * residual_hess
            )
        )
        state_hessian[2:, 2:] = velocity_weight * np.eye(2)
        knot_map = mapping[knot]
        gradient += scale * knot_map.T @ state_gradient
        hessian += scale * knot_map.T @ state_hessian @ knot_map
    total += protocol.dt * 0.5 * protocol.running_control_weight * float(
        flat @ flat
    )
    gradient += protocol.dt * protocol.running_control_weight * flat
    hessian += (
        protocol.dt * protocol.running_control_weight * np.eye(width)
    )
    return total, gradient, hessian, states


def objective_cost(task, controls, protocol=PROTOCOL):
    return objective_control_derivatives(task, controls, protocol)[0]


def signed_turns(path, center):
    delta = np.asarray(path, dtype=np.float64) - np.asarray(
        center, dtype=np.float64
    )[None, :]
    angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    turns = float((angles[-1] - angles[0]) / (2.0 * np.pi))
    if turns <= -0.25:
        return "clockwise", turns
    if turns >= 0.25:
        return "counterclockwise", turns
    return "neutral", turns
