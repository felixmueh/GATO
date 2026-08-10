"""Frozen IIWA same-problem multimodal SQP development protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
import time

import numpy as np
import pinocchio as pin


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "examples/iiwa_description/iiwa14.urdf"
BUDGET = 16
DEVELOPMENT_TASK_SEEDS = (170, 171, 172)
DEVELOPMENT_METHOD_SEEDS = (5000, 5001, 5002)
FINAL_TASK_SEEDS = (180, 181, 182)
FINAL_METHOD_SEEDS = tuple(range(6000, 6020))
CENTRAL_FINAL_TASK_SEED = 181
FINAL_PROTOCOL = None


@dataclass(frozen=True)
class IiwaMultimodalProtocol:
    knots: int = 16
    dt: float = 0.04
    budget: int = BUDGET
    joint_amplitude_rad: float = 0.01
    travel_m: float = 0.13
    goal_dz_m: float = 0.02
    pillar_radius_m: float = 0.025
    lateral_offset_m: float = 0.008
    terminal_cuda_tolerance_m: float = 0.020
    terminal_pinocchio_tolerance_m: float = 0.022
    minimum_clearance_margin_m: float = 0.003
    max_sqp_iters: int = 60
    max_pcg_iters: int = 500
    pcg_tol: float = 1e-4
    q_cost: float = 2.0
    qd_cost: float = 0.01
    u_cost: float = 1e-4
    terminal_cost: float = 100.0
    pillar_cost: float = 800.0
    mu: float = 1000.0
    rho: float = 1.0


DEVELOPMENT_PROTOCOL = IiwaMultimodalProtocol()


def load_model():
    return pin.buildModelFromUrdf(str(MODEL_PATH))


def tool_position(model, data, q):
    pin.forwardKinematics(model, data, np.asarray(q, dtype=np.float64))
    return data.oMi[model.njoints - 1].translation.copy()


def tool_path(model, q_path):
    data = model.createData()
    return np.asarray([tool_position(model, data, q) for q in q_path])


def pillar_residual_and_gradient(position_xy, pillar_xy, radius):
    delta = np.asarray(position_xy, dtype=np.float64) - np.asarray(
        pillar_xy, dtype=np.float64
    )
    inv_radius_sq = 1.0 / float(radius) ** 2
    residual = 1.0 - float(delta @ delta) * inv_radius_sq
    if residual <= 0.0:
        return 0.0, np.zeros(2, dtype=np.float64)
    return residual, -2.0 * delta * inv_radius_sq


def generate_instance(task_seed, protocol=DEVELOPMENT_PROTOCOL, model=None):
    """Generate one exact no-retry frozen IIWA pillar problem."""
    if model is None:
        model = load_model()
    rng = np.random.default_rng(int(task_seed))
    q_draw = rng.uniform(-0.10, 0.10, size=model.nq)
    q0 = np.clip(
        q_draw,
        model.lowerPositionLimit.astype(np.float64) + 0.25,
        model.upperPositionLimit.astype(np.float64) - 0.25,
    )
    start = tool_position(model, model.createData(), q0)
    base_angle = float(np.arctan2(0.04, 0.07))
    angle_jitter = float(rng.uniform(-0.12, 0.12))
    angle = base_angle + angle_jitter
    direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
    goal = start + np.asarray(
        [protocol.travel_m * direction[0], protocol.travel_m * direction[1], protocol.goal_dz_m]
    )
    lateral_sign = float(rng.choice((-1.0, 1.0)))
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    pillar = (
        0.5 * (start[:2] + goal[:2])
        + lateral_sign * protocol.lateral_offset_m * normal
    )
    endpoint_clearances = (
        np.linalg.norm(
            np.vstack([start[:2], goal[:2]]) - pillar[None, :], axis=1
        )
        - protocol.pillar_radius_m
    )
    if float(np.min(endpoint_clearances)) < 0.035:
        raise RuntimeError("frozen IIWA pillar violates endpoint-clearance assertion")
    reference = np.asarray(
        [*goal, *pillar, protocol.pillar_radius_m], dtype=np.float32
    )
    return {
        "task_seed": int(task_seed),
        "q0": q0.astype(np.float32),
        "start_position_m": start,
        "goal_position_m": goal,
        "pillar_xy_m": pillar,
        "pillar_radius_m": protocol.pillar_radius_m,
        "reference": reference,
        "metadata": {
            "rng": "numpy.default_rng",
            "q0_draw": q_draw.tolist(),
            "q0_clipped": q0.tolist(),
            "base_planar_angle_rad": base_angle,
            "angle_jitter_draw_rad": angle_jitter,
            "achieved_planar_angle_rad": angle,
            "lateral_sign_draw": lateral_sign,
            "endpoint_clearances_m": endpoint_clearances.tolist(),
            "minimum_endpoint_clearance_m": float(np.min(endpoint_clearances)),
            "rejection_or_retry_count": 0,
            "protocol": asdict(protocol),
        },
    }


def discrete_rnea_candidate_labels():
    labels = ["stationary_inverse_dynamics_hold"]
    for pair in range(7):
        labels.extend([f"discrete_rnea_{pair}_plus", f"discrete_rnea_{pair}_minus"])
    return labels + ["discrete_rnea_7_independent"]


def conditioning_candidate_labels():
    labels = ["stationary_gravity_hold"]
    for pair in range(7):
        labels.extend(
            [f"conditioning_{pair}_plus", f"conditioning_{pair}_minus"]
        )
    return labels + ["conditioning_7_independent"]


def candidate_directions(model, method_seed):
    rng = np.random.default_rng(int(method_seed))
    draws = rng.normal(size=(8, model.nv))
    draws /= np.linalg.norm(draws, axis=1, keepdims=True)
    rows = [np.zeros(model.nv)]
    for direction in draws[:7]:
        rows.extend([direction, -direction])
    return np.asarray(rows + [draws[7]])


def discrete_acceleration_profile(protocol=DEVELOPMENT_PROTOCOL):
    intervals = protocol.knots - 1
    indices = np.arange(intervals, dtype=np.float64)
    raw = np.cos(2.0 * np.pi * (indices + 0.5) / intervals)
    constraints = np.vstack(
        [np.ones(intervals), intervals - indices - 0.5]
    )
    projected = raw - constraints.T @ np.linalg.solve(
        constraints @ constraints.T, constraints @ raw
    )
    displacement = np.zeros(intervals + 1, dtype=np.float64)
    velocity = np.zeros(intervals + 1, dtype=np.float64)
    for knot in range(intervals):
        velocity[knot + 1] = velocity[knot] + protocol.dt * projected[knot]
        displacement[knot + 1] = (
            displacement[knot]
            + protocol.dt * velocity[knot]
            + 0.5 * protocol.dt**2 * projected[knot]
        )
    scale = protocol.joint_amplitude_rad / np.max(np.abs(displacement))
    acceleration = scale * projected
    displacement *= scale
    velocity *= scale
    return displacement, velocity, acceleration, {
        "interval_count": intervals,
        "raw_profile": raw.tolist(),
        "projection_constraint_rows": constraints.tolist(),
        "raw_constraint_residual": (constraints @ raw).tolist(),
        "projected_constraint_residual_before_scale": (
            constraints @ projected
        ).tolist(),
        "scaled_constraint_residual": (
            constraints @ acceleration
        ).tolist(),
        "single_scale_factor": float(scale),
        "maximum_scaled_displacement_rad": float(
            np.max(np.abs(displacement))
        ),
        "terminal_displacement_residual_rad": float(displacement[-1]),
        "terminal_velocity_residual_rad_s": float(velocity[-1]),
    }


def generate_discrete_rnea_seeds(
    model, q0, method_seed, protocol=DEVELOPMENT_PROTOCOL
):
    """Create task-neutral, discrete-consistent inverse-dynamics seeds."""
    started = time.perf_counter()
    profile_protocol = replace(protocol, joint_amplitude_rad=0.04)
    directions = candidate_directions(model, method_seed)
    displacement, velocity, acceleration, profile = discrete_acceleration_profile(
        profile_protocol
    )
    controls = np.empty(
        (BUDGET, protocol.knots - 1, model.nv), dtype=np.float32
    )
    nx, nu = model.nq + model.nv, model.nv
    width = protocol.knots * (nx + nu) - nu
    seeds = np.zeros((BUDGET, width), dtype=np.float32)
    q_paths = np.empty((BUDGET, protocol.knots, model.nq), dtype=np.float64)
    qd_paths = np.empty((BUDGET, protocol.knots, model.nv), dtype=np.float64)
    inverse_dynamics_calls = 0
    for candidate, direction in enumerate(directions):
        q_paths[candidate] = (
            np.asarray(q0, dtype=np.float64)[None, :]
            + displacement[:, None] * direction
        )
        qd_paths[candidate] = velocity[:, None] * direction
        qdd_path = acceleration[:, None] * direction
        data = model.createData()
        for knot in range(protocol.knots - 1):
            controls[candidate, knot] = pin.rnea(
                model,
                data,
                q_paths[candidate, knot],
                qd_paths[candidate, knot],
                qdd_path[knot],
            ).astype(np.float32)
            inverse_dynamics_calls += 1
        for knot in range(protocol.knots):
            offset = knot * (nx + nu)
            seeds[candidate, offset : offset + model.nq] = q_paths[candidate, knot]
            seeds[
                candidate,
                offset + model.nq : offset + nx,
            ] = qd_paths[candidate, knot]
            if knot + 1 < protocol.knots:
                seeds[candidate, offset + nx : offset + nx + nu] = controls[
                    candidate, knot
                ]
    return seeds, controls, {
        "method_seed": int(method_seed),
        "generation_latency_ms": 1e3 * (time.perf_counter() - started),
        "inverse_dynamics_calls": inverse_dynamics_calls,
        "expected_inverse_dynamics_calls": BUDGET * (protocol.knots - 1),
        "candidate_directions": directions.tolist(),
        "candidate_labels": discrete_rnea_candidate_labels(),
        "acceleration_profile": profile,
        "planned_state_basis": (
            "projected_cosine_acceleration_with_exact_discrete_closure"
        ),
        "planned_velocity": "integrated projected acceleration times direction",
        "control_generation": "one Pinocchio RNEA per candidate and interval",
        "rollout_used_to_populate_planned_states": False,
        "task_reference_or_obstacle_inputs": 0,
        "feedback_or_calibration_iterations": 0,
        "rejection_or_resampling_attempts": 0,
        "post_profile_rescale_or_retry_count": 0,
    }


def generate_planned_seeds(model, q0, method_seed, protocol=DEVELOPMENT_PROTOCOL):
    """Create the final cheap, task-neutral conditioning multistarts."""
    started = time.perf_counter()
    directions = candidate_directions(model, method_seed)
    progress = np.linspace(0.0, 1.0, protocol.knots, dtype=np.float64)
    envelope = np.sin(np.pi * progress) ** 2
    gravity = pin.rnea(
        model,
        model.createData(),
        np.asarray(q0, dtype=np.float64),
        np.zeros(model.nv),
        np.zeros(model.nv),
    ).astype(np.float32)
    controls = np.tile(
        gravity, (BUDGET, protocol.knots - 1, 1)
    ).astype(np.float32)
    nx, nu = model.nq + model.nv, model.nv
    seeds = np.zeros(
        (BUDGET, protocol.knots * (nx + nu) - nu), dtype=np.float32
    )
    for candidate, direction in enumerate(directions):
        q_path = (
            np.asarray(q0, dtype=np.float64)[None, :]
            + protocol.joint_amplitude_rad * envelope[:, None] * direction
        )
        q_path[0] = q0
        q_path[-1] = q0
        for knot in range(protocol.knots):
            offset = knot * (nx + nu)
            seeds[candidate, offset : offset + model.nq] = q_path[knot]
            if knot + 1 < protocol.knots:
                seeds[candidate, offset + nx : offset + nx + nu] = controls[
                    candidate, knot
                ]
    return seeds, controls, {
        "method_seed": int(method_seed),
        "generation_latency_ms": 1e3 * (time.perf_counter() - started),
        "inverse_dynamics_calls": 1,
        "expected_inverse_dynamics_calls": 1,
        "candidate_directions": directions.tolist(),
        "candidate_labels": conditioning_candidate_labels(),
        "planned_state_basis": "0.01*sin(pi*t/T)^2*unit_direction",
        "planned_velocity": "zero",
        "control_generation": "one stationary-gravity RNEA call per case",
        "identical_controls_all_candidates": True,
        "rollout_used_to_populate_planned_states": False,
        "task_reference_or_obstacle_inputs": 0,
        "inverse_kinematics_calls": 0,
        "feedback_or_calibration_iterations": 0,
        "rejection_or_resampling_attempts": 0,
    }


def unpack_planned_seeds(model, seeds, protocol=DEVELOPMENT_PROTOCOL):
    nx, nu = model.nq + model.nv, model.nv
    q = np.empty((len(seeds), protocol.knots, model.nq))
    qd = np.empty((len(seeds), protocol.knots, model.nv))
    controls = np.empty((len(seeds), protocol.knots - 1, model.nv))
    for knot in range(protocol.knots):
        offset = knot * (nx + nu)
        q[:, knot] = seeds[:, offset : offset + model.nq]
        qd[:, knot] = seeds[:, offset + model.nq : offset + nx]
        if knot + 1 < protocol.knots:
            controls[:, knot] = seeds[:, offset + nx : offset + nx + nu]
    return q, qd, controls


def planned_seed_screen(
    model,
    q0,
    seeds,
    protocol=DEVELOPMENT_PROTOCOL,
    initializer_kind="final_conditioning",
):
    q, qd, controls = unpack_planned_seeds(model, seeds, protocol)
    hashes = [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in seeds
    ]
    pairwise_rms = [
        float(np.sqrt(np.mean((q[left] - q[right]) ** 2)))
        for left in range(BUDGET)
        for right in range(left + 1, BUDGET)
    ]
    defects = []
    for candidate in range(BUDGET):
        data = model.createData()
        maximum = 0.0
        for knot in range(protocol.knots - 1):
            state = np.hstack([q[candidate, knot], qd[candidate, knot]])
            predicted = pinocchio_step(
                model, data, state, controls[candidate, knot], protocol.dt
            )
            actual = np.hstack([q[candidate, knot + 1], qd[candidate, knot + 1]])
            maximum = max(maximum, float(np.linalg.norm(predicted - actual)))
        defects.append(maximum)
    joint_violation = float(
        max(
            np.max(model.lowerPositionLimit - q, initial=0.0),
            np.max(q - model.upperPositionLimit, initial=0.0),
            0.0,
        )
    )
    velocity_ratio = float(np.max(np.abs(qd) / model.velocityLimit))
    control_ratio = float(np.max(np.abs(controls) / model.effortLimit))
    finite = bool(
        np.all(np.isfinite(q))
        and np.all(np.isfinite(qd))
        and np.all(np.isfinite(controls))
    )
    common_controls = all(
        np.array_equal(controls[0], controls[index])
        for index in range(1, BUDGET)
    )
    common_x0 = bool(
        np.all(q[:, 0] == np.asarray(q0)[None, :]) and np.all(qd[:, 0] == 0.0)
    )
    terminal_q_residual = float(
        np.max(np.abs(q[:, -1] - np.asarray(q0)[None, :]))
    )
    terminal_qd_residual = float(np.max(np.abs(qd[:, -1])))
    endpoints = bool(
        terminal_q_residual <= 1e-6 and terminal_qd_residual <= 1e-6
    )
    if initializer_kind == "final_conditioning":
        defect_profile_pass = bool(
            defects[0] <= 1e-4 and 0.04 <= max(defects) <= 0.08
        )
        control_structure_pass = common_controls
    elif initializer_kind == "discrete_rnea":
        defect_profile_pass = max(defects) <= 1e-4
        control_structure_pass = True
    else:
        raise ValueError(f"unknown initializer_kind={initializer_kind}")
    return {
        "initializer_kind": initializer_kind,
        "unique_planned_seed_count": len(set(hashes)),
        "planned_seed_sha256": hashes,
        "minimum_pairwise_q_path_rms_rad": min(pairwise_rms),
        "common_exact_x0": common_x0,
        "common_controls_all_candidates": common_controls,
        "terminal_q_closure_residual_rad": terminal_q_residual,
        "terminal_qd_closure_residual_rad_s": terminal_qd_residual,
        "terminal_closure_within_1e_6": endpoints,
        "planned_joint_violation_rad": joint_violation,
        "maximum_planned_velocity_limit_ratio": velocity_ratio,
        "maximum_planned_control_limit_ratio": control_ratio,
        "all_planned_values_finite": finite,
        "planned_pinocchio_one_step_defect": defects,
        "maximum_planned_pinocchio_one_step_defect": max(defects),
        "cold_candidate_pinocchio_one_step_defect": defects[0],
        "predeclared_defect_profile_pass": defect_profile_pass,
        "passes": bool(
            len(set(hashes)) == BUDGET
            and min(pairwise_rms) >= 1e-4
            and control_structure_pass
            and common_x0
            and endpoints
            and joint_violation <= 0.0
            and velocity_ratio <= 1.0
            and control_ratio <= 1.0
            and defect_profile_pass
            and finite
        ),
    }


def normalized_final_control_change(final_controls, seed_controls, effort):
    """Measure optimizer contribution independently of planned-state repair."""
    delta = (
        np.asarray(final_controls, dtype=np.float64)
        - np.asarray(seed_controls, dtype=np.float64)
    ) / np.asarray(effort, dtype=np.float64)[None, :]
    return float(np.sqrt(np.mean(delta * delta)))


def batch_lane_repeatability(output, stats, relative_merit_tolerance=1e-6):
    """Compare every identical-input batch lane against lane zero."""
    output = np.asarray(output)
    discrete_fields = (
        "sqp_iterations",
        "total_pcg_iterations",
        "pcg_cap_hits",
        "reached_requested_sqp_iteration_limit",
    )
    output_equal = [
        bool(np.array_equal(output[0], output[index]))
        for index in range(len(output))
    ]
    discrete_equal = {
        field: [
            bool(
                np.array_equal(
                    np.asarray(stats[field])[0], np.asarray(stats[field])[index]
                )
            )
            for index in range(len(output))
        ]
        for field in discrete_fields
    }
    merit_equal = {}
    for field in ("initial_merit", "final_merit"):
        values = np.asarray(stats[field], dtype=np.float64)
        scale = max(abs(values[0]), 1.0)
        merit_equal[field] = [
            bool(abs(values[index] - values[0]) / scale <= relative_merit_tolerance)
            for index in range(len(output))
        ]
    return {
        "candidate_count": len(output),
        "relative_merit_tolerance": relative_merit_tolerance,
        "exact_output_equal_to_lane0": output_equal,
        "exact_discrete_telemetry_equal_to_lane0": discrete_equal,
        "merit_numerically_equal_to_lane0": merit_equal,
        "passes": bool(
            len(output) == BUDGET
            and all(output_equal)
            and all(all(rows) for rows in discrete_equal.values())
            and all(all(rows) for rows in merit_equal.values())
        ),
    }


def pinocchio_step(model, data, state, control, dt):
    q = np.asarray(state[: model.nq], dtype=np.float64)
    qd = np.asarray(state[model.nq :], dtype=np.float64)
    qdd = pin.aba(model, data, q, qd, np.asarray(control, dtype=np.float64))
    return np.hstack([q + dt * qd + 0.5 * dt * dt * qdd, qd + dt * qdd])


def pinocchio_rollout(model, x0, controls, dt):
    states = np.empty((len(controls), controls.shape[1] + 1, len(x0)))
    for candidate in range(len(controls)):
        current = np.asarray(x0, dtype=np.float64).copy()
        data = model.createData()
        for knot in range(states.shape[1]):
            states[candidate, knot] = current
            if knot + 1 < states.shape[1]:
                current = pinocchio_step(
                    model, data, current, controls[candidate, knot], dt
                )
    return states


def cuda_rollout(sim_forward, x0, controls, dt):
    states = np.empty(
        (len(controls), controls.shape[1] + 1, len(x0)), dtype=np.float32
    )
    current = np.tile(np.asarray(x0, dtype=np.float32), (len(controls), 1))
    for knot in range(states.shape[1]):
        states[:, knot] = current
        if knot + 1 < states.shape[1]:
            current = np.asarray(
                sim_forward(current, controls[:, knot], dt), dtype=np.float32
            )
    return states


def model_pair_preflight(sim_forward, model, q0, dt, seed=20260811):
    """Fixed benign and bounded-random one-step CUDA/Pinocchio check."""
    rng = np.random.default_rng(seed)
    zeros = np.zeros(model.nv)
    gravity = pin.rnea(model, model.createData(), q0, zeros, zeros)
    x0 = np.hstack([q0, zeros]).astype(np.float32)
    benign = []
    for name, control in (("zero", zeros), ("gravity", gravity)):
        cuda = np.asarray(sim_forward(x0, control.astype(np.float32), dt))[0]
        expected = pinocchio_step(model, model.createData(), x0, control, dt)
        difference = np.asarray(cuda, dtype=np.float64) - expected
        benign.append(
            {
                "name": name,
                "l2_error": float(np.linalg.norm(difference)),
                "max_abs_error": float(np.max(np.abs(difference))),
            }
        )
    q = np.tile(np.asarray(q0, dtype=np.float64), (BUDGET, 1))
    q += rng.uniform(-0.70, 0.70, size=q.shape)
    q = np.clip(q, model.lowerPositionLimit + 0.05, model.upperPositionLimit - 0.05)
    qd = rng.uniform(-0.25, 0.25, size=(BUDGET, model.nv))
    controls = rng.uniform(-0.20, 0.20, size=(BUDGET, model.nv)) * model.effortLimit
    states = np.hstack([q, qd]).astype(np.float32)
    controls = controls.astype(np.float32)
    cuda = np.asarray(sim_forward(states, controls, dt), dtype=np.float64)
    expected = np.asarray(
        [
            pinocchio_step(
                model, model.createData(), states[index], controls[index], dt
            )
            for index in range(BUDGET)
        ]
    )
    difference = cuda - expected
    norms = np.linalg.norm(difference, axis=1)
    random_cases = [
        {
            "candidate_index": index,
            "l2_error": float(norms[index]),
            "max_abs_error": float(np.max(np.abs(difference[index]))),
            "finite": bool(
                np.all(np.isfinite(cuda[index]))
                and np.all(np.isfinite(expected[index]))
                and np.all(np.isfinite(difference[index]))
            ),
        }
        for index in range(BUDGET)
    ]
    maximum_l2 = max(float(np.max(norms)), *(row["l2_error"] for row in benign))
    maximum_abs = max(
        float(np.max(np.abs(difference))),
        *(row["max_abs_error"] for row in benign),
    )
    return {
        "seed": seed,
        "case_count": BUDGET,
        "dt": dt,
        "distribution": {
            "q_jitter_rad": [-0.70, 0.70],
            "qd_rad_s": [-0.25, 0.25],
            "control_effort_fraction": [-0.20, 0.20],
        },
        "benign": benign,
        "random_cases": random_cases,
        "all_random_cases_retained": len(random_cases) == BUDGET,
        "maximum_l2_error": maximum_l2,
        "maximum_abs_error": maximum_abs,
        "passes": bool(
            len(random_cases) == BUDGET
            and all(row["finite"] for row in random_cases)
            and maximum_l2 <= 1e-3
            and maximum_abs <= 1e-3
        ),
    }


def seed_model_screen(
    model, instance, controls, cuda_states, pin_states, preflight,
    protocol=DEVELOPMENT_PROTOCOL,
):
    rows = []
    for candidate in range(BUDGET):
        difference = np.asarray(cuda_states[candidate], dtype=np.float64) - pin_states[candidate]
        state_l2 = np.linalg.norm(difference, axis=1)
        cuda_tool = tool_path(model, cuda_states[candidate, :, : model.nq])
        pin_tool = tool_path(model, pin_states[candidate, :, : model.nq])
        ee_disagreement = np.linalg.norm(cuda_tool - pin_tool, axis=1)
        cuda_terminal = float(np.linalg.norm(cuda_tool[-1] - instance["goal_position_m"]))
        pin_terminal = float(np.linalg.norm(pin_tool[-1] - instance["goal_position_m"]))
        finite = bool(
            np.all(np.isfinite(cuda_states[candidate]))
            and np.all(np.isfinite(pin_states[candidate]))
            and np.all(np.isfinite(cuda_tool))
            and np.all(np.isfinite(pin_tool))
        )
        cuda_q = cuda_states[candidate, :, : model.nq]
        cuda_qd = cuda_states[candidate, :, model.nq :]
        pin_q = pin_states[candidate, :, : model.nq]
        pin_qd = pin_states[candidate, :, model.nq :]
        limits = bool(
            np.max(model.lowerPositionLimit - cuda_q, initial=0.0) <= 0.0
            and np.max(cuda_q - model.upperPositionLimit, initial=0.0) <= 0.0
            and np.max(model.lowerPositionLimit - pin_q, initial=0.0) <= 0.0
            and np.max(pin_q - model.upperPositionLimit, initial=0.0) <= 0.0
            and np.max(np.abs(cuda_qd) / model.velocityLimit) <= 1.0
            and np.max(np.abs(pin_qd) / model.velocityLimit) <= 1.0
            and np.max(np.abs(controls[candidate]) / model.effortLimit) <= 1.0
        )
        rows.append(
            {
                "candidate_index": candidate,
                "finite": finite,
                "limits_pass": limits,
                "cuda_terminal_error_m": cuda_terminal,
                "pinocchio_terminal_error_m": pin_terminal,
                "imperfect": bool(cuda_terminal >= 0.05 and pin_terminal >= 0.05),
                "maximum_cuda_pin_state_l2": float(np.max(state_l2)),
                "maximum_cuda_pin_ee_disagreement_m": float(
                    np.max(ee_disagreement)
                ),
                "model_agreement": bool(
                    np.max(state_l2) <= 0.002
                    and np.max(ee_disagreement) <= 0.001
                ),
            }
        )
    return rows, {
        "all_candidates_retained": len(rows) == BUDGET,
        "all_candidates_finite": all(row["finite"] for row in rows),
        "all_candidates_imperfect": all(row["imperfect"] for row in rows),
        "all_candidates_within_limits": all(row["limits_pass"] for row in rows),
        "all_full_path_model_gates_pass": all(
            row["model_agreement"] for row in rows
        ),
        "one_step_preflight_pass": bool(preflight["passes"]),
        "passes": bool(
            preflight["passes"]
            and len(rows) == BUDGET
            and all(
                row["finite"]
                and row["limits_pass"]
                and row["imperfect"]
                and row["model_agreement"]
                for row in rows
            )
        ),
    }


def independent_pillar_cost(model, states, pillar_xy, radius, weight):
    path = tool_path(model, np.asarray(states)[:, : model.nq])
    residuals = np.asarray(
        [
            pillar_residual_and_gradient(position[:2], pillar_xy, radius)[0]
            for position in path
        ]
    )
    return {
        "weighted_cost": float(0.5 * weight * np.sum(residuals * residuals)),
        "sum_positive_hinge": float(np.sum(residuals)),
        "maximum_positive_hinge": float(np.max(residuals)),
        "minimum_knot_clearance_m": float(
            np.min(np.linalg.norm(path[:, :2] - pillar_xy, axis=1)) - radius
        ),
        "residuals": residuals,
        "tool_path": path,
    }


DENSE_SUBSTEPS = 4
MODE_TURN_THRESHOLD = 0.25
PAIRWISE_INTEGER_RESIDUAL = 0.10
CONTROL_UNIQUENESS_RMS = 1e-4
TOOL_UNIQUENESS_RMS_M = 0.001


def dense_pinocchio_rollout(model, x0, controls, dt, substeps=DENSE_SUBSTEPS):
    """Replay piecewise-constant controls at fixed dense substeps."""
    controls = np.asarray(controls, dtype=np.float64)
    step_dt = float(dt) / int(substeps)
    states = np.empty(
        (len(controls), controls.shape[1] * substeps + 1, len(x0)),
        dtype=np.float64,
    )
    for candidate in range(len(controls)):
        current = np.asarray(x0, dtype=np.float64).copy()
        data = model.createData()
        dense_index = 0
        states[candidate, dense_index] = current
        for control in controls[candidate]:
            for _ in range(substeps):
                current = pinocchio_step(model, data, current, control, step_dt)
                dense_index += 1
                states[candidate, dense_index] = current
    return states


def dense_cuda_rollout(sim_forward, x0, controls, dt, substeps=DENSE_SUBSTEPS):
    controls = np.asarray(controls, dtype=np.float32)
    step_dt = float(dt) / int(substeps)
    states = np.empty(
        (len(controls), controls.shape[1] * substeps + 1, len(x0)),
        dtype=np.float32,
    )
    current = np.tile(np.asarray(x0, dtype=np.float32), (len(controls), 1))
    dense_index = 0
    states[:, dense_index] = current
    for knot in range(controls.shape[1]):
        for _ in range(substeps):
            current = np.asarray(
                sim_forward(current, controls[:, knot], step_dt), dtype=np.float32
            )
            dense_index += 1
            states[:, dense_index] = current
    return states


def external_replay_cost(
    model,
    knot_states,
    controls,
    goal_position,
    pillar_xy,
    radius,
    protocol=DEVELOPMENT_PROTOCOL,
):
    """Reconstruct the frozen nonnegative task/motion/pillar objective."""
    states = np.asarray(knot_states, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    q, qd = states[:, : model.nq], states[:, model.nq :]
    path = tool_path(model, q)
    goal_error_sq = np.sum(
        (path - np.asarray(goal_position, dtype=np.float64)[None, :]) ** 2,
        axis=1,
    )
    pillar_residual = np.asarray(
        [
            pillar_residual_and_gradient(point[:2], pillar_xy, radius)[0]
            for point in path
        ]
    )
    components = {
        "running_position": float(
            0.5 * protocol.q_cost * np.sum(goal_error_sq[:-1])
        ),
        "terminal_position": float(
            0.5 * protocol.terminal_cost * goal_error_sq[-1]
        ),
        "velocity": float(0.5 * protocol.qd_cost * np.sum(qd * qd)),
        "control": float(0.5 * protocol.u_cost * np.sum(controls * controls)),
        "pillar": float(
            0.5 * protocol.pillar_cost * np.sum(pillar_residual**2)
        ),
    }
    total = float(sum(components.values()))
    return {
        "total": total,
        "components": components,
        "pillar_residual": pillar_residual,
        "tool_path": path,
        "finite": bool(
            np.isfinite(total)
            and np.all(np.isfinite(states))
            and np.all(np.isfinite(controls))
            and np.all(np.isfinite(path))
        ),
    }


def dense_replay_metrics(
    model,
    instance,
    dense_states,
    controls,
    protocol=DEVELOPMENT_PROTOCOL,
    substeps=DENSE_SUBSTEPS,
):
    dense_states = np.asarray(dense_states, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    knot_states = dense_states[::substeps]
    q = dense_states[:, : model.nq]
    qd = dense_states[:, model.nq :]
    dense_tool = tool_path(model, q)
    cost = external_replay_cost(
        model,
        knot_states,
        controls,
        instance["goal_position_m"],
        instance["pillar_xy_m"],
        instance["pillar_radius_m"],
        protocol,
    )
    clearance = float(
        np.min(
            np.linalg.norm(
                dense_tool[:, :2]
                - np.asarray(instance["pillar_xy_m"])[None, :],
                axis=1,
            )
        )
        - instance["pillar_radius_m"]
    )
    return {
        "external_cost": cost["total"],
        "external_cost_components": cost["components"],
        "terminal_position_m": dense_tool[-1],
        "terminal_error_m": float(
            np.linalg.norm(dense_tool[-1] - instance["goal_position_m"])
        ),
        "minimum_dense_clearance_m": clearance,
        "joint_violation_rad": float(
            max(
                np.max(model.lowerPositionLimit - q, initial=0.0),
                np.max(q - model.upperPositionLimit, initial=0.0),
                0.0,
            )
        ),
        "velocity_ratio": float(np.max(np.abs(qd) / model.velocityLimit)),
        "control_ratio": float(np.max(np.abs(controls) / model.effortLimit)),
        "finite": bool(cost["finite"] and np.all(np.isfinite(dense_tool))),
        "dense_tool_path": dense_tool,
    }


def signed_xy_turns(path, pillar_xy):
    delta = np.asarray(path, dtype=np.float64)[:, :2] - np.asarray(
        pillar_xy, dtype=np.float64
    )[None, :]
    angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    turns = float((angles[-1] - angles[0]) / (2.0 * np.pi))
    if turns >= MODE_TURN_THRESHOLD:
        return "counterclockwise", turns
    if turns <= -MODE_TURN_THRESHOLD:
        return "clockwise", turns
    return "neutral", turns


def normalized_control_path_rms(left, right, effort_limit):
    delta = (
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    ) / np.asarray(effort_limit, dtype=np.float64)[None, :]
    return float(np.sqrt(np.mean(delta * delta)))


def tool_path_rms(left, right):
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def mode_support_certificate(
    rows,
    pillar_xy,
    effort_limit,
    path_key,
    cost_key,
):
    candidates = [row for row in rows if row["productive_certified"]]
    for row in candidates:
        label, turns = signed_xy_turns(row[path_key], pillar_xy)
        row.setdefault("mode_labels", {})[path_key] = label
        row.setdefault("signed_turns", {})[path_key] = turns
    supported = {}
    for mode in ("clockwise", "counterclockwise"):
        unique = []
        ordered = sorted(
            (
                row
                for row in candidates
                if row["mode_labels"][path_key] == mode
            ),
            key=lambda row: row[cost_key],
        )
        for row in ordered:
            distinct_from_all = all(
                normalized_control_path_rms(
                    row["controls"], other["controls"], effort_limit
                )
                >= CONTROL_UNIQUENESS_RMS
                or tool_path_rms(row[path_key], other[path_key])
                >= TOOL_UNIQUENESS_RMS_M
                for other in unique
            )
            if distinct_from_all:
                unique.append(row)
        supported[mode] = unique

    def pair_class(left, right):
        delta = left["signed_turns"][path_key] - right["signed_turns"][path_key]
        nearest = int(np.rint(delta))
        return {
            "delta_turns": float(delta),
            "nearest_integer": nearest,
            "integer_residual": float(abs(delta - nearest)),
            "valid_integer": bool(abs(delta - nearest) <= PAIRWISE_INTEGER_RESIDUAL),
        }

    cross_pairs = []
    for clockwise in supported["clockwise"]:
        for counterclockwise in supported["counterclockwise"]:
            cross_pairs.append(
                {
                    "clockwise_candidate": clockwise["candidate_index"],
                    "counterclockwise_candidate": counterclockwise["candidate_index"],
                    **pair_class(clockwise, counterclockwise),
                }
            )
    same_pairs = []
    for mode, values in supported.items():
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                same_pairs.append(
                    {
                        "mode": mode,
                        "left_candidate": left["candidate_index"],
                        "right_candidate": right["candidate_index"],
                        **pair_class(left, right),
                    }
                )
    cross_pass = bool(cross_pairs) and all(
        row["valid_integer"] and abs(row["nearest_integer"]) == 1
        for row in cross_pairs
    )
    same_pass = all(
        row["valid_integer"] and row["nearest_integer"] == 0
        for row in same_pairs
    )
    support = {mode: len(values) for mode, values in supported.items()}
    return {
        "path_key": path_key,
        "support": support,
        "supported_candidate_indices_by_mode": {
            mode: [row["candidate_index"] for row in values]
            for mode, values in supported.items()
        },
        "candidate_modes": {
            str(row["candidate_index"]): row["mode_labels"][path_key]
            for row in candidates
        },
        "candidate_signed_turns": {
            str(row["candidate_index"]): row["signed_turns"][path_key]
            for row in candidates
        },
        "cross_mode_pair_classes": cross_pairs,
        "same_mode_pair_classes": same_pairs,
        "cross_mode_abs_one_integer": cross_pass,
        "same_mode_zero_integer": same_pass,
        "two_mode_support": bool(
            support["clockwise"] >= 2
            and support["counterclockwise"] >= 2
            and cross_pass
            and same_pass
        ),
    }


def topology_signature(certificate):
    return {
        "support": certificate["support"],
        "supported_candidate_indices_by_mode": certificate[
            "supported_candidate_indices_by_mode"
        ],
        "candidate_modes": certificate["candidate_modes"],
        "cross": sorted(
            (
                row["clockwise_candidate"],
                row["counterclockwise_candidate"],
                row["nearest_integer"],
                row["valid_integer"],
            )
            for row in certificate["cross_mode_pair_classes"]
        ),
        "same": sorted(
            (
                row["mode"],
                row["left_candidate"],
                row["right_candidate"],
                row["nearest_integer"],
                row["valid_integer"],
            )
            for row in certificate["same_mode_pair_classes"]
        ),
        "two_mode_support": certificate["two_mode_support"],
    }


def mode_certificates_agree(left, right):
    return topology_signature(left) == topology_signature(right)


def productivity_gate(evidence):
    required = (
        "seed_not_fully_feasible",
        "finite",
        "planned_trajectory_change",
        "final_control_change",
        "solver_merit_decrease",
        "cuda_external_cost_decrease",
        "pinocchio_external_cost_decrease",
        "terminal_reduction_and_final_tolerance",
        "dense_clearance",
        "limits",
        "dense_model_agreement",
        "cap_free",
    )
    return bool(all(bool(evidence[key]) for key in required))


def execute_after_seed_gate(seed_gate_passes, callback):
    """Fail closed while making zero solver calls on a rejected seed screen."""
    if not seed_gate_passes:
        return None, {
            "completed_b1_solve_once_calls": 0,
            "completed_b16_solve_once_calls": 0,
        }
    result = callback()
    return result, result["solve_call_audit"]


def solve_call_audit_passes(audit):
    return bool(
        audit["completed_b1_solve_once_calls"] == BUDGET
        and audit["completed_b16_solve_once_calls"] == 1
        and sum(audit.values()) == BUDGET + 1
    )


def _array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def captured_solve_input_identity(serial_inputs, batch_inputs):
    """Byte/shape/dtype identity for the exact tensors passed to solves."""
    fields = ("x0", "reference", "seeds")
    widths = {
        "x0": 14,
        "reference": DEVELOPMENT_PROTOCOL.knots * 6,
        "seeds": DEVELOPMENT_PROTOCOL.knots * (14 + 7) - 7,
    }
    field_rows = {}
    for field in fields:
        serial = np.asarray(serial_inputs[field])
        batch = np.asarray(batch_inputs[field])
        structural_pass = bool(
            serial.shape == (BUDGET, 1, widths[field])
            and batch.shape == (BUDGET, widths[field])
            and serial.dtype == np.dtype(np.float32)
            and batch.dtype == np.dtype(np.float32)
        )
        rows = []
        for candidate in range(BUDGET):
            serial_candidate = serial[candidate, 0]
            batch_candidate = batch[candidate]
            rows.append(
                {
                    "candidate_index": candidate,
                    "serial_shape": list(serial_candidate.shape),
                    "batch_shape": list(batch_candidate.shape),
                    "serial_dtype": str(serial_candidate.dtype),
                    "batch_dtype": str(batch_candidate.dtype),
                    "serial_sha256": _array_sha256(serial_candidate),
                    "batch_sha256": _array_sha256(batch_candidate),
                    "passes": bool(
                        serial_candidate.shape == batch_candidate.shape
                        and serial_candidate.dtype == batch_candidate.dtype
                        and _array_sha256(serial_candidate)
                        == _array_sha256(batch_candidate)
                    ),
                }
            )
        field_rows[field] = {
            "serial_captured_shape": list(serial.shape),
            "batch_captured_shape": list(batch.shape),
            "structural_pass": structural_pass,
            "candidate_rows": rows,
        }
    return {
        "candidate_count": BUDGET,
        "fields": field_rows,
        "passes": bool(
            all(value["structural_pass"] for value in field_rows.values())
            and all(
                row["passes"]
                for value in field_rows.values()
                for row in value["candidate_rows"]
            )
        ),
    }


def certify_real_execution(
    model,
    instance,
    seeds,
    outputs,
    stats,
    sim_forward,
    protocol=DEVELOPMENT_PROTOCOL,
):
    """Certify optimizer contribution from independently replayed controls."""
    _, _, seed_controls = unpack_planned_seeds(model, seeds, protocol)
    _, _, final_controls = unpack_planned_seeds(model, outputs, protocol)
    x0 = np.hstack([instance["q0"], np.zeros(model.nv)])
    seed_cuda = dense_cuda_rollout(
        sim_forward, x0, seed_controls, protocol.dt
    )
    seed_pin = dense_pinocchio_rollout(
        model, x0, seed_controls, protocol.dt
    )
    final_cuda = dense_cuda_rollout(
        sim_forward, x0, final_controls, protocol.dt
    )
    final_pin = dense_pinocchio_rollout(
        model, x0, final_controls, protocol.dt
    )
    rows = []
    for index in range(BUDGET):
        seed_cuda_metrics = dense_replay_metrics(
            model, instance, seed_cuda[index], seed_controls[index], protocol
        )
        seed_pin_metrics = dense_replay_metrics(
            model, instance, seed_pin[index], seed_controls[index], protocol
        )
        final_cuda_metrics = dense_replay_metrics(
            model, instance, final_cuda[index], final_controls[index], protocol
        )
        final_pin_metrics = dense_replay_metrics(
            model, instance, final_pin[index], final_controls[index], protocol
        )
        initial_merit = float(stats["initial_merit"][index])
        final_merit = float(stats["final_merit"][index])
        merit_decrease = initial_merit - final_merit
        trajectory_delta = np.asarray(outputs[index]) - np.asarray(seeds[index])
        trajectory_rms = float(
            np.linalg.norm(trajectory_delta) / np.sqrt(trajectory_delta.size)
        )
        control_change = normalized_final_control_change(
            final_controls[index], seed_controls[index], model.effortLimit
        )
        cuda_cost_decrease = (
            seed_cuda_metrics["external_cost"]
            - final_cuda_metrics["external_cost"]
        )
        pin_cost_decrease = (
            seed_pin_metrics["external_cost"]
            - final_pin_metrics["external_cost"]
        )
        cuda_terminal_decrease = (
            seed_cuda_metrics["terminal_error_m"]
            - final_cuda_metrics["terminal_error_m"]
        )
        pin_terminal_decrease = (
            seed_pin_metrics["terminal_error_m"]
            - final_pin_metrics["terminal_error_m"]
        )
        state_difference = (
            np.asarray(final_cuda[index], dtype=np.float64) - final_pin[index]
        )
        state_l2 = np.linalg.norm(state_difference, axis=1)
        cuda_tool = final_cuda_metrics["dense_tool_path"]
        pin_tool = final_pin_metrics["dense_tool_path"]
        ee_difference = np.linalg.norm(cuda_tool - pin_tool, axis=1)
        finite = bool(
            seed_cuda_metrics["finite"]
            and seed_pin_metrics["finite"]
            and final_cuda_metrics["finite"]
            and final_pin_metrics["finite"]
            and np.isfinite(initial_merit)
            and np.isfinite(final_merit)
            and np.all(np.isfinite(trajectory_delta))
        )
        limits = all(
            metric["joint_violation_rad"] <= 0.0
            and metric["velocity_ratio"] <= 1.0
            and metric["control_ratio"] <= 1.0
            for metric in (
                seed_cuda_metrics,
                seed_pin_metrics,
                final_cuda_metrics,
                final_pin_metrics,
            )
        )
        model_agreement = bool(
            np.max(state_l2) <= 0.002 and np.max(ee_difference) <= 0.001
        )
        seed_fully_feasible = bool(
            seed_cuda_metrics["terminal_error_m"]
            <= protocol.terminal_cuda_tolerance_m
            and seed_pin_metrics["terminal_error_m"]
            <= protocol.terminal_pinocchio_tolerance_m
            and seed_cuda_metrics["minimum_dense_clearance_m"]
            >= protocol.minimum_clearance_margin_m
            and seed_pin_metrics["minimum_dense_clearance_m"]
            >= protocol.minimum_clearance_margin_m
            and finite
            and limits
        )
        merit_pass = bool(
            merit_decrease >= max(1e-3, 0.01 * max(abs(initial_merit), 1.0))
        )
        cuda_cost_pass = bool(
            cuda_cost_decrease
            >= max(
                1e-3,
                0.01 * max(seed_cuda_metrics["external_cost"], 1.0),
            )
        )
        pin_cost_pass = bool(
            pin_cost_decrease
            >= max(
                1e-3,
                0.01 * max(seed_pin_metrics["external_cost"], 1.0),
            )
        )
        terminal_pass = bool(
            cuda_terminal_decrease >= 0.005
            and pin_terminal_decrease >= 0.005
            and final_cuda_metrics["terminal_error_m"]
            <= 0.5 * seed_cuda_metrics["terminal_error_m"]
            and final_pin_metrics["terminal_error_m"]
            <= 0.5 * seed_pin_metrics["terminal_error_m"]
            and final_cuda_metrics["terminal_error_m"]
            <= protocol.terminal_cuda_tolerance_m
            and final_pin_metrics["terminal_error_m"]
            <= protocol.terminal_pinocchio_tolerance_m
        )
        clearance_pass = bool(
            final_cuda_metrics["minimum_dense_clearance_m"]
            >= protocol.minimum_clearance_margin_m
            and final_pin_metrics["minimum_dense_clearance_m"]
            >= protocol.minimum_clearance_margin_m
        )
        cap_free = bool(
            int(stats["pcg_cap_hits"][index]) == 0
            and int(stats["sqp_iterations"][index]) < protocol.max_sqp_iters
        )
        evidence = {
            "seed_not_fully_feasible": not seed_fully_feasible,
            "finite": finite,
            "planned_trajectory_change": trajectory_rms >= 1e-4,
            "final_control_change": control_change >= 1e-4,
            "solver_merit_decrease": merit_pass,
            "cuda_external_cost_decrease": cuda_cost_pass,
            "pinocchio_external_cost_decrease": pin_cost_pass,
            "terminal_reduction_and_final_tolerance": terminal_pass,
            "dense_clearance": clearance_pass,
            "limits": limits,
            "dense_model_agreement": model_agreement,
            "cap_free": cap_free,
        }
        productive = productivity_gate(evidence)
        rows.append(
            {
                "candidate_index": index,
                "productive_certified": productive,
                "productivity_evidence": evidence,
                "seed_fully_feasible_before_sqp": seed_fully_feasible,
                "initial_solver_merit": initial_merit,
                "final_solver_merit": final_merit,
                "solver_merit_decrease": merit_decrease,
                "solver_merit_pass": merit_pass,
                "planned_trajectory_normalized_rms_change": trajectory_rms,
                "effort_normalized_final_control_rms_change": control_change,
                "cuda_external_cost_decrease": cuda_cost_decrease,
                "pinocchio_external_cost_decrease": pin_cost_decrease,
                "cuda_external_cost_pass": cuda_cost_pass,
                "pinocchio_external_cost_pass": pin_cost_pass,
                "cuda_terminal_error_decrease_m": cuda_terminal_decrease,
                "pinocchio_terminal_error_decrease_m": pin_terminal_decrease,
                "terminal_pass": terminal_pass,
                "dense_clearance_pass": clearance_pass,
                "finite": finite,
                "limits_pass": limits,
                "maximum_cuda_pin_state_l2": float(np.max(state_l2)),
                "maximum_cuda_pin_ee_disagreement_m": float(
                    np.max(ee_difference)
                ),
                "dense_model_agreement_pass": model_agreement,
                "pcg_cap_hits": int(stats["pcg_cap_hits"][index]),
                "sqp_iterations": int(stats["sqp_iterations"][index]),
                "max_sqp_reached": bool(
                    int(stats["sqp_iterations"][index])
                    >= protocol.max_sqp_iters
                ),
                "seed_cuda": seed_cuda_metrics,
                "seed_pinocchio": seed_pin_metrics,
                "final_cuda": final_cuda_metrics,
                "final_pinocchio": final_pin_metrics,
                "cuda_external_cost": final_cuda_metrics["external_cost"],
                "pinocchio_external_cost": final_pin_metrics["external_cost"],
                "controls": final_controls[index],
                "cuda_dense_tool_path": cuda_tool,
                "pinocchio_dense_tool_path": pin_tool,
            }
        )
    return rows, {
        "seed_controls": seed_controls,
        "final_controls": final_controls,
        "seed_cuda_dense_rollout": seed_cuda,
        "seed_pinocchio_dense_rollout": seed_pin,
        "final_cuda_dense_rollout": final_cuda,
        "final_pinocchio_dense_rollout": final_pin,
    }


def compare_serial_batch(serial_rows, batch_rows):
    rows = []
    for serial, batch in zip(serial_rows, batch_rows):
        serial_success = bool(serial["productive_certified"])
        batch_success = bool(batch["productive_certified"])
        row = {
            "candidate_index": serial["candidate_index"],
            "same_productive_outcome": serial_success == batch_success,
            "no_serial_success_lost": not serial_success or batch_success,
            "cuda_cost_noninferior": bool(
                batch["cuda_external_cost"]
                <= serial["cuda_external_cost"]
                + max(1e-3, 0.01 * max(serial["cuda_external_cost"], 1.0))
            ),
            "pinocchio_cost_noninferior": bool(
                batch["pinocchio_external_cost"]
                <= serial["pinocchio_external_cost"]
                + max(
                    1e-3,
                    0.01 * max(serial["pinocchio_external_cost"], 1.0),
                )
            ),
            "cuda_terminal_noninferior": bool(
                batch["final_cuda"]["terminal_error_m"]
                <= serial["final_cuda"]["terminal_error_m"] + 0.001
            ),
            "pinocchio_terminal_noninferior": bool(
                batch["final_pinocchio"]["terminal_error_m"]
                <= serial["final_pinocchio"]["terminal_error_m"] + 0.001
            ),
            "same_finite_limits_caps": bool(
                serial["finite"] == batch["finite"]
                and serial["limits_pass"] == batch["limits_pass"]
                and serial["pcg_cap_hits"] == batch["pcg_cap_hits"]
                and serial["max_sqp_reached"] == batch["max_sqp_reached"]
            ),
        }
        row["passes"] = all(value for key, value in row.items() if key != "candidate_index")
        rows.append(row)
    return rows, {
        "candidate_count": len(rows),
        "all_same_productive_outcomes": all(
            row["same_productive_outcome"] for row in rows
        ),
        "all_candidates_noninferior": all(row["passes"] for row in rows),
        "passes": bool(len(rows) == BUDGET and all(row["passes"] for row in rows)),
    }


def all_rows_cap_free(rows):
    return bool(
        rows
        and all(
            row["pcg_cap_hits"] == 0 and not row["max_sqp_reached"]
            for row in rows
        )
    )


def seed_kinematic_screen(model, instance, controls, protocol=DEVELOPMENT_PROTOCOL):
    x0 = np.hstack([instance["q0"], np.zeros(model.nv)])
    states = pinocchio_rollout(model, x0, controls, protocol.dt)
    rows = []
    for candidate in range(BUDGET):
        q, qd = states[candidate, :, : model.nq], states[candidate, :, model.nq :]
        path = tool_path(model, q)
        terminal_error = float(np.linalg.norm(path[-1] - instance["goal_position_m"]))
        clearance = float(
            np.min(
                np.linalg.norm(path[:, :2] - instance["pillar_xy_m"][None, :], axis=1)
            )
            - protocol.pillar_radius_m
        )
        finite = bool(
            np.all(np.isfinite(states[candidate]))
            and np.all(np.isfinite(controls[candidate]))
            and np.all(np.isfinite(path))
        )
        joint_violation = float(
            max(
                np.max(model.lowerPositionLimit - q, initial=0.0),
                np.max(q - model.upperPositionLimit, initial=0.0),
                0.0,
            )
        )
        velocity_ratio = float(np.max(np.abs(qd) / model.velocityLimit))
        control_ratio = float(
            np.max(np.abs(controls[candidate]) / model.effortLimit)
        )
        limits = bool(
            joint_violation <= 0.0
            and velocity_ratio <= 1.0
            and control_ratio <= 1.0
        )
        rows.append(
            {
                "candidate_index": candidate,
                "finite": finite,
                "limits_pass": limits,
                "terminal_error_m": terminal_error,
                "imperfect_terminal_seed": terminal_error >= 0.05,
                "minimum_knot_clearance_m": clearance,
                "joint_violation_rad": joint_violation,
                "velocity_ratio": velocity_ratio,
                "control_ratio": control_ratio,
            }
        )
    return rows, {
        "all_candidates_retained": len(rows) == BUDGET,
        "all_candidates_finite": all(row["finite"] for row in rows),
        "all_candidates_within_limits": all(row["limits_pass"] for row in rows),
        "all_candidates_imperfect": all(
            row["imperfect_terminal_seed"] for row in rows
        ),
        "all_candidate_controls_retained": len(controls) == BUDGET,
        "passes": bool(
            len(rows) == BUDGET
            and all(
                row["finite"]
                and row["limits_pass"]
                and row["imperfect_terminal_seed"]
                for row in rows
            )
        ),
    }


def cpu_dense_pin_seed_screen(
    model, instance, controls, protocol=DEVELOPMENT_PROTOCOL
):
    """Dense CPU-only safety screen used before any CUDA seed diagnostic."""
    x0 = np.hstack([instance["q0"], np.zeros(model.nv)])
    states = dense_pinocchio_rollout(model, x0, controls, protocol.dt)
    rows = []
    for candidate in range(BUDGET):
        metrics = dense_replay_metrics(
            model, instance, states[candidate], controls[candidate], protocol
        )
        rows.append(
            {
                "candidate_index": candidate,
                "finite": metrics["finite"],
                "limits_pass": bool(
                    metrics["joint_violation_rad"] <= 0.0
                    and metrics["velocity_ratio"] <= 1.0
                    and metrics["control_ratio"] <= 1.0
                ),
                "imperfect": metrics["terminal_error_m"] >= 0.05,
                "terminal_error_m": metrics["terminal_error_m"],
                "joint_violation_rad": metrics["joint_violation_rad"],
                "velocity_ratio": metrics["velocity_ratio"],
                "control_ratio": metrics["control_ratio"],
            }
        )
    return rows, {
        "candidate_count": len(rows),
        "all_finite": all(row["finite"] for row in rows),
        "all_limits_pass": all(row["limits_pass"] for row in rows),
        "all_imperfect": all(row["imperfect"] for row in rows),
        "passes": bool(
            len(rows) == BUDGET
            and all(
                row["finite"] and row["limits_pass"] and row["imperfect"]
                for row in rows
            )
        ),
    }
