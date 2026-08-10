"""Cheap imperfect Tiago seeds and replay-derived multimodal certificates.

This module is intentionally separate from the route-informed pillar workflow.
Candidate controls depend only on the robot model, x0, and a method seed.  One
open-loop dynamics pass may populate state knots; there is no feedback,
collision query, pillar query, IK, rejection, or calibration in generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pinocchio as pin

from gato_tiago.multimodal_pillar import (
    Difficulty,
    MODEL_PATH,
    PillarInstance,
    TIAGO_RIGHT_START_CONFIGS,
    dynamics_defect,
    load_model,
    tool_path,
    tool_position,
    trajectory_costs,
    unpack_trajectory,
)


BUDGET = 16
DEVELOPMENT_TASK_SEEDS = (70, 71, 72)
FINAL_TASK_SEEDS = (80, 81, 82)
DEVELOPMENT_METHOD_SEEDS = (3000, 3001, 3002)
FINAL_METHOD_SEEDS = tuple(range(4000, 4020))
CENTRAL_FINAL_TASK_SEED = 81
CONTROL_PATH_RMS_SEPARATION = 1e-4
TOOL_PATH_RMS_SEPARATION_M = 0.003
OPEN_WINDING_NEUTRAL_BAND_RAD = 2.0
CLOSED_WINDING_INTEGER_RESIDUAL = 0.10


@dataclass(frozen=True)
class ColdMultimodalProtocol:
    difficulty: Difficulty
    budget: int = BUDGET
    joint_amplitude_rad: float = 0.03
    terminal_cuda_tolerance_m: float = 0.020
    terminal_pinocchio_tolerance_m: float = 0.022
    minimum_clearance_margin_m: float = 0.003


# Development geometry only. It may be revised using DEVELOPMENT_* seeds, but
# must be frozen before FINAL_* seeds are collected.
DEVELOPMENT_PROTOCOL = ColdMultimodalProtocol(
    difficulty=Difficulty(
        name="cold_multimodal_development",
        lane="cheap_open_loop_multistart",
        knots=128,
        dt=0.006,
        travel_m=0.15,
        radius_m=0.030,
        route_margin_m=0.0,
        asymmetry_m=0.008,
        max_sqp_iters=60,
        max_pcg_iters=500,
        pcg_tol=8e-4,
        q_cost=2.0,
        terminal_cost=260.0,
        qd_cost=0.15,
        u_cost=7.5e-5,
        pillar_cost=800.0,
        terminal_tolerance_m=0.020,
        minimum_clearance_margin_m=0.003,
    )
)

# Assigned only after development-seed tuning is complete. Final harnesses
# must refuse to run while this remains unset.
FINAL_PROTOCOL = None


def generate_cold_instance(task_seed, protocol=DEVELOPMENT_PROTOCOL, model=None):
    """Generate the frozen no-retry development pillar task.

    This isolates B from the older task family's radius/travel jitter.  The
    initializer is deliberately given only the resulting common ``x0``.
    """
    if model is None:
        model = load_model(MODEL_PATH)
    rng = np.random.default_rng(int(task_seed))
    base_q = np.asarray(
        TIAGO_RIGHT_START_CONFIGS["comfortable_high_clearance"],
        dtype=np.float64,
    )
    joint_jitter = rng.uniform(-0.025, 0.025, size=7)
    if model.nq != 7 or base_q.shape != (7,):
        raise ValueError("frozen Tiago task generator requires seven joints")
    q0 = np.clip(
        base_q + joint_jitter,
        model.lowerPositionLimit.astype(np.float64) + 0.08,
        model.upperPositionLimit.astype(np.float64) - 0.08,
    )
    start = tool_position(model, model.createData(), q0)
    planar_angle = float(rng.uniform(-0.16, 0.16))
    direction = np.asarray(
        [np.cos(planar_angle), np.sin(planar_angle)], dtype=np.float64
    )
    goal = start + np.asarray(
        [
            protocol.difficulty.travel_m * direction[0],
            protocol.difficulty.travel_m * direction[1],
            0.0,
        ]
    )
    lateral_sign = float(rng.choice((-1.0, 1.0)))
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    pillar = (
        0.5 * (start[:2] + goal[:2])
        + lateral_sign * protocol.difficulty.asymmetry_m * normal
    )
    radius = float(protocol.difficulty.radius_m)
    endpoint_clearances = (
        np.linalg.norm(
            np.vstack([start[:2], goal[:2]]) - pillar[None, :], axis=1
        )
        - radius
    )
    minimum_endpoint_clearance = float(np.min(endpoint_clearances))
    if minimum_endpoint_clearance < 0.035:
        raise RuntimeError("frozen pillar task violates endpoint-clearance assertion")
    instance = PillarInstance(
        seed=int(task_seed),
        difficulty=protocol.difficulty.name,
        start_q=tuple(float(value) for value in q0),
        start_position=tuple(float(value) for value in start),
        goal_position=tuple(float(value) for value in goal),
        pillar_xy=tuple(float(value) for value in pillar),
        clearance_radius_m=radius,
    )
    return instance, {
        "task_seed": int(task_seed),
        "rng": "numpy.default_rng",
        "base_configuration": "comfortable_high_clearance",
        "joint_jitter_draw_rad": joint_jitter.tolist(),
        "joint_jitter_distribution_rad": [-0.025, 0.025],
        "clipped_start_q": q0.tolist(),
        "planar_angle_draw_rad": planar_angle,
        "planar_angle_distribution_rad": [-0.16, 0.16],
        "lateral_sign_draw": lateral_sign,
        "exact_xy_travel_m": protocol.difficulty.travel_m,
        "exact_goal_dz_m": 0.0,
        "exact_pillar_fraction": 0.5,
        "exact_lateral_offset_m": (
            lateral_sign * protocol.difficulty.asymmetry_m
        ),
        "exact_pillar_radius_m": radius,
        "endpoint_clearances_m": endpoint_clearances.tolist(),
        "minimum_endpoint_clearance_m": minimum_endpoint_clearance,
        "rejection_or_retry_count": 0,
    }


def candidate_labels(budget=BUDGET):
    if budget != BUDGET:
        raise ValueError(f"cold multimodal protocol freezes budget at {BUDGET}")
    labels = ["stationary_inverse_dynamics_hold"]
    for pair in range(7):
        labels.extend([f"analytic_{pair}_plus", f"analytic_{pair}_minus"])
    labels.append("analytic_7_independent")
    return labels


def candidate_directions(model, method_seed):
    """Return the frozen task-neutral stationary/antithetic direction family."""
    rng = np.random.default_rng(int(method_seed))
    draws = rng.normal(size=(8, model.nv))
    draws /= np.linalg.norm(draws, axis=1, keepdims=True)
    directions = [np.zeros(model.nv, dtype=np.float64)]
    for direction in draws[:7]:
        directions.extend([direction, -direction])
    directions.append(draws[7])
    return np.asarray(directions, dtype=np.float64)


def generate_controls(model, q0, method_seed, protocol=DEVELOPMENT_PROTOCOL):
    """Generate a one-shot analytic joint family with inverse-dynamics feedforward.

    The desired motion is a fixed low-frequency ``sin^2`` bump returning to
    ``q0``.  Generation uses no task reference or obstacle geometry and performs
    exactly one inverse-dynamics call per candidate and control knot.
    """
    if protocol.budget != BUDGET:
        raise ValueError(f"cold multimodal protocol freezes budget at {BUDGET}")
    q0 = np.asarray(q0, dtype=np.float64)
    knots = protocol.difficulty.knots
    dt = protocol.difficulty.dt
    duration = (knots - 1) * dt
    amplitude = protocol.joint_amplitude_rad
    directions = candidate_directions(model, method_seed)
    controls = np.empty((BUDGET, knots - 1, model.nv), dtype=np.float64)
    desired_q = np.empty((BUDGET, knots, model.nq), dtype=np.float64)
    desired_qd = np.empty((BUDGET, knots, model.nv), dtype=np.float64)
    desired_qdd = np.empty((BUDGET, knots - 1, model.nv), dtype=np.float64)
    times = np.arange(knots, dtype=np.float64) * dt
    phase = np.pi * times / duration
    position_basis = np.sin(phase) ** 2
    velocity_basis = (np.pi / duration) * np.sin(2.0 * phase)
    acceleration_basis = (
        2.0 * (np.pi / duration) ** 2 * np.cos(2.0 * phase[:-1])
    )
    inverse_dynamics_calls = 0
    for candidate, direction in enumerate(directions):
        desired_q[candidate] = q0 + amplitude * position_basis[:, None] * direction
        desired_qd[candidate] = amplitude * velocity_basis[:, None] * direction
        desired_qdd[candidate] = amplitude * acceleration_basis[:, None] * direction
        data = model.createData()
        for knot in range(knots - 1):
            controls[candidate, knot] = pin.rnea(
                model,
                data,
                desired_q[candidate, knot],
                desired_qd[candidate, knot],
                desired_qdd[candidate, knot],
            )
            inverse_dynamics_calls += 1
    return controls.astype(np.float32), {
        "method_seed": int(method_seed),
        "budget": BUDGET,
        "joint_amplitude_rad": amplitude,
        "analytic_basis": "sin(pi*t/T)^2",
        "analytic_duration_s": duration,
        "candidate_labels": candidate_labels(),
        "candidate_directions": directions.tolist(),
        "inverse_dynamics_calls": inverse_dynamics_calls,
        "expected_inverse_dynamics_calls": BUDGET * (knots - 1),
        "generation_inputs": [
            "robot_model",
            "common_x0",
            "fixed_method_seed",
            "joint_and_effort_limits",
            "frozen_time_grid",
        ],
        "task_reference_or_obstacle_inputs": 0,
        "feedback_or_calibration_iterations": 0,
        "rejection_or_resampling_attempts": 0,
    }


def one_pass_rollout_and_pack(
    sim_forward: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    x0,
    controls,
    protocol=DEVELOPMENT_PROTOCOL,
):
    """Populate seed states with exactly one open-loop step per interval."""
    controls = np.asarray(controls, dtype=np.float32)
    if controls.shape != (
        protocol.budget,
        protocol.difficulty.knots - 1,
        controls.shape[-1],
    ):
        raise ValueError("controls do not match the frozen batch/horizon")
    x0 = np.asarray(x0, dtype=np.float32)
    nx, nu = x0.size, controls.shape[-1]
    width = protocol.difficulty.knots * (nx + nu) - nu
    packed = np.zeros((protocol.budget, width), dtype=np.float32)
    current = np.tile(x0, (protocol.budget, 1))
    for knot in range(protocol.difficulty.knots):
        offset = knot * (nx + nu)
        packed[:, offset : offset + nx] = current
        if knot + 1 < protocol.difficulty.knots:
            packed[:, offset + nx : offset + nx + nu] = controls[:, knot]
            current = np.asarray(
                sim_forward(current, controls[:, knot], protocol.difficulty.dt),
                dtype=np.float32,
            )
    return packed, {
        "open_loop_sim_forward_calls": protocol.difficulty.knots - 1,
        "feedback_or_calibration_iterations": 0,
        "collision_or_pillar_queries": 0,
        "rejection_or_resampling_attempts": 0,
    }


def pinocchio_step(model, data, state, control, dt):
    q = np.asarray(state[: model.nq], dtype=np.float64)
    qd = np.asarray(state[model.nq :], dtype=np.float64)
    qdd = pin.aba(model, data, q, qd, np.asarray(control, dtype=np.float64))
    return np.hstack([q + dt * qd + 0.5 * dt * dt * qdd, qd + dt * qdd])


def cuda_rollout(sim_forward, x0, controls, dt):
    controls = np.asarray(controls, dtype=np.float32)
    states = np.empty(
        (controls.shape[0], controls.shape[1] + 1, np.asarray(x0).size),
        dtype=np.float32,
    )
    current = np.tile(np.asarray(x0, dtype=np.float32), (controls.shape[0], 1))
    for knot in range(states.shape[1]):
        states[:, knot] = current
        if knot + 1 < states.shape[1]:
            current = np.asarray(
                sim_forward(current, controls[:, knot], dt), dtype=np.float32
            )
    return states


def pinocchio_rollout(model, x0, controls, dt):
    controls = np.asarray(controls, dtype=np.float64)
    states = np.empty(
        (controls.shape[0], controls.shape[1] + 1, np.asarray(x0).size),
        dtype=np.float64,
    )
    for candidate in range(controls.shape[0]):
        current = np.asarray(x0, dtype=np.float64).copy()
        data = model.createData()
        for knot in range(states.shape[1]):
            states[candidate, knot] = current
            if knot + 1 < states.shape[1]:
                current = pinocchio_step(
                    model, data, current, controls[candidate, knot], dt
                )
    return states


def dense_state_path(states, dt, samples_per_interval=16):
    states = np.asarray(states, dtype=np.float64)
    nq = states.shape[1] // 2
    rows = []
    for knot in range(states.shape[0] - 1):
        q0, qd0 = states[knot, :nq], states[knot, nq:]
        q1, qd1 = states[knot + 1, :nq], states[knot + 1, nq:]
        for alpha in np.linspace(0.0, 1.0, samples_per_interval, endpoint=False):
            a2, a3 = alpha * alpha, alpha * alpha * alpha
            q = (
                (2.0 * a3 - 3.0 * a2 + 1.0) * q0
                + (a3 - 2.0 * a2 + alpha) * dt * qd0
                + (-2.0 * a3 + 3.0 * a2) * q1
                + (a3 - a2) * dt * qd1
            )
            qd = (
                (6.0 * a2 - 6.0 * alpha) * q0 / dt
                + (3.0 * a2 - 4.0 * alpha + 1.0) * qd0
                + (-6.0 * a2 + 6.0 * alpha) * q1 / dt
                + (3.0 * a2 - 2.0 * alpha) * qd1
            )
            rows.append(np.hstack([q, qd]))
    rows.append(states[-1].copy())
    return np.asarray(rows)


def maximum_cuda_step_pinocchio_defect(model, states, controls, dt):
    maximum = 0.0
    data = model.createData()
    for knot in range(np.asarray(controls).shape[0]):
        predicted = pinocchio_step(
            model, data, states[knot], controls[knot], dt
        )
        maximum = max(maximum, float(np.linalg.norm(predicted - states[knot + 1])))
    return maximum


def rollout_metrics(model, instance, protocol, states, controls):
    difficulty = protocol.difficulty
    states = np.asarray(states, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    q, qd = states[:, : model.nq], states[:, model.nq :]
    dense = dense_state_path(states, difficulty.dt)
    dense_q, dense_qd = dense[:, : model.nq], dense[:, model.nq :]
    dense_tool = tool_path(model, dense_q)
    goal = np.asarray(instance.goal_position, dtype=np.float64)
    pillar = np.asarray(instance.pillar_xy, dtype=np.float64)
    clearance = float(
        np.min(np.linalg.norm(dense_tool[:, :2] - pillar[None, :], axis=1))
        - instance.clearance_radius_m
    )
    violation = max(0.0, protocol.minimum_clearance_margin_m - clearance)
    costs = trajectory_costs(model, instance, difficulty, q, qd, controls)
    terminal_position = dense_tool[-1]
    return {
        "terminal_position_m": terminal_position.tolist(),
        "terminal_error_m": float(np.linalg.norm(terminal_position - goal)),
        "minimum_dense_clearance_m": clearance,
        "maximum_clearance_violation_m": violation,
        "joint_violation_rad": float(
            max(
                np.max(model.lowerPositionLimit - dense_q, initial=0.0),
                np.max(dense_q - model.upperPositionLimit, initial=0.0),
                0.0,
            )
        ),
        "velocity_ratio": float(np.max(np.abs(dense_qd) / model.velocityLimit)),
        "control_ratio": float(np.max(np.abs(controls) / model.effortLimit)),
        "external_cost": costs["nonnegative_task_motion"],
        "external_cost_components": costs,
        "finite": bool(
            np.all(np.isfinite(states))
            and np.all(np.isfinite(controls))
            and np.all(np.isfinite(dense_tool))
            and np.isfinite(costs["nonnegative_task_motion"])
        ),
        "dense_tool_path": dense_tool,
    }


def model_pair_preflight(
    sim_forward,
    model,
    q0,
    dt,
    count=BUDGET,
    seed=20260811,
):
    rng = np.random.default_rng(seed)
    q0 = np.asarray(q0, dtype=np.float64)
    zeros = np.zeros(model.nv, dtype=np.float64)
    x0 = np.hstack([q0, zeros]).astype(np.float32)
    gravity = pin.rnea(model, model.createData(), q0, zeros, zeros)
    benign = []
    for name, control in (("zero", zeros), ("gravity", gravity)):
        cuda = np.asarray(
            sim_forward(x0, control.astype(np.float32), dt), dtype=np.float64
        )[0]
        expected = pinocchio_step(
            model, model.createData(), x0, control, dt
        )
        difference = cuda - expected
        benign.append(
            {
                "name": name,
                "l2_error": float(np.linalg.norm(difference)),
                "max_abs_error": float(np.max(np.abs(difference))),
            }
        )
    q = np.tile(q0, (count, 1)) + rng.uniform(
        -0.05, 0.05, size=(count, model.nq)
    )
    q = np.clip(q, model.lowerPositionLimit + 0.05, model.upperPositionLimit - 0.05)
    qd = rng.uniform(-0.25, 0.25, size=(count, model.nv))
    controls = (
        rng.uniform(-0.20, 0.20, size=(count, model.nv)) * model.effortLimit
    )
    states = np.hstack([q, qd]).astype(np.float32)
    controls = controls.astype(np.float32)
    cuda_next = np.asarray(sim_forward(states, controls, dt), dtype=np.float64)
    pin_next = np.asarray(
        [
            pinocchio_step(
                model, model.createData(), states[index], controls[index], dt
            )
            for index in range(count)
        ]
    )
    difference = cuda_next - pin_next
    norms = np.linalg.norm(difference, axis=1)
    maximum_l2 = max(
        float(np.max(norms)), max(row["l2_error"] for row in benign)
    )
    maximum_abs = max(
        float(np.max(np.abs(difference))),
        max(row["max_abs_error"] for row in benign),
    )
    return {
        "seed": seed,
        "case_count": count,
        "dt": dt,
        "distribution": {
            "q_jitter_rad": [-0.05, 0.05],
            "qd_rad_s": [-0.25, 0.25],
            "control_effort_fraction": [-0.20, 0.20],
        },
        "benign": benign,
        "per_case_l2_error": norms.tolist(),
        "maximum_l2_error": maximum_l2,
        "maximum_abs_error": maximum_abs,
        "passes_1e-3_gate": bool(maximum_l2 <= 1e-3 and maximum_abs <= 1e-3),
    }


def seed_rollout_screen(
    model,
    instance,
    protocol,
    x0,
    controls,
    sim_forward,
    preflight,
    cuda_states=None,
):
    """Evaluate the frozen seed family without invoking SQP.

    This diagnostic retains every candidate.  It checks both independent
    open-loop replay models, but performs no filtering, retry, or proposal
    adaptation based on the results.
    """
    controls = np.asarray(controls, dtype=np.float32)
    if cuda_states is None:
        cuda_states = cuda_rollout(
            sim_forward, x0, controls, protocol.difficulty.dt
        )
    else:
        cuda_states = np.asarray(cuda_states, dtype=np.float32)
    pin_states = pinocchio_rollout(
        model, x0, controls, protocol.difficulty.dt
    )
    rows = []
    for candidate in range(protocol.budget):
        cuda_metrics = rollout_metrics(
            model, instance, protocol, cuda_states[candidate], controls[candidate]
        )
        pin_metrics = rollout_metrics(
            model, instance, protocol, pin_states[candidate], controls[candidate]
        )
        state_difference = (
            np.asarray(cuda_states[candidate], dtype=np.float64)
            - np.asarray(pin_states[candidate], dtype=np.float64)
        )
        state_l2 = np.linalg.norm(state_difference, axis=1)
        cuda_tool = tool_path(model, cuda_states[candidate, :, : model.nq])
        pin_tool = tool_path(model, pin_states[candidate, :, : model.nq])
        ee_disagreement = np.linalg.norm(cuda_tool - pin_tool, axis=1)
        finite = bool(
            cuda_metrics["finite"]
            and pin_metrics["finite"]
            and np.all(np.isfinite(state_difference))
            and np.all(np.isfinite(ee_disagreement))
        )
        limits = bool(
            cuda_metrics["joint_violation_rad"] <= 0.0
            and pin_metrics["joint_violation_rad"] <= 0.0
            and cuda_metrics["velocity_ratio"] <= 1.0
            and pin_metrics["velocity_ratio"] <= 1.0
            and cuda_metrics["control_ratio"] <= 1.0
            and pin_metrics["control_ratio"] <= 1.0
        )
        imperfect = bool(
            cuda_metrics["terminal_error_m"] >= 0.050
            and pin_metrics["terminal_error_m"] >= 0.050
        )
        model_agreement = bool(
            finite
            and np.max(state_l2, initial=0.0) <= 0.002
            and np.max(ee_disagreement, initial=0.0) <= 0.001
        )
        rows.append(
            {
                "candidate_index": candidate,
                "candidate_label": candidate_labels()[candidate],
                "finite_cuda_and_pinocchio": finite,
                "limits_pass": limits,
                "imperfect_terminal_seed": imperfect,
                "not_productively_certified_before_sqp": imperfect,
                "maximum_cuda_pin_state_l2": float(
                    np.max(state_l2, initial=0.0)
                ),
                "maximum_cuda_pin_ee_disagreement_m": float(
                    np.max(ee_disagreement, initial=0.0)
                ),
                "cuda": cuda_metrics,
                "pinocchio": pin_metrics,
                "model_agreement_pass": model_agreement,
                "passes": bool(finite and limits and imperfect and model_agreement),
            }
        )
    nonstationary_controls = [
        np.ascontiguousarray(controls[index]).tobytes()
        for index in range(1, protocol.budget)
    ]
    unique_nonstationary = len(set(nonstationary_controls)) == protocol.budget - 1
    all_rows_pass = all(row["passes"] for row in rows)
    aggregate = {
        "candidate_count": len(rows),
        "retained_candidate_count": len(rows),
        "dropped_or_retried_candidate_count": 0,
        "unique_nonstationary_lanes": unique_nonstationary,
        "all_candidates_finite": all(
            row["finite_cuda_and_pinocchio"] for row in rows
        ),
        "all_candidates_within_limits": all(row["limits_pass"] for row in rows),
        "all_candidates_imperfect": all(
            row["imperfect_terminal_seed"] for row in rows
        ),
        "all_full_path_model_agreement_gates_pass": all(
            row["model_agreement_pass"] for row in rows
        ),
        "random_one_step_preflight_pass": bool(preflight["passes_1e-3_gate"]),
        "all_seed_only_diagnostic_gates_pass": bool(
            len(rows) == BUDGET
            and unique_nonstationary
            and preflight["passes_1e-3_gate"]
            and all_rows_pass
        ),
    }
    return rows, aggregate, {
        "cuda_seed_rollout": cuda_states,
        "pinocchio_seed_rollout": pin_states,
        "seed_controls": controls,
    }


def require_seed_screen_before_sqp(screen_aggregate):
    """Fail closed before constructing or invoking an SQP execution path."""
    if not screen_aggregate.get("all_seed_only_diagnostic_gates_pass", False):
        raise RuntimeError("seed-only CUDA/Pinocchio prerequisite failed; SQP disabled")


def certify_batch(
    model,
    instance: PillarInstance,
    protocol,
    x0,
    seeds,
    outputs,
    stats,
    sim_forward,
):
    difficulty = protocol.difficulty
    seed_unpack = [
        unpack_trajectory(row, model, difficulty.knots) for row in seeds
    ]
    final_unpack = [
        unpack_trajectory(row, model, difficulty.knots) for row in outputs
    ]
    seed_controls = np.asarray([item[2] for item in seed_unpack])
    final_controls = np.asarray([item[2] for item in final_unpack])
    seed_cuda = cuda_rollout(
        sim_forward, x0, seed_controls, difficulty.dt
    )
    final_cuda = cuda_rollout(
        sim_forward, x0, final_controls, difficulty.dt
    )
    seed_pin = pinocchio_rollout(model, x0, seed_controls, difficulty.dt)
    final_pin = pinocchio_rollout(model, x0, final_controls, difficulty.dt)
    rows = []
    for index in range(len(seeds)):
        seed_cuda_metrics = rollout_metrics(
            model, instance, protocol, seed_cuda[index], seed_controls[index]
        )
        final_cuda_metrics = rollout_metrics(
            model, instance, protocol, final_cuda[index], final_controls[index]
        )
        seed_pin_metrics = rollout_metrics(
            model, instance, protocol, seed_pin[index], seed_controls[index]
        )
        final_pin_metrics = rollout_metrics(
            model, instance, protocol, final_pin[index], final_controls[index]
        )
        seed_defect = maximum_cuda_step_pinocchio_defect(
            model, seed_cuda[index], seed_controls[index], difficulty.dt
        )
        final_defect = maximum_cuda_step_pinocchio_defect(
            model, final_cuda[index], final_controls[index], difficulty.dt
        )
        initial_merit = float(stats["initial_merit"][index])
        final_merit = float(stats["final_merit"][index])
        merit_decrease = initial_merit - final_merit
        merit_scale = max(abs(initial_merit), 1.0)
        external_decrease = (
            seed_cuda_metrics["external_cost"]
            - final_cuda_metrics["external_cost"]
        )
        external_scale = max(seed_cuda_metrics["external_cost"], 1.0)
        delta = np.asarray(outputs[index]) - np.asarray(seeds[index])
        cuda_terminal_decrease = (
            seed_cuda_metrics["terminal_error_m"]
            - final_cuda_metrics["terminal_error_m"]
        )
        pin_terminal_decrease = (
            seed_pin_metrics["terminal_error_m"]
            - final_pin_metrics["terminal_error_m"]
        )
        terminal_disagreement = float(
            np.linalg.norm(
                np.asarray(final_cuda_metrics["terminal_position_m"])
                - np.asarray(final_pin_metrics["terminal_position_m"])
            )
        )
        limits_ok = all(
            metric["finite"]
            and metric["joint_violation_rad"] <= 0.0
            and metric["velocity_ratio"] <= 1.0
            and metric["control_ratio"] <= 1.0
            for metric in (final_cuda_metrics, final_pin_metrics)
        )
        seed_already_feasible = bool(
            seed_cuda_metrics["terminal_error_m"]
            <= protocol.terminal_cuda_tolerance_m
            and seed_pin_metrics["terminal_error_m"]
            <= protocol.terminal_pinocchio_tolerance_m
            and seed_cuda_metrics["maximum_clearance_violation_m"] <= 0.0
            and seed_pin_metrics["maximum_clearance_violation_m"] <= 0.0
            and seed_defect <= 1e-3
            and all(
                metric["finite"]
                and metric["joint_violation_rad"] <= 0.0
                and metric["velocity_ratio"] <= 1.0
                and metric["control_ratio"] <= 1.0
                for metric in (seed_cuda_metrics, seed_pin_metrics)
            )
        )
        cuda_clearance_reduction_ok = bool(
            seed_cuda_metrics["maximum_clearance_violation_m"] <= 0.0
            or final_cuda_metrics["maximum_clearance_violation_m"]
            <= 0.5 * seed_cuda_metrics["maximum_clearance_violation_m"]
        )
        pinocchio_clearance_reduction_ok = bool(
            seed_pin_metrics["maximum_clearance_violation_m"] <= 0.0
            or final_pin_metrics["maximum_clearance_violation_m"]
            <= 0.5 * seed_pin_metrics["maximum_clearance_violation_m"]
        )
        productive = bool(
            not seed_already_feasible
            and np.max(np.abs(delta)) >= 1e-4
            and np.linalg.norm(delta) / np.sqrt(delta.size) >= 1e-4
            and np.isfinite(initial_merit)
            and np.isfinite(final_merit)
            and merit_decrease >= max(1e-3, 0.01 * merit_scale)
            and external_decrease >= max(1e-3, 0.01 * external_scale)
            and cuda_terminal_decrease >= 0.005
            and final_cuda_metrics["terminal_error_m"]
            <= 0.5 * seed_cuda_metrics["terminal_error_m"]
            and pin_terminal_decrease >= 0.005
            and final_pin_metrics["terminal_error_m"]
            <= 0.5 * seed_pin_metrics["terminal_error_m"]
            and cuda_clearance_reduction_ok
            and pinocchio_clearance_reduction_ok
            and final_cuda_metrics["terminal_error_m"]
            <= protocol.terminal_cuda_tolerance_m
            and final_pin_metrics["terminal_error_m"]
            <= protocol.terminal_pinocchio_tolerance_m
            and final_cuda_metrics["maximum_clearance_violation_m"] <= 0.0
            and final_pin_metrics["maximum_clearance_violation_m"] <= 0.0
            and terminal_disagreement <= 0.01
            and final_defect <= 1e-3
            and limits_ok
            and int(stats["pcg_cap_hits"][index]) == 0
        )
        q_planned, qd_planned, u_planned = final_unpack[index]
        rows.append(
            {
                "candidate_index": index,
                "productive_certified": productive,
                "seed_already_feasible_before_sqp": seed_already_feasible,
                "initial_solver_merit": initial_merit,
                "final_solver_merit": final_merit,
                "solver_merit_decrease": merit_decrease,
                "solver_merit_reduction_fraction": merit_decrease / merit_scale,
                "external_cost": final_cuda_metrics["external_cost"],
                "external_cost_decrease": external_decrease,
                "external_cost_reduction_fraction": external_decrease / external_scale,
                "max_abs_trajectory_delta": float(np.max(np.abs(delta))),
                "normalized_rms_trajectory_delta": float(
                    np.linalg.norm(delta) / np.sqrt(delta.size)
                ),
                "cuda_terminal_error_decrease_m": cuda_terminal_decrease,
                "pinocchio_terminal_error_decrease_m": pin_terminal_decrease,
                "terminal_cuda_pin_disagreement_m": terminal_disagreement,
                "seed_cuda_one_step_pinocchio_defect": seed_defect,
                "final_cuda_one_step_pinocchio_defect": final_defect,
                "planned_pinocchio_defect_report_only": dynamics_defect(
                    model,
                    q_planned,
                    qd_planned,
                    u_planned,
                    difficulty.dt,
                ),
                "cuda_clearance_violation_reduction_pass": (
                    cuda_clearance_reduction_ok
                ),
                "pinocchio_clearance_violation_reduction_pass": (
                    pinocchio_clearance_reduction_ok
                ),
                "seed_cuda": seed_cuda_metrics,
                "final_cuda": final_cuda_metrics,
                "seed_pinocchio": seed_pin_metrics,
                "final_pinocchio": final_pin_metrics,
                "pcg_cap_hits": int(stats["pcg_cap_hits"][index]),
                "controls": final_controls[index],
                "cuda_tool_path": final_cuda_metrics["dense_tool_path"],
                "pinocchio_tool_path": final_pin_metrics["dense_tool_path"],
            }
        )
    return rows, {
        "seed_controls": seed_controls,
        "optimized_controls": final_controls,
        "seed_cuda_rollout": seed_cuda,
        "optimized_cuda_rollout": final_cuda,
        "seed_pinocchio_rollout": seed_pin,
        "optimized_pinocchio_rollout": final_pin,
    }


def signed_open_winding(path, pillar_xy):
    delta = np.asarray(path, dtype=np.float64)[:, :2] - np.asarray(
        pillar_xy, dtype=np.float64
    )
    angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    angle = float(angles[-1] - angles[0])
    if angle > OPEN_WINDING_NEUTRAL_BAND_RAD:
        return "counterclockwise", angle
    if angle < -OPEN_WINDING_NEUTRAL_BAND_RAD:
        return "clockwise", angle
    return "neutral", angle


def pairwise_closed_loop_winding(path_a, path_b, pillar_xy):
    """Integer winding of path A joined to reverse(path B) about the pillar."""
    a = np.asarray(path_a, dtype=np.float64)[:, :2]
    b = np.asarray(path_b, dtype=np.float64)[:, :2]
    loop = np.vstack([a, b[-2::-1]])
    vectors = loop - np.asarray(pillar_xy, dtype=np.float64)[None, :]
    following = np.roll(vectors, -1, axis=0)
    increments = np.arctan2(
        vectors[:, 0] * following[:, 1]
        - vectors[:, 1] * following[:, 0],
        np.sum(vectors * following, axis=1),
    )
    winding = float(np.sum(increments) / (2.0 * np.pi))
    integer = int(np.rint(winding))
    return {
        "winding": winding,
        "nearest_integer": integer,
        "integer_residual": float(abs(winding - integer)),
        "valid_integer": bool(
            abs(winding - integer) <= CLOSED_WINDING_INTEGER_RESIDUAL
        ),
    }


def normalized_control_rms(control_a, control_b, effort_limit):
    difference = (
        np.asarray(control_a, dtype=np.float64)
        - np.asarray(control_b, dtype=np.float64)
    ) / np.asarray(effort_limit, dtype=np.float64)[None, :]
    return float(np.sqrt(np.mean(difference * difference)))


def tool_path_rms(path_a, path_b):
    difference = np.asarray(path_a, dtype=np.float64) - np.asarray(
        path_b, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def certify_mode_support(
    rows,
    pillar_xy,
    effort_limit,
    min_support=2,
    path_key="cuda_tool_path",
):
    """Require replay labels, independent candidates, and integer topology."""
    candidates = [row for row in rows if row.get("productive_certified", False)]
    for row in candidates:
        label, angle = signed_open_winding(row[path_key], pillar_xy)
        row["replay_mode"] = label
        row["open_winding_angle_rad"] = angle
    supported = {}
    for mode in ("clockwise", "counterclockwise"):
        unique = []
        for row in sorted(
            (item for item in candidates if item["replay_mode"] == mode),
            key=lambda item: item["external_cost"],
        ):
            duplicate = any(
                normalized_control_rms(
                    row["controls"], other["controls"], effort_limit
                )
                < CONTROL_PATH_RMS_SEPARATION
                or tool_path_rms(row[path_key], other[path_key])
                < TOOL_PATH_RMS_SEPARATION_M
                for other in unique
            )
            if not duplicate:
                unique.append(row)
        supported[mode] = unique

    pairwise = []
    for clockwise in supported["clockwise"]:
        for counterclockwise in supported["counterclockwise"]:
            certificate = pairwise_closed_loop_winding(
                clockwise[path_key],
                counterclockwise[path_key],
                pillar_xy,
            )
            pairwise.append(
                {
                    "clockwise_candidate": clockwise["candidate_index"],
                    "counterclockwise_candidate": counterclockwise[
                        "candidate_index"
                    ],
                    **certificate,
                }
            )
    same_mode_integer_zero = True
    same_mode_pairwise = []
    for mode, mode_rows in supported.items():
        for left_index, left in enumerate(mode_rows):
            for right in mode_rows[left_index + 1 :]:
                certificate = pairwise_closed_loop_winding(
                    left[path_key], right[path_key], pillar_xy
                )
                same_mode_pairwise.append(
                    {
                        "mode": mode,
                        "left_candidate": left["candidate_index"],
                        "right_candidate": right["candidate_index"],
                        **certificate,
                    }
                )
                same_mode_integer_zero &= bool(
                    certificate["valid_integer"]
                    and certificate["nearest_integer"] == 0
                )
    cross_mode_nonzero = bool(pairwise) and all(
        row["valid_integer"] and abs(row["nearest_integer"]) >= 1
        for row in pairwise
    )
    support = {mode: len(values) for mode, values in supported.items()}
    two_mode = bool(
        all(value >= min_support for value in support.values())
        and cross_mode_nonzero
        and same_mode_integer_zero
    )
    best = {
        mode: (
            min(values, key=lambda row: row["external_cost"]) if values else None
        )
        for mode, values in supported.items()
    }
    return {
        "support": support,
        "minimum_support_per_mode": min_support,
        "path_key": path_key,
        "two_mode_support": two_mode,
        "cross_mode_pairwise_closed_loop_winding": pairwise,
        "same_mode_pairwise_closed_loop_winding": same_mode_pairwise,
        "all_cross_mode_windings_nonzero_integer": cross_mode_nonzero,
        "all_same_mode_windings_zero_integer": same_mode_integer_zero,
        "best_external_cost_by_mode": {
            mode: None if row is None else row["external_cost"]
            for mode, row in best.items()
        },
        "best_candidate_by_mode": {
            mode: None if row is None else row["candidate_index"]
            for mode, row in best.items()
        },
        "supported_candidate_indices_by_mode": {
            mode: [row["candidate_index"] for row in values]
            for mode, values in supported.items()
        },
        "candidate_modes": {
            str(row["candidate_index"]): row["replay_mode"] for row in candidates
        },
    }


def topology_signature(certificate):
    cross_pairs = tuple(
        sorted(
            (
                row["clockwise_candidate"],
                row["counterclockwise_candidate"],
                row["nearest_integer"],
                row["valid_integer"],
            )
            for row in certificate["cross_mode_pairwise_closed_loop_winding"]
        )
    )
    same_pairs = tuple(
        sorted(
            (
                row["mode"],
                row["left_candidate"],
                row["right_candidate"],
                row["nearest_integer"],
                row["valid_integer"],
            )
            for row in certificate["same_mode_pairwise_closed_loop_winding"]
        )
    )
    return {
        "candidate_modes": certificate["candidate_modes"],
        "support": certificate["support"],
        "supported_candidate_indices_by_mode": certificate[
            "supported_candidate_indices_by_mode"
        ],
        "two_mode_support": certificate["two_mode_support"],
        "cross_pair_classes": cross_pairs,
        "same_pair_classes": same_pairs,
        "all_cross_mode_windings_nonzero_integer": certificate[
            "all_cross_mode_windings_nonzero_integer"
        ],
        "all_same_mode_windings_zero_integer": certificate[
            "all_same_mode_windings_zero_integer"
        ],
    }


def mode_certificates_agree(left, right):
    return topology_signature(left) == topology_signature(right)


def bootstrap_median_interval(values, seed=20260811, resamples=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    medians = np.median(
        values[rng.integers(0, values.size, size=(resamples, values.size))], axis=1
    )
    return {
        "confidence": 0.95,
        "resamples": resamples,
        "lower": float(np.percentile(medians, 2.5)),
        "upper": float(np.percentile(medians, 97.5)),
    }


def paired_mode_cost_summary(method_rows):
    paired = [row for row in method_rows if row.get("two_mode_support", False)]
    costs = {
        mode: np.asarray(
            [row["best_external_cost_by_mode"][mode] for row in paired],
            dtype=np.float64,
        )
        for mode in ("clockwise", "counterclockwise")
    }
    if not paired:
        return {
            "declared_method_seed_count": len(method_rows),
            "paired_support_count": 0,
            "passes": False,
        }
    medians = {mode: float(np.median(values)) for mode, values in costs.items()}
    suboptimal_mode = max(medians, key=medians.get)
    better_mode = min(medians, key=medians.get)
    gap_fraction = (
        costs[suboptimal_mode] - costs[better_mode]
    ) / np.maximum(costs[better_mode], 1e-12)
    interval = bootstrap_median_interval(gap_fraction)
    worse_fraction = float(np.mean(gap_fraction > 0.0))
    passes = bool(
        len(method_rows) >= 20
        and len(paired) >= 15
        and np.median(gap_fraction) >= 0.05
        and interval["lower"] > 0.01
        and worse_fraction >= 0.75
    )
    return {
        "declared_method_seed_count": len(method_rows),
        "paired_support_count": len(paired),
        "paired_support_fraction": len(paired) / len(method_rows),
        "median_external_cost_by_mode": medians,
        "suboptimal_mode": suboptimal_mode,
        "better_mode": better_mode,
        "paired_suboptimal_gap_fraction": gap_fraction.tolist(),
        "paired_median_suboptimal_gap_fraction": float(np.median(gap_fraction)),
        "paired_bootstrap_95_ci_median_gap": interval,
        "suboptimal_mode_worse_fraction": worse_fraction,
        "passes": passes,
    }
