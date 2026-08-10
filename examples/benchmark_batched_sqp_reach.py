#!/usr/bin/env python3
"""Benchmark productive batched SQP from cheap imperfect IIWA multistarts.

All lanes in a case share x0, reference, objective, constraints, and stopping
rules. Candidate construction uses only a cold repeated-state/zero-control seed
and bounded smooth random state/control perturbations. It performs no IK,
rollout, feedback calibration, collision query, or optimization.

The executable artifact is each optimized control sequence replayed from the
common x0 in CUDA and independently in Pinocchio. Solver-planned state defects
are retained as telemetry and are never described as dynamically consistent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pinocchio as pin


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from bsqp.interface import BSQP


MODEL_PATH = REPO_ROOT / "examples/iiwa_description/iiwa14.urdf"
DEFAULT_OUTPUT = REPO_ROOT / "example_artifacts/iiwa_batched_sqp_reach"
KNOTS = 8
DT = 0.05


@dataclass(frozen=True)
class Lane:
    name: str
    goal_offset: tuple[float, float, float]


LANES = {
    "near": Lane("near", (0.04, 0.02, 0.02)),
    "far": Lane("far", (0.07, 0.04, 0.03)),
}


def solver_parameters():
    return {
        "max_sqp_iters": 60,
        "kkt_tol": 1e-3,
        "max_pcg_iters": 500,
        "pcg_tol": 1e-4,
        "solve_ratio": 1.0,
        "mu": 1000.0,
        "q_cost": 2.0,
        "qd_cost": 1e-2,
        "u_cost": 1e-4,
        "N_cost": 100.0,
        "ee_orient_cost": 0.0,
        "ee_orient_N_cost": 0.0,
        "q_lim_cost": 0.01,
        "vel_lim_cost": 0.0,
        "ctrl_lim_cost": 0.0,
        "rho": 1.0,
    }


def sha256_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def tool_position(model, data, q):
    pin.forwardKinematics(model, data, np.asarray(q, dtype=np.float64))
    return data.oMi[model.njoints - 1].translation.copy()


def generate_instance(model, lane, instance_seed):
    rng = np.random.default_rng(int(instance_seed))
    q0 = rng.uniform(-0.10, 0.10, size=model.nq)
    q0 = np.clip(q0, model.lowerPositionLimit + 0.25, model.upperPositionLimit - 0.25)
    start = tool_position(model, model.createData(), q0)
    goal = start + np.asarray(lane.goal_offset, dtype=np.float64)
    return q0.astype(np.float32), start.astype(np.float32), goal.astype(np.float32)


def generate_seeds(model, q0, method_seed, budget):
    if budget not in (1, 2, 4, 8, 16, 32):
        raise ValueError("budget must be a supported power of two up to 32")
    started = time.perf_counter()
    nx, nu = model.nq + model.nv, model.nv
    width = KNOTS * (nx + nu) - nu
    seeds = np.zeros((budget, width), dtype=np.float32)
    progress = np.linspace(0.0, 1.0, KNOTS, dtype=np.float64)
    envelope = np.sin(np.pi * progress) ** 2
    rng = np.random.default_rng(int(method_seed))
    effort = model.effortLimit.astype(np.float64)
    for candidate in range(budget):
        q_direction = rng.normal(size=model.nq)
        q_direction /= max(float(np.linalg.norm(q_direction)), 1e-12)
        u_direction = rng.normal(size=model.nv)
        u_direction /= max(float(np.linalg.norm(u_direction)), 1e-12)
        state_amplitude = 0.0 if candidate == 0 else 0.020
        control_fraction = 0.0 if candidate == 0 else 0.010
        q_path = q0[None, :].astype(np.float64) + (
            state_amplitude * envelope[:, None] * q_direction[None, :]
        )
        q_path = np.clip(
            q_path,
            model.lowerPositionLimit.astype(np.float64) + 0.05,
            model.upperPositionLimit.astype(np.float64) - 0.05,
        )
        q_path[0] = q0
        controls = (
            control_fraction
            * effort[None, :]
            * np.sin(np.pi * progress[:-1])[:, None]
            * u_direction[None, :]
        )
        for knot in range(KNOTS):
            offset = knot * (nx + nu)
            seeds[candidate, offset : offset + model.nq] = q_path[knot]
            if knot + 1 < KNOTS:
                seeds[candidate, offset + nx : offset + nx + nu] = controls[knot]
    return seeds, {
        "state_amplitude_rad": 0.020,
        "control_effort_fraction": 0.010,
        "cold_candidate_index": 0,
        "generation_wall_ms": 1e3 * (time.perf_counter() - started),
    }


def unpack(trajectory, model):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    nx, nu = model.nq + model.nv, model.nv
    q = np.empty((KNOTS, model.nq), dtype=np.float64)
    qd = np.empty((KNOTS, model.nv), dtype=np.float64)
    controls = np.empty((KNOTS - 1, model.nv), dtype=np.float64)
    for knot in range(KNOTS):
        offset = knot * (nx + nu)
        q[knot] = trajectory[offset : offset + model.nq]
        qd[knot] = trajectory[offset + model.nq : offset + nx]
        if knot + 1 < KNOTS:
            controls[knot] = trajectory[offset + nx : offset + nx + nu]
    return q, qd, controls


def controls_from_packed(trajectories, model):
    return np.asarray([unpack(row, model)[2] for row in trajectories], dtype=np.float32)


def pin_step(model, data, state, control):
    q = np.asarray(state[: model.nq], dtype=np.float64)
    qd = np.asarray(state[model.nq :], dtype=np.float64)
    qdd = pin.aba(model, data, q, qd, np.asarray(control, dtype=np.float64))
    return np.hstack([q + DT * qd + 0.5 * DT * DT * qdd, qd + DT * qdd])


def model_pair_preflight(solver, model, seed=20260810, count=8):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-0.7, 0.7, size=(count, model.nq)).astype(np.float32)
    qd = rng.uniform(-0.25, 0.25, size=(count, model.nv)).astype(np.float32)
    controls = (
        rng.uniform(-0.20, 0.20, size=(count, model.nv)) * model.effortLimit
    ).astype(np.float32)
    states = np.hstack([q, qd]).astype(np.float32)
    cuda_next = np.asarray(solver.sim_forward(states, controls, DT), dtype=np.float64)
    pin_next = np.asarray(
        [pin_step(model, model.createData(), states[i], controls[i]) for i in range(count)]
    )
    difference = cuda_next - pin_next
    result = {
        "seed": int(seed),
        "case_count": int(count),
        "q_range_rad": [-0.7, 0.7],
        "qd_range_rad_s": [-0.25, 0.25],
        "signed_control_effort_fraction": 0.20,
        "per_case_l2_error": np.linalg.norm(difference, axis=1).tolist(),
        "per_case_max_abs_error": np.max(np.abs(difference), axis=1).tolist(),
        "maximum_l2_error": float(np.max(np.linalg.norm(difference, axis=1))),
        "maximum_abs_error": float(np.max(np.abs(difference))),
    }
    result["passes_1e-3_gate"] = result["maximum_l2_error"] <= 1e-3
    return result


def cuda_rollout(solver, x0, controls):
    controls = np.asarray(controls, dtype=np.float32)
    batch = controls.shape[0]
    states = np.empty((batch, KNOTS, x0.size), dtype=np.float32)
    current = np.tile(np.asarray(x0, dtype=np.float32), (batch, 1))
    for knot in range(KNOTS):
        states[:, knot] = current
        if knot + 1 < KNOTS:
            current = np.asarray(
                solver.sim_forward(current, controls[:, knot], DT), dtype=np.float32
            )
    return states


def pinocchio_rollout(model, x0, controls):
    controls = np.asarray(controls, dtype=np.float64)
    states = np.empty((controls.shape[0], KNOTS, x0.size), dtype=np.float64)
    for candidate in range(controls.shape[0]):
        data = model.createData()
        current = np.asarray(x0, dtype=np.float64).copy()
        for knot in range(KNOTS):
            states[candidate, knot] = current
            if knot + 1 < KNOTS:
                current = pin_step(model, data, current, controls[candidate, knot])
    return states


def independent_step_defect(model, states, controls):
    defects = np.zeros(states.shape[0], dtype=np.float64)
    for candidate in range(states.shape[0]):
        data = model.createData()
        for knot in range(KNOTS - 1):
            predicted = pin_step(
                model, data, states[candidate, knot], controls[candidate, knot]
            )
            defects[candidate] = max(
                defects[candidate],
                float(np.linalg.norm(predicted - states[candidate, knot + 1])),
            )
    return defects


def planned_defect(model, trajectories):
    values = []
    for trajectory in trajectories:
        q, qd, controls = unpack(trajectory, model)
        states = np.hstack([q, qd])[None, :, :]
        values.append(independent_step_defect(model, states, controls[None, :, :])[0])
    return np.asarray(values)


def dense_q(states, samples=8):
    nq = states.shape[1] // 2
    result = []
    for knot in range(KNOTS - 1):
        q0, qd0 = states[knot, :nq], states[knot, nq:]
        q1, qd1 = states[knot + 1, :nq], states[knot + 1, nq:]
        for alpha in np.linspace(0.0, 1.0, samples, endpoint=False):
            a2, a3 = alpha * alpha, alpha * alpha * alpha
            result.append(
                (2 * a3 - 3 * a2 + 1) * q0
                + (a3 - 2 * a2 + alpha) * DT * qd0
                + (-2 * a3 + 3 * a2) * q1
                + (a3 - a2) * DT * qd1
            )
    result.append(states[-1, :nq])
    return np.asarray(result)


def rollout_metrics(model, goal, states, controls):
    params = solver_parameters()
    data = model.createData()
    positions = np.asarray(
        [tool_position(model, data, state[: model.nq]) for state in states]
    )
    target_cost = velocity_cost = control_cost = 0.0
    for knot in range(KNOTS):
        error = positions[knot] - goal
        weight = params["N_cost"] if knot + 1 == KNOTS else params["q_cost"]
        target_cost += 0.5 * weight * float(error @ error)
        qd = states[knot, model.nq :]
        velocity_cost += 0.5 * params["qd_cost"] * float(qd @ qd)
        if knot + 1 < KNOTS:
            control_cost += 0.5 * params["u_cost"] * float(
                controls[knot] @ controls[knot]
            )
    dense = dense_q(states)
    return {
        "terminal_position_m": positions[-1].tolist(),
        "terminal_error_m": float(np.linalg.norm(positions[-1] - goal)),
        "target_cost": float(target_cost),
        "velocity_cost": float(velocity_cost),
        "control_cost": float(control_cost),
        "nonnegative_task_motion_cost": float(
            target_cost + velocity_cost + control_cost
        ),
        "joint_violation_rad": float(
            max(
                np.max(model.lowerPositionLimit - dense, initial=0.0),
                np.max(dense - model.upperPositionLimit, initial=0.0),
                0.0,
            )
        ),
        "velocity_ratio": float(
            np.max(np.abs(states[:, model.nq :]) / model.velocityLimit)
        ),
        "control_ratio": float(np.max(np.abs(controls) / model.effortLimit)),
        "finite": bool(np.all(np.isfinite(states)) and np.all(np.isfinite(controls))),
    }


def solver_merit_improvement(initial_merit, final_merit):
    initial_merit = float(initial_merit)
    final_merit = float(final_merit)
    decrease = initial_merit - final_merit
    scale = max(abs(initial_merit), 1.0)
    required = max(1e-3, 0.01 * scale)
    return {
        "initial_solver_merit": initial_merit,
        "final_solver_merit": final_merit,
        "solver_merit_decrease": float(decrease),
        "solver_merit_required_decrease": float(required),
        "solver_merit_reduction_fraction": float(decrease / scale),
        "passes_solver_merit_decrease_gate": bool(
            np.isfinite(initial_merit)
            and np.isfinite(final_merit)
            and decrease >= required
        ),
    }


def make_solver(batch_size):
    return BSQP(
        model_path=str(MODEL_PATH),
        batch_size=batch_size,
        N=KNOTS,
        dt=DT,
        plant_type="iiwa14",
        **solver_parameters(),
    )


def stats_snapshot(solver):
    stats = solver.get_stats()
    pcg = np.asarray(stats.get("pcg_iters", []), dtype=np.int32)
    return {
        "initial_merit": np.asarray(stats["initial_merit"], dtype=np.float64),
        "final_merit": np.asarray(stats["final_merit"], dtype=np.float64),
        "sqp_iterations": np.asarray(stats["sqp_iters"], dtype=np.int32),
        "total_pcg_iterations": np.sum(pcg, axis=0, dtype=np.int64),
        "pcg_cap_hits": np.sum(
            pcg >= solver_parameters()["max_pcg_iters"], axis=0, dtype=np.int64
        ),
    }


def solve_batched(solver, x0_batch, references, seeds):
    workflow_started = time.perf_counter()
    solver.reset_dual()
    solver.reset_rho()
    seed_input = seeds.copy()
    started = time.perf_counter()
    trajectories, reported_us = solver.solve(x0_batch, references, seed_input)
    solve_wall_ms = 1e3 * (time.perf_counter() - started)
    snapshot = stats_snapshot(solver)
    return (
        trajectories,
        snapshot,
        solve_wall_ms,
        float(reported_us) / 1e3,
        1e3 * (time.perf_counter() - workflow_started),
    )


def solve_independent(solver, x0_batch, references, seeds):
    outputs, snapshots = [], []
    reported_ms = 0.0
    solve_wall_ms = 0.0
    workflow_started = time.perf_counter()
    for candidate in range(seeds.shape[0]):
        solver.reset_dual()
        solver.reset_rho()
        candidate_x0 = x0_batch[candidate : candidate + 1]
        candidate_reference = references[candidate : candidate + 1]
        candidate_seed = seeds[candidate : candidate + 1].copy()
        started = time.perf_counter()
        output, reported_us = solver.solve(
            candidate_x0,
            candidate_reference,
            candidate_seed,
        )
        solve_wall_ms += 1e3 * (time.perf_counter() - started)
        outputs.append(output[0])
        snapshots.append(stats_snapshot(solver))
        reported_ms += float(reported_us) / 1e3
    merged = {
        key: np.concatenate([snapshot[key] for snapshot in snapshots])
        for key in snapshots[0]
    }
    return (
        np.asarray(outputs, dtype=np.float32),
        merged,
        solve_wall_ms,
        reported_ms,
        1e3 * (time.perf_counter() - workflow_started),
    )


def certify(model, goal, x0, seeds, outputs, stats, rollout_solver):
    started = time.perf_counter()
    seed_controls = controls_from_packed(seeds, model)
    final_controls = controls_from_packed(outputs, model)
    seed_cuda = cuda_rollout(rollout_solver, x0, seed_controls)
    final_cuda = cuda_rollout(rollout_solver, x0, final_controls)
    seed_pin = pinocchio_rollout(model, x0, seed_controls)
    final_pin = pinocchio_rollout(model, x0, final_controls)
    seed_cuda_defect = independent_step_defect(model, seed_cuda, seed_controls)
    final_cuda_defect = independent_step_defect(model, final_cuda, final_controls)
    solver_planned_defect = planned_defect(model, outputs)
    rows = []
    for candidate in range(seeds.shape[0]):
        seed_cuda_metrics = rollout_metrics(
            model, goal, seed_cuda[candidate], seed_controls[candidate]
        )
        final_cuda_metrics = rollout_metrics(
            model, goal, final_cuda[candidate], final_controls[candidate]
        )
        seed_pin_metrics = rollout_metrics(
            model, goal, seed_pin[candidate], seed_controls[candidate]
        )
        final_pin_metrics = rollout_metrics(
            model, goal, final_pin[candidate], final_controls[candidate]
        )
        delta = outputs[candidate] - seeds[candidate]
        external_cost_decrease = (
            seed_cuda_metrics["nonnegative_task_motion_cost"]
            - final_cuda_metrics["nonnegative_task_motion_cost"]
        )
        merit = solver_merit_improvement(
            stats["initial_merit"][candidate], stats["final_merit"][candidate]
        )
        cuda_terminal_decrease = (
            seed_cuda_metrics["terminal_error_m"]
            - final_cuda_metrics["terminal_error_m"]
        )
        pin_terminal_decrease = (
            seed_pin_metrics["terminal_error_m"] - final_pin_metrics["terminal_error_m"]
        )
        terminal_disagreement = float(
            np.linalg.norm(
                np.asarray(final_cuda_metrics["terminal_position_m"])
                - np.asarray(final_pin_metrics["terminal_position_m"])
            )
        )
        limits_ok = all(
            metrics["finite"]
            and metrics["joint_violation_rad"] <= 0.0
            and metrics["velocity_ratio"] <= 1.0
            and metrics["control_ratio"] <= 1.0
            for metrics in (final_cuda_metrics, final_pin_metrics)
        )
        improved = bool(
            np.max(np.abs(delta)) >= 1e-4
            and np.linalg.norm(delta) / np.sqrt(delta.size) >= 1e-4
            and merit["passes_solver_merit_decrease_gate"]
            and external_cost_decrease >= 1e-3
            and external_cost_decrease
            >= 0.01
            * max(seed_cuda_metrics["nonnegative_task_motion_cost"], 1.0)
            and cuda_terminal_decrease >= 0.005
            and final_cuda_metrics["terminal_error_m"]
            <= 0.5 * seed_cuda_metrics["terminal_error_m"]
            and pin_terminal_decrease >= 0.005
            and final_pin_metrics["terminal_error_m"]
            <= 0.5 * seed_pin_metrics["terminal_error_m"]
            and final_cuda_metrics["terminal_error_m"] <= 0.10
            and final_pin_metrics["terminal_error_m"] <= 0.11
            and terminal_disagreement <= 0.01
            and final_cuda_defect[candidate] <= 1e-3
            and limits_ok
            and stats["pcg_cap_hits"][candidate] == 0
        )
        rows.append(
            {
                "candidate_index": candidate,
                "seed_cuda": seed_cuda_metrics,
                "final_cuda": final_cuda_metrics,
                "seed_pinocchio": seed_pin_metrics,
                "final_pinocchio": final_pin_metrics,
                "external_cost_decrease": float(external_cost_decrease),
                "cuda_terminal_error_decrease_m": float(cuda_terminal_decrease),
                "pin_terminal_error_decrease_m": float(pin_terminal_decrease),
                "terminal_error_cuda_pin_disagreement_m": float(terminal_disagreement),
                "seed_cuda_one_step_pinocchio_defect": float(
                    seed_cuda_defect[candidate]
                ),
                "final_cuda_one_step_pinocchio_defect": float(
                    final_cuda_defect[candidate]
                ),
                "solver_planned_pinocchio_defect_report_only": float(
                    solver_planned_defect[candidate]
                ),
                **merit,
                "max_abs_trajectory_delta": float(np.max(np.abs(delta))),
                "normalized_rms_trajectory_delta": float(
                    np.linalg.norm(delta) / np.sqrt(delta.size)
                ),
                "sqp_iterations": int(stats["sqp_iterations"][candidate]),
                "total_pcg_iterations": int(stats["total_pcg_iterations"][candidate]),
                "pcg_cap_hits": int(stats["pcg_cap_hits"][candidate]),
                "optimizer_improved_success": improved,
            }
        )
    arrays = {
        "seed_controls": seed_controls,
        "optimized_controls": final_controls,
        "seed_cuda_rollout": seed_cuda,
        "optimized_cuda_rollout": final_cuda,
        "seed_pinocchio_rollout": seed_pin,
        "optimized_pinocchio_rollout": final_pin,
    }
    return rows, arrays, 1e3 * (time.perf_counter() - started)


def timing_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "samples_ms": values.tolist(),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95.0)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
    }


def bootstrap_interval(values, seed=0, resamples=10000):
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


def merit_repeatability(raw, method):
    result = {}
    for name in ("initial_merit", "final_merit"):
        values = np.asarray(
            [row[f"{method}_{name}"] for row in raw], dtype=np.float64
        )
        span = np.max(values, axis=0) - np.min(values, axis=0)
        scale = np.maximum(np.max(np.abs(values), axis=0), 1.0)
        result[name] = {
            "per_candidate_absolute_span": span.tolist(),
            "per_candidate_relative_span": (span / scale).tolist(),
            "maximum_absolute_span": float(np.max(span)),
            "maximum_relative_span": float(np.max(span / scale)),
        }
    return result


def multistart_quality_vs_cold(rows, cold_candidate_index=0):
    cold = rows[cold_candidate_index]
    successes = [row for row in rows if row["optimizer_improved_success"]]
    best = min(
        successes,
        key=lambda row: row["final_cuda"]["nonnegative_task_motion_cost"],
        default=None,
    )
    result = {
        "cold_candidate_index": int(cold_candidate_index),
        "cold_optimizer_improved_success": bool(
            cold["optimizer_improved_success"]
        ),
        "cold_final_cuda_terminal_error_m": float(
            cold["final_cuda"]["terminal_error_m"]
        ),
        "cold_final_cuda_nonnegative_task_motion_cost": float(
            cold["final_cuda"]["nonnegative_task_motion_cost"]
        ),
        "best_successful_candidate_index": None,
        "best_is_non_cold": False,
        "best_final_cuda_terminal_error_m": None,
        "best_final_cuda_nonnegative_task_motion_cost": None,
        "terminal_error_reduction_vs_cold_m": None,
        "terminal_error_reduction_vs_cold_fraction": None,
        "external_cost_reduction_vs_cold": None,
        "external_cost_reduction_vs_cold_fraction": None,
    }
    if best is None:
        return result
    cold_error = cold["final_cuda"]["terminal_error_m"]
    cold_cost = cold["final_cuda"]["nonnegative_task_motion_cost"]
    best_error = best["final_cuda"]["terminal_error_m"]
    best_cost = best["final_cuda"]["nonnegative_task_motion_cost"]
    result.update(
        {
            "best_successful_candidate_index": int(best["candidate_index"]),
            "best_is_non_cold": bool(
                best["candidate_index"] != cold_candidate_index
            ),
            "best_final_cuda_terminal_error_m": float(best_error),
            "best_final_cuda_nonnegative_task_motion_cost": float(best_cost),
            "terminal_error_reduction_vs_cold_m": float(cold_error - best_error),
            "terminal_error_reduction_vs_cold_fraction": float(
                (cold_error - best_error) / max(cold_error, 1e-12)
            ),
            "external_cost_reduction_vs_cold": float(cold_cost - best_cost),
            "external_cost_reduction_vs_cold_fraction": float(
                (cold_cost - best_cost) / max(cold_cost, 1e-12)
            ),
        }
    )
    return result


def final_rollouts_finite_and_within_limits(row):
    metrics = (row["final_cuda"], row["final_pinocchio"])
    return bool(
        all(
            value["finite"]
            and np.isfinite(value["terminal_error_m"])
            and np.isfinite(value["nonnegative_task_motion_cost"])
            and value["joint_violation_rad"] <= 0.0
            and value["velocity_ratio"] <= 1.0
            and value["control_ratio"] <= 1.0
            for value in metrics
        )
        and np.isfinite(row["final_cuda_one_step_pinocchio_defect"])
    )


def select_winner(rows):
    started = time.perf_counter()
    successes = [row for row in rows if row["optimizer_improved_success"]]
    winner = min(
        successes,
        key=lambda row: row["final_cuda"]["nonnegative_task_motion_cost"],
        default=None,
    )
    return winner, 1e3 * (time.perf_counter() - started)


def run(args):
    if args.repeats < 1 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    lane = LANES[args.lane]
    model = pin.buildModelFromUrdf(str(MODEL_PATH))
    q0, start, goal = generate_instance(model, lane, args.instance_seed)
    seeds, seed_metadata = generate_seeds(model, q0, args.method_seed, args.budget)
    x0 = np.hstack([q0, np.zeros(model.nv, dtype=np.float32)])
    x0_batch = np.tile(x0, (args.budget, 1)).astype(np.float32)
    reference = np.tile(np.hstack([goal, np.zeros(3, dtype=np.float32)]), KNOTS)
    references = np.tile(reference, (args.budget, 1)).astype(np.float32)
    independent_solver = make_solver(1)
    batched_solver = make_solver(args.budget)
    preflight = model_pair_preflight(batched_solver, model, count=args.budget)
    if not preflight["passes_1e-3_gate"]:
        raise RuntimeError("CUDA/Pinocchio random-state model preflight failed")
    methods = {
        "independent": lambda: solve_independent(
            independent_solver, x0_batch, references, seeds
        ),
        "batched": lambda: solve_batched(batched_solver, x0_batch, references, seeds),
    }
    for _ in range(args.warmups):
        for method in ("independent", "batched"):
            methods[method]()
    first_methods = np.asarray(
        ["independent", "batched"] * ((args.repeats + 1) // 2), dtype=object
    )[: args.repeats]
    order_rng = np.random.default_rng(
        np.random.SeedSequence(
            [args.instance_seed, args.method_seed, args.budget, KNOTS]
        )
    )
    order_rng.shuffle(first_methods)
    wall = {"independent": [], "batched": []}
    reported = {"independent": [], "batched": []}
    solve_workflow_wall = {"independent": [], "batched": []}
    raw = []
    reference_results = None
    for repeat in range(args.repeats):
        first = str(first_methods[repeat])
        order = (first, "batched" if first == "independent" else "independent")
        pair = {}
        for method in order:
            output, stats, wall_ms, reported_ms, workflow_ms = methods[method]()
            pair[method] = (output, stats)
            wall[method].append(wall_ms)
            reported[method].append(reported_ms)
            solve_workflow_wall[method].append(workflow_ms)
        independent_output, independent_stats = pair["independent"]
        batched_output, batched_stats = pair["batched"]
        parity = {
            "trajectory_bitwise_equal": bool(
                np.array_equal(independent_output, batched_output)
            ),
            "initial_merit_bitwise_equal": bool(
                np.array_equal(
                    independent_stats["initial_merit"], batched_stats["initial_merit"]
                )
            ),
            "final_merit_bitwise_equal": bool(
                np.array_equal(
                    independent_stats["final_merit"], batched_stats["final_merit"]
                )
            ),
            "sqp_iterations_equal": bool(
                np.array_equal(
                    independent_stats["sqp_iterations"],
                    batched_stats["sqp_iterations"],
                )
            ),
            "pcg_iterations_equal": bool(
                np.array_equal(
                    independent_stats["total_pcg_iterations"],
                    batched_stats["total_pcg_iterations"],
                )
            ),
            "pcg_cap_hits_equal": bool(
                np.array_equal(
                    independent_stats["pcg_cap_hits"], batched_stats["pcg_cap_hits"]
                )
            ),
            "maximum_initial_merit_relative_difference": float(
                np.max(
                    np.abs(
                        independent_stats["initial_merit"]
                        - batched_stats["initial_merit"]
                    )
                    / np.maximum(np.abs(independent_stats["initial_merit"]), 1.0)
                )
            ),
        }
        raw.append(
            {
                "repeat": repeat,
                "order": list(order),
                "independent_wall_ms": wall["independent"][-1],
                "batched_wall_ms": wall["batched"][-1],
                "independent_reported_ms": reported["independent"][-1],
                "batched_reported_ms": reported["batched"][-1],
                "independent_reset_solve_stats_workflow_ms": (
                    solve_workflow_wall["independent"][-1]
                ),
                "batched_reset_solve_stats_workflow_ms": (
                    solve_workflow_wall["batched"][-1]
                ),
                "independent_initial_merit": independent_stats[
                    "initial_merit"
                ].tolist(),
                "batched_initial_merit": batched_stats["initial_merit"].tolist(),
                "independent_final_merit": independent_stats[
                    "final_merit"
                ].tolist(),
                "batched_final_merit": batched_stats["final_merit"].tolist(),
                "independent_sqp_iterations": independent_stats[
                    "sqp_iterations"
                ].tolist(),
                "batched_sqp_iterations": batched_stats[
                    "sqp_iterations"
                ].tolist(),
                "independent_total_pcg_iterations": independent_stats[
                    "total_pcg_iterations"
                ].tolist(),
                "batched_total_pcg_iterations": batched_stats[
                    "total_pcg_iterations"
                ].tolist(),
                "independent_pcg_cap_hits": independent_stats[
                    "pcg_cap_hits"
                ].tolist(),
                "batched_pcg_cap_hits": batched_stats["pcg_cap_hits"].tolist(),
                "independent_output_sha256": sha256_array(independent_output),
                "batched_output_sha256": sha256_array(batched_output),
                "independent_stats_sha256": sha256_array(
                    np.concatenate(
                        [
                            independent_stats["initial_merit"],
                            independent_stats["final_merit"],
                            independent_stats["sqp_iterations"],
                            independent_stats["total_pcg_iterations"],
                            independent_stats["pcg_cap_hits"],
                        ]
                    )
                ),
                "batched_stats_sha256": sha256_array(
                    np.concatenate(
                        [
                            batched_stats["initial_merit"],
                            batched_stats["final_merit"],
                            batched_stats["sqp_iterations"],
                            batched_stats["total_pcg_iterations"],
                            batched_stats["pcg_cap_hits"],
                        ]
                    )
                ),
                "parity": parity,
            }
        )
        if reference_results is None:
            reference_results = {
                "independent": (independent_output, independent_stats),
                "batched": (batched_output, batched_stats),
            }
    independent_output, independent_stats = reference_results["independent"]
    reference_output, reference_stats = reference_results["batched"]
    batched_quality, batched_rollout_arrays, batched_certification_ms = certify(
        model, goal, x0, seeds, reference_output, reference_stats, batched_solver
    )
    independent_quality, independent_rollout_arrays, independent_certification_ms = certify(
        model,
        goal,
        x0,
        seeds,
        independent_output,
        independent_stats,
        batched_solver,
    )
    quality_noninferiority = []
    for serial, batch in zip(independent_quality, batched_quality):
        initial_scale = max(abs(serial["initial_solver_merit"]), 1.0)
        initial_merit_relative_difference = abs(
            batch["initial_solver_merit"] - serial["initial_solver_merit"]
        ) / initial_scale
        cost_tolerance = max(
            1e-3, 0.01 * max(serial["final_cuda"]["nonnegative_task_motion_cost"], 1.0)
        )
        row = {
            "candidate_index": serial["candidate_index"],
            "initial_merit_relative_difference": float(
                initial_merit_relative_difference
            ),
            "serial_success_preserved": bool(
                not serial["optimizer_improved_success"]
                or batch["optimizer_improved_success"]
            ),
            "batch_terminal_within_serial_plus_2mm": bool(
                batch["final_cuda"]["terminal_error_m"]
                <= serial["final_cuda"]["terminal_error_m"] + 0.002
            ),
            "batch_external_cost_noninferior": bool(
                batch["final_cuda"]["nonnegative_task_motion_cost"]
                <= serial["final_cuda"]["nonnegative_task_motion_cost"]
                + cost_tolerance
            ),
            "batch_has_no_extra_cap": bool(
                batch["pcg_cap_hits"] <= serial["pcg_cap_hits"]
            ),
            "serial_final_rollouts_finite_and_within_limits": (
                final_rollouts_finite_and_within_limits(serial)
            ),
            "batch_final_rollouts_finite_and_within_limits": (
                final_rollouts_finite_and_within_limits(batch)
            ),
            "batch_has_no_added_nonfinite_or_limit_failure": bool(
                not final_rollouts_finite_and_within_limits(serial)
                or final_rollouts_finite_and_within_limits(batch)
            ),
            "max_abs_serial_batch_planned_delta": float(
                np.max(
                    np.abs(
                        reference_output[serial["candidate_index"]]
                        - independent_output[serial["candidate_index"]]
                    )
                )
            ),
        }
        row["passes"] = bool(
            initial_merit_relative_difference <= 1e-6
            and row["serial_success_preserved"]
            and row["batch_terminal_within_serial_plus_2mm"]
            and row["batch_external_cost_noninferior"]
            and row["batch_has_no_extra_cap"]
            and row["batch_has_no_added_nonfinite_or_limit_failure"]
        )
        quality_noninferiority.append(row)
    serial_winner, serial_selection_ms = select_winner(independent_quality)
    batched_winner, batched_selection_ms = select_winner(batched_quality)
    selected_winner_noninferior = bool(
        serial_winner is None
        or (
            batched_winner is not None
            and batched_winner["final_cuda"]["terminal_error_m"]
            <= serial_winner["final_cuda"]["terminal_error_m"] + 0.002
            and batched_winner["final_cuda"]["nonnegative_task_motion_cost"]
            <= serial_winner["final_cuda"]["nonnegative_task_motion_cost"]
            + max(
                1e-3,
                0.01
                * max(
                    serial_winner["final_cuda"]["nonnegative_task_motion_cost"],
                    1.0,
                ),
            )
        )
    )
    independent_wall = np.asarray(wall["independent"])
    batched_wall = np.asarray(wall["batched"])
    reduction = (independent_wall - batched_wall) / independent_wall
    extension = Path(batched_solver.lib.__file__)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    repeatability = {}
    for method in ("independent", "batched"):
        output_hash_count = len({row[f"{method}_output_sha256"] for row in raw})
        stats_hash_count = len({row[f"{method}_stats_sha256"] for row in raw})
        exact_integer_telemetry = all(
            all(
                row[f"{method}_{name}"] == raw[0][f"{method}_{name}"]
                for name in (
                    "sqp_iterations",
                    "total_pcg_iterations",
                    "pcg_cap_hits",
                )
            )
            for row in raw[1:]
        )
        merit_spans = merit_repeatability(raw, method)
        numerical_pass = bool(
            output_hash_count == 1
            and exact_integer_telemetry
            and merit_spans["initial_merit"]["maximum_relative_span"] <= 1e-6
            and merit_spans["final_merit"]["maximum_relative_span"] <= 1e-6
        )
        repeatability[method] = {
            "output_hash_count": output_hash_count,
            "stats_hash_count": stats_hash_count,
            "exact_sqp_pcg_and_cap_telemetry": exact_integer_telemetry,
            "merit_spans": merit_spans,
            "numerical_repeatability_pass": numerical_pass,
        }
    all_numerical_repeatability = all(
        value["numerical_repeatability_pass"] for value in repeatability.values()
    )
    all_output_quality_noninferiority = bool(
        all(row["passes"] for row in quality_noninferiority)
        and selected_winner_noninferior
    )
    all_initial_merits_match = all(
        row["parity"]["maximum_initial_merit_relative_difference"] <= 1e-6
        for row in raw
    )
    revised_numerical_and_noninferiority = bool(
        all_numerical_repeatability
        and all_initial_merits_match
        and all_output_quality_noninferiority
    )
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    untracked_status = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    summary = {
        "schema_version": 1,
        "benchmark": "iiwa_same_problem_imperfect_multistart_sqp",
        "accepted_artifact": (
            "optimized controls replayed from common x0; solver-planned state defects "
            "are report-only and not claimed dynamically consistent"
        ),
        "lane": asdict(lane),
        "instance_seed": args.instance_seed,
        "method_seed": args.method_seed,
        "budget": args.budget,
        "start_configuration": q0.tolist(),
        "start_position": start.tolist(),
        "goal_position": goal.tolist(),
        "seed_distribution": {
            "description": (
                "cold repeated common state/zero control plus bounded smooth random "
                "intermediate joint states and controls; no IK, rollout, calibration, "
                "collision query, route template, or optimization"
            ),
            **seed_metadata,
        },
        "solver_parameters": solver_parameters(),
        "model_pair_preflight": preflight,
        "input_hashes": {
            "x0_batch": sha256_array(x0_batch),
            "reference_batch": sha256_array(references),
            "seed_trajectories": sha256_array(seeds),
        },
        "provenance": {
            "git_head": git_head,
            "git_tracked_dirty": bool(tracked_status),
            "untracked_paths": untracked_status,
            "model_sha256": sha256_file(MODEL_PATH),
            "extension": str(extension),
            "extension_sha256": sha256_file(extension),
        },
        "solver_only_wall": {
            "independent": timing_stats(independent_wall),
            "batched": timing_stats(batched_wall),
            "paired_reduction_fraction": reduction.tolist(),
            "median_reduction_fraction": float(np.median(reduction)),
            "bootstrap_95_ci_median_reduction": bootstrap_interval(reduction),
        },
        "native_reported_solver": {
            "independent": timing_stats(reported["independent"]),
            "batched": timing_stats(reported["batched"]),
        },
        "reset_solve_stats_workflow_wall": {
            "independent": timing_stats(solve_workflow_wall["independent"]),
            "batched": timing_stats(solve_workflow_wall["batched"]),
        },
        "secondary_workflow_components": {
            "seed_generation_ms": seed_metadata["generation_wall_ms"],
            "independent_certification_ms": independent_certification_ms,
            "batched_certification_ms": batched_certification_ms,
            "independent_selection_ms": serial_selection_ms,
            "batched_selection_ms": batched_selection_ms,
            "independent_estimated_end_to_end_ms": float(
                seed_metadata["generation_wall_ms"]
                + np.median(solve_workflow_wall["independent"])
                + independent_certification_ms
                + serial_selection_ms
            ),
            "batched_estimated_end_to_end_ms": float(
                seed_metadata["generation_wall_ms"]
                + np.median(solve_workflow_wall["batched"])
                + batched_certification_ms
                + batched_selection_ms
            ),
        },
        "all_repeats_bitwise_full_parity_diagnostic": bool(
            all(
                all(
                    value
                    for key, value in row["parity"].items()
                    if key != "maximum_initial_merit_relative_difference"
                )
                for row in raw
            )
        ),
        "all_repeats_initial_merit_within_relative_1e-6": all_initial_merits_match,
        "within_mode_repeatability": repeatability,
        "all_within_mode_numerical_repeatability_checks_pass": (
            all_numerical_repeatability
        ),
        "all_output_quality_noninferiority_checks_pass": (
            all_output_quality_noninferiority
        ),
        "revised_numerical_and_noninferiority_aggregate_pass": (
            revised_numerical_and_noninferiority
        ),
        "selected_winner_noninferior": selected_winner_noninferior,
        "serial_selected_winner": serial_winner,
        "batched_selected_winner": batched_winner,
        "multistart_quality_vs_cold": {
            "independent": multistart_quality_vs_cold(independent_quality),
            "batched": multistart_quality_vs_cold(batched_quality),
        },
        "quality_noninferiority": quality_noninferiority,
        "optimizer_improved_success_count": int(
            sum(row["optimizer_improved_success"] for row in batched_quality)
        ),
        "serial_optimizer_improved_success_count": int(
            sum(row["optimizer_improved_success"] for row in independent_quality)
        ),
        "submitted_lane_count": len(batched_quality),
        "pcg_cap_lane_count": int(
            sum(row["pcg_cap_hits"] > 0 for row in batched_quality)
        ),
        "serial_quality": independent_quality,
        "batched_quality": batched_quality,
        "raw_repeats": raw,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable_summary = json_safe(summary)
    args.output.write_text(
        json.dumps(serializable_summary, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        seed_trajectories=seeds,
        optimized_planned_trajectories=reference_output,
        x0=x0,
        reference=references[0],
        **{f"batched_{key}": value for key, value in batched_rollout_arrays.items()},
        **{
            f"independent_{key}": value
            for key, value in independent_rollout_arrays.items()
        },
    )
    print(args.output)
    print(
        json.dumps(
            {
                "optimizer_improved_success_count": summary[
                    "optimizer_improved_success_count"
                ],
                "submitted_lane_count": summary["submitted_lane_count"],
                "pcg_cap_lane_count": summary["pcg_cap_lane_count"],
                "all_repeats_bitwise_full_parity_diagnostic": summary[
                    "all_repeats_bitwise_full_parity_diagnostic"
                ],
                "all_output_quality_noninferiority_checks_pass": summary[
                    "all_output_quality_noninferiority_checks_pass"
                ],
                "within_mode_repeatability": summary["within_mode_repeatability"],
                "median_solver_only_reduction": summary["solver_only_wall"][
                    "median_reduction_fraction"
                ],
                "bootstrap_95_ci": summary["solver_only_wall"][
                    "bootstrap_95_ci_median_reduction"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=tuple(LANES), default="near")
    parser.add_argument("--instance-seed", type=int, default=40)
    parser.add_argument("--method-seed", type=int, default=1000)
    parser.add_argument("--budget", type=int, choices=(1, 2, 4, 8, 16, 32), default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "probe.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
