#!/usr/bin/env python3
"""Generate, solve, certify, and visualize Tiago pillar-routing benchmarks.

PRIMARY BENCHMARK PROPERTY: candidates share x0, reference, and objective;
only their full trajectory initialization differs. Poor initializations fail
or return worse best-known solutions. In the reference hard case, SQP leaves
every accepted informed initialization exactly unchanged. Do not interpret the
outputs as solver-converged local optima or a proven global optimum.

The task sends the Tiago tool from one side of an infinite vertical pillar to
the other.  Clockwise and counterclockwise trajectories use the same objective
and are distinguished by their winding angle around the pillar.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "tiago_src"))

from bsqp.interface import BSQP
from gato_tiago.multimodal_pillar import (
    DIFFICULTIES,
    MODEL_PATH,
    PillarInstance,
    arm_link_positions,
    candidate_specs,
    certify_results,
    evaluate_trajectory,
    generate_instance,
    load_model,
    pack_trajectory,
    pack_warm_start,
    random_candidate_specs,
    random_task_candidate_specs,
    reference_batch,
    tool_position,
    unpack_trajectory,
    dynamics_defect,
)


DEFAULT_OUTPUT = REPO_ROOT / "example_artifacts" / "tiago_multimodal_pillar" / "v2"
SCHEMA_VERSION = 2


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def solver_parameters(difficulty):
    return {
        "max_sqp_iters": difficulty.max_sqp_iters,
        "kkt_tol": 1e-3,
        "max_pcg_iters": difficulty.max_pcg_iters,
        "pcg_tol": difficulty.pcg_tol,
        "solve_ratio": 1.0,
        "mu": 20.0,
        "q_cost": difficulty.q_cost,
        "qd_cost": difficulty.qd_cost,
        "u_cost": difficulty.u_cost,
        "N_cost": difficulty.terminal_cost,
        # In tiago_right_multimodal these are running/terminal pillar weights.
        "ee_orient_cost": difficulty.pillar_cost,
        "ee_orient_N_cost": difficulty.pillar_cost,
        "q_lim_cost": 0.01,
        "vel_lim_cost": 0.001,
        "ctrl_lim_cost": 0.003,
        "rho": 0.01,
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _provenance(difficulty):
    source_paths = (
        REPO_ROOT / "gato/dynamics/tiago_right/tiago_right_plant.cuh",
        REPO_ROOT / "tiago_src/gato_tiago/multimodal_pillar.py",
        Path(__file__).resolve(),
    )
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_head, git_dirty = None, None
    module_name = f"bsqp.bsqpN{difficulty.knots}_tiago_right_multimodal"
    module_spec = importlib.util.find_spec(module_name)
    extension_path = Path(module_spec.origin) if module_spec and module_spec.origin else None
    model_path = REPO_ROOT / MODEL_PATH
    return {
        "git_head": git_head,
        "git_worktree_dirty": git_dirty,
        "model_sha256": _sha256_file(model_path),
        "solver_extension": str(extension_path) if extension_path else None,
        "solver_extension_sha256": _sha256_file(extension_path) if extension_path else None,
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256_file(path) for path in source_paths
        },
    }


def _calibrate_warm_starts_to_cuda(solver, model, difficulty, packed_batch, iterations=3):
    """Create executable CUDA rollouts for all candidates in parallel."""

    packed_batch = np.asarray(packed_batch, dtype=np.float32)
    if packed_batch.ndim != 2 or packed_batch.shape[0] != solver.batch_size:
        raise ValueError("packed warm starts must match the solver batch size")
    desired = [
        unpack_trajectory(packed, model, difficulty.knots) for packed in packed_batch
    ]
    desired_q = np.asarray([item[0] for item in desired], dtype=np.float64)
    desired_qd = np.asarray([item[1] for item in desired], dtype=np.float64)
    data = [model.createData() for _ in range(solver.batch_size)]
    nx = model.nq + model.nv
    stride = nx + model.nv
    calibrated = np.zeros_like(packed_batch)
    rollout_state = np.hstack([desired_q[:, 0], desired_qd[:, 0]]).astype(np.float32)
    controls = np.empty(
        (solver.batch_size, difficulty.knots - 1, model.nv), dtype=np.float64
    )
    max_tracking_error = np.zeros(solver.batch_size, dtype=np.float64)
    for knot in range(difficulty.knots - 1):
        q = rollout_state[:, : model.nq].astype(np.float64)
        qd = rollout_state[:, model.nq :].astype(np.float64)
        feedforward = (desired_qd[:, knot + 1] - desired_qd[:, knot]) / difficulty.dt
        target_acceleration = (
            feedforward
            + 100.0 * (desired_q[:, knot] - q)
            + 20.0 * (desired_qd[:, knot] - qd)
        )
        control = np.asarray(
            [
                pin.rnea(model, data[index], q[index], qd[index], target_acceleration[index])
                for index in range(solver.batch_size)
            ],
            dtype=np.float64,
        )
        mass = np.asarray(
            [pin.crba(model, data[index], q[index]).copy() for index in range(solver.batch_size)]
        )
        mass = np.triu(mass) + np.swapaxes(np.triu(mass, 1), 1, 2)
        for _ in range(iterations):
            predicted = np.asarray(
                solver.sim_forward(
                    rollout_state, control.astype(np.float32), difficulty.dt
                ),
                dtype=np.float64,
            )
            acceleration = (predicted[:, model.nq :] - qd) / difficulty.dt
            control += np.einsum(
                "bij,bj->bi", mass, target_acceleration - acceleration
            )
        predicted = np.asarray(
            solver.sim_forward(rollout_state, control.astype(np.float32), difficulty.dt),
            dtype=np.float64,
        )
        target_state = np.hstack(
            [desired_q[:, knot + 1], desired_qd[:, knot + 1]]
        )
        max_tracking_error = np.maximum(
            max_tracking_error, np.linalg.norm(predicted - target_state, axis=1)
        )
        controls[:, knot] = control
        offset = knot * stride
        calibrated[:, offset : offset + nx] = rollout_state
        calibrated[:, offset + nx : offset + stride] = control
        rollout_state = predicted.astype(np.float32)
    calibrated[:, (difficulty.knots - 1) * stride :] = rollout_state
    torque_ratios = np.max(
        np.abs(controls) / model.effortLimit.astype(np.float64), axis=(1, 2)
    )
    metadata = [
        {
            "seed_cuda_max_tracking_error": float(max_tracking_error[index]),
            "seed_cuda_exact_rollout": True,
            "seed_cuda_max_torque_ratio": float(torque_ratios[index]),
        }
        for index in range(solver.batch_size)
    ]
    return calibrated, metadata


def _calibrate_warm_start_to_cuda(solver, model, difficulty, packed, iterations=3):
    calibrated, metadata = _calibrate_warm_starts_to_cuda(
        solver, model, difficulty, np.asarray(packed, dtype=np.float32)[None, :], iterations
    )
    return calibrated[0], metadata[0]


def _rollout_and_gpu_defect(solver, model, difficulty, trajectory, x0):
    rollouts, defects = _rollout_and_gpu_defect_batch(
        solver,
        model,
        difficulty,
        np.asarray(trajectory, dtype=np.float32)[None, :],
        x0,
    )
    return rollouts[0], float(defects[0])


def _rollout_and_gpu_defect_batch(solver, model, difficulty, trajectories, x0):
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim != 2 or trajectories.shape[0] != solver.batch_size:
        raise ValueError("trajectories must match the solver batch size")
    unpacked = [
        unpack_trajectory(trajectory, model, difficulty.knots)
        for trajectory in trajectories
    ]
    planned_q = np.asarray([item[0] for item in unpacked], dtype=np.float64)
    planned_qd = np.asarray([item[1] for item in unpacked], dtype=np.float64)
    controls = np.asarray([item[2] for item in unpacked], dtype=np.float32)
    nx = model.nq + model.nv
    stride = nx + model.nv
    rollouts = np.zeros_like(trajectories)
    rollout_state = np.tile(np.asarray(x0, dtype=np.float32), (solver.batch_size, 1))
    max_gpu_defect = np.zeros(solver.batch_size, dtype=np.float64)
    for knot in range(difficulty.knots):
        offset = knot * stride
        rollouts[:, offset : offset + nx] = rollout_state
        if knot >= difficulty.knots - 1:
            continue
        rollouts[:, offset + nx : offset + stride] = controls[:, knot]
        planned_state = np.hstack(
            [planned_q[:, knot], planned_qd[:, knot]]
        ).astype(np.float32)
        predicted_planned = np.asarray(
            solver.sim_forward(planned_state, controls[:, knot], difficulty.dt),
            dtype=np.float64,
        )
        planned_next = np.hstack(
            [planned_q[:, knot + 1], planned_qd[:, knot + 1]]
        )
        max_gpu_defect = np.maximum(
            max_gpu_defect,
            np.linalg.norm(predicted_planned - planned_next, axis=1),
        )
        finite = np.all(np.isfinite(rollout_state), axis=1)
        predicted_rollout = np.asarray(
            solver.sim_forward(rollout_state, controls[:, knot], difficulty.dt),
            dtype=np.float32,
        )
        predicted_rollout[~finite] = np.nan
        rollout_state = predicted_rollout
    return rollouts, max_gpu_defect


def _modeled_limit_preflight(model, difficulty, trajectory):
    q, qd, controls = unpack_trajectory(trajectory, model, difficulty.knots)
    finite = bool(
        np.all(np.isfinite(q))
        and np.all(np.isfinite(qd))
        and np.all(np.isfinite(controls))
    )
    joint_violation = float(
        max(
            np.max(model.lowerPositionLimit - q, initial=0.0),
            np.max(q - model.upperPositionLimit, initial=0.0),
            0.0,
        )
    )
    velocity_ratio = float(np.max(np.abs(qd) / model.velocityLimit))
    torque_ratio = float(np.max(np.abs(controls) / model.effortLimit))
    return {
        "finite": finite,
        "joint_violation_rad": joint_violation,
        "velocity_ratio": velocity_ratio,
        "torque_ratio": torque_ratio,
        "safe_for_solver": bool(
            finite
            and joint_violation <= 0.0
            and velocity_ratio <= 1.0
            and torque_ratio <= 1.0
        ),
    }


def _solve_candidate_specs(instance: PillarInstance, specs):
    benchmark_start = time.perf_counter()
    difficulty = DIFFICULTIES[instance.difficulty]
    specs = list(specs)
    budget = len(specs)
    model = load_model(REPO_ROOT / MODEL_PATH)
    warm_starts = []
    seed_metadata = []
    for spec in specs:
        warm_start, metadata = pack_warm_start(model, instance, difficulty, spec)
        warm_starts.append(warm_start)
        seed_metadata.append({"name": spec.name, **metadata})
    warm_starts = np.asarray(warm_starts, dtype=np.float32)
    geometric_warm_start_done = time.perf_counter()
    x0 = np.hstack(
        [np.asarray(instance.start_q, dtype=np.float32), np.zeros(model.nv, dtype=np.float32)]
    )
    initial_states = np.tile(x0, (budget, 1))
    references = reference_batch(instance, difficulty.knots, budget)
    if not np.all(references == references[0]):
        raise AssertionError("All multimodal candidates must receive the same reference")

    try:
        solver = BSQP(
            model_path=str(REPO_ROOT / MODEL_PATH),
            batch_size=budget,
            N=difficulty.knots,
            dt=difficulty.dt,
            plant_type="tiago_right_multimodal",
            **solver_parameters(difficulty),
        )
    except ValueError as exc:
        command = (
            "./tools/build.sh --plant tiago_right_multimodal "
            f"--knots {difficulty.knots} --target "
            f"bsqpN{difficulty.knots}_tiago_right_multimodal --native-cuda-arch"
        )
        raise RuntimeError(f"Multimodal solver extension is unavailable. Build it with: {command}") from exc

    warm_starts, calibration_metadata = _calibrate_warm_starts_to_cuda(
        solver, model, difficulty, warm_starts
    )
    for index, calibration in enumerate(calibration_metadata):
        seed_metadata[index].update(calibration)
    warm_start_done = time.perf_counter()

    preflights = [
        _modeled_limit_preflight(model, difficulty, warm_start)
        for warm_start in warm_starts
    ]
    x0_sha256 = _sha256_array(x0.astype(np.float32))
    reference_sha256 = _sha256_array(references[0].astype(np.float32))
    warm_start_sha256 = [_sha256_array(row.astype(np.float32)) for row in warm_starts]
    unsafe_indices = [
        index for index, preflight in enumerate(preflights)
        if not preflight["safe_for_solver"]
    ]
    if unsafe_indices:
        if budget != 1:
            raise RuntimeError(
                "batched execution contains a calibrated seed outside modeled limits; "
                "use independent execution so it is rejected before SQP"
            )
        rollout, gpu_defect = _rollout_and_gpu_defect(
            solver, model, difficulty, warm_starts[0], x0
        )
        q, qd, controls = unpack_trajectory(
            warm_starts[0], model, difficulty.knots
        )
        result = evaluate_trajectory(model, instance, difficulty, rollout)
        result.update(
            candidate_index=0,
            candidate_name=specs[0].name,
            requested_route_side=specs[0].route_side,
            final_solver_merit=None,
            sqp_iterations=0,
            total_pcg_iterations=0,
            pcg_cap_hits=0,
            pcg_cap_free=True,
            solver_zero_pcg_exit=False,
            rollout_finite=bool(np.all(np.isfinite(rollout))),
            max_cuda_planned_dynamics_defect=float(gpu_defect),
            max_independent_planned_dynamics_defect=float(
                dynamics_defect(model, q, qd, controls, difficulty.dt)
            ),
            planned_initial_state_error=0.0,
            max_planned_state_to_rollout_error=0.0,
            certifiable=False,
            calibrated_seed_reconstructed_objective=float(result["objective"]),
            reconstructed_objective_improvement=0.0,
            solution_l2_delta_from_calibrated_seed=0.0,
            solution_max_abs_delta_from_calibrated_seed=0.0,
            solver_modified_calibrated_seed=False,
            solver_skipped_reason="calibrated_seed_outside_modeled_limits",
            seed_limit_preflight=preflights[0],
            input_x0_sha256=x0_sha256,
            input_reference_sha256=reference_sha256,
            input_warm_start_sha256=warm_start_sha256[0],
            **seed_metadata[0],
        )
        certificate = certify_results([result], min_support_per_mode=2)
        done = time.perf_counter()
        timings = {
            "geometric_warm_start_generation_ms": 1e3
            * (geometric_warm_start_done - benchmark_start),
            "cuda_dynamics_calibration_ms": 1e3
            * (warm_start_done - geometric_warm_start_done),
            "solver_wall_time_ms": 0.0,
            "solver_reported_gpu_time_ms": 0.0,
            "evaluation_ms": 1e3 * (done - warm_start_done),
            "benchmark_total_ms_excluding_render": 1e3 * (done - benchmark_start),
            "candidate_execution": "rejected_before_sqp",
        }
        return model, warm_starts, [result], certificate, timings

    solver.reset_dual()
    solver.reset_rho()
    solve_wall_start = time.perf_counter()
    trajectories, solve_time_us = solver.solve(initial_states, references, warm_starts)
    solve_wall_done = time.perf_counter()
    stats = solver.get_stats()
    kkt = np.asarray(stats["kkt_converged"], dtype=np.int32)
    sqp = np.asarray(stats["sqp_iters"], dtype=np.int32)
    final_merit = np.asarray(stats["final_merit"], dtype=np.float64)
    pcg = np.asarray(stats.get("pcg_iters", []), dtype=np.int32)
    rollout_batch, gpu_planned_defects = _rollout_and_gpu_defect_batch(
        solver, model, difficulty, trajectories, x0
    )

    results = []
    for index, spec in enumerate(specs):
        cap_hits = int(np.sum(pcg[:, index] >= difficulty.max_pcg_iters)) if pcg.size else 0
        planned_q, planned_qd, controls = unpack_trajectory(
            trajectories[index], model, difficulty.knots
        )
        independent_planned_defect = dynamics_defect(
            model, planned_q, planned_qd, controls, difficulty.dt
        )
        rollout = rollout_batch[index]
        gpu_planned_defect = float(gpu_planned_defects[index])
        rollout_q, rollout_qd, _ = unpack_trajectory(
            rollout, model, difficulty.knots
        )
        planned_state = np.hstack([planned_q, planned_qd])
        rollout_states = np.hstack([rollout_q, rollout_qd])
        planned_initial_error = float(np.linalg.norm(planned_state[0] - x0))
        planned_to_rollout_error = float(
            np.max(np.linalg.norm(planned_state - rollout_states, axis=1), initial=0.0)
        )
        result = evaluate_trajectory(model, instance, difficulty, rollout)
        seed_result = evaluate_trajectory(
            model, instance, difficulty, warm_starts[index]
        )
        solution_delta = np.asarray(trajectories[index]) - warm_starts[index]
        result["max_independent_planned_dynamics_defect"] = float(
            independent_planned_defect
        )
        result["max_cuda_planned_dynamics_defect"] = float(gpu_planned_defect)
        result["rollout_finite"] = bool(np.all(np.isfinite(rollout)))
        result["planned_initial_state_error"] = planned_initial_error
        result["max_planned_state_to_rollout_error"] = planned_to_rollout_error
        result["pcg_cap_free"] = cap_hits == 0
        result["certifiable"] = bool(
            result["physical_feasible"]
            and result["rollout_finite"]
            and result["pcg_cap_free"]
            and planned_initial_error <= 1e-6
            and planned_to_rollout_error <= 1e-3
            and gpu_planned_defect <= 1e-4
            and independent_planned_defect <= 1e-3
        )
        result.update(
            candidate_index=index,
            candidate_name=spec.name,
            requested_route_side=spec.route_side,
            final_solver_merit=float(final_merit[index]),
            sqp_iterations=int(sqp[index]),
            total_pcg_iterations=int(np.sum(pcg[:, index])) if pcg.size else 0,
            pcg_cap_hits=cap_hits,
            solver_zero_pcg_exit=bool(kkt[index]),
            calibrated_seed_reconstructed_objective=float(seed_result["objective"]),
            reconstructed_objective_improvement=float(
                seed_result["objective"] - result["objective"]
            ),
            solution_l2_delta_from_calibrated_seed=float(
                np.linalg.norm(solution_delta)
            ),
            solution_max_abs_delta_from_calibrated_seed=float(
                np.max(np.abs(solution_delta), initial=0.0)
            ),
            solver_modified_calibrated_seed=bool(
                np.max(np.abs(solution_delta), initial=0.0) > 1e-7
            ),
            solver_skipped_reason=None,
            seed_limit_preflight=preflights[index],
            input_x0_sha256=x0_sha256,
            input_reference_sha256=reference_sha256,
            input_warm_start_sha256=warm_start_sha256[index],
            **seed_metadata[index],
        )
        results.append(result)
    certificate = certify_results(results, min_support_per_mode=2)
    evaluation_done = time.perf_counter()
    timings = {
        "geometric_warm_start_generation_ms": 1e3 * (
            geometric_warm_start_done - benchmark_start
        ),
        "cuda_dynamics_calibration_ms": 1e3 * (
            warm_start_done - geometric_warm_start_done
        ),
        "solver_wall_time_ms": 1e3 * (solve_wall_done - solve_wall_start),
        "solver_reported_gpu_time_ms": float(solve_time_us) / 1e3,
        "evaluation_ms": 1e3 * (evaluation_done - solve_wall_done),
        "benchmark_total_ms_excluding_render": 1e3 * (evaluation_done - benchmark_start),
    }
    return model, trajectories, results, certificate, timings


def _proposal_specs(
    instance: PillarInstance,
    budget: int,
    strategy: str,
    proposal_seed: int | None = None,
):
    if proposal_seed is None:
        proposal_seed = instance.seed
    if strategy == "informed":
        return candidate_specs(budget)
    if strategy == "random-joint":
        return random_candidate_specs(budget, seed=proposal_seed)
    if strategy == "random-task":
        return random_task_candidate_specs(budget, seed=proposal_seed)
    raise ValueError(
        "proposal strategy must be 'informed', 'random-joint', or 'random-task'"
    )


def solve_instance(
    instance: PillarInstance,
    *,
    budget: int,
    execution="independent",
    proposal_strategy="informed",
    proposal_seed=None,
):
    specs = _proposal_specs(instance, budget, proposal_strategy, proposal_seed)
    if execution == "batched":
        return _solve_candidate_specs(instance, specs)
    if execution != "independent":
        raise ValueError("execution must be 'independent' or 'batched'")

    benchmark_start = time.perf_counter()
    all_trajectories = []
    all_results = []
    accumulated_timings = {}
    model = None
    for index, spec in enumerate(specs):
        model, trajectories, results, _, timings = _solve_candidate_specs(
            instance, [spec]
        )
        result = results[0]
        result["candidate_index"] = index
        all_trajectories.append(trajectories[0])
        all_results.append(result)
        for key, value in timings.items():
            if isinstance(value, (int, float)):
                accumulated_timings[key] = accumulated_timings.get(key, 0.0) + value

    certificate = certify_results(all_results, min_support_per_mode=2)
    accumulated_timings["benchmark_total_ms_excluding_render"] = 1e3 * (
        time.perf_counter() - benchmark_start
    )
    accumulated_timings["candidate_execution"] = "independent_batch1"
    return (
        model,
        np.asarray(all_trajectories, dtype=np.float32),
        all_results,
        certificate,
        accumulated_timings,
    )


def _benchmark_result_snapshot(trajectories, results, certificate, timings, wall_ms):
    return {
        "wall_time_ms": float(wall_ms),
        "reported_timings": timings,
        "trajectory_sha256": hashlib.sha256(
            np.asarray(trajectories, dtype=np.float32).tobytes()
        ).hexdigest(),
        "candidate_names": [row["candidate_name"] for row in results],
        "certifiable": [bool(row["certifiable"]) for row in results],
        "modes": [row["mode"] for row in results],
        "objectives": [float(row["objective"]) for row in results],
        "total_pcg_iterations": [int(row["total_pcg_iterations"]) for row in results],
        "pcg_cap_hits": [int(row["pcg_cap_hits"]) for row in results],
        "sqp_iterations": [int(row["sqp_iterations"]) for row in results],
        "input_x0_sha256": [row["input_x0_sha256"] for row in results],
        "input_reference_sha256": [row["input_reference_sha256"] for row in results],
        "input_warm_start_sha256": [row["input_warm_start_sha256"] for row in results],
        "seed_limit_preflights": [row["seed_limit_preflight"] for row in results],
        "certificate": certificate,
        "trajectories": np.asarray(trajectories, dtype=np.float32),
    }


def _run_benchmark_execution(instance, budget, strategy, proposal_seed, execution):
    start = time.perf_counter()
    model, trajectories, results, certificate, timings = solve_instance(
        instance,
        budget=budget,
        execution=execution,
        proposal_strategy=strategy,
        proposal_seed=proposal_seed,
    )
    wall_ms = 1e3 * (time.perf_counter() - start)
    return model, _benchmark_result_snapshot(
        trajectories, results, certificate, timings, wall_ms
    )


def _parity_snapshot(independent, batched):
    independent_trajectories = independent["trajectories"]
    batched_trajectories = batched["trajectories"]
    independent_objectives = np.asarray(independent["objectives"], dtype=np.float64)
    batched_objectives = np.asarray(batched["objectives"], dtype=np.float64)
    finite = np.isfinite(independent_objectives) & np.isfinite(batched_objectives)
    same_nonfinite_pattern = bool(
        np.array_equal(np.isfinite(independent_objectives), np.isfinite(batched_objectives))
        and np.array_equal(np.isnan(independent_objectives), np.isnan(batched_objectives))
        and np.array_equal(np.isposinf(independent_objectives), np.isposinf(batched_objectives))
        and np.array_equal(np.isneginf(independent_objectives), np.isneginf(batched_objectives))
    )
    finite_objective_delta = (
        batched_objectives[finite] - independent_objectives[finite]
    )
    return {
        "same_candidate_order": independent["candidate_names"] == batched["candidate_names"],
        "same_input_x0_hashes": independent["input_x0_sha256"] == batched["input_x0_sha256"],
        "same_input_reference_hashes": independent["input_reference_sha256"] == batched["input_reference_sha256"],
        "same_input_warm_start_hashes": independent["input_warm_start_sha256"] == batched["input_warm_start_sha256"],
        "same_certifiable_flags": independent["certifiable"] == batched["certifiable"],
        "same_modes": independent["modes"] == batched["modes"],
        "same_total_pcg_iterations": independent["total_pcg_iterations"] == batched["total_pcg_iterations"],
        "same_pcg_cap_hits": independent["pcg_cap_hits"] == batched["pcg_cap_hits"],
        "same_sqp_iterations": independent["sqp_iterations"] == batched["sqp_iterations"],
        "same_full_certificate": _jsonable(independent["certificate"]) == _jsonable(batched["certificate"]),
        "same_objective_nonfinite_pattern": same_nonfinite_pattern,
        "finite_objectives_allclose_rtol_1e-6_atol_1e-6": bool(
            np.allclose(
                independent_objectives[finite],
                batched_objectives[finite],
                rtol=1e-6,
                atol=1e-6,
            )
        ),
        "max_abs_trajectory_delta": float(
            np.max(np.abs(batched_trajectories - independent_trajectories), initial=0.0)
        ),
        "max_abs_objective_delta": float(
            np.max(np.abs(finite_objective_delta), initial=0.0)
        ),
        "trajectory_allclose_rtol_1e-5_atol_1e-6": bool(
            np.allclose(
                batched_trajectories,
                independent_trajectories,
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
            )
        ),
    }


def _timing_statistics(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {
        "samples_ms": values.tolist(),
        "median_ms": float(np.median(values)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
        "p95_ms": float(np.percentile(values, 95.0)),
    }


def _bootstrap_median_interval(values, *, seed=0, resamples=10000):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    medians = np.median(values[indices], axis=1)
    return {
        "resamples": int(resamples),
        "confidence": 0.95,
        "lower": float(np.percentile(medians, 2.5)),
        "upper": float(np.percentile(medians, 97.5)),
    }


def _best_for_mode(results, mode):
    candidates = [row for row in results if row["certifiable"] and row["mode"] == mode]
    return min(candidates, key=lambda row: row["objective"]) if candidates else None


def save_summary_plot(path, model, instance, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"clockwise": "#00693E", "counterclockwise": "#C90016", "no_winding": "#777777"}
    fig = plt.figure(figsize=(14.5, 4.8))
    ax_xy = fig.add_subplot(131)
    pillar = np.asarray(instance.pillar_xy)
    radius = instance.clearance_radius_m
    circle = plt.Circle(pillar, radius, color="#3B3B3B", alpha=0.32, label="keep-out pillar")
    ax_xy.add_patch(circle)
    seen_modes = set()
    for result in results:
        if not result["certifiable"]:
            continue
        path_xy = result["tool_path"][:, :2]
        mode = result["mode"]
        ax_xy.plot(
            path_xy[:, 0],
            path_xy[:, 1],
            color=colors[mode],
            alpha=0.8,
            linewidth=2.0,
            label=mode if mode not in seen_modes else None,
        )
        seen_modes.add(mode)
    start = np.asarray(instance.start_position)
    goal = np.asarray(instance.goal_position)
    ax_xy.plot(start[0], start[1], "o", color="#003192", markersize=8, label="start")
    ax_xy.plot(goal[0], goal[1], "*", color="#D97817", markersize=11, label="goal")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    support = {
        mode: sum(row["certifiable"] and row["mode"] == mode for row in results)
        for mode in ("clockwise", "counterclockwise")
    }
    ax_xy.set_title(
        "executable tool-center routes\n"
        f"{support['clockwise']} clockwise / {support['counterclockwise']} counterclockwise"
    )
    ax_xy.legend(loc="best", fontsize=8)
    ax_xy.grid(alpha=0.2)

    best_rows = [_best_for_mode(results, mode) for mode in ("clockwise", "counterclockwise")]
    for plot_index, (mode, result) in enumerate(
        zip(("clockwise", "counterclockwise"), best_rows), start=2
    ):
        ax = fig.add_subplot(1, 3, plot_index, projection="3d")
        theta = np.linspace(0.0, 2.0 * np.pi, 64)
        z = np.linspace(-0.75, 0.15, 2)
        cylinder_x = np.broadcast_to(
            pillar[0] + radius * np.cos(theta)[:, None], (theta.size, z.size)
        )
        cylinder_y = np.broadcast_to(
            pillar[1] + radius * np.sin(theta)[:, None], (theta.size, z.size)
        )
        cylinder_z = np.broadcast_to(z[None, :], (theta.size, z.size))
        ax.plot_surface(cylinder_x, cylinder_y, cylinder_z, color="#555555", alpha=0.20)
        if result is not None:
            q = result["q"]
            data = model.createData()
            for snapshot, alpha in zip(
                np.linspace(0, q.shape[0] - 1, 5, dtype=int),
                np.linspace(0.2, 1.0, 5),
            ):
                links = arm_link_positions(model, data, q[snapshot])
                ax.plot(
                    links[:, 0], links[:, 1], links[:, 2], "-o",
                    color=colors[mode], alpha=float(alpha), linewidth=2.0, markersize=2.5,
                )
            tool = result["tool_path"]
            ax.plot(tool[:, 0], tool[:, 1], tool[:, 2], color=colors[mode], linewidth=2.2)
            title = (
                f"{mode}\nsolver objective={result['objective']:.2f}"
            )
        else:
            title = f"{mode}\nnot covered"
        ax.scatter(*start, color="#003192", s=30)
        ax.scatter(*goal, color="#D97817", marker="*", s=60)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=22, azim=-62)

    certifiable = [row for row in results if row["certifiable"]]
    unchanged = bool(certifiable) and all(
        not row.get("solver_modified_calibrated_seed", False)
        for row in certifiable
    )
    initialization_note = (
        "SQP left every accepted proposal unchanged — initialization determines the result"
        if unchanged
        else "inspect reported seed-to-solution deltas for SQP initialization sensitivity"
    )
    fig.suptitle(
        f"Tiago tool-center pillar benchmark — {instance.difficulty}, seed {instance.seed}\n"
        f"{initialization_note}\n"
        "tool-center keep-out only; arm-link collision is not modeled"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _result_summary(result):
    excluded = {"tool_path", "q", "qd", "u", "pillar_xy"}
    return {key: _jsonable(value) for key, value in result.items() if key not in excluded}


def write_artifacts(
    output_dir,
    model,
    instance,
    trajectories,
    results,
    certificate,
    timings,
    *,
    evaluation_metadata=None,
    proposal_metadata=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    difficulty = DIFFICULTIES[instance.difficulty]
    external_submission = (
        timings.get("candidate_execution") == "external_submission_evaluation"
    )
    certifiable_rows = [row for row in results if row["certifiable"]]
    unchanged_rows = [
        row for row in certifiable_rows
        if not row.get("solver_modified_calibrated_seed", False)
    ]
    initialization_sensitivity = None
    if not external_submission:
        all_unchanged = bool(certifiable_rows) and len(unchanged_rows) == len(
            certifiable_rows
        )
        initialization_sensitivity = {
            "same_initial_state_reference_and_objective": True,
            "only_full_trajectory_initialization_varies": True,
            "certifiable_candidate_count": len(certifiable_rows),
            "certifiable_candidates_unchanged_by_sqp": len(unchanged_rows),
            "all_certifiable_candidates_unchanged_by_sqp": all_unchanged,
            "maximum_solution_delta_from_seed": max(
                (
                    row["solution_max_abs_delta_from_calibrated_seed"]
                    for row in certifiable_rows
                ),
                default=None,
            ),
            "maximum_reconstructed_objective_improvement": max(
                (
                    row["reconstructed_objective_improvement"]
                    for row in certifiable_rows
                ),
                default=None,
            ),
            "interpretation": (
                "SQP returned every certifiable proposal unchanged; trajectory "
                "initialization determines feasibility, winding mode, and best-known quality."
                if all_unchanged
                else "SQP changed at least one certifiable proposal; inspect per-candidate "
                "seed deltas and objective improvement."
            ),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "tiago_multimodal_infinite_pillar",
        "primary_benchmark_property": (
            "All built-in candidates share x0, reference, and objective; only the full "
            "trajectory initialization varies. Poor initialization can fail or retain a "
            "worse best-known solution."
            if not external_submission
            else "External trajectories are validated on one fixed x0, reference, and objective."
        ),
        "initialization_sensitivity": initialization_sensitivity,
        "claim": (
            "validation of externally submitted executable tool routes; stationarity "
            "and global optimality are not claimed"
            if external_submission
            else "finite-budget coverage of executable clockwise and counterclockwise "
            "tool routes by a route-informed proposal baseline; solver improvement, "
            "stationarity, and global optimality are not required or claimed"
        ),
        "baseline_note": (
            "External method trajectories were evaluated without running the built-in SQP baseline."
            if external_submission
            else "Built-in warm starts are dynamically executable proposals. Seed-to-solution "
            "deltas and reconstructed objective improvement show whether SQP changed them."
        ),
        "collision_scope": "tool-center clearance against an infinite-z pillar; arm-link collision is not modeled",
        "solver_convergence_note": (
            "The repository kkt_converged flag is not a KKT residual test. Acceptance instead "
            "requires candidate support, exact CUDA rollout, strict CUDA and independent "
            "shooting defects, dense tool clearance, target, and modeled limit checks."
        ),
        "limit_handling_note": (
            "The inherited log-barrier merit and derivative are inconsistent outside "
            "modeled limits. Built-in calibrated seeds are therefore rejected before SQP "
            "when any joint, velocity, or control limit is exceeded; accepted exact "
            "rollouts use strict ratio <= 1.0 gates."
        ),
        "reference_layout": [
            "goal_x", "goal_y", "goal_z", "pillar_x", "pillar_y", "clearance_radius",
        ],
        "instance": instance.metadata(),
        "difficulty": _jsonable(difficulty.__dict__),
        "benchmark_lane": difficulty.lane,
        "candidate_budget": len(results),
        "candidate_execution": timings.get("candidate_execution", "batched"),
        "solver_parameters": solver_parameters(difficulty),
        "provenance": _provenance(difficulty),
        "timings": timings,
        "pcg_cap_rate": float(
            sum(row["pcg_cap_hits"] for row in results)
            / max(sum(row["sqp_iterations"] for row in results), 1)
        ),
        "certificate": certificate,
        "results": [_result_summary(result) for result in results],
    }
    if proposal_metadata is not None:
        summary["proposal_metadata"] = _jsonable(proposal_metadata)
    if evaluation_metadata is not None:
        summary["evaluation_metadata"] = _jsonable(evaluation_metadata)
    with (output_dir / "instance.json").open("w", encoding="utf-8") as stream:
        json.dump(
            _jsonable(instance.metadata()), stream, indent=2, sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            _jsonable(summary), stream, indent=2, sort_keys=True, allow_nan=False
        )
        stream.write("\n")
    rollout_trajectories = np.asarray(
        [
            pack_trajectory(row["q"], row["qd"], row["u"], model, difficulty.knots)
            for row in results
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        output_dir / "trajectories.npz",
        planned_packed_trajectories=np.asarray(trajectories, dtype=np.float32),
        exact_rollout_packed_trajectories=rollout_trajectories,
        candidate_names=np.asarray([row["candidate_name"] for row in results]),
    )
    save_summary_plot(output_dir / "modes.png", model, instance, results)
    return summary


def run_instance(
    instance,
    budget,
    output_root,
    *,
    write=True,
    execution="independent",
    proposal_strategy="informed",
    proposal_seed=None,
):
    difficulty_name = instance.difficulty
    difficulty = DIFFICULTIES[difficulty_name]
    model = load_model(REPO_ROOT / MODEL_PATH)
    model, trajectories, results, certificate, timings = solve_instance(
        instance,
        budget=budget,
        execution=execution,
        proposal_strategy=proposal_strategy,
        proposal_seed=proposal_seed,
    )
    output_dir = (
        output_root
        / difficulty_name
        / f"seed_{instance.seed:06d}"
        / f"budget_{budget:02d}"
    )
    summary = None
    if write:
        resolved_proposal_seed = (
            instance.seed if proposal_seed is None else int(proposal_seed)
        )
        summary = write_artifacts(
            output_dir,
            model,
            instance,
            trajectories,
            results,
            certificate,
            timings,
            proposal_metadata={
                "strategy": proposal_strategy,
                "seed": resolved_proposal_seed,
                "seed_was_implicit_instance_seed": proposal_seed is None,
            },
        )
    return output_dir, instance, certificate, summary


def run_seed(
    seed,
    difficulty_name,
    budget,
    output_root,
    *,
    write=True,
    execution="independent",
    proposal_strategy="informed",
    proposal_seed=None,
):
    difficulty = DIFFICULTIES[difficulty_name]
    model = load_model(REPO_ROOT / MODEL_PATH)
    instance = generate_instance(seed, difficulty, model=model)
    return run_instance(
        instance,
        budget,
        output_root,
        write=write,
        execution=execution,
        proposal_strategy=proposal_strategy,
        proposal_seed=proposal_seed,
    )


def load_instance_file(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "instance" in payload:
        payload = payload["instance"]
    instance = PillarInstance.from_metadata(payload)
    if instance.difficulty not in DIFFICULTIES:
        raise ValueError(f"unsupported difficulty: {instance.difficulty}")
    model = load_model(REPO_ROOT / MODEL_PATH)
    # The generator intentionally perturbs starts, so validate directly instead
    # of requiring the seed to regenerate the same instance.
    actual_start = tool_position(model, model.createData(), instance.start_q)
    if not np.allclose(actual_start, instance.start_position, atol=1e-6):
        raise ValueError("instance start_position does not match start_q and model")
    return instance


def run_command(args):
    if args.instance_file:
        instance = load_instance_file(args.instance_file)
        output_dir, _, certificate, summary = run_instance(
            instance,
            args.candidate_budget,
            args.output_root,
            execution=args.execution,
            proposal_strategy=args.proposal_strategy,
            proposal_seed=args.proposal_seed,
        )
    else:
        output_dir, _, certificate, summary = run_seed(
            args.seed, args.difficulty, args.candidate_budget, args.output_root,
            execution=args.execution,
            proposal_strategy=args.proposal_strategy,
            proposal_seed=args.proposal_seed,
        )
    print(f"artifacts: {output_dir}")
    print(f"PRIMARY BENCHMARK PROPERTY: {summary['primary_benchmark_property']}")
    if summary["initialization_sensitivity"]:
        print(summary["initialization_sensitivity"]["interpretation"])
    print(json.dumps(_jsonable(certificate), indent=2, sort_keys=True, allow_nan=False))
    if args.require_two_mode and not certificate["finite_budget_two_mode_coverage"]:
        raise SystemExit("instance did not cover two executable winding modes")
    if args.require_ranked and not certificate["ranked_finite_budget_benchmark"]:
        raise SystemExit("instance did not pass ranked finite-budget acceptance")


def compare_command(args):
    rows = []
    for budget in (1, 2, 4, 8):
        output_dir, _, certificate, summary = run_seed(
            args.seed, args.difficulty, budget, args.output_root, write=True
        )
        rows.append(
            {
                "candidate_budget": budget,
                "output_dir": str(output_dir),
                "executable_mode_count": sum(
                    count > 0 for count in certificate["certifiable_support"].values()
                ),
                "best_known_solver_objective": certificate[
                    "best_known_solver_objective"
                ],
                "task_motion_cost_at_best_known": certificate[
                    "task_motion_cost_at_best_known"
                ],
                "best_known_mode": certificate["best_known_mode"],
                "solver_wall_time_ms": summary["timings"]["solver_wall_time_ms"],
                "total_wall_time_ms": summary["timings"][
                    "benchmark_total_ms_excluding_render"
                ],
            }
        )
    oracle = rows[-1]["best_known_solver_objective"]
    oracle_scale = rows[-1]["task_motion_cost_at_best_known"]
    for row in rows:
        value = row["best_known_solver_objective"]
        row["normalized_regret_to_budget_8"] = (
            None if value is None or oracle is None
            else (value - oracle) / max(float(oracle_scale), 1.0)
        )
    comparison = {
        "seed": args.seed,
        "difficulty": args.difficulty,
        "oracle_note": (
            "budget 8 is a route-template reference, not an independent oracle or "
            "proven global optimum"
        ),
        "budgets": rows,
    }
    path = (
        args.output_root / args.difficulty / f"seed_{args.seed:06d}" / "comparison.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            _jsonable(comparison), stream, indent=2, sort_keys=True, allow_nan=False
        )
        stream.write("\n")
    print(f"comparison: {path}")
    print(json.dumps(_jsonable(comparison), indent=2, sort_keys=True, allow_nan=False))


def suite_command(args):
    if args.candidate_budget != 8:
        raise SystemExit("suite certification requires --candidate-budget 8")
    accepted = []
    attempted = 0
    seed = args.seed_start
    while len(accepted) < args.count and attempted < args.max_attempts:
        output_dir, instance, certificate, summary = run_seed(
            seed, args.difficulty, args.candidate_budget, args.output_root,
            write=True,
        )
        attempted += 1
        if certificate["ranked_finite_budget_benchmark"]:
            instance_bytes = json.dumps(
                instance.metadata(), sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            split_bucket = hashlib.sha256(instance_bytes).digest()[0]
            split = (
                "train" if split_bucket < 179
                else ("validation" if split_bucket < 218 else "test")
            )
            summary_path = output_dir / "summary.json"
            accepted.append(
                {
                    "seed": seed,
                    "split": split,
                    "output_dir": str(output_dir),
                    "summary_sha256": _sha256_file(summary_path),
                    "certificate": certificate,
                }
            )
            print(f"accepted {len(accepted)}/{args.count}: seed {seed}")
        else:
            print(
                f"rejected seed {seed}: support={certificate['certifiable_support']} "
                f"gap={certificate['mode_objective_gap_fraction']}"
            )
        seed += 1
    suite = {
        "schema_version": SCHEMA_VERSION,
        "difficulty": args.difficulty,
        "requested_count": args.count,
        "attempted_count": attempted,
        "accepted_count": len(accepted),
        "accepted": accepted,
    }
    suite_path = args.output_root / args.difficulty / "suite.json"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    with suite_path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(suite), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"suite: {suite_path}")
    if len(accepted) != args.count:
        raise SystemExit(
            f"only accepted {len(accepted)}/{args.count} instances in {attempted} attempts"
        )


def evaluate_command(args):
    """Evaluate method-produced trajectories without running the built-in seeds."""

    instance = load_instance_file(args.instance_file)
    difficulty = DIFFICULTIES[instance.difficulty]
    model = load_model(REPO_ROOT / MODEL_PATH)
    archive = np.load(args.trajectories, allow_pickle=False)
    key = (
        "planned_packed_trajectories"
        if "planned_packed_trajectories" in archive
        else "packed_trajectories"
    )
    if key not in archive:
        raise ValueError(
            "trajectory archive needs planned_packed_trajectories or packed_trajectories"
        )
    trajectories = np.atleast_2d(np.asarray(archive[key], dtype=np.float32))
    expected_width = difficulty.knots * (model.nq + 2 * model.nv) - model.nv
    if trajectories.shape[1] != expected_width:
        raise ValueError(
            f"expected packed trajectory width {expected_width}, got {trajectories.shape[1]}"
        )
    if "candidate_names" in archive:
        names = [str(value) for value in archive["candidate_names"]]
    else:
        names = [f"submission_{index:03d}" for index in range(len(trajectories))]
    if len(names) != len(trajectories):
        raise ValueError("candidate_names length does not match trajectory count")
    if args.proposal_budget != len(trajectories):
        raise ValueError(
            f"declared proposal budget {args.proposal_budget} does not match "
            f"{len(trajectories)} submitted trajectories"
        )
    if args.method_runtime_ms < 0.0:
        raise ValueError("method runtime must be nonnegative")

    solver = BSQP(
        model_path=str(REPO_ROOT / MODEL_PATH),
        batch_size=1,
        N=difficulty.knots,
        dt=difficulty.dt,
        plant_type="tiago_right_multimodal",
        **solver_parameters(difficulty),
    )
    x0 = np.hstack(
        [np.asarray(instance.start_q, dtype=np.float32), np.zeros(model.nv, dtype=np.float32)]
    )
    results = []
    evaluation_start = time.perf_counter()
    for index, (name, trajectory) in enumerate(zip(names, trajectories)):
        planned_q, planned_qd, controls = unpack_trajectory(
            trajectory, model, difficulty.knots
        )
        independent_defect = dynamics_defect(
            model, planned_q, planned_qd, controls, difficulty.dt
        )
        rollout, cuda_defect = _rollout_and_gpu_defect(
            solver, model, difficulty, trajectory, x0
        )
        rollout_q, rollout_qd, _ = unpack_trajectory(
            rollout, model, difficulty.knots
        )
        planned_state = np.hstack([planned_q, planned_qd])
        rollout_states = np.hstack([rollout_q, rollout_qd])
        planned_initial_error = float(np.linalg.norm(planned_state[0] - x0))
        planned_to_rollout_error = float(
            np.max(np.linalg.norm(planned_state - rollout_states, axis=1), initial=0.0)
        )
        result = evaluate_trajectory(model, instance, difficulty, rollout)
        result.update(
            candidate_index=index,
            candidate_name=name,
            requested_route_side=None,
            final_solver_merit=None,
            sqp_iterations=0,
            total_pcg_iterations=0,
            pcg_cap_hits=0,
            pcg_cap_free=True,
            solver_zero_pcg_exit=False,
            rollout_finite=bool(np.all(np.isfinite(rollout))),
            max_cuda_planned_dynamics_defect=float(cuda_defect),
            max_independent_planned_dynamics_defect=float(independent_defect),
            planned_initial_state_error=planned_initial_error,
            max_planned_state_to_rollout_error=planned_to_rollout_error,
            submission_source="external_trajectory_archive",
        )
        result["certifiable"] = bool(
            result["physical_feasible"]
            and result["rollout_finite"]
            and planned_initial_error <= 1e-6
            and planned_to_rollout_error <= 1e-3
            and cuda_defect <= 1e-4
            and independent_defect <= 1e-3
        )
        results.append(result)
    certificate = certify_results(results, min_support_per_mode=2)
    evaluation_metadata = {
        "method_name": args.method_name,
        "declared_method_runtime_ms": float(args.method_runtime_ms),
        "proposal_budget": int(args.proposal_budget),
        "instance_file_sha256": _sha256_file(args.instance_file),
        "trajectory_archive_sha256": _sha256_file(args.trajectories),
        "oracle_summary": str(args.oracle_summary) if args.oracle_summary else None,
        "oracle_summary_sha256": (
            _sha256_file(args.oracle_summary) if args.oracle_summary else None
        ),
        "normalized_regret_to_oracle": None,
    }
    if args.oracle_summary:
        oracle = json.loads(args.oracle_summary.read_text(encoding="utf-8"))
        if oracle.get("instance") != _jsonable(instance.metadata()):
            raise ValueError("oracle summary describes a different instance")
        oracle_certificate = oracle["certificate"]
        oracle_objective = oracle_certificate["best_known_solver_objective"]
        method_objective = certificate["best_known_solver_objective"]
        oracle_scale = oracle_certificate["task_motion_cost_at_best_known"]
        if oracle_objective is not None and method_objective is not None:
            evaluation_metadata["normalized_regret_to_oracle"] = float(
                (method_objective - oracle_objective) / max(float(oracle_scale), 1.0)
            )
    timings = {
        "evaluation_ms": 1e3 * (time.perf_counter() - evaluation_start),
        "benchmark_total_ms_excluding_render": 1e3
        * (time.perf_counter() - evaluation_start),
        "candidate_execution": "external_submission_evaluation",
    }
    summary = write_artifacts(
        args.output_dir,
        model,
        instance,
        trajectories,
        results,
        certificate,
        timings,
        evaluation_metadata=evaluation_metadata,
    )
    print(f"evaluation: {args.output_dir}")
    print(
        json.dumps(
            _jsonable(summary["certificate"]), indent=2, sort_keys=True,
            allow_nan=False,
        )
    )


def batch_benchmark_command(args):
    """Benchmark one batched solve against exact-candidate batch-1 solves."""

    if args.repeats < 1 or args.warmup_repeats < 0:
        raise ValueError("repeats must be positive and warmup repeats nonnegative")
    difficulty = DIFFICULTIES[args.difficulty]
    model = load_model(REPO_ROOT / MODEL_PATH)
    instance = generate_instance(args.seed, difficulty, model=model)
    expected_names = [
        spec.name
        for spec in _proposal_specs(
            instance,
            args.candidate_budget,
            args.proposal_strategy,
            args.proposal_seed,
        )
    ]

    # Warm both paths independently. Measured first position is balanced and
    # deterministically shuffled so load cannot consistently favor one path.
    for _ in range(args.warmup_repeats):
        for execution in ("independent", "batched"):
            _run_benchmark_execution(
                instance,
                args.candidate_budget,
                args.proposal_strategy,
                args.proposal_seed,
                execution,
            )

    raw_repeats = []
    measured_by_execution = {"independent": [], "batched": []}
    first_execution = np.asarray(
        ["independent", "batched"] * ((args.repeats + 1) // 2), dtype=object
    )[: args.repeats]
    order_rng = np.random.default_rng(
        np.random.SeedSequence([args.seed, args.proposal_seed, args.candidate_budget])
    )
    order_rng.shuffle(first_execution)
    for repeat in range(args.repeats):
        first = str(first_execution[repeat])
        second = "batched" if first == "independent" else "independent"
        order = (first, second)
        paired = {}
        for execution in order:
            _, snapshot = _run_benchmark_execution(
                instance,
                args.candidate_budget,
                args.proposal_strategy,
                args.proposal_seed,
                execution,
            )
            if snapshot["candidate_names"] != expected_names:
                raise AssertionError("candidate order changed between benchmark executions")
            paired[execution] = snapshot
            measured_by_execution[execution].append(snapshot)
        parity = _parity_snapshot(paired["independent"], paired["batched"])
        serializable_pair = {}
        for execution, snapshot in paired.items():
            serializable_pair[execution] = {
                key: value for key, value in snapshot.items() if key != "trajectories"
            }
        raw_repeats.append(
            {
                "repeat": repeat,
                "execution_order": list(order),
                "runs": serializable_pair,
                "parity": parity,
            }
        )

    timing_summary = {}
    for execution, snapshots in measured_by_execution.items():
        timing_summary[execution] = {
            "end_to_end": _timing_statistics(
                [snapshot["wall_time_ms"] for snapshot in snapshots]
            ),
            "solver_wall": _timing_statistics(
                [snapshot["reported_timings"]["solver_wall_time_ms"] for snapshot in snapshots]
            ),
            "candidate_generation_calibration": _timing_statistics(
                [
                    snapshot["reported_timings"]["geometric_warm_start_generation_ms"]
                    + snapshot["reported_timings"]["cuda_dynamics_calibration_ms"]
                    for snapshot in snapshots
                ]
            ),
            "evaluation": _timing_statistics(
                [snapshot["reported_timings"]["evaluation_ms"] for snapshot in snapshots]
            ),
            "unique_trajectory_hashes": sorted(
                {snapshot["trajectory_sha256"] for snapshot in snapshots}
            ),
        }
    independent_median = timing_summary["independent"]["end_to_end"]["median_ms"]
    batched_median = timing_summary["batched"]["end_to_end"]["median_ms"]
    independent_solver_median = timing_summary["independent"]["solver_wall"]["median_ms"]
    batched_solver_median = timing_summary["batched"]["solver_wall"]["median_ms"]
    independent_wall = np.asarray(
        [snapshot["wall_time_ms"] for snapshot in measured_by_execution["independent"]],
        dtype=np.float64,
    )
    batched_wall = np.asarray(
        [snapshot["wall_time_ms"] for snapshot in measured_by_execution["batched"]],
        dtype=np.float64,
    )
    paired_delta_ms = independent_wall - batched_wall
    paired_reduction_fraction = paired_delta_ms / independent_wall
    summary = {
        "schema_version": 1,
        "benchmark": "tiago_multimodal_exact_candidate_batch_ab",
        "instance": instance.metadata(),
        "difficulty": args.difficulty,
        "candidate_budget": args.candidate_budget,
        "candidate_names": expected_names,
        "proposal_strategy": args.proposal_strategy,
        "proposal_seed": args.proposal_seed,
        "initializer_distribution": {
            "random-joint": (
                "cold, straight task-space IK, then fixed-seed smooth antithetic "
                "joint-space perturbations independent of pillar geometry"
            ),
            "random-task": (
                "cold, straight task-space IK, then fixed-seed smooth antithetic "
                "task-space bumps transverse to start-goal travel and independent of "
                "pillar geometry"
            ),
            "informed": "fixed route-informed reference candidate family",
        }[args.proposal_strategy],
        "warmup_repeats_per_execution": args.warmup_repeats,
        "measured_repeats_per_execution": args.repeats,
        "solver_parameters": solver_parameters(difficulty),
        "provenance": _provenance(difficulty),
        "timings": timing_summary,
        "median_end_to_end_speedup": float(independent_median / batched_median),
        "median_solver_speedup": float(independent_solver_median / batched_solver_median),
        "paired_end_to_end_improvement": {
            "delta_ms_independent_minus_batched": paired_delta_ms.tolist(),
            "reduction_fraction": paired_reduction_fraction.tolist(),
            "median_delta_ms": float(np.median(paired_delta_ms)),
            "median_reduction_fraction": float(np.median(paired_reduction_fraction)),
            "bootstrap_95_ci_median_delta_ms": _bootstrap_median_interval(
                paired_delta_ms, seed=0
            ),
            "bootstrap_95_ci_median_reduction_fraction": _bootstrap_median_interval(
                paired_reduction_fraction, seed=1
            ),
        },
        "all_repeats_candidate_and_certificate_parity": bool(
            all(
                row["parity"]["same_candidate_order"]
                and row["parity"]["same_input_x0_hashes"]
                and row["parity"]["same_input_reference_hashes"]
                and row["parity"]["same_input_warm_start_hashes"]
                and row["parity"]["same_certifiable_flags"]
                and row["parity"]["same_modes"]
                and row["parity"]["same_total_pcg_iterations"]
                and row["parity"]["same_pcg_cap_hits"]
                and row["parity"]["same_sqp_iterations"]
                and row["parity"]["same_full_certificate"]
                and row["parity"]["same_objective_nonfinite_pattern"]
                and row["parity"]["finite_objectives_allclose_rtol_1e-6_atol_1e-6"]
                for row in raw_repeats
            )
        ),
        "all_repeats_trajectory_allclose": bool(
            all(
                row["parity"]["trajectory_allclose_rtol_1e-5_atol_1e-6"]
                for row in raw_repeats
            )
        ),
        "raw_repeats": raw_repeats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(summary), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"benchmark: {args.output}")
    print(
        json.dumps(
            {
                "median_end_to_end_speedup": summary["median_end_to_end_speedup"],
                "median_solver_speedup": summary["median_solver_speedup"],
                "all_repeats_candidate_and_certificate_parity": summary[
                    "all_repeats_candidate_and_certificate_parity"
                ],
                "all_repeats_trajectory_allclose": summary[
                    "all_repeats_trajectory_allclose"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="solve and visualize one generated instance")
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--difficulty", choices=tuple(DIFFICULTIES), default="medium")
    run.add_argument("--candidate-budget", type=int, choices=(1, 2, 4, 8), default=8)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--instance-file", type=Path)
    run.add_argument("--execution", choices=("independent", "batched"), default="independent")
    run.add_argument(
        "--proposal-strategy",
        choices=("informed", "random-joint", "random-task"),
        default="informed",
    )
    run.add_argument("--proposal-seed", type=int)
    run.add_argument("--require-two-mode", action="store_true")
    run.add_argument("--require-ranked", action="store_true")
    run.set_defaults(func=run_command)

    compare = subparsers.add_parser(
        "compare", help="compare candidate budgets 1, 2, 4, and 8 on one instance"
    )
    compare.add_argument("--seed", type=int, default=17)
    compare.add_argument("--difficulty", choices=tuple(DIFFICULTIES), default="easy")
    compare.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    compare.set_defaults(func=compare_command)

    suite = subparsers.add_parser(
        "suite", help="generate-and-reject until a certified family is collected"
    )
    suite.add_argument("--count", type=int, default=10)
    suite.add_argument("--seed-start", type=int, default=1000)
    suite.add_argument("--max-attempts", type=int, default=100)
    suite.add_argument("--difficulty", choices=tuple(DIFFICULTIES), default="medium")
    suite.add_argument("--candidate-budget", type=int, choices=(1, 2, 4, 8), default=8)
    suite.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    suite.set_defaults(func=suite_command)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate externally generated packed trajectories"
    )
    evaluate.add_argument("--instance-file", type=Path, required=True)
    evaluate.add_argument("--trajectories", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--method-name", required=True)
    evaluate.add_argument("--method-runtime-ms", type=float, required=True)
    evaluate.add_argument("--proposal-budget", type=int, required=True)
    evaluate.add_argument("--oracle-summary", type=Path)
    evaluate.set_defaults(func=evaluate_command)

    benchmark = subparsers.add_parser(
        "batch-benchmark",
        help="measure one batched solve against exact-candidate independent solves",
    )
    benchmark.add_argument("--seed", type=int, default=30)
    benchmark.add_argument("--difficulty", choices=tuple(DIFFICULTIES), default="easy")
    benchmark.add_argument(
        "--candidate-budget", type=int, choices=(1, 2, 4, 8), default=8
    )
    benchmark.add_argument(
        "--proposal-strategy",
        choices=("informed", "random-joint", "random-task"),
        default="random-task",
    )
    benchmark.add_argument("--proposal-seed", type=int, default=1000)
    benchmark.add_argument("--warmup-repeats", type=int, default=1)
    benchmark.add_argument("--repeats", type=int, default=20)
    benchmark.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT / "batch_benchmark.json",
    )
    benchmark.set_defaults(func=batch_benchmark_command)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
