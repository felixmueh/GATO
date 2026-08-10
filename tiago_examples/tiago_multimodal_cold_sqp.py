#!/usr/bin/env python3
"""Run one productive Tiago cold-multistart pillar experiment.

The output is optimized controls replayed from common x0.  Packed planned
states and their shooting defects are retained as report-only telemetry.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "python"), str(REPO_ROOT / "tiago_src")]

from bsqp.interface import BSQP
from gato_tiago.multimodal_cold_sqp import (
    BUDGET,
    CENTRAL_FINAL_TASK_SEED,
    DEVELOPMENT_METHOD_SEEDS,
    DEVELOPMENT_PROTOCOL,
    DEVELOPMENT_TASK_SEEDS,
    FINAL_METHOD_SEEDS,
    FINAL_PROTOCOL,
    FINAL_TASK_SEEDS,
    candidate_labels,
    certify_batch,
    certify_mode_support,
    generate_cold_instance,
    generate_controls,
    model_pair_preflight,
    mode_certificates_agree,
    one_pass_rollout_and_pack,
    require_seed_screen_before_sqp,
    seed_rollout_screen,
)
from gato_tiago.multimodal_pillar import (
    MODEL_PATH,
    load_model,
    reference_batch,
    unpack_trajectory,
)


DEFAULT_OUTPUT = REPO_ROOT / "example_artifacts/tiago_multimodal_cold_sqp/probe.json"


def protocol_for_phase(phase):
    if phase == "development":
        return DEVELOPMENT_PROTOCOL
    if FINAL_PROTOCOL is None:
        raise RuntimeError("final protocol is not frozen; final data collection is disabled")
    return FINAL_PROTOCOL


def allowed_seeds(phase):
    if phase == "development":
        return set(DEVELOPMENT_TASK_SEEDS), set(DEVELOPMENT_METHOD_SEEDS)
    return set(FINAL_TASK_SEEDS), set(FINAL_METHOD_SEEDS)


def solver_parameters(protocol):
    difficulty = protocol.difficulty
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
        "ee_orient_cost": difficulty.pillar_cost,
        "ee_orient_N_cost": difficulty.pillar_cost,
        "q_lim_cost": 0.01,
        "vel_lim_cost": 0.001,
        "ctrl_lim_cost": 0.003,
        "rho": 0.01,
    }


def make_solver(batch_size, protocol):
    difficulty = protocol.difficulty
    return BSQP(
        model_path=str(MODEL_PATH),
        batch_size=batch_size,
        N=difficulty.knots,
        dt=difficulty.dt,
        plant_type="tiago_right_multimodal",
        **solver_parameters(protocol),
    )


def sha256_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def stats_snapshot(solver, protocol):
    stats = solver.get_stats()
    pcg = np.asarray(stats.get("pcg_iters", []), dtype=np.int32)
    return {
        "initial_merit": np.asarray(stats["initial_merit"], dtype=np.float64),
        "final_merit": np.asarray(stats["final_merit"], dtype=np.float64),
        "sqp_iterations": np.asarray(stats["sqp_iters"], dtype=np.int32),
        "total_pcg_iterations": np.sum(pcg, axis=0, dtype=np.int64),
        "pcg_cap_hits": np.sum(
            pcg >= protocol.difficulty.max_pcg_iters, axis=0, dtype=np.int64
        ),
    }


def solve_batch(solver, x0_batch, references, seeds, protocol):
    workflow_started = time.perf_counter()
    solver.reset_dual()
    solver.reset_rho()
    seed_input = seeds.copy()
    started = time.perf_counter()
    output, native_us = solver.solve(x0_batch, references, seed_input)
    solve_ms = 1e3 * (time.perf_counter() - started)
    stats = stats_snapshot(solver, protocol)
    return (
        output,
        stats,
        solve_ms,
        float(native_us) / 1e3,
        1e3 * (time.perf_counter() - workflow_started),
    )


def solve_serial(solver, x0_batch, references, seeds, protocol):
    workflow_started = time.perf_counter()
    outputs, snapshots = [], []
    solve_ms = native_ms = 0.0
    for candidate in range(len(seeds)):
        solver.reset_dual()
        solver.reset_rho()
        x0 = x0_batch[candidate : candidate + 1]
        reference = references[candidate : candidate + 1]
        seed = seeds[candidate : candidate + 1].copy()
        started = time.perf_counter()
        output, native_us = solver.solve(x0, reference, seed)
        solve_ms += 1e3 * (time.perf_counter() - started)
        native_ms += float(native_us) / 1e3
        outputs.append(output[0])
        snapshots.append(stats_snapshot(solver, protocol))
    stats = {
        key: np.concatenate([snapshot[key] for snapshot in snapshots])
        for key in snapshots[0]
    }
    return (
        np.asarray(outputs, dtype=np.float32),
        stats,
        solve_ms,
        native_ms,
        1e3 * (time.perf_counter() - workflow_started),
    )


def timing_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "samples_ms": values.tolist(),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95.0)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
    }


def bootstrap_median(values, seed=20260811, resamples=10000):
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
        values = np.asarray([row[f"{method}_{name}"] for row in raw])
        span = np.max(values, axis=0) - np.min(values, axis=0)
        scale = np.maximum(np.max(np.abs(values), axis=0), 1.0)
        result[name] = {
            "maximum_absolute_span": float(np.max(span)),
            "maximum_relative_span": float(np.max(span / scale)),
        }
    return result


def repeatability(raw, method):
    merits = merit_repeatability(raw, method)
    exact_telemetry = all(
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
    output_hash_count = len({row[f"{method}_output_sha256"] for row in raw})
    passed = bool(
        output_hash_count == 1
        and exact_telemetry
        and merits["initial_merit"]["maximum_relative_span"] <= 1e-6
        and merits["final_merit"]["maximum_relative_span"] <= 1e-6
    )
    return {
        "output_hash_count": output_hash_count,
        "exact_sqp_pcg_and_cap_telemetry": exact_telemetry,
        "merit_spans": merits,
        "passes": passed,
    }


def final_valid(row):
    return bool(
        all(
            metrics["finite"]
            and metrics["joint_violation_rad"] <= 0.0
            and metrics["velocity_ratio"] <= 1.0
            and metrics["control_ratio"] <= 1.0
            and metrics["maximum_clearance_violation_m"] <= 0.0
            for metrics in (row["final_cuda"], row["final_pinocchio"])
        )
        and row["final_cuda_one_step_pinocchio_defect"] <= 1e-3
    )


def numerical_noninferiority(serial_rows, batch_rows, serial_output, batch_output):
    rows = []
    for serial, batch in zip(serial_rows, batch_rows):
        initial_relative = abs(
            batch["initial_solver_merit"] - serial["initial_solver_merit"]
        ) / max(abs(serial["initial_solver_merit"]), 1.0)
        serial_cost = serial["external_cost"]
        row = {
            "candidate_index": serial["candidate_index"],
            "initial_merit_relative_difference": float(initial_relative),
            "serial_productive_success_preserved": bool(
                not serial["productive_certified"]
                or batch["productive_certified"]
            ),
            "batch_terminal_within_serial_plus_2mm": bool(
                batch["final_cuda"]["terminal_error_m"]
                <= serial["final_cuda"]["terminal_error_m"] + 0.002
            ),
            "batch_external_cost_noninferior": bool(
                batch["external_cost"]
                <= serial_cost + max(1e-3, 0.01 * max(serial_cost, 1.0))
            ),
            "batch_has_no_extra_cap": bool(
                batch["pcg_cap_hits"] <= serial["pcg_cap_hits"]
            ),
            "batch_has_no_added_invalidity": bool(
                not final_valid(serial) or final_valid(batch)
            ),
            "max_abs_serial_batch_planned_delta": float(
                np.max(
                    np.abs(
                        serial_output[serial["candidate_index"]]
                        - batch_output[serial["candidate_index"]]
                    )
                )
            ),
        }
        row["passes"] = bool(
            initial_relative <= 1e-6
            and row["serial_productive_success_preserved"]
            and row["batch_terminal_within_serial_plus_2mm"]
            and row["batch_external_cost_noninferior"]
            and row["batch_has_no_extra_cap"]
            and row["batch_has_no_added_invalidity"]
        )
        rows.append(row)
    return rows


def select_winner(rows):
    successful = [row for row in rows if row["productive_certified"]]
    return min(successful, key=lambda row: row["external_cost"], default=None)


def selected_winner_noninferior(serial, batch):
    if serial is None:
        return True
    if batch is None:
        return False
    return bool(
        batch["final_cuda"]["terminal_error_m"]
        <= serial["final_cuda"]["terminal_error_m"] + 0.002
        and batch["external_cost"]
        <= serial["external_cost"]
        + max(1e-3, 0.01 * max(serial["external_cost"], 1.0))
    )


def strip_arrays(value):
    if isinstance(value, dict):
        return {
            key: strip_arrays(item)
            for key, item in value.items()
            if key not in ("controls", "cuda_tool_path", "pinocchio_tool_path", "dense_tool_path")
        }
    if isinstance(value, list):
        return [strip_arrays(item) for item in value]
    return value


def run(args):
    protocol = protocol_for_phase(args.phase)
    task_seeds, method_seeds = allowed_seeds(args.phase)
    if args.task_seed not in task_seeds or args.method_seed not in method_seeds:
        raise ValueError("task/method seed is outside the declared phase set")
    if args.budget != BUDGET:
        raise ValueError(f"protocol freezes budget at {BUDGET}")
    model = load_model(MODEL_PATH)
    instance, task_generation = generate_cold_instance(
        args.task_seed, protocol, model=model
    )
    q0 = np.asarray(instance.start_q, dtype=np.float32)
    x0 = np.hstack([q0, np.zeros(model.nv, dtype=np.float32)])
    x0_batch = np.tile(x0, (BUDGET, 1)).astype(np.float32)
    references = reference_batch(
        instance, protocol.difficulty.knots, BUDGET
    )
    batch_solver = make_solver(BUDGET, protocol)
    preflight = model_pair_preflight(
        batch_solver.sim_forward,
        model,
        q0,
        protocol.difficulty.dt,
    )
    if not preflight["passes_1e-3_gate"]:
        raise RuntimeError("Tiago CUDA/Pinocchio model-pair preflight failed")
    generation_started = time.perf_counter()
    controls, control_metadata = generate_controls(
        model, q0, args.method_seed, protocol
    )
    seeds, rollout_metadata = one_pass_rollout_and_pack(
        batch_solver.sim_forward, x0, controls, protocol
    )
    generation_ms = 1e3 * (time.perf_counter() - generation_started)
    seed_cuda_states = np.asarray(
        [
            np.hstack(unpack_trajectory(row, model, protocol.difficulty.knots)[:2])
            for row in seeds
        ],
        dtype=np.float32,
    )
    screen_rows, screen_aggregate, screen_arrays = seed_rollout_screen(
        model,
        instance,
        protocol,
        x0,
        controls,
        batch_solver.sim_forward,
        preflight,
        cuda_states=seed_cuda_states,
    )
    if args.seed_screen_only:
        extension = Path(batch_solver.lib.__file__)
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        summary = {
            "schema_version": 1,
            "diagnostic": "tiago_task_neutral_seed_full_horizon_screen",
            "phase": args.phase,
            "task_seed": args.task_seed,
            "method_seed": args.method_seed,
            "sqps_or_optimization_calls": 0,
            "protocol": asdict(protocol),
            "instance": instance.metadata(),
            "task_generation": task_generation,
            "candidate_generation": {
                **control_metadata,
                **rollout_metadata,
                "generation_wall_ms": generation_ms,
                "pinocchio_open_loop_rollout_count": BUDGET,
                "cuda_open_loop_rollout_count": BUDGET,
                "cuda_seed_rollout_reused_from_packed_seed": True,
                "description": (
                    "stationary inverse-dynamics hold plus seven antithetic "
                    "analytic sin-squared joint pairs and one independent draw; "
                    "one RNEA call per candidate/control knot and exactly one "
                    "open-loop rollout per replay model; no reference, pillar, "
                    "collision, IK, feedback, calibration, optimization, "
                    "rejection, or resampling"
                ),
            },
            "candidate_labels": candidate_labels(),
            "model_pair_preflight": preflight,
            "seed_screen": strip_arrays(screen_rows),
            "seed_screen_aggregate": screen_aggregate,
            "input_hashes": {
                "x0_batch": sha256_array(x0_batch),
                "reference_batch_report_only": sha256_array(references),
                "seed_controls": sha256_array(controls),
                "seed_trajectories": sha256_array(seeds),
            },
            "provenance": {
                "git_head": git_head,
                "git_tracked_dirty": tracked_dirty,
                "model_sha256": sha256_file(MODEL_PATH),
                "extension": str(extension),
                "extension_sha256": sha256_file(extension),
                "source_sha256": {
                    "module": sha256_file(
                        REPO_ROOT / "tiago_src/gato_tiago/multimodal_cold_sqp.py"
                    ),
                    "harness": sha256_file(Path(__file__).resolve()),
                },
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            x0=x0,
            references_report_only=references,
            seed_trajectories=seeds,
            **screen_arrays,
        )
        print(args.output)
        print(json.dumps(screen_aggregate, indent=2, sort_keys=True))
        return
    require_seed_screen_before_sqp(screen_aggregate)
    serial_solver = make_solver(1, protocol)
    methods = {
        "serial": lambda: solve_serial(
            serial_solver, x0_batch, references, seeds, protocol
        ),
        "batch": lambda: solve_batch(
            batch_solver, x0_batch, references, seeds, protocol
        ),
    }
    for _ in range(args.warmups):
        for method in methods.values():
            method()
    rng = np.random.default_rng(
        np.random.SeedSequence([args.task_seed, args.method_seed, 20260811])
    )
    wall = {name: [] for name in methods}
    native = {name: [] for name in methods}
    workflow = {name: [] for name in methods}
    raw = []
    reference_results = None
    for repeat in range(args.repeats):
        order = list(methods)
        rng.shuffle(order)
        pair = {}
        for method in order:
            output, stats, wall_ms, native_ms, workflow_ms = methods[method]()
            pair[method] = (output, stats)
            wall[method].append(wall_ms)
            native[method].append(native_ms)
            workflow[method].append(workflow_ms)
        row = {"repeat": repeat, "order": order}
        for method, (output, stats) in pair.items():
            row.update(
                {
                    f"{method}_solve_wall_ms": wall[method][-1],
                    f"{method}_native_ms": native[method][-1],
                    f"{method}_workflow_ms": workflow[method][-1],
                    f"{method}_output_sha256": sha256_array(output),
                    f"{method}_initial_merit": stats["initial_merit"].tolist(),
                    f"{method}_final_merit": stats["final_merit"].tolist(),
                    f"{method}_sqp_iterations": stats["sqp_iterations"].tolist(),
                    f"{method}_total_pcg_iterations": stats[
                        "total_pcg_iterations"
                    ].tolist(),
                    f"{method}_pcg_cap_hits": stats["pcg_cap_hits"].tolist(),
                }
            )
        raw.append(row)
        if reference_results is None:
            reference_results = pair
    serial_output, serial_stats = reference_results["serial"]
    batch_output, batch_stats = reference_results["batch"]
    serial_quality, serial_arrays = certify_batch(
        model,
        instance,
        protocol,
        x0,
        seeds,
        serial_output,
        serial_stats,
        batch_solver.sim_forward,
    )
    batch_quality, batch_arrays = certify_batch(
        model,
        instance,
        protocol,
        x0,
        seeds,
        batch_output,
        batch_stats,
        batch_solver.sim_forward,
    )
    serial_cuda_modes = certify_mode_support(
        serial_quality,
        instance.pillar_xy,
        model.effortLimit,
        path_key="cuda_tool_path",
    )
    serial_pin_rows = copy.deepcopy(serial_quality)
    serial_pin_modes = certify_mode_support(
        serial_pin_rows,
        instance.pillar_xy,
        model.effortLimit,
        path_key="pinocchio_tool_path",
    )
    cuda_modes = certify_mode_support(
        batch_quality,
        instance.pillar_xy,
        model.effortLimit,
        path_key="cuda_tool_path",
    )
    pin_rows = copy.deepcopy(batch_quality)
    pin_modes = certify_mode_support(
        pin_rows,
        instance.pillar_xy,
        model.effortLimit,
        path_key="pinocchio_tool_path",
    )
    mode_agreement = mode_certificates_agree(cuda_modes, pin_modes)
    serial_mode_agreement = mode_certificates_agree(
        serial_cuda_modes, serial_pin_modes
    )
    batch_mode_support_noninferior = bool(
        all(
            cuda_modes["support"][mode] >= serial_cuda_modes["support"][mode]
            for mode in ("clockwise", "counterclockwise")
        )
        and (not serial_cuda_modes["two_mode_support"] or cuda_modes["two_mode_support"])
    )
    noninferiority = numerical_noninferiority(
        serial_quality, batch_quality, serial_output, batch_output
    )
    serial_winner = select_winner(serial_quality)
    batch_winner = select_winner(batch_quality)
    winner_noninferior = selected_winner_noninferior(serial_winner, batch_winner)
    repeatability_summary = {
        method: repeatability(raw, method) for method in methods
    }
    serial_wall = np.asarray(wall["serial"])
    batch_wall = np.asarray(wall["batch"])
    reductions = (serial_wall - batch_wall) / serial_wall
    extension = Path(batch_solver.lib.__file__)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    summary = {
        "schema_version": 1,
        "benchmark": "tiago_productive_cold_multistart_pillar",
        "phase": args.phase,
        "task_seed": args.task_seed,
        "method_seed": args.method_seed,
        "declared_task_seeds": sorted(task_seeds),
        "declared_method_seeds": sorted(method_seeds),
        "central_final_task_seed": CENTRAL_FINAL_TASK_SEED,
        "protocol": asdict(protocol),
        "solver_parameters": solver_parameters(protocol),
        "instance": instance.metadata(),
        "task_generation": task_generation,
        "candidate_generation": {
            **control_metadata,
            **rollout_metadata,
            "generation_wall_ms": generation_ms,
            "description": (
                "stationary inverse-dynamics hold plus seven antithetic analytic "
                "sin-squared joint pairs and one independent draw; one RNEA call "
                "per candidate/control knot and exactly one open-loop rollout per "
                "replay model; no reference, pillar, collision, IK, feedback, "
                "calibration, optimization, rejection, or resampling"
            ),
        },
        "candidate_labels": candidate_labels(),
        "model_pair_preflight": preflight,
        "seed_screen": strip_arrays(screen_rows),
        "seed_screen_aggregate": screen_aggregate,
        "input_hashes": {
            "x0_batch": sha256_array(x0_batch),
            "reference_batch": sha256_array(references),
            "seed_trajectories": sha256_array(seeds),
        },
        "common_problem_invariance": bool(
            all(np.array_equal(x0_batch[0], row) for row in x0_batch[1:])
            and all(np.array_equal(references[0], row) for row in references[1:])
        ),
        "serial_cuda_mode_certificate": serial_cuda_modes,
        "serial_pinocchio_mode_certificate": serial_pin_modes,
        "serial_cuda_pin_mode_and_topology_agreement": serial_mode_agreement,
        "cuda_mode_certificate": cuda_modes,
        "pinocchio_mode_certificate": pin_modes,
        "cuda_pin_mode_and_topology_agreement": mode_agreement,
        "batch_mode_support_noninferior_to_serial": batch_mode_support_noninferior,
        "serial_quality": strip_arrays(serial_quality),
        "batch_quality": strip_arrays(batch_quality),
        "numerical_noninferiority": noninferiority,
        "selected_serial_winner": strip_arrays(serial_winner),
        "selected_batch_winner": strip_arrays(batch_winner),
        "selected_winner_noninferior": winner_noninferior,
        "all_numerical_noninferiority_checks_pass": bool(
            all(row["passes"] for row in noninferiority)
            and winner_noninferior
            and serial_mode_agreement
            and mode_agreement
            and batch_mode_support_noninferior
        ),
        "repeatability": repeatability_summary,
        "all_repeatability_checks_pass": all(
            row["passes"] for row in repeatability_summary.values()
        ),
        "timing": {
            "solve_only_wall": {
                method: timing_stats(values) for method, values in wall.items()
            },
            "native_reported_solver": {
                method: timing_stats(values)
                for method, values in native.items()
            },
            "reset_solve_stats_workflow": {
                method: timing_stats(values)
                for method, values in workflow.items()
            },
            "paired_reduction_fraction": reductions.tolist(),
            "median_reduction_fraction": float(np.median(reductions)),
            "bootstrap_95_ci_median_reduction": bootstrap_median(reductions),
            "generation_wall_ms": generation_ms,
        },
        "raw_repeats": raw,
        "provenance": {
            "git_head": git_head,
            "git_tracked_dirty": tracked_dirty,
            "model_sha256": sha256_file(MODEL_PATH),
            "extension": str(extension),
            "extension_sha256": sha256_file(extension),
            "source_sha256": {
                "module": sha256_file(
                    REPO_ROOT / "tiago_src/gato_tiago/multimodal_cold_sqp.py"
                ),
                "harness": sha256_file(Path(__file__).resolve()),
                "plant": sha256_file(
                    REPO_ROOT
                    / "gato/dynamics/tiago_right/tiago_right_plant.cuh"
                ),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        x0=x0,
        references=references,
        seed_trajectories=seeds,
        serial_planned_trajectories=serial_output,
        batch_planned_trajectories=batch_output,
        **{f"serial_{key}": value for key, value in serial_arrays.items()},
        **{f"batch_{key}": value for key, value in batch_arrays.items()},
    )
    print(args.output)
    print(
        json.dumps(
            {
                "cuda_mode_certificate": strip_arrays(cuda_modes),
                "cuda_pin_mode_and_topology_agreement": mode_agreement,
                "productive_candidate_count": sum(
                    row["productive_certified"] for row in batch_quality
                ),
                "pcg_cap_lane_count": sum(
                    row["pcg_cap_hits"] > 0 for row in batch_quality
                ),
                "median_solver_reduction": float(np.median(reductions)),
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "final"), default="development")
    parser.add_argument("--task-seed", type=int, default=DEVELOPMENT_TASK_SEEDS[0])
    parser.add_argument("--method-seed", type=int, default=DEVELOPMENT_METHOD_SEEDS[0])
    parser.add_argument("--budget", type=int, choices=(BUDGET,), default=BUDGET)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--seed-screen-only",
        action="store_true",
        help="run the frozen seed-only CUDA/Pinocchio diagnostic without SQP",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
