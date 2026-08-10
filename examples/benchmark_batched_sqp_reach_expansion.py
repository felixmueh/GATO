#!/usr/bin/env python3
"""Evaluate one held-out IIWA reach with cold batch-1 and batch-8 multistart.

This extension imports, without retuning, the accepted N=8 IIWA solver,
initializer, and productive-control certificate.  A problem has one x0 and
goal.  Its batch contains the exact cold candidate plus seven generic bounded
perturbations generated with the frozen primary method seed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pinocchio as pin


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

import benchmark_batched_sqp_reach as accepted


OFFSETS = {
    "near": accepted.Lane("near", (0.04, 0.02, 0.02)),
    "far": accepted.Lane("far", (0.07, 0.04, 0.03)),
    "cross": accepted.Lane("cross", (0.03, -0.04, 0.02)),
}
INSTANCE_SEEDS = tuple(range(43, 51))
PRIMARY_METHOD_SEED = 1000
BUDGET = 8
CENTRAL_TIMING_INSTANCE = 46
ACCEPTED_HARNESS_COMMIT = "746d96fcfd85fe98071005e920b0773b0e450f73"
ACCEPTED_HARNESS_SHA256 = (
    "277af13dbf277bf6bbcc2ea0259b2865bacbde08f20bae968535647b3a1570dd"
)
DEFAULT_OUTPUT = REPO_ROOT / "example_artifacts/iiwa_reach_expansion/probe.json"


def problem_specs():
    return [
        (offset, instance_seed)
        for offset in OFFSETS
        for instance_seed in INSTANCE_SEEDS
    ]


def prepare_problem(model, offset_name, instance_seed):
    lane = OFFSETS[offset_name]
    q0, start, goal = accepted.generate_instance(model, lane, instance_seed)
    seeds, seed_metadata = accepted.generate_seeds(
        model, q0, PRIMARY_METHOD_SEED, BUDGET
    )
    x0 = np.hstack([q0, np.zeros(model.nv, dtype=np.float32)])
    x0_batch = np.tile(x0, (BUDGET, 1)).astype(np.float32)
    reference = np.tile(
        np.hstack([goal, np.zeros(3, dtype=np.float32)]), accepted.KNOTS
    )
    references = np.tile(reference, (BUDGET, 1)).astype(np.float32)
    return {
        "lane": lane,
        "q0": q0,
        "start": start,
        "goal": goal,
        "seeds": seeds,
        "seed_metadata": seed_metadata,
        "x0": x0,
        "x0_batch": x0_batch,
        "reference": reference,
        "references": references,
    }


def exact_cold_candidate(model, q0):
    started = time.perf_counter()
    nx, nu = model.nq + model.nv, model.nv
    cold = np.zeros(accepted.KNOTS * (nx + nu) - nu, dtype=np.float32)
    for knot in range(accepted.KNOTS):
        cold[knot * (nx + nu) : knot * (nx + nu) + model.nq] = q0
    return cold, 1e3 * (time.perf_counter() - started)


def repeatability(raw, method):
    merit_spans = accepted.merit_repeatability(raw, method)
    output_hash_count = len({row[f"{method}_output_sha256"] for row in raw})
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
    passed = bool(
        output_hash_count == 1
        and exact_integer_telemetry
        and merit_spans["initial_merit"]["maximum_relative_span"] <= 1e-6
        and merit_spans["final_merit"]["maximum_relative_span"] <= 1e-6
    )
    return {
        "output_hash_count": output_hash_count,
        "exact_sqp_pcg_and_cap_telemetry": exact_integer_telemetry,
        "merit_spans": merit_spans,
        "passes": passed,
    }


def quality_noninferiority(serial_rows, batch_rows, serial_output, batch_output):
    rows = []
    for serial, batch in zip(serial_rows, batch_rows):
        initial_relative = abs(
            batch["initial_solver_merit"] - serial["initial_solver_merit"]
        ) / max(abs(serial["initial_solver_merit"]), 1.0)
        cost_scale = serial["final_cuda"]["nonnegative_task_motion_cost"]
        cost_tolerance = max(1e-3, 0.01 * max(cost_scale, 1.0))
        row = {
            "candidate_index": serial["candidate_index"],
            "initial_merit_relative_difference": float(initial_relative),
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
                <= cost_scale + cost_tolerance
            ),
            "batch_has_no_extra_cap": bool(
                batch["pcg_cap_hits"] <= serial["pcg_cap_hits"]
            ),
            "batch_has_no_added_nonfinite_or_limit_failure": bool(
                not accepted.final_rollouts_finite_and_within_limits(serial)
                or accepted.final_rollouts_finite_and_within_limits(batch)
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
            and row["serial_success_preserved"]
            and row["batch_terminal_within_serial_plus_2mm"]
            and row["batch_external_cost_noninferior"]
            and row["batch_has_no_extra_cap"]
            and row["batch_has_no_added_nonfinite_or_limit_failure"]
        )
        rows.append(row)
    return rows


def selected_winner_noninferior(serial, batch):
    if serial is None:
        return True
    if batch is None:
        return False
    serial_cost = serial["final_cuda"]["nonnegative_task_motion_cost"]
    return bool(
        batch["final_cuda"]["terminal_error_m"]
        <= serial["final_cuda"]["terminal_error_m"] + 0.002
        and batch["final_cuda"]["nonnegative_task_motion_cost"]
        <= serial_cost + max(1e-3, 0.01 * max(serial_cost, 1.0))
    )


def seed_is_already_certified(row):
    cuda = row["seed_cuda"]
    pinocchio = row["seed_pinocchio"]
    return bool(
        cuda["finite"]
        and pinocchio["finite"]
        and cuda["terminal_error_m"] <= 0.10
        and pinocchio["terminal_error_m"] <= 0.11
        and row["seed_cuda_one_step_pinocchio_defect"] <= 1e-3
        and cuda["joint_violation_rad"] <= 0.0
        and pinocchio["joint_violation_rad"] <= 0.0
        and cuda["velocity_ratio"] <= 1.0
        and pinocchio["velocity_ratio"] <= 1.0
        and cuda["control_ratio"] <= 1.0
        and pinocchio["control_ratio"] <= 1.0
    )


def annotate_extension_success(rows):
    for row in rows:
        row["seed_already_certified_before_sqp"] = seed_is_already_certified(row)
        row["extension_optimizer_improved_success"] = bool(
            row["optimizer_improved_success"]
            and not row["seed_already_certified_before_sqp"]
        )


def select_extension_winner(rows):
    successes = [row for row in rows if row["extension_optimizer_improved_success"]]
    return min(
        successes,
        key=lambda row: row["final_cuda"]["nonnegative_task_motion_cost"],
        default=None,
    )


def run(args):
    if args.method_seed != PRIMARY_METHOD_SEED:
        raise ValueError(
            f"headline expansion freezes method seed at {PRIMARY_METHOD_SEED}"
        )
    if args.budget != BUDGET:
        raise ValueError(f"headline expansion freezes budget at {BUDGET}")
    model = pin.buildModelFromUrdf(str(accepted.MODEL_PATH))
    accepted_harness_path = Path(accepted.__file__).resolve()
    accepted_harness_sha256 = accepted.sha256_file(accepted_harness_path)
    if accepted_harness_sha256 != ACCEPTED_HARNESS_SHA256:
        raise RuntimeError(
            "accepted IIWA harness differs from its frozen 746d96f content"
        )
    problem = prepare_problem(model, args.offset, args.instance_seed)
    exact_cold, cold_generation_ms = exact_cold_candidate(model, problem["q0"])
    np.testing.assert_array_equal(exact_cold, problem["seeds"][0])

    cold_solver = accepted.make_solver(1)
    serial_solver = accepted.make_solver(1)
    batch_solver = accepted.make_solver(BUDGET)
    preflight = accepted.model_pair_preflight(batch_solver, model, count=BUDGET)
    if not preflight["passes_1e-3_gate"]:
        raise RuntimeError("CUDA/Pinocchio random-state model preflight failed")

    methods = {
        "cold": lambda: accepted.solve_batched(
            cold_solver,
            problem["x0_batch"][:1],
            problem["references"][:1],
            problem["seeds"][:1],
        ),
        "serial8": lambda: accepted.solve_independent(
            serial_solver,
            problem["x0_batch"],
            problem["references"],
            problem["seeds"],
        ),
        "batch8": lambda: accepted.solve_batched(
            batch_solver,
            problem["x0_batch"],
            problem["references"],
            problem["seeds"],
        ),
    }
    for _ in range(args.warmups):
        for method in methods.values():
            method()

    order_rng = np.random.default_rng(
        np.random.SeedSequence(
            [args.instance_seed, args.method_seed, accepted.KNOTS, 20260811]
        )
    )
    wall = {name: [] for name in methods}
    native = {name: [] for name in methods}
    workflow = {name: [] for name in methods}
    raw = []
    reference_results = None
    for repeat in range(args.repeats):
        order = list(methods)
        order_rng.shuffle(order)
        pair = {}
        for method in order:
            output, stats, solve_ms, native_ms, workflow_ms = methods[method]()
            pair[method] = (output, stats)
            wall[method].append(solve_ms)
            native[method].append(native_ms)
            workflow[method].append(workflow_ms)
        row = {"repeat": repeat, "order": order}
        for method, (output, stats) in pair.items():
            row.update(
                {
                    f"{method}_solve_wall_ms": wall[method][-1],
                    f"{method}_native_ms": native[method][-1],
                    f"{method}_workflow_ms": workflow[method][-1],
                    f"{method}_output_sha256": accepted.sha256_array(output),
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

    cold_output, cold_stats = reference_results["cold"]
    serial_output, serial_stats = reference_results["serial8"]
    batch_output, batch_stats = reference_results["batch8"]
    cold_quality, cold_arrays, cold_cert_ms = accepted.certify(
        model,
        problem["goal"],
        problem["x0"],
        problem["seeds"][:1],
        cold_output,
        cold_stats,
        cold_solver,
    )
    serial_quality, serial_arrays, serial_cert_ms = accepted.certify(
        model,
        problem["goal"],
        problem["x0"],
        problem["seeds"],
        serial_output,
        serial_stats,
        batch_solver,
    )
    batch_quality, batch_arrays, batch_cert_ms = accepted.certify(
        model,
        problem["goal"],
        problem["x0"],
        problem["seeds"],
        batch_output,
        batch_stats,
        batch_solver,
    )
    annotate_extension_success(cold_quality)
    annotate_extension_success(serial_quality)
    annotate_extension_success(batch_quality)
    noninferiority = quality_noninferiority(
        serial_quality, batch_quality, serial_output, batch_output
    )
    cold_batch0_noninferiority = quality_noninferiority(
        cold_quality, batch_quality[:1], cold_output, batch_output
    )[0]
    selection_started = time.perf_counter()
    cold_winner = select_extension_winner(cold_quality)
    cold_selection_ms = 1e3 * (time.perf_counter() - selection_started)
    selection_started = time.perf_counter()
    serial_winner = select_extension_winner(serial_quality)
    serial_selection_ms = 1e3 * (time.perf_counter() - selection_started)
    selection_started = time.perf_counter()
    batch_winner = select_extension_winner(batch_quality)
    batch_selection_ms = 1e3 * (time.perf_counter() - selection_started)
    winner_noninferior = selected_winner_noninferior(serial_winner, batch_winner)
    cold_success = bool(cold_quality[0]["extension_optimizer_improved_success"])
    batch_success = batch_winner is not None
    batch_candidate0 = batch_quality[0]
    cold_serial0_parity = {
        "trajectory_bitwise_equal": bool(
            np.array_equal(cold_output[0], serial_output[0])
        ),
        "initial_merit_relative_difference": float(
            abs(
                cold_stats["initial_merit"][0]
                - serial_stats["initial_merit"][0]
            )
            / max(abs(cold_stats["initial_merit"][0]), 1.0)
        ),
        "final_merit_relative_difference": float(
            abs(cold_stats["final_merit"][0] - serial_stats["final_merit"][0])
            / max(abs(cold_stats["final_merit"][0]), 1.0)
        ),
        "sqp_iterations_equal": bool(
            cold_stats["sqp_iterations"][0]
            == serial_stats["sqp_iterations"][0]
        ),
        "total_pcg_iterations_equal": bool(
            cold_stats["total_pcg_iterations"][0]
            == serial_stats["total_pcg_iterations"][0]
        ),
        "pcg_cap_hits_equal": bool(
            cold_stats["pcg_cap_hits"][0] == serial_stats["pcg_cap_hits"][0]
        ),
        "productive_success_equal": bool(
            cold_quality[0]["extension_optimizer_improved_success"]
            == serial_quality[0]["extension_optimizer_improved_success"]
        ),
    }
    cold_serial0_parity["passes"] = bool(
        cold_serial0_parity["trajectory_bitwise_equal"]
        and cold_serial0_parity["initial_merit_relative_difference"] <= 1e-6
        and cold_serial0_parity["final_merit_relative_difference"] <= 1e-6
        and cold_serial0_parity["sqp_iterations_equal"]
        and cold_serial0_parity["total_pcg_iterations_equal"]
        and cold_serial0_parity["pcg_cap_hits_equal"]
        and cold_serial0_parity["productive_success_equal"]
    )
    nested_cold_preserved = bool(
        not cold_success
        or batch_candidate0["extension_optimizer_improved_success"]
    )
    noncold_batch_success = any(
        row["candidate_index"] != 0
        and row["extension_optimizer_improved_success"]
        for row in batch_quality
    )
    cold_cost = cold_quality[0]["final_cuda"]["nonnegative_task_motion_cost"]
    material_non_cold_gain = bool(
        not cold_success
        and batch_winner is not None
        and batch_winner["candidate_index"] != 0
        and batch_winner["final_cuda"]["nonnegative_task_motion_cost"]
        <= 0.99 * cold_cost
    )

    repeatability_summary = {
        method: repeatability(raw, method) for method in methods
    }
    serial_wall = np.asarray(wall["serial8"], dtype=np.float64)
    batch_wall = np.asarray(wall["batch8"], dtype=np.float64)
    throughput_reduction = (serial_wall - batch_wall) / serial_wall
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
        "benchmark": "iiwa_distinct_reach_cold_vs_batch8_multistart",
        "frozen_protocol": {
            "source_acceptance_commit": ACCEPTED_HARNESS_COMMIT,
            "accepted_harness_path": str(accepted_harness_path),
            "accepted_harness_sha256": accepted_harness_sha256,
            "required_accepted_harness_sha256": ACCEPTED_HARNESS_SHA256,
            "knots": accepted.KNOTS,
            "dt": accepted.DT,
            "solver_parameters": accepted.solver_parameters(),
            "budget": BUDGET,
            "method_seed": PRIMARY_METHOD_SEED,
            "initializer": problem["seed_metadata"],
            "certificate": "accepted productive-control certificate imported unchanged",
        },
        "problem": {
            "offset": asdict(problem["lane"]),
            "instance_seed": args.instance_seed,
            "start_configuration": problem["q0"].tolist(),
            "start_position": problem["start"].tolist(),
            "goal_position": problem["goal"].tolist(),
        },
        "model_pair_preflight": preflight,
        "input_hashes": {
            "x0": accepted.sha256_array(problem["x0"]),
            "reference": accepted.sha256_array(problem["reference"]),
            "batch8_seeds": accepted.sha256_array(problem["seeds"]),
            "nested_cold_candidate": accepted.sha256_array(problem["seeds"][0]),
            "explicit_cold_candidate": accepted.sha256_array(exact_cold),
        },
        "exact_cold_candidate_is_nested": bool(
            np.array_equal(exact_cold, problem["seeds"][0])
        ),
        "outcomes": {
            "cold_success": cold_success,
            "batch8_success": batch_success,
            "rescued_by_batch8": bool(not cold_success and batch_success),
            "rescued_by_non_cold_candidate": bool(
                not cold_success and noncold_batch_success
            ),
            "cold_success_lost": bool(cold_success and not batch_success),
            "nested_cold_success_preserved_in_batch": nested_cold_preserved,
            "material_non_cold_gain_on_rescue": material_non_cold_gain,
            "cold_final_external_cost": float(cold_cost),
            "batch_selected_candidate_index": (
                None if batch_winner is None else batch_winner["candidate_index"]
            ),
            "batch_selected_final_external_cost": (
                None
                if batch_winner is None
                else batch_winner["final_cuda"]["nonnegative_task_motion_cost"]
            ),
        },
        "repeatability": repeatability_summary,
        "all_repeatability_checks_pass": all(
            value["passes"] for value in repeatability_summary.values()
        ),
        "quality_noninferiority": noninferiority,
        "cold_batch1_vs_serial8_candidate0_numerical_parity": (
            cold_serial0_parity
        ),
        "cold_batch1_vs_batch8_candidate0_noninferiority": (
            cold_batch0_noninferiority
        ),
        "all_batch8_vs_serial8_noninferiority_checks_pass": bool(
            all(row["passes"] for row in noninferiority) and winner_noninferior
        ),
        "selected_winner_noninferior": winner_noninferior,
        "timing": {
            "proposal_budget_ratio_batch8_to_cold": 8,
            "solve_only_wall": {
                method: accepted.timing_stats(values) for method, values in wall.items()
            },
            "native_reported_solver": {
                method: accepted.timing_stats(values)
                for method, values in native.items()
            },
            "reset_solve_stats_workflow": {
                method: accepted.timing_stats(values)
                for method, values in workflow.items()
            },
            "batch8_vs_serial8_paired_reduction": throughput_reduction.tolist(),
            "batch8_vs_serial8_median_reduction": float(
                np.median(throughput_reduction)
            ),
            "batch8_vs_serial8_bootstrap_95_ci": accepted.bootstrap_interval(
                throughput_reduction
            ),
            "certification_ms": {
                "cold": cold_cert_ms,
                "serial8": serial_cert_ms,
                "batch8": batch_cert_ms,
            },
            "selection_ms": {
                "cold": cold_selection_ms,
                "serial8": serial_selection_ms,
                "batch8": batch_selection_ms,
            },
            "complete_latency_ms": {
                "cold_generation_plus_workflow_plus_certification_plus_selection": float(
                    cold_generation_ms
                    + np.median(workflow["cold"])
                    + cold_cert_ms
                    + cold_selection_ms
                ),
                "batch8_generation_plus_workflow_plus_certification_plus_selection": float(
                    problem["seed_metadata"]["generation_wall_ms"]
                    + np.median(workflow["batch8"])
                    + batch_cert_ms
                    + batch_selection_ms
                ),
                "serial8_generation_plus_workflow_plus_certification_plus_selection": float(
                    problem["seed_metadata"]["generation_wall_ms"]
                    + np.median(workflow["serial8"])
                    + serial_cert_ms
                    + serial_selection_ms
                ),
            },
        },
        "cold_only_generation_ms": cold_generation_ms,
        "cold_quality": cold_quality[0],
        "serial8_quality": serial_quality,
        "batch8_quality": batch_quality,
        "serial8_selected_winner": serial_winner,
        "batch8_selected_winner": batch_winner,
        "raw_repeats": raw,
        "provenance": {
            "git_head": git_head,
            "git_tracked_dirty": tracked_dirty,
            "model_sha256": accepted.sha256_file(accepted.MODEL_PATH),
            "extension": str(extension),
            "extension_sha256": accepted.sha256_file(extension),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable = accepted.json_safe(summary)
    args.output.write_text(
        json.dumps(serializable, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        seed_trajectories=problem["seeds"],
        cold_optimized_planned_trajectory=cold_output,
        serial8_optimized_planned_trajectories=serial_output,
        batch8_optimized_planned_trajectories=batch_output,
        x0=problem["x0"],
        reference=problem["reference"],
        **{f"cold_{key}": value for key, value in cold_arrays.items()},
        **{f"serial8_{key}": value for key, value in serial_arrays.items()},
        **{f"batch8_{key}": value for key, value in batch_arrays.items()},
    )
    print(args.output)
    print(
        json.dumps(
            {
                "outcomes": summary["outcomes"],
                "all_repeatability_checks_pass": summary[
                    "all_repeatability_checks_pass"
                ],
                "all_batch8_vs_serial8_noninferiority_checks_pass": summary[
                    "all_batch8_vs_serial8_noninferiority_checks_pass"
                ],
                "batch8_vs_serial8_median_reduction": summary["timing"][
                    "batch8_vs_serial8_median_reduction"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset", choices=tuple(OFFSETS), default="near")
    parser.add_argument("--instance-seed", type=int, default=43)
    parser.add_argument("--method-seed", type=int, default=PRIMARY_METHOD_SEED)
    parser.add_argument("--budget", type=int, default=BUDGET, choices=(BUDGET,))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
