#!/usr/bin/env python3
"""Run and aggregate the predeclared 24-problem IIWA reach expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

import benchmark_batched_sqp_reach_expansion as expansion


BENCHMARK = REPO_ROOT / "examples/benchmark_batched_sqp_reach_expansion.py"
DEFAULT_OUTPUT = REPO_ROOT / "example_artifacts/iiwa_reach_expansion"


def case_name(offset, instance_seed):
    return f"{offset}_instance{instance_seed}_method{expansion.PRIMARY_METHOD_SEED}"


def required_repeats(offset, instance_seed, ordinary_repeats=5):
    del offset
    return 20 if instance_seed == expansion.CENTRAL_TIMING_INSTANCE else ordinary_repeats


def bootstrap_mean_interval(values, seed=20260811, resamples=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(
        values[rng.integers(0, values.size, size=(resamples, values.size))], axis=1
    )
    return {
        "confidence": 0.95,
        "resamples": resamples,
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_at_least(value, threshold):
    return bool(
        isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= threshold
    )


def finite_greater_than(value, threshold):
    return bool(
        isinstance(value, (int, float))
        and math.isfinite(value)
        and value > threshold
    )


def load_case(path, offset, instance_seed, ordinary_repeats=5):
    spec = (offset, instance_seed)
    if not path.exists():
        return {
            "spec": list(spec),
            "artifact": str(path),
            "load_error": "missing artifact",
            "failures": ["missing_artifact"],
            "gates": {},
        }
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "spec": list(spec),
            "artifact": str(path),
            "load_error": str(error),
            "failures": ["unreadable_artifact"],
            "gates": {},
        }
    repeat_count = len(summary.get("raw_repeats", []))
    frozen = summary.get("frozen_protocol", {})
    problem = summary.get("problem", {})
    outcomes = summary.get("outcomes", {})
    hashes = summary.get("input_hashes", {})
    provenance = summary.get("provenance", {})
    timing = summary.get("timing", {})
    initializer = frozen.get("initializer", {})
    interval = timing.get("batch8_vs_serial8_bootstrap_95_ci", {})
    complete_latency = timing.get("complete_latency_ms", {})
    cold_complete_latency = complete_latency.get(
        "cold_generation_plus_workflow_plus_certification_plus_selection"
    )
    batch_complete_latency = complete_latency.get(
        "batch8_generation_plus_workflow_plus_certification_plus_selection"
    )
    gates = {
        "identity_and_frozen_protocol": bool(
            problem.get("offset", {}).get("name") == offset
            and problem.get("instance_seed") == instance_seed
            and frozen.get("knots") == 8
            and frozen.get("dt") == 0.05
            and frozen.get("budget") == expansion.BUDGET
            and frozen.get("method_seed") == expansion.PRIMARY_METHOD_SEED
            and frozen.get("solver_parameters")
            == expansion.accepted.solver_parameters()
            and initializer.get("state_amplitude_rad") == 0.020
            and initializer.get("control_effort_fraction") == 0.010
            and initializer.get("cold_candidate_index") == 0
        ),
        "accepted_harness_content_frozen": bool(
            frozen.get("source_acceptance_commit")
            == expansion.ACCEPTED_HARNESS_COMMIT
            and frozen.get("accepted_harness_sha256")
            == expansion.ACCEPTED_HARNESS_SHA256
            and frozen.get("required_accepted_harness_sha256")
            == expansion.ACCEPTED_HARNESS_SHA256
        ),
        "required_repeat_count": repeat_count
        >= required_repeats(offset, instance_seed, ordinary_repeats),
        "exact_cold_candidate_nested": bool(
            summary.get("exact_cold_candidate_is_nested")
            and hashes.get("nested_cold_candidate")
            and hashes.get("nested_cold_candidate")
            == hashes.get("explicit_cold_candidate")
        ),
        "model_pair_preflight": bool(
            summary.get("model_pair_preflight", {}).get("passes_1e-3_gate")
        ),
        "tracked_provenance_clean": not provenance.get("git_tracked_dirty", True),
        "all_repeatability_checks": bool(
            summary.get("all_repeatability_checks_pass")
        ),
        "batch8_vs_serial8_noninferiority": bool(
            summary.get("all_batch8_vs_serial8_noninferiority_checks_pass")
        ),
        "cold_batch1_matches_serial8_candidate0": bool(
            summary.get(
                "cold_batch1_vs_serial8_candidate0_numerical_parity", {}
            ).get("passes")
        ),
        "batch8_candidate0_noninferior_to_cold_batch1": bool(
            summary.get("cold_batch1_vs_batch8_candidate0_noninferiority", {}).get(
                "passes"
            )
        ),
        "cold_success_not_lost": not outcomes.get("cold_success_lost", True),
        "nested_cold_success_preserved": bool(
            outcomes.get("nested_cold_success_preserved_in_batch")
        ),
    }
    if instance_seed == expansion.CENTRAL_TIMING_INSTANCE:
        gates["central_batch8_vs_serial8_timing"] = bool(
            finite_at_least(timing.get("batch8_vs_serial8_median_reduction"), 0.10)
            and finite_greater_than(interval.get("lower"), 0.05)
        )
    failures = [name for name, passed in gates.items() if not passed]
    return {
        "spec": list(spec),
        "artifact": str(path),
        "artifact_sha256": sha256_file(path),
        "repeat_count": repeat_count,
        "cold_success": bool(outcomes.get("cold_success")),
        "batch8_success": bool(outcomes.get("batch8_success")),
        "rescued_by_batch8": bool(outcomes.get("rescued_by_batch8")),
        "rescued_by_non_cold_candidate": bool(
            outcomes.get("rescued_by_non_cold_candidate")
        ),
        "cold_success_lost": bool(outcomes.get("cold_success_lost")),
        "material_non_cold_gain_on_rescue": bool(
            outcomes.get("material_non_cold_gain_on_rescue")
        ),
        "cold_final_external_cost": outcomes.get("cold_final_external_cost"),
        "batch_selected_candidate_index": outcomes.get(
            "batch_selected_candidate_index"
        ),
        "batch_selected_final_external_cost": outcomes.get(
            "batch_selected_final_external_cost"
        ),
        "cold_solve_wall_median_ms": timing.get("solve_only_wall", {})
        .get("cold", {})
        .get("median_ms"),
        "batch8_solve_wall_median_ms": timing.get("solve_only_wall", {})
        .get("batch8", {})
        .get("median_ms"),
        "serial8_solve_wall_median_ms": timing.get("solve_only_wall", {})
        .get("serial8", {})
        .get("median_ms"),
        "batch8_vs_serial8_median_reduction": timing.get(
            "batch8_vs_serial8_median_reduction"
        ),
        "batch8_vs_serial8_bootstrap_95_ci": interval,
        "cold_complete_latency_ms": cold_complete_latency,
        "batch8_complete_latency_ms": batch_complete_latency,
        "batch8_to_cold_complete_latency_ratio": (
            batch_complete_latency / cold_complete_latency
            if isinstance(cold_complete_latency, (int, float))
            and cold_complete_latency > 0.0
            and isinstance(batch_complete_latency, (int, float))
            else None
        ),
        "git_head": provenance.get("git_head"),
        "extension_sha256": provenance.get("extension_sha256"),
        "model_sha256": provenance.get("model_sha256"),
        "gates": gates,
        "failures": failures,
    }


def aggregate(rows):
    completed = [row for row in rows if "load_error" not in row]
    expected_count = len(expansion.OFFSETS) * len(expansion.INSTANCE_SEEDS)
    cold_successes = sum(row["cold_success"] for row in completed)
    batch_successes = sum(row["batch8_success"] for row in completed)
    rescue_count = sum(row["rescued_by_batch8"] for row in completed)
    noncold_rescue_count = sum(
        row["rescued_by_non_cold_candidate"] for row in completed
    )
    lost_count = sum(row["cold_success_lost"] for row in completed)
    both_success = sum(
        row["cold_success"] and row["batch8_success"] for row in completed
    )
    both_fail = sum(
        not row["cold_success"] and not row["batch8_success"] for row in completed
    )
    material_rescues = sum(
        row["rescued_by_batch8"] and row["material_non_cold_gain_on_rescue"]
        for row in completed
    )
    paired = np.asarray(
        [int(row["batch8_success"]) - int(row["cold_success"]) for row in completed],
        dtype=np.float64,
    )
    paired_gain = float(np.mean(paired)) if paired.size else float("nan")
    paired_interval = (
        bootstrap_mean_interval(paired)
        if paired.size
        else {"confidence": 0.95, "resamples": 0, "lower": None, "upper": None}
    )
    central_rows = [
        row
        for row in completed
        if row["spec"][1] == expansion.CENTRAL_TIMING_INSTANCE
    ]
    offsets_seen = {row["spec"][0] for row in completed}
    seeds_seen = {row["spec"][1] for row in completed}
    git_heads = {row.get("git_head") for row in completed}
    extension_hashes = {row.get("extension_sha256") for row in completed}
    model_hashes = {row.get("model_sha256") for row in completed}
    cold_complete_latencies = [
        row["cold_complete_latency_ms"]
        for row in completed
        if isinstance(row.get("cold_complete_latency_ms"), (int, float))
    ]
    batch_complete_latencies = [
        row["batch8_complete_latency_ms"]
        for row in completed
        if isinstance(row.get("batch8_complete_latency_ms"), (int, float))
    ]
    paired_complete_latencies = [
        (row["cold_complete_latency_ms"], row["batch8_complete_latency_ms"])
        for row in completed
        if isinstance(row.get("cold_complete_latency_ms"), (int, float))
        and row["cold_complete_latency_ms"] > 0.0
        and isinstance(row.get("batch8_complete_latency_ms"), (int, float))
    ]
    gates = {
        "all_24_unique_problems_present": bool(
            len(completed) == expected_count
            and len({tuple(row["spec"]) for row in completed}) == expected_count
            and offsets_seen == set(expansion.OFFSETS)
            and seeds_seen == set(expansion.INSTANCE_SEEDS)
        ),
        "batch8_success_rate_at_least_80_percent": bool(
            completed and batch_successes / expected_count >= 0.80
        ),
        "paired_success_gain_at_least_20_percentage_points": bool(
            finite_at_least(paired_gain, 0.20)
        ),
        "paired_problem_bootstrap_lower_above_5_percentage_points": bool(
            finite_greater_than(paired_interval.get("lower"), 0.05)
        ),
        "at_least_five_cold_successes": cold_successes >= 5,
        "at_least_five_cold_failures": len(completed) - cold_successes >= 5,
        "zero_cold_success_losses": lost_count == 0,
        "material_non_cold_gain_in_at_least_half_of_rescues": bool(
            rescue_count > 0 and material_rescues / rescue_count >= 0.50
        ),
        "all_exact_cold_nesting_checks_pass": bool(completed)
        and all(row["gates"].get("exact_cold_candidate_nested", False) for row in completed),
        "all_model_pair_preflights_pass": bool(completed)
        and all(row["gates"].get("model_pair_preflight", False) for row in completed),
        "all_repeatability_checks_pass": bool(completed)
        and all(row["gates"].get("all_repeatability_checks", False) for row in completed),
        "all_numerical_noninferiority_checks_pass": bool(completed)
        and all(
            row["gates"].get("batch8_vs_serial8_noninferiority", False)
            for row in completed
        ),
        "all_case_identity_and_frozen_protocol_checks_pass": bool(completed)
        and all(
            row["gates"].get("identity_and_frozen_protocol", False)
            for row in completed
        ),
        "all_accepted_harness_content_hashes_match": bool(completed)
        and all(
            row["gates"].get("accepted_harness_content_frozen", False)
            for row in completed
        ),
        "all_artifacts_have_clean_tracked_provenance": bool(completed)
        and all(
            row["gates"].get("tracked_provenance_clean", False)
            for row in completed
        ),
        "all_artifacts_share_git_extension_and_model_hashes": bool(completed)
        and len(git_heads) == 1
        and None not in git_heads
        and len(extension_hashes) == 1
        and None not in extension_hashes
        and len(model_hashes) == 1
        and None not in model_hashes,
        "all_cases_have_required_repeat_counts": bool(completed)
        and all(row["gates"].get("required_repeat_count", False) for row in completed),
        "one_20_repeat_timing_case_per_offset_passes": bool(
            len(central_rows) == len(expansion.OFFSETS)
            and {row["spec"][0] for row in central_rows} == set(expansion.OFFSETS)
            and all(
                row["gates"].get("central_batch8_vs_serial8_timing", False)
                for row in central_rows
            )
        ),
        "all_required_per_case_gates_pass": bool(completed)
        and all(all(row["gates"].values()) for row in completed),
    }
    gates["all_expansion_acceptance_gates_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "campaign": "iiwa_distinct_reach_cold_vs_batch8_multistart",
        "predeclared_grid": {
            "offsets": {
                name: list(lane.goal_offset)
                for name, lane in expansion.OFFSETS.items()
            },
            "instance_seeds": list(expansion.INSTANCE_SEEDS),
            "primary_method_seed": expansion.PRIMARY_METHOD_SEED,
            "central_20_repeat_instance_per_offset": (
                expansion.CENTRAL_TIMING_INSTANCE
            ),
            "problem_count": expected_count,
        },
        "completed_problem_count": len(completed),
        "cold_success_count": cold_successes,
        "cold_failure_count": len(completed) - cold_successes,
        "batch8_success_count": batch_successes,
        "paired_rescue_table": {
            "cold_fail_batch_success": rescue_count,
            "both_success": both_success,
            "both_fail": both_fail,
            "cold_success_batch_fail": lost_count,
        },
        "paired_success_rate_gain": paired_gain,
        "paired_problem_bootstrap_95_ci": paired_interval,
        "material_non_cold_rescue_count": material_rescues,
        "noncold_candidate_rescue_count": noncold_rescue_count,
        "material_non_cold_fraction_of_rescues": (
            material_rescues / rescue_count if rescue_count else None
        ),
        "proposal_budget_ratio_batch8_to_cold": 8,
        "complete_latency_across_problems_ms": {
            "cold_batch1_median": (
                float(np.median(cold_complete_latencies))
                if cold_complete_latencies
                else None
            ),
            "cold_batch1_p95": (
                float(np.percentile(cold_complete_latencies, 95.0))
                if cold_complete_latencies
                else None
            ),
            "batch8_multistart_median": (
                float(np.median(batch_complete_latencies))
                if batch_complete_latencies
                else None
            ),
            "batch8_multistart_p95": (
                float(np.percentile(batch_complete_latencies, 95.0))
                if batch_complete_latencies
                else None
            ),
            "median_paired_batch8_to_cold_ratio": (
                float(
                    np.median(
                        [batch / cold for cold, batch in paired_complete_latencies]
                    )
                )
                if paired_complete_latencies
                else None
            ),
            "median_paired_batch8_minus_cold_ms": (
                float(
                    np.median(
                        [batch - cold for cold, batch in paired_complete_latencies]
                    )
                )
                if paired_complete_latencies
                else None
            ),
            "batch8_to_cold_proposal_budget_ratio": 8,
        },
        "acceptance_gates": gates,
        "consistent_provenance": {
            "git_heads": sorted(value for value in git_heads if value is not None),
            "extension_sha256": sorted(
                value for value in extension_hashes if value is not None
            ),
            "model_sha256": sorted(value for value in model_hashes if value is not None),
        },
        "failed_cases": [row for row in rows if row["failures"]],
        "cases": rows,
    }


def run_case(output_dir, offset, instance_seed, repeats, warmups):
    destination = output_dir / f"{case_name(offset, instance_seed)}.json"
    command = [
        sys.executable,
        "-B",
        str(BENCHMARK),
        "--offset",
        offset,
        "--instance-seed",
        str(instance_seed),
        "--method-seed",
        str(expansion.PRIMARY_METHOD_SEED),
        "--budget",
        str(expansion.BUDGET),
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
        "--output",
        str(destination),
    ]
    result = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    destination.with_suffix(".log.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination, result.returncode


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    execution_failures = []
    if not args.summarize_only:
        for offset, instance_seed in expansion.problem_specs():
            repeats = required_repeats(offset, instance_seed, args.ordinary_repeats)
            path, returncode = run_case(
                args.output_dir, offset, instance_seed, repeats, args.warmups
            )
            if returncode:
                execution_failures.append(
                    {
                        "spec": [offset, instance_seed],
                        "artifact": str(path),
                        "returncode": returncode,
                    }
                )
    rows = [
        load_case(
            args.output_dir / f"{case_name(offset, instance_seed)}.json",
            offset,
            instance_seed,
            args.ordinary_repeats,
        )
        for offset, instance_seed in expansion.problem_specs()
    ]
    summary = aggregate(rows)
    summary["execution_failures"] = execution_failures
    output = args.output_dir / "expansion_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(json.dumps(summary["acceptance_gates"], indent=2, sort_keys=True))
    return 0 if summary["acceptance_gates"]["all_expansion_acceptance_gates_pass"] else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--ordinary-repeats", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
