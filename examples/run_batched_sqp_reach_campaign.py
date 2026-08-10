#!/usr/bin/env python3
"""Run and aggregate the predeclared IIWA same-goal multistart campaign."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "examples/benchmark_batched_sqp_reach.py"
DEFAULT_OUTPUT = REPO_ROOT / "example_artifacts/iiwa_batched_sqp_reach_campaign"
LANES = ("near", "far")
INSTANCE_SEEDS = (40, 41, 42)
METHOD_SEEDS = (1000, 1001, 1002)
CENTRAL_TIMING_CASES = {("near", 41, 1001), ("far", 41, 1001)}
SPOT_TIMING_CASES = {
    (lane, instance_seed, 1000)
    for lane, instance_seed in itertools.product(LANES, INSTANCE_SEEDS)
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_name(lane, instance_seed, method_seed):
    return f"{lane}_instance{instance_seed}_method{method_seed}"


def required_repeats(spec, quality_repeats):
    if spec in CENTRAL_TIMING_CASES:
        return 20
    if spec in SPOT_TIMING_CASES:
        return 5
    return quality_repeats


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


def load_case(path, spec, required_repeat_count):
    lane, instance_seed, method_seed = spec
    if not path.exists():
        return {
            "spec": list(spec),
            "artifact": str(path),
            "load_error": "missing artifact",
            "gates": {},
            "failures": ["missing_artifact"],
        }
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "spec": list(spec),
            "artifact": str(path),
            "load_error": str(error),
            "gates": {},
            "failures": ["unreadable_artifact"],
        }
    repeats = len(summary.get("raw_repeats", []))
    timing = summary.get("solver_only_wall", {})
    interval = timing.get("bootstrap_95_ci_median_reduction", {})
    winner = summary.get("batched_selected_winner")
    gates = {
        "identity_matches": bool(
            summary.get("lane", {}).get("name") == lane
            and summary.get("instance_seed") == instance_seed
            and summary.get("method_seed") == method_seed
            and summary.get("budget") == 8
            and summary.get("submitted_lane_count") == 8
        ),
        "required_repeat_count": repeats >= required_repeat_count,
        "model_pair_preflight": bool(
            summary.get("model_pair_preflight", {}).get("passes_1e-3_gate")
        ),
        "tracked_provenance_clean": not summary.get("provenance", {}).get(
            "git_tracked_dirty", True
        ),
        "revised_numerical_noninferiority": bool(
            summary.get("revised_numerical_and_noninferiority_aggregate_pass")
        ),
        "has_optimizer_improved_success": (
            summary.get("optimizer_improved_success_count", 0) >= 1
        ),
        "selected_winner_has_no_cap": bool(
            winner is None or winner.get("pcg_cap_hits", 1) == 0
        ),
    }
    timing_required = spec in CENTRAL_TIMING_CASES or spec in SPOT_TIMING_CASES
    if timing_required:
        gates["solver_only_timing"] = bool(
            finite_at_least(timing.get("median_reduction_fraction"), 0.10)
            and finite_greater_than(interval.get("lower"), 0.05)
        )
    failures = [name for name, passed in gates.items() if not passed]
    multistart = summary.get("multistart_quality_vs_cold", {}).get("batched", {})
    return {
        "spec": list(spec),
        "artifact": str(path),
        "artifact_sha256": sha256_file(path),
        "repeat_count": repeats,
        "required_repeat_count": required_repeat_count,
        "optimizer_improved_success_count": summary.get(
            "optimizer_improved_success_count", 0
        ),
        "submitted_lane_count": summary.get("submitted_lane_count", 0),
        "pcg_cap_lane_count": summary.get("pcg_cap_lane_count", 0),
        "git_head": summary.get("provenance", {}).get("git_head"),
        "extension_sha256": summary.get("provenance", {}).get("extension_sha256"),
        "model_sha256": summary.get("provenance", {}).get("model_sha256"),
        "best_is_non_cold": multistart.get("best_is_non_cold", False),
        "terminal_error_reduction_vs_cold_fraction": multistart.get(
            "terminal_error_reduction_vs_cold_fraction"
        ),
        "external_cost_reduction_vs_cold_fraction": multistart.get(
            "external_cost_reduction_vs_cold_fraction"
        ),
        "material_non_cold_multistart_gain": bool(
            multistart.get("best_is_non_cold", False)
            and multistart.get("external_cost_reduction_vs_cold_fraction") is not None
            and multistart["external_cost_reduction_vs_cold_fraction"] >= 0.01
        ),
        "median_solver_only_reduction_fraction": timing.get(
            "median_reduction_fraction"
        ),
        "bootstrap_95_ci_median_solver_reduction": interval,
        "gates": gates,
        "failures": failures,
    }


def aggregate(case_rows):
    completed = [row for row in case_rows if "load_error" not in row]
    successful_cases = [
        row
        for row in completed
        if row["gates"].get("has_optimizer_improved_success", False)
    ]
    lane_successes = {
        lane: sum(
            row["gates"].get("has_optimizer_improved_success", False)
            for row in completed
            if row["spec"][0] == lane
        )
        for lane in LANES
    }
    lane_optimizer_successes = {
        lane: sum(
            row["optimizer_improved_success_count"]
            for row in completed
            if row["spec"][0] == lane
        )
        for lane in LANES
    }
    submitted_lanes = sum(row["submitted_lane_count"] for row in completed)
    cap_lanes = sum(row["pcg_cap_lane_count"] for row in completed)
    cap_rate = cap_lanes / submitted_lanes if submitted_lanes else 1.0
    expected_count = len(LANES) * len(INSTANCE_SEEDS) * len(METHOD_SEEDS)
    timing_rows = [
        row
        for row in completed
        if tuple(row["spec"]) in CENTRAL_TIMING_CASES | SPOT_TIMING_CASES
    ]
    useful_multistart_cases = [
        row for row in completed if row.get("material_non_cold_multistart_gain", False)
    ]
    useful_multistart_by_lane = {
        lane: sum(
            row.get("material_non_cold_multistart_gain", False)
            for row in completed
            if row["spec"][0] == lane
        )
        for lane in LANES
    }
    git_heads = {row.get("git_head") for row in completed}
    extension_hashes = {row.get("extension_sha256") for row in completed}
    model_hashes = {row.get("model_sha256") for row in completed}
    gates = {
        "all_18_cases_present": len(completed) == expected_count,
        "all_case_identities_and_batch_sizes_match": bool(completed)
        and all(row["gates"].get("identity_matches", False) for row in completed),
        "at_least_15_of_18_cases_have_improved_success": (
            len(successful_cases) >= 15
        ),
        "at_least_7_of_9_cases_succeed_in_each_lane": all(
            count >= 7 for count in lane_successes.values()
        ),
        "multiple_optimizer_successes_in_each_lane": all(
            count >= 2 for count in lane_optimizer_successes.values()
        ),
        "at_least_9_of_18_cases_have_material_non_cold_gain": (
            len(useful_multistart_cases) >= 9
        ),
        "at_least_4_of_9_cases_have_material_non_cold_gain_in_each_lane": all(
            count >= 4 for count in useful_multistart_by_lane.values()
        ),
        "submitted_lane_cap_rate_below_10_percent": cap_rate < 0.10,
        "all_selected_winners_have_no_cap": bool(completed)
        and all(
            row["gates"].get("selected_winner_has_no_cap", False)
            for row in completed
        ),
        "all_model_pair_preflights_pass": bool(completed)
        and all(row["gates"].get("model_pair_preflight", False) for row in completed),
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
        "all_revised_numerical_noninferiority_checks_pass": bool(completed)
        and all(
            row["gates"].get("revised_numerical_noninferiority", False)
            for row in completed
        ),
        "all_predeclared_timing_cases_present_and_pass": (
            len(timing_rows) == len(CENTRAL_TIMING_CASES | SPOT_TIMING_CASES)
            and all(row["gates"].get("solver_only_timing", False) for row in timing_rows)
        ),
        "all_cases_have_required_repeat_counts": bool(completed)
        and all(
            row["gates"].get("required_repeat_count", False) for row in completed
        ),
    }
    gates["all_campaign_acceptance_gates_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "campaign": "iiwa_same_problem_imperfect_multistart_sqp",
        "scope": (
            "same-goal cold multistart trajectory optimization; this benchmark does "
            "not claim multimodality"
        ),
        "predeclared_matrix": {
            "lanes": list(LANES),
            "instance_seeds": list(INSTANCE_SEEDS),
            "method_seeds": list(METHOD_SEEDS),
            "central_20_repeat_timing_cases": [
                list(spec) for spec in sorted(CENTRAL_TIMING_CASES)
            ],
            "five_repeat_timing_spot_cases": [
                list(spec) for spec in sorted(SPOT_TIMING_CASES)
            ],
        },
        "completed_case_count": len(completed),
        "successful_case_count": len(successful_cases),
        "successful_case_fraction": (
            len(successful_cases) / expected_count if expected_count else 0.0
        ),
        "successful_case_count_by_lane": lane_successes,
        "optimizer_improved_candidate_count_by_lane": lane_optimizer_successes,
        "material_non_cold_gain_case_count": len(useful_multistart_cases),
        "material_non_cold_gain_case_count_by_lane": useful_multistart_by_lane,
        "submitted_lane_count": submitted_lanes,
        "pcg_cap_lane_count": cap_lanes,
        "pcg_cap_lane_fraction": cap_rate,
        "case_count_with_non_cold_best": sum(
            row.get("best_is_non_cold", False) for row in completed
        ),
        "consistent_provenance": {
            "git_heads": sorted(value for value in git_heads if value is not None),
            "extension_sha256": sorted(
                value for value in extension_hashes if value is not None
            ),
            "model_sha256": sorted(value for value in model_hashes if value is not None),
        },
        "acceptance_gates": gates,
        "failed_cases": [row for row in case_rows if row["failures"]],
        "cases": case_rows,
    }


def run_case(output_dir, spec, repeats, budget, warmups):
    lane, instance_seed, method_seed = spec
    destination = output_dir / f"{case_name(*spec)}.json"
    command = [
        sys.executable,
        "-B",
        str(BENCHMARK),
        "--lane",
        lane,
        "--instance-seed",
        str(instance_seed),
        "--method-seed",
        str(method_seed),
        "--budget",
        str(budget),
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
    log = {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    destination.with_suffix(".log.json").write_text(
        json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination, result.returncode


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = list(itertools.product(LANES, INSTANCE_SEEDS, METHOD_SEEDS))
    execution_failures = []
    if not args.summarize_only:
        for spec in specs:
            repeats = required_repeats(spec, args.quality_repeats)
            path, returncode = run_case(
                args.output_dir, spec, repeats, args.budget, args.warmups
            )
            if returncode:
                execution_failures.append(
                    {"spec": list(spec), "artifact": str(path), "returncode": returncode}
                )
    rows = []
    for spec in specs:
        path = args.output_dir / f"{case_name(*spec)}.json"
        rows.append(load_case(path, spec, required_repeats(spec, args.quality_repeats)))
    summary = aggregate(rows)
    summary["execution_failures"] = execution_failures
    output = args.output_dir / "campaign_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(json.dumps(summary["acceptance_gates"], indent=2, sort_keys=True))
    return 0 if summary["acceptance_gates"]["all_campaign_acceptance_gates_pass"] else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=8, choices=(8,))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--quality-repeats", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
