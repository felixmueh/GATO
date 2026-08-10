import importlib.util
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXPANSION = load_module(
    "benchmark_batched_sqp_reach_expansion",
    "examples/benchmark_batched_sqp_reach_expansion.py",
)
CAMPAIGN = load_module(
    "run_batched_sqp_reach_expansion",
    "examples/run_batched_sqp_reach_expansion.py",
)


def passing_rows():
    rows = []
    specs = EXPANSION.problem_specs()
    for index, spec in enumerate(specs):
        cold_success = index < 8
        batch_success = index < 20
        rescued = not cold_success and batch_success
        row = {
            "spec": list(spec),
            "artifact": f"{CAMPAIGN.case_name(*spec)}.json",
            "repeat_count": CAMPAIGN.required_repeats(*spec),
            "cold_success": cold_success,
            "batch8_success": batch_success,
            "rescued_by_batch8": rescued,
            "rescued_by_non_cold_candidate": rescued,
            "cold_success_lost": False,
            "material_non_cold_gain_on_rescue": rescued and index < 14,
            "cold_complete_latency_ms": 10.0 + index,
            "batch8_complete_latency_ms": 30.0 + index,
            "git_head": "a" * 40,
            "extension_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "gates": {
                "identity_and_frozen_protocol": True,
                "accepted_harness_content_frozen": True,
                "required_repeat_count": True,
                "exact_cold_candidate_nested": True,
                "model_pair_preflight": True,
                "tracked_provenance_clean": True,
                "all_repeatability_checks": True,
                "batch8_vs_serial8_noninferiority": True,
                "cold_batch1_matches_serial8_candidate0": True,
                "batch8_candidate0_noninferior_to_cold_batch1": True,
                "cold_success_not_lost": True,
                "nested_cold_success_preserved": True,
                **(
                    {"central_batch8_vs_serial8_timing": True}
                    if spec[1] == EXPANSION.CENTRAL_TIMING_INSTANCE
                    else {}
                ),
            },
            "failures": [],
        }
        rows.append(row)
    return rows


def test_predeclared_grid_contains_24_unique_problems_not_method_repeats():
    specs = EXPANSION.problem_specs()

    assert len(specs) == 24
    assert len(set(specs)) == 24
    assert {offset for offset, _ in specs} == {"near", "far", "cross"}
    assert {seed for _, seed in specs} == set(range(43, 51))
    assert EXPANSION.PRIMARY_METHOD_SEED == 1000


def test_frozen_source_hash_and_exact_cold_nesting():
    assert (
        EXPANSION.accepted.sha256_file(Path(EXPANSION.accepted.__file__))
        == EXPANSION.ACCEPTED_HARNESS_SHA256
    )
    model = pin.buildModelFromUrdf(str(EXPANSION.accepted.MODEL_PATH))
    problem = EXPANSION.prepare_problem(model, "cross", 43)
    cold, _ = EXPANSION.exact_cold_candidate(model, problem["q0"])

    np.testing.assert_array_equal(cold, problem["seeds"][0])
    q, qd, controls = EXPANSION.accepted.unpack(cold, model)
    np.testing.assert_array_equal(q, np.tile(problem["q0"], (8, 1)))
    np.testing.assert_array_equal(qd, 0.0)
    np.testing.assert_array_equal(controls, 0.0)


def test_all_24_problem_tuples_have_unique_x0_goal_pairs():
    model = pin.buildModelFromUrdf(str(EXPANSION.accepted.MODEL_PATH))
    hashes = set()
    for offset, instance_seed in EXPANSION.problem_specs():
        problem = EXPANSION.prepare_problem(model, offset, instance_seed)
        hashes.add(
            (
                EXPANSION.accepted.sha256_array(problem["x0"]),
                EXPANSION.accepted.sha256_array(problem["goal"]),
            )
        )

    assert len(hashes) == 24


def test_problem_level_bootstrap_uses_binary_paired_gain():
    paired = np.asarray([1.0] * 12 + [0.0] * 12)
    first = CAMPAIGN.bootstrap_mean_interval(paired)
    repeated = CAMPAIGN.bootstrap_mean_interval(paired)

    assert first == repeated
    assert first["lower"] > 0.05
    assert first["upper"] <= 1.0


def test_aggregate_accepts_only_when_all_frozen_extension_gates_pass():
    summary = CAMPAIGN.aggregate(passing_rows())

    assert summary["completed_problem_count"] == 24
    assert summary["cold_success_count"] == 8
    assert summary["cold_failure_count"] == 16
    assert summary["batch8_success_count"] == 20
    assert summary["paired_rescue_table"] == {
        "cold_fail_batch_success": 12,
        "both_success": 8,
        "both_fail": 4,
        "cold_success_batch_fail": 0,
    }
    assert summary["material_non_cold_fraction_of_rescues"] == 0.5
    latency = summary["complete_latency_across_problems_ms"]
    assert latency["cold_batch1_median"] == 21.5
    assert latency["batch8_multistart_median"] == 41.5
    assert latency["median_paired_batch8_minus_cold_ms"] == 20.0
    assert latency["median_paired_batch8_to_cold_ratio"] > 1.0
    assert summary["acceptance_gates"]["all_expansion_acceptance_gates_pass"]


def test_aggregate_rejects_insufficient_gain_with_a_mixed_cold_suite():
    rows = passing_rows()
    for row in rows:
        if row["rescued_by_batch8"]:
            row["batch8_success"] = False
            row["rescued_by_batch8"] = False
            row["rescued_by_non_cold_candidate"] = False
            row["material_non_cold_gain_on_rescue"] = False
    summary = CAMPAIGN.aggregate(rows)

    assert summary["cold_success_count"] == 8
    assert summary["cold_failure_count"] == 16
    assert not summary["acceptance_gates"][
        "paired_success_gain_at_least_20_percentage_points"
    ]
    assert not summary["acceptance_gates"]["all_expansion_acceptance_gates_pass"]


def test_aggregate_rejects_no_cold_mix_and_any_unconsumed_case_gate():
    rows = passing_rows()
    for row in rows:
        row["cold_success"] = True
        row["batch8_success"] = True
        row["rescued_by_batch8"] = False
        row["rescued_by_non_cold_candidate"] = False
        row["material_non_cold_gain_on_rescue"] = False
    rows[0]["gates"]["nested_cold_success_preserved"] = False
    rows[0]["failures"] = ["nested_cold_success_preserved"]
    summary = CAMPAIGN.aggregate(rows)

    assert not summary["acceptance_gates"]["at_least_five_cold_failures"]
    assert not summary["acceptance_gates"]["all_required_per_case_gates_pass"]
    assert len(summary["failed_cases"]) == 1
    assert not summary["acceptance_gates"]["all_expansion_acceptance_gates_pass"]
