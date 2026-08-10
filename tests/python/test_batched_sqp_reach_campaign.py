import importlib.util
import itertools
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples/run_batched_sqp_reach_campaign.py"
SPEC = importlib.util.spec_from_file_location(
    "run_batched_sqp_reach_campaign", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def passing_case(spec):
    return {
        "spec": list(spec),
        "artifact": f"{MODULE.case_name(*spec)}.json",
        "repeat_count": MODULE.required_repeats(spec, 5),
        "required_repeat_count": MODULE.required_repeats(spec, 5),
        "optimizer_improved_success_count": 2,
        "submitted_lane_count": 8,
        "pcg_cap_lane_count": 0,
        "git_head": "a" * 40,
        "extension_sha256": "b" * 64,
        "model_sha256": "c" * 64,
        "best_is_non_cold": True,
        "material_non_cold_multistart_gain": True,
        "gates": {
            "identity_matches": True,
            "required_repeat_count": True,
            "model_pair_preflight": True,
            "tracked_provenance_clean": True,
            "revised_numerical_noninferiority": True,
            "has_optimizer_improved_success": True,
            "selected_winner_has_no_cap": True,
            **(
                {"solver_only_timing": True}
                if spec in MODULE.CENTRAL_TIMING_CASES | MODULE.SPOT_TIMING_CASES
                else {}
            ),
        },
        "failures": [],
    }


def all_cases():
    return [
        passing_case(spec)
        for spec in itertools.product(
            MODULE.LANES, MODULE.INSTANCE_SEEDS, MODULE.METHOD_SEEDS
        )
    ]


def test_predeclared_repeat_counts_distinguish_central_and_spot_cases():
    assert MODULE.required_repeats(("near", 41, 1001), 5) == 20
    assert MODULE.required_repeats(("far", 42, 1000), 5) == 5
    assert MODULE.required_repeats(("far", 42, 1002), 5) == 5


def test_aggregate_enforces_full_matrix_and_all_campaign_gates():
    summary = MODULE.aggregate(all_cases())

    assert summary["completed_case_count"] == 18
    assert summary["successful_case_count"] == 18
    assert summary["acceptance_gates"]["all_campaign_acceptance_gates_pass"]


def test_aggregate_retains_failures_and_rejects_missing_quality_successes():
    cases = all_cases()
    for row in cases[:4]:
        row["gates"]["has_optimizer_improved_success"] = False
        row["optimizer_improved_success_count"] = 0
        row["failures"] = ["has_optimizer_improved_success"]
    summary = MODULE.aggregate(cases)

    assert summary["successful_case_count"] == 14
    assert not summary["acceptance_gates"][
        "at_least_15_of_18_cases_have_improved_success"
    ]
    assert not summary["acceptance_gates"]["all_campaign_acceptance_gates_pass"]
    assert len(summary["failed_cases"]) == 4


def test_aggregate_rejects_cap_rate_and_inconsistent_provenance():
    cases = all_cases()
    cases[0]["pcg_cap_lane_count"] = 8
    cases[1]["pcg_cap_lane_count"] = 8
    cases[-1]["extension_sha256"] = "d" * 64
    summary = MODULE.aggregate(cases)

    assert not summary["acceptance_gates"][
        "submitted_lane_cap_rate_below_10_percent"
    ]
    assert not summary["acceptance_gates"][
        "all_artifacts_share_git_extension_and_model_hashes"
    ]
    assert not summary["acceptance_gates"]["all_campaign_acceptance_gates_pass"]


def test_unsuccessful_case_has_vacuous_no_cap_winner_gate(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "lane": {"name": "near"},
                "instance_seed": 40,
                "method_seed": 1000,
                "budget": 8,
                "submitted_lane_count": 8,
                "raw_repeats": [{}] * 5,
                "batched_selected_winner": None,
            }
        ),
        encoding="utf-8",
    )
    row = MODULE.load_case(path, ("near", 40, 1000), 5)

    assert row["gates"]["selected_winner_has_no_cap"]
    assert not row["gates"]["has_optimizer_improved_success"]
