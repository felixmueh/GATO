from copy import deepcopy
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "pointmass_src")]

from pointmass_examples.pointmass2d_stage0_oracle import (  # noqa: E402
    ORACLE_RESTART_MARGINS_M,
    TRUST_CONSTR_OPTIONS,
)
from pointmass_examples.pointmass2d_stage0_v2 import (  # noqa: E402
    PROTOCOL,
    STAGE0_V2_CENTRAL_TASK_SEED,
    STAGE0_V2_DEVELOPMENT_METHOD_SEEDS,
    STAGE0_V2_DEVELOPMENT_TASK_SEEDS,
    STAGE0_V2_HELDOUT_METHOD_SEEDS,
    STAGE0_V2_HELDOUT_TASK_SEEDS,
    STAGE0_V2_STATUS,
    stage0_v2_campaign_aggregate,
    stage0_v2_expected_identities,
    stage0_v2_task_aggregate,
)


def _opposite(mode):
    return "counterclockwise" if mode == "clockwise" else "clockwise"


def _turn(mode):
    return -0.5 if mode == "clockwise" else 0.5


def _synthetic_rows(task_seed, miss_restarts=(4,)):
    rows = []
    for expected_mode, mode_side in (("clockwise", 1), ("counterclockwise", -1)):
        for restart_index in range(5):
            actual_mode = (
                _opposite(expected_mode)
                if restart_index in miss_restarts
                else expected_mode
            )
            correctly_intended = actual_mode == expected_mode
            objective = (
                (1.10 if actual_mode == "clockwise" else 1.00)
                + 0.0002 * restart_index
                if correctly_intended
                else 1000.0 + restart_index
            )
            phase_common = {
                "success": True,
                "finite": True,
                "objective": objective,
                "mode": actual_mode,
                "signed_turns": _turn(actual_mode),
            }
            rows.append(
                {
                    "identity_key": (
                        f"task{task_seed}_{expected_mode}_restart{restart_index}"
                    ),
                    "mode_side": mode_side,
                    "expected_mode": expected_mode,
                    "restart_index": restart_index,
                    "restart_margin_m": ORACLE_RESTART_MARGINS_M[restart_index],
                    "anchor_used_for_acquisition_only": True,
                    "anchor_present_in_certifying_polish": False,
                    "acquisition": {
                        **phase_common,
                        "reported_optimality": 1e-10,
                        "reported_constraint_violation": 0.0,
                    },
                    "unanchored_polish": {
                        **phase_common,
                        "terminal_position_error_m": 1e-12,
                        "terminal_velocity_error_m_s": 1e-12,
                        "minimum_physical_clearance_m": 0.0201,
                        "maximum_position_component_m": 0.35,
                        "maximum_velocity_component_m_s": 0.7,
                        "maximum_control_component_m_s2": 2.8,
                    },
                    "independent_acquisition_violation": 0.0,
                    "independent_original_task_violation": 0.0,
                    "independent_original_task_stationarity_inf": 1e-10,
                    "independent_original_task_dual_sign_inf": 0.0,
                    "independent_original_task_complementarity_inf": 1e-9,
                    "acquisition_to_polish_cost_change_fraction": (
                        0.005 if correctly_intended else 99.0
                    ),
                    "acquisition_to_polish_dense_path_rms_m": (
                        0.005 if correctly_intended else 99.0
                    ),
                }
            )
    return rows


def _task(task_seed, predicted_worse_mode="clockwise"):
    return {
        "task_seed": task_seed,
        "predicted_worse_mode": predicted_worse_mode,
    }


def test_v2_fresh_identities_are_frozen_without_task_instantiation():
    assert STAGE0_V2_DEVELOPMENT_TASK_SEEDS == (7200, 7201, 7202)
    assert STAGE0_V2_HELDOUT_TASK_SEEDS == (7300, 7301, 7302, 7303, 7304)
    assert STAGE0_V2_DEVELOPMENT_METHOD_SEEDS == (8200, 8201, 8202, 8203, 8204)
    assert STAGE0_V2_HELDOUT_METHOD_SEEDS == tuple(range(8300, 8320))
    assert STAGE0_V2_CENTRAL_TASK_SEED == 7302
    assert STAGE0_V2_STATUS == "static_only_not_executed"
    identities = stage0_v2_expected_identities()
    assert len(identities) == 80
    assert len({row["key"] for row in identities}) == 80
    assert tuple(dict.fromkeys(row["task_seed"] for row in identities)) == (
        STAGE0_V2_DEVELOPMENT_TASK_SEEDS + STAGE0_V2_HELDOUT_TASK_SEEDS
    )
    assert ORACLE_RESTART_MARGINS_M == (0.0, 0.015, 0.030, 0.045, 0.060)
    assert PROTOCOL.knots == 32 and PROTOCOL.dt == 0.05
    assert TRUST_CONSTR_OPTIONS["maxiter"] == 2000
    source = (
        REPO_ROOT / "pointmass_examples/pointmass2d_stage0_v2.py"
    ).read_text()
    assert "generate_task" not in source
    assert "run_stage0" not in source


def test_v2_four_of_five_each_side_passes_and_unintended_costs_do_not_contaminate():
    task = _task(STAGE0_V2_DEVELOPMENT_TASK_SEEDS[0])
    rows = _synthetic_rows(task["task_seed"], miss_restarts=(4,))
    aggregate = stage0_v2_task_aggregate(task, rows)
    assert aggregate["all_ten_acquisitions_numerical_finite_and_feasible"]
    assert aggregate["all_ten_frozen_two_phase_provenance_gates_pass"]
    assert aggregate["all_ten_unanchored_polishes_certified_in_two_modes"]
    assert aggregate["intended_match_count_by_requested_mode"] == {
        "clockwise": 4,
        "counterclockwise": 4,
    }
    assert aggregate["at_least_four_of_five_intended_matches_each_side"]
    assert aggregate["all_intended_matches_stable"]
    assert aggregate["actual_mode_count"] == {
        "clockwise": 5,
        "counterclockwise": 5,
    }
    assert aggregate["actual_mode_topology_gate"]
    assert len(aggregate["same_mode_pair_classes"]) == 20
    assert len(aggregate["cross_mode_pair_classes"]) == 25
    assert aggregate["correctly_intended_cost_spread_at_most_1_percent"]
    assert aggregate["predicted_worse_cost_strictly_greater"]
    assert aggregate["worse_mode_gap_at_least_5_percent_and_0_005"]
    assert aggregate["all_required_stage0_v2_task_gates_pass"]

    baseline_gap = aggregate["worse_mode_absolute_gap"]
    baseline_costs = aggregate["correctly_intended_costs_by_mode"]
    contaminated = deepcopy(rows)
    for row in contaminated:
        if row["acquisition"]["mode"] != row["expected_mode"]:
            row["unanchored_polish"]["objective"] = -1e12
    contaminated_aggregate = stage0_v2_task_aggregate(task, contaminated)
    assert contaminated_aggregate["correctly_intended_costs_by_mode"] == baseline_costs
    assert contaminated_aggregate["worse_mode_absolute_gap"] == baseline_gap
    assert contaminated_aggregate["predicted_worse_cost_strictly_greater"]


def test_v2_rejects_each_mutated_or_missing_two_phase_provenance_field():
    task = _task(STAGE0_V2_DEVELOPMENT_TASK_SEEDS[0])
    valid_rows = _synthetic_rows(task["task_seed"], miss_restarts=(4,))
    mutations = (
        ("restart_margin_m", 0.123),
        ("anchor_used_for_acquisition_only", False),
        ("anchor_present_in_certifying_polish", True),
    )
    for field, value in mutations:
        mutated = deepcopy(valid_rows)
        mutated[0][field] = value
        aggregate = stage0_v2_task_aggregate(task, mutated)
        assert not aggregate["frozen_phase_provenance_gates"][
            mutated[0]["identity_key"]
        ]
        assert not aggregate["all_ten_frozen_two_phase_provenance_gates_pass"]
        assert not aggregate["all_required_stage0_v2_task_gates_pass"]

        missing = deepcopy(valid_rows)
        del missing[0][field]
        missing_aggregate = stage0_v2_task_aggregate(task, missing)
        assert not missing_aggregate["frozen_phase_provenance_gates"][
            missing[0]["identity_key"]
        ]
        assert not missing_aggregate[
            "all_ten_frozen_two_phase_provenance_gates_pass"
        ]
        assert not missing_aggregate["all_required_stage0_v2_task_gates_pass"]


def test_v2_three_of_five_intended_matches_fails_even_with_valid_actual_modes():
    task = _task(STAGE0_V2_DEVELOPMENT_TASK_SEEDS[0])
    rows = _synthetic_rows(task["task_seed"], miss_restarts=(3, 4))
    aggregate = stage0_v2_task_aggregate(task, rows)
    assert aggregate["all_ten_acquisitions_numerical_finite_and_feasible"]
    assert aggregate["all_ten_unanchored_polishes_certified_in_two_modes"]
    assert aggregate["intended_match_count_by_requested_mode"] == {
        "clockwise": 3,
        "counterclockwise": 3,
    }
    assert not aggregate["at_least_four_of_five_intended_matches_each_side"]
    assert aggregate["actual_mode_count"] == {
        "clockwise": 5,
        "counterclockwise": 5,
    }
    assert aggregate["actual_mode_topology_gate"]
    assert not aggregate["all_required_stage0_v2_task_gates_pass"]


def test_v2_campaign_is_exact_eight_tasks_fail_closed_and_never_reverses_worse():
    task_seeds = STAGE0_V2_DEVELOPMENT_TASK_SEEDS + STAGE0_V2_HELDOUT_TASK_SEEDS
    task_rows = []
    for task_seed in task_seeds:
        task = _task(task_seed)
        aggregate = stage0_v2_task_aggregate(
            task, _synthetic_rows(task_seed, miss_restarts=(4,))
        )
        task_rows.append(
            {"task_seed": task_seed, "task": task, "aggregate": aggregate}
        )
    campaign = stage0_v2_campaign_aggregate(task_rows)
    assert campaign["exact_eight_fresh_task_identities"]
    assert campaign["all_eight_aggregate_versions_exact"]
    assert campaign["predicted_worse_cost_greater_on_all_eight_tasks"]
    assert campaign["heldout_task_count"] == 5
    assert campaign["heldout_tasks_with_predeclared_cost_gap"] == 5
    assert campaign["at_least_four_of_five_heldout_gaps"]
    assert campaign["all_stage0_v2_gates_pass"]
    assert campaign["v2_failure_closes_pointmass_without_v3"]

    reversed_rows = deepcopy(task_rows)
    reversed_rows[0]["task"]["predicted_worse_mode"] = "counterclockwise"
    reversed_rows[0]["aggregate"] = stage0_v2_task_aggregate(
        reversed_rows[0]["task"],
        _synthetic_rows(reversed_rows[0]["task_seed"], miss_restarts=(4,)),
    )
    rejected = stage0_v2_campaign_aggregate(reversed_rows)
    assert not rejected["predicted_worse_cost_greater_on_all_eight_tasks"]
    assert not rejected["all_stage0_v2_gates_pass"]

    missing = stage0_v2_campaign_aggregate(task_rows[:-1])
    assert not missing["exact_eight_fresh_task_identities"]
    assert not missing["all_stage0_v2_gates_pass"]

    wrong_version = deepcopy(task_rows)
    wrong_version[3]["aggregate"]["stage0_version"] = "stage0_v1"
    version_rejected = stage0_v2_campaign_aggregate(wrong_version)
    assert not version_rejected["all_eight_aggregate_versions_exact"]
    assert not version_rejected["all_stage0_v2_gates_pass"]
