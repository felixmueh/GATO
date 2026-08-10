"""Static-only Stage-0 v2 contract for the frozen point-mass family.

V1 remains a rejected pilot. This module deliberately contains no task
generation or execution entry point; it only freezes fresh identities and the
v2 fail-closed aggregation semantics for adversarial review.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from gato_pointmass.pointmass2d import PROTOCOL
from pointmass_examples.pointmass2d_stage0_oracle import (
    MODE_SIDES,
    ORACLE_RESTART_MARGINS_M,
    TRUST_CONSTR_OPTIONS,
    restart_identity,
)


STAGE0_V2_DEVELOPMENT_TASK_SEEDS = (7200, 7201, 7202)
STAGE0_V2_HELDOUT_TASK_SEEDS = (7300, 7301, 7302, 7303, 7304)
STAGE0_V2_DEVELOPMENT_METHOD_SEEDS = (8200, 8201, 8202, 8203, 8204)
STAGE0_V2_HELDOUT_METHOD_SEEDS = tuple(range(8300, 8320))
STAGE0_V2_CENTRAL_TASK_SEED = 7302
STAGE0_V2_VERSION = "stage0_v2_fresh_acquisition_robustness"
STAGE0_V2_STATUS = "static_only_not_executed"


def stage0_v2_expected_identities():
    return [
        restart_identity(task_seed, mode_side, restart_index)
        for task_seed in (
            STAGE0_V2_DEVELOPMENT_TASK_SEEDS + STAGE0_V2_HELDOUT_TASK_SEEDS
        )
        for mode_side in MODE_SIDES
        for restart_index in range(len(ORACLE_RESTART_MARGINS_M))
    ]


def _finite_scalars(*values):
    return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))


def _mode_threshold_pass(phase):
    return bool(
        (
            phase["mode"] == "clockwise"
            and phase["signed_turns"] <= -0.25
        )
        or (
            phase["mode"] == "counterclockwise"
            and phase["signed_turns"] >= 0.25
        )
    )


def acquisition_numeric_gate(row):
    acquisition = row["acquisition"]
    return bool(
        acquisition["success"]
        and acquisition["finite"]
        and _finite_scalars(
            acquisition["objective"],
            acquisition["reported_optimality"],
            acquisition["reported_constraint_violation"],
            row["independent_acquisition_violation"],
        )
        and row["independent_acquisition_violation"] <= 1e-6
    )


def unanchored_original_task_gate(row):
    final = row["unanchored_polish"]
    return bool(
        final["success"]
        and final["finite"]
        and _finite_scalars(
            final["objective"],
            final["signed_turns"],
            final["terminal_position_error_m"],
            final["terminal_velocity_error_m_s"],
            final["minimum_physical_clearance_m"],
            final["maximum_position_component_m"],
            final["maximum_velocity_component_m_s"],
            final["maximum_control_component_m_s2"],
            row["independent_original_task_violation"],
            row["independent_original_task_stationarity_inf"],
            row["independent_original_task_dual_sign_inf"],
            row["independent_original_task_complementarity_inf"],
        )
        and row["independent_original_task_violation"] <= 1e-6
        and row["independent_original_task_stationarity_inf"] <= 1e-5
        and row["independent_original_task_dual_sign_inf"] <= 1e-5
        and row["independent_original_task_complementarity_inf"] <= 1e-5
        and final["terminal_position_error_m"] <= 0.01
        and final["terminal_velocity_error_m_s"] <= 0.02
        and final["minimum_physical_clearance_m"] >= 0.02
        and final["maximum_position_component_m"] <= PROTOCOL.position_limit
        and final["maximum_velocity_component_m_s"] <= PROTOCOL.velocity_limit
        and final["maximum_control_component_m_s2"]
        <= PROTOCOL.acceleration_limit
        and _mode_threshold_pass(final)
    )


def intended_mode_match(row):
    expected = row["expected_mode"]
    acquisition = row["acquisition"]
    final = row["unanchored_polish"]
    return bool(
        acquisition["mode"] == expected
        and final["mode"] == expected
        and _mode_threshold_pass(acquisition)
        and _mode_threshold_pass(final)
    )


def intended_stability_gate(row):
    if not intended_mode_match(row):
        return True
    return bool(
        row["acquisition_to_polish_cost_change_fraction"] <= 0.01
        and row["acquisition_to_polish_dense_path_rms_m"] <= 0.01
    )


def frozen_phase_provenance_gate(row):
    restart_index = row.get("restart_index")
    if not isinstance(restart_index, (int, np.integer)):
        return False
    if not 0 <= int(restart_index) < len(ORACLE_RESTART_MARGINS_M):
        return False
    return bool(
        row.get("restart_margin_m")
        == ORACLE_RESTART_MARGINS_M[int(restart_index)]
        and row.get("anchor_used_for_acquisition_only") is True
        and row.get("anchor_present_in_certifying_polish") is False
    )


def _pair_record(first, second, same_mode):
    first_turn = first["unanchored_polish"]["signed_turns"]
    second_turn = second["unanchored_polish"]["signed_turns"]
    delta = float(second_turn - first_turn)
    nearest = int(np.rint(delta))
    residual = float(abs(delta - nearest))
    return {
        "first_key": first["identity_key"],
        "second_key": second["identity_key"],
        "delta_turns": delta,
        "nearest_integer": nearest,
        "nearest_integer_residual": residual,
        "class_gate": bool(
            residual <= 0.10
            and (nearest == 0 if same_mode else abs(nearest) == 1)
        ),
    }


def stage0_v2_task_aggregate(task, rows):
    expected_modes = ("clockwise", "counterclockwise")
    requested = {
        mode: [row for row in rows if row["expected_mode"] == mode]
        for mode in expected_modes
    }
    actual = {
        mode: [
            row
            for row in rows
            if row["unanchored_polish"]["mode"] == mode
        ]
        for mode in expected_modes
    }
    exact_requested_identities = bool(
        len(rows) == 10
        and all(len(requested[mode]) == 5 for mode in expected_modes)
        and all(
            sorted(row["restart_index"] for row in requested[mode])
            == list(range(5))
            for mode in expected_modes
        )
        and len({row["identity_key"] for row in rows}) == 10
        and all(
            row["identity_key"]
            == (
                f"task{int(task['task_seed'])}_{row['expected_mode']}_"
                f"restart{int(row['restart_index'])}"
            )
            and row["mode_side"]
            == (1 if row["expected_mode"] == "clockwise" else -1)
            for row in rows
        )
    )
    acquisition_gates = {
        row["identity_key"]: acquisition_numeric_gate(row) for row in rows
    }
    phase_provenance_gates = {
        row["identity_key"]: frozen_phase_provenance_gate(row) for row in rows
    }
    final_gates = {
        row["identity_key"]: unanchored_original_task_gate(row) for row in rows
    }
    intended_matches = {
        mode: [row for row in requested[mode] if intended_mode_match(row)]
        for mode in expected_modes
    }
    intended_counts = {
        mode: len(values) for mode, values in intended_matches.items()
    }
    intended_stability = {
        row["identity_key"]: intended_stability_gate(row)
        for mode in expected_modes
        for row in intended_matches[mode]
    }

    same_pairs = {}
    for mode in expected_modes:
        for first, second in combinations(actual[mode], 2):
            key = f"{first['identity_key']}__{second['identity_key']}"
            same_pairs[key] = _pair_record(first, second, same_mode=True)
    cross_pairs = {}
    for clockwise in actual["clockwise"]:
        for counterclockwise in actual["counterclockwise"]:
            key = (
                f"{clockwise['identity_key']}__"
                f"{counterclockwise['identity_key']}"
            )
            cross_pairs[key] = _pair_record(
                clockwise, counterclockwise, same_mode=False
            )
    expected_same_pair_count = {
        mode: len(actual[mode]) * (len(actual[mode]) - 1) // 2
        for mode in expected_modes
    }
    retained_same_pair_count = {
        mode: sum(
            first["unanchored_polish"]["mode"] == mode
            for first, second in combinations(actual[mode], 2)
        )
        for mode in expected_modes
    }
    topology_gate = bool(
        all(len(actual[mode]) >= 4 for mode in expected_modes)
        and retained_same_pair_count == expected_same_pair_count
        and len(same_pairs) == sum(expected_same_pair_count.values())
        and len(cross_pairs)
        == len(actual["clockwise"]) * len(actual["counterclockwise"])
        and all(pair["class_gate"] for pair in same_pairs.values())
        and all(pair["class_gate"] for pair in cross_pairs.values())
    )

    correct_costs = {
        mode: [row["unanchored_polish"]["objective"] for row in intended_matches[mode]]
        for mode in expected_modes
    }
    cost_spreads = {
        mode: (
            (max(values) - min(values)) / max(abs(float(np.mean(values))), 1e-12)
            if values
            else np.inf
        )
        for mode, values in correct_costs.items()
    }
    predicted_worse = task["predicted_worse_mode"]
    better = (
        "counterclockwise" if predicted_worse == "clockwise" else "clockwise"
    )
    worse_cost = float(np.mean(correct_costs[predicted_worse]))
    better_cost = float(np.mean(correct_costs[better]))
    absolute_gap = worse_cost - better_cost
    fractional_gap = absolute_gap / max(abs(better_cost), 1e-12)

    aggregate = {
        "stage0_version": STAGE0_V2_VERSION,
        "task_seed": int(task["task_seed"]),
        "exact_ten_unique_requested_rows": exact_requested_identities,
        "acquisition_numeric_gates": acquisition_gates,
        "frozen_phase_provenance_gates": phase_provenance_gates,
        "all_ten_frozen_two_phase_provenance_gates_pass": bool(
            exact_requested_identities and all(phase_provenance_gates.values())
        ),
        "all_ten_acquisitions_numerical_finite_and_feasible": bool(
            exact_requested_identities and all(acquisition_gates.values())
        ),
        "unanchored_original_task_gates": final_gates,
        "all_ten_unanchored_polishes_certified_in_two_modes": bool(
            exact_requested_identities and all(final_gates.values())
        ),
        "intended_match_count_by_requested_mode": intended_counts,
        "at_least_four_of_five_intended_matches_each_side": all(
            intended_counts[mode] >= 4 for mode in expected_modes
        ),
        "intended_match_stability_gates": intended_stability,
        "all_intended_matches_stable": bool(
            intended_stability and all(intended_stability.values())
        ),
        "actual_mode_count": {
            mode: len(actual[mode]) for mode in expected_modes
        },
        "same_mode_pair_classes": same_pairs,
        "same_mode_expected_pair_count": expected_same_pair_count,
        "same_mode_retained_pair_count": retained_same_pair_count,
        "cross_mode_pair_classes": cross_pairs,
        "actual_mode_topology_gate": topology_gate,
        "correctly_intended_costs_by_mode": correct_costs,
        "correctly_intended_cost_spread_fraction": cost_spreads,
        "correctly_intended_cost_spread_at_most_1_percent": all(
            spread <= 0.01 for spread in cost_spreads.values()
        ),
        "predicted_worse_mode": predicted_worse,
        "better_mode": better,
        "worse_mode_correctly_intended_mean_cost": worse_cost,
        "better_mode_correctly_intended_mean_cost": better_cost,
        "worse_mode_absolute_gap": absolute_gap,
        "worse_mode_gap_fraction": fractional_gap,
        "predicted_worse_cost_strictly_greater": bool(absolute_gap > 0.0),
        "worse_mode_gap_at_least_5_percent_and_0_005": bool(
            absolute_gap >= 0.005 and fractional_gap >= 0.05
        ),
    }
    aggregate["all_required_stage0_v2_task_gates_pass"] = bool(
        aggregate["exact_ten_unique_requested_rows"]
        and aggregate["all_ten_frozen_two_phase_provenance_gates_pass"]
        and aggregate["all_ten_acquisitions_numerical_finite_and_feasible"]
        and aggregate["all_ten_unanchored_polishes_certified_in_two_modes"]
        and aggregate["at_least_four_of_five_intended_matches_each_side"]
        and aggregate["all_intended_matches_stable"]
        and aggregate["actual_mode_topology_gate"]
        and aggregate["correctly_intended_cost_spread_at_most_1_percent"]
        and aggregate["predicted_worse_cost_strictly_greater"]
    )
    return aggregate


def stage0_v2_campaign_aggregate(task_rows):
    expected_task_seeds = (
        STAGE0_V2_DEVELOPMENT_TASK_SEEDS + STAGE0_V2_HELDOUT_TASK_SEEDS
    )
    exact_tasks = bool(
        len(task_rows) == 8
        and tuple(row["task_seed"] for row in task_rows) == expected_task_seeds
        and len({row["task_seed"] for row in task_rows}) == 8
        and all(
            row["task_seed"] == row["task"]["task_seed"]
            == row["aggregate"]["task_seed"]
            for row in task_rows
        )
    )
    heldout = [
        row for row in task_rows if row["task_seed"] in STAGE0_V2_HELDOUT_TASK_SEEDS
    ]
    heldout_gap_count = sum(
        row["aggregate"]["worse_mode_gap_at_least_5_percent_and_0_005"]
        for row in heldout
    )
    all_tasks_pass = bool(
        exact_tasks
        and all(
            row["aggregate"].get("stage0_version") == STAGE0_V2_VERSION
            and row["aggregate"]["all_required_stage0_v2_task_gates_pass"]
            for row in task_rows
        )
    )
    all_aggregate_versions_exact = bool(
        exact_tasks
        and all(
            row["aggregate"].get("stage0_version") == STAGE0_V2_VERSION
            for row in task_rows
        )
    )
    predicted_worse_never_reverses = bool(
        exact_tasks
        and all(
            row["aggregate"]["predicted_worse_cost_strictly_greater"]
            for row in task_rows
        )
    )
    return {
        "stage0_version": STAGE0_V2_VERSION,
        "status_before_execution": STAGE0_V2_STATUS,
        "exact_eight_fresh_task_identities": exact_tasks,
        "all_eight_aggregate_versions_exact": all_aggregate_versions_exact,
        "all_eight_tasks_pass_required_v2_gates": all_tasks_pass,
        "predicted_worse_cost_greater_on_all_eight_tasks": (
            predicted_worse_never_reverses
        ),
        "heldout_task_count": len(heldout),
        "heldout_tasks_with_predeclared_cost_gap": int(heldout_gap_count),
        "at_least_four_of_five_heldout_gaps": bool(
            len(heldout) == 5 and heldout_gap_count >= 4
        ),
        "v2_failure_closes_pointmass_without_v3": True,
        "all_stage0_v2_gates_pass": bool(
            all_tasks_pass
            and all_aggregate_versions_exact
            and predicted_worse_never_reverses
            and len(heldout) == 5
            and heldout_gap_count >= 4
        ),
    }
