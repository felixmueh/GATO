import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "pointmass_src")]

from gato_pointmass.pointmass2d import solver_task_identity_hash  # noqa: E402
from pointmass_examples.pointmass2d_stage0_oracle import (  # noqa: E402
    ORACLE_RESTART_MARGINS_M,
    partial_generation_paths,
    partial_latest_path,
    sha256_file,
)
from pointmass_examples.pointmass2d_stage0_v2 import (  # noqa: E402
    STAGE0_V2_VERSION,
)
import pointmass_examples.pointmass2d_stage0_v2_runner as v2_runner  # noqa: E402
from pointmass_examples.pointmass2d_stage0_v2_runner import (  # noqa: E402
    V1_PILOT_TASK_SEEDS,
    V2_TASK_SEEDS,
    run_stage0_v2,
)


def _fake_task(task_seed):
    x0 = np.asarray([task_seed * 1e-6, 0.0, 0.0, 0.0], dtype=np.float32)
    reference = np.asarray([0.35, 0.0, 0.0, 0.03, 0.10, 0.0], dtype=np.float32)
    return {
        "task_seed": int(task_seed),
        "x0": x0,
        "reference": reference,
        "solver_task_identity_sha256": solver_task_identity_hash(x0, reference),
        "canonical_numeric_source": "exact_solver_bound_float32_bytes",
        "predicted_worse_mode": "clockwise",
    }


def _opposite(mode):
    return "counterclockwise" if mode == "clockwise" else "clockwise"


def _fake_solve(problem, mode_side, restart_index):
    expected = "clockwise" if mode_side > 0 else "counterclockwise"
    actual = _opposite(expected) if restart_index == 4 else expected
    intended = actual == expected
    turns = -0.5 if actual == "clockwise" else 0.5
    objective = (
        (1.10 if actual == "clockwise" else 1.00) + 0.0002 * restart_index
        if intended
        else 1000.0 + restart_index
    )
    kkt = {
        "linear_multiplier": np.zeros(2),
        "clearance_multiplier": np.zeros(2),
        "linear_lower_slack": np.ones(2),
        "linear_upper_slack": np.ones(2),
        "clearance_lower_slack": np.ones(2),
        "stationarity_vector": np.zeros(2),
        "dual_sign_residual_vector": np.zeros(4),
        "complementarity_residual_vector": np.zeros(6),
    }
    phase_arrays = {
        "z": np.zeros(2),
        "controls": np.zeros((1, 2)),
        "dense_states": np.zeros((2, 4)),
        "independent_kkt": kkt,
    }
    phase_common = {
        "success": True,
        "finite": True,
        "objective": objective,
        "mode": actual,
        "signed_turns": turns,
    }
    return {
        "mode_side": int(mode_side),
        "expected_mode": expected,
        "restart_index": int(restart_index),
        "restart_margin_m": ORACLE_RESTART_MARGINS_M[restart_index],
        "anchor_used_for_acquisition_only": True,
        "anchor_present_in_certifying_polish": False,
        "acquisition_initial_z": np.zeros(2),
        "unanchored_polish_initial_z": np.zeros(2),
        "acquisition": {
            **phase_common,
            **phase_arrays,
            "reported_optimality": 1e-10,
            "reported_constraint_violation": 0.0,
        },
        "unanchored_polish": {
            **phase_common,
            **phase_arrays,
            "terminal_position_error_m": 0.0,
            "terminal_velocity_error_m_s": 0.0,
            "minimum_physical_clearance_m": 0.021,
            "maximum_position_component_m": 0.35,
            "maximum_velocity_component_m_s": 0.7,
            "maximum_control_component_m_s2": 2.8,
        },
        "independent_acquisition_violation": 0.0,
        "independent_original_task_violation": 0.0,
        "independent_original_task_stationarity_inf": 1e-10,
        "independent_original_task_dual_sign_inf": 0.0,
        "independent_original_task_complementarity_inf": 1e-9,
        "acquisition_to_polish_cost_change_fraction": 0.005 if intended else 99.0,
        "acquisition_to_polish_dense_path_rms_m": 0.005 if intended else 99.0,
    }


def test_v2_mock_runner_retains_final_but_rejects_simulated_head_drift(
    tmp_path, monkeypatch
):
    called_task_seeds = []
    solve_calls = []
    head_calls = 0

    def drifting_head():
        nonlocal head_calls
        head_calls += 1
        return ("a" if head_calls == 1 else "b") * 40

    monkeypatch.setattr(v2_runner, "_git_head", drifting_head)

    def task_factory(task_seed):
        called_task_seeds.append(task_seed)
        return _fake_task(task_seed)

    def solve_function(problem, mode_side, restart_index):
        solve_calls.append((problem.task["task_seed"], mode_side, restart_index))
        return _fake_solve(problem, mode_side, restart_index)

    output = tmp_path / "stage0-v2.json"
    summary = run_stage0_v2(
        output,
        dependency_overrides={
            "task_factory": task_factory,
            "problem_factory": lambda task: SimpleNamespace(task=task),
            "solve_function": solve_function,
        },
    )
    assert tuple(called_task_seeds) == V2_TASK_SEEDS
    assert not (set(called_task_seeds) & set(V1_PILOT_TASK_SEEDS))
    assert len(solve_calls) == 80
    assert summary["stage0_version"] == STAGE0_V2_VERSION
    assert summary["incomplete"] is False
    assert summary["expected_restart_count"] == 80
    assert summary["completed_restart_count"] == 80
    assert summary["pending_ordered_identities"] == []
    assert summary["completed_ordered_identities"] == summary[
        "expected_ordered_identities"
    ]
    assert all(
        row["identity_key"] == identity["key"]
        for task_row in summary["task_rows"]
        for row, identity in zip(
            task_row["rows"],
            [
                item
                for item in summary["expected_ordered_identities"]
                if item["task_seed"] == task_row["task_seed"]
            ],
        )
    )
    assert summary["campaign_aggregate"]["all_stage0_v2_gates_pass"]
    assert summary["v1_pilot_excluded"]
    assert summary["v1_rows_or_aggregates_consumed"] == 0
    assert summary["v1_artifacts_consumed"] == 0
    assert summary["source_hashes_stable_during_execution"]
    assert summary["provenance"]["source_hashes_stable"]
    assert summary["provenance"]["git_head_at_start"] == "a" * 40
    assert summary["provenance"]["git_head_at_end"] == "b" * 40
    assert not summary["provenance"]["git_head_stable_during_execution"]
    assert not summary["git_head_stable_during_execution"]
    assert summary["provenance"]["dependency_overrides_for_test_only"]
    assert not summary["all_stage0_v2_runner_gates_pass"]

    assert output.exists() and output.with_suffix(".npz").exists()
    manifest = output.with_suffix(".sha256.json")
    assert manifest.exists()
    manifest_row = json.loads(manifest.read_text())
    assert manifest_row["json_sha256"] == sha256_file(output)
    assert manifest_row["npz_sha256"] == sha256_file(output.with_suffix(".npz"))
    generation1_json, generation1_npz = partial_generation_paths(output, 1)
    generation80_json, generation80_npz = partial_generation_paths(output, 80)
    assert generation1_json.exists() and generation1_npz.exists()
    assert generation80_json.exists() and generation80_npz.exists()
    generation1 = json.loads(generation1_json.read_text())
    assert generation1["incomplete"] is True
    assert generation1["all_stage0_gates_pass"] is False
    assert generation1["all_stage0_gates_status"] == (
        "not_applicable_while_incomplete"
    )
    assert generation1["completed_restart_count"] == 1
    latest = json.loads(partial_latest_path(output).read_text())
    assert latest["latest_complete_generation"] == 80
    assert latest["generation_json_sha256"] == sha256_file(generation80_json)
    assert latest["generation_npz_sha256"] == sha256_file(generation80_npz)

    source_hashes = summary["provenance"]["source_hashes_at_start"]
    assert set(source_hashes) == {
        "task_schema",
        "v1_solver",
        "v2_contract",
        "v2_runner",
        "v1_tests",
        "v2_contract_tests",
        "v2_runner_tests",
    }
    assert all(len(value) == 64 for value in source_hashes.values())
    assert len(summary["provenance"]["task_identity_sha256_by_seed"]) == 8
    with np.load(output.with_suffix(".npz")) as arrays:
        assert len(arrays.files) > 80
        for task_seed in V2_TASK_SEEDS:
            assert f"task{task_seed}_solver_x0_float32" in arrays
            assert f"task{task_seed}_solver_reference_float32" in arrays

    runner_source = (
        REPO_ROOT / "pointmass_examples/pointmass2d_stage0_v2_runner.py"
    ).read_text()
    assert "stage0_v2_task_aggregate" in runner_source
    assert "stage0_v2_campaign_aggregate" in runner_source
    assert re.search(r"\btask_aggregate\b", runner_source) is None
    for v1_seed in V1_PILOT_TASK_SEEDS:
        assert str(v1_seed) not in [str(seed) for seed in called_task_seeds]


def test_v2_default_dirty_path_aborts_before_task_or_solve_and_writes_nothing(
    tmp_path, monkeypatch
):
    task_calls = 0
    solve_calls = 0

    def forbidden_task_factory(_task_seed):
        nonlocal task_calls
        task_calls += 1
        raise AssertionError("dirty default path instantiated a task")

    def forbidden_solve(*_args):
        nonlocal solve_calls
        solve_calls += 1
        raise AssertionError("dirty default path invoked a solve")

    monkeypatch.setattr(v2_runner, "generate_task", forbidden_task_factory)
    monkeypatch.setattr(v2_runner, "solve_restart", forbidden_solve)
    monkeypatch.setattr(
        v2_runner,
        "_git_status",
        lambda: [" M pointmass_examples/pointmass2d_stage0_v2_runner.py"],
    )
    output = tmp_path / "dirty-default.json"
    with pytest.raises(RuntimeError, match="tracked-clean provenance commit"):
        run_stage0_v2(output)
    assert task_calls == 0
    assert solve_calls == 0
    assert not output.exists()
    assert not output.with_suffix(".npz").exists()
    assert not output.with_suffix(".sha256.json").exists()
    assert not list(tmp_path.glob("dirty-default.partial.*"))


def test_v2_mock_interruption_after_two_keeps_incomplete_and_never_publishes_final(
    tmp_path,
):
    calls = 0

    def solve_function(problem, mode_side, restart_index):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected after two complete pairs")
        return _fake_solve(problem, mode_side, restart_index)

    output = tmp_path / "stage0-v2-incomplete.json"
    with pytest.raises(RuntimeError, match="injected after two complete pairs"):
        run_stage0_v2(
            output,
            dependency_overrides={
                "task_factory": _fake_task,
                "problem_factory": lambda task: SimpleNamespace(task=task),
                "solve_function": solve_function,
            },
        )
    assert calls == 3
    assert not output.exists()
    assert not output.with_suffix(".npz").exists()
    assert not output.with_suffix(".sha256.json").exists()
    latest = json.loads(partial_latest_path(output).read_text())
    assert latest["incomplete"] is True
    assert latest["all_stage0_gates_pass"] is False
    assert latest["all_stage0_gates_status"] == "not_applicable_while_incomplete"
    assert latest["latest_complete_generation"] == 2
    generation2_json, generation2_npz = partial_generation_paths(output, 2)
    assert latest["generation_json_sha256"] == sha256_file(generation2_json)
    assert latest["generation_npz_sha256"] == sha256_file(generation2_npz)
    generation2 = json.loads(generation2_json.read_text())
    assert generation2["completed_restart_count"] == 2
    assert len(generation2["pending_ordered_identities"]) == 78
