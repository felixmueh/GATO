import ast
from pathlib import Path

import numpy as np
import pytest

from gato_tiago import multimodal_toll as toll
from gato_tiago import multimodal_toll_oracle_schema as v01_oracle
from gato_tiago import multimodal_toll_v0_runner as v0_runner
from gato_tiago import multimodal_toll_v0_worker as v0_worker
from gato_tiago import multimodal_toll_v1_runner as v1_runner
from gato_tiago import multimodal_toll_v1_worker as v1_worker
from gato_tiago import multimodal_toll_v2 as v2
from gato_tiago import multimodal_toll_v2_oracle_schema as oracle
from gato_tiago import multimodal_toll_v2_runner as runner


def _linear_callback(counter=None):
    jacobian = np.column_stack((np.eye(3), np.zeros((3, 4)))).astype(np.float64)

    def evaluate(q):
        if counter is not None:
            counter.append(np.asarray(q).copy())
        return np.asarray(q[:3], dtype=np.float64), jacobian.copy()

    return evaluate


def test_v2_exact_identity_and_public_protocol_are_frozen_without_rng():
    assert v2.DEVELOPMENT_TASK_SEEDS == (12200, 12201, 12202, 12203)
    assert v2.HELDOUT_TASK_SEEDS == tuple(range(12300, 12308))
    assert len(v2.EXPECTED_TASK_IDENTITIES) == 12
    assert v2.DLS_ITERATIONS == 8
    assert v2.DLS_DAMPING == toll.DAMPING == 0.05
    assert v2.DLS_UNIT_STEP == 1.0
    metadata = v2.frozen_v2_metadata()
    assert metadata["construction"]["early_exit"] is False
    assert metadata["construction"]["step_cap"] is False
    assert metadata["construction"]["clip"] is False
    assert metadata["construction"]["projection"] is False
    assert metadata["construction"]["line_search"] is False
    assert metadata["construction"]["target"] == "p0 + 0.15*[cos(phi),sin(phi),0]"
    assert metadata["v2_task_seed_override"] == {
        "development": [12200, 12201, 12202, 12203],
        "heldout": list(range(12300, 12308)),
    }
    assert metadata["unchanged_non_seed_protocol"] == {
        name: toll.frozen_protocol_metadata()[name]
        for name in v2.NON_SEED_PUBLIC_PROTOCOL_FIELDS
    }
    assert metadata["unchanged_non_seed_protocol_matches_base"] is True
    assert len(metadata["unchanged_non_seed_protocol_sha256"]) == 64
    assert "protocol" not in metadata["unchanged_non_seed_protocol"]


def test_exact_dls_runs_eight_unit_steps_and_retains_every_array():
    q0 = np.zeros(7, dtype=np.float64)
    target = np.asarray([0.15, -0.04, 0.02], dtype=np.float64)
    calls = []
    callback = _linear_callback(calls)
    final_calls = []
    history = oracle.exact_eight_dls(
        q0,
        target,
        callback,
        final_position=lambda q: final_calls.append(np.asarray(q).copy())
        or np.asarray(q[:3], dtype=np.float64),
    )
    assert len(calls) == 8
    assert len(final_calls) == 1
    assert history.q_float64.shape == (9, 7)
    assert history.tool_position_float64.shape == (9, 3)
    assert history.residual_float64.shape == (9, 3)
    assert history.jacobian_float64.shape == (8, 3, 7)
    assert history.dq_float64.shape == (8, 7)
    for index in range(8):
        jacobian = history.jacobian_float64[index]
        expected = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.05**2 * np.eye(3),
            history.residual_float64[index],
        )
        np.testing.assert_array_equal(history.dq_float64[index], expected)
        np.testing.assert_array_equal(
            history.q_float64[index + 1], history.q_float64[index] + expected
        )
    assert set(history.hashes()) == set(history.arrays())
    np.testing.assert_array_equal(
        history.residual_float64[-1], target - history.tool_position_float64[-1]
    )


def test_exact_dls_has_no_hidden_step_cap_or_early_exit():
    calls = []
    callback = _linear_callback(calls)
    history = oracle.exact_eight_dls(
        np.zeros(7, dtype=np.float64),
        np.asarray([10.0, 0.0, 0.0], dtype=np.float64),
        callback,
        final_position=lambda q: np.asarray(q[:3], dtype=np.float64),
    )
    assert len(calls) == 8
    assert history.dq_float64[0, 0] > 9.0
    assert history.dq_float64.shape[0] == 8


def test_dls_unit_step_constant_is_applied_in_every_recurrence(monkeypatch):
    monkeypatch.setattr(oracle, "DLS_UNIT_STEP", 0.5)
    history = oracle.exact_eight_dls(
        np.zeros(7, dtype=np.float64),
        np.asarray([0.15, 0.0, 0.0], dtype=np.float64),
        _linear_callback(),
        final_position=lambda q: np.asarray(q[:3], dtype=np.float64),
    )
    for index in range(8):
        np.testing.assert_array_equal(
            history.q_float64[index + 1],
            history.q_float64[index] + 0.5 * history.dq_float64[index],
        )
    source = Path(oracle.__file__).read_text()
    assert "q = q + DLS_UNIT_STEP * dq" in source


def test_history_regeneration_is_exact_and_mutations_fail():
    q0 = np.zeros(7, dtype=np.float64)
    target = np.asarray([0.15, 0.0, 0.0], dtype=np.float64)
    final = lambda q: np.asarray(q[:3], dtype=np.float64)
    first = oracle.exact_eight_dls(
        q0, target, _linear_callback(), final_position=final
    )
    second = oracle.exact_eight_dls(
        q0, target, _linear_callback(), final_position=final
    )
    assert oracle.certify_history_regeneration(first, second)[
        "all_history_regeneration_gates_pass"
    ]
    witness = oracle.V2ConstructionWitness(
        q0_jitter=tuple(np.zeros(7)),
        phi=0.0,
        offset_sign=1,
        requested_target_xyz=tuple(target),
        actual_travel_m=0.15,
        planar_travel_m=0.15,
        endpoint_physical_clearance_m=0.04,
        chord_min_distance_to_cylinder_m=0.008,
        chord_intersects_physical_cylinder=True,
        chord_intersects_optimizer_keepout=True,
        history=first,
        history_hashes=first.hashes(),
    )
    assert oracle.certify_witness_regeneration(witness, second)[
        "all_witness_regeneration_gates_pass"
    ]
    changed = oracle.DLSHistory(
        **{name: value.copy() for name, value in second.__dict__.items()}
    )
    changed.dq_float64[0, 0] = np.nextafter(changed.dq_float64[0, 0], np.inf)
    assert not oracle.certify_history_regeneration(first, changed)[
        "all_history_regeneration_gates_pass"
    ]
    assert not oracle.certify_witness_regeneration(witness, changed)[
        "all_witness_regeneration_gates_pass"
    ]
    changed_final_residual = oracle.DLSHistory(
        **{name: value.copy() for name, value in second.__dict__.items()}
    )
    changed_final_residual.residual_float64[-1, 0] = np.nextafter(
        changed_final_residual.residual_float64[-1, 0], np.inf
    )
    assert not oracle.certify_witness_regeneration(
        witness, changed_final_residual
    )["all_witness_regeneration_gates_pass"]


def test_q_history_gate_requires_all_nine_finite_margin_safe_iterates():
    history = oracle.exact_eight_dls(
        np.zeros(7, dtype=np.float64),
        np.asarray([0.01, 0.0, 0.0], dtype=np.float64),
        _linear_callback(),
        final_position=lambda q: np.asarray(q[:3], dtype=np.float64),
    )
    lower = np.full(7, -1.0, dtype=np.float64)
    upper = np.full(7, 1.0, dtype=np.float64)
    assert oracle.q_history_gate(history, lower, upper)["all_q_history_gates_pass"]
    history.q_float64[4, 6] = np.nan
    assert not oracle.q_history_gate(history, lower, upper)["all_q_history_gates_pass"]


def test_all_runtime_capabilities_are_closed_before_any_seed_or_model_action(monkeypatch):
    assert v0_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v0_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert v1_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v1_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert v01_oracle.TASK_CONSTRUCTION_AUTHORIZATION is None
    assert v2.V2_EXECUTION_AUTHORIZATION is None
    assert oracle.TASK_CONSTRUCTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    frozen = {seed for _, seed in v2.EXPECTED_TASK_IDENTITIES}
    opened = []
    original = np.random.default_rng

    def guarded(seed=None):
        if seed in frozen:
            opened.append(seed)
        return original(seed)

    monkeypatch.setattr(np.random, "default_rng", guarded)
    for _, seed in v2.EXPECTED_TASK_IDENTITIES:
        with pytest.raises(RuntimeError, match="blocked"):
            oracle.generate_task(seed, object())
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_v2_runner(v2.V2_OUTPUT_PATH)
    assert opened == []


def test_v2_sources_are_isolated_and_benchmark_cannot_import_oracle():
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "tiago_src/gato_tiago/multimodal_toll_v2.py",
        root / "tiago_src/gato_tiago/multimodal_toll_v2_oracle_schema.py",
        root / "tiago_src/gato_tiago/multimodal_toll_v2_runner.py",
    ]
    trees = [ast.parse(path.read_text(), filename=str(path)) for path in paths]
    imported = {
        alias.name
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not any("multimodal_toll_v0" in name for name in imported)
    assert not any("multimodal_toll_v1" in name for name in imported)
    benchmark = (root / "tiago_examples/tiago_multimodal_toll_benchmark.py").read_text()
    initializer = (root / "tiago_src/gato_tiago/multimodal_toll.py").read_text()
    assert "multimodal_toll_v2_oracle_schema" not in benchmark
    assert "multimodal_toll_v2_oracle_schema" not in initializer
    assert "history" not in toll.TollTask.__dataclass_fields__
    assert "q_goal" not in toll.TollTask.__dataclass_fields__
    assert hasattr(oracle, "regenerate_witness_history")


def test_runner_declaration_is_static_and_fail_closed():
    metadata = runner.describe_v2_runner()
    assert metadata["authorization"] is None
    assert metadata["expected_identities"] == [
        list(row) for row in v2.EXPECTED_TASK_IDENTITIES
    ]
    assert metadata["task_instantiation_calls"] == 0
    assert metadata["model_calls"] == 0
    assert metadata["rng_calls"] == 0
    assert metadata["cuda_calls"] == 0
    assert metadata["sqp_calls"] == 0
    assert metadata["execution_implemented"] is False
    assert set(metadata["required_source_paths"]) == set(v2.REQUIRED_SOURCE_PATHS)
    source = Path(runner.__file__).read_text()
    assert "default_rng" not in source
    assert "load_model" not in source
    assert "generate_task(" not in source
