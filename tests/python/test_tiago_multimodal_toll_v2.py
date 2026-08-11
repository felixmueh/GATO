import ast
from dataclasses import replace
import json
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
from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS


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
    assert metadata["expected_model_load_calls"] == 1
    assert metadata["expected_task_construction_calls"] == 12
    assert metadata["expected_history_regeneration_calls"] == 12
    assert metadata["worker_calls"] == 0
    assert metadata["cuda_calls"] == 0
    assert metadata["sqp_calls"] == 0
    assert metadata["route_oracle_calls"] == 0
    assert metadata["construction_only"] is True
    assert metadata["benchmark_evidence"] is False
    assert set(metadata["required_source_paths"]) == set(v2.REQUIRED_SOURCE_PATHS)
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert oracle.TASK_CONSTRUCTION_AUTHORIZATION is None
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_v2_runner(v2.V2_OUTPUT_PATH)


def _synthetic_ledger():
    return tuple(("synthetic", 99000 + index) for index in range(12))


def _synthetic_certifiable_task(task_seed=99000):
    draws = runner._expected_task_draws(task_seed)
    q0 = np.asarray(
        TIAGO_RIGHT_START_CONFIGS["comfortable_high_clearance"],
        dtype=np.float64,
    ) + draws[:7]
    start = q0[:3].copy()
    target = start + toll.REQUESTED_TRAVEL_M * np.asarray(
        [np.cos(draws[7]), np.sin(draws[7]), 0.0], dtype=np.float64
    )
    callback = _linear_callback()
    history = oracle.exact_eight_dls(
        q0,
        target,
        callback,
        final_position=lambda q: np.asarray(q[:3], dtype=np.float64),
    )
    regenerated = oracle.exact_eight_dls(
        q0,
        target,
        callback,
        final_position=lambda q: np.asarray(q[:3], dtype=np.float64),
    )
    goal = history.tool_position_float64[-1]
    planar = float(np.linalg.norm(goal[:2] - start[:2]))
    direction = (goal[:2] - start[:2]) / planar
    normal = np.asarray([-direction[1], direction[0]])
    sign = int(draws[8])
    cylinder = 0.5 * (start[:2] + goal[:2]) + sign * toll.CYLINDER_OFFSET_M * normal
    default_side = -sign
    toll_xy = cylinder + default_side * toll.TOLL_OFFSET_FROM_CYLINDER_M * normal
    reference = toll.TollReference(
        goal_xyz=tuple(goal),
        cylinder_xy=tuple(cylinder),
        physical_radius_m=toll.PHYSICAL_RADIUS_M,
        toll_xy=tuple(toll_xy),
        toll_sigma_m=toll.TOLL_SIGMA_M,
        clearance_margin_m=toll.CLEARANCE_MARGIN_M,
    )
    ref32 = reference.as_float32()
    x0 = np.concatenate([q0, np.zeros(7)]).astype(np.float32)
    endpoint = min(
        toll.physical_clearance(start[:2], ref32),
        toll.physical_clearance(goal[:2], ref32),
    )
    chord_delta = cylinder - start[:2]
    chord_distance = float(
        abs(direction[0] * chord_delta[1] - direction[1] * chord_delta[0])
    )
    task = toll.TollTask(
        task_seed=task_seed,
        x0=tuple(x0),
        reference=toll.TollReference.from_solver_bytes(ref32),
        default_side=default_side,
        solver_x0_sha256=toll._sha256_array(x0),
        solver_reference_sha256=toll._sha256_array(ref32),
    )
    witness = oracle.V2ConstructionWitness(
        q0_jitter=tuple(draws[:7]),
        phi=float(draws[7]),
        offset_sign=sign,
        requested_target_xyz=tuple(target),
        actual_travel_m=float(np.linalg.norm(goal - start)),
        planar_travel_m=planar,
        endpoint_physical_clearance_m=endpoint,
        chord_min_distance_to_cylinder_m=chord_distance,
        chord_intersects_physical_cylinder=chord_distance < toll.PHYSICAL_RADIUS_M,
        chord_intersects_optimizer_keepout=chord_distance
        < toll.PHYSICAL_RADIUS_M + toll.CLEARANCE_MARGIN_M,
        history=history,
        history_hashes=history.hashes(),
    )

    class Model:
        lowerPositionLimit = np.full(7, -5.0, dtype=np.float64)
        upperPositionLimit = np.full(7, 5.0, dtype=np.float64)

    return task, witness, regenerated, Model()


def test_task_certificate_independently_gates_exact_seed_draws_and_public_bytes():
    identity = ("synthetic", 99000)
    task, witness, regenerated, model = _synthetic_certifiable_task(identity[1])
    row, arrays = runner._task_row(identity, task, witness, regenerated, model)
    assert row["passes"] is True
    assert row["raw_draws_exact_for_task_seed"] is True
    assert row["public_task_bytes_gate"] is True
    assert row["history_regeneration"]["all_witness_regeneration_gates_pass"]
    assert arrays
    assert row["public_task"]["default_side"] == -int(
        arrays["task_99000_raw_draws_float64"][8]
    )
    assert runner._row_arrays_bound(row, arrays)

    jitter = list(witness.q0_jitter)
    jitter[0] = np.nextafter(jitter[0], np.inf)
    altered = replace(witness, q0_jitter=tuple(jitter))
    changed, _ = runner._task_row(identity, task, altered, regenerated, model)
    assert changed["raw_draws_exact_for_task_seed"] is False
    assert changed["passes"] is False
    changed_public = json.loads(json.dumps(row))
    changed_public["public_task"]["default_side"] *= -1
    assert not runner._row_arrays_bound(changed_public, arrays)
    altered_task = replace(task, default_side=-task.default_side)
    altered_row, altered_arrays = runner._task_row(
        identity, altered_task, witness, regenerated, model
    )
    assert altered_row["public_task"]["default_side"] == -task.default_side
    assert altered_row["public_task_bytes_gate"] is False
    assert altered_row["passes"] is False
    assert not runner._row_arrays_bound(altered_row, altered_arrays)


def _synthetic_model_factory(output, call_order):
    def factory():
        call_order.append("model")
        pointer = json.loads(
            output.with_name(f"{output.stem}.partial.latest.json").read_text()
        )
        assert pointer["generation"] == 0
        return object(), {
            "synthetic_model_lower": np.full(7, -1.0, dtype=np.float64),
            "synthetic_model_upper": np.full(7, 1.0, dtype=np.float64),
        }

    return factory


def test_transaction_gen0_precedes_model_and_exact_twelve_attempts(tmp_path):
    output = tmp_path / "v2.json"
    ledger = _synthetic_ledger()
    call_order = []

    def task_factory(identity, model):
        assert model is not None
        call_order.append(identity)
        return {
            "identity": list(identity),
            "attempted": True,
            "passes": True,
            "retry_count": 0,
            "replacement_count": 0,
        }, {f"synthetic_{identity[1]}": np.asarray([identity[1]], dtype=np.int64)}

    result = runner._run_pipeline(
        output,
        ledger=ledger,
        provenance={"test_override": True},
        model_factory=_synthetic_model_factory(output, call_order),
        task_factory=task_factory,
        test_override=True,
    )
    assert call_order == ["model", *ledger]
    assert result["incomplete"] is False
    assert result["all_v2_gates_pass"] is False
    assert result["task_construction_attempt_count"] == 12
    assert result["worker_calls"] == result["cuda_calls"] == 0
    assert result["sqp_calls"] == result["route_oracle_calls"] == 0
    assert result["benchmark_evidence"] is False
    assert result["optimization_evidence"] is False
    assert all(
        (tmp_path / f"v2.partial.{generation:04d}.json").exists()
        for generation in range(14)
    )
    assert output.exists() and output.with_suffix(".npz").exists()
    assert output.with_suffix(".manifest.json").exists()


def test_failed_task_is_retained_all_twelve_are_attempted_and_no_final_is_published(tmp_path):
    output = tmp_path / "v2.json"
    ledger = _synthetic_ledger()
    call_order = []

    def task_factory(identity, model):
        del model
        call_order.append(identity)
        if identity == ledger[1]:
            raise ValueError("synthetic construction failure")
        return {
            "identity": list(identity),
            "attempted": True,
            "passes": True,
            "retry_count": 0,
            "replacement_count": 0,
        }, {}

    with pytest.raises(RuntimeError, match="V2 closes"):
        runner._run_pipeline(
            output,
            ledger=ledger,
            provenance={"test_override": False},
            model_factory=_synthetic_model_factory(output, []),
            task_factory=task_factory,
            test_override=False,
        )
    assert call_order == list(ledger)
    pointer = json.loads((tmp_path / "v2.partial.latest.json").read_text())
    assert pointer["generation"] == 13
    latest = json.loads(Path(pointer["json_path"]).read_text())
    assert latest["incomplete"] is True
    assert latest["task_construction_attempt_count"] == 12
    assert latest["pending_identities"] == []
    failed = [row for row in latest["rows"] if row.get("passes") is False]
    assert len(failed) == 1
    assert failed[0]["error_type"] == "ValueError"
    assert failed[0]["identity"] == list(ledger[1])
    assert not output.exists()
    with pytest.raises(RuntimeError, match="resume, overwrite, or rerun"):
        runner._no_existing_artifacts(output)


def test_interruption_after_second_attempt_retains_generation_and_no_final(tmp_path):
    output = tmp_path / "v2.json"
    ledger = _synthetic_ledger()

    def task_factory(identity, model):
        del model
        return {
            "identity": list(identity),
            "attempted": True,
            "passes": True,
            "retry_count": 0,
            "replacement_count": 0,
        }, {}

    def interrupt(index, row):
        del row
        if index == 2:
            raise KeyboardInterrupt("synthetic interruption")

    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        runner._run_pipeline(
            output,
            ledger=ledger,
            provenance={"test_override": True},
            model_factory=_synthetic_model_factory(output, []),
            task_factory=task_factory,
            test_override=True,
            after_attempt=interrupt,
        )
    pointer = json.loads((tmp_path / "v2.partial.latest.json").read_text())
    assert pointer["generation"] == 3
    latest = json.loads(Path(pointer["json_path"]).read_text())
    assert latest["task_construction_attempt_count"] == 2
    assert len(latest["pending_identities"]) == 10
    assert not output.exists()


def test_false_aggregate_certificate_never_publishes_final_artifacts(tmp_path):
    output = tmp_path / "v2.json"
    ledger = _synthetic_ledger()

    def task_factory(identity, model):
        del model
        return {
            "identity": list(identity),
            "attempted": True,
            "passes": True,
            "retry_count": 0,
            "replacement_count": 0,
        }, {f"synthetic_{identity[1]}": np.asarray([1.0], dtype=np.float64)}

    with pytest.raises(RuntimeError, match="aggregate certificate failed"):
        runner._run_pipeline(
            output,
            ledger=ledger,
            provenance={"tracked_tree_clean_at_start": False},
            model_factory=_synthetic_model_factory(output, []),
            task_factory=task_factory,
            test_override=False,
        )
    pointer = json.loads((tmp_path / "v2.partial.latest.json").read_text())
    assert pointer["generation"] == 13
    latest = json.loads(Path(pointer["json_path"]).read_text())
    assert latest["task_construction_attempt_count"] == 12
    assert all(row["passes"] is True for row in latest["rows"])
    assert not output.exists()
    assert not output.with_suffix(".npz").exists()
    assert not output.with_suffix(".manifest.json").exists()


def test_task_only_runner_has_no_worker_cuda_optimizer_or_predecessor_data_path():
    source = Path(runner.__file__).read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("multimodal_toll_v0" in name for name in imported_modules)
    assert not any("multimodal_toll_v1" in name for name in imported_modules)
    for forbidden in (
        "worker_launcher",
        "sim_forward",
        ".solve(",
        "np.load(",
        "import bsqp",
        "route_template",
        "planned_initializer",
    ):
        assert forbidden not in source
    assert runner.REPORT_ONLY_EXTENSION_HASHES
    assert runner.EXPECTED_WORKER_CALLS == runner.EXPECTED_CUDA_CALLS == 0
    assert runner.EXPECTED_SQP_CALLS == runner.EXPECTED_ROUTE_ORACLE_CALLS == 0
