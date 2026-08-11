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
from gato_tiago import multimodal_toll_v2_oracle_schema as v2_oracle
from gato_tiago import multimodal_toll_v2_runner as v2_runner
from gato_tiago import multimodal_toll_v3 as v3
from gato_tiago import multimodal_toll_v3_oracle_schema as v3_oracle
from gato_tiago import multimodal_toll_v3_runner as v3_runner
from gato_tiago import multimodal_toll_v4 as v4
from gato_tiago import multimodal_toll_v4_oracle_schema as oracle
from gato_tiago import multimodal_toll_v4_runner as runner
from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS


def _linear_callback(counter=None):
    jacobian = np.column_stack((np.eye(3), np.zeros((3, 4)))).astype(np.float64)

    def evaluate(q):
        if counter is not None:
            counter.append(np.asarray(q).copy())
        return np.asarray(q[:3], dtype=np.float64), jacobian.copy()

    return evaluate


def _limits():
    return (
        np.full(7, -1.0, dtype=np.float64),
        np.full(7, 1.0, dtype=np.float64),
    )


def _box_history(target=None, *, independent=False):
    lower, upper = _limits()
    function = (
        oracle.independently_reenumerate_eight_box_dls
        if independent
        else oracle.exact_eight_box_dls
    )
    target = (
        np.asarray([0.15, -0.04, 0.02], dtype=np.float64)
        if target is None
        else np.asarray(target, dtype=np.float64)
    )
    return function(
        np.zeros(7, dtype=np.float64),
        target,
        _linear_callback(),
        lower,
        upper,
        final_position=lambda q: np.asarray(q[:3], dtype=np.float64),
    )


def _copy_history(history):
    return oracle.DLSHistory(
        **{name: value.copy() for name, value in history.__dict__.items()}
    )


def test_v4_exact_identity_and_active_set_protocol_are_frozen_without_rng():
    assert v4.DEVELOPMENT_TASK_SEEDS == (12600, 12601, 12602, 12603)
    assert v4.HELDOUT_TASK_SEEDS == tuple(range(12700, 12708))
    assert len(v4.EXPECTED_TASK_IDENTITIES) == 12
    assert v4.DLS_ITERATIONS == 8
    assert v4.DLS_DAMPING == toll.DAMPING == 0.05
    assert v4.DLS_UNIT_STEP == 1.0
    assert v4.FACE_STATUS_ORDER == (-1, 0, 1)
    assert v4.FACE_COUNT == 2187
    assert v4.FACE_FEASIBILITY_TOLERANCE == 1e-12
    assert v4.FREE_NORMAL_RESIDUAL_TOLERANCE == 1e-10
    assert v4.OBJECTIVE_TIE_ABSOLUTE_TOLERANCE == 1e-12
    assert v4.VERIFY_DQ_MAXABS_TOLERANCE == 1e-10
    assert v4.VERIFY_OBJECTIVE_TOLERANCE == 1e-12
    assert v4.VERIFY_GLOBAL_DOMINANCE_TOLERANCE == 1e-12
    assert v4.SELECTED_KKT_TOLERANCE == 1e-9
    metadata = v4.frozen_v4_metadata()
    construction = metadata["construction"]
    assert construction["exhaust_all_faces_without_pruning"] is True
    assert construction["status_order"] == [-1, 0, 1]
    assert construction["faces_per_step"] == 2187
    assert construction["free_solver"] == "float64_spd_solve_no_inverse"
    assert construction["tie_break"] == "lexicographically_first_lower_free_upper"
    assert construction["independent_verifier_calls_constructor_enumerator"] is False
    assert construction["all_q0_through_q8_reserve_margin_is_acceptance_gate"] is True
    assert construction["all_numerical_ik_iterates_are_public_task_data"] is False
    assert construction["all_numerical_ik_iterates_are_initializer_inputs"] is False
    assert construction["all_numerical_ik_iterates_are_sqp_inputs"] is False
    assert metadata["v4_task_seed_override"] == {
        "development": [12600, 12601, 12602, 12603],
        "heldout": list(range(12700, 12708)),
    }
    assert metadata["unchanged_non_seed_protocol"] == {
        name: toll.frozen_protocol_metadata()[name]
        for name in v4.NON_SEED_PUBLIC_PROTOCOL_FIELDS
    }
    assert metadata["unchanged_non_seed_protocol_matches_base"] is True
    assert metadata["predecessor_artifacts_consumed"] is False
    assert set(metadata["rejected_predecessor_artifact_hashes_report_only"]) >= {
        "v3_latest_pointer",
        "v3_generation13_json",
        "v3_generation13_npz",
    }


def test_constructor_exhausts_all_faces_in_lex_order_and_uses_global_tie_rule():
    jacobian = np.column_stack((np.eye(3), np.zeros((3, 4)))).astype(np.float64)
    residual = np.asarray([10.0, -10.0, 0.2], dtype=np.float64)
    lower = np.full(7, -0.1, dtype=np.float64)
    upper = np.full(7, 0.1, dtype=np.float64)
    table = oracle._enumerate_constructor_faces(jacobian, residual, lower, upper)
    assert table["status"].shape == (2187, 7)
    assert table["dq"].shape == (2187, 7)
    assert table["objective"].shape == (2187,)
    assert table["primal"].shape == (2187,)
    assert table["free_residual"].shape == (2187,)
    assert table["feasible"].shape == (2187,)
    np.testing.assert_array_equal(table["status"][0], np.full(7, -1, dtype=np.int8))
    np.testing.assert_array_equal(table["status"][-1], np.full(7, 1, dtype=np.int8))
    assert table["selected"] == int(
        np.flatnonzero(
            table["feasible"]
            & (
                table["objective"]
                <= np.min(table["objective"][table["feasible"]]) + 1e-12
            )
        )[0]
    )
    selected = table["dq"][table["selected"]]
    assert selected[0] == 0.1
    assert selected[1] == -0.1
    assert np.all(table["primal"][table["feasible"]] <= 1e-12)
    assert np.all(table["free_residual"][table["feasible"]] <= 1e-10)

    tied = oracle._enumerate_constructor_faces(
        np.zeros((3, 7), dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros(7, dtype=np.float64),
        np.zeros(7, dtype=np.float64),
    )
    assert np.all(tied["feasible"])
    assert tied["selected"] == 0


def test_independent_reenumerator_never_calls_constructor_enumerator(monkeypatch):
    jacobian = np.column_stack((np.eye(3), np.zeros((3, 4)))).astype(np.float64)
    residual = np.asarray([0.2, -0.1, 0.05], dtype=np.float64)
    lower = np.full(7, -0.1, dtype=np.float64)
    upper = np.full(7, 0.1, dtype=np.float64)
    expected = oracle._enumerate_constructor_faces(jacobian, residual, lower, upper)

    def forbidden(*args, **kwargs):
        raise AssertionError("constructor enumerator called by verifier")

    monkeypatch.setattr(oracle, "_enumerate_constructor_faces", forbidden)
    actual = oracle._enumerate_verifier_faces(jacobian, residual, lower, upper)
    np.testing.assert_array_equal(actual["status"], expected["status"])
    np.testing.assert_array_equal(actual["feasible"], expected["feasible"])
    assert np.max(np.abs(actual["dq"] - expected["dq"])) <= 1e-10
    assert np.max(np.abs(actual["objective"] - expected["objective"])) <= 1e-12
    source = Path(oracle.__file__).read_text()
    tree = ast.parse(source)
    verifier = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_enumerate_verifier_faces"
    )
    assert "_enumerate_constructor_faces" not in ast.unparse(verifier)


def test_full_history_retains_every_face_and_independent_certificate_passes():
    constructor = _box_history()
    verifier = _box_history(independent=True)
    assert constructor.q_float64.shape == (9, 7)
    assert constructor.face_status_int8.shape == (8, 2187, 7)
    assert constructor.face_dq_float64.shape == (8, 2187, 7)
    assert constructor.face_objective_float64.shape == (8, 2187)
    assert constructor.face_primal_violation_float64.shape == (8, 2187)
    assert constructor.face_free_residual_float64.shape == (8, 2187)
    assert constructor.face_feasible_bool.shape == (8, 2187)
    assert constructor.selected_status_int8.shape == (8, 7)
    assert set(constructor.hashes()) == set(constructor.arrays())
    certificate = oracle.certify_history_regeneration(constructor, verifier)
    assert certificate["independent_verifier_calls_constructor_enumerator"] is False
    assert certificate["selected_dq_agrees"]
    assert certificate["selected_objective_agrees"]
    assert certificate["global_dominance_pass"]
    assert certificate["retained_global_dominance_pass"]
    assert certificate["selected_face_table_binding_exact"]
    assert certificate["independent_selected_identity_exact"]
    assert certificate["selected_kkt_pass"]
    assert certificate["all_history_regeneration_gates_pass"]


def test_certificate_rejects_face_selected_and_kkt_mutations():
    constructor = _box_history()
    verifier = _box_history(independent=True)

    changed_face = _copy_history(constructor)
    changed_face.face_objective_float64[0, 0] -= 1.0
    assert not oracle.certify_history_regeneration(changed_face, verifier)[
        "all_history_regeneration_gates_pass"
    ]

    changed_selected = _copy_history(constructor)
    changed_selected.dq_float64[0, 0] = np.nextafter(
        changed_selected.dq_float64[0, 0], np.inf
    )
    assert not oracle.certify_history_regeneration(changed_selected, verifier)[
        "all_history_regeneration_gates_pass"
    ]

    changed_status = _copy_history(constructor)
    changed_status.face_status_int8[0, 0, 0] = 0
    assert not oracle.certify_history_regeneration(changed_status, verifier)[
        "all_history_regeneration_gates_pass"
    ]

    changed_gradient = _copy_history(constructor)
    changed_gradient.selected_gradient_float64[0, 0] = 1.0
    assert not oracle.certify_history_regeneration(changed_gradient, verifier)[
        "all_history_regeneration_gates_pass"
    ]

    malformed_arrays = {
        name: value.copy() for name, value in constructor.__dict__.items()
    }
    malformed_arrays["face_status_int8"] = malformed_arrays[
        "face_status_int8"
    ][:, :-1]
    malformed = oracle.DLSHistory(**malformed_arrays)
    malformed_certificate = oracle.certify_history_regeneration(
        malformed, verifier
    )
    assert malformed_certificate["exact_array_schema"] is False
    assert not malformed_certificate["all_history_regeneration_gates_pass"]


def test_q_history_gate_requires_exact_recurrence_and_all_nine_reserve():
    history = _box_history(target=[0.5, 0.0, 0.0])
    lower, upper = _limits()
    report = oracle.q_history_gate(history, lower, upper)
    assert report["all_q0_through_q8_finite"]
    assert report["exact_recurrence"]
    assert report["all_q0_through_q8_inside_frozen_reserve"]
    assert report["all_q_history_gates_pass"]

    bad_intermediate = _copy_history(history)
    bad_intermediate.q_float64[4, 6] = np.nextafter(
        bad_intermediate.q_float64[4, 6], np.inf
    )
    assert not oracle.q_history_gate(bad_intermediate, lower, upper)[
        "all_q_history_gates_pass"
    ]

    bad_reserve = _copy_history(history)
    bad_reserve.q_float64[4, 6] = 0.93
    bad_reserve.dq_float64[3, 6] = (
        bad_reserve.q_float64[4, 6] - bad_reserve.q_float64[3, 6]
    )
    assert not oracle.q_history_gate(bad_reserve, lower, upper)[
        "all_q_history_gates_pass"
    ]


def test_all_v4_and_predecessor_capabilities_are_closed_without_opening_rng(monkeypatch):
    assert v0_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v0_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert v1_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v1_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert v01_oracle.TASK_CONSTRUCTION_AUTHORIZATION is None
    assert v2.V2_EXECUTION_AUTHORIZATION is None
    assert v2_oracle.TASK_CONSTRUCTION_AUTHORIZATION is None
    assert v2_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v3.V3_EXECUTION_AUTHORIZATION is None
    assert v3_oracle.TASK_CONSTRUCTION_AUTHORIZATION is None
    assert v3_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v4.V4_EXECUTION_AUTHORIZATION is None
    assert oracle.TASK_CONSTRUCTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    frozen = {seed for _, seed in v4.EXPECTED_TASK_IDENTITIES}
    opened = []
    original = np.random.default_rng

    def guarded(seed=None):
        if seed in frozen:
            opened.append(seed)
        return original(seed)

    monkeypatch.setattr(np.random, "default_rng", guarded)
    for _, seed in v4.EXPECTED_TASK_IDENTITIES:
        with pytest.raises(RuntimeError, match="blocked"):
            oracle.generate_task(seed, object())
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_v4_runner(v4.V4_OUTPUT_PATH)
    assert opened == []


def test_v4_sources_are_isolated_tainted_and_benchmark_cannot_import_oracle():
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "tiago_src/gato_tiago/multimodal_toll_v4.py",
        root / "tiago_src/gato_tiago/multimodal_toll_v4_oracle_schema.py",
        root / "tiago_src/gato_tiago/multimodal_toll_v4_runner.py",
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
    for predecessor in ("v0", "v1", "v2", "v3"):
        assert not any(f"multimodal_toll_{predecessor}" in name for name in imported)
    benchmark = (root / "tiago_examples/tiago_multimodal_toll_benchmark.py").read_text()
    initializer = (root / "tiago_src/gato_tiago/multimodal_toll.py").read_text()
    assert "multimodal_toll_v4_oracle_schema" not in benchmark
    assert "multimodal_toll_v4_oracle_schema" not in initializer
    assert "history" not in toll.TollTask.__dataclass_fields__
    assert "q_goal" not in toll.TollTask.__dataclass_fields__
    assert set(v4.frozen_v4_metadata()["forbidden_public_or_initializer_fields"]) <= set(
        toll.FORBIDDEN_INITIALIZER_FIELDS
    )
    for field in ("box_dls_history", "box_dls_face_table", "box_dls_active_set"):
        with pytest.raises(ValueError, match="tainted"):
            toll.validate_initializer_inputs({field: np.zeros(1)})


def test_runner_declaration_is_static_and_fail_closed():
    metadata = runner.describe_v4_runner()
    assert metadata["authorization_enabled"] is False
    assert metadata["expected_identities"] == [
        list(row) for row in v4.EXPECTED_TASK_IDENTITIES
    ]
    assert metadata["expected_model_load_calls"] == 1
    assert metadata["expected_task_construction_calls"] == 12
    assert metadata["expected_history_regeneration_calls"] == 12
    assert metadata["worker_calls"] == metadata["cuda_calls"] == 0
    assert metadata["sqp_calls"] == metadata["route_oracle_calls"] == 0
    assert metadata["construction_only"] is True
    assert metadata["benchmark_evidence"] is False
    assert set(metadata["required_source_paths"]) == set(v4.REQUIRED_SOURCE_PATHS)
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_v4_runner(v4.V4_OUTPUT_PATH)


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
    lower = np.full(7, -5.0, dtype=np.float64)
    upper = np.full(7, 5.0, dtype=np.float64)
    history = oracle.exact_eight_box_dls(
        q0,
        target,
        callback,
        lower,
        upper,
        final_position=lambda q: np.asarray(q[:3], dtype=np.float64),
    )
    regenerated = oracle.independently_reenumerate_eight_box_dls(
        q0,
        target,
        callback,
        lower,
        upper,
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
    witness = oracle.V4ConstructionWitness(
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
        lowerPositionLimit = lower
        upperPositionLimit = upper

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
    output = tmp_path / "v4.json"
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
    assert result["all_v4_gates_pass"] is False
    assert result["task_construction_attempt_count"] == 12
    assert result["worker_calls"] == result["cuda_calls"] == 0
    assert result["sqp_calls"] == result["route_oracle_calls"] == 0
    assert result["benchmark_evidence"] is False
    assert result["optimization_evidence"] is False
    assert all(
        (tmp_path / f"v4.partial.{generation:04d}.json").exists()
        for generation in range(14)
    )
    assert output.exists() and output.with_suffix(".npz").exists()
    assert output.with_suffix(".manifest.json").exists()


def test_failed_task_is_retained_all_twelve_are_attempted_and_no_final_is_published(tmp_path):
    output = tmp_path / "v4.json"
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

    with pytest.raises(RuntimeError, match="V4 closes"):
        runner._run_pipeline(
            output,
            ledger=ledger,
            provenance={"test_override": False},
            model_factory=_synthetic_model_factory(output, []),
            task_factory=task_factory,
            test_override=False,
        )
    assert call_order == list(ledger)
    pointer = json.loads((tmp_path / "v4.partial.latest.json").read_text())
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
    output = tmp_path / "v4.json"
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
    pointer = json.loads((tmp_path / "v4.partial.latest.json").read_text())
    assert pointer["generation"] == 3
    latest = json.loads(Path(pointer["json_path"]).read_text())
    assert latest["task_construction_attempt_count"] == 2
    assert len(latest["pending_identities"]) == 10
    assert not output.exists()


def test_false_aggregate_certificate_never_publishes_final_artifacts(tmp_path):
    output = tmp_path / "v4.json"
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
    pointer = json.loads((tmp_path / "v4.partial.latest.json").read_text())
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
    assert not any("multimodal_toll_v2" in name for name in imported_modules)
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
