import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shlex

import numpy as np
import pytest

from gato_tiago import multimodal_toll_oracle_v1_runner as v1_runner
from gato_tiago import multimodal_toll_oracle_v1_worker as v1_worker
from gato_tiago import multimodal_toll_oracle_v2 as schema
from gato_tiago import multimodal_toll_oracle_v2_prerequisite_runner as runner
from gato_tiago.multimodal_toll_v4_model_preflight_v4 import array_hash as model_hash
from gato_tiago.multimodal_toll_v4_runner import _array_hash as task_hash


def _old_oracle_hash(value):
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(
        array.dtype.str.encode() + repr(array.shape).encode() + array.tobytes()
    ).hexdigest()


def _producer_summary(arrays, key):
    return {
        "incomplete": False,
        key: True,
        "certificate": {key: True},
        "array_names": sorted(arrays),
        "array_hashes": {
            name: schema.producer_array_hash(value)
            for name, value in arrays.items()
        },
        "rows": [],
    }


def _large_arrays(count, prefix):
    return {
        f"{prefix}_{index:04d}": np.asarray(index, dtype=np.int64)
        for index in range(count)
    }


def _independent_predecessors():
    identities = schema.EXPECTED_TASK_IDENTITIES
    task_arrays = {
        "model_lower_float64": -np.ones(7, dtype=np.float64),
        "model_upper_float64": np.ones(7, dtype=np.float64),
    }
    rows = []
    for index, identity in enumerate(identities):
        seed = identity[1]
        task_arrays[f"task_{seed}_x0_float32"] = np.full(
            14, index / 100.0, dtype=np.float32
        )
        task_arrays[f"task_{seed}_reference_float32"] = np.full(
            10, index / 50.0, dtype=np.float32
        )
        task_arrays[f"task_{seed}_history_q_float64"] = np.full(
            (9, 7), index / 25.0, dtype=np.float64
        )
        rows.append(
            {
                "identity": list(identity),
                "public_task": {"default_side": (-1, 1)[index % 2]},
            }
        )
    while len(task_arrays) < schema.TASK_ARRAY_COUNT:
        index = len(task_arrays)
        task_arrays[f"task_padding_{index:04d}"] = np.asarray(
            index, dtype=np.int64
        )
    task_summary = _producer_summary(task_arrays, "all_v4_gates_pass")
    task_summary["rows"] = rows
    model_arrays = {
        "public_solver_x0_float32": np.stack(
            [task_arrays[f"task_{seed}_x0_float32"] for _, seed in identities]
        ),
        "public_reference_float32": np.stack(
            [
                task_arrays[f"task_{seed}_reference_float32"]
                for _, seed in identities
            ]
        ),
        "public_default_side_int8": np.asarray(
            [row["public_task"]["default_side"] for row in rows], np.int8
        ),
        "quarantined_q8_float64": np.stack(
            [
                task_arrays[f"task_{seed}_history_q_float64"][8]
                for _, seed in identities
            ]
        ),
        "model_lower_float64": task_arrays["model_lower_float64"].copy(),
        "model_upper_float64": task_arrays["model_upper_float64"].copy(),
        "model_velocity_float64": np.full(7, 2.0, dtype=np.float64),
        "model_effort_float64": np.full(7, 3.0, dtype=np.float64),
        "quarantined_q8_oracle_only_bool": np.ones(12, dtype=np.bool_),
        "quarantined_q8_initializer_eligible_bool": np.zeros(12, dtype=np.bool_),
        "quarantined_q8_solver_seed_eligible_bool": np.zeros(12, dtype=np.bool_),
        "v4_artifact_authentication_gate_bool": np.asarray(True, np.bool_),
    }
    while len(model_arrays) < schema.MODEL_ARRAY_COUNT:
        index = len(model_arrays)
        model_arrays[f"model_padding_{index:04d}"] = np.asarray(
            index, dtype=np.int64
        )
    model_summary = _producer_summary(
        model_arrays, "all_model_preflight_gates_pass"
    )
    return task_summary, task_arrays, model_summary, model_arrays


def _publish_independent_synthetic(output):
    task_summary, task_arrays, model_summary, model_arrays = (
        _independent_predecessors()
    )

    def loader(pins):
        if pins is schema.TASK_PINS:
            return task_summary, task_arrays, {}, {}
        assert pins is schema.MODEL_PINS
        return model_summary, model_arrays, {}, {}

    result = runner._run_pipeline(
        output,
        loader=loader,
        task_recert=lambda *_: True,
        model_recert=lambda *_: True,
        provenance=_synthetic_provenance(),
        finish=_finish,
    )
    return result, loader


def _refresh_final_document_hashes(output):
    manifest_path = output.with_suffix(".manifest.json")
    pointer_path = output.with_name(f"{output.stem}.partial.latest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["json_sha256"] = runner.sha256_file(output)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    pointer = json.loads(pointer_path.read_text())
    pointer["json_sha256"] = runner.sha256_file(output)
    pointer["manifest_sha256"] = runner.sha256_file(manifest_path)
    pointer_path.write_text(json.dumps(pointer, sort_keys=True, indent=2) + "\n")


def _synthetic_provenance():
    source_rows = {
        label: {"path": path, "sha256": "a" * 64}
        for label, path in runner.REQUIRED_SOURCE_PATHS.items()
    }
    return {
        "git_head_at_start": "b" * 40,
        "git_head_at_end": None,
        "tracked_tree_clean_at_start": True,
        "tracked_tree_clean_at_end": None,
        "full_git_status_at_start": [],
        "full_git_status_at_end": None,
        "source_hashes_at_start": source_rows,
        "source_hashes_at_end": None,
        "cwd": str(schema.AUTHORIZED_CWD),
        "orig_argv": list(schema.AUTHORIZED_ORIG_ARGV),
        "exact_command": shlex.join(schema.AUTHORIZED_ORIG_ARGV),
        "python_version": "test",
        "numpy_version": np.__version__,
        "task_artifact_loads": 0,
        "task_pure_recert_calls": 0,
        "task_array_hashes_checked": 0,
        "model_artifact_loads": 0,
        "model_pure_recert_calls": 0,
        "model_array_hashes_checked": 0,
        "cross_bind_calls": 0,
        "rejected_v1_artifact_loads": 0,
        **{name: 0 for name in schema.FORBIDDEN_CALL_COUNTS},
    }


def _finish(provenance):
    provenance["git_head_at_end"] = provenance["git_head_at_start"]
    provenance["tracked_tree_clean_at_end"] = True
    provenance["full_git_status_at_end"] = []
    provenance["source_hashes_at_end"] = copy.deepcopy(
        provenance["source_hashes_at_start"]
    )


def test_tokens_paths_pins_and_rejected_v1_report_are_frozen():
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is not None
    assert v1_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v1_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert schema.AUTHORIZED_OUTPUT == Path(
        "/tmp/tiago-tool-center-toll-oracle-v2-prerequisite-authorized-once/prerequisite.json"
    )
    assert len(schema.TASK_PINS) == 4 and len(schema.MODEL_PINS) == 5
    assert schema.TASK_ARRAY_COUNT == 603 and schema.MODEL_ARRAY_COUNT == 150
    report = schema.V1_REJECTED_REPORT_ONLY
    assert report["exit_code"] == 1
    assert report["wall_seconds"] == pytest.approx(0.965679851, abs=0)
    assert report["completed_pairs"] == 0
    assert report["pending_pairs"] == 120
    assert report["artifact_loads_in_v2"] == 0
    assert all(value == 0 for value in report["retained_gen0_counts"].values())
    reconstructed = report["independently_reconstructed_execution_counts"]
    assert reconstructed["task_artifact_loads"] == 1
    assert reconstructed["task_pure_recert_calls"] == 1
    assert all(
        value == 0
        for name, value in reconstructed.items()
        if name not in {"task_artifact_loads", "task_pure_recert_calls"}
    )


def test_v1_reconstructed_counts_follow_the_failed_source_path():
    pipeline = inspect.getsource(v1_runner._run_pipeline)
    task_loader = inspect.getsource(
        v1_runner.ProductionOracleDependencies.task_loader
    )
    pure_loader = inspect.getsource(v1_runner.load_authenticated_task_artifact)
    assert pipeline.index("task_loader()") < pipeline.index(
        'authenticate_summary(task_summary,task_arrays,kind="task")'
    )
    assert "load_authenticated_task_artifact()" in task_loader
    assert "certify_final(" in pure_loader


@pytest.mark.parametrize(
    "value",
    [
        np.asarray(True, dtype=np.bool_),
        np.asarray([1, -2, 3], dtype=np.int16),
        np.arange(12, dtype=np.float32).reshape(3, 4),
        np.arange(6, dtype=np.float64).reshape(2, 3)[:, ::-1],
    ],
)
def test_producer_native_hash_exactly_matches_both_producers(value):
    expected = schema.producer_array_hash(value)
    assert expected == task_hash(np.ascontiguousarray(value))
    assert expected == model_hash(np.ascontiguousarray(value))
    assert expected != _old_oracle_hash(value)


def test_producer_authentication_is_exact_and_fail_closed():
    arrays = {
        "a": np.arange(4, dtype=np.float32),
        "b": np.asarray(3, dtype=np.int64),
        "c": np.eye(2, dtype=np.float64),
    }
    summary = _producer_summary(arrays, "pass")
    assert runner.authenticate_producer(
        summary, arrays, expected_count=3, certificate_key="pass"
    )["passes"]
    mutations = []
    changed = {**arrays, "a": arrays["a"].copy()}
    changed["a"].view(np.uint8)[0] ^= 1
    mutations.append((summary, changed))
    for name, value in (
        ("dtype", arrays["a"].astype(np.float64)),
        ("shape", arrays["a"].reshape(2, 2)),
    ):
        changed = dict(arrays)
        changed["a"] = value
        mutations.append((summary, changed))
    duplicate = copy.deepcopy(summary)
    duplicate["array_names"].append("a")
    mutations.append((duplicate, arrays))
    missing = copy.deepcopy(summary)
    del missing["array_hashes"]["a"]
    mutations.append((missing, arrays))
    extra = copy.deepcopy(summary)
    extra["array_hashes"]["extra"] = "0" * 64
    mutations.append((extra, arrays))
    legacy = copy.deepcopy(summary)
    legacy["array_hashes"] = {
        name: _old_oracle_hash(value) for name, value in arrays.items()
    }
    mutations.append((legacy, arrays))
    for mutated_summary, mutated_arrays in mutations:
        assert not runner.authenticate_producer(
            mutated_summary,
            mutated_arrays,
            expected_count=3,
            certificate_key="pass",
        )["passes"]


def _cross_bind_fixture():
    identities = (("synthetic", 1), ("synthetic", 2))
    rows = []
    task = {
        "model_lower_float64": -np.ones(7),
        "model_upper_float64": np.ones(7),
    }
    for index, (_, seed) in enumerate(identities):
        x0 = np.full(14, index, dtype=np.float32)
        reference = np.full(10, index + 2, dtype=np.float32)
        history = np.full((9, 7), index + 4, dtype=np.float64)
        task[f"task_{seed}_x0_float32"] = x0
        task[f"task_{seed}_reference_float32"] = reference
        task[f"task_{seed}_history_q_float64"] = history
        rows.append(
            {
                "identity": list(identities[index]),
                "public_task": {"default_side": (-1, 1)[index]},
            }
        )
    model = {
        "public_solver_x0_float32": np.stack(
            [task[f"task_{seed}_x0_float32"] for _, seed in identities]
        ),
        "public_reference_float32": np.stack(
            [task[f"task_{seed}_reference_float32"] for _, seed in identities]
        ),
        "public_default_side_int8": np.asarray([-1, 1], dtype=np.int8),
        "quarantined_q8_float64": np.stack(
            [task[f"task_{seed}_history_q_float64"][8] for _, seed in identities]
        ),
        "model_lower_float64": task["model_lower_float64"].copy(),
        "model_upper_float64": task["model_upper_float64"].copy(),
        "model_velocity_float64": np.full(7, 2.0, dtype=np.float64),
        "model_effort_float64": np.full(7, 3.0, dtype=np.float64),
        "quarantined_q8_oracle_only_bool": np.ones(2, dtype=np.bool_),
        "quarantined_q8_initializer_eligible_bool": np.zeros(2, dtype=np.bool_),
        "quarantined_q8_solver_seed_eligible_bool": np.zeros(2, dtype=np.bool_),
        "v4_artifact_authentication_gate_bool": np.asarray(True, dtype=np.bool_),
    }
    return identities, rows, task, model


def test_cross_bind_covers_public_q8_limits_and_all_quarantine_flags():
    identities, rows, task, model = _cross_bind_fixture()
    assert runner.cross_bind_prerequisites(
        rows, task, model, expected_identities=identities
    )["passes"]
    mutations = [
        ("public_solver_x0_float32", (0, 0)),
        ("public_reference_float32", (0, 0)),
        ("public_default_side_int8", (0,)),
        ("quarantined_q8_float64", (0, 0)),
        ("model_lower_float64", (0,)),
        ("model_upper_float64", (0,)),
        ("quarantined_q8_oracle_only_bool", (0,)),
        ("quarantined_q8_initializer_eligible_bool", (0,)),
        ("quarantined_q8_solver_seed_eligible_bool", (0,)),
    ]
    for name, index in mutations:
        changed = {key: value.copy() for key, value in model.items()}
        if changed[name].dtype == np.bool_:
            changed[name][index] = ~changed[name][index]
        else:
            changed[name][index] += 1
        assert not runner.cross_bind_prerequisites(
            rows, task, changed, expected_identities=identities
        )["passes"]
    for name in ("model_velocity_float64", "model_effort_float64"):
        for bad in (
            np.zeros(7, dtype=np.float64),
            np.ones(6, dtype=np.float64),
            np.ones(7, dtype=np.float32),
            np.asarray([1, 1, 1, np.nan, 1, 1, 1], dtype=np.float64),
        ):
            changed = {key: value.copy() for key, value in model.items()}
            changed[name] = bad
            assert not runner.cross_bind_prerequisites(
                rows, task, changed, expected_identities=identities
            )["passes"]


def test_synthetic_transaction_has_gen0_to_gen3_and_success_only_finals(tmp_path):
    task_arrays = _large_arrays(schema.TASK_ARRAY_COUNT, "task")
    model_arrays = _large_arrays(schema.MODEL_ARRAY_COUNT, "model")
    task_summary = _producer_summary(task_arrays, "all_v4_gates_pass")
    model_summary = _producer_summary(
        model_arrays, "all_model_preflight_gates_pass"
    )
    artifacts = {
        id(schema.TASK_PINS): (task_summary, task_arrays, {}, {}),
        id(schema.MODEL_PINS): (model_summary, model_arrays, {}, {}),
    }
    load_order = []

    def loader(pins):
        load_order.append(pins)
        return artifacts[id(pins)]

    output = tmp_path / "prerequisite.json"
    result = runner._run_pipeline(
        output,
        loader=loader,
        task_recert=lambda *_: True,
        model_recert=lambda *_: True,
        cross_binder=lambda *_: {"gates": {"synthetic": True}, "passes": True},
        provenance=_synthetic_provenance(),
        finish=_finish,
    )
    assert load_order == [schema.TASK_PINS, schema.MODEL_PINS]
    assert result["prerequisite_auth_only"] is True
    assert result["oracle_evidence"] is result["benchmark_evidence"] is False
    assert result["provenance"]["task_artifact_loads"] == 1
    assert result["provenance"]["model_artifact_loads"] == 1
    assert result["provenance"]["rejected_v1_artifact_loads"] == 0
    assert all(result["provenance"][name] == 0 for name in schema.FORBIDDEN_CALL_COUNTS)
    for generation in range(4):
        assert (tmp_path / f"prerequisite.partial.{generation:04d}.json").is_file()
        assert (tmp_path / f"prerequisite.partial.{generation:04d}.npz").is_file()
    assert output.is_file()
    assert output.with_suffix(".npz").is_file()
    assert output.with_suffix(".manifest.json").is_file()
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["side_artifact_count"] == 8
    assert [(row["generation"], row["kind"]) for row in manifest["side_artifacts"]] == [
        (generation, kind)
        for generation in range(4)
        for kind in ("json", "npz")
    ]
    assert all(
        runner.sha256_file(row["path"]) == row["sha256"]
        for row in manifest["side_artifacts"]
    )
    pointer = json.loads((tmp_path / "prerequisite.partial.latest.json").read_text())
    assert pointer["incomplete"] is False and pointer["generation"] == 3
    assert pointer["superseded_by"] == str(output)
    retained = runner._certify_retained_documents(
        output,
        output.with_suffix(".npz"),
        output.with_suffix(".manifest.json"),
        tmp_path / "prerequisite.partial.latest.json",
        expected_output=output,
        task_auth=result["task_authentication"],
        model_auth=result["model_authentication"],
        cross_bind=result["cross_bind"],
    )
    assert retained["passes"]
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as archive:
        changed = {name: archive[name] for name in archive.files}
    changed["task_array_count_int64"] = np.asarray(602, dtype=np.int64)
    output.with_suffix(".npz").unlink()
    runner._atomic_npz(output.with_suffix(".npz"), changed)
    assert not runner._certify_retained_documents(
        output,
        output.with_suffix(".npz"),
        output.with_suffix(".manifest.json"),
        tmp_path / "prerequisite.partial.latest.json",
        expected_output=output,
        task_auth=result["task_authentication"],
        model_auth=result["model_authentication"],
        cross_bind=result["cross_bind"],
    )["passes"]


def test_failure_is_retained_and_never_publishes_final(tmp_path):
    task_arrays = _large_arrays(schema.TASK_ARRAY_COUNT, "task")
    task_summary = _producer_summary(task_arrays, "all_v4_gates_pass")
    calls = []

    def loader(_pins):
        calls.append("load")
        return task_summary, task_arrays, {}, {}

    output = tmp_path / "prerequisite.json"
    with pytest.raises(RuntimeError, match="task prerequisite"):
        runner._run_pipeline(
            output,
            loader=loader,
            task_recert=lambda *_: False,
            model_recert=lambda *_: pytest.fail("model recert reached"),
            provenance=_synthetic_provenance(),
            finish=lambda *_: pytest.fail("finish reached"),
        )
    assert calls == ["load"]
    assert not output.exists() and not output.with_suffix(".npz").exists()
    pointer = json.loads((tmp_path / "prerequisite.partial.latest.json").read_text())
    assert pointer["generation"] == 1 and pointer["incomplete"] is True


def test_injected_final_certificate_failure_leaves_gen3_and_no_finals(
    tmp_path, monkeypatch
):
    task_summary, task_arrays, model_summary, model_arrays = (
        _independent_predecessors()
    )

    def loader(pins):
        if pins is schema.TASK_PINS:
            return task_summary, task_arrays, {}, {}
        return model_summary, model_arrays, {}, {}

    monkeypatch.setattr(
        runner,
        "_certify_retained_documents",
        lambda *_args, **_kwargs: {"passes": False},
    )
    output = tmp_path / "prerequisite.json"
    with pytest.raises(RuntimeError, match="staged recertification"):
        runner._run_pipeline(
            output,
            loader=loader,
            task_recert=lambda *_: True,
            model_recert=lambda *_: True,
            provenance=_synthetic_provenance(),
            finish=_finish,
        )
    pointer = json.loads((tmp_path / "prerequisite.partial.latest.json").read_text())
    assert pointer["generation"] == 3 and pointer["incomplete"] is True
    assert pointer["stage"] == "honest_end_provenance"
    assert not output.exists()
    assert not output.with_suffix(".npz").exists()
    assert not output.with_suffix(".manifest.json").exists()
    assert not any(tmp_path.glob(".prerequisite.final-candidate.*"))


def test_public_recertifier_reloads_predecessors_and_recomputes_full_detail(
    tmp_path, monkeypatch
):
    output = tmp_path / "prerequisite.json"
    monkeypatch.setattr(runner, "AUTHORIZED_OUTPUT", output)
    _result, loader = _publish_independent_synthetic(output)
    calls = []

    def counted_loader(pins):
        calls.append(pins)
        return loader(pins)

    monkeypatch.setattr(runner, "_load_pinned_artifact", counted_loader)
    monkeypatch.setattr(runner, "_task_pure_recert", lambda *_: True)
    monkeypatch.setattr(runner, "_model_pure_recert", lambda *_: True)
    retained_paths = sorted(tmp_path.glob("prerequisite*"))
    before = {path.name: runner.sha256_file(path) for path in retained_paths}
    certificate = runner.recertify_retained_prerequisite(output)
    assert certificate["passes"]
    assert calls == [schema.TASK_PINS, schema.MODEL_PINS]
    assert before == {
        path.name: runner.sha256_file(path) for path in retained_paths
    }

    summary = json.loads(output.read_text())
    summary["task_authentication"]["passes"] = False
    output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    _refresh_final_document_hashes(output)
    assert not runner.recertify_retained_prerequisite(output)["passes"]


def test_public_recertifier_rejects_refreshed_checkpoint_provenance(
    tmp_path, monkeypatch
):
    output = tmp_path / "prerequisite.json"
    monkeypatch.setattr(runner, "AUTHORIZED_OUTPUT", output)
    _result, loader = _publish_independent_synthetic(output)
    monkeypatch.setattr(runner, "_load_pinned_artifact", loader)
    monkeypatch.setattr(runner, "_task_pure_recert", lambda *_: True)
    monkeypatch.setattr(runner, "_model_pure_recert", lambda *_: True)
    checkpoint = tmp_path / "prerequisite.partial.0002.json"
    payload = json.loads(checkpoint.read_text())
    payload["provenance"]["task_artifact_loads"] = 0
    checkpoint.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["side_artifacts"]:
        if row["path"] == str(checkpoint):
            row["sha256"] = runner.sha256_file(checkpoint)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    pointer_path = output.with_name("prerequisite.partial.latest.json")
    pointer = json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = runner.sha256_file(manifest_path)
    pointer_path.write_text(json.dumps(pointer, sort_keys=True, indent=2) + "\n")
    assert not runner.recertify_retained_prerequisite(output)["passes"]


@pytest.mark.parametrize("target", ["summary_extra", "manifest_count", "pointer_stage"])
def test_public_recertifier_rejects_exact_document_boundary_mutations(
    tmp_path, monkeypatch, target
):
    output = tmp_path / "prerequisite.json"
    monkeypatch.setattr(runner, "AUTHORIZED_OUTPUT", output)
    _result, loader = _publish_independent_synthetic(output)
    monkeypatch.setattr(runner, "_load_pinned_artifact", loader)
    monkeypatch.setattr(runner, "_task_pure_recert", lambda *_: True)
    monkeypatch.setattr(runner, "_model_pure_recert", lambda *_: True)
    manifest_path = output.with_suffix(".manifest.json")
    pointer_path = output.with_name("prerequisite.partial.latest.json")
    if target == "summary_extra":
        summary = json.loads(output.read_text())
        summary["extra"] = True
        output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
        _refresh_final_document_hashes(output)
    elif target == "manifest_count":
        manifest = json.loads(manifest_path.read_text())
        manifest["side_artifact_count"] = 7
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        pointer = json.loads(pointer_path.read_text())
        pointer["manifest_sha256"] = runner.sha256_file(manifest_path)
        pointer_path.write_text(
            json.dumps(pointer, sort_keys=True, indent=2) + "\n"
        )
    else:
        pointer = json.loads(pointer_path.read_text())
        pointer["stage"] = "model_prerequisite_authenticated"
        pointer_path.write_text(
            json.dumps(pointer, sort_keys=True, indent=2) + "\n"
        )
    assert not runner.recertify_retained_prerequisite(output)["passes"]


def test_public_recertifier_rejects_refreshed_predecessor_public_bytes(
    tmp_path, monkeypatch
):
    output = tmp_path / "prerequisite.json"
    monkeypatch.setattr(runner, "AUTHORIZED_OUTPUT", output)
    _result, loader = _publish_independent_synthetic(output)
    task_summary, task_arrays, model_summary, model_arrays = (
        _independent_predecessors()
    )
    task_arrays = {name: value.copy() for name, value in task_arrays.items()}
    name = f"task_{schema.EXPECTED_TASK_IDENTITIES[0][1]}_x0_float32"
    task_arrays[name][0] += np.float32(0.25)
    task_summary = copy.deepcopy(task_summary)
    task_summary["array_hashes"][name] = schema.producer_array_hash(
        task_arrays[name]
    )

    def changed_loader(pins):
        if pins is schema.TASK_PINS:
            return task_summary, task_arrays, {}, {}
        return model_summary, model_arrays, {}, {}

    monkeypatch.setattr(runner, "_load_pinned_artifact", changed_loader)
    monkeypatch.setattr(runner, "_task_pure_recert", lambda *_: True)
    monkeypatch.setattr(runner, "_model_pure_recert", lambda *_: True)
    assert not runner.recertify_retained_prerequisite(output)["passes"]


def test_provenance_exact_counts_command_head_and_sources_fail_closed():
    provenance = _synthetic_provenance()
    provenance.update(
        {
            "task_artifact_loads": 1,
            "task_pure_recert_calls": 1,
            "task_array_hashes_checked": 603,
            "model_artifact_loads": 1,
            "model_pure_recert_calls": 1,
            "model_array_hashes_checked": 150,
            "cross_bind_calls": 1,
        }
    )
    _finish(provenance)
    assert runner.certify_provenance(provenance)
    for key, value in (
        ("git_head_at_end", "c" * 40),
        ("exact_command", "wrong"),
        ("cwd", "/tmp"),
        ("task_array_hashes_checked", 602),
        ("model_pure_recert_calls", 0),
        ("rejected_v1_artifact_loads", 1),
        ("cuda_calls", 1),
    ):
        changed = copy.deepcopy(provenance)
        changed[key] = value
        assert not runner.certify_provenance(changed)
    changed = copy.deepcopy(provenance)
    changed["source_hashes_at_end"]["v2_schema"]["sha256"] = "d" * 64
    assert not runner.certify_provenance(changed)


def test_execution_boundary_blocks_before_root_or_artifact_access(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner, "_load_pinned_artifact", lambda *_: calls.append("artifact")
    )
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_prerequisite(str(tmp_path / "prerequisite.json"), object())
    assert calls == [] and list(tmp_path.iterdir()) == []


def test_authorized_boundary_is_hard_path_fresh_and_exact_token(
    tmp_path, monkeypatch
):
    output = tmp_path / "fresh" / "prerequisite.json"
    calls = []
    monkeypatch.setattr(runner, "AUTHORIZED_OUTPUT", output)
    monkeypatch.setattr(runner, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        runner,
        "start_provenance",
        lambda *_args, **_kwargs: {"tracked_tree_clean_at_start": True},
    )
    monkeypatch.setattr(
        runner,
        "_run_pipeline",
        lambda path, **_kwargs: calls.append(Path(path)) or {"passes": True},
    )
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_prerequisite(str(output), object())
    assert calls == []
    assert runner.execute_prerequisite(
        str(output), runner.RUNNER_EXECUTION_AUTHORIZATION
    ) == {"passes": True}
    assert calls == [output]
    output.parent.mkdir()
    with pytest.raises(RuntimeError, match="root must be absent"):
        runner.execute_prerequisite(
            str(output), runner.RUNNER_EXECUTION_AUTHORIZATION
        )
    assert calls == [output]


def test_public_recertifier_rejects_wrong_path_before_predecessor_load(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        runner, "_load_pinned_artifact", lambda *_: calls.append("load")
    )
    assert not runner.recertify_retained_prerequisite(
        tmp_path / "prerequisite.json"
    )["passes"]
    assert calls == []


def test_static_sources_have_no_forbidden_runtime_or_v1_artifact_access():
    sources = "\n".join(
        Path(module.__file__).read_text() for module in (schema, runner)
    )
    tree = ast.parse(Path(runner.__file__).read_text())
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "generate_task" not in calls
    assert "default_rng" not in calls
    assert "minimize" not in calls
    assert "sim_forward" not in sources
    assert "BSQP_" not in sources
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert not any("multimodal_toll_oracle_v1" in name for name in imports)
    assert "/tmp/tiago-tool-center-toll-oracle-v1-authorized-once" in sources
    assert "artifact_loads_in_v2\": 0" in sources


def test_source_declares_exact_logical_counts_and_no_full_campaign():
    source = inspect.getsource(runner._run_pipeline)
    assert 'provenance["task_artifact_loads"] = 1' in source
    assert 'provenance["model_artifact_loads"] = 1' in source
    assert 'provenance["cross_bind_calls"] = 1' in source
    assert "acquisition" not in source and "polish" not in source
