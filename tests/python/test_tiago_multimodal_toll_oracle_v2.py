import inspect
import json
from pathlib import Path
import re

import numpy as np
import pytest

from gato_tiago import multimodal_toll_oracle_v1 as v1
from gato_tiago import multimodal_toll_oracle_v1_runner as v1_runner
from gato_tiago import multimodal_toll_oracle_v1_worker as v1_worker
from gato_tiago import multimodal_toll_oracle_v2 as prerequisite_schema
from gato_tiago import multimodal_toll_oracle_v2_prerequisite_runner as prerequisite_runner
from gato_tiago import multimodal_toll_oracle_v2_full as schema
from gato_tiago import multimodal_toll_oracle_v2_runner as runner
from gato_tiago import multimodal_toll_oracle_v2_worker as worker


def _producer_summary(arrays, kind):
    key = "all_v4_gates_pass" if kind == "task" else "all_model_preflight_gates_pass"
    return {
        "incomplete": False,
        key: True,
        "certificate": {key: True},
        "array_names": sorted(arrays),
        "array_hashes": {
            name: prerequisite_schema.producer_array_hash(value)
            for name, value in arrays.items()
        },
    }


def _canonical_prerequisite_authentication():
    return {
        "pin_results": {
            name: {
                "path": str(path),
                "expected_sha256": digest,
                "actual_sha256": digest,
                "passes": True,
            }
            for name, (path, digest) in schema.PREREQUISITE_ARTIFACT_PINS.items()
        },
        "all_four_pins_pass": True,
        "independent_recertification": {
            "gates": {
                name: True
                for name in runner.PREREQUISITE_RECERTIFICATION_GATE_NAMES
            },
            "passes": True,
        },
        "passes": True,
    }


def test_full_v2_tokens_are_closed_after_runtime_rejection():
    assert schema.ORACLE_EXECUTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert prerequisite_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v1_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v1_worker.WORKER_EXECUTION_AUTHORIZATION is None
    root = Path(__file__).resolve().parents[2]
    pattern = re.compile(r"^[A-Z][A-Z0-9_]*AUTHORIZATION = object\(\)$")
    enabled = []
    for path in sorted((root / "tiago_src/gato_tiago").glob("*.py")):
        enabled.extend(
            (path.name, line)
            for line in path.read_text().splitlines()
            if pattern.fullmatch(line)
        )
    assert enabled == []


def test_v2_paths_pins_and_nonconsumption_records_are_exact():
    assert schema.ORACLE_OUTPUT_PATH == Path(
        "/tmp/tiago-tool-center-toll-oracle-v2-authorized-once/oracle.json"
    )
    assert runner.WORKER_MODULE == "gato_tiago.multimodal_toll_oracle_v2_worker"
    assert {name: digest for name, (_path, digest) in schema.PREREQUISITE_ARTIFACT_PINS.items()} == {
        "final_json": "b1805a61517805e3315c857d4192feef851e54cdd97338177c83e04c28a1b49b",
        "final_npz": "d6705e1d741a192dd5c897ffa879b2c2c58d273b4c5fb88a1990c608015264ab",
        "final_manifest": "1b1ca64e5eef51c4c16d8798b8314b44023ba4498b42a24406e6c703dfd0ddbe",
        "latest_pointer": "3d9ba295d252ab7bcd0f48d9c585cfc99238ab9068b732377b6cacf4f6e68d36",
    }
    assert schema.V1_REJECTED_REPORT_ONLY == prerequisite_schema.V1_REJECTED_REPORT_ONLY
    assert schema.V1_REJECTED_REPORT_ONLY["artifact_loads_in_v2"] == 0


def test_science_ledger_options_costs_and_watchdog_are_identity_parity_with_v1():
    constant_names = (
        "EXPECTED_TASK_IDENTITIES", "EXPECTED_LEDGER", "EXPECTED_PAIR_COUNT",
        "OBJECTIVE_WEIGHTS", "TRUST_CONSTR_OPTIONS", "TEMPLATE_RADII_M",
        "ACQUISITION_MAXITER", "POLISH_MAXITER", "ACQUISITION_WALL_LIMIT_S",
        "POLISH_WALL_LIMIT_S", "CAMPAIGN_WALL_LIMIT_S", "BOOTSTRAP_SEED",
        "BOOTSTRAP_RESAMPLES", "DENSE_SAMPLES", "Z_WIDTH",
    )
    for name in constant_names:
        assert getattr(schema, name) == getattr(v1, name)
    assert runner.OriginalOracleNLP is v1_runner.OriginalOracleNLP
    assert runner.solve_anchored_acquisition is v1_runner.solve_anchored_acquisition
    assert runner.solve_unanchored_polish is v1_runner.solve_unanchored_polish
    assert runner.certify_retained_row is v1_runner.certify_retained_row
    assert runner.aggregate_campaign is v1_runner.aggregate_campaign
    assert runner.bootstrap_heldout_gaps is v1_runner.bootstrap_heldout_gaps


def test_every_inherited_v1_frozen_constant_is_exact_except_namespace_fields():
    excluded = {
        "ORACLE_PROTOCOL_VERSION",
        "ORACLE_EXECUTION_AUTHORIZATION",
        "ORACLE_OUTPUT_PATH",
        "AUTHORIZED_ORIG_ARGV",
        "REQUIRED_SOURCE_PATHS",
    }
    inherited = {
        name
        for name in vars(v1)
        if name.isupper() and name not in excluded
    }
    assert inherited
    for name in inherited:
        left = getattr(schema, name)
        right = getattr(v1, name)
        if isinstance(left, np.ndarray):
            assert np.array_equal(left, right)
        else:
            assert left == right


@pytest.mark.parametrize("kind,count", [("task", 603), ("model", 150)])
def test_v2_predecessor_auth_uses_only_producer_native_hash(kind, count):
    arrays = {
        f"array_{index:04d}": np.asarray(index, dtype=np.int64)
        for index in range(count)
    }
    summary = _producer_summary(arrays, kind)
    assert runner.authenticate_summary(summary, arrays, kind=kind)
    name = next(iter(arrays))
    summary["array_hashes"][name] = v1_runner.array_hash(arrays[name])
    assert not runner.authenticate_summary(summary, arrays, kind=kind)


def test_prerequisite_authenticator_hard_pins_and_calls_public_recert(monkeypatch):
    expected = {str(path): digest for path, digest in schema.PREREQUISITE_ARTIFACT_PINS.values()}
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) in expected)
    monkeypatch.setattr(runner, "sha256_file", lambda path: expected[str(path)])
    monkeypatch.setattr(
        prerequisite_runner,
        "recertify_retained_prerequisite",
        lambda: _canonical_prerequisite_authentication()[
            "independent_recertification"
        ],
    )
    detail = runner.authenticate_prerequisite_artifact()
    assert detail["passes"] and detail["all_four_pins_pass"]
    changed = dict(expected)
    changed[str(schema.PREREQUISITE_ARTIFACT_PINS["final_json"][0])] = "0" * 64
    monkeypatch.setattr(runner, "sha256_file", lambda path: changed[str(path)])
    with pytest.raises(RuntimeError, match="prerequisite authentication"):
        runner.authenticate_prerequisite_artifact()


def test_full_pipeline_authenticates_prerequisite_before_v1_pipeline(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        v1_runner,
        "_run_pipeline",
        lambda *_args, **_kwargs: calls.append("v1_pipeline") or {"ok": True},
    )
    provenance = {"prerequisite_artifact_loads": 0, "prerequisite_artifact_authentication": None}
    result = runner._run_pipeline(
        tmp_path / "oracle.json",
        provenance=provenance,
        prerequisite_authenticator=lambda: calls.append("prerequisite") or {"passes": True},
        task_loader=lambda: None,
        model_loader=lambda: None,
        acquisition_solver=lambda *_: None,
        polish_solver=lambda *_: None,
        replay_solver=lambda *_: None,
        finish_provenance=lambda *_: None,
        test_override=True,
    )
    assert result == {"ok": True}
    assert calls == ["prerequisite", "v1_pipeline"]
    assert provenance["prerequisite_artifact_loads"] == 1
    assert provenance["prerequisite_artifact_authentication"] == {"passes": True}


def test_failed_prerequisite_never_enters_v1_pipeline(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(v1_runner, "_run_pipeline", lambda *_a, **_k: calls.append("v1"))
    with pytest.raises(RuntimeError, match="prerequisite authentication"):
        runner._run_pipeline(
            tmp_path / "oracle.json", provenance={},
            prerequisite_authenticator=lambda: {"passes": False},
            task_loader=lambda: calls.append("task"), model_loader=lambda: calls.append("model"),
            acquisition_solver=lambda *_: calls.append("acq"),
            polish_solver=lambda *_: calls.append("polish"),
            replay_solver=lambda *_: calls.append("worker"),
            finish_provenance=lambda *_: calls.append("finish"), test_override=True,
        )
    assert calls == []


def test_v2_provenance_requires_exact_preflight_pin_load_and_v1_exclusion(
    monkeypatch,
):
    monkeypatch.setattr(runner, "_V1_CERTIFY_PROVENANCE", lambda _row: True)
    detail = _canonical_prerequisite_authentication()
    provenance = {
        "prerequisite_artifact_pins": {
            name: {"path": str(path), "sha256": digest}
            for name, (path, digest) in schema.PREREQUISITE_ARTIFACT_PINS.items()
        },
        "prerequisite_artifact_loads": 1,
        "prerequisite_artifact_authentication": detail,
        "rejected_v1_artifact_loads": 0,
        "v1_rejected_report_only": schema.V1_REJECTED_REPORT_ONLY,
    }
    assert runner.certify_provenance(provenance)
    for key, value in (
        ("prerequisite_artifact_loads", 0),
        ("prerequisite_artifact_authentication", {"passes": False}),
        ("rejected_v1_artifact_loads", 1),
        ("v1_rejected_report_only", {}),
        ("prerequisite_artifact_pins", {}),
    ):
        changed = dict(provenance)
        changed[key] = value
        assert not runner.certify_provenance(changed)


def test_v2_context_is_scoped_and_restores_every_v1_global():
    names = (
        "ORACLE_PROTOCOL_VERSION", "ORACLE_OUTPUT_PATH", "AUTHORIZED_ORIG_ARGV",
        "RUNNER_PROTOCOL_VERSION", "REQUIRED_SOURCE_PATHS", "WORKER_PROTOCOL_VERSION",
        "validate_request", "certify_replay", "authenticate_summary",
        "certify_worker_row_boundary", "certify_provenance",
    )
    before = {name: getattr(v1_runner, name) for name in names}
    with runner._v2_runner_context():
        assert v1_runner.ORACLE_PROTOCOL_VERSION == schema.ORACLE_PROTOCOL_VERSION
        assert v1_runner.ORACLE_OUTPUT_PATH == schema.ORACLE_OUTPUT_PATH
        assert v1_runner.WORKER_PROTOCOL_VERSION == worker.WORKER_PROTOCOL_VERSION
        assert v1_runner.authenticate_summary is runner.authenticate_summary
    assert {name: getattr(v1_runner, name) for name in names} == before


def test_v2_runner_and_worker_contexts_restore_globals_after_exceptions():
    runner_names = (
        "ORACLE_PROTOCOL_VERSION", "ORACLE_OUTPUT_PATH", "AUTHORIZED_ORIG_ARGV",
        "RUNNER_PROTOCOL_VERSION", "REQUIRED_SOURCE_PATHS", "WORKER_PROTOCOL_VERSION",
        "validate_request", "certify_replay", "authenticate_summary",
        "certify_worker_row_boundary", "certify_provenance",
    )
    runner_before = {name: getattr(v1_runner, name) for name in runner_names}
    with pytest.raises(RuntimeError, match="synthetic runner"):
        with runner._v2_runner_context():
            raise RuntimeError("synthetic runner")
    assert {name: getattr(v1_runner, name) for name in runner_names} == runner_before
    worker_names = ("ORACLE_OUTPUT_PATH", "AUTHORIZED_ROOT", "WORKER_PROTOCOL_VERSION")
    worker_before = {name: getattr(v1_worker, name) for name in worker_names}
    with pytest.raises(RuntimeError, match="synthetic worker"):
        with worker._v2_worker_context():
            raise RuntimeError("synthetic worker")
    assert {name: getattr(v1_worker, name) for name in worker_names} == worker_before


def test_disk_recert_authenticates_prerequisite_before_v1_evidence(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner,
        "authenticate_prerequisite_artifact",
        lambda: calls.append("prerequisite") or _canonical_prerequisite_authentication(),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _self: json.dumps(
            {"provenance": {"prerequisite_artifact_authentication": _canonical_prerequisite_authentication()}}
        ),
    )
    monkeypatch.setattr(
        runner,
        "_V1_RECERTIFY_RETAINED_ORACLE",
        lambda *_args: calls.append("v1_recert") or {"all_disk_recertification_gates_pass": True},
    )
    result = runner.recertify_retained_oracle("a", "b", "c", "d")
    assert result["all_disk_recertification_gates_pass"] is True
    assert calls == ["prerequisite", "v1_recert"]
    calls.clear()
    monkeypatch.setattr(
        runner, "authenticate_prerequisite_artifact", lambda: {"passes": False}
    )
    assert not runner.recertify_retained_oracle("a", "b", "c", "d")[
        "all_disk_recertification_gates_pass"
    ]
    assert calls == []


def test_disk_recert_is_fail_closed_on_real_prerequisite_exception_and_restores(monkeypatch):
    names = ("ORACLE_PROTOCOL_VERSION", "ORACLE_OUTPUT_PATH", "authenticate_summary")
    before = {name: getattr(v1_runner, name) for name in names}
    monkeypatch.setattr(
        runner,
        "authenticate_prerequisite_artifact",
        lambda: (_ for _ in ()).throw(RuntimeError("real pin failure")),
    )
    result = runner.recertify_retained_oracle("a", "b", "c", "d")
    assert result == {
        "prerequisite_artifact_authentication_exact": False,
        "all_disk_recertification_gates_pass": False,
    }
    assert {name: getattr(v1_runner, name) for name in names} == before


def test_v2_worker_context_is_scoped_and_uses_v2_root_protocol():
    before = {
        name: getattr(v1_worker, name)
        for name in ("ORACLE_OUTPUT_PATH", "AUTHORIZED_ROOT", "WORKER_PROTOCOL_VERSION")
    }
    with worker._v2_worker_context():
        assert v1_worker.ORACLE_OUTPUT_PATH == schema.ORACLE_OUTPUT_PATH
        assert v1_worker.AUTHORIZED_ROOT == schema.ORACLE_OUTPUT_PATH.parent
        assert v1_worker.WORKER_PROTOCOL_VERSION == worker.WORKER_PROTOCOL_VERSION
    assert {
        name: getattr(v1_worker, name)
        for name in ("ORACLE_OUTPUT_PATH", "AUTHORIZED_ROOT", "WORKER_PROTOCOL_VERSION")
    } == before


def test_worker_request_is_exact_v2_namespace_without_constructor_calls():
    identity = schema.EXPECTED_LEDGER[0]
    root = schema.ORACLE_OUTPUT_PATH.parent
    request = {
        "identity": list(identity),
        "request_json": str(root / "oracle.worker.0000.request.json"),
        "input_npz": str(root / "oracle.worker.0000.input.npz"),
        "input_npz_sha256": "1" * 64,
        "extension_module": schema.CUDA_EXTENSION_MODULE,
        "extension_path": str(schema.CUDA_EXTENSION_PATH),
        "extension_sha256": schema.CUDA_EXTENSION_SHA256,
        "worker_protocol": worker.WORKER_PROTOCOL_VERSION,
        "output_json": str(root / "oracle.worker.0000.json"),
        "output_npz": str(root / "oracle.worker.0000.npz"),
    }
    assert worker.validate_request(request)
    request["worker_protocol"] = v1_worker.WORKER_PROTOCOL_VERSION
    assert not worker.validate_request(request)


def _worker_arrays():
    reference = np.asarray(
        [0.0, 0.0, 0.0, 0.1, 0.1, 0.03, 0.0, 0.055, 0.02, 0.005],
        dtype=np.float32,
    )
    return {
        "captured_x0_float32": np.zeros(14, np.float32),
        "captured_reference_float32": reference,
        "captured_controls_float32": np.zeros((63, 7), np.float32),
        "captured_joint_lower_float64": -np.ones(7),
        "captured_joint_upper_float64": np.ones(7),
        "captured_velocity_limit_float64": np.ones(7),
        "captured_effort_limit_float64": np.ones(7),
        "cuda_knot_states_float32": np.zeros((64, 14), np.float32),
        "cuda_dense_states_float32": np.zeros((4033, 14), np.float32),
        "cuda_dense_tool_float32": np.zeros((4033, 3), np.float32),
        "dense_time_float64": np.arange(4033, dtype=np.float64) * schema.DT / 64,
    }


def _worker_summary():
    return {
        "protocol": worker.WORKER_PROTOCOL_VERSION,
        "module_attributes": schema.CUDA_MODULE_ATTRIBUTES,
        "constructor_calls": 2,
        "sim_forward_calls": 4032,
        "tool_position_calls": 253,
        "solve_calls": 0,
        "sqp_calls": 0,
    }


def _reference():
    return np.asarray(
        [0.0, 0.0, 0.0, 0.1, 0.1, 0.03, 0.0, 0.055, 0.02, 0.005],
        dtype=np.float32,
    )


def test_v2_worker_replay_certificate_is_v1_science_with_v2_protocol():
    arrays = _worker_arrays()
    summary = _worker_summary()
    v2_result = worker.certify_replay(summary, arrays)
    assert v2_result["passes"]
    with worker._v2_worker_context():
        inherited = v1_worker.certify_replay(summary, arrays)
    assert v2_result == inherited
    summary["sim_forward_calls"] = 1
    assert not worker.certify_replay(summary, arrays)["passes"]


def test_worker_and_runner_are_closed_after_runtime_rejection(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "_run_authorized_worker", lambda *_: calls.append("worker"))
    monkeypatch.setattr(runner, "_production_pipeline", lambda *_: calls.append("runner"))
    with pytest.raises(RuntimeError, match="blocked"):
        worker.execute_worker(tmp_path / "request.json", tmp_path / "output.json", object())
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_oracle(tmp_path / "oracle.json", object())
    with pytest.raises(RuntimeError, match="blocked"):
        worker.execute_worker(
            tmp_path / "request.json",
            tmp_path / "output.json",
            worker.WORKER_EXECUTION_AUTHORIZATION,
        )
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_oracle(
            tmp_path / "oracle.json", runner.RUNNER_EXECUTION_AUTHORIZATION
        )
    assert calls == [] and list(tmp_path.iterdir()) == []

    assert calls == []


def test_full_v2_sources_refuse_resume_overwrite_and_existing_worker_artifacts():
    runner_source = inspect.getsource(runner.execute_oracle)
    replay_source = inspect.getsource(runner.ProductionOracleDependencies.replay)
    inherited_worker_source = inspect.getsource(v1_worker._run_authorized_worker)
    assert "output.parent.exists()" in runner_source
    assert "output root must be absent" in runner_source
    assert "if any(path.exists()" in replay_source
    assert "worker side artifact exists" in replay_source
    assert "exists()" in inherited_worker_source


def test_source_has_v2_worker_command_and_no_v1_artifact_consumption():
    replay_source = inspect.getsource(runner.ProductionOracleDependencies.replay)
    assert "WORKER_MODULE" in replay_source
    assert "multimodal_toll_oracle_v1_worker\"" not in replay_source
    sources = "\n".join(
        Path(module.__file__).read_text() for module in (schema, runner, worker)
    )
    assert "/tmp/tiago-tool-center-toll-oracle-v1-authorized-once" not in sources
    assert schema.V1_REJECTED_REPORT_ONLY["artifact_loads_in_v2"] == 0
    assert "generate_task(" not in sources


def test_source_order_authenticates_preflight_before_any_campaign_entry():
    source = inspect.getsource(runner._run_pipeline)
    assert source.index("prerequisite_authenticator()") < source.index(
        "_v1._run_pipeline("
    )
    production = inspect.getsource(runner._production_pipeline)
    assert "authenticate_prerequisite_artifact" in production
    assert "ProductionOracleDependencies" in production


def test_v2_real_disk_roundtrip_uses_adapter_and_binds_fresh_prerequisite_detail(
    tmp_path, monkeypatch
):
    """Exercise inherited V1 disk semantics through the real V2 public adapter."""

    identity = schema.EXPECTED_LEDGER[0]
    output = tmp_path / "oracle.json"
    authentication = _canonical_prerequisite_authentication()
    monkeypatch.setattr(
        runner, "authenticate_prerequisite_artifact", lambda: authentication
    )
    monkeypatch.setattr(v1_runner, "EXPECTED_LEDGER", (identity,))
    monkeypatch.setattr(worker, "validate_request", lambda _request: True)
    final_arrays = {
        name: np.zeros(shape, dtype)
        for name, (shape, dtype) in schema.BASE_ARRAY_SPECS.items()
    }
    final_arrays["public_reference_float32"][0] = _reference()
    final_arrays["joint_lower_float64"][:] = -1
    final_arrays["joint_upper_float64"][:] = 1
    final_arrays["velocity_limit_float64"][:] = 1
    final_arrays["effort_limit_float64"][:] = 1
    retained = {
        name: np.zeros(shape, dtype)
        for name, (shape, dtype) in schema.ROW_ARRAY_SPECS.items()
    }
    retained["dense_time_float64"][:] = (
        np.arange(schema.DENSE_SAMPLES) * schema.DT / schema.DENSE_SUBSTEPS
    )
    for name, value in retained.items():
        final_arrays[f"row_000_{name}"] = value
    acquisition = {
        name: retained[name] for name in schema.ACQUISITION_ARRAY_NAMES
    }

    with runner._v2_runner_context():
        acquisition_artifact = v1_runner.publish_acquisition(
            output, 0, identity, {"finite": True, "success": True}, acquisition
        )
        pair_artifact = v1_runner.publish_pair(output, 0, identity, retained)
        request_path = tmp_path / "oracle.worker.0000.request.json"
        input_path = tmp_path / "oracle.worker.0000.input.npz"
        worker_json = tmp_path / "oracle.worker.0000.json"
        worker_npz = tmp_path / "oracle.worker.0000.npz"
        worker_input = {
            "x0_float32": final_arrays["public_x0_float32"][0],
            "reference_float32": final_arrays["public_reference_float32"][0],
            "controls_float32": retained["controls_float64"].astype(np.float32),
            "joint_lower_float64": final_arrays["joint_lower_float64"],
            "joint_upper_float64": final_arrays["joint_upper_float64"],
            "velocity_limit_float64": final_arrays["velocity_limit_float64"],
            "effort_limit_float64": final_arrays["effort_limit_float64"],
        }
        v1_runner._atomic_npz(input_path, worker_input)
        request = {
            "identity": list(identity),
            "request_json": str(request_path),
            "input_npz": str(input_path),
            "input_npz_sha256": v1_runner.sha256_file(input_path),
            "extension_module": schema.CUDA_EXTENSION_MODULE,
            "extension_path": str(schema.CUDA_EXTENSION_PATH),
            "extension_sha256": schema.CUDA_EXTENSION_SHA256,
            "worker_protocol": worker.WORKER_PROTOCOL_VERSION,
            "output_json": str(worker_json),
            "output_npz": str(worker_npz),
        }
        v1_runner._atomic_json(request_path, request)
        worker_arrays = _worker_arrays()
        worker_arrays["captured_reference_float32"][:] = final_arrays[
            "public_reference_float32"
        ][0]
        v1_runner._atomic_npz(worker_npz, worker_arrays)
        worker_summary = {
            **_worker_summary(),
            "identity": list(identity),
            "request_path": str(request_path),
            "request_sha256": v1_runner.sha256_file(request_path),
            "input_npz": str(input_path),
            "input_npz_sha256": v1_runner.sha256_file(input_path),
            "extension_path": str(schema.CUDA_EXTENSION_PATH),
            "extension_sha256": schema.CUDA_EXTENSION_SHA256,
            "array_names": sorted(worker_arrays),
            "array_hashes": {
                name: schema.array_hash(value)
                for name, value in worker_arrays.items()
            },
            "npz_sha256": v1_runner.sha256_file(worker_npz),
        }
        worker_summary["certificate"] = worker.certify_replay(
            worker_summary, worker_arrays
        )
        v1_runner._atomic_json(worker_json, worker_summary)
        artifact = {
            "request_path": str(request_path),
            "request_sha256": v1_runner.sha256_file(request_path),
            "input_path": str(input_path),
            "input_sha256": v1_runner.sha256_file(input_path),
            "json_path": str(worker_json),
            "json_sha256": v1_runner.sha256_file(worker_json),
            "npz_path": str(worker_npz),
            "npz_sha256": v1_runner.sha256_file(worker_npz),
        }
        row = {
            "identity": list(identity),
            "acquisition": {"finite": True, "success": True},
            "acquisition_side_artifact": acquisition_artifact,
            "pair_side_artifact": pair_artifact,
            "worker_artifact": artifact,
            "worker": {
                **worker_summary,
                "artifact": artifact,
                "command": [
                    "python", "-B", "-m", runner.WORKER_MODULE,
                    "--request", str(request_path), "--output", str(worker_json),
                ],
                "cwd": str(schema.AUTHORIZED_CWD),
                "exit_code": 0,
            },
        }
        base_counts = {
            "task_artifact_loads": 0,
            "model_artifact_loads": 0,
            "acquisition_optimizer_calls": 0,
            "polish_optimizer_calls": 0,
            "worker_subprocess_calls": 0,
            "cuda_replay_calls": 0,
            "bootstrap_rng_calls": 0,
            "git_head_at_end": None,
            "tracked_tree_clean_at_end": None,
            "full_git_status_at_end": None,
            "source_hashes_at_end": None,
            "extension_sha256_at_end": None,
            "extension_size_bytes_at_end": None,
            "prerequisite_artifact_pins": {
                name: {"path": str(path), "sha256": digest}
                for name, (path, digest) in schema.PREREQUISITE_ARTIFACT_PINS.items()
            },
            "prerequisite_artifact_loads": 1,
            "prerequisite_artifact_authentication": authentication,
            "rejected_v1_artifact_loads": 0,
            "v1_rejected_report_only": schema.V1_REJECTED_REPORT_ONLY,
        }
        v1_runner._checkpoint(output, 0, "before_prerequisite_loads", [], {}, dict(base_counts))
        provenance = {**base_counts, "task_artifact_loads": 1}
        v1_runner._checkpoint(output, 1, "task_authenticated", [], {}, provenance)
        provenance = {**provenance, "model_artifact_loads": 1}
        v1_runner._checkpoint(output, 2, "model_authenticated", [], {}, provenance)
        provenance = {
            **provenance,
            "acquisition_optimizer_calls": 1,
            "polish_optimizer_calls": 1,
            "worker_subprocess_calls": 1,
            "cuda_replay_calls": 1,
        }
        v1_runner._checkpoint(output, 3, "pair_completed", [row], {}, provenance)
        provenance = {**provenance, "bootstrap_rng_calls": 1}
        v1_runner._checkpoint(output, 4, "end_provenance", [row], {}, provenance)

        side_paths = sorted(
            [
                *tmp_path.glob("oracle.partial.[0-9][0-9][0-9][0-9].json"),
                *tmp_path.glob("oracle.partial.[0-9][0-9][0-9][0-9].npz"),
                *tmp_path.glob("oracle.acquisition.*"),
                *tmp_path.glob("oracle.pair.*"),
                *tmp_path.glob("oracle.worker.*"),
            ]
        )
        manifest = {
            "all_side_artifacts": [
                {"path": str(path), "sha256": v1_runner.sha256_file(path)}
                for path in side_paths
            ]
        }
        fake_certificate = {
            "canonical_rows": [{"identity": list(identity)}],
            "all_oracle_gates_pass": True,
        }
        monkeypatch.setattr(
            v1_runner,
            "_authenticated_recertification_factory",
            lambda _arrays: (lambda _index: None),
        )
        monkeypatch.setattr(
            v1_runner, "certify_final_oracle", lambda *_a, **_k: fake_certificate
        )
        monkeypatch.setattr(
            v1_runner,
            "exact_final_array_schema",
            lambda arrays: set(arrays) == set(final_arrays),
        )
        npz_path = output.with_suffix(".npz")
        v1_runner._atomic_npz(npz_path, final_arrays)
        summary = {
            "protocol": schema.ORACLE_PROTOCOL_VERSION,
            "incomplete": False,
            "all_oracle_gates_pass": True,
            "rows": [row],
            "provenance": provenance,
            "canonical_rows": fake_certificate["canonical_rows"],
            "certificate": fake_certificate,
            "array_names": sorted(final_arrays),
            "array_hashes": {
                name: schema.array_hash(value)
                for name, value in final_arrays.items()
            },
            "npz_path": str(npz_path),
            "npz_sha256": v1_runner.sha256_file(npz_path),
        }
        v1_runner._atomic_json(output, summary)
        manifest.update(
            {
                "protocol": schema.ORACLE_PROTOCOL_VERSION,
                "incomplete": False,
                "all_oracle_gates_pass": True,
                "json_path": str(output),
                "json_sha256": v1_runner.sha256_file(output),
                "npz_path": str(npz_path),
                "npz_sha256": v1_runner.sha256_file(npz_path),
                "side_artifact_count": len(manifest["all_side_artifacts"]),
                "acquisition_side_artifacts": [acquisition_artifact],
            }
        )
        manifest_path = output.with_suffix(".manifest.json")
        v1_runner._atomic_json(manifest_path, manifest)
        pointer_path = tmp_path / "oracle.final.latest.json"
        v1_runner._atomic_json(
            pointer_path,
            {
                "generation": 123,
                "incomplete": False,
                "superseded_by": str(output),
                "json_path": str(output),
                "json_sha256": v1_runner.sha256_file(output),
                "npz_path": str(npz_path),
                "npz_sha256": v1_runner.sha256_file(npz_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": v1_runner.sha256_file(manifest_path),
            },
        )
        monkeypatch.setattr(
            v1_runner, "EXPECTED_SIDE_ARTIFACT_COUNT", len(manifest["all_side_artifacts"])
        )
        result = runner.recertify_retained_oracle(
            output, npz_path, manifest_path, pointer_path
        )
        assert result["all_disk_recertification_gates_pass"], json.dumps(
            result, indent=2, sort_keys=True, default=str
        )
        assert result["prerequisite_artifact_authentication_exact"]

        original_summary = output.read_bytes()
        changed = json.loads(original_summary)
        changed["provenance"]["prerequisite_artifact_authentication"][
            "pin_results"
        ]["final_json"]["actual_sha256"] = "0" * 64
        output.write_text(json.dumps(changed, sort_keys=True))
        manifest["json_sha256"] = v1_runner.sha256_file(output)
        v1_runner._atomic_json(manifest_path, manifest)
        pointer = json.loads(pointer_path.read_text())
        pointer["json_sha256"] = v1_runner.sha256_file(output)
        pointer["manifest_sha256"] = v1_runner.sha256_file(manifest_path)
        v1_runner._atomic_json(pointer_path, pointer)
        assert not runner.recertify_retained_oracle(
            output, npz_path, manifest_path, pointer_path
        )["all_disk_recertification_gates_pass"]


@pytest.mark.parametrize(
    "elapsed,trigger", [(181.0, "projected_total"), (21600.0, "total_deadline")]
)
def test_v2_campaign_watchdog_is_unchanged_and_stops_after_one_pair(
    tmp_path, monkeypatch, elapsed, trigger
):
    output = tmp_path / "oracle.json"
    calls = {"acquisition": 0, "polish": 0, "worker": 0}
    times = iter((0.0, elapsed))
    monkeypatch.setattr(runner, "authenticate_summary", lambda *_a, **_k: True)
    monkeypatch.setattr(v1_runner, "cross_bind_prerequisites", lambda *_: True)
    model = {
        "public_solver_x0_float32": np.zeros((12, 14), np.float32),
        "public_reference_float32": np.zeros((12, 10), np.float32),
        "public_default_side_int8": np.ones(12, np.int8),
        "quarantined_q8_float64": np.zeros((12, 7)),
        "model_lower_float64": -np.ones(7),
        "model_upper_float64": np.ones(7),
        "model_velocity_float64": np.ones(7),
        "model_effort_float64": np.ones(7),
    }

    def acquisition(*_args):
        calls["acquisition"] += 1
        return {"finite": True, "passes": True}, {
            name: np.zeros(shape, dtype)
            for name, (shape, dtype) in schema.ROW_ARRAY_SPECS.items()
            if name in schema.ACQUISITION_ARRAY_NAMES
        }

    def polish(_payload):
        calls["polish"] += 1
        return {"passes": True}, {
            name: np.zeros(shape, dtype)
            for name, (shape, dtype) in schema.ROW_ARRAY_SPECS.items()
            if name in schema.POLISH_ARRAY_NAMES
        }

    def replay(*_args):
        calls["worker"] += 1
        return {"artifact": {}}, {
            name: np.zeros(shape, dtype)
            for name, (shape, dtype) in schema.ROW_ARRAY_SPECS.items()
            if name in schema.REPLAY_ARRAY_NAMES
        }

    deadlines = []
    provenance = {
        "prerequisite_artifact_loads": 0,
        "prerequisite_artifact_authentication": None,
    }
    with pytest.raises(RuntimeError, match="permanently rejected"):
        runner._run_pipeline(
            output,
            provenance=provenance,
            prerequisite_authenticator=lambda: {"passes": True},
            task_loader=lambda: ({}, [], {}),
            model_loader=lambda: ({}, model),
            acquisition_solver=acquisition,
            polish_solver=polish,
            replay_solver=replay,
            finish_provenance=lambda _row: None,
            test_override=True,
            monotonic=lambda: next(times),
            campaign_deadline_setter=deadlines.append,
        )
    assert calls == {"acquisition": 1, "polish": 1, "worker": 1}
    assert deadlines == [schema.CAMPAIGN_WALL_LIMIT_S]
    pointer = json.loads((tmp_path / "oracle.partial.latest.json").read_text())
    retained = json.loads(Path(pointer["json_path"]).read_text())
    assert retained["stage"] == "runtime_watchdog_rejected"
    assert retained["completed_count"] == 1
    assert retained["runtime_watchdog"]["trigger"] == trigger
    assert not output.exists() and not output.with_suffix(".npz").exists()
