import inspect
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from gato_tiago import l2_portability_smoke as schema
from gato_tiago import l2_portability_smoke_runner as runner
from gato_tiago import l2_portability_smoke_worker as worker


def _arrays(xyz=(0.1, 0.2, 0.3)):
    q = np.zeros(7, dtype=np.float32)
    fk1 = np.asarray(xyz, dtype=np.float32)[None, :]
    return {
        "q_float32": q,
        "q16_float32": np.repeat(q[None, :], 16, axis=0),
        "fk_b1_float32": fk1,
        "fk_b16_float32": np.repeat(fk1, 16, axis=0),
    }


def _rows_and_arrays():
    rows = []
    arrays = {}
    for spec in schema.FROZEN_MODULES:
        rows.append(
            {
                "module_name": spec["module_name"],
                "passes": True,
                "constructor_calls": 2,
                "fk_calls": 2,
                "forbidden_call_counts": dict(schema.FORBIDDEN_CALL_COUNTS),
            }
        )
        arrays[spec["module_name"]] = _arrays()
    return rows, arrays


def _provenance():
    sources = {
        name: {"path": path, "sha256": "b" * 64}
        for name, path in schema.REQUIRED_SOURCE_PATHS.items()
    }
    return {
        "git_head_at_start": "a" * 40,
        "git_head_at_end": "a" * 40,
        "tracked_clean_at_start": True,
        "tracked_clean_at_end": True,
        "source_hashes_at_start": copy.deepcopy(sources),
        "source_hashes_at_end": copy.deepcopy(sources),
        "cwd": str(schema.AUTHORIZED_CWD),
        "orig_argv": list(schema.AUTHORIZED_ORIG_ARGV),
        "exact_command": " ".join(schema.AUTHORIZED_ORIG_ARGV),
        "frozen_build_head": schema.FROZEN_BUILD_HEAD,
        "frozen_build_source_hashes": dict(schema.FROZEN_BUILD_SOURCE_HASHES),
        "frozen_cuda_arch": schema.FROZEN_CUDA_ARCH,
        "extension_hashes": {
            row["module_name"]: row["extension_sha256"]
            for row in schema.FROZEN_MODULES
        },
        "extension_sizes": {
            row["module_name"]: row["extension_size_bytes"]
            for row in schema.FROZEN_MODULES
        },
        "forbidden_call_counts": dict(schema.FORBIDDEN_CALL_COUNTS),
    }


def test_completed_one_shot_authorizations_are_closed():
    assert schema.SMOKE_EXECUTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert schema.AUTHORIZED_OUTPUT_PATH == Path(
        "/tmp/tiago-tool-center-l2-portability-smoke-authorized-once/smoke.json"
    )
    assert len(schema.FROZEN_MODULES) == 3
    assert [row["reference_size"] for row in schema.FROZEN_MODULES] == [10, 6, 6]
    assert all(len(row["extension_sha256"]) == 64 for row in schema.FROZEN_MODULES)
    assert [row["extension_size_bytes"] for row in schema.FROZEN_MODULES] == [
        6686384,
        6764208,
        6678192,
    ]
    assert schema.FROZEN_CUDA_ARCH == "61-real"
    assert schema.AUTHORIZED_CWD == Path("/workspace/GATO")
    assert list(schema.AUTHORIZED_ORIG_ARGV) == [
        "python",
        "-B",
        "-m",
        "gato_tiago.l2_portability_smoke_runner",
        "--execute",
        "--output",
        str(schema.AUTHORIZED_OUTPUT_PATH),
    ]
    assert schema.EXPECTED_GLOBAL_CONSTRUCTOR_CALLS == 6
    assert schema.EXPECTED_GLOBAL_FK_CALLS == 6
    assert all(value == 0 for value in schema.FORBIDDEN_CALL_COUNTS.values())


def test_authorized_entrypoints_reject_wrong_paths_and_refuse_overwrite(
    tmp_path, monkeypatch
):
    runner_authorization = object()
    worker_authorization = object()
    monkeypatch.setattr(runner, "RUNNER_EXECUTION_AUTHORIZATION", runner_authorization)
    monkeypatch.setattr(worker, "WORKER_EXECUTION_AUTHORIZATION", worker_authorization)
    authorized = tmp_path / "authorized" / "smoke.json"
    monkeypatch.setattr(runner, "AUTHORIZED_OUTPUT_PATH", authorized)
    monkeypatch.setattr(worker, "AUTHORIZED_OUTPUT_PATH", authorized)
    runner_calls = []

    def fake_pipeline(output, *, token=None):
        assert token is runner._PRODUCTION_TOKEN
        runner._no_existing(output)
        runner_calls.append(Path(output))
        return {"mock": True}

    monkeypatch.setattr(runner, "_production_pipeline", fake_pipeline)
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_smoke(authorized, authorization=object())
    with pytest.raises(RuntimeError, match="not authorized"):
        runner.execute_smoke(
            tmp_path / "wrong.json",
            authorization=runner_authorization,
        )
    assert runner.execute_smoke(
        authorized,
        authorization=runner_authorization,
    ) == {"mock": True}
    authorized.parent.mkdir(parents=True, exist_ok=True)
    authorized.with_name("smoke.partial.latest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="overwrite, resume, or rerun"):
        runner.execute_smoke(
            authorized,
            authorization=runner_authorization,
        )
    assert runner_calls == [authorized]

    worker_calls = []
    monkeypatch.setattr(
        worker,
        "_run_worker",
        lambda request, output: worker_calls.append((request, output)) or {"mock": True},
    )
    with pytest.raises(RuntimeError, match="blocked"):
        worker.execute_worker(
            authorized.parent / "wrong.request.json",
            authorized.parent / "wrong.json",
            authorization=object(),
        )
    expected_pairs = []
    for spec in schema.FROZEN_MODULES:
        leaf = spec["module_name"].split(".")[-1]
        request = authorized.parent / f"smoke.{leaf}.request.json"
        output = authorized.parent / f"smoke.{leaf}.json"
        expected_pairs.append((request.resolve(), output.resolve()))
        assert worker.execute_worker(
            request,
            output,
            authorization=worker_authorization,
        ) == {"mock": True}
    with pytest.raises(RuntimeError, match="not authorized"):
        worker.execute_worker(
            expected_pairs[0][0],
            expected_pairs[1][1],
            authorization=worker_authorization,
        )
    expected_pairs[0][1].write_text("{}")
    with pytest.raises(RuntimeError, match="overwrite or rerun"):
        worker.execute_worker(
            *expected_pairs[0],
            authorization=worker_authorization,
        )
    assert worker_calls == expected_pairs


def test_worker_pure_certificate_accepts_only_exact_zero_and_repeated_fk():
    assert worker.certify_arrays(_arrays())["all_worker_array_gates_pass"]
    mutations = []
    bad = _arrays()
    bad["q_float32"][0] = 1
    mutations.append(bad)
    bad = _arrays()
    bad["q16_float32"] = bad["q16_float32"].astype(np.float64)
    mutations.append(bad)
    bad = _arrays()
    bad["fk_b1_float32"][0, 0] = np.nan
    mutations.append(bad)
    bad = _arrays()
    bad["fk_b16_float32"][7, 0] += 1
    mutations.append(bad)
    bad = _arrays()
    bad["extra"] = np.zeros(1)
    mutations.append(bad)
    for bad in mutations:
        assert not worker.certify_arrays(bad)["all_worker_array_gates_pass"]


@pytest.mark.parametrize(
    "mutation",
    [
        "count",
        "fk_count",
        "forbidden",
        "cross",
        "missing",
        "head",
        "hash",
        "source",
        "source_valid_sha",
        "cwd",
        "argv",
        "size",
        "arch",
    ],
)
def test_global_certificate_rejects_count_array_cross_and_provenance_mutations(mutation):
    rows, arrays = _rows_and_arrays()
    provenance = _provenance()
    if mutation == "count":
        rows[0]["constructor_calls"] = 1
    elif mutation == "fk_count":
        rows[1]["fk_calls"] = 3
    elif mutation == "forbidden":
        rows[2]["forbidden_call_counts"]["solve_calls"] = 1
    elif mutation == "cross":
        arrays[rows[2]["module_name"]]["fk_b1_float32"][0, 0] += 1
        arrays[rows[2]["module_name"]]["fk_b16_float32"][:, 0] += 1
    elif mutation == "missing":
        del arrays[rows[0]["module_name"]]["q_float32"]
    elif mutation == "head":
        provenance["git_head_at_end"] = "c" * 40
    elif mutation == "hash":
        provenance["extension_hashes"][rows[0]["module_name"]] = "d" * 64
    elif mutation == "source":
        provenance["source_hashes_at_start"]["worker"]["sha256"] = "not-a-hash"
    elif mutation == "source_valid_sha":
        provenance["source_hashes_at_end"]["worker"]["sha256"] = "c" * 64
    elif mutation == "cwd":
        provenance["cwd"] = "/tmp"
    elif mutation == "argv":
        provenance["orig_argv"][-1] = "/tmp/replacement.json"
    elif mutation == "size":
        provenance["extension_sizes"][rows[0]["module_name"]] += 1
    else:
        provenance["frozen_cuda_arch"] = "75-real"
    assert not runner.certify_smoke(rows, arrays, provenance)[
        "all_smoke_gates_pass"
    ]


def test_global_certificate_passes_exact_synthetic_contract():
    rows, arrays = _rows_and_arrays()
    result = runner.certify_smoke(rows, arrays, _provenance())
    assert result["all_smoke_gates_pass"]
    assert result["constructor_calls"] == 6
    assert result["fk_calls"] == 6
    assert result["cross_module_fk_bitwise_equal"]
    assert result["optimization_evidence"] is False
    assert result["timing_evidence"] is False


def test_transaction_attempts_all_three_and_retains_failures(tmp_path):
    attempts = []
    output = tmp_path / "smoke.json"

    def launcher(spec):
        attempts.append(spec["module_name"])
        if len(attempts) in (1, 3):
            raise RuntimeError(f"injected{len(attempts)}")
        return {
            "module_name": spec["module_name"],
            "passes": True,
            "constructor_calls": 2,
            "fk_calls": 2,
            "forbidden_call_counts": dict(schema.FORBIDDEN_CALL_COUNTS),
        }, _arrays()

    provenance = _provenance()
    expected_end = copy.deepcopy(provenance["source_hashes_at_end"])
    provenance["git_head_at_end"] = None
    provenance["tracked_clean_at_end"] = None
    provenance["source_hashes_at_end"] = None

    def finish(value):
        value["git_head_at_end"] = value["git_head_at_start"]
        value["tracked_clean_at_end"] = True
        value["source_hashes_at_end"] = expected_end

    result = runner._run_pipeline(
        output, provenance, launcher, test_override=True, finish=finish
    )
    assert attempts == [row["module_name"] for row in schema.FROZEN_MODULES]
    assert result["all_smoke_gates_pass"] is False
    pointer = json.loads((tmp_path / "smoke.partial.latest.json").read_text())
    assert pointer["generation"] == 4
    partial = json.loads(Path(pointer["json_path"]).read_text())
    assert partial["incomplete"] is True
    assert [row["passes"] for row in partial["rows"]] == [False, True, False]
    assert partial["provenance"]["git_head_at_end"] == "a" * 40
    assert partial["provenance"]["tracked_clean_at_end"] is True
    assert partial["provenance"]["source_hashes_at_end"] == expected_end
    assert not list(tmp_path.glob("*.tmp"))


def test_generation_zero_is_first_and_permanently_forbids_rerun(tmp_path):
    output = tmp_path / "smoke.json"
    calls = []

    def launcher(spec):
        calls.append(spec["module_name"])
        raise RuntimeError("injected")

    result = runner._run_pipeline(
        output, _provenance(), launcher, test_override=True
    )
    assert result["all_smoke_gates_pass"] is False
    generation0 = json.loads((tmp_path / "smoke.partial.0000.json").read_text())
    assert generation0["completed_worker_count"] == 0
    assert generation0["pending_modules"] == [
        row["module_name"] for row in schema.FROZEN_MODULES
    ]
    with pytest.raises(RuntimeError, match="overwrite, resume, or rerun"):
        runner._run_pipeline(
            output, _provenance(), launcher, test_override=True
        )


def test_worker_source_has_exact_two_calls_and_no_forbidden_capabilities():
    source = inspect.getsource(worker._run_worker)
    assert source.count("BSQP_1_float()") == 1
    assert source.count("BSQP_16_float()") == 1
    assert source.count(".tool_position(") == 2
    assert source.count("constructor_calls += 1") == 2
    assert source.count("fk_calls += 1") == 2
    for forbidden in (
        "default_rng",
        "generate_task",
        "sim_forward",
        ".solve(",
        "pinocchio",
        "initializer",
    ):
        assert forbidden not in source


def test_source_and_build_hash_manifests_are_exact_without_loading_extensions():
    root = Path(__file__).resolve().parents[2]
    assert runner.repository_root(Path(runner.__file__)) == root
    assert all((root / path).is_file() for path in schema.REQUIRED_SOURCE_PATHS.values())
    assert {
        path: runner.sha256_file(root / path)
        for path in schema.FROZEN_BUILD_SOURCE_HASHES
    } == schema.FROZEN_BUILD_SOURCE_HASHES
    sources = "\n".join(
        Path(module.__file__).read_text() for module in (schema, runner, worker)
    )
    assert "default_rng(" not in sources
    assert "multimodal_toll_v4_runner" not in sources
    provenance_source = inspect.getsource(runner._source_provenance)
    assert '"git_head_at_end": None' in provenance_source
    assert '"tracked_clean_at_end": None' in provenance_source
    assert '"source_hashes_at_end": None' in provenance_source


def test_worker_boundary_rejects_path_hash_protocol_command_and_array_mutations(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    root.mkdir()
    extension = root / "extension.so"
    extension.write_bytes(b"synthetic extension")
    digest = runner.sha256_file(extension)
    spec = {
        "module_name": "synthetic.module",
        "extension_path": "extension.so",
        "extension_sha256": digest,
        "extension_size_bytes": extension.stat().st_size,
        "reference_size": 6,
    }
    output_root = tmp_path / "authorized"
    authorized = output_root / "smoke.json"
    monkeypatch.setattr(runner, "AUTHORIZED_OUTPUT_PATH", authorized)
    monkeypatch.setattr(runner, "repository_root", lambda *_: root)
    leaf = "module"
    request_path = output_root / f"smoke.{leaf}.request.json"
    summary_path = output_root / f"smoke.{leaf}.json"
    npz_path = summary_path.with_suffix(".npz")
    output_root.mkdir()
    request = {
        "module_name": spec["module_name"],
        "extension_path": str(extension.resolve()),
        "extension_sha256": digest,
        "extension_size_bytes": extension.stat().st_size,
        "cuda_arch": schema.FROZEN_CUDA_ARCH,
        "protocol_version": worker.WORKER_PROTOCOL_VERSION,
    }
    request_path.write_text(json.dumps(request))
    arrays = _arrays()
    np.savez_compressed(npz_path, **arrays)
    summary = {
        "protocol_version": schema.PROTOCOL_VERSION,
        "worker_protocol_version": worker.WORKER_PROTOCOL_VERSION,
        "module_name": spec["module_name"],
        "module_file": str(extension.resolve()),
        "extension_sha256": digest,
        "extension_size_bytes": extension.stat().st_size,
        "cuda_arch": schema.FROZEN_CUDA_ARCH,
        "KNOT_POINTS": 64,
        "REFERENCE_SIZE": 6,
        "TOOL_POSITION_FRAME": "arm_right_tool_joint_origin",
        "TOOL_POSITION_SIZE": 3,
        "request_path": str(request_path),
        "request_sha256": runner.sha256_file(request_path),
        "constructor_calls": 2,
        "fk_calls": 2,
        "forbidden_call_counts": dict(schema.FORBIDDEN_CALL_COUNTS),
        "certificate": {"all_worker_array_gates_pass": True},
        "npz_path": str(npz_path),
        "npz_sha256": runner.sha256_file(npz_path),
        "array_names": sorted(arrays),
        "array_hashes": {
            name: worker.array_hash(value) for name, value in arrays.items()
        },
        "optimization_evidence": False,
        "timing_evidence": False,
    }
    summary_path.write_text(json.dumps(summary))
    row = {
        "exit_code": 0,
        "module_name": spec["module_name"],
        "command": [
            sys.executable,
            "-B",
            "-m",
            "gato_tiago.l2_portability_smoke_worker",
            "--request",
            str(request_path),
            "--output",
            str(summary_path),
            "--execute",
        ],
        "request_path": str(request_path),
        "request_sha256": runner.sha256_file(request_path),
        "summary_path": str(summary_path),
        "summary_sha256": runner.sha256_file(summary_path),
        "npz_path": str(npz_path),
        "npz_sha256": runner.sha256_file(npz_path),
    }
    assert runner.validate_worker_boundary(row, spec, request, summary, arrays)
    mutations = []
    bad = copy.deepcopy(row)
    bad["command"][-1] = "--wrong"
    mutations.append((bad, request, summary, arrays))
    bad = copy.deepcopy(row)
    bad["request_sha256"] = "0" * 64
    mutations.append((bad, request, summary, arrays))
    bad_summary = copy.deepcopy(summary)
    bad_summary["worker_protocol_version"] = "wrong"
    mutations.append((row, request, bad_summary, arrays))
    bad_request = copy.deepcopy(request)
    bad_request["extension_sha256"] = "0" * 64
    mutations.append((row, bad_request, summary, arrays))
    bad_arrays = _arrays()
    bad_arrays["fk_b16_float32"][0, 0] += 1
    mutations.append((row, request, summary, bad_arrays))
    bad_summary = copy.deepcopy(summary)
    bad_summary["TOOL_POSITION_FRAME"] = "wrong_frame"
    mutations.append((row, request, bad_summary, arrays))
    for key, bad_value in (
        ("KNOT_POINTS", 63),
        ("REFERENCE_SIZE", 10),
        ("TOOL_POSITION_SIZE", 4),
        ("cuda_arch", "75-real"),
        ("extension_size_bytes", extension.stat().st_size + 1),
    ):
        bad_summary = copy.deepcopy(summary)
        bad_summary[key] = bad_value
        mutations.append((row, request, bad_summary, arrays))
    bad_request = copy.deepcopy(request)
    bad_request["extension_size_bytes"] += 1
    mutations.append((row, bad_request, summary, arrays))
    bad_request = copy.deepcopy(request)
    bad_request["cuda_arch"] = "75-real"
    mutations.append((row, bad_request, summary, arrays))
    for bad_row, bad_request, bad_summary, bad_arrays in mutations:
        assert not runner.validate_worker_boundary(
            bad_row, spec, bad_request, bad_summary, bad_arrays
        )
