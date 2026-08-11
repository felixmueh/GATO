import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from gato_tiago import multimodal_toll_v0_runner as runner
from gato_tiago import multimodal_toll_v0_worker as worker


def _smoke_raw(reference_size=10, delta=0.25):
    xu1 = np.zeros((1, 1337), dtype=np.float32)
    xu16 = np.zeros((16, 1337), dtype=np.float32)
    initial = np.asarray(
        np.r_[np.repeat(3.0, 8), np.repeat(3.0 + delta, 8)], dtype=np.float32
    )
    baseline = np.zeros(reference_size, dtype=np.float32)
    perturbed = baseline.copy()
    perturbed[6:8] = 1 if reference_size == 10 else perturbed[6:8]
    return {
        "input_xu_b1": xu1,
        "output_xu_b1": xu1.copy(),
        "input_xu_b16": xu16,
        "output_xu_b16": xu16.copy(),
        "initial_merit_b1": np.asarray([3.0], dtype=np.float32),
        "final_merit_b1": np.asarray([3.0 + 1e-7], dtype=np.float32),
        "initial_merit_b16": initial,
        "final_merit_b16": (initial + np.float32(1e-7)).astype(np.float32),
        "sqp_iters_b1": np.zeros(1, dtype=np.int32),
        "sqp_iters_b16": np.zeros(16, dtype=np.int32),
        "pcg_iters_b1": np.empty((0, 1), dtype=np.int32),
        "pcg_iters_b16": np.empty((0, 16), dtype=np.int32),
        "ls_num_iters_b1": 0,
        "ls_num_iters_b16": 0,
        "baseline_reference": baseline,
        "perturbed_reference": perturbed,
    }


def test_worker_reference_smoke_is_zero_iteration_and_tolerance_based():
    result = worker.certify_reference_smoke(
        _smoke_raw(), variant="toll", reference_size=10
    )
    assert result["all_reference_smoke_gates_pass"]
    assert result["diagnostic_zero_iteration_solve_calls"] == 2
    assert result["sqp_optimization_calls"] == 0
    assert result["optimization_evidence"] is False
    assert result["raw_initial_merit_b16"] == [3.0] * 8 + [3.25] * 8


@pytest.mark.parametrize(
    ("variant", "width", "changed"),
    [("plain_ref6", 6, (0,)), ("pillar_ref6", 6, (0, 3, 4))],
)
def test_legacy_reference_smokes_require_exact_native_fields(variant, width, changed):
    raw = _smoke_raw(width)
    raw["perturbed_reference"] = raw["baseline_reference"].copy()
    raw["perturbed_reference"][list(changed)] = 1
    result = worker.certify_reference_smoke(
        raw, variant=variant, reference_size=width
    )
    assert result["all_reference_smoke_gates_pass"]
    raw["perturbed_reference"][next(index for index in range(width) if index not in changed)] = 1
    assert not worker.certify_reference_smoke(
        raw, variant=variant, reference_size=width
    )["all_reference_smoke_gates_pass"]


@pytest.mark.parametrize(
    "mutation", ["xu", "pcg", "merit", "reference", "dtype", "nan", "telemetry_dtype"]
)
def test_worker_reference_smoke_rejects_adversarial_mutations(mutation):
    raw = _smoke_raw()
    if mutation == "xu":
        raw["output_xu_b16"][0, 0] = 1
    elif mutation == "pcg":
        raw["pcg_iters_b16"] = np.zeros((1, 16), dtype=np.int32)
    elif mutation == "merit":
        raw["final_merit_b1"][0] += 1e-2
    elif mutation == "reference":
        raw["perturbed_reference"][:] = raw["baseline_reference"]
    elif mutation == "dtype":
        raw["input_xu_b1"] = raw["input_xu_b1"].astype(np.float64)
    elif mutation == "nan":
        raw["baseline_reference"][0] = np.nan
    else:
        raw["sqp_iters_b16"] = raw["sqp_iters_b16"].astype(np.int64)
    assert not worker.certify_reference_smoke(
        raw, variant="toll", reference_size=10
    )["all_reference_smoke_gates_pass"]


def test_worker_and_runner_block_before_read_import_or_filesystem(tmp_path, monkeypatch):
    touched = []
    monkeypatch.setattr(worker.Path, "read_text", lambda *_: touched.append("read"))
    with pytest.raises(RuntimeError, match="blocked"):
        worker.execute_worker(tmp_path / "absent.json", tmp_path / "out.json")
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_v0_runner(tmp_path / "out.json")
    assert touched == []
    assert list(tmp_path.iterdir()) == []
    assert list(inspect.signature(runner.execute_v0_runner).parameters) == [
        "output",
        "authorization",
    ]


def test_frozen_worker_specs_are_exact_and_one_module_each():
    assert [row["module_name"] for row in runner.FROZEN_EXTENSIONS] == list(
        worker.SUPPORTED_MODULES
    )
    assert [row["reference_size"] for row in runner.FROZEN_EXTENSIONS] == [10, 6, 6]
    assert len({row["extension_sha256"] for row in runner.FROZEN_EXTENSIONS}) == 3
    assert all(len(row["extension_sha256"]) == 64 for row in runner.FROZEN_EXTENSIONS)
    source = inspect.getsource(runner._production_pipeline)
    assert "importlib.import_module" not in source
    assert source.count("oracle.generate_task(") == 1


def _synthetic_ledger():
    return tuple(("synthetic", 900000 + index) for index in range(12))


def _provenance():
    return {"test_override": True, "optimization_evidence": False}


def test_mock_runner_uses_exact_order_and_one_fresh_process_per_module(tmp_path):
    task_calls = []
    worker_calls = []
    ledger = _synthetic_ledger()

    def task_factory(identity):
        task_calls.append(identity)
        return {"identity": identity, "passes": True}, {"q": np.zeros(7)}

    specs = [
        {"module_name": f"synthetic.module{index}"} for index in range(3)
    ]

    def worker_launcher(spec, arrays):
        worker_calls.append(spec["module_name"])
        identity = ("worker", spec["module_name"])
        return {
            "identity": identity,
            "passes": True,
            "command": [sys_executable(), "-m", spec["module_name"]],
        }, {"captured": np.asarray([len(arrays)])}

    result = runner._run_pipeline(
        tmp_path / "v0.json",
        ledger=ledger,
        task_factory=task_factory,
        worker_specs=specs,
        worker_launcher=worker_launcher,
        base_arrays={"base": np.zeros(1)},
        provenance=_provenance(),
        test_override=True,
    )
    assert task_calls == list(ledger)
    assert worker_calls == [row["module_name"] for row in specs]
    assert result["task_construction_calls"] == 12
    assert result["worker_subprocess_calls"] == 3
    assert result["fresh_subprocess_per_extension"]
    assert result["sqp_optimization_calls"] == 0
    assert result["all_v0_gates_pass"] is False
    assert result["optimization_evidence"] is False


def sys_executable():
    # Avoid depending on a literal interpreter path in the mock contract.
    return "python-static-mock"


def test_failure_retains_incomplete_checkpoint_and_never_final(tmp_path):
    calls = 0
    ledger = _synthetic_ledger()

    def task_factory(identity):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected task failure")
        return {"identity": identity, "passes": True}, {
            "value": np.asarray([calls], dtype=np.int64)
        }

    output = tmp_path / "v0.json"
    with pytest.raises(RuntimeError, match="injected"):
        runner._run_pipeline(
            output,
            ledger=ledger,
            task_factory=task_factory,
            worker_specs=(),
            worker_launcher=lambda *_: pytest.fail("worker called"),
            base_arrays={"base": np.zeros(1)},
            provenance=_provenance(),
            test_override=True,
        )
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
    pointer = json.loads((tmp_path / "v0.partial.latest.json").read_text())
    assert pointer["generation"] == 3
    assert Path(pointer["json_path"]).name == "v0.partial.0003.json"
    assert Path(pointer["npz_path"]).name == "v0.partial.0003.npz"
    checkpoint = json.loads(Path(pointer["json_path"]).read_text())
    assert checkpoint["incomplete"] is True
    assert checkpoint["all_v0_gates_pass"] is False
    assert len(checkpoint["completed_identities"]) == 2
    assert len(checkpoint["pending_identities"]) == 10
    assert checkpoint["task_construction_attempt_count"] == 3
    assert checkpoint["rows"][-1]["identity"] == list(ledger[2])
    assert checkpoint["rows"][-1]["error_type"] == "RuntimeError"
    assert "injected task failure" in checkpoint["rows"][-1]["traceback"]
    assert runner._sha256_file(pointer["npz_path"]) == pointer["npz_sha256"]
    assert not list(tmp_path.glob("*.tmp"))


def test_generation_zero_precedes_model_failure_and_forbids_rerun(tmp_path):
    output = tmp_path / "v0.json"
    provenance = {"stage": "pre-model-static-test", "optimization_evidence": False}
    expected = runner._publish_pre_model_checkpoint(
        output, provenance, runner.FROZEN_EXTENSIONS
    )
    assert len(expected) == 15
    pointer = json.loads((tmp_path / "v0.partial.latest.json").read_text())
    assert pointer["generation"] == 0
    checkpoint = json.loads(Path(pointer["json_path"]).read_text())
    assert checkpoint["checkpoint_stage"] == "before_model_or_task_rng"
    assert checkpoint["completed_identities"] == []
    assert checkpoint["attempted_identities"] == []
    assert checkpoint["incomplete"] is True
    assert not output.exists()
    with pytest.raises(RuntimeError, match="resume, overwrite, or rerun"):
        runner._no_existing_artifacts(output)
    source = inspect.getsource(runner._production_pipeline)
    assert source.index("_publish_pre_model_checkpoint") < source.index(
        "model = load_model"
    )
    assert source.index('stage="model_loaded"') < source.index(
        "draws = generate_v0_draws"
    )
    assert source.index('stage="frozen_draws_generated"') < source.index(
        "pin_fk = np.stack"
    )
    assert source.index('stage="pin_preflight_captured"') < source.index(
        "def task_factory"
    )


def test_worker_boundary_rejects_fabricated_or_cross_module_outputs(tmp_path):
    summary = tmp_path / "worker.json"
    archive = tmp_path / "worker.npz"
    request_path = tmp_path / "request.json"
    input_path = tmp_path / "input.npz"
    extension = tmp_path / "extension.so"
    extension.write_bytes(b"static fake extension bytes")
    np.savez_compressed(input_path, input=np.zeros(1))
    worker_arrays = {
        name: np.zeros(1) for name in worker.EXPECTED_WORKER_ARRAY_NAMES
    }
    np.savez_compressed(archive, **worker_arrays)
    input_hash = runner._sha256_file(input_path)
    spec = {
        "module_name": "synthetic.mod",
        "extension_sha256": runner._sha256_file(extension),
        "extension_path": str(extension),
        "reference_size": 10,
    }
    request = {
        "module_name": spec["module_name"],
        "extension_path": spec["extension_path"],
        "extension_sha256": spec["extension_sha256"],
        "input_npz": str(input_path),
        "input_npz_sha256": input_hash,
        "reference_size": 10,
        "protocol_version": worker.WORKER_PROTOCOL_VERSION,
    }
    request_path.write_text(json.dumps(request))
    payload = {
        "worker_protocol_version": worker.WORKER_PROTOCOL_VERSION,
        "reference_size": 10,
        "module_name": spec["module_name"],
        "module_file": str(extension.resolve()),
        "extension_sha256": spec["extension_sha256"],
        "request_path": str(request_path.resolve()),
        "request_sha256": runner._sha256_file(request_path),
        "input_npz_path": str(input_path.resolve()),
        "input_npz_sha256": input_hash,
        "worker_npz_sha256": runner._sha256_file(archive),
        "reference_smoke": {"all_reference_smoke_gates_pass": True},
        "optimization_evidence": False,
        "sqp_optimization_calls": 0,
        "array_hashes": {
            name: runner._array_hash(value) for name, value in worker_arrays.items()
        },
        "array_names": sorted(worker_arrays),
    }
    summary.write_text(json.dumps(payload))
    row = {
        "exit_code": 0,
        "module_name": spec["module_name"],
        "extension_sha256": spec["extension_sha256"],
        "input_npz_sha256": input_hash,
        "input_npz_path": str(input_path.resolve()),
        "request_path": str(request_path.resolve()),
        "request_sha256": runner._sha256_file(request_path),
        "summary_path": str(summary),
        "summary_sha256": runner._sha256_file(summary),
        "npz_path": str(archive),
        "npz_sha256": runner._sha256_file(archive),
        "command": [
            runner.sys.executable,
            "-B",
            "-m",
            "gato_tiago.multimodal_toll_v0_worker",
            "--request",
            str(request_path),
            "--output",
            str(summary),
            "--execute",
        ],
    }
    assert runner._validate_worker_boundary(row, spec, input_hash)
    row["input_npz_sha256"] = "c" * 64
    assert not runner._validate_worker_boundary(row, spec, input_hash)
    row["input_npz_sha256"] = input_hash
    payload["worker_protocol_version"] = "wrong-protocol"
    summary.write_text(json.dumps(payload))
    row["summary_sha256"] = runner._sha256_file(summary)
    assert not runner._validate_worker_boundary(row, spec, input_hash)
    payload["worker_protocol_version"] = worker.WORKER_PROTOCOL_VERSION
    payload["array_names"] = payload["array_names"][:-1]
    summary.write_text(json.dumps(payload))
    row["summary_sha256"] = runner._sha256_file(summary)
    assert not runner._validate_worker_boundary(row, spec, input_hash)
    payload["array_names"] = sorted(worker_arrays)
    summary.write_text(json.dumps(payload))
    row["summary_sha256"] = runner._sha256_file(summary)
    request["unexpected"] = True
    request_path.write_text(json.dumps(request))
    row["request_sha256"] = runner._sha256_file(request_path)
    payload["request_sha256"] = row["request_sha256"]
    summary.write_text(json.dumps(payload))
    row["summary_sha256"] = runner._sha256_file(summary)
    assert not runner._validate_worker_boundary(row, spec, input_hash)


def test_static_runner_never_opens_frozen_task_rng(monkeypatch):
    frozen = {seed for _, seed in runner.EXPECTED_TASK_IDENTITIES}
    original = np.random.default_rng

    def guarded(seed=None):
        assert seed not in frozen
        return original(seed)

    monkeypatch.setattr(np.random, "default_rng", guarded)
    # Static metadata and synthetic ledgers are inert; production is blocked.
    assert len(runner.EXPECTED_TASK_IDENTITIES) == 12
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_v0_runner("unused.json")


def test_runner_provenance_requires_clean_stable_head_and_three_processes():
    sources = {
        name: {"path": path, "sha256": "a" * 64}
        for name, path in runner.REQUIRED_SOURCE_PATHS.items()
    }
    provenance = {
        "tracked_tree_clean_at_start": True,
        "tracked_tree_clean_at_end": True,
        "git_head_at_start": "b" * 40,
        "git_head_at_end": "b" * 40,
        "orig_argv": ["python", "-B", "-m", "runner", "--execute", "--output", "evidence.json"],
        "exact_command": "python -B -m runner --execute --output evidence.json",
        "cwd": "/workspace/GATO",
        "critical_environment": {"PYTHONPATH": "tiago_src:python", "CUDA_VISIBLE_DEVICES": None},
        "model_sha256": "c" * 64,
        "extension_hashes": {name: "d" * 64 for name in runner.SUPPORTED_MODULES},
        "source_hashes": sources,
        "retry_count": 0,
        "resume_count": 0,
    }
    rows = [
        {
            "module_name": name,
            "command": ["python", "-m", "worker", name],
            "exit_code": 0,
        }
        for name in runner.SUPPORTED_MODULES
    ]
    assert runner.certify_runner_provenance(provenance, rows)[
        "all_runner_provenance_gates_pass"
    ]
    for key, bad in (
        ("tracked_tree_clean_at_end", False),
        ("git_head_at_end", "e" * 40),
        ("resume_count", 1),
    ):
        mutated = dict(provenance)
        mutated[key] = bad
        assert not runner.certify_runner_provenance(mutated, rows)[
            "all_runner_provenance_gates_pass"
        ]
    duplicate = [dict(row) for row in rows]
    duplicate[2]["command"] = duplicate[1]["command"]
    assert not runner.certify_runner_provenance(provenance, duplicate)[
        "all_runner_provenance_gates_pass"
    ]


def test_exact_invocation_uses_orig_argv_not_lossy_sys_argv(monkeypatch, tmp_path):
    original = ["python", "-B", "-m", "gato_tiago.runner", "--output", "a path/v0.json"]
    monkeypatch.setattr(runner.sys, "orig_argv", original)
    monkeypatch.setattr(runner.sys, "argv", ["runner.py", "--output", "a path/v0.json"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "tiago_src:python")
    result = runner.exact_invocation_provenance()
    assert result["orig_argv"] == original
    assert result["exact_command"] == runner.shlex.join(original)
    assert "-m gato_tiago.runner" in result["exact_command"]
    assert result["cwd"] == str(tmp_path.resolve())
    assert result["critical_environment"]["PYTHONPATH"] == "tiago_src:python"


def _dense_random_fixture():
    draws = runner.generate_v0_draws(
        np.full(7, -2.0),
        np.full(7, 2.0),
        np.arange(1.0, 8.0),
        np.zeros(7),
    )
    states = np.repeat(draws.dynamics_x_float32[:, None, :], 65, axis=1)
    tools = np.zeros((32, 65, 3), dtype=np.float32)
    time = np.arange(65, dtype=np.float64) * (runner.toll.DT / 64)
    controls = np.repeat(draws.dynamics_u_float32[:, None, :], 64, axis=1)
    rows = [
        {
            "cuda_dense_states_float32": states.copy(),
            "cuda_dense_tool_positions_float32": tools.copy(),
            "dense_time_float64": time.copy(),
            "dense_controls_float32": controls.copy(),
        }
        for _ in range(3)
    ]
    return draws, states.astype(np.float64), tools.astype(np.float64), rows


def test_dense_random_capture_reports_state_l2_but_gates_tool_and_identity():
    draws, pin_states, pin_tools, rows = _dense_random_fixture()
    pin_states[:, 1:, 0] += 0.25
    result = runner.certify_dense_random_capture(
        draws, pin_states, pin_tools, rows
    )
    assert result["all_dense_random_capture_gates_pass"]
    assert result["max_state_l2_report_only"] == pytest.approx(0.25)
    assert result["state_l2_is_acceptance_gate"] is False
    assert result["diagnostic_scope"] == "one_interval_65_samples"
    assert result["future_witness_scope"] == "full_horizon_4033_samples"
    assert set(result["retained_array_hashes"]) == {
        "pin_states_float64",
        "cuda_states_float32",
        "pin_tool_float64",
        "cuda_tool_float32",
        "time_float64",
        "controls_float32",
    }


@pytest.mark.parametrize("mutation", ["time", "control", "initial", "cross", "dtype", "tool"])
def test_dense_random_capture_rejects_boundary_mutations(mutation):
    draws, pin_states, pin_tools, rows = _dense_random_fixture()
    if mutation == "time":
        rows[0]["dense_time_float64"][1] += 1e-9
    elif mutation == "control":
        rows[0]["dense_controls_float32"][0, 0, 0] += 1e-3
    elif mutation == "initial":
        rows[0]["cuda_dense_states_float32"][0, 0, 0] += 1e-3
    elif mutation == "cross":
        rows[1]["cuda_dense_states_float32"][0, 1, 0] += 1e-3
    elif mutation == "dtype":
        rows[0]["cuda_dense_states_float32"] = rows[0][
            "cuda_dense_states_float32"
        ].astype(np.float64)
    else:
        pin_tools[0, 1, 0] += 0.002
    assert not runner.certify_dense_random_capture(
        draws, pin_states, pin_tools, rows
    )["all_dense_random_capture_gates_pass"]
