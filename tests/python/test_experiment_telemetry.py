"""CPU checks use fake native replies, never instantiate a native solver."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bsqp.interface import BSQP, _decode_experiment_telemetry


def native_reply(codes=((1, 2), (3, 4)), counts=((0, 0), (100, 100)), sqp_code=1):
    codes = np.asarray(codes, dtype=np.uint32)
    counts = np.asarray(counts, dtype=np.int32)
    batch = codes.shape[1]
    rows = codes.shape[0]
    return {
        "XU": np.zeros((batch, 3), dtype=np.float32),
        "sqp_time_us": 42,
        "sqp_iters": np.full(batch, rows, dtype=np.int32),
        "kkt_converged": np.arange(batch, dtype=np.int32) % 2,
        "final_merit": np.zeros(batch, dtype=np.float32),
        "pcg_iters": counts,
        "pcg_times_us": np.zeros(rows, dtype=np.float32),
        "ls_num_iters": max(rows - 1, 0),
        "ls_min_merit": np.zeros((max(rows - 1, 0), batch), dtype=np.float32),
        "ls_step_size": np.zeros((max(rows - 1, 0), batch), dtype=np.float32),
        "experiment_telemetry": {
            "schema_version": 1,
            "pcg_stop_reason_codes": codes,
            "sqp_stop_reason_code": sqp_code,
            "breakdown_stopping_supported": False,
            "breakdown": None,
            "breakdown_unavailable_reason": "No explicit breakdown stop exists in this solver; numerical anomalies are not instrumented.",
        },
    }


def wrapper(reply):
    controller = BSQP.__new__(BSQP)
    controller.batch_size = reply["XU"].shape[0]
    controller.nx = 1
    controller.dt = .03
    controller.XU_B = reply["XU"].copy()
    controller.stats = {}
    controller.solver = SimpleNamespace(solve=lambda *args: reply)
    controller.solve(np.zeros((controller.batch_size, 1)), np.zeros((controller.batch_size, 6)))
    return controller


def test_full_native_pcg_rows_survive_final_sqp_ratio_exit():
    reply = native_reply()
    stats = wrapper(reply).get_stats()
    assert stats["ls_num_iters"] == 1
    np.testing.assert_array_equal(stats["pcg_iters"], [[0, 0], [100, 100]])
    np.testing.assert_array_equal(stats["pcg_times_us"], [0., 0.])
    telemetry = stats["experiment_telemetry"]
    assert telemetry["pcg_stop_reasons"] == [
        ["kkt_mask_skip", "initial_absolute_rho"],
        ["absolute_relative_rho", "iteration_cap"],
    ]
    # Equal cap counts have different actual exits; do not reconstruct from counts.
    assert telemetry["sqp_stop_reason"] == "kkt_ratio"
    np.testing.assert_array_equal(stats["kkt_converged"], [0, 1])
    assert telemetry["breakdown_stopping_supported"] is False
    assert telemetry["breakdown"] is None


def test_zero_pcg_cap_retains_initial_exit_vs_cap_distinction():
    decoded = _decode_experiment_telemetry(native_reply(((2, 4),), ((0, 0),)), 2)
    assert decoded["pcg_stop_reasons"] == [["initial_absolute_rho", "iteration_cap"]]


def test_zero_sqp_cap_has_empty_pcg_matrix_and_actual_cap_reason():
    reply = native_reply(np.empty((0, 2), dtype=np.uint32), np.empty((0, 2), dtype=np.int32), 2)
    stats = wrapper(reply).get_stats()
    assert stats["experiment_telemetry"]["pcg_stop_reason_codes"].shape == (0, 2)
    assert stats["experiment_telemetry"]["pcg_stop_reasons"] == []
    assert stats["experiment_telemetry"]["sqp_stop_reason"] == "iteration_cap"


def test_ordinary_or_old_native_module_keeps_original_stats_api():
    reply = native_reply()
    del reply["experiment_telemetry"]
    assert _decode_experiment_telemetry(reply, 2) is None
    stats = wrapper(reply).get_stats()
    assert "experiment_telemetry" not in stats
    np.testing.assert_array_equal(stats["pcg_iters"], reply["pcg_iters"])


@pytest.mark.parametrize("field,value,message", [
    ("schema_version", 99, "schema"),
    ("pcg_stop_reason_codes", np.array([[1, 2]], dtype=np.int32), "uint32"),
    ("pcg_stop_reason_codes", np.array([[1, 2]], dtype=np.uint32), "every native PCG row"),
    ("pcg_stop_reason_codes", np.array([[1, 2], [3, 99]], dtype=np.uint32), "Unknown PCG"),
    ("sqp_stop_reason_code", 99, "Unknown SQP"),
])
def test_invalid_telemetry_is_rejected_without_silent_row_loss(field, value, message):
    reply = native_reply()
    reply["experiment_telemetry"][field] = value
    with pytest.raises(ValueError, match=message):
        _decode_experiment_telemetry(reply, 2)


def test_source_scope_off_projection_and_on_sync_copy_kernel_counts():
    path = Path(__file__).resolve().parents[2] / "experiments/frequency_campaign/telemetry/audit_scope.py"
    spec = importlib.util.spec_from_file_location("experiment_telemetry_scope", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.audit()
    assert report["cmake_default_off"]
    assert report["device_bytes_delta_B16"] == 64
    assert all(row["off_tokens_equal_baseline"] for row in report["files"].values())

@pytest.fixture(scope="module")
def comparison_driver():
    path = Path(__file__).resolve().parents[2] / "experiments/frequency_campaign/telemetry/compare.py"
    spec = importlib.util.spec_from_file_location("experiment_telemetry_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_bank_and_schedule_lock_seventy_calls(comparison_driver):
    d = comparison_driver
    protocol = d.freeze_check()
    queue = d.schedule()
    assert len(queue) == 24
    assert len({row["id"] for row in queue}) == 24
    assert 2 * protocol["direct_PCG"]["calls_per_variant"] + sum(row["max_native_calls"] for row in queue) == 70
    assert [(row["variant"], row["repeat"]) for row in queue[:6]] == list(d.ORDER)
    for kind in ("cold", "warm", "unrelated"):
        b1 = d.tiled_inputs(kind, 1)
        b16 = d.tiled_inputs(kind, 16)
        for field in b1:
            assert b16[field].dtype == np.float32
            np.testing.assert_array_equal(b16[field], np.repeat(b1[field], 16, axis=0))
        np.testing.assert_array_equal(b1["warm_start"][:, :14], b1["state"])


def test_cpu_only_default_does_not_submit_or_import_native(comparison_driver, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU dry-run attempted a subprocess/native module import")
    monkeypatch.setattr(comparison_driver.subprocess, "run", forbidden)
    monkeypatch.setattr(comparison_driver.importlib.util, "spec_from_file_location", forbidden)
    assert comparison_driver.main([]) == 0
    assert '"CPU_PREPARATION_ONLY"' in capsys.readouterr().out


def test_raw_payload_round_trip_preserves_all_bytes_and_strict_json(comparison_driver, tmp_path):
    d = comparison_driver
    value = {"arrays": [np.array([np.nan, np.inf, -np.inf], np.float32), np.array([[1, 2, 3, 4]], np.uint32)],
             "scalar": float("inf"), "breakdown": None, "flag": False}
    d.save_payload(tmp_path, "raw", value)
    restored = d.load_payload(tmp_path, "raw")
    assert restored["arrays"][0].tobytes() == value["arrays"][0].tobytes()
    assert restored["arrays"][1].tobytes() == value["arrays"][1].tobytes()
    assert np.isposinf(restored["scalar"])
    assert restored["breakdown"] is None and restored["flag"] is False
    assert "Infinity" not in (tmp_path / "raw.json").read_text()
    assert "NaN" not in (tmp_path / "raw.json").read_text()
    with (tmp_path / "raw.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        d.load_payload(tmp_path, "raw")


def _fake_receipt(d, root, variant):
    import shutil
    build = root / variant
    source = build / "source"
    source.mkdir(parents=True)
    hashes = {}
    # Previously omitted compiled solver/dynamics headers and Python package
    # files exercise coverage beyond the old narrow critical list.
    extras = ("gato/bsqp/kernels/line_search.cuh", "gato/dynamics/integrator.cuh",
              "gato/dynamics/tiago_right/tiago_right_grid.cuh",
              "python/bsqp/offline_mpc.py", "python/bsqp/common.py")
    for relative in (*d.SOURCE_FILES, *extras):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(d.REPO / relative, target)
        hashes[relative] = d.sha(target)
    flags = {"CMAKE_BUILD_TYPE": "Release", "CMAKE_CUDA_ARCHITECTURES": "61-real",
             "N": 32, "plant": "tiago_right", "compiler": "CPU-only-matched",
             "cuda_toolkit": "CPU-only-matched", "cuda_flags": "CPU-only", "cxx_flags": "CPU-only",
             "resource_reporting_flags": "CPU-only"}
    patch = root / "reviewed-candidate.patch"
    patch.write_text("CPU-only synthetic archived patch artifact; no solver build\n")
    external = root / "synthetic-toolkit-header.h"
    external.write_text("CPU-only header receipt\n")
    authority = root / "reviewed-source-manifest.json"
    d.strict_json(authority, dict(schema_version=2, base_commit="6"*40,
        candidate_diff={"path": str(patch), "sha256": d.sha(patch)},
        source_file_sha256=hashes, build_flags=flags,
        external_dependency_sha256={str(external): d.sha(external)}))
    artifacts = {}
    for key in ("module", "direct_executable", "cmake_cache", "direct_cmake_cache", "compiler_resource_log", "compiled_dependencies"):
        target = build / ("bsqpN32_tiago_right.cpu_fixture.so" if key == "module" else key)
        if "cmake_cache" in key:
            target.write_text(f"GATO_EXPERIMENT_TELEMETRY:BOOL={variant}\nCMAKE_BUILD_TYPE:STRING=Release\nCMAKE_CUDA_ARCHITECTURES:STRING=61-real\n"
                              + ("KNOTS:STRING=32\nPLANT:STRING=tiago_right\n" if key == "cmake_cache" else ""))
        elif key == "compiled_dependencies":
            depfile = build / "synthetic-compiler.d"
            depfile.write_text("CPU-only compiler-dependency fixture\n")
            d.strict_json(target, dict(local_source_files=list(hashes), external_dependency_sha256={str(external):d.sha(external)},
                                      depfile_sha256={str(depfile):d.sha(depfile)}))
        else:
            target.write_text("CPU fixture artifact; never executable or imported\n")
        artifacts[key] = {"path": str(target), "sha256": d.sha(target)}
    receipt = dict(schema_version=2, variant=variant, build_root=str(build), source_root=str(source),
                   base_commit="6"*40, candidate_diff_sha256=d.sha(patch), source_file_sha256=hashes,
                   source_manifest_path=str(authority), source_manifest_sha256=d.sha(authority),
                   flags=dict(flags, GATO_EXPERIMENT_TELEMETRY=variant == "ON"), **artifacts)
    path = build / "receipt.json"
    d.strict_json(path, receipt)
    return path


def _validate(d, path, variant):
    r = d.read_json(path)
    return d.validate_receipt(path, variant, r['source_manifest_path'], r['source_manifest_sha256'])


def _pair(d, off, on):
    r = d.read_json(off)
    return d.validate_pair(off, on, r['source_manifest_path'], r['source_manifest_sha256'])


def test_receipts_verify_separate_build_owned_bytes_and_flags(comparison_driver, tmp_path):
    d = comparison_driver
    off, on = [_fake_receipt(d, tmp_path, v) for v in ("OFF", "ON")]
    assert set(_pair(d, off, on)) == {"OFF", "ON"}
    raw = d.read_json(on)
    raw["flags"]["compiler"] = "different"
    d.strict_json(on, raw)
    with pytest.raises(ValueError, match="expected flags"):
        _pair(d, off, on)


def test_receipt_rejects_shared_main_source_and_false_cache_option(comparison_driver, tmp_path):
    d = comparison_driver
    path = _fake_receipt(d, tmp_path, "ON")
    receipt = d.read_json(path)
    receipt["source_root"] = str(d.REPO)
    d.strict_json(path, receipt)
    with pytest.raises(ValueError, match="build-owned"):
        _validate(d, path, "ON")
    receipt["source_root"] = str(tmp_path / "ON/source")
    cache = Path(receipt["cmake_cache"]["path"])
    cache.write_text(cache.read_text().replace("BOOL=ON", "BOOL=OFF"))
    receipt["cmake_cache"]["sha256"] = d.sha(cache)
    d.strict_json(path, receipt)
    with pytest.raises(ValueError, match="option"):
        _validate(d, path, "ON")


def _direct_saved_data(d, folder, variant, protocol):
    folder.mkdir()
    d.strict_json(folder / "configuration.json", {"N": 32, "batch_size": 1, "state_size": 14, "vector_size_padded": 476,
                   "stats_words": 2 if variant == "ON" else 1, "telemetry_enabled": variant == "ON"})
    for fixture in protocol["direct_PCG"]["fixtures"]:
        row = folder / fixture["id"]
        row.mkdir()
        matrix = np.zeros((32, 14, 42), np.float32)
        matrix[:, np.arange(14), 14 + np.arange(14)] = 1
        matrix.tofile(row / "M_inv.f32")
        if fixture["id"] == "one-iteration-cap-nonconverged":
            matrix[0, 1, 15] = 2
        matrix.tofile(row / "A.f32")
        b = np.zeros(476, np.float32)
        b[14:16] = fixture["b_first_components"]
        b.tofile(row / "b.f32")
        np.zeros(476, np.float32).tofile(row / "initial_x.f32")
        x = np.zeros(476, np.float32)
        if fixture["max_pcg_iters"]:
            x[14:16] = [1, 1] if fixture["id"] == "tolerance-exactly-at-cap" else [2 / 3, 2 / 3]
        x.tofile(row / "output_x.f32")
        values = [fixture["expected_iterations"]]
        if variant == "ON":
            values.append(fixture["expected_reason_code"])
        np.asarray(values, np.uint32).tofile(row / "stats.u32")
        d.strict_json(row / "request.json", {"status": "started", "max_pcg_iters": fixture["max_pcg_iters"], "epsilon": 0, "kkt_mask": fixture["kkt_mask"]})
        d.strict_json(row / "kernel_submission.json", {"status": "pending kernel launch; process death leaves completion unknown"})
        d.strict_json(row / "completion.json", {"status": "completed", "iterations": fixture["expected_iterations"],
                      "stop_reason_code": fixture["expected_reason_code"] if variant == "ON" else None,
                      "breakdown_stopping_supported": False, "breakdown": None})


def test_direct_finite_spd_saved_layout_and_last_cap_tolerance(comparison_driver, tmp_path):
    d = comparison_driver
    protocol = d.freeze_check()
    folder = tmp_path / "ON"
    _direct_saved_data(d, folder, "ON", protocol)
    outputs = d.check_direct(folder, "ON", protocol)
    np.testing.assert_array_equal(outputs["tolerance-exactly-at-cap"][14:16], [1, 1])
    # Both count1 paths have distinct actual reasons. Count0 has three paths.
    cap = folder / "one-iteration-cap-nonconverged/stats.u32"
    np.array([1, 3], np.uint32).tofile(cap)
    with pytest.raises(ValueError, match="stop path"):
        d.check_direct(folder, "ON", protocol)


def test_native_branch_validation_distinguishes_last_cap_and_batch_gate(comparison_driver):
    d = comparison_driver
    protocol = d.freeze_check()
    reply = native_reply(((3, 4),), ((120, 120),), sqp_code=1)
    reply["kkt_converged"] = np.array([1, 1], np.int32)
    d.check_native_reply(reply, wrapper(reply).get_stats(), "ON", 2, protocol)
    reply["kkt_converged"] = np.array([1, 0], np.int32)
    with pytest.raises(ValueError, match="Batch KKT-ratio"):
        d.check_native_reply(reply, wrapper(reply).get_stats(), "ON", 2, protocol)


def _fake_processes(d, protocol, zero_call_worker=False, fail_direct=False):
    from subprocess import CompletedProcess
    def run(command, stdout, stderr, check, env=None):
        assert check is False
        if "--worker" in command:
            request_path = Path(command[-1])
            request = d.read_json(request_path)  # request must precede submission
            folder = Path(request["output"])
            count = 0 if zero_call_worker else request["max_native_calls"]
            d.strict_json(folder / "worker_status.json", {"status": "completed", "native_calls_started": count,
                         "native_calls_completed": count, "call_submission": "returned"})
            batch = request["batch_size"]
            reply = native_reply(np.full((1, batch), 2), np.zeros((1, batch), np.int32), sqp_code=1)
            reply["kkt_converged"] = np.ones(batch, np.int32)
            reply["initial_merit"] = np.ones(batch, np.float32)
            if request["variant"] == "OFF":
                del reply["experiment_telemetry"]
            d.save_payload(folder, "measured_output", {"native_reply": reply, "public_stats": wrapper(reply).get_stats(), "wrapper_wall_s": .001})
        else:
            output = Path(command[-1])
            variant = output.name.split("-")[-1]
            assert (output.parent / (output.name + "-request.json")).exists()
            if fail_direct:
                return CompletedProcess(command, 7)
            _direct_saved_data(d, output, variant, protocol)
        return CompletedProcess(command, 0)
    return run


def test_orchestration_failure_preserves_attempt_and_stops_without_retry(comparison_driver, tmp_path, monkeypatch):
    d = comparison_driver
    protocol = d.freeze_check()
    receipts = {v: {"direct_executable": {"path": v}, "receipt_path": v, "receipt_sha256": v, "source_manifest_path": v, "source_manifest_sha256": v} for v in ("OFF", "ON")}
    calls = []
    fake = _fake_processes(d, protocol, fail_direct=True)
    def run(*args, **kwargs):
        calls.append(args[0])
        return fake(*args, **kwargs)
    monkeypatch.setattr(d.subprocess, "run", run)
    output = tmp_path / "attempt"
    assert d.execute(output, receipts, protocol) == 1
    assert len(calls) == 1
    assert d.read_json(output / "status.json")["status"] == "failed_validation_held"
    assert (output / "direct-OFF-request.json").exists()
    assert not (output / "direct-ON-request.json").exists()
    assert d.read_json(output / "call_inventory.json")["completed_native_calls_known"] == 0
    assert (output / "SHA256SUMS.json").is_file()


def test_zero_native_call_success_receipt_is_rejected(comparison_driver, tmp_path, monkeypatch):
    d = comparison_driver
    protocol = d.freeze_check()
    receipts = {v: {"direct_executable": {"path": v}, "receipt_path": v, "receipt_sha256": v, "source_manifest_path": v, "source_manifest_sha256": v} for v in ("OFF", "ON")}
    monkeypatch.setattr(d.subprocess, "run", _fake_processes(d, protocol, zero_call_worker=True))
    output = tmp_path / "attempt"
    assert d.execute(output, receipts, protocol) == 1
    assert d.read_json(output / "status.json")["workers_completed"] == 0
    assert "zero-native-call" in d.read_json(output / "status.json")["error"]
    assert len(list(output.glob("B*"))) == 1


def test_frozen_successful_orchestration_uses_24_isolated_workers_and_70_calls(comparison_driver, tmp_path, monkeypatch):
    d = comparison_driver
    protocol = d.freeze_check()
    receipts = {v: {"direct_executable": {"path": v}, "receipt_path": v, "receipt_sha256": v, "source_manifest_path": v, "source_manifest_sha256": v} for v in ("OFF", "ON")}
    commands = []
    fake = _fake_processes(d, protocol)
    def run(*args, **kwargs):
        commands.append(args[0])
        return fake(*args, **kwargs)
    monkeypatch.setattr(d.subprocess, "run", run)
    output = tmp_path / "attempt"
    assert d.execute(output, receipts, protocol) == 0
    assert len(commands) == 26
    assert sum("--worker" in command for command in commands) == 24
    assert d.read_json(output / "status.json")["actual_native_calls"] == 70
    assert d.read_json(output / "call_inventory.json")["completed_native_calls_known"] == 70
    assert d.read_json(output / "summary.json")["status"] == "passed_bounded_comparison"
    for relative, expected in d.read_json(output / "SHA256SUMS.json").items():
        assert d.sha(output / relative) == expected

@pytest.mark.parametrize("kind,expected_calls", [("cold", 2), ("warm", 3)])
def test_worker_uses_frozen_warmup_optional_prefix_and_measured_input_only(comparison_driver, tmp_path, monkeypatch, kind, expected_calls):
    import sys
    d = comparison_driver
    receipt_path = _fake_receipt(d, tmp_path, "ON")
    receipt = d.read_json(receipt_path)
    native_inputs = []

    class FakeNative:
        def solve(self, xu, dt, state, reference):
            native_inputs.append({"state": state.copy(), "reference": reference.copy(), "warm_start": xu.copy()})
            batch = state.shape[0]
            reply = native_reply(np.full((1, batch), 2), np.zeros((1, batch), np.int32), 1)
            reply["kkt_converged"] = np.ones(batch, np.int32)
            reply["XU"] = xu.copy()
            reply["initial_merit"] = np.ones(batch, np.float32)
            return reply

    class FakePublic:
        solve = BSQP.solve
        get_stats = BSQP.get_stats
        def __init__(self, *, batch_size, dt, **kwargs):
            self.batch_size = batch_size
            self.nx = 14
            self.dt = dt
            self.XU_B = np.zeros((batch_size, 665), np.float32)
            self.stats = {}
            self.solver = FakeNative()

    monkeypatch.setitem(sys.modules, "bsqp.interface", SimpleNamespace(BSQP=FakePublic, __file__=str(Path(receipt["source_root"]) / "python/bsqp/interface.py")))
    monkeypatch.setattr(d.importlib.util, "spec_from_file_location", lambda *args: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None)))
    monkeypatch.setattr(d.importlib.util, "module_from_spec", lambda spec: SimpleNamespace())
    output = tmp_path / "worker"
    output.mkdir()
    request = {"id": f"test-{kind}", "variant": "ON", "batch_size": 16, "input": kind, "max_native_calls": expected_calls,
               "output": str(output), "receipt_path": str(receipt_path), "receipt_sha256": d.sha(receipt_path), "source_manifest_path": receipt["source_manifest_path"], "source_manifest_sha256": receipt["source_manifest_sha256"]}
    d.strict_json(output / "request.json", request)
    assert d.worker(output / "request.json") == 0
    assert len(native_inputs) == expected_calls
    for actual, label in zip(native_inputs, ["unrelated", "cold", "warm"] if kind == "warm" else ["unrelated", "cold"]):
        for field, value in d.tiled_inputs(label, 16).items():
            np.testing.assert_array_equal(actual[field], value)
    status = d.read_json(output / "worker_status.json")
    assert status["native_calls_started"] == status["native_calls_completed"] == expected_calls
    assert (output / "measured_input.npz").is_file()
    assert (output / "cold_prefix_input.npz").exists() == (kind == "warm")
    raw = d.load_payload(output, "measured_output")
    assert raw["native_reply"]["pcg_iters"].shape == (1, 16)
    assert raw["native_reply"]["experiment_telemetry"]["breakdown"] is None


def test_worker_native_exception_keeps_input_and_failure_without_additional_calls(comparison_driver, tmp_path, monkeypatch):
    import sys
    d = comparison_driver
    receipt_path = _fake_receipt(d, tmp_path, "ON")
    receipt = d.read_json(receipt_path)
    calls = []
    class FailedNative:
        def solve(self, *args):
            calls.append(True)
            raise RuntimeError("CPU-only synthetic native exception")
    class FakePublic:
        solve = BSQP.solve
        get_stats = BSQP.get_stats
        def __init__(self, *, batch_size, dt, **kwargs):
            self.batch_size = batch_size
            self.nx = 14
            self.dt = dt
            self.XU_B = np.zeros((batch_size, 665), np.float32)
            self.stats = {}
            self.solver = FailedNative()
    monkeypatch.setitem(sys.modules, "bsqp.interface", SimpleNamespace(BSQP=FakePublic, __file__=str(Path(receipt["source_root"]) / "python/bsqp/interface.py")))
    monkeypatch.setattr(d.importlib.util, "spec_from_file_location", lambda *args: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None)))
    monkeypatch.setattr(d.importlib.util, "module_from_spec", lambda spec: SimpleNamespace())
    output = tmp_path / "worker"
    output.mkdir()
    d.strict_json(output / "request.json", {"id": "failed", "variant": "ON", "batch_size": 1, "input": "cold", "max_native_calls": 2,
                  "output": str(output), "receipt_path": str(receipt_path), "receipt_sha256": d.sha(receipt_path), "source_manifest_path": receipt["source_manifest_path"], "source_manifest_sha256": receipt["source_manifest_sha256"]})
    assert d.worker(output / "request.json") == 1
    assert len(calls) == 1
    status = d.read_json(output / "worker_status.json")
    assert status["status"] == "failed"
    assert status["native_calls_started"] == 1 and status["native_calls_completed"] == 0
    assert (output / "unrelated_warmup_input.npz").exists()
    assert not (output / "measured_input.npz").exists()
    assert not (output / "failed_call_return.npz").exists()

@pytest.mark.parametrize('relative', ['gato/bsqp/kernels/line_search.cuh',
                                     'gato/dynamics/integrator.cuh',
                                     'gato/dynamics/tiago_right/tiago_right_grid.cuh',
                                     'python/bsqp/common.py'])
def test_both_receipts_cannot_omit_or_substitute_dependencies(comparison_driver, tmp_path, relative):
    d = comparison_driver
    paths = [_fake_receipt(d, tmp_path, v) for v in ('OFF', 'ON')]
    original = d.read_json(paths[0])
    for path in paths:
        r = d.read_json(path)
        del r['source_file_sha256'][relative]
        d.strict_json(path, r)
    with pytest.raises(ValueError, match='complete authoritative'):
        d.validate_pair(*paths, original['source_manifest_path'], original['source_manifest_sha256'])
    # Replacing both receipt maps and the self-hashed manifest cannot substitute
    # a separately pinned review authority.
    manifest = Path(original['source_manifest_path'])
    forged = d.read_json(manifest)
    del forged['source_file_sha256'][relative]
    d.strict_json(manifest, forged)
    for path in paths:
        r = d.read_json(path)
        r['source_manifest_sha256'] = d.sha(manifest)
        d.strict_json(path, r)
    with pytest.raises(ValueError, match='manifest hash mismatch'):
        d.validate_pair(*paths, manifest, original['source_manifest_sha256'])


@pytest.mark.parametrize('key,bad', [
    ('GATO_EXPERIMENT_TELEMETRY', 'ON-malicious'), ('CMAKE_BUILD_TYPE', 'ReleaseDebug'),
    ('CMAKE_CUDA_ARCHITECTURES', '61-real;999-malicious'), ('KNOTS', '32;64'),
    ('PLANT', 'tiago_right;indy7'),
])
def test_cache_exact_values_reject_rehashed_substring_deceptions(comparison_driver, tmp_path, key, bad):
    d = comparison_driver
    path = _fake_receipt(d, tmp_path, 'ON')
    r = d.read_json(path)
    cache = Path(r['cmake_cache']['path'])
    lines = cache.read_text().splitlines()
    lines = [line.split('=', 1)[0]+'='+bad if line.startswith(key+':') else line for line in lines]
    # A comment containing the desired entry is never an actual cache field.
    cache.write_text('\n'.join(lines)+'\n# '+key+':STRING=Release\n')
    r['cmake_cache']['sha256'] = d.sha(cache)
    d.strict_json(path, r)
    with pytest.raises(ValueError, match='cache option'):
        _validate(d, path, 'ON')


@pytest.mark.parametrize('cache_key', ['cmake_cache', 'direct_cmake_cache'])
@pytest.mark.parametrize('mutation', ['duplicate', 'wrongtype', 'commentonly'])
def test_both_caches_reject_duplicate_wrong_type_and_comment_only_options(comparison_driver, tmp_path, cache_key, mutation):
    d = comparison_driver
    path = _fake_receipt(d, tmp_path, 'ON')
    r = d.read_json(path)
    cache = Path(r[cache_key]['path'])
    text = cache.read_text()
    if mutation == 'duplicate':
        text += 'GATO_EXPERIMENT_TELEMETRY:BOOL=OFF\n'
    elif mutation == 'wrongtype':
        text = text.replace('GATO_EXPERIMENT_TELEMETRY:BOOL=', 'GATO_EXPERIMENT_TELEMETRY:STRING=')
    else:
        text = text.replace('GATO_EXPERIMENT_TELEMETRY:BOOL=', '//GATO_EXPERIMENT_TELEMETRY:BOOL=')
    cache.write_text(text)
    r[cache_key]['sha256'] = d.sha(cache)
    d.strict_json(path, r)
    with pytest.raises(ValueError, match='Duplicate|cache option'):
        _validate(d, path, 'ON')


def test_base_commit_patch_artifact_and_module_horizon_are_truthful(comparison_driver, tmp_path):
    d = comparison_driver
    path = _fake_receipt(d, tmp_path, 'ON')
    original = d.read_json(path)
    wrong = dict(original, source_commit='1'*40)
    d.strict_json(path, wrong)
    with pytest.raises(ValueError, match='must not claim source_commit'):
        _validate(d, path, 'ON')
    wrong = dict(original, candidate_diff_sha256='0'*64)
    d.strict_json(path, wrong)
    with pytest.raises(ValueError, match='diff identity'):
        _validate(d, path, 'ON')
    wrong = dict(original)
    wrong['module'] = dict(original['module'])
    moved = Path(original['module']['path']).with_name('bsqpN64_tiago_right.cpu_fixture.so')
    Path(original['module']['path']).rename(moved)
    wrong['module']['path'] = str(moved)
    d.strict_json(path, wrong)
    with pytest.raises(ValueError, match='basename'):
        _validate(d, path, 'ON')
    Path(tmp_path/'reviewed-candidate.patch').write_text('substituted candidate patch')
    with pytest.raises(ValueError, match='diff artifact'):
        _validate(d, path, 'ON')


def test_only_exact_receipt_bound_module_output_allowed_in_source_inventory(comparison_driver, tmp_path):
    d = comparison_driver
    path = _fake_receipt(d, tmp_path, 'ON')
    r = d.read_json(path)
    source = Path(r['source_root'])
    module = source/'python/bsqp'/Path(r['module']['path']).name
    Path(r['module']['path']).rename(module)
    r['module']['path'] = str(module)
    d.strict_json(path, r)
    _validate(d, path, 'ON')
    extra = source/'python/bsqp/__pycache__'
    extra.mkdir()
    (extra/'unreviewed.pyc').write_bytes(b'extra')
    with pytest.raises(ValueError, match='inventory differs'):
        _validate(d, path, 'ON')


def _worker_request(d, receipt_path, output, kind='cold'):
    r = d.read_json(receipt_path)
    output.mkdir()
    request = dict(id='CPU-only-worker', variant='ON', batch_size=1, input=kind,
                   max_native_calls=3 if kind=='warm' else 2, output=str(output),
                   receipt_path=str(receipt_path), receipt_sha256=d.sha(receipt_path),
                   source_manifest_path=r['source_manifest_path'], source_manifest_sha256=r['source_manifest_sha256'])
    d.strict_json(output/'request.json', request)
    return output/'request.json'


@pytest.mark.parametrize('drift', ['gato/dynamics/integrator.cuh', 'gato/bsqp/kernels/line_search.cuh', 'python/bsqp/common.py'])
def test_complete_dependency_drift_fails_before_native_import(comparison_driver, tmp_path, monkeypatch, drift):
    d = comparison_driver
    receipt = _fake_receipt(d, tmp_path, 'ON')
    r = d.read_json(receipt)
    (Path(r['source_root'])/drift).write_text('CPU-only mutation of omitted-old-list dependency')
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid provenance reached native import')
    monkeypatch.setattr(d.importlib.util, 'spec_from_file_location', forbidden)
    output = tmp_path/'worker'
    request = _worker_request(d, receipt, output)
    assert d.worker(request) == 1
    status = d.read_json(output/'worker_status.json')
    assert status['native_calls_started'] == status['native_calls_completed'] == 0
    assert 'Source snapshot bytes changed' in status['error']


@pytest.mark.parametrize('failure', ['nonfinite-warmup', 'nonfinite-prefix', 'fresh-constructor'])
def test_all_warmup_prefix_returns_validate_and_stale_capture_is_released(comparison_driver, tmp_path, monkeypatch, failure):
    import sys
    import weakref
    d = comparison_driver
    receipt_path = _fake_receipt(d, tmp_path, 'ON')
    r = d.read_json(receipt_path)
    calls, constructed, refs = [], [], []
    class FakeNative:
        def solve(self, xu, dt, state, reference):
            calls.append(True)
            reply = native_reply(((2,),), ((0,),), 1)
            reply['kkt_converged'] = np.ones(1, np.int32)
            reply['XU'] = xu.copy()
            reply['initial_merit'] = np.ones(1, np.float32)
            if (failure=='nonfinite-warmup' and len(calls)==1) or (failure=='nonfinite-prefix' and len(calls)==2):
                reply['initial_merit'][0] = np.nan
            return reply
    class FakePublic:
        solve, get_stats = BSQP.solve, BSQP.get_stats
        def __init__(self, *, batch_size, dt, **kwargs):
            constructed.append(True)
            if len(constructed)==2:
                assert refs[0]() is None, 'Disposable warmup solver/native retained'
                if failure=='fresh-constructor':
                    raise RuntimeError('CPU-only fresh measured constructor failure')
            self.batch_size, self.nx, self.dt = batch_size, 14, dt
            self.XU_B, self.stats, self.solver = np.zeros((batch_size, 665), np.float32), {}, FakeNative()
            refs.append(weakref.ref(self))
    monkeypatch.setitem(sys.modules, 'bsqp.interface', SimpleNamespace(BSQP=FakePublic, __file__=str(Path(r['source_root'])/'python/bsqp/interface.py')))
    monkeypatch.setattr(d.importlib.util, 'spec_from_file_location', lambda *args: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None)))
    monkeypatch.setattr(d.importlib.util, 'module_from_spec', lambda spec: SimpleNamespace())
    output = tmp_path/'worker'
    request = _worker_request(d, receipt_path, output, kind='warm')
    assert d.worker(request)==1
    status = d.read_json(output/'worker_status.json')
    if failure=='fresh-constructor':
        assert len(calls)==1
        assert 'constructor failure' in status['error']
        assert not (output/'failed_call_return.npz').exists(), 'Stale warmup relabeled failed current call'
    else:
        assert len(calls)==(1 if failure=='nonfinite-warmup' else 2)
        assert 'Nonfinite returned initial_merit' in status['error']
        label = 'unrelated_warmup' if failure=='nonfinite-warmup' else 'cold_prefix'
        retained = d.load_payload(output, label+'_output')
        assert np.isnan(retained['native_reply']['initial_merit'][0])
        assert (output/'failed_call_return.npz').exists()
    assert not (output/'measured_input.npz').exists()
    assert status['native_calls_started']==status['native_calls_completed']==len(calls)


def test_escaping_source_links_and_changed_external_dependencies_fail(comparison_driver, tmp_path):
    d = comparison_driver
    receipt = _fake_receipt(d, tmp_path, 'ON')
    r = d.read_json(receipt)
    source = Path(r['source_root'])
    target = source/'gato/dynamics/integrator.cuh'
    external = tmp_path/'unreviewed-integrator.cuh'
    external.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(external)
    with pytest.raises(ValueError, match='Source snapshot bytes changed'):
        _validate(d, receipt, 'ON')
    target.unlink()
    target.write_bytes(external.read_bytes())
    (tmp_path/'synthetic-toolkit-header.h').write_text('changed SDK header after review')
    with pytest.raises(ValueError, match='external dependency bytes changed'):
        _validate(d, receipt, 'ON')


def test_real_orchestrated_worker_arguments_disable_bytecode(comparison_driver, tmp_path, monkeypatch):
    d = comparison_driver
    protocol = d.freeze_check()
    receipts = {v: {'direct_executable':{'path':v}, 'receipt_path':v, 'receipt_sha256':v,
                    'source_manifest_path':v, 'source_manifest_sha256':v} for v in ('OFF','ON')}
    fake = _fake_processes(d, protocol)
    worker_commands=[]
    def run(command, **kwargs):
        if '--worker' in command:
            worker_commands.append(command)
            assert command[1]=='-B'
            assert kwargs['env']['PYTHONDONTWRITEBYTECODE']=='1'
            request=d.read_json(Path(command[-1]))
            assert request['source_manifest_sha256']==request['variant']
        return fake(command, **kwargs)
    monkeypatch.setattr(d.subprocess, 'run', run)
    assert d.execute(tmp_path/'attempt', receipts, protocol)==0
    assert len(worker_commands)==24


def test_nested_build_roots_cannot_share_snapshot_ownership(comparison_driver, tmp_path):
    d = comparison_driver
    off, on = [_fake_receipt(d, tmp_path, v) for v in ('OFF', 'ON')]
    r = d.read_json(on)
    r['build_root'] = str(tmp_path)  # Contains OFF as well as its own ON artifacts.
    d.strict_json(on, r)
    with pytest.raises(ValueError, match='independent build/source roots'):
        _pair(d, off, on)


@pytest.mark.parametrize('stage', ['unrelated_warmup', 'cold_prefix'])
@pytest.mark.parametrize('field,bad_shape', [('XU', (1, 0)), ('XU', (1, 664)),
                                          ('initial_merit', (1, 1)), ('final_merit', (1, 1))])
def test_every_warmup_prefix_reply_requires_exact_shapes_after_raw_preservation(comparison_driver, tmp_path, monkeypatch, stage, field, bad_shape):
    import sys
    d = comparison_driver
    receipt = _fake_receipt(d, tmp_path, 'ON')
    r = d.read_json(receipt)
    calls = []
    malformed = np.zeros(bad_shape, dtype=np.float32)
    fail_call = 1 if stage=='unrelated_warmup' else 2
    class FakeNative:
        def solve(self, xu, dt, state, reference):
            calls.append(True)
            reply = native_reply(((2,),), ((0,),), 1)
            reply['XU'] = xu.copy()
            reply['kkt_converged'] = np.ones(1, np.int32)
            reply['initial_merit'] = np.ones(1, np.float32)
            if len(calls)==fail_call:
                reply[field] = malformed.copy()  # Finite, including empty: shape is the failure.
            return reply
    class FakePublic:
        solve, get_stats = BSQP.solve, BSQP.get_stats
        def __init__(self, *, batch_size, dt, **kwargs):
            self.batch_size, self.nx, self.dt = batch_size, 14, dt
            self.XU_B, self.stats, self.solver = np.zeros((batch_size,665),np.float32), {}, FakeNative()
    monkeypatch.setitem(sys.modules,'bsqp.interface',SimpleNamespace(BSQP=FakePublic,__file__=str(Path(r['source_root'])/'python/bsqp/interface.py')))
    monkeypatch.setattr(d.importlib.util,'spec_from_file_location',lambda *args: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module:None)))
    monkeypatch.setattr(d.importlib.util,'module_from_spec',lambda spec:SimpleNamespace())
    output=tmp_path/'worker'
    request=_worker_request(d,receipt,output,kind='warm')
    assert d.worker(request)==1
    assert len(calls)==fail_call
    status=d.read_json(output/'worker_status.json')
    assert status['status']=='failed'
    assert f'Malformed returned {field} shape in {stage}' in status['error']
    assert status['native_calls_started']==status['native_calls_completed']==fail_call
    for label in [stage+'_output','failed_call_return']:
        retained=d.load_payload(output,label)['native_reply']
        assert set(retained)==set(native_reply())|{'initial_merit'}
        assert retained[field].shape==bad_shape
        assert retained[field].tobytes()==malformed.tobytes()
        assert np.isfinite(retained[field]).all()
        assert 'experiment_telemetry' in retained
    assert (output/(stage+'_input.npz')).exists()
    assert not (output/'measured_input.npz').exists()
    assert not (output/'measured_output.npz').exists()
