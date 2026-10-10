"""Frozen CPU preparation and root-run 70-call maximum ON/OFF comparison.

The default command validates receipts and prints the queue without importing a
solver. --execute-reviewed is reserved for the reviewed root-controlled run.
Every production-module import happens in a fresh worker subprocess.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE_FILES = (
    "CMakeLists.txt", "gato/types.cuh", "gato/bsqp/bsqp.cuh",
    "gato/bsqp/kernels/pcg.cuh", "python/bindings.cu", "python/bsqp/interface.py",
    "gato/settings.h", "gato/constants.h", "gato/utils/linalg.cuh",
    "gato/dynamics/tiago_right/tiago_right_arm.urdf",
    *("experiments/frequency_campaign/telemetry/" + name for name in (
        "CMakeLists.txt", "direct_pcg.cu", "compare.py", "comparison_protocol.json",
        "comparison_inputs.npz", "comparison_input_receipt.json")),
)
ORDER = (("OFF", 1), ("ON", 1), ("ON", 2), ("OFF", 2), ("OFF", 3), ("ON", 3))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON token {value}")))


def save_payload(folder, name, value):
    """Preserve every array byte and exceptional scalar outside solve timing."""
    arrays = {}
    def pack(item, key):
        if isinstance(item, np.ndarray):
            arrays[key] = item.copy()
            return {"npz_key": key, "shape": list(item.shape), "dtype": str(item.dtype)}
        if isinstance(item, dict):
            return {str(k): pack(v, key + "__" + str(k)) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [pack(v, key + "__" + str(i)) for i, v in enumerate(item)]
        if isinstance(item, np.generic):
            return pack(item.item(), key)
        if isinstance(item, float) and not math.isfinite(item):
            arrays[key] = np.asarray(item)
            return {"npz_key": key, "dtype": str(arrays[key].dtype), "nonfinite_scalar": True}
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f"Unsupported raw value {type(item).__name__} at {key}")
    metadata = pack(value, "record")
    path = Path(folder) / (name + ".npz")
    with path.with_suffix(".npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix(".npz.tmp").replace(path)
    strict_json(Path(folder) / (name + ".json"), {"payload": metadata, "npz_sha256": sha(path)})


def load_payload(folder, name):
    metadata = read_json(Path(folder) / (name + ".json"))
    target = Path(folder) / (name + ".npz")
    if sha(target) != metadata["npz_sha256"]:
        raise ValueError("Raw payload hash mismatch")
    with np.load(target, allow_pickle=False) as data:
        def unpack(item):
            if isinstance(item, dict) and "npz_key" in item:
                return data[item["npz_key"]].copy()
            if isinstance(item, dict):
                return {k: unpack(v) for k, v in item.items()}
            if isinstance(item, list):
                return [unpack(v) for v in item]
            return item
        return unpack(metadata["payload"])


def manifest(folder):
    files = sorted(path for path in Path(folder).rglob("*") if path.is_file() and path != Path(folder) / "SHA256SUMS.json")
    strict_json(Path(folder) / "SHA256SUMS.json", {str(path.relative_to(folder)): sha(path) for path in files})


def freeze_check():
    protocol = read_json(HERE / "comparison_protocol.json")
    receipt = read_json(HERE / "comparison_input_receipt.json")
    if sha(HERE / "comparison_inputs.npz") != receipt["comparison_inputs_sha256"]:
        raise ValueError("Frozen input bank bytes changed")
    for relative, expected in protocol["source_receipts"].items():
        if sha(REPO / relative) != expected:
            raise ValueError(f"Frozen input receipt changed: {relative}")
    with np.load(HERE / "comparison_inputs.npz", allow_pickle=False) as bank:
        for key, row in receipt["arrays"].items():
            array = bank[key]
            if list(array.shape) != row["shape"] or str(array.dtype) != row["dtype"] or hashlib.sha256(array.tobytes()).hexdigest() != row["array_sha256"]:
                raise ValueError(f"Frozen array mismatch: {key}")
    if protocol["maximum_native_calls"] != 70 or protocol["full_solver"]["repeats_per_input_batch_variant"] != 3:
        raise ValueError("Frozen native call budget changed")
    return protocol


def schedule():
    return [{"variant": variant, "repeat": repeat, "batch_size": batch, "input": kind,
             "id": f"B{batch}-{kind}-{variant}-{repeat}", "max_native_calls": 3 if kind == "warm" else 2}
            for batch in (1, 16) for kind in ("cold", "warm") for variant, repeat in ORDER]


def _hash_value(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA256: {label}")
    return value


def _relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Noncanonical source path: {value}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"Noncanonical source path: {value}")
    return path


def load_source_manifest(path, expected_sha256):
    """Caller-pinned review authority, independent of both mutable build receipts."""
    _hash_value(expected_sha256, "externally reviewed complete manifest")
    if sha(path) != expected_sha256:
        raise ValueError("Reviewed source manifest hash mismatch")
    authority = read_json(path)
    if authority.get("schema_version") != 2 or not re.fullmatch(r"[0-9a-f]{40}", authority.get("base_commit", "")):
        raise ValueError("Reviewed source manifest schema/base_commit mismatch")
    patch = authority["candidate_diff"]
    _hash_value(patch["sha256"], "candidate diff")
    if not Path(patch["path"]).is_absolute() or sha(patch["path"]) != patch["sha256"]:
        raise ValueError("Reviewed candidate diff artifact mismatch")
    hashes = authority["source_file_sha256"]
    if not hashes:
        raise ValueError("Reviewed complete source manifest is empty")
    for relative, expected in hashes.items():
        _relative_path(relative)
        _hash_value(expected, relative)
    if any(relative not in hashes for relative in SOURCE_FILES):
        raise ValueError("Reviewed source manifest omits required critical files")
    for relative, expected in authority.get("external_dependency_sha256", {}).items():
        if not Path(relative).is_absolute():
            raise ValueError("External dependency path must be absolute")
        _hash_value(expected, relative)
        if sha(relative) != expected:
            raise ValueError(f"Reviewed external dependency bytes changed: {relative}")
    authority["manifest_path"] = str(Path(path).resolve())
    authority["manifest_sha256"] = expected_sha256
    return authority


def parse_cmake_cache(path):
    """Parse full typed entries; comments/substrings never satisfy a field."""
    entries = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line or line.startswith(("//", "#")):
            continue
        match = re.fullmatch(r"([^:=\s]+):([A-Z_]+)=(.*)", line)
        if not match:
            raise ValueError(f"Malformed CMake cache entry at line {number}")
        key, kind, value = match.groups()
        if key in entries:
            raise ValueError(f"Duplicate CMake cache key: {key}")
        entries[key] = (kind, value)
    return entries


def validate_cache(path, variant, flags, main):
    entries = parse_cmake_cache(path)
    arch = flags["CMAKE_CUDA_ARCHITECTURES"]
    if not isinstance(arch, str) or not re.fullmatch(r"[1-9][0-9]*(?:-real|-virtual)?(?:;[1-9][0-9]*(?:-real|-virtual)?)*", arch):
        raise ValueError("Explicit numeric CUDA architecture required")
    expected = {"GATO_EXPERIMENT_TELEMETRY": ("BOOL", variant),
                "CMAKE_BUILD_TYPE": ("STRING", "Release"),
                "CMAKE_CUDA_ARCHITECTURES": ("STRING", arch)}
    if main:
        expected.update(KNOTS=("STRING", "32"), PLANT=("STRING", "tiago_right"))
    for key, wanted in expected.items():
        if entries.get(key) != wanted:
            raise ValueError(f"Build cache option differs from frozen receipt: {key}")


def validate_receipt(path, variant, source_manifest_path, source_manifest_sha256):
    authority = load_source_manifest(source_manifest_path, source_manifest_sha256)
    receipt = read_json(path)
    if receipt.get("schema_version") != 2 or receipt.get("variant") != variant or "source_commit" in receipt:
        raise ValueError("Build receipt schema/variant mismatch; uncommitted candidate must not claim source_commit")
    if receipt.get("base_commit") != authority["base_commit"] or receipt.get("candidate_diff_sha256") != authority["candidate_diff"]["sha256"]:
        raise ValueError("Build receipt base/candidate diff identity mismatch")
    if receipt.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("Build receipt does not bind externally reviewed complete source manifest")
    if receipt.get("source_file_sha256") != authority["source_file_sha256"]:
        raise ValueError("Receipt source map differs from complete authoritative manifest")
    if not Path(receipt["build_root"]).is_absolute() or not Path(receipt["source_root"]).is_absolute():
        raise ValueError("Build-owned source/build paths must be absolute")
    build = Path(receipt["build_root"]).resolve()
    source = Path(receipt["source_root"]).resolve()
    if not source.is_relative_to(build) or source == REPO.resolve() or ".worktrees/experiment-runner" in str(source):
        raise ValueError("Sources must belong to a separate build-owned snapshot")
    flags = receipt["flags"]
    expected_flags = dict(authority["build_flags"], GATO_EXPERIMENT_TELEMETRY=variant == "ON")
    if flags != expected_flags or flags.get("GATO_EXPERIMENT_TELEMETRY") is not (variant == "ON"):
        raise ValueError("Build flags differ from externally reviewed expected flags")
    if flags.get("CMAKE_BUILD_TYPE") != "Release" or type(flags.get("N")) is not int or flags["N"] != 32 or flags.get("plant") != "tiago_right":
        raise ValueError("Frozen build configuration mismatch")
    for key in ("module", "direct_executable", "cmake_cache", "direct_cmake_cache", "compiler_resource_log", "compiled_dependencies"):
        info = receipt[key]
        if not Path(info["path"]).is_absolute():
            raise ValueError(f"Build-owned artifact path must be absolute: {key}")
        artifact = Path(info["path"]).resolve()
        _hash_value(info["sha256"], key)
        if not artifact.is_relative_to(build) or not artifact.is_file() or sha(artifact) != info["sha256"]:
            raise ValueError(f"Build-owned artifact/hash mismatch: {key}")
    if not re.fullmatch(r"bsqpN32_tiago_right(?:\.[A-Za-z0-9_-]+)*\.so", Path(receipt["module"]["path"]).name):
        raise ValueError("Native module basename must bind N32 and tiago_right")
    validate_cache(receipt["cmake_cache"]["path"], variant, flags, main=True)
    validate_cache(receipt["direct_cmake_cache"]["path"], variant, flags, main=False)
    if Path(receipt["compiler_resource_log"]["path"]).stat().st_size == 0:
        raise ValueError("Compiler resource log is empty")
    for relative, expected in authority["source_file_sha256"].items():
        artifact = source / relative
        if not artifact.resolve().is_relative_to(source) or artifact.is_symlink() or not artifact.is_file() or sha(artifact) != expected:
            raise ValueError(f"Source snapshot bytes changed: {relative}")
    module = Path(receipt["module"]["path"]).resolve()
    actual = set()
    for artifact in source.rglob("*"):
        if artifact.is_symlink():
            raise ValueError(f"Source snapshot contains unreviewed link: {artifact}")
        if artifact.is_file():
            # Only this exact hashed receipt artifact may be an extra build output.
            if artifact.resolve() == module and str(artifact.relative_to(source)) not in authority["source_file_sha256"]:
                continue
            actual.add(str(artifact.relative_to(source)))
    if actual != set(authority["source_file_sha256"]):
        raise ValueError("Source snapshot inventory differs from complete authoritative manifest")
    dependencies = read_json(receipt["compiled_dependencies"]["path"])
    if not dependencies.get("local_source_files") or not dependencies.get("depfile_sha256"):
        raise ValueError("Compiled dependency receipt must retain local dependencies and compiler depfiles")
    for relative in dependencies["local_source_files"]:
        if relative not in authority["source_file_sha256"]:
            raise ValueError(f"Compiled dependency omitted from reviewed manifest: {relative}")
    external = dependencies.get("external_dependency_sha256", {})
    for filename, expected in external.items():
        if not Path(filename).is_absolute() or sha(filename) != _hash_value(expected, filename):
            raise ValueError(f"Compiled external dependency bytes changed: {filename}")
    for filename, expected in authority.get("external_dependency_sha256", {}).items():
        if external.get(filename) != expected:
            raise ValueError(f"Compiled dependency differs from reviewed external inventory: {filename}")
    for filename, expected in dependencies["depfile_sha256"].items():
        artifact = Path(filename).resolve()
        if not artifact.is_relative_to(build) or sha(artifact) != _hash_value(expected, filename):
            raise ValueError("Compiler depfile ownership/hash mismatch")
    # Keep the narrow reviewed-worktree byte comparison as an additional check;
    # it is not the authority for the complete snapshot inventory.
    for relative in SOURCE_FILES:
        if authority["source_file_sha256"].get(relative) != sha(REPO / relative):
            raise ValueError(f"Snapshot is not the reviewed current candidate: {relative}")
    receipt.update(receipt_path=str(Path(path).resolve()), receipt_sha256=sha(path),
                   source_manifest_path=authority["manifest_path"], source_manifest_sha256=source_manifest_sha256)
    return receipt


def validate_pair(off_path, on_path, source_manifest_path, source_manifest_sha256):
    receipts = {variant: validate_receipt(path, variant, source_manifest_path, source_manifest_sha256)
                for variant, path in (("OFF", off_path), ("ON", on_path))}
    off, on = receipts["OFF"], receipts["ON"]
    off_build, on_build = Path(off["build_root"]).resolve(), Path(on["build_root"]).resolve()
    if off_build.is_relative_to(on_build) or on_build.is_relative_to(off_build) or Path(off["source_root"]).resolve() == Path(on["source_root"]).resolve():
        raise ValueError("ON/OFF must have separate independent build/source roots")
    if off["base_commit"] != on["base_commit"] or off["candidate_diff_sha256"] != on["candidate_diff_sha256"] or off["source_file_sha256"] != on["source_file_sha256"]:
        raise ValueError("ON/OFF source identities differ")
    flags = [{k: v for k, v in row["flags"].items() if k != "GATO_EXPERIMENT_TELEMETRY"} for row in (off, on)]
    if flags[0] != flags[1]:
        raise ValueError("Nontelemetry ON/OFF build flags differ")
    return receipts


def tiled_inputs(kind, batch):
    with np.load(HERE / "comparison_inputs.npz", allow_pickle=False) as bank:
        return {field: np.repeat(bank[kind + "_" + field], batch, axis=0).copy()
                for field in ("state", "reference", "warm_start")}


def check_native_reply(raw, stats, variant, batch, protocol):
    count = np.asarray(raw["pcg_iters"])
    times = np.asarray(raw["pcg_times_us"])
    flags = np.asarray(raw["kkt_converged"])
    if count.ndim != 2 or count.shape[1] != batch or times.shape != (count.shape[0],):
        raise ValueError("Native PCG rows/lanes/times shape mismatch")
    if count.shape[0] == 0 or count.shape[0] > protocol["full_solver"]["parameters"]["max_sqp_iters"]:
        raise ValueError("Unexpected native SQP work")
    np.testing.assert_array_equal(stats["pcg_iters"], count)
    np.testing.assert_array_equal(stats["pcg_times_us"], times)
    np.testing.assert_array_equal(stats["kkt_converged"], flags)
    if flags.shape != (batch,) or np.asarray(raw["sqp_iters"]).shape != (batch,):
        raise ValueError("Native SQP flags/counts shape mismatch")
    if raw["ls_num_iters"] not in (count.shape[0], count.shape[0] - 1):
        raise ValueError("Line search/terminal PCG row mismatch")
    if variant == "OFF":
        if "experiment_telemetry" in raw or "experiment_telemetry" in stats:
            raise ValueError("DefaultOFF stats API changed")
        return
    telemetry = raw["experiment_telemetry"]
    codes = np.asarray(telemetry["pcg_stop_reason_codes"])
    if telemetry["schema_version"] != 1 or codes.dtype != np.uint32 or codes.shape != count.shape or not np.isin(codes, [1, 2, 3, 4]).all():
        raise ValueError("Actual stop codes missing/invalid")
    if telemetry["breakdown_stopping_supported"] is not False or telemetry["breakdown"] is not None or not telemetry["breakdown_unavailable_reason"]:
        raise ValueError("Breakdown unsupported/null semantics changed")
    cap = protocol["full_solver"]["parameters"]["max_pcg_iters"]
    if np.any(count[(codes == 1) | (codes == 2)] != 0) or np.any(count[codes == 4] != cap) or np.any((count[codes == 3] < 1) | (count[codes == 3] > cap)):
        raise ValueError("Actual branch/count inconsistency")
    sqp_code = telemetry["sqp_stop_reason_code"]
    if sqp_code == 1:
        if flags.sum() < batch * protocol["full_solver"]["parameters"]["solve_ratio"] or raw["ls_num_iters"] != count.shape[0] - 1:
            raise ValueError("Batch KKT-ratio branch inconsistent with flags/last row")
    elif sqp_code == 2:
        if count.shape[0] != protocol["full_solver"]["parameters"]["max_sqp_iters"]:
            raise ValueError("SQP cap branch inconsistent with work")
    else:
        raise ValueError("Unknown SQP stop branch")


def worker(request_path):
    request = read_json(request_path)
    output = Path(request["output"])
    status = {"status": "started", "native_calls_started": 0, "native_calls_completed": 0, "id": request["id"]}
    strict_json(output / "worker_status.json", status)
    captured = None
    last_wall = None
    try:
        protocol = freeze_check()
        if (request["variant"] not in ("OFF", "ON") or request["batch_size"] not in (1, 16)
                or request["input"] not in ("cold", "warm")
                or request["max_native_calls"] != (3 if request["input"] == "warm" else 2)):
            raise ValueError("Worker request changes frozen scope/call budget")
        # Validate everything before loading any production native module.
        receipt = validate_receipt(request["receipt_path"], request["variant"], request["source_manifest_path"], request["source_manifest_sha256"])
        if sha(request["receipt_path"]) != request["receipt_sha256"]:
            raise ValueError("Reviewed receipt changed between parent and worker")
        sys.dont_write_bytecode = True
        source = Path(receipt["source_root"])
        sys.path.insert(0, str(source / "python"))
        module_name = "bsqp.bsqpN32_tiago_right"
        spec = importlib.util.spec_from_file_location(module_name, receipt["module"]["path"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        from bsqp.interface import BSQP
        interface_file = Path(sys.modules["bsqp.interface"].__file__).resolve()
        if interface_file != source / "python/bsqp/interface.py":
            raise ValueError("Imported wrapper is not the build-owned reviewed source")
        params = protocol["full_solver"]["parameters"]
        batch = request["batch_size"]

        class CaptureNative:
            def __init__(self, native):
                self.native = native
                self.last_reply = None
                self.calls_started = 0
                self.calls_returned = 0
            def __getattr__(self, name):
                return getattr(self.native, name)
            def solve(self, *args):
                self.calls_started += 1
                self.last_reply = None
                self.last_reply = self.native.solve(*args)
                self.calls_returned += 1
                return self.last_reply

        def new_solver():
            solver = BSQP(model_path=str(source / "gato/dynamics/tiago_right/tiago_right_arm.urdf"),
                          batch_size=batch, N=32, dt=.03, plant_type="tiago_right", **params)
            solver.solver = CaptureNative(solver.solver)
            return solver

        def invoke(solver, label, kind):
            nonlocal captured, last_wall
            captured = None
            last_wall = None
            inputs = tiled_inputs(kind, batch)
            save_payload(output, label + "_input", inputs)
            status["active_call"] = label
            status["call_submission"] = "pending; process death may leave native completion unknown"
            strict_json(output / "worker_status.json", status)
            captured = solver.solver
            captured.last_reply = None
            calls_before = captured.calls_started
            returned_before = captured.calls_returned
            before = time.perf_counter()
            try:
                solver.solve(inputs["state"], inputs["reference"], inputs["warm_start"])
            finally:
                last_wall = time.perf_counter() - before
                status["native_calls_started"] += captured.calls_started - calls_before
                status["native_calls_completed"] += captured.calls_returned - returned_before
            # Public BSQP.solve result/telemetry decoding is inside the timer.
            # Subsequent get_stats, raw copying/archiving and validation are outside.
            save_payload(output, label + "_output", {"native_reply": captured.last_reply,
                         "public_stats": solver.get_stats(), "wrapper_wall_s": last_wall})
            status["call_submission"] = "returned"
            strict_json(output / "worker_status.json", status)
            expected_shapes = {"XU": (batch, 32 * (14 + 7) - 7),
                               "initial_merit": (batch,), "final_merit": (batch,)}
            for field, expected_shape in expected_shapes.items():
                if np.asarray(captured.last_reply[field]).shape != expected_shape:
                    raise ValueError(f"Malformed returned {field} shape in {label}; expected {expected_shape}; raw preserved, validation held")
                if not np.isfinite(np.asarray(captured.last_reply[field])).all():
                    raise ValueError(f"Nonfinite returned {field} in {label}; raw preserved, validation held")
            check_native_reply(captured.last_reply, solver.get_stats(), request["variant"], batch, protocol)
            return {"wrapper_wall_s": last_wall, "native_solve_us": float(captured.last_reply["sqp_time_us"])}

        warmup = new_solver()
        invoke(warmup, "unrelated_warmup", "unrelated")
        captured = None
        last_wall = None
        del warmup  # Release disposable native/result ownership before constructor.
        # Measured cold/prefix state belongs to a fresh solver.
        measured = new_solver()
        if request["input"] == "warm":
            invoke(measured, "cold_prefix", "cold")
        timing = invoke(measured, "measured", request["input"])
        if status["native_calls_completed"] != request["max_native_calls"]:
            raise ValueError("Frozen worker call count mismatch")
        status.update(status="completed", timing=timing)
        strict_json(output / "worker_status.json", status)
        manifest(output)
        return 0
    except BaseException as error:
        # Include returned native data even if public normalization/validation failed.
        if captured is not None and captured.last_reply is not None:
            save_payload(output, "failed_call_return", {"native_reply": captured.last_reply, "wrapper_wall_s": last_wall})
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        strict_json(output / "worker_status.json", status)
        (output / "traceback.txt").write_text(traceback.format_exc())
        manifest(output)
        return 1


def check_direct(folder, variant, protocol):
    config = read_json(Path(folder) / "configuration.json")
    if config != {"N": 32, "batch_size": 1, "state_size": 14, "vector_size_padded": 476,
                  "stats_words": 2 if variant == "ON" else 1, "telemetry_enabled": variant == "ON"}:
        raise ValueError("Direct fixture configuration mismatch")
    outputs = {}
    for fixture in protocol["direct_PCG"]["fixtures"]:
        row = Path(folder) / fixture["id"]
        request = read_json(row / "request.json")
        if request != {"status": "started", "max_pcg_iters": fixture["max_pcg_iters"], "epsilon": 0, "kkt_mask": fixture["kkt_mask"]}:
            raise ValueError("Direct fixture input protocol mismatch")
        A = np.fromfile(row / "A.f32", dtype=np.float32).reshape(32, 14, 42)
        M = np.fromfile(row / "M_inv.f32", dtype=np.float32).reshape(32, 14, 42)
        expected = np.zeros((32, 14, 42), np.float32)
        expected[:, np.arange(14), 14 + np.arange(14)] = 1
        np.testing.assert_array_equal(M, expected)
        if fixture["id"] == "one-iteration-cap-nonconverged":
            expected[0, 1, 15] = 2
        np.testing.assert_array_equal(A, expected)
        b = np.fromfile(row / "b.f32", dtype=np.float32)
        expected_b = np.zeros(476, np.float32)
        expected_b[14:16] = fixture["b_first_components"]
        np.testing.assert_array_equal(b, expected_b)
        np.testing.assert_array_equal(np.fromfile(row / "initial_x.f32", dtype=np.float32), np.zeros(476, np.float32))
        completion = read_json(row / "completion.json")
        stats = np.fromfile(row / "stats.u32", dtype=np.uint32)
        if completion["status"] != "completed" or stats.size != config["stats_words"] or int(stats[0]) != fixture["expected_iterations"] or completion["iterations"] != int(stats[0]):
            raise ValueError("Direct fixture iteration count/shape mismatch")
        expected_reason = fixture["expected_reason_code"] if variant == "ON" else None
        if completion["stop_reason_code"] != expected_reason or (variant == "ON" and int(stats[1]) != expected_reason):
            raise ValueError("Direct actual stop path mismatch")
        if completion["breakdown_stopping_supported"] is not False or completion["breakdown"] is not None:
            raise ValueError("Direct unsupported breakdown semantics changed")
        x = np.fromfile(row / "output_x.f32", dtype=np.float32)
        if x.shape != (476,) or not np.isfinite(x).all():
            raise ValueError("Direct fixture finite output shape mismatch")
        expected_x = np.zeros(476, np.float32)
        if fixture["max_pcg_iters"]:
            expected_x[14:16] = [2 / 3, 2 / 3] if fixture["id"] == "one-iteration-cap-nonconverged" else [1, 1]
        np.testing.assert_allclose(x, expected_x, atol=1e-6, rtol=1e-6)
        outputs[fixture["id"]] = x
    if {folder.name for folder in Path(folder).iterdir() if folder.is_dir()} != set(outputs):
        raise ValueError("Unexpected direct fixture directories/call count")
    return outputs


def summarize(output, protocol):
    groups = []
    for batch in (1, 16):
        for kind in ("cold", "warm"):
            collected = {variant: [load_payload(output / f"B{batch}-{kind}-{variant}-{rep}", "measured_output")
                                  for rep in (1, 2, 3)] for variant in ("OFF", "ON")}
            row = {"batch_size": batch, "input": kind, "numeric_comparisons": {}, "timings": {}, "work_and_flags": {}}
            for field in ("XU", "final_merit", "initial_merit"):
                values = {variant: np.stack([sample["native_reply"][field] for sample in collected[variant]]) for variant in collected}
                if not all(np.isfinite(array).all() for array in values.values()):
                    raise ValueError(f"Nonfinite returned {field}; raw data retained, validation held")
                tolerances = protocol["full_solver"]["practical_tolerances"]
                abs_tol, rel_tol = (tolerances["XU_abs"], tolerances["XU_rel"]) if field == "XU" else (tolerances["merit_abs"], tolerances["merit_rel"])
                repeat_range = np.maximum(np.ptp(values["OFF"], axis=0), np.ptp(values["ON"], axis=0))
                differences = np.abs(values["OFF"] - values["ON"])
                threshold = abs_tol + rel_tol * np.maximum(np.abs(values["OFF"]), np.abs(values["ON"])) + repeat_range
                passed = bool(np.all(differences <= threshold))
                row["numeric_comparisons"][field] = {"passed": passed, "max_absolute_difference": float(differences.max()), "max_within_variant_repeat_range": float(repeat_range.max())}
                if not passed:
                    raise ValueError(f"Practical ON/OFF disagreement beyond tolerance+repeat range: B{batch}/{kind}/{field}; no rerun")
            for variant, samples in collected.items():
                row["timings"][variant] = {}
                for metric, getter in (("wrapper_wall_s", lambda sample: sample["wrapper_wall_s"]),
                                       ("native_solve_us", lambda sample: sample["native_reply"]["sqp_time_us"])):
                    values = np.asarray([getter(sample) for sample in samples], dtype=float)
                    if not np.isfinite(values).all() or np.any(values < 0):
                        raise ValueError("Invalid timing values")
                    row["timings"][variant][metric] = {"samples": values.tolist(), "mean": float(values.mean()), "median": float(np.median(values))}
                row["work_and_flags"][variant] = [
                    {"sqp_iters": sample["native_reply"]["sqp_iters"].tolist(),
                     "kkt_converged": sample["native_reply"]["kkt_converged"].tolist(),
                     "pcg_iters": sample["native_reply"]["pcg_iters"].tolist(),
                     "ls_num_iters": sample["native_reply"]["ls_num_iters"]} for sample in samples]
            row["paired_ON_minus_OFF_wrapper_s"] = [collected["ON"][i]["wrapper_wall_s"] - collected["OFF"][i]["wrapper_wall_s"] for i in range(3)]
            row["paired_ON_minus_OFF_native_us"] = [float(collected["ON"][i]["native_reply"]["sqp_time_us"]) - float(collected["OFF"][i]["native_reply"]["sqp_time_us"]) for i in range(3)]
            groups.append(row)
    return {"status": "passed_bounded_comparison", "groups": groups,
            "timing_limit": "Three repeats describe samples only; no precise overhead interval or broad benchmark claim.",
            "variant_scope": "ON adds one stop word/lane and changes existing pageable count copy destination to pinned; tiny register/host decode allocations also included."}


def call_inventory(output):
    """Completed counts plus explicit uncertainty, never count receipts as success."""
    output = Path(output)
    direct_submitted = len(list(output.glob("direct-*/*/kernel_submission.json")))
    direct_completed = len(list(output.glob("direct-*/*/completion.json")))
    worker_started = worker_completed = 0
    uncertain = []
    for folder in output.glob("B*"):
        if not folder.is_dir():
            continue
        path = folder / "worker_status.json"
        if not path.exists():
            uncertain.append({"worker": folder.name, "reason": "process failed before publishing native counters"})
            continue
        status = read_json(path)
        worker_started += status["native_calls_started"]
        worker_completed += status["native_calls_completed"]
        if status["status"] == "started" and status.get("call_submission", "").startswith("pending"):
            uncertain.append({"worker": folder.name, "call": status.get("active_call"), "reason": "process exited with pending submission; native completion unknown"})
    return {"direct_kernel_submissions": direct_submitted, "direct_kernels_completed": direct_completed,
            "worker_native_calls_started_known": worker_started, "worker_native_calls_returned_known": worker_completed,
            "completed_native_calls_known": direct_completed + worker_completed,
            "unresolved_worker_submissions": uncertain,
            "direct_pending_or_failed_submissions": direct_submitted - direct_completed}


def execute(output, receipts, protocol):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    progress = {"status": "started", "maximum_native_calls": 70, "workers_completed": 0, "direct_variants_completed": 0}
    strict_json(output / "queue.json", {"schedule": schedule(), "build_receipts": receipts,
                "protocol_sha256": sha(HERE / "comparison_protocol.json"), "input_sha256": sha(HERE / "comparison_inputs.npz")})
    strict_json(output / "status.json", progress)
    try:
        direct = {}
        for variant in ("OFF", "ON"):
            folder = output / ("direct-" + variant)
            strict_json(output / ("direct-" + variant + "-request.json"), {"status": "prepared", "variant": variant, "max_native_calls": 5,
                        "executable": receipts[variant]["direct_executable"], "output": str(folder)})
            with (output / ("direct-" + variant + "-stdout.log")).open("w") as stdout, (output / ("direct-" + variant + "-stderr.log")).open("w") as stderr:
                result = subprocess.run([receipts[variant]["direct_executable"]["path"], str(folder)], stdout=stdout, stderr=stderr, check=False)
            strict_json(output / ("direct-" + variant + "-exit.json"), {"returncode": result.returncode})
            if result.returncode:
                raise RuntimeError(f"Direct {variant} fixture process failed; no retry")
            direct[variant] = check_direct(folder, variant, protocol)
            progress["direct_variants_completed"] += 1
            strict_json(output / "status.json", progress)
        for name in direct["OFF"]:
            np.testing.assert_allclose(direct["ON"][name], direct["OFF"][name], atol=1e-6, rtol=1e-6)
        for item in schedule():
            folder = output / item["id"]
            folder.mkdir(exist_ok=False)
            receipt = receipts[item["variant"]]
            request = dict(item, output=str(folder), receipt_path=receipt["receipt_path"], receipt_sha256=receipt["receipt_sha256"],
                           source_manifest_path=receipt["source_manifest_path"], source_manifest_sha256=receipt["source_manifest_sha256"])
            strict_json(folder / "request.json", request)  # before subprocess submission
            with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
                result = subprocess.run([sys.executable, "-B", str(HERE / "compare.py"), "--worker", str(folder / "request.json")], stdout=stdout, stderr=stderr, check=False,
                                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            strict_json(folder / "exit.json", {"returncode": result.returncode})
            if result.returncode:
                raise RuntimeError(f"Native worker {item['id']} failed; no retry")
            state = read_json(folder / "worker_status.json")
            if state["status"] != "completed" or state["native_calls_completed"] != item["max_native_calls"]:
                raise ValueError(f"Worker {item['id']} incomplete or zero-native-call false green")
            progress["workers_completed"] += 1
            strict_json(output / "status.json", progress)
        inventory = call_inventory(output)
        if inventory["completed_native_calls_known"] != 70 or inventory["unresolved_worker_submissions"] or inventory["direct_pending_or_failed_submissions"]:
            raise ValueError("Actual known native call count differs from fixed70 or includes unresolved submission")
        strict_json(output / "summary.json", summarize(output, protocol))
        progress.update(status="completed", actual_native_calls=inventory["completed_native_calls_known"])
        strict_json(output / "status.json", progress)
        return 0
    except BaseException as error:
        progress.update(status="failed_validation_held", error_type=type(error).__name__, error=str(error))
        strict_json(output / "status.json", progress)
        (output / "traceback.txt").write_text(traceback.format_exc())
        return 1
    finally:
        strict_json(output / "call_inventory.json", call_inventory(output))
        manifest(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-receipt", type=Path)
    parser.add_argument("--on-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--execute-reviewed", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    protocol = freeze_check()
    if not args.off_receipt or not args.on_receipt:
        if args.execute_reviewed:
            parser.error("Execution requires both reviewed build receipts and a fresh output directory")
        print(json.dumps({"status": "CPU_PREPARATION_ONLY", "maximum_native_calls": 70, "schedule": schedule()}, indent=2))
        return 0
    if not args.source_manifest or not args.source_manifest_sha256:
        parser.error("Build receipt validation requires independently reviewed --source-manifest and its externally pinned --source-manifest-sha256")
    receipts = validate_pair(args.off_receipt, args.on_receipt, args.source_manifest, args.source_manifest_sha256)
    if not args.execute_reviewed:
        print(json.dumps({"status": "validated_dry_run_no_native_import", "maximum_native_calls": 70, "schedule": schedule()}, indent=2))
        return 0
    if not args.output:
        parser.error("Execution requires --output pointing to a fresh directory")
    return execute(args.output, receipts, protocol)


if __name__ == "__main__":
    sys.exit(main())
