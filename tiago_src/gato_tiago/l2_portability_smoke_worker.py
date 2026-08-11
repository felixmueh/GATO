"""Isolated extension worker for the CUDA L2 portability smoke."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from gato_tiago.l2_portability_smoke import (
    AUTHORIZED_OUTPUT_PATH,
    EXPECTED_CONSTRUCTOR_CALLS_PER_MODULE,
    EXPECTED_FK_CALLS_PER_MODULE,
    EXPECTED_WORKER_ARRAY_NAMES,
    FORBIDDEN_CALL_COUNTS,
    FROZEN_CUDA_ARCH,
    FROZEN_MODULES,
    PROTOCOL_VERSION,
)


WORKER_EXECUTION_AUTHORIZATION = None
WORKER_PROTOCOL_VERSION = PROTOCOL_VERSION + "_worker_1"
SUPPORTED_MODULES = {row["module_name"]: dict(row) for row in FROZEN_MODULES}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        f"{array.dtype.str}|{array.shape}|".encode() + array.tobytes()
    ).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_request(request):
    required = {
        "module_name",
        "extension_path",
        "extension_sha256",
        "extension_size_bytes",
        "cuda_arch",
        "protocol_version",
    }
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError("portability worker request schema mismatch")
    spec = SUPPORTED_MODULES.get(request["module_name"])
    extension = Path(request["extension_path"]).resolve()
    if spec is None or request["protocol_version"] != WORKER_PROTOCOL_VERSION:
        raise ValueError("portability worker identity/protocol mismatch")
    if request["extension_sha256"] != spec["extension_sha256"]:
        raise ValueError("portability worker frozen extension hash mismatch")
    if not extension.is_file() or sha256_file(extension) != spec["extension_sha256"]:
        raise ValueError("portability worker extension file/hash mismatch")
    if (
        request["extension_size_bytes"] != spec["extension_size_bytes"]
        or extension.stat().st_size != spec["extension_size_bytes"]
        or request["cuda_arch"] != FROZEN_CUDA_ARCH
    ):
        raise ValueError("portability worker extension size/architecture mismatch")
    return spec, extension


def certify_arrays(arrays):
    if set(arrays) != EXPECTED_WORKER_ARRAY_NAMES:
        return {"all_worker_array_gates_pass": False}
    q = np.asarray(arrays["q_float32"])
    q16 = np.asarray(arrays["q16_float32"])
    fk1 = np.asarray(arrays["fk_b1_float32"])
    fk16 = np.asarray(arrays["fk_b16_float32"])
    exact = bool(
        q.shape == (7,)
        and q.dtype == np.float32
        and np.array_equal(q, np.zeros(7, dtype=np.float32))
        and q16.shape == (16, 7)
        and q16.dtype == np.float32
        and np.array_equal(q16, np.repeat(q[None, :], 16, axis=0))
        and fk1.shape == (1, 3)
        and fk1.dtype == np.float32
        and fk16.shape == (16, 3)
        and fk16.dtype == np.float32
    )
    finite = bool(exact and np.all(np.isfinite(fk1)) and np.all(np.isfinite(fk16)))
    repeated = bool(
        finite and np.array_equal(fk16, np.repeat(fk1, 16, axis=0))
    )
    return {
        "exact_shapes_dtypes_and_zero_input": exact,
        "all_fk_finite": finite,
        "b1_b16_repeated_lane_bitwise_equal": repeated,
        "all_worker_array_gates_pass": bool(exact and finite and repeated),
    }


def _run_worker(request_path, output_path):  # pragma: no cover
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text())
    spec, extension = validate_request(request)
    module = importlib.import_module(spec["module_name"])
    if (
        Path(module.__file__).resolve() != extension
        or module.KNOT_POINTS != 64
        or module.REFERENCE_SIZE != spec["reference_size"]
        or module.TOOL_POSITION_FRAME != "arm_right_tool_joint_origin"
        or module.TOOL_POSITION_SIZE != 3
    ):
        raise RuntimeError("portability worker imported module identity mismatch")
    q = np.zeros(7, dtype=np.float32)
    q16 = np.repeat(q[None, :], 16, axis=0)
    constructor_calls = 0
    fk_calls = 0
    b1 = module.BSQP_1_float()
    constructor_calls += 1
    b16 = module.BSQP_16_float()
    constructor_calls += 1
    fk1 = np.asarray(b1.tool_position(q[None, :]))
    fk_calls += 1
    fk16 = np.asarray(b16.tool_position(q16))
    fk_calls += 1
    if (
        constructor_calls != EXPECTED_CONSTRUCTOR_CALLS_PER_MODULE
        or fk_calls != EXPECTED_FK_CALLS_PER_MODULE
    ):
        raise RuntimeError("portability worker call ledger mismatch")
    arrays = {
        "q_float32": q,
        "q16_float32": q16,
        "fk_b1_float32": fk1,
        "fk_b16_float32": fk16,
    }
    certificate = certify_arrays(arrays)
    if not certificate["all_worker_array_gates_pass"]:
        raise RuntimeError("portability worker output certificate failed")
    output = Path(output_path)
    npz_path = output.with_suffix(".npz")
    _atomic_npz(npz_path, arrays)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "module_name": spec["module_name"],
        "module_file": str(extension),
        "extension_sha256": spec["extension_sha256"],
        "extension_size_bytes": spec["extension_size_bytes"],
        "cuda_arch": FROZEN_CUDA_ARCH,
        "KNOT_POINTS": int(module.KNOT_POINTS),
        "REFERENCE_SIZE": int(module.REFERENCE_SIZE),
        "TOOL_POSITION_FRAME": str(module.TOOL_POSITION_FRAME),
        "TOOL_POSITION_SIZE": int(module.TOOL_POSITION_SIZE),
        "request_path": str(request_path),
        "request_sha256": sha256_file(request_path),
        "constructor_calls": constructor_calls,
        "fk_calls": fk_calls,
        "forbidden_call_counts": dict(FORBIDDEN_CALL_COUNTS),
        "certificate": certificate,
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "array_names": sorted(arrays),
        "array_hashes": {name: array_hash(value) for name, value in arrays.items()},
        "optimization_evidence": False,
        "timing_evidence": False,
    }
    _atomic_json(output, summary)
    return summary


def execute_worker(request_path, output_path, *, authorization=None):
    if (
        WORKER_EXECUTION_AUTHORIZATION is None
        or authorization is not WORKER_EXECUTION_AUTHORIZATION
    ):
        raise RuntimeError("L2 portability worker execution is blocked")
    request = Path(request_path).resolve()
    output = Path(output_path).resolve()
    allowed = {
        (
            AUTHORIZED_OUTPUT_PATH.parent / f"smoke.{name.split('.')[-1]}.request.json",
            AUTHORIZED_OUTPUT_PATH.parent / f"smoke.{name.split('.')[-1]}.json",
        )
        for name in SUPPORTED_MODULES
    }
    if (request, output) not in allowed:
        raise RuntimeError("L2 portability worker path is not authorized")
    if output.exists() or output.with_suffix(".npz").exists():
        raise RuntimeError("L2 portability worker refuses overwrite or rerun")
    return _run_worker(request, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute or WORKER_EXECUTION_AUTHORIZATION is None:
        raise SystemExit("L2 portability worker execution is blocked")
    execute_worker(
        args.request,
        args.output,
        authorization=WORKER_EXECUTION_AUTHORIZATION,
    )
    return 0


if __name__ == "__main__":
    main()
