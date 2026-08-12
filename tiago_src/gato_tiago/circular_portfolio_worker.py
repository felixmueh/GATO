"""Isolated N96 CUDA replay worker; execution remains statically blocked."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from gato_tiago.circular_portfolio import (
    CONTROL_LIMIT_COST, CYLINDER_WEIGHT, DENSE_SAMPLES, DENSE_SUBSTEPS, DT,
    EXPECTED_LEDGER, INTERVALS, KKT_TOL, MAX_PCG_ITERS, MU, N_COST, PCG_TOL, QD_COST,
    Q_COST, Q_LIMIT_COST, RHO, SOLVE_RATIO, U_COST, VELOCITY_LIMIT_COST,
)


WORKER_EXECUTION_AUTHORIZATION = None


def expected_worker_paths(index):
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(EXPECTED_LEDGER):
        raise ValueError("worker index is outside the canonical ledger")
    stem = f"p1.worker.{index:03d}"
    return {
        "request_path": AUTHORIZED_ROOT / f"{stem}.request.json",
        "input_path": AUTHORIZED_ROOT / f"{stem}.input.npz",
        "json_path": AUTHORIZED_ROOT / f"{stem}.json",
        "npz_path": AUTHORIZED_ROOT / f"{stem}.npz",
    }
WORKER_PROTOCOL = "tiago_tool_center_circular_portfolio_cuda_worker_p1_1"
MODULE_NAME = "bsqp.bsqpN96_tiago_right_circular_portfolio_toll"
MODULE_RELATIVE_PATH = "python/bsqp/bsqpN96_tiago_right_circular_portfolio_toll.cpython-310-x86_64-linux-gnu.so"
# A separately audited build/import stage must replace both sentinels before
# worker/full-campaign authorization. None is deliberately fail-closed.
FROZEN_EXTENSION_SHA256 = None
FROZEN_EXTENSION_SIZE_BYTES = None
AUTHORIZED_ROOT = Path(
    "/tmp/tiago-tool-center-circular-portfolio-p1-authorized-once"
)
EXPECTED_SIM_FORWARD_CALLS = INTERVALS * DENSE_SUBSTEPS
EXPECTED_TOOL_POSITION_CALLS = (DENSE_SAMPLES + 15) // 16


def sha256_file(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest


def request_array_schema(arrays: Mapping):
    expected = {
        "x0_float32": ((14,), np.float32),
        "controls_float32": ((INTERVALS, 7), np.float32),
    }
    return bool(
        set(arrays) == set(expected)
        and all(np.asarray(arrays[name]).shape == shape
                and np.asarray(arrays[name]).dtype == np.dtype(dtype)
                and np.isfinite(np.asarray(arrays[name])).all()
                for name, (shape, dtype) in expected.items())
    )


def certify_lane_binding(index, request_path, request, request_arrays,
                         output_arrays, expected_x0, expected_controls):
    try:
        paths = expected_worker_paths(index)
    except ValueError:
        return False
    expected_x0 = np.asarray(expected_x0)
    expected_controls = np.asarray(expected_controls)
    return bool(
        Path(request_path).resolve() == paths["request_path"]
        and request.get("identity") == list(EXPECTED_LEDGER[index])
        and Path(request.get("input_path", "")).resolve() == paths["input_path"]
        and Path(request.get("output_path", "")).resolve() == paths["json_path"]
        and request_array_schema(request_arrays)
        and expected_x0.shape == (14,) and expected_x0.dtype == np.float32
        and expected_controls.shape == (INTERVALS, 7)
        and expected_controls.dtype == np.float32
        and np.array_equal(request_arrays["x0_float32"], expected_x0)
        and np.array_equal(request_arrays["controls_float32"], expected_controls)
        and np.array_equal(output_arrays.get("captured_x0_float32"), expected_x0)
        and np.array_equal(
            output_arrays.get("captured_controls_float32"), expected_controls
        )
    )


def certify_worker_output(summary: Mapping, arrays: Mapping, request: Mapping,
                          request_arrays: Mapping):
    expected_summary = {
        "protocol", "identity", "request_path", "request_sha256", "input_path",
        "input_sha256", "module_name", "module_path", "extension_sha256",
        "extension_size_bytes", "KNOT_POINTS", "REFERENCE_SIZE",
        "TOOL_POSITION_FRAME", "TOOL_POSITION_SIZE", "constructor_calls",
        "sim_forward_calls", "tool_position_calls", "solve_calls", "sqp_calls",
        "array_names", "array_hashes", "npz_path", "npz_sha256", "certificate",
    }
    expected_arrays = {
        "captured_x0_float32": ((14,), np.float32),
        "captured_controls_float32": ((INTERVALS, 7), np.float32),
        "cuda_dense_state_float32": ((DENSE_SAMPLES, 14), np.float32),
        "cuda_dense_tool_float32": ((DENSE_SAMPLES, 3), np.float32),
    }
    expected_request = {
        "protocol", "identity", "input_path", "input_sha256", "module_name",
        "module_path", "extension_sha256", "extension_size_bytes", "output_path",
    }
    if (set(summary) != expected_summary or set(arrays) != set(expected_arrays)
            or set(request) != expected_request):
        return {"passes": False}
    try:
        ledger_index = EXPECTED_LEDGER.index(tuple(summary["identity"]))
        expected_paths = expected_worker_paths(ledger_index)
    except (ValueError, TypeError):
        return {"passes": False}
    array_hashes = {
        name: hashlib.sha256(
            f"{np.asarray(value).dtype.str}|{np.asarray(value).shape}|".encode()
            + np.ascontiguousarray(value).tobytes()
        ).hexdigest() for name, value in arrays.items()
    }
    try:
        request_file_gate = (
            Path(summary["request_path"]).is_file()
            and sha256_file(summary["request_path"]) == summary["request_sha256"]
            and Path(summary["input_path"]).is_file()
            and sha256_file(summary["input_path"]) == summary["input_sha256"]
            and Path(summary["npz_path"]).is_file()
            and sha256_file(summary["npz_path"]) == summary["npz_sha256"]
        )
        extension_gate = (
            FROZEN_EXTENSION_SHA256 is not None
            and FROZEN_EXTENSION_SIZE_BYTES is not None
            and Path(summary["module_path"]).is_file()
            and sha256_file(summary["module_path"]) == FROZEN_EXTENSION_SHA256
            and Path(summary["module_path"]).stat().st_size == FROZEN_EXTENSION_SIZE_BYTES
        )
    except OSError:
        request_file_gate = extension_gate = False
    gates = {
        "request_schema": request_array_schema(request_arrays),
        "protocol": summary["protocol"] == WORKER_PROTOCOL,
        "identity": summary["identity"] == request["identity"],
        "identity_paths": Path(summary["request_path"]).resolve() == expected_paths["request_path"]
        and Path(summary["input_path"]).resolve() == expected_paths["input_path"]
        and Path(summary["npz_path"]).resolve() == expected_paths["npz_path"]
        and Path(request["output_path"]).resolve() == expected_paths["json_path"],
        "request_exact": request["protocol"] == WORKER_PROTOCOL
        and request["input_path"] == summary["input_path"]
        and request["input_sha256"] == summary["input_sha256"]
        and request["module_name"] == summary["module_name"]
        and request["module_path"] == summary["module_path"]
        and request["extension_sha256"] == summary["extension_sha256"]
        and request["extension_size_bytes"] == summary["extension_size_bytes"]
        and request["output_path"] == summary["npz_path"].removesuffix(".npz") + ".json",
        "module": extension_gate and summary["module_name"] == MODULE_NAME
        and summary["extension_sha256"] == FROZEN_EXTENSION_SHA256
        and summary["extension_size_bytes"] == FROZEN_EXTENSION_SIZE_BYTES,
        "attrs": summary["KNOT_POINTS"] == 96 and summary["REFERENCE_SIZE"] == 10
        and summary["TOOL_POSITION_FRAME"] == "arm_right_tool_joint_origin"
        and summary["TOOL_POSITION_SIZE"] == 3,
        "counts": summary["constructor_calls"] == 1
        and summary["sim_forward_calls"] == EXPECTED_SIM_FORWARD_CALLS
        and summary["tool_position_calls"] == EXPECTED_TOOL_POSITION_CALLS
        and summary["solve_calls"] == summary["sqp_calls"] == 0,
        "arrays": all(np.asarray(arrays[name]).shape == shape
        and np.asarray(arrays[name]).dtype == np.dtype(dtype)
        and np.isfinite(np.asarray(arrays[name])).all()
        for name, (shape, dtype) in expected_arrays.items()),
        "input_capture": np.array_equal(arrays["captured_x0_float32"], request_arrays["x0_float32"])
        and np.array_equal(arrays["captured_controls_float32"], request_arrays["controls_float32"]),
        "array_hashes": summary["array_names"] == sorted(arrays)
        and summary["array_hashes"] == array_hashes,
        "side_files": request_file_gate,
    }
    recomputed = {"gates": gates, "passes": bool(all(gates.values()))}
    stored_equal = summary["certificate"] == recomputed
    return {**recomputed, "stored_certificate_equal": stored_equal,
            "passes": bool(recomputed["passes"] and stored_equal)}


def execute_worker(request_path, output_path, *, authorization=None):  # pragma: no cover
    if WORKER_EXECUTION_AUTHORIZATION is None or authorization is not WORKER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("circular portfolio CUDA worker is blocked")
    if FROZEN_EXTENSION_SHA256 is None or FROZEN_EXTENSION_SIZE_BYTES is None:
        raise RuntimeError("N96 CUDA extension has not passed audited build pinning")
    request_path = Path(request_path).resolve(); output_path = Path(output_path).resolve()
    identity_token = output_path.stem.removeprefix("p1.worker.")
    if not identity_token.isdecimal() or len(identity_token) != 3:
        raise RuntimeError("worker identity token is not canonical")
    index = int(identity_token)
    paths = expected_worker_paths(index)
    if request_path != paths["request_path"]:
        raise RuntimeError("worker request path is not ledger-derived")
    if output_path != paths["json_path"]:
        raise RuntimeError("worker output path is not identity-derived")
    npz_path = paths["npz_path"]
    if output_path.exists() or npz_path.exists():
        raise FileExistsError("worker output overwrite is forbidden")
    request = json.loads(request_path.read_text())
    expected_request_keys = {
        "protocol", "identity", "input_path", "input_sha256", "module_name",
        "module_path", "extension_sha256", "extension_size_bytes", "output_path",
    }
    if (set(request) != expected_request_keys or request["protocol"] != WORKER_PROTOCOL
            or request["module_name"] != MODULE_NAME
            or request["extension_sha256"] != FROZEN_EXTENSION_SHA256
            or request["extension_size_bytes"] != FROZEN_EXTENSION_SIZE_BYTES
            or request["identity"] != list(EXPECTED_LEDGER[index])
            or Path(request["input_path"]).resolve() != paths["input_path"]
            or Path(request["output_path"]).resolve() != output_path):
        raise ValueError("worker request identity/schema mismatch")
    input_path = Path(request["input_path"]).resolve()
    if sha256_file(input_path) != request["input_sha256"]:
        raise RuntimeError("worker input hash mismatch")
    with np.load(input_path, allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    if not request_array_schema(inputs):
        raise ValueError("worker input schema invalid")
    module = importlib.import_module(MODULE_NAME)
    module_path = Path(module.__file__).resolve()
    if sha256_file(module_path) != FROZEN_EXTENSION_SHA256:
        raise RuntimeError("worker extension hash mismatch")
    constructor_calls = sim_forward_calls = tool_position_calls = 0
    solver = module.BSQP_16_float(
        DT, 0, KKT_TOL, MAX_PCG_ITERS, PCG_TOL, SOLVE_RATIO, MU,
        Q_COST, QD_COST, U_COST, N_COST, CYLINDER_WEIGHT, CYLINDER_WEIGHT,
        Q_LIMIT_COST, VELOCITY_LIMIT_COST, CONTROL_LIMIT_COST, RHO,
    )
    constructor_calls += 1
    state = np.empty((DENSE_SAMPLES, 14), np.float32); state[0] = inputs["x0_float32"]
    for interval in range(INTERVALS):
        for substep in range(DENSE_SUBSTEPS):
            state_index = interval * DENSE_SUBSTEPS + substep
            batch_x = np.repeat(state[state_index:state_index+1], 16, axis=0)
            batch_u = np.repeat(inputs["controls_float32"][interval:interval+1], 16, axis=0)
            state[state_index + 1] = solver.sim_forward(batch_x, batch_u, DT / DENSE_SUBSTEPS)[0]
            sim_forward_calls += 1
    padded = np.concatenate([state[:, :7], np.repeat(state[-1:, :7], (-len(state)) % 16, axis=0)])
    tool_chunks = []
    for k in range(0, len(padded), 16):
        tool_chunks.append(solver.tool_position(padded[k:k+16]))
        tool_position_calls += 1
    tool = np.concatenate(tool_chunks)[:DENSE_SAMPLES]
    arrays = {
        "captured_x0_float32": inputs["x0_float32"],
        "captured_controls_float32": inputs["controls_float32"],
        "cuda_dense_state_float32": state,
        "cuda_dense_tool_float32": np.asarray(tool, np.float32),
    }
    with npz_path.open("xb") as stream:
        np.savez(stream, **arrays)
    module_path = Path(module.__file__).resolve()
    summary = {
        "protocol": WORKER_PROTOCOL, "identity": request["identity"],
        "request_path": str(request_path), "request_sha256": sha256_file(request_path),
        "input_path": str(input_path), "input_sha256": sha256_file(input_path),
        "module_name": MODULE_NAME, "module_path": str(module_path),
        "extension_sha256": FROZEN_EXTENSION_SHA256,
        "extension_size_bytes": FROZEN_EXTENSION_SIZE_BYTES,
        "KNOT_POINTS": int(module.KNOT_POINTS), "REFERENCE_SIZE": int(module.REFERENCE_SIZE),
        "TOOL_POSITION_FRAME": str(module.TOOL_POSITION_FRAME),
        "TOOL_POSITION_SIZE": int(module.TOOL_POSITION_SIZE),
        "constructor_calls": constructor_calls, "sim_forward_calls": sim_forward_calls,
        "tool_position_calls": tool_position_calls, "solve_calls": 0,
        "sqp_calls": 0, "array_names": sorted(arrays),
        "array_hashes": {name: hashlib.sha256(
            f"{value.dtype.str}|{value.shape}|".encode()+value.tobytes()).hexdigest()
            for name, value in arrays.items()},
        "npz_path": str(npz_path), "npz_sha256": sha256_file(npz_path),
        "certificate": {},
    }
    first = certify_worker_output(summary, arrays, request, inputs)
    summary["certificate"] = {"gates": first["gates"],
                              "passes": bool(all(first["gates"].values()))}
    if not certify_worker_output(summary, arrays, request, inputs)["passes"]:
        raise RuntimeError("worker output failed pre-publication certification")
    with output_path.open("x") as stream:
        json.dump(summary, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    return summary
