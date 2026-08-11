"""Isolated per-extension worker for Tiago toll V4 model preflight.

Importing multiple GATO extension modules in one process can collide in
pybind11's class registry.  The audited runner therefore launches one fresh
process per module.  Execution is intentionally disabled in this checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import argparse
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from gato_tiago import multimodal_toll as toll
from gato_tiago.multimodal_toll_v4_model_preflight import (
    AUTHORIZED_OUTPUT_PATH,
    BROAD_ROW_COUNT,
    DENSE_SUBSTEPS,
    FROZEN_EXTENSIONS,
    MODEL_PREFLIGHT_PROTOCOL_VERSION,
    ONE_STEP_DT,
    array_hash as _array_hash,
    is_sha256 as _is_sha256,
)


WORKER_EXECUTION_AUTHORIZATION = None
WORKER_PROTOCOL_VERSION = MODEL_PREFLIGHT_PROTOCOL_VERSION + "_worker_1"
AUTHORIZED_RUN_ROOT = AUTHORIZED_OUTPUT_PATH.parent
SUPPORTED_MODULES = {
    row["module_name"]: (row["reference_size"], row["variant"])
    for row in FROZEN_EXTENSIONS
}
MERIT_ABS_TOLERANCE = 1e-5
MERIT_REL_TOLERANCE = 1e-6
MIN_REFERENCE_PERTURBATION_MERIT_DELTA = 0.01
EXPECTED_WORKER_ARRAY_NAMES = frozenset(
    {
        "captured_public_q0_float32",
        "captured_quarantined_q8_float32",
        "cuda_q0_fk_b1_float32",
        "cuda_q0_fk_b16_float32",
        "cuda_q8_fk_b1_float32",
        "cuda_q8_fk_b16_float32",
        "captured_one_step_x_float32",
        "captured_one_step_u_float32",
        "cuda_one_step_b1_float32",
        "cuda_one_step_b16_float32",
        "captured_broad_x_float32",
        "captured_broad_q_float32",
        "captured_broad_u_float32",
        "cuda_broad_one_step_b1_float32",
        "cuda_broad_one_step_b16_float32",
        "cuda_broad_fk_b1_float32",
        "cuda_broad_fk_b16_float32",
        "captured_dense_x0_float32",
        "captured_dense_u_float32",
        "cuda_dense_states_float32",
        "cuda_dense_tool_positions_float32",
        "dense_time_float64",
        "dense_controls_float32",
        "reference_smoke_input_xu_b1",
        "reference_smoke_output_xu_b1",
        "reference_smoke_input_xu_b16",
        "reference_smoke_output_xu_b16",
        "reference_smoke_initial_merit_b1",
        "reference_smoke_final_merit_b1",
        "reference_smoke_initial_merit_b16",
        "reference_smoke_final_merit_b16",
        "reference_smoke_sqp_iters_b1",
        "reference_smoke_sqp_iters_b16",
        "reference_smoke_pcg_iters_b1",
        "reference_smoke_pcg_iters_b16",
        "reference_smoke_ls_num_iters_b1",
        "reference_smoke_ls_num_iters_b16",
        "reference_smoke_baseline_reference",
        "reference_smoke_perturbed_reference",
    }
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path, value):
    _atomic_write_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    )


def _atomic_write_npz(path, arrays):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
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


def validate_worker_request(request):
    if not isinstance(request, Mapping):
        raise ValueError("worker request must be a mapping")
    if set(request) != {
        "module_name",
        "extension_path",
        "extension_sha256",
        "input_npz",
        "input_npz_sha256",
        "reference_size",
        "protocol_version",
    }:
        raise ValueError("worker request schema mismatch")
    module_name = request.get("module_name")
    if module_name not in SUPPORTED_MODULES:
        raise ValueError("worker module is not in the frozen three-module set")
    reference_size, variant = SUPPORTED_MODULES[module_name]
    module_path = Path(request.get("extension_path", ""))
    input_npz = Path(request.get("input_npz", ""))
    if not module_path.is_file() or not input_npz.is_file():
        raise FileNotFoundError("worker extension/input artifact is missing")
    if not _is_sha256(request.get("extension_sha256")) or sha256_file(
        module_path
    ) != request["extension_sha256"]:
        raise ValueError("worker extension hash mismatch")
    if not _is_sha256(request.get("input_npz_sha256")) or sha256_file(
        input_npz
    ) != request["input_npz_sha256"]:
        raise ValueError("worker input hash mismatch")
    if request.get("reference_size") != reference_size:
        raise ValueError("worker reference width mismatch")
    if request.get("protocol_version") != WORKER_PROTOCOL_VERSION:
        raise ValueError("worker protocol mismatch")
    return {
        "module_name": module_name,
        "extension_path": module_path,
        "input_npz": input_npz,
        "reference_size": reference_size,
        "variant": variant,
    }


def reference_rows(variant, tool_xyz):
    """Construct baseline/perturbed native rows from actual worker CUDA FK."""

    tool = np.asarray(tool_xyz, dtype=np.float32)
    if tool.shape != (3,) or not np.all(np.isfinite(tool)):
        raise ValueError("tool_xyz must be one finite float32-compatible row")
    if variant == "toll":
        far = tool[:2] + np.asarray([1.0, 1.0], dtype=np.float32)
        baseline = np.asarray(
            [*tool, *far, 0.030, *far, 0.020, 0.005], dtype=np.float32
        )
        perturbed = baseline.copy()
        perturbed[6:8] = tool[:2]
        changed_fields = (6, 7)
    elif variant == "plain_ref6":
        baseline = np.asarray([*tool, 0.0, 0.0, 0.0], dtype=np.float32)
        perturbed = baseline.copy()
        perturbed[0] += np.float32(0.10)
        changed_fields = (0,)
    elif variant == "pillar_ref6":
        far = tool[:2] + np.asarray([1.0, 1.0], dtype=np.float32)
        baseline = np.asarray([*tool, *far, 0.030], dtype=np.float32)
        perturbed = baseline.copy()
        perturbed[0] += np.float32(0.10)
        perturbed[3:5] = tool[:2]
        changed_fields = (0, 3, 4)
    else:
        raise ValueError("unknown frozen worker variant")
    return baseline, perturbed, changed_fields


def _merit_close(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return bool(
        np.all(np.isfinite(left))
        and np.all(np.isfinite(right))
        and np.allclose(
            left, right, atol=MERIT_ABS_TOLERANCE, rtol=MERIT_REL_TOLERANCE
        )
    )


def certify_reference_smoke(raw, *, variant, reference_size):
    """Pure certificate for exactly two max-SQP-zero diagnostic calls."""

    required = {
        "input_xu_b1",
        "output_xu_b1",
        "input_xu_b16",
        "output_xu_b16",
        "initial_merit_b1",
        "final_merit_b1",
        "initial_merit_b16",
        "final_merit_b16",
        "sqp_iters_b1",
        "sqp_iters_b16",
        "pcg_iters_b1",
        "pcg_iters_b16",
        "ls_num_iters_b1",
        "ls_num_iters_b16",
        "baseline_reference",
        "perturbed_reference",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        return {"all_reference_smoke_gates_pass": False, "schema_exact": False}
    try:
        b1_in = np.asarray(raw["input_xu_b1"])
        b1_out = np.asarray(raw["output_xu_b1"])
        b16_in = np.asarray(raw["input_xu_b16"])
        b16_out = np.asarray(raw["output_xu_b16"])
        im1 = np.asarray(raw["initial_merit_b1"])
        fm1 = np.asarray(raw["final_merit_b1"])
        im16 = np.asarray(raw["initial_merit_b16"])
        fm16 = np.asarray(raw["final_merit_b16"])
        sqp1 = np.asarray(raw["sqp_iters_b1"])
        sqp16 = np.asarray(raw["sqp_iters_b16"])
        pcg1 = np.asarray(raw["pcg_iters_b1"])
        pcg16 = np.asarray(raw["pcg_iters_b16"])
        baseline = np.asarray(raw["baseline_reference"])
        perturbed = np.asarray(raw["perturbed_reference"])
        shapes = (
            b1_in.shape == b1_out.shape == (1, 1337)
            and b1_in.dtype == b1_out.dtype == np.float32
            and b16_in.shape == b16_out.shape == (16, 1337)
            and b16_in.dtype == b16_out.dtype == np.float32
            and im1.shape == fm1.shape == (1,)
            and im1.dtype == fm1.dtype == np.float32
            and im16.shape == fm16.shape == (16,)
            and im16.dtype == fm16.dtype == np.float32
            and sqp1.shape == (1,)
            and sqp1.dtype == np.int32
            and sqp16.shape == (16,)
            and sqp16.dtype == np.int32
            and pcg1.shape == (0, 1)
            and pcg1.dtype == np.int32
            and pcg16.shape == (0, 16)
            and pcg16.dtype == np.int32
            and baseline.shape == perturbed.shape == (reference_size,)
            and baseline.dtype == perturbed.dtype == np.float32
            and isinstance(raw["ls_num_iters_b1"], int)
            and not isinstance(raw["ls_num_iters_b1"], bool)
            and isinstance(raw["ls_num_iters_b16"], int)
            and not isinstance(raw["ls_num_iters_b16"], bool)
        )
    except (KeyError, TypeError, ValueError, IndexError):
        shapes = False
    if not shapes:
        return {"all_reference_smoke_gates_pass": False, "schema_exact": True}
    finite_domains = all(
        np.all(np.isfinite(array))
        for array in (b1_in, b1_out, b16_in, b16_out, im1, fm1, im16, fm16, baseline, perturbed)
    )
    unchanged = finite_domains and np.array_equal(b1_in, b1_out) and np.array_equal(b16_in, b16_out)
    zero_telemetry = bool(
        np.all(sqp1 == 0)
        and np.all(sqp16 == 0)
        and pcg1.size == 0
        and pcg16.size == 0
        and raw["ls_num_iters_b1"] == 0
        and raw["ls_num_iters_b16"] == 0
    )
    initial_final = _merit_close(im1, fm1) and _merit_close(im16, fm16)
    repeated_baseline = _merit_close(
        im16[:8], np.repeat(im1, 8)
    ) and _merit_close(fm16[:8], np.repeat(fm1, 8))
    repeated_perturbed = _merit_close(
        im16[8:], np.repeat(im16[8:9], 8)
    ) and _merit_close(fm16[8:], np.repeat(fm16[8:9], 8))
    perturb_delta = np.abs(im16[8:] - im16[:8])
    material = bool(
        np.all(np.isfinite(perturb_delta))
        and np.min(perturb_delta) >= MIN_REFERENCE_PERTURBATION_MERIT_DELTA
    )
    changed = tuple(np.flatnonzero(baseline != perturbed).tolist())
    expected_changed = {
        "toll": (6, 7),
        "plain_ref6": (0,),
        "pillar_ref6": (0, 3, 4),
    }[variant]
    result = {
        "schema_exact": True,
        "diagnostic_zero_iteration_solve_calls": 2,
        "sqp_optimization_calls": 0,
        "optimization_evidence": False,
        "xu_bytes_unchanged": unchanged,
        "all_numeric_domains_finite": finite_domains,
        "zero_iteration_telemetry": zero_telemetry,
        "initial_final_merit_tolerance": initial_final,
        "repeated_lane_merit_tolerance": repeated_baseline,
        "repeated_perturbed_lane_merit_tolerance": repeated_perturbed,
        "native_reference_materially_consumed": material,
        "changed_reference_fields_exact": changed == expected_changed,
        "raw_initial_merit_b1": im1.tolist(),
        "raw_initial_merit_b16": im16.tolist(),
        "raw_final_merit_b1": fm1.tolist(),
        "raw_final_merit_b16": fm16.tolist(),
        "input_xu_b1_sha256": _array_hash(b1_in),
        "output_xu_b1_sha256": _array_hash(b1_out),
        "input_xu_b16_sha256": _array_hash(b16_in),
        "output_xu_b16_sha256": _array_hash(b16_out),
    }
    result["all_reference_smoke_gates_pass"] = all(
        result[name]
        for name in (
            "xu_bytes_unchanged",
            "all_numeric_domains_finite",
            "zero_iteration_telemetry",
            "initial_final_merit_tolerance",
            "repeated_lane_merit_tolerance",
            "repeated_perturbed_lane_merit_tolerance",
            "native_reference_materially_consumed",
            "changed_reference_fields_exact",
        )
    )
    return result


def _pack_stationary_trajectory(q):
    q = np.asarray(q, dtype=np.float32)
    if q.shape != (7,):
        raise ValueError("stationary diagnostic q must have shape (7,)")
    packed = np.zeros(1337, dtype=np.float32)
    for knot in range(toll.KNOTS):
        offset = knot * 21
        packed[offset : offset + 7] = q
    return packed


def _solver(module, batch_size, variant):
    cls = getattr(module, f"BSQP_{batch_size}_float")
    obstacle_weight = 0.0 if variant == "plain_ref6" else toll.CYLINDER_WEIGHT
    return cls(
        toll.DT,
        0,
        toll.KKT_TOL,
        toll.MAX_PCG_ITERS,
        toll.PCG_TOL,
        toll.SOLVE_RATIO,
        toll.MU,
        toll.Q_COST,
        toll.QD_COST,
        toll.U_COST,
        toll.N_COST,
        obstacle_weight,
        obstacle_weight,
        toll.Q_LIMIT_COST,
        toll.VELOCITY_LIMIT_COST,
        toll.CONTROL_LIMIT_COST,
        toll.RHO,
    )


def _run_authorized_worker(request_path, output_path):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text())
    request_sha256 = sha256_file(request_path)
    validated = validate_worker_request(request)
    with np.load(validated["input_npz"], allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    expected_inputs = {
        "public_q0_float32",
        "quarantined_q8_float32",
        "one_step_x_float32",
        "one_step_u_float32",
        "dense_x0_float32",
        "dense_u_float32",
        "broad_x_float32",
        "broad_u_float32",
    }
    if set(inputs) != expected_inputs:
        raise ValueError("worker input array-name schema mismatch")

    public_q0 = np.asarray(inputs["public_q0_float32"])
    quarantined_q8 = np.asarray(inputs["quarantined_q8_float32"])
    one_step_x = np.asarray(inputs["one_step_x_float32"])
    one_step_u = np.asarray(inputs["one_step_u_float32"])
    dense_x0 = np.asarray(inputs["dense_x0_float32"])
    dense_u = np.asarray(inputs["dense_u_float32"])
    broad_x = np.asarray(inputs["broad_x_float32"])
    broad_u = np.asarray(inputs["broad_u_float32"])
    exact_inputs = (
        public_q0.shape == quarantined_q8.shape == (12, 7)
        and public_q0.dtype == quarantined_q8.dtype == np.float32
        and one_step_x.shape == (24, 14)
        and one_step_u.shape == (24, 7)
        and one_step_x.dtype == one_step_u.dtype == np.float32
        and dense_x0.shape == (12, 14)
        and dense_u.shape == (12, 7)
        and dense_x0.dtype == dense_u.dtype == np.float32
        and broad_x.shape == (BROAD_ROW_COUNT, 14)
        and broad_u.shape == (BROAD_ROW_COUNT, 7)
        and broad_x.dtype == broad_u.dtype == np.float32
        and all(
            np.all(np.isfinite(value))
            for value in (
                public_q0,
                quarantined_q8,
                one_step_x,
                one_step_u,
                dense_x0,
                dense_u,
                broad_x,
                broad_u,
            )
        )
    )
    if not exact_inputs:
        raise ValueError("worker model-preflight inputs have invalid shape/dtype/domain")

    module = importlib.import_module(validated["module_name"])
    if (
        Path(module.__file__).resolve() != validated["extension_path"].resolve()
        or module.KNOT_POINTS != 64
        or module.REFERENCE_SIZE != validated["reference_size"]
        or module.TOOL_POSITION_FRAME != "arm_right_tool_joint_origin"
        or module.TOOL_POSITION_SIZE != 3
    ):
        raise RuntimeError("imported extension identity mismatch")
    b1 = _solver(module, 1, validated["variant"])
    b16 = _solver(module, 16, validated["variant"])

    def tool_b1(rows):
        return np.stack([np.asarray(b1.tool_position(row))[0] for row in rows])

    def tool_b16(rows):
        padded = np.concatenate(
            [rows, np.repeat(rows[-1:], (-len(rows)) % 16, axis=0)]
        )
        values = np.concatenate(
            [
                np.asarray(b16.tool_position(padded[start : start + 16]))
                for start in range(0, len(padded), 16)
            ]
        )
        return values[: len(rows)]

    cuda_q0_b1 = tool_b1(public_q0)
    cuda_q0_b16 = tool_b16(public_q0)
    # q8 is quarantined here: it is used only by tool_position, never by
    # sim_forward, solve, trajectory packing, initializer, or solver seed code.
    cuda_q8_b1 = tool_b1(quarantined_q8)
    cuda_q8_b16 = tool_b16(quarantined_q8)

    cuda_one_b1 = np.stack(
        [
            np.asarray(b1.sim_forward(x, u, ONE_STEP_DT))[0]
            for x, u in zip(one_step_x, one_step_u)
        ]
    )
    padded_x = np.concatenate([one_step_x, np.repeat(one_step_x[-1:], 8, axis=0)])
    padded_u = np.concatenate([one_step_u, np.repeat(one_step_u[-1:], 8, axis=0)])
    cuda_one_b16 = np.concatenate(
        [
            np.asarray(
                b16.sim_forward(
                    padded_x[start : start + 16],
                    padded_u[start : start + 16],
                    ONE_STEP_DT,
                )
            )
            for start in (0, 16)
        ]
    )[:24]
    cuda_broad_b1 = np.stack(
        [
            np.asarray(b1.sim_forward(x, u, ONE_STEP_DT))[0]
            for x, u in zip(broad_x, broad_u)
        ]
    )
    cuda_broad_b16 = np.concatenate(
        [
            np.asarray(
                b16.sim_forward(
                    broad_x[start : start + 16],
                    broad_u[start : start + 16],
                    ONE_STEP_DT,
                )
            )
            for start in (0, 16)
        ]
    )
    cuda_broad_fk_b1 = tool_b1(broad_x[:, :7])
    cuda_broad_fk_b16 = tool_b16(broad_x[:, :7])

    dense_states = np.empty((12, 65, 14), dtype=np.float32)
    dense_tools = np.empty((12, 65, 3), dtype=np.float32)
    dense_states[:, 0] = dense_x0
    for step in range(64):
        padded_state = np.concatenate(
            [dense_states[:, step], np.repeat(dense_states[-1:, step], 4, axis=0)]
        )
        padded_control = np.concatenate(
            [dense_u, np.repeat(dense_u[-1:], 4, axis=0)]
        )
        dense_states[:, step + 1] = np.asarray(
            b16.sim_forward(
                padded_state, padded_control, ONE_STEP_DT / DENSE_SUBSTEPS
            )
        )[:12]
    for step in range(65):
        dense_tools[:, step] = tool_b16(dense_states[:, step, :7])

    q_smoke = public_q0[0]
    tool_smoke = cuda_q0_b1[0]
    baseline, perturbed, _ = reference_rows(validated["variant"], tool_smoke)
    xu = _pack_stationary_trajectory(q_smoke)
    x0 = np.concatenate([q_smoke, np.zeros(7, dtype=np.float32)])
    ref1 = np.tile(baseline, toll.KNOTS)[None, :]
    refs16 = np.empty((16, toll.KNOTS * validated["reference_size"]), np.float32)
    refs16[:8] = np.tile(baseline, toll.KNOTS)
    refs16[8:] = np.tile(perturbed, toll.KNOTS)
    xu1 = xu[None, :].copy()
    xu16 = np.tile(xu, (16, 1))
    out1 = b1.solve(xu1.copy(), toll.DT, x0[None, :], ref1)
    out16 = b16.solve(xu16.copy(), toll.DT, np.tile(x0, (16, 1)), refs16)
    wrong_width = 6 if validated["reference_size"] == 10 else 10
    wrong_rejected = False
    wrong_message = None
    try:
        b1.solve(
            xu1.copy(),
            toll.DT,
            x0[None, :],
            np.zeros((1, toll.KNOTS * wrong_width), dtype=np.float32),
        )
    except ValueError as error:
        wrong_rejected = True
        wrong_message = str(error)
    expected_wrong_message = (
        "reference input must have shape (batch_size, "
        f"{toll.KNOTS * validated['reference_size']}) for this plant"
    )
    raw_smoke = {
        "input_xu_b1": xu1,
        "output_xu_b1": np.asarray(out1["XU"]),
        "input_xu_b16": xu16,
        "output_xu_b16": np.asarray(out16["XU"]),
        "initial_merit_b1": np.asarray(out1["initial_merit"]),
        "final_merit_b1": np.asarray(out1["final_merit"]),
        "initial_merit_b16": np.asarray(out16["initial_merit"]),
        "final_merit_b16": np.asarray(out16["final_merit"]),
        "sqp_iters_b1": np.asarray(out1["sqp_iters"]),
        "sqp_iters_b16": np.asarray(out16["sqp_iters"]),
        "pcg_iters_b1": np.asarray(out1["pcg_iters"]),
        "pcg_iters_b16": np.asarray(out16["pcg_iters"]),
        "ls_num_iters_b1": int(out1["ls_num_iters"]),
        "ls_num_iters_b16": int(out16["ls_num_iters"]),
        "baseline_reference": baseline,
        "perturbed_reference": perturbed,
    }
    certificate = certify_reference_smoke(
        raw_smoke,
        variant=validated["variant"],
        reference_size=validated["reference_size"],
    )
    certificate.update(
        {
            "wrong_width_rejected_before_native_kernel": wrong_rejected,
            "wrong_width_value_error": wrong_message,
            "wrong_width_value_error_exact": wrong_message == expected_wrong_message,
            "wrong_width_rejection_calls": 1,
        }
    )
    certificate["all_reference_smoke_gates_pass"] = bool(
        certificate["all_reference_smoke_gates_pass"]
        and wrong_rejected
        and certificate["wrong_width_value_error_exact"]
    )

    arrays = {
        "captured_public_q0_float32": public_q0,
        "captured_quarantined_q8_float32": quarantined_q8,
        "cuda_q0_fk_b1_float32": cuda_q0_b1,
        "cuda_q0_fk_b16_float32": cuda_q0_b16,
        "cuda_q8_fk_b1_float32": cuda_q8_b1,
        "cuda_q8_fk_b16_float32": cuda_q8_b16,
        "captured_one_step_x_float32": one_step_x,
        "captured_one_step_u_float32": one_step_u,
        "cuda_one_step_b1_float32": cuda_one_b1,
        "cuda_one_step_b16_float32": cuda_one_b16,
        "captured_broad_x_float32": broad_x,
        "captured_broad_q_float32": broad_x[:, :7].copy(),
        "captured_broad_u_float32": broad_u,
        "cuda_broad_one_step_b1_float32": cuda_broad_b1,
        "cuda_broad_one_step_b16_float32": cuda_broad_b16,
        "cuda_broad_fk_b1_float32": cuda_broad_fk_b1,
        "cuda_broad_fk_b16_float32": cuda_broad_fk_b16,
        "captured_dense_x0_float32": dense_x0,
        "captured_dense_u_float32": dense_u,
        "cuda_dense_states_float32": dense_states,
        "cuda_dense_tool_positions_float32": dense_tools,
        "dense_time_float64": np.arange(
            DENSE_SUBSTEPS + 1, dtype=np.float64
        )
        * (ONE_STEP_DT / DENSE_SUBSTEPS),
        "dense_controls_float32": np.repeat(dense_u[:, None, :], 64, axis=1),
    }
    for name, value in raw_smoke.items():
        if isinstance(value, np.ndarray):
            arrays[f"reference_smoke_{name}"] = value
    arrays["reference_smoke_ls_num_iters_b1"] = np.asarray(
        raw_smoke["ls_num_iters_b1"], dtype=np.int32
    )
    arrays["reference_smoke_ls_num_iters_b16"] = np.asarray(
        raw_smoke["ls_num_iters_b16"], dtype=np.int32
    )
    if set(arrays) != EXPECTED_WORKER_ARRAY_NAMES:
        raise RuntimeError("internal worker output array-name schema mismatch")
    output = Path(output_path)
    npz_path = output.with_suffix(".npz")
    _atomic_write_npz(npz_path, arrays)
    summary = {
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "model_preflight_protocol_version": MODEL_PREFLIGHT_PROTOCOL_VERSION,
        "request_path": str(request_path),
        "request_sha256": request_sha256,
        "input_npz_path": str(validated["input_npz"].resolve()),
        "input_npz_sha256": request["input_npz_sha256"],
        "module_name": validated["module_name"],
        "module_file": str(Path(module.__file__).resolve()),
        "extension_sha256": request["extension_sha256"],
        "reference_size": validated["reference_size"],
        "subprocess_isolation_required": True,
        "q8_scope": "tool_position_only",
        "q8_initializer_eligible": False,
        "q8_solver_seed_eligible": False,
        "reference_smoke": certificate,
        "optimization_evidence": False,
        "sqp_optimization_calls": 0,
        "worker_npz": str(npz_path),
        "worker_npz_sha256": sha256_file(npz_path),
        "array_hashes": {name: _array_hash(value) for name, value in arrays.items()},
        "array_names": sorted(arrays),
    }
    _atomic_write_json(output, summary)
    return summary


def execute_worker(request_path, output_path, *, authorization=None):
    if (
        WORKER_EXECUTION_AUTHORIZATION is None
        or authorization is not WORKER_EXECUTION_AUTHORIZATION
    ):
        raise RuntimeError("V4 model preflight extension worker execution is blocked")
    request_path = Path(request_path).resolve()
    output_path = Path(output_path).resolve()
    allowed_pairs = {
        (
            AUTHORIZED_RUN_ROOT / f"model.{name.split('.')[-1]}.request.json",
            AUTHORIZED_RUN_ROOT / f"model.{name.split('.')[-1]}.json",
        )
        for name in SUPPORTED_MODULES
    }
    if (request_path, output_path) not in allowed_pairs:
        raise RuntimeError("V4 model preflight worker path is outside the single authorized run")
    if output_path.exists() or output_path.with_suffix(".npz").exists():
        raise RuntimeError("V4 model preflight worker refuses overwrite or rerun")
    return _run_authorized_worker(request_path, output_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # This check intentionally precedes request reads and extension imports.
    if not args.execute or WORKER_EXECUTION_AUTHORIZATION is None:
        raise SystemExit("V4 model preflight extension worker execution is blocked")
    execute_worker(
        args.request,
        args.output,
        authorization=WORKER_EXECUTION_AUTHORIZATION,
    )
    return 0


if __name__ == "__main__":
    main()
