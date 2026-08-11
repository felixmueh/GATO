"""Transactional artifact-only V4 Tiago model-preflight runner.

All execution capabilities are disabled in this static checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import traceback
from typing import Callable, Mapping, Sequence

import numpy as np

from gato_tiago import multimodal_toll as toll
from gato_tiago.multimodal_toll_v4_model_preflight import (
    AUTHORIZED_OUTPUT_PATH,
    BROAD_ROW_COUNT,
    BROAD_CONTROL_FRACTION,
    BROAD_Q_RESERVE_RAD,
    BROAD_VELOCITY_FRACTION,
    DENSE_TOOL_POSITION_TOLERANCE_M,
    DENSE_SUBSTEPS,
    EXPECTED_SQP_OPTIMIZATION_CALLS,
    EXPECTED_NONWORKER_ARRAY_NAMES,
    EXPECTED_TASK_COUNT,
    EXPECTED_TASK_IDENTITIES,
    EXPECTED_V4_ARRAY_COUNT,
    EXPECTED_WORKER_COUNT,
    FROZEN_EXTENSIONS,
    FROZEN_BUILD_COMMIT,
    FROZEN_BUILD_SOURCE_HASHES,
    FROZEN_CUDA_ARCH,
    FK_PIN_CUDA_TOLERANCE_M,
    MODEL_PREFLIGHT_PROTOCOL_VERSION,
    ONE_STEP_DT,
    ONE_STEP_STATE_L2_TOLERANCE,
    ONE_STEP_STATE_MAXABS_TOLERANCE,
    Q8_PUBLIC_GOAL_TOLERANCE_M,
    REQUIRED_SOURCE_PATHS,
    V4_ARTIFACT_PATHS_AND_HASHES,
    array_hash,
    frozen_model_preflight_metadata,
    generate_broad_algebraic_rows,
    is_sha256,
)
from gato_tiago.multimodal_toll_v4_model_preflight_worker import (
    EXPECTED_WORKER_ARRAY_NAMES,
    SUPPORTED_MODULES,
    WORKER_EXECUTION_AUTHORIZATION,
    WORKER_PROTOCOL_VERSION,
    certify_reference_smoke,
)


RUNNER_EXECUTION_AUTHORIZATION = None
RUNNER_PROTOCOL_VERSION = MODEL_PREFLIGHT_PROTOCOL_VERSION + "_runner_1"
_PRODUCTION_PIPELINE_TOKEN = object()


def repository_root(module_path=Path(__file__)) -> Path:
    root = Path(module_path).resolve().parents[2]
    sentinels = (
        root / ".git",
        root / "CMakeLists.txt",
        root / REQUIRED_SOURCE_PATHS["preflight_schema"],
        root / REQUIRED_SOURCE_PATHS["preflight_runner"],
        root / REQUIRED_SOURCE_PATHS["preflight_worker"],
    )
    if not all(path.exists() for path in sentinels):
        raise RuntimeError("V4 model-preflight repository root sentinel mismatch")
    return root


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path, payload):
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


def _atomic_json(path, value):
    _atomic_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    )


def _atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _no_existing_artifacts(output):
    output = Path(output)
    candidates = [
        output,
        output.with_suffix(".npz"),
        output.with_suffix(".manifest.json"),
        output.with_name(f"{output.stem}.partial.latest.json"),
        *output.parent.glob(f"{output.stem}.partial.*"),
        *output.parent.glob(f"{output.stem}.*.request.json"),
        *output.parent.glob(f"{output.stem}.*.json"),
        *output.parent.glob(f"{output.stem}.*.npz"),
    ]
    if any(path.exists() for path in candidates):
        raise RuntimeError("V4 model preflight refuses resume, overwrite, or rerun")


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _checkpoint(output, generation, stage, rows, arrays, provenance):
    output = Path(output)
    npz_path = output.with_name(
        f"{output.stem}.partial.{generation:04d}.npz"
    )
    json_path = output.with_name(
        f"{output.stem}.partial.{generation:04d}.json"
    )
    pointer_path = output.with_name(f"{output.stem}.partial.latest.json")
    if npz_path.exists() or json_path.exists():
        raise RuntimeError("V4 model-preflight checkpoint generation already exists")
    retained = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    _atomic_npz(npz_path, retained)
    summary = {
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "incomplete": True,
        "all_model_preflight_gates_pass": False,
        "stage": stage,
        "generation": generation,
        "completed_worker_count": len(rows),
        "pending_worker_modules": [
            row["module_name"] for row in FROZEN_EXTENSIONS[len(rows) :]
        ],
        "rows": _json_value(rows),
        "array_names": sorted(retained),
        "array_hashes": {name: array_hash(value) for name, value in retained.items()},
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "provenance": _json_value(provenance),
        "optimization_evidence": False,
        "sqp_optimization_calls": 0,
        "timing_evidence": False,
    }
    _atomic_json(json_path, summary)
    pointer = {
        "generation": generation,
        "json_path": str(json_path),
        "json_sha256": sha256_file(json_path),
        "npz_path": str(npz_path),
        "npz_sha256": summary["npz_sha256"],
    }
    _atomic_json(pointer_path, pointer)
    return pointer


def _source_provenance(repo):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    ).splitlines()
    if tracked:
        raise RuntimeError("V4 model preflight requires a tracked-clean tree")
    sources = {
        name: {"path": path, "sha256": sha256_file(repo / path)}
        for name, path in REQUIRED_SOURCE_PATHS.items()
    }
    build_source_hashes = {
        path: sha256_file(repo / path) for path in FROZEN_BUILD_SOURCE_HASHES
    }
    if build_source_hashes != FROZEN_BUILD_SOURCE_HASHES:
        raise RuntimeError("model preflight build sources differ from frozen build")
    return {
        "git_head_at_start": head,
        "git_head_at_end": head,
        "tracked_tree_clean_at_start": True,
        "tracked_tree_clean_at_end": True,
        "full_git_status_at_start": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).splitlines(),
        "full_git_status_at_end": [],
        "source_hashes": sources,
        "frozen_build_commit": FROZEN_BUILD_COMMIT,
        "frozen_build_source_hashes": dict(FROZEN_BUILD_SOURCE_HASHES),
        "current_build_source_hashes": build_source_hashes,
        "configured_cuda_arch": FROZEN_CUDA_ARCH,
        "exact_command": shlex.join(list(sys.orig_argv)),
        "orig_argv": list(sys.orig_argv),
        "sys_argv": list(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "critical_environment": {
            name: os.environ.get(name)
            for name in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")
        },
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "v4_artifact_only": True,
        "task_rng_calls": 0,
        "task_construction_calls": 0,
        "predecessor_artifact_loads": 0,
        "retry_count": 0,
        "resume_count": 0,
        "worker_retry_count": 0,
    }


def authenticate_v4_artifact():
    """Load only the four hard-pinned V4 final artifacts and exact 603 arrays."""

    for _, (path, digest) in V4_ARTIFACT_PATHS_AND_HASHES.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError("hard-pinned V4 artifact path/hash mismatch")
    final_path = V4_ARTIFACT_PATHS_AND_HASHES["final_json"][0]
    npz_path = V4_ARTIFACT_PATHS_AND_HASHES["final_npz"][0]
    manifest_path = V4_ARTIFACT_PATHS_AND_HASHES["final_manifest"][0]
    pointer_path = V4_ARTIFACT_PATHS_AND_HASHES["generation13_pointer"][0]
    summary = json.loads(final_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    pointer = json.loads(pointer_path.read_text())
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    exact_arrays = bool(
        len(arrays) == EXPECTED_V4_ARRAY_COUNT
        and set(arrays) == set(summary.get("array_names", ()))
        and set(arrays) == set(summary.get("array_hashes", ()))
        and all(
            array_hash(arrays[name]) == summary["array_hashes"][name]
            for name in arrays
        )
    )
    identities = [tuple(row.get("identity", ())) for row in summary.get("rows", ())]
    exact_metadata = bool(
        summary.get("all_v4_gates_pass") is True
        and summary.get("incomplete") is False
        and summary.get("task_construction_attempt_count") == EXPECTED_TASK_COUNT
        and identities == list(EXPECTED_TASK_IDENTITIES)
        and all(row.get("passes") is True for row in summary.get("rows", ()))
        and manifest.get("all_v4_gates_pass") is True
        and manifest.get("json_sha256")
        == V4_ARTIFACT_PATHS_AND_HASHES["final_json"][1]
        and manifest.get("npz_sha256")
        == V4_ARTIFACT_PATHS_AND_HASHES["final_npz"][1]
        and pointer.get("generation") == 13
    )
    if not exact_arrays or not exact_metadata:
        raise RuntimeError("V4 final artifact authenticity certificate failed")

    public_x0 = []
    public_reference = []
    public_default_side = []
    quarantined_q8 = []
    for row in summary["rows"]:
        task_seed = int(row["task_seed"])
        prefix = f"task_{task_seed}"
        x0 = arrays[f"{prefix}_x0_float32"]
        reference = arrays[f"{prefix}_reference_float32"]
        q_history = arrays[f"{prefix}_history_q_float64"]
        public = row["public_task"]
        if (
            x0.shape != (14,)
            or x0.dtype != np.float32
            or reference.shape != (10,)
            or reference.dtype != np.float32
            or q_history.shape != (9, 7)
            or q_history.dtype != np.float64
            or not np.array_equal(x0, np.asarray(public["x0_float32"], np.float32))
            or not np.array_equal(
                reference, np.asarray(public["reference_float32"], np.float32)
            )
            or public["default_side"] not in (-1, 1)
        ):
            raise RuntimeError("V4 public/quarantined task extraction mismatch")
        public_x0.append(x0)
        public_reference.append(reference)
        public_default_side.append(public["default_side"])
        quarantined_q8.append(q_history[8])
    extracted = {
        "public_solver_x0_float32": np.stack(public_x0),
        "public_reference_float32": np.stack(public_reference),
        "public_default_side_int8": np.asarray(public_default_side, dtype=np.int8),
        "quarantined_q8_float64": np.stack(quarantined_q8),
    }
    authentication = {
        "all_four_hard_pinned_hashes_match": True,
        "exact_603_arrays": exact_arrays,
        "final_metadata_exact": exact_metadata,
        "artifact_paths_and_hashes": {
            name: {"path": str(path), "sha256": digest}
            for name, (path, digest) in V4_ARTIFACT_PATHS_AND_HASHES.items()
        },
        "v4_array_count": len(arrays),
        "v4_array_names": sorted(arrays),
        "v4_array_hashes": dict(summary["array_hashes"]),
        "v4_task_identities": [list(row) for row in identities],
        "task_rng_calls": 0,
        "task_construction_calls": 0,
        "predecessor_artifact_loads": 0,
        "all_v4_artifact_authentication_gates_pass": True,
    }
    return extracted, authentication


def constant_acceleration_step(x, qdd, dt):
    x = np.asarray(x, dtype=np.float64)
    qdd = np.asarray(qdd, dtype=np.float64)
    q = x[:7] + dt * x[7:] + 0.5 * dt**2 * qdd
    qd = x[7:] + dt * qdd
    return np.concatenate([q, qd])


def certify_model_preflight(base, worker_rows, worker_arrays):
    if len(worker_rows) != EXPECTED_WORKER_COUNT:
        return {"all_model_preflight_gates_pass": False}
    module_names = [row.get("module_name") for row in worker_rows]
    if module_names != [row["module_name"] for row in FROZEN_EXTENSIONS]:
        return {"all_model_preflight_gates_pass": False}
    if {
        name for name in base if not name.startswith("worker_")
    } != EXPECTED_NONWORKER_ARRAY_NAMES:
        return {"all_model_preflight_gates_pass": False}

    expected_base_shapes = {
        "public_solver_x0_float32": ((12, 14), np.dtype(np.float32)),
        "public_reference_float32": ((12, 10), np.dtype(np.float32)),
        "public_default_side_int8": ((12,), np.dtype(np.int8)),
        "quarantined_q8_float64": ((12, 7), np.dtype(np.float64)),
        "quarantined_q8_float32": ((12, 7), np.dtype(np.float32)),
        "quarantined_q8_oracle_only_bool": ((12,), np.dtype(np.bool_)),
        "quarantined_q8_initializer_eligible_bool": ((12,), np.dtype(np.bool_)),
        "quarantined_q8_solver_seed_eligible_bool": ((12,), np.dtype(np.bool_)),
        "pin_q0_tool_float64": ((12, 3), np.dtype(np.float64)),
        "pin_q8_tool_float64": ((12, 3), np.dtype(np.float64)),
        "one_step_x_float32": ((24, 14), np.dtype(np.float32)),
        "one_step_u_float32": ((24, 7), np.dtype(np.float32)),
        "pin_one_step_float64": ((24, 14), np.dtype(np.float64)),
        "broad_q_codes_int8": ((32, 7), np.dtype(np.int8)),
        "broad_qd_codes_int8": ((32, 7), np.dtype(np.int8)),
        "broad_u_codes_int8": ((32, 7), np.dtype(np.int8)),
        "broad_x_float64": ((32, 14), np.dtype(np.float64)),
        "broad_u_float64": ((32, 7), np.dtype(np.float64)),
        "broad_x_float32": ((32, 14), np.dtype(np.float32)),
        "broad_u_float32": ((32, 7), np.dtype(np.float32)),
        "pin_broad_one_step_float64": ((32, 14), np.dtype(np.float64)),
        "pin_broad_tool_float64": ((32, 3), np.dtype(np.float64)),
        "dense_x0_float32": ((12, 14), np.dtype(np.float32)),
        "dense_u_float32": ((12, 7), np.dtype(np.float32)),
        "pin_dense_states_float64": ((12, 65, 14), np.dtype(np.float64)),
        "pin_dense_tool_positions_float64": ((12, 65, 3), np.dtype(np.float64)),
        "model_lower_float64": ((7,), np.dtype(np.float64)),
        "model_upper_float64": ((7,), np.dtype(np.float64)),
        "model_velocity_float64": ((7,), np.dtype(np.float64)),
        "model_effort_float64": ((7,), np.dtype(np.float64)),
        "forbidden_call_counts_int64": ((3,), np.dtype(np.int64)),
        "v4_artifact_authentication_gate_bool": ((), np.dtype(np.bool_)),
    }
    if any(
        name not in base
        or np.asarray(base[name]).shape != shape
        or np.asarray(base[name]).dtype != dtype
        for name, (shape, dtype) in expected_base_shapes.items()
    ):
        return {"all_model_preflight_gates_pass": False}
    if not all(np.all(np.isfinite(base[name])) for name in expected_base_shapes):
        return {"all_model_preflight_gates_pass": False}
    public_names = np.asarray(base.get("public_handoff_field_names_unicode"))
    if public_names.shape != (3,) or public_names.dtype.kind != "U":
        return {"all_model_preflight_gates_pass": False}

    per_module = {}
    for row in worker_rows:
        module = row["module_name"]
        if module not in worker_arrays or set(worker_arrays[module]) != EXPECTED_WORKER_ARRAY_NAMES:
            return {"all_model_preflight_gates_pass": False}
        arrays = worker_arrays[module]
        expected_worker_shapes = {
            "captured_public_q0_float32": ((12, 7), np.dtype(np.float32)),
            "captured_quarantined_q8_float32": ((12, 7), np.dtype(np.float32)),
            "cuda_q0_fk_b1_float32": ((12, 3), np.dtype(np.float32)),
            "cuda_q0_fk_b16_float32": ((12, 3), np.dtype(np.float32)),
            "cuda_q8_fk_b1_float32": ((12, 3), np.dtype(np.float32)),
            "cuda_q8_fk_b16_float32": ((12, 3), np.dtype(np.float32)),
            "captured_one_step_x_float32": ((24, 14), np.dtype(np.float32)),
            "captured_one_step_u_float32": ((24, 7), np.dtype(np.float32)),
            "cuda_one_step_b1_float32": ((24, 14), np.dtype(np.float32)),
            "cuda_one_step_b16_float32": ((24, 14), np.dtype(np.float32)),
            "captured_broad_x_float32": ((32, 14), np.dtype(np.float32)),
            "captured_broad_q_float32": ((32, 7), np.dtype(np.float32)),
            "captured_broad_u_float32": ((32, 7), np.dtype(np.float32)),
            "cuda_broad_one_step_b1_float32": ((32, 14), np.dtype(np.float32)),
            "cuda_broad_one_step_b16_float32": ((32, 14), np.dtype(np.float32)),
            "cuda_broad_fk_b1_float32": ((32, 3), np.dtype(np.float32)),
            "cuda_broad_fk_b16_float32": ((32, 3), np.dtype(np.float32)),
            "captured_dense_x0_float32": ((12, 14), np.dtype(np.float32)),
            "captured_dense_u_float32": ((12, 7), np.dtype(np.float32)),
            "cuda_dense_states_float32": ((12, 65, 14), np.dtype(np.float32)),
            "cuda_dense_tool_positions_float32": ((12, 65, 3), np.dtype(np.float32)),
            "dense_time_float64": ((65,), np.dtype(np.float64)),
            "dense_controls_float32": ((12, 64, 7), np.dtype(np.float32)),
        }
        if any(
            np.asarray(arrays[name]).shape != shape
            or np.asarray(arrays[name]).dtype != dtype
            for name, (shape, dtype) in expected_worker_shapes.items()
        ):
            return {"all_model_preflight_gates_pass": False}
        q0_b1 = arrays["cuda_q0_fk_b1_float32"]
        q0_b16 = arrays["cuda_q0_fk_b16_float32"]
        q8_b1 = arrays["cuda_q8_fk_b1_float32"]
        q8_b16 = arrays["cuda_q8_fk_b16_float32"]
        one_b1 = arrays["cuda_one_step_b1_float32"]
        one_b16 = arrays["cuda_one_step_b16_float32"]
        broad_b1 = arrays["cuda_broad_one_step_b1_float32"]
        broad_b16 = arrays["cuda_broad_one_step_b16_float32"]
        broad_fk_b1 = arrays["cuda_broad_fk_b1_float32"]
        broad_fk_b16 = arrays["cuda_broad_fk_b16_float32"]
        dense_states = arrays["cuda_dense_states_float32"]
        dense_tools = arrays["cuda_dense_tool_positions_float32"]
        q0_error = np.linalg.norm(
            q0_b1.astype(np.float64) - base["pin_q0_tool_float64"], axis=1
        )
        q8_error = np.linalg.norm(
            q8_b1.astype(np.float64) - base["pin_q8_tool_float64"], axis=1
        )
        q8_b1_goal_error = np.linalg.norm(
            q8_b1.astype(np.float64)
            - base["public_reference_float32"][:, :3].astype(np.float64),
            axis=1,
        )
        q8_b16_goal_error = np.linalg.norm(
            q8_b16.astype(np.float64)
            - base["public_reference_float32"][:, :3].astype(np.float64),
            axis=1,
        )
        one_delta = one_b1.astype(np.float64) - base["pin_one_step_float64"]
        one_l2 = np.linalg.norm(one_delta, axis=1)
        one_maxabs = np.max(np.abs(one_delta), axis=1)
        broad_delta = (
            broad_b1.astype(np.float64) - base["pin_broad_one_step_float64"]
        )
        broad_l2 = np.linalg.norm(broad_delta, axis=1)
        broad_maxabs = np.max(np.abs(broad_delta), axis=1)
        broad_fk_b1_error = np.linalg.norm(
            broad_fk_b1.astype(np.float64) - base["pin_broad_tool_float64"],
            axis=1,
        )
        broad_fk_b16_error = np.linalg.norm(
            broad_fk_b16.astype(np.float64) - base["pin_broad_tool_float64"],
            axis=1,
        )
        dense_delta = (
            dense_states.astype(np.float64) - base["pin_dense_states_float64"]
        )
        dense_state_l2 = np.linalg.norm(dense_delta, axis=2)
        dense_tool_error = np.linalg.norm(
            dense_tools.astype(np.float64)
            - base["pin_dense_tool_positions_float64"],
            axis=2,
        )
        q = dense_states[:, :, :7].astype(np.float64)
        qd = dense_states[:, :, 7:].astype(np.float64)
        u = arrays["captured_dense_u_float32"].astype(np.float64)
        cuda_limits = bool(
            np.all(q >= base["model_lower_float64"][None, None, :])
            and np.all(q <= base["model_upper_float64"][None, None, :])
            and np.all(
                np.abs(qd) <= base["model_velocity_float64"][None, None, :]
            )
            and np.all(
                np.abs(u) <= base["model_effort_float64"][None, :]
            )
        )
        pin_q = base["pin_dense_states_float64"][:, :, :7]
        pin_qd = base["pin_dense_states_float64"][:, :, 7:]
        pin_limits = bool(
            np.all(pin_q >= base["model_lower_float64"][None, None, :])
            and np.all(pin_q <= base["model_upper_float64"][None, None, :])
            and np.all(
                np.abs(pin_qd) <= base["model_velocity_float64"][None, None, :]
            )
            and np.all(np.abs(u) <= base["model_effort_float64"][None, :])
        )
        input_identity = bool(
            np.array_equal(
                arrays["captured_public_q0_float32"],
                base["public_solver_x0_float32"][:, :7],
            )
            and np.array_equal(
                arrays["captured_quarantined_q8_float32"],
                base["quarantined_q8_float64"].astype(np.float32),
            )
            and np.array_equal(
                arrays["captured_one_step_x_float32"],
                base["one_step_x_float32"],
            )
            and np.array_equal(
                arrays["captured_one_step_u_float32"],
                base["one_step_u_float32"],
            )
            and np.array_equal(
                arrays["captured_dense_x0_float32"],
                base["dense_x0_float32"],
            )
            and np.array_equal(
                arrays["captured_dense_u_float32"],
                base["dense_u_float32"],
            )
            and np.array_equal(
                arrays["captured_broad_x_float32"], base["broad_x_float32"]
            )
            and np.array_equal(
                arrays["captured_broad_q_float32"],
                base["broad_x_float32"][:, :7],
            )
            and np.array_equal(
                arrays["captured_broad_u_float32"], base["broad_u_float32"]
            )
        )
        exact_diagnostic_layout = bool(
            np.array_equal(base["one_step_x_float32"][0::2], base["dense_x0_float32"])
            and np.array_equal(base["one_step_x_float32"][1::2], base["dense_x0_float32"])
            and np.array_equal(
                base["one_step_u_float32"][0::2],
                np.zeros((12, 7), dtype=np.float32),
            )
            and np.array_equal(
                base["one_step_u_float32"][1::2], base["dense_u_float32"]
            )
            and np.array_equal(
                arrays["dense_time_float64"],
                np.arange(65, dtype=np.float64) * ONE_STEP_DT / DENSE_SUBSTEPS,
            )
            and np.array_equal(
                arrays["dense_controls_float32"],
                np.repeat(base["dense_u_float32"][:, None, :], 64, axis=1),
            )
            and np.array_equal(dense_states[:, 0], base["dense_x0_float32"])
            and np.array_equal(
                base["pin_dense_states_float64"][:, 0],
                base["dense_x0_float32"].astype(np.float64),
            )
        )
        smoke_raw = {
            name: arrays[f"reference_smoke_{name}"]
            for name in (
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
                "baseline_reference",
                "perturbed_reference",
            )
        }
        smoke_raw["ls_num_iters_b1"] = int(
            arrays["reference_smoke_ls_num_iters_b1"]
        )
        smoke_raw["ls_num_iters_b16"] = int(
            arrays["reference_smoke_ls_num_iters_b16"]
        )
        spec = next(row for row in FROZEN_EXTENSIONS if row["module_name"] == module)
        independent_reference = certify_reference_smoke(
            smoke_raw,
            variant=spec["variant"],
            reference_size=spec["reference_size"],
        )
        reported_reference = row.get("reference_smoke", {})
        gates = {
            "worker_boundary_pass": row.get("passes") is True,
            "captured_inputs_exact": input_identity,
            "zero_gravity_dense_layout_exact": exact_diagnostic_layout,
            "q0_b1_b16_lane_identity": np.array_equal(q0_b1, q0_b16),
            "q8_b1_b16_lane_identity": np.array_equal(q8_b1, q8_b16),
            "q0_pin_cuda_tool_pass": float(np.max(q0_error))
            <= FK_PIN_CUDA_TOLERANCE_M,
            "q8_pin_cuda_tool_pass": float(np.max(q8_error))
            <= FK_PIN_CUDA_TOLERANCE_M,
            "q8_b1_public_goal_pass": float(
                np.max(q8_b1_goal_error)
            )
            <= Q8_PUBLIC_GOAL_TOLERANCE_M,
            "q8_b16_public_goal_pass": float(
                np.max(q8_b16_goal_error)
            )
            <= Q8_PUBLIC_GOAL_TOLERANCE_M,
            "q8_pin_public_goal_pass": float(
                np.max(
                    np.linalg.norm(
                        base["pin_q8_tool_float64"]
                        - base["public_reference_float32"][:, :3].astype(
                            np.float64
                        ),
                        axis=1,
                    )
                )
            )
            <= Q8_PUBLIC_GOAL_TOLERANCE_M,
            "one_step_b1_b16_lane_identity": np.array_equal(one_b1, one_b16),
            "one_step_all_finite": np.all(np.isfinite(one_b1))
            and np.all(np.isfinite(base["pin_one_step_float64"])),
            "one_step_l2_pass": float(np.max(one_l2))
            <= ONE_STEP_STATE_L2_TOLERANCE,
            "one_step_maxabs_pass": float(np.max(one_maxabs))
            <= ONE_STEP_STATE_MAXABS_TOLERANCE,
            "broad_one_step_b1_b16_lane_identity": np.array_equal(
                broad_b1, broad_b16
            ),
            "broad_fk_b1_b16_lane_identity": np.array_equal(
                broad_fk_b1, broad_fk_b16
            ),
            "broad_fk_b1_pin_cuda_pass": float(np.max(broad_fk_b1_error))
            <= FK_PIN_CUDA_TOLERANCE_M,
            "broad_fk_b16_pin_cuda_pass": float(np.max(broad_fk_b16_error))
            <= FK_PIN_CUDA_TOLERANCE_M,
            "broad_one_step_all_finite": np.all(np.isfinite(broad_b1))
            and np.all(np.isfinite(base["pin_broad_one_step_float64"])),
            "broad_one_step_l2_pass": float(np.max(broad_l2))
            <= ONE_STEP_STATE_L2_TOLERANCE,
            "broad_one_step_maxabs_pass": float(np.max(broad_maxabs))
            <= ONE_STEP_STATE_MAXABS_TOLERANCE,
            "dense_cuda_all_finite": np.all(np.isfinite(dense_states))
            and np.all(np.isfinite(dense_tools)),
            "dense_pin_all_finite": np.all(
                np.isfinite(base["pin_dense_states_float64"])
            )
            and np.all(np.isfinite(base["pin_dense_tool_positions_float64"])),
            "dense_cuda_limits_pass": cuda_limits,
            "dense_pin_limits_pass": pin_limits,
            "dense_tool_pass": float(np.max(dense_tool_error))
            <= DENSE_TOOL_POSITION_TOLERANCE_M,
            "reference_smoke_pass": reported_reference.get(
                "all_reference_smoke_gates_pass"
            )
            is True,
            "reference_smoke_independently_recomputed": independent_reference.get(
                "all_reference_smoke_gates_pass"
            )
            is True,
            "wrong_width_host_rejection_exact": reported_reference.get(
                "wrong_width_rejected_before_native_kernel"
            )
            is True
            and reported_reference.get("wrong_width_value_error_exact") is True
            and reported_reference.get("wrong_width_rejection_calls") == 1,
            "zero_iteration_nonoptimization_smoke_exact": reported_reference.get(
                "diagnostic_zero_iteration_solve_calls"
            )
            == 2
            and reported_reference.get("sqp_optimization_calls") == 0
            and reported_reference.get("optimization_evidence") is False,
            "zero_sqp_optimization_calls": row.get("sqp_optimization_calls") == 0,
            "q8_tool_position_only": row.get("q8_scope") == "tool_position_only",
            "q8_not_initializer_or_solver_seed": row.get(
                "q8_initializer_eligible"
            )
            is False
            and row.get("q8_solver_seed_eligible") is False,
        }
        per_module[module] = {
            **gates,
            "max_q0_tool_error_m": float(np.max(q0_error)),
            "max_q8_tool_error_m": float(np.max(q8_error)),
            "max_q8_b1_public_goal_error_m": float(
                np.max(q8_b1_goal_error)
            ),
            "max_q8_b16_public_goal_error_m": float(
                np.max(q8_b16_goal_error)
            ),
            "max_one_step_l2": float(np.max(one_l2)),
            "max_one_step_maxabs": float(np.max(one_maxabs)),
            "max_broad_one_step_l2": float(np.max(broad_l2)),
            "max_broad_one_step_maxabs": float(np.max(broad_maxabs)),
            "max_broad_fk_b1_pin_cuda_error_m": float(
                np.max(broad_fk_b1_error)
            ),
            "max_broad_fk_b16_pin_cuda_error_m": float(
                np.max(broad_fk_b16_error)
            ),
            "max_dense_state_l2_report_only": float(np.max(dense_state_l2)),
            "max_dense_tool_error_m": float(np.max(dense_tool_error)),
            "independent_reference_smoke": independent_reference,
            "all_module_gates_pass": bool(all(gates.values())),
        }

    first = worker_arrays[module_names[0]]
    cross_module = {
        "q0_fk_exact": all(
            np.array_equal(first["cuda_q0_fk_b16_float32"], worker_arrays[name]["cuda_q0_fk_b16_float32"])
            for name in module_names[1:]
        ),
        "q8_fk_exact": all(
            np.array_equal(first["cuda_q8_fk_b16_float32"], worker_arrays[name]["cuda_q8_fk_b16_float32"])
            for name in module_names[1:]
        ),
        "one_step_exact": all(
            np.array_equal(first["cuda_one_step_b16_float32"], worker_arrays[name]["cuda_one_step_b16_float32"])
            for name in module_names[1:]
        ),
        "broad_one_step_exact": all(
            np.array_equal(
                first["cuda_broad_one_step_b16_float32"],
                worker_arrays[name]["cuda_broad_one_step_b16_float32"],
            )
            for name in module_names[1:]
        ),
        "broad_fk_b1_exact": all(
            np.array_equal(
                first["cuda_broad_fk_b1_float32"],
                worker_arrays[name]["cuda_broad_fk_b1_float32"],
            )
            for name in module_names[1:]
        ),
        "broad_fk_b16_exact": all(
            np.array_equal(
                first["cuda_broad_fk_b16_float32"],
                worker_arrays[name]["cuda_broad_fk_b16_float32"],
            )
            for name in module_names[1:]
        ),
        "dense_state_exact": all(
            np.array_equal(first["cuda_dense_states_float32"], worker_arrays[name]["cuda_dense_states_float32"])
            for name in module_names[1:]
        ),
        "dense_tool_exact": all(
            np.array_equal(first["cuda_dense_tool_positions_float32"], worker_arrays[name]["cuda_dense_tool_positions_float32"])
            for name in module_names[1:]
        ),
    }
    regenerated_broad = generate_broad_algebraic_rows(
        base["model_lower_float64"],
        base["model_upper_float64"],
        base["model_velocity_float64"],
        base["model_effort_float64"],
    )
    broad_regeneration_exact = all(
        np.array_equal(base[name], value)
        for name, value in regenerated_broad.items()
    )
    gates = {
        "hard_pinned_v4_artifact_authenticated": bool(
            np.asarray(
                base.get("v4_artifact_authentication_gate_bool", False)
            ).shape
            == ()
            and bool(base.get("v4_artifact_authentication_gate_bool"))
        ),
        "all_three_modules_pass": all(
            row["all_module_gates_pass"] for row in per_module.values()
        ),
        "cross_module_behavior_exact": all(cross_module.values()),
        "public_handoff_exact": base["public_solver_x0_float32"].shape == (12, 14)
        and base["public_reference_float32"].shape == (12, 10)
        and base["public_default_side_int8"].shape == (12,)
        and np.all(np.isin(base["public_default_side_int8"], (-1, 1)))
        and np.array_equal(
            base["public_solver_x0_float32"][:, 7:],
            np.zeros((12, 7), dtype=np.float32),
        ),
        "quarantine_arrays_exact": np.array_equal(
            base["quarantined_q8_float32"],
            base["quarantined_q8_float64"].astype(np.float32),
        )
        and np.all(base["quarantined_q8_oracle_only_bool"])
        and not np.any(base["quarantined_q8_initializer_eligible_bool"])
        and not np.any(base["quarantined_q8_solver_seed_eligible_bool"]),
        "broad_algebraic_grid_regenerates_exactly": broad_regeneration_exact,
        "broad_algebraic_grid_limits_pass": bool(
            np.all(
                base["broad_x_float64"][:, :7]
                >= base["model_lower_float64"][None, :] + BROAD_Q_RESERVE_RAD
            )
            and np.all(
                base["broad_x_float64"][:, :7]
                <= base["model_upper_float64"][None, :] - BROAD_Q_RESERVE_RAD
            )
            and np.all(
                np.abs(base["broad_x_float64"][:, 7:])
                <= BROAD_VELOCITY_FRACTION
                * base["model_velocity_float64"][None, :]
            )
            and np.all(
                np.abs(base["broad_u_float64"])
                <= BROAD_CONTROL_FRACTION
                * base["model_effort_float64"][None, :]
            )
        ),
        "q0_q8_model_limits_pass": bool(
            np.all(
                base["public_solver_x0_float32"][:, :7].astype(np.float64)
                >= base["model_lower_float64"][None, :]
            )
            and np.all(
                base["public_solver_x0_float32"][:, :7].astype(np.float64)
                <= base["model_upper_float64"][None, :]
            )
            and np.all(
                base["quarantined_q8_float64"]
                >= base["model_lower_float64"][None, :]
            )
            and np.all(
                base["quarantined_q8_float64"]
                <= base["model_upper_float64"][None, :]
            )
        ),
        "quarantined_q8_not_public_handoff": set(
            np.asarray(base.get("public_handoff_field_names_unicode", ())).tolist()
        )
        == {
            "public_solver_x0_float32",
            "public_reference_float32",
            "public_default_side_int8",
        },
        "zero_task_rng_construction_predecessor_loads": np.array_equal(
            base.get("forbidden_call_counts_int64"),
            np.zeros(3, dtype=np.int64),
        ),
        "zero_sqp_optimization_calls": all(
            row.get("sqp_optimization_calls") == EXPECTED_SQP_OPTIMIZATION_CALLS
            for row in worker_rows
        ),
    }
    return {
        "per_module": per_module,
        "cross_module": cross_module,
        "dense_state_l2_is_report_only": True,
        **gates,
        "all_model_preflight_gates_pass": bool(all(gates.values())),
    }


def _validate_worker_boundary(row, spec, input_hash):
    try:
        summary_path = Path(row["summary_path"])
        npz_path = Path(row["npz_path"])
        request_path = Path(row["request_path"])
        summary = json.loads(summary_path.read_text())
        request = json.loads(request_path.read_text())
        with np.load(npz_path, allow_pickle=False) as archive:
            names = set(archive.files)
            hashes = {name: array_hash(archive[name]) for name in archive.files}
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    expected_command = [
        sys.executable,
        "-B",
        "-m",
        "gato_tiago.multimodal_toll_v4_model_preflight_worker",
        "--request",
        str(request_path),
        "--output",
        str(summary_path),
        "--execute",
    ]
    return bool(
        row.get("exit_code") == 0
        and row.get("module_name") == spec["module_name"]
        and row.get("command") == expected_command
        and row.get("request_sha256") == sha256_file(request_path)
        and row.get("summary_sha256") == sha256_file(summary_path)
        and row.get("npz_sha256") == sha256_file(npz_path)
        and request.get("module_name") == spec["module_name"]
        and Path(request.get("extension_path", "")).resolve()
        == Path(spec["extension_path"]).resolve()
        and request.get("extension_sha256") == spec["extension_sha256"]
        and request.get("input_npz_sha256") == input_hash
        and request.get("protocol_version") == WORKER_PROTOCOL_VERSION
        and summary.get("worker_protocol_version") == WORKER_PROTOCOL_VERSION
        and summary.get("model_preflight_protocol_version")
        == MODEL_PREFLIGHT_PROTOCOL_VERSION
        and summary.get("module_name") == spec["module_name"]
        and Path(summary.get("module_file", "")).resolve()
        == Path(spec["extension_path"]).resolve()
        and summary.get("extension_sha256") == spec["extension_sha256"]
        and summary.get("reference_size") == spec["reference_size"]
        and summary.get("input_npz_sha256") == input_hash
        and Path(summary.get("input_npz_path", "")).resolve()
        == Path(row.get("input_npz_path", "")).resolve()
        and Path(summary.get("request_path", "")).resolve() == request_path.resolve()
        and summary.get("request_sha256") == row.get("request_sha256")
        and summary.get("worker_npz_sha256") == row.get("npz_sha256")
        and set(summary.get("array_names", ())) == EXPECTED_WORKER_ARRAY_NAMES
        and names == EXPECTED_WORKER_ARRAY_NAMES
        and summary.get("array_hashes") == hashes
        and summary.get("optimization_evidence") is False
        and summary.get("sqp_optimization_calls") == 0
    )


def _invoke_worker(command, environment):  # pragma: no cover
    process = subprocess.run(
        command,
        cwd=repository_root(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": list(command),
        "stdout": process.stdout,
        "stderr": process.stderr,
        "exit_code": process.returncode,
    }


def _run_pipeline(
    output,
    *,
    provenance,
    artifact_loader: Callable,
    model_preflight_factory: Callable,
    worker_launcher: Callable,
    finish_provenance: Callable | None = None,
    test_override: bool,
):
    output = Path(output)
    _no_existing_artifacts(output)
    rows = []
    arrays = {}
    _checkpoint(output, 0, "before_artifact_or_model", rows, arrays, provenance)
    try:
        artifact_arrays, authentication = artifact_loader()
        artifact_arrays = dict(artifact_arrays)
        artifact_arrays["v4_artifact_authentication_gate_bool"] = np.asarray(
            authentication.get("all_v4_artifact_authentication_gates_pass") is True,
            dtype=np.bool_,
        )
        arrays.update(artifact_arrays)
        provenance["v4_artifact_authentication"] = authentication
    except BaseException as error:
        rows.append(
            {
                "identity": ["artifact", "v4_final"],
                "passes": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _checkpoint(output, 1, "artifact_failed", rows, arrays, provenance)
        raise
    _checkpoint(output, 1, "v4_artifact_authenticated", rows, arrays, provenance)

    try:
        model_arrays = model_preflight_factory(artifact_arrays)
        arrays.update(model_arrays)
    except BaseException as error:
        rows.append(
            {
                "identity": ["model", "pinocchio"],
                "passes": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _checkpoint(output, 2, "model_failed", rows, arrays, provenance)
        raise
    _checkpoint(output, 2, "pin_model_preflight_captured", rows, arrays, provenance)

    worker_arrays = {}
    for index, spec in enumerate(FROZEN_EXTENSIONS, start=3):
        try:
            row, captured = worker_launcher(spec, arrays)
            row = dict(row)
            if row.get("module_name") != spec["module_name"]:
                raise RuntimeError("worker identity mismatch")
            worker_arrays[spec["module_name"]] = captured
            leaf = spec["module_name"].split(".")[-1]
            arrays.update(
                {
                    f"worker_{leaf}_{name}": np.ascontiguousarray(value)
                    for name, value in captured.items()
                }
            )
        except BaseException as error:
            row = {
                "identity": ["worker", spec["module_name"]],
                "module_name": spec["module_name"],
                "passes": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
        rows.append(row)
        _checkpoint(output, index, "worker_attempted", rows, arrays, provenance)

    if not test_override and not all(row.get("passes") is True for row in rows):
        raise RuntimeError("V4 model preflight worker failure; retained partial closes run")
    if finish_provenance is not None:
        finish_provenance(provenance)
    certificate = certify_model_preflight(arrays, rows, worker_arrays)
    certificate["provenance_pass"] = bool(
        provenance.get("git_head_at_start")
        == provenance.get("git_head_at_end")
        and provenance.get("tracked_tree_clean_at_start") is True
        and provenance.get("tracked_tree_clean_at_end") is True
        and set(provenance.get("source_hashes", ())) == set(REQUIRED_SOURCE_PATHS)
        and all(
            is_sha256(row.get("sha256"))
            for row in provenance.get("source_hashes", {}).values()
        )
        and provenance.get("task_rng_calls") == 0
        and provenance.get("task_construction_calls") == 0
        and provenance.get("predecessor_artifact_loads") == 0
        and provenance.get("extension_hashes")
        == {
            row["module_name"]: row["extension_sha256"]
            for row in FROZEN_EXTENSIONS
        }
        and provenance.get("frozen_build_commit") == FROZEN_BUILD_COMMIT
        and provenance.get("frozen_build_source_hashes")
        == FROZEN_BUILD_SOURCE_HASHES
        and provenance.get("current_build_source_hashes")
        == FROZEN_BUILD_SOURCE_HASHES
        and provenance.get("configured_cuda_arch") == FROZEN_CUDA_ARCH
    )
    certificate["all_model_preflight_gates_pass"] = bool(
        certificate.get("all_model_preflight_gates_pass")
        and certificate["provenance_pass"]
    )
    if not test_override and not certificate["all_model_preflight_gates_pass"]:
        raise RuntimeError("V4 model preflight certificate failed; no final published")

    summary = {
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "model_preflight_protocol_version": MODEL_PREFLIGHT_PROTOCOL_VERSION,
        "incomplete": False,
        "rows": _json_value(rows),
        "certificate": _json_value(certificate),
        "all_model_preflight_gates_pass": bool(
            certificate["all_model_preflight_gates_pass"] and not test_override
        ),
        "v4_artifact_authentication": provenance.get(
            "v4_artifact_authentication"
        ),
        "public_handoff_fields": [
            "public_solver_x0_float32",
            "public_reference_float32",
            "public_default_side_int8",
        ],
        "q8_scope": "quarantined_tool_position_model_preflight_only",
        "q8_initializer_eligible": False,
        "q8_solver_seed_eligible": False,
        "task_rng_calls": 0,
        "task_construction_calls": 0,
        "predecessor_artifact_loads": 0,
        "worker_subprocess_calls": len(rows),
        "diagnostic_zero_iteration_solve_calls": 2 * len(rows),
        "sqp_optimization_calls": 0,
        "optimization_evidence": False,
        "timing_evidence": False,
        "provenance": _json_value(provenance),
    }
    final_arrays = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    npz_path = output.with_suffix(".npz")
    _atomic_npz(npz_path, final_arrays)
    summary.update(
        {
            "npz_path": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
            "array_names": sorted(final_arrays),
            "array_hashes": {
                name: array_hash(value) for name, value in final_arrays.items()
            },
        }
    )
    _atomic_json(output, summary)
    side_paths = {
        Path(path)
        for row in rows
        for path in (
            row.get("request_path"),
            row.get("input_npz_path"),
            row.get("summary_path"),
            row.get("npz_path"),
        )
        if path
    }
    side_paths.update(output.parent.glob(f"{output.stem}.partial.*"))
    manifest = {
        "json_path": str(output),
        "json_sha256": sha256_file(output),
        "npz_path": str(npz_path),
        "npz_sha256": summary["npz_sha256"],
        "incomplete": False,
        "all_model_preflight_gates_pass": summary[
            "all_model_preflight_gates_pass"
        ],
        "side_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(side_paths, key=str)
            if path.is_file()
        ],
    }
    _atomic_json(output.with_suffix(".manifest.json"), manifest)
    return summary


def _production_pipeline(output, *, token=None):  # pragma: no cover
    if token is not _PRODUCTION_PIPELINE_TOKEN:
        raise RuntimeError("V4 model-preflight production pipeline is private")
    repo = repository_root()
    provenance = _source_provenance(repo)
    extension_specs = [dict(row) for row in FROZEN_EXTENSIONS]

    def artifact_loader():
        return authenticate_v4_artifact()

    def model_factory(artifact):
        import pinocchio as pin
        from gato_tiago.multimodal_pillar import load_model, tool_position

        model = load_model(repo / REQUIRED_SOURCE_PATHS["model"])
        data = model.createData()
        q0 = artifact["public_solver_x0_float32"][:, :7].astype(np.float64)
        q8 = artifact["quarantined_q8_float64"]
        zero = np.zeros(7, dtype=np.float64)
        pin_q0_tool = np.stack([tool_position(model, data, q) for q in q0])
        pin_q8_tool = np.stack([tool_position(model, data, q) for q in q8])
        gravity = np.stack([pin.rnea(model, data, q, zero, zero) for q in q0])
        one_x = np.repeat(
            np.concatenate([q0, np.zeros((12, 7), dtype=np.float64)], axis=1),
            2,
            axis=0,
        )
        one_u = np.empty((24, 7), dtype=np.float64)
        one_u[0::2] = 0.0
        one_u[1::2] = gravity
        pin_one = []
        for x, u in zip(one_x, one_u):
            qdd = pin.aba(model, data, x[:7], x[7:], u)
            pin_one.append(constant_acceleration_step(x, qdd, ONE_STEP_DT))
        pin_one = np.stack(pin_one)
        broad = generate_broad_algebraic_rows(
            np.asarray(model.lowerPositionLimit, np.float64),
            np.asarray(model.upperPositionLimit, np.float64),
            np.asarray(model.velocityLimit, np.float64),
            np.asarray(model.effortLimit, np.float64),
        )
        pin_broad = []
        for x, u in zip(broad["broad_x_float64"], broad["broad_u_float64"]):
            qdd = pin.aba(model, data, x[:7], x[7:], u)
            pin_broad.append(constant_acceleration_step(x, qdd, ONE_STEP_DT))
        pin_broad = np.stack(pin_broad)
        pin_broad_tool = np.stack(
            [
                tool_position(model, data, q)
                for q in broad["broad_x_float64"][:, :7]
            ]
        )

        dense_x0 = np.concatenate(
            [q0, np.zeros((12, 7), dtype=np.float64)], axis=1
        )
        pin_dense = np.empty((12, 65, 14), dtype=np.float64)
        pin_dense_tool = np.empty((12, 65, 3), dtype=np.float64)
        pin_dense[:, 0] = dense_x0
        for index in range(12):
            pin_dense_tool[index, 0] = tool_position(model, data, q0[index])
        for step in range(DENSE_SUBSTEPS):
            for index in range(12):
                state = pin_dense[index, step]
                qdd = pin.aba(
                    model,
                    data,
                    state[:7],
                    state[7:],
                    gravity[index],
                )
                pin_dense[index, step + 1] = constant_acceleration_step(
                    state, qdd, ONE_STEP_DT / DENSE_SUBSTEPS
                )
                pin_dense_tool[index, step + 1] = tool_position(
                    model, data, pin_dense[index, step + 1, :7]
                )
        return {
            **broad,
            "model_lower_float64": np.asarray(model.lowerPositionLimit, np.float64),
            "model_upper_float64": np.asarray(model.upperPositionLimit, np.float64),
            "model_velocity_float64": np.asarray(model.velocityLimit, np.float64),
            "model_effort_float64": np.asarray(model.effortLimit, np.float64),
            "pin_q0_tool_float64": pin_q0_tool,
            "pin_q8_tool_float64": pin_q8_tool,
            "one_step_x_float32": one_x.astype(np.float32),
            "one_step_u_float32": one_u.astype(np.float32),
            "pin_one_step_float64": pin_one,
            "pin_broad_one_step_float64": pin_broad,
            "pin_broad_tool_float64": pin_broad_tool,
            "dense_x0_float32": dense_x0.astype(np.float32),
            "dense_u_float32": gravity.astype(np.float32),
            "pin_dense_states_float64": pin_dense,
            "pin_dense_tool_positions_float64": pin_dense_tool,
            "quarantined_q8_float32": q8.astype(np.float32),
            "quarantined_q8_oracle_only_bool": np.ones(12, dtype=np.bool_),
            "quarantined_q8_initializer_eligible_bool": np.zeros(12, dtype=np.bool_),
            "quarantined_q8_solver_seed_eligible_bool": np.zeros(12, dtype=np.bool_),
            "public_handoff_field_names_unicode": np.asarray(
                [
                    "public_solver_x0_float32",
                    "public_reference_float32",
                    "public_default_side_int8",
                ]
            ),
            "forbidden_call_counts_int64": np.zeros(3, dtype=np.int64),
        }

    input_npz = Path(output).with_name(f"{Path(output).stem}.worker-input.npz")
    input_written = False

    def worker_launcher(spec, arrays):
        nonlocal input_written
        extension_path = (repo / spec["extension_path"]).resolve()
        if (
            not extension_path.is_file()
            or sha256_file(extension_path) != spec["extension_sha256"]
        ):
            raise RuntimeError("frozen extension hash mismatch")
        spec = {**spec, "extension_path": str(extension_path)}
        if not input_written:
            _atomic_npz(
                input_npz,
                {
                    "public_q0_float32": arrays["public_solver_x0_float32"][:, :7],
                    "quarantined_q8_float32": arrays["quarantined_q8_float32"],
                    "one_step_x_float32": arrays["one_step_x_float32"],
                    "one_step_u_float32": arrays["one_step_u_float32"],
                    "dense_x0_float32": arrays["dense_x0_float32"],
                    "dense_u_float32": arrays["dense_u_float32"],
                    "broad_x_float32": arrays["broad_x_float32"],
                    "broad_u_float32": arrays["broad_u_float32"],
                },
            )
            input_written = True
        input_hash = sha256_file(input_npz)
        leaf = spec["module_name"].split(".")[-1]
        request_path = Path(output).with_name(
            f"{Path(output).stem}.{leaf}.request.json"
        )
        worker_output = Path(output).with_name(f"{Path(output).stem}.{leaf}.json")
        request = {
            "module_name": spec["module_name"],
            "extension_path": spec["extension_path"],
            "extension_sha256": spec["extension_sha256"],
            "input_npz": str(input_npz.resolve()),
            "input_npz_sha256": input_hash,
            "reference_size": spec["reference_size"],
            "protocol_version": WORKER_PROTOCOL_VERSION,
        }
        _atomic_json(request_path, request)
        command = [
            sys.executable,
            "-B",
            "-m",
            "gato_tiago.multimodal_toll_v4_model_preflight_worker",
            "--request",
            str(request_path),
            "--output",
            str(worker_output),
            "--execute",
        ]
        process = _invoke_worker(
            command,
            {
                **os.environ,
                "PYTHONPATH": f"{repo / 'tiago_src'}:{repo / 'python'}",
            },
        )
        worker_npz = worker_output.with_suffix(".npz")
        row = {
            "identity": ["worker", spec["module_name"]],
            "module_name": spec["module_name"],
            "command": command,
            "stdout": process["stdout"],
            "stderr": process["stderr"],
            "exit_code": process["exit_code"],
            "request_path": str(request_path.resolve()),
            "request_sha256": sha256_file(request_path),
            "input_npz_path": str(input_npz.resolve()),
            "input_npz_sha256": input_hash,
            "summary_path": str(worker_output.resolve()),
            "summary_sha256": sha256_file(worker_output)
            if worker_output.is_file()
            else None,
            "npz_path": str(worker_npz.resolve()),
            "npz_sha256": sha256_file(worker_npz)
            if worker_npz.is_file()
            else None,
        }
        row["passes"] = _validate_worker_boundary(row, spec, input_hash)
        captured = {}
        if row["passes"]:
            summary = json.loads(worker_output.read_text())
            row.update(
                {
                    "reference_smoke": summary["reference_smoke"],
                    "sqp_optimization_calls": summary["sqp_optimization_calls"],
                    "q8_scope": summary["q8_scope"],
                    "q8_initializer_eligible": summary[
                        "q8_initializer_eligible"
                    ],
                    "q8_solver_seed_eligible": summary[
                        "q8_solver_seed_eligible"
                    ],
                }
            )
            with np.load(worker_npz, allow_pickle=False) as archive:
                captured = {name: archive[name] for name in archive.files}
        return row, captured

    def finish_provenance(value):
        value["git_head_at_end"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        value["tracked_tree_clean_at_end"] = (
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo,
                text=True,
            ).splitlines()
            == []
        )
        value["full_git_status_at_end"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).splitlines()
        value["extension_hashes"] = {
            row["module_name"]: row["extension_sha256"]
            for row in extension_specs
        }

    return _run_pipeline(
        output,
        provenance=provenance,
        artifact_loader=artifact_loader,
        model_preflight_factory=model_factory,
        worker_launcher=lambda spec, arrays: worker_launcher(
            next(
                row
                for row in extension_specs
                if row["module_name"] == spec["module_name"]
            ),
            arrays,
        ),
        finish_provenance=finish_provenance,
        test_override=False,
    )


def describe_model_preflight():
    return {
        **frozen_model_preflight_metadata(),
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "runner_execution_authorized": RUNNER_EXECUTION_AUTHORIZATION is not None,
        "worker_execution_authorized": WORKER_EXECUTION_AUTHORIZATION is not None,
        "expected_worker_modules": [row["module_name"] for row in FROZEN_EXTENSIONS],
        "transaction_generations": [
            "gen0_before_artifact_or_model",
            "gen1_v4_artifact_authenticated",
            "gen2_pin_model_preflight_captured",
            "gen3_to_gen5_worker_attempts",
        ],
    }


def execute_model_preflight(output, *, authorization=None):
    if (
        RUNNER_EXECUTION_AUTHORIZATION is None
        or authorization is not RUNNER_EXECUTION_AUTHORIZATION
    ):
        raise RuntimeError("V4 model-preflight runner execution is blocked")
    if Path(output).resolve() != AUTHORIZED_OUTPUT_PATH.resolve():
        raise RuntimeError("V4 model-preflight output is not the authorized path")
    return _production_pipeline(output, token=_PRODUCTION_PIPELINE_TOKEN)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.describe and not args.execute:
        print(json.dumps(describe_model_preflight(), indent=2, sort_keys=True))
        return 0
    if not args.execute or RUNNER_EXECUTION_AUTHORIZATION is None:
        raise SystemExit("V4 model-preflight execution is blocked")
    execute_model_preflight(
        args.output, authorization=RUNNER_EXECUTION_AUTHORIZATION
    )
    return 0


if __name__ == "__main__":
    main()
