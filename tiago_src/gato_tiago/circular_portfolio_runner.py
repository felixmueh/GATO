"""Fail-closed static transaction boundary for circular portfolio P1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from gato_tiago.circular_portfolio import (
    DENSE_SAMPLES, EXPECTED_LEDGER, INTERVALS, MODEL_ARTIFACT_PINS,
    PROTOCOL_VERSION, TASK_ARTIFACT_PINS,
)
from gato_tiago.circular_portfolio_constructor import (
    CSR_FIELDS, DERIVATIVE_ARRAY_SPECS, GLOBAL_ARRAY_SPECS,
    PROFILE_ARRAY_SPECS, watchdog,
)


RUNNER_EXECUTION_AUTHORIZATION = None
OUTPUT_PATH = Path(
    "/tmp/tiago-tool-center-circular-portfolio-p1-authorized-once/p1.json"
)
SOURCE_PATHS = (
    "CMakeLists.txt", "python/bindings.cu",
    "gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "gato/dynamics/tiago_right/tiago_right_grid.cuh",
    "gato/dynamics/tiago_right/tiago_right_arm.urdf",
    "gato/dynamics/integrator.cuh", "gato/bsqp/bsqp.cuh",
    "gato/bsqp/kernels/tool_position.cuh", "gato/utils/cuda.cuh",
    "python/bsqp/interface.py", "tools/build.sh",
    "tiago_src/gato_tiago/multimodal_pillar.py",
    "tiago_src/gato_tiago/circular_portfolio.py",
    "tiago_src/gato_tiago/circular_portfolio_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_worker.py",
    "tests/python/test_tiago_circular_portfolio_static.py",
)
PROVENANCE_KEYS = {
    "protocol", "cwd", "orig_argv", "exact_command", "git_head_at_start",
    "git_head_at_end", "tracked_clean_at_start", "tracked_clean_at_end",
    "source_hashes_at_start", "source_hashes_at_end", "task_artifact_pins",
    "model_artifact_pins", "build_commit", "cuda_arch", "extension",
    "runtime_versions",
}
AUTHORIZED_CWD = "/workspace/GATO"
AUTHORIZED_ORIG_ARGV = (
    "python", "-B", "-m", "gato_tiago.circular_portfolio_runner", "--execute",
    "--output", str(OUTPUT_PATH),
)
FROZEN_BUILD_COMMIT = "2f1011da2a240fe8eae9ff25b2b3ee991c11c6c2"
FROZEN_CUDA_ARCH = "61-real"
FROZEN_EXTENSION = {
    "path": "/workspace/GATO/python/bsqp/bsqpN96_tiago_right_circular_portfolio_toll.cpython-310-x86_64-linux-gnu.so",
    "sha256": "b079410ade9e7de19ed3d4b7ed6f6ace27172cd442ccea0d5a46bb7277067a2f",
    "size_bytes": 6_690_480,
    "KNOT_POINTS": 96, "REFERENCE_SIZE": 10,
    "TOOL_POSITION_FRAME": "arm_right_tool_joint_origin",
    "TOOL_POSITION_SIZE": 3, "B1": "BSQP_1_float", "B16": "BSQP_16_float",
}
COUNTER_KEYS = {
    "task_artifact_loads", "task_pure_recert_calls", "task_array_hashes_checked",
    "model_artifact_loads", "model_pure_recert_calls", "model_array_hashes_checked",
    "cross_bind_calls", "pinocchio_model_calls", "geometry_fk_calls",
    "profile_attempts", "optimizer_calls", "pin_replay_calls", "worker_subprocess_calls",
    "cuda_replay_calls", "task_rng_calls", "task_construction_calls",
    "sqp_calls", "initializer_calls",
}


def authenticate_accepted_prerequisites():  # pragma: no cover - runtime boundary
    """Own every accepted-artifact load, producer hash, pure recert, and bind."""
    from gato_tiago.multimodal_toll_oracle_v2 import (
        MODEL_ARRAY_COUNT, MODEL_PINS, TASK_ARRAY_COUNT, TASK_PINS,
    )
    from gato_tiago.multimodal_toll_oracle_v2_prerequisite_runner import (
        _load_pinned_artifact, _model_pure_recert, _task_pure_recert,
        authenticate_producer, cross_bind_prerequisites,
    )
    task_summary, task_arrays, _task_manifest, _task_pointer = _load_pinned_artifact(TASK_PINS)
    task_auth = authenticate_producer(
        task_summary, task_arrays, expected_count=TASK_ARRAY_COUNT,
        certificate_key="all_v4_gates_pass",
    )
    task_auth["pure_recert"] = _task_pure_recert(task_summary, task_arrays)
    task_auth["passes"] = bool(task_auth["passes"] and task_auth["pure_recert"])
    model_summary, model_arrays, _model_manifest, _model_pointer = _load_pinned_artifact(MODEL_PINS)
    model_auth = authenticate_producer(
        model_summary, model_arrays, expected_count=MODEL_ARRAY_COUNT,
        certificate_key="all_model_preflight_gates_pass",
    )
    model_auth["pure_recert"] = _model_pure_recert(model_summary, model_arrays)
    model_auth["passes"] = bool(model_auth["passes"] and model_auth["pure_recert"])
    cross_bind = cross_bind_prerequisites(task_summary.get("rows", ()), task_arrays, model_arrays)
    detail = {"task": task_auth, "model": model_auth, "cross_bind": cross_bind}
    detail["passes"] = bool(task_auth["passes"] and model_auth["passes"] and cross_bind["passes"])
    return detail, task_summary, task_arrays, model_summary, model_arrays


def _expected_counters(generation):
    completed = max(0, min(len(EXPECTED_LEDGER), generation - 3))
    task = generation >= 1
    model = generation >= 2
    geometry = generation >= 3
    return {
        "task_artifact_loads": int(task), "task_pure_recert_calls": int(task),
        "task_array_hashes_checked": 603 if task else 0,
        "model_artifact_loads": int(model), "model_pure_recert_calls": int(model),
        "model_array_hashes_checked": 150 if model else 0,
        "cross_bind_calls": int(model), "pinocchio_model_calls": int(geometry),
        "geometry_fk_calls": 24 if geometry else 0,
        "profile_attempts": completed, "optimizer_calls": completed,
        "pin_replay_calls": completed, "worker_subprocess_calls": completed,
        "cuda_replay_calls": completed,
        "task_rng_calls": 0, "task_construction_calls": 0,
        "sqp_calls": 0, "initializer_calls": 0,
    }


def _checkpoint_array_names(generation):
    completed = max(0, min(len(EXPECTED_LEDGER), generation - 3))
    names = {
        "generation_int64", "incomplete_bool", "completed_count_int64",
        "task_authentication_bool", "model_authentication_bool",
        "geometry_authentication_bool",
    }
    if generation >= 1:
        names.update(("task_array_hash_digest_uint8",))
    if generation >= 2:
        names.update(("model_array_hash_digest_uint8", "cross_bind_bool"))
    retained_indices = (generation - 4,) if 4 <= generation <= len(EXPECTED_LEDGER) + 3 else ()
    for index in retained_indices:
        identity = EXPECTED_LEDGER[index]
        prefix = f"profile_{index:03d}_{identity[1]}_{identity[2]}_{identity[3]}"
        names.update(f"{prefix}_{name}" for name in PROFILE_ARRAY_SPECS)
    return tuple(sorted(names))


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


def expected_final_array_names():
    names = {
        "public_x0_float32", "public_reference_float32", "public_default_side_int8",
        "quarantined_q8_float64", "joint_lower_float64", "joint_upper_float64",
        "velocity_limit_float64", "effort_limit_float64",
        "b1_x0_float32", "b1_reference_float32", "b1_seed_xu_float32",
        "b16_x0_float32", "b16_reference_float32", "b16_seed_xu_float32",
    }
    names.update(DERIVATIVE_ARRAY_SPECS)
    names.update(GLOBAL_ARRAY_SPECS)
    names.update(
        f"derivative_probe{probe}_inequality_csr_{field}"
        for probe in range(3) for field in CSR_FIELDS
    )
    for index, identity in enumerate(EXPECTED_LEDGER):
        prefix = f"profile_{index:03d}_{identity[1]}_{identity[2]}_{identity[3]}"
        names.update(f"{prefix}_{name}" for name in PROFILE_ARRAY_SPECS)
    return tuple(sorted(names))


def transaction_schema():
    return {
        "protocol": PROTOCOL_VERSION,
        "generation_zero_before_artifact_or_model_load": True,
        "checkpoint_stages": [
            "gen0", "task_authenticated", "model_authenticated", "geometry_authenticated",
            *["profile_completed"] * len(EXPECTED_LEDGER), "honest_end_provenance",
        ],
        "final_generation": len(EXPECTED_LEDGER) + 4,
        "completed_identities": [],
        "pending_identities": [list(identity) for identity in EXPECTED_LEDGER],
        "resume_allowed": False,
        "overwrite_allowed": False,
        **{name: 0 for name in COUNTER_KEYS},
        "oracle_evidence": False,
        "benchmark_evidence": False,
    }


def certify_provenance(provenance: Mapping, *, final: bool):
    if set(provenance) != PROVENANCE_KEYS:
        return False
    head_start = provenance["git_head_at_start"]
    head_end = provenance["git_head_at_end"]
    starts = provenance["source_hashes_at_start"]
    ends = provenance["source_hashes_at_end"]
    return bool(
        provenance["protocol"] == PROTOCOL_VERSION
        and isinstance(head_start, str) and len(head_start) == 40
        and provenance["cwd"] == AUTHORIZED_CWD
        and tuple(provenance["orig_argv"]) == AUTHORIZED_ORIG_ARGV
        and provenance["exact_command"] == " ".join(AUTHORIZED_ORIG_ARGV)
        and provenance["tracked_clean_at_start"] is True
        and set(starts) == set(SOURCE_PATHS)
        and all(isinstance(value, str) and len(value) == 64 for value in starts.values())
        and provenance["task_artifact_pins"] == TASK_ARTIFACT_PINS
        and provenance["model_artifact_pins"] == MODEL_ARTIFACT_PINS
        and provenance["build_commit"] == FROZEN_BUILD_COMMIT
        and provenance["cuda_arch"] == FROZEN_CUDA_ARCH
        and provenance["extension"] == FROZEN_EXTENSION
        and set(provenance["runtime_versions"]) == {"python", "numpy", "scipy", "pinocchio", "cuda"}
        and all(isinstance(value, str) and value for value in provenance["runtime_versions"].values())
        and ((not final and head_end is None and ends is None
              and provenance["tracked_clean_at_end"] is None)
             or (final and head_end == head_start and ends == starts
                 and provenance["tracked_clean_at_end"] is True))
    )


def certify_checkpoint(document: Mapping, generation: int):
    required = {
        "protocol", "generation", "stage", "incomplete", "completed_identities",
        "pending_identities", "counters", "provenance", "evidence_flags",
        "watchdog", "array_names", "array_hashes", "npz_path", "npz_sha256",
        "profile_certificate",
        "task_array_hash_digest_hex", "model_array_hash_digest_hex",
        "geometry_digest_hex",
    }
    if set(document) != required or generation < 0 or generation > len(EXPECTED_LEDGER) + 4:
        return False
    completed = max(0, min(len(EXPECTED_LEDGER), generation - 3))
    if generation == 0:
        stage = "gen0"
    elif generation == 1:
        stage = "task_authenticated"
    elif generation == 2:
        stage = "model_authenticated"
    elif generation == 3:
        stage = "geometry_authenticated"
    elif generation <= len(EXPECTED_LEDGER) + 3:
        stage = "profile_completed"
    else:
        stage = "honest_end_provenance"
    counters = document["counters"]
    final = generation == len(EXPECTED_LEDGER) + 4
    profile_certificate = document["profile_certificate"]
    profile_gate = (
        isinstance(profile_certificate, Mapping)
        and tuple(profile_certificate.get("identity", ())) == EXPECTED_LEDGER[generation - 4]
        and profile_certificate.get("passes") is True
    ) if 4 <= generation <= len(EXPECTED_LEDGER) + 3 else profile_certificate is None
    return bool(
        document["protocol"] == PROTOCOL_VERSION
        and document["generation"] == generation and document["stage"] == stage
        and document["incomplete"] is True
        and document["completed_identities"] == [list(row) for row in EXPECTED_LEDGER[:completed]]
        and document["pending_identities"] == [list(row) for row in EXPECTED_LEDGER[completed:]]
        and counters == _expected_counters(generation)
        and document["evidence_flags"] == {"oracle_evidence": False, "benchmark_evidence": False}
        and certify_provenance(document["provenance"], final=final)
        and document["watchdog"] == watchdog(
            document["watchdog"].get("elapsed_s", -1), completed
        )
        and document["array_names"] == list(_checkpoint_array_names(generation))
        and isinstance(document["array_hashes"], Mapping)
        and set(document["array_hashes"]) == set(document["array_names"])
        and isinstance(document["npz_path"], str)
        and isinstance(document["npz_sha256"], str) and len(document["npz_sha256"]) == 64
        and profile_gate
    )


def certify_checkpoint_arrays(document: Mapping, arrays: Mapping):
    names = document.get("array_names")
    generation = document.get("generation", -1)
    task_expected = generation >= 1
    model_expected = generation >= 2
    geometry_expected = generation >= 3
    fixed = bool(
        isinstance(names, list) and names == sorted(arrays)
        and set(document.get("array_hashes", {})) == set(arrays)
        and document["array_hashes"] == {name: array_hash(arrays[name]) for name in names}
        and all(np.isfinite(np.asarray(value)).all() for value in arrays.values())
        and np.asarray(arrays["generation_int64"]).shape == ()
        and int(np.asarray(arrays["generation_int64"]).item()) == document["generation"]
        and np.asarray(arrays["completed_count_int64"]).shape == ()
        and int(np.asarray(arrays["completed_count_int64"]).item())
        == len(document["completed_identities"])
        and np.asarray(arrays["incomplete_bool"]).shape == ()
        and bool(np.asarray(arrays["incomplete_bool"]).item()) is True
        and np.asarray(arrays["task_authentication_bool"]).dtype == np.bool_
        and np.asarray(arrays["task_authentication_bool"]).shape == ()
        and bool(arrays["task_authentication_bool"]) is task_expected
        and np.asarray(arrays["model_authentication_bool"]).dtype == np.bool_
        and np.asarray(arrays["model_authentication_bool"]).shape == ()
        and bool(arrays["model_authentication_bool"]) is model_expected
        and np.asarray(arrays["geometry_authentication_bool"]).dtype == np.bool_
        and np.asarray(arrays["geometry_authentication_bool"]).shape == ()
        and bool(arrays["geometry_authentication_bool"]) is geometry_expected
    )
    if not fixed:
        return False
    if task_expected:
        value = np.asarray(arrays["task_array_hash_digest_uint8"])
        fixed &= value.shape == (32,) and value.dtype == np.uint8
        fixed &= document.get("task_array_hash_digest_hex") == value.tobytes().hex()
    if model_expected:
        value = np.asarray(arrays["model_array_hash_digest_uint8"])
        cross = np.asarray(arrays["cross_bind_bool"])
        fixed &= value.shape == (32,) and value.dtype == np.uint8
        fixed &= document.get("model_array_hash_digest_hex") == value.tobytes().hex()
        fixed &= cross.shape == () and cross.dtype == np.bool_ and bool(cross)
    return bool(fixed)


def certify_checkpoint_authentication(document: Mapping, arrays: Mapping,
                                      prerequisite_detail, geometry_digest_hex):
    generation = document.get("generation", -1)
    if not certify_checkpoint_arrays(document, arrays):
        return False
    def canonical(value):
        if isinstance(value, Mapping): return {str(key): canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)): return [canonical(item) for item in value]
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        return value
    task_digest = hashlib.sha256(json.dumps(canonical(
        prerequisite_detail["task"]), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    model_digest = hashlib.sha256(json.dumps(canonical({
        "model": prerequisite_detail["model"],
        "cross_bind": prerequisite_detail["cross_bind"],
    }), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return bool(
        (generation < 1 or (
            document["task_array_hash_digest_hex"] == task_digest
            and np.array_equal(arrays["task_array_hash_digest_uint8"],
                               np.frombuffer(bytes.fromhex(task_digest), np.uint8))
        ))
        and (generation < 2 or (
            document["model_array_hash_digest_hex"] == model_digest
            and np.array_equal(arrays["model_array_hash_digest_uint8"],
                               np.frombuffer(bytes.fromhex(model_digest), np.uint8))
        ))
        and (generation < 3 or geometry_digest_hex == document.get("geometry_digest_hex"))
    )


def certify_array_manifest(arrays: Mapping):
    expected = expected_final_array_names()
    if tuple(sorted(arrays)) != expected:
        return {"passes": False}
    gates = {}
    for index, identity in enumerate(EXPECTED_LEDGER):
        prefix = f"profile_{index:03d}_{identity[1]}_{identity[2]}_{identity[3]}"
        for name, (shape, dtype) in PROFILE_ARRAY_SPECS.items():
            value = np.asarray(arrays[f"{prefix}_{name}"])
            gates[f"{prefix}_{name}"] = bool(
                value.shape == shape and value.dtype == dtype and np.isfinite(value).all()
            )
    base = {
        "public_x0_float32": ((12, 14), np.float32),
        "public_reference_float32": ((12, 960), np.float32),
        "public_default_side_int8": ((12,), np.int8),
        "quarantined_q8_float64": ((12, 7), np.float64),
        "joint_lower_float64": ((7,), np.float64),
        "joint_upper_float64": ((7,), np.float64),
        "velocity_limit_float64": ((7,), np.float64),
        "effort_limit_float64": ((7,), np.float64),
        "b1_x0_float32": ((12, 1, 14), np.float32),
        "b1_reference_float32": ((12, 1, 960), np.float32),
        "b1_seed_xu_float32": ((12, 1, 2009), np.float32),
        "b16_x0_float32": ((12, 16, 14), np.float32),
        "b16_reference_float32": ((12, 16, 960), np.float32),
        "b16_seed_xu_float32": ((12, 16, 2009), np.float32),
    }
    for name, (shape, dtype) in base.items():
        value = np.asarray(arrays[name])
        gates[name] = value.shape == shape and value.dtype == np.dtype(dtype) and np.isfinite(value).all()
    for name, (shape, dtype) in DERIVATIVE_ARRAY_SPECS.items():
        value = np.asarray(arrays[name])
        gates[name] = value.shape == shape and value.dtype == dtype and np.isfinite(value).all()
    for name, (shape, dtype) in GLOBAL_ARRAY_SPECS.items():
        value = np.asarray(arrays[name])
        gates[name] = value.shape == shape and value.dtype == dtype and np.isfinite(value).all()
    for probe in range(3):
        base_name = f"derivative_probe{probe}_inequality_csr_"
        dynamic = {
            "data_float64": (np.float64, 1), "indices_int32": (np.int32, 1),
            "indptr_int32": (np.int32, 1), "shape_int64": (np.int64, 1),
            "nnz_int64": (np.int64, 0),
        }
        for field, (dtype, ndim) in dynamic.items():
            value = np.asarray(arrays[base_name + field])
            gates[base_name + field] = value.dtype == dtype and value.ndim == ndim and np.isfinite(value).all()
    return {"gates": gates, "passes": bool(all(gates.values()))}


def certify_final_document(summary: Mapping, arrays: Mapping):
    required = {
        "protocol", "incomplete", "overall_pass", "canonical_profile_count",
        "array_names", "array_hashes", "npz_path", "npz_sha256", "provenance",
        "checkpoint_count", "watchdog", "prerequisite_authentication",
        "canonical_rows", "task_certificates", "campaign_certificate",
        "b1_b16_certificate", "derivative_certificate",
        "storage_report", "cuda_worker_rows",
    }
    schema = certify_array_manifest(arrays)
    names = list(expected_final_array_names())
    hashes = {name: array_hash(arrays[name]) for name in names} if schema["passes"] else {}
    structural = bool(schema["passes"] and set(summary) == required)
    fresh_prerequisite = None
    semantic = {"passes": False}
    if structural:
        try:
            fresh_prerequisite = authenticate_accepted_prerequisites()
            semantic = _production_semantic_recert(summary, arrays, fresh_prerequisite)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, ImportError):
            semantic = {"passes": False}
    retained_storage = summary.get("storage_report", {})
    try:
        worker_side_bytes = worker_side_npz_array_bytes(summary.get("cuda_worker_rows", ()))
    except (OSError, ValueError, KeyError, TypeError):
        worker_side_bytes = -1
    storage = storage_report(
        arrays, retained_storage.get("checkpoint_bytes", ()), worker_side_bytes,
    )
    gates = {
        "exact_keys": set(summary) == required,
        "protocol": summary.get("protocol") == PROTOCOL_VERSION,
        "complete": summary.get("incomplete") is False and summary.get("overall_pass") is True,
        "count": summary.get("canonical_profile_count") == len(EXPECTED_LEDGER),
        "arrays": schema["passes"] and summary.get("array_names") == names
        and summary.get("array_hashes") == hashes,
        "provenance": certify_provenance(summary.get("provenance", {}), final=True),
        "checkpoints": summary.get("checkpoint_count") == len(EXPECTED_LEDGER) + 5,
        "prerequisites": fresh_prerequisite is not None
        and summary.get("prerequisite_authentication") == fresh_prerequisite[0]
        and fresh_prerequisite[0].get("passes") is True,
        "watchdog": summary.get("watchdog", {}).get("runtime_watchdog_rejected") is False,
        "semantic_recertification": semantic.get("passes") is True,
        "storage": summary.get("storage_report") == storage and storage["passes"],
    }
    return {"gates": gates, "semantic": semantic, "passes": bool(all(gates.values()))}


FINAL_NPZ_BYTE_CAP = 512 * 1024**2
CHECKPOINT_NPZ_BYTE_CAP = 4 * 1024**2
CUMULATIVE_NPZ_BYTE_CAP = 1024**3
# Pre-data algebraic sparsity projection using the frozen 4114x665 Jacobian
# block structure. Runtime publication records and gates actual nbytes.
def _schema_nbytes(shape, dtype):
    return int(np.prod(shape, dtype=np.int64) if shape else 1) * np.dtype(dtype).itemsize


PROJECTED_PROFILE_BYTES = sum(
    _schema_nbytes(shape, dtype) for shape, dtype in PROFILE_ARRAY_SPECS.values()
)
PROJECTED_CSR_NNZ_PER_PROBE = 1_075_970
PROJECTED_FINAL_BYTES = 368_330_356
_CHECKPOINT_FIXED_PROJECTED_BYTES = {
    "generation_int64": 8, "incomplete_bool": 1, "completed_count_int64": 8,
    "task_authentication_bool": 1, "model_authentication_bool": 1,
    "geometry_authentication_bool": 1, "task_array_hash_digest_uint8": 32,
    "model_array_hash_digest_uint8": 32, "cross_bind_bool": 1,
}


def projected_checkpoint_array_bytes(generation):
    total = 0
    for name in _checkpoint_array_names(generation):
        if name in _CHECKPOINT_FIXED_PROJECTED_BYTES:
            total += _CHECKPOINT_FIXED_PROJECTED_BYTES[name]
            continue
        matched = sorted(
            (suffix for suffix in PROFILE_ARRAY_SPECS if name.endswith("_" + suffix)),
            key=len, reverse=True,
        )
        if not matched:
            raise AssertionError(f"unprojected checkpoint array {name}")
        total += _schema_nbytes(*PROFILE_ARRAY_SPECS[matched[0]])
    return total


PROJECTED_CHECKPOINT_BYTES = tuple(
    projected_checkpoint_array_bytes(generation)
    for generation in range(len(EXPECTED_LEDGER) + 5)
)
PROJECTED_CHECKPOINT_CUMULATIVE_BYTES = sum(PROJECTED_CHECKPOINT_BYTES)
PROJECTED_NAMESPACE_BYTES = PROJECTED_FINAL_BYTES + PROJECTED_CHECKPOINT_CUMULATIVE_BYTES
PROJECTED_WORKER_INPUT_NPZ_ARRAY_BYTES = (14 + INTERVALS * 7) * 4 * len(EXPECTED_LEDGER)
PROJECTED_WORKER_OUTPUT_NPZ_ARRAY_BYTES = (
    14 + INTERVALS * 7 + DENSE_SAMPLES * 14 + DENSE_SAMPLES * 3
) * 4 * len(EXPECTED_LEDGER)
PROJECTED_WORKER_SIDE_NPZ_ARRAY_BYTES = (
    PROJECTED_WORKER_INPUT_NPZ_ARRAY_BYTES + PROJECTED_WORKER_OUTPUT_NPZ_ARRAY_BYTES
)
PROJECTED_NAMESPACE_WITH_WORKER_SIDE_BYTES = (
    PROJECTED_NAMESPACE_BYTES + PROJECTED_WORKER_SIDE_NPZ_ARRAY_BYTES
)


def uncompressed_array_bytes(arrays: Mapping):
    return int(sum(np.asarray(value).nbytes for value in arrays.values()))


def storage_report(final_arrays: Mapping, checkpoint_bytes, worker_side_npz_bytes=0):
    final_bytes = uncompressed_array_bytes(final_arrays)
    checkpoints = [int(value) for value in checkpoint_bytes]
    cumulative = final_bytes + sum(checkpoints)
    candidate_peak = cumulative + max([final_bytes, *checkpoints], default=0)
    gates = {
        "final_within_512_mib": final_bytes <= FINAL_NPZ_BYTE_CAP,
        "each_checkpoint_within_4_mib": bool(checkpoints)
        and max(checkpoints) <= CHECKPOINT_NPZ_BYTE_CAP,
        "cumulative_within_1_gib": cumulative <= CUMULATIVE_NPZ_BYTE_CAP,
        "candidate_peak_within_1_gib": candidate_peak <= CUMULATIVE_NPZ_BYTE_CAP,
        "worker_side_npz_bytes_nonnegative": int(worker_side_npz_bytes) >= 0,
    }
    return {
        "per_array_bytes": {name: int(np.asarray(value).nbytes)
                            for name, value in sorted(final_arrays.items())},
        "checkpoint_bytes": checkpoints, "final_bytes": final_bytes,
        "cumulative_bytes": cumulative,
        "worker_side_npz_array_bytes": int(worker_side_npz_bytes),
        "namespace_including_worker_side_npz_array_bytes": cumulative + int(worker_side_npz_bytes),
        "candidate_peak_projected_bytes": candidate_peak,
        "caps": {"final": FINAL_NPZ_BYTE_CAP, "checkpoint": CHECKPOINT_NPZ_BYTE_CAP,
                 "cumulative": CUMULATIVE_NPZ_BYTE_CAP},
        "operational_runtime_projection_source": "campaign_watchdog_elapsed/completed*192",
        "gates": gates, "passes": bool(all(gates.values())),
    }


def worker_side_npz_array_bytes(worker_rows):
    total = 0
    for index, row in enumerate(worker_rows):
        from gato_tiago.circular_portfolio_worker import expected_worker_paths
        paths = expected_worker_paths(index)
        if (row.get("input_path") != str(paths["input_path"])
                or row.get("npz_path") != str(paths["npz_path"])):
            raise ValueError("worker NPZ paths are not canonical")
        for path in (paths["input_path"], paths["npz_path"]):
            with np.load(path, allow_pickle=False) as archive:
                total += sum(int(archive[name].nbytes) for name in archive.files)
    return total


def _production_pin_context():  # pragma: no cover - runtime model boundary
    import pinocchio as pin
    from gato_tiago.multimodal_pillar import (
        TOOL_FRAME, load_model, tool_position,
    )
    repo = Path(__file__).resolve().parents[2]
    model = load_model(repo / "gato/dynamics/tiago_right/tiago_right_arm.urdf")
    tool_id = model.getFrameId(TOOL_FRAME)
    def kinematics(q):
        data = model.createData()
        position = tool_position(model, data, q)
        jacobian = pin.computeFrameJacobian(
            model, data, np.asarray(q), tool_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3]
        return position, jacobian
    def rnea(q, qd, qdd):
        return pin.rnea(model, model.createData(), q, qd, qdd).copy()
    def rnea_derivatives(q, qd, qdd):
        data = model.createData()
        dq, dv, da = pin.computeRNEADerivatives(model, data, q, qd, qdd)
        return dq.copy(), dv.copy(), da.copy()
    def aba(q, qd, u):
        return pin.aba(model, model.createData(), q, qd, u).copy()
    def velocity(_model, q, qd):
        return kinematics(q)[1] @ np.asarray(qd)
    def dense_replay(x0, controls):
        from gato_tiago.circular_portfolio import DENSE_SUBSTEPS, DT
        state = np.empty((len(controls) * DENSE_SUBSTEPS + 1, 14), np.float64)
        tool = np.empty((len(state), 3), np.float64)
        state[0] = x0; tool[0] = kinematics(x0[:7])[0]
        index = 0; step = DT / DENSE_SUBSTEPS
        for control in controls:
            for _ in range(DENSE_SUBSTEPS):
                q, qd = state[index, :7], state[index, 7:]
                qdd = aba(q, qd, control)
                state[index + 1, :7] = q + step * qd + .5 * step**2 * qdd
                state[index + 1, 7:] = qd + step * qdd
                index += 1; tool[index] = kinematics(state[index, :7])[0]
        return state, tool
    return model, kinematics, rnea, rnea_derivatives, aba, velocity, dense_replay


def _derivative_evaluation(x0, tool_reference, pillar, lower, upper, velocity,
                           effort, kinematics, rnea, rnea_derivatives):
    from gato_tiago.circular_portfolio_constructor import (
        integrate_acceleration, position_and_control_sensitivities,
        reconstruct_controls, shooting_inequality_jacobian,
        shooting_inequalities, shooting_objective, shooting_objective_gradient,
    )
    x0 = np.asarray(x0).copy(); tool_reference = np.asarray(tool_reference).copy()
    pillar = np.asarray(pillar).copy()
    def evaluate(flat):
        qdd = np.asarray(flat).reshape(95, 7)
        q, qd = integrate_acceleration(x0[:7], np.zeros(7), qdd)
        controls = reconstruct_controls(q, qd, qdd, rnea)
        positions, dp, du = position_and_control_sensitivities(
            q, qd, qdd, kinematics, rnea_derivatives
        )
        return qdd, q, qd, controls, positions, dp, du
    def objective(flat):
        qdd, q, _qd, controls, positions, _dp, _du = evaluate(flat)
        return shooting_objective(q, qdd, controls, positions, tool_reference, effort)
    def objective_gradient(flat):
        qdd, _q, _qd, controls, positions, dp, du = evaluate(flat)
        return shooting_objective_gradient(qdd, controls, positions, tool_reference, effort, dp, du)
    def inequality(flat):
        _qdd, q, qd, controls, positions, _dp, _du = evaluate(flat)
        return shooting_inequalities(q, qd, controls, positions, lower, upper, velocity, effort, pillar)
    def inequality_jacobian(flat):
        _qdd, q, qd, controls, positions, dp, du = evaluate(flat)
        return shooting_inequality_jacobian(q, qd, controls, positions, pillar, dp, du)
    return objective, objective_gradient, inequality, inequality_jacobian


def _production_semantic_recert(summary, arrays, prerequisite_bundle):  # pragma: no cover
    """Canonical final evidence is rebuilt from NPZ and authenticated Pin callbacks."""
    from gato_tiago.circular_portfolio import (
        TASK_IDENTITIES, construct_geometry, circular_reference,
        reference_from_geometry,
    )
    from gato_tiago.circular_portfolio_constructor import (
        ShootingResult, aggregate_campaign, aggregate_task_profiles,
        certify_b1_b16_binding, certify_independent_replay,
        certify_portfolio_topology, certify_production_derivative_evidence,
        certify_proxy_construction, certify_proxy_kkt, certify_seed_reversal,
        certify_shooting_result, integrate_acceleration, position_and_control_sensitivities,
        reconstruct_controls, pack_solver_seed, unpack_solver_seed,
        shooting_inequality_jacobian,
        shooting_inequalities, shooting_objective, shooting_objective_gradient,
    )
    detail, _task_summary, task_arrays, _model_summary, model_arrays = prerequisite_bundle
    if not detail.get("passes"):
        return {"passes": False}
    model, kinematics, rnea, rnea_derivatives, aba, tool_velocity, dense_replay = _production_pin_context()
    lower = np.asarray(model_arrays["model_lower_float64"])
    upper = np.asarray(model_arrays["model_upper_float64"])
    velocity = np.asarray(model_arrays["model_velocity_float64"])
    effort = np.asarray(model_arrays["model_effort_float64"])
    canonical_rows = []; task_certificates = []; derivative_evaluations = []
    worker_rows = summary.get("cuda_worker_rows")
    if not isinstance(worker_rows, list) or len(worker_rows) != len(EXPECTED_LEDGER):
        return {"cuda_worker_ledger": False, "passes": False}
    b1_b16_details = []
    geometry_hash_inputs = []
    for task_index, identity in enumerate(TASK_IDENTITIES):
        seed = identity[1]; x0 = np.asarray(task_arrays[f"task_{seed}_x0_float32"])
        accepted_ref = np.asarray(task_arrays[f"task_{seed}_reference_float32"])
        q_goal = np.asarray(task_arrays[f"task_{seed}_history_q_float64"])[8]
        default_side = int(_task_summary["rows"][task_index]["public_task"]["default_side"])
        if not (np.array_equal(arrays["public_x0_float32"][task_index], x0)
                and int(arrays["public_default_side_int8"][task_index]) == default_side
                and np.array_equal(arrays["quarantined_q8_float64"][task_index], q_goal)):
            return {"passes": False}
        if not np.array_equal(x0[7:], np.zeros(7, np.float32)):
            return {"accepted_x0_velocity_zero": False, "passes": False}
        geometry = construct_geometry(kinematics(x0[:7])[0], accepted_ref[:3], default_side)
        expected_solver_reference = np.tile(
            reference_from_geometry(geometry).as_float32(), (96, 1)
        ).ravel()
        if not np.array_equal(arrays["public_reference_float32"][task_index], expected_solver_reference):
            return {"passes": False}
        geometry_hash_inputs.append(
            np.ascontiguousarray(expected_solver_reference).tobytes()
            + np.asarray(default_side, np.int8).tobytes()
            + np.ascontiguousarray(q_goal).tobytes()
        )
        profile_rows = []; topology_rows = []; reversal_rows = {}
        for lane in range(16):
            index = task_index * 16 + lane; ledger = EXPECTED_LEDGER[index]
            prefix = f"profile_{index:03d}_{ledger[1]}_{ledger[2]}_{ledger[3]}"
            retained = {name: arrays[f"{prefix}_{name}"] for name in PROFILE_ARRAY_SPECS}
            route, profile = ledger[2], ledger[3]
            tool_reference = circular_reference(geometry, route, profile)
            proxy = {name: retained[name] for name in (
                "proxy_q_before_float64", "proxy_position_float64", "proxy_residual_float64",
                "proxy_jacobian_float64", "proxy_dq_float64", "proxy_q_float64")}
            kkt = {name: retained[name] for name in retained if name.startswith("proxy_kkt_")}
            result = ShootingResult(
                ledger, retained["acceleration_float64"], retained["q_float64"],
                retained["qd_float64"], retained["controls_float64"],
                retained["positions_float64"], float(retained["shooting_objective_float64"]),
                retained["endpoint_residual_float64"], retained["inequalities_float64"],
                bool(retained["optimizer_success_bool"]), int(retained["optimizer_status_int64"]),
                int(retained["optimizer_iterations_int64"]), float(retained["optimizer_wall_float64"]),
            )
            gates = {
                "proxy": certify_proxy_construction(proxy, x0[:7], q_goal, tool_reference, kinematics)["passes"],
                "proxy_kkt": certify_proxy_kkt(
                    kkt, x0[:7], q_goal, proxy["proxy_q_float64"],
                    {name: arrays[name] for name in GLOBAL_ARRAY_SPECS},
                )["passes"],
                "shooting": certify_shooting_result(
                    result, x0[:7], q_goal, tool_reference, lower, upper, velocity,
                    effort, geometry.pillar_xy, kinematics, rnea, aba,
                )["passes"],
            }
            applied_controls = retained["controls_float64"].astype(np.float32)
            if not np.array_equal(retained["applied_controls_float32"], applied_controls):
                return {"applied_control_cast": False, "passes": False}
            pin_applied_controls = applied_controls.astype(np.float64)
            pin = {"state": retained["pin_dense_state_float64"], "control": applied_controls,
                   "tool": retained["pin_dense_tool_float64"], "joint_lower": lower,
                   "joint_upper": upper, "velocity_limit": velocity, "effort_limit": effort,
                   "pillar_xy": geometry.pillar_xy, "physical_radius_m": .03,
                   "x0": x0, "reference_float32": expected_solver_reference}
            cuda = {**pin, "state": retained["cuda_dense_state_float32"],
                    "tool": retained["cuda_dense_tool_float32"]}
            try:
                worker_row = worker_rows[index]
                from gato_tiago.circular_portfolio_worker import (
                    certify_lane_binding, certify_worker_output, expected_worker_paths,
                )
                expected_paths = expected_worker_paths(index)
                worker_npz = expected_paths["npz_path"]
                worker_json = expected_paths["json_path"]
                request_path = expected_paths["request_path"]
                input_path = expected_paths["input_path"]
                worker_summary = json.loads(worker_json.read_text())
                request = json.loads(request_path.read_text())
                with np.load(worker_npz, allow_pickle=False) as archive:
                    worker_arrays = {name: archive[name] for name in archive.files}
                with np.load(input_path, allow_pickle=False) as archive:
                    worker_inputs = {name: archive[name] for name in archive.files}
                worker_gate = bool(
                    worker_row == {
                        "identity": list(ledger), "json_path": str(worker_json),
                        "json_sha256": sha256_file(worker_json), "npz_path": str(worker_npz),
                        "npz_sha256": sha256_file(worker_npz), "request_path": str(request_path),
                        "request_sha256": sha256_file(request_path), "input_path": str(input_path),
                        "input_sha256": sha256_file(input_path),
                    }
                    and certify_worker_output(worker_summary, worker_arrays, request, worker_inputs)["passes"]
                    and certify_lane_binding(
                        index, request_path, request, worker_inputs, worker_arrays,
                        x0, applied_controls,
                    )
                    and np.array_equal(worker_arrays["cuda_dense_state_float32"], retained["cuda_dense_state_float32"])
                    and np.array_equal(worker_arrays["cuda_dense_tool_float32"], retained["cuda_dense_tool_float32"])
                )
            except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
                worker_gate = False
            gates["owned_cuda_worker"] = worker_gate
            gates["replay"] = certify_independent_replay(
                pin, cuda, tool_reference, route, expected_solver_reference, tool_velocity
            )["passes"]
            replay_state, replay_tool = dense_replay(x0, pin_applied_controls)
            gates["pin_replay_recomputed"] = bool(
                np.array_equal(replay_state, retained["pin_dense_state_float64"])
                and np.array_equal(replay_tool, retained["pin_dense_tool_float64"])
            )
            if not all(gates.values()):
                return {"failed_identity": list(ledger), "gates": gates, "passes": False}
            row = {"identity": list(ledger), "route": route, "profile": profile,
                   "attempt_count": 1, "retained": True, "filtered": False,
                   "replacement": None, "passes": bool(all(gates.values())), "gates": gates}
            canonical_rows.append(row); profile_rows.append(row)
            if index in (0, 8, 191):
                distances = np.linalg.norm(
                    retained["positions_float64"][:, :2] - geometry.pillar_xy, axis=1
                )
                knot = 48 if index == 0 else int(np.argmin(distances)) if index == 8 else 94
                objective, objective_gradient, inequality, inequality_jacobian = _derivative_evaluation(
                    x0, tool_reference, geometry.pillar_xy, lower, upper,
                    velocity, effort, kinematics, rnea, rnea_derivatives,
                )
                derivative_evaluations.append({
                    "q": retained["q_float64"][knot], "qd": retained["qd_float64"][knot],
                    "qdd": retained["acceleration_float64"][min(knot, 94)],
                    "rnea": rnea, "rnea_derivatives": rnea_derivatives,
                    "objective": objective, "objective_gradient": objective_gradient,
                    "inequality": inequality, "inequality_jacobian": inequality_jacobian,
                })
            topology_rows.append({"lane": lane, "route": route, "pillar_xy": geometry.pillar_xy,
                                  "pin_tool": retained["pin_dense_tool_float64"],
                                  "cuda_tool": retained["cuda_dense_tool_float32"]})
            reversal_rows[(route, profile)] = {
                "pin_path": retained["pin_dense_tool_float64"], "cuda_path": retained["cuda_dense_tool_float32"],
                "pin_state": retained["pin_dense_state_float64"], "cuda_state": retained["cuda_dense_state_float32"],
                "controls": applied_controls, "reference_float32": expected_solver_reference,
                "joint_lower": lower, "joint_upper": upper, "velocity_limit": velocity, "effort_limit": effort,
                "pin_base_cost": retained["pin_base_cost_float64"],
                "cuda_base_cost": retained["cuda_base_cost_float64"],
                "pin_full_cost": retained["pin_full_cost_float64"],
                "cuda_full_cost": retained["cuda_full_cost_float64"],
                "pin_toll_residual": retained["pin_toll_residual_float64"],
                "cuda_toll_residual": retained["cuda_toll_residual_float64"],
            }
        topology = certify_portfolio_topology(topology_rows)
        reversal = {"passes": all(certify_seed_reversal(
            reversal_rows[("short", profile)], reversal_rows[("long", profile)]
        )["passes"] for profile in range(8))}
        aggregate = aggregate_task_profiles(profile_rows, topology, reversal)
        task_certificates.append({"identity": list(identity), "profile_count": 16,
                                  "passes": aggregate["passes"], "certificate": aggregate})
        seeds = []
        for lane in range(16):
            index = task_index * 16 + lane; ledger = EXPECTED_LEDGER[index]
            prefix = f"profile_{index:03d}_{ledger[1]}_{ledger[2]}_{ledger[3]}"
            seed = pack_solver_seed(
                arrays[f"{prefix}_q_float64"], arrays[f"{prefix}_qd_float64"],
                arrays[f"{prefix}_applied_controls_float32"],
            )
            seed_q, seed_qd, seed_u = unpack_solver_seed(seed)
            if not (
                np.array_equal(seed_q, arrays[f"{prefix}_q_float64"].astype(np.float32))
                and np.array_equal(seed_qd, arrays[f"{prefix}_qd_float64"].astype(np.float32))
                and np.array_equal(seed_u, arrays[f"{prefix}_applied_controls_float32"])
                and np.array_equal(seed_qd[0], x0[7:])
            ):
                return {"solver_seed_layout": False, "passes": False}
            seeds.append(seed)
        b16 = {"x0_float32": arrays["b16_x0_float32"][task_index],
               "reference_float32": arrays["b16_reference_float32"][task_index],
               "seed_xu_float32": arrays["b16_seed_xu_float32"][task_index]}
        b1 = {"x0_float32": arrays["b1_x0_float32"][task_index],
              "reference_float32": arrays["b1_reference_float32"][task_index],
              "seed_xu_float32": arrays["b1_seed_xu_float32"][task_index]}
        if not (
            np.array_equal(b16["x0_float32"], np.repeat(x0[None], 16, axis=0))
            and np.array_equal(b16["reference_float32"], np.repeat(expected_solver_reference[None], 16, axis=0))
            and np.array_equal(b16["seed_xu_float32"], np.stack(seeds))
            and all(np.array_equal(b1[name], b16[name][:1]) for name in b1)
        ):
            return {"solver_boundary_bytes": False, "passes": False}
        b1_b16_details.append(certify_b1_b16_binding(b1, b16))
    campaign = aggregate_campaign(task_certificates)
    derivative_names = set(DERIVATIVE_ARRAY_SPECS) | {
        f"derivative_probe{probe}_inequality_csr_{field}"
        for probe in range(3) for field in CSR_FIELDS
    }
    derivative_arrays = {name: arrays[name] for name in derivative_names}
    derivative = certify_production_derivative_evidence(
        derivative_arrays, derivative_evaluations
    )
    b1_b16 = {"task_details": b1_b16_details,
              "passes": bool(all(row["passes"] for row in b1_b16_details))}
    semantic = {
        "canonical_rows": canonical_rows, "task_certificates": task_certificates,
        "campaign_certificate": campaign,
        "b1_b16_certificate": b1_b16, "derivative_certificate": derivative,
        "geometry_digest": hashlib.sha256(b"".join(geometry_hash_inputs)).hexdigest(),
    }
    gates = {
        "rows": summary["canonical_rows"] == canonical_rows,
        "tasks": summary["task_certificates"] == task_certificates,
        "campaign": summary["campaign_certificate"] == campaign and campaign["passes"],
        "b1_b16": summary["b1_b16_certificate"] == b1_b16 and b1_b16["passes"],
        "derivatives": summary["derivative_certificate"] == derivative and derivative["passes"],
    }
    return {**semantic, "gates": gates, "passes": bool(all(gates.values()))}


def recertify_retained_transaction(output):
    output = Path(output).resolve()
    try:
        summary = json.loads(output.read_text())
        npz_path = Path(summary["npz_path"]).resolve()
        if npz_path != output.with_suffix(".npz") or sha256_file(npz_path) != summary["npz_sha256"]:
            return {"passes": False}
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        certificate = certify_final_document(summary, arrays)
        manifest_path = output.with_name(output.stem + ".manifest.json")
        pointer_path = output.with_name(output.stem + ".partial.latest.json")
        manifest = json.loads(manifest_path.read_text())
        pointer = json.loads(pointer_path.read_text())
        side = manifest.get("side_artifacts")
        expected_side_count = 2 * (len(EXPECTED_LEDGER) + 5) + 4 * len(EXPECTED_LEDGER)
        checkpoints_ok = isinstance(side, list) and len(side) == expected_side_count
        fresh = authenticate_accepted_prerequisites()
        for generation in range(len(EXPECTED_LEDGER) + 5):
            json_path = output.with_name(f"{output.stem}.partial.gen{generation}.json")
            npz_checkpoint = output.with_name(f"{output.stem}.partial.gen{generation}.npz")
            row = json.loads(json_path.read_text())
            checkpoints_ok &= certify_checkpoint(row, generation)
            with np.load(npz_checkpoint, allow_pickle=False) as archive:
                checkpoint_arrays = {name: archive[name] for name in archive.files}
            checkpoints_ok &= certify_checkpoint_arrays(row, checkpoint_arrays)
            checkpoints_ok &= certify_checkpoint_authentication(
                row, checkpoint_arrays, fresh[0],
                certificate["semantic"].get("geometry_digest"),
            )
            checkpoints_ok &= row["npz_path"] == str(npz_checkpoint)
            checkpoints_ok &= row["npz_sha256"] == sha256_file(npz_checkpoint)
            if 4 <= generation <= len(EXPECTED_LEDGER) + 3:
                profile_index = generation - 4
                checkpoints_ok &= row["profile_certificate"] == certificate["semantic"]["canonical_rows"][profile_index]
                prefix = f"profile_{profile_index:03d}_{EXPECTED_LEDGER[profile_index][1]}_{EXPECTED_LEDGER[profile_index][2]}_{EXPECTED_LEDGER[profile_index][3]}"
                checkpoints_ok &= all(
                    np.array_equal(checkpoint_arrays[name], arrays[name])
                    for name in checkpoint_arrays if name.startswith(prefix + "_")
                )
            expected_side = [
                {"path": str(json_path), "sha256": sha256_file(json_path)},
                {"path": str(npz_checkpoint), "sha256": sha256_file(npz_checkpoint)},
            ]
            checkpoints_ok &= all(item in side for item in expected_side)
        for worker_row in summary.get("cuda_worker_rows", ()):
            for path_key, hash_key in (
                ("request_path", "request_sha256"), ("input_path", "input_sha256"),
                ("json_path", "json_sha256"), ("npz_path", "npz_sha256"),
            ):
                checkpoints_ok &= {
                    "path": worker_row[path_key], "sha256": worker_row[hash_key]
                } in side
        document_ok = bool(
            manifest.get("protocol") == PROTOCOL_VERSION
            and manifest.get("json_path") == str(output)
            and manifest.get("json_sha256") == sha256_file(output)
            and manifest.get("npz_path") == str(npz_path)
            and manifest.get("npz_sha256") == sha256_file(npz_path)
            and pointer == {
                "protocol": PROTOCOL_VERSION, "incomplete": False,
                "json_path": str(output), "json_sha256": sha256_file(output),
                "npz_path": str(npz_path), "npz_sha256": sha256_file(npz_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
        return {"final": certificate, "checkpoints": checkpoints_ok,
                "documents": document_ok,
                "passes": bool(certificate["passes"] and checkpoints_ok and document_ok)}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return {"passes": False}


def refuse_existing_artifacts(output):
    output = Path(output).resolve()
    candidates = [output, output.with_suffix(".npz"),
                  output.with_name(output.stem + ".manifest.json"),
                  output.with_name(output.stem + ".partial.latest.json")]
    candidates.extend(output.parent.glob(output.stem + ".partial.gen*"))
    if any(path.exists() for path in candidates):
        raise FileExistsError("circular portfolio transaction forbids overwrite/resume")


def _write_json_exclusive(path, document):
    def convert(value):
        if isinstance(value, Mapping): return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)): return [convert(item) for item in value]
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, Path): return str(value)
        return value
    payload = (json.dumps(convert(document), sort_keys=True, separators=(",", ":")) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(payload)


def publish_checkpoint(output, document: Mapping, arrays: Mapping, *,
                       prerequisite_detail=None, geometry_digest_hex=None):
    """Atomically publish one immutable, still-incomplete generation."""
    output = Path(output).resolve()
    generation = document.get("generation")
    if not isinstance(generation, int):
        raise ValueError("checkpoint generation invalid")
    if uncompressed_array_bytes(arrays) > CHECKPOINT_NPZ_BYTE_CAP:
        raise ValueError("checkpoint exceeds frozen 4 MiB uncompressed cap")
    json_path = output.with_name(f"{output.stem}.partial.gen{generation}.json")
    npz_path = output.with_name(f"{output.stem}.partial.gen{generation}.npz")
    pointer = output.with_name(output.stem + ".partial.latest.json")
    if json_path.exists() or npz_path.exists():
        raise FileExistsError("checkpoint generation already exists")
    json_tmp = json_path.with_suffix(json_path.suffix + ".candidate")
    npz_tmp = npz_path.with_suffix(npz_path.suffix + ".candidate")
    pointer_tmp = pointer.with_suffix(pointer.suffix + ".candidate")
    if json_tmp.exists() or npz_tmp.exists() or pointer_tmp.exists():
        raise FileExistsError("orphan candidate blocks a new transaction")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with npz_tmp.open("xb") as stream:
            np.savez(stream, **{name: np.asarray(value) for name, value in arrays.items()})
        bound = dict(document)
        bound["npz_path"] = str(npz_path)
        bound["npz_sha256"] = sha256_file(npz_tmp)
        bound["array_names"] = sorted(arrays)
        bound["array_hashes"] = {
            name: array_hash(arrays[name]) for name in sorted(arrays)
        }
        bound["task_array_hash_digest_hex"] = (
            np.asarray(arrays["task_array_hash_digest_uint8"]).tobytes().hex()
            if generation >= 1 else None
        )
        bound["model_array_hash_digest_hex"] = (
            np.asarray(arrays["model_array_hash_digest_uint8"]).tobytes().hex()
            if generation >= 2 else None
        )
        if not certify_checkpoint(bound, generation) or not certify_checkpoint_arrays(bound, arrays):
            raise ValueError("checkpoint failed its exact pre-publication certificate")
        if generation >= 1 and (
            prerequisite_detail is None
            or not certify_checkpoint_authentication(
                bound, arrays, prerequisite_detail, geometry_digest_hex
            )
        ):
            raise ValueError("checkpoint authentication bytes are not owned evidence")
        _write_json_exclusive(json_tmp, bound)
        os.replace(npz_tmp, npz_path)
        os.replace(json_tmp, json_path)
        _write_json_exclusive(pointer_tmp, {
            "protocol": PROTOCOL_VERSION, "incomplete": True,
            "generation": generation, "json_path": str(json_path),
            "json_sha256": sha256_file(json_path), "npz_path": str(npz_path),
            "npz_sha256": sha256_file(npz_path), "superseded_by": None,
        })
        os.replace(pointer_tmp, pointer)
    finally:
        for candidate in (json_tmp, npz_tmp, pointer_tmp):
            if candidate.exists():
                candidate.unlink()
    return json_path, npz_path, pointer


def publish_final_documents(output, summary: Mapping, arrays: Mapping,
                            checkpoint_pairs):
    """Certify candidates before any success-named artifact or pointer exists."""
    output = Path(output).resolve()
    npz_path = output.with_suffix(".npz")
    manifest_path = output.with_name(output.stem + ".manifest.json")
    pointer_path = output.with_name(output.stem + ".partial.latest.json")
    if output.exists() or npz_path.exists() or manifest_path.exists():
        raise FileExistsError("final artifact already exists")
    if len(checkpoint_pairs) != len(EXPECTED_LEDGER) + 5:
        raise ValueError("final publication requires every immutable checkpoint")
    checkpoint_bytes = []
    for _json_checkpoint, npz_checkpoint in checkpoint_pairs:
        with np.load(npz_checkpoint, allow_pickle=False) as archive:
            checkpoint_bytes.append(sum(archive[name].nbytes for name in archive.files))
    worker_side_bytes = worker_side_npz_array_bytes(summary.get("cuda_worker_rows", ()))
    projected_storage = storage_report(arrays, checkpoint_bytes, worker_side_bytes)
    if not projected_storage["passes"]:
        raise ValueError("frozen uncompressed namespace storage cap exceeded")
    npz_tmp = npz_path.with_suffix(".npz.candidate")
    json_tmp = output.with_suffix(".json.candidate")
    manifest_tmp = manifest_path.with_suffix(".json.candidate")
    pointer_tmp = pointer_path.with_suffix(pointer_path.suffix + ".candidate")
    for candidate in (npz_tmp, json_tmp, manifest_tmp, pointer_tmp):
        if candidate.exists():
            raise FileExistsError("orphan final candidate blocks publication")
    try:
        with npz_tmp.open("xb") as stream:
            np.savez(stream, **{name: np.asarray(value) for name, value in arrays.items()})
        candidate_summary = dict(summary)
        candidate_summary["storage_report"] = projected_storage
        candidate_summary["npz_path"] = str(npz_path)
        candidate_summary["npz_sha256"] = sha256_file(npz_tmp)
        certificate = certify_final_document(candidate_summary, arrays)
        if not certificate["passes"]:
            raise ValueError("final candidate failed before publication")
        _write_json_exclusive(json_tmp, candidate_summary)
        side = []
        for json_checkpoint, npz_checkpoint in checkpoint_pairs:
            side.extend((
                {"path": str(json_checkpoint), "sha256": sha256_file(json_checkpoint)},
                {"path": str(npz_checkpoint), "sha256": sha256_file(npz_checkpoint)},
            ))
        for worker_row in candidate_summary["cuda_worker_rows"]:
            for path_key, hash_key in (
                ("request_path", "request_sha256"), ("input_path", "input_sha256"),
                ("json_path", "json_sha256"), ("npz_path", "npz_sha256"),
            ):
                path = Path(worker_row[path_key])
                if not path.is_file() or sha256_file(path) != worker_row[hash_key]:
                    raise ValueError("CUDA worker side artifact changed before final publication")
                side.append({"path": str(path), "sha256": worker_row[hash_key]})
        manifest = {
            "protocol": PROTOCOL_VERSION, "json_path": str(output),
            "json_sha256": sha256_file(json_tmp), "npz_path": str(npz_path),
            "npz_sha256": sha256_file(npz_tmp), "side_artifacts": side,
        }
        _write_json_exclusive(manifest_tmp, manifest)
        os.replace(npz_tmp, npz_path)
        os.replace(json_tmp, output)
        os.replace(manifest_tmp, manifest_path)
        _write_json_exclusive(pointer_tmp, {
            "protocol": PROTOCOL_VERSION, "incomplete": False,
            "json_path": str(output), "json_sha256": sha256_file(output),
            "npz_path": str(npz_path), "npz_sha256": sha256_file(npz_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        })
        os.replace(pointer_tmp, pointer_path)
    finally:
        for candidate in (npz_tmp, json_tmp, manifest_tmp, pointer_tmp):
            if candidate.exists():
                candidate.unlink()


def execute(output, authorization=None):
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("circular portfolio P1 runner is blocked")
    if Path(output).resolve() != OUTPUT_PATH.resolve():
        raise RuntimeError("circular portfolio P1 output path is not authorized")
    refuse_existing_artifacts(output)
    raise RuntimeError("production P1 runner is absent from static checkpoint")


__all__ = ["OUTPUT_PATH", "RUNNER_EXECUTION_AUTHORIZATION", "transaction_schema",
           "certify_checkpoint", "certify_final_document", "recertify_retained_transaction",
           "refuse_existing_artifacts", "publish_checkpoint", "publish_final_documents",
           "watchdog", "execute"]
