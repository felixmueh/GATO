"""Transactional, task-construction-only Tiago toll V2 preflight.

The production boundary is blocked in this checkpoint.  A future authorized
run constructs each frozen task exactly once on CPU, retains every DLS witness
and independent replay, and never imports a CUDA extension or optimizer.
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
from gato_tiago import multimodal_toll_v2_oracle_schema as oracle
from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS
from gato_tiago.multimodal_toll_v2 import (
    EXPECTED_TASK_IDENTITIES,
    REJECTED_PREDECESSOR_ARTIFACT_HASHES,
    REQUIRED_SOURCE_PATHS,
    V2_OUTPUT_PATH,
    V2_PROTOCOL_VERSION,
    frozen_v2_metadata,
)


RUNNER_EXECUTION_AUTHORIZATION = None
AUTHORIZED_OUTPUT_PATH = Path(V2_OUTPUT_PATH)
RUNNER_PROTOCOL_VERSION = "tiago_tool_center_toll_v2_construction_runner_1"
EXPECTED_TASK_COUNT = 12
EXPECTED_MODEL_LOAD_CALLS = 1
EXPECTED_TASK_CONSTRUCTION_CALLS = 12
EXPECTED_HISTORY_REGENERATION_CALLS = 12
EXPECTED_WORKER_CALLS = 0
EXPECTED_CUDA_CALLS = 0
EXPECTED_SQP_CALLS = 0
EXPECTED_ROUTE_ORACLE_CALLS = 0
_PRODUCTION_PIPELINE_TOKEN = object()

# These already-built modules are report-only. This runner never imports or
# launches them; a later separately authorized CUDA stage must consume the
# retained task bytes instead of reopening task RNG.
REPORT_ONLY_EXTENSION_HASHES = {
    "bsqp.bsqpN64_tiago_right_multimodal_toll": "3d3c1e0808a57b6189a0267e701a64d0b11ed13cda37b65873781954ffaa8551",
    "bsqp.bsqpN64_tiago_right": "f0944080523a212edce0af72d323f5eeffd587adde2f0bea6063a31fabff2f41",
    "bsqp.bsqpN64_tiago_right_multimodal": "75650c37b3cd0e729922cdd82e65dd81214bb29f9591d0a7466aac197908b1fe",
}


def repository_root(module_path=Path(__file__)) -> Path:
    root = Path(module_path).resolve().parents[2]
    sentinels = (
        root / ".git",
        root / "CMakeLists.txt",
        root / REQUIRED_SOURCE_PATHS["v2_schema"],
        root / REQUIRED_SOURCE_PATHS["v2_runner"],
        root / REQUIRED_SOURCE_PATHS["model"],
    )
    if not all(path.exists() for path in sentinels):
        raise RuntimeError("Stage V2 repository root sentinel mismatch")
    return root


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        f"{array.dtype.str}|{array.shape}|".encode() + array.tobytes()
    ).hexdigest()


def _is_hex_digest(value, length: int) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
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


def _atomic_json(path: Path, value: Mapping) -> None:
    _atomic_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    )


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
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


def _no_existing_artifacts(output: Path) -> None:
    output = Path(output)
    candidates = [output, output.with_suffix(".npz"), output.with_suffix(".manifest.json")]
    candidates.extend(output.parent.glob(f"{output.stem}.partial.*"))
    if any(path.exists() for path in candidates):
        raise RuntimeError("Stage V2 does not permit resume, overwrite, or rerun")


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _checkpoint(
    output: Path,
    *,
    generation: int,
    stage: str,
    expected: Sequence[tuple],
    attempted: Sequence[tuple],
    completed: Sequence[tuple],
    rows: Sequence[Mapping],
    arrays: Mapping[str, np.ndarray],
    provenance: Mapping,
) -> dict:
    output = Path(output)
    stem = output.parent / f"{output.stem}.partial.{generation:04d}"
    npz_path = Path(f"{stem}.npz")
    json_path = Path(f"{stem}.json")
    pointer_path = output.parent / f"{output.stem}.partial.latest.json"
    if npz_path.exists() or json_path.exists():
        raise RuntimeError("Stage V2 checkpoint generation already exists")
    payload_arrays = {
        str(name): np.ascontiguousarray(value) for name, value in arrays.items()
    }
    _atomic_npz(npz_path, payload_arrays)
    npz_hash = _sha256_file(npz_path)
    attempted_set = set(attempted)
    summary = {
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "v2_protocol_version": V2_PROTOCOL_VERSION,
        "incomplete": True,
        "checkpoint_generation": int(generation),
        "checkpoint_stage": str(stage),
        "expected_identities": [list(row) for row in expected],
        "attempted_identities": [list(row) for row in attempted],
        "completed_identities": [list(row) for row in completed],
        "pending_identities": [list(row) for row in expected if row not in attempted_set],
        "task_construction_attempt_count": len(attempted),
        "model_load_calls": int(generation >= 1),
        "task_rng_generation_calls": len(attempted),
        "task_rng_authentication_calls": sum(
            "raw_draws_exact_for_task_seed" in row for row in rows
        ),
        "history_regeneration_calls": sum(
            "history_regeneration" in row for row in rows
        ),
        "rows": _json_value(list(rows)),
        "array_names": sorted(payload_arrays),
        "array_hashes": {
            name: _array_hash(value) for name, value in payload_arrays.items()
        },
        "npz_path": str(npz_path),
        "npz_sha256": npz_hash,
        "provenance": _json_value(dict(provenance)),
        "worker_calls": 0,
        "cuda_calls": 0,
        "sqp_calls": 0,
        "route_oracle_calls": 0,
        "retry_count": 0,
        "replacement_count": 0,
        "construction_only": True,
        "benchmark_evidence": False,
        "optimization_evidence": False,
        "all_v2_gates_pass": False,
    }
    _atomic_json(json_path, summary)
    json_hash = _sha256_file(json_path)
    pointer = {
        "generation": int(generation),
        "json_path": str(json_path),
        "json_sha256": json_hash,
        "npz_path": str(npz_path),
        "npz_sha256": npz_hash,
    }
    _atomic_json(pointer_path, pointer)
    return pointer


def _source_provenance(repo: Path) -> dict:
    source_hashes = {
        label: {"path": path, "sha256": _sha256_file(repo / path)}
        for label, path in REQUIRED_SOURCE_PATHS.items()
    }
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    ).splitlines()
    return {
        "git_head_at_start": head,
        "tracked_tree_clean_at_start": not tracked,
        "full_git_status_at_start": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).splitlines(),
        "source_hashes": source_hashes,
        "model_path": REQUIRED_SOURCE_PATHS["model"],
        "model_sha256": source_hashes["model"]["sha256"],
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pinocchio_version": str(getattr(oracle.pin, "__version__", "unknown")),
        "exact_command": shlex.join(list(sys.orig_argv)),
        "orig_argv": list(sys.orig_argv),
        "sys_argv": list(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "critical_environment": {
            name: os.environ.get(name) for name in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")
        },
        "predecessor_artifacts_consumed": False,
        "rejected_predecessor_artifact_hashes_report_only": dict(
            REJECTED_PREDECESSOR_ARTIFACT_HASHES
        ),
        "extension_hashes_report_only": dict(REPORT_ONLY_EXTENSION_HASHES),
        "retry_count": 0,
        "replacement_count": 0,
    }


def _expected_task_draws(task_seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(task_seed))
    jitter = rng.uniform(-toll.Q0_JITTER_RAD, toll.Q0_JITTER_RAD, size=7)
    phi = float(rng.uniform(-toll.PLANAR_ANGLE_RAD, toll.PLANAR_ANGLE_RAD))
    sign = int(rng.choice((-1, 1)))
    return np.asarray([*jitter, phi, float(sign)], dtype=np.float64)


def _task_row(identity, task, witness, regenerated, model) -> tuple[dict, dict]:
    phase, task_seed = identity
    history = witness.history
    history_gate = oracle.certify_witness_regeneration(witness, regenerated)
    lower = np.asarray(model.lowerPositionLimit, dtype=np.float64)
    upper = np.asarray(model.upperPositionLimit, dtype=np.float64)
    q_gate = oracle.q_history_gate(history, lower, upper)
    x0 = np.asarray(task.x0, dtype=np.float32)
    reference = task.reference.as_float32()
    q0_expected = history.q_float64[0].astype(np.float32)
    x0_expected = np.concatenate([q0_expected, np.zeros(7, dtype=np.float32)])
    goal_expected = history.tool_position_float64[-1].astype(np.float32)
    start = history.tool_position_float64[0]
    goal = history.tool_position_float64[-1]
    actual_travel = float(np.linalg.norm(goal - start))
    planar_travel = float(np.linalg.norm(goal[:2] - start[:2]))
    vertical = abs(float(goal[2] - start[2]))
    raw_draws = np.asarray(
        [*witness.q0_jitter, witness.phi, float(witness.offset_sign)],
        dtype=np.float64,
    )
    expected_draws = _expected_task_draws(int(task_seed))
    q0_from_draws = np.asarray(
        TIAGO_RIGHT_START_CONFIGS["comfortable_high_clearance"], dtype=np.float64
    ) + expected_draws[:7]
    requested_target = start + toll.REQUESTED_TRAVEL_M * np.asarray(
        [np.cos(expected_draws[7]), np.sin(expected_draws[7]), 0.0],
        dtype=np.float64,
    )
    direction = (goal[:2] - start[:2]) / planar_travel
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    offset_sign = int(expected_draws[8])
    cylinder = (
        0.5 * (start[:2] + goal[:2])
        + offset_sign * toll.CYLINDER_OFFSET_M * normal
    )
    default_side = -offset_sign
    toll_xy = (
        cylinder
        + default_side * toll.TOLL_OFFSET_FROM_CYLINDER_M * normal
    )
    reference_expected = np.asarray(
        [
            *goal,
            *cylinder,
            toll.PHYSICAL_RADIUS_M,
            *toll_xy,
            toll.TOLL_SIGMA_M,
            toll.CLEARANCE_MARGIN_M,
        ],
        dtype=np.float32,
    )
    endpoint_clearance = min(
        toll.physical_clearance(start[:2], reference_expected),
        toll.physical_clearance(goal[:2], reference_expected),
    )
    cylinder_delta = cylinder - start[:2]
    chord_distance = float(
        abs(direction[0] * cylinder_delta[1] - direction[1] * cylinder_delta[0])
    )
    draw_gate = bool(
        raw_draws.shape == expected_draws.shape == (9,)
        and raw_draws.dtype == expected_draws.dtype == np.float64
        and np.array_equal(raw_draws, expected_draws)
        and np.array_equal(history.q_float64[0], q0_from_draws)
        and np.array_equal(
            np.asarray(witness.requested_target_xyz, dtype=np.float64),
            requested_target,
        )
    )
    public_bytes_gate = bool(
        x0.shape == (14,)
        and x0.dtype == np.float32
        and reference.shape == (10,)
        and reference.dtype == np.float32
        and np.array_equal(x0, x0_expected)
        and np.array_equal(reference[:3], goal_expected)
        and task.solver_x0_sha256 == toll._sha256_array(x0)
        and task.solver_reference_sha256 == toll._sha256_array(reference)
        and task.task_seed == int(task_seed)
        and task.default_side == default_side
        and np.array_equal(reference, reference_expected)
    )
    geometry_gates = {
        "actual_travel_in_frozen_range": bool(
            toll.ACTUAL_TRAVEL_RANGE_M[0]
            <= actual_travel
            <= toll.ACTUAL_TRAVEL_RANGE_M[1]
            and witness.actual_travel_m == actual_travel
        ),
        "vertical_travel_in_frozen_range": bool(
            vertical <= toll.MAX_VERTICAL_TRAVEL_M
        ),
        "endpoint_physical_clearance_pass": bool(
            endpoint_clearance >= toll.MIN_ENDPOINT_PHYSICAL_CLEARANCE_M
            and witness.endpoint_physical_clearance_m == endpoint_clearance
        ),
        "chord_intersects_physical_cylinder": bool(
            chord_distance < toll.PHYSICAL_RADIUS_M
            and witness.chord_intersects_physical_cylinder is True
        ),
        "chord_intersects_optimizer_keepout": bool(
            chord_distance < toll.PHYSICAL_RADIUS_M + toll.CLEARANCE_MARGIN_M
            and witness.chord_intersects_optimizer_keepout is True
        ),
        "planar_travel_matches_witness": bool(
            witness.planar_travel_m == planar_travel
        ),
        "chord_distance_matches_witness": bool(
            witness.chord_min_distance_to_cylinder_m == chord_distance
        ),
    }
    row = {
        "identity": list(identity),
        "phase": phase,
        "task_seed": int(task_seed),
        "attempted": True,
        "passes": False,
        "raw_draws_sha256": _array_hash(raw_draws),
        "expected_raw_draws_sha256": _array_hash(expected_draws),
        "raw_draws_exact_for_task_seed": draw_gate,
        "raw_draws": raw_draws.tolist(),
        "history_hashes": dict(witness.history_hashes),
        "regenerated_history_hashes": regenerated.hashes(),
        "history_regeneration": history_gate,
        "q_history_gate": q_gate,
        "public_task_bytes_gate": public_bytes_gate,
        "solver_x0_sha256": task.solver_x0_sha256,
        "solver_reference_sha256": task.solver_reference_sha256,
        "public_task": {
            "task_seed": int(task.task_seed),
            "default_side": int(task.default_side),
            "x0_float32": x0.tolist(),
            "reference_float32": reference.tolist(),
            "solver_x0_sha256": task.solver_x0_sha256,
            "solver_reference_sha256": task.solver_reference_sha256,
        },
        "geometry_gates": geometry_gates,
        "final_residual_norm_m_report_only": float(
            np.linalg.norm(history.residual_float64[-1])
        ),
        "actual_travel_m": witness.actual_travel_m,
        "vertical_travel_m": vertical,
        "endpoint_physical_clearance_m": witness.endpoint_physical_clearance_m,
        "chord_min_distance_to_cylinder_m": witness.chord_min_distance_to_cylinder_m,
        "oracle_only": witness.oracle_only,
        "benchmark_seed_eligible": witness.benchmark_seed_eligible,
        "retry_count": 0,
        "replacement_count": 0,
    }
    row["passes"] = bool(
        history_gate["all_witness_regeneration_gates_pass"]
        and q_gate["all_q_history_gates_pass"]
        and draw_gate
        and public_bytes_gate
        and all(geometry_gates.values())
        and witness.oracle_only is True
        and witness.benchmark_seed_eligible is False
    )
    prefix = f"task_{int(task_seed)}"
    arrays = {
        f"{prefix}_x0_float32": x0,
        f"{prefix}_reference_float32": reference,
        f"{prefix}_raw_draws_float64": raw_draws,
        f"{prefix}_expected_raw_draws_float64": expected_draws,
    }
    for name, value in history.arrays().items():
        arrays[f"{prefix}_history_{name}"] = value
    for name, value in regenerated.arrays().items():
        arrays[f"{prefix}_regenerated_{name}"] = value
    return row, arrays


def _expected_final_array_names() -> set[str]:
    names = {
        "model_lower_float64",
        "model_upper_float64",
        "comfortable_q_float64",
    }
    for _, task_seed in EXPECTED_TASK_IDENTITIES:
        prefix = f"task_{task_seed}"
        names.update(
            {
                f"{prefix}_x0_float32",
                f"{prefix}_reference_float32",
                f"{prefix}_raw_draws_float64",
                f"{prefix}_expected_raw_draws_float64",
            }
        )
        for source in ("history", "regenerated"):
            for field in (
                "q_float64",
                "tool_position_float64",
                "residual_float64",
                "jacobian_float64",
                "dq_float64",
            ):
                names.add(f"{prefix}_{source}_{field}")
    return names


def _row_arrays_bound(row: Mapping, arrays: Mapping[str, np.ndarray]) -> bool:
    task_seed = row.get("task_seed")
    if not isinstance(task_seed, int):
        return False
    prefix = f"task_{task_seed}"
    public = row.get("public_task")
    try:
        raw_draws = arrays[f"{prefix}_raw_draws_float64"]
        expected_raw_draws = arrays[f"{prefix}_expected_raw_draws_float64"]
        x0 = arrays[f"{prefix}_x0_float32"]
        reference = arrays[f"{prefix}_reference_float32"]
        history_arrays = {
            field: arrays[f"{prefix}_history_{field}"]
            for field in (
                "q_float64",
                "tool_position_float64",
                "residual_float64",
                "jacobian_float64",
                "dq_float64",
            )
        }
        regenerated_arrays = {
            field: arrays[f"{prefix}_regenerated_{field}"]
            for field in history_arrays
        }
        public_x0 = np.asarray(public["x0_float32"], dtype=np.float32)
        public_reference = np.asarray(public["reference_float32"], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        isinstance(public, Mapping)
        and raw_draws.shape == (9,)
        and raw_draws.dtype == np.float64
        and public.get("task_seed") == task_seed
        and public.get("default_side") == -int(raw_draws[8])
        and public_x0.shape == x0.shape == (14,)
        and public_reference.shape == reference.shape == (10,)
        and np.array_equal(public_x0, x0)
        and np.array_equal(public_reference, reference)
        and row.get("raw_draws_sha256") == _array_hash(raw_draws)
        and row.get("expected_raw_draws_sha256")
        == _array_hash(expected_raw_draws)
        and public.get("solver_x0_sha256") == toll._sha256_array(x0)
        and public.get("solver_reference_sha256")
        == toll._sha256_array(reference)
        and row.get("solver_x0_sha256") == public.get("solver_x0_sha256")
        and row.get("solver_reference_sha256")
        == public.get("solver_reference_sha256")
        and all(
            row.get("history_hashes", {}).get(field)
            == _array_hash(history_arrays[field])
            and row.get("regenerated_history_hashes", {}).get(field)
            == _array_hash(regenerated_arrays[field])
            for field in (
                "q_float64",
                "tool_position_float64",
                "residual_float64",
                "jacobian_float64",
                "dq_float64",
            )
        )
    )


def certify_final(
    rows: Sequence[Mapping],
    provenance: Mapping,
    arrays: Mapping[str, np.ndarray],
    *,
    test_override: bool,
) -> dict:
    identities = [tuple(row.get("identity", ())) for row in rows]
    source_rows = provenance.get("source_hashes", {})
    source_gate = bool(
        isinstance(source_rows, Mapping)
        and set(source_rows) == set(REQUIRED_SOURCE_PATHS)
        and all(
            source_rows[label].get("path") == path
            and _is_hex_digest(source_rows[label].get("sha256"), 64)
            for label, path in REQUIRED_SOURCE_PATHS.items()
        )
    )
    provenance_gate = bool(
        provenance.get("tracked_tree_clean_at_start") is True
        and provenance.get("tracked_tree_clean_at_end") is True
        and _is_hex_digest(provenance.get("git_head_at_start"), 40)
        and provenance.get("git_head_at_start") == provenance.get("git_head_at_end")
        and _is_hex_digest(provenance.get("model_sha256"), 64)
        and isinstance(provenance.get("exact_command"), str)
        and bool(provenance.get("exact_command"))
        and all(
            isinstance(provenance.get(name), str) and bool(provenance.get(name))
            for name in ("python_version", "numpy_version", "pinocchio_version")
        )
        and provenance.get("predecessor_artifacts_consumed") is False
        and provenance.get("rejected_predecessor_artifact_hashes_report_only")
        == REJECTED_PREDECESSOR_ARTIFACT_HASHES
        and provenance.get("extension_hashes_report_only")
        == REPORT_ONLY_EXTENSION_HASHES
        and source_gate
    )
    exact_array_schema = bool(
        set(arrays) == _expected_final_array_names()
        and all(
            isinstance(value, np.ndarray) and np.all(np.isfinite(value))
            for value in arrays.values()
        )
        and arrays["model_lower_float64"].shape == (7,)
        and arrays["model_lower_float64"].dtype == np.float64
        and arrays["model_upper_float64"].shape == (7,)
        and arrays["model_upper_float64"].dtype == np.float64
        and arrays["comfortable_q_float64"].shape == (7,)
        and arrays["comfortable_q_float64"].dtype == np.float64
    )
    arrays_bound_to_rows = exact_array_schema
    if arrays_bound_to_rows:
        for row in rows:
            arrays_bound_to_rows = _row_arrays_bound(row, arrays)
            if not arrays_bound_to_rows:
                break
    gates = {
        "exact_ordered_twelve_identities": identities == list(EXPECTED_TASK_IDENTITIES),
        "exact_twelve_attempts": len(rows) == EXPECTED_TASK_COUNT,
        "all_task_rows_pass": len(rows) == EXPECTED_TASK_COUNT
        and all(row.get("passes") is True for row in rows),
        "provenance_pass": provenance_gate,
        "exact_retained_array_schema": exact_array_schema,
        "all_retained_arrays_bound_to_rows": arrays_bound_to_rows,
        "zero_retry_replacement": all(
            row.get("retry_count") == row.get("replacement_count") == 0 for row in rows
        ),
        "zero_worker_cuda_sqp_route_calls": True,
        "production_not_test_override": test_override is False,
    }
    gates["all_v2_gates_pass"] = all(gates.values())
    return gates


def _finalize(
    output: Path,
    *,
    expected: Sequence[tuple],
    rows: Sequence[Mapping],
    arrays: Mapping[str, np.ndarray],
    provenance: Mapping,
    test_override: bool,
    latest_generation: int,
    certificate: Mapping | None = None,
) -> dict:
    output = Path(output)
    certificate = dict(certificate) if certificate is not None else certify_final(
        rows, provenance, arrays, test_override=test_override
    )
    summary = {
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "v2_protocol_version": V2_PROTOCOL_VERSION,
        "incomplete": False,
        "expected_identities": [list(row) for row in expected],
        "attempted_identities": [row["identity"] for row in rows],
        "completed_identities": [
            row["identity"] for row in rows if row.get("passes") is True
        ],
        "pending_identities": [],
        "task_construction_attempt_count": len(rows),
        "model_load_calls": 1,
        "task_rng_generation_calls": len(rows),
        "task_rng_authentication_calls": sum(
            "raw_draws_exact_for_task_seed" in row for row in rows
        ),
        "history_regeneration_calls": sum(
            "history_regeneration" in row for row in rows
        ),
        "rows": _json_value(list(rows)),
        "certificate": certificate,
        "all_v2_gates_pass": certificate["all_v2_gates_pass"],
        "construction_only": True,
        "benchmark_evidence": False,
        "optimization_evidence": False,
        "worker_calls": 0,
        "cuda_calls": 0,
        "sqp_calls": 0,
        "route_oracle_calls": 0,
        "retry_count": 0,
        "replacement_count": 0,
        "provenance": _json_value(dict(provenance)),
        "supersedes_partial_generation": int(latest_generation),
    }
    payload_arrays = {
        str(name): np.ascontiguousarray(value) for name, value in arrays.items()
    }
    npz_path = output.with_suffix(".npz")
    manifest_path = output.with_suffix(".manifest.json")
    _atomic_npz(npz_path, payload_arrays)
    summary["npz_path"] = str(npz_path)
    summary["npz_sha256"] = _sha256_file(npz_path)
    summary["array_names"] = sorted(payload_arrays)
    summary["array_hashes"] = {
        name: _array_hash(value) for name, value in payload_arrays.items()
    }
    _atomic_json(output, summary)
    manifest = {
        "incomplete": False,
        "json_path": str(output),
        "json_sha256": _sha256_file(output),
        "npz_path": str(npz_path),
        "npz_sha256": summary["npz_sha256"],
        "supersedes_partial_generation": int(latest_generation),
        "all_v2_gates_pass": summary["all_v2_gates_pass"],
    }
    _atomic_json(manifest_path, manifest)
    return summary


def _run_pipeline(
    output: Path,
    *,
    ledger: Sequence[tuple],
    provenance: Mapping,
    model_factory: Callable[[], tuple[object, Mapping[str, np.ndarray]]],
    task_factory: Callable[[tuple, object], tuple[Mapping, Mapping[str, np.ndarray]]],
    test_override: bool,
    after_attempt: Callable[[int, Mapping], None] | None = None,
) -> dict:
    output = Path(output)
    expected = tuple(tuple(row) for row in ledger)
    _no_existing_artifacts(output)
    rows = []
    arrays: dict[str, np.ndarray] = {}
    attempted: list[tuple] = []
    completed: list[tuple] = []
    _checkpoint(
        output,
        generation=0,
        stage="before_model_or_rng",
        expected=expected,
        attempted=attempted,
        completed=completed,
        rows=rows,
        arrays=arrays,
        provenance=provenance,
    )
    try:
        model, model_arrays = model_factory()
        arrays.update(model_arrays)
    except BaseException as error:
        rows.append(
            {
                "identity": ["model", "tiago_right"],
                "attempted": True,
                "passes": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _checkpoint(
            output,
            generation=1,
            stage="model_failed",
            expected=expected,
            attempted=attempted,
            completed=completed,
            rows=rows,
            arrays=arrays,
            provenance=provenance,
        )
        raise
    _checkpoint(
        output,
        generation=1,
        stage="model_and_config_captured",
        expected=expected,
        attempted=attempted,
        completed=completed,
        rows=rows,
        arrays=arrays,
        provenance=provenance,
    )
    for index, identity in enumerate(expected, start=1):
        attempted.append(identity)
        fatal_error = None
        try:
            row, task_arrays = task_factory(identity, model)
            row = dict(row)
            if tuple(row.get("identity", ())) != identity:
                raise RuntimeError("task row identity mismatch")
            arrays.update(task_arrays)
            if row.get("passes") is True:
                completed.append(identity)
        except BaseException as error:
            row = {
                "identity": list(identity),
                "attempted": True,
                "passes": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "retry_count": 0,
                "replacement_count": 0,
            }
            if not isinstance(error, Exception):
                fatal_error = error
        rows.append(row)
        _checkpoint(
            output,
            generation=index + 1,
            stage="task_attempted",
            expected=expected,
            attempted=attempted,
            completed=completed,
            rows=rows,
            arrays=arrays,
            provenance=provenance,
        )
        if fatal_error is not None:
            raise fatal_error
        if after_attempt is not None:
            after_attempt(index, row)
    if not test_override and not all(row.get("passes") is True for row in rows):
        raise RuntimeError(
            "Stage V2 task construction failed; generation 13 is retained and V2 closes"
        )
    certificate = certify_final(
        rows, provenance, arrays, test_override=test_override
    )
    if not test_override and not certificate["all_v2_gates_pass"]:
        raise RuntimeError(
            "Stage V2 aggregate certificate failed; generation 13 is retained and no final is published"
        )
    return _finalize(
        output,
        expected=expected,
        rows=rows,
        arrays=arrays,
        provenance=provenance,
        test_override=test_override,
        latest_generation=EXPECTED_TASK_COUNT + 1,
        certificate=certificate,
    )


def _production_pipeline(output: Path, *, token=None):
    if token is not _PRODUCTION_PIPELINE_TOKEN:
        raise RuntimeError("Stage V2 production pipeline is private")
    repo = repository_root()
    provenance = _source_provenance(repo)
    if not provenance["tracked_tree_clean_at_start"]:
        raise RuntimeError("Stage V2 requires a tracked-clean tree before generation zero")

    def model_factory():
        from gato_tiago.multimodal_pillar import load_model

        model = load_model(repo / REQUIRED_SOURCE_PATHS["model"])
        lower = np.asarray(model.lowerPositionLimit, dtype=np.float64)
        upper = np.asarray(model.upperPositionLimit, dtype=np.float64)
        comfortable = np.asarray(
            TIAGO_RIGHT_START_CONFIGS["comfortable_high_clearance"], dtype=np.float64
        )
        return model, {
            "model_lower_float64": lower,
            "model_upper_float64": upper,
            "comfortable_q_float64": comfortable,
        }

    def task_factory(identity, model):
        phase, task_seed = identity
        task, witness = oracle.generate_task(
            task_seed, model, authorization=oracle.TASK_CONSTRUCTION_AUTHORIZATION
        )
        regenerated = oracle.regenerate_witness_history(model, witness)
        row, arrays = _task_row(identity, task, witness, regenerated, model)
        row["phase"] = phase
        return row, arrays

    def capture_end():
        provenance["git_head_at_end"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        tracked = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            text=True,
        ).splitlines()
        provenance["tracked_tree_clean_at_end"] = not tracked
        provenance["full_git_status_at_end"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).splitlines()

    # The end provenance must be captured before final certification. A wrapper
    # around the twelfth checkpoint performs no task/model/RNG work.
    def after_attempt(index, row):
        del row
        if index == EXPECTED_TASK_COUNT:
            capture_end()

    return _run_pipeline(
        output,
        ledger=EXPECTED_TASK_IDENTITIES,
        provenance=provenance,
        model_factory=model_factory,
        task_factory=task_factory,
        test_override=False,
        after_attempt=after_attempt,
    )


def describe_v2_runner() -> dict:
    return {
        "protocol_version": RUNNER_PROTOCOL_VERSION,
        "authorization": RUNNER_EXECUTION_AUTHORIZATION,
        "authorized_output_path": str(AUTHORIZED_OUTPUT_PATH),
        "expected_identities": [list(row) for row in EXPECTED_TASK_IDENTITIES],
        "schema": frozen_v2_metadata(),
        "required_source_paths": dict(REQUIRED_SOURCE_PATHS),
        "report_only_extension_hashes": dict(REPORT_ONLY_EXTENSION_HASHES),
        "expected_model_load_calls": EXPECTED_MODEL_LOAD_CALLS,
        "expected_task_construction_calls": EXPECTED_TASK_CONSTRUCTION_CALLS,
        "expected_history_regeneration_calls": EXPECTED_HISTORY_REGENERATION_CALLS,
        "worker_calls": 0,
        "cuda_calls": 0,
        "sqp_calls": 0,
        "route_oracle_calls": 0,
        "construction_only": True,
        "benchmark_evidence": False,
    }


def execute_v2_runner(output, *, authorization=None):
    if (
        RUNNER_EXECUTION_AUTHORIZATION is None
        or authorization is not RUNNER_EXECUTION_AUTHORIZATION
    ):
        raise RuntimeError("Stage V2 runner execution is blocked")
    if Path(output).resolve() != AUTHORIZED_OUTPUT_PATH:
        raise RuntimeError("Stage V2 runner output is not the single authorized path")
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
        print(json.dumps(describe_v2_runner(), indent=2, sort_keys=True))
        return 0
    if not args.execute or RUNNER_EXECUTION_AUTHORIZATION is None:
        raise SystemExit("Stage V2 task/model execution is blocked")
    execute_v2_runner(args.output, authorization=RUNNER_EXECUTION_AUTHORIZATION)
    return 0


if __name__ == "__main__":
    main()
