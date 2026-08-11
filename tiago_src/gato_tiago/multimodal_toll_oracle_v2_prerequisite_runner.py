"""Transactional CPU-only prerequisite authenticator for Oracle V2.

Execution is disabled in this static checkpoint.  The production boundary
loads the accepted predecessor artifacts, but never constructs a model/task or
opens any predecessor/V1 output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np

from gato_tiago.multimodal_toll_oracle_v2 import (
    AUTHORIZED_CWD,
    AUTHORIZED_ORIG_ARGV,
    AUTHORIZED_OUTPUT,
    EXPECTED_TASK_IDENTITIES,
    FORBIDDEN_CALL_COUNTS,
    MODEL_ARRAY_COUNT,
    MODEL_PINS,
    PRODUCER_HASH_CONVENTION,
    PROTOCOL,
    TASK_ARRAY_COUNT,
    TASK_PINS,
    V1_REJECTED_REPORT_ONLY,
    producer_array_hash,
)


RUNNER_EXECUTION_AUTHORIZATION = object()
REQUIRED_SOURCE_PATHS = {
    "v2_schema": "tiago_src/gato_tiago/multimodal_toll_oracle_v2.py",
    "v2_prerequisite_runner": (
        "tiago_src/gato_tiago/multimodal_toll_oracle_v2_prerequisite_runner.py"
    ),
    "v2_tests": (
        "tests/python/test_tiago_multimodal_toll_oracle_v2_prerequisite.py"
    ),
    "task_producer": "tiago_src/gato_tiago/multimodal_toll_v4_runner.py",
    "model_producer_schema": (
        "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight_v4.py"
    ),
    "model_producer_runner": (
        "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight_v4_runner.py"
    ),
    "closed_v1_runner": (
        "tiago_src/gato_tiago/multimodal_toll_oracle_v1_runner.py"
    ),
    "closed_v1_worker": (
        "tiago_src/gato_tiago/multimodal_toll_oracle_v1_worker.py"
    ),
}
_PRODUCTION_TOKEN = object()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_root(module_path=Path(__file__)) -> Path:
    root = Path(module_path).resolve().parents[2]
    if not (root / ".git").exists() or not (root / "CMakeLists.txt").is_file():
        raise RuntimeError("Oracle V2 prerequisite repository root mismatch")
    return root


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def source_hashes(root: Path) -> dict:
    return {
        label: {"path": path, "sha256": sha256_file(root / path)}
        for label, path in REQUIRED_SOURCE_PATHS.items()
    }


def start_provenance(root=None, *, orig_argv=None) -> dict:
    root = repository_root() if root is None else Path(root).resolve()
    argv = tuple(sys.orig_argv if orig_argv is None else orig_argv)
    tracked = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "git_head_at_start": _git(root, "rev-parse", "HEAD"),
        "git_head_at_end": None,
        "tracked_tree_clean_at_start": tracked == "",
        "tracked_tree_clean_at_end": None,
        "full_git_status_at_start": _git(root, "status", "--porcelain=v1"),
        "full_git_status_at_end": None,
        "source_hashes_at_start": source_hashes(root),
        "source_hashes_at_end": None,
        "cwd": str(Path.cwd().resolve()),
        "orig_argv": list(argv),
        "exact_command": shlex.join(argv),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "task_artifact_loads": 0,
        "task_pure_recert_calls": 0,
        "task_array_hashes_checked": 0,
        "model_artifact_loads": 0,
        "model_pure_recert_calls": 0,
        "model_array_hashes_checked": 0,
        "cross_bind_calls": 0,
        "rejected_v1_artifact_loads": 0,
        **{name: 0 for name in FORBIDDEN_CALL_COUNTS},
    }


def finish_provenance(provenance: dict, root=None) -> None:
    root = repository_root() if root is None else Path(root).resolve()
    provenance["git_head_at_end"] = _git(root, "rev-parse", "HEAD")
    tracked = _git(root, "status", "--porcelain", "--untracked-files=no")
    provenance["tracked_tree_clean_at_end"] = tracked == ""
    provenance["full_git_status_at_end"] = _git(
        root, "status", "--porcelain=v1"
    )
    provenance["source_hashes_at_end"] = source_hashes(root)


def certify_provenance(provenance: Mapping) -> bool:
    start = provenance.get("git_head_at_start")
    sources = provenance.get("source_hashes_at_start")
    return bool(
        isinstance(start, str)
        and len(start) == 40
        and all(char in "0123456789abcdef" for char in start)
        and start == provenance.get("git_head_at_end")
        and provenance.get("tracked_tree_clean_at_start") is True
        and provenance.get("tracked_tree_clean_at_end") is True
        and provenance.get("full_git_status_at_start")
        == provenance.get("full_git_status_at_end")
        and provenance.get("cwd") == str(AUTHORIZED_CWD)
        and tuple(provenance.get("orig_argv", ())) == AUTHORIZED_ORIG_ARGV
        and provenance.get("exact_command") == shlex.join(AUTHORIZED_ORIG_ARGV)
        and isinstance(provenance.get("python_version"), str)
        and bool(provenance.get("python_version"))
        and isinstance(provenance.get("numpy_version"), str)
        and bool(provenance.get("numpy_version"))
        and isinstance(sources, Mapping)
        and set(sources) == set(REQUIRED_SOURCE_PATHS)
        and sources == provenance.get("source_hashes_at_end")
        and all(
            row.get("path") == REQUIRED_SOURCE_PATHS[label]
            and isinstance(row.get("sha256"), str)
            and len(row["sha256"]) == 64
            for label, row in sources.items()
        )
        and provenance.get("task_artifact_loads") == 1
        and provenance.get("task_pure_recert_calls") == 1
        and provenance.get("task_array_hashes_checked") == TASK_ARRAY_COUNT
        and provenance.get("model_artifact_loads") == 1
        and provenance.get("model_pure_recert_calls") == 1
        and provenance.get("model_array_hashes_checked") == MODEL_ARRAY_COUNT
        and provenance.get("cross_bind_calls") == 1
        and provenance.get("rejected_v1_artifact_loads") == 0
        and all(provenance.get(name) == 0 for name in FORBIDDEN_CALL_COUNTS)
    )


def authenticate_producer(
    summary: Mapping,
    arrays: Mapping[str, np.ndarray],
    *,
    expected_count: int,
    certificate_key: str,
) -> dict:
    names = summary.get("array_names")
    hashes = summary.get("array_hashes")
    exact_names = bool(
        isinstance(names, list)
        and names == sorted(arrays)
        and len(names) == len(set(names)) == expected_count
        and set(arrays) == set(names)
    )
    exact_hash_map = bool(
        isinstance(hashes, Mapping)
        and list(sorted(hashes)) == sorted(arrays)
        and set(hashes) == set(arrays)
    )
    listed_names = names if isinstance(names, list) else []
    hash_keys = set(hashes) if isinstance(hashes, Mapping) else set()
    array_keys = set(arrays)
    computed = {
        name: producer_array_hash(value) for name, value in arrays.items()
    }
    mismatches = sorted(
        name
        for name in array_keys & hash_keys
        if computed[name] != hashes[name]
    )
    certificate = summary.get("certificate")
    stored_gate = bool(
        summary.get("incomplete") is False
        and summary.get(certificate_key) is True
        and isinstance(certificate, Mapping)
        and certificate.get(certificate_key) is True
    )
    return {
        "expected_count": expected_count,
        "actual_count": len(arrays),
        "ordered_array_names": list(names) if isinstance(names, list) else None,
        "computed_array_hashes": computed,
        "mismatch_names": mismatches,
        "mismatch_count": len(mismatches),
        "duplicate_names": sorted(
            {name for name in listed_names if listed_names.count(name) > 1}
        ),
        "names_missing_from_arrays": sorted(set(listed_names) - array_keys),
        "names_extra_in_arrays": sorted(array_keys - set(listed_names)),
        "hash_keys_missing_from_arrays": sorted(hash_keys - array_keys),
        "array_names_missing_hash_keys": sorted(array_keys - hash_keys),
        "exact_ordered_unique_names": exact_names,
        "exact_hash_key_map": exact_hash_map,
        "stored_certificate_gate": stored_gate,
        "producer_native_hashes_pass": bool(
            exact_names and exact_hash_map and not mismatches
        ),
        "passes": bool(
            exact_names and exact_hash_map and not mismatches and stored_gate
        ),
    }


def cross_bind_prerequisites(
    task_rows: Sequence[Mapping],
    task_arrays: Mapping[str, np.ndarray],
    model_arrays: Mapping[str, np.ndarray],
    *,
    expected_identities: Sequence[tuple] = EXPECTED_TASK_IDENTITIES,
) -> dict:
    identities = [tuple(row.get("identity", ())) for row in task_rows]
    try:
        x0 = np.stack(
            [task_arrays[f"task_{seed}_x0_float32"] for _, seed in expected_identities]
        )
        reference = np.stack(
            [
                task_arrays[f"task_{seed}_reference_float32"]
                for _, seed in expected_identities
            ]
        )
        default_side = np.asarray(
            [row["public_task"]["default_side"] for row in task_rows],
            dtype=np.int8,
        )
        q8 = np.stack(
            [
                task_arrays[f"task_{seed}_history_q_float64"][8]
                for _, seed in expected_identities
            ]
        )
        task_lower = np.asarray(task_arrays["model_lower_float64"])
        task_upper = np.asarray(task_arrays["model_upper_float64"])
        velocity = np.asarray(model_arrays["model_velocity_float64"])
        effort = np.asarray(model_arrays["model_effort_float64"])
        scalar_gate = np.asarray(
            model_arrays["v4_artifact_authentication_gate_bool"]
        )
        gates = {
            "identities": identities == list(expected_identities),
            "public_x0": np.array_equal(
                model_arrays["public_solver_x0_float32"], x0
            ),
            "public_reference": np.array_equal(
                model_arrays["public_reference_float32"], reference
            ),
            "public_default_side": np.array_equal(
                model_arrays["public_default_side_int8"], default_side
            ),
            "quarantined_q8": np.array_equal(
                model_arrays["quarantined_q8_float64"], q8
            ),
            "model_lower": np.array_equal(
                model_arrays["model_lower_float64"], task_lower
            ),
            "model_upper": np.array_equal(
                model_arrays["model_upper_float64"], task_upper
            ),
            "model_velocity_limit_domain": bool(
                velocity.shape == (7,)
                and velocity.dtype == np.dtype(np.float64)
                and np.all(np.isfinite(velocity))
                and np.all(velocity > 0.0)
            ),
            "model_effort_limit_domain": bool(
                effort.shape == (7,)
                and effort.dtype == np.dtype(np.float64)
                and np.all(np.isfinite(effort))
                and np.all(effort > 0.0)
            ),
            "q8_oracle_only": bool(
                np.all(model_arrays["quarantined_q8_oracle_only_bool"])
            ),
            "q8_not_initializer": bool(
                not np.any(
                    model_arrays["quarantined_q8_initializer_eligible_bool"]
                )
            ),
            "q8_not_solver_seed": bool(
                not np.any(
                    model_arrays["quarantined_q8_solver_seed_eligible_bool"]
                )
            ),
            "model_disk_recert": bool(
                scalar_gate.shape == ()
                and scalar_gate.dtype == np.dtype(np.bool_)
                and scalar_gate.item()
            ),
        }
    except (KeyError, TypeError, ValueError, IndexError):
        gates = {"malformed_input": False}
    return {"gates": gates, "passes": bool(gates and all(gates.values()))}


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


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool = False) -> None:
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Mapping, *, replace: bool = False) -> None:
    _atomic_bytes(
        path,
        (json.dumps(_json_value(value), sort_keys=True, indent=2) + "\n").encode(),
        replace=replace,
    )


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz"
    )
    os.close(fd)
    try:
        np.savez(temporary, **arrays)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checkpoint(
    output: Path,
    generation: int,
    stage: str,
    provenance: Mapping,
    *,
    detail: Mapping | None = None,
) -> dict:
    stem = output.with_suffix("")
    json_path = stem.with_name(f"{stem.name}.partial.{generation:04d}.json")
    npz_path = stem.with_name(f"{stem.name}.partial.{generation:04d}.npz")
    arrays = {
        "generation_int64": np.asarray(generation, dtype=np.int64),
        "prerequisite_auth_only_bool": np.asarray(True, dtype=np.bool_),
        "oracle_evidence_bool": np.asarray(False, dtype=np.bool_),
        "benchmark_evidence_bool": np.asarray(False, dtype=np.bool_),
    }
    _atomic_npz(npz_path, arrays)
    payload = {
        "protocol": PROTOCOL,
        "generation": generation,
        "stage": stage,
        "incomplete": True,
        "prerequisite_auth_only": True,
        "oracle_evidence": False,
        "benchmark_evidence": False,
        "all_oracle_gates_pass": False,
        "provenance": dict(provenance),
        "detail": {} if detail is None else dict(detail),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "array_names": sorted(arrays),
        "array_hashes": {
            name: producer_array_hash(value) for name, value in arrays.items()
        },
    }
    _atomic_json(json_path, payload)
    pointer = output.with_name(f"{output.stem}.partial.latest.json")
    _atomic_json(
        pointer,
        {
            "protocol": PROTOCOL,
            "generation": generation,
            "stage": stage,
            "incomplete": True,
            "json_path": str(json_path),
            "json_sha256": sha256_file(json_path),
            "npz_path": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
        },
        replace=pointer.exists(),
    )
    return payload


def _load_pinned_artifact(pins: Mapping) -> tuple[dict, dict, dict, dict]:
    for path, digest in pins.values():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError("hard-pinned prerequisite artifact mismatch")
    summary = json.loads(pins["final_json"][0].read_text())
    manifest = json.loads(pins["final_manifest"][0].read_text())
    pointer = json.loads(pins["latest_pointer"][0].read_text())
    with np.load(pins["final_npz"][0], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return summary, arrays, manifest, pointer


def _task_pure_recert(summary, arrays) -> bool:  # pragma: no cover
    from gato_tiago.multimodal_toll_v4_runner import certify_final

    recomputed = certify_final(
        summary.get("rows", ()),
        summary.get("provenance", {}),
        arrays,
        test_override=False,
    )
    return bool(
        recomputed == summary.get("certificate")
        and recomputed.get("all_v4_gates_pass") is True
    )


def _model_pure_recert(summary, arrays) -> bool:  # pragma: no cover
    from gato_tiago.multimodal_toll_v4_model_preflight_v4 import (
        EXPECTED_NONWORKER_ARRAY_NAMES,
        FROZEN_EXTENSIONS,
    )
    from gato_tiago.multimodal_toll_v4_model_preflight_v4_runner import (
        certify_model_preflight,
    )
    from gato_tiago.multimodal_toll_v4_model_preflight_v4_worker import (
        EXPECTED_WORKER_ARRAY_NAMES,
    )

    base = {name: arrays[name] for name in EXPECTED_NONWORKER_ARRAY_NAMES}
    worker_arrays = {}
    for spec in FROZEN_EXTENSIONS:
        module = spec["module_name"]
        prefix = f"worker_{module.split('.')[-1]}_"
        worker_arrays[module] = {
            name: arrays[f"{prefix}{name}"] for name in EXPECTED_WORKER_ARRAY_NAMES
        }
    recomputed = certify_model_preflight(base, summary.get("rows", ()), worker_arrays)
    stored = dict(summary.get("certificate", {}))
    provenance_pass = stored.pop("provenance_pass", None)
    return bool(
        stored == recomputed
        and provenance_pass is True
        and recomputed.get("all_model_preflight_gates_pass") is True
    )


def _side_artifacts(output: Path) -> list[dict]:
    rows = []
    for generation in range(4):
        for kind, suffix in (("json", "json"), ("npz", "npz")):
            path = output.with_name(
                f"{output.stem}.partial.{generation:04d}.{suffix}"
            )
            rows.append(
                {
                    "generation": generation,
                    "kind": kind,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    return rows


def _expected_checkpoint_provenance(final_provenance: Mapping, generation: int) -> dict:
    expected = dict(final_provenance)
    if generation < 3:
        for name in (
            "git_head_at_end",
            "tracked_tree_clean_at_end",
            "full_git_status_at_end",
            "source_hashes_at_end",
        ):
            expected[name] = None
    if generation == 0:
        for name in (
            "task_artifact_loads",
            "task_pure_recert_calls",
            "task_array_hashes_checked",
            "model_artifact_loads",
            "model_pure_recert_calls",
            "model_array_hashes_checked",
            "cross_bind_calls",
        ):
            expected[name] = 0
    elif generation == 1:
        for name in (
            "model_artifact_loads",
            "model_pure_recert_calls",
            "model_array_hashes_checked",
            "cross_bind_calls",
        ):
            expected[name] = 0
    return expected


def _certify_checkpoints(
    output: Path,
    provenance: Mapping,
    task_auth: Mapping,
    model_auth: Mapping,
    cross_bind: Mapping,
) -> dict:
    stages = (
        "before_prerequisite_loads",
        "task_prerequisite_authenticated",
        "model_prerequisite_authenticated",
        "honest_end_provenance",
    )
    details = (
        {},
        dict(task_auth),
        {"model_authentication": dict(model_auth), "cross_bind": dict(cross_bind)},
        {"provenance_pass": True},
    )
    gates = {}
    for generation, (stage, detail) in enumerate(zip(stages, details)):
        json_path = output.with_name(
            f"{output.stem}.partial.{generation:04d}.json"
        )
        npz_path = output.with_name(
            f"{output.stem}.partial.{generation:04d}.npz"
        )
        try:
            payload = json.loads(json_path.read_text())
            with np.load(npz_path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            expected_arrays = {
                "generation_int64": np.asarray(generation, dtype=np.int64),
                "prerequisite_auth_only_bool": np.asarray(True, dtype=np.bool_),
                "oracle_evidence_bool": np.asarray(False, dtype=np.bool_),
                "benchmark_evidence_bool": np.asarray(False, dtype=np.bool_),
            }
            expected_payload = {
                "protocol": PROTOCOL,
                "generation": generation,
                "stage": stage,
                "incomplete": True,
                "prerequisite_auth_only": True,
                "oracle_evidence": False,
                "benchmark_evidence": False,
                "all_oracle_gates_pass": False,
                "provenance": _expected_checkpoint_provenance(
                    provenance, generation
                ),
                "detail": detail,
                "npz_path": str(npz_path),
                "npz_sha256": sha256_file(npz_path),
                "array_names": sorted(expected_arrays),
                "array_hashes": {
                    name: producer_array_hash(value)
                    for name, value in expected_arrays.items()
                },
            }
            gates[f"generation_{generation}_exact"] = bool(
                set(arrays) == set(expected_arrays)
                and all(
                    arrays[name].shape == expected.shape
                    and arrays[name].dtype == expected.dtype
                    and np.array_equal(arrays[name], expected)
                    for name, expected in expected_arrays.items()
                )
                and payload == _json_value(expected_payload)
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            gates[f"generation_{generation}_exact"] = False
    return {"gates": gates, "passes": bool(all(gates.values()))}


def _final_arrays(task_auth, model_auth, cross_bind) -> dict:
    return {
        "task_authentication_bool": np.asarray(task_auth["passes"], np.bool_),
        "model_authentication_bool": np.asarray(model_auth["passes"], np.bool_),
        "cross_bind_bool": np.asarray(cross_bind["passes"], np.bool_),
        "task_array_count_int64": np.asarray(task_auth["actual_count"], np.int64),
        "model_array_count_int64": np.asarray(model_auth["actual_count"], np.int64),
        "prerequisite_auth_only_bool": np.asarray(True, np.bool_),
        "oracle_evidence_bool": np.asarray(False, np.bool_),
        "benchmark_evidence_bool": np.asarray(False, np.bool_),
    }


def _certify_retained_documents(
    summary_path: Path,
    npz_path: Path,
    manifest_path: Path,
    pointer_path: Path,
    *,
    expected_output: Path,
    task_auth: Mapping,
    model_auth: Mapping,
    cross_bind: Mapping,
) -> dict:
    """Recertify staged or final documents against independently derived detail."""

    false_gates = {
        "exact_array_schema_and_hashes": False,
        "exact_scalar_values": False,
        "exact_document_paths_and_hashes": False,
        "exact_non_evidence_semantics_and_provenance": False,
        "exact_checkpoint_chain": False,
    }
    try:
        expected_output = Path(expected_output)
        final_npz = expected_output.with_suffix(".npz")
        final_manifest = expected_output.with_suffix(".manifest.json")
        summary = json.loads(Path(summary_path).read_text())
        manifest = json.loads(Path(manifest_path).read_text())
        pointer = json.loads(Path(pointer_path).read_text())
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        expected_arrays = _final_arrays(task_auth, model_auth, cross_bind)
        expected_names = sorted(expected_arrays)
        names_gate = bool(
            set(arrays) == set(expected_arrays)
            and summary.get("array_names") == expected_names
            and summary.get("array_hashes")
            == {
                name: producer_array_hash(value)
                for name, value in expected_arrays.items()
            }
            and all(
                arrays[name].shape == expected.shape
                and arrays[name].dtype == expected.dtype
                and np.array_equal(arrays[name], expected)
                for name, expected in expected_arrays.items()
            )
        )
        scalar_gate = bool(
            all(value.shape == () for value in arrays.values())
            and bool(arrays["task_authentication_bool"])
            and bool(arrays["model_authentication_bool"])
            and bool(arrays["cross_bind_bool"])
            and bool(arrays["prerequisite_auth_only_bool"])
            and not bool(arrays["oracle_evidence_bool"])
            and not bool(arrays["benchmark_evidence_bool"])
            and int(arrays["task_array_count_int64"].item()) == TASK_ARRAY_COUNT
            and int(arrays["model_array_count_int64"].item()) == MODEL_ARRAY_COUNT
        )
        side_artifacts = _side_artifacts(expected_output)
        checkpoint = _certify_checkpoints(
            expected_output,
            summary.get("provenance", {}),
            task_auth,
            model_auth,
            cross_bind,
        )
        expected_manifest = {
            "protocol": PROTOCOL,
            "incomplete": False,
            "prerequisite_auth_only": True,
            "oracle_evidence": False,
            "benchmark_evidence": False,
            "json_path": str(expected_output),
            "json_sha256": sha256_file(summary_path),
            "npz_path": str(final_npz),
            "npz_sha256": sha256_file(npz_path),
            "side_artifact_count": 8,
            "side_artifacts": side_artifacts,
        }
        expected_pointer = {
            "protocol": PROTOCOL,
            "generation": 3,
            "stage": "honest_end_provenance",
            "incomplete": False,
            "superseded_by": str(expected_output),
            "json_path": str(expected_output),
            "json_sha256": sha256_file(summary_path),
            "npz_path": str(final_npz),
            "npz_sha256": sha256_file(npz_path),
            "manifest_path": str(final_manifest),
            "manifest_sha256": sha256_file(manifest_path),
        }
        paths_gate = bool(
            summary.get("npz_path") == str(final_npz)
            and summary.get("npz_sha256") == sha256_file(npz_path)
            and manifest == expected_manifest
            and pointer == expected_pointer
        )
        semantic_gate = bool(
            set(summary)
            == {
                "protocol",
                "incomplete",
                "prerequisite_auth_only",
                "prerequisites_authenticated",
                "oracle_evidence",
                "benchmark_evidence",
                "all_oracle_gates_pass",
                "task_authentication",
                "model_authentication",
                "cross_bind",
                "producer_hash_convention",
                "task_pins",
                "model_pins",
                "v1_rejected_report_only",
                "rejected_v1_artifact_loads",
                "provenance",
                "npz_path",
                "npz_sha256",
                "array_names",
                "array_hashes",
            }
            and summary.get("protocol") == PROTOCOL
            and summary.get("incomplete") is False
            and summary.get("prerequisite_auth_only") is True
            and summary.get("prerequisites_authenticated") is True
            and summary.get("oracle_evidence") is False
            and summary.get("benchmark_evidence") is False
            and summary.get("all_oracle_gates_pass") is False
            and summary.get("producer_hash_convention")
            == PRODUCER_HASH_CONVENTION
            and summary.get("task_pins")
            == {
                key: {"path": str(path), "sha256": digest}
                for key, (path, digest) in TASK_PINS.items()
            }
            and summary.get("model_pins")
            == {
                key: {"path": str(path), "sha256": digest}
                for key, (path, digest) in MODEL_PINS.items()
            }
            and summary.get("task_authentication") == task_auth
            and summary.get("model_authentication") == model_auth
            and summary.get("cross_bind") == cross_bind
            and summary.get("rejected_v1_artifact_loads") == 0
            and summary.get("v1_rejected_report_only") == V1_REJECTED_REPORT_ONLY
            and certify_provenance(summary.get("provenance", {}))
        )
        gates = {
            "exact_array_schema_and_hashes": names_gate,
            "exact_scalar_values": scalar_gate,
            "exact_document_paths_and_hashes": paths_gate,
            "exact_non_evidence_semantics_and_provenance": semantic_gate,
            "exact_checkpoint_chain": checkpoint["passes"],
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        gates = false_gates
    return {"gates": gates, "passes": bool(all(gates.values()))}


def _candidate_paths(output: Path) -> tuple[Path, Path, Path, Path]:
    return tuple(
        output.with_name(f".{output.stem}.final-candidate.{suffix}")
        for suffix in ("json", "npz", "manifest.json", "pointer.json")
    )


def _finalize(
    output: Path,
    provenance: Mapping,
    task_auth: Mapping,
    model_auth: Mapping,
    cross_bind: Mapping,
) -> dict:
    output = Path(output)
    final_npz = output.with_suffix(".npz")
    final_manifest = output.with_suffix(".manifest.json")
    latest = output.with_name(f"{output.stem}.partial.latest.json")
    candidate_json, candidate_npz, candidate_manifest, candidate_pointer = (
        _candidate_paths(output)
    )
    candidates = (candidate_json, candidate_npz, candidate_manifest, candidate_pointer)
    if any(path.exists() for path in (output, final_npz, final_manifest)):
        raise RuntimeError("Oracle V2 final artifact already exists")
    if any(path.exists() for path in candidates):
        raise RuntimeError("Oracle V2 final candidate already exists")
    final_arrays = _final_arrays(task_auth, model_auth, cross_bind)
    try:
        _atomic_npz(candidate_npz, final_arrays)
        summary = {
            "protocol": PROTOCOL,
            "incomplete": False,
            "prerequisite_auth_only": True,
            "prerequisites_authenticated": True,
            "oracle_evidence": False,
            "benchmark_evidence": False,
            "all_oracle_gates_pass": False,
            "task_authentication": task_auth,
            "model_authentication": model_auth,
            "cross_bind": cross_bind,
            "producer_hash_convention": PRODUCER_HASH_CONVENTION,
            "task_pins": {
                key: {"path": str(path), "sha256": digest}
                for key, (path, digest) in TASK_PINS.items()
            },
            "model_pins": {
                key: {"path": str(path), "sha256": digest}
                for key, (path, digest) in MODEL_PINS.items()
            },
            "v1_rejected_report_only": V1_REJECTED_REPORT_ONLY,
            "rejected_v1_artifact_loads": 0,
            "provenance": provenance,
            "npz_path": str(final_npz),
            "npz_sha256": sha256_file(candidate_npz),
            "array_names": sorted(final_arrays),
            "array_hashes": {
                name: producer_array_hash(value)
                for name, value in final_arrays.items()
            },
        }
        _atomic_json(candidate_json, summary)
        manifest = {
            "protocol": PROTOCOL,
            "incomplete": False,
            "prerequisite_auth_only": True,
            "oracle_evidence": False,
            "benchmark_evidence": False,
            "json_path": str(output),
            "json_sha256": sha256_file(candidate_json),
            "npz_path": str(final_npz),
            "npz_sha256": sha256_file(candidate_npz),
            "side_artifact_count": 8,
            "side_artifacts": _side_artifacts(output),
        }
        _atomic_json(candidate_manifest, manifest)
        _atomic_json(
            candidate_pointer,
            {
                "protocol": PROTOCOL,
                "generation": 3,
                "stage": "honest_end_provenance",
                "incomplete": False,
                "superseded_by": str(output),
                "json_path": str(output),
                "json_sha256": sha256_file(candidate_json),
                "npz_path": str(final_npz),
                "npz_sha256": sha256_file(candidate_npz),
                "manifest_path": str(final_manifest),
                "manifest_sha256": sha256_file(candidate_manifest),
            },
        )
        retained = _certify_retained_documents(
            candidate_json,
            candidate_npz,
            candidate_manifest,
            candidate_pointer,
            expected_output=output,
            task_auth=task_auth,
            model_auth=model_auth,
            cross_bind=cross_bind,
        )
        if not retained["passes"]:
            raise RuntimeError("Oracle V2 prerequisite staged recertification failed")
        for source, destination in (
            (candidate_npz, final_npz),
            (candidate_json, output),
            (candidate_manifest, final_manifest),
        ):
            os.replace(source, destination)
        os.replace(candidate_pointer, latest)
        return summary
    except Exception:
        for path in candidates:
            if path.exists():
                path.unlink()
        raise


def recertify_retained_prerequisite(
    summary_path: Path = AUTHORIZED_OUTPUT,
    npz_path: Path | None = None,
    manifest_path: Path | None = None,
    pointer_path: Path | None = None,
) -> dict:
    """Independently reload predecessors and recertify all retained V2 bytes."""

    summary_path = Path(summary_path)
    if summary_path != AUTHORIZED_OUTPUT:
        return {
            "gates": {"exact_authorized_output_path": False},
            "passes": False,
        }
    npz_path = summary_path.with_suffix(".npz") if npz_path is None else Path(npz_path)
    manifest_path = (
        summary_path.with_suffix(".manifest.json")
        if manifest_path is None else Path(manifest_path)
    )
    pointer_path = (
        summary_path.with_name(f"{summary_path.stem}.partial.latest.json")
        if pointer_path is None else Path(pointer_path)
    )
    try:
        task_summary, task_arrays, _task_manifest, _task_pointer = (
            _load_pinned_artifact(TASK_PINS)
        )
        task_auth = authenticate_producer(
            task_summary,
            task_arrays,
            expected_count=TASK_ARRAY_COUNT,
            certificate_key="all_v4_gates_pass",
        )
        task_auth["pure_recert_pass"] = bool(
            _task_pure_recert(task_summary, task_arrays)
        )
        task_auth["passes"] = bool(
            task_auth["passes"] and task_auth["pure_recert_pass"]
        )
        model_summary, model_arrays, _model_manifest, _model_pointer = (
            _load_pinned_artifact(MODEL_PINS)
        )
        model_auth = authenticate_producer(
            model_summary,
            model_arrays,
            expected_count=MODEL_ARRAY_COUNT,
            certificate_key="all_model_preflight_gates_pass",
        )
        model_auth["pure_recert_pass"] = bool(
            _model_pure_recert(model_summary, model_arrays)
        )
        model_auth["passes"] = bool(
            model_auth["passes"] and model_auth["pure_recert_pass"]
        )
        cross_bind = cross_bind_prerequisites(
            task_summary.get("rows", ()), task_arrays, model_arrays
        )
        documents = _certify_retained_documents(
            summary_path,
            npz_path,
            manifest_path,
            pointer_path,
            expected_output=summary_path,
            task_auth=task_auth,
            model_auth=model_auth,
            cross_bind=cross_bind,
        )
        gates = {
            "task_producer_and_pure_recert": task_auth["passes"],
            "model_producer_and_pure_recert": model_auth["passes"],
            "cross_bind": cross_bind["passes"],
            "stored_detail_equals_independent_recomputation": bool(
                json.loads(summary_path.read_text()).get("task_authentication")
                == task_auth
                and json.loads(summary_path.read_text()).get(
                    "model_authentication"
                )
                == model_auth
                and json.loads(summary_path.read_text()).get("cross_bind")
                == cross_bind
            ),
            "retained_document_and_checkpoint_chain": documents["passes"],
        }
    except Exception:
        gates = {
            "task_producer_and_pure_recert": False,
            "model_producer_and_pure_recert": False,
            "cross_bind": False,
            "stored_detail_equals_independent_recomputation": False,
            "retained_document_and_checkpoint_chain": False,
        }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def _run_pipeline(
    output: Path,
    *,
    loader: Callable,
    task_recert: Callable,
    model_recert: Callable,
    cross_binder: Callable = cross_bind_prerequisites,
    provenance: dict,
    finish: Callable[[dict], None],
) -> dict:
    output = Path(output)
    _checkpoint(output, 0, "before_prerequisite_loads", provenance)

    task_summary, task_arrays, _task_manifest, _task_pointer = loader(TASK_PINS)
    provenance["task_artifact_loads"] = 1
    task_auth = authenticate_producer(
        task_summary,
        task_arrays,
        expected_count=TASK_ARRAY_COUNT,
        certificate_key="all_v4_gates_pass",
    )
    provenance["task_array_hashes_checked"] = len(task_auth["computed_array_hashes"])
    provenance["task_pure_recert_calls"] = 1
    task_auth["pure_recert_pass"] = bool(task_recert(task_summary, task_arrays))
    task_auth["passes"] = bool(task_auth["passes"] and task_auth["pure_recert_pass"])
    _checkpoint(output, 1, "task_prerequisite_authenticated" if task_auth["passes"] else "task_prerequisite_rejected", provenance, detail=task_auth)
    if not task_auth["passes"]:
        raise RuntimeError("Oracle V2 task prerequisite authentication failed")

    model_summary, model_arrays, _model_manifest, _model_pointer = loader(MODEL_PINS)
    provenance["model_artifact_loads"] = 1
    model_auth = authenticate_producer(
        model_summary,
        model_arrays,
        expected_count=MODEL_ARRAY_COUNT,
        certificate_key="all_model_preflight_gates_pass",
    )
    provenance["model_array_hashes_checked"] = len(model_auth["computed_array_hashes"])
    provenance["model_pure_recert_calls"] = 1
    model_auth["pure_recert_pass"] = bool(model_recert(model_summary, model_arrays))
    model_auth["passes"] = bool(model_auth["passes"] and model_auth["pure_recert_pass"])
    provenance["cross_bind_calls"] = 1
    cross = cross_binder(
        task_summary.get("rows", ()), task_arrays, model_arrays
    )
    model_stage_pass = bool(model_auth["passes"] and cross["passes"])
    _checkpoint(output, 2, "model_prerequisite_authenticated" if model_stage_pass else "model_prerequisite_rejected", provenance, detail={"model_authentication": model_auth, "cross_bind": cross})
    if not model_stage_pass:
        raise RuntimeError("Oracle V2 model prerequisite authentication failed")

    finish(provenance)
    provenance_pass = certify_provenance(provenance)
    _checkpoint(output, 3, "honest_end_provenance" if provenance_pass else "provenance_rejected", provenance, detail={"provenance_pass": provenance_pass})
    if not provenance_pass:
        raise RuntimeError("Oracle V2 prerequisite provenance failed")
    return _finalize(output, provenance, task_auth, model_auth, cross)


def execute_prerequisite(output, authorization=None):  # pragma: no cover
    if (
        RUNNER_EXECUTION_AUTHORIZATION is None
        or authorization is not RUNNER_EXECUTION_AUTHORIZATION
        or output != str(AUTHORIZED_OUTPUT)
    ):
        raise RuntimeError("Oracle V2 prerequisite execution is blocked")
    output = Path(output)
    if output.parent.exists():
        raise RuntimeError("Oracle V2 prerequisite root must be absent")
    root = repository_root()
    provenance = start_provenance(root)
    if not provenance["tracked_tree_clean_at_start"]:
        raise RuntimeError("Oracle V2 prerequisite requires tracked-clean source")
    return _run_pipeline(
        output,
        loader=_load_pinned_artifact,
        task_recert=_task_pure_recert,
        model_recert=_model_pure_recert,
        provenance=provenance,
        finish=lambda row: finish_provenance(row, root),
    )


def main(argv=None):  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not args.execute or RUNNER_EXECUTION_AUTHORIZATION is None:
        raise SystemExit("Oracle V2 prerequisite execution is blocked")
    execute_prerequisite(args.output, RUNNER_EXECUTION_AUTHORIZATION)


if __name__ == "__main__":
    main()
