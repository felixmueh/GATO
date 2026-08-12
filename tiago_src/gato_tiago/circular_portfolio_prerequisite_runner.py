"""CPU-only accepted-artifact and circular-geometry prerequisite transaction.

The capability is intentionally disabled in the static checkpoint.  A later
one-shot CPU authorization may authenticate the accepted V4 task/model
artifacts and Pin tool-origin geometry, but may not import the N96 extension or
run task RNG, optimization, CUDA, or a portfolio worker.
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
from typing import Callable, Mapping

import numpy as np

from gato_tiago.circular_portfolio import (
    EXPECTED_LEDGER, MODEL_ARRAY_COUNT, TASK_ARRAY_COUNT, TASK_IDENTITIES,
    circular_reference, certify_geometry, construct_geometry,
    reference_from_geometry,
)


PROTOCOL = "tiago_tool_center_circular_portfolio_prerequisite_p1_1"
OUTPUT = Path(
    "/tmp/tiago-tool-center-circular-portfolio-p1-prerequisite-authorized-once/prerequisite.json"
)
AUTHORIZED_CWD = Path("/workspace/GATO")
AUTHORIZED_ORIG_ARGV = (
    "python", "-B", "-m", "gato_tiago.circular_portfolio_prerequisite_runner",
    "--execute", "--output", str(OUTPUT),
)
RUNNER_EXECUTION_AUTHORIZATION = None
FROZEN_BUILD_HEAD = "2f1011da2a240fe8eae9ff25b2b3ee991c11c6c2"
FROZEN_CUDA_ARCH = "61-real"
FROZEN_EXTENSION = {
    "path": "/workspace/GATO/python/bsqp/bsqpN96_tiago_right_circular_portfolio_toll.cpython-310-x86_64-linux-gnu.so",
    "sha256": "b079410ade9e7de19ed3d4b7ed6f6ace27172cd442ccea0d5a46bb7277067a2f",
    "size_bytes": 6_690_480,
    "KNOT_POINTS": 96, "REFERENCE_SIZE": 10,
    "TOOL_POSITION_FRAME": "arm_right_tool_joint_origin",
    "TOOL_POSITION_SIZE": 3, "B1": True, "B16": True,
}
SOURCE_PATHS = (
    "CMakeLists.txt", "python/bindings.cu",
    "gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "gato/dynamics/tiago_right/tiago_right_grid.cuh",
    "gato/dynamics/tiago_right/tiago_right_arm.urdf",
    "tiago_src/gato_tiago/circular_portfolio.py",
    "tiago_src/gato_tiago/circular_portfolio_prerequisite_runner.py",
    "tiago_src/gato_tiago/multimodal_toll_oracle_v2.py",
    "tiago_src/gato_tiago/multimodal_toll_oracle_v2_prerequisite_runner.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_runner.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight_v4_runner.py",
    "tiago_src/gato_tiago/multimodal_pillar.py",
    "tests/python/test_tiago_circular_portfolio_prerequisite_static.py",
)
COUNTER_KEYS = (
    "task_artifact_loads", "task_pure_recert_calls", "task_array_hashes_checked",
    "model_artifact_loads", "model_pure_recert_calls", "model_array_hashes_checked",
    "cross_bind_calls", "pinocchio_model_calls", "primary_pin_fk_calls",
    "primary_geometry_constructions", "primary_reference_constructions",
    "recert_pin_fk_calls", "recert_geometry_constructions",
    "recert_reference_constructions", "task_rng_calls",
    "task_construction_calls", "optimizer_calls", "worker_subprocess_calls",
    "cuda_calls", "sqp_calls", "initializer_calls",
)

ARRAY_SPECS = {
    "public_x0_float32": ((12, 14), np.dtype(np.float32)),
    "accepted_task_reference_float32": ((12, 10), np.dtype(np.float32)),
    "public_default_side_int8": ((12,), np.dtype(np.int8)),
    "quarantined_q8_float64": ((12, 7), np.dtype(np.float64)),
    "joint_lower_float64": ((7,), np.dtype(np.float64)),
    "joint_upper_float64": ((7,), np.dtype(np.float64)),
    "velocity_limit_float64": ((7,), np.dtype(np.float64)),
    "effort_limit_float64": ((7,), np.dtype(np.float64)),
    "q0_tool_float64": ((12, 3), np.dtype(np.float64)),
    "q8_tool_float64": ((12, 3), np.dtype(np.float64)),
    "portfolio_reference_float32": ((12, 10), np.dtype(np.float32)),
    "route_tool_reference_float64": ((192, 96, 3), np.dtype(np.float64)),
    "ledger_seed_int64": ((192,), np.dtype(np.int64)),
    "ledger_route_int8": ((192,), np.dtype(np.int8)),
    "ledger_profile_int8": ((192,), np.dtype(np.int8)),
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(
        f"{value.dtype.str}|{value.shape}|".encode() + value.tobytes()
    ).hexdigest()


def _canonical_json(value):
    if isinstance(value, Mapping):
        return {str(key): _canonical_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _source_hashes(root):
    return {path: _sha256_file(root / path) for path in SOURCE_PATHS}


def _extension_measurement():
    path = Path(FROZEN_EXTENSION["path"])
    stat = path.stat()
    return {
        "path": str(path.resolve()), "sha256": _sha256_file(path),
        "size_bytes": int(stat.st_size),
        "accepted_attributes": {
            key: FROZEN_EXTENSION[key] for key in (
                "KNOT_POINTS", "REFERENCE_SIZE", "TOOL_POSITION_FRAME",
                "TOOL_POSITION_SIZE", "B1", "B16",
            )
        },
    }


def certify_extension_measurement(value):
    expected_keys = {"path", "sha256", "size_bytes", "accepted_attributes"}
    expected_attributes = {
        key: FROZEN_EXTENSION[key] for key in (
            "KNOT_POINTS", "REFERENCE_SIZE", "TOOL_POSITION_FRAME",
            "TOOL_POSITION_SIZE", "B1", "B16",
        )
    }
    return bool(
        isinstance(value, Mapping) and set(value) == expected_keys
        and value["path"] == str(Path(FROZEN_EXTENSION["path"]).resolve())
        and value["sha256"] == FROZEN_EXTENSION["sha256"]
        and value["size_bytes"] == FROZEN_EXTENSION["size_bytes"]
        and value["accepted_attributes"] == expected_attributes
    )


def _runtime_versions():
    return {
        "python": sys.version, "numpy": np.__version__,
        "pinocchio": importlib.metadata.version("pin"),
    }


def _start_provenance(root, orig_argv=None):
    argv = tuple(sys.orig_argv if orig_argv is None else orig_argv)
    return {
        "protocol": PROTOCOL, "cwd": str(Path.cwd().resolve()),
        "orig_argv": list(argv), "exact_command": shlex.join(argv),
        "git_head_at_start": _git(root, "rev-parse", "HEAD"),
        "git_head_at_end": None,
        "tracked_clean_at_start": _git(root, "status", "--porcelain", "--untracked-files=no") == "",
        "tracked_clean_at_end": None,
        "source_hashes_at_start": _source_hashes(root),
        "source_hashes_at_end": None,
        "build_head": FROZEN_BUILD_HEAD, "cuda_arch": FROZEN_CUDA_ARCH,
        "extension_at_start": _extension_measurement(),
        "extension_at_end": None, "runtime_versions": _runtime_versions(),
    }


def _finish_provenance(value, root):
    value["git_head_at_end"] = _git(root, "rev-parse", "HEAD")
    value["tracked_clean_at_end"] = _git(
        root, "status", "--porcelain", "--untracked-files=no"
    ) == ""
    value["source_hashes_at_end"] = _source_hashes(root)
    value["extension_at_end"] = _extension_measurement()


def certify_provenance(value, *, final):
    required = {
        "protocol", "cwd", "orig_argv", "exact_command", "git_head_at_start",
        "git_head_at_end", "tracked_clean_at_start", "tracked_clean_at_end",
        "source_hashes_at_start", "source_hashes_at_end", "build_head",
        "cuda_arch", "extension_at_start", "extension_at_end",
        "runtime_versions",
    }
    if set(value) != required:
        return False
    start = value["git_head_at_start"]
    return bool(
        value["protocol"] == PROTOCOL and value["cwd"] == str(AUTHORIZED_CWD)
        and tuple(value["orig_argv"]) == AUTHORIZED_ORIG_ARGV
        and value["exact_command"] == shlex.join(AUTHORIZED_ORIG_ARGV)
        and isinstance(start, str) and len(start) == 40
        and value["tracked_clean_at_start"] is True
        and set(value["source_hashes_at_start"]) == set(SOURCE_PATHS)
        and all(len(item) == 64 for item in value["source_hashes_at_start"].values())
        and value["build_head"] == FROZEN_BUILD_HEAD
        and value["cuda_arch"] == FROZEN_CUDA_ARCH
        and certify_extension_measurement(value["extension_at_start"])
        and value["runtime_versions"] == _runtime_versions()
        and ((not final and value["git_head_at_end"] is None
              and value["tracked_clean_at_end"] is None
              and value["source_hashes_at_end"] is None
              and value["extension_at_end"] is None)
             or (final and value["git_head_at_end"] == start
                 and value["tracked_clean_at_end"] is True
                 and value["source_hashes_at_end"] == value["source_hashes_at_start"]
                 and certify_extension_measurement(value["extension_at_end"])
                 and value["extension_at_end"] == value["extension_at_start"]))
    )


def _expected_counters(stage):
    task = stage >= 1; model = stage >= 2; geometry = stage >= 3
    return {
        "task_artifact_loads": int(task), "task_pure_recert_calls": int(task),
        "task_array_hashes_checked": TASK_ARRAY_COUNT if task else 0,
        "model_artifact_loads": int(model), "model_pure_recert_calls": int(model),
        "model_array_hashes_checked": MODEL_ARRAY_COUNT if model else 0,
        "cross_bind_calls": int(model), "pinocchio_model_calls": int(geometry),
        "primary_pin_fk_calls": 24 if geometry else 0,
        "primary_geometry_constructions": 204 if geometry else 0,
        "primary_reference_constructions": 204 if geometry else 0,
        "recert_pin_fk_calls": 24 if geometry else 0,
        "recert_geometry_constructions": 204 if geometry else 0,
        "recert_reference_constructions": 204 if geometry else 0,
        "task_rng_calls": 0, "task_construction_calls": 0,
        "optimizer_calls": 0, "worker_subprocess_calls": 0, "cuda_calls": 0,
        "sqp_calls": 0, "initializer_calls": 0,
    }


def authenticate_task_artifact():  # pragma: no cover - authorized CPU boundary
    from gato_tiago.multimodal_toll_oracle_v2 import TASK_PINS
    from gato_tiago.multimodal_toll_oracle_v2_prerequisite_runner import (
        _load_pinned_artifact, _task_pure_recert, authenticate_producer,
    )
    summary, arrays, manifest, pointer = _load_pinned_artifact(TASK_PINS)
    detail = authenticate_producer(
        summary, arrays, expected_count=TASK_ARRAY_COUNT,
        certificate_key="all_v4_gates_pass",
    )
    detail["pure_recert"] = _task_pure_recert(summary, arrays)
    detail["passes"] = bool(detail["passes"] and detail["pure_recert"])
    return detail, summary, arrays, manifest, pointer


def authenticate_model_artifact(task_summary, task_arrays):  # pragma: no cover
    from gato_tiago.multimodal_toll_oracle_v2 import MODEL_PINS
    from gato_tiago.multimodal_toll_oracle_v2_prerequisite_runner import (
        _load_pinned_artifact, _model_pure_recert, authenticate_producer,
        cross_bind_prerequisites,
    )
    summary, arrays, manifest, pointer = _load_pinned_artifact(MODEL_PINS)
    detail = authenticate_producer(
        summary, arrays, expected_count=MODEL_ARRAY_COUNT,
        certificate_key="all_model_preflight_gates_pass",
    )
    detail["pure_recert"] = _model_pure_recert(summary, arrays)
    detail["passes"] = bool(detail["passes"] and detail["pure_recert"])
    cross = cross_bind_prerequisites(task_summary.get("rows", ()), task_arrays, arrays)
    return detail, cross, summary, arrays, manifest, pointer


def production_kinematics():  # pragma: no cover - authorized CPU Pin boundary
    import pinocchio as pin
    from gato_tiago.multimodal_pillar import TOOL_FRAME, load_model, tool_position
    root = Path(__file__).resolve().parents[2]
    model = load_model(root / "gato/dynamics/tiago_right/tiago_right_arm.urdf")
    frame = model.getFrameId(TOOL_FRAME)
    def evaluate(q):
        data = model.createData()
        position = tool_position(model, data, np.asarray(q, np.float64))
        jacobian = pin.computeFrameJacobian(
            model, data, np.asarray(q, np.float64), frame,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3]
        return position.copy(), jacobian.copy()
    return evaluate


def construct_geometry_arrays(task_summary, task_arrays, model_arrays,
                              kinematics: Callable, *, call_counts=None):
    counts = {"pin_fk_calls": 0, "geometry_constructions": 0,
              "reference_constructions": 0} if call_counts is None else call_counts
    if set(counts) != {"pin_fk_calls", "geometry_constructions", "reference_constructions"}:
        raise ValueError("geometry call ledger has invalid keys")
    def fk(q):
        counts["pin_fk_calls"] += 1
        return kinematics(q)
    def geometry_for(start, goal, side):
        counts["geometry_constructions"] += 1
        return construct_geometry(start, goal, side)
    def public_reference(geometry):
        counts["reference_constructions"] += 1
        return reference_from_geometry(geometry).as_float32()
    def route_reference(geometry, route, profile):
        counts["reference_constructions"] += 1
        return circular_reference(geometry, route, profile)
    arrays = {name: np.empty(shape, dtype) for name, (shape, dtype) in ARRAY_SPECS.items()}
    arrays["joint_lower_float64"][:] = model_arrays["model_lower_float64"]
    arrays["joint_upper_float64"][:] = model_arrays["model_upper_float64"]
    arrays["velocity_limit_float64"][:] = model_arrays["model_velocity_float64"]
    arrays["effort_limit_float64"][:] = model_arrays["model_effort_float64"]
    task_index_by_seed = {identity[1]: index for index, identity in enumerate(TASK_IDENTITIES)}
    geometry_details = []
    for task_index, identity in enumerate(TASK_IDENTITIES):
        seed = identity[1]
        x0 = np.asarray(task_arrays[f"task_{seed}_x0_float32"])
        accepted = np.asarray(task_arrays[f"task_{seed}_reference_float32"])
        q8 = np.asarray(task_arrays[f"task_{seed}_history_q_float64"])[8]
        default = int(task_summary["rows"][task_index]["public_task"]["default_side"])
        p0, _ = fk(x0[:7]); p8, _ = fk(q8)
        geometry = geometry_for(p0, p8, default)
        detail = certify_geometry(geometry)
        if not detail["passes"]:
            raise RuntimeError(f"circular geometry failed for task {seed}")
        arrays["public_x0_float32"][task_index] = x0
        arrays["accepted_task_reference_float32"][task_index] = accepted
        arrays["public_default_side_int8"][task_index] = default
        arrays["quarantined_q8_float64"][task_index] = q8
        arrays["q0_tool_float64"][task_index] = p0
        arrays["q8_tool_float64"][task_index] = p8
        arrays["portfolio_reference_float32"][task_index] = public_reference(geometry)
        geometry_details.append({"identity": list(identity), "certificate": detail})
    for index, ledger in enumerate(EXPECTED_LEDGER):
        task_index = task_index_by_seed[ledger[1]]
        geometry = geometry_for(
            arrays["q0_tool_float64"][task_index], arrays["q8_tool_float64"][task_index],
            int(arrays["public_default_side_int8"][task_index]),
        )
        arrays["route_tool_reference_float64"][index] = route_reference(
            geometry, ledger[2], ledger[3]
        )
        arrays["ledger_seed_int64"][index] = ledger[1]
        arrays["ledger_route_int8"][index] = 0 if ledger[2] == "short" else 1
        arrays["ledger_profile_int8"][index] = ledger[3]
    return arrays, geometry_details, dict(counts)


def certify_geometry_arrays(arrays, task_summary, task_arrays, model_arrays,
                            kinematics: Callable):
    schema = bool(
        set(arrays) == set(ARRAY_SPECS)
        and all(np.asarray(arrays[name]).shape == shape
                and np.asarray(arrays[name]).dtype == dtype
                and np.isfinite(np.asarray(arrays[name])).all()
                for name, (shape, dtype) in ARRAY_SPECS.items())
    )
    if not schema:
        return {"schema": False, "passes": False}
    try:
        regenerated, details, calls = construct_geometry_arrays(
            task_summary, task_arrays, model_arrays, kinematics,
            call_counts={"pin_fk_calls": 0, "geometry_constructions": 0,
                         "reference_constructions": 0},
        )
    except (ValueError, RuntimeError, KeyError, IndexError):
        return {"schema": True, "passes": False}
    exact = {name: np.array_equal(arrays[name], regenerated[name]) for name in ARRAY_SPECS}
    gates = {
        "schema": schema, "exact_regeneration": bool(all(exact.values())),
        "all12_geometry": len(details) == 12 and all(row["certificate"]["passes"] for row in details),
        "all192_references": arrays["route_tool_reference_float64"].shape == (192, 96, 3),
        "accepted_x0_velocity_zero": np.array_equal(
            arrays["public_x0_float32"][:, 7:], np.zeros((12, 7), np.float32)
        ),
        "exact_recert_call_ledger": calls == {
            "pin_fk_calls": 24, "geometry_constructions": 204,
            "reference_constructions": 204,
        },
    }
    return {"gates": gates, "array_gates": exact, "geometry_details": details,
            "recert_call_counts": calls,
            "passes": bool(all(gates.values()))}


def _write_npz_exclusive(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez(stream, **arrays)


def _write_json_exclusive(path, value):
    with Path(path).open("x") as stream:
        json.dump(_canonical_json(value), stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def _checkpoint_paths(output, generation):
    return (
        output.with_name(f"{output.stem}.partial.gen{generation}.json"),
        output.with_name(f"{output.stem}.partial.gen{generation}.npz"),
        output.with_name(f"{output.stem}.partial.latest.json"),
    )


def checkpoint_document(generation, stage, counters, provenance, arrays,
                        task_auth=None, model_auth=None, cross_bind=None,
                        geometry_certificate=None):
    return {
        "protocol": PROTOCOL, "generation": generation, "stage": stage,
        "incomplete": True, "counters": counters, "provenance": provenance,
        "task_authentication": task_auth, "model_authentication": model_auth,
        "cross_bind": cross_bind, "geometry_certificate": geometry_certificate,
        "array_names": sorted(arrays),
        "array_hashes": {name: _array_hash(arrays[name]) for name in sorted(arrays)},
        "oracle_evidence": False, "benchmark_evidence": False,
        "prerequisite_auth_only": True,
    }


def certify_checkpoint(document, arrays, generation, owned_detail=None):
    stages = ("gen0", "task_authenticated", "model_cross_bound", "geometry_authenticated", "honest_end")
    exact_keys = {
        "protocol", "generation", "stage", "incomplete", "counters", "provenance",
        "task_authentication", "model_authentication", "cross_bind",
        "geometry_certificate", "array_names", "array_hashes",
        "oracle_evidence", "benchmark_evidence", "prerequisite_auth_only",
    }
    stage = stages[generation] if 0 <= generation < len(stages) else None
    expected_names = [] if generation < 3 else sorted(ARRAY_SPECS)
    hashes = {name: _array_hash(arrays[name]) for name in expected_names} if set(arrays) == set(expected_names) else {}
    owned = owned_detail or {}
    task_expected = None if generation == 0 else owned.get("task")
    model_expected = None if generation < 2 else owned.get("model")
    cross_expected = None if generation < 2 else owned.get("cross_bind")
    geometry_expected = None if generation < 3 else owned.get("geometry")
    return bool(
        set(document) == exact_keys and stage is not None
        and document["protocol"] == PROTOCOL and document["generation"] == generation
        and document["stage"] == stage and document["incomplete"] is True
        and document["counters"] == _expected_counters(generation)
        and certify_provenance(document["provenance"], final=generation == 4)
        and document["array_names"] == expected_names
        and document["array_hashes"] == hashes
        and _canonical_json(document["task_authentication"]) == _canonical_json(task_expected)
        and _canonical_json(document["model_authentication"]) == _canonical_json(model_expected)
        and _canonical_json(document["cross_bind"]) == _canonical_json(cross_expected)
        and _canonical_json(document["geometry_certificate"]) == _canonical_json(geometry_expected)
        and (generation == 0 or task_expected.get("passes") is True)
        and (generation < 2 or model_expected.get("passes") is True)
        and (generation < 2 or cross_expected.get("passes") is True)
        and (generation < 3 or geometry_expected.get("passes") is True)
        and document["oracle_evidence"] is False
        and document["benchmark_evidence"] is False
        and document["prerequisite_auth_only"] is True
    )


def publish_checkpoint(output, document, arrays, *, owned_detail):
    generation = document["generation"]
    if not certify_checkpoint(document, arrays, generation, owned_detail):
        raise ValueError("prerequisite checkpoint failed certification")
    json_path, npz_path, pointer = _checkpoint_paths(Path(output), generation)
    if json_path.exists() or npz_path.exists():
        raise FileExistsError("prerequisite checkpoint overwrite forbidden")
    _write_npz_exclusive(npz_path, arrays)
    stored = dict(document, npz_path=str(npz_path), npz_sha256=_sha256_file(npz_path))
    _write_json_exclusive(json_path, stored)
    pointer_tmp = pointer.with_suffix(pointer.suffix + ".candidate")
    _write_json_exclusive(pointer_tmp, {
        "protocol": PROTOCOL, "incomplete": True, "generation": generation,
        "json_path": str(json_path), "json_sha256": _sha256_file(json_path),
        "npz_path": str(npz_path), "npz_sha256": _sha256_file(npz_path),
    })
    os.replace(pointer_tmp, pointer)
    return json_path, npz_path


def certify_final(summary, arrays, task_bundle, model_bundle, semantic,
                  expected_npz_path, actual_npz_sha256):
    task_auth, task_summary, task_arrays, _tm, _tp = task_bundle
    model_auth, cross, _model_summary, model_arrays, _mm, _mp = model_bundle
    names = sorted(ARRAY_SPECS)
    hashes = {name: _array_hash(arrays[name]) for name in names} if semantic["passes"] else {}
    required = {
        "protocol", "incomplete", "overall_pass", "prerequisite_auth_only",
        "oracle_evidence", "benchmark_evidence", "task_authentication",
        "model_authentication", "cross_bind", "geometry_certificate", "counters",
        "provenance", "array_names", "array_hashes", "npz_path", "npz_sha256",
        "checkpoint_count",
    }
    gates = {
        "keys": set(summary) == required, "protocol": summary.get("protocol") == PROTOCOL,
        "complete_non_evidence": summary.get("incomplete") is False
        and summary.get("overall_pass") is True
        and summary.get("prerequisite_auth_only") is True
        and summary.get("oracle_evidence") is False
        and summary.get("benchmark_evidence") is False,
        "task_auth": _canonical_json(summary.get("task_authentication")) == _canonical_json(task_auth) and task_auth.get("passes") is True,
        "model_auth": _canonical_json(summary.get("model_authentication")) == _canonical_json(model_auth) and model_auth.get("passes") is True,
        "cross": _canonical_json(summary.get("cross_bind")) == _canonical_json(cross) and cross.get("passes") is True,
        "geometry": _canonical_json(summary.get("geometry_certificate")) == _canonical_json(semantic) and semantic["passes"],
        "counters": summary.get("counters") == _expected_counters(4),
        "provenance": certify_provenance(summary.get("provenance", {}), final=True),
        "arrays": summary.get("array_names") == names and summary.get("array_hashes") == hashes,
        "npz_identity": summary.get("npz_path") == str(Path(expected_npz_path).resolve())
        and summary.get("npz_sha256") == actual_npz_sha256
        and isinstance(actual_npz_sha256, str) and len(actual_npz_sha256) == 64,
        "checkpoints": summary.get("checkpoint_count") == 5,
    }
    return {"gates": gates, "semantic": semantic, "passes": bool(all(gates.values()))}


def refuse_existing(output):
    output = Path(output).resolve()
    candidates = [output, output.with_suffix(".npz"), output.with_name(output.stem + ".manifest.json"),
                  output.with_name(output.stem + ".partial.latest.json")]
    candidates.extend(output.parent.glob(output.stem + ".partial.gen*"))
    if any(path.exists() for path in candidates):
        raise FileExistsError("prerequisite transaction forbids overwrite/resume")


def recertify_retained_prerequisite(output=OUTPUT, *, return_payload=False):  # pragma: no cover - audit boundary
    output = Path(output).resolve()
    try:
        summary = json.loads(output.read_text())
        npz_path = output.with_suffix(".npz")
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        task = authenticate_task_artifact()
        model = authenticate_model_artifact(task[1], task[2])
        kinematics = production_kinematics()
        semantic = certify_geometry_arrays(arrays, task[1], task[2], model[3], kinematics)
        owned = {"task": task[0], "model": model[0], "cross_bind": model[1],
                 "geometry": semantic}
        final = certify_final(
            summary, arrays, task, model, semantic, npz_path, _sha256_file(npz_path)
        )
        manifest_path = output.with_name(output.stem + ".manifest.json")
        pointer_path = output.with_name(output.stem + ".partial.latest.json")
        manifest = json.loads(manifest_path.read_text())
        pointer = json.loads(pointer_path.read_text())
        sides = []
        checkpoints = True
        for generation in range(5):
            json_checkpoint, npz_checkpoint, _ = _checkpoint_paths(output, generation)
            row = json.loads(json_checkpoint.read_text())
            with np.load(npz_checkpoint, allow_pickle=False) as archive:
                checkpoint_arrays = {name: archive[name] for name in archive.files}
            stored = dict(row); stored.pop("npz_path"); stored.pop("npz_sha256")
            checkpoints &= certify_checkpoint(stored, checkpoint_arrays, generation, owned)
            checkpoints &= row["npz_path"] == str(npz_checkpoint)
            checkpoints &= row["npz_sha256"] == _sha256_file(npz_checkpoint)
            sides.extend((
                {"path": str(json_checkpoint), "sha256": _sha256_file(json_checkpoint)},
                {"path": str(npz_checkpoint), "sha256": _sha256_file(npz_checkpoint)},
            ))
        documents = bool(
            manifest == {"protocol": PROTOCOL, "json_path": str(output),
                         "json_sha256": _sha256_file(output), "npz_path": str(npz_path),
                         "npz_sha256": _sha256_file(npz_path), "side_artifacts": sides}
            and pointer == {"protocol": PROTOCOL, "incomplete": False,
                            "json_path": str(output), "json_sha256": _sha256_file(output),
                            "npz_path": str(npz_path), "npz_sha256": _sha256_file(npz_path),
                            "manifest_path": str(manifest_path),
                            "manifest_sha256": _sha256_file(manifest_path)}
        )
        result = {"final": final, "checkpoints": bool(checkpoints), "documents": documents,
                  "passes": bool(final["passes"] and checkpoints and documents)}
        if return_payload:
            result["_retained_summary"] = summary
            result["_retained_arrays"] = arrays
        return result
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError):
        return {"passes": False}


def execute(output=OUTPUT, authorization=None):  # pragma: no cover - one-shot CPU boundary
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("circular portfolio prerequisite execution is blocked")
    output = Path(output).resolve()
    if output != OUTPUT:
        raise RuntimeError("circular portfolio prerequisite output path is not authorized")
    refuse_existing(output); output.parent.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    provenance = _start_provenance(root)
    checkpoints = []
    owned = {"task": None, "model": None, "cross_bind": None, "geometry": None}
    checkpoints.append(publish_checkpoint(output, checkpoint_document(
        0, "gen0", _expected_counters(0), provenance, {}
    ), {}, owned_detail=owned))
    task = authenticate_task_artifact()
    owned["task"] = task[0]
    checkpoints.append(publish_checkpoint(output, checkpoint_document(
        1, "task_authenticated", _expected_counters(1), provenance, {}, task_auth=task[0]
    ), {}, owned_detail=owned))
    model = authenticate_model_artifact(task[1], task[2])
    owned["model"] = model[0]; owned["cross_bind"] = model[1]
    checkpoints.append(publish_checkpoint(output, checkpoint_document(
        2, "model_cross_bound", _expected_counters(2), provenance, {},
        task_auth=task[0], model_auth=model[0], cross_bind=model[1]
    ), {}, owned_detail=owned))
    kinematics = production_kinematics()
    primary_calls = {"pin_fk_calls": 0, "geometry_constructions": 0,
                     "reference_constructions": 0}
    arrays, _details, primary_calls = construct_geometry_arrays(
        task[1], task[2], model[3], kinematics, call_counts=primary_calls
    )
    if primary_calls != {"pin_fk_calls": 24, "geometry_constructions": 204,
                         "reference_constructions": 204}:
        raise RuntimeError("primary geometry call ledger mismatch")
    geometry = certify_geometry_arrays(arrays, task[1], task[2], model[3], kinematics)
    owned["geometry"] = geometry
    checkpoints.append(publish_checkpoint(output, checkpoint_document(
        3, "geometry_authenticated", _expected_counters(3), provenance, arrays,
        task_auth=task[0], model_auth=model[0], cross_bind=model[1],
        geometry_certificate=geometry,
    ), arrays, owned_detail=owned))
    _finish_provenance(provenance, root)
    checkpoints.append(publish_checkpoint(output, checkpoint_document(
        4, "honest_end", _expected_counters(4), provenance, arrays,
        task_auth=task[0], model_auth=model[0], cross_bind=model[1],
        geometry_certificate=geometry,
    ), arrays, owned_detail=owned))
    npz_path = output.with_suffix(".npz")
    summary = {
        "protocol": PROTOCOL, "incomplete": False, "overall_pass": True,
        "prerequisite_auth_only": True, "oracle_evidence": False,
        "benchmark_evidence": False, "task_authentication": task[0],
        "model_authentication": model[0], "cross_bind": model[1],
        "geometry_certificate": geometry, "counters": _expected_counters(4),
        "provenance": provenance, "array_names": sorted(arrays),
        "array_hashes": {name: _array_hash(arrays[name]) for name in sorted(arrays)},
        "npz_path": str(npz_path), "npz_sha256": None, "checkpoint_count": 5,
    }
    npz_candidate = npz_path.with_suffix(npz_path.suffix + ".candidate")
    json_candidate = output.with_suffix(output.suffix + ".candidate")
    if npz_candidate.exists() or json_candidate.exists():
        raise FileExistsError("orphan prerequisite final candidate exists")
    _write_npz_exclusive(npz_candidate, arrays)
    summary["npz_sha256"] = _sha256_file(npz_candidate)
    if not certify_final(
        summary, arrays, task, model, geometry, npz_path, summary["npz_sha256"]
    )["passes"]:
        npz_candidate.unlink()
        raise RuntimeError("prerequisite failed before final publication")
    _write_json_exclusive(json_candidate, summary)
    os.replace(npz_candidate, npz_path); os.replace(json_candidate, output)
    sides = [{"path": str(path), "sha256": _sha256_file(path)} for pair in checkpoints for path in pair]
    manifest_path = output.with_name(output.stem + ".manifest.json")
    _write_json_exclusive(manifest_path, {
        "protocol": PROTOCOL, "json_path": str(output), "json_sha256": _sha256_file(output),
        "npz_path": str(npz_path), "npz_sha256": _sha256_file(npz_path),
        "side_artifacts": sides,
    })
    pointer = output.with_name(output.stem + ".partial.latest.json")
    pointer_tmp = pointer.with_suffix(pointer.suffix + ".candidate")
    _write_json_exclusive(pointer_tmp, {
        "protocol": PROTOCOL, "incomplete": False,
        "json_path": str(output), "json_sha256": _sha256_file(output),
        "npz_path": str(npz_path), "npz_sha256": _sha256_file(npz_path),
        "manifest_path": str(manifest_path), "manifest_sha256": _sha256_file(manifest_path),
    })
    os.replace(pointer_tmp, pointer)
    return summary


def main(argv=None):  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.execute:
        raise SystemExit("--execute is required")
    execute(args.output, RUNNER_EXECUTION_AUTHORIZATION)


if __name__ == "__main__":  # pragma: no cover
    main()
