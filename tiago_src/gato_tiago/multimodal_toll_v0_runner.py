"""Transactional Stage V0 runner for Tiago tool-center model validation.

This module is deliberately fail closed.  The public entry point has no way to
accept caller-produced evidence: after a future source authorization it will
generate the frozen draws and tasks itself and launch each CUDA extension in a
fresh Python process.  Static tests use the private mock seam and can never
produce evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shlex
import tempfile
import traceback
from typing import Callable, Mapping, Sequence

import numpy as np

from gato_tiago import multimodal_toll as toll
from gato_tiago.multimodal_toll_v0 import (
    EXPECTED_TASK_IDENTITIES,
    REQUIRED_SOURCE_PATHS,
    V0_PROTOCOL_VERSION,
    certify_v0,
    constant_acceleration_step,
    generate_v0_draws,
)
from gato_tiago.multimodal_toll_v0_worker import (
    EXPECTED_WORKER_ARRAY_NAMES,
    SUPPORTED_MODULES,
    WORKER_PROTOCOL_VERSION,
)


RUNNER_EXECUTION_AUTHORIZATION = None
RUNNER_PROTOCOL_VERSION = "tiago_tool_center_toll_v0_runner_1"
AUTHORIZED_OUTPUT_PATH = Path(
    "/tmp/tiago-tool-center-toll-v0-authorized-once/v0.json"
)
EXPECTED_TASK_COUNT = 12
EXPECTED_WORKER_COUNT = 3
EXPECTED_SQP_OPTIMIZATION_CALLS = 0
_PRODUCTION_PIPELINE_TOKEN = object()
FROZEN_EXTENSIONS = (
    {
        "module_name": "bsqp.bsqpN64_tiago_right_multimodal_toll",
        "extension_path": "python/bsqp/bsqpN64_tiago_right_multimodal_toll.cpython-310-x86_64-linux-gnu.so",
        "extension_sha256": "3d3c1e0808a57b6189a0267e701a64d0b11ed13cda37b65873781954ffaa8551",
        "reference_size": 10,
        "build_command": "./tools/build.sh --plant tiago_right_multimodal_toll --knots 64 --target bsqpN64_tiago_right_multimodal_toll --native-cuda-arch",
    },
    {
        "module_name": "bsqp.bsqpN64_tiago_right",
        "extension_path": "python/bsqp/bsqpN64_tiago_right.cpython-310-x86_64-linux-gnu.so",
        "extension_sha256": "f0944080523a212edce0af72d323f5eeffd587adde2f0bea6063a31fabff2f41",
        "reference_size": 6,
        "build_command": "./tools/build.sh --plant tiago_right --knots 64 --target bsqpN64_tiago_right --native-cuda-arch",
    },
    {
        "module_name": "bsqp.bsqpN64_tiago_right_multimodal",
        "extension_path": "python/bsqp/bsqpN64_tiago_right_multimodal.cpython-310-x86_64-linux-gnu.so",
        "extension_sha256": "75650c37b3cd0e729922cdd82e65dd81214bb29f9591d0a7466aac197908b1fe",
        "reference_size": 6,
        "build_command": "./tools/build.sh --plant tiago_right_multimodal --knots 64 --target bsqpN64_tiago_right_multimodal --native-cuda-arch",
    },
)


def repository_root(module_path=Path(__file__)):
    root = Path(module_path).resolve().parents[2]
    sentinels = (
        root / ".git",
        root / "CMakeLists.txt",
        root / "gato/bsqp/bsqp.cuh",
        root / "tiago_src/gato_tiago/multimodal_toll_v0_runner.py",
    )
    if not all(path.exists() for path in sentinels):
        raise RuntimeError("Stage V0 repository root sentinel mismatch")
    return root


def exact_invocation_provenance():
    original = list(sys.orig_argv)
    return {
        "exact_command": shlex.join(original),
        "orig_argv": original,
        "sys_argv": list(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "critical_environment": {
            name: os.environ.get(name)
            for name in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")
        },
        "python_executable": sys.executable,
    }


def certify_runner_provenance(provenance, worker_rows):
    """Pure, fail-closed provenance gate for a future completed execution."""

    if not isinstance(provenance, Mapping) or len(worker_rows) != 3:
        return {"all_runner_provenance_gates_pass": False}
    source_hashes = provenance.get("source_hashes")
    exact_sources = bool(
        isinstance(source_hashes, Mapping)
        and set(source_hashes) == set(REQUIRED_SOURCE_PATHS)
        and all(
            isinstance(source_hashes[name], Mapping)
            and source_hashes[name].get("path") == path
            and isinstance(source_hashes[name].get("sha256"), str)
            and len(source_hashes[name]["sha256"]) == 64
            for name, path in REQUIRED_SOURCE_PATHS.items()
        )
    )
    commands = [tuple(row.get("command", ())) for row in worker_rows]
    modules = [row.get("module_name") for row in worker_rows]
    worker_exact = bool(
        modules == list(SUPPORTED_MODULES)
        and len(set(commands)) == 3
        and all(row.get("exit_code") == 0 for row in worker_rows)
    )
    gates = {
        "tracked_clean_at_both_boundaries": provenance.get(
            "tracked_tree_clean_at_start"
        )
        is True
        and provenance.get("tracked_tree_clean_at_end") is True,
        "git_head_stable": isinstance(provenance.get("git_head_at_start"), str)
        and len(provenance["git_head_at_start"]) == 40
        and provenance.get("git_head_at_end") == provenance["git_head_at_start"],
        "exact_command_retained": isinstance(provenance.get("exact_command"), str)
        and bool(provenance["exact_command"]),
        "exact_command_reconstructs_orig_argv": isinstance(
            provenance.get("orig_argv"), list
        )
        and provenance.get("exact_command")
        == shlex.join(provenance.get("orig_argv", [])),
        "cwd_and_environment_retained": isinstance(provenance.get("cwd"), str)
        and bool(provenance.get("cwd"))
        and isinstance(provenance.get("critical_environment"), Mapping)
        and set(provenance["critical_environment"])
        == {"PYTHONPATH", "CUDA_VISIBLE_DEVICES"},
        "model_hash_retained": isinstance(provenance.get("model_sha256"), str)
        and len(provenance["model_sha256"]) == 64,
        "all_extension_hashes_retained": isinstance(
            provenance.get("extension_hashes"), Mapping
        )
        and set(provenance["extension_hashes"]) == set(SUPPORTED_MODULES)
        and all(len(value) == 64 for value in provenance["extension_hashes"].values()),
        "source_hashes_exact": exact_sources,
        "fresh_worker_processes_exact": worker_exact,
        "zero_retry_resume": provenance.get("retry_count") == 0
        and provenance.get("resume_count") == 0,
    }
    gates["all_runner_provenance_gates_pass"] = all(gates.values())
    return gates


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        f"{array.dtype.str}|{array.shape}|".encode() + array.tobytes()
    ).hexdigest()


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
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


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
    candidates = [output, output.with_suffix(".npz"), output.with_suffix(".manifest.json")]
    candidates += list(output.parent.glob(f"{output.stem}.partial.*"))
    if any(path.exists() for path in candidates):
        raise RuntimeError("Stage V0 does not permit resume, overwrite, or rerun")


def _checkpoint(
    output, completed, expected, rows, arrays, provenance, *, generation=None, stage=None
):
    output = Path(output)
    count = len(completed)
    generation = count if generation is None else int(generation)
    stem = output.parent / f"{output.stem}.partial.{generation:04d}"
    npz_path = Path(f"{stem}.npz")
    json_path = Path(f"{stem}.json")
    _atomic_npz(npz_path, arrays)
    summary = {
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "incomplete": True,
        "all_v0_gates_pass": False,
        "optimization_evidence": False,
        "sqp_optimization_calls": 0,
        "expected_identities": [list(row) for row in expected],
        "completed_identities": [list(row) for row in completed],
        "attempted_identities": [
            list(row["identity"])
            for row in rows
            if isinstance(row, Mapping) and row.get("attempted") is True
        ],
        "task_construction_attempt_count": sum(
            isinstance(row, Mapping)
            and row.get("attempted") is True
            and row.get("identity", (None,))[0] != "worker"
            for row in rows
        ),
        "pending_identities": [list(row) for row in expected[count:]],
        "checkpoint_stage": stage,
        "rows": rows,
        "provenance": provenance,
        "npz_path": str(npz_path),
        "npz_sha256": _sha256_file(npz_path),
        "array_hashes": {name: _array_hash(value) for name, value in arrays.items()},
    }
    _atomic_json(json_path, summary)
    pointer = {
        "generation": generation,
        "json_path": str(json_path),
        "json_sha256": _sha256_file(json_path),
        "npz_path": str(npz_path),
        "npz_sha256": summary["npz_sha256"],
    }
    _atomic_json(output.parent / f"{output.stem}.partial.latest.json", pointer)


def _publish_pre_model_checkpoint(output, provenance, extension_specs):
    expected = tuple(EXPECTED_TASK_IDENTITIES) + tuple(
        ("worker", row["module_name"]) for row in extension_specs
    )
    _checkpoint(
        output,
        [],
        expected,
        [],
        {},
        provenance,
        generation=0,
        stage="before_model_or_task_rng",
    )
    return expected


def _invoke_worker(command, *, environment=None):
    """Only subprocess boundary used by production; stdout/stderr are retained."""

    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    return {
        "command": list(command),
        "exit_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _validate_worker_boundary(row, spec, input_npz_sha256):
    """Reject fabricated, stale, or cross-module worker artifacts."""

    if not isinstance(row, Mapping) or row.get("exit_code") != 0:
        return False
    summary_path = Path(row.get("summary_path", ""))
    npz_path = Path(row.get("npz_path", ""))
    request_path = Path(row.get("request_path", ""))
    input_path = Path(row.get("input_npz_path", ""))
    if not all(path.is_file() for path in (summary_path, npz_path, request_path, input_path)):
        return False
    try:
        summary = json.loads(summary_path.read_text())
        request = json.loads(request_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            actual_hashes = {name: _array_hash(archive[name]) for name in archive.files}
    except (OSError, ValueError):
        return False
    expected_command = [
        sys.executable,
        "-B",
        "-m",
        "gato_tiago.multimodal_toll_v0_worker",
        "--request",
        str(request_path),
        "--output",
        str(summary_path),
        "--execute",
    ]
    expected_request = {
        "module_name": spec["module_name"],
        "extension_path": spec["extension_path"],
        "extension_sha256": spec["extension_sha256"],
        "input_npz": str(input_path),
        "input_npz_sha256": input_npz_sha256,
        "reference_size": spec["reference_size"],
        "protocol_version": WORKER_PROTOCOL_VERSION,
    }
    return bool(
        row.get("module_name") == spec["module_name"]
        and row.get("extension_sha256") == spec["extension_sha256"]
        and row.get("input_npz_sha256") == input_npz_sha256
        and row.get("request_sha256") == _sha256_file(request_path)
        and row.get("command") == expected_command
        and request == expected_request
        and row.get("summary_sha256") == _sha256_file(summary_path)
        and row.get("npz_sha256") == _sha256_file(npz_path)
        and summary.get("module_name") == spec["module_name"]
        and summary.get("extension_sha256") == spec["extension_sha256"]
        and summary.get("worker_protocol_version") == WORKER_PROTOCOL_VERSION
        and summary.get("reference_size") == spec["reference_size"]
        and summary.get("module_file") == str(Path(spec["extension_path"]).resolve())
        and summary.get("request_path") == str(request_path.resolve())
        and summary.get("request_sha256") == row.get("request_sha256")
        and summary.get("input_npz_path") == str(input_path.resolve())
        and summary.get("input_npz_sha256") == input_npz_sha256
        and summary.get("worker_npz_sha256") == row.get("npz_sha256")
        and summary.get("array_hashes") == actual_hashes
        and summary.get("array_names") == sorted(actual_hashes)
        and set(actual_hashes) == EXPECTED_WORKER_ARRAY_NAMES
        and summary.get("reference_smoke", {}).get(
            "all_reference_smoke_gates_pass"
        )
        is True
        and summary.get("optimization_evidence") is False
        and summary.get("sqp_optimization_calls") == 0
    )


def certify_dense_random_capture(draws, pin_states, pin_tools, worker_arrays):
    """Certify the 65-sample, one-interval random model diagnostic."""

    try:
        pin_states = np.asarray(pin_states)
        pin_tools = np.asarray(pin_tools)
        if (
            pin_states.shape != (32, 65, 14)
            or pin_states.dtype != np.float64
            or pin_tools.shape != (32, 65, 3)
            or pin_tools.dtype != np.float64
            or len(worker_arrays) != 3
        ):
            raise ValueError("dense Pin/worker shape mismatch")
        expected_time = np.arange(65, dtype=np.float64) * (toll.DT / 64)
        expected_controls = np.repeat(
            draws.dynamics_u_float32[:, None, :], 64, axis=1
        )
        cuda_states = []
        cuda_tools = []
        for row in worker_arrays:
            states = np.asarray(row["cuda_dense_states_float32"])
            tools = np.asarray(row["cuda_dense_tool_positions_float32"])
            time = np.asarray(row["dense_time_float64"])
            controls = np.asarray(row["dense_controls_float32"])
            if (
                states.shape != (32, 65, 14)
                or states.dtype != np.float32
                or tools.shape != (32, 65, 3)
                or tools.dtype != np.float32
                or time.shape != (65,)
                or time.dtype != np.float64
                or controls.shape != (32, 64, 7)
                or controls.dtype != np.float32
                or not np.array_equal(states[:, 0], draws.dynamics_x_float32)
                or not np.array_equal(time, expected_time)
                or not np.array_equal(controls, expected_controls)
            ):
                raise ValueError("dense worker boundary identity mismatch")
            cuda_states.append(states)
            cuda_tools.append(tools)
        numeric = [pin_states, pin_tools, *cuda_states, *cuda_tools]
        if not all(np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("dense capture contains nonfinite values")
        cross_exact = all(
            np.array_equal(cuda_states[0], value) for value in cuda_states[1:]
        ) and all(np.array_equal(cuda_tools[0], value) for value in cuda_tools[1:])
        state_l2 = np.linalg.norm(
            pin_states - cuda_states[0].astype(np.float64), axis=2
        )
        tool_error = np.linalg.norm(
            pin_tools - cuda_tools[0].astype(np.float64), axis=2
        )
    except (KeyError, TypeError, ValueError):
        return {
            "arrays_valid": False,
            "all_dense_random_capture_gates_pass": False,
            "diagnostic_scope": "one_interval_65_samples",
            "future_witness_scope": "full_horizon_4033_samples",
            "state_l2_is_acceptance_gate": False,
        }
    result = {
        "arrays_valid": True,
        "diagnostic_scope": "one_interval_65_samples",
        "future_witness_scope": "full_horizon_4033_samples",
        "exact_initial_control_time_identity": True,
        "cross_module_outputs_exact": cross_exact,
        "max_tool_disagreement_m": float(np.max(tool_error)),
        "max_state_l2_report_only": float(np.max(state_l2)),
        "state_l2_is_acceptance_gate": False,
        "retained_array_hashes": {
            "pin_states_float64": _array_hash(pin_states),
            "cuda_states_float32": _array_hash(cuda_states[0]),
            "pin_tool_float64": _array_hash(pin_tools),
            "cuda_tool_float32": _array_hash(cuda_tools[0]),
            "time_float64": _array_hash(expected_time),
            "controls_float32": _array_hash(expected_controls),
        },
    }
    result["all_dense_random_capture_gates_pass"] = bool(
        cross_exact and result["max_tool_disagreement_m"] <= 1e-3
    )
    return result


def _run_pipeline(
    output,
    *,
    ledger,
    task_factory: Callable,
    worker_specs: Sequence[Mapping],
    worker_launcher: Callable,
    base_arrays: Mapping,
    provenance: Mapping,
    test_override: bool,
    final_certifier: Callable | None = None,
    before_finalize: Callable | None = None,
    production_token=None,
    preexisting_checkpoint_generation=None,
):
    """Private orchestration seam; test use is permanently non-evidence."""

    output = Path(output)
    if preexisting_checkpoint_generation is None:
        _no_existing_artifacts(output)
    expected = tuple(ledger) + tuple(("worker", row["module_name"]) for row in worker_specs)
    arrays = {name: np.asarray(value) for name, value in base_arrays.items()}
    rows = []
    completed = []
    tasks = []
    worker_rows = []
    generation = (
        0
        if preexisting_checkpoint_generation is None
        else int(preexisting_checkpoint_generation)
    )
    if preexisting_checkpoint_generation is None:
        _checkpoint(
            output,
            completed,
            expected,
            rows,
            arrays,
            provenance,
            generation=generation,
            stage="runner_initialized",
        )
    try:
        for identity in ledger:
            try:
                row, task_arrays = task_factory(identity)
                if tuple(row.get("identity", ())) != tuple(identity):
                    raise RuntimeError("task factory identity/order mismatch")
            except BaseException as error:
                rows.append(
                    {
                        "identity": identity,
                        "attempted": True,
                        "passes": False,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                generation += 1
                _checkpoint(
                    output,
                    completed,
                    expected,
                    rows,
                    arrays,
                    provenance,
                    generation=generation,
                    stage="task_construction_failed",
                )
                raise
            row["attempted"] = True
            rows.append(row)
            tasks.append(row)
            for name, value in task_arrays.items():
                arrays[f"task_{len(tasks)-1:02d}_{name}"] = np.asarray(value)
            completed.append(identity)
            generation += 1
            _checkpoint(
                output,
                completed,
                expected,
                rows,
                arrays,
                provenance,
                generation=generation,
                stage="task_constructed",
            )
        for spec in worker_specs:
            row, worker_arrays = worker_launcher(spec, arrays)
            identity = ("worker", spec["module_name"])
            if tuple(row.get("identity", ())) != identity:
                raise RuntimeError("worker identity/order mismatch")
            rows.append(row)
            worker_rows.append(row)
            for name, value in worker_arrays.items():
                arrays[f"worker_{len(worker_rows)-1:02d}_{name}"] = np.asarray(value)
            completed.append(identity)
            generation += 1
            _checkpoint(
                output,
                completed,
                expected,
                rows,
                arrays,
                provenance,
                generation=generation,
                stage="worker_completed",
            )
    except BaseException:
        # The most recent completed boundary remains immutable and recoverable.
        raise

    if before_finalize is not None:
        before_finalize()
    certificate = (
        final_certifier(tasks, worker_rows, arrays)
        if final_certifier is not None
        else {"all_v0_gates_pass": False, "test_override": True}
    )
    exact = (
        len(tasks) == EXPECTED_TASK_COUNT
        and len(worker_rows) == EXPECTED_WORKER_COUNT
        and tuple(ledger) == EXPECTED_TASK_IDENTITIES
        and all(row.get("passes") is True for row in rows)
        and certificate.get("all_v0_gates_pass") is True
    )
    production_boundary = bool(
        not test_override and production_token is _PRODUCTION_PIPELINE_TOKEN
    )
    all_gates = bool(exact and production_boundary)
    npz_path = output.with_suffix(".npz")
    _atomic_npz(npz_path, arrays)
    summary = {
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "v0_protocol_version": V0_PROTOCOL_VERSION,
        "incomplete": False,
        "all_v0_gates_pass": all_gates,
        "optimization_evidence": False,
        "test_override": bool(test_override),
        "production_call_boundary_authenticated": production_boundary,
        "sqp_optimization_calls": EXPECTED_SQP_OPTIMIZATION_CALLS,
        "task_construction_calls": len(tasks),
        "worker_subprocess_calls": len(worker_rows),
        "fresh_subprocess_per_extension": len(
            {tuple(row.get("command", ())) for row in worker_rows}
        )
        == EXPECTED_WORKER_COUNT,
        "expected_identities": [list(row) for row in expected],
        "completed_identities": [list(row) for row in completed],
        "rows": rows,
        "certificate": certificate,
        "provenance": provenance,
        "npz_path": str(npz_path),
        "npz_sha256": _sha256_file(npz_path),
        "array_hashes": {name: _array_hash(value) for name, value in arrays.items()},
    }
    _atomic_json(output, summary)
    manifest = {
        "json_path": str(output),
        "json_sha256": _sha256_file(output),
        "npz_path": str(npz_path),
        "npz_sha256": summary["npz_sha256"],
        "incomplete": False,
        "supersedes_partial_generation": generation,
        "side_artifacts": [],
    }
    side_paths = {
        Path(path)
        for row in worker_rows
        for path in (
            row.get("request_path"),
            row.get("input_npz_path"),
            row.get("summary_path"),
            row.get("npz_path"),
        )
        if path
    }
    side_paths.update(output.parent.glob(f"{output.stem}.partial.*"))
    manifest["side_artifacts"] = [
        {"path": str(path), "sha256": _sha256_file(path)}
        for path in sorted(side_paths, key=str)
        if path.is_file()
    ]
    _atomic_json(output.with_suffix(".manifest.json"), manifest)
    return summary


def _production_pipeline(output):  # pragma: no cover - execution is blocked
    """Future source-matched implementation; reachable only after a new audit."""

    # Imports that instantiate the model or open frozen task RNG are deliberately
    # inside this guarded function, never module import or static describe paths.
    import pinocchio as pin
    from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS
    from gato_tiago.multimodal_pillar import load_model, tool_position
    from gato_tiago import multimodal_toll_oracle_schema as oracle

    output = Path(output).resolve()
    _no_existing_artifacts(output)
    repo = repository_root()
    head_start = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    tracked_status_start = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    ).splitlines()
    if tracked_status_start:
        raise RuntimeError("tracked tree must be clean before any V0 model/task action")
    full_status_start = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).splitlines()
    source_hashes = {
        name: {"path": relative, "sha256": _sha256_file(repo / relative)}
        for name, relative in REQUIRED_SOURCE_PATHS.items()
    }
    extension_specs = [dict(row) for row in FROZEN_EXTENSIONS]
    for spec in extension_specs:
        spec["extension_path"] = str((repo / spec["extension_path"]).resolve())
        if _sha256_file(spec["extension_path"]) != spec["extension_sha256"]:
            raise RuntimeError("frozen extension hash mismatch before model/task action")

    provenance = {
        "git_head_at_start": head_start,
        "git_head_at_end": head_start,
        "tracked_tree_clean_at_start": True,
        "tracked_tree_clean_at_end": True,
        "full_git_status_at_start": full_status_start,
        "full_git_status_at_end": full_status_start,
        "source_hashes": source_hashes,
        "model_path": str(toll.MODEL_PATH),
        "model_sha256": _sha256_file(repo / toll.MODEL_PATH),
        "extension_sha256": extension_specs[0]["extension_sha256"],
        "extension_hashes": {
            row["module_name"]: row["extension_sha256"] for row in extension_specs
        },
        "extension_source_commits": {
            row["module_name"].split(".")[-1]: "9bb1ceaf64787597f1ea7df5566f84d0635c4b7f"
            for row in extension_specs
        },
        **exact_invocation_provenance(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pinocchio_version": pin.__version__,
        "retry_count": 0,
        "replacement_count": 0,
        "resume_count": 0,
    }
    expected = _publish_pre_model_checkpoint(output, provenance, extension_specs)

    model = load_model(toll.MODEL_PATH)
    comfortable = np.asarray(
        TIAGO_RIGHT_START_CONFIGS["comfortable_high_clearance"], dtype=np.float64
    )
    effort = np.asarray(model.effortLimit, dtype=np.float64)
    model_arrays = {
        "model_lower_float64": np.asarray(model.lowerPositionLimit, dtype=np.float64),
        "model_upper_float64": np.asarray(model.upperPositionLimit, dtype=np.float64),
        "model_effort_float64": effort,
        "comfortable_q_float64": comfortable,
    }
    _checkpoint(
        output,
        [],
        expected,
        [],
        model_arrays,
        provenance,
        generation=1,
        stage="model_loaded",
    )
    draws = generate_v0_draws(
        np.asarray(model.lowerPositionLimit, dtype=np.float64),
        np.asarray(model.upperPositionLimit, dtype=np.float64),
        effort,
        comfortable,
    )
    draw_arrays = {**model_arrays, **draws.__dict__}
    _checkpoint(
        output,
        [],
        expected,
        [],
        draw_arrays,
        provenance,
        generation=2,
        stage="frozen_draws_generated",
    )
    data = model.createData()
    pin_fk = np.stack([tool_position(model, data, q) for q in draws.fk_q_float64])
    pin_qdd = np.stack(
        [pin.aba(model, data, q, qd, u) for q, qd, u in zip(
            draws.dynamics_q_float64, draws.dynamics_qd_float64, draws.dynamics_u_float64
        )]
    )
    pin_next = np.stack(
        [constant_acceleration_step(x, qdd) for x, qdd in zip(
            np.concatenate([draws.dynamics_q_float64, draws.dynamics_qd_float64], axis=1),
            pin_qdd,
        )]
    )
    dense_pin_states = np.empty((32, 65, 14), dtype=np.float64)
    dense_pin_tools = np.empty((32, 65, 3), dtype=np.float64)
    dense_pin_states[:, 0] = np.concatenate(
        [draws.dynamics_q_float64, draws.dynamics_qd_float64], axis=1
    )
    dense_pin_tools[:, 0] = np.stack(
        [tool_position(model, data, x[:7]) for x in dense_pin_states[:, 0]]
    )
    for step in range(64):
        for index in range(32):
            x = dense_pin_states[index, step]
            qdd = pin.aba(model, data, x[:7], x[7:], draws.dynamics_u_float64[index])
            dense_pin_states[index, step + 1] = constant_acceleration_step(
                x, qdd, toll.DT / 64
            )
            dense_pin_tools[index, step + 1] = tool_position(
                model, data, dense_pin_states[index, step + 1, :7]
            )
    base_arrays = {
        **draw_arrays,
        "pin_fk_positions_float64": pin_fk,
        "pin_qdd_float64": pin_qdd,
        "pin_next_states_float64": pin_next,
        "pin_dense_states_float64": dense_pin_states,
        "pin_dense_tool_positions_float64": dense_pin_tools,
    }
    _checkpoint(
        output,
        [],
        expected,
        [],
        base_arrays,
        provenance,
        generation=3,
        stage="pin_preflight_captured",
    )

    def task_factory(identity):
        phase, task_seed = identity
        task, witness = oracle.generate_task(
            task_seed, model, authorization=oracle.TASK_CONSTRUCTION_AUTHORIZATION
        )
        q0 = comfortable + np.asarray(witness.q0_jitter, dtype=np.float64)
        q_goal = np.asarray(witness.q_goal, dtype=np.float64)
        pin.computeJointJacobians(model, data, q0)
        pin.updateFramePlacements(model, data)
        frame = model.getFrameId(oracle.TOOL_FRAME)
        jacobian = pin.getFrameJacobian(
            model, data, frame, pin.LOCAL_WORLD_ALIGNED
        )[:3].copy()
        arrays = {
            "solver_x0": np.asarray(task.x0, dtype=np.float32),
            "solver_reference": task.reference.as_float32(),
            "q_goal_float32": q_goal.astype(np.float32),
            "q_goal_float64": q_goal,
            "dq_float64": np.asarray(witness.dq),
            "requested_delta_xyz_float64": np.asarray(witness.requested_delta_xyz),
            "pin_position_jacobian_float64": jacobian,
            "q0_jitter_float64": np.asarray(witness.q0_jitter),
            "pin_start_tool_xyz": tool_position(model, data, q0),
            "pin_q_goal_tool_xyz": tool_position(model, data, q_goal),
        }
        row = {
            "identity": identity,
            "phase": phase,
            "task_seed": task_seed,
            "phi_float64": witness.phi,
            "offset_sign": witness.offset_sign,
            "default_side": task.default_side,
            "retry_count": 0,
            "replacement_count": 0,
            "passes": True,
            "solver_x0_sha256": task.solver_x0_sha256,
            "solver_reference_sha256": task.solver_reference_sha256,
            "q_goal_sha256": _array_hash(q_goal.astype(np.float32)),
            "construction_draws_sha256": _array_hash(
                np.concatenate(
                    [np.asarray(witness.q0_jitter), [witness.phi, witness.offset_sign]]
                )
            ),
        }
        return row, arrays

    input_npz = Path(output).with_name(f"{Path(output).stem}.worker-input.npz")
    input_written = False

    def worker_launcher(spec, arrays):
        nonlocal input_written
        if not input_written:
            task_q = np.concatenate(
                [
                    np.stack([arrays[f"task_{index:02d}_solver_x0"][:7] for index in range(12)]),
                    np.stack([arrays[f"task_{index:02d}_q_goal_float32"] for index in range(12)]),
                ]
            ).astype(np.float32)
            _atomic_npz(
                input_npz,
                {
                    "fk_q_float32": arrays["fk_q_float32"],
                    "dynamics_x_float32": arrays["dynamics_x_float32"],
                    "dynamics_u_float32": arrays["dynamics_u_float32"],
                    "task_q_float32": task_q,
                },
            )
            input_written = True
        input_hash = _sha256_file(input_npz)
        module_leaf = spec["module_name"].split(".")[-1]
        request_path = Path(output).with_name(f"{Path(output).stem}.{module_leaf}.request.json")
        worker_output = Path(output).with_name(f"{Path(output).stem}.{module_leaf}.json")
        request = {
            "module_name": spec["module_name"],
            "extension_path": spec["extension_path"],
            "extension_sha256": spec["extension_sha256"],
            "input_npz": str(input_npz),
            "input_npz_sha256": input_hash,
            "reference_size": spec["reference_size"],
            "protocol_version": WORKER_PROTOCOL_VERSION,
        }
        _atomic_json(request_path, request)
        command = [
            sys.executable,
            "-B",
            "-m",
            "gato_tiago.multimodal_toll_v0_worker",
            "--request",
            str(request_path),
            "--output",
            str(worker_output),
            "--execute",
        ]
        process = _invoke_worker(
            command,
            environment={**os.environ, "PYTHONPATH": f"{repo / 'tiago_src'}:{repo / 'python'}"},
        )
        worker_npz = worker_output.with_suffix(".npz")
        row = {
            "identity": ("worker", spec["module_name"]),
            "module_name": spec["module_name"],
            "extension_sha256": spec["extension_sha256"],
            "input_npz_sha256": input_hash,
            "input_npz_path": str(input_npz.resolve()),
            "request_path": str(request_path.resolve()),
            "request_sha256": _sha256_file(request_path),
            "summary_path": str(worker_output),
            "summary_sha256": _sha256_file(worker_output) if worker_output.is_file() else None,
            "npz_path": str(worker_npz),
            "npz_sha256": _sha256_file(worker_npz) if worker_npz.is_file() else None,
            **process,
        }
        row["passes"] = _validate_worker_boundary(row, spec, input_hash)
        worker_arrays = {}
        if row["passes"]:
            with np.load(worker_npz, allow_pickle=False) as archive:
                worker_arrays = {name: archive[name] for name in archive.files}
        return row, worker_arrays

    def final_certifier(task_rows, worker_rows, arrays):
        toll_prefix = "worker_00_"
        cuda_tasks = arrays[f"{toll_prefix}cuda_task_fk_b1_float32"]
        if not np.array_equal(
            cuda_tasks, arrays[f"{toll_prefix}cuda_task_fk_b16_float32"]
        ):
            return {"all_v0_gates_pass": False, "task_fk_b1_b16_exact": False}
        certificate_rows = []
        for index, metadata in enumerate(task_rows):
            prefix = f"task_{index:02d}_"
            row = dict(metadata)
            row.update(
                {
                    name: arrays[f"{prefix}{name}"]
                    for name in (
                        "solver_x0",
                        "solver_reference",
                        "q_goal_float32",
                        "q_goal_float64",
                        "dq_float64",
                        "requested_delta_xyz_float64",
                        "pin_position_jacobian_float64",
                        "q0_jitter_float64",
                        "pin_start_tool_xyz",
                        "pin_q_goal_tool_xyz",
                    )
                }
            )
            row["cuda_start_tool_xyz"] = cuda_tasks[index]
            row["cuda_q_goal_tool_xyz"] = cuda_tasks[index + 12]
            certificate_rows.append(row)
        manifest = {}
        for index, (spec, process_row) in enumerate(zip(extension_specs, worker_rows)):
            prefix = f"worker_{index:02d}_"
            fk1 = arrays[f"{prefix}cuda_fk_b1_float32"]
            fk16 = arrays[f"{prefix}cuda_fk_b16_float32"]
            summary = json.loads(Path(process_row["summary_path"]).read_text())
            smoke = summary["reference_smoke"]
            manifest[spec["module_name"].split(".")[-1]] = {
                "knots": 64,
                "reference_size": spec["reference_size"],
                "tool_position": True,
                "tool_position_frame": "arm_right_tool_joint_origin",
                "extension_sha256": spec["extension_sha256"],
                "extension_path": spec["extension_path"],
                "import_smoke_pass": True,
                "native_reference_shape_accepted": smoke["all_reference_smoke_gates_pass"],
                "wrong_reference_shape_rejected": smoke["wrong_width_rejected_before_native_kernel"],
                "accepted_reference_width": spec["reference_size"],
                "rejected_reference_width": 6 if spec["reference_size"] == 10 else 10,
                "b1_reference_cost_output_parity": smoke["initial_final_merit_tolerance"],
                "b16_reference_cost_output_parity": smoke["initial_final_merit_tolerance"],
                "b1_broadcast_fk_parity": np.array_equal(
                    arrays[f"{prefix}cuda_fk_b1_broadcast_float32"], fk1[:1]
                ),
                "b16_broadcast_fk_parity": bool(
                    np.all(arrays[f"{prefix}cuda_fk_b16_broadcast_float32"] == fk16[0])
                ),
                "b16_per_lane_fk_parity": np.array_equal(fk1, fk16),
                "pin_fk_max_error_m": float(
                    np.max(np.linalg.norm(pin_fk - fk1.astype(np.float64), axis=1))
                ),
                "cuda_reference_smoke_pass": smoke["all_reference_smoke_gates_pass"],
                "build_command": spec["build_command"],
                "test_command": "python -B -m gato_tiago.multimodal_toll_v0_worker --execute",
                "source_commit": provenance["extension_source_commits"][
                    spec["module_name"].split(".")[-1]
                ],
            }
        worker_input_and_model_parity = True
        for index in range(3):
            prefix = f"worker_{index:02d}_"
            worker_input_and_model_parity = bool(
                worker_input_and_model_parity
                and np.array_equal(arrays[f"{prefix}captured_fk_q_float32"], draws.fk_q_float32)
                and np.array_equal(arrays[f"{prefix}captured_dynamics_x_float32"], draws.dynamics_x_float32)
                and np.array_equal(arrays[f"{prefix}captured_dynamics_u_float32"], draws.dynamics_u_float32)
                and np.array_equal(arrays[f"{prefix}cuda_fk_b1_float32"], arrays[f"{prefix}cuda_fk_b16_float32"])
                and np.array_equal(arrays[f"{prefix}cuda_next_b1_float32"], arrays[f"{prefix}cuda_next_b16_float32"])
            )
        dense_capture = certify_dense_random_capture(
            draws,
            dense_pin_states,
            dense_pin_tools,
            [
                {
                    name: arrays[f"worker_{index:02d}_{name}"]
                    for name in (
                        "cuda_dense_states_float32",
                        "cuda_dense_tool_positions_float32",
                        "dense_time_float64",
                        "dense_controls_float32",
                    )
                }
                for index in range(3)
            ],
        )
        result = certify_v0(
            draws=draws,
            captured_fk_q_float32=arrays[f"worker_00_captured_fk_q_float32"],
            captured_dynamics_x_float32=arrays[f"worker_00_captured_dynamics_x_float32"],
            captured_dynamics_u_float32=arrays[f"worker_00_captured_dynamics_u_float32"],
            pin_fk_positions=pin_fk,
            cuda_fk_positions=arrays[f"worker_00_cuda_fk_b1_float32"],
            pin_qdd_float64=pin_qdd,
            pin_next_states=pin_next,
            cuda_next_states=arrays[f"worker_00_cuda_next_b1_float32"],
            task_rows=certificate_rows,
            extension_manifest=manifest,
            provenance=provenance,
        )
        result["worker_boundaries_pass"] = all(row["passes"] for row in worker_rows)
        result["task_fk_b1_b16_exact"] = True
        result["all_worker_input_and_b1_b16_model_parity"] = worker_input_and_model_parity
        result["dense_random_capture"] = dense_capture
        result["runner_provenance"] = certify_runner_provenance(provenance, worker_rows)
        result["all_v0_gates_pass"] = bool(
            result["all_v0_gates_pass"]
            and result["worker_boundaries_pass"]
            and worker_input_and_model_parity
            and dense_capture["all_dense_random_capture_gates_pass"]
            and result["runner_provenance"]["all_runner_provenance_gates_pass"]
        )
        return result

    def capture_end_provenance():
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

    result = _run_pipeline(
        output,
        ledger=EXPECTED_TASK_IDENTITIES,
        task_factory=task_factory,
        worker_specs=extension_specs,
        worker_launcher=worker_launcher,
        base_arrays=base_arrays,
        provenance=provenance,
        test_override=False,
        final_certifier=final_certifier,
        before_finalize=capture_end_provenance,
        production_token=_PRODUCTION_PIPELINE_TOKEN,
        preexisting_checkpoint_generation=3,
    )
    return result


def execute_v0_runner(output, *, authorization=None):
    """Public entry point.  It accepts no task rows, evidence arrays, or booleans."""

    if (
        RUNNER_EXECUTION_AUTHORIZATION is None
        or authorization is not RUNNER_EXECUTION_AUTHORIZATION
    ):
        raise RuntimeError("Stage V0 runner execution is blocked")
    if Path(output).resolve() != AUTHORIZED_OUTPUT_PATH:
        raise RuntimeError("Stage V0 runner output is not the single authorized path")
    return _production_pipeline(output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.execute or RUNNER_EXECUTION_AUTHORIZATION is None:
        raise SystemExit("Stage V0 runner execution is blocked")
    execute_v0_runner(args.output, authorization=RUNNER_EXECUTION_AUTHORIZATION)
    return 0


if __name__ == "__main__":
    main()
