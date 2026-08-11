"""Transactional runner for the isolated CUDA L2 portability smoke."""

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

import numpy as np

from gato_tiago.l2_portability_smoke import (
    AUTHORIZED_OUTPUT_PATH,
    AUTHORIZED_CWD,
    AUTHORIZED_ORIG_ARGV,
    EXPECTED_CONSTRUCTOR_CALLS_PER_MODULE,
    EXPECTED_FK_CALLS_PER_MODULE,
    EXPECTED_GLOBAL_CONSTRUCTOR_CALLS,
    EXPECTED_GLOBAL_FK_CALLS,
    EXPECTED_WORKER_ARRAY_NAMES,
    FORBIDDEN_CALL_COUNTS,
    FROZEN_BUILD_HEAD,
    FROZEN_BUILD_SOURCE_HASHES,
    FROZEN_CUDA_ARCH,
    FROZEN_MODULES,
    PROTOCOL_VERSION,
    REQUIRED_SOURCE_PATHS,
    frozen_metadata,
)
from gato_tiago.l2_portability_smoke_worker import (
    WORKER_EXECUTION_AUTHORIZATION,
    WORKER_PROTOCOL_VERSION,
    array_hash,
    certify_arrays,
)


RUNNER_EXECUTION_AUTHORIZATION = None
RUNNER_PROTOCOL_VERSION = PROTOCOL_VERSION + "_runner_1"
_PRODUCTION_TOKEN = object()


def repository_root(module_path=Path(__file__)):
    root = Path(module_path).resolve().parents[2]
    if not (root / ".git").exists() or not (root / "CMakeLists.txt").is_file():
        raise RuntimeError("L2 portability smoke repository root mismatch")
    return root


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _no_existing(output):
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
        raise RuntimeError("L2 portability smoke refuses overwrite, resume, or rerun")


def _checkpoint(output, generation, rows, arrays, provenance):
    output = Path(output)
    npz = output.with_name(f"{output.stem}.partial.{generation:04d}.npz")
    summary_path = output.with_name(
        f"{output.stem}.partial.{generation:04d}.json"
    )
    retained = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    _atomic_npz(npz, retained)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "incomplete": True,
        "all_smoke_gates_pass": False,
        "generation": generation,
        "completed_worker_count": len(rows),
        "pending_modules": [
            row["module_name"] for row in FROZEN_MODULES[len(rows) :]
        ],
        "rows": rows,
        "array_names": sorted(retained),
        "array_hashes": {name: array_hash(value) for name, value in retained.items()},
        "npz_path": str(npz),
        "npz_sha256": sha256_file(npz),
        "provenance": provenance,
        "optimization_evidence": False,
        "timing_evidence": False,
    }
    _atomic_json(summary_path, summary)
    pointer = {
        "generation": generation,
        "json_path": str(summary_path),
        "json_sha256": sha256_file(summary_path),
        "npz_path": str(npz),
        "npz_sha256": summary["npz_sha256"],
    }
    _atomic_json(output.with_name(f"{output.stem}.partial.latest.json"), pointer)


def _source_provenance(repo):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    ).splitlines()
    if tracked:
        raise RuntimeError("L2 portability smoke requires a tracked-clean tree")
    source_hashes = {
        name: {"path": path, "sha256": sha256_file(repo / path)}
        for name, path in REQUIRED_SOURCE_PATHS.items()
    }
    build_hashes = {
        path: sha256_file(repo / path) for path in FROZEN_BUILD_SOURCE_HASHES
    }
    if build_hashes != FROZEN_BUILD_SOURCE_HASHES:
        raise RuntimeError("L2 portability smoke build-source hash mismatch")
    return {
        "git_head_at_start": head,
        "git_head_at_end": None,
        "tracked_clean_at_start": True,
        "tracked_clean_at_end": None,
        "full_status_at_start": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).splitlines(),
        "full_status_at_end": None,
        "exact_command": shlex.join(list(sys.orig_argv)),
        "orig_argv": list(sys.orig_argv),
        "cwd": str(Path.cwd().resolve()),
        "source_hashes_at_start": source_hashes,
        "source_hashes_at_end": None,
        "frozen_build_head": FROZEN_BUILD_HEAD,
        "frozen_build_source_hashes": build_hashes,
        "frozen_cuda_arch": FROZEN_CUDA_ARCH,
        "extension_hashes": {
            row["module_name"]: row["extension_sha256"] for row in FROZEN_MODULES
        },
        "extension_sizes": {
            row["module_name"]: row["extension_size_bytes"]
            for row in FROZEN_MODULES
        },
        "forbidden_call_counts": dict(FORBIDDEN_CALL_COUNTS),
    }


def certify_smoke(rows, worker_arrays, provenance):
    if len(rows) != 3 or [row.get("module_name") for row in rows] != [
        row["module_name"] for row in FROZEN_MODULES
    ]:
        return {"all_smoke_gates_pass": False}
    reports = {}
    for row in rows:
        name = row["module_name"]
        arrays = worker_arrays.get(name, {})
        independent = certify_arrays(arrays)
        reports[name] = {
            "worker_boundary_pass": row.get("passes") is True,
            "arrays_independently_certified": independent.get(
                "all_worker_array_gates_pass"
            )
            is True,
            "constructor_calls_exact": row.get("constructor_calls")
            == EXPECTED_CONSTRUCTOR_CALLS_PER_MODULE,
            "fk_calls_exact": row.get("fk_calls") == EXPECTED_FK_CALLS_PER_MODULE,
            "forbidden_calls_zero": row.get("forbidden_call_counts")
            == FORBIDDEN_CALL_COUNTS,
        }
        reports[name]["all_module_gates_pass"] = all(reports[name].values())
    cross = False
    if set(worker_arrays) == {row["module_name"] for row in FROZEN_MODULES}:
        first = worker_arrays[FROZEN_MODULES[0]["module_name"]]
        cross = all(
            np.array_equal(
                first["fk_b1_float32"],
                worker_arrays[row["module_name"]]["fk_b1_float32"],
            )
            and np.array_equal(
                first["fk_b16_float32"],
                worker_arrays[row["module_name"]]["fk_b16_float32"],
            )
            for row in FROZEN_MODULES[1:]
        )
    constructor_count = sum(row.get("constructor_calls", -1000) for row in rows)
    fk_count = sum(row.get("fk_calls", -1000) for row in rows)
    start_head = provenance.get("git_head_at_start")
    end_head = provenance.get("git_head_at_end")
    source_start = provenance.get("source_hashes_at_start", {})
    source_end = provenance.get("source_hashes_at_end", {})
    provenance_pass = bool(
        isinstance(start_head, str)
        and len(start_head) == 40
        and all(character in "0123456789abcdef" for character in start_head)
        and start_head == end_head
        and provenance.get("tracked_clean_at_start") is True
        and provenance.get("tracked_clean_at_end") is True
        and set(source_start) == set(REQUIRED_SOURCE_PATHS)
        and source_start == source_end
        and all(
            row.get("path") == REQUIRED_SOURCE_PATHS[name]
            and isinstance(row.get("sha256"), str)
            and len(row["sha256"]) == 64
            and all(character in "0123456789abcdef" for character in row["sha256"])
            for name, row in source_start.items()
        )
        and provenance.get("cwd") == str(AUTHORIZED_CWD)
        and provenance.get("orig_argv") == list(AUTHORIZED_ORIG_ARGV)
        and provenance.get("exact_command") == shlex.join(AUTHORIZED_ORIG_ARGV)
        and provenance.get("frozen_build_head") == FROZEN_BUILD_HEAD
        and provenance.get("frozen_build_source_hashes") == FROZEN_BUILD_SOURCE_HASHES
        and provenance.get("frozen_cuda_arch") == FROZEN_CUDA_ARCH
        and provenance.get("extension_hashes")
        == {row["module_name"]: row["extension_sha256"] for row in FROZEN_MODULES}
        and provenance.get("extension_sizes")
        == {
            row["module_name"]: row["extension_size_bytes"]
            for row in FROZEN_MODULES
        }
        and provenance.get("forbidden_call_counts") == FORBIDDEN_CALL_COUNTS
    )
    gates = {
        "all_modules_pass": all(row["all_module_gates_pass"] for row in reports.values()),
        "cross_module_fk_bitwise_equal": cross,
        "global_constructor_calls_exact": constructor_count
        == EXPECTED_GLOBAL_CONSTRUCTOR_CALLS,
        "global_fk_calls_exact": fk_count == EXPECTED_GLOBAL_FK_CALLS,
        "provenance_pass": provenance_pass,
        "optimization_evidence": False,
        "timing_evidence": False,
    }
    return {
        "per_module": reports,
        "constructor_calls": constructor_count,
        "fk_calls": fk_count,
        **gates,
        "all_smoke_gates_pass": bool(
            all(
                value
                for key, value in gates.items()
                if key not in {"optimization_evidence", "timing_evidence"}
            )
        ),
    }


def validate_worker_boundary(row, spec, request, summary, arrays):
    """Authenticate one exact subprocess request/output boundary."""

    try:
        leaf = spec["module_name"].split(".")[-1]
        request_path = AUTHORIZED_OUTPUT_PATH.parent / f"smoke.{leaf}.request.json"
        summary_path = AUTHORIZED_OUTPUT_PATH.parent / f"smoke.{leaf}.json"
        npz_path = summary_path.with_suffix(".npz")
        expected_extension = (
            repository_root() / spec["extension_path"]
        ).resolve()
        expected_command = [
            sys.executable,
            "-B",
            "-m",
            "gato_tiago.l2_portability_smoke_worker",
            "--request",
            str(request_path),
            "--output",
            str(summary_path),
            "--execute",
        ]
        hashes = {name: array_hash(value) for name, value in arrays.items()}
        return bool(
            row.get("exit_code") == 0
            and row.get("module_name") == spec["module_name"]
            and row.get("command") == expected_command
            and Path(row.get("request_path", "")).resolve() == request_path
            and Path(row.get("summary_path", "")).resolve() == summary_path
            and Path(row.get("npz_path", "")).resolve() == npz_path
            and row.get("request_sha256") == sha256_file(request_path)
            and row.get("summary_sha256") == sha256_file(summary_path)
            and row.get("npz_sha256") == sha256_file(npz_path)
            and request
            == {
                "module_name": spec["module_name"],
                "extension_path": str(expected_extension),
                "extension_sha256": spec["extension_sha256"],
                "extension_size_bytes": spec["extension_size_bytes"],
                "cuda_arch": FROZEN_CUDA_ARCH,
                "protocol_version": WORKER_PROTOCOL_VERSION,
            }
            and summary.get("protocol_version") == PROTOCOL_VERSION
            and summary.get("worker_protocol_version") == WORKER_PROTOCOL_VERSION
            and summary.get("module_name") == spec["module_name"]
            and Path(summary.get("module_file", "")).resolve()
            == expected_extension
            and summary.get("extension_sha256") == spec["extension_sha256"]
            and summary.get("extension_size_bytes")
            == spec["extension_size_bytes"]
            and summary.get("cuda_arch") == FROZEN_CUDA_ARCH
            and summary.get("KNOT_POINTS") == 64
            and summary.get("REFERENCE_SIZE") == spec["reference_size"]
            and summary.get("TOOL_POSITION_FRAME")
            == "arm_right_tool_joint_origin"
            and summary.get("TOOL_POSITION_SIZE") == 3
            and Path(summary.get("request_path", "")).resolve() == request_path
            and summary.get("request_sha256") == row.get("request_sha256")
            and summary.get("constructor_calls")
            == EXPECTED_CONSTRUCTOR_CALLS_PER_MODULE
            and summary.get("fk_calls") == EXPECTED_FK_CALLS_PER_MODULE
            and summary.get("forbidden_call_counts") == FORBIDDEN_CALL_COUNTS
            and summary.get("certificate", {}).get("all_worker_array_gates_pass")
            is True
            and Path(summary.get("npz_path", "")).resolve() == npz_path
            and summary.get("npz_sha256") == row.get("npz_sha256")
            and set(summary.get("array_names", ())) == EXPECTED_WORKER_ARRAY_NAMES
            and set(arrays) == EXPECTED_WORKER_ARRAY_NAMES
            and summary.get("array_hashes") == hashes
            and summary.get("optimization_evidence") is False
            and summary.get("timing_evidence") is False
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False
def _run_pipeline(output, provenance, launcher, *, test_override=False, finish=None):
    output = Path(output)
    _no_existing(output)
    rows = []
    retained = {}
    worker_arrays = {}
    _checkpoint(output, 0, rows, retained, provenance)
    for generation, spec in enumerate(FROZEN_MODULES, start=1):
        try:
            row, arrays = launcher(spec)
            row = dict(row)
            if row.get("module_name") != spec["module_name"]:
                raise RuntimeError("L2 portability worker identity mismatch")
            worker_arrays[spec["module_name"]] = arrays
            leaf = spec["module_name"].split(".")[-1]
            retained.update(
                {f"worker_{leaf}_{name}": value for name, value in arrays.items()}
            )
        except BaseException as error:
            row = {
                "module_name": spec["module_name"],
                "passes": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
        rows.append(row)
        _checkpoint(output, generation, rows, retained, provenance)
    if finish is not None:
        finish(provenance)
    _checkpoint(output, 4, rows, retained, provenance)
    certificate = certify_smoke(rows, worker_arrays, provenance)
    if not test_override and not certificate["all_smoke_gates_pass"]:
        raise RuntimeError("L2 portability smoke failed; retained partial closes run")
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "incomplete": False,
        "rows": rows,
        "certificate": certificate,
        "all_smoke_gates_pass": bool(
            certificate["all_smoke_gates_pass"] and not test_override
        ),
        "provenance": provenance,
        "optimization_evidence": False,
        "timing_evidence": False,
    }
    npz = output.with_suffix(".npz")
    _atomic_npz(npz, retained)
    summary.update(
        {
            "npz_path": str(npz),
            "npz_sha256": sha256_file(npz),
            "array_names": sorted(retained),
            "array_hashes": {name: array_hash(value) for name, value in retained.items()},
        }
    )
    _atomic_json(output, summary)
    side = set(output.parent.glob(f"{output.stem}.partial.*"))
    for row in rows:
        for key in ("request_path", "summary_path", "npz_path"):
            if row.get(key):
                side.add(Path(row[key]))
    manifest = {
        "json_path": str(output),
        "json_sha256": sha256_file(output),
        "npz_path": str(npz),
        "npz_sha256": summary["npz_sha256"],
        "all_smoke_gates_pass": summary["all_smoke_gates_pass"],
        "side_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(side, key=str)
            if path.is_file()
        ],
    }
    _atomic_json(output.with_suffix(".manifest.json"), manifest)
    return summary


def _production_pipeline(output, *, token=None):  # pragma: no cover
    if token is not _PRODUCTION_TOKEN:
        raise RuntimeError("L2 portability production pipeline is private")
    repo = repository_root()
    provenance = _source_provenance(repo)

    def launcher(spec):
        extension = (repo / spec["extension_path"]).resolve()
        if (
            not extension.is_file()
            or sha256_file(extension) != spec["extension_sha256"]
            or extension.stat().st_size != spec["extension_size_bytes"]
        ):
            raise RuntimeError("L2 portability frozen extension hash mismatch")
        leaf = spec["module_name"].split(".")[-1]
        request_path = Path(output).with_name(f"{Path(output).stem}.{leaf}.request.json")
        worker_output = Path(output).with_name(f"{Path(output).stem}.{leaf}.json")
        request = {
            "module_name": spec["module_name"],
            "extension_path": str(extension),
            "extension_sha256": spec["extension_sha256"],
            "extension_size_bytes": spec["extension_size_bytes"],
            "cuda_arch": FROZEN_CUDA_ARCH,
            "protocol_version": WORKER_PROTOCOL_VERSION,
        }
        _atomic_json(request_path, request)
        command = [
            sys.executable,
            "-B",
            "-m",
            "gato_tiago.l2_portability_smoke_worker",
            "--request",
            str(request_path),
            "--output",
            str(worker_output),
            "--execute",
        ]
        process = subprocess.run(
            command,
            cwd=repo,
            env={**os.environ, "PYTHONPATH": f"{repo / 'tiago_src'}:{repo / 'python'}"},
            capture_output=True,
            text=True,
            check=False,
        )
        worker_npz = worker_output.with_suffix(".npz")
        row = {
            "module_name": spec["module_name"],
            "command": command,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "exit_code": process.returncode,
            "request_path": str(request_path),
            "request_sha256": sha256_file(request_path),
            "summary_path": str(worker_output),
            "summary_sha256": sha256_file(worker_output) if worker_output.is_file() else None,
            "npz_path": str(worker_npz),
            "npz_sha256": sha256_file(worker_npz) if worker_npz.is_file() else None,
        }
        arrays = {}
        if process.returncode == 0 and worker_output.is_file() and worker_npz.is_file():
            worker_summary = json.loads(worker_output.read_text())
            with np.load(worker_npz, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            row.update(
                {
                    "constructor_calls": worker_summary["constructor_calls"],
                    "fk_calls": worker_summary["fk_calls"],
                    "forbidden_call_counts": worker_summary["forbidden_call_counts"],
                }
            )
            row["passes"] = validate_worker_boundary(
                row, spec, request, worker_summary, arrays
            )
        else:
            row["passes"] = False
        return row, arrays

    def finish(value):
        value["git_head_at_end"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        value["tracked_clean_at_end"] = not subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            text=True,
        ).splitlines()
        value["full_status_at_end"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).splitlines()
        value["source_hashes_at_end"] = {
            name: {"path": path, "sha256": sha256_file(repo / path)}
            for name, path in REQUIRED_SOURCE_PATHS.items()
        }

    return _run_pipeline(output, provenance, launcher, finish=finish)


def describe_smoke():
    return {
        **frozen_metadata(),
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "runner_execution_authorized": RUNNER_EXECUTION_AUTHORIZATION is not None,
        "worker_execution_authorized": WORKER_EXECUTION_AUTHORIZATION is not None,
        "transaction_generations": 5,
    }


def execute_smoke(output, *, authorization=None):
    if (
        RUNNER_EXECUTION_AUTHORIZATION is None
        or authorization is not RUNNER_EXECUTION_AUTHORIZATION
    ):
        raise RuntimeError("L2 portability smoke execution is blocked")
    if Path(output).resolve() != AUTHORIZED_OUTPUT_PATH.resolve():
        raise RuntimeError("L2 portability smoke output path is not authorized")
    return _production_pipeline(output, token=_PRODUCTION_TOKEN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.describe and not args.execute:
        print(json.dumps(describe_smoke(), indent=2, sort_keys=True))
        return 0
    if not args.execute or RUNNER_EXECUTION_AUTHORIZATION is None:
        raise SystemExit("L2 portability smoke execution is blocked")
    execute_smoke(args.output, authorization=RUNNER_EXECUTION_AUTHORIZATION)
    return 0


if __name__ == "__main__":
    main()
