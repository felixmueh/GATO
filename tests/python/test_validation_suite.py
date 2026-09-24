"""CPU-only orchestration regressions; no CUDA module or subprocess is run."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def suite(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "examples/randomized_multimodal/validation_suite.py"
    spec = importlib.util.spec_from_file_location("validation_suite_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "repo"
    binaries = root / "python/bsqp"
    binaries.mkdir(parents=True)
    # Deliberately leave an old binary: a failed rebuild must not run it.
    (binaries / "bsqpN16_tiago_right_multimodal.so").write_bytes(b"old binary")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "source_hashes", lambda: {"solver.cuh": "fixed-source"})
    monkeypatch.setattr(module, "environment", lambda: {
        "python": "test-python", "executable": sys.executable, "packages": {},
        "gpu": {"stdout": "mock device"}, "nvcc": {"stdout": "mock compiler"},
    })
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "schema_version": 1, "seed": 1401,
        "build": {"knots": [16], "architecture": "native", "jobs": 1},
        "solver": {"iterations": 2, "passes": 1, "rho": .01},
        "tasks": [0], "batches": [1],
        "studies": {"discretization": {"durations": [.45], "knots": [16]}},
    }))
    output = tmp_path / "run"

    def invoke(*options):
        monkeypatch.setattr(sys, "argv", [str(source), "--protocol", str(protocol),
                            "--output", str(output), "--build-dir", str(tmp_path / "build"),
                            *options])
        return module.main()

    return module, invoke, output


def write_child_result(command, *, failure=False):
    folder = Path(command[command.index("--output") + 1])
    cell = folder / "N16_T0.45_task0"
    cell.mkdir(parents=True, exist_ok=True)
    value = {"failure": "deliberate solver exception"} if failure else {"branches": {"1": {}}}
    (cell / "summary.json").write_text(json.dumps(value))
    if not failure:
        # The coordinator treats NPZ payloads as opaque artifacts, as intended.
        (cell / "b1.npz").write_bytes(b"recorded trajectory bytes")
    (folder / "index.json").write_text(json.dumps([value]))


def test_failed_rebuild_never_runs_stale_solver(suite, monkeypatch):
    module, invoke, output = suite
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "-B" in command:
            build_dir = Path(command[command.index("-B") + 1])
            build_dir.mkdir(parents=True)
            (build_dir / "CMakeCache.txt").write_text("mock configured build")
        return subprocess.CompletedProcess(command, 2 if "--build" in command else 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(SystemExit):
        invoke()
    assert calls and all(command[0] == "cmake" for command in calls)
    record = json.loads((output / "suite.json").read_text())
    assert record["status"] == "build_failed"
    assert record["process_failures"]
    assert not (output / "discretization").exists()


@pytest.mark.parametrize("damage", ["modify", "delete"])
def test_resume_rejects_damaged_trajectory_without_overwriting(suite, monkeypatch, damage):
    module, invoke, output = suite
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        write_child_result(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    invoke("--skip-build")
    artifact = next(output.rglob("b1.npz"))
    if damage == "delete":
        artifact.unlink()
    else:
        artifact.write_bytes(b"changed outside the experiment")
    calls.clear()
    with pytest.raises(SystemExit, match="artifact missing/changed"):
        invoke("--skip-build", "--resume")
    assert not calls
    if damage == "delete":
        assert not artifact.exists()
    else:
        assert artifact.read_bytes() == b"changed outside the experiment"


def test_zero_exit_with_child_exception_remains_incomplete_and_retryable(suite, monkeypatch):
    module, invoke, output = suite
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        write_child_result(command, failure=True)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    for options in [("--skip-build",), ("--skip-build", "--resume")]:
        with pytest.raises(SystemExit) as error:
            invoke(*options)
        assert error.value.code != 0
    assert len(calls) == 2
    manifest = json.loads((output / "suite.json").read_text())
    assert manifest["status"] == "incomplete"
    assert len(manifest["process_failures"]) == 1
    record = manifest["records"][manifest["process_failures"][0]]
    assert record["returncode"] == 0
    assert record["status"] == "incomplete_child"
    assert len(list((output / "logs").glob("*.log"))) == 2
