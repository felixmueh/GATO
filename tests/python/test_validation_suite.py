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


def test_first_run_without_openloop_trajectory_is_incomplete(suite, monkeypatch):
    module, invoke, output = suite

    def fake_run(command, **kwargs):
        write_child_result(command)
        next(output.rglob("b1.npz")).unlink()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(SystemExit):
        invoke("--skip-build")
    manifest = json.loads((output / "suite.json").read_text())
    assert manifest["status"] == "incomplete"
    record = manifest["records"][manifest["process_failures"][0]]
    assert any("b1.npz" in reason for reason in record["child_failures"])


def set_study(output, name, config):
    path = output.parent / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["studies"] = {name: config}
    path.write_text(json.dumps(protocol))


@pytest.mark.parametrize("scenario", ["missing_branch", "missing_stage", "truncated", "physical_failure", "integration_failure"])
def test_mpc_requires_recorded_branches_and_stages_but_retains_failure(suite, monkeypatch, scenario):
    module, invoke, output = suite
    set_study(output, "mpc-interval", {
        "cell": [16, .45], "tasks": [0], "clocks": [.06],
        "execution_duration": .12, "initializations": ["warm"], "batches": [1, 4],
    })

    def fake_run(command, **kwargs):
        if "--source" not in command:
            write_child_result(command)
        else:
            for flag, expected in [("--iters", "2"), ("--passes", "1"), ("--rho", "0.01")]:
                assert command[command.index(flag)+1] == expected
            folder = Path(command[command.index("--output") + 1])
            folder.mkdir(parents=True)
            branches = {}
            for batch in [1, 4]:
                if batch == 4 and scenario == "missing_branch":
                    continue
                failed = scenario in ("physical_failure", "integration_failure")
                count = 1 if failed or scenario == "truncated" else 2
                branches[str(batch)] = dict(stages=[{} for _ in range(count)],
                    executed={"physical": not failed}, failure="collision" if failed else None,
                    simulated_seconds=count*.06, complete=not failed)
                (folder / f"b{batch}_executed.npz").write_bytes(b"executed state")
                if scenario == "integration_failure":
                    branches[str(batch)]["stages"] = [{"failure": "integration diverged"}]
                    branches[str(batch)]["simulated_seconds"] = 0.
                    continue
                for step in range(count):
                    if not (scenario == "missing_stage" and batch == 4 and step == 1):
                        (folder / f"b{batch}_stage{step}.npz").write_bytes(b"solver trajectory")
            (folder / "summary.json").write_text(json.dumps({"branches": branches}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    failed_observation = scenario in ("physical_failure", "integration_failure")
    if failed_observation:
        invoke("--skip-build")
    else:
        with pytest.raises(SystemExit):
            invoke("--skip-build")
    manifest = json.loads((output / "suite.json").read_text())
    assert manifest["status"] == ("completed" if failed_observation else "incomplete")


@pytest.mark.parametrize("scenario", ["missing_repeat", "duplicate_repeat", "invalid_numerical"])
def test_latency_requires_exact_workload_coverage_not_numerical_success(suite, monkeypatch, scenario):
    module, invoke, output = suite
    set_study(output, "latency", {
        "cells": [[16, .45]], "batches": [1, 4], "iterations": 2, "repeats": 2,
    })

    def fake_run(command, **kwargs):
        folder = Path(command[command.index("--output") + 1])
        folder.mkdir(parents=True)
        records = [dict(batch=batch, repeat=repeat, starts=16, calls=16//batch,
                        solve_wall_seconds=.01, finite_outputs=False, budget_counts_valid=True,
                        pcg_breakdown_events=1, work_valid=False, output_sha256="mock")
                   for batch in [1, 4] for repeat in range(2)]
        if scenario == "missing_repeat":
            records.pop()
        elif scenario == "duplicate_repeat":
            records[-1] = records[0].copy()
        (folder / "summary.json").write_text(json.dumps({
            "records": records, "summary": [{"batch": 1}, {"batch": 4}],
        }))
        (folder / "inputs.npz").write_bytes(b"initial trajectories")
        (folder / "inputs.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    if scenario == "invalid_numerical":
        invoke("--skip-build")
    else:
        with pytest.raises(SystemExit):
            invoke("--skip-build")
    manifest = json.loads((output / "suite.json").read_text())
    assert manifest["status"] == ("completed" if scenario == "invalid_numerical" else "incomplete")


@pytest.mark.parametrize("relationship", ["same", "ancestor", "descendant"])
def test_input_and_output_trees_cannot_overlap(suite, monkeypatch, relationship):
    module, invoke, output = suite
    inputs = {"same": output, "ancestor": output.parent,
              "descendant": output / "imported"}[relationship]
    inputs.mkdir(parents=True, exist_ok=True)
    def unexpected_process(*args, **kwargs):
        pytest.fail("An overlapping input/output tree must be rejected before launching anything")
    monkeypatch.setattr(module.subprocess, "run", unexpected_process)
    with pytest.raises(SystemExit):
        invoke("--skip-build", "--inputs-root", str(inputs))
