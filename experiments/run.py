#!/usr/bin/env python3
"""Run committed experiment cases sequentially; receipts are the queue state."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: fail(f"Invalid JSON number: {x}"))


def fail(message):
    raise ValueError(message)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
        return result.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        fail(f"Expected a relative path inside the experiment: {value}")
    return path


def name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        fail(f"Invalid experiment/machine/case name: {value}")
    return value


def machine(requested, profiles):
    """Jetson need not provide nvidia-smi; explicit selection is also supported."""
    snapshot = {"hostname": socket.gethostname(), "system": platform.platform()}
    gpu = ""
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True)
        gpu = result.stdout.strip()
        telemetry = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,clocks.sm,power.draw",
                                    "--format=csv,noheader"], capture_output=True, text=True)
        snapshot["telemetry"] = telemetry.stdout.strip() or None
    model = Path("/proc/device-tree/model")
    if model.exists():
        snapshot["device_model"] = model.read_text().strip("\x00")
    snapshot["gpu"] = gpu
    detected = "local"
    if "1070" in gpu:
        detected = "gtx1070"
    elif "5060" in gpu:
        detected = "rtx5060"
    elif "Orin" in snapshot.get("device_model", "") or "Orin" in gpu:
        detected = "jetson"
    selected = requested or detected
    if selected not in profiles:
        fail(f"Unknown machine profile {selected}; choose one of {list(profiles)}")
    return selected, snapshot


def command(argv, values):
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        fail("Commands must be nonempty arrays of arguments")
    # Never interpret experiment arguments through a shell.
    return [arg.format_map(values) for arg in argv]


def execute(argv, cwd, output, env):
    """Stop descendants too, so interrupting a build cannot leave GPU workers behind."""
    with (output / "stdout.log").open("a") as stdout, (output / "stderr.log").open("a") as stderr:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        fail(f"Command exited {code}; see {output / 'stderr.log'}")


def worktree(commit):
    # Use the common checkout so all agent worktrees share the same source cache.
    common = Path(git("rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = (ROOT / common).resolve()
    else:
        common = common.resolve()
    path = common.parent / ".worktrees" / "experiment-runner" / commit
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(path), commit)
    if subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() != commit:
        fail(f"Source cache points to the wrong commit: {path}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain",
                                     "--untracked-files=no"], text=True).strip()
    if dirty:
        fail(f"Pinned source worktree has been modified: {path}")
    return path


def build_source(experiment, commit, profile, python, environment, cache):
    source = worktree(commit)
    recipe = experiment.get("build", {})
    if recipe.get("commands") and not recipe.get("artifacts"):
        fail("Build commands must declare their output artifact patterns")
    key = digest({"commit": commit, "recipe": recipe, "profile": profile,
                  "python": python, "environment": environment})
    build = cache / "builds" / key
    build.mkdir(parents=True, exist_ok=True)
    receipt = build / "build.json"
    if receipt.exists():
        saved = read(receipt)
        if all(Path(p).exists() and file_hash(p) == h for p, h in saved["artifacts"].items()):
            return source, build, saved
    values = dict(source=str(source), build=str(build), python=python,
                  cuda_arch=str(profile.get("cuda_arch", "")), runner=str(HERE))
    env = dict(os.environ, GATO_BUILD_DIR=str(build))
    if recipe.get("commands"):
        print(f"Building {experiment['id']}; logs: {build}", flush=True)
    for argv in recipe.get("commands", []):
        execute(command(argv, values), source, build, env)
    artifacts = {}
    for pattern in recipe.get("artifacts", []):
        matches = [p for p in build.glob(str(relative(pattern))) if p.is_file()]
        if not matches:
            fail(f"Build produced no artifacts matching {pattern}")
        artifacts.update({str(p): file_hash(p) for p in matches})
    saved = {"commit": commit, "recipe": recipe, "profile": profile, "artifacts": artifacts}
    write(receipt, saved)
    return source, build, saved


def collect(output):
    """Derive summaries from receipts, including failed and interrupted attempts."""
    records = []
    for path in sorted(output.glob("*/*/*/attempt-*/receipt.json")):
        receipt = read(path)
        metrics = path.parent / "metrics.json"
        records.append({"machine": receipt["machine"], "experiment": receipt["experiment"],
                        "case": receipt["case"]["id"], "repeat": receipt["case"]["repeat"],
                        "status": receipt["status"], "attempt": str(path.parent.relative_to(output)),
                        "metrics": read(metrics) if receipt["status"] == "completed" and metrics.exists() else {}})
    write(output / "index.json", records)
    with (output / "summary.csv.tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["machine", "experiment", "case", "repeat",
                                                   "status", "attempt", "metrics"])
        writer.writeheader()
        for row in records:
            writer.writerow(dict(row, metrics=json.dumps(row["metrics"], sort_keys=True)))
    (output / "summary.csv.tmp").replace(output / "summary.csv")
    lines = ["# Experiment results", "", "| Machine | Experiment | Case/repeat | Status | Artifacts |",
             "| --- | --- | --- | --- | --- |"]
    lines.extend(f"| {r['machine']} | {r['experiment']} | {r['case']}/{r['repeat']} | "
                 f"{r['status']} | [attempt]({r['attempt']}/) |" for r in records)
    (output / "README.md").write_text("\n".join(lines) + "\n")


def run(args):
    plan = read(args.plan)
    if plan.get("version") != 1:
        fail("Expected experiment plan version 1")
    config = read(HERE / "local.json") if (HERE / "local.json").exists() else {}
    profiles = read(HERE / "machines.json")
    selected, hardware = machine(args.machine or config.get("machine"), profiles)
    print(f"Machine: {selected}", flush=True)
    profile = profiles[selected]
    # A venv executable commonly symlinks to the system Python. Resolving it
    # discards the venv and silently changes both packages and build headers.
    python = str(Path(config.get("python", sys.executable)).expanduser().absolute())
    environment = {"executable": python,
                   "python": subprocess.check_output([python, "-c", "import sys; print(sys.version)"], text=True).strip()}
    environment["packages"] = json.loads(subprocess.check_output([
        python, "-c", "import importlib.metadata as m, json; "
        "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))"], text=True))
    if profile.get("cuda_arch"):
        environment["nvcc"] = subprocess.check_output(["nvcc", "--version"], text=True).strip() if shutil.which("nvcc") else "missing"
    output = Path(args.output or config.get("output", "~/gato-results")).expanduser().resolve()
    cache = Path(config.get("cache", "~/.cache/gato-experiments")).expanduser().resolve()
    runner_hashes = {str(p.relative_to(HERE)): file_hash(p) for p in
                     [HERE / "run.py", *sorted((HERE / "adapters").glob("*.py"))]}
    pending = []
    seen = set()
    failures = 0
    for experiment in plan["experiments"]:
        eid = name(experiment["id"])
        if eid in seen:
            fail(f"Duplicate experiment id: {eid}")
        seen.add(eid)
        if not experiment.get("enabled", True) or selected not in experiment["machines"]:
            continue
        if args.only and eid not in args.only:
            continue
        commit = experiment["source_commit"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            fail(f"{eid}: source_commit must be a full immutable Git commit")
        try:
            git("cat-file", "-e", commit + "^{commit}")
        except subprocess.CalledProcessError:
            git("fetch", "origin")
            git("cat-file", "-e", commit + "^{commit}")
        inputs = experiment.get("inputs", {})
        for path, expected in inputs.items():
            if file_hash(ROOT / relative(path)) != expected:
                fail(f"Frozen input hash mismatch: {path}")
        spec = {k: v for k, v in experiment.items() if k not in ("cases", "repeats", "enabled", "machines")}
        case_names = set()
        repeats = experiment.get("repeats", 1)
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            fail(f"{eid}: repeats must be a positive integer")
        for case in experiment["cases"]:
            cid = name(case["id"])
            if cid in case_names:
                fail(f"{eid}: duplicate case id {cid}")
            case_names.add(cid)
            for repeat in range(repeats):
                resolved = dict(case, repeat=repeat)
                identity = {"experiment": spec, "case": resolved, "profile": profile,
                            "environment": environment, "runner": runner_hashes,
                            "device": {k: hardware.get(k) for k in ("hostname", "gpu", "device_model")}}
                folder = output / name(selected) / eid / digest(identity)
                attempts = sorted(folder.glob("attempt-*/receipt.json"))
                status = read(attempts[-1])["status"] if attempts else "pending"
                if status == "completed" or (status == "failed" and not args.retry_failed):
                    print(f"{status}: {eid}/{cid}/{repeat}", flush=True)
                    failures += status == "failed"
                    continue
                print(f"pending: {eid}/{cid}/{repeat}", flush=True)
                pending.append((experiment, resolved, identity, folder))
    if args.only and set(args.only) - seen:
        fail(f"Unknown experiment ids: {sorted(set(args.only) - seen)}")
    if args.dry_run:
        return 0
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    build_errors = {}
    # Shared across agent worktrees/clones using this cache. Manual GPU jobs must
    # still be kept out of timing runs; a filesystem lock cannot control them.
    with (cache / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("Another experiment runner is active; wait for it to finish")
        try:
            for experiment, case, identity, folder in pending:
                # Another invocation may have completed this case between plan
                # inspection and acquisition of the shared lock.
                existing = sorted(folder.glob("attempt-*/receipt.json"))
                if existing and read(existing[-1])["status"] == "completed":
                    continue
                if existing and read(existing[-1])["status"] == "running":
                    abandoned = read(existing[-1])
                    abandoned.update(status="interrupted", error="Previous invocation did not finish its receipt")
                    write(existing[-1], abandoned)
                attempt = folder / f"attempt-{len(list(folder.glob('attempt-*'))) + 1:03d}"
                attempt.mkdir(parents=True)
                receipt = {"experiment": experiment["id"], "machine": selected, "hardware": hardware,
                           "identity": identity, "case": case, "status": "running", "started": time.time()}
                write(attempt / "case.json", case)
                write(attempt / "receipt.json", receipt)
                try:
                    build_key = digest([experiment["source_commit"], experiment.get("build", {})])
                    if build_key in build_errors:
                        fail(build_errors[build_key])
                    try:
                        source, build, build_receipt = build_source(experiment, experiment["source_commit"],
                                                                   profile, python, environment, cache)
                    except (ValueError, OSError, subprocess.CalledProcessError) as error:
                        build_errors[build_key] = str(error)
                        raise
                    receipt["build"] = build_receipt
                    input_dir = attempt / "inputs"
                    input_dir.mkdir()
                    for path, expected in experiment.get("inputs", {}).items():
                        destination = input_dir / relative(path)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(ROOT / path, destination)
                        # A long build may allow another agent to edit an input.
                        if file_hash(destination) != expected:
                            fail(f"Frozen input changed before execution: {path}")
                    values = dict(source=str(source), build=str(build), python=python, runner=str(HERE),
                                  case=str(attempt / "case.json"), output=str(attempt), inputs=str(input_dir),
                                  cuda_arch=str(profile.get("cuda_arch", "")))
                    argv = command(experiment["command"], values)
                    receipt["command"] = argv
                    write(attempt / "receipt.json", receipt)
                    print(f"Running {experiment['id']}/{case['id']}/{case['repeat']}; logs: {attempt}", flush=True)
                    execute(argv, source, attempt, dict(os.environ, GATO_BUILD_DIR=str(build),
                                                       GATO_SOURCE_DIR=str(source), GATO_MACHINE=selected))
                    metrics = read(attempt / "metrics.json")
                    if not isinstance(metrics, dict):
                        fail("Experiment must write a JSON object to metrics.json")
                    receipt["status"] = "completed"
                except KeyboardInterrupt:
                    receipt["status"] = "interrupted"
                    raise
                except (ValueError, OSError, subprocess.CalledProcessError) as error:
                    receipt.update(status="failed", error=str(error))
                    failures += 1
                finally:
                    receipt["finished"] = time.time()
                    receipt["hardware_after"] = machine(selected, profiles)[1]
                    receipt["artifacts"] = {str(p.relative_to(attempt)): file_hash(p) for p in attempt.rglob("*")
                                             if p.is_file() and p.name != "receipt.json"}
                    write(attempt / "receipt.json", receipt)
                    print(f"{receipt['status']}: {experiment['id']}/{case['id']}/{case['repeat']}", flush=True)
        finally:
            collect(output)
        if config.get("sync_to"):
            # rsync updates preserved attempts without deleting the receiver's
            # other machines. Delivery failure never invalidates local results.
            destination = config["sync_to"].rstrip("/") + "/" + selected + "/"
            (output / selected).mkdir(exist_ok=True)
            result = subprocess.run(["rsync", "-a", str(output / selected) + "/", destination])
            if result.returncode:
                print("Result upload failed; run again to retry delivery", file=sys.stderr)
                failures += 1
    print(f"Results: {output / 'README.md'}", flush=True)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE / "plan.json")
    parser.add_argument("--machine", help="Override automatic machine detection")
    parser.add_argument("--output", help="Override local.json/default output folder")
    parser.add_argument("--only", nargs="+", help="Run only these experiment ids")
    parser.add_argument("--dry-run", action="store_true", help="List pending cases without builds or execution")
    parser.add_argument("--retry-failed", action="store_true", help="Make new attempts for failed executions")
    parser.add_argument("--collect-only", action="store_true", help="Regenerate the index after combining machine folders")
    args = parser.parse_args()
    if args.collect_only:
        config = read(HERE / "local.json") if (HERE / "local.json").exists() else {}
        output = Path(args.output or config.get("output", "~/gato-results")).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        collect(output)
        return 0
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Experiment runner: {error}", file=sys.stderr)
        sys.exit(1)
