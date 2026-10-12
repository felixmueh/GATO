#!/usr/bin/env python3
"""Run committed experiment cases sequentially; receipts are the queue state."""
import argparse
import fcntl
import hashlib
import json
import math
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


def execute(argv, cwd, output, env, timeout=None):
    """Stop descendants too, so interrupting a build cannot leave GPU workers behind."""
    with (output / "stdout.log").open("a") as stdout, (output / "stderr.log").open("a") as stderr:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (KeyboardInterrupt, subprocess.TimeoutExpired) as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if isinstance(error, subprocess.TimeoutExpired):
                fail(f"Execution wall-clock timeout after {timeout} s; partial output retained at {output}; "
                     "distinct from the scientific task timeout")
            raise
    if code:
        fail(f"Command exited {code}; see {output / 'stderr.log'}")


def worktree(commit, cache):
    """Each checkout owns its source cache; never use the private legacy runner."""
    path = cache / "sources" / commit
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


def build_key(experiment, profile, python, environment):
    return digest({"commit": experiment["source_commit"], "recipe": experiment.get("build", {}),
                   "cuda_arch": profile.get("cuda_arch"), "python": python,
                   "python_abi": environment.get("python_abi")})


def build_source(experiment, commit, profile, python, environment, cache):
    recipe = experiment.get("build", {})
    for path, expected in recipe.get("inputs", {}).items():
        if file_hash(HERE / relative(path)) != expected:
            fail(f"Frozen build input hash mismatch: {path}")
    source = worktree(commit, cache)
    if recipe.get("commands") and not recipe.get("artifacts"):
        fail("Build commands must declare their output artifact patterns")
    build = cache / "builds" / build_key(experiment, profile, python, environment)
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
    saved = {"commit": commit, "recipe": recipe, "profile": profile, "artifacts": artifacts,
             "python": python, "environment": environment}
    write(receipt, saved)
    return source, build, saved


def collect(output):
    from collect import collect as collect_results
    return collect_results(output)


def scientific_identity(experiment, case, selected):
    """Protocol/source/input revisions change evidence; bookkeeping edits do not."""
    identity = {"machine": selected, "experiment": experiment.get("id", experiment.get("experiment")),
                "source_commit": experiment["source_commit"], "protocol": experiment.get("protocol"),
                "inputs": experiment.get("inputs", {}), "timing_mode": experiment.get("timing_mode"),
                "backend": experiment.get("backend", experiment.get("execution_backend")),
                "case": {"id": case["id"], "parameters": case.get("parameters", {}), "repeat": case["repeat"]}}
    if experiment.get("parameters"):
        identity["parameters"] = experiment["parameters"]
    return identity


def receipt_identity(receipt, metrics=None):
    """Read old/new receipts, retaining their recorded timing and backend."""
    identity = receipt.get("identity", {})
    spec = identity.get("experiment") if isinstance(identity.get("experiment"), dict) else dict(identity or receipt)
    spec = dict(spec)
    metrics = metrics or {}
    spec.setdefault("timing_mode", receipt.get("timing_mode", metrics.get("timing_mode")))
    spec.setdefault("backend", spec.get("execution_backend", receipt.get("backend",
                    receipt.get("execution_backend", metrics.get("backend", metrics.get("execution_backend"))))))
    return scientific_identity(spec, receipt["case"], receipt["machine"])


def discover(output):
    latest = {}
    for path in sorted(output.glob("*/*/*/attempt-*/receipt.json")):
        try:
            receipt = read(path)
            if receipt.get("kind") == "preflight":
                continue
            metrics = {}
            spec = receipt.get("identity", {})
            if isinstance(spec.get("experiment"), dict):
                spec = spec["experiment"]
            if "timing_mode" not in spec or ("backend" not in spec and "execution_backend" not in spec):
                metrics_path = path.parent / "metrics.json"
                if metrics_path.exists():
                    try:
                        metrics = read(metrics_path)
                    except (ValueError, OSError) as error:
                        print(f"Warning: cannot read legacy metrics {metrics_path}: {error}", file=sys.stderr)
            key = digest(receipt_identity(receipt, metrics))
            previous = latest.get(key)
            if previous is None or receipt.get("started", 0) >= previous[1].get("started", 0):
                latest[key] = (path, receipt)
        except (ValueError, OSError, KeyError, TypeError) as error:
            print(f"Warning: cannot read receipt {path}: {error}", file=sys.stderr)
    imports = output / "imports.json"
    if imports.exists():
        try:
            entries = read(imports)
            if not isinstance(entries, list):
                fail("expected a list of imported completion registrations")
            for entry in entries:
                if entry.get("status") == "completed":
                    path = output / relative(entry["path"])
                    if not path.exists():
                        print(f"Warning: imported path is missing; keeping its completed status: {path}", file=sys.stderr)
                    latest[digest(scientific_identity(entry, entry["case"], entry["machine"]))] = (path, entry)
        except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
            fail(f"Cannot determine saved progress from {imports}: {error}; repair the registration before execution")
    return latest


def copy_inputs(experiment, output, selected):
    inputs = experiment.get("inputs", {})
    destination = output / selected / experiment["id"] / "_inputs" / digest(inputs)[:16]
    destination.mkdir(parents=True, exist_ok=True)
    for value, expected in inputs.items():
        path = relative(value)
        original = ROOT / path
        if file_hash(original) != expected:
            fail(f"Frozen input hash mismatch: {value}")
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or file_hash(target) != expected:
            shutil.copyfile(original, target)
        if file_hash(target) != expected:
            fail(f"Frozen input changed before execution: {value}")
    return destination


def environment_snapshot(python, profile):
    environment = json.loads(subprocess.check_output([python, "-c",
        "import sys, importlib.metadata as m, json; print(json.dumps({"
        "'executable': sys.executable, 'python': sys.version, 'python_abi': list(sys.version_info[:2]), "
        "'packages': {d.metadata['Name']: d.version for d in m.distributions()}}))"], text=True))
    if profile.get("cuda_arch"):
        environment["nvcc"] = subprocess.check_output(["nvcc", "--version"], text=True).strip() if shutil.which("nvcc") else "missing"
    return environment


def run(args):
    plan = read(args.plan)
    if plan.get("version") != 1:
        fail("Expected experiment plan version 1")
    config = read(HERE / "local.json") if (HERE / "local.json").exists() else {}
    profiles = read(HERE / "machines.json")
    selected, hardware = machine(args.machine or config.get("machine"), profiles)
    print(f"Machine: {selected}", flush=True)
    profile = profiles[selected]
    # Keep the venv path: resolving executable symlinks discards its packages.
    python = str(Path(getattr(args, "python", None) or config.get("python", sys.executable)).expanduser().absolute())
    output = Path(args.output or config.get("output", ROOT / ".experiment-results")).expanduser().resolve()
    cache = Path(getattr(args, "cache", None) or config.get("cache", ROOT / ".experiment-cache")).expanduser().resolve()
    previous = discover(output)
    pending, seen, failures = [], set(), 0
    for experiment in plan["experiments"]:
        eid = name(experiment["id"])
        if eid in seen:
            fail(f"Duplicate experiment id: {eid}")
        seen.add(eid)
        if not experiment.get("enabled", True) or selected not in experiment["machines"]:
            continue
        if args.only and eid not in args.only:
            continue
        repeats = experiment.get("repeats", 1)
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            fail(f"{eid}: repeats must be a positive integer")
        case_names = set()
        for case in experiment["cases"]:
            cid = name(case["id"])
            if cid in case_names:
                fail(f"{eid}: duplicate case id {cid}")
            case_names.add(cid)
            for repeat in range(repeats):
                resolved = dict(case, repeat=repeat)
                identity = scientific_identity(experiment, resolved, selected)
                key = digest(identity)
                status = previous[key][1]["status"] if key in previous else "pending"
                if status == "completed" or (status in ("failed", "blocked") and not args.retry_failed):
                    print(f"{status}: {eid}/{cid}/{repeat}", flush=True)
                    failures += status in ("failed", "blocked")
                    continue
                folder = output / name(selected) / eid / f"{cid}-r{repeat:03d}-{key[:16]}"
                if key in previous:
                    folder = previous[key][0].parent.parent
                print(f"pending: {eid}/{cid}/{repeat}", flush=True)
                pending.append((experiment, resolved, identity, folder))
    if args.only and set(args.only) - seen:
        fail(f"Unknown experiment ids: {sorted(set(args.only) - seen)}")
    if args.dry_run:
        return 0
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    # Worktrees own their builds but share the repository's execution lock.
    common = Path(git("rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = (ROOT / common).resolve()
    with (common / "experiment-runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("Another experiment runner is active; wait for it to finish")
        previous = discover(output)
        builds, preparations, preflights = {}, {}, {}
        environment = None
        try:
            for experiment, case, identity, folder in pending:
                key = digest(identity)
                if key in previous and previous[key][1]["status"] == "completed":
                    continue
                if key in previous and previous[key][1]["status"] in ("failed", "blocked") and not args.retry_failed:
                    failures += 1
                    continue
                if key in previous and previous[key][1]["status"] == "running":
                    abandoned = previous[key][1]
                    abandoned.update(status="interrupted", error="Previous invocation did not finish its receipt")
                    write(previous[key][0], abandoned)
                attempt = folder / f"attempt-{len(list(folder.glob('attempt-*'))) + 1:03d}"
                attempt.mkdir(parents=True)
                receipt = {"experiment": experiment["id"], "machine": selected, "hardware": hardware,
                           "identity": identity, "case": case, "status": "running", "started": time.time(),
                           **{field: experiment[field] for field in ("research_question", "protocol", "backend", "timing_mode")
                              if field in experiment}}
                write(attempt / "case.json", case)
                write(attempt / "receipt.json", receipt)
                stage = "preparation"
                try:
                    if environment is None:
                        environment = environment_snapshot(python, profile)
                    receipt["environment"] = environment
                    eid = experiment["id"]
                    if eid not in preparations:
                        try:
                            commit = experiment["source_commit"]
                            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                                fail(f"{eid}: source_commit must be a full immutable Git commit")
                            try:
                                git("cat-file", "-e", commit + "^{commit}")
                            except subprocess.CalledProcessError:
                                git("fetch", "origin")
                                git("cat-file", "-e", experiment["source_commit"] + "^{commit}")
                            preparations[eid] = copy_inputs(experiment, output, selected)
                        except (ValueError, OSError, subprocess.CalledProcessError) as error:
                            preparations[eid] = error
                    if isinstance(preparations[eid], Exception):
                        raise preparations[eid]
                    input_dir = preparations[eid]
                    stage = "build"
                    shared = build_key(experiment, profile, python, environment)
                    if shared not in builds:
                        try:
                            builds[shared] = build_source(experiment, experiment["source_commit"],
                                                          profile, python, environment, cache)
                        except (ValueError, OSError, subprocess.CalledProcessError) as error:
                            builds[shared] = error
                    if isinstance(builds[shared], Exception):
                        raise builds[shared]
                    source, build, build_receipt = builds[shared]
                    # Validate the original input after a potentially long build,
                    # once per family. Use the already frozen copy for execution.
                    if not preparations.get((eid, "checked")):
                        for path, expected in experiment.get("inputs", {}).items():
                            if file_hash(ROOT / relative(path)) != expected:
                                error = ValueError(f"Frozen input changed before execution: {path}")
                                preparations[eid] = error
                                raise error
                        preparations[(eid, "checked")] = True
                    receipt["build"] = build_receipt
                    receipt["inputs_path"] = str(input_dir.relative_to(output))
                    values = dict(source=str(source), build=str(build), python=python, runner=str(HERE),
                                  case=str(attempt / "case.json"), output=str(attempt), inputs=str(input_dir),
                                  cuda_arch=str(profile.get("cuda_arch", "")))
                    preflight = experiment.get("preflight")
                    if preflight:
                        stage = "preflight"
                        group = name(preflight.get("group", eid))
                        signature = digest({"source": experiment["source_commit"], "build": shared,
                                            "inputs": experiment.get("inputs", {}), "preflight": preflight,
                                            "parameters": experiment.get("parameters", {}),
                                            "timing_mode": identity["timing_mode"], "backend": identity["backend"],
                                            "artifacts": {str(Path(p).relative_to(build)): h for p, h in build_receipt["artifacts"].items()},
                                            "case_parameters": case.get("parameters", {})
                                            if "group" in preflight and any("{case}" in arg for arg in preflight["command"]) else None})
                        if group in preflights and preflights[group][0] != signature:
                            fail(f"Preflight group {group} has conflicting configuration")
                        root = output / selected / "_preflight" / group
                        if group not in preflights:
                            for previous_pf in sorted(root.glob("attempt-*/receipt.json"), reverse=True):
                                saved = read(previous_pf)
                                if saved.get("signature") != signature:
                                    continue
                                if saved["status"] == "completed":
                                    preflights[group] = (signature, None, previous_pf.parent)
                                elif saved["status"] == "failed" and not args.retry_failed:
                                    preflights[group] = (signature, ValueError(saved.get("error", "Prior preflight failed")), previous_pf.parent)
                                break
                        if group not in preflights:
                            pf_output = root / f"attempt-{len(list(root.glob('attempt-*'))) + 1:03d}"
                            pf_output.mkdir(parents=True)
                            pf_receipt = {"kind": "preflight", "group": group, "status": "running",
                                          "started": time.time(), "machine": selected, "source_commit": experiment["source_commit"],
                                          "experiment": eid, "signature": signature,
                                          "case": {"id": "preflight", "repeat": 0}}
                            write(pf_output / "receipt.json", pf_receipt)
                            try:
                                argv = command(preflight["command"], dict(values, output=str(pf_output)))
                                pf_receipt["command"] = argv
                                execute(argv, source, pf_output, dict(os.environ, GATO_BUILD_DIR=str(build),
                                                                    GATO_SOURCE_DIR=str(source), GATO_MACHINE=selected))
                                pf_receipt["status"] = "completed"
                                preflights[group] = (signature, None, pf_output)
                            except (ValueError, OSError, subprocess.CalledProcessError) as error:
                                pf_receipt.update(status="failed", error=str(error))
                                preflights[group] = (signature, error, pf_output)
                            except KeyboardInterrupt:
                                pf_receipt["status"] = "interrupted"
                                raise
                            finally:
                                pf_receipt["finished"] = time.time()
                                write(pf_output / "receipt.json", pf_receipt)
                        receipt["preflight_path"] = str(preflights[group][2].relative_to(output))
                        if preflights[group][1]:
                            raise preflights[group][1]
                    stage = "execution"
                    argv = command(experiment["command"], values)
                    receipt["command"] = argv
                    write(attempt / "receipt.json", receipt)
                    print(f"Running {eid}/{case['id']}/{case['repeat']}; logs: {attempt}", flush=True)
                    timeout = experiment.get("timeout_seconds")
                    if timeout is not None and (not isinstance(timeout, (float, int)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0):
                        fail("timeout_seconds must be a positive number")
                    receipt["timeout_seconds"] = timeout
                    execute(argv, source, attempt, dict(os.environ, GATO_BUILD_DIR=str(build),
                                                       GATO_SOURCE_DIR=str(source), GATO_MACHINE=selected), timeout=timeout)
                    metrics = read(attempt / "metrics.json")
                    if not isinstance(metrics, dict):
                        fail("Experiment must write a JSON object to metrics.json")
                    receipt["status"] = "completed"
                except KeyboardInterrupt:
                    receipt["status"] = "interrupted"
                    raise
                except (ValueError, OSError, subprocess.CalledProcessError, KeyError, TypeError) as error:
                    receipt.update(status="failed" if stage == "execution" else "blocked", error=str(error),
                                   failure_stage=stage)
                    failures += 1
                finally:
                    receipt["finished"] = time.time()
                    # Metadata failures must not erase a valid measured outcome.
                    try:
                        receipt["artifacts"] = {filename: file_hash(attempt / filename)
                                                for filename in ("case.json", "metrics.json", "stdout.log", "stderr.log")
                                                if (attempt / filename).is_file()}
                    except OSError as error:
                        receipt["metadata_warning"] = str(error)
                    write(attempt / "receipt.json", receipt)
                    print(f"{receipt['status']}: {experiment['id']}/{case['id']}/{case['repeat']}", flush=True)
        finally:
            try:
                collect(output)
            except (ValueError, OSError, KeyError, TypeError) as error:
                print(f"Collection failed; raw receipts retained: {error}", file=sys.stderr)
                failures += 1
        if config.get("sync_to"):
            destination = config["sync_to"].rstrip("/") + "/" + selected + "/"
            (output / selected).mkdir(exist_ok=True)
            try:
                result = subprocess.run(["rsync", "-a", str(output / selected) + "/", destination])
                if result.returncode:
                    fail(f"rsync exited {result.returncode}")
            except (ValueError, OSError) as error:
                print(f"Result upload failed; local results retained: {error}", file=sys.stderr)
                failures += 1
    print(f"Results: {output}", flush=True)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="Explicit reviewed experiment plan (required for runs)")
    parser.add_argument("--machine", help="Override automatic machine detection")
    parser.add_argument("--output", help="Override local.json/default output folder")
    parser.add_argument("--cache", help="Override local.json/default source/build cache")
    parser.add_argument("--python", help="Python executable with the experiment dependencies")
    parser.add_argument("--only", nargs="+", help="Run only these experiment ids")
    parser.add_argument("--dry-run", action="store_true", help="List pending cases without fetching, builds or execution")
    parser.add_argument("--retry-failed", action="store_true", help="Make new attempts for failed/blocked executions")
    parser.add_argument("--collect-only", action="store_true", help="Regenerate the index after combining machine folders")
    args = parser.parse_args()
    if args.collect_only:
        config = read(HERE / "local.json") if (HERE / "local.json").exists() else {}
        output = Path(args.output or config.get("output", ROOT / ".experiment-results")).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        collect(output)
        return 0
    if args.plan is None:
        parser.error("--plan is required; the legacy default queue is not the campaign")
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Experiment runner: {error}", file=sys.stderr)
        sys.exit(1)
