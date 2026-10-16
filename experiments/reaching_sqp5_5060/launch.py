#!/usr/bin/env python3
"""Launch exactly the frozen eight SQP5/RTX5060 reaching arms locally."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
COHORT = "reaching-sqp5-5060-20261016"
ARMS = ("NORMAL", "FLAT", "TERMINAL10", "TERMINAL10_FLAT")
DEFINES = "-DTIAGO_EXPERIMENT_JOINT_COST_MODES=1 -DTIAGO_COST_JOINT_MARGIN_RAD=0.20 -DTIAGO_COST_VELOCITY_LIMIT_FRACTION=0.80 -DTIAGO_COST_CONTROL_LIMIT_FRACTION=0.85"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_plan(source_commit):
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in sorted((HERE / "inputs").iterdir()) if p.is_file()}
    frozen = json.loads((HERE / "inputs/FREEZE_RECEIPT.json").read_text())
    for name, expected in frozen["files"].items():
        if sha(HERE / "inputs" / name) != expected:
            raise ValueError("Frozen study input changed: " + name)
    tasks = json.loads((HERE / "inputs/tasks.json").read_text())
    recipe = dict(inputs={"native/CMakeLists.txt": sha(ROOT / "experiments/native/CMakeLists.txt")},
                  commands=[["cmake", "-S", "{runner}/native", "-B", "{build}",
                             "-DGATO_SOURCE_DIR={source}", "-DKNOTS=32", "-DCMAKE_BUILD_TYPE=Release",
                             "-DGATO_EXPERIMENT_TELEMETRY=ON", "-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}-real",
                             "-DCMAKE_CUDA_FLAGS=" + DEFINES, "-DPython3_EXECUTABLE={python}"],
                            ["cmake", "--build", "{build}", "--target", "bsqpN32_tiago_right", "--parallel", "2"]],
                  artifacts=["modules/bsqpN32_tiago_right*.so"])
    experiments = []
    for family in ("cone", "random"):
        for arm in ARMS:
            common = ["{python}", "{source}/experiments/reaching_sqp5_5060/adapter.py",
                      "--source", "{source}", "--build", "{build}", "--inputs", "{inputs}", "--arm", arm]
            experiments.append(dict(id=family + "_" + arm, source_commit=source_commit,
                protocol=COHORT + "-" + arm, research_question="RQ-reaching", backend="internal",
                machines=["rtx5060"], timing_mode="A", repeats=1, inputs=inputs, build=recipe,
                preflight=dict(group=COHORT + "-" + arm,
                    command=common + ["--family", "random", "--preflight", "--output", "{output}"]),
                command=common + ["--family", family, "--case", "{case}", "--output", "{output}"],
                cases=[dict(id=f"pair_{row['pair_id']:05d}", parameters=dict(pair_id=row["pair_id"])) for row in tasks["families"][family]]))
    return dict(version=1, cohort=COHORT, scientific_episodes=800, diagnostic_canonicals=4,
                canonical_policy="One finite execution diagnostic per variant shared across both families. Retain arrival and physical negatives; only setup/nonfinite/incomplete failure holds dependent work.",
                no_other_experiments=True, experiments=experiments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="/opt/gato-venv/bin/python3" if Path("/opt/gato-venv/bin/python3").exists() else sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Inspect the explicit800-case plan; no build or native execution")
    parser.add_argument("--write-plan-only", action="store_true", help="Freeze and validate the plan without device queries or native execution")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Explicit new attempts for retained failed cases")
    args = parser.parse_args()
    source = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if dirty:
        raise ValueError("Study checkout has tracked edits; use the clean reviewed published commit")
    plan = make_plan(source)
    work = ROOT / ".experiment-work" / COHORT
    output = ROOT / "_expr_results" / COHORT / "rtx5060"
    cache = ROOT / ".experiment-cache" / COHORT
    archive = ROOT / "_expr_results" / (COHORT + "-backup") / "results.tar.gz"
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / "plan.json"
    encoded = json.dumps(plan, indent=2, allow_nan=False) + "\n"
    if plan_path.exists() and plan_path.read_text() != encoded:
        raise ValueError("This cohort already has a different source/plan; retain it and prepare an explicitly identified successor")
    plan_path.write_text(encoded)
    print(f"Source {source}\nPlan {plan_path}\nPlan SHA256 {sha(plan_path)}\nRaw {output}\nShared build/cache {cache}\nSeparate raw archive {archive}", flush=True)
    print("Exactly800 scientific episodes:100cone+100originalrandom × NORMAL/FLAT/TERMINAL10/TERMINAL10_FLAT. Four shared diagnostic canonicals. N32,30ms knots/feedback,3ms RK4,max30s,modeA,SQP5,PCG120,natural zero-PCG exit; orientation/final approach off. Arrival50mm and speedL1<=1 at feedback boundaries; raw limits counted separately.", flush=True)
    if args.write_plan_only:
        return 0
    sys.path.insert(0, str(ROOT / "experiments"))
    # Exact copies of the reviewed210d0de runner/collector are retained on this
    # isolated branch; neither the central runner nor its private cache is edited.
    runner = load("study_runner", ROOT / "experiments/run.py")
    collector = load("study_collector", ROOT / "experiments/collect.py")
    python = shutil.which(args.python) or str(Path(args.python).expanduser().absolute())
    invocation = SimpleNamespace(plan=plan_path, machine="rtx5060", output=str(output),
                                 cache=str(cache), python=python, dry_run=True,
                                 retry_failed=args.retry_failed, only=None)
    if not args.collect_only:
        runner.run(invocation)
        if args.dry_run:
            return 0
        missing = [name for name in ("git", "cmake", "nvcc", "c++") if not shutil.which(name)]
        if missing:
            raise ValueError("Missing tools in the existing CUDA environment: " + ", ".join(missing))
        subprocess.run([python, "-c", "import numpy, pinocchio, pybind11"], check=True)
        output.mkdir(parents=True, exist_ok=True)
        deployment = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_commit=source,
                          plan_sha256=sha(plan_path), plan=plan,
                          note="Manual RTX5060 internal simulation; no remote command, Gazebo or physical robot. All attempts and scientific negatives retained.")
        deployment_path = output / "deployment.json"
        if deployment_path.exists():
            if json.loads(deployment_path.read_text())["plan_sha256"] != deployment["plan_sha256"]:
                raise ValueError("Existing raw cohort has a different plan; preserve it")
        else:
            deployment_path.write_text(json.dumps(deployment, indent=2) + "\n")
        invocation.dry_run = False
        try:
            status = runner.run(invocation)
        except KeyboardInterrupt:
            status = 130
        finally:
            print("All study worker processes have returned; GPU released. Collecting saved data and backup.", flush=True)
            collector.collect(output)
            collector.export_results(output, archive)
            print(f"GPU work finished/stopped. All local raw attempts and failures archived: {archive}", flush=True)
        return status
    if not output.is_dir():
        raise ValueError("No local raw results to collect")
    collector.collect(output)
    collector.export_results(output, archive)
    print(f"Saved-only collection/archive complete: {archive}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
