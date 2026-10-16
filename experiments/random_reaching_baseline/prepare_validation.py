#!/usr/bin/env python3
"""Freeze 100 additional random reaches and a selected configuration; CPU only.

This preparer never imports the solver, launches a GPU process or chooses a
configuration from validation outcomes. The existing run.py owns execution.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys

import numpy as np

SEED = 20261015
COUNT = 100


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def verify_assignment(assignment, historical, pairs, pool_size):
    """Check retained identities without sampling, filtering or changing order."""
    ids, excluded = assignment["task_ids"], historical["task_ids"]
    size = len(pairs["start_index"])
    if (len(ids) != COUNT or any(type(i) is not int for i in ids)
            or len(set(ids)) != COUNT or any(i < 0 or i >= size for i in ids)
            or set(ids).intersection(excluded)):
        raise ValueError("Assignment must contain100 unique valid pairs disjoint from all original100")
    if (assignment["task_ids_sha256"] != ids_sha(ids)
            or assignment["excluded_historical_task_ids"] != excluded
            or assignment["excluded_historical_task_ids_sha256"] != ids_sha(excluded)
            or assignment["input_sha256"] != {name:historical["input_sha256"][name] for name in ("pool.npz", "pairs.npz")}
            or assignment["development_task_ids"] != historical.get("development_task_ids", excluded[:12])
            or assignment["population_size"] != size
            or assignment["remaining_population_size"] != size-len(excluded)
            or assignment["seed"] != SEED):
        raise ValueError("Frozen assignment metadata differs from this study's retained inputs")
    records = [dict(order=i+1, pair_id=pair, start_index=int(pairs["start_index"][pair]),
                    goal_index=int(pairs["goal_index"][pair])) for i, pair in enumerate(ids)]
    if assignment["records"] != records:
        raise ValueError("Frozen assignment records differ from the saved pair archive")
    if any(not (0 <= r[key] < pool_size) for r in records for key in ("start_index", "goal_index")):
        raise ValueError("Frozen pair references an invalid configuration index")
    if assignment["same_configuration_index_pairs"] != [r["pair_id"] for r in records if r["start_index"] == r["goal_index"]]:
        raise ValueError("Frozen assignment's same-configuration record differs")


def retained_assignment(path):
    """Read exact frozen bytes and their existing sibling SHA256 receipt."""
    payload = path.read_bytes()
    receipt = path.with_suffix(".sha256").read_text().split()
    if not receipt or hashlib.sha256(payload).hexdigest() != receipt[0]:
        raise ValueError("Frozen assignment bytes do not match their existing SHA256 receipt")
    return json.loads(payload), payload


def estimate(development, output):
    cases = [p for p in development.glob("pair_*/metrics.json") if read(p).get("complete")]
    if not cases:
        return
    metrics = [read(p) for p in cases]
    seconds = sum(m["episode_wall_s"] for m in metrics)
    size = sum(p.stat().st_size for metric in cases for p in metric.parent.rglob("*") if p.is_file())
    write(output / "resource_estimate.json", dict(reference=str(development), completed_cases=len(cases),
        observed_episode_wall_s=seconds, estimated_100_episode_wall_s=seconds/len(cases)*100,
        observed_case_bytes=size, estimated_100_case_bytes=size/len(cases)*100,
        notes=["Linear estimate from the final development configuration; fresh task difficulty can change time and storage.",
               "Native build/source caches already exist and are excluded; a separate raw backup requires additional storage.",
               "All cases and per-solve records retained; estimates do not authorize pruning, filtering or retries."]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "pool", "historical-protocol", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--final-protocol", type=Path)
    p.add_argument("--development-run", type=Path)
    p.add_argument("--reused-preflight", type=Path)
    p.add_argument("--preparation-name", default="preparation",
        help="Fresh versioned directory beneath output; the legacy preparation is preserved")
    p.add_argument("--assignment", type=Path,
        help="Reuse exact existing assignment.json bytes and sibling assignment.sha256; never resample")
    p.add_argument("--selected-tag", help="Configuration label for the validation protocol ID; defaults to selected protocol ID")
    args = p.parse_args()
    source, pool, output = args.source.resolve(), args.pool.resolve(), args.output.resolve()
    if args.preparation_name in ("", ".", "..") or Path(args.preparation_name).name != args.preparation_name:
        raise ValueError("Preparation name must be one directory name beneath output")
    prep = output / args.preparation_name
    if args.assignment and prep.resolve() == args.assignment.resolve().parent:
        raise ValueError("Reused assignment requires a separate versioned preparation; preserve its original directory")
    historical = read(args.historical_protocol)
    excluded = historical["task_ids"]
    if len(excluded) != 100 or len(set(excluded)) != 100:
        raise ValueError("Exclude exactly the complete original100 task roster")
    for name in ("pool.npz", "pairs.npz"):
        if sha(pool/name) != historical["input_sha256"][name]:
            raise ValueError("Original frozen task archive differs: " + name)
    with np.load(pool / "pairs.npz", allow_pickle=False) as pairs, np.load(pool/"pool.npz", allow_pickle=False) as configurations:
        size = len(pairs["start_index"])
        if size != 10000 or len(pairs["goal_index"]) != size:
            raise ValueError("Expected the retained10000 pair archive")
        population = [i for i in range(size) if i not in set(excluded)]
        existing = retained_assignment(args.assignment.resolve()) if args.assignment else None
        ids = existing[0]["task_ids"] if existing else random.Random(SEED).sample(population, COUNT)
        if existing:
            verify_assignment(existing[0], historical, pairs, len(configurations["q"]))
        records = [dict(order=i+1, pair_id=pair, start_index=int(pairs["start_index"][pair]),
                        goal_index=int(pairs["goal_index"][pair])) for i, pair in enumerate(ids)]
    assignment = dict(schema_version=1, role="additional_validation_only", seed=SEED,
        sampler="random.Random(seed).sample(population, 100)", python_version=sys.version,
        population_size=size, remaining_population_size=len(population),
        population_order="Ascending pair IDs0..9999 excluding every original100 ID; no geometry/outcome filtering",
        task_ids=ids, task_ids_sha256=ids_sha(ids),
        excluded_historical_task_ids=excluded, excluded_historical_task_ids_sha256=ids_sha(excluded),
        development_task_ids=historical["development_task_ids"] if "development_task_ids" in historical else excluded[:12],
        historical_protocol=str(args.historical_protocol.resolve()), historical_protocol_sha256=sha(args.historical_protocol),
        input_sha256={name: sha(pool/name) for name in ("pool.npz", "pairs.npz")}, records=records,
        same_configuration_index_pairs=[r["pair_id"] for r in records if r["start_index"] == r["goal_index"]],
        stopping_policy="One pass of all100 assigned tasks; preserve failures; no tuning, retries, filtering or decisions based on validation outcomes.")
    if existing:
        assignment, assignment_bytes = existing
    else:
        assignment_bytes = (json.dumps(assignment, indent=2, allow_nan=False)+"\n").encode()
    prep.mkdir(parents=True, exist_ok=True)
    manifest = prep / "assignment.json"
    if manifest.exists():
        if manifest.read_bytes() != assignment_bytes:
            raise ValueError("Existing frozen assignment differs; never overwrite or resample")
    else:
        manifest.write_bytes(assignment_bytes)
        (prep/"task_ids.txt").write_text(" ".join(map(str, ids))+"\n")
        (prep/"assignment.sha256").write_text(sha(manifest)+"  assignment.json\n")
        (prep/"frozen_before_validation_utc.txt").write_text(datetime.now(timezone.utc).isoformat()+"\n")
    if args.development_run:
        estimate(args.development_run.resolve(), prep)
    if not args.final_protocol:
        print(json.dumps(dict(assignment=str(manifest), task_ids_sha256=ids_sha(ids),
            sampled_count=len(ids), excluded_count=len(excluded), execution="not launched"), indent=2));return
    selected = read(args.final_protocol)
    if selected["task_ids"] != excluded or selected["development_task_ids"] != assignment["development_task_ids"]:
        raise ValueError("Final selected development configuration belongs to another task family")
    protocol = dict(selected)
    protocol.update(id="random-reaching-validation100-"+(args.selected_tag or selected["id"])+"-v2",
        evaluation_role="100 additional validation reaches; no further configuration decisions",
        task_ids=ids, task_ids_sha256=ids_sha(ids), validation_task_ids=ids,
        validation_assignment_sha256=sha(manifest), validation_seed=SEED,
        development_task_ids_are_provenance_only=True,
        selected_final_configuration=dict(path=str(args.final_protocol.resolve()), sha256=sha(args.final_protocol),
            original_id=selected["id"], selection="Final development configuration selected before any validation solve; known development failures remain failures."),
        comparison="100 additional tasks disjoint from all original100, not a12+88 split. Validation only: preserve and identify failures; no tuning, filtering, rescue, retry or policy changes from these outcomes.")
    protocol.pop("historical_manifest_sha256_original_unavailable", None)
    if args.reused_preflight:
        preflight = read(args.reused_preflight)
        if not preflight.get("preflight_pass"):
            raise ValueError("The selected configuration's canonical preflight did not pass")
        protocol["reused_canonical_preflight"] = dict(path=str(args.reused_preflight.resolve()),
            metrics_sha256=sha(args.reused_preflight), preflight_pass=True,
            policy="Reuse the unchanged selected configuration's source/module/parameter preflight; no additional GPU test")
    frozen_protocol = prep / "validation_protocol.json"
    if frozen_protocol.exists() and read(frozen_protocol) != protocol:
        raise ValueError("Existing validation configuration differs; never overwrite after freezing")
    write(frozen_protocol, protocol)
    argv = [sys.executable, str(source/"experiments/random_reaching_baseline/run.py"),
        "--source", str(source), "--module", selected["native_module_path"], "--pool", str(pool),
        "--protocol", str(frozen_protocol), "--pairs", *map(str, ids), "--output", str(output/"pilot")]
    write(prep/"launch_argv.json", argv)
    (prep/"launch_command.txt").write_text(shlex.join(argv)+"\n")
    dry = subprocess.run([*argv, "--dry-run"], capture_output=True, text=True)
    (prep/"dryrun.stdout.json").write_text(dry.stdout)
    (prep/"dryrun.stderr.log").write_text(dry.stderr)
    if dry.returncode:
        raise RuntimeError("Exact existing-harness CPU dry run failed; dependent launch held")
    write(prep/"PREPARED.json", dict(assignment_sha256=sha(manifest), protocol_sha256=sha(frozen_protocol),
        reused_assignment_path=str(args.assignment.resolve()) if args.assignment else None,
        selected_protocol_sha256=sha(args.final_protocol), task_ids_sha256=ids_sha(ids),
        dryrun_sha256=sha(prep/"dryrun.stdout.json"), launch_argv_sha256=sha(prep/"launch_argv.json"),
        source_commit=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        wrapper_sha256=sha(source/"experiments/random_reaching_baseline/run.py"),
        native_module_sha256=sha(selected["native_module_path"]),
        preparation_only=True, GPU_execution_started=False))
    print(json.dumps(dict(preparation=str(prep), assignment_sha256=sha(manifest), task_ids_sha256=ids_sha(ids),
        selected_protocol=str(frozen_protocol), dryrun="passed without native imports", execution="not launched"), indent=2))


if __name__ == "__main__":
    main()
