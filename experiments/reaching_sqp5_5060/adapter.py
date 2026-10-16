#!/usr/bin/env python3
"""Adapt one frozen reaching worker to the unchanged experiment runner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback


def benchmark_readiness(metrics):
    return bool(metrics.get("complete") and not metrics.get("execution_error")
                and not metrics.get("interrupted") and metrics.get("physical", {}).get("finite"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--family", choices=("random", "cone"), required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = args.inputs / "experiments/reaching_sqp5_5060/inputs"
    modules = list((args.build / "modules").glob("bsqpN32_tiago_right*.so"))
    if len(modules) != 1:
        raise ValueError("Expected exactly one shared N32 TIAGo module")
    pair = -1 if args.preflight else json.loads(args.case.read_text())["parameters"]["pair_id"]
    episode = args.output / "episode"
    argv = [sys.executable, str(args.source / "experiments/reaching_sqp5_5060/run_case.py"),
            "--source", str(args.source), "--module", str(modules[0]),
            "--tasks", str(inputs / "tasks.json"), "--family", args.family,
            "--protocol", str(inputs / (args.arm + ".json")), "--output", str(episode)]
    argv += ["--preflight-input", str(inputs / "canonical.json")] if args.preflight else ["--pairs", str(pair)]
    try:
        completed = subprocess.run(argv, check=False)
        metrics_path = episode / ("preflight" if args.preflight else f"pair_{pair:05d}") / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics.update(campaign_id="reaching-sqp5-5060-20261016", execution_backend="internal",
                       source_commit=subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
                       adapter_argv=argv)
        if args.preflight:
            # This is a diagnostic benchmark. Preserve canonical arrival/physical
            # negatives as outcomes; only incomplete/nonfinite/setup failures hold
            # dependent tasks. The worker's original preflight_pass stays intact.
            metrics["benchmark_readiness_pass"] = benchmark_readiness(metrics)
            metrics["canonical_role"] = "Diagnostic; finite complete execution is readiness, arrival and raw limits are separate measured outcomes."
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        return (0 if metrics["benchmark_readiness_pass"] else 1) if args.preflight else completed.returncode
    except BaseException:
        (args.output / "metrics.json").write_text(json.dumps(dict(
            campaign_id="reaching-sqp5-5060-20261016", family=args.family,
            arm=args.arm, pair_id=pair, execution_error=traceback.format_exc(),
            task_success=False, physical_pass=False, outcome="execution_failed"), indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
