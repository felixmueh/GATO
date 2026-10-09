"""Adapt one existing heatmap cell to the outer runner's case/output interface."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--case", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
case = json.loads(args.case.read_text())
params = case["parameters"]
source = Path(os.environ["GATO_SOURCE_DIR"])
artifacts = args.output / "artifacts"
argv = [sys.executable, str(source / "examples/solve_time_heatmap/run.py"),
        "--build-dir", os.environ["GATO_BUILD_DIR"], "--output", str(artifacts),
        "--gpu-name", os.environ["GATO_MACHINE"], "--plant", "tiago_right",
        "--knots", str(params["N"]), "--batches", str(params["batch"]),
        "--horizon-time", str(params["horizon_seconds"]),
        "--max-sqp-iters", str(params["sqp_iterations"]), "--initialization", "single",
        "--wall-time", str(params["wall_seconds"]), "--max-solve-ms", "0",
        "--repeats", "1", "--seed", str(params["seed"] + case["repeat"])]
subprocess.run(argv, cwd=source, check=True)
result = json.loads((artifacts / "results.json").read_text())
rows = result["results"]
if len(rows) != 1:
    raise ValueError("Expected exactly one heatmap cell")
row = rows[0]
if row["status"] == "error":
    raise RuntimeError(row.get("error", "Heatmap worker failed"))
milliseconds = row.get("avg_native_ms")
metrics = {
    "outcome": row["status"], "batch_size": params["batch"],
    "mean_native_ms": milliseconds,
    "batch_solves_per_second": 1000 / milliseconds if milliseconds and milliseconds > 0 else None,
    "timing_scope": "synchronized native batch solve; excludes initialization and binding transfers",
    "configuration": result["config"], "samples": row.get("samples", 0),
    "scope": "offline tracking pilot; wall-budget samples are not matched input/trajectory coverage across GPUs"
}
(args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
