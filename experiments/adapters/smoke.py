"""CPU-only receipt smoke case; this is not a solver performance experiment."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--case", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
case = json.loads(args.case.read_text())
(args.output / "metrics.json").write_text(json.dumps({
    "outcome": "recorded", "message": case["parameters"]["message"]
}, indent=2) + "\n")
