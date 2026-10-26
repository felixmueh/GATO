#!/usr/bin/env python3
"""The bdsv path on this GPU, against the golden problem of test/test_parity_golden.py.

Solves the suite's arm reach problem (TEST_PARAMS and arm_problem from
test/conftest.py: ARM_START at rest, ARM_GOAL at every knot, 10 SQP iterations)
for indy7 on the pcg and the bdsv path, each on a fresh solver, and prints the
initial and final merit next to the stored goldens (captured on the receipt
GPU), whether xu is bitwise equal to the golden, and its largest difference
relative to the largest golden entry. The goldens are unconverged 10-iteration
solves, so a pcg plan on another architecture may drift from them without
being wrong; the final merit is the quantity to compare. Per SQP iteration it
also prints the linear-system count (on bdsv: 1 a direct solve, 2 a skipped
update) and the accepted step size (-1 marks a line-search failure).

Needs pytest for the conftest import (pip install pytest); run from the root of
the GATO checkout with its venv active:
    python issue19/bdsv_golden_check.py
"""
import sys
from pathlib import Path

import numpy as np

import gato

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))
try:
    from conftest import TEST_PARAMS, arm_problem
except ImportError as exc:
    raise SystemExit(f"cannot import test/conftest.py ({exc}); pip install pytest")

PLANT = "indy7"
print(f"GATO {gato.__version__}, {PLANT}, the golden problem (TEST_PARAMS, arm_problem), batch 1\n")
print("| N | linsys | initial merit | golden | final merit | golden | xu bitwise = golden | xu diff / max golden "
      "| linsys count per SQP iteration | accepted step per SQP iteration |")
print("|---|---|---|---|---|---|---|---|---|---|")
for N in (32, 64, 128):
    if (PLANT, N) not in gato.available():
        print(f"| {N} | – | module not built | | | | | | | |")
        continue
    for linsys in ("pcg", "bdsv"):
        golden = np.load(ROOT / "test" / "golden" / f"{PLANT}_N{N}_{linsys}.npz")
        s = gato.BSQP(str(gato.robot_urdf(PLANT)), batch_size=1, N=N, dt=0.01, params=TEST_PARAMS, plant_type=PLANT)
        s.set_linsys(linsys)
        x, goals = arm_problem(PLANT, N)
        r = s.solve(x, goals)
        xu = np.asarray(r.xu, np.float32)
        diff = np.abs(xu - golden["xu"]).max() / np.abs(golden["xu"]).max()
        counts = np.asarray(r.stats.pcg_iters)[:, 0].tolist()
        steps = [round(float(v), 4) for v in np.asarray(r.stats.step_size)[:, 0]]
        print(f"| {N} | {r.stats.linsys} | {r.stats.initial_merit[0]:.4f} | {golden['initial_merit'][0]:.4f} "
              f"| {r.stats.final_merit[0]:.4f} | {golden['final_merit'][0]:.4f} "
              f"| {'yes' if np.array_equal(xu, golden['xu']) else 'no'} | {diff:.1e} | {counts} | {steps} |")
