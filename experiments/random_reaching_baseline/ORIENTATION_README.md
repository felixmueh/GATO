# Orientation and ordinary torque cost comparison

This isolated worktree retains the original uniform-joint/FK sampler and exact
ordered development12. OFF uses orientation0/0; LOW uses.01/.2. Both use the
primary position<=5cm and speedL1<=1rad/s arrival gate. The30degree axis
threshold is used only by descriptive first_pose_touch_seconds. Orientation
errors remain recorded, and final approach stays off.

The original M14 orientation-on motion is reused and rescored, with original
labels unchanged: position+speed gives1/12. M14 is not rerun here. OFF canonical
timed out at54.0717mm with a physical pass, so its pilot is held. LOW canonical
passed at2.73s; its full12 yielded6arrivals/12physical/0errors. A separate
one-value follow-up LOW_TORQUE changes ordinary u_cost.0001 to.00001 only,
keeping LOW orientation, ctrl_lim_cost.02 and all gates fixed. It requires its
own passed canonical preflight before its full12 pilot. OFF_TORQUE separately
uses orientation0/0 at the same u_cost.00001; its canonical failed at55.7404mm
while passing physical checks. Felix explicitly approved exactly its prepared
ordered12 diagnostic pilot despite that failure. The raw
OFF_TORQUE/diagnostic_user_waiver.json receipt is pinned at SHA256
b6cf8f70c6f404ca2327453d0e14c4950b7d916f6d4b35457fbb29913d7afc8d.
The analyzer accepts only this exact arm/protocol/canonical/IDs/criteria/answer
exception. The failed canonical remains false and the original5cm criteria
remain unchanged. Original OFF remains held. No retries, automatic selection
or further tuning are authorized here.

The M14 core85ef2ea5 and immutable q020 module
d9bee23f90a1d7816b32db23749bd651db2a7537f94c3dfec15f1a448bb34cc8
are reused. All runs use B1/N32,30ms knots/feedback,3ms RK4,8s maximum,
modeA/controller_dt, SQP1000/PCG120, existing natural zero-PCG stopping,
unchanged cold seeds/warm shifts and once-per-episode dual/rho reset.
No KKT acceptance gate, new sampler, horizon run or solver change is added.
Original run.py and run_orientation.py remain immutable for their saved runs;
run_torque_regularization.py is an auditable copy with only its frozen
variant/parameter validation changed. Complete protocols and sibling hashes
are in protocols/{OFF,LOW,LOW_TORQUE,OFF_TORQUE}-orientation.{json,sha256}.

Original orientation/position weight ratios are.025:1rad wrapped-RPY error
norm has the same quadratic cost as about158mm Cartesian error (2rad≈316mm).
LOW ratios.0025 reduce these equivalents to50/100mm; OFF removes orientation
goal cost and leaves positional IK redundancy. RPY norms are not physical
rotation angles, and these scales are not success guarantees. Reducing u_cost
10x retains the torque barrier; zero-torque total curvature falls about3.25x
for joints1/2 and1.98x for joints3–7, with larger curvature at nonzero torque.

The former additional100 validation was user-stopped after46complete episodes
plus one interrupted episode and is not resumed or presented as complete.
Horizon execution and new samplers remain held. Raw histories stay under
/workspace/GATO/_expr_results/random-reaching-orientation-20261015.

## Exact launch commands and conditions

These are the four original launches plus the bounded torque contrasts.
GPU calls are sequential and owned by the designated execution agent.
Each pilot requires its own matching successful canonical preflight; OFF
therefore remains held. CPU preparation appends --dry-run to these commands
and records receipts in the raw preparation directory without native imports.
The exact written OFF_TORQUE diagnostic waiver above is the sole exception to
that canonical condition; it changes neither the failed result nor its5cm gate.

```sh
# OFF preflight; Root-reviewed canonical launch
/opt/gato-venv/bin/python3 /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/run_orientation.py --source /workspace/GATO-random-reaching-orientation --module /workspace/GATO-random-reaching-joint-margins/.experiment-modules/q020/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching --protocol /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/protocols/OFF-orientation.json --preflight-input /workspace/GATO-experiment-runner-simple/experiments/inputs/batching/ba_preflight.json --output /workspace/GATO/_expr_results/random-reaching-orientation-20261015/OFF/preflight

# OFF pilot; Matching same-arm canonical preflight_pass true; root review before dependent pilot
/opt/gato-venv/bin/python3 /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/run_orientation.py --source /workspace/GATO-random-reaching-orientation --module /workspace/GATO-random-reaching-joint-margins/.experiment-modules/q020/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching --protocol /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/protocols/OFF-orientation.json --pairs 3626 7733 7132 7093 1265 9651 5264 279 8054 8270 2497 2342 --output /workspace/GATO/_expr_results/random-reaching-orientation-20261015/OFF/pilot

# LOW preflight; Root-reviewed canonical launch
/opt/gato-venv/bin/python3 /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/run_orientation.py --source /workspace/GATO-random-reaching-orientation --module /workspace/GATO-random-reaching-joint-margins/.experiment-modules/q020/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching --protocol /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/protocols/LOW-orientation.json --preflight-input /workspace/GATO-experiment-runner-simple/experiments/inputs/batching/ba_preflight.json --output /workspace/GATO/_expr_results/random-reaching-orientation-20261015/LOW/preflight

# LOW pilot; Matching same-arm canonical preflight_pass true; root review before dependent pilot
/opt/gato-venv/bin/python3 /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/run_orientation.py --source /workspace/GATO-random-reaching-orientation --module /workspace/GATO-random-reaching-joint-margins/.experiment-modules/q020/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching --protocol /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/protocols/LOW-orientation.json --pairs 3626 7733 7132 7093 1265 9651 5264 279 8054 8270 2497 2342 --output /workspace/GATO/_expr_results/random-reaching-orientation-20261015/LOW/pilot

# LOW_TORQUE preflight; Root-reviewed canonical launch
/opt/gato-venv/bin/python3 /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/run_torque_regularization.py --source /workspace/GATO-random-reaching-orientation --module /workspace/GATO-random-reaching-joint-margins/.experiment-modules/q020/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching --protocol /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/protocols/LOW_TORQUE-orientation.json --preflight-input /workspace/GATO-experiment-runner-simple/experiments/inputs/batching/ba_preflight.json --output /workspace/GATO/_expr_results/random-reaching-orientation-20261015/LOW_TORQUE/preflight

# LOW_TORQUE pilot; Matching LOW_TORQUE canonical preflight_pass true; no timeout waiver
/opt/gato-venv/bin/python3 /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/run_torque_regularization.py --source /workspace/GATO-random-reaching-orientation --module /workspace/GATO-random-reaching-joint-margins/.experiment-modules/q020/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching --protocol /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/protocols/LOW_TORQUE-orientation.json --pairs 3626 7733 7132 7093 1265 9651 5264 279 8054 8270 2497 2342 --output /workspace/GATO/_expr_results/random-reaching-orientation-20261015/LOW_TORQUE/pilot
# OFF_TORQUE preflight; Root-reviewed OFF_TORQUE canonical only
/opt/gato-venv/bin/python3 /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/run_off_torque.py --source /workspace/GATO-random-reaching-orientation --module /workspace/GATO-random-reaching-joint-margins/.experiment-modules/q020/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching --protocol /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/protocols/OFF_TORQUE-orientation.json --preflight-input /workspace/GATO-experiment-runner-simple/experiments/inputs/batching/ba_preflight.json --output /workspace/GATO/_expr_results/random-reaching-orientation-20261015/OFF_TORQUE/preflight

# OFF_TORQUE pilot; NOT authorized unless matching OFF_TORQUE canonical preflight_pass true, or explicit later user waiver; no inferred waiver
/opt/gato-venv/bin/python3 /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/run_off_torque.py --source /workspace/GATO-random-reaching-orientation --module /workspace/GATO-random-reaching-joint-margins/.experiment-modules/q020/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching --protocol /workspace/GATO-random-reaching-orientation/experiments/random_reaching_baseline/protocols/OFF_TORQUE-orientation.json --pairs 3626 7733 7132 7093 1265 9651 5264 279 8054 8270 2497 2342 --output /workspace/GATO/_expr_results/random-reaching-orientation-20261015/OFF_TORQUE/pilot
```

## Saved-record analysis

Temporary CPU previews use --allow-partial and an output outside thesis_results.
A final export requires each new arm to be complete or explicitly held.
The report preserves held/unrecorded labels and the fixed12 denominator,
with no OFF pilot rate inferred. Saved-trace5/7.5/10cm sensitivity is
descriptive only: earlier5cm arrival truncates later observations, and no
rescore changes original labels, rescues OFF or implies eventual arrival.

```sh
python experiments/random_reaching_baseline/analyze_orientation.py --raw-root /workspace/GATO/_expr_results/random-reaching-orientation-20261015 --source /workspace/GATO-random-reaching-orientation --output /tmp/gato-orientation-review --allow-partial
```

Final destination after completion/review is
/workspace/GATO/thesis_results/random-reaching-orientation-20261015, with
self-contained REPORT.md/PDF, compact case/sensitivity/canonical-trace CSVs,
figures and editable analysis/renderer scripts.
