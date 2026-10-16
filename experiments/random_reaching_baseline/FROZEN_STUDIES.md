# Frozen OFF_TORQUE validation and SQP-cap studies

Preparation is CPU only. Sole-GPU execution follows root's focused review and
completion of the current selected-four batching pilot. No new native module,
solver change or calibration episode is needed.

The research baseline is OFF_TORQUE: orientation0/0, ordinary torque1e-5,
qd.05, q/v/u limit weights.02125/1.275/.02, mu10, rho.01, q inset.20rad,
speed/effort fractions.80/.85; B1/N32, .03s knots/feedback, .003s RK4,8s.
Position <=5cm and speed L1 <=1rad/s are the primary arrival criterion.
The immutable module SHA256 is
`d9bee23f90a1d7816b32db23749bd651db2a7537f94c3dfec15f1a448bb34cc8`.
Numerical source remains85ef2ea. The new worker copies the completed
run_off_torque.py runtime exactly; only prepare validates the two new study
roles/assignments/caps. The preparer verifies unchanged runtime AST.

Fresh100 is separate from the stopped M14 validation and from development.
Sample100 without replacement with stdlib Random(20261015), ascending original
10000-pair population excluding ALL historical100 IDs (including development12)
and ALL old stopped-assignment100 IDs, leaving9800 eligible. No geometry,
clearance, reachability, endpoint-separation or outcome filter is applied. The
original uniform-q pool and fixed pair table supply source-goal FK targets;
there is no new IK sampler. No validation outcome may retune/select the frozen
configuration. Preserve all100 outcomes and fixed100 denominator.

SQP caps1/5/100 each run the exact ordered development12; reuse saved cap1000.
Only the outer allowance changes. The existing natural zero-PCG exit remains;
a cap is not a requirement to execute that many iterations. PCG120 and all other
parameters are fixed. These36 episodes and reused12 are a development comparison,
not additional independent validation samples.

The saved canonical remains an8s timeout at55.7404mm, physical pass, original
preflight false. It is openly reused as diagnostic context, never as a passed
qualification or as evidence that a changed cap passed canonical. The latest user
request authorizes these diagnostic studies despite that known miss. No redundant
canonical is run. Shared setup/import/module/controller-constructor failure holds
the affected run; individual task/execution/nonfinite failures are recorded and
remaining independent cases continue. No retries, pruning, physical veto,
threshold relaxation, new KKT gate or stopping change is added.

Preparation command (use fresh raw roots):

```sh
python experiments/random_reaching_baseline/prepare_frozen_studies.py \
  --source /workspace/GATO-frozen-reaching-validation \
  --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching \
  --validation-root /workspace/GATO/_expr_results/random-reaching-off-torque-validation-100-20261015 \
  --sqp-root /workspace/GATO/_expr_results/random-reaching-off-torque-sqp-20261015
```

Both raw roots contain preparation/PREPARED.json, exact launch_commands.txt/JSON,
protocols, dryrun outputs and resource_estimate.json. Four CPU dryruns perform
source/module/input/assignment checks without importing the native extension.
The selected baseline freeze, canonical and worker hashes are retained. Assignment
and protocols are frozen before any new solve. Preparation refuses replacement
of an existing preparation; no resampling or silent update after outcomes.

Observed baseline12:167.14s episode wall, about23MB case data. Linear100 estimate
is23min/193MB. Plan30–120min and1GiB raw plus separate backup because new difficulty
and natural stopping are unknown. New SQP36 episodes: planning5–20min/0.5GiB plus
backup; lower caps can lengthen episodes and alter task success. These are planning
estimates, not deadlines or reasons to drop cases. Existing builds are excluded.

Final compact destinations, after complete collection/review:

- thesis_results/random-reaching-off-torque-validation-100-20261015/
- thesis_results/random-reaching-off-torque-sqp-20261015/

Keep raw arrays, source snapshots and archives outside thesis_results. Rates from
this fresh100, development12 and selected-four batching pilot remain separate.
