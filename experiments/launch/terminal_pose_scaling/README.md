# Terminal pose weights at long reaching horizons — preparation only

Question: on the retained goal-3 task, does increasing the terminal position and
orientation weights in proportion to the number of running-cost knots improve
the failed long-horizon reaches? This is a small sensitivity test, not a claim
that terminal scaling is the cause of the failures or a mathematically correct
continuous-time discretization. The cited external
`research/task-parameter-results-2026-10-14.md` was not found locally; the retained
task-parameter reports and frozen source/input files are the available evidence.

Only **two new RTX5060/internal science episodes** are prepared. No build, GPU
solve, Gazebo run or robot motion was performed. No native, worker or runner code
was changed. Unchanged matching baselines are references, not queued reruns.

| N | Preview at dt=0.03 s | Scale alpha=(N-1)/31 | N_cost | ee_orient_N_cost |
|---:|---:|---:|---:|---:|
| 32, retained anchor | 0.93 s | 1 | 80 | 2 |
| 128, new scaled case | 3.81 s | 127/31 | 327.741935483871 | 8.193548387097 |
| 256, new scaled case | 7.65 s | 255/31 | 658.064516129032 | 16.451612903226 |

The native merit sums knot costs without a dt quadrature factor. Scaling just
these two pose-terminal weights preserves their N32 ratio to the aggregate
running pose weights (q_cost=4 and ee_orient_cost=0.1). It does **not** scale
terminal velocity or barrier terms, and does not add quadrature. Fixed dt makes
the knot-count ratio also the preview-duration ratio; this design cannot isolate
time discretization from horizon growth. A separate dt study is outside scope.

## Frozen controls and retained evidence

Each new case uses the original goal-3 pose, exact full initial state, public
cold constant-state/zero-control initialization and shifted warm starts. N alone
selects the previously tested module; dt=0.03 s, maximum RK4 step=0.003 s,
mode B/solve-time-paid plant, SQP5, PCG120, qd_cost=0.1 and all other costs and
settings stay fixed. Orientation is ON, final approach OFF. Maximum task duration
is 8 s, with zero dwell and rough thresholds 50 mm / 30 degrees / joint-speed
L1<=1 rad/s. A success can end early; this is the original task protocol, not a
fixed eight-second settling observation.

Only N_cost and ee_orient_N_cost differ from each matching unscaled input.
Frozen copies retain just goal-3 and shared metadata, dropping irrelevant
tracking arrays. `v01.json` is byte-identical to the original, and the effective
canonical protocol is identical. The worker's inherited campaign label remains
`task-parameters-20261014`; new plan/case identities and input hashes distinguish
this successor cohort. See the copied inputs and `retained_baselines.json` for
exact values, original attempts and provenance.

- N32 anchor: three retained RTX SQP5 passes, about 1.478 s simulated time.
- N128/SQP5: retained RTX task and physical failure at 8 s; final distance
  676.631 mm. Native median **18.361 ms**, wrapper median 18.436 ms. The supplied
  approximate 7 ms statement does not match this case: N256/SQP1 reports 7.118 ms.
- N256/SQP5: retained RTX task and physical failure at 8 s; final distance
  374.272 mm; native median 31.370 ms, wrapper median 31.453 ms.

N128 runtime/collector pins remain fa6f6be / 90bdb825, with the existing
N16/32/64/128 build recipe. N256 remains a17e3e6 / f6cb96f, with the existing
N32/256 recipe. These pairs compare within each unchanged numerical source and
loaded native module; N128 and N256 are separate source/build cohorts. The N256
successor adds only the approved conditional shared-memory launch policy to the
internal baseline. Prefer the retained cached module and record actual loaded source, path, module
hash, compiler and build options. A rebuild from the same numerical source and
settings may have different binary bytes without changing the scientific
comparison. Record rebuild provenance; material source/compiler/options or
behavior changes require reassessing baseline comparability, not automatic
reruns for a hash difference alone.

RTX capacity and finite execution are already demonstrated for N256. GTX1070
cannot run N256; GTX N128, Jetson and Gazebo are not queued in this proposal.

## Review and future manual commands

The isolated preparation branch is `experiment/terminal-pose-scaling` in
`/workspace/GATO-terminal-pose-scaling`. It is not pushed by this task. After
separate publication or an agreed cherry-pick into the deployment runner, use
`git switch experiment/terminal-pose-scaling` (or the chosen integration branch).
This guide adds only plans, frozen inputs and references; it does not replace
runner code. Use the existing CUDA/Python shell and
the **existing cache directories** from the retained studies. No publication is
performed by this preparation. This single loop prints the two dry runs:

```bash
TERMINAL_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
TERMINAL_OUTPUT="$PWD/.experiment-results/terminal-pose-scaling-20261014-rtx5060_internal"
for TERMINAL_N in 128 256; do
  if [ "$TERMINAL_N" = 128 ]; then
    TERMINAL_CACHE="$PWD/.experiment-cache/task-parameters-20261014-rtx5060_internal"
  else
    TERMINAL_CACHE="$PWD/.experiment-cache/task-parameters-large-optin-20261014-rtx5060_internal"
  fi
  "$TERMINAL_PYTHON" experiments/run.py \
    --plan experiments/launch/terminal_pose_scaling/rtx5060_internal.json \
    --machine rtx5060 --python "$TERMINAL_PYTHON" \
    --output "$TERMINAL_OUTPUT" --cache "$TERMINAL_CACHE" \
    --only "TerminalPose-N$TERMINAL_N" --dry-run
done
```

**After separate execution authorization and exclusive idle-GPU review**, remove
`--dry-run` from the same loop. Each episode runs once; preserve all outcomes.
Existing compatible build artifacts are reused. Missing artifacts would cause
the runner to build; inspect the cache first rather than treating this guide as
authorization to compile. The runner's preflight signature includes the changed
T01 input hashes, so its existing guard performs one brief unchanged N32
canonical check per original build family. It cannot transparently reuse the
old receipt despite the identical effective canonical task; no reuse framework
or receipt rewriting was added. A shared canonical/import/nonfinite execution
failure holds its dependent case. Scientific task/physical negatives remain
results; no automatic retry or tuning.

Retained N128 and N256 case wall times were 10.378 and 10.227 s respectively
(including existing process/recording overhead, not builds). Budget about one
minute for the two new science episodes plus brief canonical checks with warm
caches; allow a few minutes for environment and collection overhead. This is a
planning estimate, not a prediction from native solve medians. No remote build
time is included. Keep raw recordings and a separate raw archive/backup; compact
motion/timing/provenance only belong in the final thesis export.

```bash
"$TERMINAL_PYTHON" experiments/run.py --collect-only --output "$TERMINAL_OUTPUT"
"$TERMINAL_PYTHON" experiments/collect.py --output "$TERMINAL_OUTPUT" \
  --export "$PWD/.experiment-work/terminal-pose-scaling-20261014-rtx5060_internal.tar.gz"
```

## Outcomes and later thesis export

Use the existing recorder/schema: protocol, case, metrics, provenance and
receipt; complete executed motion and compact per-call timings/quality. Preserve
execution errors, partial traces and failures. Report rough success/time,
terminal position/axis/full-orientation errors, wall/native timing distributions
and update intervals, separately from actual URDF position/velocity/applied
effort validation (original tolerances 0.0001 rad / 0.001 rad/s / 0.001 Nm).
The actual URDF effort limits are 43,43,26,26,26,26,26 Nm; this is not the square
experiment's uniform 26 Nm gate. Collision clearance is unassessed, exactly as
in the retained obstacle-free arm-only cohort. KKT/convergence is descriptive.

The later compact export should have four comparison rows (unscaled/scaled at
N128 and N256) and the three retained N32 anchor repeats, with a column marking
`retained` versus `new`, original attempt identities and module/input hashes.
Retain full executed motion for position/axis/physical plots, separate task and
physical columns, and all timing samples needed for distributions. Raw data
stays under `_expr_results`; finalized report/PDF and editable compact plotting
data/scripts belong in `thesis_results/terminal-pose-scaling-20261014/`.

Two new observations can support a bounded sensitivity finding. They cannot
establish general reliability, an optimal scale or a unique explanation of
previous long-horizon failures. Any new failure is preserved and discussed
before changing the selected scale, solver budget, target or stopping gates.
