# Public native/LDLT verification extension

This experiment reuses the built public Indy7 N128 module containing the Schur
input-preservation contribution and isolated block LDLT candidate. Both methods
use the **same binary**, solver settings, production controller and independent
Pinocchio plant. No native, binding or core changes are made by this extension.
The source checkout of that binary and the verification harness commit are
recorded separately, together with the binary and driver hashes.

Six new reaching scenarios use public `tasks.pickplace_scenarios`, seeds 2–7,
three goals each, step references. Parameters retain the original public task:
10 ms feedback, 1 ms plant, 4 s per goal, 5 cm distance, speed L2 below 1 rad/s,
100 ms dwell, telemetry-only limits, B1, PCG, native default cost/budgets.
These are six scenarios / eighteen distinct goals, each repeated by both methods.

Three tracking scenarios use public `tasks.fig8_scenarios`, seeds 11/12/13 and
fixed requested joint-speed ratios 0.6/0.8/1.0 respectively. Public
`tasks.fig8_reference(..., periods=2)` supplies two full periods plus sufficient
reference padding for the complete N128 window. The exact scenario, initial
state, full reference, period and requested duration are frozen before timing.
Tracking metrics use the production `goal_distances_knot0` convention: Euclidean
position error at feedback timestamps, with RMS/p95/max and a terminal endpoint
error against the exact frozen knot0 reference. Duration completion, finite
outputs and URDF limits are separate from these descriptive errors. No strict
tracking-error, KKT or exact-equality acceptance gate is added. Natural final
time overshoot by up to a feedback step is retained and reported.

## Preparation and execution

Use an active Python environment with the public dependencies and the existing
built native/LDLT checkout. `freeze` calls only CPU public scenario generators;
it does not construct a solver or launch GPU work. Any generation failure is
saved in a failed manifest and holds the shared campaign setup. It is not retried.

```bash
python research/preconditioners/verification_worker.py freeze \
  --checkout <built-native-ldlt-checkout> --output <fresh-frozen-directory>
```

The default campaign has nine cases, two methods and three separate process
repeats: **54 processes**, alternating method order by repeat. Run only in the
root-owned quiet GPU window with clocks recorded during measured tasks:

```bash
python research/preconditioners/verification_worker.py campaign \
  --checkout <built-native-ldlt-checkout> --tasks <frozen-directory> \
  --output <fresh-campaign-directory>
```

Individual runs can use `run --case reach-seed2 --method normal` with the same
checkout/tasks/output arguments. `--cases reach-seed2,tracking-seed11` and
`--repeats 3` explicitly select a smaller campaign. `--preflight` explicitly
selects the first selected reaching and tracking case, one process per method
per case: four processes with the default pair. The exact requested schedule
is saved before execution. These switches never change task parameters.
Independent case failures are preserved and do not stop other cases; no tuning,
retry-until-green, omitted failed episodes or solver fallback is implemented.

Task timing includes reset, required initial-plan solve, all feedback solves,
selection, independent simulation and shared in-memory recording. Every solve
retains complete PCG arrays including warmup. Native solve-wall, wrapper and
controller wall, PCG GPU events and CPU setup remain separate. File writing and
construction are outside the task interval; setup wall is recorded separately,
and campaign `status.json` records full subprocess duration including startup,
setup and final serialization. Per-call and per-physical-simulated-second task
cost are descriptive, so reduced reaching call counts remain visible. Tracking
compares the fixed requested duration and reports actual call counts rather
than assuming equality.

Desktop default primary timing subtracts the all-copy counter, retaining raw
task wall as well. `--primary-timing wall` makes actual whole-task wall primary
and subtracts nothing; it is the Jetson launcher default. Any memory migration,
cache maintenance or synchronization inside the task therefore stays charged.
This interval excludes one-time process construction/prepare; full subprocess
duration is reported separately and must not be mislabeled as the same boundary.

Run raw data belong in a new extension output directory. Existing desktop
cohorts and their finalized report remain unchanged. Preserve every result and
report paired-completed-only speed ratios separately if used. Jetson execution
is not performed during source preparation; the user will run its portable
experiment branch. Jetson MODE_40W is user-confirmed, not freshly measured;
no host power/clock checks or changes are requested.

The portable Jetson launcher additionally includes the existing GPU direct
`gpu_bdsv` control. `verification_worker.py run --method gpu_bdsv` maps to native
method `normal` plus `--linsys bdsv`; `run --method normal --linsys bdsv` is an
equivalent explicit invocation. Its iteration fields are retained native
bookkeeping, not CG counts. The original desktop default remains the declared
native-PCG/LDLT pair and 54 processes.
