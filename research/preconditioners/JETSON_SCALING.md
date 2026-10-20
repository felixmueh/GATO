# Existing preconditioners on Jetson, N128/256/512

Branch `felix-precond-timing/jetson-scaling` is a public-upstream experiment,
based on public `aeacf16` plus our Schur input-preservation contribution
`9b0df42`. Existing algorithms are collected into the same binary per horizon;
PCG is retained in every method. No private TIAGo implementation is used.
No Jetson execution has been performed during preparation.

This document and `jetson_scaling.py` are the current transfer-subtracted scaling
protocol. Older `JETSON.md` describes a separate managed-memory/actual-wall
preparation and does not define this cohort.

## Methods and boundaries

The eleven selected methods are native damped stair (`normal`), GPU polynomial
orders2/3, CPU native-P control, block SGS, overlapping Schwarz, incomplete
Cholesky, block Jacobi, CPU polynomial orders2/3, and exact block LDLT factors
applied on GPU **inside PCG** (`block_ldlt`). No new direct algorithm is added.
The native-P host control distinguishes backend overhead from operator quality.
CPU methods use the existing host public-GLASS-formula PCG with GPU matvecs;
GPU polynomial and exact LDLT use the existing GPU callback recurrence.
AMG failed all four earlier N128 episodes; those results remain evidence and
AMG is excluded explicitly from this scaling plan. No AMGCL dependency is needed.

Primary `whole_task_compute_us` equals actual whole-task wall minus **all measured
explicit H2D/D2H copy walls**, including existing native I/O and candidate bridge
copies. Previous GPU computation is fenced before each copy timer; that waiting,
CPU setup/application/packing, launch costs, D2D copies and remaining synchronization
stay charged. Whole-task wall includes reset/initial plan, feedback solves,
controller, Pinocchio plant and in-memory recording; construction/preparation and
serialization are outside. Actual instrumented whole wall and full subprocess
wall are also retained. Compute subtraction estimates this particular execution
with explicit copy stages removed; it is not an exact prediction of a zero-copy
implementation, since instrumentation/ownership fences and changed scheduling
can still cost time. Native/candidate GPU events are separate from host wall;
CPU-PCG event totals cover GPU matvecs only.

Use explicit device allocations/copies for this cohort. Jetson shared physical
RAM does not automatically remove `cudaMemcpy`; managed runtime migration cannot
be measured as an explicit copy and is deliberately outside this primary cohort.
The inherited CPU bridge measures complete copy walls but does not count its
additional copy bytes/calls; metadata flags that limitation. Zero additional
byte/call counts for CPU methods do not mean zero transfers.

## Inputs and horizon meaning

Nine frozen public cases contain six three-goal reaches (seeds2–7) and three
two-period figure-eights (seeds11/12/13, requested speed ratios0.6/0.8/1.0).
Inputs/scenarios are generated once, with reference padding for the largest N,
and shared across horizons/methods/repeats. Each N's native module hash is pinned.
Feedback/knot dt stays10ms and plant dt1ms. Prediction durations are therefore
1.27/2.55/5.11s at N128/256/512: this is a horizon-length scaling study, not a
fixed-duration discretization study. SQP1, PCG200 and all original protocol costs,
zero-PCG stopping and telemetry-only constraints are retained. The full config
is saved; goal arrival and physical-limit compliance are reported separately.
Tracking duration/error, finiteness, controller-call counts and merits are saved.
No KKT or exact-determinism gate is added. Failed cases stay failed without
retuning/retries and independent scheduled cases continue.

## Obtain the prepared branch

The preparation branch is local and has not been pushed. Import the supplied
Git bundle inside the Jetson container; no remote branch is assumed. In an
existing public GATO clone, fetch it and create an isolated checkout:

```bash
git fetch /path/to/jetson-preconditioner-scaling.bundle \
  felix-precond-timing/jetson-scaling:felix-precond-timing/jetson-scaling
git worktree add ../GATO-preconditioner-scaling felix-precond-timing/jetson-scaling
cd ../GATO-preconditioner-scaling
```

The supplied receipt identifies the final branch commit and bundle hash. Keep
existing local modifications separate; do not reset an occupied branch. The
bundle contains source, not desktop binaries or raw results.

## Build and frozen inputs

Run inside the existing Jetson container from this checkout, using the Python
environment with public GATO dependencies (`grid-rbd==0.5.1`, NumPy, Pinocchio),
pybind11, CMake3.25+ and Eigen3 headers available to the compiler. CUDA and a
C++17 toolchain are required. Architecture defaults to CMake `native`; `--arch 87`
can be supplied explicitly for the Orin target. No desktop paths are hardcoded.
A single compiler job limits Jetson build memory use. Use fresh output paths.

```bash
python research/preconditioners/jetson_scaling.py run \
  --output _expr_results/jetson-scaling-plan --dry-run
python research/preconditioners/jetson_scaling.py build \
  --output _expr_results/jetson-scaling-build
python research/preconditioners/jetson_scaling.py freeze \
  --output _expr_results/jetson-scaling-frozen
```

Build logs/commands, module hashes, source commit/dirty state and frozen input
hashes are retained. Modules are Indy7 float B1, N128/256/512. Native PCG scratch
is about31/62/123KiB respectively (plus static bytes); the exact resource query
is saved per run. N512 requires a device supporting that per-block opt-in carve;
GTX1070 cannot launch it. Preparation checks actual kernel static+dynamic bytes
against normal/opt-in limits and configures opt-in before task timing, without
launching a null-pointer kernel. The existing launch helpers remain intact.
Target resource/finite-solve validation remains pending until Jetson execution.

## Bounded checks, short protocol and full selection

The launcher executes nothing for `--dry-run`; the exact job list and rough budget
are printed. For a first bounded target check, one full frozen reaching episode
per existing method at N128 is explicit:

```bash
python research/preconditioners/jetson_scaling.py run \
  --tasks _expr_results/jetson-scaling-frozen \
  --output _expr_results/jetson-scaling-preflight \
  --knots 128 --cases reach-seed2 --preflight
```

Dummy data round-trip calibration at the smallest/largest N is four episodes,
with no algorithm change. Check that native and dummy transfer-subtracted times
agree reasonably; retain all outcomes and uncertainty rather than requiring
precise equality:

```bash
python research/preconditioners/jetson_scaling.py run \
  --tasks _expr_results/jetson-scaling-frozen \
  --output _expr_results/jetson-scaling-calibration \
  --knots 128,512 --methods normal,dummy --cases reach-seed2 --preflight
```

The clearly named **short protocol** is the default: one frozen reach and one
frozen tracking case, all11 methods, N128/256/512,3 separate process repeats:
**198 episodes**. It retains complete episode durations and alternates method
order by repeat. Review the observed target preflight duration/storage and discuss
this selection with Felix before execution. Preparation does not run it.

```bash
python research/preconditioners/jetson_scaling.py run \
  --tasks _expr_results/jetson-scaling-frozen \
  --output _expr_results/jetson-scaling-short --dry-run
# After reviewing the plan and target preflight budget, run the same command
# with a new output path and without --dry-run.
```

`--full` explicitly requests all9 frozen cases:891 episodes. `--methods`,
`--knots`, `--cases`, and `--repeats` provide visible selections; all exact jobs
are saved, never silently pruned. With no Jetson measurements yet, a deliberately
rough planning allowance is5–30s/process and2–20MiB/process, based on retained
GTX1070N128 successful0.4–0.8s compute plus startup/copies/larger horizons.
Short: roughly17–99min/0.4–4GiB raw; full:74–446min/1.7–17GiB. These are not
measured Jetson predictions; slow/failed cases can exceed them and builds are
additional. The target preflight supersedes these estimates.

Quiet locks prevent overlapping participating builds/runs. During tasks the
launcher records container-visible `tegrastats` or `nvidia-smi` clock samples;
unavailable monitoring is recorded and is not a launch gate. MODE_40W is
user-confirmed, not freshly measured. No `nvpmodel`, `jetson_clocks`, host checks,
power changes, ROS or physical motion are required. Preserve complete raw/failures
and source/build receipts; copy raw results to a separate backup before analysis.
