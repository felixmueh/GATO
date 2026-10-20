# Public upstream whole-reaching worker

This worker reuses the public example12 Indy7 pick-and-place scenario and
`MPC_GATO.run_mpc_goals`, including the production controller's reset, initial
plan, shifted warm starts and feedback solves. The simulated plant is the
upstream Pinocchio RK4 world. No TIAGo code or physical motion is involved.

The starting source is the public Schur contribution
`9b0df42b5ba27a63fc1b7be05568bf59ac61533e`, a single commit over unchanged
upstream `aeacf16e7733df1a17dfd6627f3b5d15c2ba4387`. The independent public
nonfinite retained-PCG-seed contribution is
`ef527349700b2ef614dab5748ab4204d4b84326e`, also based directly on `aeacf16`.
Combining the two requires a separately recorded source pin. Both contributions
modify GATO, with grid-rbd 0.5.1 and its GLASS commit
`9e57178ca146fa1cecae7e27c22260603b993aa6` unchanged.

## Frozen task

`upstream_reach_config.json` explicitly records the existing example12 paper
configuration, extended to N128: B1, seed1 scenario0, three position goals,
step reference, 10 ms feedback/knots, 1 ms simulation, 4 seconds per goal,
5 cm position tolerance, joint-speed L2 below 1 rad/s, and 100 ms dwell.
The initial pose is the paper-ready pose with the generator's seeded 0.05 rad
jitter. The generated initial state and exact goals are saved per run.

The public example explicitly selects `linsys="pcg"` and
`limits="telemetry"`, retaining the upstream native solver parameters. Limit
violations are reported separately from goal arrival. Generic controller
defaults instead select ADMM; at N128 its inner default is `bdsv_factor`,
which would bypass the intended PCG comparison. This worker preserves the
example's explicit paper arm. No position-only orientation gate is invented.

Fixed physical pacing (`pace_by_solve_time=False`) keeps simulation time
independent of the measured implementation speed. Each method still performs
a fresh complete reaching episode; iteration counts and feedback inputs may
differ. No exact-equality or KKT acceptance gate is applied. N128 baseline
task success must be established by the actual preflight, not inferred from
the existing smaller-horizon examples.

## Invocation

Run only in the root-owned quiet GPU window with both established timing/GPU
locks held, sequentially with builds and other GPU tasks. The worker does not
acquire locks or launch a clock monitor: the campaign owner records clocks
during the measured interval and retains the monitor log. Use fresh output
directories; existing output directories are rejected.

For the clean, uninstrumented reference module already built in its own checkout:

```bash
/workspace/GATO-upstream-contributions/.venv/bin/python \
  /workspace/GATO-upstream-precond-base/research/preconditioners/upstream_reach.py \
  --checkout /workspace/GATO-upstream-precond-reference \
  --method native --output <fresh-reference-output>
```

For an experimental extension:

```bash
/workspace/GATO-upstream-contributions/.venv/bin/python \
  /workspace/GATO-upstream-precond-base/research/preconditioners/upstream_reach.py \
  --checkout /workspace/GATO-upstream-precond-base \
  --modulepath <absolute-extension-path.so> --method normal \
  --output <fresh-instrumented-output>
```

The extension is loaded using its filename's Python module name and registered
as `gato.bsqpN128_indy7`. It must retain the native `BSQP_float` runtime-batch
interface and module dimensions. Without `--modulepath`, direct standard
module import deliberately avoids the interface's automatic-build fallback.
The optional API is `set_method(str)`, `prepare()`, `reset_metrics()` and
`metrics()`. Preparation and controller construction precede timing; metrics
reset immediately precedes the task. Metrics export follows timing.

## Timing and retained data

The primary complete-task interval is entry/exit of `run_mpc_goals`, including
the necessary initial plan, every feedback solve, selection, independent
simulation, arrival checks and shared in-memory recording overhead. Printing
is captured in memory; serialization and compression occur after timing.
The native module must provide cumulative **all** H2D/D2H copy wall time as
`metrics()["transfer_wall_us"]` (or under `metrics()["totals"]`). Only then
is `whole_task_compute_us = whole_task_wall_us - all_transfer_wall_us`
reported. Missing measurement produces `null` and `unmeasuredtransfer`.
Native solve-wall and Python solve/controller wall aggregates are also saved,
without pretending that a task-wide transfer counter has their boundaries.
CUDA-event stage metrics remain separately identified in `metrics.json`.

Each directory retains:

- `config.json`: source commit/status, explicit settings, generated scenario,
  URDF/module/driver hashes, native source identity and dependency versions.
- `task.npz`: exact initial state and three goals; copied runner/config sources.
- `solve_records.json`: every attempted call, including initial warmup, full
  native SQP/PCG arrays and timing, line-search steps, row telemetry, initial
  and final merit. Merit includes penalties; it is not a separate objective
  evaluation. KKT flags are retained as descriptive native output only.
- `raw.npz`: per-call state/reference/warm seed/returned plans; all 1 ms plant
  states and applied torques including terminal state. Missing plans on failed
  calls remain explicit through attempted/completed call counts.
- `task_stats.json`, `controller_records.json`, `task_stdout.txt`, native
  `metrics.json` and `summary.json`: original task outcomes/events, separate
  task/limits/finiteness results and errors.

Exceptions preserve collected records before returning nonzero. There are no
retries or solver fallbacks. The limits check uses URDF bounds with 1e-5
comparison tolerance; it does not certify collisions or physical robot use.
Use separate process repeats for uncertainty, retain every outcome, and report
clock-comparable analyses separately. A frozen linear solve or an open-loop
predicted-state sweep is supplementary and is not this whole-reaching task.

Source syntax and CLI-help checks are complete. The root-owned uninstrumented
Schur-contribution N128 preflight reached all three goals in 1.61 s simulated
time: 161 solve calls including the initial plan, 4171 PCG iterations, and all
returned plans finite. URDF speed compliance failed (maximum excess
2.4017 rad/s), as retained separately for this telemetry-only paper task;
position and torque bounds passed. Raw evidence is preserved at
`/workspace/GATO-upstream-precond-reference/_expr_results/native-preflight/data`.
Its 453.64 ms task wall and 332.18 ms native solve-wall totals are unadjusted;
transfers were unmeasured. Instrumented comparison and calibration remain
pending. This is task-arrival evidence, not physical qualification.
