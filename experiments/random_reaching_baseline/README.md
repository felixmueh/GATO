# Random-reaching B1 baseline

This isolated study tests the original frozen random reaching tasks with the
public pose-reaching parameterization and a generous SQP cap. It uses the
unchanged production `gato_tiago.tiago_mpc_controller.MPC_GATO` constructor and
solver, plus `bsqp.common` initialization, warm-start shift and RK4 integration.
The numerical source is pinned to `f0cafcde47a675d0deb17724005c3416a327e7db`; experiment-only commits may follow
it. Native module path and SHA256, actual imported source hashes, exact inputs,
effective configuration and original mode-A helper provenance are recorded.

`protocol.json` freezes N32, 30 ms knots and feedback, 3 ms RK4 plant steps,
eight seconds maximum observation, B1 and mode A: solver computation consumes no
simulated plant time. The first solve supplies the first executed control and
is included in all work/timing records. Cold initialization repeats the actual
rest state with zero controls. Subsequent seeds shift the previous optimizer
output and reanchor at the actual state. Duals and rho reset once per episode;
adaptive rho and duals persist across feedback solves, matching public reaching.
There is no unrecorded warmup or final-approach switch.

The initial cap is SQP1000, PCG120, with ordinary zero-PCG stopping enabled.
Public full-pose costs are explicit, including qd `.1`, velocity-limit `.1`,
control-limit `.003` and running/terminal orientation `.1`/`2`. Historical BA
used control-limit `.05` and reset rho every solve, so these are new observations.
The exact reviewed C2 quadratic continuation from candidate `17100c0` is applied in the isolated numerical source, with a 1% limit-interval relaxation width. It remains a soft penalty; no production merge is implied. KKT flags are saved
as descriptive existing outputs and never gate this study.

Run from the worktree root after building and reviewing a matching native N32
module. A CPU dry run validates source/configuration/input hashes without
importing NumPy, Pinocchio, the native module or controller, and writes no files:

```sh
python experiments/random_reaching_baseline/run.py \
  --source /workspace/GATO-random-reaching-continuation \
  --module /workspace/GATO-random-reaching-continuation/python/bsqp/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so \
  --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching \
  --preflight-input /workspace/GATO-experiment-runner-simple/experiments/inputs/batching/ba_preflight.json \
  --output /workspace/GATO/_expr_results/random-reaching-continuation-20261015/preflight \
  --dry-run
```

Remove `--dry-run` for the reviewed canonical preflight. The retained JSON is
used for initial state and goal geometry only. `preflight_effective.json` records
the actual new costs, SQP budget and retained-rho policy. The historical file
remains immutable. Preflight passes only with rough arrival, sampled physical
compliance, at least 100 mm initial error and at least 50 mm executed endpoint
displacement. A failed preflight holds dependent task launches.

For the ordered development pilot, replace `--preflight-input ...` with:

```sh
--pairs 3626 7733 7132 7093 1265 9651 5264 279 8054 8270 2497 2342
```

Use a fresh `--output` directory for every invocation. `--pairs` requires
distinct IDs in the frozen100 roster. The remaining88 are reserved from new
parameter tuning. `--protocol PATH` accepts a new explicitly named configuration
with a complete `solver_params` dictionary; `--max-sqp-iters N` changes only the
cap and appends it to the recorded configuration identifier. Other initial-study
timing, gates and policies remain fixed. No parameter grid or later budget/batch
campaign is launched automatically.

Each episode retains `inputs.npz`, preparation hashes, complete `executed.npz`
timestamps/states/applied torques and per-solve seed/returned horizons. Typed
per-iteration PCG counts, line-search steps/merits and existing flags are stored
once per returned NPZ, including a terminal zero-PCG row with no line search.
`solves.jsonl` contains scalar work/timing and physical diagnostics; there are no
repeated thousand-row statistics in JSON. `metrics.json` and `summary.json`
separate task success, physical compliance and execution status. Original NaNs
remain in numeric artifacts; JSON uses null for nonfinite descriptive values.

Seed and optimized prediction checks cover modeled knots; executed-prefix and
complete executed-motion checks cover RK4 samples. First violations identify
joint, sample and time, with raw maximum excesses retained. Predicted timestamps
refer to the current feedback time plus prediction-grid offsets. Physical
definitions use the URDF limits, including 43 Nm on the first two joints and
26 Nm on the remaining five; no universal26 Nm robot limit is substituted.
Boolean tolerances preserve historical checks: q excess at most `1e-6` rad,
velocity/effort ratios at most `1+1e-6`. These are sampled checks, with no
obstacle or continuous-time collision certificate.

Rough arrival requires position error at most50 mm, tool-axis error at most30°
and joint-speed L1 at most1 rad/s at a30 ms feedback boundary, with zero dwell.
First position/pose-region touches are measured at dense3 ms observations.
Full SO(3) error is recorded separately. Last-second error range/progress and
joint motion help describe hovering; stopping at rough arrival provides no
evidence of subsequent settling or a verified local minimum.

Physical violations are retained and do not trigger clipping, a candidate veto
or an early limit-stop. A finite timeout is separate from a failed execution.
Task-specific execution failures retain partial histories and allow independent
assigned cases to continue. Shared constructor/import failures hold dependent
work. Interruptions retain the current partial episode and stop subsequent cases.
There is no robot or Gazebo execution path in this script.

## Saved pilot analysis

`analyze.py` reads saved recordings only and imports no native solver or
Pinocchio. It requires all twelve assigned metrics and checks them against the
saved summary before exporting figures, compact editable CSVs, a Markdown report
and its self-contained PDF. The separate SQP1 diagnostic retains all failures,
partial histories, enormous finite extrema and nullable nonfinite maxima.

```sh
python experiments/random_reaching_baseline/analyze.py \
  --raw /workspace/GATO/_expr_results/random-reaching-continuation-20261015/pilot/run-001 \
  --comparison-raw /workspace/GATO/_expr_results/random-reaching-continuation-20261015/pilot/sqp1-run-001 \
  --comparison-preflight /workspace/GATO/_expr_results/random-reaching-continuation-20261015/preflight-sqp1/preflight/metrics.json \
  --reference-root /workspace/GATO/_expr_results/random-reaching-baseline-20261015 \
  --preflight /workspace/GATO/_expr_results/random-reaching-continuation-20261015/preflight/preflight/metrics.json \
  --source /workspace/GATO-random-reaching-continuation \
  --output /workspace/GATO/thesis_results/random-reaching-continuation-20261015 \
  --pdf-helper /workspace/GATO/thesis_results/reaching-landscape/report_to_pdf.py \
  --previews /workspace/GATO/_expr_results/random-reaching-continuation-render-check-20261015
```

Full horizons, raw histories, module/build outputs and the backup archive remain
outside the thesis export. The export includes the compact plotting inputs,
analysis script and PDF helper so figures can be adjusted without raw recordings:

```sh
python analyze.py --plot-only .
python report_to_pdf.py --report REPORT.md
```

The `--reference-root` comparison retains the original helper results separately.
It checks matched per-cap settings, input hashes and all twelve initial/goal hashes,
then exports old/new outcome counts, paired gains/losses and per-case goal/limit
changes. Null full-history extrema stay null; finite values are not substituted
for them. The report describes the helper using recorded `limit_penalty` metadata
and derives cohort outcomes and failure counts from the saved records. Original
reports and raw recordings are never overwritten.
