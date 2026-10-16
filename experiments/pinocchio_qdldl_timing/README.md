# Jetson Pinocchio + QDLDL CPU-reference timing matrix

**PREPARED, NOT RUN.** The CPU-only backend is implemented. A native C++ QDLDL
bridge built successfully and ten bounded functional CPU checks passed. No
matrix cohort, Jetson experiment, CUDA operation or GPU benchmark was run.

**This measures Python-orchestrated CPU wall time**, including Python loops,
Pinocchio, NumPy, block assembly, QDLDL and merit search. It is a new experimental
CPU reference, not a native GATO CPU build or an optimized/best CPU baseline.
Initialization, plant RK4 integration, recorder I/O and input/output binding
array copies are outside the timer. Existing GPU and hybrid measurements time
native host code instead; this material scope difference belongs in comparisons.

## Frozen study

`matrix_proposal.json` retains every original cell's exact N/B/horizon/budget
arguments from `examples/solve_time_heatmap/nb_cells.json`. Its filename remains
for provenance; implementation status inside is explicit. The full grid is
N16/32/64/128/256/512 × B1/2/4/8/16, two repeats of 300 measured updates, one
excluded single-solve initialization and SQP cap 1. Prediction knots are 30 ms,
control updates 10 ms, plant RK4 steps at most 1 ms. Reference times are exactly
0.01–3.00 s, mode A: solver time is not charged to the simulated plant. There is
no slow-solve cutoff. Each repeat retains the original 180 s worker timeout,
including imports and initialization; slow CPU cells may therefore be incomplete.

`inputs.npz` is the original hybrid N16/B1/R0 input file, shared by all horizons.
Array SHA256s: initial state `6c54a01a71e18d67edae8fda665c28704bc423e820eb559e0e1415901e7ad5dd`;
full reference `476e4225a3b3789279bcecfe83522ff5b47877fbb6998bf6c3a8223bbe6dcb90`.
The 24 s horizontal figure-eight and comfortable-high-clearance start are
unchanged. Every measured solve reanchors the first seed state only; no temporal
warm shift occurs. The previous batch-0 plan is broadcast unshifted to all lanes
following initialization and every update, exactly as the timing tool does.
Rho/drho reset every call; dual state resets once and QDLDL ignores a dual seed.
Every lane is actually assembled and solved independently and serially inside
the whole-batch timer, including identical lanes. No solve-once/copy shortcut.

## CPU implementation and numerical scope

`cpu_solver.py` implements the frozen one-SQP, zero-wrench, position-only path.
Pinocchio ABA/analytic derivatives and frame Jacobians replace GRiD CUDA calls.
The prediction uses native type-2 trapezoidal dynamics, distinct from the RK4
plant. The raw URDF coordinate convention needs no GRiD joint-6 sign remapping.
Position Gauss–Newton Hessians, terminal velocity/limit terms, raw historical
clipped-log limit penalties (including their outside-limit derivative mismatch),
Q-only rho regularization, Schur signs, direction recovery and eight global-min
merit candidates with strict improvement are retained. Running and limit costs
use 0.03/0.008 scaling; terminal position weight remains 80. No C2 continuation,
interior bounds, tuning or current reaching defaults are substituted.

Pinocchio/NumPy assembly is float64. The unchanged `gato/bsqp/qdldl_host.h`
bridge casts Schur/RHS to float32, uses pinned QDLDL
`df481005111f2864770a3eccb05ed4120727eff6`, and checks the full Schur residual
against max(1e-6, pcg_tol × RHS norm) after its float32 solve. This differs from
the original all-float32 CUDA assembly and must not be called identical arithmetic.
An existing optional componentwise KKT shortcut is preserved internally; it is
not a new study gate. With direct QDLDL there is no zero-PCG shortcut. PCG cap
120 remains a frozen-protocol identifier, not QDLDL work telemetry.

Thread-library counts are set to one before numerical imports. Effective CPU
affinity, environment, platform, package versions, source and loaded CPU bridge
SHA256 are recorded. This does not fix affinity or change clocks/power. MODE_40W
is user-confirmed only; no host `nvpmodel`/clock query or setting change is required.
CUDA and Torch are not imported by this CPU path. The CMake project enables
only C and C++; no GPU extension is built or reused.

## Prepare and run manually after publication/review

The root agent owns publication and the pinned command in `/workspace/GATO/commands_felix.md`.
On the target, use a separate worktree. The same command builds the CPU bridge,
runs the bounded functional tests, then measures all fixed attempts and exports:

```sh
python3 experiments/pinocchio_qdldl_timing/run_matrix.py --only cpu-pinocchio-qdldl \
  --raw "$PWD/.experiment-results/nb-matrix-jetson-pinocchio-qdldl-20261016" \
  --cache "$PWD/.experiment-cache/nb-matrix-jetson-pinocchio-qdldl-20261016" \
  --export "$PWD/.experiment-work/nb-matrix-jetson-pinocchio-qdldl-20261016-compact"
```

Append `--dry-run` for the exact 60-attempt plan with no build or measurements.
Append `--prepare-only` for CPU bridge build and bounded tests only; this emits
PREPARED_NOT_RUN and performs no timing cohort. Required existing dependencies
are NumPy, Pinocchio, Matplotlib, pybind11, Git, CMake >=3.24 and a C/C++ compiler.
QDLDL is fetched at its pinned commit; no package install is performed.

`--resume` checks identical source/module/cache and skips every existing fixed
attempt, including failures. An unfinished output directory is an incomplete
attempt, never rerun. Shared setup failures hold dependent work. Individual
numerical/timeout failures preserve their data and independent cells continue.
Interruption stops the active worker and exports flushed partial results.
No source/manifest deletion, automatic retry or outcome tuning occurs.
Run without other Jetson CPU/GPU experiments; the manual command guard is owned
by the root running guide. No parallel timing workers.

The fixed worker budget is at most 60 × 180 s = 3 h, plus build/startup/export.
This is a bound, not a measured duration forecast. A conservative 1 GiB raw plus
separate backup allowance is sufficient for planning; no per-call horizons are
saved. Actual duration/storage and target outcomes remain unknown.

## Records and compact export

Each repeat reuses the original `Recorder` and `run_instrumented` for initialization,
plant evolution, clock, warm starts, NPZ state/timing/work arrays and flushed
`solves.jsonl`. The retained `native_ms` column now stores the CPU solve wall
boundary described above; metadata and report state that explicitly. Initialization
is separately recorded. CPU direct-solve counts cover every lane. Nonfinite
inputs/plans/plant states have nonfinite labels; other execution/setup failures
remain incomplete and keep their tracebacks.

`cells.csv` keeps the requested existing columns. GPU UUID/SM count and PCG-only/
CUDA shared-memory fields are blank for this CPU measurement. CPU bridge SHA is
in module_sha256, and actual frontend/helper hashes are in metadata/per-repeat
results. Partial summary statistics are retained as diagnostics, but only two
completed 300-update repeats make a completed heatmap cell. CPU-only cells do
not borrow GPU SMEM classifications.

Raw goes to `_expr_results/nb-matrix-jetson-pinocchio-qdldl-20261016/` on desktop
with a separate backup. Compact goes to
`thesis_results/nb-matrix-jetson-pinocchio-qdldl-20261016/`: REPORT.md/self-contained
PDF, quicklook, cells.csv, repeat statuses, protocol and provenance. Add
`--export-only` to regenerate against saved raw without executing a solver.
Comparisons to old PCG/hybrid data are descriptive: precision, Python overhead
and evolving closed-loop inputs differ. No equal-work, best-CPU-performance,
task-success, physical safety or real-time-controller qualification is implied.

## Preparation checks

Native CPU CMake build passed with GCC 11.4 and float32/int32 QDLDL. Ten bounded
CPU tests pass: analytic prediction/FK Jacobians, interior cost gradients,
historical outside-limit behavior, known-system direct solve, dense KKT signs,
actual serial two-lane work, a two-update shared-loop/Recorder reset/clock test,
exact grid and incomplete-stream preservation. The shared-loop test also checks
that Torch is absent. These are functional checks, not benchmark results.
Build/test logs remain in `build/pinocchio-qdldl-cpu/` and the preparation receipt.
