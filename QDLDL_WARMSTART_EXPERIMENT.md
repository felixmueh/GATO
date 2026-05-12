# QDLDL Warmstart Experiment

This repository is a full copy of `/workspace/GATO` made for testing whether PCG behaves better when initialized from QDLDL solutions. Treat this as an experiment branch/worktree: it is fine to add focused test scripts and temporary artifacts here.

## Current State

- The GATO QDLDL backend works on the current `large_easy_tracking` experiment.
- The active compiled Python modules in `python/bsqp/*.so` report whichever
  backend was built most recently; after the 2026-05-07 replay, the Tiago N=32
  module reports `LINSYS_SOLVER=PCG`.
- PCG and QDLDL are selected at CMake configure time with `GATO_LINSYS_SOLVER=PCG|QDLDL`.
- The compiled extension module names are the same for both backends, for example `bsqp.bsqpN32_tiago_right`. Rebuilding one backend overwrites the active `.so` files under `python/bsqp/`.
- Because of that module-name collision, do not try to import PCG and QDLDL backends in the same normal Python process. Use a two-pass disk handoff, or separate processes with careful module loading.

## Relevant Files

- `examples/large_easy_tracking.py`
  - Smooth joint-space waypoint trajectory.
  - Converts waypoints to end-effector references.
  - Runs closed-loop tracking for `tiago_right`, `iiwa14`, and `indy7`.
  - Resets dual and rho every MPC tick:
    - `solver.reset_dual()`
    - `solver.reset_rho()`
  - Initializes warm start each tick with `initialize_warm_start(x, horizon, solver.nx, solver.nu)`.
  - Saves summaries, CSVs, PNGs, and GIFs under `test-artifacts/large_easy_tracking/<plant>/<backend>/`.
- `python/bsqp/interface.py`
  - Wrapper class `BSQP`.
  - `BSQP.solve(xcur_B, eepos_goals_B, XU_B=None)` accepts an explicit warm-start trajectory `XU_B`.
  - The wrapper overwrites `XU_B[:, :nx]` with the current state before calling CUDA.
  - `solver.linsys_solver` reports `PCG` or `QDLDL`.
- `CMakeLists.txt`
  - `GATO_LINSYS_SOLVER` config option.
  - `PLANT` and `KNOTS` CMake options can restrict build scope.
- `gato/bsqp/bsqp.cuh`
  - Backend dispatch:
    - QDLDL path calls `gato::qdldl_linsys::solveQDLDLBatched`.
    - PCG path calls `solvePCGBatched`.
  - `reset_dual()` zeros `d_lambda_batch_`.
  - `reset_rho()` restores initial rho and drho.
- `gato/bsqp/kernels/qdldl_linsys.cuh`
  - Host-side QDLDL solver bridge for the Schur system.

## Baseline Results Already Observed

These were produced by `examples/large_easy_tracking.py` with `N=32`, `DT=0.03`, `SIM_DT=0.003`, `SEGMENT_TIME=3.0`.

Tiago right:

- PCG:
  - mean error: `0.0211032162 m`
  - p95 error: `0.0584229936 m`
  - max error: `0.0909552563 m`
  - final error: `0.0020714603 m`
  - rejected solves: `143`
  - mean PCG iterations: `705.0954`
  - PCG cap-hit fraction: `0.6908`
  - mean SQP time: `115.573 ms`
- QDLDL:
  - mean error: `0.0113208264 m`
  - p95 error: `0.0253416228 m`
  - max error: `0.0280597188 m`
  - final error: `0.0002332198 m`
  - rejected solves: `0`
  - mean SQP time: `9.067694 ms`
  - mean QDLDL solve time: `0.2953738 ms`

For comparison, IIWA PCG did not hit the cap in the same experiment, while Indy7 had moderate PCG trouble. This makes Tiago the most useful plant for the warmstart test.

## QDLDL-to-PCG Warmstart Result

Implemented and ran the exact replay experiment in
`examples/qdldl_pcg_warmstart_experiment.py` on 2026-05-07.

Workflow:

1. Captured the full Tiago `large_easy_tracking` QDLDL closed-loop run:
   - `500` subproblems
   - `0` rejected solves
   - `0` non-finite outputs
   - artifact:
     `test-artifacts/qdldl_warmstart/tiago_right/qdldl_capture.npz`
2. Rebuilt the Tiago N=32 module with `GATO_LINSYS_SOLVER=PCG`.
3. Replayed each captured subproblem with:
   - the same `x_before`
   - the same reference window
   - `XU_B = qdldl_solution`
   - `sigma = 0`
   - dual and rho reset before every solve

PCG replay result:

- Replayed subproblems: `500`
- Linear solves recorded: `10000`
- Mean PCG iterations per linear solve: `879.6204`
- p95 PCG iterations per linear solve: `1000`
- Max PCG iterations: `1000`
- Linear-solve PCG cap-hit fraction: `0.867`
- Subproblem cap-hit fraction: `0.994` (`497/500` subproblems hit the cap at
  least once)
- KKT-converged fraction: `0.0`
- Accepted-step fraction: `0.186`
- Non-finite outputs: `0`
- Mean SQP time: `142.674 ms`
- Mean final merit: `7.64761`
- Mean PCG-vs-QDLDL solution L2 norm: `0.0593993`
- Max PCG-vs-QDLDL solution Linf norm: `1.19021`
- artifact:
  `test-artifacts/qdldl_warmstart/tiago_right/pcg_replay_sigma_0/pcg_replay.npz`

Answer to the first question: no, Tiago PCG in GATO does not converge reliably
when warm-started with the QDLDL solution of the same captured subproblem. The
failure is not just the closed-loop warm-start basin: PCG still hits the linear
iteration cap on almost every replayed subproblem even from the QDLDL solution.

## Build Commands

Build only the Tiago N=32 module for QDLDL:

```bash
cmake -S . -B build-qdldl-tiago-n32 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGATO_LINSYS_SOLVER=QDLDL \
  -DPLANT=tiago_right \
  -DKNOTS=32
cmake --build build-qdldl-tiago-n32 --target bsqpN32_tiago_right -j
PYTHONPATH=$PWD/python python3 - <<'PY'
from bsqp import bsqpN32_tiago_right as m
print(m.LINSYS_SOLVER)
PY
```

Build only the Tiago N=32 module for PCG:

```bash
cmake -S . -B build-pcg-tiago-n32 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGATO_LINSYS_SOLVER=PCG \
  -DPLANT=tiago_right \
  -DKNOTS=32
cmake --build build-pcg-tiago-n32 --target bsqpN32_tiago_right -j
PYTHONPATH=$PWD/python python3 - <<'PY'
from bsqp import bsqpN32_tiago_right as m
print(m.LINSYS_SOLVER)
PY
```

The existing build dirs `build-pcg-all-n32/` and `build-qdldl-all-n32/` are present in this copy, but the active import is always whichever backend most recently wrote `python/bsqp/bsqpN32_tiago_right*.so`.

## Proposed Test

Use a two-pass workflow.

1. QDLDL capture pass:
   - Rebuild QDLDL.
   - Run the Tiago large-easy trajectory.
   - At every MPC tick, save:
     - `x_before`: current state passed to the solver.
     - `reference_window`: flattened `(1, 6*N)` end-effector target window.
     - `warm_start_in`: the initial trajectory passed to QDLDL.
     - `qdldl_solution`: returned `XU` trajectory.
     - QDLDL stats: SQP iterations, step sizes, final merit, solve time.
   - Save to something like `test-artifacts/qdldl_warmstart/tiago_right/qdldl_capture.npz`.

2. PCG replay pass:
   - Rebuild PCG.
   - Load the QDLDL capture file.
   - For each captured tick, solve the exact same subproblem:
     - `xcur_B = x_before`
     - `eepos_goals_B = reference_window`
     - `XU_B = qdldl_solution`
   - Reset dual and rho before each solve unless testing carry-over explicitly.
   - Collect PCG iterations, cap-hit fraction, SQP step acceptance, final merit, and solve time.

3. Perturbed warmstart pass:
   - Repeat PCG replay with perturbations to the QDLDL solution:
     - `sigma = 0.0`
     - `sigma = 1e-5`
     - `sigma = 1e-4`
     - `sigma = 1e-3`
     - optionally `sigma = 1e-2`
   - Do not perturb the first `nx` entries unless intentionally testing current-state mismatch. The Python wrapper overwrites those entries anyway.
   - Perturb only finite entries and keep `float32`.

The first pass answers: "Can PCG converge when it starts essentially at the QDLDL solution of the same nonlinear subproblem?" If yes, the PCG issue is warm-start/conditioning/basin-related. If no, PCG is likely failing on the Schur systems even near a good SQP trajectory.

## Useful Implementation Shape

Add a new script, for example:

```text
examples/qdldl_pcg_warmstart_experiment.py
```

Recommended CLI:

```bash
PYTHONPATH=$PWD/python python3 examples/qdldl_pcg_warmstart_experiment.py capture --plant tiago_right --N 32
PYTHONPATH=$PWD/python python3 examples/qdldl_pcg_warmstart_experiment.py replay --plant tiago_right --N 32 --sigma 0
PYTHONPATH=$PWD/python python3 examples/qdldl_pcg_warmstart_experiment.py replay --plant tiago_right --N 32 --sigma 1e-3
```

Keep the trajectory generation and plant config identical to `examples/large_easy_tracking.py`. The cleanest approach is to import or lightly refactor helpers from that file rather than duplicating all constants by hand.

## Metrics To Report

For each PCG replay variant:

- number of replayed subproblems
- mean, p95, and max PCG iterations
- PCG cap-hit fraction
- SQP accepted-step fraction
- mean, p95, and max SQP solve time
- mean final merit
- count of non-finite outputs
- optional comparison of `PCG_solution` vs `QDLDL_solution` norm

Also save per-tick CSV/NPZ data so we can inspect where cap hits occur along the trajectory.

## Important Caveats

- This test is about the GATO implementation, not MPCGPU. Do not assume MPCGPU core bugs here.
- The current GATO Tiago shared-memory overlap bug has already been fixed in this copy:
  - `gato/dynamics/tiago_right/tiago_right_plant.cuh`
  - `grid::DEE_POS_SHARED_MEM_COUNT = DEE_POS_DYNAMIC_SHARED_MEM_COUNT + 16`
  - `s_temp = s_dXmatsHom + 128`
- The large-easy experiment is intentionally smooth in joint space. It is not an IK or discontinuous pose-hopping test.
- The current experiment resets dual and rho every tick. If a future test carries duals across ticks, report that separately because it changes the question.
- `pcg_iters` is also used as a placeholder stats field for QDLDL and is expected to be `1` for QDLDL in the current bridge.

## Suggested Next Decision

Start with replaying the exact QDLDL-captured subproblems with `sigma=0`. That gives the highest-signal answer before spending time on closed-loop variants.
