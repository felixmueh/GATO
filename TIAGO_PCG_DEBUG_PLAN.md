# Tiago PCG Debug Plan

This note records the next debugging path after the QDLDL-to-PCG warmstart
experiment.

## Current Observation

Tiago PCG does not converge reliably even when replaying the exact same
subproblems captured from QDLDL and using the QDLDL solution as `XU_B`.

The `sigma=0` replay result:

- Replayed subproblems: `500`
- Linear solves recorded: `10000`
- Mean PCG iterations per linear solve: `879.6204`
- p95 PCG iterations: `1000`
- Linear-solve cap-hit fraction: `0.867`
- Subproblem cap-hit fraction: `0.994`
- KKT-converged fraction: `0.0`
- Accepted-step fraction: `0.186`
- Non-finite outputs: `0`

This makes the issue look like a Tiago Schur-system or GPU PCG issue, not just
a closed-loop MPC warm-start issue.

## Why IIWA Can Still Work

IIWA and Tiago do not produce the same linear systems. The same PCG kernel can
work on IIWA if those Schur systems are sufficiently SPD, well scaled, and well
matched to the current preconditioner, while failing on Tiago if the Tiago
systems are indefinite, badly conditioned, poorly scaled, or exposing a
plant-specific layout/derivative bug.

The useful comparison is therefore not solver-level behavior alone. Dump one
representative IIWA Schur system and one representative Tiago Schur system and
compare their matrix properties directly.

## First Principle

Reduce the problem before instrumenting deeply:

1. Avoid closed-loop MPC.
2. Avoid rolling horizons.
3. Avoid moving references.
4. Avoid non-obvious warm starts.
5. Start with one static single-solve trajectory optimization task.

If Tiago PCG breaks on a static current-pose hold task with a
gravity-compensated warm start, the problem is very local. If that works, add
one small source of difficulty at a time.

## Reduction Ladder

Run a single Tiago solve with `examples/tiago_single_to_smoke.py`:

1. `hold`
   - Start at `TIAGO_RIGHT_START_CONFIGS["comfortable"]`.
   - Target is the current tool position at every knot.
   - Warm start repeats the current state.
   - Controls are gravity compensation at the current pose.
2. `small-x`
   - Same state and warm start.
   - Target is current tool position plus `1 cm` in local x.
3. `small-z`
   - Same state and warm start.
   - Target is current tool position plus `1 cm` in local z.
4. Larger offsets only after the above are understood.

For each case, run both PCG and QDLDL with the same solver parameters and
compare:

- PCG iteration distribution
- cap-hit fraction
- accepted line-search steps
- initial/final merit
- output finiteness
- first control
- final predicted tool position

## Initial Single-Solve Results

Implemented the reduction as `examples/tiago_single_to_smoke.py` and ran the
first two cases on 2026-05-07.

### `hold`

This is the most boring possible Tiago task:

- current comfortable pose
- constant current tool-position target at every knot
- gravity-compensated static warm start
- no MPC rollout
- one solver call

PCG:

- Initial tool error: `0.0 m`
- Final predicted tool error: `0.0 m`
- Accepted steps: `0/20`
- Line-search failures: `20/20`
- PCG iterations:
  `[17, 12, 21, 14, 116, 1000, 1000, ...]`
- PCG cap-hit fraction: `0.75`
- Output stayed finite
- Artifact:
  `test-artifacts/qdldl_warmstart/tiago_right/single_to_hold_pcg.json`

QDLDL:

- Same unchanged trajectory and rejected line searches, as expected for an
  already-satisfied static target
- Linear-solver placeholder iterations: all `1`
- PCG/QDLDL cap-hit fraction: `0.0`
- Output stayed finite
- Artifact:
  `test-artifacts/qdldl_warmstart/tiago_right/single_to_hold_qdldl.json`

Interpretation: even a solved static hold case can produce Tiago Schur systems
where GPU PCG starts hitting the cap. This is useful for linear-system
debugging, but not a good behavioral optimization test because the initial
trajectory already satisfies the end-effector target.

### `small-x`

This is the smallest useful nonzero tracking task:

- same static pose and gravity-compensated warm start
- constant target shifted by `+1 cm` in local x
- no MPC rollout
- one solver call

QDLDL:

- Initial tool error: `0.00999999 m`
- Final predicted tool error: `0.00586651 m`
- Accepted steps: `7/20`
- Linear-solver placeholder iterations: all `1`
- Output stayed finite
- Artifact:
  `test-artifacts/qdldl_warmstart/tiago_right/single_to_small_x_qdldl.json`

PCG:

- Initial tool error: `0.00999999 m`
- Final predicted tool error: `0.00999999 m`
- Accepted steps: `0/20`
- Line-search failures: `20/20`
- Mean PCG iterations: `804.35`
- p95 PCG iterations: `1000`
- PCG cap-hit fraction: `0.8`
- Output stayed finite
- Artifact:
  `test-artifacts/qdldl_warmstart/tiago_right/single_to_small_x_pcg.json`

Interpretation: Tiago PCG breaks on a one-shot, non-MPC, 1 cm tracking task.
QDLDL can make progress on the same task. The next debugging step should dump
the Tiago Schur system from this `small-x` PCG case, starting with the first
SQP iteration that hits the cap.

## Schur Dump Findings

Added a debug Schur dump path in `gato/bsqp/bsqp.cuh` and a CPU analyzer in
`tools/analyze_schur_dump.py`.

Debug environment variables:

- `GATO_DEBUG_DUMP_SCHUR_DIR=/path/to/output`
- `GATO_DEBUG_DUMP_SQP_ITER=<iter>` to restrict to one SQP iteration
- `GATO_DEBUG_DUMP_ALL_SCHUR=1` to dump non-cap iterations too

For the default `small-x` PCG case with `u_cost=1e-6`, the first cap hit is
SQP iteration `2`.

Analysis of that cap-hit Schur system:

- Matrix dimension: `448`
- Values are finite
- Worst raw Schur entry: about `6.38e10`
- The largest entries are the main diagonal local `(13, 13)` at nearly every
  knot, i.e. the last velocity-state coordinate
- Symmetric-part eigenvalues are all negative:
  - min: about `-6.39e10`
  - max: about `-6.15e-3`
- Absolute condition estimate: about `1.04e13`
- Direct CPU solve succeeds with tiny residual
- GPU PCG after `1000` iterations has a residual L2 around `3278`
- Artifact:
  `test-artifacts/qdldl_warmstart/tiago_right/schur_debug_small_x_pcg_trace/schur_dump_0_sqp2_analysis.json`

The CUDA PCG scalar trace shows that, on this ill-conditioned float32 system,
warp-level accumulation order materially changes `rho`, `p^T A p`, and
`alpha`. Emulating the CUDA warp reduction in Python matches the GPU trace.
This points to single-precision arithmetic sensitivity in the Schur solve,
rather than a bad dump layout.

## Control-Cost Probe

The default Tiago examples use `u_cost=1e-6`. Running the same one-shot
`small-x` PCG task with only `u_cost` changed gives:

- `u_cost=1e-5`
  - cap-hit fraction: `0.0`
  - mean PCG iterations: `8.375`
  - accepted steps: `0`
  - KKT-converged: `1`
- `u_cost=1e-4`
  - cap-hit fraction: `0.0`
  - mean PCG iterations: `9.0`
  - accepted steps: `1`
  - final predicted tool error: `0.00946 m`
- `u_cost=1e-3`
  - cap-hit fraction: `0.0`
  - mean PCG iterations: `7.35`
  - accepted steps: `12`
  - final predicted tool error: `0.00906 m`

The `u_cost=1e-4` SQP-0 Schur dump shows:

- Worst raw Schur entry drops to about `6.38e8`
- Symmetry L-inf drops from `256` to `2`
- Absolute condition estimate drops from about `1.04e13` to `1.89e11`
- GPU PCG exits in `19` iterations
- Artifact:
  `test-artifacts/qdldl_warmstart/tiago_right/schur_debug_small_x_pcg_ucost_1e-4/schur_dump_0_sqp0_analysis.json`

Current working hypothesis: the Tiago PCG failure is primarily numerical. The
combination of Tiago dynamics and the very small control cost (`u_cost=1e-6`)
creates a huge Schur scale in the last velocity coordinate. QDLDL handles the
system directly, but single-precision PCG is too sensitive to accumulation
order and conditioning. Increasing control regularization by two or three
orders of magnitude makes the same single-solve task PCG-friendly.

## Large-Easy Tracking Retest

Extended `examples/large_easy_tracking.py` with solver-parameter CLI overrides
and `--no-media` so control-cost probes can run without rendering plots/GIFs.

Ran Tiago N=32 PCG on the full `large_easy_tracking` trajectory:

| `u_cost` | mean err [m] | p95 err [m] | max err [m] | final err [m] | rejected | mean PCG iters | cap fraction | mean SQP [ms] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-6` baseline | `0.021103` | `0.058423` | `0.090955` | `0.002071` | `143` | `705.10` | `0.6908` | `115.57` |
| `1e-5` | `0.011321` | `0.025343` | `0.028110` | `0.000793` | `0` | `13.38` | `0.0` | `24.83` |
| `1e-4` | `0.011269` | `0.025835` | `0.029055` | `0.000457` | `0` | `18.14` | `0.0` | `28.24` |
| `1e-3` | `0.012879` | `0.027495` | `0.048407` | `0.001633` | `0` | `24.70` | `0.0` | `26.57` |

Artifacts:

- `test-artifacts/large_easy_tracking/tiago_right/pcg_ucost_1em05/`
- `test-artifacts/large_easy_tracking/tiago_right/pcg_ucost_0p0001/`
- `test-artifacts/large_easy_tracking/tiago_right/pcg_ucost_0p001/`

Conclusion: yes, the PCG cap issue also disappears on the full large Tiago
example when `u_cost` is increased. The smallest tested successful value is
`1e-5`, and it is the best of the tested values on this trajectory because it
removes cap hits/rejections while keeping tracking close to the QDLDL result.
The timing numbers are useful for direction but not benchmark-grade because
other agents may share the GPU and the build currently includes debug-dump
hooks guarded by environment variables.

## Schur-System Dump

If the single-solve task still causes cap hits, dump the linear system around
the first cap-hit SQP iteration.

Add a debug-only dump path in `gato/bsqp/bsqp.cuh` after
`formSchurSystemBatched(...)` and before/after `solvePCGBatched(...)`.

Dump at least:

- `S` from `schur_system_batch_.d_S_batch`
- `gamma` from `schur_system_batch_.d_gamma_batch`
- `P_inv` from `schur_system_batch_.d_P_inv_batch`
- initial `lambda` from `d_lambda_batch_`
- final `lambda` after PCG
- `d_pcg_iterations_`
- plant, horizon, tick/case, SQP iteration

Prefer an environment variable or compile-time debug macro so normal builds do
not write large artifacts.

## CPU Analysis

For each dumped system:

- Check finite values.
- Check ranges and norms of `S`, `gamma`, `P_inv`, and `lambda`.
- Reconstruct the padded block-tridiagonal matrix into a dense CPU matrix.
- Check symmetry: `||S - S.T||`.
- Check eigenvalues.
- Estimate condition number when SPD.
- Try dense direct solve.
- Run SciPy `cg`, `minres`, and `gmres`.
- Compare residuals for:
  - QDLDL solution
  - GPU PCG solution
  - CPU direct solution
  - CPU iterative solutions

Interpretation:

- If `S` is indefinite or strongly asymmetric, PCG is mathematically the wrong
  method for that system.
- If `S` is SPD but extremely ill-conditioned, scaling or preconditioning is
  the likely problem.
- If CPU CG converges but GPU PCG does not, suspect GPU PCG implementation,
  padded layout, or preconditioner application.
- If CPU CG also fails while QDLDL works, suspect conditioning or SPD
  assumptions rather than a CUDA bug.

## GPU PCG Residual Trace

If the matrix looks PCG-appropriate on CPU, instrument
`gato/bsqp/kernels/pcg.cuh` for a single solve and record:

- initial residual norm
- per-iteration `rho`
- per-iteration `rho_new`
- per-iteration `p^T A p`
- per-iteration `alpha`
- per-iteration relative residual test value
- final residual norm
- exit reason

This distinguishes:

- slow convergence
- stagnation
- divergence
- non-SPD breakdown
- NaN/Inf contamination
- bad stopping criterion

## Preconditioner Checks

Inspect `P_inv` separately:

- finite values
- diagonal sign/range
- symmetry where expected
- norm of `P_inv * r`
- whether preconditioning improves or worsens condition number on CPU

The current preconditioner may be adequate for IIWA and ineffective for Tiago.

## IIWA Comparison

Repeat the same static single-solve and Schur dump for IIWA using an equivalent
current-pose hold task.

Compare Tiago vs IIWA:

- Schur symmetry error
- eigenvalue spread
- condition number
- `gamma` norm
- preconditioned residual behavior
- CPU CG iteration count

This comparison should explain whether IIWA works because its systems are
better conditioned or because Tiago exposes a plant-specific bug.
