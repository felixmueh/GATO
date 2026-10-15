# Optional Jetson hybrid QDLDL comparison

This isolated branch selectively adapts backend math from
`123f11d5cc95bec786ecea06d24941964cb21805` (original mixed WIP
`dc62927e9fa43f6328009ee042a4aa7fe2257a56`) onto the published fixed-update
matrix source `04a6ecd`. It does not merge the historical branch or its other
solver/controller changes. The standalone tool build supports the optional
backend; root CMake remains unchanged.

**QDLDL is a hybrid backend:** CPU sparse direct Schur factorization, with GPU
assembly, direction recovery, merit and line search. It still requires CUDA.
It is not a full CPU MPC baseline or an equal-compute replacement for PCG120.
Default PCG arithmetic, stopping and initialization remain unchanged.

`gato/bsqp/qdldl_host.h` retains the historical upper-CSC conversion, float/int32
factorization and full-row residual check after conversion back to solver
precision. Acceptance is `||rhs-A lambda|| <= max(1e-6, pcg_tol*||rhs||)`.
This differs from PCG's initial-residual-scaled stopping. Direct factorization
does not consume the dual initial iterate. Outer primal warm starts remain the
same. There is no new symbolic cache, threading or numerical tuning.

The thin CUDA bridge throws if any lane is rejected, before publishing a new
lambda or executing direction recovery. Accepted direct work occupies an
internal native count slot as 1, preventing the zero-PCG shortcut; the Python
wrapper exposes it as `direct_solve_work`, not PCG iterations. Recorder PCG
iteration/cap summaries are inapplicable and null; measured successful direct
lane solves are counted separately. Failed calls retain the existing worker
error/partial-recording behavior. KKT flags remain separate from direct linear
acceptance and do not certify full nonlinear convergence.

## Isolation and bounded scope

The tool's `--backend PCG` default retains its original build-directory default.
`--backend QDLDL` selects a distinct default directory ending `-qdldl`, and is
bounded to TIAGo N32/B1. Both the resource probe and loaded extension must match
the requested backend. CMake refuses switching backends inside one build root.
The extension names are unchanged, so workers remain separate processes and
build/module roots must remain separate. Resource probes omit the unused PCG
kernel and mark its memory policy inapplicable for QDLDL; remaining GPU resource
metadata does not describe CPU cost.

QDLDL is pinned to `df481005111f2864770a3eccb05ed4120727eff6` in
`cmake/qdldl.cmake`. CMake can fetch this revision or reuse an existing exact
checkout with `-DFETCHCONTENT_SOURCE_DIR_QDLDL=/path/to/qdldl`. No BLAS dependency
is introduced. ARM64/sm87 build and runtime behavior remain unvalidated here.

## CPU/build validation commands

Local checks used `/opt/gato-venv/bin/python3`, x86_64, CUDA compilation target
sm61 explicitly; no GPU execution. Local dependency checkout used below was
verified at the exact library pin. Build/compiler logs are retained locally in
`build/qdldl-validation/` (not tracked).

```sh
cmake -S tests/qdldl_host -B build/qdldl-host \
  -DFETCHCONTENT_SOURCE_DIR_QDLDL=/workspace/GATO-reaching-qdldl/build/qdldl-n64/_deps/qdldl-src
cmake --build build/qdldl-host --parallel 2
ctest --test-dir build/qdldl-host --output-on-failure
/opt/gato-venv/bin/python3 -m unittest discover -s examples/solve_time_heatmap -p 'test_*.py'

cmake -S examples/solve_time_heatmap -B build/solve-time-heatmap-tiago_right-qdldl \
  -DPLANT=tiago_right -DKNOTS=32 -DGATO_LINSYS_SOLVER=QDLDL \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DPython3_EXECUTABLE=/opt/gato-venv/bin/python3 \
  -DFETCHCONTENT_SOURCE_DIR_QDLDL=/workspace/GATO-reaching-qdldl/build/qdldl-n64/_deps/qdldl-src
cmake --build build/solve-time-heatmap-tiago_right-qdldl --parallel 2 \
  --target bsqpN32_tiago_right resources_N32

cmake -S examples/solve_time_heatmap -B build/solve-time-heatmap-tiago_right \
  -DPLANT=tiago_right -DKNOTS=32 -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DPython3_EXECUTABLE=/opt/gato-venv/bin/python3
cmake --build build/solve-time-heatmap-tiago_right --parallel 2 \
  --target bsqpN32_tiago_right resources_N32
```

Both N32 extension/resource targets compiled and linked successfully for PCG
and QDLDL. Neither extension was instantiated and neither executable was run.

The CPU known-system fixture reuses the historical dense off-diagonal/padded
layout test with TIAGo's 14-state block size: max solution error `3.57628e-7`,
relative residual `1.60101e-7`. Singular/nonfinite inputs, invalid tolerance and
asymmetric full rows are rejected; zero RHS is accepted. All 53 CPU tool tests
pass, including unchanged reference/default measurement semantics, fixed300
counting, backend identity/scope, inapplicable PCG fields and partial-data
retention after a mocked direct rejection. This does not test CUDA transfers,
actual current Schur systems, Jetson portability or scientific task behavior.

## Later Jetson steps, pending target build/runtime validation

Review the local commit and publish/provision its exact source before use.
Build the isolated N32 backend using `-DCMAKE_CUDA_ARCHITECTURES=87` with Jetson's
Python/compiler environment. Estimate duration/storage before execution as
required by the Jetson workflow. Check backend/module identity and a bounded
bridge/runtime preflight when GPU access is explicitly available. Preserve
all negatives; stop for setup failures rather than changing solver math.

The proposed smallest scientific invocation, **not executed or authorized by
this implementation note**, is a fresh output cohort using the same frozen
reference segment and budgets:

```sh
python3 examples/solve_time_heatmap/run.py --backend QDLDL --plant tiago_right \
  --gpu-name 'Jetson hybrid CPU QDLDL / GPU remainder' \
  --build-dir build/solve-time-heatmap-tiago_right-qdldl \
  --output test-artifacts/jetson-qdldl-n32-b1-NEW \
  --knots 32 --batches 1 --updates 300 --repeats 2 --max-sqp-iters 1 \
  --horizon-time 0.93 --initialization single --max-solve-ms 0 --timeout 180
```

For comparison preserve existing PCG results; any newly needed PCG cohort uses
its own build/output and the same arguments except backend/labels/paths.
Closed-loop states may diverge: this compares backend behavior and measured full solve
time over a matched reference segment, not identical per-step linear inputs.
No state-rate frontend changes or full N×B sweep are included.
