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
bounded to the same TIAGo N16/32/64/128/256/512 × B1/2/4/8/16 grid as the timing matrix. Both the resource probe and loaded extension must match
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

## Full matched Jetson matrix preparation

The optional existing driver now takes `--backend QDLDL`; PCG remains its default.
`nb_cells.json` is byte-identical, SHA256
`1dd1c88734f48dc59074a6bc2d1adfa542170060bc0c4fd817ef7b11665f1446`.
All30cells retain300 measured updates, two repeats, SQP1, .03s knot spacing,
.01s control interval/.001s maximum integration step, single excluded startup,
no measured cutoff, worker timeout180s and seed17092026 (+repeat index).
The matrix does not charge solve wall time to plant advancement: native solve
latency is measured while reference/state updates advance by fixed10ms.

The host block/CSC format is not tied to PCG shared memory. At N512 with14states,
Schur dimension is7168 and upper CSC storage153916 entries, within int32 indexing.
CPU format dimensions were inspected and the N512 extension (all registered batch
templates, including B16) plus resource executable compiled/linked for explicit
sm87 on x86_64; neither was executed. All55CPU tool tests pass, and a saved-only
copy/export of the actual returned N32/B1 archive preserves its600samples,
source/module/GPU identities and hashes. Logs: `build/qdldl-matrix-validation/`.
This is not an ARM64 build or a new target runtime qualification.
Batch lanes use the same host math serially; no solver/native/binding changes are
needed for this extension. GPU assembly/recovery/etc still have their own resource
requirements, so probes may mark an unsupported cell without launching it.
Unused PCG resources must not reject a QDLDL cell. Direct CPU work is serial across
lanes; this is a hybrid full-batch comparison, not CPU parallel throughput.

`--reuse-n32-b1 ORIGINAL_RUN_ROOT` validates the existing completed matching
cohort and copies its entire folder unchanged to `RAW/_retained/n32-b1/`.
Original manifest, receipts, streams and samples retain recorded hashes and their
own source/module identities. The driver skips both N32/B1 workers. The combined
view covers60repeat slots:58 newly planned slots and2retained repeats; new-results
records do not pretend to include the old measurements. `cells.csv` and repeat
rows preserve the retained source18859b414eab3fff87bd58fd0c4abc70327634b3 and
module1f66c7b1382f9c01ec577955c44e3743ef46016cce504061456fa0f5833d35c3.
Export regeneration reads the archive-local retained files, not an external path.
The original cohort itself remains untouched. Each new repeat gets one attempt;
resume preserves failed/interrupted records, with no automatic retry.

Completed N32/B1 evidence has600 measured samples, mean2.780268ms. Its inspected
raw folder occupies3,173,975bytes including source snapshots. Preparation itself
runs no GPU/native benchmarks. New horizons/batches require target build/runtime
validation; do not label them scientifically qualified from CPU/build checks.

### Ready command strategy (not executed here)

Root will publish the reviewed source and write machine commands. On Jetson,
use a fresh raw/export root and a separate QDLDL build cache:

```sh
python3 examples/solve_time_heatmap/nb_matrix.py --only nb-matrix --gpu jetson \
  --backend QDLDL --raw .experiment-results/jetson-qdldl-nb-NEW \
  --cache .experiment-cache/jetson-qdldl-nb-NEW \
  --export thesis-results/jetson-qdldl-nb-NEW \
  --reuse-n32-b1 /workspace/GATO-jetson-qdldl/.experiment-results/jetson-qdldl-n32-b1-20261015 \
  --dry-run
```

After source review and target-readiness decisions, the same invocation without
`--dry-run` builds per-horizon shared modules for sm87, probes resources without
kernel launches, and schedules sequential workers. Failed independent cells keep
their data while others continue; a shared build/probe failure holds dependent
measurement. `--resume` reuses the same source/config/modules and retained hashes.
No edits to existing PCG or N32 result directories are needed.

Unmeasured planning estimates:58workers ×180s gives174minutes of worker-timeout
ceiling plus build/probe/export time. Actual runtime growth with N and B is unknown;
the N32/B1 mean alone must not be extrapolated as a measured prediction. Larger
serial batches may hit the unchanged timeout and retain partial observations.
Existing per-repeat streams are~228KB, suggesting~14MB of new measured JSON streams;
allow~100–300MB raw including compressed samples/snapshots, conservatively1GB,
excluding source/build caches and backup/archive overhead. Reassess target free
space/build time before execution; no silent timeout or budget changes.
