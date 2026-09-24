# What limits GATO on TIAGo?

This study separates hardware launch support, throughput, discretization error,
long-horizon optimization, receding-horizon feedback, and batch correctness.
The shared coordinator is `examples/randomized_multimodal/validation_suite.py`.
`protocol.json` defines the planned matrix. It is **not a completed-results
claim**. Measured, bounded runs are recorded separately under `results/`.

Branch: `experiment/tiago-solver-limits`. The earlier multimodal experiment is
preserved on `experiment/randomized-multimodal-mpc`. Configuration fuzzing and
staggered-cylinder experiments are deferred until a horizon/grid is selected.

## c3po WIP quickstart

Transfer branch: `wip/tiago-solver-limits-c3po`. It contains the complete
experiment source, protocol, build targets and compact GTX 1070 results.
No local `felixnotes/` or `felixtools/` files are required to run the study.
The older raw trajectory archive is optional when collecting new data.

After pushing this branch, on c3po:

```sh
git fetch origin
git switch --track origin/wip/tiago-solver-limits-c3po
```

Inside the existing GPU-enabled GATO environment, from the checkout root, use
its Python interpreter (for the Felix container, `/opt/gato-venv/bin/python`):

```sh
python examples/randomized_multimodal/validation_suite.py \
  --architecture native --jobs 2 \
  --output example_artifacts/solver_limits/c3po_full
```

This builds all seven knot counts for c3po's GPU and runs all seven studies.
It can take a substantial amount of time. Solver latency does not advance the
simulated plant clock. Keep the GPU free of other workloads for useful latency
measurements. The environment/toolkit requirements and staged commands below
also apply to this one-command run.

After interruption, repeat the exact command with `--resume` appended. Keep the
source and protocol unchanged for that run. Child logs, failures, trajectories,
hashes and machine metadata are saved under the chosen output directory; a
nonzero suite exit can mean recorded failed cases and does not discard results.
To send all collected data back, including failures and partial runs:

```sh
tar -czf c3po_solver_limits.tar.gz -C example_artifacts/solver_limits c3po_full
```

Use a new output directory after code or protocol changes. For exact paired
cross-GPU comparisons, use the saved-input workflow below instead of regenerating
inputs; the historical raw archive has its own pilot layout and is not directly
a full-suite `--inputs-root` directory.

## Completed pilot records

- [Numerical pilot](results/numerical_pilot.txt): 36 selected open-loop outputs,
  three exact-dt probes, two effort rescues, and six equal-duration feedback runs.
- [Batch equivalence](results/batch_equivalence_gtx1070.json): full N16 and
  preliminary N32 corresponding-lane checks; larger grids remain unqualified.
- [Resource probe](results/resources_gtx1070.json): actual B1/B16 launches at
  N8 through N512, with explicit N512 opt-in diagnostics. All seven Python
  modules also [import successfully](results/extension_imports_gtx1070.json).
- [Latency smoke](results/latency_smoke_gtx1070.json): matched-input execution
  plumbing only, not a publishable timing benchmark.
- [Raw archive inventory](results/artifact_archive.json): location and checksum
  of the locally preserved numerical/batching archive. Copy that archive
  separately for exact-input transfer; it is not included in Git.

## Questions and controlled variables

| Study | Vary | Hold fixed | What the result can establish |
|---|---|---|---|
| `resources` | N and B | Float precision, compiled source, one diagnostic SQP iteration | Build/import/launch support, shared memory, registers, occupancy; not convergence |
| `latency` | Batch shape and selected N | Same ordered 16 starts, fixed SQP work, one warm-up, repeated measurements | Cost of serial versus batched work on the recorded GPU; not deadline safety |
| `discretization` | N | Physical prediction duration, task, limiting objective, proposal coefficients | Whether finer discretization changes fidelity or optimization |
| `horizon` | N and T together | Exact dt, task, normalized objective, solver settings | Effect of extending prediction; N and T are intentionally coupled here |
| `mpc-interval` | Time executed between replans | N, T, task, solver settings, equal total simulated duration | Feedback/warm-start degradation as update intervals grow |
| `effort` | SQP passes | Task, N, T, initialization, damping | Whether a fixed-work failure recovers with more optimization; failure alone does not prove an inherent limit |
| `batch` | B, lane order, duplicates, neighboring lane difficulty | Exact serialized lane inputs and solver reset state | Per-lane equivalence to independent B1 solves |

The fixed-T and fixed-dt studies complement each other; neither alone separates
all numerical mechanisms. Different horizon lengths also have different total
costs, so raw objective values across T are not a ranking of controller quality.
Compare equal-duration closed-loop outcomes instead.

Default tasks travel 12, 22, and 30 cm from a common arm configuration, with
slightly different goal directions/heights and cylinder sizes. A difficult
short-horizon task is retained. Do not shorten tasks or filter failed cells to
produce a clean table. These are development probes, not a held-out distribution
of robot tasks.

## Reproduce on another machine

Use the project's Python environment with NumPy, SciPy, Pinocchio, Matplotlib,
pybind11 and the CUDA development toolkit. The selected Python interpreter is
passed directly to CMake. CMake >=3.24 is required by this repository. No script
installs packages, edits Docker settings, or operates a robot.

The default build uses `CMAKE_CUDA_ARCHITECTURES=native`; it does not reuse a GTX
1070 binary. Native RTX 5060 compilation needs a Blackwell-capable toolkit (CUDA
12.8 introduced that support). Check the actual `nvcc`, driver, and GPU recorded
in `suite.json`; the CUDA version shown by `nvidia-smi` is not the installed
compiler version. See [NVIDIA's toolkit announcement](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support)
and [GPU capability table](https://developer.nvidia.com/cuda/gpus).

From the repository root, preview the full plan without builds or GPU execution:

```sh
python examples/randomized_multimodal/validation_suite.py \
  --output example_artifacts/solver_limits/c3po_full --dry-run
```

Build all requested knot counts and run the initial resource/batching gates:

```sh
python examples/randomized_multimodal/validation_suite.py \
  --output example_artifacts/solver_limits/c3po_gates \
  --only resources batch
```

Then run numerical studies using the same build:

```sh
python examples/randomized_multimodal/validation_suite.py \
  --output example_artifacts/solver_limits/c3po_numerical --skip-build \
  --only discretization horizon effort
python examples/randomized_multimodal/validation_suite.py \
  --output example_artifacts/solver_limits/c3po_feedback --skip-build \
  --only mpc-interval
python examples/randomized_multimodal/validation_suite.py \
  --output example_artifacts/solver_limits/c3po_latency --skip-build \
  --only latency
```

Omit `--only` to run the entire planned suite serially through one command. This
is a substantial matrix, especially on the GTX 1070. For a bounded pilot, copy
`protocol.json`, reduce its case lists, and supply `--protocol your_copy.json`.
The complete chosen protocol is embedded in the run manifest. Do not call a
reduced pilot the full suite.

Append `--resume` to the **same command and output path** after interruption.
Completed subprocess cases are skipped; failed processes are retried. The
runner refuses changed protocol, source, input-root, or binary hashes in the
same run. After a code/configuration change, use a new output directory.
Use `--jobs`, `--architecture`, or `--build-dir` to override local build choices.
`--reference` adds the costly independent CPU local-reference calculations.
A separate `--build-only` invocation is available; use a separate output folder
if its later experiment selection will differ.

The scripts use subprocess isolation for knot modules (pybind registration can
conflict in one interpreter) and failed CUDA launches (`gpuErrchk` exits the
process). Unsupported shapes stay visible in the manifest and logs. Process
completion alone never means a numerical or scientific pass.

## Exact cross-GPU inputs and provenance

Each batch audit exports `inputs.npz` and matching `inputs.json`, including a
content hash over names, shapes, dtypes, and bytes. Each open-loop cell exports
its actual task/IK result, initial XU, initial state and reference with hashes.
This allows comparison without rerunning inverse kinematics on another machine.

Copy a prior suite output tree to c3po, then use the same protocol/selection with
`--inputs-root /path/to/prior_suite`. The relative study/cell structure must
match. Use a new output directory. Source and input identity should match across
machines; compiled binary hashes and GPU metadata are expected to differ.
Cross-architecture bit identity is not required, but material changes in route,
feasibility, objective, or convergence must be investigated.

`suite.json` records the protocol, source and binary hashes, commit, dirty state,
Python/package versions, GPU/driver/toolchain, every command, process exit,
elapsed wall time and log path. Child results retain lane-level diagnostics and
trajectories. A missing module, a resource-limit launch failure, a numerically
invalid solution, a task failure and a missed latency budget are different
outcomes and must not be merged into a single success count.

Compact measured summaries and their manifests belong in tracked `results/`.
Large trajectories remain in `example_artifacts/`; preserve them externally and
retain their hashes in the compact report. Git preserves code and summaries,
not ignored large artifacts. Local process pauses or shared-GPU contention must
be recorded and excluded from uncontended timing claims.

## Numerical interpretation

Physical horizon is `T=(N-1)*dt`. MPC's execution interval is a separate clock;
the plant advances by that interval regardless of solver wall time. The interval
need not divide dt: a partially executed control segment is integrated only to
the exact update time. Episode comparisons use equal total physical time.

The numerical study scales running terms with dt relative to the old .03 s
normalization, while terminal position and cylinder weights remain fixed. The
velocity sample at the terminal knot belongs to the running quadrature; there
is no separate fixed terminal-velocity penalty. Endpoint quadrature error is
explicitly acknowledged. Goal error and final speed are independently reported.

Keep three different trajectories visible: optimizer state knots, coarse
control replay under its discrete dynamics, and independent fine-step RK4
control replay (1 ms with .5 ms refinement). Full-horizon open-loop drift is a
model/discretization diagnostic. It does not alone prove that a controller
fails when only a short prefix is executed before replanning. Report prefix
accuracy and executed MPC feasibility separately. Fine-replay selection is an
offline diagnostic, not an unreported oracle used by the online controller.

For the current compact float PCG kernel on the seven-joint plant, dynamic
shared memory is `4 * (3 * 14 * (N+2) + 32)` bytes per block, independent of B.
N256 uses 43,472 bytes; N512 uses 86,480, plus compiled static shared memory.
The resource probe queries the actual kernel attributes and device limits. Its
optional `--pcg-opt-in` run changes only the probe process's kernel attribute;
it does **not** modify the production Python solver. This separates the default
CUDA allocation limit from hardware opt-in capacity. See the [CUDA execution API](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__EXECUTION.html).

Batch comparisons use exact corresponding lanes, not just a better batch winner.
They include nested B2/4/8/16, reversed lanes, duplicate lanes, a finite harder
neighbor and solver reuse after reset. Repeated B1 runs estimate repeatability.
Fixed-work runs disable early termination; production-stopping runs separately
expose the solver's batch-wide stopping semantics. Prespecified strict and
material tolerances, per-lane results, and roundoff diagnostics are retained.

Broader research attribution is collected in the local mirrored
`felixnotes/REFERENCES.md` (GATO, KELLY-TO, NVIDIA-CC, NVIDIA-BLACKWELL and CUDA
resource references). Our matrix and diagnostic acceptance rules are engineering
design choices, not results established by those sources.
