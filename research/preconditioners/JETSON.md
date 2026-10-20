# Portable Jetson experiment entry point

The user runs this experiment branch manually inside the existing container.
No Jetson execution was performed during preparation. Activate the environment
that provides the public GATO build dependencies, NumPy and Pinocchio. Commands
below use its `python` and discover CMake/pybind11/CUDA through that environment;
there are no desktop checkout, venv or CUDA architecture paths to replace.

This launcher must be integrated with the explicit managed-memory experiment
source before building. It checks for `PRECOND_MANAGED_S` support and requests
that option, rather than assuming that CPU/GPU shared physical memory makes
ordinary CUDA copies disappear. The primary algorithm comparison is:

- `normal`: original native GPU PCG, original `cudaMalloc` Schur S.
- `block_ldlt_shared`: CPU block LDLT setup reading explicitly managed Schur S,
  followed by GPU triangular solves reading explicitly managed factors inside
  PCG. This algorithm combines CPU exact factors with GPU triangular solves.
- `gpu_bdsv`: existing public GPU direct BDSV, native method `normal` and
  `linsys="bdsv"`, with original `cudaMalloc` S. Its iteration fields are
  direct-solve bookkeeping, not CG iterations.

Optional controls are `normal_shared` (GPU PCG with managed S) and `block_ldlt`
(explicit-copy CPU-setup/GPU-apply control). The shared API records device
capabilities, requested/actual allocation and verified pointer attributes.
Unsupported shared requests fail; there is no silent copy fallback. A device
with `concurrentManagedAccess=0` uses explicit idle ownership fences. Physical
sharing and zero migration are not inferred from an allocation API name; any
faults, migration, cache maintenance and synchronization inside an episode
remain charged to actual wall time.

## Build and freeze inputs

From the experiment checkout, using new local output directories:

```bash
python research/preconditioners/jetson_experiment.py build \
  --output _expr_results/jetson-build
python research/preconditioners/jetson_experiment.py freeze \
  --output _expr_results/jetson-frozen
```

The build defaults to N128 Indy7, one compiler job and `--arch native`.
`--arch 87` is an explicit alternative if the installed CMake cannot detect
the architecture. Build logs, exact configuration and resulting binary hash
are retained. No fast-math, solver parameter or task default is overridden.
Freeze uses only the existing public CPU scenario generators; all nine cases
and full padded tracking references are saved once. All methods/repeats then
read those exact assets, without repeated IK generation.

## Explicit short preflight and selected/full campaigns

```bash
python research/preconditioners/jetson_experiment.py run \
  --tasks _expr_results/jetson-frozen --output _expr_results/jetson-preflight \
  --preflight
```

The explicit preflight is **six separate processes**: the first frozen
reaching case and first tracking case, once per primary algorithm. It retains
the full task parameters and two-period tracking duration. Its schedule is
printed/saved; it does not silently shorten tasks. The full Jetson default is
nine cases × three algorithms × three repeats = **81 processes**, which may be costly
on Jetson. Review preflight duration/storage and discuss a shorter selection
before executing a larger Jetson campaign. This preparation does not execute
or implicitly authorize that larger campaign.

An explicit selected campaign, preserving three repeats per algorithm:

```bash
python research/preconditioners/jetson_experiment.py run \
  --tasks _expr_results/jetson-frozen --output _expr_results/jetson-selected \
  --cases reach-seed2,tracking-seed11 --repeats 3
```

Other frozen case IDs are `reach-seed3` through `reach-seed7` and
`tracking-seed12`, `tracking-seed13`. To include allocation/copy controls,
pass `--methods normal,normal_shared,block_ldlt,block_ldlt_shared,gpu_bdsv` explicitly.
Failures remain failed and all independent scheduled cases continue. There
are no retries, retuning, physical robot actions, KKT gates or strict tracking
error gates. Tracking errors and physical limits remain separate observations.

## Timing and operation

`primary_task_time_us` is **actual whole-task wall**. Nothing is subtracted.
The module's end-to-end mode is selected before preparation/construction so
extra transfer/event instrumentation is disabled where the experiment API
supports it. One-time module/controller setup is outside the episode, reported
as `episode_setup_wall_us`; full subprocess duration, including startup and
serialization, is recorded separately in campaign status. Every episode still
includes controller reset and the required initial-plan solve. Per-call and
per-physical-simulation-second descriptors expose changes in reaching work;
tracking requests exactly two periods with its actual call count retained.
Terminal output prominently reports whole-episode milliseconds, total solve
calls split into initial/feedback calls, wall milliseconds per solve, native
solve-wall median/p95, simulated duration and algorithm. Task wall per call
includes plant/controller work and is not a CUDA kernel or SQP-only time.

The launcher obtains container-local GPU/timing locks with a bounded 45 s wait.
It records clocks during execution using available `tegrastats` or `nvidia-smi`,
with host observation timestamps. Missing monitor access is recorded, not a
host configuration gate. Samples do not prove constant clocks. CPU BLAS/OpenMP
thread counts are explicitly one. The raw recorders save every plan/input and
all 1 ms plant states, so disk use scales with feedback calls and task duration.
All outputs stay in the selected local result directory.

Jetson MODE_40W is **user-confirmed, not freshly measured**. No host power-mode
or clock checks, mode changes or clock-setting commands are requested.
