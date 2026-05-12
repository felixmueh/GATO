# Bugfixes

This file separates likely real repository fixes from local-environment
workarounds discovered on the `bugfixes` branch and in the current uncommitted
changes.

## Likely Real Fixes

These are the changes most worth keeping when rebuilding cleanly in Docker.

### 1. Discover `pybind11` from the active Python environment

Commits:

- `f3f22cd`

Why it likely matters:

- the original build assumed `pybind11Config.cmake` was globally available
- host-side builds worked reliably once CMake asked the active Python
  environment for the `pybind11` CMake directory

Relevant file:

- [CMakeLists.txt](/home/felix/Uni/GATO/CMakeLists.txt:33)

### 2. Keep CUDA error checks enabled in release builds

Commits:

- `f3f22cd`

Why it likely matters:

- disabling `gpuErrchk` in release mode hid real CUDA failures behind generic
  crashes
- leaving CUDA error checks on made the later solver bugs diagnosable

Relevant file:

- [gato/utils/cuda.cuh](/home/felix/Uni/GATO/gato/utils/cuda.cuh:7)

### 3. Fix Python tracking/reference indexing and timing

Commits:

- `00198cf`

Why it likely matters:

- the benchmark/controller path was reading the wrong goal slice
- it was also mixing GPU solve time and wall-clock loop timing

Relevant files:

- [python/bsqp/common.py](/home/felix/Uni/GATO/python/bsqp/common.py:106)
- [python/bsqp/mpc_controller.py](/home/felix/Uni/GATO/python/bsqp/mpc_controller.py:186)

### 4. Fix BSQP merit/integrator shared-memory bugs

Commits:

- `bfc4e01`

Why it likely matters:

- this was a real handwritten CUDA solver bug, not just a benchmark issue
- the integrator workspace overlapped in shared memory
- the merit kernel also under-allocated dynamic shared memory for the defect
  path
- re-validation on a GTX 1070 (`sm_61`) showed this is separate from the
  wrong-architecture build problem: even after compiling for `sm_61`, the
  strict CUDA checks exposed invalid shared-memory writes in the merit defect
  path
- the plant wrappers also need to report enough forward-dynamics scratch memory
  for their generated helpers; do this in handwritten wrappers rather than
  editing generated `grid.cuh` files

Relevant files:

- [gato/dynamics/integrator.cuh](/home/felix/Uni/GATO/gato/dynamics/integrator.cuh:218)
- [gato/bsqp/kernels/merit.cuh](/home/felix/Uni/GATO/gato/bsqp/kernels/merit.cuh:94)
- [gato/dynamics/indy7/indy7_plant.cuh](/home/felix/Uni/GATO/gato/dynamics/indy7/indy7_plant.cuh:175)
- [gato/dynamics/iiwa14/iiwa14_plant.cuh](/home/felix/Uni/GATO/gato/dynamics/iiwa14/iiwa14_plant.cuh:182)

### 5. Tiago joint-6 sign mapping is a real integration requirement

Commits:

- `abe4b8a`
- `d153f48`

Why it likely matters:

- the GRiD-compatible Tiago arm export normalizes joint 6 from axis `0 0 -1`
  to `0 0 1`
- that changes the scalar joint convention in the generated dynamics model
- the wrapper must flip joint 6 consistently for `q`, `qd`, `u`, `qdd`, and
  the relevant derivatives

Relevant file:

- [gato/dynamics/tiago_right/tiago_right_plant.cuh](/home/felix/Uni/GATO/gato/dynamics/tiago_right/tiago_right_plant.cuh:10)

### 6. Tiago fixed-target helper calls needed explicit shared-memory management

Commits:

- `d153f48`

Why it likely matters:

- naïvely calling the generated fixed-target `*_device_arm_right_tool_joint`
  helpers from inside solver kernels caused shared-memory collisions
- the stable fix was to call the corresponding `_inner` helpers with scratch
  memory controlled by the wrapper

Relevant file:

- [gato/dynamics/tiago_right/tiago_right_plant.cuh](/home/felix/Uni/GATO/gato/dynamics/tiago_right/tiago_right_plant.cuh:160)

### 7. Tiago derivative debugging surface is worth keeping

Commits:

- `d153f48`

Why it likely matters:

- the added debug kernels made it possible to distinguish rollout bugs from
  tracking-derivative bugs quickly
- this is reusable infrastructure for future plant integrations

Relevant files:

- [gato/bsqp/kernels/debug.cuh](/home/felix/Uni/GATO/gato/bsqp/kernels/debug.cuh:1)
- [python/bsqp/interface.py](/home/felix/Uni/GATO/python/bsqp/interface.py:246)
- [examples/tiago_debug_checks.py](/home/felix/Uni/GATO/examples/tiago_debug_checks.py:1)

## Likely Setup Artifacts Or Local Workflow Changes

These changes may still be useful, but they should be re-justified in a clean
Docker rebuild instead of being assumed correct by default.

### 1. `CMAKE_CUDA_ARCHITECTURES=native`

Commits:

- `f3f22cd`

This is good for local iteration speed, but it is not a generally portable repo
default.

Relevant file:

- [CMakeLists.txt](/home/felix/Uni/GATO/CMakeLists.txt:24)

### 2. Reduced default build matrix in `tools/build.sh`

Commits:

- `f3f22cd`

This is a local-development convenience, not a solver bug fix.

Relevant file:

- [tools/build.sh](/home/felix/Uni/GATO/tools/build.sh:8)

### 3. CUDA 13.2 compatibility around `memoryClockRate`

Commits:

- `f3f22cd`

This is a real compatibility tweak for newer CUDA toolkits, but it is only
relevant if the clean environment uses a toolkit where that field is gone.

Relevant file:

- [gato/utils/cuda.cuh](/home/felix/Uni/GATO/gato/utils/cuda.cuh:28)

### 4. Tiago tracking tuning and the `comfortable` start pose

Commits:

- `d153f48`
- current uncommitted `python/bsqp/config.py` change

This is useful example tuning, but it is not evidence of a solver bug by
itself. The key takeaway is that the original Tiago demo default started too
close to a joint limit and under-penalized velocity/limit violations.

Relevant file:

- [python/bsqp/config.py](/home/felix/Uni/GATO/python/bsqp/config.py:34)

## Re-Check Carefully

These areas contain useful ideas but still need cleanup or re-validation.

### Runtime compute-capability guard

Commits:

- `ade3c02`

The underlying motivation is real: unsupported GPU/toolkit combinations can
produce misleading failures. But the current implementation should be revisited.

Problems:

- it records only the first entry from `CMAKE_CUDA_ARCHITECTURES`
- for a multi-arch build, that can reject a GPU even when a matching binary was
  actually compiled
- the same guard was added for `indy7` and `iiwa14`, but not for Tiago

Relevant files:

- [CMakeLists.txt](/home/felix/Uni/GATO/CMakeLists.txt:28)
- [gato/utils/cuda.cuh](/home/felix/Uni/GATO/gato/utils/cuda.cuh:72)
- [gato/dynamics/tiago_right/tiago_right_plant.cuh](/home/felix/Uni/GATO/gato/dynamics/tiago_right/tiago_right_plant.cuh:181)
