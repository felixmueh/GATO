# Setup Without Docker

This repository can be built and run directly on the host. The notes here are
for local development only. They should not be confused with the repository's
reference Docker workflow.

## What Was Verified

A host-side workflow was verified with:

- Fedora Linux
- Python `3.10.12`
- `uv`
- CMake `3.31`
- a working NVIDIA driver
- a visible CUDA toolkit on `PATH`

The most important host-side build changes already in the repo are:

- `CMakeLists.txt` resolves `pybind11` from the active Python environment via
  `python -m pybind11 --cmakedir`
- `tools/build.sh` is tuned for local development and uses a reduced build
  matrix by default
- CUDA runtime error checking stays enabled in release builds

## Prerequisites

You need:

- `nvidia-smi` working on the host
- `nvcc` available on `PATH`
- a working C++ toolchain
- CMake
- `uv`

Sanity checks:

```sh
nvidia-smi -L
nvcc --version
cmake --version
```

## Host Build

From the repo root:

```sh
uv sync
source .venv/bin/activate
./tools/build.sh
```

`tools/build.sh` currently:

- deletes and recreates `build/`
- runs the repository's CUDA compatibility validator after a successful build

Override the defaults when needed:

```sh
PLANT="indy7;iiwa14" KNOTS="8;16;32;64;128" CMAKE_BUILD_PARALLEL_LEVEL=2 ./tools/build.sh
```

If you want explicit CMake commands instead:

```sh
cmake -S . -B build -DPLANT="indy7;iiwa14" -DKNOTS="16;32;64"
cmake --build build --parallel 4
./tools/test_cuda_compatibility.sh --strict
```

## Smoke Tests

C++:

```sh
./build/bsqp
```

Python:

```sh
source .venv/bin/activate
PYTHONPATH=python python examples/benchmark_fig8.py --plant indy7 --batch-sizes 1 --no-save
```

## Important Caveats

### CUDA Toolkit Version Matters

Some host-side failures were caused by a bad toolkit/GPU pairing, not by the
solver itself.

The clearest known example:

- `CUDA 13.x` no longer supports Pascal `sm_60/sm_61`
- on a Pascal GPU, a build with CUDA 13.x can produce misleading runtime
  behavior before you ever reach the real solver bug
- rebuilding with a toolkit that still supports the GPU exposed the real
  shared-memory bug in the solver

So if host results look nonsensical, verify the compiled architectures and
toolkit support before assuming the math is wrong.

### `CMAKE_CUDA_ARCHITECTURES`

The repo now defaults to:

```cmake
set(CMAKE_CUDA_ARCHITECTURES "61-real;75-real;86-real;89-real;89-virtual")
```

That is a portable default for the GPUs currently relevant to this repository,
but local builds should still override it when you know the exact target GPU.

Examples:

```sh
# Build only for the local Pascal GPU
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=61

# Build a portable binary with real cubins for common targets plus PTX for newer GPUs
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="61-real;75-real;86-real;89-real;89-virtual"
```

### Sandbox GPU Access

If a sandboxed run reports:

```text
no CUDA-capable device is detected
```

that may be a sandbox limitation rather than a real host problem. Re-check on
the actual host before changing code.

### CUDA Compatibility Validator

Use:

```sh
./tools/test_cuda_compatibility.sh --strict
```

It checks the built executable and Python extension modules against the visible
GPU(s) using `nvidia-smi` and `cuobjdump`.

By default `tools/build.sh` runs it automatically after building. To skip that
automatic validation for a build, set:

```sh
GATO_SKIP_CUDA_COMPATIBILITY_TEST=1 ./tools/build.sh
```

## Notes For A Fresh Docker Rebuild

These host-side notes should be treated as convenience guidance, not as the
canonical environment definition.

Likely real repo improvements that still matter in Docker:

- venv-based `pybind11` discovery
- always-on CUDA runtime error checks
- Python benchmark/reference indexing fixes
- the BSQP merit/integrator shared-memory fix

Likely host-specific or local-development choices that should be re-evaluated
under Docker:

- `CMAKE_CUDA_ARCHITECTURES=native`
- reduced default build matrix in `tools/build.sh`
- any behavior specific to unsupported host GPU / CUDA combinations
