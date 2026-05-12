# Testing

This repository uses a small build-backed test setup intended for confident
agent-driven development. Test style should follow the local TDD skill:
behavior through public interfaces, one focused test at a time, and no broad
imagined suites.

## Entry Points

Use the Make targets instead of raw `ctest` when you need compiled artifacts to
be current. These targets ask CMake to build the needed binaries/extensions
before running tests.

```sh
make test
make test-fast
make test-tracking
make test-performance
```

Targets:

- `make test`: builds `cuda_access_smoke` and `bsqpN8_indy7`, then runs the
  default CTest suite.
- `make test-fast`: runs the fast Python behavior suite.
- `make test-tracking`: builds the CUDA smoke binary and `bsqpN8_indy7`, then
  runs the short tracking smoke test.
- `make test-performance`: builds the tracking dependencies and writes tracking
  performance artifacts.

For a non-default build directory, set `GATO_BUILD_DIR`:

```sh
GATO_BUILD_DIR=/tmp/gato-build make test
```

## Test Layers

- Fast Python behavior tests live under `tests/python/` and cover small,
  deterministic public behavior such as warm-start construction, figure-8
  reference generation, and public constructor errors.
- CUDA smoke coverage lives in `tests/cpp/cuda_access_smoke.cu` and verifies
  that the runtime can see a CUDA device and allocate/free device memory.
- Tracking smoke coverage uses the public `MPC_GATO.run_mpc_fig8(...)` path on
  a very short Indy7 figure-8 scenario and asserts finite tracking/solve
  metrics.
- Performance tracking is opt-in and records a JSON summary plus a PNG plot
  under `test-artifacts/tracking/`.

## Direct Pytest Use

Direct pytest is useful for quick Python-only iteration, but it does not build
CUDA binaries or extensions.

```sh
python -m pytest -m "not cuda and not tracking and not performance" tests/python
```

Tracking tests can also be invoked directly after the required extension has
already been built:

```sh
GATO_RUN_TRACKING_TESTS=1 \
python -m pytest -m "tracking and not performance" tests/python
```

## Adding Tests

Prefer tests that describe observable behavior through a public entry point:
Python APIs, C++/CUDA executables, CTest targets, or small solver runs. When a
test needs a compiled artifact, add that artifact as a dependency of the
appropriate CMake `check*` target so the supported Make command keeps it fresh.
