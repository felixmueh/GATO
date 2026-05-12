# Rebase Tiago Work Onto ICRA-26 Plan

## Overview

- [ ] 4a149fc * origin/tiago [tiago] Add pickplace validation examples
- [ ] c5d7f01 * [tiago] tiago impl + cuda smoke test
- [x] a08f48d * [gato] Track uv lockfile for reproducible setup
- [ ] fe5a659 * [gato] Add build-backed smoke tests
- [ ] 5657e63 * Build nice to haves + incremental build
- [x] 55ce2a5 * [gato] validate correct cuda architecture compile
- [ ] 1b093d8 * Compile for arch 61 and use env var
- [x] 6475ad3 * [gato] bugfix: merit integrator memory overlap
- [x] 9870d51 * [gato] bugfix: Match GRiD forward dynamics memory access
- [x] ae481d6 * [gato] Keep CUDA error checks in release
- [x] cb5b7e0 * [git] gitignore local tools
- [x] 7a6276b * [gato] Fix setup problems.
- [x] 14411cd * [git] gitignore markdown files except README.md

## Goal

Move the current `tiago` work onto `origin/ICRA-26` without doing a blind
mechanical rebase. The ICRA branch changed the same build, Docker, Python, and
solver files that the Tiago branch touched, so the safer path is a new
integration branch plus selective cherry-picks and manual commits.

## Starting Point

Create a new integration branch from ICRA:

```sh
git switch -c tiago-icra origin/ICRA-26
```

Keep the existing `tiago` branch intact until the new branch builds and the
Tiago pick/place examples pass.

## ICRA Changes To Preserve

Prefer the ICRA version for:

- `CMakeLists.txt` CMake 3.24 setup
- default CUDA architecture policy using `CMAKE_CUDA_ARCHITECTURES=native`
- Python/pybind lookup from the selected virtual environment
- Dockerfile direction based on CUDA 12.9.1, `.venv`, and uv dependency sync
- PyTorch CUDA uv source configuration in `pyproject.toml`
- non-default CUDA stream plumbing in BSQP and kernel launch wrappers

Do not overwrite these wholesale with older Tiago-branch infrastructure.

## Cherry-Pick Real Solver Bugfixes

These commits are still relevant and should be replayed first:

```sh
git cherry-pick ae481d6   # [gato] Keep CUDA error checks in release
git cherry-pick 9870d51   # [gato] bugfix: Match GRiD forward dynamics memory access
git cherry-pick 6475ad3   # [gato] bugfix: merit integrator memory overlap
```

Expected conflict:

- `6475ad3` conflicts with ICRA's stream changes in
  `gato/bsqp/kernels/merit.cuh`.

Resolution policy:

- Keep ICRA's `cudaStream_t` argument and async launch plumbing.
- Also keep the Tiago-branch merit shared-memory fix:
  `STATE_SIZE / 2 + STATE_SIZE + forwardDynamics_TempMemSize_Shared()`.
- Keep the integrator fix:
  `s_extra_temp = s_err + STATE_SIZE`.

## Reapply Build And Test Helpers Manually

Do not blindly commit these. Stage them without committing, resolve them into
the ICRA structure, then make one coherent commit.

```sh
git cherry-pick -n 55ce2a5 5657e63 fe5a659
```

Keep useful pieces:

- `tools/build.sh` incremental/focused-target workflow
- `tools/test_cuda_compatibility.sh`
- root `Makefile` convenience targets
- `tests/cpp/cuda_access_smoke.cu`
- Python smoke tests under `tests/python`
- CTest targets from `fe5a659`

Conflict policy:

- Preserve ICRA's CMake 3.24, `native` arch default, Python venv selection, and
  pybind discovery.
- Add test targets and helper functions around that ICRA CMake base.
- Make CUDA compatibility checks aware of ICRA's `native` default.
- Keep Tiago out of this commit unless it is needed only as a generic plant
  list extension.

Commit after cleanup:

```sh
git add CMakeLists.txt Makefile tools/build.sh tools/test_cuda_compatibility.sh tests pyproject.toml
git commit -m "[gato] Add build-backed tests and build helpers"
```

## Regenerate Lockfile

Do not reuse the old `uv.lock` blindly. ICRA changed `pyproject.toml` to add
CUDA PyTorch source configuration and `torchvision` / `torchaudio`.

After the build/test helper commit:

```sh
uv lock
git add uv.lock .gitignore
git commit -m "[gato] Track uv lockfile"
```

If `uv lock` selects unexpected future package versions, inspect before
committing.

## Cherry-Pick Tiago Work

Replay Tiago-specific commits after the generic infrastructure is stable:

```sh
git cherry-pick c5d7f01   # [tiago] tiago impl + cuda smoke test
git cherry-pick 4a149fc   # [tiago] Add pickplace validation examples
```

Expected conflict areas:

- `CMakeLists.txt`
- `python/bindings.cu`
- `python/bsqp/interface.py`
- `python/bsqp/config.py`
- `pyproject.toml`

Resolution policy:

- Preserve ICRA CMake structure and add only the `tiago_right` plant default,
  compile definition, and test-required target wiring.
- Keep ICRA CUDA stream plumbing.
- Keep Tiago generated files and plant wrapper:
  - `gato/dynamics/tiago_right/tiago_right_arm.urdf`
  - `gato/dynamics/tiago_right/tiago_right_grid.cuh`
  - `gato/dynamics/tiago_right/tiago_right_limits.cuh`
  - `gato/dynamics/tiago_right/tiago_right_plant.cuh`
- Keep Tiago generator:
  - `tools/generate_tiago_dynamics.py`
- Keep pick/place examples:
  - `examples/tiago_arm_pickplace_goals.py`
  - `examples/iiwa14_pickplace_goals.py`

## Do Not Cherry-Pick As Commits

Skip these commits as direct history:

```sh
14411cd   # local markdown ignore policy
cb5b7e0   # local agent/tool ignore policy
7a6276b   # older setup/Docker changes, partially superseded by ICRA
1b093d8   # older CUDA arch policy, superseded by ICRA native default
a08f48d   # old lockfile commit; regenerate instead
```

Borrow individual ideas manually only if still useful.

## Validation

After integration:

```sh
cmake -S . -B build-tiago-icra \
  -DPLANT=tiago_right \
  -DKNOTS=16 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-tiago-icra --target bsqpN16_tiago_right -j 4
python examples/tiago_arm_pickplace_goals.py
```

Expected Tiago pick/place behavior from the current branch:

- reaches `4/4` goals
- final error below `0.01 m`
- reports `rejected_solves` explicitly

Also run generic tests where feasible:

```sh
cmake --build build-tiago-icra --target check-fast
cmake --build build-tiago-icra --target cuda_access_smoke
```

If CUDA arch compatibility fails under ICRA's `native` policy, decide whether
to keep `native` as default and document an override, or restore a portable
multi-arch default.

## Final Shape

The new branch should have clear commits in this order:

1. solver/runtime bugfixes
2. build/test helper integration
3. regenerated lockfile
4. Tiago implementation
5. Tiago/IIWA pick/place validation examples

Once validated, replace the old `tiago` branch with the new `tiago-icra`
history.
