# Session Log

## 2026-05-07

### Tiago figure-8 timing check

- Temporarily instrumented the Tiago figure-8 artifact path to separate
  wall-clock solver-call timing from best-trajectory evaluation timing, ran one
  media-disabled timing check, then removed that instrumentation and its
  generated run artifacts at the user's request.

## 2026-04-28

### Tiago dynamics generation setup

- Started the Tiago arm integration by adding upstream GRiD as a `GRiD/`
  submodule.
- Initialized GRiD's nested submodules through an HTTPS URL rewrite because
  upstream records those nested URLs as SSH remotes, which are not usable in the
  non-interactive container.
- Removed the stale local `TiagoProURDF/README.md` from the previous trial run;
  Tiago implementation notes should live in `felixnotes/IMPL_TIAGO.md`.
- Decided with the user that the new generation script should orchestrate the
  full source-URDF-to-generated-header flow and default the header output to
  `gato/dynamics/tiago_right/tiago_right_grid.cuh`.
- Identified one open design issue before implementation: the current upstream
  GRiD checkout does not expose the old trial's documented fixed-target `-t`
  CLI, so the script should either rely on generic single-leaf end-effector
  output or fail until fixed-target generation is provided outside GRiD.
- Added `GRiD_SUPPORT_NEG_AXES.md` as a handoff document describing the
  negative-axis parser limitation, the Tiago joint-6 case, and a proposed
  upstream-style GRiD fix.
- The user decided not to patch GRiD for negative axes. Implemented
  `tools/generate_tiago_dynamics.py` as a vanilla-GRiD generator that extracts
  the right arm into a temporary URDF, flips negative principal axes only in
  that temporary input, runs GRiD, and writes
  `gato/dynamics/tiago_right/tiago_right_grid.cuh`.
- Verified generation with a repo-local `.venv_grid` because the default
  `/opt/gato-venv` is root-owned and missing GRiD's `lxml` dependency.
  `.venv_grid/` is excluded locally through `.git/info/exclude`.
- Confirmed the only axis flip for Tiago right arm generation is
  `arm_right_6_joint`, where the temporary GRiD convention requires
  `q_grid=-q_tiago`, `qd_grid=-qd_tiago`, `u_grid=-u_tiago`, and
  `qdd_tiago=-qdd_grid`.
- Important follow-up: the current vanilla GRiD checkout generates the core
  dynamics helpers but not the old trial's end-effector pose/gradient helpers,
  so Tiago tracking wrapper work must either add those kinematics another way
  or avoid relying on generated end-effector helper functions.
- Implemented `gato/dynamics/tiago_right/tiago_right_plant.cuh` following the
  existing plant wrapper API:
  - Tiago joint, velocity, and torque limits
  - GRiD robot-model allocation/free hooks
  - forward dynamics and forward-dynamics gradient wrappers
  - explicit `arm_right_6_joint` sign conversion at every coordinate boundary
  - a small handwritten FK/Jacobian helper for `arm_right_tool_link`, because
    vanilla GRiD does not generate end-effector helpers
- Wired `tiago_right` into `gato/settings.h`, `CMakeLists.txt`,
  `python/bindings.cu`, and the Python `BSQP` interface.
- Added `examples/tiago_arm_tracking_simple.py` as a one-shot end-effector
  tracking smoke example.
- Built `bsqpN16_tiago_right` in `build-tiago/` and ran the simple example.
  The example reduced merit from about `83.68` to `11.71` over 5 SQP
  iterations on the local CUDA runtime.

### Felix Docker workspace mount

- Changed `felixtools/docker.sh` to mount the full parent directory at
  `/workspace` instead of mounting the GATO repository and `GATO-notes`
  separately.
- Kept the container working directory inside the GATO checkout under that
  parent mount, so repo-relative commands still start from the project.
- Removed the explicit `NOTES_MIRROR_REPO` environment override; the notes sync
  helper now uses its default `<repo>/../GATO-notes` path inside the shared
  workspace mount.
- Set the container `PYTHONPATH` from `docker.sh` to the repository path under
  the parent mount, avoiding the old baked `/workspace/python` path when the
  parent directory is mounted instead.

## 2026-04-27

### Codex runtime environment note

- Documented at the top of `AGENTS.md` that Codex usually runs inside the
  Felix Docker container defined by `felixtools/Dockerfile` and
  `felixtools/docker.sh`.
- Kept the `AGENTS.md` note short and moved the practical container-detection
  checks to `felixnotes/RUNTIME_ENVIRONMENT.md`.
- Clarified that Codex should treat commands as having broad access to mounted
  host paths, and should ask the user to modify Docker configuration when
  package, mount, device, or security settings need to change.

## 2026-04-19

### Docker-first installation rewrite

- Reworked the repository setup flow so Docker is the primary environment.
- `tools/docker.sh` is now the authoritative container lifecycle script.
- Added `--rebuild-image` and `--no-attach` to `tools/docker.sh`.
- `tools/install.sh` no longer installs or activates a host-side Python
  environment.
- The Docker image now installs `uv` and creates the Python environment inside
  the image at `/opt/gato-venv`.
- Added `.dockerignore` so local artifacts like `.git`, `build/`, and `.venv`
  do not get copied into the Docker build context.
- Updated `../README.md` to document:
  - `./tools/install.sh`
  - `./tools/docker.sh`
  - Python example execution with `PYTHONPATH=python`

### Optional X11 setup handling

- Made `tools/docker.sh` treat `xhost +local:docker` as best-effort.
- X11 forwarding setup no longer aborts the script when `xhost` is missing or
  when no X server is available.

### ROS shell cleanup

- Stopped auto-sourcing `/opt/ros/humble/setup.bash` in the default container
  shell.
- The default shell now only activates `/opt/gato-venv`.
- This fixes the interactive-shell import failure where ROS `PYTHONPATH`
  injected ROS `eigenpy`, which then conflicted with the venv `pinocchio`.
- Also cleared the inherited `/ros_entrypoint.sh` from the ROS base image,
  because that entrypoint was still sourcing ROS before the shell started.

### Default PYTHONPATH

- Added `/workspace/python` to `PYTHONPATH` in the default container shell.
- Python examples can now be run as `python examples/...` without manually
  prefixing `PYTHONPATH=python`.

### Validate `BUGFIXES.md` point 2 (`gpuErrchk` in release builds)

- Re-checked the current tree against `BUGFIXES.md` item 2.
- Confirmed `gato/utils/cuda.cuh` still compiles `gpuErrchk(...)` away under
  `NDEBUG`.
- Confirmed `CMakeLists.txt` still defines `-DNDEBUG` for release builds, so
  the checked-in release configuration still disables CUDA error checks.
- Built a minimal local probe that forces CUDA error statuses in both debug and
  release configurations.
- Observed the debug probe abort immediately with a `GPUassert:` message.
- Observed the `-DNDEBUG` probe continue past the same forced errors and exit
  successfully, which confirms the failure is currently hidden in release mode.
- Local-only probe artifacts were written under `tmp/` during validation.

### Validate performance impact of enabling `gpuErrchk` in release

- Built isolated release-mode comparison harnesses for:
  - the current local behavior with `gpuErrchk` always enabled
  - a control copy of `gato/` with the previous `NDEBUG`-gated macro
- End-to-end solver benchmarking is not possible in this environment because
  the current always-check build now aborts immediately on the first real CUDA
  allocation with `OS call failed or operation not supported on this OS`,
  while the old release behavior hides that failure and continues.
- Added a host-side microbenchmark to isolate the extra successful-call cost of
  the check itself in release builds.
- Across 100,000,000 calls per run, measured per-call costs were:
  - always-check: 1.250 ns, 1.370 ns, 1.616 ns
  - old `NDEBUG` behavior: 1.597 ns, 1.603 ns, 1.659 ns
- Within run-to-run noise, there is no evidence of a meaningful performance
  regression from keeping `gpuErrchk` enabled in release. Any added host-side
  overhead is below nanosecond-scale noise on this machine and far below the
  solver's real CUDA work.

### Rebuild for compute capability 6.1

- The main `CMakeLists.txt` hardcodes `CMAKE_CUDA_ARCHITECTURES` to
  `89 86 75`, so a direct `-DCMAKE_CUDA_ARCHITECTURES=61` configure does not
  override it.
- Created a temporary source copy under `/tmp/gato_sm61_src`, changed only its
  `CMakeLists.txt` to target `61`, and built the `bsqp` executable there.
- An unrestricted run of the rebuilt `sm_61` binary no longer failed at the
  first `cudaMalloc` in `indy7_grid.cuh`.
- The new failure is `GPUassert: an illegal memory access was encountered` at
  the post-kernel copy in `gato/bsqp/bsqp.cuh`, which indicates the earlier
  architecture mismatch was masking a later kernel-side bug.
- That new failure is consistent with the existing `BUGFIXES.md` item about
  BSQP merit/integrator shared-memory bugs being real solver issues.

### Codex-in-Docker sandbox wrapper

- Confirmed the Docker launcher already passes `--gpus all`; the remaining GPU
  access problem was the nested Codex sandbox inside the container.
- Added `felixtools/codex.sh` as an explicit wrapper that starts Codex with
  `--dangerously-bypass-approvals-and-sandbox`, relying on the outer Docker
  container as the isolation boundary.
- Updated `felixtools/docker.sh --help` to point users at that wrapper for
  CUDA/device-access debugging inside the container.

### CUDA architecture selection and runtime guard

- Reworked `CMakeLists.txt` so `CMAKE_CUDA_ARCHITECTURES` is initialized before
  `project(...)`, which lets the repo provide a real default while still
  honoring user overrides from `-D...` or `CUDAARCHS`.
- The repo now defaults to a portable architecture set:
  `61-real;75-real;86-real;89-real;89-virtual`.
- Added a generated header that records the configured CUDA architecture list in
  the binary so runtime code can validate against what was actually built.
- Added a runtime compatibility guard in `gato/utils/cuda.cuh` and called it
  from both plant `initializeDynamicsConstMem()` entry points.
- The guard accepts:
  - same-major real cubins with sufficient minor capability
  - PTX entries whose virtual architecture is less than or equal to the device
- Verified the matching default build on the local `sm_61` GPU gets past the
  architecture guard and reaches the pre-existing later solver bug.
- Verified an intentionally wrong `75-real` build now fails early with a clear
  message telling the user to rebuild with `-DCMAKE_CUDA_ARCHITECTURES=<your-sm>`
  or a portable list.
- Updated `../README.md` and `SETUP_WITHOUT_DOCKER.md` with explicit local and
  portable build examples.
- Added an `../AGENTS.md` note that multi-architecture CUDA builds and Python
  extension builds can be slow, so long compile times alone should not be
  treated as evidence of a hang.

### CUDA compatibility validator script

- Replaced the in-process CUDA architecture guard with an external developer
  validation script, `tools/test_cuda_compatibility.sh`.
- The script compares the visible GPU compute capability from `nvidia-smi`
  against the embedded cubin/PTX architectures reported by `cuobjdump`.
- Compatibility rules now live in that script:
  - cubins must match major version and not exceed the GPU minor version
  - PTX is accepted only when its target architecture is less than or equal to
    the GPU compute capability
- Wired the validator into `tools/build.sh` so setup/builds automatically fail
  early when the produced artifacts are not suitable for the visible GPU.
- Added `GATO_SKIP_CUDA_COMPATIBILITY_TEST=1` as an escape hatch for builds
  where the compatibility check should be skipped explicitly.
- Verified the script correctly rejects the current stale `build/bsqp` artifact
  on the local `sm_61` GPU because it only contains `75/86/89` cubin/PTX code.

### Testing plan

- Wrote `PLAN_TESTING.md` to outline a phased automated testing strategy for
  upstream GATO.
- The plan separates:
  - `ctest`-driven CUDA/C++ smoke tests
  - `pytest`-driven Python tests
  - heavier optional integration coverage
- Documented that Python test dependencies should be installed through the main
  repository `Dockerfile`, not only through Felix-specific tooling.
- Noted the intended installation path:
  - declare test modules in `pyproject.toml`
  - refresh `uv.lock`
  - make the main Docker build install the test group via `uv`

### Upstream test dependencies in Docker

- Kept `pytest` in the repository dependency metadata under the dev group in
  `pyproject.toml`.
- Refreshed `uv.lock` so frozen installs include the new Python test
  dependency chain (`pytest`, `pluggy`, `iniconfig`).
- Updated the main repository `Dockerfile` to install the dev group with
  `uv sync --frozen --no-install-project --group dev`, so future upstream test
  setup lives in the primary image rather than the Felix overlay.

### Keep `gpuErrchk` active in release builds

- Removed the `NDEBUG` gating around `gpuErrchk(...)` in `gato/utils/cuda.cuh`
  so release builds keep surfacing CUDA API failures.
- Marked `gpuAssert(...)` as `inline` because it is defined in a header and is
  intended to be included from multiple translation units.
- Verified behavior with a local `-DNDEBUG` probe: the forced CUDA error now
  aborts immediately with `GPUassert:` instead of being silently ignored.
- Verified the project still builds in release mode by configuring a fresh
  CMake tree in `/workspace/tmpbuild_release` and building the `bsqp` target
  there, because the pre-existing `/workspace/build` tree is not writable.

## 2026-04-20

### Notes mirror scope

- Extended the post-commit notes mirror hook to include `/felixtools/*` in
  addition to the existing mirrored Markdown files.
- Kept the change confined to the hook include rules so the main repository
  structure and upstream-facing files remain unchanged.
- Corrected the `rsync` filter order so `../README.md` stays excluded as intended
  after broadening the mirrored file set.

## 2026-04-25

### Benchmark path for strict CUDA error checks

- Reviewed the existing examples to identify a practical regression benchmark
  for always-on `gpuErrchk(...)` behavior.
- Confirmed `examples/benchmark_fig8.py` is the best in-tree option for a
  full-MPC regression check because it reports per-step solver timings from the
  native BSQP layer.
- Built two isolated `sm_61` release variants of `bsqpN64_indy7` under `/tmp`,
  differing only in whether `gpuErrchk(...)` stays active in release mode.
- Added a local helper at `tmp/run_short_fig8_benchmark.py` to run a short
  `benchmark_fig8.py` configuration against an arbitrary repo root without
  editing the example itself.
- On the current GTX 1070 setup, the enabled-checks variant aborts immediately
  with `GPUassert: an illegal memory access was encountered` from
  `gato/bsqp/bsqp.cuh:137`, so this example cannot provide a clean A/B timing
  comparison yet.
- The disabled-checks variant completes the same short run and reports
  `avg_gpu_time_ms ~= 0.025`, which confirms the old release behavior hides the
  fault rather than proving there is no performance regression.

### Shared-memory bug validation

- Could not find `../GATO_OLD` in this container, and the commit IDs referenced
  in `BUGFIXES.md` are not present in this checkout, so the memory-bug
  validation was done from the current source and local reproducer builds.
- Built isolated `sm_61` release variants of `bsqpN64_indy7` to test the two
  suspected shared-memory fixes independently.
- The integrator-only and merit-allocation-only variants both still failed with
  `GPUassert: an illegal memory access was encountered`, so neither fix is
  sufficient on its own.
- `compute-sanitizer` showed invalid shared-memory writes in
  `computeMeritBatchedKernel` through `compute_integrator_error` and the
  generated Indy7 `direct_minv_inner` path, confirming the failure is a real
  shared-memory sizing/layout bug after compiling for the correct `sm_61`
  architecture.
- Implemented the validated fix set in handwritten code: move the integrator
  extra scratch past the full error vector, size merit shared memory for the
  defect path explicitly, and have plant wrappers report the larger generated
  forward-dynamics scratch requirement.
- Verified the workspace `bsqpN64_indy7` build now completes the short
  figure-8 benchmark with strict CUDA checks enabled, and compiled
  `bsqpN8_iiwa14` to cover the analogous IIWA14 wrapper change.

### Error-checking performance benchmark

- Built two isolated `sm_61` release copies of `bsqpN64_indy7`, differing only
  in whether `gpuErrchk(...)` is compiled out under `NDEBUG`.
- Used `tmp/run_short_fig8_benchmark.py` to run the same reduced
  `examples/benchmark_fig8.py` path: Indy7, `N=64`, `batch_size=1`,
  `sim_time=0.2`, `sim_dt=0.002`, five repeats per measured pass.
- After warm-up, the enabled-checks build averaged `0.554 ms` GPU solve time;
  the disabled-checks build averaged `0.568 ms`.
- Conclusion: this benchmark shows no measurable solve-time regression from
  keeping `gpuErrchk(...)` active in release builds.
- As a red-test sanity check, an artificial `cudaDeviceSynchronize()` inserted
  into the temporary enabled-checks build raised average GPU solve time to
  `0.707 ms` versus `0.582 ms` for the unchanged disabled-checks build; the
  temporary slowdown was then reverted.

### Forward-dynamics shared-memory layout correction

- Revisited the memory fix using the more detailed finding that the plant
  wrappers should match GRiD's generated plain forward-dynamics shared-memory
  layout instead of over-allocating via `FD_DU_MAX_SHARED_MEM_COUNT`.
- Updated the Indy7 and IIWA14 plain `forwardDynamics()` wrappers to place
  `s_temp` after `72 * grid::NUM_JOINTS` XI entries, matching the generated
  GRiD kernels, and restored `forwardDynamics_TempMemSize_Shared()` to
  `grid::FD_DYNAMIC_SHARED_MEM_COUNT`.
- Kept the separate integrator scratch fix because `integrator_error_inner`
  writes a full `STATE_SIZE` error vector and needs non-overlapping scratch.
- Kept merit shared-memory sizing aligned with the corrected integrator layout:
  `STATE_SIZE / 2 + STATE_SIZE + max(STATE_SIZE, forwardDynamics temp)`.
- Rebuilt `bsqpN64_indy7` and verified the short figure-8 reproducer completes
  with strict CUDA checks enabled. `compute-sanitizer` no longer reports the
  invalid shared-memory writes; only the pre-existing GTX 1070 unsupported
  persisting-L2 limit report remains.
- Rebuilt `bsqpN8_iiwa14` successfully to cover the analogous IIWA14 wrapper
  change.

### Commit style documentation

- Reviewed recent commit subjects and bodies to capture the local message
  style.
- Added `GIT_STYLE.md` with terse scoped subject guidance, body conventions,
  and examples from recent history.
- Linked `GIT_STYLE.md` from the commit-labeling section in `../AGENTS.md`.

### Default figure-8 tracking check

- Ran the unmodified default `examples/benchmark_fig8.py` after the CUDA error
  checking and shared-memory fixes.
- The default benchmark completed and saved
  `benchmark_fig8_20260426_105628.pkl`.
- Tracking is reasonable for the smaller default batch sizes:
  batch 1 `0.0328 m`, batch 4 `0.0330 m`, batch 8 `0.0330 m`,
  batch 16 `0.0344 m`, batch 32 `0.0421 m`, batch 64 `0.0758 m`.
- The full default sweep is not fully healthy: batch 2 fails with
  `Batch size must be > 3 for exploitation + exploration strategy`, batches
  128 and 256 produce `nan` tracking error, and batches 512 and 1024 track at
  roughly `0.99 m` average error.
- Conclusion: the core tracking path works with default controller settings for
  practical small/medium batch sizes, but the default benchmark's full batch
  list still contains invalid or poor-tracking configurations.

### Shared-memory bug identification summary

- The bug first became visible only after keeping `gpuErrchk(...)` active in
  release builds. Before that, release builds compiled the check out and the
  solver could continue after CUDA failures, making the symptom look like a
  later generic crash or bad result.
- Initial failures were separated into environment/setup issues and real solver
  issues: sandbox GPU access produced an `OS call failed` allocation error, and
  the stale build targeted newer GPUs than the local GTX 1070. After rebuilding
  for `sm_61` with real GPU access, the remaining failure was an illegal memory
  access during the BSQP solve.
- The short `benchmark_fig8.py` reproducer narrowed the failure to the Indy7
  `bsqpN64_indy7` solver path. With strict CUDA checks enabled, it failed after
  solver kernels ran rather than during allocation or module load, pointing away
  from architecture mismatch and toward kernel memory behavior.
- Controlled temporary builds tested the suspected fixes independently:
  integrator-only and merit-allocation-only variants both still failed, showing
  the two handwritten shared-memory issues were related but neither was
  sufficient on its own.
- `compute-sanitizer` located the bad access in `computeMeritBatchedKernel`
  through `compute_integrator_error()` and then into the generated Indy7
  `direct_minv_inner()` call. That proved the merit defect path was under-sizing
  or mis-laying-out dynamic shared memory.
- Source inspection then identified two concrete layout mistakes: the integrator
  error path placed `s_extra_temp` halfway through the full `STATE_SIZE` error
  vector, and the merit kernel sized its temp region for only `max(tracking,
  forwardDynamics)` instead of the integrator-local qdd/error workspace plus
  forward-dynamics scratch.
- A later deeper review showed the plant wrappers also used hard-coded XI
  offsets (`864` for Indy7, `1008` for IIWA14) that matched a larger gradient
  layout, not the generated plain forward-dynamics layout. The correct plain
  GRiD layout starts temp storage after `72 * grid::NUM_JOINTS` XI entries and
  can use `grid::FD_DYNAMIC_SHARED_MEM_COUNT`.
- The final fix is therefore three-part: move the integrator extra scratch after
  the full error vector, size merit shared memory for the full defect path, and
  make the plain plant `forwardDynamics()` wrappers match the generated GRiD
  dynamic shared-memory layout.
- Validation after the final fix: `bsqpN64_indy7` rebuilt and completed the
  short figure-8 reproducer with strict CUDA checks enabled; `compute-sanitizer`
  reported no invalid shared-memory writes; `bsqpN8_iiwa14` compiled
  successfully for the analogous wrapper change.

### GTX 1070 large-batch tracking diagnosis

- Verified the current `bsqpN64_indy7` Python extension contains an `sm_61`
  cubin, so the default figure-8 large-batch behavior is not the old
  wrong-architecture build problem.
- Confirmed GPU access on the local GTX 1070 reports compute capability 6.1.
  PyTorch warns that its CUDA build does not support `sm_61`, but GATO uses its
  own CUDA extension for the solver path tested here.
- Short large-batch runs complete with CUDA error checks enabled. A sanitizer
  run with API-limit reporting disabled reported `ERROR SUMMARY: 0 errors`.
- The sanitizer still reports `cudaErrorUnsupportedLimit` for
  `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)` on the GTX 1070.
  That is a real portability issue in the unconditional persisting-L2 setup,
  but it is separate from the observed tracking degradation.
- The default MPC loop advances simulated time by measured wall-clock solver
  latency (`solve_time = time.time() - start`). On this 1070, large batches
  exceed the controller `dt=0.01 s`, so the simulated controller effectively
  runs slower than intended.
- Batch 128, 1.0 s simulation, current timing: 26 MPC updates, average solve
  time `41.0 ms`, average tracking error `0.550 m`.
- Batch 128, 1.0 s simulation, same solver but fixed simulated period of
  `dt=10 ms`: 101 MPC updates, average solve time `33.9 ms`, average tracking
  error `0.070 m`.
- Conclusion: the poor large-batch default tracking on the GTX 1070 is mainly a
  compute-throughput/benchmark timing artifact, not evidence of remaining CUDA
  memory corruption. There is still a separate GTX 1070 compatibility bug in
  persisting-L2 setup that should be guarded or ignored on unsupported GPUs.

### CUDA architecture default simplification

- Simplified the CMake CUDA architecture default by removing the extra
  `GATO_CUDA_ARCHITECTURES_DEFAULT` cache variable.
- Kept the default guarded by `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` so
  developers can still override architectures with
  `-DCMAKE_CUDA_ARCHITECTURES=...`.
- Verified configure behavior with a default configure and with an explicit
  `-DCMAKE_CUDA_ARCHITECTURES=61` override.

### Local notes layout and hook extraction

- Moved local ignored Markdown notes under `felixnotes/`, leaving
  `../AGENTS.md` at the repository root as the entry point.
- Updated `../AGENTS.md` to point at the new `felixnotes/` paths.
- Extracted the local notes mirror logic from `.githooks/post-commit` into
  `../felixtools/commit_notes.sh`; the hook now only invokes that script.
- The mirror script syncs `../AGENTS.md`, `felixnotes/`, and
  `../felixtools/` with `rsync --delete --delete-excluded` so subdirectories,
  deletions, and moves are reflected in the notes repository.

### Felix Docker notes repo mount

- Updated `../felixtools/docker.sh` to mount a sibling `GATO-notes` repository
  into Felix dev containers when present.
- The host path defaults to `../GATO-notes` and can be overridden with
  `NOTES_REPO_HOST`.
- Inside the container, the notes repo is mounted at `/GATO-notes` and
  `NOTES_MIRROR_REPO=/GATO-notes` is exported so
  `../felixtools/commit_notes.sh` can sync notes without relying on the
  sibling path being mounted next to `/workspace`.
- Added stale-container detection to the Felix Docker wrapper: if the named
  container already exists but does not have the `/GATO-notes` mount or
  `NOTES_MIRROR_REPO=/GATO-notes`, the wrapper recreates it so new bind mounts
  take effect.
- Confirmed `../felixtools/commit_notes.sh` already includes
  `../felixtools/**` in the GATO-notes mirror.
- Simplified `../felixtools/docker.sh` for the fixed local environment:
  removed environment-variable configuration and defensive multi-environment
  branches, kept the existing command-line flags, and added `--restart`.
- The simplified wrapper always creates containers with the GATO repo mounted
  at `/workspace`, sibling notes repo at `/GATO-notes`, Codex state under the
  container user's home, and `NOTES_MIRROR_REPO=/GATO-notes`.

### Felix Docker tooling packages

- Added `rsync` to `../felixtools/Dockerfile` because
  `../felixtools/commit_notes.sh` requires it to mirror local notes.
- Added a compact set of local interactive/debugging tools to the Felix overlay
  image: `htop`, `jq`, `less`, `lsof`, `ncdu`, `strace`, `tmux`, and `tree`.

### Build wrapper organization and incremental builds

- Changed `tools/build.sh` from an always-clean rebuild helper into an
  incremental CMake wrapper. The default path now reuses `build/`, while
  `--fresh` preserves the old clean-build behavior.
- Added focused build options for CMake targets, plant/knot configuration,
  explicit CUDA architectures, and opt-in native GPU architecture detection.
- Added a root `Makefile` as the organized command surface, with explicit
  targets for `bsqp` and each default `indy7`/`iiwa14` Python extension target.
- Left runtime build verification to the `/workspace/build` environment per
  request; local verification was limited to static/syntax checks.

### Incremental build validation and cleanup

- Validated the new incremental build wrapper on the local GTX 1070
  (`sm_61`) using isolated build directories under `/tmp`.
- A fresh focused `bsqp` build with `--native-cuda-arch` compiled successfully;
  repeating the same target build reused the existing CMake tree and did not
  recompile the CUDA object.
- A narrowed full build in a temporary source copy
  (`--plant indy7 --knots 8`) built both `bsqp` and `bsqpN8_indy7`, then passed
  strict CUDA artifact validation against the non-default build directory.
- Repeating that narrowed full build completed as an incremental no-op for
  both targets while still running compatibility validation.
- Found and fixed one build-directory consistency problem: full builds honored
  `GATO_BUILD_DIR`, but `tools/test_cuda_compatibility.sh`, `make test-cuda`,
  and `make clean-build` still assumed `build/`.
- Simplified the new root `Makefile` by generating the repeated plant/knot
  Python extension targets from `PLANTS` and `KNOTS` lists instead of spelling
  each target rule out manually.

## 2026-04-27

### Felix Docker agents mount

- Updated `../felixtools/docker.sh` to create and mount host `~/.agents` into
  the Felix dev container at `/home/felix/.agents`, matching the existing
  Codex state mount pattern.

## 2026-04-28

### Testing plan TDD re-evaluation

- Reworked `felixnotes/PLAN_TESTING.md` around the local TDD skill's
  behavior-first, vertical tracer-bullet workflow.
- Updated the plan to reflect current test-related files already present in
  the tree: Python utility tests, CUDA access smoke source, pytest dependency
  metadata, Docker dev dependency install, and the CUDA artifact compatibility
  script.
- Clarified the desired AI-driven development loop: identify the public
  behavior, add one failing test, make the narrow fix, run focused checks, and
  only then refactor.

### Basic testing setup

- Wired CMake/CTest test registration for a fast Python behavior suite, CUDA
  access smoke test, and short Python tracking smoke test.
- Reworked the Python tests as behavior examples: warm-start construction,
  figure-8 reference generation, public BSQP constructor error reporting, and
  a short end-to-end Indy7 figure-8 tracking run.
- Added pytest markers for `cuda`, `tracking`, and `performance` so the fast
  default suite stays small while GPU/tracking/performance checks remain
  explicit.
- Added an opt-in tracking performance test that writes a JSON summary and
  tracking plot artifact when `GATO_RUN_PERFORMANCE_TESTS=1` is set.
- Verified the narrowed CTest setup after rebuilding `bsqpN8_indy7` from the
  current source.
- Added CMake `check`, `check-fast`, `check-python`, `check-tracking`, and
  `check-performance` targets so supported test entrypoints build the
  appropriate CUDA binaries/extensions before running tests.
- Added root Makefile wrappers for those CMake targets, with `make test` as the
  build-backed default test command.

### README policy note

- Added an `AGENTS.md` documentation policy note that `README.md` should stay
  aligned with upstream and only be modified for `[gato]` upstream-facing
  changes.
- Clarified that local workflow information may be newer in `AGENTS.md`,
  `felixnotes/`, or tool help than in `README.md`.

### Testing notes rename

- Replaced the long testing plan with a short current-state
  `felixnotes/TESTING.md` reference describing the build-backed test entry
  points, test layers, direct pytest use, and how to add tests.
- Updated `AGENTS.md` to reference `felixnotes/TESTING.md` from the recommended
  local workflow notes.

### GRID Negative Axis Inquiry

Investigated native support for signed principal URDF axes in GRiD, specifically Tiago Pro's “arm_right_6_joint” with
“<axis xyz="0 @ -1" />>.

Findings:
- “URDFParser/Joint.py” only handles positive principal axes.

- A parser-only fix is not enough: generated CUDA assumes “S* is encoded as the index of a “+1° entry via
*s.tolist() .index(1)*.

- That index is used throughout generated dynamics code for “S * qd’, “S * qdd’, “SAT * f°, “IA * S*, “SAT * U, “SAI
* F, and “mxs”.

- “RBDReference’ and “GRiDCodeGenerator/_test.py’ also assume positive “S*, so tests would need updates too.

- A proper GRiD fix would need signed sparse motion subspace support, likely “S_ind + S_sign*, across “URDFParser’,
“GRiDCodeGenerator™, and “RBDReference’.

Decision:
- Do not pursue native signed-axis support in GRiD for now.

- Keep the GATO-side workaround: rewrite negative axes to positive, negate/swap limits, and compensate in wrapper
mappings.

### Tiago right-arm implementation pass

- Added vanilla GRiD as the `GRiD/` submodule and generated
  `gato/dynamics/tiago_right/tiago_right_grid.cuh` from
  `TiagoProURDF/tiago_pro.urdf`.
- Added `tools/generate_tiago_dynamics.py` as the Tiago generation entrypoint.
  It extracts the right arm into a temporary GRiD URDF, flips unsupported
  negative principal axes only in that temporary file, runs GRiD, and deletes
  the temporary URDF afterward.
- Implemented the Tiago right-arm plant wrapper around the generated GRiD
  dynamics. The wrapper keeps GATO and robot-facing code in native Tiago joint
  coordinates and maps only `arm_right_6_joint` across the GRiD convention
  boundary.
- Wired `PLANT_TIAGO_RIGHT` through CMake, settings, Python bindings, and the
  Python BSQP interface. The interface now reduces the full Tiago URDF to the
  seven right-arm joints with Pinocchio instead of relying on a persistent
  extracted arm URDF.
- Added `examples/tiago_arm_tracking_simple.py` as a small tracking smoke test
  using the full Tiago URDF.
- Verified:
  - `GATO_BUILD_DIR=build-tiago ./tools/build.sh --plant tiago_right --knots 16 --target bsqpN16_tiago_right`
  - Python compile checks for the generator, interface, config, and example
  - `PYTHONPATH=python python examples/tiago_arm_tracking_simple.py`
  - a direct `sim_forward` call on the Tiago right-arm solver
- Added an explicit Python API guard so `plant_type="tiago_right"` raises
  `NotImplementedError` for nonzero external wrenches instead of silently
  ignoring them. Zero wrenches remain allowed for normal initialization and
  reset paths.
- Verified the closed-loop `MPC_GATO` path for Tiago. The first attempt used a
  Pinocchio model reduced from the full Tiago URDF and drifted because that
  model does not match the arm-only GRiD dynamics closely enough for simulation
  validation. The working smoke test now builds a temporary native right-arm
  URDF for Pinocchio simulation, matching the generated GRiD dynamics without
  leaving an artifact.
- Added `examples/tiago_arm_mpc_tracking_simple.py`; it runs a one-goal
  closed-loop MPC smoke test and fails if the goal is not reached.
- Added `examples/tiago_arm_mpc_fig8_tracking.py` as the proper closed-loop
  Tiago figure-8 tracking smoke test. It builds the temporary native arm URDF,
  runs `MPC_GATO.run_mpc_fig8` for 2 seconds, checks that the reference and
  actual motion have nontrivial span, and fails on excessive mean or 95th
  percentile tracking error.
- Added `examples/plot_figure8_tracking.py` to run the figure-8 tracking
  examples and generate static Matplotlib PNG plots. The `--plant all` mode
  runs each plant in a separate Python subprocess because generated pybind
  solver modules register the same `BSQP_1_float` type name in-process.
  Generated plots go to ignored `test-artifacts/figure8_plots/`.
- Added `examples/tiago_arm_pickplace_goals.py`, a scripted Tiago multi-goal
  pick/place-style example. It creates a temporary native right-arm URDF,
  tracks five reachable Cartesian goals, writes a static PNG and Matplotlib GIF
  to ignored `test-artifacts/tiago_pickplace/`, and fails if all goals are not
  reached.
- Tightened the pick/place example so each goal must reach a 3 cm threshold
  before switching to the next goal, with a 5 second per-goal timeout acting as
  a failure guard rather than normal control flow.
- Added `goal_dwell_time` support to `MPC_GATO.run_mpc_goals`; default is zero
  so existing callers keep their behavior. The Tiago pick/place script was then
  changed to the stricter requested scenario: roughly 30 cm Cartesian goal
  spacing, 1 cm threshold, and 0.5 s dwell before switching. This currently
  exposes a real limitation: the example times out after reaching only 1 of 4
  goals with the current Tiago controller/horizon/tuning.
- Compared the tracking path against existing plants. The existing Indy7
  tracking smoke test passes but is very weak: it runs for 0.03 s and accepts
  about 0.41 m average error. A normal N16 Indy7 one-shot solve hit a CUDA
  illegal memory access. IIWA14 N8 centered figure-8 tracking over 0.5 s worked
  with about 1.8 cm mean error, though motion was mostly in X and very small in
  Y/Z. This suggests the large Tiago pick/place failure is not simply because
  all existing examples are healthy; the example suite has mixed quality.
- Added `examples/iiwa14_pickplace_goals.py` as a working large-step
  pick/place-style baseline. It uses segmented Cartesian references toward
  each goal, a 1 cm threshold, and 0.3 s dwell. Verified it reaches 4/4 goals
  and writes PNG/GIF artifacts under ignored `test-artifacts/iiwa_pickplace/`.
- Revisited the Tiago large-goal failure from a debugging perspective rather
  than continuing blind tuning. Direct `sim_forward` comparisons show Tiago
  CUDA forward dynamics match Pinocchio ABA closely at sampled states, so the
  first-order suspicion moved away from raw dynamics/sign mapping. However,
  zero-torque Tiago poses can have very large accelerations, and the current
  Tiago multi-goal script still times out on the first goal without an
  IK-shaped warm start. An IK-shaped warm start can drive the first reduced
  target under 1 cm, but this should be treated as a diagnostic result until
  the dynamics gradients, tracking cost derivatives, Hessian convention, and
  reference layout are explicitly validated.
- Added temporary `@TODO: remove` debug bindings/kernels to expose plant
  dynamics gradients and tracking derivatives from the compiled solver. Tiago
  checks passed: qdd matched Pinocchio ABA to about `1e-4`, dynamics gradient
  relative error was about `1e-7`, tracking gradient relative error was about
  `5e-6`, and the Gauss-Newton tracking Hessian matched finite differences at
  zero residual. This makes a basic dynamics/sign/gradient bug less likely.
- Tested the Tiago pick/place example at N16, N32, and N64. With the current
  IK-shaped segment warm start and roughly 28-31 cm goal spacing: N16 reaches
  goal 1 then stalls about 9.6 cm from goal 2; N32 reaches about 5.1 cm from
  goal 1; N64 reaches about 16.6 cm from goal 1. Scaling local segment length
  by horizon improves N32 to about 2.8 cm but still misses the 1 cm threshold,
  while N64 remains worse. Longer horizon alone is therefore not solving the
  Tiago large-goal task.
- Added `examples/debug_tiago_goal_step.py` to inspect the first failing Tiago
  goal transition. It showed the solver was often preserving the warm start
  exactly: PCG hit the iteration cap, line search step size stayed at `-1`, and
  the first control did not change. The deeper issue was the warm-start IK
  branch: the original goal sequence pushed the local IK to joint limits, where
  the next segment was still kinematically reachable from other seeds but not
  from the local branch.
- Replaced the Tiago pick/place goals with a roughly 33-35 cm Cartesian cycle
  sampled from one smooth, moderate joint-space branch. Verified
  `PYTHONPATH=python python examples/tiago_arm_pickplace_goals.py` reaches
  4/4 goals with a 1 cm threshold and 0.3 s dwell, producing PNG/GIF artifacts
  under ignored `test-artifacts/tiago_pickplace/`.
- Removed the temporary derivative-debug surface after it served its purpose:
  deleted the debug kernel/example and removed the BSQP/pybind debug methods.
  The remaining `python/bindings.cu` diff is only the real Tiago module suffix.
- Reviewed the current Tiago right-arm implementation changes, excluding the
  new example scripts. Main review focus was the plant wrapper's sign mapping
  and custom FK/Jacobian path, the Python Tiago model reduction, and the
  external-force-disabled behavior. Noted that batch MPC can still enable the
  force estimator and then fail on Tiago even when no external force was
  requested, and that the handwritten Tiago FK constants are the largest
  simplicity/maintenance risk.
- Added `tiago_right` to the default CMake plant matrix. Moved the Tiago
  no-external-wrench guard into C++ as well: the pybind method now rejects
  nonzero Tiago wrench batches before copying to the solver, and the Tiago
  plant external-wrench overloads assert that any device wrench pointer
  contains only zeros before falling back to the no-force dynamics path.
  Verified with `cmake --build build-tiago -j2` and a direct pybind smoke check
  that nonzero Tiago wrench input raises `ValueError`.
- Revisited the current Tiago URDF support decisions across the generator,
  Python BSQP interface, and plant wrapper. The non-GRiD-required choices to
  re-evaluate are the arm-only `torso_lift_link` root, hard-coded handwritten
  FK/Jacobian constants for `arm_right_tool_link`, tracking position-only goals
  inside a 6D end-effector API shape, uniform 0.1 shrinking of joint/velocity
  and torque limits, omission of URDF damping/friction, and use of a reduced
  full-URDF Pinocchio model for Python FK while CUDA dynamics use the extracted
  arm-only GRiD model.
- Checked limit handling against Indy7 and IIWA14. Those wrappers also hard-code
  URDF-derived joint, velocity, and effort limits and apply the same `-0.1`
  `JOINT_LIMIT_MARGIN()` to all three limit families. This is shared legacy
  wrapper behavior rather than a Tiago-specific requirement. For Tiago, the
  generation script already has the selected URDF joints, so emitting a small
  wrapper constants header from the source URDF would be cleaner than hand
  maintaining the arrays in `tiago_right_plant.cuh`.
- Implemented Tiago limit extraction in `tools/generate_tiago_dynamics.py`.
  The script now writes `tiago_<arm>_limits.cuh` from the native source URDF,
  with raw position limits and symmetric velocity/effort limits and no solver
  safety margins. `tiago_right_plant.cuh` now includes that generated header and
  no longer defines `JOINT_LIMIT_MARGIN()`. Verified the Python extraction path
  and a direct `nvcc` include compile of the Tiago plant. A full CMake build is
  currently blocked before compilation by the missing `tests/cpp/cuda_access_smoke.cu`
  source, and a full generator smoke run in the default Python environment is
  blocked by missing GRiD dependency `lxml`.
- Added a repo-local GRiD venv hint to `tools/generate_tiago_dynamics.py` and
  regenerated Tiago headers with `.venv_grid/bin/python`. Added the missing
  `tests/cpp/cuda_access_smoke.cu` source expected by CMake, then verified
  `cmake --build build-tiago -j2` completes.
- Reworked the Tiago pick/place example circuit to start from a configuration
  with meaningful raw-limit clearance, added a preflight IK limit-clearance
  check, and recorded arm link positions during rollout. Verified the circuit
  before solving: minimum raw joint-limit clearance was `0.584 rad`, maximum
  joint-space segment was `0.537 rad`. Ran
  `PYTHONPATH=python python examples/tiago_arm_pickplace_goals.py`; it reached
  `4/4` goals and generated the existing path artifacts plus a new 300-frame
  3D arm GIF at `test-artifacts/tiago_pickplace/tiago_pickplace_arm3d.gif`.
- Checked whether the Tiago pick/place example still needs the IK-shaped warm
  start by monkey-patching `build_segment_warm_start` to return `None`. With the
  same safer circuit and solver settings, the no-IK-warm-start run reached only
  `2/4` goals and timed out on goal 3 with final error about `6.4 cm`. The
  IK-shaped warm start is therefore still needed for the current strict
  `1 cm` pick/place example, though future solver/reference tuning might remove
  that dependency.
- Changed Tiago URDF extraction so `tools/generate_tiago_dynamics.py` now emits
  a native arm-only Pinocchio URDF at
  `gato/dynamics/tiago_right/tiago_right_arm.urdf` in addition to the
  GRiD-normalized temporary URDF and generated CUDA headers. Tiago examples now
  use that generated arm URDF directly instead of recreating temporary arm URDFs
  at runtime, and the Python BSQP interface validates a 7-DoF generated Tiago
  arm model instead of relying on reduced full-URDF dynamics for the normal
  path.
- Moved the Tiago default `comfortable` start pose away from the raw joint
  limits. The previous pose was only about `0.034 rad` from joint 4's upper raw
  limit, which became more visible after removing the artificial wrapper limit
  margin. Verified `examples/tiago_arm_tracking_simple.py` and
  `examples/tiago_arm_mpc_tracking_simple.py` against the generated arm-only
  URDF, and reran the pick/place example successfully with `4/4` goals reached.
- Simplified `python/bsqp/interface.py` back toward the original implementation:
  it now directly builds the URDF path supplied by the caller instead of
  carrying Tiago-specific full-URDF reduction/fallback logic. With
  `tiago_right_arm.urdf` generated and used by Tiago examples, alignment between
  Pinocchio and the generated CUDA plant is an input artifact responsibility
  rather than hidden interface behavior. Verified the simple Tiago tracking and
  one-goal MPC smoke examples still run.
- Investigated why the regenerated Tiago GRiD header lacks end-effector helper
  functions. The current checked-in `GRiD/` submodule's `gen_all_code()` only
  emits dynamics, Minv, forward-dynamics, and derivative helpers; there are no
  `end_effector_*` codegen routines in that source tree. Regenerating from the
  existing IIWA grid URDF with the same submodule also omits EE helpers, so the
  checked-in Indy7/IIWA14 `*_grid.cuh` files came from a different or manually
  modified GRiD output path. This is not caused by Tiago's URDF shape.
- Switched the `GRiD` submodule URL and checkout from `robot-acceleration/GRiD`
  to `A2R-Lab/GRiD` at `0a6c18e`, with nested A2R submodules initialized over
  HTTPS locally. Updated the Tiago generator for the A2R CLI and pass
  `--fixed-target-names arm_right_tool_joint`, which generates fixed tool-frame
  EE pose and gradient helpers. Regenerated `tiago_right_grid.cuh`.
- Removed the handwritten Tiago CUDA FK/Jacobian constants and now call the
  generated fixed-target `arm_right_tool_joint` EE helpers from the Tiago plant,
  keeping only the joint-6 sign mapping at the wrapper boundary. Verified an
  arch-61-only build, simple Tiago tracking, one-goal MPC tracking, and the
  pick/place example; pick/place still reached `4/4` goals and regenerated the
  path and 3D arm GIF artifacts.
- Re-evaluated the GRiD-specific Tiago preprocessing after switching to
  A2R-Lab GRiD. Negative-axis normalization is still required: running A2R GRiD
  directly on the native generated arm URDF with `arm_right_6_joint` axis
  `0 0 -1` still fails URDF parsing. The joint-origin defaulting is also still
  required because A2R's parser indexes both `origin["xyz"]` and
  `origin["rpy"]`, while Tiago's `arm_right_tool_joint` source origin only has
  `xyz`.

## 2026-04-29

### Tiago pick/place evaluation

- Evaluated `examples/tiago_arm_pickplace_goals.py` with the existing
  `bsqpN16_tiago_right` Python extension. CUDA access smoke and direct module
  import both passed before the example run.
- The pick/place example completed successfully: reached `4/4` goals,
  preflight minimum joint-limit clearance was `0.584 rad`, maximum joint-space
  segment was `0.537 rad`, and final end-effector error was `0.008884 m`,
  under the built-in `0.01 m` success threshold.
- The run generated/updated artifacts under
  `test-artifacts/tiago_pickplace/`, including the summary PNG, path GIF, and
  3D arm GIF.

### Adaptive Tiago benchmark patch

- Applied `/workspace/GATO_benchmark.patch` in the main `/workspace/GATO`
  worktree. The patch adds SQP/PCG timing and PCG residual reporting, fixes the
  merit-integrator shared-memory overlap needed for reliable benchmarking, and
  updates the Tiago pick/place benchmark to simulate for measured solve time
  with `N=32`, `max_sqp_iters=10`, `max_pcg_iters=400`, and `pcg_tol=1e-2`.
- Reconfigured `build-tiago` for the narrow benchmark build
  (`PLANT=tiago_right`, `KNOTS=32`, `CMAKE_CUDA_ARCHITECTURES=61-real`) and
  rebuilt `bsqpN32_tiago_right`.
- Reran `examples/tiago_arm_pickplace_goals.py`. The adaptive benchmark timed
  out on the first goal after about `20.027 s` simulated time. Mean control
  period was `30.297 ms` (`33.006 Hz`), mean SQP time was `30.132 ms`, mean SQP
  iterations were `10.00`, mean PCG time was `2.729 ms`, mean PCG iterations
  were `399.47`, and `99.9%` of PCG solves hit the 400-iteration cap.

### Disturbance-rejection example review

- Reviewed the figure-8 disturbance-rejection implementation. The example uses
  one batched solver with per-batch external-wrench parameters, not separate
  compiled dynamics models. The controller generates a force batch, runs a full
  batched SQP solve under those force hypotheses, then selects the optimized
  trajectory whose one-step dynamics prediction best matches the measured state
  transition.
- The model-consistency test itself only needs batched one-step dynamics
  simulation, but the current example precomputes optimized controls for every
  hypothesis so the chosen model already has a matching trajectory available.

## 2026-05-02

### IIWA14 pick/place PCG convergence check

- Instrumented `examples/iiwa14_pickplace_goals.py` at runtime by wrapping
  `BSQP.solve`, leaving the example's goals, horizon, costs, dwell, and timeout
  behavior unchanged.
- The IIWA14 circuit reached `4/4` goals with final error `0.005852 m`.
  Across `495` solve calls and `2429` recorded PCG solves, no PCG solve hit the
  configured `160`-iteration cap. PCG iterations ranged from `1` to `23`, with
  mean `16.54`, median `19`, and 95th percentile `21`.

### Tiago PCG failure capture

- Added `examples/debug_tiago_pcg_failure.py`, a replay/capture harness for the
  current Tiago pick/place example. It reuses the same model, reference,
  horizon, costs, warm start, dwell, and timeout setup, then writes the exact
  `x`, reference, warm start, solution, and exposed solver stats for a selected
  solve under ignored `test-artifacts/tiago_pickplace/`.
- Captured the first solve where all ten SQP iterations hit the `400` PCG
  iteration cap at `test-artifacts/tiago_pickplace/first_pcg_cap_solve.npz`.
  Replaying the capture reproduces the zero accepted trajectory update and all
  line-search step sizes at `-1`; PCG cap fraction varies between runs, but the
  rejected line search is deterministic at the trajectory level.
- Also captured the initial rejected solve at
  `test-artifacts/tiago_pickplace/first_rejected_solve.npz`. Its first
  Cartesian segment is only `0.08 m`, the warm start moves in the correct
  direction, but the solver still returns the warm start unchanged with
  line-search step sizes all `-1`.
- A/B tested the initial rejected solve with rho adaptation disabled. PCG then
  converged easily (`0` cap hits, max `17` iterations), but line search still
  rejected every step and the solution remained identical to the warm start.
  This suggests the earliest failure is the SQP step/merit acceptance path;
  PCG cap hits are likely downstream from repeated rejection and rho
  adaptation rather than the primary cause.
- Extended the debug stats to expose raw line-search alpha merits, the last
  proposed `dz`, and the last KKT dynamics `A`, `B`, and `c` arrays. Rebuilt
  focused debug modules in `build-tiago-debug-n32/` and
  `build-iiwa-debug-n16/`.
- Replayed `first_rejected_solve.npz` and recomputed the merit terms on CPU
  with Pinocchio. The CPU merit matched the GPU alpha merits within float
  precision, which rejects the hypothesis that the line-search merit kernel is
  stale or computing a different cost/constraint. For the Tiago warm start,
  merit was about `-6.553`, with dynamics constraint only about `0.0108`.
  The best tiny alpha from one replay improved cost by only about `0.00013`
  while increasing dynamics constraint to about `0.292`, so rejection was
  correct under the current merit.
- Validated the exposed Tiago KKT `A`/`B` layout against Pinocchio finite
  differences at several knots; relative errors were around `1e-5` or lower.
  With rho adaptation enabled, the later proposed `dz` can violate even the
  linearized dynamics constraints by thousands in L1 after PCG/rho struggles.
  With rho adaptation disabled, PCG remains easy and the linearized residual is
  much smaller, but all line-search candidates still increase merit.
- Compared the first IIWA14 pick/place solve with the same debug surface. IIWA
  initially rejects one SQP candidate, but later iterations find accepted
  alphas and reduce merit from about `15.58` to `8.22`. This reinforces that
  the Tiago failure is not a generic line-search or merit-kernel failure.

### Remove Tiago IK warm start

- Removed the IK-shaped solver initialization from
  `examples/tiago_arm_pickplace_goals.py`. The Tiago example now follows the
  IIWA14 pattern: initialize the solver trajectory from the current state once,
  then shift the previous solution forward after each control update. The IK
  helper remains only for preflight reachability/limit-clearance validation.
- Updated `examples/debug_tiago_pcg_failure.py` so new captures use the same
  no-IK warm-start policy as the Tiago example.
- Verified the IIWA14 example already uses no IK warm start:
  `initialize_warm_start(...)` plus `shift_warm_start(...)`.
- Ran the no-IK Tiago pick/place example. It timed out on the first goal with
  `0/4` goals reached, but the result is numerically cleaner: final active-goal
  error was `0.016251 m`, mean error was `0.025649 m`, mean SQP time was
  `1.015 ms`, mean SQP iterations were `3.46`, mean PCG iterations were
  `2.45`, and PCG max-iteration fraction was `0.000`.
- Relaxed the Tiago pick/place goal threshold from `0.01 m` to `0.05 m`.
  With the no-IK warm start, the example reached `2/4` goals and then timed
  out on goal 3. The run reported mean SQP time `9.042 ms`, mean SQP
  iterations `6.01`, mean PCG iterations `177.77`, PCG cap-hit fraction
  `0.441`, and final active-goal error `1.191228 m`.
- Compared IIWA14 and Tiago pick/place setup before continuing the bug hunt.
  IIWA uses `N=16`, fixed `DT=0.03 s` simulation, `0.01 m` threshold,
  `0.05 m` segment reference cap, `max_sqp_iters=5`,
  `max_pcg_iters=160`, and reaches `4/4`. Tiago currently uses `N=32`,
  measured-solve-time simulation despite solver `DT=0.03 s`, `0.05 m`
  threshold, `0.08 m` segment cap, `max_sqp_iters=10`,
  `max_pcg_iters=400`, and reaches `2/4`. The biggest setup confounder is
  that Tiago advances simulated time by wall-clock solve duration while IIWA
  advances by the model timestep.
- Normalized Tiago to the IIWA setup while keeping the requested `0.05 m`
  threshold: `N=16`, fixed `DT=0.03 s` control simulation, `0.05 m`
  reference segment cap, `max_sqp_iters=5`, `kkt_tol=1e-4`,
  `max_pcg_iters=160`, `pcg_tol=1e-4`, `q_cost=20`, `qd_cost=5e-2`,
  `u_cost=1e-6`, `N_cost=400`, `q_lim_cost=0.01`, `rho=0.02`, `mu=1`,
  and zero velocity/control limit costs. Rebuilt `bsqpN16_tiago_right` for
  this configuration.
- The normalized Tiago run timed out on goal 1 with `0/4` goals reached.
  It reported mean SQP time `5.239 ms`, mean SQP iterations `5.00`, mean PCG
  iterations `159.90`, PCG cap-hit fraction `0.999`, mean error `0.242111 m`,
  and final active-goal error `0.192004 m`. This makes the PCG/rho failure
  reappear when Tiago uses the same tighter IIWA solver parameters.
- Changed the Tiago pick/place loop to continue to the next goal after a
  per-goal timeout instead of aborting the circuit. Under the normalized setup,
  the all-goals run produced outcomes `['timeout', 'reached', 'timeout',
  'timeout']`, so only goal 2 reached the `0.05 m` dwell condition. Aggregate
  stats were mean SQP time `5.156 ms`, mean PCG iterations `159.97`, PCG
  cap-hit fraction `1.000`, mean error `0.196598 m`, and final active-goal
  error `0.121274 m`.
- Rechecked the "figure-eight by accident" hypothesis. The Tiago pick/place
  code path does not call `MPC_GATO.run_mpc_fig8`, but the Tiago pick/place
  target offsets were still sampled from the old right-arm loop, with norms
  up to about `0.254 m`. That meant the earlier "5 cm goal" change had only
  changed the success threshold, not the requested motion. Replaced those
  loop-derived offsets with a small `0.05 m` Cartesian circuit and restored the
  reach threshold to `0.01 m`.
- Tested solver-state variants for the first Tiago goal. Stateful shifted
  warm starts kept PCG near the cap and produced cycling. Fresh current-state
  warm starts plus `reset_dual()` and fixed rho tracked the first old large
  goal to about `0.0085 m`, while stale dual state or rho adaptation made the
  result unstable. Updated the Tiago pick/place script to use fresh
  current-state warm starts, `reset_dual()`, `reset_rho()`, and
  `adapt_rho=False`.
- The corrected 5 cm circuit still does not satisfy the strict `0.01 m` dwell
  condition. The first goal initially moves in the right direction from
  `0.0500 m` to about `0.0167 m`, then drifts away; the full four-goal run
  timed out on all goals with mean error `0.124118 m`, final error
  `0.097973 m`, and PCG cap-hit fraction `0.841`. A quick check showed Tiago's
  scalar tracking cost and KKT gradient/Hessian both use XYZ only, so zero
  orientation slots are not the immediate cost mismatch.
- Confirmed the apparent "constant point" tracking came from executing a
  fresh zero-control warm start whenever a solve rejected every line-search
  candidate. Those zero-torque bursts let the gravity-loaded arm drift along a
  goal-independent passive path. Added a Tiago pick/place fallback that uses
  Pinocchio gravity compensation when no SQP step is accepted. With the 5 cm
  circuit and strict `0.01 m` threshold, the example now reaches `4/4` goals in
  `4.860 s`; mean error is `0.019240 m`, final error is `0.008675 m`, mean SQP
  time is `2.770 ms`, mean PCG iterations are `61.35`, and PCG cap-hit
  fraction is `0.309`.
- Changed the Tiago pick/place circuit to use a strict `0.01 m` reach
  threshold and `0.25 m` Cartesian spacing between consecutive goals. A naive
  25 cm square is not reachable from the comfortable start, so the script now
  normalizes the previously validated reachable workspace directions to 25 cm
  per segment. The resulting circuit passes the IK preflight but the MPC run
  still times out on all four goals: `0/4`, mean error `0.194346 m`, final
  error `0.093527 m`, mean SQP time `1.880 ms`, mean PCG iterations `26.72`,
  and PCG cap-hit fraction `0.097`.
- Diagnosed the 25 cm local-state behavior. The 5 cm rolling reference cap was
  too local for this larger move: goal 1 got down to about `0.051 m`, then left
  the goal basin. Setting the reference cap to the full `GOAL_DISTANCE`, raising
  `qd_cost` to `0.1`, and using damped gravity fallback on rejected solves
  (`qdd_desired = -5 qd`, clipped to `20 rad/s^2`) made the 25 cm circuit
  reach all goals. The artifact-generating run reported `4/4` goals reached in
  `6.690 s`, mean error `0.062246 m`, final error `0.008239 m`, mean SQP time
  `3.061 ms`, mean PCG iterations `66.83`, and PCG cap-hit fraction `0.343`.
- Re-enabled rho adaptation for the same corrected 25 cm Tiago setup. The run
  still reached `4/4` goals in `6.630 s`; mean error was `0.059997 m`, final
  error `0.008551 m`, mean SQP time `3.130 ms`, mean PCG iterations `68.26`,
  and PCG cap-hit fraction `0.344`. Rho diagnostics are still much larger with
  adaptation enabled (`mean_pcg_rho_initial` about `1.73e18`,
  `mean_pcg_rho_final` about `2.00e21`), so the task works but rho scaling
  remains suspicious.
- Documented a branch-repair plan in `REFACTOR_REPOSITORY_PLAN.md`. Current
  assessment: `db73d15` is the cleaner Tiago implementation that landed on
  `main` by mistake; local `tiago` has the important build/runtime bugfix chain;
  the dirty worktree mixes useful pick/place experiment fixes with debug
  instrumentation. The plan preserves current work in a WIP commit, recreates
  `main` at `a06607a`, cherry-picks the clean Tiago implementation onto
  `tiago`, reapplies experiment fixes selectively, rebuilds ignored extension
  artifacts, validates the pick/place circuit, then cleans old branches.
- Started the branch repair. Created safety branch
  `safety/pre-repo-refactor-main`, committed the dirty state as
  `98624f6 [tiago] WIP preserve pickplace debugging state`, renamed the
  misplaced `main` to `felix_experiments`, recreated `main` at `a06607a`, and
  cherry-picked the clean Tiago implementation onto `tiago` as
  `c5d7f01 [tiago] tiago impl + cuda smoke test`. Reapplied the WIP changes on
  `tiago` as a dirty worktree and kept the already-fixed `tiago` version of
  `merit.cuh`.
- Rebuilt `bsqpN16_tiago_right` from the repaired `tiago` worktree using
  `build-tiago-n16`, then reran `examples/tiago_arm_pickplace_goals.py`. The
  repaired branch plus dirty experiment fixes reached `4/4` goals in
  `6.630 s`; mean error `0.059997 m`, final error `0.008551 m`, mean SQP time
  `3.091 ms`, mean PCG iterations `68.26`, and PCG cap-hit fraction `0.344`.

### Track uv lockfile

- Removed the root `uv.lock` ignore rule so the lockfile can be committed for a
  reproducible Python setup.
- Reverted that commit from `main` after realizing it belonged on the `tiago`
  branch, and force-updated `origin/main` back to the previous commit.
- Created the lockfile-tracking change on `tiago` and pushed the remote branch
  via the GitHub connector because the container lacks SSH push credentials.

## 2026-05-05

### Rebase onto ICRA-26

- Created `felix-devel` from `origin/ICRA-26` and started replaying the
  upstreamable GATO changes in semantic units.
- Replayed the local workflow ignore changes as `[dev]`, then integrated the
  Docker/setup and lockfile work as `[gato] Fix build setup`.
- Added the CUDA architecture verifier, release-mode CUDA error checks, and the
  two shared-memory access bugfixes as separate `[gato]` commits.
- Cherry-picked the remaining test/build-helper work onto `felix-devel` as
  `[gato] Add incremental build helpers` and
  `[gato] Add build-backed smoke tests`.
- Regenerated `uv.lock` after adding the pytest-backed smoke tests so the
  tracked lockfile remains consistent with `pyproject.toml`.
- Cherry-picked the Tiago implementation and pick/place validation examples
  onto `felix-devel` after resolving the `CMakeLists.txt` plant-list conflict
  by keeping the ICRA/test build structure and adding `tiago_right` to the
  supported plant targets.

### Tiago pick/place cleanup

- Cleaned the branch-repair worktree so the remaining main-repo changes are the
  two pick/place examples. Removed the temporary solver debug API changes from
  the core BSQP, pybind, and Python interface files, deleted exploratory debug
  examples, and cleared ignored build output.
- Rebuilt `bsqpN16_tiago_right` from the cleaned `tiago` worktree and reran
  `examples/tiago_arm_pickplace_goals.py`. The example still reached `4/4`
  goals in `6.630 s` with final error `0.008551 m`, mean SQP time about
  `3.19 ms`, mean PCG iterations `68.26`, and PCG cap-hit fraction `0.344`.
- Checked whether the rejected-step fallback was dead code by monkey-patching
  it to raise. The run still uses that fallback, so the example now reports
  `rejected_solves` explicitly instead of hiding the behavior in the summary.
- Removed unreported timing/debug bookkeeping from the pick/place examples:
  unused PCG timing capture, Tiago `q_history`, and unreported reached-time
  arrays. Reran the Tiago pick/place example successfully after the cleanup.
- Committed the cleaned pick/place examples on `tiago` as
  `4a149fc [tiago] Add pickplace validation examples`.
- Removed obsolete local cleanup branches:
  `felix_experiments`, `safety/pre-repo-refactor-main`,
  `tiago_experiments`, and `wip/tiago-arm-snapshot`. Also removed the local
  `origin/tiago_experiments` remote-tracking ref.
- Repaired GitHub `main` through the connector by moving it back to
  `a06607a`. A normal SSH push from the container is still blocked by missing
  SSH credentials, so publishing local `tiago` remains a credentialed-push
  follow-up.
- Validated the checked-in Tiago pick/place example on the GTX 1070 container.
  Recreated the normal `build/` cache for `/workspace/GATO` with
  `--plant tiago_right --knots 16 --native-cuda-arch`, rebuilt
  `bsqpN16_tiago_right`, and ran `examples/tiago_arm_pickplace_goals.py`.
  The run reached `4/4` goals in `6.630 s`, final error `0.008551 m`, mean
  control period `3.370 ms`, mean SQP time `3.227 ms`, mean PCG iterations
  `68.26`, PCG cap-hit fraction `0.344`, and `26` rejected solves. Artifacts
  were generated under `test-artifacts/tiago_pickplace/`.
- Tried the same Tiago pick/place circuit with `32` knots by rebuilding
  `bsqpN32_tiago_right` and overriding the example `N` global at runtime. The
  default `MAX_PCG_ITERS=160` run failed validation: `0/4` goals reached, all
  goals timed out over `80.040 s`, final error `0.053475 m`, mean SQP time
  `7.338 ms`, `2638` rejected solves, and PCG cap-hit fraction `0.935`. A
  focused retry with `MAX_PCG_ITERS=320` and artifact rendering disabled also
  failed: `0/4` goals reached, final error `0.082002 m`, mean SQP time
  `9.625 ms`, `2099` rejected solves, and PCG cap-hit fraction `0.607`.
- Added `examples/tiago_arm_timed_pose_circuit.py`, a separate Tiago timed
  end-effector position circuit. Each waypoint has an xyz target and a segment
  duration; the first duration is `None`. The example validates all waypoint
  positions with IK, then constructs each MPC reference from the current tool
  position to the active target over the remaining segment time and holds the
  target through the rest of the horizon. It waits `0.5 s` at each reached
  waypoint and writes an arm-motion GIF with IK ghost arms and expected target
  positions. Validation reached `5/5` poses in `7.470 s`, final error
  `0.006066 m`, mean SQP time `2.978 ms`, mean PCG iterations `62.27`, PCG
  cap-hit fraction `0.309`, and `43` rejected solves.
- Pivoted away from the timed-target prototype and removed the untracked
  example before committing it. Added `examples/plot_fig8_tracking.py` instead,
  a deterministic time-indexed figure-8 tracking plotter that runs one plant
  per Python process to avoid pybind class-registration collisions. Rebuilt and
  ran `bsqpN16_iiwa14` first: mean error `0.022628 m`, max error
  `0.036670 m`, final error `0.004725 m`, mean SQP time `1.418 ms`, and `11`
  rejected solves. The initial Tiago run was poor, so the script now uses the
  Tiago-stable fresh-warm-start/reset-dual policy and a stronger Tiago
  tracking weight (`q_cost=80`). The corrected `bsqpN16_tiago_right` run had
  mean error `0.007534 m`, max error `0.016468 m`, final error `0.002387 m`,
  mean SQP time `2.904 ms`, and `24` rejected solves. Plots and coordinate
  time-series views were written under `test-artifacts/fig8_tracking/`.
- Checked the copied `examples/trajfiles/` data against Tiago before building a
  larger visual trajectory demo. The files are IIWA-format trajectories:
  `*_traj.csv` rows have `21` columns (`q`, `qd`, `u` for a 7-DoF arm), and the
  only copied `*_eepos.traj` is `0_0_eepos.traj`. The C++ benchmark shape would
  require matching `*_eepos.traj` files for the other start/goal pairs, which
  are missing. The `0_0_eepos.traj` positions match IIWA FK for
  `0_0_traj.csv`, but are not usable as raw Tiago torso-frame targets: sampled
  Tiago IK over the raw path had mean error about `0.78 m` and max error about
  `1.27 m`.
- Added `examples/tiago_large_trajectory_tracking.py`, a Tiago-native visual
  tracking demo that recreates the copied trajectory-file idea with a reachable
  torso-frame end-effector path instead of reusing IIWA poses. The script
  generates five smooth 0.22 m Cartesian segments, validates sampled targets
  with Tiago IK, runs N=16 MPC, and writes a 3D plot, GIF, reference CSV, and
  actual tracked path under `test-artifacts/tiago_large_tracking/`.
- Tuned the Tiago large tracking demo after finding that carrying only the
  shifted primal warm start caused almost all solves to reject because the
  Python wrapper does not expose a matching dual shift. The example now resets
  dual/rho and initializes a fresh primal each MPC tick. With `q_cost=160`,
  `qd_cost=0.02`, and `N_cost=800`, the validated run produced IK mean/max
  errors of `0.000248 m`/`0.000613 m`, tracking mean/p95/max/final errors of
  `0.002790 m`/`0.004411 m`/`0.010141 m`/`0.000326 m`, mean SQP time
  `2.681 ms`, `43` rejected solves, and PCG cap-hit fraction `0.266`.
- Investigated why the Tiago large-tracking GIF still appears small. The demo
  uses five 0.22 m segments for about `1.10 m` of total end-effector arc length,
  but the directions double back, so the axis-aligned footprint is only about
  `0.19 m x 0.17 m x 0.05 m`. The GIF also frames all arm-link positions, whose
  z-span is about `0.92 m`, so the target path occupies a small part of the
  rendered cube. IK remains reliable up to about `0.35 m` segments for this
  shape; `0.40 m` segments already showed sampled IK error around `2.8 cm`.
- Replaced the small doubled-back Tiago path with a workspace-extreme circuit.
  A deterministic FK sample showed the generated Tiago right-arm model has a
  broad end-effector envelope, roughly `1.5 m` in each torso-frame axis over
  joint limits. The updated example interpolates through reachable joint-space
  configurations for forward, left, high, back, right, and return waypoints,
  then tracks the FK end-effector trace. The regenerated reference footprint is
  `1.514 m x 1.537 m x 1.514 m` with about `8.20 m` of arc length over
  `29.880 s`. The validated run produced IK mean/max errors of
  `0.000372 m`/`0.000994 m`, tracking mean/p95/max/final errors of
  `0.011352 m`/`0.022694 m`/`0.066044 m`/`0.002440 m`, mean SQP time
  `2.551 ms`, `88` rejected solves, and PCG cap-hit fraction `0.250`.
- Reran the Tiago workspace-reach example with the same path and solver tuning
  at larger horizons using the existing `bsqpN32_tiago_right` and
  `bsqpN64_tiago_right` modules. N=32 produced tracking mean/p95/max/final
  errors of `0.021847 m`/`0.053105 m`/`0.108481 m`/`0.003810 m`, mean SQP time
  `4.809 ms`, `272` rejected solves, and PCG cap-hit fraction `0.461`.
  N=64 produced `0.029040 m`/`0.061719 m`/`0.159888 m`/`0.064916 m`, mean SQP
  time `6.865 ms`, `245` rejected solves, and PCG cap-hit fraction `0.372`.
  Artifacts were written under `test-artifacts/tiago_large_tracking/N32/` and
  `test-artifacts/tiago_large_tracking/N64/`. With unchanged tuning, increasing
  knots worsened tracking and solver acceptance on this example.
- Tested higher SQP/PCG iteration caps for the same Tiago workspace-reach
  trajectory. For N=32, `max_sqp_iters=10` and `max_pcg_iters=320` was the best
  sweep setting but only modestly changed the result (`0.020521 m` mean,
  `0.043319 m` p95, `0.099007 m` max, `252` rejected solves) at a much higher
  `18.384 ms` mean SQP time; a repeat artifact run was worse
  (`0.023851 m`/`0.063760 m`/`0.168804 m`, `337` rejected solves). For N=64,
  increased caps did not help: `max_sqp_iters=8`, `max_pcg_iters=320` produced
  about `0.032348 m` mean and `0.144137 m` max in the sweep, while
  `max_sqp_iters=10`, `max_pcg_iters=320` diverged and `10`/`640` was slower and
  still worse. Artifact runs were written under
  `test-artifacts/tiago_large_tracking/N32_sqp10_pcg320/` and
  `test-artifacts/tiago_large_tracking/N64_sqp8_pcg320/`.
- Ran an N=32 proof-of-concept where solver wall time is allowed to exceed the
  MPC tick by a lot. The full workspace-reach run with `max_sqp_iters=20` and
  `max_pcg_iters=1000` completed with tracking mean/p95/max/final errors of
  `0.021836 m`/`0.049389 m`/`0.082640 m`/`0.004483 m`, mean SQP time
  `122.177 ms`, `297` rejected solves, mean PCG iterations `743.09`, and PCG
  cap-hit fraction `0.729`. This is not better on mean error than the N=32
  baseline, though it reduced the worst spike. Short 150-step prefix checks
  with `20` SQP and `2000`/`4000` PCG iterations improved prefix errors but
  took `232.731 ms`/`467.676 ms` per solve and still hit the PCG cap about
  `71`-`73%` of the time, so full `4000`-PCG runs would be minutes-long without
  addressing the underlying formulation issue.
- Compared the N=32 high-cap workspace-reach experiment against the other
  plant modules. Generated IIWA14 and Indy7 workspace traces by sampling FK
  extremes and interpolating through the corresponding joint configurations.
  IIWA14 completed cleanly with `max_sqp_iters=20`, `max_pcg_iters=1000`,
  moderate plant weights (`q_cost=5`, `N_cost=100`, `qd_cost=0.05`), and a
  `1.790 m x 1.798 m x 1.523 m` footprint over `55.860 s`: tracking
  mean/p95/max/final errors were
  `0.023257 m`/`0.046254 m`/`0.059247 m`/`0.022001 m`, mean SQP time
  `9.342 ms`, `0` rejected solves, mean PCG iterations `28.15`, and PCG
  cap-hit fraction `0.000`. The existing Indy7 module crashed even on a
  one-step solve, so `bsqpN32_indy7` was rebuilt in `build-indy-n32/`. After the
  rebuild, Indy7 ran but the full workspace path was not successfully tracked:
  with the same moderate weights over a `1.973 m x 1.963 m x 1.956 m` footprint
  it produced `0.876232 m` mean error, `1594` rejected solves, mean PCG
  iterations `851.79`, and PCG cap-hit fraction `0.844`; a shorter/faster
  variant still had `0.255` cap-hit fraction and poor tracking. This suggests
  the frequent PCG cap hits are not universal across plants: IIWA14 is easy for
  PCG here, Tiago is cap-heavy, and Indy7 becomes cap-heavy on the extreme path
  but in a failed-tracking regime.
- Added `examples/large_easy_tracking.py`, an isolated experiment that builds
  smooth joint-space waypoint paths for Tiago right arm, IIWA14, and Indy7,
  converts them to end-effector xyz references with FK, resets dual/rho and
  uses a fresh primal warm start every tick, and writes CSV/plot artifacts under
  `test-artifacts/large_easy_tracking/`. With default N=32, DT=0.03,
  `max_sqp_iters=20`, and `max_pcg_iters=1000`, Tiago still hit the PCG cap
  heavily on the smoother large path (`0.744` cap-hit fraction, mean PCG
  `758.63`, mean error `0.023750 m`, `167` rejected solves). IIWA14 remained
  PCG-easy (`0.000` cap-hit fraction, mean PCG `30.06`, mean error
  `0.031928 m`, no rejected solves). Indy7 ran with the existing N=32 module
  but tracked poorly (`0.157698 m` mean error, `32` rejected solves) and had a
  smaller but nonzero PCG cap-hit fraction of `0.072`.
- Extended `examples/large_easy_tracking.py` to capture arm link positions and
  render `tracking.gif` alongside `tracking.png`. Reran all three plants with
  default N=32 high-cap settings. The rendered outputs are
  `test-artifacts/large_easy_tracking/tiago_right/tracking.gif`,
  `test-artifacts/large_easy_tracking/iiwa14/tracking.gif`, and
  `test-artifacts/large_easy_tracking/indy7/tracking.gif`. The rerun metrics
  were consistent with the previous observation: Tiago cap-hit fraction
  `0.709` with mean error `0.021811 m`, IIWA cap-hit fraction `0.000` with mean
  error `0.031928 m`, and Indy cap-hit fraction `0.072` with mean error
  `0.157698 m`.
- Inspected the sibling `/workspace/MPCGPU` repository for QDLDL integration.
  MPCGPU selects PCG vs QDLDL with the compile-time `LINSYS_SOLVE` define and
  its QDLDL path is host-side: it forms Schur values on the GPU, copies CSR
  values and the RHS to host, calls `QDLDL_factor`/`QDLDL_solve`, then copies
  lambda back to device. In GATO, `BSQP::solve` currently always calls
  `solvePCGBatched` after `formSchurSystemBatched`. The least invasive QDLDL
  option would therefore be a compile-time CMake backend that reuses GATO's
  KKT/Schur/dz/line-search code and swaps only the Schur solve boundary, with a
  conversion from GATO's padded block-row Schur storage to QDLDL CSC/CSR lower
  triangle. The MPCGPU QDLDL submodule is present but not initialized in the
  mounted checkout, so a real build would also need vendoring or fetching
  QDLDL.
- Implemented an optional CMake linear-system backend with
  `GATO_LINSYS_SOLVER=PCG|QDLDL`. The QDLDL path fetches a pinned OSQP/QDLDL
  revision, links it into the Python BSQP modules, exposes `LINSYS_SOLVER` on
  the generated modules, and swaps only the Schur solve call in `BSQP::solve`.
  The new QDLDL wrapper converts GATO's padded block-row Schur storage to
  upper-triangular CSC on the host, solves with QDLDL, and copies lambda back to
  device. This is a diagnostic direct-solver backend rather than a real-time GPU
  implementation because it copies and factors on the host every SQP iteration.
- Generated backend-separated large-easy-tracking diagrams for Tiago right arm,
  IIWA14, and Indy7 under `test-artifacts/large_easy_tracking/<plant>/pcg/` and
  `/qdldl/`, plus a combined
  `test-artifacts/large_easy_tracking/tracking_backend_comparison.png` and
  `backend_timing_summary.csv`. With N=32 and DT=0.03, QDLDL removed rejected
  solves in this experiment and gave nonzero measured direct-solve timing:
  Tiago `0.296 ms` mean linear solve and `9.077 ms` mean SQP time, IIWA14
  `0.293 ms` and `8.652 ms`, Indy7 `0.229 ms` and `6.490 ms`. Tiago tracking
  improved substantially versus PCG on this path (`0.0113 m` mean error vs
  `0.0256 m`, no rejected solves vs `186`). IIWA14 remained easy for both
  backends, while Indy7 still tracked poorly but QDLDL removed the rejected
  solves seen with PCG.
- Created a throwaway `/workspace/MPCGPU` branch
  `tiago-easy-pcg-validation` to check whether Tiago's PCG behavior reproduces
  in the original MPCGPU code path. The validation port copies GATO's Tiago
  generated dynamics/limits/URDF into MPCGPU, adds a Tiago plant selection path
  and a single `track_tiago_easy_pcg.cu` executable, generates the same smooth
  Tiago easy trajectory into MPCGPU's `traj.csv`/`eepos.traj` format, and runs
  N=32 PCG with `PCG_MAX_ITER=1000`. The run completed and dumped
  `tmp/results/tiago_easy_32_PCG_summary.json`: `9389` linear solves,
  mean PCG iterations `646.56`, p95 `1000`, max `1000`, and cap-hit fraction
  `0.601`. This confirms the cap-hit issue reproduces in MPCGPU on the copied
  Tiago easy path, though the MPCGPU rolling-window/simulation semantics are not
  identical to GATO's fresh-warm-start Python experiment.
- Tried to validate the same copied Tiago easy path with MPCGPU's QDLDL backend
  by adding `track_tiago_easy_qdldl.cu`. The first QDLDL runs produced NaNs
  partway through the path. A concrete validation-port bug was found in the
  tracking-error FK path: `mpcsim.cuh` called the generated Tiago GRiD
  end-effector kernel with native Tiago coordinates, while the plant cost and
  dynamics wrapper map `arm_right_6_joint` through the required sign flip before
  calling GRiD. Added a native-coordinate Tiago FK wrapper for the recording
  path. Even after that fix, matched rho (`INITIAL_RHO=0.02`), a larger QDLDL
  SQP budget (`SQP_MAX_TIME_US=20000`), and extra control regularization
  (`TIAGO_U_COST=1e-3`, `TIAGO_CTRL_LIM_COST=0.01`), QDLDL still diverged:
  finite state rows `1725/7501`, first non-finite recorded error at reference
  shift `114/500`, mean SQP time `17.49 ms`, max SQP iterations `38`, and zero
  `QDLDL_factor` failure messages. This means QDLDL tracking is not validated in
  the current MPCGPU Tiago port; the copied Tiago experiment/setup is still
  suspect independently of the direct solver.
- Continued debugging the MPCGPU Tiago validation port bottom-up with a new
  `examples/tiago_port_probe.cu` and `scripts/check_tiago_port.py` in
  `/workspace/MPCGPU`. The probe compares Tiago FK, FK gradients, forward
  dynamics, and forward-dynamics derivatives against Pinocchio and finite
  differences on sampled `tiago_easy_traj.csv` states. This found a real
  shared-memory layout bug in the copied Tiago pose-gradient path: the generated
  `load_update_XmatsHom_helpers(..., s_dXmatsHom, ...)` writes 128 floats into
  `s_dXmatsHom`, but the wrapper placed `s_temp` after only 112 floats. That
  caused nondeterministic end-effector pose corruption in the pose+gradient
  wrapper. Changed the Tiago wrapper to reserve 128 floats for `s_dXmatsHom` and
  increased `grid::DEE_POS_SHARED_MEM_COUNT` accordingly. After the fix, the
  probe reports FK agreement with Pinocchio at `2.35e-7 m` max, zero mismatch
  between pose-gradient FK and native FK for all probe rows, EE Jacobian finite
  difference max error `5.13e-4`, forward dynamics vs Pinocchio ABA max error
  `4.50e-4`, and forward-dynamics Jacobian vs Pinocchio ABA derivatives max
  error `2.46e-3`.
- Reran the MPCGPU Tiago easy tracking examples after the shared-memory fix.
  The low-level Tiago model math now checks out, but the high-level tracking
  behavior is still bad: PCG remains cap-heavy (`9131` linear solves, mean PCG
  iterations `659.84`, cap-hit fraction `0.599`) and QDLDL still goes
  non-finite (`1650/7501` finite state rows, first non-finite recorded tracking
  error at `109/500`). The trajectory itself is inside the URDF joint, velocity,
  and torque limits. The remaining failure appears to be formulation/setup
  rather than a Tiago FK/dynamics port bug: the copied MPCGPU end-effector
  tracking objective has no joint-reference/posture term, uses only weak
  velocity/control/limit regularization, and the simulated joint states leave
  the robot limits by very large margins before NaNs appear.
- Added tracking-cost derivative output to the MPCGPU Tiago probe after
  questioning whether the remaining issue could still be a different bug.
  With a larger finite-difference step appropriate for single precision
  (`1e-3`), `trackingCostGradientAndHessian` agrees with finite differences of
  `trackingcost` to max `1.79e-2` and mean `2.99e-3` on sampled rows, so the
  local cost gradient path is not showing a gross sign/layout bug. Also noted a
  likely formulation bug in the Tiago wrapper's barrier model: outside-limit
  `jointBarrier` cost clamps the violated side to a constant while the gradient
  still pushes back, so line search merit can under-penalize already-violated
  joint states. This can compound the EE-only underconstraint once states leave
  limits.
- Checked whether the MPCGPU Tiago shared-memory bug also exists in the GATO
  experiment. It does: `gato/dynamics/tiago_right/tiago_right_plant.cuh` places
  `s_temp` after `s_dXmatsHom + 112`, but the generated Tiago helper in
  `tiago_right_grid.cuh` copies/writes 128 floats into `s_dXmatsHom`. Because
  the GATO BSQP Tiago KKT path calls `trackingCostGradientAndHessian`, which
  calls `computeTiagoToolPoseAndGradient`, Tiago tracking experiments in GATO
  can be affected by the same end-effector pose/gradient shared-memory
  corruption. The equivalent fix is to reserve 128 floats for `s_dXmatsHom` in
  the wrapper and add 16 floats to the Tiago DEE shared-memory count used by the
  KKT shared-memory sizing.
- Fixed the GATO Tiago shared-memory overlap in
  `gato/dynamics/tiago_right/tiago_right_plant.cuh` by adding
  `grid::DEE_POS_SHARED_MEM_COUNT = DEE_POS_DYNAMIC_SHARED_MEM_COUNT + 16`,
  moving `s_temp` to `s_dXmatsHom + 128`, and using the corrected count in
  `trackingCostGradientAndHessian_TempMemSize_Shared`. Rebuilt the existing
  N=32 PCG and QDLDL builds, then reran
  `examples/large_easy_tracking.py --plant tiago_right` for both backends. Fresh
  artifacts were written under
  `test-artifacts/large_easy_tracking/tiago_right/pcg/` and `/qdldl/`. Post-fix
  PCG: mean error `0.021103 m`, p95 `0.058423 m`, max `0.090955 m`, final
  `0.002071 m`, `143` rejected solves, mean PCG iterations `705.10`, cap-hit
  fraction `0.6908`, mean SQP time `115.57 ms`. Post-fix QDLDL: mean error
  `0.011321 m`, p95 `0.025342 m`, max `0.028060 m`, final `0.000233 m`, no
  rejected solves, mean SQP time `9.068 ms`, mean QDLDL solve time `0.295 ms`.
  The fix is real, but these metrics are close to the earlier GATO Tiago
  large-easy results; PCG remains cap-heavy, while QDLDL remains clean on this
  GATO experiment.
- Found another concrete bug in the `/workspace/MPCGPU` Tiago validation
  experiment's rolling-window implementation. In `include/mpcsim.cuh`, after
  shifting `d_xu`, the code filled the horizon tail from
  `(state_size+control_size)*traj_offset - control_size`, which pulls controls
  and states from the beginning of the global precomputed trajectory. The tail
  should match the shifted reference tail and use
  `(state_size+control_size)*(traj_offset+knot_points-1) - control_size`.
  Patched the MPCGPU branch accordingly and reran the Tiago easy examples. The
  fix improved but did not solve the MPCGPU validation: PCG finite throughout
  with mean recorded L1 error `1.0373` vs `1.1004` before, mean PCG iterations
  `593.24` vs `659.84`, cap-hit fraction `0.557` vs `0.599`; QDLDL now stays
  finite to row `2363/7501` and recorded error `157/500`, compared with
  `1650/7501` and `109/500` before, but still eventually diverges. The
  remaining MPCGPU failure therefore has at least one more cause beyond the
  tail-fill bug.
- This directory was created as a full sibling copy of `/workspace/GATO` for an
  isolated QDLDL-to-PCG warmstart experiment. Added
  `QDLDL_WARMSTART_EXPERIMENT.md` with the current backend state, relevant
  entry points, baseline large-easy Tiago metrics, build commands, and a
  proposed two-pass capture/replay design. The intended first test is to run a
  QDLDL capture pass for Tiago N=32, then rebuild PCG and replay the exact same
  subproblems using each QDLDL `XU` solution as the PCG warm start.
- Implemented `examples/qdldl_pcg_warmstart_experiment.py` for that two-pass
  test. The script imports the existing `large_easy_tracking.py` Tiago
  trajectory and solver settings, captures per-tick QDLDL subproblems and
  solutions to `test-artifacts/qdldl_warmstart/tiago_right/qdldl_capture.npz`,
  then replays the captured subproblems with PCG using the QDLDL trajectory as
  `XU_B`. It records per-tick CSVs plus JSON summaries for PCG cap-hit rate,
  KKT-converged fraction, accepted line-search fraction, final merit, solve
  time, non-finite outputs, and PCG-vs-QDLDL solution norms.
- Ran the exact Tiago N=32 QDLDL-to-PCG warmstart experiment. The QDLDL capture
  produced `500` subproblems with no rejected solves and no non-finite outputs.
  After rebuilding `bsqpN32_tiago_right` as PCG, replaying those same
  subproblems with `sigma=0` and `XU_B=qdldl_solution` showed that PCG still
  does not converge reliably: `10000` linear solves, mean PCG iterations
  `879.6204`, p95 `1000`, linear-solve cap-hit fraction `0.867`, and
  subproblem cap-hit fraction `0.994`. Only `3/500` subproblems avoided the
  cap entirely; KKT-converged fraction was `0.0`, accepted-step fraction was
  `0.186`, and outputs stayed finite. This points to PCG failing on the Tiago
  Schur systems even very near the QDLDL trajectory, not merely poor closed-loop
  initialization.
- Added `TIAGO_PCG_DEBUG_PLAN.md` to capture the proposed linear-system
  diagnosis path and implemented `examples/tiago_single_to_smoke.py` as a
  single-solve Tiago trajectory-optimization reduction. The default `hold` case
  uses the comfortable pose, current tool position at every knot, and a
  gravity-compensated static warm start. PCG already hits the cap on `15/20`
  linear solves in that static case, while QDLDL records placeholder iteration
  count `1` throughout. The `small-x` case shifts the target by only `1 cm`;
  QDLDL accepts `7/20` steps and reduces the predicted final tool error from
  `0.0100 m` to `0.00587 m`, while PCG accepts `0/20` steps and leaves the
  predicted final tool error at `0.0100 m` with mean PCG iterations `804.35`
  and cap-hit fraction `0.8`. This reduces the Tiago problem to a one-shot,
  non-MPC Schur-system failure.
- Continued debugging the one-shot `small-x` fixture by adding a debug Schur
  dump path in `gato/bsqp/bsqp.cuh`, scalar PCG trace output in
  `gato/bsqp/kernels/pcg.cuh`, and `tools/analyze_schur_dump.py`. The first
  cap-hit dump is SQP iteration `2`: the Schur matrix is finite but has worst
  entries around `6.38e10`, all concentrated on local `(13,13)` of the main
  diagonal, the last velocity-state coordinate. The symmetric part is negative
  definite with absolute condition estimate about `1.04e13`; direct CPU solve
  succeeds, while GPU PCG's residual after `1000` iterations is enormous
  (`~3278` L2). Emulating CUDA warp-reduction order in Python matches the GPU
  PCG scalar trace, so the behavior is consistent with float32 arithmetic
  sensitivity on an extremely ill-conditioned Tiago Schur system rather than a
  bad dump layout.
- Probed the one-shot `small-x` PCG fixture by increasing only `u_cost` from
  the default `1e-6`. At `u_cost=1e-5`, cap hits disappear but no line-search
  step is accepted; at `u_cost=1e-4`, PCG has no cap hits, mean iterations
  `9.0`, one accepted step, and final predicted error `0.00946 m`; at
  `u_cost=1e-3`, PCG has no cap hits, mean iterations `7.35`, twelve accepted
  steps, and final predicted error `0.00906 m`. The `u_cost=1e-4` Schur dump
  reduces the worst Schur entry to `6.38e8`, symmetry L-inf to `2`, and
  condition estimate to `1.89e11`, with GPU PCG exiting in `19` iterations.
  Current working hypothesis: Tiago plus `u_cost=1e-6` creates a Schur scale
  that is numerically hostile to single-precision PCG; QDLDL can still solve it
  directly.
- Retested the full Tiago `large_easy_tracking` example with PCG and higher
  control regularization. Added solver-parameter CLI overrides and `--no-media`
  to `examples/large_easy_tracking.py`. With `u_cost=1e-5`, the full N=32 Tiago
  run has zero rejected solves, zero PCG cap hits, mean tracking error
  `0.011321 m`, p95 `0.025343 m`, max `0.028110 m`, final `0.000793 m`, and
  mean PCG iterations `13.38`. With `u_cost=1e-4`, the run also has zero
  rejected solves/cap hits, mean error `0.011269 m`, and mean PCG iterations
  `18.14`. With `u_cost=1e-3`, cap hits remain gone but tracking degrades
  slightly: mean error `0.012879 m`, max `0.048407 m`, mean PCG iterations
  `24.70`. The smallest tested successful value is therefore `u_cost=1e-5`,
  which fixes the large-example PCG cap issue while keeping QDLDL-level
  tracking on this trajectory.
- Ran a one-at-a-time ablation sweep against the working Tiago large-easy PCG
  setup to identify which differences from default `MPC_GATO.run_mpc_fig8`
  still matter. Recorded the table in `TIAGO_PCG_ABLATIONS.md`. Current
  classification: `u_cost=1e-5`, dual reset every tick, repeated-current warm
  starts, more than one SQP iteration, and the high Tiago running tracking
  weight are still important. `max_pcg_iters` can drop to `120` once `u_cost`
  is fixed; `kkt_tol=1e-3`, `rho=0.01`, and `qd_cost=1e-2` are fine; `N_cost=50`
  is mostly fine. Carrying previous primal and dual together is very bad
  (`0.998` cap fraction and `494` rejected solves), and default-like carried
  primal/dual with `max_sqp_iters=1` diverges.
- Ran the existing `examples/gato_fig8_tracking.ipynb` Indy7 notebook and saved
  the executed copy plus artifact-run outputs under
  `test-artifacts/gato_fig8_tracking/indy7/`. Added
  `examples/gato_fig8_tracking_artifacts.py` to save comparable JSON, CSV, and
  PNG outputs for this notebook workflow. Added a Tiago version,
  `examples/gato_fig8_tracking_tiago.ipynb`, and kept its differences to the
  required plant-specific pieces: Tiago URDF/start pose, zero external wrench,
  a small figure-8 centered at the current Tiago tool pose, tuned PCG
  parameters, dual reset every tick, and repeated-current warm starts. The
  Tiago artifact run completed with batch size `1`, mean error `0.000602 m`,
  max error `0.001304 m`, and mean solve time `1.569 ms`.
- Added a Tiago offset-start variant to the artifact runner and rendered
  `test-artifacts/gato_fig8_tracking/tiago_right/offset_start_dx_m0p12_dy_0p08_dz_0p06/`.
  The figure-8 was shifted by `[-0.12, 0.08, 0.06] m` from the current Tiago
  tool pose. PCG still completed cleanly with mean error `0.000662 m`, max
  transient error `0.137315 m`, final error `0.000342 m`, and mean solve time
  `1.164 ms`.
- Re-ran the Tiago offset-start artifact with the exact Indy7 start-to-first
  reference offset vector, `[-0.35355339, 0.54005339, -0.4675] m` (norm about
  `0.797 m`). The run did not numerically fail, but it did not track: mean
  error `0.537949 m`, p95 `0.561565 m`, max `0.760099 m`, final
  `0.531246 m`, and mean SQP iterations nearly saturated at `19.98/20`.
  Artifacts are under
  `test-artifacts/gato_fig8_tracking/tiago_right/offset_start_same_as_indy7/`.
- Added an artifact mode that rigidly transforms the exact Indy7 figure-8 point
  cloud into Tiago's torso-relative workspace while preserving the initial
  start-to-first-reference offset length (`0.797003 m`) and the curve arc
  length (`12.194309 m`). A reachable-looking first reference at
  `[0.557576, -0.672736, -0.276456] m` gives Tiago start delta
  `[0.240823, -0.428772, 0.627194] m`. The transformed full-size task tracks
  with mean error `0.005288 m`, p95 `0.007175 m`, final `0.003827 m`, and a
  max transient catch-up error `0.760642 m`. Artifacts are under
  `test-artifacts/gato_fig8_tracking/tiago_right/transformed_indy7_shape_reachable/`.
- Added joint-position limit reporting and joint CSV output to the figure-8
  artifact runner. Re-ran the transformed Indy7-shape Tiago task and confirmed
  no position joint limits are violated (`max_joint_position_violation_rad=0`).
  The closest margins are joint 3 upper margin about `0.192 rad` and joint 5
  upper margin about `0.237 rad`.
- Added applied-control logging to `MPC_GATO.run_mpc_fig8` and control CSV plus
  torque/velocity/finite-difference acceleration summaries to the figure-8
  artifact runner. The transformed Indy7-shape Tiago task is dynamically
  infeasible with the current unconstrained settings: max velocity violation is
  `53.27 rad/s` over the `2.5 rad/s` URDF limit, max applied torque violation
  is `359.48 Nm`, peak applied torques are `[207.9, 402.5, 119.5, 236.4, 7.5,
  19.5, 0.8] Nm` versus limits `[43, 43, 26, 26, 26, 26, 26] Nm`, and
  finite-difference accelerations reach thousands of `rad/s^2`.
- Created a more physically meaningful transformed-Indy7 Tiago tracking test:
  fixed control ticks (`fixed_dt`), actuator saturation in the simulator,
  planned/applied torque logging, and nonzero soft velocity/torque limit costs.
  The chosen review artifact is
  `test-artifacts/gato_fig8_tracking/tiago_right/physical_sensible_q40_limits1_saturated_fixeddt/`
  with `q_cost=40`, `N_cost=50`, `u_cost=1e-5`, `vel_lim_cost=1`, and
  `ctrl_lim_cost=1`. It preserves the full Indy7 figure-8 shape and initial
  offset distance, starts with a hard `0.792 m` catch-up error, then lags rather
  than violating limits: mean error `0.0651 m`, p95 `0.1064 m`, final
  `0.0743 m`, no position/velocity/torque violations, max joint speed
  `1.93 rad/s`, and max applied torque `35.35 Nm`.
- Copied `examples/gato_fig8_tracking_artifacts.py` to
  `examples/gato_fig8_tracking_artifacts_anim.py` and added Tiago arm-link
  reconstruction plus a 3D GIF renderer. Ran the physical `q_cost=40` case
  through the animation script and wrote
  `test-artifacts/gato_fig8_tracking/tiago_right/physical_sensible_q40_limits1_saturated_fixeddt_anim/tracking_arm_3d.gif`
  with `420` frames at `850x720`. The rerun preserved the physical-limit
  result: mean error about `0.0651 m` and no position, velocity, or torque
  violations.
- Extended `examples/large_easy_tracking.py` with velocity/torque limit-cost
  CLI overrides, actuator saturation, and joint/control artifact summaries.
  Re-ran Tiago large-easy with the physical figure-8 parameter family:
  `q_cost=40`, `N_cost=50`, `qd_cost=1e-2`, `u_cost=1e-5`,
  `vel_lim_cost=1`, `ctrl_lim_cost=1`, `max_pcg_iters=120`, `kkt_tol=1e-3`,
  `rho=0.01`, dual reset, repeated-current warm start, and saturated controls.
  Final rendered artifact:
  `test-artifacts/large_easy_tracking/tiago_right/pcg_physical_q40_limits1_saturated/`.
  It tracks with mean error `0.01913 m`, p95 `0.04005 m`, max `0.04489 m`,
  zero rejected solves, PCG cap fraction `0.0163`, and no position, velocity,
  planned-torque, or applied-torque violations. Max joint speed was
  `2.329 rad/s`; max applied torque was `18.39 Nm`.
- Rendered an Indy7 figure-8 baseline with batch size `1` and zero external
  wrench via a temporary in-process override of the artifact runner, leaving
  `examples/gato_fig8_tracking.ipynb` unchanged. Artifact directory:
  `test-artifacts/gato_fig8_tracking/indy7/batch1_no_external_force/`.
  The run produced `tracking_xz.png` and `tracking_3d.png`; mean tracking
  error was `0.04575 m`, p95 `0.08038 m`, max `0.79355 m`, final
  `0.07005 m`, and mean solve time `0.219 ms`.
- Changed `MPC_GATO.run_mpc_fig8` so external-force hypothesis selection is
  evaluated before trajectory optimization. For `batch_size > 1`, the batched
  solver is now used for one-step force-hypothesis simulation, while a
  separate batch-1 solver optimizes the trajectory for the selected wrench.
  Rendered Indy7 figure-8 artifacts at
  `test-artifacts/gato_fig8_tracking/indy7/selected_force_single_solve/`.
  Results: batch `1` mean error `0.07938 m`, mean solve `0.217 ms`; batch
  `32` mean error `0.05658 m`, mean solve `0.484 ms`; batch `128` mean error
  `0.05027 m`, mean solve `0.503 ms`.
