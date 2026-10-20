# Jetson scaling preparation checks

2026-10-20. Public Indy7 only. These are bounded desktop functional checks,
not Jetson measurements, a new performance ranking, or physical qualification.
The prepared protocol is [JETSON_SCALING.md](JETSON_SCALING.md); the separate
research decision is [CPU_SPIKE_RECOMMENDATION.md](CPU_SPIKE_RECOMMENDATION.md).

## ARM Eigen/NVCC build correction

User's target compilation reports undefined `__builtin_aarch64_*` intrinsics
from `arm_neon.h`. The added CPU Eigen headers are compiled inside bindings.cu;
`EIGEN_NO_CUDA` enables host-only Eigen but leaves host explicit SIMD enabled.
ARM experiment targets now add `EIGEN_DONT_VECTORIZE`, following Eigen's CUDA
guidance. Stock upstream aeacf16 bindings/CMake have no added Eigen integration.
This is an experiment portability correction, not an upstream solver repair.

Retained `_expr_results/arm-eigen-build-fix/` checks: existing CPU IC/SGS/Schwarz/
Jacobi mathematical tests and LDLT reconstruction tests pass with the new flag.
The N128/D12 CUDA application test compiles and passes on desktop sm61 (maximum
absolute error2.08616e-7). CMake configuration with simulated ARM processor
selects the definition for all three N128/256/512 targets; the existing x86
configuration has no such definition. The simulated processor check does not
cross-compile or validate ARM. Actual target rebuild remains pending.

CPU timings on ARM will reflect disabled Eigen explicit SIMD; host compiler
auto-vectorization is unchanged. Fully NEON-optimized CPU timings would need
Eigen code moved into separate host C++ translation units. No timing ranking or
whole-episode performance claim is made by these functional build checks.

## SQP-iteration reporting update

C++ timing hooks at0ee8922 and reporting ata38f25c were checked in a fresh
N128 desktop cohort under `_expr_results/sqp-iteration-timing/`. Release sm61
build passes in the environment below. Native, dummy and block-LDLT each finish
reach-seed2 and tracking-seed11; CPU native control and incomplete Cholesky each
finish reach-seed2. All eight episodes have finite saved plans and no reporting
or execution error. Across1,764 attempted SQP iterations, native wall matches
returned solve wall exactly; in-interval copies are positive, compute is
positive and equals wall minus copies, and B1/SQP1 counts agree. Three CPU
reporting tests cover mismatched mappings, unmeasured copies and zero-PCG
shortcut retention. CSV collection passes on the native/dummy/LDLT runs.

Reproduction after `build --knots 128 --arch 61` and `freeze --knots 128`:

```bash
python research/preconditioners/jetson_scaling.py run \
  --tasks _expr_results/sqp-iteration-timing/frozen \
  --output _expr_results/sqp-iteration-timing/native-check \
  --knots 128 --methods normal,dummy,block_ldlt \
  --cases reach-seed2,tracking-seed11 --preflight
python research/preconditioners/jetson_scaling.py run \
  --tasks _expr_results/sqp-iteration-timing/frozen \
  --output _expr_results/sqp-iteration-timing/cpu-check \
  --knots 128 --methods cpu_native_control,incomplete_cholesky \
  --cases reach-seed2 --preflight
python -m unittest discover -s research/preconditioners/tests \
  -p test_sqp_iteration_reporting.py -v
```

Module SHA256 is
`1a0a2426e76dfd392a0d147d7875705ed7c379c94f4ff480448c484808f84fcd`.
Build began at0ee8922 while Python reporting was being implemented; C++ files
remained unchanged through a38f25c. Clocks were sampled during runs. These are
one-process functional checks, not a timing ranking. Larger-N binaries and
Jetson execution of the new instrumentation remain pending; rebuild and use a
fresh frozen-input directory on target.

## Completed

- Python syntax/help and dry-run schedules pass: short 198 episodes, full 891,
  native/dummy N128/N512 calibration four episodes.
- Existing CPU strip checks pass for incomplete Cholesky, SGS, Schwarz and
  Jacobi. AMG was not requested or checked.
- Release CUDA builds compile/link all three modules at N128/N256/N512 for
  sm61. Desktop environment: GTX1070, CUDA12.9.86, GCC11.4, CMake3.31.10,
  public grid-rbd0.5.1, GLASS9e57178, Eigen3 and pybind11 3.0.4.
- Safe resource queries launch no kernels. N128 needs 31,328 dynamic plus
  32 compiled static shared-memory bytes and passes. N256/N512 are rejected
  before launch on this GTX1070's 49,152-byte limit, as expected.
- Fourteen N128 episodes complete without execution errors or nonfinite saved
  plans/merits: all eleven methods reach all three goals for reach-seed2;
  native and full-LDLT complete tracking-seed11; the dummy reaches all goals.
  Native/dummy each record 158 calls and 4,031 PCG iterations.
- Both tracking runs complete 3.48 seconds against a requested
  3.470920240288425 seconds. Position RMS is approximately 43.7 mm native and
  43.2 mm LDLT. One process each supplies a functional check, not uncertainty
  or comparative performance evidence. Existing telemetry-only limits remain
  in use; arrival does not certify physical-limit compliance.
- For every episode, measured copy wall is positive, compute equals episode
  wall minus that copy wall, saved call counts match native records, and
  during-run clock samples are retained. CPU setup/application are charged.

Root owns all CUDA builds/queries/runs under the common-repository GPU and
timing locks. No clock or power settings were changed. MODE_40W in the portable
launcher's metadata is the user-confirmed **Jetson target** configuration;
it does not describe the desktop GTX1070 used for these functional checks.

## Commands and provenance

From this checkout, with the active public environment and CMake3.31 on PATH:

```bash
python research/preconditioners/jetson_scaling.py build \
  --output _expr_results/desktop-validation/build-all --arch 61 --jobs 1
python research/preconditioners/jetson_scaling.py freeze \
  --output _expr_results/desktop-validation/frozen
python research/preconditioners/jetson_scaling.py run \
  --tasks _expr_results/desktop-validation/frozen \
  --output _expr_results/desktop-validation/reach-all-methods \
  --knots 128 --cases reach-seed2 --preflight
python research/preconditioners/jetson_scaling.py run \
  --tasks _expr_results/desktop-validation/frozen \
  --output _expr_results/desktop-validation/tracking-controls \
  --knots 128 --cases tracking-seed11 --methods normal,block_ldlt --preflight
python research/preconditioners/jetson_scaling.py run \
  --tasks _expr_results/desktop-validation/frozen \
  --output _expr_results/desktop-validation/dummy-control \
  --knots 128 --cases reach-seed2 --methods dummy --preflight
```

Logs, exact environment-specific commands, binary/source hashes and raw episodes
are retained under `_expr_results/desktop-validation/`; the source-only checks
are under `_expr_results/source-checks/`. `audit.json` records episode accounting
and `resources.json` records the safe resource queries. Build configuration
began at parent 5751e53 with the integration uncommitted; all eighteen C++/CMake
file hashes were saved and verified unchanged after the clean integration
commit 0c7c0ad. Later changes are Python, documentation and research notes.
The per-module hashes are pinned in the frozen manifest. Local binaries are
desktop artifacts and are not supplied as Jetson binaries.

## Pending on target

Jetson ARM64 build, actual N256/N512 resource/finite-task checks, target
normal/dummy calibration and the selected repeated comparison remain pending.
No Jetson execution occurred. The short and full schedules are prepared plans,
not completed results. Astra's approximate triangular-application spike was
tested separately with a negative N128 replay outcome; it is excluded here.
PCG and all existing preconditioner algorithms remain unchanged.
