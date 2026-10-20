# Public GATO block LDLT preconditioner candidate

Based on reviewed public timing harness `d66dba5` and unchanged public core
`9b0df42` (no TIAGo fork source). Dependency: grid-rbd 0.5.1, pinned GLASS
`9e57178ca146fa1cecae7e27c22260603b993aa6`. Its actual installed header root is
`/workspace/GATO-upstream-contributions/.venv/lib/python3.10/site-packages/grid_codegen/_data/GLASS`.

The CPU factors H=-S=L D L^T directly from public [N,D,3D] row-major strips.
Storage and conversion are O(N D²), factorization O(N D³); there is no global
dense adapter. Positive D is checked with Eigen float LDLT, without shifts,
floors, thresholded sparsity, rescue, or modifications to device S. Symmetric
diagonal blocks use Eigen's lower-triangle convention; off-diagonal symmetry
is checked at a float roundoff tolerance. Mathematical nonfinite input fails.

Uploaded row-major L and diagonal inverses are applied on the GPU with serial
dependent block substitution and parallel within-block rows. Native stores S
negative, so application returns -H^-1 r. Original public GLASS recurrence,
dot reductions, initial-guess handling, stopping and iteration accounting are
retained in the shared provenance-pinned callback extraction. It imports
glass::block helpers, matching the original function's namespace lookup. The
normal/dummy paths still call the original native GLASS solver directly.

Only method `block_ldlt`, float B1, Indy7 N128 is enabled by this branch's CMake
target definitions. CMake finds Eigen headers and defines EIGEN_NO_CUDA and
EIGEN_DONT_PARALLELIZE. All candidate/native solver arithmetic build flags stay
the public ones. Prepare allocates CUDA/pinned buffers before measurement.

Timing records distinguish CPU setup wall, PCG CUDA-event kernel total, PCG
compute wall and additional transfers. CPU application is zero because this
backend applies on the GPU. The hook is included after copy macros are unset:
its D2H S/skip and H2D factor copy walls contribute only to additional transfers,
disjoint from instrumented native copies. CPU validation/factorization, GPU
application and ordinary host overhead remain in whole-reach transfer-excluded
wall. Skip flags retain native behavior and bypass factor setup.

Root owns all builds, GPU tests and whole-reach runs. Source preparation ran no
builds or GPU calls. Relevant commands after obtaining the shared quiet window:

```bash
g++ -std=c++17 -O3 -DNDEBUG -I/usr/include/eigen3 research/preconditioners/test_block_ldlt_factors.cpp -o build/test_block_ldlt_factors
build/test_block_ldlt_factors
nvcc -std=c++17 -O3 -arch=sm_61 -DEIGEN_NO_CUDA -DEIGEN_DONT_PARALLELIZE -I/usr/include/eigen3 research/preconditioners/tests/block_ldlt_apply.cu -o build/test_block_ldlt_apply
build/test_block_ldlt_apply
```

The CPU test reconstructs independent SPD matrices from factors. The GPU test
checks the actual N128/D12 callback against an independently constructed known
solution, including sign and padded zeros. Neither test establishes whole-reach
performance or task qualification; root must separately run the existing public
whole-reaching worker with `--method block_ldlt`, retaining every outcome.
