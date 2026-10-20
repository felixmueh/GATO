# Public CPU preconditioner backend

This isolated variant derives its host PCG formula from the public pinned GLASS
`src/base/pcg/solve.cuh` (SHA256 recorded in `upstream_cpu_bridge.cuh`). It retains
native negative-S signs, the supplied warm seed, update ordering, initial
`abs(rho)<1e-6` shortcut, subsequent `abs(rho_new)<1e-6+epsilon*abs(rho_init)`
stopping, and the caller's cap/tolerance. There is no retained-seed guard or
fallback. Nonfinite CPU inputs/results fail and remain failed runs.

CPU Eigen dots/vector updates replace the fused device reductions; each raw-S
matvec still uses public GLASS `bdmv` on the GPU. Launch/synchronization and host
overhead remain in total cost after separately measured CPU/GPU copies are
subtracted. The mandatory `cpu_native_control` applies the actual downloaded
native P, including all off-diagonals, on this same host backend. Compare it to
native `normal` before attributing differences to a CPU preconditioner. This is
an execution-backend comparison, not an exact-arithmetic parity claim.

CPU method branches are explicit aliases of this reviewed common runtime
implementation. Choose the method in the worker; the branch name alone does
not switch the solver:

| Branch suffix under `felix-precond-timing/` | Worker `--method` |
|---|---|
| `incomplete_cholesky` | `incomplete_cholesky` |
| `block_sgs` | `block_sgs` |
| `schwarz` | `schwarz` |
| `block_jacobi` | `block_jacobi` |
| `amg` | `amg` |
| `polynomial2` | `polynomial2` |
| `polynomial3` | `polynomial3` |
| `cpu_native_control` | `cpu_native_control` |
| `symmetric_stair` | `symmetric_stair` (same actual-P host control) |

`symmetric_stair` is an alias of that actual-native-P control; it does not
reconstruct P using a fixed damping assumption. GPU native formation, including
the complete fused preconditioner setup, remains charged before every candidate.
The CPU polynomial methods likewise use actual raw P and S: start z=P*r and
repeat z+=P*(r-S*z) once or twice. All their host strip products and vector work
are included in CPU application cost; this is the same fixed operator as the
GPU polynomial variants with different host arithmetic and execution costs.

Other methods factor H=-0.5*(S+S^T) directly from block strips in O(N*D^2)
conversion space/time, without allocating an N*D by N*D dense matrix. GPU matvec
continues to use raw S. IC builds the exact-nonzero scalar sparse pattern and
retains Eigen's default ordering/scaling/limited-fill/adaptive-shift behavior;
the configured initial shift is 1e-3 and its actual shift is unknown. SGS uses
(D+L)D^-1(D+L)^T, Jacobi retains complete diagonal blocks, and additive Schwarz
uses four-knot local LLT solves with stride three and a truncated final domain.

Optional AMGCL uses pinned revision `1fbc24e687b404962c31bc4db598f6c73c0f4333`:
scalar smoothed aggregation, one pre/post Jacobi sweep and one V-cycle, checked
LLT coarse solve at size32. Per-level Jacobi damping is 0.9 divided by the
maximum absolute row sum scaled by its positive diagonal (at least one).
No global dense conversion is introduced; only the coarse solve is dense.
The method is unavailable unless its external headers are on the include path.

Build this variant with Eigen include path and host-only/single-thread flags:
`-I/usr/include/eigen3 -DEIGEN_NO_CUDA -DEIGEN_DONT_PARALLELIZE`. For AMG add its
external pinned checkout include path; no dependency or binary is vendored.
Root owns all builds, GPU runs and benchmark comparisons.

CMake exposes target-scoped `PRECOND_CPU_BENCH=ON` for N128 Indy7 and optional
`PRECOND_AMGCL=ON`, with `PRECOND_AMGCL_ROOT=build_deps/amgcl` by default. The
AMG option verifies the exact external checkout revision at configure time;
dependency details are in `cpu_amg_dependency.json`. Other native targets keep
their existing backend. No OpenMP flags are added.

`test_cpu_strips.cpp` supplies small independent SPD strip-conversion, inverse
symmetry/SPD, linearity, SGS/Jacobi formula and IC shift-metadata checks. Whole
public reaching and the native-P host control remain separate required evidence.
The default CPU test requires IC/SGS/Schwarz/Jacobi; run it with `--require-amg`
when qualifying AMG, which fails if the requested dependency is unavailable and
adds a connected Poisson97 hierarchy case. `test_cpu_native_control.cu` compares
the actual public native solve against the host control, including nonzero-P
off-diagonals, converged seed, cap0, skip flag and a passed one-iteration cap.
