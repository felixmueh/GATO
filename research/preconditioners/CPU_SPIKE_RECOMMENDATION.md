# Next CPU preconditioning spike for public GATO

Research recommendation, 2026-10-20. **Proposed; not implemented or measured.**
Public upstream `aeacf16e7733df1a17dfd6627f3b5d15c2ba4387` plus Schur
contribution `9b0df42b5ba27a63fc1b7be05568bf59ac61533e`; Indy7, B1,
12-dimensional state blocks. Private TIAGo results do not establish performance
for this target.

**Test CPU block LDLT construction with a fixed, symmetric, approximate
triangular application on the GPU. Begin with a small replay of identical
captured Schur systems, before integrating another closed-loop candidate.**
Use the existing factors and PCG callback, replacing the two serial triangular
passes by a few parallel Jacobi sweeps. Screen only sweep counts 2, 4 and 8.
This is the most useful next research spike because it directly tests whether
the strong existing factor can become an economical *approximate* inverse,
without adding another generic sparse factorization package or changing PCG.

The recommendation is an engineering inference, not a demonstrated ranking.
There is a credible failure mode: short sweeps may discard precisely the
long-range coupling that made the exact factor effective. A negative replay
result would answer the question cheaply.

## What the existing evidence actually establishes

The [completed public report](/workspace/GATO/thesis_results/upstream-preconditioners-20261020/REPORT.md)
and [source ledger](/workspace/GATO/thesis_results/upstream-preconditioners-20261020/public-branches-methods.json)
are the measured evidence. Full-episode transfer-subtracted means are
422.80 ms for native and 338.41 ms for CPU LDLT/GPU application, but the
respective trajectories require 161 and 126 controller solves. Rough division
gives 2.626 and 2.686 ms per controller call, including other episode work.
Therefore the 20% episode improvement does **not** establish a 20% faster
linear solve. Higher speeds and the different trajectory remain part of that
result.

The closest available breakdown is also on different systems: native GPU PCG
averages 110.18/161 = 0.684 ms per call; LDLT CPU setup plus GPU PCG averages
(49.15 + 31.20)/126 = 0.638 ms. These are motivation for a fair replay, not a
matched speed comparison. CPU setup alone is approximately 0.390 ms/call.

Source review gives three useful facts:

* [Block LDLT construction](/workspace/GATO-upstream-precond-block-ldlt/research/preconditioners/block_ldlt_factors.hpp)
  already exploits the block chain, with no global dense matrix, symbolic
  ordering, pivot floor or adaptive shift. Its main cost is O(N d³).
* [Its GPU application](/workspace/GATO-upstream-precond-block-ldlt/research/preconditioners/block_ldlt_apply.cuh)
  serializes the forward and backward passes across N knots, with a block
  barrier per knot and only d independent rows within each current block.
  This is a concrete parallelism limitation, especially as N grows.
* [Current CPU IC](/workspace/GATO-upstream-precond-cpu/research/preconditioners/cpu_strip_ic.hpp)
  reconstructs a sparse matrix and calls Eigen `compute()` each setup.
  Its 411.32 ms episode setup overwhelms its 17.90 ms application and
  6.59 ms GPU matvec times. The host native-P control is already much slower
  than fused GPU PCG. Those results mix mathematics with backend overhead.

The native formation code has damped diagonal inverses and adjacent inverse
blocks. Its stale “block Jacobi” comment is not a suitable baseline
description, and the existing `symmetric_stair` CPU alias is not a new method.

## The one proposed method

Use the existing positive convention H = -S and CPU factorization

    H = L D Lᵀ,       L = I + E,

where E is strictly block lower bidiagonal and D is positive block diagonal.
For fixed integer m define

    T_m = I - E + E² - ... + (-E)^(m-1),
    B_m = T_mᵀ D⁻¹ T_m.

Apply `-B_m` in public GATO's negative-S PCG convention. Compute the factors
fresh for each system in this first spike. Do not materialize powers of E,
construct a wider sparse inverse, or add factor reuse yet.

Application consists of m forward Jacobi iterates starting from zero,
`y_next = r - E y`, one block-diagonal multiplication `v = D⁻¹ y`, and m
transpose iterates starting from zero, `z_next = v - Eᵀ z`. All knots in one
iterate are independent. Identity-only initialization can eliminate the
first trivial iterate, but the meaning of m must remain exactly the number
of polynomial terms above.

This gives a short mathematical correctness argument: T_m is unit lower
triangular and invertible; consequently B_m is symmetric positive definite
whenever D is positive definite. Matched forward/transpose operators, the
same fixed m and deterministic zero initialization are essential. Adaptive
inner stopping or reusing an application vector changes this operator and
is outside the spike. Roundoff still requires an application oracle.

For m < N this is an approximate inverse, even though its CPU factorization
is exact. At m = N, nilpotence of E recovers the full inverse in exact
arithmetic. This limiting case is a useful small CPU oracle, not a proposed
runtime setting. A block IC(0) label alone would not create this
approximation: eliminating a block-tridiagonal chain with dense retained
blocks introduces no additional block fill.

The practical opportunity is exchanging approximately 2N serial dependency
stages for O(m) parallel stages. Arithmetic increases to O(m N d²) per
application. There is no guarantee that fewer stages compensate for more
outer iterations or memory traffic. The construction floor remains
O(N d³), and an O(m) stage count is not an O(m) wall-time claim.

[Chow, Anzt, Scott and Dongarra (2018)](https://doi.org/10.1016/j.jpdc.2018.04.017)
study approximate Jacobi triangular solves for incomplete-factor PCG and
show that their utility is problem-dependent; blocking can improve
robustness. Their [author manuscript](https://icl.utk.edu/files/publications/2018/icl-utk-1067-2018.pdf)
motivates this application strategy. Our choice to use GATO's existing exact
block LDLT factors and its natural state blocks is a separate adaptation.
That paper supplies no performance guarantee for this short robot chain.

## Smallest experiment that can decide

**1. Capture a bounded set of real public systems.** The inspected retained
episode NPZ contains controller inputs, warm seeds and returned plans, but
not the assembled S, RHS and incoming multiplier needed for exact linear
replay. Do not reconstruct them approximately from saved states. Add a
research-only capture point immediately before PCG, outside timed replay.

At N128 capture four predeclared calls per goal—first, second, middle and
last—from one native and one full-LDLT episode: at most 24 systems. Include
goal/reset transitions and any zero-iteration or capped solve among those
selected calls; retain labels and do not replace inconvenient inputs.
Save raw S, actual native P, RHS, incoming lambda, epsilon/cap/skip state,
rho, call index and source/config hashes. Capture any factors needed for
optional primal-step comparison. Sample selection uses call positions,
not favorable iteration counts.

The existing raw episodes remain unchanged. Capturing new systems needs a
coordinated GPU window; none was requested or executed for this memo.

**2. Run a CPU mathematical screen, then one small GPU callback.** Verify
the fixed-sweep operator against its independently formed expression on
small blocks, symmetry/positive quadratic forms, signs, boundaries and
the m = N limit. Replay the actual captured systems with the same incoming
lambda and public stopping formula to estimate iteration inflation.
CPU timings of this reference do not predict fused GPU performance.
If all three m values are numerically poor or even optimistic cost bounds
cannot beat the controls, stop before GPU implementation.

For viable m values, implement only a fused application callback around
the existing factors. Keep the raw-S product, recurrence, absolute rho
tolerance, relative tolerance, cap and zero-iteration shortcut unchanged.
The performance controls are native fused PCG and existing full-LDLT PCG
on the **same captured arrays**. Do not use host-PCG timings as a substitute
for these controls.

The straightforward Jacobi implementation may need an additional vector
for safe double buffering and the fixed transpose RHS. Audit actual scratch
liveness and compiled resource usage. Do not assume the current callback's
two temporary vectors suffice or permit in-place races to save storage.
N512 needs its own resource/finite-application check; a valid full-LDLT
launch does not automatically validate a callback with more scratch.

**3. Report cost and quality separately.** For each captured system, report

    setup + factor packing + PCG application/matvec/reductions
          + launch/synchronization and other solve-wrapper wall.

Measure CPU setup, GPU PCG event time, application time in a separate
diagnostic pass, and actual host solve-stage wall. Also report the requested
wall minus all disjoint measured H2D/D2H copy intervals. Fence preceding
compute before timing a transfer; never subtract CPU factorization or a
compute wait as copy time. Keep GPU event and host-wall boundaries distinct.
Retain the common native preconditioner formation cost or clearly report
its shared upstream boundary; do not silently remove it for one candidate.

The break-even test for each m is

    setup_m + K_m * (application_m + other_PCG_iteration_work)
            + fixed_wrapper_cost < corresponding_control_cost,

with the initial application and final iteration accounting taken from
actual counters, rather than assuming exactly K applications. Use measured
total cost for the decision. If CPU construction already exceeds native's
entire matched solve-stage cost, this fresh-factor candidate has no path
to a win on that cohort, even with a free application.

The production stopping quantity rᵀPr changes with P; equal tolerances and
identical input arrays still do not establish equal solution accuracy.
Retain the public stopping policy, and additionally compute a common
relative Euclidean linear residual `||b-S lambda||₂ / ||b||₂` after replay
(report the absolute residual when b is zero/negligible). A loose reference
band such as 1e-2 is a descriptive comparison, not a new KKT or task gate.
Compare solution/primal-step differences with a double-precision structured
solve using sensible percent-scale, physically scaled summaries. Report
native's errors too. If a timing advantage accompanies materially poorer
linear answers, label the tradeoff and do not call it a comparable-quality
speedup. No tight residual pursuit, exact agreement or KKT investigation.

Use four independently started replay repeats, with a predeclared warmup
and alternating method order. Repeated calls inside one process may
stabilize a short timing interval but are not independent uncertainty
samples. Preserve every run, report paired aggregate differences and
repeat spread, and keep per-system tails visible. Use a quiet CPU/GPU
window and record clocks during measured runs without changing them.

**4. Decide, then scale.** Promote one m only if the matched solve-stage
cost improves by roughly 15% over the better of native and full LDLT,
comfortably exceeds repeat uncertainty, and shows no material linear
quality loss across the selected systems. This is an engineering threshold
for investing further, not a statistical theorem. A smaller or noisy gain
does not justify a tuning campaign. If none wins, retain the negative
result and stop at m = 8; do not add sweep counts, coarse levels or shifts
to chase success.

Then screen that one m at N256 and N512 on Jetson using the same capture
rule and the separately prepared public as-is controls. Desktop comparisons
use only horizon sizes supported by their compiled resource limits.
Increasing N at fixed dt increases the physical prediction horizon, so
interpret each N within its own matched cohort. Keep Jetson MODE_40W as
user-confirmed; no host power/clock checks or changes are required.

Only after a positive replay result run complete public example12 reaches,
with the existing protocol and four independent process repeats. Report
actual and transfer-subtracted episode wall, calls, simulated duration,
goals, speeds and limits separately. Do not use changed call count as proof
of linear-solver acceleration. Managed-memory behavior is a separate
measurement; shared physical RAM does not remove explicit copies.

## Why other next steps have lower priority

| Option | Decision and reason |
| --- | --- |
| Lag full LDLT factors across MPC calls | Defer. This is a legitimate approximate preconditioner, fixed within each solve. But halving the current setup frequency saves only about 0.195 ms/call before iteration growth, against an existing full-chain kernel of about 0.248 ms/call. Only a small number of extra full-chain applications can consume that saving. Matrix changes, rho changes and horizon alignment also need evidence. |
| Reuse IC symbolic analysis and buffers | Worth future engineering, but the current IC is a generic sparse setup on a dense-block chain. It also retains triangular application dependencies. The existing structured LDLT already provides a much cheaper, stronger factor baseline. Do not infer that the IC mathematical family is intrinsically slow. |
| Larger Schwarz windows, grouped stair or segmented LDLT | Plausible second choice. They shorten dependencies but add a window/overlap design choice and discard or approximate boundary coupling. Four-knot Schwarz already had modest iteration reduction. The selected spike holds CPU factors fixed and isolates application cost more cleanly. |
| More polynomial steps with native P | Defer: measured order 2/3 GPU variants already lost because extra products outweighed fewer iterations. The selected polynomial acts on triangular factors, has explicit symmetric factor composition, and targets an observed serial bottleneck. It is not a claim that polynomial methods generally win. |
| AMG, adaptive coarse spaces or low-rank corrections | Defer. The four current AMG failures are retained evidence, but do not identify a missing near-nullspace or a profitable correction rank. New multilevel tuning introduces more unsupported choices than this decision needs. |
| Direct full block solve, cyclic reduction or parallel scan | Strong separate exact-solver direction, especially to remove redundant PCG around an exact inverse. It changes the solver path rather than testing approximate preconditioning. Preserve it as a later comparison; this memo does not implement it. |

The historical private-TIAGo QDLDL comparison included explicit copies and
generic packing/analysis costs, and used a different robot and protocol.
It neither rejects a carefully implemented public direct solver nor supplies
the missing matched-system measurement here.

## Source and status record

Reviewed public implementations: base `f7d456c`, full LDLT `16dd835`,
CPU candidates `15198f1`, and callback recurrence pinned to GLASS
`9e57178ca146fa1cecae7e27c22260603b993aa6`. The receiving verification
checkout was `c7bf5ed` when this memo was prepared.

Primary sources for the decision:

* **CHOW-TRIANGULAR-JACOBI-2018:** Chow, Anzt, Scott and Dongarra,
  [Using Jacobi iterations and blocking for solving sparse triangular systems in incomplete factorization preconditioning](https://doi.org/10.1016/j.jpdc.2018.04.017).
  Direct inspiration for approximate parallel application; transfer to GATO
  remains untested.
* **SYMSTAIR-2024:** Bu and Plancher,
  [Symmetric Stair Preconditioning of Linear Systems for Parallel Trajectory Optimization](https://arxiv.org/abs/2309.06427).
  Background for the existing parallel neighboring-block baseline and the
  importance of symmetry; not evidence for the proposed sweep count.
* **BLOCK-TRIDIAGONAL-2013:** Aravkin, Bell, Burke and Pillonetto,
  [Kalman smoothing and block tridiagonal systems: new connections and numerical stability results](https://arxiv.org/abs/1303.5237).
  Structural context for exact forward/backward block solves; no imported
  Kalman-specific stability guarantee for the GATO implementation.
* **PATH-PCG-2020:** Zafar, Cantoni and Farokhi,
  [Structured preconditioning of conjugate gradients for path-graph network optimal control problems](https://arxiv.org/abs/2010.05616).
  Background on fixed structured inner iterations; its network model and
  conditioning bounds are not claimed for this candidate.

No implementation, CUDA build, GPU benchmark, new numerical result, solver
repair, merge or push was performed for this recommendation. CPU-only
synthetic implementation was not delegated: without captured public Schur
arrays it would not resolve the decisive iteration-versus-cost uncertainty.
