# Interior cost bounds: independent review

Reviewed 2026-10-15 against continuation study parent `34d7a4b`. CPU/source review only; root owns GPU measurements. This is an explicitly authorized experiment configuration change, not a production merge or a hard constraint controller.

## Source findings

`gato/dynamics/tiago_right/tiago_right_cost_limits.cuh` supplies bounds used only by soft costs. Defaults of joint margin zero and speed/effort fractions one preserve the previous intervals. All nine calls in `tiago_right_plant.cuh` consistently use these bounds: position, velocity and effort in each of value, gradient and Hessian. The existing C2 continuation remains unchanged and uses 1% of the resulting cost interval.

The initial M0–M3 joint margin is 0.02 rad on both ends; velocity/effort fractions 0.90 give ±2.25 rad/s and ±38.7/23.4 Nm. M4 uses q margin 0.10 rad and fractions 0.80/0.85, giving ±2.0 rad/s and ±36.55/22.1 Nm. Every cost interval remains nonempty. Raw physical limits remain the original q intervals, ±2.5 rad/s and ±43/26 Nm. Initial states remain unchanged; there is no filtering, clipping or candidate veto.

Focused CPU compilation of the actual header with `g++ -std=c++17 -fsyntax-only` passed for both default and selected definitions. Compile-time assertions checked unchanged default bounds and the selected transformed bounds. No compiled artifact or GPU call was required. This does not qualify the native build or task behavior.

## Merit parameter

`gato/bsqp/kernels/merit.cuh:73–92` adds `cost_k + mu * constraint_k`: the constraint term is the summed absolute modeled dynamics defects, plus the initial-state constraint once per horizon. `gato/dynamics/integrator.cuh:212` computes that modeled defect. Mu does not multiply physical-limit penalties and is not adapted automatically. The constructor initializes the per-lane value; `BSQP::set_mu_batch` in `gato/bsqp/bsqp.cuh:82` and the existing binding can replace it directly. The current study can simply use its explicit constructor parameter.

`gato/bsqp/kernels/line_search.cuh:59–85` accepts a strict merit decrease and adapts rho, not mu. Existing retained-rho behavior and zero-PCG stopping remain unchanged. Increasing mu changes accepted-step tradeoffs; it does not establish physical feasibility or a convergence certificate. Native final merits from different mu or weight configurations are different quantities and should not be ranked as a common objective.

## Bounded development sequence

Use the same frozen twelve development tasks, B1, SQP cap 1000, PCG120 and mode A. Keep the 30 ms feedback clock, 3 ms RK4 execution, initialization and stopping fixed. Preserve every configuration and outcome; downstream candidates depend on the preceding evidence.

| Candidate | Recorded configuration and scope |
| --- | --- |
| M0 | q inset 0.02 rad, velocity/effort fractions 0.90; public weights, mu=1; twelve-task pilot completed |
| M1 | Speed-only weight 1.0, effort weight 0.003; other M0 settings; twelve-task pilot completed |
| M2 | q/speed/effort weights 0.1/1/0.03, qd=0.05, mu=1 at M0 margins; canonical failed, pilot held |
| M3 | Same as M2 with mu=10; canonical failed, pilot held |
| M4 | q inset 0.10 rad, velocity/effort fractions 0.80/0.85; q/speed/effort weights 0.01/0.2/0.01, qd=0.05, mu=10; canonical passed, pilot 3 rough / 11 physical / 3 both |
| M5 | Same wide bounds/module, q/speed/effort weights 0.015/0.6/0.02, other M4 settings; canonical passed, pilot 3 rough / 11 physical / 3 both |

M2/M3 each reached the full eight-second canonical observation, finite and within
physical limits, but timed out at about 90.6/90.7 mm position error. Their 267
zero-PCG exits are practical stopping events, not proof of task success; no
dependent twelve-task pilot ran. The original qd-only M2 and q inset 0.04 rad
wide build were unused preparation, preserved separately from measured cohorts.

M4 balances local velocity damping (zero-speed curvature 0.15 versus M0 0.1395)
with earlier boundary deterrence (physical-speed limit gradient 72.46 versus
M0 16.77). Its q weight stays 0.01 because the wider inset already increases
physical-bound gradients and q weight 0.02 would nearly reproduce M2's large
canonical J4 centering gradient. The 0.10 rad inset is motivated by 0.075 rad
travel per fixed 30 ms interval and approximate braking scales, not a safety
certificate. The frozen random pool has a separate 0.12 rad sampling inset:
all 100 starts/goals remain inside the 0.10 rad cost inset. The canonical start
is separate and J4 is outside that cost inset while physically valid.

This is combined parameter development; M4 does not isolate a margin effect.
Its canonical pass at 0.54 s is separate from its completed pilot's joint/speed
failure on task 2342. M5's canonical passed at 0.84 s, then all twelve assigned
tasks completed with no execution errors. All M5 speed/effort checks passed,
but task 2342 retained q excess 0.0473932 rad (M4: 0.0395858 rad).
M5 therefore remains an incomplete physical baseline, not an all-safe setup.
The cost grid is stopped and M5 frozen. A separately authorized set of 100
additional random validation tasks excludes the historical 100-task roster and
cannot inform further parameter decisions or retuning; results are pending.
The old reserved 88 remain held. These new outcomes cannot erase development
failures.
Mu=10 showed no material canonical improvement for M3 over M2, and M4's result
cannot be attributed to mu. No mu=100 sweep is queued. Matching preflights,
original physical limits, immutable tasks and all failures remain recorded.
See [the cost accounting](COST_SCALING.md) for the actual C2 formula, local
canonical gradients and finite-increment scales. No residual gate is added.

## Descriptive one-step consistency from existing records

Compare the first future optimizer state `xhat[1]` in each saved returned horizon with the saved executed state at `feedback_time + 0.03`. Keep q errors in rad and velocity errors in rad/s separately. This measures their composite practical mismatch, including discretization and model differences. The anchored `xhat[0]` is not the future prediction.

The native default integrator type 2 is labeled trapezoidal, but explicitly uses frozen initial acceleration: `v1=v0+a0*dt`, `q1=q0+v0*dt+0.5*a0*dt^2` (`integrator.cuh:20–40`). Execution uses ten 3 ms RK4 substeps. If useful, CPU Pinocchio ABA at the saved initial state and applied control can split optimizer-versus-frozen-acceleration and frozen-acceleration-versus-RK4 differences. That remains approximate because generated/native float dynamics and CPU Pinocchio need not agree exactly.

Match times explicitly. Label final partial prefixes shorter than 30 ms as unavailable for the xhat[1] comparison; retain nonfinite and missing-record counts. Use this only to describe a fixed-weight mu comparison when it helps interpret observed behavior. It is not a new acceptance condition, residual instrument, required diagnostic or solver investigation.

## Deployment identity checks

- Record the new source commit and hashes of the cost-bound header and plant call sites; include the header in wrapper source provenance.
- Confirm the selected cohort's actual native compile flags: M0–M3 use q/velocity/effort values `0.02/0.90/0.90`; M4/M5 use `0.10/0.80/0.85`. Header defaults alone do not activate margins. M4 module SHA256 is `67fe78c95f68a619571b8f783cc8b456d1aa16b8ac268adb1290cd708b0e62b7`, with source/header identity still `85ef2ea`.
- Record exact module path/SHA256 and the build configuration receipt separately from source identity. Keep the old continuation module and cohorts intact.
- Confirm the numerical diff is limited to the authorized header and nine cost-bound uses; raw limits, generated dynamics, PCG arithmetic and stopping are unchanged.
- Record complete effective weights, mu, unchanged task/input hashes and the matching preflight result for each candidate. Preserve failed preflights and hold their dependent work.
