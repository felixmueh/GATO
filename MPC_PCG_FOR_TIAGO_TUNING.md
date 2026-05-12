# MPC PCG For Tiago Tuning

This note summarizes the current findings for making the GATO PCG backend work
on Tiago right-arm tracking.

## Main Result

The original Tiago PCG failures are not primarily caused by closed-loop MPC
warm-start quality. The reduced one-shot trajectory-optimization test and the
full large-easy tracking test both point to a numerical conditioning problem in
the Tiago Schur systems.

The most important tuning change is:

```text
u_cost = 1e-5
```

The previous/default-like values around `1e-6` to `2e-6` make the Tiago Schur
system badly scaled for single-precision PCG. With `u_cost=1e-5`, the full
large-easy Tiago PCG run has zero rejected solves and zero PCG cap hits.

## Best Current Large-Easy Setup

For `examples/large_easy_tracking.py --plant tiago_right --N 32`:

- PCG backend
- `u_cost=1e-5`
- `max_sqp_iters=20`
- `max_pcg_iters=120` is enough, though `1000` was used for many probes
- `pcg_tol=1e-3`
- `q_cost=160`
- `qd_cost=1e-2` or `2e-2` both work
- `N_cost=50` or `800` both work on this trajectory
- `rho=0.01` or `0.02` both work
- reset dual and rho every MPC tick
- warm start by repeating the current state across the horizon

Representative result with `u_cost=1e-5`, `max_pcg_iters=1000`:

- rejected solves: `0`
- PCG cap-hit fraction: `0.0`
- mean PCG iterations: `13.38`
- mean tracking error: `0.011321 m`
- p95 tracking error: `0.025343 m`
- max tracking error: `0.028110 m`

## What Was Actually Required

Required or important:

- **Increase `u_cost`**.
  - `1e-5` works cleanly.
  - `5e-6` is marginal.
  - `2e-6` largely brings cap hits back.
- **Reset duals every tick**.
  - Carrying duals across ticks without shifting them is very bad.
- **Use repeated-current-state warm starts**.
  - Reusing the previous primal solution reintroduces cap hits and worse
    tracking in the current wrapper.
- **Use more than one SQP iteration**.
  - `max_sqp_iters=1` is not enough in this reset-every-tick setup.
  - `5` is mostly usable.
  - `20` is clean.
- **Keep a strong running tracking weight**.
  - Reverting `q_cost` to `2` worsens tracking and brings back some cap hits.

Probably not required for this smooth large-easy trajectory:

- `max_pcg_iters=1000`; `120` is enough once `u_cost=1e-5`.
- `kkt_tol=1e-4`; `1e-3` behaves the same here.
- `rho=0.02`; `0.01` works.
- `qd_cost=2e-2`; `1e-2` works.
- `N_cost=800`; `50` is mostly fine here.
- Gravity fallback does not matter in the clean tuned baseline because there
  are no rejected solves.

## Why `u_cost` Matters

`u_cost` is the quadratic control-effort penalty on joint torques.

For Tiago, very small `u_cost` makes the control block weakly regularized. In
the Schur system this creates very large values, especially in the last
velocity-state coordinate. A dumped cap-hit system with `u_cost=1e-6` had:

- worst Schur entry: about `6.38e10`
- absolute condition estimate: about `1.04e13`
- direct CPU solve succeeds
- single-precision GPU PCG hits the `1000` iteration cap

Raising `u_cost` to `1e-4` in the same one-shot test changed the Schur system to:

- worst Schur entry: about `6.38e8`
- absolute condition estimate: about `1.89e11`
- GPU PCG exits in `19` iterations

So the control penalty is acting as necessary numerical regularization for
single-precision PCG on Tiago.

## Dual Reset Explanation

The dual variables are the Lagrange multipliers for the linearized equality
constraints in the SQP/KKT solve. In this solver they live along the whole
horizon, one block per knot/constraint stage.

In MPC, the problem at tick `t+1` is not the same indexed problem as at tick
`t`. The horizon moves forward:

```text
tick t:     [x0, x1, x2, ..., xN]
tick t+1:   [x1, x2, x3, ..., xN+1]
```

If we want to carry duals across ticks, we must shift them the same way:

```text
lambda_t[1] -> lambda_{t+1}[0]
lambda_t[2] -> lambda_{t+1}[1]
...
```

and then choose a sensible value for the new tail dual. The current Python
wrapper does not expose or implement this dual shift.

If we simply keep the internal dual array as-is, then each multiplier remains
attached to the wrong stage of the next MPC problem. That means the solve starts
with stale dual information for constraints from the previous time index and
previous reference window. For Tiago PCG this is especially damaging because
the Schur systems are already numerically sensitive.

Observed result:

- With the tuned baseline and dual reset:
  - rejected solves: `0`
  - cap fraction: `0.0`
  - mean tracking error: `0.011321 m`
- Carrying duals without shifting:
  - rejected solves: `324`
  - cap fraction: `0.735`
  - mean tracking error: `0.366185 m`
- Carrying both previous primal and unshifted duals:
  - rejected solves: `494`
  - cap fraction: `0.998`
  - mean tracking error: `0.598947 m`

Therefore, until a correct dual horizon shift is implemented and tested, Tiago
MPC should call:

```python
solver.reset_dual()
```

at every MPC tick.

## Open Caveats

- These conclusions are for the smooth large-easy Tiago trajectory, not yet for
  the original default figure-8 reference.
- Timing numbers are directional only because this machine may have other
  agents sharing the GPU.
- The debug dump hooks currently in the working tree are gated by environment
  variables, but they are still instrumentation code and should not be confused
  with the tuning itself.
