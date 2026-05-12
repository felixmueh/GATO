# Tiago PCG Ablations

This note tracks which Tiago-specific choices are actually needed for the
large-easy Tiago PCG run, compared with the default `MPC_GATO.run_mpc_fig8`
controller style.

The fixed working baseline for these ablations is:

- `examples/large_easy_tracking.py`
- `plant=tiago_right`
- `N=32`
- `DT=0.03`
- smooth FK-generated reference from Tiago joint waypoints
- PCG backend
- `u_cost=1e-5`
- `max_sqp_iters=20`
- `max_pcg_iters=1000`
- `pcg_tol=1e-3`
- reset dual and rho every MPC tick
- warm start repeats the current state at every knot
- rejected line-search solves use gravity compensation

Baseline result:

- rejected solves: `0`
- PCG cap-hit fraction: `0.0`
- mean PCG iterations: `13.38`
- mean error: `0.011321 m`
- p95 error: `0.025343 m`
- max error: `0.028110 m`

## Results

| Case | What changed toward default `MPC_GATO` | Rejected | Cap frac | Mean iters | Mean err [m] | p95 err [m] | Max err [m] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ablate_base` | nothing | `0` | `0.000` | `13.38` | `0.011321` | `0.025343` | `0.028110` |
| `ablate_u_cost_mpc` | `u_cost=2e-6` | `93` | `0.506` | `520.97` | `0.017467` | `0.041424` | `0.073034` |
| `ablate_u_cost_5em6` | `u_cost=5e-6` | `1` | `0.077` | `89.19` | `0.011363` | `0.025539` | `0.031026` |
| `ablate_max_sqp_1` | `max_sqp_iters=1` | `33` | `0.064` | `79.09` | `0.019197` | `0.046789` | `0.121212` |
| `ablate_max_sqp_2` | `max_sqp_iters=2` | `19` | `0.030` | `45.02` | `0.019741` | `0.060467` | `0.105427` |
| `ablate_max_sqp_5` | `max_sqp_iters=5` | `3` | `0.004` | `16.74` | `0.011718` | `0.026981` | `0.030655` |
| `ablate_max_pcg_120` | `max_pcg_iters=120` | `0` | `0.000` | `13.38` | `0.011321` | `0.025343` | `0.028110` |
| `ablate_max_pcg_200` | `max_pcg_iters=200` | `0` | `0.000` | `13.38` | `0.011321` | `0.025343` | `0.028110` |
| `ablate_pcg_tol_1em4` | `pcg_tol=1e-4` | `1` | `0.006` | `22.59` | `0.011321` | `0.025351` | `0.028088` |
| `ablate_kkt_tol_1em3` | `kkt_tol=1e-3` | `0` | `0.000` | `13.38` | `0.011321` | `0.025343` | `0.028110` |
| `ablate_mu_10` | `mu=10` | `0` | `0.003` | `14.45` | `0.011284` | `0.025452` | `0.028172` |
| `ablate_rho_0p01` | `rho=0.01` | `0` | `0.000` | `14.05` | `0.011316` | `0.025367` | `0.028064` |
| `ablate_q_cost_2` | `q_cost=2` | `0` | `0.021` | `29.78` | `0.029789` | `0.047188` | `0.049903` |
| `ablate_N_cost_50` | `N_cost=50` | `0` | `0.004` | `16.75` | `0.011329` | `0.025369` | `0.028104` |
| `ablate_qd_cost_1em2` | `qd_cost=1e-2` | `0` | `0.000` | `11.73` | `0.011238` | `0.025363` | `0.028043` |
| `ablate_no_reset_dual` | carry duals across ticks | `324` | `0.735` | `738.45` | `0.366185` | `0.965962` | `1.001539` |
| `ablate_previous_solution` | warm start from previous solution | `0` | `0.070` | `77.41` | `0.017943` | `0.035239` | `0.064601` |
| `ablate_prev_no_reset` | carry previous solution and duals | `494` | `0.998` | `997.80` | `0.598947` | `0.981144` | `0.986994` |
| `ablate_no_reject_fallback` | apply rejected solution controls | `0` | `0.000` | `13.38` | `0.011321` | `0.025343` | `0.028110` |

Extra interaction checks:

- `previous_solution + no_reset_dual + max_sqp_iters=1` diverged at step `48`.
- `previous_solution + max_sqp_iters=1` with dual reset diverged at step `61`.

## Current Classification

Required for this Tiago large-easy PCG task:

- **Higher control regularization**: `u_cost=1e-5` works; `5e-6` is marginal;
  `2e-6` largely brings cap hits back.
- **Reset dual every tick**: carrying duals across ticks is not usable with the
  current wrapper because there is no dual horizon shift.
- **Current-state repeated warm start**: using the previous primal solution is
  not immediately catastrophic if duals are reset, but it reintroduces cap hits
  and worse tracking. Carrying previous primal and dual together is very bad.
- **More than one SQP iteration**: `max_sqp_iters=1` is not enough in this
  reset-every-tick setup. `5` is mostly usable; `20` is clean.
- **High running tracking weight**: `q_cost=160` still matters for tracking
  quality. Reverting to `q_cost=2` keeps the run finite but produces much worse
  tracking and some cap hits.

Probably not required, or safe to move toward default:

- `max_pcg_iters=120` is enough once `u_cost=1e-5`.
- `kkt_tol=1e-3` behaves the same as `1e-4` in this run.
- `rho=0.01` behaves about the same as `0.02`.
- `qd_cost=1e-2` is fine and slightly lowers mean PCG iterations.
- `N_cost=50` is mostly fine for this trajectory, though it introduces a tiny
  cap-hit fraction.
- The gravity fallback does not matter in the clean baseline because there are
  no rejected solves.

Open items:

- These results are for the smooth large-easy trajectory, not the default
  figure-8 reference.
- The default `MPC_GATO` rolling primal warm start would need a proper dual
  shift before carrying duals can be fairly tested.
- Timing numbers are not benchmark-grade because the machine may have shared
  GPU usage.
