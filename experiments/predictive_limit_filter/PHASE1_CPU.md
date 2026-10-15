# Predictive control filter — phase 1 CPU functional test

This is an isolated experiment helper, not a solver change or deployed controller. No native module, GPU solve, full controller episode, physical robot or Jetson execution ran. Source starts from the frozen Jetson frontend `72533b237a55732aaa11b7747f204eee0ff16725`; the URDF and reused RK4 helper are unchanged.

The filter forecasts a proposed constant torque for 30 ms with the existing coupled Pinocchio RK4 dynamics, at samples no more than 3 ms apart. Prediction guards are physical q limits inset 0.02 rad, physical speed limits minus 0.10 rad/s (2.4 rad/s), and 95% of physical effort limits. A passing proposed torque is retained exactly. Otherwise a fixed small candidate set uses coupled RNEA bias compensation, acceleration projected into approximate q/v endpoint bounds, braking and stronger braking, and three fixed proposal/braking blends. Each candidate is torque-bounded and checked by the actual forward forecast. The closest passing torque in normalized squared effort distance is selected. There is no assumption that scaling torque toward zero brakes the robot, and no new optimizer dependency.

A **guard forecast pass** requires all sampled states, including the initial state, inside the guard interval. If the initial state is physically valid but already outside the guard band, a **recovery-only** candidate may pass raw physical limits and reduce its guard violation without increasing its peak; this is explicitly not guard-certified. `no_candidate_certified` means the fixed candidates supplied no accepted forecast, not proof that every possible torque is infeasible. Already physically invalid or nonfinite initial states receive no certificate. Any later episode stopped by the filter must retain its incomplete status and must not become a full-episode physical pass.

The CPU test uses all 16 J1 and 13 J2 finite, error-free, physically unsafe episodes from the original Jetson studies. Each contributes two saved state/control anchors: immediately before the first guard entry, and before the first recorded raw violation. Original numerical errors are excluded separately (4 J1, 9 J2) rather than being counted as physical passes. The original successful/unsafe/error labels remain unchanged.

| Saved anchor | J1: guard / recovery / no candidate | J2: guard / recovery / no candidate |
|---|---:|---:|
| Before first guard entry | 14 / 0 / 2 of 16 | 12 / 0 / 1 of 13 |
| Before first raw violation | 8 / 6 / 2 of 16 | 7 / 5 / 1 of 13 |

All 29 first-guard anchors start inside the guard interval. Thus the fixed search finds modified, guard-passing forecasts for 26 of them. These are **30 ms constant-command counterfactual predictions**, not 26 rescued episodes. The original command at each saved point was actually applied for its recorded segment (at most 3 ms), and later original controls need not equal that held command.

For example, J1 pair502 at 1.038939 s has a proposed held-torque 30 ms forecast with peak speed ratio 1.878. A fixed blend returns a forecast with peak ratio 0.936 and effort ratio 0.277, compared with proposed effort ratio 0.430. This establishes a useful modeled correction at that one state, not its future task outcome.

The search finds no guard-certified candidate for J1 pairs6566/7843 and J2 pair3925 at either anchor. For pair7843 some tested controls keep the forecast within raw physical limits but enter the 0.02 rad q guard band: for example stronger braking has minimum physical q clearance 0.01537 rad. That is a guard-policy rejection, not proof of physical infeasibility. All candidate predictions and rejections remain in the raw JSON.

A separate deterministic coupled two-joint/gravity test verifies that a bias-balanced command is retained, zero torque can cause a speed violation, a nonzero inverse-dynamics correction passes, consumed-buffer recovery remains distinct, and an already physically invalid state is not certified. It uses stub dynamics only; no native solver call.

## Timing and next decision

On this desktop, these difficult boundary samples take a median roughly 5.8 ms per search. Per-group p95 is 5.9–7.2 ms; maximum observed is 8.13 ms. All 58 exceed the nominal 3 ms control step. This is descriptive, not a deadline acceptance gate. These are desktop Python/Pinocchio CPU timings, not Jetson measurements or evidence of real concurrent scheduling.

The original Mode-B native binding retains the Python GIL and its previous plan runs while the solve wrapper is charged. A same-process Python safety thread would not provide the assumed continuous filtering. A quick simulation may explicitly model an independent high-rate safety loop applying the filter to every actual integration/control segment, including stale-plan intervals, but serialized replay cannot establish concurrent scheduling, resource contention or full computation-paid behavior. Root must settle that architecture/timing scope before closed-loop integration. No worker has been changed in this phase.

No robust/model-error, continuous-time, collision or physical-robot safety guarantee is claimed. No KKT stopping criterion, solver parameter or original task acceptance gate changes. Native/GPU execution remains with the sole owner and waits root review/handoff.

## Reproduction and records

```sh
cd /workspace/GATO-jetson-predictive-limit-filter
python experiments/predictive_limit_filter/check_cpu.py
python experiments/predictive_limit_filter/replay_cpu.py --output /workspace/GATO/_expr_results/random-reaching-predictive-filter-20261015/phase1-cpu-new
```

Existing output: `/workspace/GATO/_expr_results/random-reaching-predictive-filter-20261015/phase1-cpu/`. `replay_samples.csv` retains all 58 anchors; `replay_details.json` retains exact saved states, original/filtered torques and each candidate forecast. `counts.json`, `functional_checks.json`, `excluded_error_episodes.json` and `metadata.json` preserve scopes, timings and source/input hashes. Original Jetson data and all prior experiment sources remain untouched. The separate directed two-target and one-shot horizon tuning studies remain held/prepared as previously instructed.
