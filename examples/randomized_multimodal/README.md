# Randomized multimodal trajectory optimization and MPC

This example uses actual GATO batched SQP on a planar double integrator. Cheap
random trajectory initializations receive the initial state, goal, dynamics,
limits and RNG seed, but no obstacle information. Their low-frequency uniform
control perturbations preserve the nominal start/goal boundary conditions.

The static cylinder case is an explicit control experiment: an ordinary nominal
single solve can already find the best route. The changed-scene experiment
reflects the cylinder offset across the original start–goal chord after one MPC
tick. The single solver naturally retains its previous solution; the batch also
receives fresh blind proposals. The displacement and time of this change depend
only on the scene, never on the solver's chosen route.

The `changed_state_runs` records provide the clean multistart comparison. They
freeze the single controller's measured state immediately after the prescribed
change, including its exact shifted warm initialization. Every batch size gets
that same lane zero and a nested set of random proposals, and solves the same
objective from the same physical state. Full MPC episodes are separate policy
comparisons: their states may differ, so their decision regrets use their own
state-specific references.

## Run

Build the `pointmass2d` N32 extension using the repository's CMake workflow. Its
raw module is `python/bsqp/bsqpN32_pointmass2d*.so`. Then:

```sh
python -m unittest discover -s examples/randomized_multimodal -p test_model.py
python examples/randomized_multimodal/experiment.py \
  --extension-dir python/bsqp --reference-starts 16 \
  --scene-seeds 24092400 24092401 24092402 \
  --proposal-seeds 901 902 903 --batches 1 4 8 16 \
  --mpc --switch-tick 1 --changed-state --random-single \
  --output /tmp/randomized-multimodal-development.json --plot
```

Default physical horizon is 31 × 0.08 = 2.48 seconds. Each MPC tick executes two
controls (0.16 simulated seconds), irrespective of measured compute duration.
This is a simulation experiment, not a real-time performance claim. Each solve
uses up to 80 SQP iterations. `--dt` controls physical discretization;
`--iterations` changes optimization effort. Knot count is fixed by the compiled
N32 module, not by any real-time deadline.

Omit `--switch-tick` for static MPC. `--cpu-only` runs only independent offline
reference solves. The output JSON is checkpointed after every scene. Optional
`--plot` writes publication-style PNG/PDF figures alongside it.

## Evidence and limits

GATO outputs are independently replayed. Clearance is checked at the exact
stationary points of distance along every piecewise-quadratic acceleration
segment, including endpoints. Success also requires final position error at
most 2 cm, componentwise acceleration at most 4 m/s², and velocity at most
2 m/s. MPC completion additionally requires terminal speed below 0.08 m/s.
The 15 mm optimization buffer is distinct from physical collision feasibility.

The objective matches the CUDA plant: summed position tracking, velocity,
acceleration and soft cylinder residual costs, with terminal position weight.
There is no additional `dt` multiplier. Independent L-BFGS-B multistart provides
a **best-known feasible reference**, not a global-optimality certificate.
Near-best means within 3% of that reference with all feasibility checks passing.
At the changed-state decision, the selected GATO control is independently
polished; mode, projected gradient and cost change are recorded. This polish
does not modify the online result. Report GATO's convergence flags and finite
iteration cap rather than equating low cost with proven stationarity.

`cost` in an MPC episode is the realized schedule-dependent running objective
plus terminal terms. It cannot be compared directly to a fixed-horizon
reference. `changed_decision` and `changed_state_runs` have matching-horizon
references. The `cw`/`ccw` labels denote route modes based on angular traversal;
they are not a formal topology certificate for tolerance-bounded endpoints.

Report static nominal, random-single, same-state multistart, and whole-episode
outcomes separately. Repeated proposal seeds within one scene are not
independent scene samples. Include failed scenes and missing references in the
denominators. Strong probability claims require independent held-out scene
counts and uncertainty intervals, including non-degenerate uncertainty when
every observed scene succeeds.
