# Solve-time and tracking matrix

The default plant is **TIAGo's seven-joint right arm**: native arm URDF,
`comfortable_high_clearance` start, horizontal 24-second figure-eight, and
`TIAGO_TRACKING_SOLVER_PARAMS`. Tracking measures `arm_right_tool_link` relative
to `torso_lift_link`. The reference generator is shared with
`tiago_examples/tiago_fig8_tracking.py`.

This is an offline Pinocchio simulation. It needs CUDA, but neither ROS nor
Gazebo. It measures solver timing and offline tracking; it does not run the
live controller's torque clamp, collision/velocity monitors or communication
path. It cannot establish hardware safety or end-to-end control frequency.

## Build

From the repository root, in the same Python environment used for running.
The TIAGo development container provides `/opt/gato-venv`; dependencies are in
[requirements.txt](requirements.txt).

```sh
cmake -S examples/solve_time_heatmap -B build/solve-time-heatmap-tiago_right \
  -DPLANT=tiago_right -DKNOTS='8;16;32;64;128;256' \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build/solve-time-heatmap-tiago_right --parallel 2
python examples/solve_time_heatmap/smoke.py
```

`native` requires CMake 3.24+ and a visible GPU. Alternatively specify the
architecture explicitly: 61 for GTX 1070, 87 for Jetson Orin, 120 for RTX 5060,
with a CUDA toolkit supporting that architecture. The standalone modules live
under the build directory; existing `python/bsqp` extensions are untouched.
For a smaller build, restrict `KNOTS` and pass matching `--knots` to the runner.

Resource probes use the selected plant and check each horizon's kernel shared
memory and thread requirements. Unsupported horizons appear as `SMEM`, not
numeric timings; their extension is not required. **Old Indy7 resource limits
do not apply to TIAGo.** Use separate build directories for different plants.
The runner rejects probes or modules belonging to another plant.

## Quick knot comparison

Keep prediction duration fixed. `0.504 s` matches the TIAGo N64/8 ms setup.
This pilot uses the controller's five-SQP cap and one excluded initialization
call, so a difficult initialization does not consume five large solve budgets.

```sh
python examples/solve_time_heatmap/run.py \
  --gpu-name 'RTX 5060' --horizon-time .504 \
  --knots 16 32 64 128 --batches 1 --wall-time 2 \
  --max-sqp-iters 5 --initialization single \
  --output test-artifacts/tiago-knot-pilot
```

Use the GPU's actual name. Omit `--knots` and `--batches` for the full matrix.
Use `--max-sqp-iters 1` for one-SQP operation. Omit the SQP and initialization
options to investigate native stopping: the benchmark retains its **1000-SQP
cap**, with up to five excluded initialization calls. This differs from the
live controller's five-SQP default. `INIT CAP` means initialization never
reported native stopping; that cell has no measured samples.

Defaults: knots 8/16/32/64/128/256; batches 1/2/4/8/16/32/64/128/256/512;
10-second measured wall budget per cell; one repeat; 1000 ms slow-solve cutoff.
`--repeats 3` repeats each cell. `--save-plans` also retains batch-0 plans.

`--horizon-time T` sets prediction spacing to `T/(N-1)`. TIAGo running position,
velocity, control and limit weights scale by `prediction_dt/.008`; terminal
position weight stays 80. Thus N64 at 0.504 s uses the tracking configuration's
weights exactly. Velocity and state-limit terms also occur at the terminal
knot in the native objective, retaining a resolution-dependent endpoint term.
Without `--horizon-time`, spacing remains 10 ms and weights stay unchanged:
prediction duration then grows with N.

Control updates remain 10 ms apart, with RK4 substeps of at most 1 ms. Fixed
horizons interpolate one shared periodic reference and split integration at
control-knot boundaries. Batch 0 supplies control; its unshifted plan is
broadcast to the next batch warm start. External force is zero. Rho resets
before each solve; duals reset once.

## Results and limits

Each output contains `results.json`, CSV summaries, a Markdown report,
solve-time/frequency PNG/PDF/SVG plots, per-cell inputs and NPZ samples,
initialization records, worker logs, and source/module hashes.

- Timing is synchronized native host time for a **whole batch**, excluding
  initialization, Python integration, recording, and binding transfers.
  Frequency is `1000 / mean_native_ms`.
- Faster cells cover more simulated time and figure-eight phases within the
  wall budget. Tracking means are per update, include motion startup, and are
  not phase-matched. Use this pilot to shortlist N, then compare closed-loop
  tracking over matched trajectories in the real controller.
- Native stopping currently means zero inner PCG iterations, not a nonlinear
  KKT certificate. Per-member flags, SQP counts and cap hits are retained.
- Slow solves, incomplete initialization, nonfinite trajectories and resource
  limits remain explicit outcomes. Partial samples do not become completed
  heatmap values. An asterisk marks incomplete repeats; a dagger marks SQP caps.
- Wall and slow-solve limits are checked between completed updates. An in-flight
  solve finishes and is retained. `--max-solve-ms 0` disables that cutoff;
  `--timeout SECONDS` optionally bounds the entire worker, including startup.

Run without competing GPU workloads or `CUDA_LAUNCH_BLOCKING`. The runner
checks module locations and refuses dirty solver/Python sources. It puts the
active Python environment's Torch libraries first in `LD_LIBRARY_PATH`.

Use a fresh output directory. `--resume` requires the same configuration,
sources, modules and build path. Older Indy7 manifests remain plot-readable
but cannot be resumed with this version.

```sh
python examples/solve_time_heatmap/run.py --plot-only \
  --output test-artifacts/tiago-knot-pilot
```

Plot labels come from the saved plant and GPU, not the current CLI defaults.

## Historical Indy7 experiment

Old measurements and `plots/gato_solve_time_heatmap.png` are **Indy7**.
To rerun, build separately with `-DPLANT=indy7` in
`build/solve-time-heatmap-indy7`, then select `--plant indy7` for running/smoke.

```sh
python examples/solve_time_heatmap/run.py --plant indy7 \
  --gpu-name 'GeForce GTX 1070' --protocol reference \
  --sim-time 10 --max-sqp-iters 1 --max-solve-ms 0 \
  --output test-artifacts/indy7-reference
```

Reference mode retains the protocol from `4173824`: ready pose, original
figure-eight, 10 ms prediction spacing, capped previous-solve-time simulation
clock, and joint-6 error against the next reference knot. It uses this
checkout's solver, not the historical binary. `--horizon-time` is unavailable
in that mode. Fixed-duration Indy7 runs retain their historical N32/10 ms
cost anchor; TIAGo tuning is never substituted into old results.

## CPU checks

```sh
python -m unittest discover -s examples/solve_time_heatmap -p 'test_*.py'
```

These cover TIAGo model/reference/frames, seven-joint execution, tuning,
plant selection, historical parity, initialization exclusion, time and cutoff
semantics, aggregation, and output preservation. The smoke and short pilot
exercise the compiled CUDA path.

Port validation: 34 CPU tests; N8/N64 CUDA smoke tests and short offline
pilots on c3po (RTX 5060, CUDA 12.9). N8–N128 resource probes fit that GPU;
N256 exceeds PCG shared memory. These checks establish execution, not a
completed matrix benchmark.
