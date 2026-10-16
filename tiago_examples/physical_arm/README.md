# Manual TIAGo experiments

Each script below directly calls an existing example. Read its top-level settings
and the explicit `example.run(...)` or `example.run_experiment(...)` call:

- `figure8_24s.py`: one 24 s figure-eight using `tiago_fig8_tracking.py`.
- `figure8_12s.py`: one 12 s figure-eight using the same example.
- `reach_goals.py`: three calls to `tiago_reach_target.py`, one goal per call.
- `preflight_tracking.py` / `preflight_reaching.py`: one modest 100.636 mm reach
  using the selected family's costs and native module.

The examples import their ordinary costs from `tiago_src/gato_tiago/config.py`.
Reaching uses the agreed defaults, including `u_cost=.0001`, `qd_cost=.05` and
`mu=10`. Tracking retains its own defaults. Both use N32, .03 s prediction knots,
SQP5/PCG120, B1 and orientation off. The live loop uses observed time; .03 s is
not a scheduled feedback interval. No solver changes or final-approach tuning.
There is no dispatcher, runner queue or preflight cache.

## Prepare and inspect on RTX5060

Run from this checkout's root, inside the existing container. Start the usual
PAL Gazebo instance separately.

```bash
source tiago_tools/ros_simulation.sh
python3 tiago_examples/physical_arm/native_build.py --cuda-architecture 120
python3 tiago_examples/physical_arm/figure8_24s.py --preview
python3 tiago_examples/physical_arm/reach_goals.py --preview
```

Preview imports no ROS/native modules and causes no motion. The build command
only compiles; CUDA must support sm120 (CUDA12.8 or newer). It prepares isolated
N32 modules under `build/physical-arm-native/`: ordinary tracking cost intervals
and reaching's .20 rad/.80 velocity/.85 effort interior intervals. Each launcher
loads its appropriate module before calling the example. Installed extensions
are not overwritten; unchanged matching builds can be reused.

## Run by hand

```bash
python3 tiago_examples/physical_arm/preflight_tracking.py
python3 tiago_examples/physical_arm/figure8_24s.py
python3 tiago_examples/physical_arm/figure8_12s.py

python3 tiago_examples/physical_arm/preflight_reaching.py
python3 tiago_examples/physical_arm/reach_goals.py
```

Review each family preflight's outcome before running its tasks. A failed
preflight raises and retains data; preserve it and discuss the setup before
continuing that family. Task launches do not enforce a cached receipt.

Each call prints its settings and actual task/cost configuration, then asks you
to type `MOVE` before reset or effort motion. Every reaching goal creates a new
controller, resets to comfortable, verifies measured arrival and hands back.
Arrival requires <=.05 rad maximum joint error and <=.1 rad/s joint-speed L1.
The three target distances are 100.636/207.933/317.052 mm from nominal comfortable;
they remain candidates awaiting current-setting simulation/collision review.
Each gets eight seconds, <=5 cm position arrival, speed L1<=1 rad/s, zero dwell.
An ordinary timeout is recorded; the next goal still requires confirmation.
Errors stop the sequence. Repeat a tracking run with another manual invocation.

## Data and collision review

Each invocation uses a fresh UTC-stamped folder in
`_expr_results/physical-arm/gazebo/`. Existing example outputs are retained:
tracking data are in `<run>/tiago_right/`; reaching data in `<run>/goal-N/data/`.
Those leaf directories also contain:

- `command_history.jsonl`: every effort publication attempt, including priming
  and handback zeros, seven commanded torques in Nm, timestamps and outcomes.
  These are commanded torques, not measured actuator torques.
- Full joint-state history, lifecycle boundaries, completeness metadata,
  resolved settings, source/config identity and `result.json`.
- `task_stats.json`: existing task statistics and partial statistics on failure;
  `solve_calls` retains native and wall milliseconds per returned solve, even
  when the following controller operation rejects it.

Command recording is buffered in the worker and exported after handback, with
no per-tick disk IO. Cleanup runs before summaries/rendering and exports are
attempted independently on failure. The backend's `manual-run-index.jsonl`
indexes these recordings. Raw data and failures need a separate backup.

Run the separate CPU collision replay on each leaf directory:

```bash
python3 tiago_examples/physical_arm/check_collisions.py /absolute/path/to/run/tiago_right
python3 tiago_examples/physical_arm/check_collisions.py /absolute/path/to/run/goal-1/data
```

Replay checks full named robot states, reset/task/handback coverage, joint limits
and the 4 cm monitored-pair margin. It distinguishes clearance from completeness,
reports original gaps >100 ms and never interpolates across them. <=3 ms linear
interpolation within supported gaps is sampled review, not a continuous-motion
guarantee. Task error/timeout labels remain separate. Inspect paths visually;
external workspace objects are outside this self-collision model.

Runtime collision, joint, speed, torque, nonfinite and stale-state guards remain
active. The PAL position-controller reset is recorded and arrival checked; the
GATO effort collision monitor starts after reset. Normal cleanup does not cancel
an accepted PAL reset trajectory. Physical reset paths need supervised review.

## Later Jetson use

Review the Gazebo task/reset paths and collision reports, actual tool/full-body
configuration and workspace. Then change each relevant script's `BACKEND` to
`"physical_robot"` and set `PHYSICAL_PATH_AND_WORKSPACE_REVIEWED=True`.

```bash
source tiago_tools/ros_tiago.sh
python3 tiago_examples/physical_arm/native_build.py --cuda-architecture 87
```

Run the same family preflights/tasks manually. Physical launches require the
Jetson ARM64/sm87 build and the TIAGo/domain2 profile. MODE_40W is user-confirmed;
no host power/clock commands or changes. No KKT acceptance gate or automatic
retries/tuning. Preparation performs no robot motion.
