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

## Optional SEA absolute-position feedback

The normal commands above keep the existing JointState position/velocity path.
Add `--ros-sea-absolute` to a manual launcher to select the existing arm SEA
publisher instead. For inspection without imports or motion:

```bash
python3 tiago_examples/physical_arm/reach_goals.py --preview --ros-sea-absolute
```

This option selects `/arm_right_sea_state_broadcaster_controller/sea_joint_states`
(type `sea_state_msgs/msg/SeaState`). It uses unfiltered `absolute_position` for
arm q and estimates qd by differentiating advancing message timestamps with a
light time-based low-pass filter. No 1 ms sample interval or 1 kHz delivery is
assumed. The initial velocity-filter time constant is .01 s; the short scripts
expose `SEA_STATE_TOPIC` and `SEA_VELOCITY_FILTER_TAU_S` at the top. The ordinary
examples also expose `--ros-sea-state-topic` and
`--ros-sea-velocity-filter-tau` (seconds; zero disables the velocity filter).
`--ros-joint-states-topic` selects the full-body JointState source.

The adapter still consumes full-body JointState data for collision checking,
replacing only the right-arm entries with the SEA estimate. Each source must
advance independently and remain within the existing stale timeout. It waits
for velocity initialization before reset, never substitutes motor coordinates
when SEA is unavailable, and uses the existing stop/handback behavior on loss.
The .01 s filter is an initial configurable setting, not a live-qualified tuning.
Absolute encoder zero/sign conventions must match the robot model; no automatic
encoder-offset calibration is performed. The 125 Hz effort publication setting
and existing MPC solver behavior are unchanged.

The client container needs the `sea_state_msgs` Python/type-support package only
when this option is enabled. The existing image's client-interface overlay now
builds the minimal inspected SeaState schema alongside the PAL controller types;
see `tiago_tools/pal_interfaces/README.md`. No robot package/controller is changed.
Gazebo may not provide this topic; the normal path remains usable independently.

Full-state history retains optional feedback metadata for the raw motor/absolute
channels, estimated velocity and separate source timestamps. These are encoder
and estimated-velocity data; `expected_gravity_torque` is not measured effort.
The option has local CPU/message-serialization validation only until explicit
read-only stream validation and separately authorized physical testing.

Local tests cover sustained 50 Hz SEA/full-body input with 125 Hz polling,
irregular delivery, short losses, stale streams and two-sample recovery. At 50 Hz
the same timestamped state is held between arrivals; velocity uses the actual
source interval. A 24-case seeded synthetic study also covers stationary/smooth
motion, 50/125/1000 Hz sources, regular/jittered delivery and .1/1 mrad independent
position noise. These are estimator checks, not closed-loop qualification.
With 1 mrad noise, the .01 s filter leaves roughly .055–.081 rad/s velocity error
RMS pooled across joints/time, and only 0–.83% of stationary samples meet the reset's speed-only
condition (L1 <=.1 rad/s). At .1 mrad, 99.8–100% meet it. Smooth low-noise 50 Hz
motion also shows a small filtering-lag penalty. We therefore need passive real
stream measurements before choosing a filter setting; defaults are unchanged.
Reproduce without ROS/GPU:

```bash
PYTHONPATH=tiago_src:python python3 tests/python/sea_feedback_synthetic.py --output _expr_results/sea-noise-check
```

## Prepare the Jetson before the lab

Stay inside the existing TIAGo ROS container, at this checkout's root (normally
`/workspace` in our container). Check out `felix-final-tiago-expr` and fast-forward
it; preserve any local edits. The archived overbuilt branch is a different
history, so do not merge it into this branch. The following preparation only
builds client software and previews configuration; it requires no robot access.

An older container can acquire the optional SeaState client type without an
image rebuild. Load the existing PAL controller-interface overlay first, select
loopback discovery, then build this small additional package:

```bash
source /opt/gato-pal-interfaces/setup.bash
source tiago_tools/ros_simulation.sh
colcon --log-base build/sea-client/log build \
  --base-paths tiago_tools/pal_interfaces/sea_state_msgs \
  --build-base build/sea-client/build \
  --install-base build/sea-client/install --merge-install \
  --cmake-args -DBUILD_TESTING=OFF \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3
source build/sea-client/install/local_setup.bash
python3 -c 'from sea_state_msgs.msg import SeaState; from gato_tiago.ros_tiago import validate_controller_manager_interfaces; validate_controller_manager_interfaces(); print("client interfaces OK")'
python3 tiago_examples/physical_arm/native_build.py --cuda-architecture 87
python3 tiago_examples/physical_arm/reach_goals.py --preview
python3 tiago_examples/physical_arm/reach_goals.py --preview --ros-sea-absolute
python3 tiago_examples/physical_arm/observe_sea_feedback.py --help
PYTHONPATH="tiago_src:python:$PYTHONPATH" python3 -m pytest -q tests/python/test_tiago_sea_feedback.py tests/python/test_tiago_passive_sea_observer.py
```

Use the same Python environment for build and run, preserving the sourced ROS
`PYTHONPATH`. The native command prepares both isolated N32 profiles for sm87;
it only compiles and can reuse unchanged builds. The normal feedback path needs
no SeaState package. MODE_40W is user-confirmed; no host power/clock checks or
changes are required. Do not launch a task/preflight as a preparation check:
those scripts reset and move the arm after confirmation.

## At the lab: passive feedback check

In each new container shell, source the robot profile and then the additional
client overlay. Before any motion, while the robot is already stationary, run:

```bash
source tiago_tools/ros_tiago.sh
source build/sea-client/install/local_setup.bash
python3 tiago_examples/physical_arm/observe_sea_feedback.py --duration 20
```

This standalone observer creates a subscription only: no arm client, command
publishers or controller calls. It records raw encoder channels and the same
velocity estimator at nominal 125 Hz into a fresh
`_expr_results/physical-arm/passive-sea/*.jsonl`, plus a summary of delivered
rates/gaps, position variation and velocity RMS. JSONL contains consumed samples;
callback/interval summaries cover all callbacks received by this observer. Its
delivery rates do not qualify production delivery. It does not check full-body
freshness or qualify the controller. Position variation is a noise estimate
only if the arm is stationary. Review those measurements, encoder/model
conventions and the existing full-body stream before enabling the option.

## Later physical runs

Review the Gazebo task/reset paths and collision reports, actual tool/full-body
configuration and workspace. Then change each relevant script's `BACKEND` to
`"physical_robot"` and set `PHYSICAL_PATH_AND_WORKSPACE_REVIEWED=True`.

```bash
source tiago_tools/ros_tiago.sh
source build/sea-client/install/local_setup.bash  # Only needed for SEA feedback.
```

Run the same family preflights/tasks manually. Physical launches require the
Jetson ARM64/sm87 build and the TIAGo/domain2 profile. Add `--ros-sea-absolute`
only when selecting the optional feedback path. No KKT acceptance gate or
automatic retries/tuning. Preparation performs no robot motion.
