# Manual TIAGo tasks: Gazebo first

Branch `felix-final-tiago-expr` starts from `felix-devel` at `aa10e48`.
The scripts use the existing MPC, ROS controller, collision monitor and handback.
There is no experiment runner or task queue. All commands below run inside the
existing container. Start the usual PAL Gazebo instance separately.

## RTX 5060 preparation

From this checkout's root:

```bash
source tiago_tools/ros_simulation.sh
python3 tiago_examples/physical_arm/native_build.py --cuda-architecture 120
python3 tiago_examples/physical_arm/preflight_tracking.py --preview
python3 tiago_examples/physical_arm/preflight_reaching.py --preview
```

The build command only compiles. It produces two isolated N32 modules under
`build/physical-arm-native/`: tracking uses original cost intervals, reaching
uses the .20 rad/.80 velocity/.85 effort intervals. Every task automatically
loads its verified module; no manual binary copying or swapping. Preparation
does not overwrite the repository's installed extensions. Changes to relevant
native source require preparation again. Use the same Python environment for
preparation and execution. CUDA must support sm120 (CUDA 12.8 or newer).

`--preview` prints all task constants without loading ROS/native code or moving
anything. Review the top of each script before running. `BACKEND = "gazebo"`
is the initial setting. ROS profile/domain mismatches fail before motion.

## Manually run one task at a time

```bash
python3 tiago_examples/physical_arm/preflight_tracking.py
python3 tiago_examples/physical_arm/figure8_24s.py
python3 tiago_examples/physical_arm/figure8_12s.py

python3 tiago_examples/physical_arm/preflight_reaching.py
python3 tiago_examples/physical_arm/reach_goals.py
```

Each invocation prints its configuration and requires the exact displayed
confirmation before **any motion**, including reset. For the three reaches,
confirm each goal separately; each one creates a fresh controller, resets to
comfortable, checks arrival, executes and hands back. A task error, physical
failure or unconfirmed handback stops the remaining goals. An ordinary timeout
is retained and printed; the next goal still requires a new confirmation.

Both preflights use the existing modest 100.636 mm T01 target, with the selected
family's exact costs and native module, position-only arrival <=5 cm and speed
L1 <=1 rad/s. Successful same-backend/source/build/settings evidence is reused
automatically; no preflight after every small run. A failed new preflight
invalidates the cached pass for that family. Preserve the failure and review it
before dependent runs. These are new preflights; older orientation-on or
infrastructure-only canonical results are not substituted.

Tracking: one horizontal figure-eight cycle at 24 s or 12 s, N32/.03 s,
SQP5/PCG120, orientation off. Repeat the 24 s script manually for the proposed
second run. The original 30/60 mm RMS/maximum labels remain descriptive task
criteria and do not change physical guards. Timing/error metrics in result.json
are feedback-sample metrics; full controller observations/reference are saved
for subsequent time-weighted analysis. All native/wall solve times are retained.

Reaching: three T01 FK-derived targets 100.636/207.933/317.052 mm from the nominal
comfortable start, eight seconds each, no final approach or orientation gate.
All costs are explicit and match the new reaching defaults including u=.0001.
These are candidates awaiting new-setting task/collision review. The J1 random
tasks used different initial states, so their successes do not qualify these
comfortable-start reaches. No additional far goal is included.

## Inspect collision clearance and preserve data

Each run prints its fresh timestamped directory under
`_expr_results/physical-arm/gazebo/`. Run the CPU-only replay on each directory:

```bash
python3 tiago_examples/physical_arm/check_collisions.py /absolute/path/to/run
```

It checks full named robot states and <=3 ms linear interpolation, including
the recorded reset, against the same collision geometry/excluded pairs and
4 cm monitored-pair margin. `collision_review.json` reports the limiting pair,
clearance/time, configuration and geometry hashes, missing/dropped data and
separate q/v violations. Nonzero exit means failed/incomplete review. A sampled
interpolated path is not a continuous-motion guarantee. Inspect the trajectory
visually as well; external lab objects are outside this self-collision model.

Runtime effort control keeps collision, joint, velocity, torque, stale-state,
nonfinite and worker checks active. There is no report-only safety bypass or
torque clamping. The reset runs under the existing PAL position controller;
its observations are recorded and arrival is checked before effort setup.
The GATO effort collision monitor starts after reset. Existing cleanup restores
controller mode; it does not cancel an already accepted PAL position trajectory.
Supervised motion, a reviewed reset path and the normal manual stop remain
necessary for physical operation.

Saved files include resolved_config/provenance/result, compact per-solve work and
timing JSONL, failed-call arrays where applicable, reference, task stats, full
joint/publication history, history completeness and a manual-run index. Published
`applied_tau` is distinct from measured actuator torque, which is unavailable.
Keep raw data and a separate backup; finalized reports/figures go to thesis_results.
No KKT acceptance gate, retries or automatic tuning is added.

## Later Jetson use

Review the Gazebo results and collision reports first, including reset paths.
For physical execution, review the actual full-body/tool configuration and clear
workspace; edit the relevant scripts' `BACKEND` to `"physical_robot"` and set
`PHYSICAL_PATH_AND_WORKSPACE_REVIEWED = True` only after that review. This flag
records the operator's review; it does not certify geometry or predict a real
path. Physical runs require ARM64/sm87 and the TIAGo/domain2 profile:

```bash
source tiago_tools/ros_tiago.sh
python3 tiago_examples/physical_arm/native_build.py --cuda-architecture 87
```

Then run the same family preflight and task scripts manually. Comfortable arrival
must be within .05 rad max joint error and .1 rad/s L1 speed, with bounded fresh
state checks. Joint-state rate is recorded, not assumed from the 125 Hz command
rate or 30 ms prediction knots. MODE_40W is user-confirmed; no power/clock commands
or changes are needed. Physical scripts and results remain subject to Felix's
manual review; this preparation performs no robot motion.
