# Remaining Gazebo motions and accepted four-corner square

Felix accepted the passing internal four-corner motion. This [separate plan](rtx5060_gazebo_four_corner_v10.json)
enables only the selected upward 12-second figure-eight and the accepted square.
Goal1 is disabled: Felix-supplied attempt002 metrics report task, observed
physical and handback passes at runner `666560f`/live runtime `8dbb8b8`.
Its archive and exact remote receipt path have not been imported; collect the
existing attempt002 evidence rather than rerun it. Previous plans and failures
remain intact. Optional M03 faults use their separate successor plan.

The square is 10 × 10 cm, centered at torso `[0.50,-0.35,-0.15]` m, with the four
inward-pointing corners in the passing order and full rotations. **There is no
center reach.** Input bytes, N32/30 ms knots/3 ms internal plant/15 s per goal,
costs, final approach, strict settling, initialization and controller settings
are unchanged from executed `a968603`. All four rough/strict reaches and recorded
plant checks passed; 1,449 sampled actual states passed the 40 mm collision
margin, with 43.05 mm minimum. This is one internal run and sampled clearance,
not live or physical qualification. No GPU rerun was needed for packaging.

Square collector `91118e32b380e5b19220b460574dbfb052c763f5` explicitly defaults
to `proposal_right10_roll.json`, and all standalone scripts explicitly bind
that selected input. A collector repin alone would have selected the old
proposal. Runner frozen input SHA256 is
`8f1725b1ee38ee8b4b479be4d10637ed112f08e83398bdcf0c5fa091f1ddafd2`.
The proposal's inherited `accepted_square_evidence` describes historical v5;
the current plan points to the actual v10 evidence. The square has no tracking
tasks. Upward tracking keeps its separate collector `2574409` and original
reference; its saved internal task failure/physical pass and startup-inclusive
100 mm maximum remain unchanged. Live runtime stays `8dbb8b8` for both motions.

Publish the runner branch and all three immutable source refs from a host with
working origin access. Publication is unverified: the container encountered
SSH host-key verification failure; no security settings were changed and no
push occurred. Exact source refspecs are:

```sh
git push origin experiment/simple-runner \
  91118e32b380e5b19220b460574dbfb052c763f5:refs/heads/experiment/deployment-source-91118e32b380 \
  8dbb8b8d6dc1d65e902373a9aa58693e33329c14:refs/heads/experiment/deployment-source-8dbb8b8d6dc1 \
  2574409ee4236b7c32035ec27dbaca19cd27dfeb:refs/heads/experiment/deployment-source-2574409ee423
```

After pulling that branch on RTX5060, finish any active invocation and use the
existing PAL/ROS-capable **active Python interpreter**, simulation environment
and manual setup review from the main handoff. The runner explicitly selects
RTX5060 (`120-real`) and reuses matching cached native builds. It does not start
Gazebo or change controller/ROS settings. From the runner checkout:

```sh
git pull --ff-only origin experiment/simple-runner
GAZEBO_EXPERIMENT_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
"$GAZEBO_EXPERIMENT_PYTHON" -c 'import numpy,pinocchio,rclpy; import json,sys; sys.path.insert(0,"experiments"); import machine_launch; machine_launch.verify_sources(json.load(open("experiments/launch/rtx5060_gazebo_four_corner_v10.json")))'
"$GAZEBO_EXPERIMENT_PYTHON" experiments/run.py --machine rtx5060 \
  --plan experiments/launch/rtx5060_gazebo_four_corner_v10.json \
  --python "$GAZEBO_EXPERIMENT_PYTHON" --cache .experiment-cache/rtx5060-gazebo \
  --output .experiment-results/rtx5060-gazebo-four-corner-v10-20261012 --dry-run
"$GAZEBO_EXPERIMENT_PYTHON" experiments/run.py --machine rtx5060 \
  --plan experiments/launch/rtx5060_gazebo_four_corner_v10.json \
  --python "$GAZEBO_EXPERIMENT_PYTHON" --cache .experiment-cache/rtx5060-gazebo \
  --output .experiment-results/rtx5060-gazebo-four-corner-v10-20261012
"$GAZEBO_EXPERIMENT_PYTHON" experiments/collect.py \
  --output .experiment-results/rtx5060-gazebo-four-corner-v10-20261012 \
  --export .experiment-work/rtx5060-gazebo-four-corner-v10-20261012-results.tar.gz
```

The two-case dry-run should list `UpwardFigure8-12s-Gazebo/figure8-12s` and
`SquareFourCorner-v10-Gazebo/square`. Canonical, collision and handback gates
remain. Keep unexpected failures and partial attempts; collect after interruption
too. Transfer this archive and the existing goal1 attempt002 archive to desktop
`_expr_results/`, with a separate raw backup.

Physical execution remains standalone and manual, after live review. On Jetson,
use the existing robot ROS profile and active ROS-capable interpreter; modules
must be the reviewed Jetson N32 build (`87-real`), never an RTX binary. Check out
collector `91118e3` and live runtime `8dbb8b8` separately. In the collector's
`experiments/deployment/demo/demo_robot.py`, set the top-level `RUNTIME_SOURCE`
to the **absolute live-runtime checkout path**, and `N32_MODULE_PATH` to the
reviewed Jetson module. The default `None` source fallback is the collector
checkout and must not be used as an implicit live runtime. Keep preview on first:

```sh
python3 /absolute/collector-91118e3/experiments/deployment/demo/demo_robot.py
```

Review its four exact goals/constants, configure `PREFLIGHT_ONLY` and a fresh
matching canonical attempt, then supply `PREFLIGHT_OUTPUT` before reviewing
execution. The script retains per-task explicit motion confirmation, robot
environment checks, collision safety, handback, fresh timestamped raw output
and a separate manual-robot index. The selected default is square-only; upward
tracking retains its separate collector. No physical motion was run here.

The separate manual checkout commands, from the published runner checkout, are:

```sh
git worktree add --detach .manual-deployment/four-corner-collector-91118e3 91118e32b380e5b19220b460574dbfb052c763f5
git worktree add --detach .manual-deployment/live-runtime-8dbb8b8 8dbb8b8d6dc1d65e902373a9aa58693e33329c14
```

Set the script's `RUNTIME_SOURCE` to the absolute second checkout path and use
the absolute first checkout path for its preview command. Fetch missing source
refs with the same existing `machine_launch.verify_sources` helper shown above;
it provisions source objects only and does not launch motion. These worktrees
are manual deployment checkouts, separate from runner-owned caches.

Preparation checks passed: exact accepted input and all resolved scientific
fields; three standalone previews, including explicit robot binding; one CPU
test of the archived collector/input/standalone contract; two-case dry-run;
source/input/build pins and command syntax. Earlier20 unchanged adapter CPU
checks were reused. No production runtime/solver change, native build, GPU rerun,
live execution or push was performed for this packaging.
