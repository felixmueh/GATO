# Selected upward-pointing Gazebo motion

Felix chose the existing upward-pointing 12-second demo on 2026-10-12. This
[separate successor plan](rtx5060_gazebo_upward_selected.json) keeps goal1 and
selects only that demo figure-eight. The previous 23-case Gazebo plan and all
its attempted outcomes remain unchanged. The old square is explicitly disabled
while the requested front placement and rotation are prepared; it cannot run
from this plan. Optional fault cases are excluded.

The selected figure-eight uses unchanged live runtime
`8dbb8b8d6dc1d65e902373a9aa58693e33329c14`, collector
`2574409ee4236b7c32035ec27dbaca19cd27dfeb` and frozen proposal SHA256
`46e6b66efdfae2c6365d004e129d0f18fbd06da4f7043c0b3c8679897ed6f068`.
It retains N32, 30 ms knots, 8 ms tracking feedback, the full comfortable-yaw
reference with tool axis +Z, tracking costs, initial state, physical checks,
canonical preflight and controller handback. A narrow adapter selection change
allows this single existing task without launching the historical four-task
sequence. No goal, reference, solver, timing parameter or gate is tuned.

Matching scientific internal evidence is already saved at
`_expr_results/gtx1070-zero-pcg-20261011T214000Z/gtx1070/demo_visual_square_internal/figure8-12s-r000-a81be32ccae2a73e/attempt-003`.
The bounded comparison checked 25 motion fields including the complete packed
reference and both tracking gates; they match collector `2574409`. The saved
internal runtime and the collector's internal pin are both `fa6f6be`. Tracking
execution/evaluation and the float32 force boundary agree; collector and
recording identities still distinguish the cohorts. A new internal run is
unnecessary solely for that collector migration.

**The matching saved internal task flag is false; physical checks passed.**
Startup-inclusive RMS error is 19.0519 mm and maximum error is 100 mm, exceeding
the frozen 60 mm maximum gate; maximum upward-axis error is 19.822 degrees.
Retain those outcomes and the original gate. This is limited internal evidence,
not a complete task-success qualification or a Gazebo pass. Felix selected this
setup with its recorded limitations. New live outcomes must be retained as-is.

After pulling the published runner branch, use the existing RTX5060 PAL/ROS
environment and simulator prerequisites in the main remote-machine handoff.
Finish the active invocation before switching plans. Use a fresh result cohort
and the existing unchanged Gazebo cache to reuse matching builds. From the
runner checkout, prepare and inspect this selection:

```sh
GAZEBO_EXPERIMENT_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
"$GAZEBO_EXPERIMENT_PYTHON" -c 'import json,sys; sys.path.insert(0,"experiments"); import machine_launch; machine_launch.verify_sources(json.load(open("experiments/launch/rtx5060_gazebo_upward_selected.json")))'
"$GAZEBO_EXPERIMENT_PYTHON" experiments/run.py --machine rtx5060 \
  --plan experiments/launch/rtx5060_gazebo_upward_selected.json \
  --python "$GAZEBO_EXPERIMENT_PYTHON" \
  --cache .experiment-cache/rtx5060-gazebo \
  --output .experiment-results/rtx5060-gazebo-upward-selected-20261012 --dry-run
```

Expected selection: **two motion cases**, `D01/goal-1` and
`UpwardFigure8-12s-Gazebo/figure8-12s`. The square stays held. After the usual
manual setup review, the launch and collection commands are:

```sh
"$GAZEBO_EXPERIMENT_PYTHON" experiments/run.py --machine rtx5060 \
  --plan experiments/launch/rtx5060_gazebo_upward_selected.json \
  --python "$GAZEBO_EXPERIMENT_PYTHON" \
  --cache .experiment-cache/rtx5060-gazebo \
  --output .experiment-results/rtx5060-gazebo-upward-selected-20261012
"$GAZEBO_EXPERIMENT_PYTHON" experiments/collect.py \
  --output .experiment-results/rtx5060-gazebo-upward-selected-20261012 \
  --export .experiment-work/rtx5060-gazebo-upward-selected-20261012-results.tar.gz
```

Existing canonical, collision safety and handback gates apply. Unexpected
setup or validation failures retain partial data and require review; no
retry-until-green, startup filtering or gate relabeling. A future square
successor needs its reviewed collector/input identity and fresh cohort choice.
Collect existing partial attempts after interruption too; copy the archive to
the desktop `_expr_results/` and keep a separate raw backup.

Publication requires the runner branch and immutable source refs
`experiment/deployment-source-8dbb8b8d6dc1` and
`experiment/deployment-source-2574409ee423` on the same origin. Preparation
does not claim publication, a target build, live motion or physical robot work.

Preparation checks (2026-10-12): 20 focused demo/recording/deployment CPU checks
passed, including single-task dispatch with the explicit collector and preserved
preflight handback path. Static source/input/build and protocol checks passed;
the two-case dry-run passed. The original 23-case plan and the Jetson supplement
were unchanged. No native build, solver call or live execution ran here.
