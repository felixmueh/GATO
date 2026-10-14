# Task parameter studies — prepared launch instructions

These are new cohorts using PCG120 throughout. Collector/live source: `90bdb825d059e22b709e2e53a785d7d034aad1b0`; internal numerical runtime: `fa6f6bede286c122cc41ac65f42a583da84290e0`. The live successor starts from `8dbb8b8d6dc1d65e902373a9aa58693e33329c14`; native solver/model files are unchanged. No builds or task runs were performed during preparation. Run GPUs sequentially on each machine. Start with the separate readiness/cost pilots; discuss their duration and Jetson storage/runtime before launching the full queue.

| Setting | Horizon length and solver budget | Knot spacing and horizon length (new cells) | Total |
|---|---:|---:|---:|
| GTX1070 internal | 42 | 40 | 82 |
| RTX5060 internal | 42 | 40 | 82 |
| Jetson internal | 42 | 40 | 82 |
| RTX5060 Gazebo | 27 | 20 | 47 |

Total: 293 scientific episodes. Readiness is separate. Pilot cases are identical science cells and resume from the same output directory. Study B reuses Study A's 30 ms/SQP5 tracking cells. The guarded historical D01 anchor is context; the new Gazebo figure-eight anchor has two fresh report-only repetitions.

Each source/backend uses one existing all-four-N native build recipe. N16/32/64/128 modules are checked as artifacts, and canonical N32 remains available independently of the grid cell's N. SQP and knot spacing select runtime parameters, not new native builds. The new live source may require a fresh all-four-N build; internal fa6 modules can reuse an identical existing cache. Safety policy changes attempt/preflight identity and directories, while sharing native builds.

Native import/finite-solve readiness for N16/32/64/128 on every GPU is **untested during preparation and pending execution**. Artifact presence is not qualification. Review the first scheduled cell for each N before running its dependent cells; pilots cover N16/N128 and canonical uses N32, so N64 remains untested. Run `--only A-N64-SQP1` separately and inspect import/execution/finiteness before the full queue. Preserve expected task/physical negatives; they do not indicate a module-readiness failure. Genuine module/import/nonfinite execution failures hold dependent cells for that N.

Use the existing environment; the selected Python must import the task dependencies and, for Gazebo, ROS. Inside the current checkout on `experiment/simple-runner`:

```bash
git pull --ff-only origin experiment/simple-runner
TASK_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
```

Choose exactly one setting below, then keep its output/cache for both pilots and full queue:

```bash
# GTX1070 internal
SETTING=gtx1070
MACHINE=gtx1070

# RTX5060 internal: SETTING=rtx5060_internal; MACHINE=rtx5060
# Jetson internal: SETTING=jetson; MACHINE=jetson
# RTX5060 Gazebo: SETTING=rtx5060_gazebo; MACHINE=rtx5060

TASK_OUTPUT="$PWD/.experiment-results/task-parameters-20261014-$SETTING"
TASK_CACHE="$PWD/.experiment-cache/task-parameters-20261014-$SETTING"
TASK_PLAN="experiments/launch/task_parameters/$SETTING.json"
TASK_PILOTS="experiments/launch/task_parameters/${SETTING}_pilots.json"
"$TASK_PYTHON" -c 'import json,sys; from pathlib import Path; sys.path.insert(0,"experiments"); from machine_launch import verify_sources; verify_sources(json.loads(Path(sys.argv[1]).read_text()))' "$TASK_PLAN"
```

Internal pilot review and execution (the first command executes nothing):

```bash
"$TASK_PYTHON" experiments/run.py --plan "$TASK_PILOTS" --machine "$MACHINE" --python "$TASK_PYTHON" --output "$TASK_OUTPUT" --cache "$TASK_CACHE" --dry-run
"$TASK_PYTHON" experiments/run.py --plan "$TASK_PILOTS" --machine "$MACHINE" --python "$TASK_PYTHON" --output "$TASK_OUTPUT" --cache "$TASK_CACHE"
```

Gazebo uses the existing simulation ROS shell (`GATO_ROS_PROFILE=simulation`, `ROS_DOMAIN_ID=1`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`). The user selected **report-only physical safety**: joint/velocity/collision/proposed-torque violations stay visible but do not stop the motion. There is no torque clamp. Nonfinite data, stale state/trajectory, monitor/execution errors and handback remain enforced. This mode is refused outside Gazebo simulation. The plan cannot enable it; the execution command must carry the explicit dangerous option:

```bash
"$TASK_PYTHON" experiments/run.py --plan "$TASK_PILOTS" --machine rtx5060 --python "$TASK_PYTHON" --output "$TASK_OUTPUT" --cache "$TASK_CACHE" --dangerously-disable-safety-stops --dry-run
"$TASK_PYTHON" experiments/run.py --plan "$TASK_PILOTS" --machine rtx5060 --python "$TASK_PYTHON" --output "$TASK_OUTPUT" --cache "$TASK_CACHE" --dangerously-disable-safety-stops
```

After pilot review, use the same command with `--plan "$TASK_PLAN"` (keep the dangerous option for the selected Gazebo report-only cohort). Completed pilot cells are reused. Completed task/physical failures remain scientific evidence and independent cells continue. Canonical task or shared execution/setup failures hold dependent work; physical-only canonical violations are recorded without holding the report-only family. `--retry-failed` is an explicit new attempt, not a way to repeat scientific negative outcomes.

Collected data stays beneath `TASK_OUTPUT`. Regenerate the compact index and create a raw archive after runs:

```bash
"$TASK_PYTHON" experiments/run.py --collect-only --output "$TASK_OUTPUT"
"$TASK_PYTHON" -c 'import sys; from pathlib import Path; sys.path.insert(0,"experiments"); from collect import export_results; export_results(Path(sys.argv[1]),Path(sys.argv[2]))' "$TASK_OUTPUT" "$PWD/.experiment-work/task-parameters-20261014-$SETTING.tar.gz"
```

Publish the runner branch and the exact collector/live source refs listed in the main handoff before target execution. `verify_sources` fetches missing immutable source refs. No remote publication was performed here.
