# Additional large-horizon studies — prepared, execution pending

These separate plans add N256 on RTX5060 internal/Gazebo and N256/N512 on Jetson. Collector/live source: `f6cb96f1aa0c014ffc221b1b81adb2beebbb9564`; internal source: `a17e3e631e52e927bb6b81675b9a8f958d473d3d`. GTX1070 remains at N16–128. PCG120, task settings, SQP/dt grids and the N32 repetition count are unchanged. Base plans, pins, caches and completed results remain valid; no completed N16–128 campaign is repeated.

| Setting | New horizons | Study A | New Study B cells | New episodes |
|---|---|---:|---:|---:|
| RTX5060 internal | 256 | 9 | 10 | 19 |
| RTX5060 Gazebo | 256 | 6 | 5 | 11 |
| Jetson internal | 256, 512 | 18 | 20 | 38 |

68 new scientific episodes; 361 combined with the 293 base episodes if all supported cells execute. Study B's 30 ms column reuses this extension's Study A SQP5 tracking cells. Capacity checks and single-solve readiness are separate observations. Pilots are identical science cells and resume from the same output folder.

The isolated test sources enable shared-memory opt-in only when the PCG request exceeds the default allowance and fits the actual device/kernel limits. The numerical PCG body is unchanged. GTX cannot fit either new N; RTX N512 cannot fit its block limit. Each task build contains N32 plus exactly one new N, so a failed N512 build cannot block N256 or the original small-N builds. Canonical groups are separate per new build; N32 canonical availability is preserved. Changed source/builds need fresh bounded readiness; no native or task qualification is claimed during preparation.

Felix supplied unchanged-source N256 resource rejections from both RTX and Jetson (runner 7572137, source fa6, 76,516 total shared bytes against a configured 49,152 allowance). Their raw archives have not been imported here. Preserve those observations and **repeat N256 capacity under the opt-in successor** as a separate cohort; Jetson N512 capacity is new.

Use the existing target environment and its active Python. For Gazebo, use the existing simulation ROS shell with domain 1, CycloneDDS and `GATO_ROS_PROFILE=simulation`. First pull the published runner:

```bash
git pull --ff-only origin experiment/simple-runner
LARGE_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
LARGE_SETTING=jetson # alternatively rtx5060_internal or rtx5060_gazebo
LARGE_MACHINE=jetson # rtx5060 for either RTX setting
LARGE_ROOT="experiments/launch/task_parameters_large"
LARGE_OUTPUT="$PWD/.experiment-results/task-parameters-large-optin-20261014-$LARGE_SETTING"
LARGE_CACHE="$PWD/.experiment-cache/task-parameters-large-optin-20261014-$LARGE_SETTING"
"$LARGE_PYTHON" -c 'import json,sys; from pathlib import Path; sys.path.insert(0,"experiments"); from machine_launch import verify_sources; verify_sources(json.loads(Path(sys.argv[1]).read_text()))' "$LARGE_ROOT/$LARGE_SETTING.json"
```

1. Follow the separate [opt-in capacity commands](../task_parameter_capacity_optin/README.md) for N256 on that target.
2. Run the native readiness observation. It executes one cold public B1/SQP1/PCG120/30 ms solve using frozen canonical state/goal; no integration, ROS or motion. Inspect import, finite packed output, timing and provenance before the task pilot:

```bash
"$LARGE_PYTHON" experiments/run.py --plan "$LARGE_ROOT/${LARGE_SETTING}_native_readiness.json" --machine "$LARGE_MACHINE" --python "$LARGE_PYTHON" --output "$LARGE_OUTPUT" --cache "$LARGE_CACHE" --only native-N256 --dry-run
"$LARGE_PYTHON" experiments/run.py --plan "$LARGE_ROOT/${LARGE_SETTING}_native_readiness.json" --machine "$LARGE_MACHINE" --python "$LARGE_PYTHON" --output "$LARGE_OUTPUT" --cache "$LARGE_CACHE" --only native-N256
```

3. Run the same-cell SQP5 task pilot after readiness. It also performs the new build's canonical N32 family check. Internal pilot:

```bash
"$LARGE_PYTHON" experiments/run.py --plan "$LARGE_ROOT/${LARGE_SETTING}_pilots.json" --machine "$LARGE_MACHINE" --python "$LARGE_PYTHON" --output "$LARGE_OUTPUT" --cache "$LARGE_CACHE" --only A-N256-SQP5 --dry-run
"$LARGE_PYTHON" experiments/run.py --plan "$LARGE_ROOT/${LARGE_SETTING}_pilots.json" --machine "$LARGE_MACHINE" --python "$LARGE_PYTHON" --output "$LARGE_OUTPUT" --cache "$LARGE_CACHE" --only A-N256-SQP5
```

For the selected Gazebo report-only pilot, add `--dangerously-disable-safety-stops` to both pilot commands. The option is unnecessary for the native readiness observation. Physical/collision/proposed-torque negatives remain visible without stopping Gazebo; finite/stale/execution checks and handback remain enforced. Plans cannot enable this policy automatically.

On Jetson, review N256 capacity, finite execution and measured pilot cost before trying N512. Build/launch/OOM or unreasonable measured cost can justify stopping; preserve the evidence and record N512 skipped with a reason. A bad reach or isolated numerical failure alone is not a hardware ceiling. If N256 is sensible, repeat capacity→native readiness→pilot with N512. There is no adaptive runner or silent budget pruning.

After readiness/pilot wall-cost review, run the main extension plan with the same output/cache; completed pilots are reused. To limit Jetson to N256 after a justified ceiling, use existing `--only` with `A-N256-SQP1 A-N256-SQP2 A-N256-SQP5 B-N256-dt10ms B-N256-dt15ms B-N256-dt20ms B-N256-dt45ms B-N256-dt60ms`. Run the complete plan without `--only` only after both N values are cleared. Keep the explicit dangerous option for Gazebo main execution. Preserve scientific task/physical negatives; module/import/nonfinite or shared readiness failures hold dependent cells. No retry-until-green.

Source refs must be published before target execution; `verify_sources` fetches missing immutable refs. Original F01 timing, F04/F05, N32 random-reaching and all historical receipts remain untouched. The active large-N consumers are these extension/readiness plans and the separate opt-in capacity plans.

Publish from a host with the existing working SSH setup (these commands were not executed during preparation):

```bash
git push origin experiment/simple-runner
git push origin a17e3e631e52e927bb6b81675b9a8f958d473d3d:refs/heads/experiment/deployment-source-a17e3e631e52
git push origin f6cb96f1aa0c014ffc221b1b81adb2beebbb9564:refs/heads/experiment/deployment-source-f6cb96f1aa0c
```

The source owner's local resource-only validation and the unchanged-source remote receipts are recorded in the [current machine handoff](/workspace/GATO/felixnotes/HANDOFF_REMOTE_MACHINE_RUNS.md). Local resource checks do not qualify supported-device numerical solves or task behavior.
