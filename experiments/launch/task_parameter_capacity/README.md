# N256/N512 capacity checks — prepared, not executed

The 293 task episodes remain at N16/32/64/128. A complete extension would contain 429 episodes (120 per internal GPU, 69 Gazebo), but it is not queued: the unchanged PCG launch has a concrete shared-memory boundary. These optional checks use the existing `examples/solve_time_heatmap/resources.cu`, pinned to `fa6f6be`, with isolated single-N builds. Failed large builds cannot block the smaller task builds. No solver repair, shared-memory opt-in change, task motion or ROS action is included.

For TIAGo float, PCG requests `4*(5*14*(N+2)+32+5+1024)` dynamic bytes per block, with 1024 threads. N128:40,644 bytes; N256:76,484; N512:148,164. Compiled static shared memory must be added (32 bytes in the historical records). Current PCG has no function-specific opt-in above 48 KiB. Generated GRiD kernel opt-ins configure different kernels. Pascal's block limit is 48 KiB; NVIDIA lists opt-in block limits of 163 KiB for sm87 and 99 KiB for sm120. These are source/specification predictions for current unrun targets. N512 therefore also exceeds the RTX block limit even with opt-in. B1 solver/binding device buffers are roughly 2 MiB at N256/4 MiB at N512, excluding CUDA context, plant constants, host data and compilation; global memory is a separate limit.

The existing **RTX5060**, N256 receipt at `/workspace/GATO/test-artifacts/tiago-matrix-port-validation-20260922/resource-N256.json` reports 76,484 dynamic+32 static bytes, a configured 49,152-byte limit, `fits=false` and `pcg_launch_error="invalid argument"`. SHA256:`1415a94c48ea550fd729a2b92e0d40b8288d933cb4ba949e89fab1cac9805672`. The R01 review in `/workspace/GATO-tiago-r01-review/research/r01_review/` preserves this observation. It is not a GTX measurement or a current whole-module qualification. Current resource allocation/thread/static-scalar declarations match that boundary; zero-PCG/telemetry source changes mean the entire PCG file is not identical. Prefer this retained resource evidence over a redundant RTX check unless a current compiled observation is needed. GTX/Jetson large-N launch outcomes remain unmeasured.

On a target after the GPU execution owner releases it, pull the published runner and select its existing Python. Preparation did not perform these commands:

```bash
git pull --ff-only origin experiment/simple-runner
CAPACITY_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
CAPACITY_MACHINE=jetson # alternatively gtx1070 or rtx5060
CAPACITY_PLAN="experiments/launch/task_parameter_capacity/$CAPACITY_MACHINE.json"
CAPACITY_OUTPUT="$PWD/.experiment-results/task-parameter-capacity-20261014-$CAPACITY_MACHINE"
CAPACITY_CACHE="$PWD/.experiment-cache/task-parameter-capacity-20261014-$CAPACITY_MACHINE"
"$CAPACITY_PYTHON" -c 'import json,sys; from pathlib import Path; sys.path.insert(0,"experiments"); from machine_launch import verify_sources; verify_sources(json.loads(Path(sys.argv[1]).read_text()))' "$CAPACITY_PLAN"
"$CAPACITY_PYTHON" experiments/run.py --plan "$CAPACITY_PLAN" --machine "$CAPACITY_MACHINE" --python "$CAPACITY_PYTHON" --output "$CAPACITY_OUTPUT" --cache "$CAPACITY_CACHE" --only capacity-N256 --dry-run
"$CAPACITY_PYTHON" experiments/run.py --plan "$CAPACITY_PLAN" --machine "$CAPACITY_MACHINE" --python "$CAPACITY_PYTHON" --output "$CAPACITY_OUTPUT" --cache "$CAPACITY_CACHE" --only capacity-N256
```

Review the build logs, `metrics.json`, `probe.stdout.json`, `probe.stderr.log` and `probe_returncode.json`. Valid JSON with a known resource rejection is a completed capacity measurement (only exit 0 or 2); signal or other exit codes remain execution failures. The pinned producer normally returns valid JSON with exit 0 for the expected invalid launch. Its unexpected-launch-error exit 2 can leave an unclosed JSON object; preserve that partial stdout and mark execution failure, without repairing it. A build, malformed output or unrelated process failure is an execution failure. Preserve all outcomes. The probe does not execute an optimizer solve; its rejected null-pointer launch never reaches the kernel, so its placeholder one-iteration argument is not a changed PCG120 task budget. If all kernels unexpectedly fit, report that finding before scheduling a finite solve or large task cells.

On Jetson, stop increasing N when N256 demonstrates an operational ceiling (build/OOM/launch rejection or unreasonable measured cost). Write the reason for **N512 skipped**, retaining its planned identity; do not mark it completed. A bad reach or isolated numerical failure alone is not a hardware ceiling. Only if N256 remains sensible, repeat the final two commands with `--only capacity-N512`. No adaptive runner or silent budget pruning is used. The same explicit selection avoids redundant known-impossible GTX/RTX grid executions. RTX capacity evidence concerns the native solver shared by internal/Gazebo; these checks never enter Gazebo.

Primary sources: [NVIDIA Pascal tuning guide](https://docs.nvidia.com/cuda/archive/11.6.0/pascal-tuning-guide/index.html), [CUDA compute-capability resource limits](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html). See the shared reference entry `NVIDIA-SHARED-LIMIT` for historical interpretation and limitations.
