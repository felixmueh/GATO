# Jetson N128 B1/B16 stage timing — two missing counterparts

Exactly two timing-only workers: N128, B1/B16, SQP5, PCG100, dt0.03 s,
internal mode A. Each uses one excluded unrelated warmup and one measured call.
Every call creates a fresh solver and explicitly resets dual and rho before
the timed public solve. Four total native solves, no controller/plant/ROS motion,
resource-query solve, task preflight or extra readiness solve. One attempt each;
retain failures and continue the independent other worker without automatic retry.

Runtime is unchanged `852b0cbcfe6e56fe5337a3b6e5fe8e777f959b77`. The exact
GTX1070/H100 F01ab N128 bank is reused: arrays SHA256
`a674b748a058538fe7c6b0e93a836a0b82394196028788f09284d29b5dd13ed4`.
B1 uses candidate0; B16 repeats the identical input in all16 lanes. This is
timing-only batching and does not compare diverse candidate quality.

The reviewed H100 resource-profile adapter is reused without modification.
Build only N128 with sm87, telemetry and Stage ON. Existing N32 modules are
insufficient. Both cases share the identical native build recipe/module.
Stage windows are stream elapsed time, not exclusive kernel execution.
Native `pcg_times_us` literal zeros remain unavailable; use the `pcg` stage
window. Named-stage sums divided by actual SQP count match the existing
GTX/H100 comparison schema; preserve not-reached slots and actual PCG/SQP work.
Single measured solves do not estimate variability or thermal steady state.

Preparation contains no build or GPU execution. Planning allowance: 5–20 minutes
for the fresh target N128 build (uncalibrated), then under one minute for the two
workers; allow up to 2 GiB source/build space and 50 MiB raw/archive space.
Actual duration/storage must be recorded, not relabeled as these estimates.
Use the existing power mode; record host nvpmodel/tegrastats before and after.
Check exclusive idle GPU before execution. No power-mode changes are requested.

## Git publication (credentialed host, not target container)

```bash
git push git@github.com:felixmueh/GATO.git experiment/jetson-stage-timing-n128:refs/heads/experiment/jetson-stage-timing-n128 852b0cbcfe6e56fe5337a3b6e5fe8e777f959b77:refs/heads/experiment/deployment-source-852b0cbcfe6e
git ls-remote git@github.com:felixmueh/GATO.git refs/heads/experiment/jetson-stage-timing-n128 refs/heads/experiment/deployment-source-852b0cbcfe6e
```

Use this self-contained branch, not just the JSON in a legacy adapter checkout.
The H100 parent carries the N128/B16 adapter support and frozen bank.
Root supplies exact target Git/host observation commands. Within the selected
Jetson checkout and active Python environment, prepare only:

```bash
STAGE_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
"$STAGE_PYTHON" -c 'import numpy,pinocchio; import json,sys; sys.path.insert(0,"experiments"); from machine_launch import verify_sources; verify_sources(json.load(open("experiments/launch/jetson_stage_timing_n128/jetson.json")))'
"$STAGE_PYTHON" experiments/run.py --machine jetson --plan experiments/launch/jetson_stage_timing_n128/jetson.json --python "$STAGE_PYTHON" --cache .experiment-cache/jetson-stage-timing-n128-20261014 --output .experiment-results/jetson-stage-timing-n128-20261014 --dry-run
```

After root reviews setup and the existing power-mode/idle observations:

```bash
STAGE_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
"$STAGE_PYTHON" experiments/run.py --machine jetson --plan experiments/launch/jetson_stage_timing_n128/jetson.json --python "$STAGE_PYTHON" --cache .experiment-cache/jetson-stage-timing-n128-20261014 --output .experiment-results/jetson-stage-timing-n128-20261014
"$STAGE_PYTHON" experiments/collect.py --output .experiment-results/jetson-stage-timing-n128-20261014 --export .experiment-work/jetson-stage-timing-n128-20261014-results.tar.gz
```

Existing passed attempts are skipped automatically. Do not add retry-failed.
Root collects the archive into `_expr_results` and runs the prepared CPU exporter
after transfer. Final report destination:
`thesis_results/jetson-stage-timing-n128-2026-10-14/`.
