# GTX1070 F05 N128 stage profile extension

Explicit three-worker successor, separate from historical F05 results:
N128/B1/SQP5/PCG100, N128/B1/SQP1/PCG100, and a fresh N32/B1/SQP5/PCG100
anchor. Each runs one excluded warmup and one measured call, using a fresh
solver owner and explicit dual/rho resets for both calls. Internal modeA;
stage timing, telemetry and resource-query flagsON. No sustained run or extra
task episode. The relevant warmup is the short execution check.

The unchanged runtime is `852b0cbcfe6e56fe5337a3b6e5fe8e777f959b77`, matching
the previous F05 cohort. Only runner/adapter/plan files change. Compile the
existing stage-event instrumentation without solver, plant or generated edits.
Builds share one N32/N128 recipe with two CPU jobs; GPU workers run sequentially.

N32 retains the original F01c frozen bank and candidate0. N128 explicitly uses
retained F01ab `full24` first reaching episode start0/goal0 and its unrelated
warmup. The existing F01ab `cold_seed` primitive makes its holding/zero-control
N128 input. No MPC episode, captured warm start, seed filtering or tuning occurs.
Both banks and their hashes are frozen by `experiments/f05_gtx1070_n128_stage.json`.
This is a stage-window comparison with different N32/N128 input constructions;
it is not a strict matched-input horizon sweep. Effective reaching weights are
retained, including qd0.1 and the frozen pose-task ctrl_limit0.05 override.

Each call records actual SQP/PCG work, stop metadata, native host-chrono timing,
public solve-call wall timing and existing stream-elapsed stage windows. The
legacy literal-zero `pcg_times_us` field is unavailable as a timing measurement.
Skipped stage windows retain null/not_reached. Main per-SQP groups are merit,
PCG, KKT, Schur and other: stage totals divided by actual SQP iterations; other
is native total minus those four groups, divided by actual SQP. Native-minus-all
event windows is separately labeled unattributed, without clipping. Event
windows include stream idle gaps and are not pure kernel-active timings.

```sh
cd /workspace/GATO-f05-n128-stage-profile
/opt/gato-venv/bin/python experiments/resource_profiles_plan.py --gtx1070-stage-extension
/opt/gato-venv/bin/python experiments/run.py --plan experiments/f05_gtx1070_n128_stage.json --machine gtx1070 --python /opt/gato-venv/bin/python --cache /workspace/GATO-f05-n128-stage-profile/.experiment-cache/f05-n128 --output /workspace/GATO/_expr_results/f05-gtx1070-n128-stage-20261014 --dry-run
/opt/gato-venv/bin/python experiments/run.py --plan experiments/f05_gtx1070_n128_stage.json --machine gtx1070 --python /opt/gato-venv/bin/python --cache /workspace/GATO-f05-n128-stage-profile/.experiment-cache/f05-n128 --output /workspace/GATO/_expr_results/f05-gtx1070-n128-stage-20261014
```

The plan contains exact CMake configure/build commands, source pin and declared
module artifacts. One attempt only; no retries. Preserve failures. Export raw
measurements below `_expr_results/`, plus a separate backup; compact tables,
commands and self-contained report/PDF belong under
`thesis_results/f05-gtx1070-n128-stage-2026-10-14/`. Preserve the existing
`stage_windows.csv` column schema in this separate cohort.
