# J1 conditional20 predictive filter — simulation runner prepared, not run

Status: **PREPARED_NOT_RUN**. This is an experiment-only functional simulation, not a physical robot controller or a production/solver change. No native/GPU run or Jetson launch occurred during preparation. Jetson is reserved for a separate QDLDL job; root/sole-owner handoff must precede any GPU launch.

The assignment contains **all20 original J1 physical non-passes**:16 finite unsafe episodes and4 original execution errors. Original labels remain attached to every new row; none is dropped or reclassified. Execution checks case502 first, then the other19 in original assignment order, without rerunning502. The original 100-case J1 scope remains historical; this conditional20 cannot become a new100-case reliability estimate. Original unfiltered histories are reused as the matched reference, not rerun unnecessarily.

The frozen J1 settings remain B1, SQP5/PCG120/natural zero-PCG exit, N32/.03s knots, 8s maximum, modeB/solve_time, orientationoff, position<=5cm/speedL1<=1 arrival and finalapproachoff. Original cold initialization, first recorded/applied solve, temporal warm shift, once-only dual/rho reset and persistent state are unchanged. Numerical source85ef/tree-equivalentfd3 and retained q020 sm87 module remain unchanged.

Every applied RK4 segment, including the unpaid bootstrap and the **old/stale active plan during charged native wrapper time**, passes through the CPU torque filter. Proposed and applied controls are both saved aligned in `executed.npz`; `filter_samples.jsonl` records every forecast, modification, guard/recovery flag, candidate outcome and wall time. Forecast uses zero external forces, checked at setup, matching actual plant force; no disturbance/force-safety claim.

Filter guards remain physical q inset.02rad, speed minus.10rad/s, effort95%. A proposed constant command is forecast30ms with max3ms RK4 samples. Fixed coupled RNEA/project/brake/blend candidates are forward validated. The phase2 helper tests the **same** candidates in normalized torque-distance order and stops at the first passing candidate, preserving the closest-passing result. On58 saved phase1 anchors, status/candidate/command match within1e-6; old phase1 source/results are preserved. Desktop phase2 median/p95 wall:3.82/5.87ms,44/58 above nominal3ms, descriptive only.

`no_candidate_certified` is a fixed-search outcome, not a proof of infeasibility. A consumed guard can yield recovery-only, not a guard certificate. No returned command means no corresponding plant step: the new row is an **incomplete filter-stop diagnostic**, physical_pass=false, observed-prefix flag retained separately. `failure_record_count_including_filter_stops`/`execution_errors` includes those records; `numerical_errors`, `infrastructure_errors` and `incomplete_filter_stops` distinguish causes and are not summed as disjoint with the total. Individual numerical/filter failures remain in the20 denominator and independent cases continue. Unexpected canonical numerical/nonfinite or shared setup failure holds dependent stages; finite scientific timeout/physical negatives and filter search stops retain negative labels but permit diagnostics. Remaining19 requires a terminal first1 stage with no shared hold.

## Timing scope

This explicitly models an **independent high-rate execution safety loop** with access to each simulated state. Native BSQP.solve wrapper time still advances the old plan exactly as in original modeB. Filter CPU wall is measured separately and **not added** to that clock; overlap is an architecture assumption, not proof that filtering is free or fully computation-paid. The native binding retains the Python GIL, so the runner does not claim that a same-process Python thread works concurrently. No actual concurrent scheduling/contention or Jetson real-time feasibility is established. Nominal3ms overruns are descriptive, never a deadline acceptance gate. Prediction is sampled/model-based, not robust/continuous-time/collision/physical-robot safety.

## CPU evidence and GTX handoff

`check_filtered_loop.py` uses fake native returns but executes the actual loop and verifies bootstrap/stale-plan filtering, applied filtered controls, once-only resets, no command=>no plant step, incomplete/error classification, retained native work, and canonical filter-stop allowed versus unexpected numerical hold. `check_cpu.py` supplies the coupled/gravity functional checks. No new native regression or source repair.

Exact GTX commands/CPU dry-runs are retained under `/workspace/GATO/_expr_results/random-reaching-predictive-filter-20261015/filtered20-preparation/GTX/preparation-final/launch_commands.json`. They request only the matching8s canonical and case502, using the exact retained desktop d9bee module. Sole GPU owner executes only after root review/handoff. No other experiment is launched by these scripts.

## Portable Jetson install and manual commands

Preparation produces a compact Git bundle whose prerequisite is the already installed frontend72533b2. Transfer that bundle using the existing desktop/laptop/Jetson file-copy route to `/workspace/GATO-filtered20-source.bundle` **inside the container**. Source install is independent of GPU execution:

```sh
git -C /workspace/GATO-jetson-fresh100 fetch /workspace/GATO-filtered20-source.bundle refs/heads/experiment/jetson-filtered20-simulation:refs/heads/experiment/jetson-filtered20-simulation
git -C /workspace/GATO-jetson-fresh100 worktree add /workspace/GATO-jetson-filtered20-simulation experiment/jetson-filtered20-simulation
cd /workspace/GATO-jetson-filtered20-simulation
python experiments/predictive_limit_filter/prepare_filtered.py \
  --platform Jetson \
  --module /workspace/GATO-jetson-fresh100/.experiment-cache/fresh100-native-sm87/modules/bsqpN32_tiago_right.cpython-310-aarch64-linux-gnu.so \
  --module-receipt /workspace/GATO-jetson-fresh100/_expr_results/random-reaching-jetson-fresh100-20261015/build_receipts/jetson.json \
  --output /workspace/GATO-jetson-filtered20-simulation/_expr_results/filtered20/preparation
```

This reuses exact ARM/sm87 module SHA256 `3ba05f3cafe4038c88fbdc77acd891dbe99899cf1db6d2bcde9de6984312035a`, no build required. Target protocols/commands regenerate locally, so no desktop-only paths enter the Jetson launch. The frozen portable input bundle is already part of the prerequisite frontend. Actual runs reject missing/wrong module or receipt. Host power/clock checks remain waived;40W is user-confirmed, no query/change.

**Only after QDLDL release and root handoff**, define the manual stage launcher below. Each invocation runs only the named stage; no automatic optional-study chain:

```sh
run_filter_stage () {
python - /workspace/GATO-jetson-filtered20-simulation/_expr_results/filtered20/preparation/launch_commands.json "$1" <<'PY'
import json,subprocess,sys
commands=json.load(open(sys.argv[1]))
entry=next(c for c in commands if c['stage']==sys.argv[2])
subprocess.run(entry['argv'],cwd=entry['cwd'],check=True)
PY
}
run_filter_stage canonical
run_filter_stage first1
```

Read those saved diagnostics and confirm setup before the remaining stage:

```sh
run_filter_stage remaining19
```

The launcher/worker refuses missing, changed or shared-failed predecessors; it never relabels a failed canonical as passed. `launch_receipt.json`, provenance/protocol, every per-case metric, original labels, full executed history, filter records, solve/work logs and `cohort_status.json` are written per stage. Setup/import failures preserve `setup_failure.json`. Preserve all outputs/partial stages in a raw archive and separate backup; do not overwrite or retry until green.

Original J1 selected20 episode-wall sum was212.30s for119.90s recorded simulated motion/14,155 solves. The new policy changes motion/stopping and adds substantial CPU/I/O work. Provision **15–45min and up to1GiB raw plus a separate backup** for Jetson20, a planning allowance rather than a bound. GTX canonical+502 is a small minutes-scale smoke; its observed timing/storage will refine estimates without shortening8s or dropping cases. Raw arrays/per-filter/per-call records stay outside thesis sync. A final outcome report requires actual recorded results and root interpretation; none is claimed now.
