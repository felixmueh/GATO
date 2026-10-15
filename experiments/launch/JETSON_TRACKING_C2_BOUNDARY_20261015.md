# Jetson continuation tracking recheck — prepared only

Felix explicitly selected three boundary cases: the original N128/N256/N512
6-second tracking episodes. This preparation does not execute GPU tasks or alter
`commands_felix.md`; the overnight QDLDL matrix remains separate and unchanged.

## Frozen sources and scope

- Published native source: `c15303b2dd68529376487cf6a5465807b9205b38`.
- Side-source alias: `experiment/deployment-source-c15303b2dd68`.
- Native tree: `1dc500c821aded75252453fdcef95d0463ac7c7c`.
- Local native branch/worktree: `experiment/jetson-tracking-c2`,
  `/workspace/GATO-jetson-tracking-c2`; local metadata commit
  `f868fc21f4a3c22fbda98d38dc69e184046df5b5` has the same tree.
- Exact native parent: `a17e3e631e52e927bb6b81675b9a8f958d473d3d`.
  The only changed file is `gato/dynamics/tiago_right/tiago_right_plant.cuh`,
  exactly the reviewed helper patch from
  `f0cafcde47a675d0deb17724005c3416a327e7db` (38 additions/31 deletions).
  Patch SHA256: `9c2896e86ca81a30ab429bf9e0d3b004eb5196cdb6fd38e9e41a883e8eb241eb`.
- Runner preparation starts from published
  `c7dbe0915da03729336a3345ffa7cfdaf41c0d5d`, with no adapter/build changes.

This uses the complete reviewed 1%-interval C2 quadratic continuation, not just
the historical17100c0 width-only commit. There are no interior cost margins,
new weights, controller/plant integration changes or other solver changes.
Conditional native PCG shared-memory opt-in remains exactly a17. At N128 it is
inactive, preserving the fa6 numerical kernel configuration. The scalar helpers
serve q/velocity/effort soft costs at their original raw bounds; this does not
make them hard physical constraints or guarantee successful tracking.

## Exact three-case protocol

Plan: `experiments/launch/jetson_tracking_c2_boundary_20261015.json`.
Cases are `C2-N128-SQP5`, `C2-N256-SQP5`, `C2-N512-SQP5`, each
`figure8-6s`, one planned attempt. B1/SQP5/PCG120, .03s knots and nominal feedback,
.003s maximum RK4 plant steps, six-second full period and25/50mm RMS/maximum
tracking gates remain frozen. Mode B charges BSQP.solve wrapper wall time while
the previous active plan continues. Other recorder/controller/plant work is
excluded from charged time. Native timings remain separate and deadline excess
is descriptive. No state sampling option or fixed300 timing-matrix protocol is
introduced.

Tracking q/qd/u costs remain4/.01/.0001, terminal80, orientation0/0, limit costs
.01/.001/.003, rho.01, PCG/KKT tolerances.001. Cold repeated current-state/zero
control initialization, shifted previous plans, persistent dual/rho with adaptive
rho, original reference phase and full-preview coverage remain unchanged.
Original reference hashes are asserted by CPU tests; preview is3.81/7.65/15.33s.
Task and physical gates remain separate, including raw URDF physical limits.

N128 retains collector `90bdb825d059e22b709e2e53a785d7d034aad1b0`.
N256/N512 retain `f6cb96f1aa0c014ffc221b1b81adb2beebbb9564`.
Both already contain the same ordinary finite-event guard. The newer resolver
adds N256/N512 support; its optional native-readiness entry is not invoked.

One identical shared build contains N32/128/256/512 and the original access-smoke
executable. One new-source N32 canonical uses the f6 collector for all three
cases; CPU resolution proves its canonical protocol equals the90b version.
It is an explicit shared preflight, not old binary evidence or a new scientific
pilot. A canonical task/physical failure remains failed and holds dependent
cases; there is no diagnostic waiver. Shared build/import/setup failures also
hold dependent work. Independent task execution failures retain their events
and allow the other selected cases to proceed. No retries, fallback to1ms,
expanded N512 grid, tuning or baseline reruns are included.

## Retained original comparators

| N | Original runtime / collector | Original outcome | Median wrapper/native ms |
| --- | --- | --- | --- |
|128|fa6f6be /90bdb825|RMS25.429mm; speed excess2.207rad/s, task/physical false|30.012/29.375|
|256|a17e3e6 /f6cb96f|RMS37.649mm; speed excess11.694rad/s, task/physical false|37.183/36.564|
|512|a17e3e6 /f6cb96f|nonfinite state_after at4.677729s/57solves; RMS unavailable|77.799/77.083|

These saved episodes are reused only as historical comparators, with their
original labels and timings. They are not fresh paired controls or proof that
compute latency alone caused failure. Changed helper arithmetic may also change
compute time and the Mode-B plant trajectory. The other18unrun N512 cells remain
held. Existing C2 tracking regression evidence uses a different fixture/interior
cost policy and does not qualify these Jetson episodes.

## Preparation checks and pending target validation

CPU-only checks: two focused plan/protocol contracts pass; exactly3pending
scientific tasks in runner dry-run. Each resolved protocol was compared with
its original saved JSON:27effective fields and the complete packed reference
match, with exact original reference hashes. Shared build/preflight signatures,
original per-case collectors, costs, safety policy and input/build hashes are
checked. Source audit proves the exact helper-only diff and published tree.
Logs: runner worktree `.experiment-cache/jetson-tracking-c2-preparation/`.

Compile-only checks passed for N32 and representative N512 extension modules
and the access-smoke executable, using explicit sm87 on x86_64 and the exact
native tree. Neither the executable nor either extension was run.
No CUDA binary, task suite or benchmark is executed here. Required target GPU
task regressions and the new-source canonical remain pending GPU coordination;
CPU/build checks are not task/physical/convergence qualification. The actual
Jetson command builds all four horizon modules with its ARM64 toolchain.

## Target command strategy and estimates

Root publishes the exact runner successor and writes the separate user block.
From that dedicated runner checkout, after fetching the published side-source
alias (and retaining original collector objects), inspect:

```sh
python3 experiments/run.py --machine jetson \
  --plan experiments/launch/jetson_tracking_c2_boundary_20261015.json \
  --output .experiment-results/jetson-tracking-c2-boundary-20261015 \
  --cache .experiment-cache/jetson-tracking-c2-boundary-20261015 --dry-run
```

The same arguments without `--dry-run` are the future launch; no such command
was executed here. Use a fresh raw/cache cohort, preserve failed receipts, and
resume without `--retry-failed`. After execution, existing `experiments/collect.py`
can index/export the entire result root including canonical/failed/partial data.
Keep an independent raw/archive backup; finalized exports belong in thesis_results.

Unmeasured planning allowance: originals took18.46/26.14/22.56s per receipt and
occupied6.82/6.60/4.81MB raw. The scientific trio might take~1–3minutes plus
canonical/build/export, but C2 behavior/time is unknown; allow~10–30minutes overall
and1GB raw/return space conservatively, excluding build caches and backup copies.
Original per-case hard timeout1800s is retained; this estimate is not a runtime
bound. Review target free space/build conditions before running. No budget
pruning or timeout changes are authorized by this preparation.
