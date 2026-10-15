# Fixed-update TIAGo N × B matrix

One driver, no legacy queue. The committed `nb_cells.json` is identical across
GTX1070/RTX5060/Jetson/H100: N16/32/64/128/256/512 × B1/2/4/8/16, prediction
spacing30 ms (`T=.03*(N-1)`), SQP1, existing TIAGo tracking PCG120 defaults,
300 measured updates/repeat, two repeats, one excluded single-solve initialization.
No slow-solve cutoff; worker timeout180s. Control stays10ms, RK4max1ms; the
reference offsets are0.01…3.00s for every repeat. States and actual PCG work
can vary. All lanes get the previous unshifted batch0 plan; this does not assert
identical solver outputs. With this tool's horizon-time option, running/limit
weights scale0.03/0.008=3.75 while terminal position remains80. It is separate
from F01ab/F01c and the earlier task-parameter study.

The source is based on authorized a17 conditional PCG shared-memory opt-in.
Resource-only probes compile/query all six N first, with zero kernel launches;
only fitting numerical modules are built. GTX has no N256/N512 hardware capacity;
the target probes, not a hardware assumption, determine the retained SMEM cells.
No null operands are launched. N32/B1 and N512/B1 are the first original grid
cells, reused as smoke observations instead of extra measurements.

The driver requires clean measurement sources/exact committed cells. Publish the
complete reviewed branch and notify Felix/H100 agent of its exact commit before
any target build/probe/measurement. Root owns target idle and Jetson power-mode
observations. There is no native/task/readiness solve beyond each worker's
single initialization plus its300 planned measured updates.

## Commands

Supply absolute paths for raw, shared per-GPU build cache and compact export.
The raw root may contain `_host_conditions/` collected before launch. Optional
`--power-mode-file PATH` copies an existing host log there without changing mode.
The dry-run is CPU-only and writes nothing:

```bash
python3 examples/solve_time_heatmap/nb_matrix.py --only nb-matrix --gpu gtx1070 --raw /workspace/GATO/_expr_results/nb-matrix-gtx1070-2026-10-15 --cache /workspace/GATO-tiago-nb-matrix/test-artifacts/nb-cache-gtx1070 --export /workspace/GATO/thesis_results/nb-matrix-gtx1070-2026-10-15 --dry-run
```

After publication and root go, remove `--dry-run`. GTX starts with `--smoke-only`;
then use `--resume` for the same command to finish all other cells without
repeating the two smoke cells. Targets use `--gpu rtx5060`, `--gpu jetson` or
`--gpu h100` with matching distinct directories; native architectures are61/120/87/90.
All use the same committed cells file and per-cell arguments. Do not add local
budget overrides or retry failed repeats. A single resource-compatible cell's
scientific failure does not alter independent cases.

The timeout is per repeat, including imports and initialization. Each attempted
repeat runs once; interrupted/failed folders are retained and never recycled by
the driver. No silent fallback or retry-until-success. Hard-timeout measured
JSONL partials stay raw even when the last NPZ checkpoint is older. Incremental
CSV updates do not plot between workers; plots/report/PDF run only at the final
or smoke-only boundary.

## Storage, duration and interpretation

No new GPU measurements are claimed during preparation. Fixed300 updates cost
90s/cell for two repeats at150ms/solve, plus Python integration/recording/imports.
The strict worst-case timeout budget is3h for60workers, excluding builds. Source
inspection predicts10/5/0/0 resource-rejected cells onGTX/RTX/Jetson/H100; probes
must confirm. Build time is uncalibrated (allow tens of minutes on Jetson for
six resource probes plus supported modules). Allow2GiBraw/export space and
5GiBbuild/cache until observed; full plans are disabled. No budget/sample pruning.

`cells.csv` has exactly the requested30-row schema. Status is completed, SMEM,
nonfinite or incomplete. Only two completed300-update repeats with600 retained
finite timing samples receive headline timing/SQP/PCG aggregates. Failed cells'
partial timing means stay labelled in `repeat_status.csv`; they are never pooled
into completed cells. `samples` counts retained measured JSONL rows even when
the last timed-out worker checkpoint reports fewer. Population SD is over the
600 individual timings, not uncertainty across independent experiments.
PCG cap share uses all actual lane-SQP slots, including zeros; missing telemetry
stays empty. A completed budget is not nonlinear convergence, physical/task
qualification or a guaranteed control rate. Reciprocal native time is solve-only
frequency, not batch-member throughput or end-to-end control frequency.

CPU regeneration after collection:

```bash
python3 examples/solve_time_heatmap/nb_export.py --raw /workspace/GATO/_expr_results/nb-matrix-gtx1070-2026-10-15 --output /workspace/GATO/thesis_results/nb-matrix-gtx1070-2026-10-15
```

The compact export includes cells/repeat-status CSVs, exact source/module/probe
provenance, editable quick-look exporter and self-contained REPORT.md/PDF.
Archive raw output separately and keep a backup; preserve all failures and the
common source/cell argument identities for cross-GPU comparisons.
