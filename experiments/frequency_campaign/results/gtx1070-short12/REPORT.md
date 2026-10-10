# F01ab: GTX 1070 solve time and realized solver work

The GTX 1070 completed all **64 B1 cells**: N16/32/64/128 × SQP1/2/5/10 ×
PCG100/200/400/800. Each cell used the same frozen 12 short episode definitions
in mode A: computation time was measured but did not advance the plant.
There were **15,360 measured calls** (768 cold, 14,208 warm, 384 goal switch)
and 64 separately retained, excluded process/CUDA warm-ups. No execution failed.
The matching RTX 5060/internal and Jetson results remain pending; this report
makes no cross-GPU comparison.

These are .6-second timing subsets, with two .3-second reaching segments or
.6 seconds of an approved figure-eight reference. Feedback and prediction knot
spacing were 30 ms, with 3 ms plant steps. Prediction duration therefore changes
with N (.45/.93/1.89/3.81 seconds). Each cap cell generates its own trajectory
and subsequent inputs; the timing comparisons do not hold all solve inputs
identical. PCG100 is an explicit matrix setting, distinct from the public preset120.

## Timing observations

The following example holds SQP5/PCG100 fixed. Times are **native milliseconds**;
each cold/warm/switch mean uses 12/222/6 calls respectively. All per-cell/category
native and wrapper mean/median/p95 values are in
[cell_category_summary.csv](cell_category_summary.csv).

| N | Cold mean | Warm mean | Goal-switch mean | Warm p95 | Mean total PCG iterations per warm call |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 1.770 | 1.381 | 1.745 | 1.551 | 21.12 |
| 32 | 2.579 | 1.962 | 2.753 | 2.415 | 37.50 |
| 64 | 4.861 | 4.482 | 5.063 | 6.866 | 161.31 |
| 128 | 7.073 | 7.278 | 8.905 | 11.305 | 210.30 |

Across all 64 cells, native per-cell cold means range from .559 to 46.529 ms;
warm means from .379 to 45.169 ms; goal-switch means from .419 to 98.380 ms.
Cold mean exceeds warm mean in 50/64 cells, so a blanket claim that cold is
always slower is unsupported. For N128/SQP10/PCG800, goal-switch mean is
98.380 ms versus warm mean45.169 ms; warm p95 is111.586 ms. Distributions and
category histograms should be inspected without presuming bimodality.

The 15,360 measured calls total77.678 native seconds and80.074 wrapper-wall
seconds. The median per-call difference is .149 ms. Wrapper time includes the
public solve wrapper; recorder I/O, controller bookkeeping and plant integration
are excluded. These totals are not whole campaign duration. Solve time is the
primary view; reciprocal summaries are **1000/aggregate milliseconds**, not
mean instantaneous Hz or an actuation rate. No deadline pass/fail is imposed.

## Realized work and retained outcomes

Every measured SQP call reports `iteration_cap`: actual SQP work equals the
configured cap, totaling69,120 SQP iterations. The 69,120 recorded PCG stages
report `absolute_relative_rho`50,947 times (73.71%), `initial_absolute_rho`12,197
(17.65%), and `iteration_cap`5,976 (8.65%). Thus a PCG budget is not equivalent
to realized PCG work. These instrumented stop reasons are not a nonlinear
convergence certificate. Breakdown is explicitly **null**: this solver has no
explicit breakdown stop; numerical anomalies were not instrumented.

All 768 episodes and153,600 plant steps remain in the timing cohort, including
730 sampled physical passes and38 physical negatives (N16:31, N64:1, N128:6).
Maximum recorded sampled excesses are1.013661 rad position,4.043923 rad/s speed
and7.034943 Nm torque. All384 rough-reaching subset episodes fail the two-segment
criterion; all384 tracking subset task statuses remain unknown. These short
windows do not qualify or invalidate the separate full-duration canonical
reaching/T01 protocols. No tuning, solver defect cause or viability conclusion
is inferred. Known SB-001/SB-002 and the preserved Q02 negative remain relevant;
no full numerical certificate is claimed.

## Provisional F01c budget proposal

For later discussion, three **unfrozen** N32 budgets span inexpensive, ordinary
and expensive timing/work in this GTX-only data:

| SQP / PCG | Warm native mean / p95 (ms) | Warm wrapper mean (ms) | Reciprocal native mean (Hz) | Mean total PCG iterations |
| --- | ---: | ---: | ---: | ---: |
| 1 / 100 | .540 / .636 | .676 | 1853.31 | 17.38 |
| 5 / 100 | 1.962 / 2.415 | 2.118 | 509.75 | 37.50 |
| 10 / 800 | 4.140 / 10.417 | 4.295 | 241.55 | 111.93 |

This is a proposal, not an approved or queued F01c protocol, a task-quality
ranking or a same-input speedup. The other GPU timing results and Jetson
runtime discussion remain pending. F01c must separately freeze candidate inputs,
B1/2/4/8/16 scaling and identical 16-candidate serial-B1 versus B16 comparison.

## Evidence and plotting

Source: `502fa29c8b0d649dcb347f9534bf29c6935fa6dd`, cohort `F01ab-short12`,
GTX1070/internal/A. The pinned index hash is
`ccda341698b8ca55feb68f846b8c36d6ae869409a331def0e1ad65f21bfa825f`;
bank JSON hash `9c3a3a06b53079af04ca99a6d4146ff741161d1bf82013283447ab2d5422d7b9`.
[receipt.json](/workspace/gato-campaign-recovery-20261010/f01-gtx1070-report/receipt.json) records exact attempt/module/build/source/input
hashes and commands. The independently accepted
[saved-data review](saved_data_review.json) supplies coverage and
physical-outcome counts. This extraction checked metadata and selected solve
NPZ hashes, decoding only the SQP/PCG iteration arrays; it did not repeat the
plotter's complete555,840-file integrity audit or run a solver.

[Primary/reciprocal plots, category histograms and full coverage CSVs](/workspace/gato-campaign-recovery-20261010/analysis/f01ab-gtx1070/)
are generated by the separate saved-data plotter. Plot completion/review belongs
to that output's provenance. Raw attempts remain unchanged. Per-call timing/work
CSV is [samples_timing_work.csv](/workspace/gato-campaign-recovery-20261010/f01-gtx1070-report/samples_timing_work.csv). Reproduce this compact
summary with `/opt/gato-venv/bin/python3 summarize.py` in `/workspace/gato-campaign-recovery-20261010/f01-gtx1070-report/`;
this is saved-data analysis only. Independent report and plot review is approved: [analysis_review.json](analysis_review.json). Runtime source and raw artifacts remain unchanged.
