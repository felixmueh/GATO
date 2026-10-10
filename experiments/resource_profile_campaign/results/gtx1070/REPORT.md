# F04/F05: GTX 1070 resources and response timing

All 38 approved cases completed once with **44,385 native calls, zero execution errors**, and 901.473 s whole-queue wall time. The cohort contains eight query-only cases, 18 process-first measurements (three per budget/B arm), six measured profiles, 44,349 measured sustained calls and 12 separately retained excluded warmups. No solve timing is missing. Matching RTX5060/internal and Jetson cohorts remain pending.

First response means the first public solve in each fresh process; import and constructor are outside the solve timer. It is not hardware power-on cold. Stage timing is OFF for first-response and sustained arms. Each sustained call creates a complete owner, resets dual/rho, solves and records; its throughput differs from the solve-only reciprocal. B1 rotates individual candidates across six inputs; B16 processes whole pools. These are not same-candidate/equal-quality batch speedups or controller actuation rates.

Process-first latency, three independent process samples per arm (ms):

| SQP/PCG | B | Native mean / median / p95 | Wrapper mean / median / p95 |
|---|---:|---:|---:|
| 1/100 | 1 | 4.289 / 4.300 / 4.319 | 4.451 / 4.451 / 4.490 |
| 1/100 | 16 | 7.572 / 7.544 / 7.676 | 7.810 / 7.762 / 7.930 |
| 5/100 | 1 | 6.346 / 6.081 / 6.837 | 6.562 / 6.268 / 7.128 |
| 5/100 | 16 | 22.112 / 22.153 / 22.170 | 22.394 / 22.440 / 22.457 |
| 10/800 | 1 | 8.269 / 8.152 / 8.588 | 8.520 / 8.371 / 8.888 |
| 10/800 | 16 | 89.822 / 89.364 / 97.222 | 90.150 / 89.677 / 97.550 |

Sustained measurements (all calls retained, one excluded warmup per arm):

| SQP/PCG | B | Measured calls | Native mean / median / p95 ms | Wrapper mean / median / p95 ms | 1000/wrapper mean (solve calls/s) | Actual whole-loop calls/s |
|---|---:|---:|---:|---:|---:|---:|
| 1/100 | 1 | 13,359 | 0.771 / 0.690 / 1.279 | 0.900 / 0.819 / 1.405 | 1111.51 | 105.88 |
| 1/100 | 16 | 5,879 | 4.258 / 4.214 / 4.596 | 4.427 / 4.381 / 4.754 | 225.87 | 48.99 |
| 5/100 | 1 | 11,188 | 2.837 / 2.816 / 3.347 | 2.986 / 2.965 / 3.503 | 334.88 | 93.23 |
| 5/100 | 16 | 3,714 | 15.959 / 15.920 / 16.438 | 16.150 / 16.120 / 16.628 | 61.92 | 30.94 |
| 10/800 | 1 | 9,009 | 5.503 / 4.313 / 9.808 | 5.663 / 4.469 / 9.989 | 176.57 | 75.07 |
| 10/800 | 16 | 1,200 | 83.423 / 82.819 / 89.428 | 83.667 / 83.060 / 89.664 | 11.95 | 10.00 |

Actual loop durations were 126.166 s for B1/SQP1/PCG100 and 120.003–120.037 s for the other five arms. The respective boundary overruns were 6.166 s and .003–.037 s: the worker stops starting calls after 120 s and retains completion/recording of the in-flight loop. This is descriptive, with no deadline gate, retry or bottleneck attribution. Reciprocal means above are **1000 divided by mean latency**, not the mean of instantaneous frequencies. [latency_summary.csv](latency_summary.csv) also retains median/p95 reciprocals and separate profile/warmup groups; [first_response_samples.csv](first_response_samples.csv) contains all 18 individual first-call measurements.

F04 queried the seven actual-module kernels at N16/32/64/128 × B1/B16: 56 rows. All device/driver/runtime/property, kernel-attribute and occupancy API statuses were successful; every row predicts a fit within default launch limits. Dynamic shared memory spans 64–40,644 bytes, static shared memory 0–176 bytes, registers/thread 18–54, and predicted active blocks/SM 2–16; reported local bytes/thread are zero. These are compiler/API resource attributes and theoretical occupancy, not observed launches, occupancy utilization or proof of task feasibility. Query workers create no solver or kernel launch; no unsupported actual launch is demonstrated. Complete per-kernel values and query/device statuses are in [kernel_resources.csv](kernel_resources.csv) and [query_devices.csv](query_devices.csv).

F05 profiling is a separate stage-ON cohort: one measured call per budget/B after an excluded warmup. All 236 expected measured windows were reached with elapsed microseconds (472 including warmups), with zero missing windows: initial reset/merit/snapshot, seven windows per SQP iteration (setup KKT, Schur system, PCG, Dz, residual/count D2H, candidate merit, line search/stats publication), and final reset/merit. For example, PCG stream-elapsed totals across iterations for B1/B16 are .328/.806 ms at1/100, .883/3.834 ms at5/100, and .997/50.183 ms at10/800. These are single-call observations, not repeated stage estimates. [stage_windows_summary.csv](stage_windows_summary.csv) retains every stage/category mean/median/p95 and missingness; [profile_totals.csv](profile_totals.csv) retains stream-window sums alongside native/wrapper times.

Stream windows include idle host submission gaps; their sum is a diagnostic, not a complete additive decomposition of native chrono or pure kernel time. The existing `pcg_times_us` field remains a literal-zero placeholder and is explicitly **unavailable**, distinct from the new PCG stream window. Stage-ON timing is not pooled with OFF timing, and no causal instrumentation-overhead or ON/OFF numerical-equivalence claim is made.

Passive `nvidia-smi`/NVML sampling across the entire queue returned 899 rows, nominally once per second. Temperature ranged32–57°C, SM clocks139–1987MHz, memory clocks405–4006MHz, power6.50–83.57W, sampled GPU utilization0–90%, and reported used GPU memory208–302MiB. All six numeric CSV fields are available in all899 rows; these include startup/query/idle intervals. Command, field names, source CSV hash and scope are pinned; [passive_conditions.csv](passive_conditions.csv) reports ranges and missingness. No clock/power settings changed. Observer overhead was not measured. These observations do not establish thermal steady state, saturation, per-call energy or a cause of timing differences.

Provenance: runtime source `4e5301feaea5f2d06530922ba46aa543e1995d47`, science-plan checkpoint `714736a`, production core `16b33719b4f0fd463eec7262c99b16b0b324366f`; bank manifest `f8a8217982f3fa2b1b4fd2629e8cb45949d372f6d43c6137bbdaa7e52acc7007`. All builds enable accepted stop telemetry and resource queries; N16/32/64/128 OFF modules and the separate N32 ON module each retain exact build/module identities. [provenance.json](provenance.json) pins these five module paths/hashes, flags, build receipts, all 38 absolute raw paths/manifests, plan `32480d8e9664937591cee40035f14439fba42229bc1f344d035162436934066c`, target review `b8b38e8bf3112647a478e61c40aa0a2a9ea326b6f3b1844b2779e714a62986a4`, and the validated export index `260ed100e64ee1849552c331b0d83d19c331adaa3f5f00c73fa3017bb4170fcf`.

Raw roots: `/home/felix/gato-results/gtx1070/F04-F05-gtx1070/`; validated plots/full call and stage CSVs: `/workspace/gato-campaign-recovery-20261010/analysis/f04-f05-gtx1070/`; selection index: `/workspace/gato-campaign-recovery-20261010/f04-f05-gtx1070-analysis-index.json`. Total raw bytes/file count are not indexed in the consumed exports and remain explicitly unavailable; no second manifest sweep or filesystem walk was performed. Task success, physical compliance and numerical certification are unmeasured in these static timing/query studies.

Reproduce this report’s tables by copying `summarize_exports.py` alone to a fresh directory and running `python3 <fresh-directory>/summarize_exports.py`. It consumes validated CSV/provenance, performs eight bounded metadata checks, and refuses output overwrites. It performs no native import, GPU query/solve, build, replay or test. The source/raw cohort is unchanged; no duplicate figures were generated.
