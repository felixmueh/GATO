# Reused PCG reaching comparison on current source

This run reuses the original 100 ordered TIAGo reaching tasks and B1/B16
candidate policy on felix-devel 2db284c. It is one paired pass on the GTX1070,
with no extra repeats: 200 task episodes after the rough-reaching preflight.
The before/after statistics are an ancillary report, not a new main study.

The current cohort follows the established V01/T01 pose protocol: N32, .03 s
knots and feedback, .003 s RK4 plant, 8 s task allowance, 5 cm position,
30 degree tool-axis and joint-speed L1 <= 1 rad/s, no dwell or final approach.
Use the full frozen solver configuration in protocol.json. It includes
qd_cost=.1, vel_lim_cost=.1, orientation costs .1/2 and the established
campaign ctrl_lim_cost=.05 override, with SQP5/PCG120. The latter is the
public TIAGo preset, distinct from F01 original-PCG100 budgets.

Preserve the tested portfolio policy. The first solve has identical
constant-state/zero-control seeds. Later B1 shifts its selected plan; B16
keeps that same warm lane and draws 15 fresh goal-independent cosine/squared
progress perturbations, mapped to torques using RNEA around nominal ABA.
Choose the lowest native final merit among finite full plans, merits and
nominal-feedback RK4 prefixes, breaking ties by lane ID. Physical limits are
checked separately and do not veto selection. Reset rho each solve and retain
native duals per slot, as the original experiment did. This loop uses the
current generic constructor, common helpers and solver interface; it does not
use the ordinary live force-hypothesis selector.

historical_manifest.json and inputs/ preserve the exact ordered tasks, input
bytes, seeds and historical protocol/source identities. Do not resume an old
cohort under the new source or overwrite its raw results. The old cohort used
N64/.001 s plant/5 s/4 cm/qd=.01/vel_lim=.3/ctrl_lim=.003. Its bound-valid
successes were 17/100 B1 and 25/100 B16, with eight rescues and zero regressions;
goal-only successes were 37/100 and 51/100. Changes in the new statistics combine
source and setup changes, and cannot be attributed to the fixes alone.

Retain every input, seed, optimized output, merit, native/wall timing,
selection, step metadata, executed state/control, outcome and failure. Use
fresh raw directories and capture exact commands, source and module hashes.
The audited evaluation-only task tests from 75b17ce/4b6fa25 are included here;
no historical solver, QDLDL or B01 remedy is adopted.

This paired run is the existing BA task comparison only. The supplementary
unoptimized-input diagnostics, nominal larger-budget B1 comparison and mode B
BB implementation remain active campaign work; this run does not complete
those requirements. The existing F01c serial/batch timing is separate-budget
and separate-input evidence. For future BB, Felix approved one Jetson pass
through the same 100 B1/B16 pairs; target duration/storage discussion and
mode B readiness are still required before execution. Retained moving-pillar
cases keep their different proposal/objective policy and await compatibility
review; they are not silently replaced by obstacle-free tasks.
# GTX1070 historical/current-default comparison, 2026-10-10

The reused 100 tasks completed once per B1/B16 method: 200 episodes, no
technical failures, no saved-data integrity exclusions. This is the requested
incidental comparison, not the full BA study. The current candidate construction,
selection and RNG policy are preserved; current reaching defaults and source
versions differ from the historical cohort.

| Outcome out of 100 tasks | Historical B1 | Historical B16 | Current B1 | Current B16 |
| --- | ---: | ---: | ---: | ---: |
| Goal reached | 37 | 51 | 51 | 62 |
| Goal reached within sampled physical limits | 17 | 25 | 23 | 22 |
| Timeout | 59 | 48 | 44 | 36 |
| Completed numerical failure: no finite selectable plan | 4 | 1 | 5 | 2 |

B16's bound-valid rescues/regressions changed from 8/0 to 1/2. Thus B16 still
raises goal-only success, but its net bound-valid advantage is absent in this
current setup. These are separate measures; there is no collision or continuous
physical-validity certificate. Among pairs where both methods reached within
sampled limits (historical 17, current 21), median B16 cost gain was 0.0371%
and 0.2711%, respectively; median reaching-time gain was 0% in both cohorts.
Cost is integrated until each method's own stopping time and excludes terminal
and bound penalties. These conditional gains do not compare raw costs across
different weights or stopping criteria.

N changed 64→32, plant step 1→3 ms, timeout 5→8 s, position gate 4→5 cm,
velocity cost .01→.1, velocity-limit cost .3→.1 and control-limit cost
.003→.05. Knot spacing stays 30 ms; the shorter prediction changes physical
seed timing despite preserving the perturbation policy. This cannot isolate
the effect of the solver fixes. Matching RTX5060/internal and BB remain pending.
Known numerical limitations SB-001/SB-002 remain recorded.

`RESULTS.json` pins source/module/protocol identities, commands and external raw,
analysis and review receipts. Collection used source `2cc7bd9`; analysis used
`972aa1f`. One same-machine unchanged task suite passed 3/3, and the canonical
modeA B1/B16 rough-reaching and sampled physical preflight passed before the
200 episodes. Collection took 1835.316 s. Original failed postprocessing is
retained: a strict horizon metadata comparison rejected .93 versus
.9299999999999999 seconds. The separately reviewed correction validates the
duration derived from exact N/knot spacing; no episodes or raw files changed.

Saved-data analysis, using a fresh derived output directory:

```sh
python3 -B examples/randomized_multimodal/analyze_reaching_campaign.py \
  --campaign /home/felix/gato-results/gtx1070/BA-current-defaults-100-A-20261010T192610Z \
  --output /path/to/fresh/analysis
```

## BB: solver-wrapper wall paid, GTX1070 harness qualified

`bb_protocol.json` reuses the exact100 ordered tasks, input/model hashes,
current-default costs and portfolio policy. It is a separately identified B
cohort, not a continuation of the A results. One Jetson pass of200 episodes is
approved. Implementation and GTX1070 preflight are reviewed; the full BB task
cohort has not run. Target runtime/storage discussion and exact deployment
still precede large execution. RTX5060/internal and Jetson need their own suite
and canonical B preflight.

The bootstrap solve is recorded but unpaid and executes one nominal30ms prefix.
Thereafter the old active plan runs for each current public solver wrapper's
measured wall seconds; the returned winner starts u0 on arrival. Feedback has
no30ms minimum wait. Warm shift uses the previous solve-start origin, separately
from control activation age. Replay splits3ms substeps at control knots and
fractional boundaries, holds the last valid control beyond the plan, and clips
only at the terminal8s episode boundary. Late/unneeded outputs remain raw and
never activate. Candidate selection remains the original finite full plan,
finite native merit and nominal-feedback prefix from solve-start state; physical
bounds do not veto. No eligible paid return still pays old-plan execution;
task success, selection/controller failure and physical compliance are separate.

The paid scope is **public solver-wrapper wall only**, including work inside
that call. Candidate generation/shifting, rho setup, get_stats/native defect
checks, selection/prefix screening, replay, metrics, recording and process work
are unpaid. Report generation, selection, diagnostic, raw-recording and replay
wall fields plus whole episode elapsed time. The component fields cover their
named sections, not an exhaustive accounting of initialization, reset/setup,
metrics outside replay, JSON preparation, or every final receipt write. Never
claim full Python pipeline latency paid. Native microseconds are saved and
converted to milliseconds separately; they do not drive the plant.

The local `tiago_batching_execution.py` helper adapts only accepted12dc7cc
old-plan boundary/remainder logic. Existing common sampling/shift helpers are
unchanged. Raw `execution.jsonl` records attempted/completed/failed substeps;
`executed.npz` keeps typed executed states/controls/times and active-plan IDs/ages.
Per-solve clock/input/returned/stats/candidate files retain all outputs, proposal
coefficients, full timings and activation/discard reasons, including typed
nonfinite native arrays. `*_prefixes.npz` is candidate diagnostic motion, not
executed plant motion. Internal dual/rho buffers remain unexposed.

The existing pilot defaults to A. To inspect BB CPU policy/timing checks:

```sh
/opt/gato-venv/bin/python -B -m pytest -q tests/python/test_tiago_batching_mode_b.py tests/python/test_tiago_batching_policy.py tests/python/test_reaching_prediction_provenance.py tests/python/test_reaching_campaign_analysis.py
```

After independent review, the coordinator runs the required unchanged
`make test-tiago-tasks` and canonical B preflight through the portfolio pilot,
with explicit `--protocol experiments/tiago_batching/bb_protocol.json` and
`--preflight-input experiments/tiago_batching/bb_preflight.json`. Use a fresh
checkout-local timestamped output path, B1/B16, and the frozen CLI defaults.
Failures are retained/reported and hold that target family. No solver/native
suite, preflight, calibration or cohort run is implied by the CPU tests.

The manager selects B only with the explicit protocol path and requires an
explicit calibrated positive `--wall-timeout`; it cannot silently inherit the
A120s guard for Jetson. All100 pairs, attempts and failure holds remain frozen.
There are no automatic retries, task filters, strict deadline gates or hidden
feedback floors. Faster-than30ms B1 calls can multiply call count/storage;
measure actual target overhead/storage and discuss duration before a full run.
Existing source/module pins and a reviewed deployment/dry-run remain required.
Known SB-001/SB-002 and obstacle-free simulated-plant limitations remain.

GTX1070 validation on 2026-10-10 used the unchanged N32 Release sm61 module
SHA256 `5d21dc1d887f24dd66aa2dce77fe0a3606f66b178deea7c9b6148892c227b7b8`.
Independent focused CPU checks passed 43 tests plus 25 analyzer subtests; two
metadata correction checks passed. The required `make test-tiago-tasks` passed
all three unchanged cases in 20.919 s. One canonical same-loop B preflight per
arm then passed both rough reaching and sampled physical limits:

| Arm | Solves including bootstrap | Simulated reach time | Position error | Tool-axis error | Speed L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 232 | 0.484714 s | 49.8723 mm | 8.1451 degrees | 0.60175 rad/s |
| B16 | 25 | 0.507115 s | 48.4665 mm | 8.2507 degrees | 0.58793 rad/s |

Both arms had zero sampled position-limit excess, peak velocity ratios below
0.223 and peak effort ratios below 0.34. All actual inputs, returned plans,
telemetry, controls, states and clocks are retained under
`example_artifacts/bb_validation/20261010T203804.657656Z/`. No retries, solver
changes, robot runs or full 100-task BB campaign occurred. `BB_VALIDATION.json`
pins the exact source/module/protocol, task-suite and preflight receipts.
Only solver-wrapper wall is paid. B1 file recording dominates its recorded
episode-body time; whole child time also includes startup. These separate
measurements provide context for later runtime/storage estimates.
