> This isolated worktree now owns the orientation/ordinary-torque comparison.
> See [ORIENTATION_README.md](ORIENTATION_README.md) for current scope and exact
> commands. The margin-study notes below are historical; the former100validation
> was user-stopped and is not resumed here.

# Random reaching with interior cost bounds

This branch develops a batch-one parameterization for the original random reaches
that stays within the robot's physical limits. It preserves the twelve ordered
development tasks and the historical failures. A finite timeout within limits is
informative; arrival, physical compliance and execution status remain separate.
M5's stopping decision/report is a preserved checkpoint. Felix subsequently
reopened joint-only development before validation. M6 and M7 completed the same
twelve tasks with the remaining task 2342 J1 failure; M8 restored q weight at the
same margin and reduced, but did not remove, that violation. M14 subsequently passed every physical check on all twelve development tasks,
with one rough arrival and eleven timeouts. This is a bounded physical baseline,
not reliable general reaching.

The latest priority is large-first physical screening at joint inset 0.20 rad:
find a physically passing endpoint even if reaching is poor, then reduce/narrow
the weight range using recorded outcomes. M9 (q weight 0.05) and M10 (q weight
0.10) passed q but failed speed on the fixed task 2342 screen. M10 reached the goal despite its earlier physical
violation. M11 supplied an all-physical passing screen endpoint at common factor
10 on M8's joint/speed weights. M12–M15 narrowed that factor using the same
fixed task. M14 (factor 2.125, q 0.02125 / speed 1.275) completed the full
twelve-task pilot after its rough canonical preflight passed. It was selected
over the lower passing M15 screen for greater observed J1 clearance. This
development selection was frozen at 2026-10-15 11:43:04 UTC after all twelve
physical checks passed. The separate additional 100-task validation is now running
without further tuning. No finer midpoint or per-joint change is queued. The additional 100-task
validation reuses its previously frozen assignment; horizon execution remains
held while the 100-task run owns the GPU. Its original preparation and the old M5
checkpoint stay intact; no validation
outcome informed tuning. The historical reserved 88 remain held too.

The numerical source is `85ef2ea5cf5b96726be6f4b5441f4a7c0ca041b4`, based on
the reviewed 1%-width C2 quadratic continuation. The only additional numerical
change supplies interior bounds consistently to all nine position, velocity and
effort cost value/gradient/Hessian calls. Separately recorded N32 builds use
that same numerical source:

| Cohort | Cost inset/fractions: q / velocity / effort | Module SHA256 |
| --- | --- | --- |
| M0–M3 | 0.02 rad / 0.90 / 0.90 | `c7dfad734cf5b25ca4a4eb7984f6ca7201da09a970a3b94caea1f79c8929c6b7` |
| M4/M5 | 0.10 rad / 0.80 / 0.85 | `67fe78c95f68a619571b8f783cc8b456d1aa16b8ac268adb1290cd708b0e62b7` |
| M6 | 0.15 rad / 0.80 / 0.85 | `a7cf90854fd581fc3f9eaa24a6d28a5d2fab163522911a3b0e8e8c6fc46b5415` |
| M7–M15 | 0.20 rad / 0.80 / 0.85 | `d9bee23f90a1d7816b32db23749bd651db2a7537f94c3dfec15f1a448bb34cc8` |

M0–M3 use the module in this worktree's `python/bsqp/` directory. M4/M5 use:

```text
/workspace/GATO-random-reaching-wide-margins/python/bsqp/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so
```

The later joint-margin builds preserve immutable module copies:

```text
/workspace/GATO-random-reaching-joint-margins/.experiment-modules/q015/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so
/workspace/GATO-random-reaching-joint-margins/.experiment-modules/q020/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so
```

M7–M15 use the q020 copy with:

```text
-DTIAGO_COST_JOINT_MARGIN_RAD=0.20
-DTIAGO_COST_VELOCITY_LIMIT_FRACTION=0.80
-DTIAGO_COST_CONTROL_LIMIT_FRACTION=0.85
```

These affect soft costs only. M4 position cost bounds move inward by 0.10 rad
at both ends; velocity cost bounds are ±2.0 rad/s; effort cost bounds are
±36.55 Nm for joints 1–2 and ±22.1 Nm for joints 3–7. M0–M3 instead use
±2.25 rad/s and ±38.7/23.4 Nm. The C2 continuation width is 1% of the
resulting cost interval. Header defaults are zero margin and unit fractions;
source identity alone does not establish the active configuration.
`build_receipt.json` and `build_receipt_wide.json` retain actual compile flags
and source/module hashes. The earlier 0.04 rad wider build was unused
preparation, preserved separately as `build_receipt_wide_q004_unused.json`.
`build_receipt_q015.json` and `build_receipt_q020.json` identify the later
margin builds. The wrapper verifies selected source, new header, module and
frozen inputs. The q020 native regression passed 3/3 plus CUDA access; runtime
weight changes reuse that exact module. A static source review traced
`q_lim_cost` from the protocol through the TIAGo controller and binding to the
native value/gradient/Hessian; no override or omission was found.

Physical validation still uses the raw URDF joint intervals, ±2.5 rad/s and
±43 Nm for joints 1–2 / ±26 Nm for joints 3–7. It retains the historical
q-excess tolerance of 1e-6 rad and velocity/effort ratio tolerance of 1+1e-6.
The pool's sampling box is already inset 0.12 rad from raw URDF joint bounds.
All 100 sampled starts and goal witnesses remain inside the 0.10 rad cost inset;
the twelve development starts have at least 0.131623 rad physical clearance.
The separate canonical start has J4 clearance 0.093461 rad. At inset0.15 one
of the twelve development starts is outside a soft cost interval; at inset0.20,
two starts and three FK witnesses are outside. None is excluded or clipped;
a witness is not a required joint endpoint. The canonical J4 also lies outside
these cost intervals while physically valid. All starts remain unchanged. No
clipping, filtering, physical candidate veto or early limit-stop masks behavior.
There are no obstacles or collision claims.

## Fixed execution and bounded candidates

All candidates use B1, SQP cap 1000, PCG120, N32 with 30 ms knots and feedback,
3 ms RK4 plant execution, and an eight-second observation limit. Mode A
(`offline_timing=controller_dt`) advances no plant time during computation;
measured wall/native solve times are descriptive, not a real-time claim.
Existing zero-PCG stopping remains active. The cap is an allowance, and every
solve's actual SQP/PCG work is recorded. KKT flags never gate this study.

The production TIAGo controller constructor and solver remain in use. Cold seeds
repeat the actual rest state with zero controls; later seeds shift the previous
output with the public helper and reanchor at the measured state. Duals and rho
reset once per episode; adaptive rho persists across feedback solves. The first
t0 solve supplies the first executed prefix and is included in all work records.
There is no unrecorded warmup or final approach.

| Candidate / protocol file | q limit cost | Speed limit cost | Effort limit cost | qd | Mu |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 / `protocols/M0-margin-only.json` | 0.01 | 0.1 | 0.003 | 0.1 | 1 |
| M1 / `protocols/M1-speed-penalty.json` | 0.01 | 1.0 | 0.003 | 0.1 | 1 |
| M2 / `protocols/M2-all-limit-penalties.json` | 0.1 | 1.0 | 0.03 | 0.05 | 1 |
| M3 / `protocols/M3-merit-mu10.json` | 0.1 | 1.0 | 0.03 | 0.05 | 10 |
| M4 / `protocols/M4-balanced-wide.json` | 0.01 | 0.2 | 0.01 | 0.05 | 10 |
| M5 / `protocols/M5-firmer-limits.json` | 0.015 | 0.6 | 0.02 | 0.05 | 10 |
| M6 / `protocols/M6-joint-margin015.json` | 0.01 | 0.6 | 0.02 | 0.05 | 10 |
| M7 / `protocols/M7-joint-margin020.json` | 0.005 | 0.6 | 0.02 | 0.05 | 10 |
| M8 / `protocols/M8-joint-margin020-weight010.json` | 0.01 | 0.6 | 0.02 | 0.05 | 10 |
| M9 / `protocols/M9-joint-margin020-weight050.json` | 0.05 | 0.6 | 0.02 | 0.05 | 10 |
| M10 / `protocols/M10-joint-margin020-weight100.json` | 0.10 | 0.6 | 0.02 | 0.05 | 10 |
| M11 / `protocols/M11-joint-speed-large.json` | 0.10 | 6.0 | 0.02 | 0.05 | 10 |
| M12 / `protocols/M12-joint-speed-mid055.json` | 0.055 | 3.3 | 0.02 | 0.05 | 10 |
| M13 / `protocols/M13-joint-speed-mid0325.json` | 0.0325 | 1.95 | 0.02 | 0.05 | 10 |
| M14 / `protocols/M14-joint-speed-mid02125.json` | 0.02125 | 1.275 | 0.02 | 0.05 | 10 |
| M15 / `protocols/M15-joint-speed-mid015625.json` | 0.015625 | 0.9375 | 0.02 | 0.05 | 10 |

M0 isolates the initial margin change; M1 isolates stronger speed cost. Both
completed their assigned twelve-task pilots. The earlier qd-only M2 and joint
0.04 rad margin proposals were preparation, not measured cohorts. After Felix
broadened the tuning scope, M2 combined stronger limit costs with lower qd;
M3 changed only mu to 10. Both failed their matching canonical preflights:
finite, physical eight-second timeouts with final position errors 90.568 and
90.694 mm. All 267 solves in each used the zero-PCG shortcut. Their dependent
pilots were held, so neither has twelve-task results. These failures remain
recorded and do not trigger solver repairs or retries until success.

M4 combines wider margins with balanced weights. At zero speed its combined
velocity curvature is 0.15, close to M0's 0.1395, while the limit gradient at
physical speed 2.5 rad/s is 72.46 versus M0's 16.77. The comparison is local:
it still discourages approaching the new 2.0 rad/s cost boundary strongly.
A 0.10 rad joint inset exceeds one 30 ms interval's 0.075 rad travel at maximum
speed and matches a typical frozen-posture braking scale, without covering every
braking estimate or providing a safety guarantee. M4 retains q weight 0.01:
combining q weight 0.02 with the wider inset would nearly reproduce M2/M3's
large canonical J4 centering gradient. Effort weight 0.01 is a moderate increase
around unchanged `u_cost=0.0001`. This is combined parameter development, not
an isolated margin-effect comparison. Its canonical preflight passed at 0.54 s
with physical checks passing. Its complete pilot retained a joint/speed failure
on task 2342.

M5 was the bounded firmer candidate at the same wide margins and module.
Its q/speed/effort coefficients rise 1.5/3/2 times M4, leaving all other settings
fixed. Its local total velocity curvature is 0.35, below M1's 0.4951, while its
physical-speed limit gradient is 217.37. The matching canonical preflight passed
at 0.84 s. Its complete development pilot had no execution errors but retained
a 0.0473932 rad maximum joint-limit excess on task 2342. All sampled speed and
effort checks passed: maximum ratios 0.778744 and 0.833065. Compared with M4,
speed compliance improved while the largest q excess increased from 0.0395858
rad. This is not a complete physical qualification or a monotonic improvement.

| Completed development cohort | Rough arrivals / 12 | Physical passes / 12 | Both / 12 |
| --- | ---: | ---: | ---: |
| M0 | 3 | 10 | 3 |
| M1 | 1 | 10 | 1 |
| M4 | 3 | 11 | 3 |
| M5 | 3 | 11 | 3 |
| M6 | 3 | 11 | 3 |
| M7 | 3 | 11 | 3 |
| M8 | 3 | 11 | 3 |
| M14 | 1 | 12 | 1 |

M5's valid arrivals were 3626/9651/8270 at 4.53/7.65/7.26 simulated seconds.
At that checkpoint the cost search was stopped and M5 frozen. Later user
steering reopened only joint margins/weights. M6's maximum task 2342 q excess
was 0.0479143 rad; M7's was 0.0648886 rad; M8's was 0.0260036 rad. Each full pilot
still had 11/12 physical passes, with speed/effort passing throughout. M7/M8
isolates weight at 0.20 rad; M6/M8 isolates margin at weight 0.01. These observations
remain separate from physical-only screens and from validation.

M9's canonical preflight was a finite physical eight-second timeout, ending
80.507 mm from the target. Its original `preflight_pass=false` and return code
remain unchanged. The expressly authorized task 2342 screen had zero sampled q
excess, but velocity ratio 1.139542 (first J3 violation at 0.75 s), torque ratio
0.806789 and a timeout at 121.787 mm/64.48 degrees tool-axis error. It is not a
physically passing endpoint or a task qualification. M9 has no full twelve-task
pilot. M10 changed only q weight 0.05→0.10 from M9, at the same inset/module.
Its canonical remained a finite physical timeout at 119.426 mm/32.28 degrees.
Task 2342 then reached the rough goal, but still failed speed: maximum ratio
1.185194, first J3 violation at 0.687 s. q and effort passed (torque ratio 0.804732).
Thus q-only doubling found no physically passing upper endpoint.

M11 changes only speed weight 0.6→6.0 from M10. Joint/speed weights 0.10/6.0
are a common factor 10 above M8's 0.01/0.6. Effort weight stays 0.02 because its
checks have consistently passed. Near-zero total velocity curvature rises from
0.35 to 3.05; total velocity gradient at the physical 2.5 rad/s limit rises from
217.49 to 2173.79. Poor reaching is plausible, but physical-first screening is
explicitly authorized. Its actual outcomes remain separate; no guarantee is
inferred from these cost scales.

## Completed bracket screens and selected full pilot

The large-first search found a passing fixed-task endpoint at M11. Its canonical
was a finite physical timeout; its task 2342 screen passed q, velocity and effort
(maximum ratios 0.364620 / 0.441742 for velocity / effort) but timed out at
129.330 mm / 108.997 degrees tool-axis error. The following arithmetic midpoints
all passed that same physical screen. Their q excess was zero throughout the
sampled screen, and each screen timed out; reaching remained a separate outcome:

| Candidate | Common factor over M8 q/speed | Canonical rough outcome | Screen maximum v ratio | Screen maximum effort ratio | Screen final position / axis error |
| --- | ---: | --- | ---: | ---: | --- |
| M11 | 10 | Physical timeout | 0.364620 | 0.441742 | 129.330 mm / 108.997° |
| M12 | 5.5 | Physical timeout, 69.913 mm | 0.458841 | 0.618869 | 111.633 mm / 102.876° |
| M13 | 3.25 | Physical timeout, 56.536 mm | 0.572971 | 0.742765 | 103.721 mm / 99.425° |
| M14 | 2.125 | Passed at 1.77 s, physical | 0.663382 | 0.787807 | 97.332 mm / 100.441° |
| M15 | 1.5625 | Passed at 1.35 s, physical | 0.730451 | 0.801041 | 83.954 mm / 98.868° |

These screens cover only the declared development task 2342. M14's minimum J1
physical clearance there was 0.0273191 rad, compared with M15's 0.00496562 rad.
The search stopped refining the bracket and selected M14 for the complete ordered
twelve-task pilot, favoring that observed buffer despite M15's faster canonical
arrival. M14 completed twelve cases with twelve physical passes, one rough
arrival (task3626 at 6.84 s), eleven timeouts and no execution errors. Its minimum
sampled joint clearance was 0.0273528 rad, maximum velocity ratio 0.660626 and
maximum effort ratio 0.826864. It was frozen at 11:43:04 UTC before the separate
100-task validation began. Physical compliance and reaching need not vary monotonically with
weights, and neither one screen nor greater local clearance is a safety guarantee.
No new midpoint is queued. Validation is running without tuning; horizon
execution remains held while CPU-only horizon preparation proceeds separately.

Only explicitly flagged protocols M9–M15 set
`physical_search_allow_preflight_timeout=true` and
`search_screen_task_ids=[2342]`. The saved stage launcher permits a canonical
failure solely for stage `screen`, and only when the canonical record is complete,
finite/physically compliant, has no execution error and has outcome `timeout`.
It preserves the failed rough-reaching result and writes a separate authorization
receipt. This is no waiver for full `pilot`, evaluation, validation or horizons.
A physical/nonfinite/execution failure still holds the dependent screen.

Once a useful candidate is selected, full qualification still requires rough
canonical success, the complete assigned twelve-task physical checks and
nontrivial, clearly reported reaching behavior. Passing one fixed failure screen
cannot establish all-task compliance or erase earlier failures. Validation remains
separate from development: M14 is now frozen and its additional 100-task
validation is running. No per-joint source changes or further cost trial is queued.
The completed development pilot demonstrates observed physical compliance, while
its low arrival count remains a limitation.

Other public full-pose weights stay fixed: running/terminal orientation 0.1/2,
running position 4 and terminal position 80. Mu weights L1 modeled dynamics and
initial-state defects in line-search merit; it does not multiply physical-limit
costs or adapt automatically. Rho is the separate adaptive damping parameter.
M2/M3 established no canonical reaching benefit from mu=10; M4's pass cannot be
attributed to mu alone. Mu=100 is not queued. Merits from different weights/mu
are different quantities, not a common objective ranking.

Each changed configuration requires its matching canonical preflight before
its assigned pilot. See [the independent review](MARGIN_REVIEW.md) and
[physical cost scales](COST_SCALING.md). The wide native regression passed all
three frozen task cases in 16.81 s, with CUDA access smoke passing; its source,
module, flags and log identity are retained in `build_receipt_wide.json`.
No new acceptance gate or high-precision residual requirement is introduced.

## Reproduction commands

Only one process may use the GPU for measurements. Run from this worktree with
the exact module required by the selected protocol. To inspect the selected M14 candidate, choose a fresh output
root for each invocation group:

```sh
STUDY_SOURCE=/workspace/GATO-random-reaching-margins
STUDY_RAW=/workspace/GATO/_expr_results/random-reaching-margins-20261015/manual-$(date -u +%Y%m%dT%H%M%SZ)
STUDY_PROTOCOL="$STUDY_SOURCE/experiments/random_reaching_baseline/protocols/M14-joint-speed-mid02125.json"
STUDY_MODULE=/workspace/GATO-random-reaching-joint-margins/.experiment-modules/q020/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so
```

Validate the exact configuration without native imports, GPU calls or output
writes. The module must already exist because its bytes are checked:

```sh
/opt/gato-venv/bin/python "$STUDY_SOURCE/experiments/random_reaching_baseline/run.py" \
  --source "$STUDY_SOURCE" \
  --module "$STUDY_MODULE" \
  --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching \
  --protocol "$STUDY_PROTOCOL" \
  --preflight-input /workspace/GATO-experiment-runner-simple/experiments/inputs/batching/ba_preflight.json \
  --output "$STUDY_RAW/preflight" --dry-run
```

Remove `--dry-run` from that exact command to execute the canonical preflight.
The old BA file supplies geometry only; `preflight_effective.json` records the
new configuration, including q/speed-limit costs 0.02125/1.275 for M14 and retained rho.
The historical file stays immutable. Preflight requires rough arrival, sampled
physical compliance, at least 100 mm initial error and 50 mm endpoint movement.
Check `"preflight_pass": true` in
`$STUDY_RAW/preflight/preflight/metrics.json` before launching the pilot. A failed
preflight holds its dependent work and remains recorded.

For the selected candidate, the recorded stage launcher commands
are below. Each stage uses fresh output/log paths and preserves canonical failure;
completed stages must not be overwritten. The screen command applies its narrow
explicit timeout authorization and assigns only2342:

```sh
STUDY_RECORDS=/workspace/GATO/_expr_results/random-reaching-margins-20261015
/opt/gato-venv/bin/python "$STUDY_RECORDS/execute_stage.py" M14-joint-speed-mid02125 preflight
/opt/gato-venv/bin/python "$STUDY_RECORDS/execute_stage.py" M14-joint-speed-mid02125 screen
```

The complete ordered development pilot command below uses selected M14, whose
rough canonical preflight passed. The recorded pilot is complete; use a
fresh output only for an explicitly requested reproduction. It does not use the
physical-only timeout waiver:

```sh
/opt/gato-venv/bin/python "$STUDY_SOURCE/experiments/random_reaching_baseline/run.py" \
  --source "$STUDY_SOURCE" \
  --module "$STUDY_MODULE" \
  --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching \
  --protocol "$STUDY_PROTOCOL" \
  --pairs 3626 7733 7132 7093 1265 9651 5264 279 8054 8270 2497 2342 \
  --output "$STUDY_RAW/pilot"
```

For an explicitly assigned subset of development tasks, use a separate output:

```sh
/opt/gato-venv/bin/python "$STUDY_SOURCE/experiments/random_reaching_baseline/run.py" \
  --source "$STUDY_SOURCE" \
  --module "$STUDY_MODULE" \
  --pool /workspace/GATO-experiment-runner-simple/experiments/inputs/batching \
  --protocol "$STUDY_PROTOCOL" \
  --pairs 3626 7733 \
  --output "$STUDY_RAW/custom-pairs"
```

The CLI accepts distinct IDs from the frozen 100-task roster; that flexibility
does not authorize tuning on the reserved 88 or replacing the complete pilot
with a favorable subset. For an explicit reproduction of the historical M1
configuration, change both the protocol and module, and use a fresh root:

```sh
STUDY_PROTOCOL="$STUDY_SOURCE/experiments/random_reaching_baseline/protocols/M1-speed-penalty.json"
STUDY_MODULE="$STUDY_SOURCE/python/bsqp/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so"
STUDY_RAW=/workspace/GATO/_expr_results/random-reaching-margins-20261015/M1-speed-penalty-manual-$(date -u +%Y%m%dT%H%M%SZ)
```

The commands above then use that exact M1 configuration. Existing completed
runs need no repeat merely for documentation or format changes. The wrapper
does not launch later candidates or enforce the preflight dependency itself.

## Records and interpretation

Each episode retains exact inputs, per-solve seed/returned horizons and typed
iteration arrays, complete 3 ms executed t/q/v/applied-control histories,
source/module/configuration hashes, timings and partial failure records.
`solves.jsonl`, per-case `metrics.json` and root `summary.json` keep goal success,
physical compliance and execution status separate. Seed, predicted horizon,
executed prefix and complete executed-motion limit checks are distinct.
Nonfinite numeric artifacts remain preserved; JSON descriptive values use null.

Rough arrival is checked at 30 ms feedback boundaries: position error ≤50 mm,
tool-axis error ≤30 degrees and joint-speed L1 ≤1 rad/s, with zero dwell.
Position/pose-region first touches use dense 3 ms observations. Full SO(3) error
and last-second progress/motion are descriptive; stopping on arrival supplies no
post-arrival settling evidence. A clean final pose does not erase an earlier
violation. Task-specific execution failures retain data and allow independent
cases to continue; shared setup failures hold dependent work. Interruptions stop
subsequent cases. This script controls neither a physical robot nor Gazebo.

The current raw/build receipt destination is
`/workspace/GATO/_expr_results/random-reaching-margins-20261015/`.
Preflight/pilot directories are named by configuration, for example
`M14-joint-speed-mid02125/preflight`,
`M14-joint-speed-mid02125/screen` and `M14-joint-speed-mid02125/pilot`. Keep large data,
full horizons, builds and separate raw backups outside the thesis export. Finalized reports, self-contained PDFs,
figures, tables and compact editable plotting inputs belong in
`/workspace/GATO/thesis_results/random-reaching-margins-20261015/` once generated.
`analyze_margins.py` exports saved margin cohorts and the physical/cost accounting;
the finalized report records its exact regeneration command and held-cohort labels.
The earlier `analyze.py` and its original cohorts remain separate evidence.

The original baseline and continuation results remain separate evidence:
[original-helper report](/workspace/GATO/thesis_results/random-reaching-baseline-20261015/REPORT.md)
and [continuation report](/workspace/GATO/thesis_results/random-reaching-continuation-20261015/REPORT.md).
Their raw roots and reports are preserved; this successor does not overwrite or
retroactively relabel either cohort.
