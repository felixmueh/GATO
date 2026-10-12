# Experiment runner

Current campaign decisions are in the main
[results review](/workspace/GATO/felixnotes/RESULTS_REVIEW_2026-10-11.md) and
[revision addendum](/workspace/GATO/EXPERIMENT_PLAN_REVISION_2026-10-10.md).
Use the reviewed machine plans in `experiments/launch/`. Historical plans
and raw results remain preserved with their original stopping policy and inputs.
The runner does not launch remote jobs or physical robots.

Current manual handoff: [Git-based machine instructions](/workspace/GATO/felixnotes/HANDOFF_REMOTE_MACHINE_RUNS.md).
Felix wants a branch pull and local commands; source archives are historical
handoff artifacts, not the current provisioning workflow. Publication of the
runner and exact side-source refs remains required; no push is claimed.

## Manual Jetson and RTX5060 launch

Latest machine selection is [the question-based coverage table](machine_coverage.md):
**89 RTX5060/internal cases,84 Jetson cases,23 separate Gazebo cases**.
The full earlier successor plans remain provenance. **Completed main Jetson
F01ab is reused and never rerun by this launch**; its older source/stopping-policy
labels remain separate. The accepted v5 square is one separate Gazebo case;
older failed square protocols remain historical and are not queued. Four
original-force tasks per target stay held. No remote agents or automatic
Gazebo/physical robot launch.

Run in the existing CUDA development environment with Git, CMake >=3.24,
a C++ compiler, CUDA `nvcc`, Python headers, NumPy, Pinocchio, pybind11 and pytest.
RTX5060 needs a toolkit supporting `120-real`; Jetson uses `87-real`.
The launcher uses `/opt/gato-venv/bin/python3` when present, otherwise its invoking
interpreter. Override with `--python /path/to/python3` if needed. It checks
missing dependencies before measurements and does not alter system packages.

From the published `experiment/simple-runner` checkout, use the command for
that machine:

```sh
python3 experiments/machine_launch.py rtx5060
python3 experiments/machine_launch.py jetson
```

The launcher validates frozen inputs/source pins, fetches missing immutable
source refs, prints the explicit plan/dry-run, then uses the existing runner
for shared local builds, canonical preflights, sequential execution and resume.
Scientific negatives remain results; failed preflights hold dependent families,
and independent cases continue. The plan choices/settings and limitations are
in the coverage table. The Jetson planning estimate is roughly2–8h/10–50GiB raw
plus builds/archive/headroom, with120GiB free a conservative recommendation,
not a hard gate. This is not measured successor Jetson performance. Put the
checkout on a volume with room; the launcher prints free
storage. The old6–24h estimate applied to the superseded broad plan.

Results stay under `.experiment-results/<machine>-zero-pcg-20261011/`; source
worktrees/builds stay in `.experiment-cache/<machine>-zero-pcg-20261011/`.
Explicit paths override legacy `local.json`. Repeating the command resumes
completed work and preserves failures. `--retry-failed` explicitly makes new
attempts for failed/blocked cases. `--dry-run` validates sources/inputs without
builds or GPU work. Use an existing persistent terminal if the shell may
otherwise disconnect; no background supervisor is created.

On completion or handled interruption, all local raw attempts, partial data,
inputs, logs and deployment identities are indexed and exported with the existing
gzip-level1 exporter to `.experiment-work/<machine>-zero-pcg-20261011-results.tar.gz`.
Build caches stay out. Keep room for raw and the archive; replacing an archive
also temporarily retains the old one. Regenerate collection/export alone with:

```sh
python3 experiments/machine_launch.py jetson --collect-only
```

Substitute `rtx5060` on that machine. Copy the printed archive to the desktop's
`/workspace/GATO/_expr_results/` and keep a separate raw backup. There are no
existing remote results requiring migration. Future demo recordings use
[compact raw references](demo_recording.md); the internal plans contain no demos.

Gazebo is separate. With the existing Gazebo/ROS environment already running,
inspect then manually run the three-family23-case integration plan:

```sh
python3 experiments/run.py --machine rtx5060 --plan experiments/launch/rtx5060_gazebo.json --dry-run
python3 experiments/run.py --machine rtx5060 --plan experiments/launch/rtx5060_gazebo.json --python /opt/gato-venv/bin/python3 --cache .experiment-cache/rtx5060-gazebo --output .experiment-results/rtx5060-gazebo
python3 experiments/collect.py --output .experiment-results/rtx5060-gazebo --export .experiment-work/rtx5060-gazebo-results.tar.gz
```

It measures two reaching/tracking integration tasks, one accepted five-goal
square and the20 distinct fault/baseline conditions; no GPU timing/performance
studies. Run Gazebo after RTX5060 internal success. M03 requires host-assisted,
separately logged simulator restoration before every trial and fresh PAL
observations; see the Git handoff for actual ROS/configuration prerequisites.
The runner does not start Gazebo, change the environment or run a robot.

**Git publication:** pull-only needs the runner branch and every advertised
`experiment/deployment-source-<pin12>` source ref published to the same origin.
Exact publication/clone/pull commands and pending remote status are in the
[Git handoff](/workspace/GATO/felixnotes/HANDOFF_REMOTE_MACHINE_RUNS.md).
All frozen inputs/native helpers are tracked; the launcher imports missing
sources through Git and owns builds/worktrees. No source archive copy is required.

## Revised campaign preparation


Historical broad transfer package:
[prepared-transfer-zero-pcg-20261012T105900Z](/workspace/GATO-experiment-runner-simple/.experiment-work/prepared-transfer-zero-pcg-20261012T105900Z/inventory.json).
Its inventory records the exact corrected runner/source pins and asset hashes.
The previous101500Z/efd190f package remains provenance. The new demo adapter
overrides finalization only (`recorded-event-sha-v1`): own immutable raw files
reuse their already-recorded hashes, all other finalized files are freshly
hashed. Recording/controller behavior and manifest schema remain unchanged.
The earlier213600Z/ee6a388 package remains provenance, superseded for transfer.
This package prepares700 GTX1070,713 RTX5060/internal,368 Jetson and27 Gazebo
cases; four original E02 cases per internal target are retained as held
artifacts and explicitly omitted from executable plans, not permanently dropped.
No remote deployment or new force execution is authorized by packaging.

`zero_pcg_plan.py` prepares full successor cohorts using reviewed source pins,
fresh matching native builds and target preflights. This includes all frozen
BA/BB task pairs rather than only the old pending continuation. It also retains
still-pending solver-independent checks on their required machines, without
rerunning completed GTX1070 model/resource/collision measurements merely because
the stopping policy changed.

```sh
python3 experiments/zero_pcg_plan.py
python3 experiments/package_plans.py --cohort zero-pcg --output experiments/successor/merged
python3 experiments/run.py --plan experiments/successor/merged/gtx1070_desktop.json --machine gtx1070 --dry-run
python3 experiments/run.py --plan experiments/successor/merged/rtx5060_desktop.json --machine rtx5060 --dry-run
python3 experiments/run.py --plan experiments/successor/merged/jetson_proposed.json --machine jetson --dry-run
```

Use a fresh raw result directory for a new cohort. Inspect the explicit plan
before execution; the old `campaign.json`, merged plans and native-copy recipes
remain historical. The restored solver can stop at zero PCG iterations while
KKT flags remain residual-based. Report actual SQP/PCG work and never pool the
old and new stopping policies as one measurement cohort.

The faster tracking proposal adds **8, 6 and 4 s periods** in internal mode B,
with provisional common **25 mm RMS / 50 mm maximum** position-error gates.
Its references preserve the original figure-eight geometry and are frozen in
`inputs/t01_faster/`. These cases are simulation-only; Felix reviews their
results and gates before any live adoption. Existing live and physical scripts
retain their approved tasks. Jetson V01/T01 in mode B comes before live work;
Jetson duration/storage discussion and manual physical-motion review remain.

The accepted force redesign is prepared separately because it first needs an
original-arm 0 N smoke and a fixed 5/10/20 N screen against a new 0 N baseline.
`successor/force_stages.json` records the status of that prerequisite, selected
force and revised E01/E02/E03 coverage. A pending stage is not an executable
old-force substitute. Persistent solvers and GPU prediction replace the old
fresh-owner/CPU-predictor study; their data remain separate protocols.

**Current force status: original E02 held; F*=10 N bank preserved.** The revised original-method 0 N canonical
reached its target but exceeded the sampled joint-speed limit by 0.127113 rad/s.
The matching v2 bank and final E02/E03 plans are prepared. The separate
`successor/force_smoke.json` is not queued in combined campaign plans.
`force_stages.json` retains both failed canonicals. Felix authorized the narrow
rho-policy correction: reaching now carries rho as requested, while tracking
continues to reset it each feedback solve. Source `bbbb047` reused the unchanged
native module for one unchanged canonical retry; it reached the target but
again failed physical compliance (joint-speed excess **0.465362 rad/s**, no
position/torque excess or execution error). That retry is retained at
`/workspace/GATO/_expr_results/force-rho-policy-20261011T201734Z/canonical-original0/`;
the first failure and previous package remain preserved. The correction did not
resolve the physical failure. Felix subsequently authorized ignoring **speed-only
violations for internal force-study eligibility**. Raw `physical_pass=false` and
speed excess remain unchanged; position/torque tolerances, finite data, full
completion and sampled-effort checks still apply. This does not apply to robots
or any other study. The earlier corrected canonical was reused for the bounded screen. The later
campaign canonical produced nonfinite plant/control; speed-only waiver does not
waive that execution gate. E02-original is now disabled on all targets pending
Felix's explicit decision, retaining four cases per target as held definitions.
No prior-canonical bypass, fresh force run or remote deployment is authorized.
Original0N smoke completed24s with finite data and zero physical excesses
(RMS21.323mm/max35.933mm); its old20mm task-gate failure stays reported. The
no-estimator0N canonical passed. The fixed 0/5/10/20 N screen completed with
all four physical gates passing. Against baseline RMS19.869mm, 5 N added
9.179mm and did not meet the frozen10mm degradation criterion; 10 N added
25.275mm and is the smallest qualifier. Final plans add12 E02 tasks and24 E03
method cases per machine, using reviewed recording-only source61a8135 and the
10 N v2 bank. Both failures and earlier packages are retained.
`force_successor_plan.py` and `force_screen.py` implement the subsequent fixed
screen and final-plan generation. `--allow-speed-only` records this explicit
internal-force waiver; it does not relabel physical compliance. Regenerating
`zero_pcg_plan.py` preserves the installed force stage and any final force plans.
The current merged inventories list held original-force definitions and omit
their four cases per target: GTX700/RTX713/Jetson368/Gazebo27 prepared cases.
Passed task-suite evidence is reused across families only in the same campaign
and machine with identical source pin, native module, CUDA smoke, fixtures,
helper and bootstrap; family canonical preflights remain required. Reuse writes
a reference receipt, never a second cache or failed-evidence pass.

Saved-analysis updates live in `analysis/results_review.py`, `analysis/f01ab.py`
and `analysis/control_choices.py`. The historical BB report combines all 100
pairs, including 11 B16 crash outcomes, while retaining raw error/unavailable
fields. BA seeds-alone is dropped; own-seed-to-output progress remains.
Full SO(3) errors, task-colored timing histograms and PCG cap shares supplement
the original reports. The N128 Jetson discrepancy remains a limitation without
a diagnostic replay.

## Earlier prepared inventory

The sections below retain the original family commands and protocol context.
Their old pinned plans are not substitutes for the successor cohort. The
historical `campaign.json` selects remaining RTX5060/internal T01 and F01ab
measurements at source `502fa29`; completed GTX1070/Jetson exclusions in that
plan do not define the restored-solver campaign. `plan.json` remains a legacy
pilot, never the new campaign default.

## Combined machine plans and transfer package

`package_plans.py` combines the explicitly listed prepared study plans without
changing their cases or scientific settings. Its inventory lists missing studies
and retains each source plan's prerequisites. Internal desktop runs, Gazebo
runs and Jetson proposals remain separate; physical scripts never enter a queue.
The generator is still being extended as the remaining study adapters finish.

```sh
python3 experiments/package_plans.py
python3 experiments/run.py --plan experiments/merged/rtx5060_desktop.json --machine rtx5060 --dry-run
python3 experiments/run.py --plan experiments/merged/rtx5060_gazebo.json --machine rtx5060 --dry-run
```

After the final plan/code checkpoint, create a fresh transfer package:

```sh
python3 experiments/package_plans.py --package .experiment-work/prepared-transfer
```

The package inventory gives the runner branch to clone from `sources.bundle`.
Extract `runner-assets.tar.gz` into that clone, set the local machine/Python
configuration and inspect the explicit plan before running. The runner manages
source checkouts and shared builds. Working-tree measurements, build caches and
local settings are excluded; exact Git source history includes any historically
tracked objects. The GTX1070 batching continuation still needs its retained native
module at the declared path to reuse that machine's existing readiness evidence.
Jetson plans require the agreed duration/storage discussion before execution.

## What the runner does

- Runs pending cases sequentially, with one relevant family preflight when
  configured, reusing unchanged successful preflight evidence on resume. Preflight failures block that family's cases, not other studies.
- Keeps scientific negatives as completed measurements. Execution failures keep
  their logs and partial data; subsequent independent cases continue.
- Resumes completed cases. `--retry-failed` explicitly retries failed/blocked
  work in fresh attempts; previous attempts remain intact.
- Shares compatible builds and frozen inputs. It checks input integrity when
  loading and native artifacts once per invocation, without re-running input
  generation or comparing environment versions as scientific gates.
- Keeps Ctrl-C meaningful: stops the active process group and records interruption.
  An optional `timeout_seconds` is a process hang guard, separate from simulated
  task duration or control deadlines.

Default directories are `.experiment-results/` and `.experiment-cache/`.
An ignored `experiments/local.json` may set `machine`, `python`, `output`, `cache`,
and optional `sync_to` (an rsync destination). Keep results and cache separate;
worktrees of this repository share one runner lock in the common Git directory.
Separate clones and manual GPU jobs must also be kept separate from measurements.

```json
{"machine":"rtx5060","python":"/opt/gato-venv/bin/python3"}
```

## Development workflow

Comparisons should be **good enough for the scientific question**. Use sensible,
fairly loose tolerances and check that behavior and conclusions are similar.
High numerical precision, exact equality and bitwise determinism are not
required. Small numerical differences alone should not block an experiment or
start another investigation. Physical-safety and task-success gates remain
separate. Preserve original failed checks rather than changing their tolerances
after the result to turn them into passes.

1. Start with the experiment's existing thesis question and retained evidence.
   State the comparison, required measurements and intended figure/table briefly;
   reuse a working study when it answers the question.
2. Reuse the existing scientific worker and source revision where suitable.
   Several experiments can share one source/build. Create a separate source
   branch only when actual runtime changes require it; do not create a new
   manager or framework for each family.
3. Implement the smallest script/adapter using the interface below. Keep raw
   measurements and failures, write a compact summary, and provide saved-data
   analysis. Build/source/cache directories never belong in collected results.
4. Add explicit cases, source/input pins and a protocol revision to the plan.
   Preserve historical cohorts. Select only pending work; completed results can
   be imported without repeating measurements.
5. Use focused checks for changed setup, recording and failure handling, and
   one focused review for a substantive implementation batch. Reuse passed
   checks when their relevant code and modules are unchanged. A scientific
   negative is evidence, not a reason to tune or retry until success.
6. Before a large run, use a short sensibility preflight where relevant. Reuse
   an unchanged successful preflight. Keep actual task and physical gates;
   package-version equality and harmless floating-point differences are not
   scientific gates. Inspect the explicit plan with `--dry-run`.
7. Run locally on the selected machine, sequentially on its GPU. Preserve an
   individual failure and continue independent cases. Shared setup/preflight
   failures block only dependent work. Changes to task settings define a new
   recorded cohort, never a silent rescue of a failed one.
8. Collect raw results separately from builds; regenerate plots/indexes without
   rerunning measurements. Update the thesis-question result ledger with paths,
   outcomes and practical limitations.

Keep the runner **and the experiment harness** simple. Fix demonstrated faults
that compromise the study; do not turn experiment development into solver repair,
exhaustive certification, repeated artifact audits or a general-purpose framework.

## Add an experiment

Implement a normal script accepting `--case FILE --output DIRECTORY`. Read the
case JSON, save raw measurements, and write a small `metrics.json` object. Exit
zero for a completed experiment, including a negative result; exit nonzero for
an execution failure. Keep task success, physical compliance and numerical
results separate. Preserve unavailable values as null with an explanation.

A plan entry supplies `id`, `research_question`, `protocol`, `source_commit`,
`machines`, `command`, and `cases`. Optional `build` has command arrays and artifact
patterns. `build.inputs` pins only the runner-relative helper files actually used
by that build. Optional `preflight` has one command array; `group` shares it between
experiments with identical source/build/input/config. Commands run without a shell.
Available substitutions are `{python}`, `{source}`, `{runner}`, `{build}`,
`{cuda_arch}`, `{case}`, `{output}`, `{inputs}`. Preflight gets its own output folder.

Case identity is the experiment, protocol, source, frozen input hashes, machine,
timing mode/backend and case parameters/repeat. Environment versions are provenance. Unrelated
package, runner, prose or plotting edits do not make completed cases pending.
**Bump the protocol when changing scientific adapter behavior or build settings
without changing the pinned source or inputs.** Resume is intentionally not an
inference about whether arbitrary code edits affect science.

Use `recording.RawRecorder` for repeated raw events. It writes bounded array
chunks and append-only metadata; `close()`, a context manager, or `manifest()`
flushes partial chunks on handled failures. A hard kill may lose the current
buffer (normally at most 64 events / 16 MiB, plus one large event). Earlier
chunks survive. `iter_events()` reads both this format and historical per-event
NPZ recordings. It retains arrays, dtypes, shapes and nonfinite values.

## Collect existing results and export measurements

After copying machine result directories into one collection, rebuild its index:

```sh
python3 experiments/collect.py --output .experiment-results
```

For the manually fetched Jetson F01ab archive, extract it first, then import the
portable directory. It already contains results, inputs and provenance without
build/source/cache trees:

```sh
python3 experiments/collect.py --output .experiment-results \
  --import-results /path/to/jetson-f01ab-results --name jetson-f01ab
```

Imports preserve original receipt/raw bytes under `_imports/<name>/`. The index
uses relative paths; no old absolute Jetson paths need to be rewritten. Importing
only indexes evidence by default. For an explicit migration of matching case
parameters under the same experiment ID, add `--plan FILE --experiment ID` to
register completed cases against that plan. It refuses ambiguous case mappings
or mismatched recorded timing/backend. Historical F01ab used per-cell experiment
IDs; its index-only import is sufficient because Jetson F01ab is excluded from
the current queue. Do not manufacture new runner receipts for old results.

`imports.json` registrations prevent reruns even if their data are temporarily
unavailable. A corrupt registration file stops execution with an error instead
of silently forgetting completed work. Fix the index from preserved evidence;
do not delete it just to make the queue run.

```sh
python3 experiments/collect.py --output .experiment-results \
  --export experiment-results.tar.gz
```

Export contains only the result collection. Build/cache/source directories are
rejected if accidentally included. `index.json`, `summary.csv`, and the generated
README show all attempts, including errors and partial metrics. Export/collection
can be repeated independently of measurements. The generic index carries thesis
question, source, backend and timing mode; family-specific raw formats and plots
remain local to each study. The saved-data F01 tool reads both historical per-event and new chunked recordings:

```sh
python3 experiments/analysis/f01ab.py --index .experiment-results/index.json \
  --output .experiment-results/analysis/f01ab
```

It writes all samples, per-cell/category summaries, solve-time histograms, and
reciprocal-frequency views. Warmups remain labeled and excluded from aggregates;
failed/nonfinite samples remain visible. It never pools distinct attempts or
cohorts. No universal scientific schema is required.

## Thesis exports and laptop plotting

Use `/workspace/GATO/thesis_results/<study>/` as the finalized export destination
for desktop/laptop sync. Generate thesis figures and tables on the desktop,
where full raw recordings remain with a separate backup. Keep raw recordings,
builds and source caches outside this directory. The full result archive from
`collect.py --export` belongs in raw-data transfer/backup, not thesis sync.

Export each finalized Markdown report as a self-contained PDF beside its
editable source in `thesis_results/<study>/`. Embed the important figures with
captions, key tables, findings and limitations so the essential results can be
read on another device without following local filesystem links. Regenerate the
PDF when the report or included figures change. Use an ordinary Markdown-to-PDF
tool and briefly inspect the output for missing images, unreadable figures or
clipped content; no separate publishing framework is needed.

Where straightforward, also export compact plotting data and its plotting script
for laptop edits. Preserve individual samples needed for distributions, units,
cohort identities and failure labels. Include source runs/raw locations,
protocol, script revision and regeneration command in a small provenance file
or text note. Use relative paths within the export. Analyses needing full raw
recordings run on the desktop and refresh the figures/tables.

Start with existing analysis outputs; do not introduce a new export framework,
universal data format or requirement to copy all raw data for plotting. Record
finalized export paths alongside raw locations in the main `experiment_result.md`.
Sync and backup configuration are separate from this directory policy.

## BB integration

`adapters/bb.py` runs one frozen pair/B1-or-B16 episode directly, with the same
candidate policy and mode B clock. Its recording hooks live at source commit
`4f6c8fd4be428c135fc16564c5dd56eb61ce5660` (branch
`experiment/bb-recording-simple`); historical worker defaults are unchanged.
Use case parameters `{"pair_id": 2342, "batch": 16}` and supply frozen `--pool`
and `--protocol` paths from the plan's declared inputs, plus `--source`, `--build`,
`--case`, and `--output`. Use `timeout_seconds: 600` for the existing process guard.
An optional canonical preflight uses `--preflight-input FILE --batch 1` (or 16),
returning nonzero if either task success or physical compliance fails.

BB uses the separate explicit plans described below. Its old 23 completed
episodes and retained failed episode remain untouched; the GTX1070 continuation
selects only the 176 pending episodes and links the earlier cohort. The runner handles independent failures;
there is no second campaign manager.

## F01c same-candidate timing

The separate `f01c.json` plan selects only the 15 pending RTX5060/internal cells:
N32, B1/2/4/8/16, budgets SQP/PCG 1/100, 5/100 and 10/800. Each cell retains the
same six frozen inputs, 16 candidate pool, three repeats, disposable public BSQP
owner and explicit dual/rho/zero-force reset per call. B1 supplies the serial
16-candidate baseline. The middle-budget B16 cell includes six extra duplicate
and permutation checks. All 1695 calls, including 15 excluded unrelated warmups,
remain identifiable. Canonical mode A preflight runs before this family; the
runner continues independent cells if one worker fails.

```sh
python3 experiments/f01c_plan.py
python3 experiments/run.py --plan experiments/f01c.json --machine rtx5060 --dry-run
python3 experiments/run.py --plan experiments/f01c.json --machine rtx5060
```

Generation preserves `campaign.json` and copies the existing F01c frozen bank
without regenerating seeds or CPU screening. Native source remains `502fa29`;
the build configuration matches the existing F01ab/T01 campaign for shared
modules. Scientific collector source is `9f2ab6f`: `adapters/f01c_collect.py`
retains its exact `jobs` and `collect` functions with the episode helper injected,
and `adapters/f01c_diagnostics.py` is byte-identical. Original source hashes and
the actual new adapter hashes are recorded separately. Protocol `simple-f01c-v1`
identifies the new recording/scheduling adapter; the old release/family-hold
framework is replaced by the current runner's preflight and independent cases.

GTX1070 F01c is completed and reviewed: import its historical results instead
of rerunning them. Jetson is excluded from this plan; its target preparation and
duration/storage discussion precede execution. Jetson F01ab remains completed
and awaiting Felix's manual collection.

Saved-data analysis accepts old per-event and new chunked raw recordings:

```sh
python3 experiments/analysis/f01c.py --index .experiment-results/index.json \
  --output .experiment-results/analysis/f01c
```

Solve time is primary and its reciprocal is the secondary figure. Compare total
time for the same 16 candidates across batch sizes; report setup/recording and
whole-collection time separately. Existing GTX1070 outputs differ materially
between scalar and batch solves, so speedup alone does not establish equal
quality, task success or physical feasibility. Preserve failed/missing calls,
actual work, duplicate/permutation differences and repeat dispersion.

## GPU resources, stage timings and sustained response (F04/F05)

`resource_profiles.json` prepares the 38 pending RTX5060 workers: eight resource
queries, 18 fresh-process first responses, six stage profiles and six sustained
blocks of 120 seconds each. GTX1070 results are completed and excluded. Stage
OFF modules share a four-horizon build; stage ON has its own N32 build. Canonical
mode A reaching checks the ordinary module; the original three-case task suite
and CUDA access probe check the stage-profile module. Both retain failures.

```sh
python3 experiments/run.py --plan experiments/resource_profiles.json --machine rtx5060 --dry-run
python3 experiments/resource_profiles_monitor.py --plan experiments/resource_profiles.json \
  --machine rtx5060 --output .experiment-results/resource-conditions-001
python3 experiments/analysis/resource_profiles.py --index .experiment-results/index.json \
  --output .experiment-results/analysis/resource-profiles
```

Use a fresh monitor output directory. The wrapper records passive clock,
temperature and power observations outside solve timers. The separate
`resource_profiles-jetson-proposed.json` retains the same approved coverage;
discuss target duration/storage before its execution. A CPU preparation check
is not a native build or target qualification.

## Batching quality and computation-paid tasks (BA/BB)

Use the explicit machine plan, not the historical manager. The GTX1070
computation-paid continuation (`bb_gtx1070.json`) selects only 176 pending
episodes and links the original 23 completed episodes plus one retained
execution failure. The failed episode is not retried. `bb_rtx5060.json` selects
the full 200 matching episodes. Both use mode B and the existing 600-second
process guard. Each failed case keeps its data; independent cases continue.

```sh
python3 experiments/run.py --plan experiments/bb_gtx1070.json --machine gtx1070 --dry-run
python3 experiments/run.py --plan experiments/bb_gtx1070.json --machine gtx1070
python3 experiments/run.py --plan experiments/bb_rtx5060.json --machine rtx5060 --dry-run
python3 experiments/run.py --plan experiments/bb_rtx5060.json --machine rtx5060
```

The GTX1070 plan copies and verifies the recorded native module from its retained
local path and reuses matching passed readiness. Preserve that module when
moving this plan to another checkout. RTX5060 builds its own module and runs
canonical B1/B16 readiness. The separate `bb_jetson_proposed.json` prepares one
pass through the same 100 pairs; Jetson repeat/duration discussion remains due.

The computation-ignored quality comparison uses `ba_gtx1070.json` (100 pending
nominal-compute B1 episodes) or `ba_rtx5060.json` (300 default/nominal-compute
episodes). These are mode A. The nominal arm gives B1 SQP80 versus B16 SQP5,
both PCG120; actual work and elapsed time are recorded, not asserted equal.
Its SQP80 canonical preflight is separate from the reused default readiness.
Use the same explicit `run.py --plan ... --machine ... --dry-run` then run
command pattern. Retained GTX1070 default comparisons are not repeated.

`analysis/ba_seed.py --raw PATH --source PINNED_SOURCE --output NEW_DIRECTORY`
reads the existing per-step candidate/output files without solving. It exports
own-seed/output merits, endpoint errors, predicted-knot bounds and nominal-prefix
diagnostics. These are not seed-only closed-loop success or convergence claims.
Same-candidate serial/batched solve timing comes from F01c, with its distinct
input and budget scope. Historical raw files and failures remain unchanged.

## Model consistency and bounded control choices

`model_checks.json` runs the original eight-state, 351-call model/orientation/cost
probe on RTX5060. GTX1070's completed 2/560 numerical negatives stay in the
historical evidence. A completed numerical failure is a result; an incomplete
probe or missing raw manifest is an execution failure. Tolerances are unchanged.

`control_studies.json` prepares 38 cases on each desktop: running position and
orientation costs, two historical cost-switch distances, existing final approach
with one small starting-state perturbation, damping/speed-limit/terminal weights,
nominal holding and warm-start choices. Reaching uses an explicit eight-second
observation so settling switches can act; zero-dwell rough task success is
reported separately from observation timeout and physical compliance. These
reuse the frozen two reaches and 24-second figure eight. Holding is a finite
observation, not an equilibrium certificate. The extra gripper holding case uses
the existing include-gripper URDF with its six finger joints locked at the
0.8028514559173916 rad driver upper limit and the declared mimic relationships.
It pairs with the shared nominal-zero holding baseline. These are the wider
nominal-zero and narrower-upper geometric configurations; only the Pinocchio
plant changes, while the native optimizer keeps its nominal model. This is a
small intentional inertial mismatch, with no physical gripper motion. The
original target task suite and canonical mode B preflight precede execution.

`analysis/control_choices.py --raw RESULT_COLLECTION --output NEW_DIRECTORY`
exports per-attempt outcomes and reaching traces, preserving retries, failures,
source/module identities and separate rough-success/physical outcomes.

```sh
python3 experiments/run.py --plan experiments/model_checks.json --machine rtx5060 --dry-run
python3 experiments/run.py --plan experiments/control_studies.json --machine rtx5060 --dry-run
```

The same commands without `--dry-run` execute the selected plan on that machine.
CPU preparation is not target qualification. Shared readiness supports the
repository's namespace package and requires the selected native module.

## Saved visual demonstration review

After internal or Gazebo recording, render the actual feedback path, active
goal/tool axes, position/axis errors and the existing arm skeleton animation:

```sh
python3 experiments/analysis/demo_visual.py --raw RESULT_COLLECTION --source PINNED_RUNTIME_SOURCE --output NEW_DIRECTORY
```

The output keeps attempts, backends, task/settling/physical outcomes and explicit
skipped/held selections separate. Every feedback row is exported; the GIF uses
at most 220 display frames. These drawings do not establish swept collision
clearance. Add `--no-gifs` for paths/error traces only. The source supplies the
recorded model and existing `tiago_reach_target.py` animation helper; no native
solver or robot motion is invoked.

## Remaining Jetson proposals

`jetson_remaining_plan.py` copies unchanged study definitions into
`t01_jetson_proposed.json` (six reaching/tracking tasks),
`f01c_jetson_proposed.json` (15 same-candidate timing cells) and
`ba_jetson_proposed.json` (300 batching-quality/nominal-compute episodes).
Together with the existing batching, resource and warm-start proposal plans,
these are prepared coverage, not execution authorization. Inspect retained
Jetson timings, estimate duration/storage and discuss any shorter protocol with
Felix before execution. Repeats and coverage have not been silently reduced.
Jetson cold/warm timing (F01ab) is completed and remains excluded.

## Visual demonstration and collision monitoring

`demo_internal.json` prepares the five-reach vertical-square sequence and three
upward figure eights. `demo_gazebo.json` runs the same four tasks as one live
sequence, holding remaining motion after an execution or handback failure.
The 30 cm square uses inward corner orientations and a normal centre approach;
each reach has a 15-second timeout and the existing settling parameter switch.
Rough success is reported separately from stricter settling. Endpoint IK and
collision screens passed; full-motion simulation and Felix's review remain due.

The manually reviewed physical script is
`experiments/deployment/demo/demo_robot.py` at source `2c7e2b1` on the deployment
branch. It is outside the runner, defaults to preview, prints named parameters,
requires matching canonical preflight evidence and confirms before robot motion.
Backend, selected/skipped tasks and fresh raw paths are recorded. Do not run an
unreviewed robot script as part of a desktop plan.

`collision_monitor.json` contains four CPU cases: original/wrapped geometry with
all-pair exact distances or the runtime's first-violation margin query. It bundles
the used geometry. These workloads differ deliberately; exact timing includes
fresh collision-data allocation. Synchronous safety-worker timing excludes
process/queue overhead and uses the runtime margin implementation in both query
cases. `collision_monitor_jetson_proposed.json` retains the same coverage.

`simulated_faults.json` preserves the original live fault protocol: a leading
baseline preflight and 20 remaining trials, including the trailing baseline.
It needs the existing local PAL simulator, loopback DDS, an actual successful
canonical Gazebo preflight and fresh PAL controller observations. Configure the
`simulator_faults` section of ignored `experiments/local.json` as described by that plan's
`prerequisites`; the runner does not launch host Docker containers or reset the
simulator. After a trial fails to restore control, a separately logged simulator
restoration is needed before another trial can pass its motion prerequisites.
Historical publication/restoration failures remain results, not solver fixes.

Saved-data commands do not execute motion or solve again:

```sh
python3 experiments/analysis/control_choices.py --raw PATH_TO_CONTROL_RESULTS --output NEW_DERIVED_DIRECTORY
python3 experiments/analysis/fault_envelope.py --raw PATH_TO_FAULT_RESULTS --output NEW_DERIVED_DIRECTORY
python3 experiments/analysis/collision_monitor.py --raw PATH_TO_MONITOR_RESULTS --output NEW_DERIVED_DIRECTORY
```

Control plots retain every attempt separately. Fault response and observed motion
are descriptive; unavailable state age, clearance and certified braking bounds
remain null. No measured hardware-safety guarantee is inferred.

## Relevant checks

```sh
python3 -m pytest -q experiments/test_*.py
```

These exercise real temporary Git/subprocess runner behavior and CPU recording,
collection and adapter wiring. They do not qualify native builds or new GPU
measurements. The family preflight remains part of actual execution. Controller,
timing, warm-start or task-behavior changes require their relevant task regressions;
packaging and recording-only work does not repeat a task campaign.

## Cylinder costs, supplied routes and moving obstacles

`cylinder_studies.json` prepares nine cases on each desktop: one mode-A
cylinder-cost check, two mode-B supplied-route cases (B1/B8), and six mode-B
moving-cylinder cases (three retained scenes, B1/B16). The Jetson proposal
contains the six moving-cylinder cases and still needs the duration discussion.
The native objective is position plus cylinder cost; its orientation slots
encode obstacle weights. Ordinary orientation-on readiness and the positional
cylinder preflight remain separate. Tool-center sampled clearance does not
certify whole-arm or continuous collision avoidance.

Analyze a saved collection without executing tasks:

```sh
python3 experiments/analysis/cylinder.py --raw RESULT_COLLECTION --output NEW_DIRECTORY --source PINNED_SOURCE
```

Each attempt retains its task/physical outcome, tool-clearance and motion traces,
and own-seed versus optimized-output differences. Retries are separate points.
`--source` is optional and supplies matching model limits for descriptive traces.
Historical failures and unchanged supplied trajectories remain evidence; the
new computation-paid cohort does not replace them. QDLDL (B02) is not selected
for this campaign, following Felix's decision.

## Force estimation and known disturbances

`force-gtx1070-proposed.json`, `force-rtx5060-proposed.json` and
`force-jetson-proposed.json` pin reviewed source `c0032ba`. Each prepares
24 original/adapted one-tick timing/memory comparisons with identical frozen
hypotheses and six 24-second tracking tasks: no estimator/original/adapted,
each with zero force or a 2 N body-fixed local-x force at the last arm joint.
Each method/disturbance task has its own canonical mode-B preflight.
The first-input timing records also supply correspondence evidence (E01),
without duplicate executions. Native solve and complete algorithm timing are
separate; memory snapshots describe observed allocation boundaries, not a
continuous peak. Disturbance tasks charge complete algorithm wall time while
the previous plan plays. These are body-fixed forces, not constant world forces.

```sh
python3 experiments/analysis/force.py --index RESULT_COLLECTION/index.json --output NEW_DIRECTORY
```

This exports per-attempt timing/memory, task and paired-correspondence tables.
Failures remain visible and duplicate completed attempts are not silently
selected or pooled. Production-float validation is accepted; the retained
additional double-precision discrepancy is closed without a tighter gate.
Jetson duration/storage discussion remains required before execution.
