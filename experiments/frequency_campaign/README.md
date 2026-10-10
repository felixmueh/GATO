# F01ab — frozen short12 protocol and held runner integration

Current implementation: the reviewed mode-A collector now has a lazy native
factory, same-mode canonical A readiness worker, normal isolated build adapter
and single-cell runner entry. Felix accepted the FIRST telemetry implementation
without a numerical-equivalence gate; its earlier negative comparison remains
descriptive evidence. The real deployment manifest and all queued entries remain
held pending reviewed committed source/build receipts, own-machine canonical A
readiness and exact plan. No native execution was performed by this CPU batch.
Use [RUNNER_README.md](RUNNER_README.md) for bounded receipt/command details and
`python3 -B experiments/frequency_campaign/run.py --dry-run` for the exact64-cell
short12 coverage. [held_plan_template.json](held_plan_template.json) contains
readiness plus all64 cells with every entry disabled; it never changes the
legacy default queue. Matching5060/internal and Jetson-specific release gates
remain separate.

This directory contains design only. No runner plan, worker, build, GPU command,
measurement or execution permission is supplied. `protocol_proposal.json` is an
exact proposed episode schedule; `duration_storage_worksheet.json` is a
recomputed historical receipt and transparent planning scenarios. Sources are
approved production16b3371 and campaign12dc7cc; no historical solver baseline.
The independent initial design review is
`/workspace/gato-campaign-recovery-20261010/f01_design_review.json`; original
proposal bytes remain in external `f01-design-initial-proposal/`.
The authoritative revision is `/workspace/GATO/EXPERIMENT_PLAN_REVISION_2026-10-10.md`,
section4. The frozen `.6s fixed-slot/four-start` bank remains a preserved
superset; Felix selected common short12/two-start coverage after the Jetson
runtime/coverage discussion. Source/input review and actual machine release
remain required; that F01 coverage decision is not reopened.
Fixed-slot collection is a routine design choice within the revision; the exact
approved goal/speed set does not need approval again.

## Concrete proposal for Felix

Use all64 cells on each GPU: N16/32/64/128 × SQP1/2/5/10 × PCG100/200/400/800,
B1, modeA. Keep30ms prediction/feedback and3ms plant. Horizon duration is
.45/.93/1.89/3.81s; this is fixed-spacing, changing-duration comparison only.
Keep each approved T01 task's public preset exactly except the two effort caps.
Public PCG120 is explicitly overridden;100 is the original solver default.

Propose24 distinct start/task pairs percell, each20solves/20executed30msupdates
(.6s). Four zero-velocity starts lie at0/.05/.10/.15 of the existing goal1
joint-endpoint witness displacement from the approved start. All exact vectors
are in the JSON, serialized as exact little-endianfloat32. FK/joint-limit checks and sampled
8sRNEA screening now pass (scope below). All first targets are outside5cm; new goal1 start distances are
100.636/95.959/91.240/86.482mm. Three are below the original T01.1m minimum,
so this subset is not the original T01 protocol or canonical task evidence. Twelve reaching episodes
pair each start with cyclic approved goals1→2,2→3,3→1; first target remains
active for10updates, second for10. Twelve tracking episodes pair each start
with24/18/12s approved periods, using only the first.6s of each unchanged public
reference. This is a short timing subset, not a full-period tracking verdict.
The first plan is standard constant-current-state/zero-control cold seed;
subsequent plans are shifted previous solutions. Each new episode restores the
entire public initial solver state includingrho; within episodes use the public
rho adaptation and no reset. Goal switch preserves the shifted warm input.

For every completed cell the exact counts are24cold,444warm,12goal-switch,
480measured solves. A fresh cell process first performs one unrelated-input
CUDA warm-up, retained and excluded from all three categories. A concrete
warm-up input must be frozen before release (suggest witness fraction.025 with
approvedgoal3; never a scheduled episode input). Recreate controller/solver
state afterward; do not call an episode first solve the CUDA warm-up.

The stopping decision matters: public reaching advances/stops after success,
which makes warm/switch counts cap-dependent. The proposed timing worker uses
fixed slots and fixed switch time; it observes success without stopping.
This explicitly defines the timing-study collection policy. Implement it through
the ordinary review/checkpoint cycle; do not silently pass it through
`run_mpc_goals` with modified tolerances.
Record first rough-success times, end errors and physical compliance for each
.3s goal segment. A segment that never meets the gates is a short-subset failure,
not a claim that its8s approved T01 task fails. Record tracking subset errors;
full-period T01 success is unmeasured here. A .3s/.6s subset failure may result
from its short window as well as optimizer behavior; do not attribute it to
inadequate budget or infer full-period tracking quality. Never hide this
difference in labels.

Felix approved the shorter common design:12 episodes, the firsttwo0/.05 witness
starts only, same20updates and all64cells on allthreeGPUs. Counts are
12cold/222warm/6switch percell. The Jetson coverage/runtime discussion is already
complete for this F01 choice. Reduced cold coverage supports only bounded
histogram interpretation; no budget cell is removed. Actual target runtime/
storage calibration and machine-specific release remain pending. Fixed slots
resolve cap-dependent public success-stop counts without altered tolerances or
replacement samples. No success-driven replan/extra-sample arm is proposed.

## Implementation design after input freeze

Reuse `MPC_GATO` construction/public solver presets, `OfflineRecorder.solve`,
`RawRecorder`, `sample_reference_horizon`,
`shift_packed_trajectory_warm_start` and
`sample_packed_trajectory_control`. A family-local readable fixed-slot modeA
loop applies the returned plan immediately, advances exactly30ms with ten
3msRK4steps, then shifts by30ms before the next solve. Match inherited public
initialization, control sampling,plant andrho behavior; do not change production
solver/controllers, ROS or generated code. Any new shared controller change
requires its own independent review and mandatory `make test-tiago-tasks` with
audited test-only transplants. CPU acceptance must compare the family loop
against the inherited modeA loop before its first success/switch event, using
mocked plans that expose shifted-control and reference indexing errors.

Tracking windows sample original hashed T01 reference arrays at actual local
feedback times andNprediction times. First.6s plusN128preview ends by4.41s,
inside every existing full-period reference array, so no extension/truncation/
resampling change is needed; assert that bound and original reference hash.
The serialized `inputs/` bank freezes both maximal24 and firsttwo-start12
variants, all four roundedfloat32 starts, original goals/fullreferencef64 bytes,
a distinct fraction.025 warm-up state/goal3, exact category schedules and
source/model/generator hashes. Sampled CPU witness screening passes. Tracking
transient/dynamic feasibility remains unverified; record that limitation rather
than retune. Screening does not optimize seeds or change public initialization. Preserve exact approved pose
vectors/periods; no goal retuning,1msfallback or effort changes to rescue a cell.

Keep one runner case/process percapcell, one fixed pass, same lexicographic
N/SQP/PCG/episode order on every machine. No timing job overlaps another GPU
job. Per-machine exact module/build/source/input hashes and reviewed explicit
runner plan/dry-run precede release. N16/N128 build/import or nonsensical task
outcomes remain recorded results; root holds/report affected coverage rather
than substitutes a new task. Scientific rough/physical failure retains timings
and every scheduled episode; execution/nonfinite plant failure retains partial
raw data and missing slots and holds affected cell/family for review. Do not
continue integration from a failed state merely to hit a count. A canonical
N32 modeA family rough-reaching readiness receipt is required on each machine;
it is not a viability requirement for all64timing cells.

## Raw contract and telemetry limitations

Retain exact per-call state, reference window, shifted warm input, rho/effective
config/caps, returned plan, native stats, optimizer/activation times, wrapper
wall and native solve time separately. Record ten actual plant states/controls
perupdate, success checks, switched-goal event and failed-call input/error/wall.
Hash full reference arrays, starts, protocols, build/module receipts and every
raw file; a later hashed replay bank supplements these actual percell episodes,
and never replaces them. Use existing NPZ/JSONL recorder with all nonfinite
array/scalar bytes retained and raw copying/telemetry/IO outside solve timer.

The accepted FIRST optional experiment telemetry retains the existing native
arrays/counts/times/flags/merits/line-search data and reports actual PCG
mask/initial-rho/tolerance/cap paths plus the batch-wide SQP ratio/cap path.
Task success and physical compliance remain separate. Breakdown is explicitly
unsupported/null because the algorithm has no breakdown-stop branch; that
never means numerical breakdown was disproved. The earlier ON/OFF numerical
comparison is preserved descriptively under Felix's as-is acceptance, without
equivalence, causal or full nonlinear-certificate claims. Current release uses
pinned acceptance/source/build/reason/ON-suite evidence, not a numericalPASS
proxy. Raw `native_return` is saved after wrapper timing even when public
normalization raises; its argument references are labeled at return, while
`solve_started` retains exact pre-call wrapper state/reference/warm bytes.

## Analysis contract

Time in ms is primary and separately labeled native/wrapper. Plot cold,warm and
goal-switch mean/median/p95 bycapcell/machine/task-kind. Secondary figures show
1000/mean_ms and1000/median_ms with definition; optional per-sample reciprocal
has a different label. Neither is an actuation rate or deadline gate. Produce
one category-colored solve-time histogram percell/machine, common category
colors/binning recorded across matched machines, finite/nonfinite/missing counts
printed, no presumed bimodality. Preserve rows with task/execution failures;
no slow-sample filtering or pooling goal-switch into warm. Derived plots belong
outside attempts, with input/script hashes, PNG/PDF and dataCSV. Add a family
plot script beside existing `experiments/analysis` only in a reviewed laterbatch.

Eachcapcell produces its own trajectory and realized inputs; hardware comparisons
match episode definitions, not identical trajectories. Numerical noise may
produce materially different trajectories; show state/error/outcome differences
alongside aggregate timing. No exact-determinism audit. Fixed slots prevent
success-stop sample-selection differences but do not prove identical inputs.
F01c budgets/inputs are selected after F01ab; no selections made here. F04/F05
remain separate active studies, not silently incorporated in these samples.

## Jetson duration and storage

The worksheet rehashes the retained `mpc_stages.csv` and recomputes300B1stages,
16episodes,2076.331669s solve-stage total,mean6.921106s,median9.905285s,max12.409853s.
Important provenance correction: selected CSV/config fields sayN64/B1 for
all300rows although `run_id=jetson_n128`. The run label is not a module identity;
blank binary/source hashes in these rows leave native module identity unknown.
The linked saved summary confirms knots64/200SQP/two passes/PCG1000; CSV
`pcg_iters` denotes that budget while `pcg_cap` counts cap events. Do not describe
this as confirmedN128 or confirmedN64 module timing. This historical
cylinder/200SQP×2pass/PCG1000 protocol is not a TIAGo/F01 runtime predictor.
The separate historical `latency.csv` does containN12820SQP rows and serial/B16
collection timings; no linearSQP/PCG/N extrapolation from either is justified.

24episode option gives30784calls/machine including64warmups. Assumed average
solve wall10ms/100ms/1s gives5.13min/51.31min/8.55h solve-only; reusing the
historical6.921s average as a cautionary scenario gives59.18h, not a bound or
prediction. Twelveepisode option gives15424calls and2.57min/25.71min/4.28h;
historical-rate scenario29.65h. Builds/imports/1536or768episode initializations,
plant integration, raw event IO/hashes and thermal/upload time are additional.

Reserve1MiB per solve event-group plus128KiB perupdate as an intentionally loose
unmeasured storage allowance:33.82GiB (24episodes) or16.95GiB (12episodes),
double for a second archival copy. This is not a measured upperbound;
compression/native telemetry/event-file overhead can change it. Existing
recorder emits multiple files per solve/plantstep, so filesystemIO may dominate
fast GPU calls. Measure actual eventbytes and total process time from reviewed
desktop runs to refine target runtime/storage calibration. Record Jetson
power/thermal conditions, builds, initialization, solves, plant, recorder and
archival time separately. Common short12 coverage was already approved after
its Jetson duration discussion; actual machine release/calibration remain,
without silently pruned cells or shortened episodes. Jetson T01/BB repeat
counts are separate pending discussions.

## Bounded stopping points

1. Common short12/two-start coverage is approved after the Jetson discussion;
   full24 remains the immutable historical superset.
2. Preserve reviewed CPU dynamic screening and serialized bank/reference/config
   hashes, selected counts and timing/native telemetry provenance.
3. Implement only that frozen design; validate meaningful CPU failure/count/
   sampling tests, isolated build identities, required task suite if applicable;
   re-review corrections and checkpoint, then explicit dry-run.
4. Root-owned sequential desktop timing cohort and matched5060; preserve/report
   surprising failures before changing scope. Update duration/storage proposal
   using actual call/IO/module/thermal measurements. Prepare actual Jetson machine
   release using the already approved common short12 coverage.
5. Complete approved64cells/machine with outcomes/coverage visible; plot and
   interpret before asking about F01c representative budgets. No continuation
   into parameterization viability, learning, force task, solver repair or robot.


## Frozen CPU bank preparation

Command: `/opt/gato-venv/bin/python3 experiments/frequency_campaign/freeze_inputs.py
--output experiments/frequency_campaign/inputs`. CLI refuses existing output
directories. Inputs contain copies of exact sourceT01/endpoints/model/generator,
`arrays.npz`, `bank.json`, `witnesses.npz`, `screening.json`, provenance and
`SHA256SUMS.json`. The bank preserves both variants; Felix selected the common short12 subset.
The bounded collector/factory/readiness/runner path is implemented on CPU,
with real native deployment held by the reviewed release and machine gates. References retain the original T01globalanchor across starts.

Fifteen8squintic routes were sampled at2668points each (maximum spacing<=3ms):
four starts→all three original FK-derived endpoints, plus each endpoint→the
next in the cyclic goal sequence. Positive sampled q/v/u margins and necessary
endpoint velocity time bounds are retained perroute. EndpointFK matches the
unchanged approved pose targets. Witnesses are screening only: not optimizer
seeds, not .3ssegment feasibility, and not actual warm goal-switch states.
Collision, acceleration-limit enforcement, between-sample compliance and
tracking transient/closed-loop feasibility remain unqualified.

Validation: `/opt/gato-venv/bin/python3 -m pytest
tests/python/test_frequency_inputs.py -q` yields12passed with one existing
hppfcl→coal dependency warning. Tests cover exactreference/shape/dtype bytes,
firsttwo-start subset and switch categories, positive sampled margins, overwrite
refusal and preserved failed preparation, and rehashed malicious array/model/
config/schedule/witness changes. A fresh-process check proves the utility's
CPU screening imports no solver/controller/native modules. The first version
of that check assumed pytest imported none; repository conftest importsbsqp,
so the isolated-process check replaced that assumption without changing data.
Initial bankv1 before semantic validator strengthening is preserved externally
in `/workspace/gato-campaign-recovery-20261010/f01-input-bank-v1-prevalidation/`.
No input failure or numeric parameter change occurred; failed mutation/source
fixtures are test-only evidence. No GPU/native calls/builds or runtimecore edits.


## Concrete held runner commands

The normal runner compiles a build-owned source archive with one literal N
per horizon group, using the existing campaign preparation helper:

```
{python} -B {source}/experiments/frequency_campaign/build_isolated.py --source {source} --build {build} --python {python} --cuda-arch {cuda_arch} --N 32
```

This configures Release/N32/tiago_right/telemetryON; N16/64/128 groups use their
literalN. It writes source-ownedmodule and `f01-build-N32.json` receipts under
`{build}`, never edits private runner sources or shares builds across variants.
Root supplies actual reviewed committed source/module receipts after review.

The worker command consumes normal case/output/input paths:

```
{python} -B {build}/campaign-source/experiments/frequency_campaign/run.py --case {case} --output {output} --inputs {inputs} --manifest /absolute/reviewed-machine-release.json --manifest-sha256 REVIEWED_HASH --family-state /absolute/machine-family-state.json
```

A separate fresh N32 canonical-A worker adds `--readiness`; generic runner
supplies `GATO_MACHINE`. Relative manifest paths resolve under `--inputs`.
Attempt logs/runner receipts stay in `{output}`; full raw evidence goes into
fresh `{output}/f01-raw`, and final metrics are mirrored at the attempt root.
Plot selections use `f01-raw` and its exactmanifest hash. Injected CPU owners,
plants/observers/readiness executors are marked `evidence_kind=synthetic_test`.

The checked-in held template contains readiness plus all64short12 cells, four
horizon groups/16budgets each, one repeat/allthree machines. Every entry remains
disabled; replace reviewed commit/hash/manifest/family-state placeholders only
in a separate explicit root-reviewed plan. Current coverage display is CPU-only:

```
python3 -B experiments/frequency_campaign/run.py --dry-run
```

After root's exact committedplan/build/readiness review, inspect it with:

```
python3 experiments/run.py --plan /absolute/reviewed-f01-plan.json --machine gtx1070 --dry-run
```

No plan was enabled or generic runner/native/build executed by this CPU batch.
The live manifest remains HELD; existing accepted-first observer/ONsuite
qualification is retained, canonical A readiness is still unmeasured, and
matching5060/internal plus Jetson release remain pending independently.

Native solve exceptions also flush `native_call_failed` after the wrapper timer,
with argument arrays labeled at failure and an explicitly unavailable return;
the standard failed event and family hold follow without another solve.
