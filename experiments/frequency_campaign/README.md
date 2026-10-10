# F01ab held protocol proposal

This directory contains design only. No runner plan, worker, build, GPU command,
measurement or execution permission is supplied. `protocol_proposal.json` is an
exact proposed episode schedule; `duration_storage_worksheet.json` is a
recomputed historical receipt and transparent planning scenarios. Sources are
approved production16b3371 and campaign12dc7cc; no historical solver baseline.
The independent initial design review is
`/workspace/gato-campaign-recovery-20261010/f01_design_review.json`; original
proposal bytes remain in external `f01-design-initial-proposal/`.
The authoritative revision is `/workspace/GATO/EXPERIMENT_PLAN_REVISION_2026-10-10.md`,
section4. This is a new `.6s fixed-slot/four-start` timing subset. It requires serialized
input freeze/dynamic screening, implementation review, and Jetson discussion.
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

Offer a shorter option for Jetson discussion: use12 episodes, the firsttwo0/.05 witness
starts only, same20updates and all64cells. Counts12cold/222warm/6switch percell.
After the required Jetson coverage/runtime discussion, freeze the same selected
design on all three GPUs for matching counts. It reduces cold
coverage and supports only bounded histogram interpretation; no budget cell is
removed. The coordinator presents these concrete choices with calibrated
duration/storage estimates for the required Jetson discussion. Fixed slots
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

Current `BSQP.get_stats` retains SQP/PCG counts/times, KKT flag, merits, step
size and line-search count. No explicit stop_reason or PCGbreakdown field exists
in current public interface. Preserve all telemetry; derived cap-hit/early-return
labels must be marked derived and cannot identify a breakdown. This is an
explicit measurement limitation: retain native_stop_reason=null and
native_breakdown=null with reasons that the current public API reports neither.
Available cap counts/flags are observed proxies only, not actual stop causes or
breakdown claims. Do not presume the native implementation has a reportable
breakdown mechanism or require instrumentation. No solver instrumentation,
NaN-cause proposal, repair or invented certificate without separate authorization.

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
desktop runs before turning this worksheet into a Jetson duration proposal.
Record Jetson power/thermal conditions, builds, initialization, solves, plant,
recorder and archival time separately. No Jetson execution before shorter
protocol discussion with Felix; no silently pruned cells or shortened episodes.

## Bounded stopping points

1. Corrected independent CPU design re-review; root checkpoints the held
   proposal and presents concrete24/12episode Jetson coverage/runtime choices.
2. Independently review prepared CPU dynamic screening and serialized bank/
   reference/config hashes; root selects maximal24 or firsttwo-start12 variants;
   independently review hashes/counts/timing loop/native telemetry collection.
3. Implement only that frozen design; validate meaningful CPU failure/count/
   sampling tests, isolated build identities, required task suite if applicable;
   re-review corrections and checkpoint, then explicit dry-run.
4. Root-owned sequential desktop timing cohort and matched5060; preserve/report
   surprising failures before changing scope. Update duration/storage proposal
   using actual call/IO/module/thermal measurements. Discuss Jetson first.
5. Complete approved64cells/machine with outcomes/coverage visible; plot and
   interpret before asking about F01c representative budgets. No continuation
   into parameterization viability, learning, force task, solver repair or robot.


## Frozen CPU bank preparation

Command: `/opt/gato-venv/bin/python3 experiments/frequency_campaign/freeze_inputs.py
--output experiments/frequency_campaign/inputs`. CLI refuses existing output
directories. Inputs contain copies of exact sourceT01/endpoints/model/generator,
`arrays.npz`, `bank.json`, `witnesses.npz`, `screening.json`, provenance and
`SHA256SUMS.json`. Both variants remain held; no runtime/controller/native loop
is implemented. References retain the original T01globalanchor across starts.

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
