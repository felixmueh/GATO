# Ordinary campaign G0/G1 harness

This batch prepares V01 and the six approved T01 episodes from landed solver core
16b3371. It changes no solver, ROS overrun policy or physical robot path.
The true old-plan controller review and required frozen suite are complete. This
batch permits canonical V01 on GTX 1070 and RTX 5060/internal after the reviewed
checkpoint and matching explicit plan dry-run. T01 additionally requires each
machine's passing preflight from the same committed source and native build.
Jetson is excluded by the worker's
frozen machine guard before dependency/native imports.

`inputs/v01.json` freezes the canonical N32,30ms knots/public feedback,3ms plant,
qd0.1,vel-limit0.1,orientation0.1/2,8s,5cm/30deg/speedL1<=1,zero dwell and disabled
final approach. It reuses the exact frozen short fixture's100.636mm goal and
labels its3ms variant; the original1ms fixture is immutable. All effective
solver weights/caps, force/gravity, seed/warm-start and physical validation
settings are explicit. Known SB-001/SB-002 remain limitations; no solver
numerical certificate is inferred.

Q02's original frozen h=.004 protocol remains **FAILED 2/560**. The independent
saved-input CPU audit reproduces both coarse-step negatives while native analytic
gradients agree much more closely with finer references (about 4e-4/3.6e-5 maximum
differences for the selected failures). Native-versus-independent and scalar/batch
subchecks pass. This bounded method-negative explanation changes no tolerance,
step, source or original outcome; Q02 is not relabeled a full pass. Solver numerical
certification remains null beside SB-001/SB-002. `inputs/v01.json` links exact
hashes for `review_timing_v2.json`, the current-suite source receipt/log/summary,
and `/workspace/gato-campaign-q02-20261010/audit-once.json`.

## Recording and clock scope

The new Q04 controller executes the old active plan's prefix during the current
measured solve and activates the returned plan afterward with its control u0.
Optimizer-reference/feedback time and control activation time are separate;
active-plan age continues while computation is paid. The first bootstrap plan
executes for the nominal 30 ms and its initial solve time is retained but unpaid.
Timeout clipping is explicit; a result arriving after task completion is discarded.
Felix's exact 2026-10-10 decision was: "Implement old-plan execution during solve".
The corrected controller/helper passed independent source/103 CPU review and the
required original-mode Make suite once (3/3), with saved-array audit. Those checks
do not qualify canonical B tasks. The historical returned-plan/producing-duration approximation
is kept separate from current B evidence.

The controller timer surrounds `BSQP.solve` only. Its input copies, telemetry
access, output copies, recorder compression/hash/IO, controller bookkeeping and
RK4 are excluded from the B clock. `record_callback` must be called outside
that timer and must treat arrays read-only. Every started/completed/failed solve
and RK4 step gets an NPZ plus JSONL metadata, times, array mapping and hash;
finite and nonfinite values are retained. Solve events include state, reference,
warm start, per-lane output and native telemetry. Final/partial solve snapshots
retain selection metadata added after completed callbacks. Interrupted SIGINT
or SIGTERM runs finalize whatever was observed and return execution failure.
SIGKILL can interrupt the current write; prior event files remain readable.
Selection, plan activation/discard, completed-interval and goal-finished events are also retained
by the generic recorder (`solve_selected`, `plan_activated`, `plan_discarded`,
`interval_completed`, `goal_finished`). Provenance includes `python/bsqp/offline_mpc.py`.
Nonterminal goals pay the entire solve wall interval and check success/timeout at
actual arrival, with requested deadlines/end/lateness retained descriptively.
Only the final overall reaching task timeout or tracking end clips an interval.

`execution.npz` stores every returned RK4 state/control/time without filtering.
Started or failed RK4 records remain raw events. `metrics.json` checks the
executed rough reaching gates and sampled q/v/u limits separately. No output
plan nonfiniteness is dropped. Numerical certification is null. Wall/native
solve timing distributions and actual intervals/above-nominal shares are
descriptive; there is no strict deadline gate. Nonfinite timing samples produce
null statistics with a reason, not a filtered estimate. The worker's zero exit
includes completed scientific failures; thrown errors return nonzero. Both
hold the dependent family on that machine.

## Explicit runner deployment after review

Do not use or modify `experiments/plan.json`, which pins legacy7187cf8.
After the complete campaign batch is reviewed and committed:

```sh
python3 experiments/campaign/make_plan.py \
  --source-commit <full-reviewed-40-character-commit> \
  --output /tmp/campaign-v01-plan.json
python3 experiments/run.py --plan /tmp/campaign-v01-plan.json --dry-run
```

The generated plan is disabled until an explicitly reviewed release uses
`--enable-reviewed`. It freezes one attempt per desktop machine, the same1070
and5060/internal protocol, and exact protocol/mode-receipt/fixture hashes.
Never queue T01 without approved goals/speeds, Jetson without duration/
storage/shorter-protocol discussion, or any Gazebo/physical task in this plan.
Failed executions require explicit runner retry; no repeat-until-green.
The plan hash can be generated after commit, avoiding circular source hashes.

`build_isolated.py` archives the committed source to `{build}/campaign-source`
and uses `{build}/campaign-native` for CMake. The stock CMake output under
source/python/bsqp therefore stays inside the runner-owned **build**, while its
cached private source remains untouched. No shared binaries/build directories,
submodule regeneration or private cache edits are used. Build/source/config and
fresh module hashes are receipts/artifacts; the worker imports only that isolated
module. The root owns actual builds, CUDA smoke and required `make test-tiago-tasks`.

Retries validate the pinned archive identity and every archived source file
before reusing an extracted source. An interrupted extraction without a receipt
is preserved under the build's `preserved-build-attempts` before reconstruction.
Modified pinned source bytes fail explicitly and stay intact. Build retries
preserve previous module bytes, receipts and configure diagnostics, then force
a clean target rebuild so newer stale modules cannot be accepted as fresh.
This operates solely inside the runner-owned build directory. The worker also
enforces the frozen held status before dependency imports or controller work;
an approved changed status defines a newly hashed protocol cohort.

An A diagnostic is a distinct manual case with hashed completed failed-B
`metrics.json` as a declared runner input:

```json
{"id":"rough-reaching-A-diagnostic", "parameters": {
  "protocol":"V01", "offline_timing":"controller_dt",
  "diagnostic_after_failed_B": {
    "path":"experiments/campaign/inputs/failed_B_metrics.json",
    "sha256":"<failed-B-metrics-hash>"}}}
```

The worker requires exact original B configuration, same machine/internal
backend, a completed scientific negative and the copied input hash: rough-reaching
failure or physical-limit failure qualifies, including rough success with physical
failure. Both gates must be known booleans and execution_error must be null.
It refuses a bare flag, both gates passing, unknown/null gates, other protocols,
errors or interrupted executions. A diagnostic is not in the automatic B plan.

## T01 proposal, CPU only

Felix approved the goals/speeds and 3 ms first protocol on 2026-10-10:
"Yes but try this first. If this does not work also try with 1ms."
The question specified the three FK goals below, 8 s allowance, public figure
eight geometry with 24/18/12 s periods, N32/.03 s timing, and one proposed
desktop episode per GPU. The conditional 1 ms variant must have its own name,
protocol hash and retained outcomes; it cannot overwrite 3 ms failures or
qualify the canonical 3 ms cohort. CPU task-worker preparation is present below.
Jetson repeats and duration/storage discussion remain pending.
The controller review/required suite are complete; T01 still requires canonical
same-machine preflight and its separately reviewed task-worker/protocol release.
No T01 run is enabled by this V01 plan.

`inputs/t01_proposal.json` and hashed `inputs/t01_witness.npz` retain the concrete
FK endpoints at1/2/3 times[.12,.4,0,.15,0,.05,0]rad from comfortable_high_clearance.
Initial distances are100.636/207.933/317.052mm. Three8s quintics have2668 samples
each at<=3ms and positive sampled model q/v/u margins. Minimum overall margins:
q.09346rad,velocity2.21875rad/s,effort17.39758Nm. This is a sampled inverse-dynamics
witness, with collision and between-sample compliance unassessed. The public
constant-current-state seed is deliberately retained to evaluate the selected
public protocol; the witness is neither seed nor tracking reference.

Proposed figure-eight periods24/18/12s preserve public.32m xspan,.08m y amplitude
and.02m edge clearance. Maximum analytic reference speeds are.059238/.078985/
.118477m/s. The agent freezes provisional RMS/maximum gates of 20/40, 25/50,
and 30/60 mm; Felix reviews them on simulation results before D01/D02.
Historical14.323mm RMS/26.193mm
maximum belongs to one24s fixed-clock1ms/qd.01 run; it does not certify B or
higher speeds. Tracking keeps its separate public preset; reaching settings
come from V01. Proposal repeats are1 desktop, Jetson/D02 undecided.

Reproduce screening without CUDA or an optimizer:

```sh
/opt/gato-venv/bin/python3 experiments/campaign/propose_t01.py \
  --output /tmp/gato-t01-screening
/opt/gato-venv/bin/python3 -m pytest tests/python/test_campaign_harness.py -q
```

Every V01 callback's independent NPZ introduces many small files (up to roughly
two events per plant step plus solves). Plant-only8s upper count is about5334
started/completed step files at3ms, with extra fractional knot cuts possible.
Timing in A could produce about268 feedback solves; B count depends on actual
wall solve times and must be estimated from measured root-owned V01 evidence.
Disk/recorder wall costs are excluded from the simulated clock but retained
in event elapsed times. Estimate actual duration/storage from those results
before any Jetson plan; no new runtime/storage measurement is claimed here.

`planning_estimates.json` gives labeled, unmeasured scenarios using 0.8/2/10/30 ms
feedback wall times, 60 kB per solve including lifecycle events/final snapshots,
5 kB per plant step, and 1–5 ms of recorder wall per event. V01's maximum 8 s
scenario spans about 267–10,000 solves and 29–650 MB; the proposed full T01 set's
78 simulated seconds spans about 2,600–97,500 solves and 286 MB–6.34 GB. These
are planning allowances, not predictions: faster B solves increase record counts,
and nominal feedback does not bound B updates. Process estimates add recorder
costs to solve time but exclude imports/build/RK4/thermal effects. T01 estimates
use equivalent shared-helper tracking instrumentation. Current estimates allow
four solve events plus one completed
interval event per update and paired events per plant step. Actual desktop
recording costs and retained Jetson timings must
inform Felix's duration/storage/shorter-protocol discussion before Jetson work.

## T01 frozen worker and prerequisite gates

`run_t01.py` executes one frozen independent episode selected from `t01.json`:
three reaching goals and figure-eight periods 24/18/12 s. `t01_1ms.json` retains
all six identical targets, references, solver presets and success gates with
separately named 1 ms plant timing. Public reaching and tracking presets remain
distinct. Exact little-endian float64 reference arrays and their byte hashes
are embedded in each tracking task; previews extend beyond the complete period.

The worker and `make_task_plan.py --enable-reviewed` reject held protocols
before native identity, build or task execution. Final desktop release markers
are checkpointed together with V01 after independent review; actual T01 execution
still requires exact hashed review/suite and passing preflight receipts. This
avoids a metadata-only later commit invalidating the original V01 source identity.
Canonical preflight receipts must match the current machine, actual shared
harness source bytes, source commit, native module and complete protocol hash.
Failed or interrupted setup is an execution failure, retained separately from
a completed negative task outcome, and cannot authorize fallback.

After a completed negative canonical preflight, a separate completed A
diagnostic is required before the conditional 1 ms preflight. Alternatively,
after passing canonical preflight, all six completed canonical task receipts
with at least one task/physical failure authorize the 1 ms preflight. A passing
1 ms preflight then permits the full matched six-task alternate cohort; a
failed alternate preflight holds that family. No failed-only selective rescue,
automatic retries or canonical 3 ms qualification follows from 1 ms results.

Every episode retains raw lifecycle events/NPZ arrays, shifted warm inputs,
native outputs/timings, states/applied controls, frozen references, effective
protocol/build/source receipts and output hashes. `executed_trace.csv` records
FK/target errors and physical violations at actual plant timestamps. Tracking
RMS uses the time-weighted trapezoidal integral over the complete trace, with
no startup exclusion. Task success, sampled physical compliance and numerical
diagnostics are independent; update intervals above nominal are descriptive.

On a released worker path, the desktop machine whitelist and frozen controller
and task-harness hashes are checked before native identity access. Exact module,
build/source receipts, shared source hashes and readiness executable bytes are
validated before any controller/native imports or construction; setup failures
retain raw failure receipts. Imported shared modules must resolve to the isolated
source paths. Required Pinocchio absence is an execution failure.

CPU contract validation:

```sh
python3 -m pytest tests/python/test_t01_harness.py -q
```

The explicit planner creates one matched six-case plan per desktop GPU with
full reviewed commit and hashed prerequisite inputs. The legacy queue remains
unchanged. Plan generation requires committed bytes; after review/checkpoint,
write a disabled plan and inspect it with the runner's `--plan ... --dry-run`
before any separately authorized release. Jetson, Gazebo and physical robot
execution are outside this desktop preparation batch.
Use `--enable-reviewed --enable-machine gtx1070` with that desktop's evidence
to enable it while retaining the matched RTX 5060 entry disabled/pending. Each
machine requires its own passing preflight before activation; omission of
`--enable-machine` requests both desktops and requires both evidence bundles.
