# Newly defined force orchestration adapter

Felix explicitly authorized a new adapter on 2026-10-10: predict each force
hypothesis first, choose the current best fitting force, then run exactly one
B1 trajectory optimization. Historical adapted implementation/result provenance
was not recovered. This directory implements that newly defined orchestration,
not a reconstruction of historical results. Recovery evidence remains in
`../force_recovery/source_inventory.json`.

`orchestration.run_tick()` accepts predictor and TO callbacks rather than loading
CUDA, native bindings, ROS or a robot. Original comparison: generate/freeze B
hypotheses, B-lane TO, predict the recorded observation interval, choose a lane,
update the estimator once, return that lane's plan. Adapted comparison:
generate/freeze the same hypotheses, predict, choose a force, update the estimator
once for the next tick, execute exactly one B1 TO using the selected *unsmoothed*
candidate force and its corresponding state/reference/frozen warm seed.
The next-tick estimator update cannot change current TO inputs. Calls are
sequential, and one adapter/estimator instance must have one tick in flight.

The selection rule is the existing controller's unweighted Euclidean norm of
the full state prediction error and NumPy argmin (earliest tie). Mixing state
units is deliberate correspondence, not a new statistical estimator.
Nonfinite predictions/errors retain raw values and fail before selection/update;
no lane filtering or replacement force is allowed. Nonfinite full plans are
retained and fail the tick. A failure after estimator update is not rolled back
and must not be retried as an unchanged estimator tick. These are execution
failures, distinct from task success and physical compliance.

`FrozenBatchForceEstimator` subclasses the unchanged existing
`examples.force_estimator.ForceEstimator`; inherited update mathematics remain
unchanged, using controller alpha=.6/beta=.5. The inherited update's internal
`generate_batch()` call receives a copy of this tick's frozen hypotheses rather
than generating another batch. A finally block clears the temporary cache,
including on an update exception; exceptions may leave partial estimator state.
The original estimator source hash appears in the CPU receipt.

## Backend contract

`TickInputs` carries previous measured state/time, current measured state/time,
and the complete actually applied control history as ordered contiguous
half-open intervals. No replacement with the first planned control is permitted.
Times use seconds on one plant clock; controls and state use the caller's
declared plant units (CPU protocol uses abstract arithmetic state/control).
The caller supplies candidate-specific current states, reference windows and
warm seeds with a leading B axis. It is responsible for their provenance and
for the relation between these states and the observation; they are frozen
without shifting, regenerating or replacing seeds. Record that seed policy.

Predictor receives the frozen hypothesis batch, the full history and a wrench
transform callback. It must replay every segment from the recorded previous
state for every hypothesis, using that hypothesis's evolving state and the same
frame contract at every dynamics evaluation. Wrench convention explicitly names
source/destination frames AND reference points, with components
`Fx,Fy,Fz,Tx,Ty,Tz`, N and N*m. The caller transform must perform the correct
rotation and reference-point torque shift; identity is valid only for identical
frame/point conventions. This module does not qualify physical transformations.

TO requests retain a leading batch axis even for B1. They carry unsmoothed source
wrenches and a transform at each candidate's initial state. A TO backend using
pose-dependent wrench frames must handle the same physical wrench at its
internal states; holding an initial rotated wrench fixed is not generally
equivalent. Caller-provided backend model/frame/config/source and native timing
receipts are prerequisites for a real comparison. There is no built-in dynamics
or BSQP backend here.

Original B16 and adapted B1 TO outputs need not match: batch-wide stopping/work
semantics may differ. Prediction and force-selection correspondence is separate
from optimizer work, outputs and closed-loop task behavior. The arithmetic stub
plans happen to match and provide no claim about native solver correspondence.

## Recording and timing

Each tick invokes its observer once in a finally block, including on failures,
outside all wrapper timers. An observer must persist synchronously and avoid
raising; observer failure is a recording failure and may propagate. Records
contain frozen inputs, hypothesis/prediction arrays, errors, selection, TO
request/full plans/selected plan when available, timing scopes and the exception.
`records.write_record()` uses a fresh per-tick directory, NPZ for lossless raw
arrays including NaN/Inf, and JSON references/metadata. Caller must add source,
config, RNG state/seed, backend timings and model receipts for real runs;
callback names in JSON are identifiers, not executable-source provenance.

- `candidate_forward`: predictor callback wall time, including whatever that
  callback allocates/transforms/replays; excludes result snapshot and observer.
- `to_backend`: optimizer callback wall time, excluding request preparation,
  transforms, returned-plan snapshot and observer. Native device timing is a
  separate backend responsibility; this is not GPU solve time.
- `selection`: norm, finiteness validation and argmin/selection snapshots.
- `estimator_update`: inherited update wrapper callback wall time.
- `total`: entry through success/failure, before observer.
- `wrapper`: total minus candidate_forward, to_backend and selection; includes
  input/hypothesis freezing, request preparation/transforms and estimator update.
  It overlaps `estimator_update` deliberately; do not sum all reported scopes.

Memory is null with an explicit not-measured reason. A predictor could allocate
a full B-lane optimizer workspace internally. No lower GPU allocation, peak
memory or speed benefit follows from issuing B1 TO in this CPU harness. E03
must measure actual backend allocations and their scope on the qualified model.

## Frozen CPU protocol and stopping point

`cpu_protocol.json` freezes B16 candidates, two different applied controls,
states/references/seeds and an arithmetic predictor/TO stub. Reproduce from the
worktree root, using a fresh output directory:

```bash
python3 -m pytest -q tests/python/test_force_adapter.py
python3 -m experiments.force_adapter.run_cpu --output /tmp/force-adapter-new-raw-directory
```

The correspondence CLI executes one original and one adapted tick, preserving
raw records and a source/raw SHA-256 receipt. Tests additionally cover ties,
nonfinite failures, backend partial records, candidate-specific seeds,
single estimator update and cache restoration after an exception. The first
CPU receipt is retained under `cpu_evidence_v1/`; it identifies uncommitted
adapter source by hash and parent commit e18bc86. These are CPU contract checks,
not native timing, memory, robot task or performance results.

Stop at CPU acceptance and independent review. Native/GPU correspondence and
Jetson time/memory runs remain pending backend/model/frame qualification and
runtime/storage discussion with Felix. E02 TIAGo disturbance remains blocked:
current Python/native nonzero-wrench guards are unchanged and no Indy7 result
can substitute for TIAGo qualification. No GPU/build/robot jobs, solver changes
or deployment are authorized by this CPU implementation.

## 2026-10-11 runtime continuation

Felix subsequently authorized bounded nonzero-force runtime implementation and
the known representation correction. [RESULTS.md](RESULTS.md) records the
production float native oracle, accepted limitation and unchanged task suite.
These later instructions supersede the earlier CPU-only implementation scope.

`backend.py` supplies actual Pinocchio prediction and native TO callbacks using
the explicit body-fixed final-joint frame/origin contract. It reuses the existing
orchestration and estimator. This physical scope is prominent: it is an attached
wrench, not a constant world-expressed disturbance. Native controller world-force
semantics remain limited and guarded for TIAGo.

`memory.py` records CUDA free/total bytes at owner/solve/close boundaries and
Linux host peak RSS. Device differences are observed boundary snapshots, not
exact per-allocation accounting or a continuous GPU peak. Jetson shares system
memory; do not sum CUDA device availability changes and host RSS. Memory failures
retain status/null reasons. Constructor, public-solve and native timings remain
distinct; the complete TO callback includes a fresh owner/model allocation.

## Prepared real studies (no measurements yet)

`run_native.py` uses frozen `native_inputs/` for 24 E03 timing/memory cases:
six retained start/goal inputs × original/adapted × zero/local-x2N truth.
Each worker has one unrelated excluded warmup and one measured tick. Both paths
receive the same frozen16 hypotheses, RNG, observed state, actual two-control
history, reference and cold seed. E01 correspondence derives from the four
first-input measured E03 records; it schedules no duplicate calls.

`run_task.py` uses byte-identical accepted502 V01/T01 inputs in `task_inputs/`.
E02 has six unchanged24s figure-eight tasks: no estimator/original/adapted ×
zero/local-x2N. Each method/truth has its canonical modeB rough-reaching
preflight. No estimator uses zero-wrench B1 on every tick. Both estimator
methods bootstrap with zero prior lane0 B1, with no truth access or estimator
update before the first observed control interval. Hidden truth only enters
the body-fixed Pinocchio plant. Failed B preflight retains evidence and holds
its dependent task; a follow-up diagnostic is discussed separately.

All tasks use the unchanged accepted `bsqp.offline_mpc.run_mode_b` engine.
Full force-algorithm wall time, including prediction/selection/update, fresh
native owner/model allocation and memory queries, advances the previous plan.
Recording occurs outside that timer. Native solve and constructor clocks are
also saved separately. Do not describe full algorithm time as pure TO speedup.
Observed device-availability boundary differences and host RSS are measured
separately; missing telemetry stays null with its reason. Full histories,
native replies, hypotheses, selections and estimator errors remain in raw data.

The simple runner's `experiments/force_plan.py` prepares an explicit plan for
each named GPU from this source pin. Jetson execution needs the duration/storage
discussion; none of these studies have been executed. Saved-data analysis is
`experiments/analysis/force.py` in the simple runner. It retains failed attempts,
exports algorithm/native time and reciprocal, memory, task/physical outcomes,
and descriptive correspondence, without another tight precision gate.

CPU study wiring checks:

```bash
python -m pytest -q tests/python/test_force_studies.py
```
