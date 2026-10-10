# Opt-in actual stop-path telemetry (CPU preparation only)

Felix authorized additional experiment instrumentation without additional
synchronization or heavy device-memory changes. This candidate uses
`-DGATO_EXPERIMENT_TELEMETRY=ON`; its default is OFF. It does not repair the
solver, change arithmetic, stop conditions, barrier order, stream usage, or ROS.
No CUDA build, production module import or GPU execution is authorized by this
preparation. Root reviews the candidate and the bounded comparison before those
steps. F01 execution remains held separately by the episode/protocol gates.

## Recorded semantics

The native return gains `experiment_telemetry` only when compiled ON. Schema1
contains the full `(number_of_actual_PCG_calls, BatchSize)` uint32 matrix
`pcg_stop_reason_codes` and scalar `sqp_stop_reason_code`. The Python wrapper
preserves these fields and adds human-readable names; unknown codes/schema or a
matrix that omits a terminal PCG row are rejected. Ordinary and older modules
retain their original stats keys. Existing counts, times, flags, merits and
line-search arrays remain available alongside telemetry. PCG native times
remain their existing zero values: no new timer is introduced.

PCG actual paths, in existing execution order:

| Code | Name | Existing branch |
| --- | --- | --- |
| 1 | `kkt_mask_skip` | Incoming device KKT mask causes immediate return. |
| 2 | `initial_absolute_rho` | `abs(rho) < 1e-6` causes initial return. |
| 3 | `absolute_relative_rho` | `abs(rho_new) < (1e-6 + epsilon * abs(rho_initial))` breaks loop. |
| 4 | `iteration_cap` | Existing loop completes, including zero iterations if cap0 and neither earlier return occurs. |

A tolerance break on the last allowed iteration is code3 even when the count
equals the cap. Count0 can mean code1,2, or4. These are actual branch records;
they are not certificates of linear-system correctness, conditioning, task
success or physical compliance. The mask path records the value passed to the
kernel; it does not claim the host currently publishes nonzero masks.

SQP code1/`kkt_ratio` records the existing **batch-wide**
`num_solved >= BatchSize * solve_ratio` break; code2/`iteration_cap` records
exhaustion, including cap0. A final-iteration ratio break is still code1. Final
per-lane `kkt_converged` remains a separate existing field and does not imply
all lanes satisfied the ratio gate. The terminal PCG row is retained even if
no corresponding line search happens.

`breakdown_stopping_supported=false`, `breakdown=null` and an explicit absence
reason report that the current algorithm has no breakdown-stop branch. They
do not report that no numerical breakdown occurred. No nonfinite reads,
per-iteration anomaly tracing, or additional numerical predicates are added.

## Memory, publication and timing scope

ON extends the existing device iteration buffer from B to2B uint32 words.
Counts occupy its first B words; one stop word per lane occupies the second B.
This adds **4B device bytes:64 bytes at B16**, with no new device allocation.
Each path writes its stop word only from thread0, next to an existing statistics
write or its existing return. Every path initializes its output; no extra
memset is needed. The existing count copy becomes one2B-word copy into a tiny
pinned host buffer (8B bytes:128 at B16). Counts and reasons are copied into
host stats after the same existing `cudaStreamSynchronize`.

The number and sequence of explicit synchronization, votes/atomics, kernel
launches and host/device copy calls are unchanged. No extra stream, event or
shared memory is introduced. There is one constructor pinned-host allocation
and corresponding destruction; those are outside solve timing. ON replaces the
existing pageable count destination with pinned storage and transfers4B extra
bytes per PCG call. That may change implicit host publication/blocking and
wrapper/native timing; it is part of the measured telemetry variant, not an
assertion of zero overhead. Local register/resource changes and small host
stats allocations/decoding may also affect time. All F01 builds must record
whether instrumentation is enabled and use matched flags.

DefaultOFF native source projects to exactly the baseline tokens, allowing
only equivalent braces around the existing `break` statements. The scope audit
checks ON synchronization/vote/atomic and copy call sequences and launch counts.
It is evidence about source scope, not a substitute for compilation or native
comparison. Source manifest hashes make the audited bytes reviewable.

## CPU validation

```
/opt/gato-venv/bin/python3 -m pytest tests/python/test_experiment_telemetry.py tests/python/test_pcg_telemetry.py tests/python/test_interface.py -q
python3 experiments/frequency_campaign/telemetry/audit_scope.py
```

2026-10-10:17 passed, one existing hppfcl deprecation warning. The existing PCG
transport test compiles a host-only pybind conversion fixture in pytest's
private temporary directory; it does not compile/import a CUDA solver or use
the GPU. New tests use fake native replies and source projection. They cover
all stop-code mappings, equal-to-cap tolerance distinction, zero-count path
distinction, terminal PCG row retention, empty zero-SQP rows, batch ratio versus
lane flags, old-module absence, and malformed data rejection. They do not prove
which branches a GPU actually takes. `comparison_protocol.json` freezes the
bounded native checks to resolve that limitation once root approves execution.

No scientific outcome or campaign release is claimed from these CPU checks.

## Bounded native comparison implementation (still CPU-only)

`direct_pcg.cu` and its standalone CMake target are test harness code, compiled
only in root-created build-owned source snapshots. The five cases use finite
real-space SPD diagonal systems with zero vector padding and fresh seeds:
mask skip, initial-rho exit with cap0, nonzero-rho cap0, exact identity solve
at cap1, and a two-diagonal system still unfinished at cap1. This directly
checks all four native exits, including a tolerance break exactly at the cap.
Every fixture saves configuration/matrix/RHS/seed before device copies, a
submission marker before launch, and full x/count/reason output afterward.
The test harness has its own publication synchronization per case, identical
in both variants; it adds no synchronization to the production solver.

`compare.py` defaults to CPU receipt validation/queue printing. Its execution
mode is reserved for root after independent review. The fixed queue has five
direct calls per variant, followed by24 fresh subprocesses: B1/B16 × cold/warm
× OFF/ON × three repeats. Each subprocess imports exactly one module and its
reviewed wrapper from that variant's separate snapshot, avoiding pybind global
class-registration conflicts. It executes one disposable unrelated warmup,
then a fresh measured solver; warm cases replay one cold prefix in that solver
before the exact retained warm primal input. The maximum is70 native calls.
The CaptureNative proxy retains the exact native return without copying it
inside the timed window; its tiny Python call/counter overhead is present in
both variants. The timer covers the actual public `BSQP.solve`, including its
normal result decoding, and excludes input/output archiving and validation.

All native returns and public stats are kept in raw NPZ/strictJSON files,
including every terminal PCG row and optional branch codes. Inputs/config and
request files are written before subprocess submission or each solve. Failures
stop the queue; exception returns/counters/errors, subprocess exits and any
unknown pending submissions remain indexed. No retries or skipped calls count
as successful validation. The full manifest covers nested worker receipts and
raw files; source/build/module/input identities are retained separately.
Matched numeric checks use the frozen tolerance plus observed within-variant
repeat range. Timing reports keep all three samples, mean/median and paired
deltas, without claiming a precise overhead confidence interval.

Run CPU preparation only:

```
python3 experiments/frequency_campaign/telemetry/compare.py
```

Root's post-review build receipts follow schema2 in
`build_receipt_schema.json`. The native candidate is not yet committed: its
identity is the actual archive `base_commit` plus the SHA256 of the saved,
reviewed six-file native patch artifact. A `source_commit` field is rejected;
the driver does not claim that the base commit contains the overlay. Root owns
archive/overlay creation and review attestations; this driver does not create
or compile snapshots.

The independent source authority is a JSON object with schema_version2,
base_commit, candidate_diff `{path, sha256}`, the COMPLETE source_file_sha256
map, matched common build_flags, and optional external_dependency_sha256.
Root creates it from the exact reviewed gitarchive plus allowed overlays and
reviews its completeness independently. Supply its reviewed digest as
`--source-manifest-sha256`; the driver does not infer that trusted digest from
either receipt. It verifies the authority digest and actual candidate patch
bytes, exact map equality in both receipts, every archived source byte, and the
complete source inventory. This covers solver/dynamics/generated headers and
all Python package/wrapper files, including dependencies absent from the older
narrow critical list. The narrow current-worktree critical-file comparison is
retained as an additional check. A receipt-bound exact module artifact is the
only permitted extra file inside a source snapshot; arbitrary bytecode/build
outputs and symlinks are rejected. Workers use `-B` and
`PYTHONDONTWRITEBYTECODE=1`.

Each receipt also binds the build-owned compiled_dependencies JSON inventory
and actual compiler depfiles. Every listed local dependency must be in the
complete reviewed source map. All recorded external dependency hashes are
verified; every extra externally pinned SDK entry must also be retained.
Root audits the actual compiler commands/dependency collection and matched
compiler/toolkit/optimization/resource flags. Optional selected SDK hashes do
not constitute a full system-header certificate. This is a review trust root,
not a digital signature: substituting both the authority and its independently
trusted command-line digest changes the trusted evidence.

Both typed CMake caches must match telemetry/Release/architecture exactly.
Duplicate keys, misleading comments, wrong types, and substring values are
rejected. The module cache additionally requires KNOTS:STRING=32 and
PLANT:STRING=tiago_right; the module basename binds N32/tiago_right. The direct
fixture's N32/plant constants are bound by its reviewed source and separately
checked output configuration. Separate build/source roots are required.

Disposable warmup capture/native ownership is released before construction of
the fresh measured solver. Each call clears stale capture before submission.
After raw preservation, every warmup, prefix and measured return is checked
for finite XU/initial/final merits; a failure holds the queue immediately.
Public BSQP.solve result and telemetry decoding are INCLUDED in wrapper wall
time. Later get_stats access, raw copies/archiving and validation are outside.

After independent review/root compilation, inspect the queue with:

```
python3 -B experiments/frequency_campaign/telemetry/compare.py \
  --off-receipt OFF.json --on-receipt ON.json \
  --source-manifest REVIEWED_SOURCE_MANIFEST.json \
  --source-manifest-sha256 REVIEWED_MANIFEST_SHA256
```

Execution additionally requires `--execute-reviewed --output NEW_DIRECTORY`.
The frozen70 comparison scope excludes the separately required single ON
`make test-tiago-tasks` regression with unchanged fixtures and its own complete
receipt. No extra comparison calls, retry-green or fixture changes are implied.
This documentation is not execution authorization.

Driver correction CPU checks cover both-receipt omissions and authority
substitution, previously omitted solver/dynamics/wrapper drift before a native
import sentinel, exact and duplicate cache entries, patch/module identities,
forbidden extra bytecode, all-call nonfinite retention, warmup disposal and
stale constructor failure capture. The initial driver/protocol/input copies
remain under `/workspace/gato-campaign-recovery-20261010/f01-telemetry-initial-driver/`.
The frozen input bank and all numerical/budget settings are unchanged.
