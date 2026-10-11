# Jetson F01ab full run

Use the existing Jetson GATO container and active GATO Python environment. Update
the standalone container checkout in place:

```bash
cd /workspace/.worktrees/jetson-f01ab
git fetch https://github.com/felixmueh/GATO.git experiment/jetson-f01-preparation
git merge --ff-only FETCH_HEAD
python3 -B experiments/jetson_preparation/prepare_jetson.py --check
python3 -B experiments/jetson_preparation/prepare_jetson.py --device jetson --reuse-qualified "$PWD/example_artifacts/jetson_f01ab/20261010T203259.571927Z"
```

The reuse command is for the recorded Jetson attempt above: all four builds,
the unchanged suite and qualification succeeded; readiness stopped on frozen
screening metadata before any native/task event. It creates a fresh timestamped
correction cohort, preserving that failure. It reuses the qualified modules and
suite, runs one actual canonical mode A task/physical preflight, then the same64
cells. It never rebuilds or repeats Make. Unrelated task/native failures and
already-started matrix attempts cannot use this option.

Runtime bank validation checks the exact frozen bytes, source/model/generator,
array definitions and schedules, and preserves the historical saved screening.
It does not rerun CPU screening, compare library versions or regenerate physical
witnesses on the deployment machine. The actual canonical task remains the gate.
New receipts distinguish original compiled source502fa29 from the new collector
commit, whose six runtime/controller/model files must equal the compiled archive.
Original build/suite/cache receipts and held markers are untouched.

For a first container checkout, clone the branch without registering a worktree,
then check and run the fresh pipeline:

```bash
git clone --branch experiment/jetson-f01-preparation https://github.com/felixmueh/GATO.git /workspace/.worktrees/jetson-f01ab
cd /workspace/.worktrees/jetson-f01ab
python3 -B experiments/jetson_preparation/prepare_jetson.py --check
python3 -B experiments/jetson_preparation/prepare_jetson.py --device jetson
```

Activate the existing GATO Python environment first, or pass `--python` with its
executable. Stop other GPU jobs and use a persistent terminal/session. The last
command performs the full approved protocol automatically; no central runner
interface is introduced here. `--check` verifies package hashes/source authority
on CPU. Every attempt defaults to a fresh UTC timestamp below this checkout's
`example_artifacts/jetson_f01ab/`. `--work` accepts another fresh path beneath that
same directory. All raw outputs, logs, isolated sources/builds, temporary files,
caches and receipts remain below this checkout. Prior HOME-based attempts remain
historical evidence and are not resumed by this launcher.

Before creating an attempt, it checks CMake >=3.24, make, git, nvcc, C++, Python
development headers, NumPy, Pinocchio, pytest, pybind11 and torch, both source pins' ancestry and target selection.
`--device jetson` explicitly selects this dedicated Jetson/87-real pipeline when
the container hides `/proc/device-tree/model`. It records user selection, host
architecture and optional model/error; it makes no CUDA or Torch CUDA query.
Omitting the flag retains automatic host Orin model detection. Explicit selection
is not hardware verification: the actual SM87 suite smoke remains mandatory
before canonical preflight or any matrix cell. It does not install dependencies. The
branch must carry runtime `502fa29c8b0d649dcb347f9534bf29c6935fa6dd` and science
`9cbf25721056e5b9dae1a5daf9e24982849b0573` in its history; no native binary or
bundle is needed. Standalone local clones avoid stale worktree registrations.

The sequence is four isolated Release/87-real/telemetryON builds (N16/32/64/128),
ONE unchanged N32 `make test-tiago-tasks`, automated target validation, ONE
canonical mode A rough-reaching preflight, then the exact64-cell F01ab short12
matrix through the existing runner, one sequential `--only` invocation per cell.
No second native modules are built by the runner. Target validation checks the
actual SM87 smoke output, all three unchanged task cases' receipts, module hashes,
source/cache identities and retained summaries. It is automated validation under
the reviewed launcher policy, not human artifact review or numerical equivalence.
The canonical readiness guard requires both task success and physical compliance;
a worker exit0 alone is insufficient.

The unchanged Make target overrides the artifact environment directory. Its
actual raw tasks under `builds/N32/campaign-source/test-artifacts/tiago-tasks` are
preserved and copied to `suite/raw`, including failures. Original builder receipts
and before/after cache/module hashes are preserved. Any cache-only Make
reconfiguration gets a separate post-Make receipt; original evidence is untouched.

The saved explicit plan and actual dry-run must contain exactly64 pending cells:
N16/32/64/128 × SQP1/2/5/10 × PCG100/200/400/800, mode A/controller_dt. Each cell
retains12 episodes,12 cold +222 warm +6 goal-switch +1 excluded initialization
call, totaling15,424 native calls. Public preflight PCG120 is distinct from
matrix100. All raw hashes, event/input/native-return counts, completed episodes
and category counts are checked before the next cell. Task/physical scientific
negatives in completed timing cells are retained. No strict deadline gate,
sample filtering, budget pruning or automatic retries is used.

`launch-state.json` records completed/held status, checked coverage and provenance.
A failure, interruption, abort or missing/changed raw artifact holds the family;
its marker blocks subsequent attempts. Preserve the whole attempt and report
`launch-state.json`, `preparation.json`, logs and raw results. Do not delete
markers or retry to obtain a pass. SIGINT/SIGTERM stop child jobs and retain the
hold. SIGKILL/power loss leaves the durable running marker; inspect remaining
processes before any later execution.

`--continue-prepared PATH` is narrowly for a successful old preparation with
`status=prepared_TARGET_REVIEW_REQUIRED`, intact build/suite receipt hashes, and
no campaign stage/marker ever started. It validates the target before preflight
without repeating builds or Make. Failed/interrupted or already launched attempts
cannot continue automatically.

Approved planning scenario: .1–1 s/call gives26min–4.3h solve-only plus builds,
plant execution and recording. Actual Jetson duration/storage remain unmeasured;
the launcher records actual tool/disk/environment and output evidence. Source
observer acceptance records are global authority, not imported Jetson results.
No robot motion is commanded. BA/BB and a future unified runner are separate.

CPU fake-pipeline checks (no native import/build/GPU):

```bash
mkdir -p example_artifacts/jetson_f01ab-tests
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q experiments/jetson_preparation/test_full_pipeline.py experiments/jetson_preparation/test_target.py experiments/jetson_preparation/test_bank_reuse.py --basetemp=example_artifacts/jetson_f01ab-tests/tmp -o cache_dir=example_artifacts/jetson_f01ab-tests/cache
python3 -B experiments/jetson_preparation/check_artifact_retention.py
```

## Download results

Successful runs automatically write `jetson-f01ab-results.tar.gz` in the checkout
root. This single portable archive includes all cell raw data, summaries, readiness,
logs, frozen inputs and compact provenance; it excludes builds, source checkouts
and caches. `EXPORT.json` records file hashes and maps original Jetson paths to
archive paths. The latest export replaces this convenience archive; original
timestamped run directories remain unchanged.

To package an already completed run without rebuilding or rerunning anything:

```bash
python3 experiments/jetson_preparation/export_results.py
```

From the destination computer, using the configured `jetson` SSH alias:

```bash
rsync -avP jetson:/home/pal/FelixGATO/GATO/.worktrees/jetson-f01ab/jetson-f01ab-results.tar.gz .
```

Extract with `tar -xzf jetson-f01ab-results.tar.gz`. Pass `--run PATH` to export
a specific run, including partial/failed evidence; its status is retained.
