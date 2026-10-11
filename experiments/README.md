# Experiment runner

Run an explicit plan locally on each machine. Several experiments can share one
pinned GATO source/build. Results and build caches are separate, beneath this
checkout by default. The runner does not launch remote jobs or physical robots.

```sh
python3 experiments/run.py --plan experiments/campaign.json --machine rtx5060 --dry-run
python3 experiments/run.py --plan experiments/campaign.json --machine rtx5060
```

The current campaign plan contains the remaining RTX 5060/internal T01 tasks and
64 F01ab timing cells. It reuses the existing frozen tasks and runtime source
`502fa29`. GTX 1070 T01/F01ab and the user-reported completed Jetson F01ab are
excluded: import their saved results. Jetson T01, other families and physical
scripts retain their separate protocols and execution decisions. The old
`plan.json` is a historical pilot, never the default campaign.

`campaign_plan.py` regenerates `campaign.json` and its small frozen input bundle
from the recorded Git source. It does not build, fetch or run anything. Changes
to this collector use protocol `simple-harness-v1`; historical cohorts remain
identified by their original source and configuration.

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

BB is not added to the automatic plan. Its old 23 completed episodes and retained
failed episode remain untouched. Select only the genuinely pending cases when
preparing a continuation plan, retain the earlier cohort links, and keep the new
recording source/protocol identifiable. The runner handles independent failures;
there is no second campaign manager.

## Relevant checks

```sh
python3 -m pytest -q experiments/test_*.py
```

These exercise real temporary Git/subprocess runner behavior and CPU recording,
collection and adapter wiring. They do not qualify native builds or new GPU
measurements. The family preflight remains part of actual execution. Controller,
timing, warm-start or task-behavior changes require their relevant task regressions;
packaging and recording-only work does not repeat a task campaign.
