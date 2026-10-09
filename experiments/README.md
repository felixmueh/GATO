# Small experiment runner

From a configured GATO checkout:

```sh
git pull --ff-only
python3 experiments/run.py
```

The runner executes pending cases sequentially, prepares isolated native builds,
and writes everything beneath `~/gato-results`. It uses Python's standard library
on Linux. CUDA experiments still need the normal GATO dependencies, CUDA toolkit
and compatible compiler/interpreter; the runner does not install system packages.

## Included queue

`plan.json` contains a CPU receipt smoke case and a small ordinary-TIAGo frequency
pilot: N16/N32, B1, three process repeats, five SQP iterations, fixed 0.504-second
prediction, one excluded initialization solve and two-second measured wall budgets.
On a machine without a recognized GPU, only the CPU smoke case is selected.

The GPU pilot adapts the existing `examples/solve_time_heatmap` tool. It uses that
pinned version's tracking configuration and timestep scaling, not the reaching
preset. It disables the slow-solve cutoff. Raw results retain the realized weights,
source/model/module hashes and scientific outcomes. The native timer measures a
whole batch; `batch_solves_per_second` is `1000 / mean_native_ms`, not candidate
throughput or a measured live control rate. Different machines cover different
reference phases within the wall budget. This is a functional pilot, **not the
final identical-input hardware comparison**. The existing heatmap tool also needs
`nvidia-smi` and `nvcc` on the GPU machines.

No large historical campaign, ROS connection or physical-robot run is scheduled.
Add a separately frozen experiment to the plan when its protocol is ready.

## Configure each machine once

Optional, ignored `experiments/local.json`:

```json
{
  "machine": "gtx1070",
  "python": "/opt/gato-venv/bin/python",
  "output": "~/gato-results",
  "cache": "~/.cache/gato-experiments"
}
```

Omit `machine` to detect GTX 1070, RTX 5060 or Jetson Orin. `--machine` overrides
it. Use the interpreter containing your GATO dependencies; virtual-environment
executable paths retain their symlinks so builds and workers use that environment.
Machine profiles in
`machines.json` select CUDA targets 61, 120 and 87 respectively; `local` has no
CUDA target. Missing GPU telemetry is recorded as unavailable. The toolkit must
support the selected architecture. Profiles do not change power modes or clocks.

Useful commands:

```sh
python3 experiments/run.py --dry-run
python3 experiments/run.py --only tiago-frequency-pilot
python3 experiments/run.py --retry-failed
python3 experiments/run.py --collect-only
```

`--output` selects another collection folder and `--plan` selects another plan.
Pulling the repository updates the queue; it does not modify previous results.

## Add an experiment from an agent worktree

1. Develop the experiment on a dedicated branch/worktree under `.worktrees/`.
   Keep builds separate and validate a small GTX 1070 case first.
2. Commit the experiment script and protocol. Publish the branch containing that
   commit to the same Git remote as the runner's checkout.
3. Add an entry to `plan.json` with its full 40-character `source_commit`, command,
   target machines, cases and repetitions. Commit/publish the updated plan too.
4. Store small frozen inputs alongside the plan and declare their SHA256 hashes.
   Every target must receive the same input bytes. Large inputs may be provisioned
   at those paths separately; their hashes are still mandatory. The runner checks
   and copies declared inputs into each attempt before execution, checking the
   copied bytes again in case an input changed during a long build.

Minimal experiment entry (replace the commit with your real published commit):

```json
{
  "id": "reaching-corrected",
  "source_commit": "<full experiment commit>",
  "machines": ["gtx1070", "rtx5060", "jetson"],
  "command": ["{python}", "{source}/experiments/reaching.py", "--case", "{case}", "--output", "{output}"],
  "repeats": 3,
  "cases": [{"id": "near", "parameters": {"vel_lim_cost": 0.1, "qd_cost": 0.01}}]
}
```

`enabled: false` leaves an experiment out of the automatic queue. `inputs` is an
optional object mapping repository-relative file paths to SHA256 strings.
Cases have unique `id` values and an experiment-defined `parameters` object;
the runner adds a zero-based `repeat`. It does not invent a matrix or retune
parameters. Adding a case/repeat leaves existing cases complete.

The script reads `--case`, writes raw artifacts below `--output`, and writes a
JSON object to `metrics.json`. Exit zero means the experiment completed, including
declared negative/numerically rejected outcomes; put that outcome in the metrics.
Exit nonzero means execution failed. Keep solver convergence, task success,
sampled limits and solve-frequency measurements separate in your metrics.

Commands are argument arrays, executed without a shell. Available substitutions:
`{python}`, `{source}`, `{runner}`, `{build}`, `{cuda_arch}`, `{case}`, `{output}`,
`{inputs}`. The last three apply to the experiment command. Environment variables
`GATO_SOURCE_DIR`, `GATO_BUILD_DIR` and `GATO_MACHINE` are also supplied.

An optional `build` object contains `commands` and required `artifacts` patterns
relative to `{build}`; see the included pilot. Use explicit build/module paths.
Builds are shared by cases with the same source/recipe/profile/interpreter and
environment, and retained artifact hashes are checked before reuse. A failed
shared build is attempted only once per invocation. Scripts needing submodules
must include their initialization in the build commands.

The runner checks out exact commits under the repository's common
`.worktrees/experiment-runner/` directory. Missing commits trigger `git fetch
origin`; they must be reachable from published branches. It refuses modified
tracked sources in these cached worktrees. Leave them private to the runner.
Small compatibility adapters may live beside the runner and use `{runner}`;
their Python source hashes participate in case identity.

## Resume, provenance and collection

A case identity includes its pinned commit, experiment definition, parameters,
repeat, input hashes, machine profile/device, interpreter/package/toolkit identity
and runner/adapter code hashes. Changed code, weights or environments therefore
produce a new cohort. Changing the runner/any bundled adapter conservatively
invalidates all case identities. Completed historical cohorts stay intact.

Results use `machine/experiment/case-hash/attempt-001/`, with `case.json`,
`receipt.json`, `metrics.json`, stdout/stderr logs and experiment artifacts.
Receipts include loaded build-artifact hashes, hardware snapshots and hashes of
collected files. `README.md`, `index.json` and `summary.csv` summarize all attempts,
including failures and earlier cohorts; the CSV's metrics column is a JSON object.

Completed cases are skipped. Failed executions stay skipped until
`--retry-failed` and keep the command's exit status nonzero. Interrupted/unfinished
cases get a new attempt on the next invocation; completed cases are not repeated.
Ctrl-C stops the active process group and writes an interrupted receipt. Case-level
restart deliberately avoids adding another checkpoint system inside each script.

One cache lock prevents simultaneous runner invocations across agent worktrees.
Keep the same cache setting on one machine. Manual GPU jobs are outside that lock;
keep the GPU free when collecting timing measurements.

To collect machines on the desktop automatically, configure SSH/rsync and create
each machine's destination directory once, then add:

```json
{"sync_to": "felix@desktop:/home/felix/gato-results"}
```

Merge that field into `local.json`. Each invocation uploads its machine subtree,
including preserved attempts, without deleting other machines. It retries uploads
even when there are no pending cases. Failed delivery leaves local results intact
and returns nonzero. On the desktop, `--collect-only` regenerates the combined
index after uploads; a normal runner invocation also regenerates the index.

## Verification

```sh
python3 -m unittest discover -s experiments -p test_runner.py -v
```

The integration tests exercise real temporary Git worktrees/subprocesses: resume
and new cohorts, explicit execution retries, frozen inputs, build reuse/stale
artifact detection, interruption/worker cleanup, cross-invocation locking and
upload retry without rerunning completed experiments, and configured/default
virtual-environment interpreters. They do not need CUDA.

GTX 1070 validation on 2026-10-09 completed the CPU smoke and all six GPU pilot
cases with CUDA 12.9.86, GCC 11.4 and `/opt/gato-venv/bin/python3`. All eight
integration tests passed. Second invocations left the seven receipts unchanged;
436 collected artifact hashes and the native build hashes were checked.
The [measured summary](validation/gtx1070-20261009.json) records exact attempts,
source/runner/module hashes, realized weights, package versions and diagnostics.
Raw results are under `/home/felix/gato-results`; archived build logs are under
`gtx1070/validation/20261009/` in that collection.

| Case | Repeats | Mean native solve time range | Reciprocal frequency range |
| --- | --- | --- | --- |
| N16, B1 | 3 | 0.574–0.592 ms | 1690–1741 Hz |
| N32, B1 | 3 | 0.788–0.861 ms | 1161–1270 Hz |

These are per-run means from the functional pilot, with different simulated
reference coverage. The recorded initial state and 10 ms feedback samples had
no URDF position/velocity limit violations. Full plans and controls were not
saved, so torque limits, modeled dynamics defects and between-knot feasibility
were not assessed. Native stopping flags are not nonlinear KKT certificates;
SQP cap hits are retained in the summary. Local package versions differ from
the example requirements. RTX 5060 and Jetson validation remain pending.

Existing scientific tooling: [heatmap protocol and limitations](../examples/solve_time_heatmap/README.md).
