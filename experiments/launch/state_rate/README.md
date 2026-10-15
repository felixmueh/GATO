# State-rate tracking preparation

Preparation only: no CUDA build or task execution until Felix explicitly says
GPU is free. Six frozen one-attempt plans are prepared separately. First execute
`sep-exact.json` and stop for review. Then root can authorize the remaining five;
no arbitrary pass tolerance or automatic retry is added. `internal.json` is an
inventory and must not be launched as an unattended six-case queue before review.

The exact retained normal b400 N32 module is copied and hash checked, never
rebuilt. Published frontend pin is1b517fce5bd4ddcbabd200fa120d72f012f40771,
whose native tree remains identical to b400 and frontend base to7cf. The .02 option is forwarded only for 50Hz; exact0 uses the old
helper call unchanged. Reference and source identities remain distinct.

September inputs preserve exact saved15000 XYZ samples/zeroRPY and saved weights,
N32/.008s; grid12 and24 use frozen input SHAaec47cfa and collector90bdb
periodic_reference interpolation with N32/.03s/SQP5/PCG120. Both preset groups
retain nominal start and actual1ms integration. Cost values are frozen in each
input t01.json. No solver/core/native modifications.

CPU checks:

```sh
cd /workspace/GATO-state-rate-runner
/opt/gato-venv/bin/python3 -m pytest experiments/test_state_rate_plan.py -q
/opt/gato-venv/bin/python3 -m py_compile experiments/adapters/state_rate.py experiments/launch/state_rate/export.py
```

Runner commands use each reviewed per-case plan with explicit GTX machine,
source/build cache and fresh raw root; --dry-run performs no native execution.
Run commands are maintained by root after the frontend pin is finalized.
No GPU smoke/build/calibration/preflight episodes are added.

CPU export:

```sh
/opt/gato-venv/bin/python3 experiments/launch/state_rate/export.py --raw <raw-root> --output <thesis-export-root>
```

Per-feedback executed_trace main q/v/FK/error columns use actual plant state at
solve start; sampled_* columns separately preserve held solver-input state.
References, selected/applied control association, actual timestamps,
sample timestamps/age/index/repeated flag and native/wall solve timings.
Actual plant-step RMS/max remains separate and supplies runs.csv. Full plant
states/controls stay raw. Report labels preserve failures; no KKT/safety claim.
No new figure is generated.

The positive path reproduces historical3aa warm shift max(sampledelta,sim_dt):
repeated sample shifts1ms; exact0 retains current behavior. Reference origins
remain sample timestamps and the newly returned first knot activates on arrival.
The runs.csv timestep_s field is prediction knot spacing (.008/.03); plant
integration step remains explicitly1ms in protocol.json.


## Single Gazebo24 period-only counterpart

`gazebo24.json` selects only `Gazebo24-N32-grid/figure8-24s`, one attempt.
It keeps source/collector90bdb825d059e22b709e2e53a785d7d034aad1b0 and the
original grid build recipe/inputs/tracking preset/start state. A small adapter
loads the immutable original worker/protocol unmodified, changes only the
reference period/duration and their derived reference/speed/coverage labels,
and records its own hash. Explicit --dangerously-disable-safety-stops retains
the grid's report-only policy, finite/stale/infrastructure stops and handback.
No joint-state or simulator configuration is changed. Reuse the passed grid
canonical at the path in the plan; original canonical task/hash/error/handback
checks still apply. If it is missing, stop and inspect evidence rather than
launching another qualification episode. Use the old task-grid cache to reuse
unchanged native modules; never rebuild or run the grid science cases.

The collector requires an already running PAL Gazebo on domain1; it does not
start a simulator. commands_felix.md includes the earlier known PAL launch in a
separate host block only if needed, and one experiment block inside GATO.

CPU export after collection:

```sh
python3 experiments/launch/state_rate/export_gazebo24.py --raw RAW_ROOT --output /workspace/GATO/thesis_results/gazebo-24s-n32-2026-10-15
```

Use `attempt-001/observed_task_trace.csv` for the observed path. It exports the
original observed_task_trace event (including actual task boundary samples),
with t/x/y/z columns used by saved September path CSVs and time_s/fk_* aliases.
The thesis evaluation extractor is not present here, so parser execution is
pending on the thesis side. No thesis figure is produced.
