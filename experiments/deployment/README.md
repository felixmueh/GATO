# D01 Gazebo task worker

This implements the approved three T01 reaching goals and figure-eight periods
24/18/12 s using current `MPC_GATO.run_mpc_goals`, `run_mpc_fig8` and
`TiagoControllerOrchestrator`. It is a CPU-validated implementation proposal;
no Gazebo, native solve or physical robot was run in this batch. Matching
RTX 5060/internal T01 and the target-machine D01 preflight remain pending.
D02 standalone physical scripts are later work, outside the runner.

`inputs/t01.json` is copied byte-for-byte from the reviewed ordinary study:
SHA256 `aec47cfa3819de5adbb8633b637f5bd9bcd327cb3afc5043a856b323af5aac41`.
The original internal timing/release metadata stays in that source input;
`protocol.py` resolves the live protocol explicitly. Frozen input, model and
reference integrity are checked at loading. Screening is reused rather than
rerun. No package equality, telemetry equivalence or determinism gate is added.

CPU-only inspection from this checkout:

```sh
python3 experiments/deployment/run_d01.py \
  --case '{"id":"preflight-goal-1","parameters":{"backend":"gazebo"}}' \
  --output experiments/deployment/raw/PROPOSED/preflight --dry-run
python3 experiments/deployment/run_d01.py \
  --case '{"id":"figure8-24s","parameters":{"backend":"gazebo","preflight_output":"/absolute/successful/preflight/attempt"}}' \
  --output experiments/deployment/raw/PROPOSED/figure8-24s --dry-run
```

`--case` also accepts a JSON file path. Dry-run prints resolved configuration
without ROS, Pinocchio, native solver imports or writes. It does not qualify
the target machine. The disabled `plan.proposal.json` lists preflight and six
task cases, each once; the deployment owner supplies its reviewed commit,
isolated build arrangement and actual preflight attempt path before producing
the runnable explicit plan and runner dry-run. The legacy default runner queue
does not represent D01. New raw, build/cache/temp and logs belong beneath the
selected deployment checkout; cached runner source and supplied raw output may
be separate directories. Output must be a fresh attempt; worker-owned files
are never overwritten.

Actual execution requires an already sourced simulation profile
(`GATO_ROS_PROFILE=simulation`, `ROS_DOMAIN_ID=1`, Cyclone DDS) and
`GATO_MACHINE=rtx5060`. The worker neither sources ROS nor changes environment.
It requires advancing `/clock` and observed `/joint_states` before creating
the motion orchestrator. Domain/profile plus simulator clock are the backend
guard; they are not a physical safety qualification. Runtime safety stays on.
The orchestrator performs the existing reset motion to the frozen T01 start q.
The actual observed post-reset initial state is retained; goals stay at their
frozen absolute torso-frame poses.

Run `preflight-goal-1` before this family on its machine/setup. Task success,
sampled physical compliance, execution failure and confirmed handback are
separate metrics. A failed preflight holds all six tasks and preserves raw
failure evidence. Each subsequent case supplies `parameters.preflight_output`
pointing to that successful attempt. Reuse checks its backend/machine/input,
canonical configuration and loaded native module identity once at the relevant
boundary. There is no automatic diagnostic, alternate parameterization or
retry. Completed task negatives exit zero; execution errors exit nonzero.

See [PROTOCOL.md](PROTOCOL.md) for actual settings and observable meanings.
The focused CPU checks are:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/python/test_deployment_experiments.py
```

Eight checks pass: runtime-free dry-run; exact solve arguments/output/exception
and timer scope; task-failure cleanup with successful or failed restoration;
separate task/physical outcomes; backend/preflight hold; and observer-error
cleanup with a sibling output; and failed-recorder cleanup/finalization with
independent task and handback outcomes retained. Ordinary pytest also passed the initial six
checks, with an hppfcl library warning during startup. No CUDA solve, ROS or
motion ran. Solver/controller source is unchanged, so no new task regression
campaign was performed for this recording-only worker batch.

Implementation references are the current public reaching/tracking examples,
`tiago_tools/capture_joint_state_recording.py`, current ROS client/history
writer, and ordinary campaign `raw.py` (the reused NPZ/JSONL writer). The
ordinary writer provenance is commit `5f271b6fb4b08835401d8c7116dfd655faa75d12`.
The public live-controller base is `2db284ca8e11f42f8f3248403b879652966893ed`.
