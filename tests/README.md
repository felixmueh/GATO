# TIAGo task regressions

This suite remains work in progress and is **not qualified as a passing commit
gate**. Its current short-reaching case exposes a material repeat failure. The
task assertions remain strict, so a failing execution makes the requested run
fail.

Run these before commits that change TIAGo dynamics, solver behavior, reaching
costs, warm starts, or MPC execution. They are opt-in and run separately from the
fast behavior tests and the short Indy7 tracking smoke test.

```sh
make test-tiago-tasks
```

This builds the native TIAGo N32 extension, checks CUDA access, and runs each
task in a fresh Python process. A requested run fails if CUDA, the extension,
Pinocchio, or another required dependency is missing. CPU-only PyTorch is
supported: CUDA access is established by the native solver and CUDA smoke check.
Ordinary pytest runs skip
these cases unless explicitly enabled. GPU task runs must be sequential.

To run one case against an already built extension:

```sh
GATO_RUN_TIAGO_TASK_TESTS=1 GATO_TEST_ARTIFACT_DIR=test-artifacts/tiago-tasks \
  python3 -m pytest -q tests/python/test_tiago_task_regression.py -k short_reach
```

Replace `short_reach` with `long_reach` or `figure8_tracking` to select another
case. The `task_regression` marker selects this suite. Every invocation creates
a new artifact directory, preserving failures and prior configurations.

| Case | Task and timing | Acceptance |
| --- | --- | --- |
| `short_reach` | 100.636 mm FK pose target; N32; 30 ms knots/feedback; 1 ms plant; 8 s timeout | Actual reach below 5 cm, tool-axis error below 30 degrees, joint-speed L1 below 1 rad/s; at least 5 cm executed movement |
| `long_reach` | 317.052 mm FK pose target, initially 36.920 degrees tool-axis error; same timing | Same reaching gates; at least 25 cm executed movement |
| `figure8_tracking` | Public horizontal figure eight: 32 cm x span, 8 cm y amplitude, 24 s period; N32; 8 ms knots/feedback; 1 ms plant | Complete 24 s; RMS position error at most 5 cm and maximum at most 10 cm; at least 25 cm executed movement |

Reaching uses the public pose-reaching weights, including the explicitly
corrected `qd_cost=0.1`, orientation costs, and `ctrl_lim_cost=0.05`. Tracking
keeps the public position-only defaults, including `qd_cost=0.01` and disabled
orientation costs. Both use the public zero-control initial guess, production
warm starts, one lane, and disabled final approach. Tracking supplies two
reference cycles to retain sufficient preview throughout the first full cycle.
The frozen case JSON files record complete inputs and effective solver weights.

Every case independently validates all recorded executed plant substeps:
finite q/v/u and returned plans, the initial state, clock increments, and URDF
position/speed/effort limits. Permitted bound excesses are 1e-4 rad, 1e-3 rad/s,
and 1e-3 Nm. Tracking errors use the reference at each actual observation time;
native convergence flags and the controller's next-knot error do not determine
acceptance. Full orientation error is recorded separately for reaching because
the tool-axis gate permits rotation about that axis.

Artifacts include `summary.json`, source/module identities in `receipt.json`,
`execution.npz` with all observed states, applied controls, plans and clocks,
public statistics, and subprocess stdout/stderr. Successful rough reaches do
not establish strict settling, collision clearance, or robustness across other
tasks. Limits are public soft penalties plus independent sampled validation.
The simulated feedback clock is fixed independently of solver wall time.

The 2026-10-09 full `make test-tiago-tasks` run used the isolated limit-cost
repair `17100c0`, corrected reaching `qd_cost=0.1`, and unchanged public tracking
weights. It produced **two passes and one failure**, so the full target failed:

| Case | Observed outcome |
| --- | --- |
| Short reach | The suite run hit the execution guard at 3.736 s with position and speed violations. Two additional fresh processes with unchanged inputs produced another guard failure at 1.208 s, then one pass at 0.510 s. Total: 1/3 passes; this is not a stable regression baseline. |
| Long reach | Passed at 1.500 s, ending at 49.708 mm position error, 8.097 degrees tool-axis error, and 0.431 rad/s joint-speed L1. No sampled q/v/u violations. |
| Tracking | Completed 24.000 s with 14.316 mm RMS and 26.199 mm maximum actual-clock position error. No sampled q/v/u violations. |

These are bounded outcomes, not estimates of success probability or proof of
robustness. The short failure must remain visible; retries until a pass,
relaxed gates, skips, or expected-failure marks do not qualify the suite.
The current case inputs and assertions remain unchanged after these results.

The full-run artifacts are in `test-artifacts/tiago-tasks/short_reach_8bh66qcd`,
`long_reach_dqbqb6a5`, and `figure8_tracking_55706yqj`. The two additional short
traces are under
`/workspace/gato-tiago-task-regression-20261009/short_corrected_repeat2` and
`short_corrected_repeat3`. Each receipt preserves source/module identities and
the exact effective protocol.

Earlier short qualification used `qd_cost=0.01`, and the 317 mm task failed
under that earlier configuration. Before the full suite, corrected
`qd_cost=0.1` plus `17100c0` produced two long-task diagnostic passes while the
matched unchanged native baseline failed both repeats. Those historical
diagnostics remain separate evidence and do not override the current short
case's qualification failure.

Three additional matched short controls used the archived original-cost native
solver `2a39ed1`, the same current Python worker/controller and frozen short
task, and unchanged `qd_cost=0.1` / `vel_lim_cost=0.1`. Fresh-process outcomes
were pass at 0.510 s, execution-guard failure at 1.228 s, and pass at 0.510 s.
Both rough reaching and sampled physical limits passed in 2/3 controls,
compared with the repaired native's existing 1/3 combined passes. Both variants
therefore have material short-task failures. These small counts establish no
statistically meaningful ordering or repair benefit; the suite remains
unqualified. No repaired cases were rerun for this comparison.

Original controls, exact wrapper commands, copied frozen input and per-run
native-provenance sidecars are preserved in
`/workspace/gato-tiago-task-regression-20261009/original_short_corrected_controls`.
The sidecars identify the original native separately from the current Python
worker's source commit.

The committed [qualification record](tiago_task_qualification.json) preserves
compact measurements, failed gates, raw artifact paths, and source/module/input
hashes for all eight episodes. The native cost repair and suite are isolated
from main; this checkpoint does not land that solver candidate.
