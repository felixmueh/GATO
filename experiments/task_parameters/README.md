# Prepared task parameter studies

This experiment-only source starts from live runtime `8dbb8b8`. Internal tasks
continue using the unchanged `fa6f6be` runtime supplied through `--source`.
Native solver, generated dynamics, soft penalties and physical URDF limits are
unchanged in arithmetic. This large-horizon testing successor adds only PCG
shared-memory launch setup; ordinary float32 N<=128 adds no CUDA queries or
attribute calls. It is separate from the original fa6/90bdb small-horizon sources.
No optimizer solve, ROS motion, Gazebo run or publication was performed
while preparing this source. Target canonical, per-module readiness and pilots
remain necessary before measurements; CPU tests do not qualify task behavior.

`worker.py` reuses the current runner's compact task/deployment adapters. Supply
the runner repository through `--runner-root`, the pinned runtime through
`--source`, and a frozen directory containing the original `t01.json` and
`v01.json` through `--inputs`. The verified collector commit is supplied explicitly
through `--collector-pin`; its actual worker and protocol hashes are recorded.
When extracting the worker from Git, extract sibling `protocol.py` as well.

Cases select `goal-3`, `figure8-6s` or `figure8-12s`, with parameters `N`, `dt_s`,
`sqp_max_iter` and `pcg_max_iter=120`. The canonical preflight always uses
N32/30 ms/SQP5/PCG120 and the original rough-reaching task, independent of the
grid case. Internal execution remains mode B, with the existing solve-time clock,
3 ms RK4 integration, initialization, warm starts and task costs.

References preserve the original frozen T01 geometry, phase, initial offset and
orientation. A complete original 12-second sampled period is interpolated
periodically onto the selected knot spacing, with phase rescaled for the
6-second task. The reference includes at least the whole task plus `(N-1)*dt`
preview, using ceiling allocation for noninteger duration/spacing. Prediction
and metric reference spacing both use `reference_dt_s=dt_s`. The 12-second
N32/30 ms overlap matches the frozen reference within the CPU check's 2e-15 m
tolerance; the old reference simply retained one additional unused endpoint.
Changing spacing introduces sampled-reference interpolation, recorded in the
protocol, rather than changing the path or shortening task duration.

The unsafe option is `--dangerously-disable-safety-stops`. It defaults to false
and can be supplied only explicitly on the invoking CLI. Case/plan parameters
cannot activate it. It requires the explicit Gazebo backend and the simulation
ROS environment/domain 1; the live worker also checks an advancing `/clock`.
Standalone reaching/tracking examples expose the same option together with
`--ros-tiago --ros-backend gazebo`. Robot/internal/non-simulation use is rejected.
The option prints a warning and is recorded in protocol and metrics.

Report-only mode bypasses stopping for proposed-trajectory torque, joint
position, joint velocity, collision margin and collision-body speed. It retains
their calculations, the original thresholds, no clamping, invalid/nonfinite
data checks, stale state/trajectory checks, monitor/service failures and handback.
Neither the old monitor-off option nor torque clamping can be combined with it.
All rows of each final trajectory accepted by the torque worker are checked;
latest-queue discarded trajectories are outside that worker summary's scope.
Accepted-horizon torque maxima/counts are separate from independently observed
published effort maxima/counts, which are not measured actuator torque.
Physical summaries retain checked/violating-state counts, kinds and first/last
events. Collision reporting uses the first threshold-violating pair's lower-bound
distance estimate, not a global minimum or an exact contact test; contact status
is explicitly unavailable. Full named state histories remain recorded.

`physical_pass` stays false or unavailable when physical criteria fail or cannot
be evaluated; `observed_urdf_q_v_effort_pass` preserves the original narrower
observed-URDF result. In report-only Gazebo preflight, these physical negatives
are recorded without holding the family. Task failure, execution errors and
unconfirmed handback still hold it. Safe preflight retains physical stopping and
physical qualification. Runtime defaults outside this explicit option are safe.

CPU validation (no native loads or ROS execution):

```sh
/opt/gato-venv/bin/python3 -m pytest -q experiments/task_parameters/test_parameters.py tests/python/test_tiago_controller_validation.py tests/python/test_tiago_controller_shutdown.py tests/python/test_tiago_controller_modes.py
```

Initial independent review recorded 53 passed; the final finite-guard followup
recorded 54 passed, with the existing hppfcl deprecation warning. The final focused
subset contains 10 checks. Both public CLI help commands
and compilation checks pass. The first new reference check incorrectly required
the old extra endpoint and failed on array length; its overlap comparison was
corrected without changing reference generation or numerical tolerances. All
scientific target runs remain held pending the separate authorized execution.

A small worker-local recorder guard preserves each event before rejecting
nonfinite final solver output, executed state/control or live feedback. The
existing failure finalization records this as an execution error. It adds no
internal torque/joint/collision gate and does not inspect intermediate SQP
diagnostics. The unchanged internal runtime alone did not enforce this policy;
this guard is part of the new study's collection setup. The followup focused
subset contains 10 passing checks.

## Large-horizon testing successor

Standalone internal core commit `3bc10fff411ad87078e94c8bfc5d96830b82a66e`
is cleanly cherry-pickable: shared resource helper plus two BSQP constructor
hooks, before allocation and outside solve timing. Equivalent live cherry-pick:
`5f258acfc074c69852719df39b080d677dd7465e`. No kernel arithmetic changed.
The resource probe is a separate commit and now queries the same policy and
actual configured function allowance without launching any kernel with null
operands. An unsupported requested dynamic+compiled-static total yields a clear
error instead of proceeding to solver allocation/execution.

Allowed N now includes 256 and 512. Hardware support is conditional; this does
not qualify either horizon for tasks. Local GTX1070 resource probes compiled
and were queried once each: N32 uses 13,764 dynamic+32 static bytes and fits
49,152 with no opt-in; N256 requests 76,484+32 and exceeds that device's actual
49,152 opt-in limit. Both performed zero kernel launches. Raw evidence:
`/workspace/GATO/_expr_results/pcg-shmem-optin-validation-20261014/`.
Successful opt-in on RTX/Orin, N512 resource support and task behavior remain
remote checks, not local qualification. No new native optimizer/task run occurred.

Before each new horizon's task pilot, manually perform capacity then
`worker.py --native-readiness` with the ordinary CLI paths, internal backend and
case parameters N=<new N>,dt_s=0.03,sqp_max_iter=1,pcg_max_iter=120.
The worker uses the frozen v01 state/goal, keeps the requested N, and executes
exactly one cold public BSQP B1 call with constant-state/zero-control seed.
It retains input/output NPZ, finite flags, wrapper/native time and exact
source/module/input provenance. There is no plant/controller/ROS/motion and no
KKT/intermediate-array convergence gate. Nonfinite final packed execution
output or a native exception fails readiness; it is not a reaching outcome.
Neither the unsafe flag nor canonical-preflight mode can be combined with it.
The two CPU binding fixtures are plumbing checks only. Combined relevant CPU
checks (parameter worker plus shared-memory API stubs) recorded 13 passed.
