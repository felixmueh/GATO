# Prepared task parameter studies

This experiment-only source starts from live runtime `8dbb8b8`. Internal tasks
continue using the unchanged `fa6f6be` runtime supplied through `--source`.
Native solver, generated dynamics, soft penalties and physical URDF limits are
unchanged. No build, GPU solve, ROS motion, Gazebo run or publication was performed
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

Independent review recorded 53 passed, with the existing hppfcl deprecation
warning. The new focused subset recorded 9 passed. Both public CLI help commands
and compilation checks pass. The first new reference check incorrectly required
the old extra endpoint and failed on array length; its overlap comparison was
corrected without changing reference generation or numerical tolerances. All
scientific target runs remain held pending the separate authorized execution.
