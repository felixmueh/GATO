# September23 figure-eight: current internal counterpart

One GTX1070 internal modeB episode,24s, N32/8ms knots/1ms plant, B1.
The saved September robot and Gazebo summaries agree on all solver weights:
q4, qd0.01, u0.0001, terminal80, limits0.01/0.001/0.003,
orientation0/0, SQP5, PCG120, rho0.01, mu1, tolerances0.001.
These saved settings override current reaching defaults. No solver changes.

Native source is current felix-devel
`b400bdf2bb6eea543b17cefcb0577d5de4d22ce2`; reuse its exact normal N32
module SHA256 `0c2cbc7cd217561cdba2ebc82cfa9e9eb12ddb3b442da7d9b4bd2ff0828c1666`.
The module was built from unchanged b400, with CPU registration receipt.
Runtime source `7cf5df8b132da3d9679681e8eb4c8a3d8545e418` overlays exactly
the previously reviewed fa6 Python modeB frontend and three pure task helpers.
The b400 solver, binding, public interface, common helpers, plant and generated
code remain unchanged. This explicit frontend/native distinction is recorded.

Freeze all15000saved XYZ samples at8ms (five24s cycles) with zeroRPY;
the current controller interpolates at actual feedback/prediction times.
The old CSV selected floor(t/8ms)+1 and old distance used ref(t+8ms), so
old mean errors and current timeweighted RMS use different definitions.
ModeB executes the old plan during each measured solve, activating its result
on arrival. Bootstrap solve wall is unpaid; tracking stops at24s. Tracking
resets rho each solve, with rho adaptation enabled, matching historical defaults.
Internal nominal state/dynamics differ from measured robot/Gazebo states and
ROS drive/clamping behavior. This is a counterpart, not an exact physical replay.

Use the existing single-case T01 task collector. Its inherited provisional
20mmRMS/40mmmax labels are descriptive; no retry or KKT gate is introduced.
Reuse existing compatible source qualification and the cached build identity;
no additional GPU task, calibration or regression episode. Plan2–5min wall and
150MiB raw as an estimate; retain all actual measurements and failures.
Jetson was not reachable by the cheap hostname/configuration check; no remote run.

```sh
cd /workspace/GATO-fig8-sep23-internal
/opt/gato-venv/bin/python experiments/run.py --plan experiments/fig8_sep23_counterpart/plan.json --machine gtx1070 --python /opt/gato-venv/bin/python --cache /workspace/GATO-fig8-sep23-internal/.experiment-cache/gtx1070 --output <fresh-raw-root> --dry-run
/opt/gato-venv/bin/python experiments/run.py --plan experiments/fig8_sep23_counterpart/plan.json --machine gtx1070 --python /opt/gato-venv/bin/python --cache /workspace/GATO-fig8-sep23-internal/.experiment-cache/gtx1070 --output <fresh-raw-root>
```

The adapter writes `feedback.csv`: actual solve-start time/reference/FK,
seven q/v/applied-torque values, newly planned u0, and wall/native solve times.
Applied torque is the control executed during the first completed plant substep
starting at that feedback timestamp, using the old active plan during the solve;
startup uses its bootstrap plan. `feedback_schema.json` states this association.
Every actual plant-substep state/control remains in raw events and execution
artifacts. Final metrics retain completion, physical limits, position errors,
actual work and timing, without claiming KKT convergence.
