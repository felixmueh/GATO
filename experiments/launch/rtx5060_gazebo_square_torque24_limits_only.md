# Limits-only square: next Gazebo trial

This [single-case successor plan](rtx5060_gazebo_square_torque24_limits_only.json)
uses the original successful four-corner goals and normal/settling costs, with
uniform24 Nm native **soft penalty thresholds**. These are not hard optimizer
bounds. The existing26 Nm controller trajectory guard, no-clamp policy,
collision safety, feedback timing and handback remain unchanged.

One v12 internal run passed all four rough/strict reaches, recorded q/v/control
checks and sampled actual collision clearance. All25,753 final31-row horizons
and26,595 applied substep controls are finite and within26 Nm; peak17.448 Nm.
All1,506 sampled full-body states clear40 mm, minimum43.293 mm. The earlier
heavy-cost v11 failure and original Gazebo square guard/handback failure remain
separate. This supports the next bounded live trial; it does not guarantee a
Gazebo pass. [Internal report](/workspace/GATO/thesis_results/square-torque24-limits-only/README.md).

Live runtime `ab00875006741688a98e76fc3a56ad3103dab7ff` starts from `8dbb8b8`
and changes **only** `gato/dynamics/tiago_right/tiago_right_limits.cuh` to the same
seven±24 constants as internal native `228cfa8`. It includes no internal callback
harness. Collector `529fd86863ddb4bc0eaa50f3d2171a2f10b92a88` explicitly selects
the exact v12 proposal and this live runtime; input SHA256 is
`ec4dc077930d19032caa197564a5fdb2def28b5e9c9f31f50ee8f86f8eff5401`.
Native source identity gives a distinct matching build in the existing cache.
Previous results/builds/plans are retained; no figure-eight, goal1 or completed
Jetson study is selected. Physical scripts/deployment remain unchanged and held
pending review of this Gazebo outcome.

Publish from a host with working origin access; publication has not been
verified from the container after its SSH host-key failure. No push or security
setting change occurred during preparation. Required refspecs:

```sh
git push origin experiment/simple-runner \
  ab00875006741688a98e76fc3a56ad3103dab7ff:refs/heads/experiment/deployment-source-ab0087500674 \
  529fd86863ddb4bc0eaa50f3d2171a2f10b92a88:refs/heads/experiment/deployment-source-529fd86863dd
```

Finish active work first. On RTX5060, use the existing PAL/Gazebo environment
and its **active ROS-capable Python**; inspect the printed machine/configuration
and usual controller-state prerequisites. From the runner checkout on the
correct branch, prepare and inspect the one-case plan:

```sh
git pull --ff-only origin experiment/simple-runner
GAZEBO_EXPERIMENT_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
"$GAZEBO_EXPERIMENT_PYTHON" -c 'import numpy,pinocchio,rclpy; import json,sys; sys.path.insert(0,"experiments"); import machine_launch; machine_launch.verify_sources(json.load(open("experiments/launch/rtx5060_gazebo_square_torque24_limits_only.json")))'
"$GAZEBO_EXPERIMENT_PYTHON" experiments/run.py --machine rtx5060 \
  --plan experiments/launch/rtx5060_gazebo_square_torque24_limits_only.json \
  --python "$GAZEBO_EXPERIMENT_PYTHON" --cache .experiment-cache/rtx5060-gazebo \
  --output .experiment-results/rtx5060-gazebo-square-torque24-limits-only-v12-20261013 --dry-run
```

Expected: only `SquareTorque24-LimitsOnly-Gazebo/square`. After manual setup
review, launch once. The runner builds the source-matched RTX120 N32 module,
runs this new cohort's canonical preflight, then the square. A preflight failure
holds the square; do not relax the26 Nm guard or retry until green.

```sh
"$GAZEBO_EXPERIMENT_PYTHON" experiments/run.py --machine rtx5060 \
  --plan experiments/launch/rtx5060_gazebo_square_torque24_limits_only.json \
  --python "$GAZEBO_EXPERIMENT_PYTHON" --cache .experiment-cache/rtx5060-gazebo \
  --output .experiment-results/rtx5060-gazebo-square-torque24-limits-only-v12-20261013
"$GAZEBO_EXPERIMENT_PYTHON" experiments/collect.py \
  --output .experiment-results/rtx5060-gazebo-square-torque24-limits-only-v12-20261013 \
  --export .experiment-work/rtx5060-gazebo-square-torque24-limits-only-v12-20261013-results.tar.gz
```

Keep any task/physical/guard/handback failure and partial output. Collect after
interruption too; copy the archive to desktop `_expr_results/` and retain a raw
backup. Review the actual live result before physical adoption. Preparation
contains no local native build, GPU/ROS call, live execution or physical motion.

Preparation checks passed: one focused CPU contract check of the exact runtime
patch, frozen input, collector/preview, preserved physical scripts and distinct
build key; the one-case dry-run; source/input/build pins, fresh result path and
shell syntax. Earlier unchanged adapter checks were reused.
