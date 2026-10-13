# Limits-only square Gazebo successor

This collector binds Gazebo to the exact `proposal_torque24_limits_only.json`
input of the passing v12 internal run and live runtime
`ab00875006741688a98e76fc3a56ad3103dab7ff`. That live runtime starts from
`8dbb8b8` and changes only the seven native soft control-penalty thresholds to
±24 Nm, exactly matching internal native source `228cfa8`. It contains no
internal callback harness or controller repair. The 26 Nm controller guard,
no-clamp policy, collision safety and handback remain unchanged.

All original v10 goals and costs remain: four inward-facing corners, 10 cm
square centered at torso `[0.50,-0.35,-0.15]` m, no center reach, N32/.03 s/
15 s per goal and original final approach/initialization. The new input differs
only in cohort metadata from the accepted v10 input; its frozen pending-status
text records preparation history, not the final outcome. The passing internal
run is `_expr_results/square-torque24-limits-only-v12-internal-20261013/`.
All four rough/strict reaches, q/v/applied-control and sampled clearance checks
pass. Every 25,753 final31-row horizon and 26,595 applied substep controls is
finite and within26 Nm, with maximum17.448 Nm. All1,506 sampled actual full-body
states satisfy40 mm clearance, minimum43.293 mm. This is one internal run,
not a guarantee that the live experiment will pass.

Standalone `demo_gazebo.py` explicitly selects the new input and still defaults
to preview. Supply the separate live-runtime checkout and its matching N32
module for execution. The original `demo_robot.py` and `demo_internal.py` are
byte-identical to collector91118e3 and still explicitly select the historical
v10 input. They are not limits-only deployment scripts. Physical deployment
remains held pending Gazebo review; no robot script or physical setting was
updated. Upward tracking keeps its separate old collector and is not rerun.

CPU checks passed for the exact selected input, all original scientific fields,
seven matching24 Nm constants, live runtime's one-file diff, preserved26 Nm
guard/clampfalse and Gazebo preview. No build, GPU/ROS call, motion or push ran.
