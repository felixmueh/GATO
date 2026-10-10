# Reused PCG reaching comparison on current source

This run reuses the original 100 ordered TIAGo reaching tasks and B1/B16
candidate policy on felix-devel 2db284c. It is one paired pass on the GTX1070,
with no extra repeats: 200 task episodes after the rough-reaching preflight.
The before/after statistics are an ancillary report, not a new main study.

The current cohort follows the established V01/T01 pose protocol: N32, .03 s
knots and feedback, .003 s RK4 plant, 8 s task allowance, 5 cm position,
30 degree tool-axis and joint-speed L1 <= 1 rad/s, no dwell or final approach.
Use the full frozen solver configuration in protocol.json. It includes
qd_cost=.1, vel_lim_cost=.1, orientation costs .1/2 and the established
campaign ctrl_lim_cost=.05 override, with SQP5/PCG120. The latter is the
public TIAGo preset, distinct from F01 original-PCG100 budgets.

Preserve the tested portfolio policy. The first solve has identical
constant-state/zero-control seeds. Later B1 shifts its selected plan; B16
keeps that same warm lane and draws 15 fresh goal-independent cosine/squared
progress perturbations, mapped to torques using RNEA around nominal ABA.
Choose the lowest native final merit among finite full plans, merits and
nominal-feedback RK4 prefixes, breaking ties by lane ID. Physical limits are
checked separately and do not veto selection. Reset rho each solve and retain
native duals per slot, as the original experiment did. This loop uses the
current generic constructor, common helpers and solver interface; it does not
use the ordinary live force-hypothesis selector.

historical_manifest.json and inputs/ preserve the exact ordered tasks, input
bytes, seeds and historical protocol/source identities. Do not resume an old
cohort under the new source or overwrite its raw results. The old cohort used
N64/.001 s plant/5 s/4 cm/qd=.01/vel_lim=.3/ctrl_lim=.003. Its bound-valid
successes were 17/100 B1 and 25/100 B16, with eight rescues and zero regressions;
goal-only successes were 37/100 and 51/100. Changes in the new statistics combine
source and setup changes, and cannot be attributed to the fixes alone.

Retain every input, seed, optimized output, merit, native/wall timing,
selection, step metadata, executed state/control, outcome and failure. Use
fresh raw directories and capture exact commands, source and module hashes.
The audited evaluation-only task tests from 75b17ce/4b6fa25 are included here;
no historical solver, QDLDL or B01 remedy is adopted.

This paired run is the existing BA task comparison only. The supplementary
unoptimized-input diagnostics, nominal larger-budget B1 comparison and mode B
BB implementation remain active campaign work; this run does not complete
those requirements. The existing F01c serial/batch timing is separate-budget
and separate-input evidence. For future BB, Felix approved one Jetson pass
through the same 100 B1/B16 pairs; target duration/storage discussion and
mode B readiness are still required before execution. Retained moving-pillar
cases keep their different proposal/objective policy and await compatibility
review; they are not silently replaced by obstacle-free tasks.
