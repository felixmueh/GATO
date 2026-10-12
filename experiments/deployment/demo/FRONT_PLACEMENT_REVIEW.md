# Front-square placement review — 2026-10-12

Successor worktree `/workspace/GATO-square-front-placement`, branch
`experiment/square-front-placement`, starts from finalized v5 documentation
`5021b644`. The accepted v5 geometry and its historical data remain unchanged.
No new square protocol is adopted, and no GPU, Gazebo or physical motion ran.

The full-body model has torso-to-base identity rotation and torso translation
`[-0.0245, 0, 0.9457]` m for torso lift 0.10 m. Torso +X is forward; -Y is robot
right. Full-body FK expressed in the torso matches the arm-only FK to 2.1 µm
and 0.00000051 rad at the initial pose. The old `[0.50,-0.40,0]` m center is
indeed 40 cm right of the centerline; no rotation/frame mismatch was found.

The bounded screen keeps a vertical 30 cm square, inward diagonal tool-Z at
corners, center tool-Z +X, original goal ordering and body posture. Each reached
goal receives twelve axial twists at 30° spacing and two IK starts (previous
witness and initial pose), at most 150 function evaluations each. Exact witness
criteria remain position <1 mm, full orientation <0.02 rad, bounds with 0.001 rad
margin, and monitored minimum collision clearance ≥4 cm. Up to three ranked
endpoint candidates receive 21 joint-interpolation collision samples from the
previous witness. Joint/wrist travel, full rotation change and joint margin
guide ranking; they are descriptive comfort proxies, not a human-comfort proof.
The collision URDF and blacklist match the existing runtime monitor, with torso
0.10 m, head zero, left arm `[.36,-1.83,.47,-2.35,0,-1.2,0]` rad, and
right initial pose `[-.39,-1.73,-.38,-2.35,0,-1.21,.04]` rad. Wheels/grippers use
the monitor's fixed neutral values. This is an assumed simulated body posture,
not a current observation of the robot or Gazebo.

| Center torso m | Sequence screened | First failed witness / limitation |
|---|---|---|
| [.50,0,-.10] | first two corners accepted | top-left best 5.13 mm / 3.05°; wrist joint6 at lower bound |
| [.45,0,-.10] | first two accepted | top-left exact alternative: 1.36 mm collision clearance; other best ≈5 mm / 3° at wrist bound |
| [.50,-.05,-.10] | first three accepted | bottom-left best 10.31 mm / 7.31°; wrist joint6 at upper bound. Prior top-left witness needs 6.62 rad joint-vector travel |
| [.35,0,-.10] | first corner fails | best 43.38 mm / 4.99°; elbow lower bound; 35.75 mm clearance |
| [.40,0,-.10] | first corner fails | best 7.75 mm / 0.98°; elbow lower bound; 35.53 mm clearance |
| [.55,0,-.10] | first two accepted | top-left exact witness, 5.61 mm clearance; wrist near upper bound |
| [.60,0,-.10] | first two accepted | top-left exact witness, 23.48 mm clearance; wrist near upper bound |

First batch preserves 240 IK attempts. Felix then questioned forward distance;
the authorized distance-only batch preserves another 192 attempts. The latter
changes only X distance to 0.35/0.40/0.55/0.60 m and keeps Y=0, Z=-0.10 m.
Neither batch yields a complete precise witness sequence. These negative strict
screens do **not** prove failure under the demo's 5 cm/30° rough reaching gates,
global IK infeasibility, a solver defect, or collision along an unexecuted MPC
trajectory. A joint-interpolation witness is not a dynamic/swept certificate.
No tolerances are relabeled. Further geometry decisions belong to Felix/root.

Raw receipts and per-attempt results:

- `/workspace/GATO/_expr_results/square-front-placement-cpu-20261012/screen-001/`
- `/workspace/GATO/_expr_results/square-front-placement-cpu-20261012/screen-002-distance/`

Each folder contains `screen.json` and its exact `screen_source.py`.
The first folder's `geometry.png` shows the actual full-body URDF at the stated
initial posture, front and top projections, old v5 and two front candidates.
It is static setup geometry, never presented as executed motion.

Regenerate the existing static drawing:

```sh
cd /workspace/GATO-square-front-placement
python experiments/deployment/demo/plot_front_screen.py /workspace/GATO/_expr_results/square-front-placement-cpu-20261012/screen-001
```

Do not rerun old screens to migrate their format. The source screen's current
four X centers represent the separately authorized distance successor, not the
first batch. No runner plan, solver, controller or task gates were changed.

## Latest user decisions: 20 cm and joint wrist relaxation

Felix explicitly authorized 20 cm and a bounded front-placement screen, then
continuous 360-degree axial wrist freedom and choosing relaxed corner poses
together. The fixed-roll 12-center successor preserved 936 additional failed
attempts. Its 4-rad comfort filter was an overly strict interpretation; it
rejected no otherwise valid failed-goal endpoints, because every failed goal
had zero endpoint candidates. Later checks keep travel descriptive/ranked.
One midline continuous-roll screen (16 calls), one final right-shoulder screen
(20 calls), and one natural historical center-seed check are preserved too.

The selected four-corner setup is a **20 cm square centered torso
[0.50,-0.10,-0.10] m**. All four corners have precise IK witnesses and ≥4 cm
monitored endpoint clearance; all fixed 21-sample joint interpolation checks
pass. The fifth center+X pose remains **HELD** after precise IK/clearance
failures. `front_corner_setup.json` is a static four-corner setup proposal,
not an enabled four-goal replacement for the original five-goal live task.
The accepted `proposal.json` and standalone motion scripts remain unchanged.

One jointly constrained four-corner CPU posture refinement converged after
15/30 allowed SLSQP iterations (57 s). Its objective combines squared joint
transitions (arm weights1.0; wrist weights1.35) and a small inverse-square
joint-limit proximity penalty; fixed position/tool-Z equalities, physical
joint bounds and the original 4 cm clearance apply. Full rotations are frozen
after the selection; runtime orientation costs are unchanged.
Objective47.17→27.70; total joint-vector length10.34→8.57 rad; largest individual
step3.784→2.936 rad; top-left vector transition5.154→3.747 rad. Minimum wrist
margin0.273→0.558 rad, minimum all-joint margin0.215→0.182 rad. Endpoint
clearances60.25/73.10/62.37/56.20 mm; transition sample minimum46.02 mm.
The remaining largest joint step is168 degrees: improved local selection,
not global optimality or a claim of fully comfortable motion.

Latest raw: `screen-007-joint-relaxation/` under the same raw root. No failed
checks or old protocols are relabeled, and no GPU/controller/robot motion ran.
Final full-body URDF four-pose image, compact poses, before/after inputs and
self-contained report/PDF are in
`/workspace/GATO/thesis_results/square-front-postures/`.

Regenerate the final drawing with:

```sh
python experiments/deployment/demo/render_front_ik.py /workspace/GATO/_expr_results/square-front-placement-cpu-20261012/screen-007-joint-relaxation
```

Center decision and matching internal motion qualification remain prerequisites
for any updated live/robot protocol. No remote runner plan was changed.

URDF confirms joint7 is axial tool roll (+Z axis, identity-rotation fixed tool
joint), with physical limits ±2.618 rad. It stays near0.040 rad at all four
refined corners. The remaining large top-right→top-left transition is joint6
+2.936 rad and joint5+2.229 rad, not a tool-roll flip. Axial orientation freedom
during selection still respects physical joint bounds; no exhaustive360-degree
or globally most relaxed claim is made.

## Five-goal internal diagnostic — final outcome

Felix authorized adding center-only collision-constrained posture selection and
one GTX1070 internal MPC diagnostic. Center s009 is an exact endpoint with
40.10 mm monitored clearance, but its 21-sample straight joint interpolation
fails. One bounded last-two posture refinement s010 has valid endpoints and
passes its coarse constraints but still fails the original21-sample witness.
All failed checks remain preserved. No more CPU search was performed.

Root released one fresh canonical and one five-goal diagnostic from collector
`9b267dde977cc2a3dd2792029a106435f7c0b278`, separate `proposal_front.json`.
It keeps all four corner target full poses exactly `089141b`, adds the s009
center+X target, and retains production `initial_guess=None`, warm starts,
N32/.03/3ms/modeB/15s pergoal/original settling costs. IK witnesses select goals;
they are not supplied as controller initializations or presumed executed paths.
Unchanged fa6 runtime/module and current reviewed compact TaskRecorder reused.

Canonical task/plant-limitPASS. One episode completes rough5/5, strict3/5;
bottom-left and center strict timeouts retained. All recorded plant-substep
q/velocity/applied-control checksPASS with zero excess; all73167feedbackstates
finite. **Actual sampled full-body collisionFAIL**:4390realfeedbacksamples at
10ms targetspacing plus boundaries,2216below40mm/1338contact-overlap samples.
Firstthreegoals clear (min46.17/57.20/52.57mm), bottomleft1351below/1327contacts,
center865below/11contacts. Dominant contactpair leftarm1/rightarm3. This is
modeled simulation under the declared body posture, not observed robot contact.
Internal arm-only plant does not enforce full-body collision monitor. Live
adoption is held; no solver repair, tuning, rerun, Gazebo or physical motion.

Finalcenter40.27mm/axis26.02°/fullrotation26.08°/speedL1.00897rad_s. Its achieved
wristmargin is.00744rad (~.43°), not the endpoint IK witness margin. Top-left
already departs>5.24rad from its witness despite accurate pose/collisionclearance,
with joint7 near its upper limit; bottomleft joint7 is~.245° below the upper
limit. Optimized static joint comfort did not transfer through EE-only MPC.
No cause/solver-defect or global-infeasibility claim is made.

Duration43.8686splant/186.42srecorderwall;73167solves. Wrappermedian/p95
.315/2.275ms, native.224/2.165ms, descriptive with recording/plant overhead.
Raw `/workspace/GATO/_expr_results/square-front20cm-v6-internal-20261012/`;
new compact report/full-robot actual-motion GIF/PDF in
`/workspace/GATO/thesis_results/square-front-center-motion/`.
The static four-posture export stays separate; old v5 and all CPU failures remain.
