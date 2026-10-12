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
