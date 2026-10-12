# Accepted four-corner square deployment binding

This packaging successor explicitly binds `run_demo.py` and all three standalone
scripts to `proposal_right10_roll.json`. The proposal bytes are unchanged from
the passing internal run at `a96860397426d82337bd57d49db8a30b33afd06b`.
The historical `proposal.json` and previous commits remain intact.

The selected motion is four inward-pointing corners of a 10 × 10 cm vertical
square centered at torso `[0.50,-0.35,-0.15]` m, in the exact passing order and
full rotations. There is no center reach. N32, 30 ms knots, 3 ms internal plant,
15 s per goal, costs, final approach, strict settling, initialization, collision
margin and controller settings are unchanged. All four reaches and recorded
plant checks passed; actual saved-state collision sampling passed at 43.05 mm
minimum against the unchanged 40 mm margin. This remains one internal run and
a sampled check, not live or physical robot qualification.

The proposal has no tracking tasks. The separately selected upward 12-second
figure-eight retains collector `2574409` and its own frozen proposal/reference.
Do not merge or infer tracking qualification from this square. The input's
inherited `accepted_square_evidence` field describes historical v5 evidence;
the current four-corner evidence is the v10 record cited above and the compact
`thesis_results/square-four-corner-final/` report.

Internal runtime stays `fa6f6bede286c122cc41ac65f42a583da84290e0`; live Gazebo
and physical runtime stays `8dbb8b8d6dc1d65e902373a9aa58693e33329c14`.
No solver/controller/runtime changes or GPU rerun occurred for this binding.
The runner uses the live source separately from this collector commit.

Standalone `demo_robot.py` defaults to preview and square-only selection.
Its top-level constants print the reviewed four goals and match the selected
input. Motion still requires the exact N32 module, successful matching canonical
preflight, manual configuration review, explicit per-task confirmation, robot
environment guard, collision safety and controller handback. Timestamped raw
directories and the separate manual-robot index remain. No robot action is
performed during packaging or preview.

Packaging checks passed for exact selected input bytes and all resolved fields,
all three standalone previews, four-goal order/center exclusion and explicit
robot binding. Production solver/controller/runtime source is unchanged.
Input SHA256: `8f1725b1ee38ee8b4b479be4d10637ed112f08e83398bdcf0c5fa091f1ddafd2`.
