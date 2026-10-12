# Machine coverage — manual deployment scope, 2026-10-12

Felix's latest instruction supersedes blanket repetition on every machine.
Each machine is selected by the thesis question. Full older prepared plans remain
in `successor/merged/`; this selection preserves their source, inputs, per-case
settings and historical outcomes. It does not create a new tuned protocol. Felix separately accepted the v5 square as-is and added
one Gazebo case; historical square failures remain preserved.

| Study/question | GTX1070 | RTX5060 internal | Jetson internal | Gazebo | Reason and limits |
| --- | --- | --- | --- | --- | --- |
| F01ab cold/warm budget/horizon timing | Saved successor64 cells |64 cells |Saved existing main timing study; **no rerun** |None |RTX fills desktop hardware comparison. N16/32/64/128 × SQP1/2/5/10 × PCG100/200/400/800. Jetson saved study retains its old source/stopping policy, so it is not a matched restored-policy hardware comparison. |
| F01c batching throughput |Saved15 |15 |15 |None |N32, B1/2/4/8/16 × SQP/PCG1/100,5/100,10/800; same16-candidate serial baseline, six frozen inputs and existing repeats. Gives an actually matched new-policy hardware/batch comparison. |
| F04 compiled resources |Saved8 |8 |8 |None |Zero-solve queries at N16/32/64/128 × B1/B16; architecture-dependent resource information. |
| F05 response/stages/sustained |Saved full study |None |10 workers |None |Onboard behavior: N32, B1/B16, SQP5/PCG100; first-response r0/1/2 (6), stageON profile (2),120s sustained (2). No full-cap or RTX thermal claim. |
| T01 reaching/tracking |Saved6+faster3 |2 |6+faster3 |D01 two |RTX goal-1 + figure8-12s. Jetson all goal-1/2/3, figure8-24/18/12s and faster8/6/4s proposals to show actual computation-paid tracking breakdown. Existing gates stay frozen; faster/common gates remain simulation-only. |
| BA optimizer/seed/equal-compute quality |Saved300 |None |None |None |Algorithm comparison already studied on1070; F01c supplies remote throughput. |
| BB computation-paid batching |Saved100 pairs |None |5 pairs (10 cases) |None |First five frozen pairs3626,7733,7132,7093,1265 each B1/B16, unchanged N32/default settings. Bounded onboard transfer examples, not a Jetson100-pair success-rate or reliability claim. |
| E03 force time/memory and E01 correspondence |Saved24 |None |24 |None |Full frozen six starts × two truths0/10N × original/adapted;20 measured ticks unchanged. Jetson cost/memory is the thesis question; no need for a5060force scaling claim. |
| E02 disturbance tasks |Saved outcomes |None |8 executable |None |No-estimator/adapted each0N once +10N three repeats, modeB. Four original-method tasks remain held; no1070failed-canonical exception transferred, no completed three-method onboard claim. |
| C01/C03–C06/C08 ControlChoices, X01/X02 extensions, Q02 model checks, TIAGo X03/X04 cylinder routes |Saved sweeps/checks/tasks |None |None |None |Setup/model/algorithm evidence reused on1070. Cylinder routing conclusions remain1070-specific. |
| M02 geometry/query comparison |Saved study |None |None |None |No added onboard monitor query-frequency claim. |
| D01/M03 live integration/fault reactions |Not a timing study |Separate Gazebo environment |None |2 task cases +20 fault/baseline cases |D01 goal-1,figure8-12s. M03 all20 conditions retained because each tests a distinct existing software/safety/handback path. No timing matrix or physical-robot guarantee. |
| Accepted v5 square |Saved rough5/5, strict2/5, physical pass |No internal rerun |No internal rerun |One separate five-goal square case |Exact accepted orientations with center+X,30cm placement,current weights,N32,15s/goal and final approach retained as-is. Target Gazebo canonical/safety/handback remain required. |
| Historical square failures, other visual demos and physical D02 |Original/v4 failures preserved |Withheld |Withheld |No old square or extra figure-eight demo queued |No automatic physical motion. Other live figure eights remain unchanged; physical scripts remain standalone/manual. |
| M04 effort/response/braking illustration |Reuse available1070 simulation evidence |None |No extra remote study |M03 provides focused integration |Effort, state-age, response, braking/clearance illustrations stay qualified by their recorded simulation scope; do not assert missing raw evidence. |
| F02 cylinder matrix, F03 Indy7, M01 geometry, C07 repair progression, L00 SciPy landscape, IIWA X03 |Retained records only |None |None |None |No rerun anywhere, including source/format migrations. C02 also remains record-only without a new concrete question. |
| B02/QDLDL, BA seeds-alone |Dropped; saved history retained |None |None |None |Latest accepted removals; no new tasks. |
| L01 learning, V02 broad parameterization viability |Deferred |None |None |None |Outside the present thesis campaign. |
| Q01/Q03/Q04/V01 target setup, original task suite, timing checks, canonical rough reach |Automatic relevant readiness |Automatic relevant readiness |Automatic relevant readiness |Relevant integration readiness |Not extra scientific cohorts; unchanged passed same-machine evidence reused where valid. |

The concrete plans contain **89 RTX5060 cases in four families**, **84 Jetson
cases in ten experiment entries**, and **23 Gazebo cases in three families**.
Case counts are workers, not families: timing workers contain frozen samples,
BB has two workers per pair, and M03's20 cases test distinct fault conditions. The square worker covers five
goals but is one case.
Required target task suites/canonical preflights are readiness checks in addition
to these scientific cases. Failed cases/holds retain raw data and do not become
passes through this scope selection.

This replaces the earlier712/368/27 broad executable handoff. In particular,
Jetson's completed main F01ab is reused, not repeated; its policy labels remain
explicit. The old6–24h estimate described the broad plan and no longer applies.
For the84-case Jetson plan, the earlier 2–8 h duration is a broad estimate,
not measured performance under the new recording contract. Future recording
follows [the smaller measurement contract](recording_contract.md). About 1 GiB
results/machine with 2 GiB planning allowance is a design target to check against
the recorded desktop-proxy estimates, not a sample-dropping cap. The previous 10–50 GiB raw and
120 GiB free recommendation is superseded. Report uncompressed results, compressed
exports and peak local storage separately; reserve additional system/build space.
Full matched-input timing and force-bank studies remain exempt, so they establish
a storage floor. Lower recording overhead can change sustained-loop call counts
and mode-B timing; retain recorder identities and do not claim equivalence.
