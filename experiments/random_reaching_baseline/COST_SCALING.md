# Physical interpretation and scalar cost scales

This is CPU arithmetic for the actual reviewed C2 helper, not another solver run.
It supports the bounded M2/M3 comparison: current cost margins, q-limit weight 0.1,
speed-limit weight 1, effort-limit weight 0.03, qd regularization 0.05,
`u_cost=0.0001`, first mu=1 (M2) and then mu=10 (M3) at identical weights. Pose costs,
physical limits, mode-A timing and stopping stay fixed. The earlier M0/M1
observations remain distinct evidence. These calculations motivate choices;
they establish no physical guarantee or formal convergence claim.

Latest direction: after the joint-only M6–M8 comparisons, Felix requested a
large-first physical search and narrowing of the joint/speed weight block.
M11–M15 passed the fixed task2342 physical screen. M14 is selected for the full
twelve-task development pilot. It completed with twelve physical passes, one
rough arrival, eleven timeouts and no execution errors, and was frozen at
2026-10-15 11:43:04 UTC before the separate 100-task validation began. Validation
is now running without further tuning; horizon execution remains held. The M5 stopping decision
and report remain preserved checkpoints. Physical checks and goal costs stay fixed.

Both M2 and M3 subsequently timed out in their matching eight-second canonical
preflights, with physical checks passing and final position errors 90.568 and
90.694 mm. Their dependent twelve-task pilots were held; these are not completed
twelve-task cohorts. M4 therefore rebalances both margins and weights, as
described below. Its canonical preflight reached at 0.54 s within physical
limits; the assigned development pilot is separate evidence.

## Formula and units

For a bound distance d and delta=0.01*(cost upper−cost lower), the helper is
`−log(d)` for d≥delta. Otherwise, with t=(delta−d)/delta, it is
`−log(delta)+t+t²/2`, with derivative `−(1+t)/delta` and curvature `1/delta²`.
The two-sided cost adds the lower/upper distances, subtracts the upper-distance
gradient, and adds both curvatures. Thus points at or outside a cost/physical
bound use the quadratic continuation; evaluating an ordinary logarithm there
would be wrong. The 1% continuation width and the interior cost margin are
different parameters.

The tables report one scalar at one running knot. Native cost sums knots without
a dt factor. Cost values below subtract the same-interval center value to remove
the arbitrary raw-log constant; derivatives are unchanged. Gradient units are
cost per rad, per rad/s or per Nm; curvature units are cost per squared variable
unit. Do not compare derivatives across these categories directly. Ordinary
velocity/effort quadratics are `0.5*qd_cost*v²` and `0.5*u_cost*u²`. The position
weight named `q_cost` acts on Cartesian meters, so there is no matching ordinary
joint-angle quadratic to compare with the joint-bound helper.

## Speed: useful limits also add damping

With current cost bounds ±2.25 rad/s and speed-limit weight 1:

| Speed rad/s | Relative limit cost | Gradient | Curvature |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0.3951 |
| 1 | 0.2201 | 0.4923 | 0.7347 |
| 2 | 1.5612 | 3.7647 | 16.0554 |
| 2.5, physical limit | 31.2080 | 167.6907 | 493.8715 |

At the previous speed-limit weight 0.1, every table entry is one tenth as large.
For qd=0.1, the ordinary quadratic at speed 1 has cost 0.05, gradient 0.1 and
curvature 0.1; qd=0.05 halves all three. Thus the strong limit cost already
dominates ordinary damping near speed 1, not only near a violation. At zero,
combined velocity curvature is 0.1395 for M0 (0.0395+0.1), versus 0.4451 for M2
(0.3951+0.05). At speed 1, halving qd from 0.1 to 0.05 reduces the combined
strong-limit gradient only 8.4%. M2 still substantially discourages motion; the
qd reduction moderates that effect rather than undoing the stronger limit cost.

Changing the velocity fraction from 0.90 to 0.80 gives cost bounds ±2.0 rad/s.
At weight 1 the curvature at zero rises to 0.5. At speed 2 the gradient becomes
49.75 and curvature 625.06; at physical speed 2.5 they become 362.28 and 625.05.
The relative cost at 2.5 becomes 107.73. A fraction of 0.80 is therefore a
substantial additional restriction in practice. Test it only if current-margin
results justify more room before the physical bound; expect possible slower
progress or more finite timeouts.

### Balanced wider candidate

A wider velocity margin need not retain the tenfold speed weight. Choosing
inner bounds ±2.0 rad/s, `vel_lim_cost=0.2` and `qd_cost=0.05` gives total
zero-speed curvature `0.05+2*0.2/2²=0.15`. That is only 7.52% above M0's
0.139506, while the physical-bound limit gradient is 72.4556, 4.321 times M0's
16.7691. This deliberately reshapes ordinary-motion drag versus boundary
deterrence rather than multiplying every coefficient by ten.

| Setup | Cost speed bound | Speed weight | qd | Total curvature at zero | Limit gradient at 2.5 rad/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | ±2.25 | 0.1 | 0.1 | 0.1395 | 16.769 |
| M1 | ±2.25 | 1 | 0.1 | 0.4951 | 167.691 |
| M2/M3 | ±2.25 | 1 | 0.05 | 0.4451 | 167.691 |
| Balanced wider M4 | ±2.0 | 0.2 | 0.05 | 0.1500 | 72.456 |
| Firmer wider M5 | ±2.0 | 0.6 | 0.05 | 0.3500 | 217.367 |

The close damping comparison is local to zero speed. At speed 1, combined
limit-plus-ordinary gradients are 0.14923 for M0 and 0.18333 for the balanced
candidate, a 22.85% increase. At speed 2, the new cost boundary, they are
0.57647 and 10.05. Thus the candidate still strongly discourages approaching
2 rad/s; it does not preserve the old cost throughout the speed range.

M4 combines this velocity choice with position inset 0.10 rad, effort fraction
0.85, `q_lim_cost=0.01`, `ctrl_lim_cost=0.01`, unchanged `u_cost=0.0001`
and mu=10. The lower q weight avoids the large canonical posture-centering
gradient examined below; effort weight rises a moderate 3.33-fold from M0,
alongside its wider cost margin.
This is a physically motivated combined parameterization change. It does not
isolate a margin effect or erase the failed M2/M3 canonical observations.
Its changed build flags/module and all retained outcomes belong
to a separate recorded cohort; physical checks and assigned tasks stay fixed.

### One final firmer candidate

M4 completed all twelve tasks with three rough arrivals, eleven physical passes
and three jointly valid arrivals. Task 2342 retained a 0.039586 rad maximum
joint-bound excess and maximum velocity ratio 1.349335. Every torque check
passed, with the largest ratio 0.852943. The reserved 88 remained held at that
decision point.

M5 is one final bounded follow-up at the same wide margins, qd=0.05,
`u_cost=0.0001` and mu=10: q-limit weight 0.015, speed-limit weight 0.6 and
effort-limit weight 0.02. Those are 1.5, 3 and 2 times the respective M4
weights. Other goal weights, assignments, source/module, timing and stopping
stay fixed. Its canonical preflight passed at 0.84 s with physical checks
passing. Its completed twelve-task pilot had three rough arrivals, eleven
physical passes, three jointly valid arrivals and no execution errors. Every
speed/effort check passed, with maximum ratios 0.778744 and 0.833065, but task
2342 retained maximum q excess 0.0473932 rad. Its q excess was larger than M4's
0.0395858 rad even though the speed violation was removed. The valid arrivals
3626/9651/8270 occurred at 4.53/7.65/7.26 simulated seconds.

At the M5 checkpoint the cost search was stopped with that known q failure:
the study had not achieved a fully limits-compliant baseline. A later explicit
user instruction reopened only joint-margin/joint-weight development before
validation. The original M5 report and validation preparation remain intact.
The 100-task validation was never launched and stays held; the historical
reserved 88 also remain held. No validation outcome informed the new decision.

Near zero speed, M5 total curvature is 0.35, between M4's 0.15 and M1's
0.4951. At physical speed 2.5 rad/s, its limit gradient is 217.3667, three
times M4 and about 12.96 times M0. At speed 1, its total gradient is 0.45,
versus 0.18333 for M4 and 0.59231 for M1. Thus it asks for firmer boundary
deterrence without restoring all of M1's small-speed drag; it can still slow
progress. A larger penalty does not imply monotonic improvement.

![One-joint velocity costs for M0, M1, M4 and M5. Each curve is its C2 limit cost
relative to the interval center plus its ordinary qd quadratic. The inset shows
the same linear cost at low speeds. Vertical lines mark the two cost bounds
and the unchanged physical speed limit.](velocity_cost_curve.png)

The figure is scalar objective accounting, not a measured trajectory, common
evaluation objective or comparison of solver performance. The log continuation
begins 1% of the cost-interval width inside each bound (2.205 or 1.96 rad/s);
the markers show the bounds themselves, not the beginning of that continuation.

## Torque: compare with holding demand and the same-variable effort term

For the same twelve initial postures, the independent CPU gravity calculation
gave representative absolute medians/maxima of 10.461/22.954 Nm on joint 2
(43 Nm physical limit) and 6.072/9.799 Nm on joint 4 (26 Nm limit). These are
rounded posture statistics, not dynamic demand or measurements of applied torque.
The following uses current 0.90 effort cost fractions and strong weight 0.03:

| Joint/point | Torque Nm | Limit gradient | Ordinary effort gradient |
| --- | ---: | ---: | ---: |
| J2 median gravity | 10.461 | 0.000452 | 0.001046 |
| J2 maximum gravity | 22.954 | 0.001419 | 0.002295 |
| J2 physical limit | 43 | 0.292484 | 0.004300 |
| J4 median gravity | 6.072 | 0.000713 | 0.000607 |
| J4 maximum gravity | 9.799 | 0.001302 | 0.000980 |
| J4 physical limit | 26 | 0.483723 | 0.002600 |

The old effort-limit weight 0.003 gives one tenth of those limit gradients;
the ordinary effort gradient stays `0.0001*u`. At zero torque, strong limit
curvature is 0.0000401 for a 43 Nm joint and 0.0001096 for a 26 Nm joint,
versus ordinary effort curvature 0.0001. Thus stronger torque cost is comparable
with ordinary regularization around typical holding demand, and much stronger
near physical effort limits. It discourages unnecessary effort while leaving
the required holding torque to be balanced by the full objective and dynamics;
it is not a torque cap or gravity compensation controller.

At either physical torque limit, relative limit cost is 0.09362 with weight
0.003 and 0.93624 with weight 0.03. The equality between 26/43 Nm here follows
from equal normalized fractions; gradients and curvatures differ with units.
With fraction 0.85, physical-bound relative cost becomes 0.18363/1.83626 at
those two weights. Strong physical-bound gradients become 0.44382 for 43 Nm
and 0.73400 for 26 Nm. The corresponding cost bounds are ±36.55 and ±22.1 Nm.
These wider margins are a conditional additional change, not needed merely
because a previous valid trajectory used nearly all available torque.

M4 instead uses the intermediate effort-limit weight 0.01 at fraction 0.85.
Its physical-bound gradients are 0.14794 for 43 Nm and 0.24467 for 26 Nm,
with zero-torque curvatures 0.00001497 and 0.00004095. At the rounded maximum
initial gravity demands above, J2/J4 gradients are 0.000567/0.000499, versus
ordinary effort gradients 0.002295/0.000980. This keeps a modest cost around
holding demand while strengthening near-bound deterrence compared with M0.
M5 doubles that limit weight to 0.02: its physical-bound gradients become
0.29588/0.48934 on the 43/26 Nm classes, while zero-torque curvatures are
0.00002994/0.00008190. At the representative maximum initial gravity demand,
its J2/J4 limit gradients are 0.001135/0.000999. The torque change is a bounded
conservative retuning choice, not evidence of a torque failure in M4.

## Joint position: margin and weight have different effects

Across all seven raw URDF joint intervals, these are the ranges at the physical
upper bound. The final column is the exact added cost for a further outward
0.01 rad, not just gradient times increment. Lower-bound magnitudes mirror
these values for the corresponding outward direction.

| Cost inset rad | Joint-limit weight | Gradient range | Curvature range | Extra cost for +0.01 rad |
| ---: | ---: | ---: | ---: | ---: |
| 0.02 | 0.01 | 0.457–0.722 | 3.704–7.990 | 0.00476–0.00762 |
| 0.02 | 0.1 | 4.571–7.223 | 37.04–79.90 | 0.04756–0.07622 |
| 0.04 | 0.01 | 0.536–0.896 | 3.762–8.174 | 0.00555–0.00937 |
| 0.04 | 0.1 | 5.364–8.959 | 37.62–81.74 | 0.05552–0.09367 |
| 0.10 | 0.01 | 0.789–1.466 | 3.943–8.765 | 0.00809–0.01509 |
| 0.10 | 0.1 | 7.895–14.656 | 39.43–87.65 | 0.08092–0.15094 |

The tenfold weight change is stronger than doubling the position inset at this
particular point. A larger inset starts discouraging proximity earlier; within
the quadratic continuation, curvature is finite and largely set by the interval
width. Increasing the inset leaves every raw interval nonempty and discourages
proximity earlier. In general an inset can place a start outside cost bounds,
where its fixed initial-state penalty cannot be removed by changing x0. That
does not occur for this frozen pool at the selected insets: the sampling-box
bounds were already inset by 0.12 rad from the actual URDF bounds. All 100 starts
and goal witnesses remain inside even the 0.10 rad URDF cost inset. Minimum
actual physical clearances are 0.120730 rad for starts and 0.120118 rad for goal
witnesses; the twelve development tasks have minima 0.131623 and 0.150541 rad.
Do not confuse small distance to the sampling-box boundary with small distance
to the physical boundary. Keep starts and actual physical outcomes unchanged;
the new cost interval is not a new robot limit.

The wider M4 candidate uses a 0.10 rad position inset, velocity fraction 0.80
and effort fraction 0.85, selected after the preceding recorded failures.
The earlier 0.04 rad candidate is an unused preparation artifact, not a measured
cohort. At the physical speed limit 2.5 rad/s, position insets of 0.02 and 0.04
rad correspond to only 8 and 16 ms of travel. A full 30 ms control interval spans
0.075 rad at that speed; 0.10 rad is a rounded scale allowing more than one such
interval. Every joint cost interval remains nonempty: the smallest width after
the two 0.10 rad insets is 3.377925 rad.

The independent CPU braking estimates at 2.25 rad/s gave median distances of
about 0.062 rad for J1 and 0.095 rad for J2, but a maximum of about 0.278 rad in
the sampled posture/joint calculation. Those approximate scales support trying
more than 0.02 rad of room; 0.10 rad does not cover the largest estimate or
provide a coupled-dynamics safety guarantee. Solving advances no plant time in
mode A, so this motivation concerns fixed feedback-prefix travel and braking,
not an added claim about wall-time latency. Starts, raw limits and physical
validation remain unchanged.

For scale only, reducing a Cartesian position error from 5 cm to 4 cm lowers
running position cost by 0.0018 per knot at weight 4, and terminal cost by
0.036 at weight 80. If that reduction happened at all 31 running knots and the
terminal knot, its combined reduction would be 0.0918. This is an objective
accounting example, not a feasible alternative trajectory or a direct conversion
from a joint displacement to Cartesian progress.

### Canonical posture accounting

The canonical preflight is distinct from the random pool. Its J4 start has only
0.093461 rad clearance to the raw lower bound, putting that coordinate 0.006539
rad outside a 0.10 rad cost inset. A CPU finite difference of the same torso-frame
Cartesian/RPY pose cost at this start gives a J4 running pose gradient of
+0.014199 cost/rad; the terminal pose gradient is +0.283982. The q-bound
gradient is negative, so its descent direction opposes the local pose-cost
descent direction in that coordinate.

| Cost inset / q weight | J4 limit gradient | Limit-gradient norm / running pose-gradient norm |
| --- | ---: | ---: |
| 0.02 / 0.01, M0 | −0.133240 | 1.26 |
| 0.02 / 0.1, M2/M3 | −1.332401 | 12.63 |
| 0.10 / 0.01, M4 | −0.646432 | 6.07 |
| 0.10 / 0.015, M5 | −0.969648 | 9.10 |
| 0.10 / 0.02, considered only | −1.292864 | 12.13 |

Thus q weight 0.02 with the wider inset would nearly recreate the strong
M2/M3 J4 centering gradient. M4 retains weight 0.01: widening the inset already
raises the physical-bound gradient approximately 1.7–2 times M0 across joints.
M5's 50% weight increase gives physical-bound gradients 1.184–2.198 across
the seven joints, versus M4's 0.789–1.466, without using the earlier tenfold
coefficient increase.
The complete seven-component pose and limit gradients, central-difference step
1e-6 rad, fixture/URDF provenance and local norm ratios are in the optional
canonical accounting section of `cost_scaling.json`.

These gradients do not prove why a preflight timed out. They describe local
objective terms at one posture, without dynamics or a feasible motion direction.
For example, along the coupled FK-witness displacement both local costs can
decrease even though their J4 gradients oppose one another. A pose objective
also does not prescribe that witness as a joint target. The calculation motivates
bounded retuning and preserves the original failure rather than diagnosing a
new solver defect.

### Joint-only follow-up after the M5 checkpoint

M6 first changes the position cost inset to 0.15 rad and joint-limit weight
to 0.01. All other M5 parameters stay fixed, including speed weight 0.6,
effort weight 0.02, qd=0.05, `u_cost=0.0001`, mu=10, velocity/effort fractions
0.80/0.85 and the fixed mode-A clock. The raw physical q limits do not move.
A 0.15 rad buffer is above the sampled J1 frozen-posture maximum braking
distance of about 0.132 rad at speed 2.0 rad/s and effort fraction 0.85.
That supports testing earlier deterrence, not a coupled braking guarantee.

The table uses the same C2 helper and unchanged development inputs. Gradients
are objective cost/rad, not torque. Delta is 1% of the resulting interval and
therefore differs by joint and inset. J1 is evaluated at its physical upper
bound; the other columns use the actual canonical and task-2342 starting q4.

| Pair: inset / weight | C2 delta J1 / J4 rad | J1 physical gradient | Canonical J4 gradient | Task2342 start J4 gradient |
| --- | ---: | ---: | ---: | ---: |
| M5: 0.10 / 0.015 | 0.050360 / 0.033779 | 1.18425 | −0.96965 | +0.46793 |
| M6: 0.15 / 0.01 | 0.049360 / 0.032779 | 1.01888 | −1.13334 | +0.77814 |
| M7: 0.20 / 0.005 | 0.048360 / 0.031779 | 0.63338 | −0.84061 | +0.65166 |
| M8: 0.20 / 0.01 | 0.048360 / 0.031779 | 1.26676 | −1.68122 | +1.30332 |
| Alternative: 0.15 / 0.005 | 0.049360 / 0.032779 | 0.50944 | −0.56667 | +0.38907 |

M6's J1 slope at the physical boundary is 86% of M5's, while at five degrees
inside that boundary it is nearly unchanged (0.66067 versus 0.66805).
At a quarter-span interior point, absolute gradient ranges across joints drop
from 0.00849–0.01307 for M5 to 0.00598–0.00948 for M6. Thus a wider inset
and lower weight reshape where the penalty acts rather than uniformly making
it stronger. Earlier deterrence and smaller ordinary interior bias trade
against a weaker late boundary slope.

Near cost edges, a lower coefficient need not reduce bias. M6 increases the
canonical J4 gradient magnitude by 17% and task2342's start J4 magnitude by
66%. Their running pose-gradient components are +0.014199 and −0.374918,
respectively; each opposes its corresponding limit gradient locally. The
0.20/0.005 pair lowers canonical magnitude by 13% relative to M5, but still
raises task2342's component by 39%. The 0.15/0.005 alternative lowers both
by about 42% and 17%. These local comparisons include neither dynamics nor a
feasible direction and do not predict the closed-loop result.

M6's canonical preflight passed at 0.90 s within physical limits. Its completed
twelve-task cohort again had three rough arrivals, eleven physical passes,
three jointly valid arrivals and no execution errors. Only task2342 failed
physical checks, with maximum q excess 0.0479143 rad; all speed/effort checks
passed. This did not remedy M5's joint-limit failure.

M7 selected the declared 0.20/0.005 physical-failure branch. Its complete cohort
also had 3/11/3 rough/physical/both passes and no execution errors, with only
task2342 failing: maximum q excess increased to 0.0648886 rad. Speed and effort
checks still passed. The alternative 0.15/0.005 remains unexecuted preparation.

Under Felix's continuing explicit priority to find a parameterization before
validation, M8 then restores weight 0.01 at the same 0.20 inset and cached
module as M7. M7/M8 changes only weight, so every scalar limit value and
derivative doubles. M6/M8 holds weight 0.01 fixed and changes only the joint
inset from 0.15 to 0.20. At the physical J1 bound this raises the gradient
24.3%, versus increases of 48.3% at canonical J4 and 67.5% at task2342's
starting J4. M8's J1 slopes one/five degrees inside the physical bound are
1.19213/0.89358; its quarter-span gradient range is 0.00633–0.01037.

This isolated comparison supersedes the earlier recommended one-branch stopping
point; it does not launch another grid. M8's canonical preflight passed at
0.96 s within physical limits, despite its larger local canonical gradient.
Its twelve-task outcomes are separate and not inferred from that accounting.
The M5 checkpoint remains intact; no 100-task validation or horizon runs have
launched. Joint-specific buffers remain a discussion only if this uniform
comparison fails; none is implemented here.

A focused static setup review confirmed that `q_lim_cost` is passed intact:
`run.py:236,258` supplies the full protocol dictionary; the TIAGo controller
forwards it explicitly at `tiago_mpc_controller.py:114`; `interface.py:81` and
`bindings.cu:40–44` retain the native argument order; `bsqp.cuh:48` stores it
and supplies it to merit/KKT setup. Plant value, gradient and Hessian use it
at `tiago_right_plant.cuh:411,535,585`. Final approach is off, this wrapper
does not call `set_cost_weights`, and dual/rho resets do not reset cost weights.
No static omission or default override was found. This source check adds no
GPU experiment, instrumentation or causal solver investigation.

Starts and FK goal witnesses are not clipped or filtered. Among the twelve
development starts/witnesses, inset 0.15 puts one start and no witnesses outside
the cost interval; inset 0.20 puts two starts and three witnesses outside.
Those are soft cost intervals, not changed physical definitions, and a goal
witness is not a required joint endpoint. The already distinct canonical J4
also lies outside these larger cost intervals.

Full scalar values, curvatures, quarter-span points, J1 checks one/five degrees
inside its bound and actual task2342 start provenance are in the independent
`joint_margin_summary.csv`, `joint_margin_scalar_points.csv` and
`joint_margin_tradeoffs.json` under
`/workspace/GATO/_expr_results/random-reaching-joint-margin-tradeoffs-20261015/`.
The source is [joint_margin_tradeoffs.py](joint_margin_tradeoffs.py). It uses
the actual task2342 start pool entry 897; goal entry 75 remains only its FK
pose witness. No new task, rollout, GPU measurement or solver instrumentation
was introduced for that accounting.

## Selected M14: one additional accounting point

M14 holds the 0.20 rad joint cost inset, ±2.0 rad/s speed cost bounds and
0.85 effort fraction. Its q/speed weights 0.02125/1.275 are a common factor
2.125 times M8's 0.01/0.6. Effort weight 0.02, qd=0.05, `u_cost=0.0001`,
mu=10 and all pose costs remain fixed. This is one selected configuration,
not a new parameter grid. The exact protocol SHA256 is
`786f98c8c4762af29342befe7ed169fdce8e9389a4799d61820849720f4ab7fb`;
source remains `85ef2ea5cf5b96726be6f4b5441f4a7c0ca041b4` and the immutable
q020 module SHA256 is
`d9bee23f90a1d7816b32db23749bd651db2a7537f94c3dfec15f1a448bb34cc8`.

| Setup | J1 physical-upper q gradient | Canonical J4 q gradient | Task2342 initial J4 q gradient | Total velocity curvature at zero | Total velocity gradient at 2.5 rad/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| M14, selected development setup | +2.691874 | −3.572594 | +2.769547 | 0.687500 | 462.029167 |

The first three columns are cost/rad; the last two are respectively
cost/(rad/s)² and cost/(rad/s). They cannot be compared directly across units.
The physical-speed gradient comprises 461.904167 from the limit cost plus
0.125 from the ordinary velocity quadratic. The near-zero curvature is
`0.05+2*1.275/2²=0.6875`, above M5/M8's 0.35. All q-limit values and derivatives
at these same bounds are exactly 2.125 times M8's. The canonical and task2342
J4 limit gradients still oppose their local running-pose gradient components
(+0.014199 / −0.374918 cost/rad); this local accounting neither predicts a
physical failure nor explains a closed-loop trajectory causally.

For J1 the continuation starts 0.20+0.048359878=0.248359878 rad inside the
physical bound. This earlier smooth transition is larger than the sampled
J1 frozen-posture braking maximum of about 0.132 rad at 2 rad/s plus one
30 ms executed prefix's travel of 0.060 rad, totaling about 0.192 rad. That
scale supports earlier deterrence. It is not a guaranteed stopping distance:
posture changes, coupled dynamics and other joints can require more room;
the sampled J2 estimate already reaches about 0.261 rad before prefix travel.
Mode A charges no plant time for solver computation, so this comparison adds
feedback-prefix travel rather than measured solve latency.

The fixed task2342 screen gave M14 minimum J1 physical clearance 0.0273191 rad,
versus M15's 0.00496562 rad. Both screens passed all sampled physical checks and
timed out. M14's rough canonical passed at 1.77 s; M15's at 1.35 s. The selected
M14 full pilot tests whether the more buffered screen transfers to all twelve
unchanged development tasks. It completed with twelve physical passes, one rough
arrival (3626 at 6.84 s), eleven timeouts and no execution errors. The sampled
minimum q clearance was 0.0273528 rad, maximum velocity ratio 0.660626 and
maximum effort ratio 0.826864. M14 was frozen at 11:43:04 UTC before separate
100-task validation started. No validation outcome informed this selection;
the low development arrival count remains an explicit limitation. The existing M0/M1/M4/M5 velocity figure is
retained as a checkpoint rather than crowded with another curve.

## Mu: equality-defect merit, not damping or a physical limit cost

Mu multiplies the L1 modeled dynamics defects and the initial-state defect in
line-search merit. The underlying modeled equalities remain part of the SQP
problem. Mu changes how the line search trades objective reduction against
nonlinear equality defects; it does not change the physical bounds, automatically
adapt, or certify exact equality. Rho instead regularizes the solve and adapts
with line-search acceptance. It is a separate parameter.

| One defect component | Mu=1 merit contribution | Mu=10 | Mu=100 |
| --- | ---: | ---: | ---: |
| q defect 0.001 rad | 0.001 | 0.01 | 0.1 |
| v defect 0.01 rad/s | 0.01 | 0.1 | 1 |
| v defect 0.1 rad/s | 0.1 | 1 | 10 |

These unnormalized units are mixed and summed across joints and knots. For an
accounting illustration, reducing one predicted speed from 2 to 1.99 rad/s saves
0.036865 of strong current-margin limit cost. A hypothetical associated single
0.01 rad/s equality defect contributes 0.01/0.1/1 at mu 1/10/100. At speed
2.5→2.49 the limit-cost saving is instead 1.65221. Those examples do not assert
that the actual SQP directions introduce that defect, nor prove a sufficient mu.
They explain why a sensible penalty magnitude depends on where the trajectory
lies and why increasing physical costs can change the useful merit tradeoff.

The M2/M3 comparison changed mu from 1 to 10 at fixed weights and bounds; both
canonical preflights remained physical but timed out at nearly the same position
error. It established no reaching benefit from mu=10 in that setup. M4 retains
mu=10 as an explicit engineering choice while changing other parameters; its
canonical pass cannot be attributed to mu alone. There is no current reason to
queue mu=100 merely to seek smaller descriptive errors. A later bounded point
would require a concrete task-relevant question from the changed setup.
Compare saved first
future knots with executed 30 ms endpoints if useful, distinguishing native
frozen-acceleration discretization from 3 ms RK4 as described in
[the independent review](MARGIN_REVIEW.md). This adds no KKT gate, precision
requirement, solver repair or new native residual instrumentation.

Two saved-prefix CPU comparisons illustrate that distinction. Returned future
knots differed from the CPU frozen-acceleration prediction by approximately
0.00014–0.00029 rad in q and 0.009–0.012 rad/s in v, whereas frozen-acceleration
versus RK4 endpoints differed by about 0.022–0.027 rad and 2.16–2.88 rad/s.
At those two selected prefixes, the much larger gap is associated with the
integration approximation, including the CPU/native model distinction, rather
than a large returned-knot defect against the frozen-acceleration model. This
bounded observation does not establish a cause for every task failure. The
mu=10 comparison can still answer a practical task question, but should not be
expected to remove that integration-scale gap by itself. No additional solver
investigation or mismatch acceptance threshold follows.

## Reproduction and provenance

```sh
python experiments/random_reaching_baseline/cost_scaling.py
```

[cost_scaling.csv](cost_scaling.csv) contains 409 scalar rows covering both speed
margins, speed weights 0.1/0.2/1, qd weights, all seven joint intervals and all three joint margins/weights,
all seven torque classes with rounded gravity points, both effort margins and
effort weights 0.003/0.01/0.03. It retains signed derivatives, units, ordinary quadratics and finite
increments (0.01 rad/s, 0.01 rad or 1 Nm). Joint rows leave ordinary joint-angle
quadratics blank. Some finite increments deliberately go outside physical bounds
to describe the soft continuation; they are not motion proposals.
Only the 46 exact M5 additions were appended: four velocity points at weight
0.6/qd=0.05/inner bound 2.0, fourteen physical joint-bound points at q weight
0.015/inset 0.10, and twenty-eight effort points at weight 0.02/fraction 0.85.
They do not extend every hypothetical weight/margin pairing.
The later 56 joint-only additions cover exactly 0.15/0.01, 0.20/0.005,
0.15/0.005 and M8's 0.20/0.01 at both physical bounds of all seven joints.
The 0.15/0.005 alternative remains unexecuted preparation. M6/M7/M8 have the
same velocity cost as M5, so no duplicate velocity curve is added.
Seven exact M14 rows add four representative velocities and only J1 physical
upper / canonical J4 / actual task2342-start J4 points. The previous 402 rows
remain numerically unchanged; no other weight/margin combinations are added.

[cost_scaling.json](cost_scaling.json) records source/helper hashes, conventions,
gravity inputs and merit examples. Its `selected_M14_accounting` object adds
the one-row summary above, exact protocol/module identities, canonical fixture
hash and task2342 actual-start pool index 897 plus frozen input hashes. Its
accounting metadata retains the original pending-pilot snapshot; this note records
the subsequent completed development result without rewriting that snapshot.
The script checks the exact reviewed helper
body hash before calculating, so a different formula requires explicit review.
The actual helper's CPU consistency checks already passed in the margin source
review; these tables introduce no new native qualification. Full physical-scale
and braking analysis remains a separate CPU calculation, not inferred from these
scalar cost derivatives.

Regenerate the additional canonical FK/RPY accounting with CPU Pinocchio:

```sh
python experiments/random_reaching_baseline/cost_scaling.py \
  --canonical-preflight /workspace/GATO-experiment-runner-simple/experiments/inputs/batching/ba_preflight.json
```

Without that optional argument, scalar cost regeneration uses only the Python
standard library. Neither mode imports the native solver, constructs a controller
or performs a GPU call.

The single velocity-cost figure has a compact 1,084-row CSV and a portable
plot-only command:

```sh
python experiments/random_reaching_baseline/velocity_cost_curve.py
python velocity_cost_curve.py --plot-only --output .
```

Use the first command in this checkout to regenerate the data and PNG/PDF.
Use the second beside the exported `velocity_cost_curve.csv` for laptop plot
adjustments. The plot-only path needs matplotlib and the CSV; it imports no
native solver or cost-generation module. `velocity_cost_curve.json` retains
the exact curves, grid, formula identity and file hashes.

The separate [physical-scale calculator](physical_scales.py) retains full M/g,
headroom, braking assumptions and selected-prefix provenance in its generated
`physical_scales.json`, with `gravity_headroom.csv`, `acceleration_capacities.csv`
and `PHYSICAL_SCALES.md`. Those records supply the detailed physical evidence;
the rounded gravity points here are only inputs for readable cost comparisons.
Their retained raw destination is
`/workspace/GATO/_expr_results/random-reaching-margins-physics-20261015/`.
