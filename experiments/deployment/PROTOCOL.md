# D01 proposed live protocol

Target: RTX 5060/Gazebo, separate from RTX 5060/internal T01. One preflight
followed by one pass of each task, with no extra repeats or calibration.

| Case | Task | Gate |
| --- | --- | --- |
| `preflight-goal-1` | canonical rough goal-1 reach | 5 cm, tool-axis 30°, speed L1 ≤ 1 rad/s, zero dwell |
| `goal-1` | frozen 100.636 mm FK-derived target | same rough-reaching gate, 8 s allowance |
| `goal-2` | frozen 207.933 mm FK-derived target | same rough-reaching gate, 8 s allowance |
| `goal-3` | frozen 317.052 mm FK-derived target | same rough-reaching gate, 8 s allowance |
| `figure8-24s` | full frozen 24 s reference/period | RMS/max 20/40 mm |
| `figure8-18s` | full frozen 18 s reference/period | RMS/max 25/50 mm |
| `figure8-12s` | full frozen 12 s reference/period | RMS/max 30/60 mm |

Goals, full reference bytes, presets and original witnesses are in the copied
input. These model-start FK offsets do not establish actual live reachability
or collision clearance. Accepted tracking gates derive from Felix's
2026-10-10 simulation review. Tracking has no orientation gate.

All cases use N32, 0.03 s knot/nominal feedback interval, 0.93 s prediction,
B1, zero external force, gravity (0,0,-9.81) m/s², no pendulum and final
approach off. Initial q in radians is
(-0.3899999857,-1.7300000191,-0.3799999952,-2.3499999046,0,-1.2100000381,0.0399999991);
requested initial qd is zero. Live state replaces the requested optimizer start
after public initialization. Public cold constant-current-state/zero-control
seed and shifted previous-plan warm starts stay unchanged. The first solve is
retained; there is no extra warmup solve. Reaching retains adaptive rho between
solves; public live tracking resets rho on feedback solves.

The public task call receives `sim_dt=0.003 s`. In live mode the public loops
observe ROS state and do not run that RK4 integration. Gazebo physics timestep
is configured externally and is not measured by `/clock` subscription spacing.
This worker does not reconfigure Gazebo or claim to establish a 3 ms Gazebo
integrator. It records this limitation alongside requested setup.

| Solver parameter | Reaching | Tracking |
| --- | --- | --- |
| SQP cap / KKT tol | 5 / .001 | 5 / .001 |
| PCG cap / tol | 120 / .001 | 120 / .001 |
| solve ratio / mu / rho | 1 / 1 / .01 | 1 / 1 / .01 |
| q / qd / u cost | 4 / .1 / .0001 | 4 / .01 / .0001 |
| terminal cost | 80 | 80 |
| orientation / terminal orientation cost | .1 / 2 | 0 / 0 |
| q / velocity / control limit cost | .01 / .1 / .05 | .01 / .001 / .003 |

These are frozen campaign public-preset settings and existing pose overrides,
not raw low-level constructor defaults. Limit costs are soft penalties; modeled
limits are independently checked on observations and published commands.

Current live runtime defaults are retained separately: reaching command target
100 Hz, tracking 125 Hz; reset duration 2 s; stale timeout .1 s; software torque
cap 26 Nm; no clamping; collision safety enabled, minimum distance .04 m,
check timeout .05 s, monitored geometry speed 1 m/s, joint position margin
0 rad, velocity scale 1. The existing comfortable-pose collision blacklist is
used. State history capacity 20,000; multiprocessing spawn; controller read/
finish timeout 8 s; startup timeout 30 s. Full resolved settings appear in
`protocol.json`; safety source/blacklist hashes appear in provenance.

Timing is `live`, `computation_paid=true`. Actual observed ROS state-stamp
elapsed time drives the public loops while the unchanged controller process
executes its held/interpolated plan. Existing latest-plan queue and horizon
expiry/stale-state/safety policy are retained. Wrapper wall measures only
public `BSQP.solve`; returned native time is separately recorded in µs. Input,
warm/output/stat copies and solve-record IO are outside that timer; their
measured overhead is reported separately. All copying, IO, observation and
bookkeeping consume live elapsed time. The experiment adds observation overhead
and does not claim equivalence to an unrecorded live run. Update intervals and
their above-nominal share are descriptive; there is no deadline acceptance gate.

Every public solve saves exact input state, reference, warm seed before any
public-interface mutation, returned packed output, existing host stats, native
time and wrapper wall. Failures retain input and exception. No extra get_stats,
device synchronization or device transfer is added. The task's existing
get_stats calls are forwarded unchanged. Raw arrays, including nonfinite values,
are preserved in NPZ; JSON metadata gives their keys and hashes. `events.jsonl`
also retains feedback reads, planned full-horizon publication requests and
finish/close/handback events. The original result object returns unchanged.

The independent subscriber saves `/clock`, `/joint_states`, and
`/gato_arm_effort_forward_runtime/commands` with original ROS stamps where
available and monotonic receive times. Effort commands have no source header:
the receive time/latest observed clock are labelled observations. They are
controller-published effort, not measured actuator torque. Public
`applied_controls` is only controller-side attribution. Full joint/mode/safety
histories from the orchestrator are retained in finally, including on errors;
its `applied_tau` likewise means the last published effort at a state callback.
Observer shutdown and confirmed public controller restoration are recorded;
an unconfirmed handback is an execution failure.
Recorder failures are execution errors. Cleanup records are best-effort:
failed writes cannot prevent controller close, either history attempt or
observer shutdown. Confirmed handback and evaluated task gates retain their
own truth when subsequent recording fails. Missing task evidence is reported
as unmeasured, and any execution error still holds the canonical preflight.

Reaching records final position, tool-axis, full SO(3) and speed errors and
checks the declared gates plus public goal outcome. Tracking reconstructs FK
on all observed task states between initial and terminal public feedback,
including both boundary feedback states. Position RMS is time-weighted
trapezoidal squared error against the frozen sampled reference; no startup
samples are excluded. Live end overshoot and repeated stamps are retained;
repeated stamps contribute zero integration weight. Full-period completion,
RMS and maximum gates determine tracking task success. Raw observations are
never removed to make metrics finite.

Sampled physical compliance checks every complete observed right-arm q/qd and
every observed published effort command for the whole session, including
reset/handback, against model limits with tolerances 1e-4 rad, 1e-3 rad/s and
1e-3 Nm. Missing/incomplete observations yield unmeasured compliance and hold
preflight; nonfinite values fail the sampled gate. `physical_pass` denotes this
observable sampled scope. Measured actuator torque and full physical compliance
remain unavailable. Sampling does not prove between-sample bounds or physical
safety. Runtime collision safety is separate from these model checks.

SB-001 retained P01 goal1 failure and SB-002 static finiteness failure remain
limitations. Successful task results do not certify full solver convergence,
all configurations, or per-fix benefit. Actual execution and release remain
separate from this implementation proposal.
