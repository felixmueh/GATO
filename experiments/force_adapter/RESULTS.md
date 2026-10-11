# Bounded TIAGo force runtime qualification — 2026-10-11

The production float TIAGo body-fixed wrench path passes the bounded native
Pinocchio checks and the unchanged three-case task regression. Felix accepted
this as sufficient on 2026-10-11. The extra double accuracy check failed its
stronger threshold and remains recorded; no further precision investigation,
changed tolerance, optimizer repair or force-performance campaign is authorized
by this qualification.

The supported physical input is a wrench fixed in `arm_right_7_joint` coordinates
about that joint's origin. Python uses `[Fx,Fy,Fz,Tx,Ty,Tz]`; GRiD receives
`[Tx,Ty,Tz,Fx,Fy,Fz]`. This is an attached/body-fixed disturbance. It does not
qualify constant world-force prediction or physical robot use. The controller's
TIAGo constant-world-wrench guard remains explicit.

The generated TIAGo header is unchanged. Two copied RNEA overload bodies apply
the local wrench before their original backward force propagation. The existing
analytic inverse-dynamics derivative consumes the force-aware v/a/f buffer;
the wrapper retains the original joint-coordinate sign mapping. Generic and
TIAGo controller transforms now rotate world-expressed components, shift the
tool-origin moment to the parent-joint origin and omit the extra inverse joint
transform. Generic Pinocchio simulation re-expresses a world/tool-point wrench at
each RK stage. A solver's initial-state world-to-local conversion still represents
a frozen local wrench, so it cannot be described as full world-force prediction.

The accepted `502fa29` controller timing implementation and `offline_mpc.py`
were transplanted without live-controller changes. CPU reaching/tracking timing
checks and force/orchestration checks pass: **91 tests**. Independent wrench
oracles cover TIAGo, Indy7 and IIWA14 using Jacobian-transpose and RNEA forces.
The body-fixed Pinocchio ABA derivatives agree with central differences at the
selected pose (preliminary maximum errors q/v/u: 1.49e-7/2.17e-7/7.53e-8).
Zero-wrench RK integration matches the existing path exactly in the CPU check.

One isolated N32 Release build used CUDA architecture 61 and the selected
uncommitted source hashes. Module SHA256:
`8c113908459b92144c5a48cc24c1056ea8db0938e1326ec5bed6563c120f6e52`.
Six retained inputs cover two poses and zero force, mixed force/torque and pure
torque. Native forward acceleration and analytic derivatives were compared
against Pinocchio using the actual float-quantized or double inputs.

| Maximum absolute error | Production float | Additional double |
| --- | ---: | ---: |
| Forward acceleration (rad/s²) | 2.2797e-4 | 1.9999e-4 |
| Gradient-path acceleration (rad/s²) | 2.2797e-4 | 1.9999e-4 |
| Acceleration derivative w.r.t. q | 6.5397e-4 | 2.9841e-4 |
| Acceleration derivative w.r.t. qd | 4.3504e-6 | 3.5250e-6 |
| Acceleration derivative w.r.t. torque | 3.4531e-4 | 3.4028e-4 |
| Comparisons passing original threshold | 30/30 | 0/30 |

Float used atol=2e-3, rtol=2e-4. The additional double check used atol=1e-6,
rtol=1e-7. All outputs were finite. Double discrepancies occur even at zero
wrench and in force-independent control derivatives; this is a clue, not a
verified cause. Its stronger accuracy result does not overturn the accepted
production float evidence. The proposed original-baseline precision check was
not built or executed.

The original frozen short reach, long reach and figure-eight task suite passed
**3/3**, with task and physical gates passing, zero skips, failures or errors.
Two earlier launches failed in shared package-bootstrap setup before any task:
first an absent namespace-package `__file__`, then duplicate identical namespace
paths. Both attempts remain intact. The corrected launch executed each actual
task once. No fixtures, module or task parameters changed to obtain these passes.

All build logs, input/native/Pinocchio arrays, source hashes, module receipt and
task artifacts are retained beneath
`/workspace/GATO-eval-force/.force-preparation/qualification-20261011/`.
`qualification.json`, `native-oracle/receipt.json`, and
`task-suite-setup3/metrics.json` identify the evidence. The full numerical solver
certificate remains unavailable, and historical solver failures remain evidence.

The new predictor replays every actually applied control segment. Native TO uses
a fresh complete owner per tick; constructor/model/allocator time and native
solve time are separate. CUDA free/total boundary snapshots and host RSS will be
recorded separately, with no claimed continuous peak or summed Jetson GPU/CPU
memory. No original/adapted runtime, memory saving or force-disturbance task result
has been collected yet. New study protocols and machine plans remain preparation.
