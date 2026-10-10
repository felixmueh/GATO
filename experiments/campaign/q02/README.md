# Q02 current normal-plant qualification

This is a new bounded **current-source** protocol, not recovered historical debug
code. Approved production core is `16b3371`; no production headers, solver,
native binding, generated model, or root build wiring are changed. Q02 remains
partial until the standalone native probe has been reviewed, compiled and run.
The prior Pinocchio reduction/gripper checks remain separate evidence; this
probe does not rerun or replace those studies.

`protocol.json` freezes eight native-coordinate interior states, goals, weights,
raw URDF limits, gravity, timing/grid conventions, finite-difference steps,
practical tolerances and a **351-call** one-pass budget. Input samples 1/2 perturb
native joint 6 to -0.35/+0.35 rad. Inputs are rounded to float32 before the
independent float64 references. The model is the current lumped 7-DoF URDF in
`torso_lift_link`; the tool is `arm_right_tool_link`. Pose uses XYZ and roll/pitch/
yaw with the same declared conventional ZYX extraction, compared with wrapped
angular differences. No damping/friction or external wrench is added: GRiD and
Pinocchio ABA here model rigid-body dynamics without those URDF annotations.

The static probe uses actual unchanged handwritten functions:

- `forwardDynamics`, `forwardDynamicsAndGradient`, `integrator_inner<2,false>`,
  `integrator_gradient_inner<2>`: acceleration, its 7×21 derivative, one 30 ms
  trapezoidal step, A and B. Direct dynamics and gradient-path accelerations
  are retained separately. Reference values/derivatives use independent
  Pinocchio ABA and float64 central differences.
- `computeTiagoToolPoseAndGradient`: native-coordinate pose and 6×7 derivative;
  the existing sign map is applied at the wrapper boundary. Independent
  Pinocchio FK supplies pose and finite-difference pose derivatives.
- `trackingcost`, `trackingCostGradientAndHessian`, and the actual
  `trackingCostGradientAndHessian_lastblock`: running scalar cost/gradient/Q/R
  plus terminal scalar cost/gradient/Q. The grid keeps **blockIdx.x = 0/30/31**
  for N32; B1/B4 lane indexing uses blockIdx.y. Lastblock receives identical
  state/goal in its running and next-state slots and outputs the next-state
  terminal blocks. This checks actual terminal promotion in that helper.

B4 groups include distinct states, duplicate lanes and two permutations.
Every declared output is compared to B1 at practical tolerances; no bitwise
comparison or repeat-variation study occurs. Each sample then gets exactly
42 centered native calls (21 inputs × plus/minus). Returned acceleration,
step, pose and costs supply independent numerical derivatives of the wrapper
values. Terminal control derivative is also checked against zero.

Q/R comparisons use **JᵀWJ plus regularizer/barrier curvature**, constructed
from independent finite-difference pose Jacobians. The pose cost's exact
nonlinear Hessian is deliberately not the expected Gauss–Newton block.
Both the Pinocchio-FD and native-value-FD GN arrays are preserved. Weights are
explicit probe weights, not a claim to qualify all public task defaults.

No optimizer, SQP solve, KKT buffers, outside-bound derivatives, NaN cause
inspection, determinism, controller task, learning, Gazebo or robot execution
is part of this protocol. Eight-sample/B4 agreement is not a full model,
solver convergence, task success or reliability certificate.

## CPU preparation and static review

CPU tests (no CUDA import or build):

```sh
/opt/gato-venv/bin/python3 -m pytest tests/python/test_q02_protocol.py -q
```

The frozen protocol must not be regenerated after GPU results. `freeze.py`
refuses overwriting it. Before first native compilation, any probe/runner
review corrections can refresh **only** the source hash manifest:

```sh
/opt/gato-venv/bin/python3 experiments/campaign/q02/freeze.py --refresh-manifest-only
/opt/gato-venv/bin/python3 experiments/campaign/q02/run.py \
  --output /workspace/gato-q02-cpu-reference-UNIQUE_TIMESTAMP
```

`run.py` verifies all frozen source hashes and saves the full model bytes,
float32 inputs/goals, float64 reference arrays, setup margins, finite-difference
reference timing, package versions and an all-file raw manifest. A failed CPU
setup holds native execution; preserve it and report before changing inputs,
tolerances, geometry or conventions.

## Root-only native execution after independent static review

Use a fresh dedicated build directory; never configure the root project or
install into `python/bsqp`. Root chooses and records the reviewed compiler/toolkit
paths and explicit architecture for the machine. Example GTX1070 commands:

```sh
cmake -S experiments/campaign/q02 -B /workspace/gato-q02-native-UNIQUE_TIMESTAMP \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DPython3_EXECUTABLE=/opt/gato-venv/bin/python3 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /workspace/gato-q02-native-UNIQUE_TIMESTAMP --parallel 1
/opt/gato-venv/bin/python3 experiments/campaign/q02/run.py \
  --module /workspace/gato-q02-native-UNIQUE_TIMESTAMP/modules/q02_probe.EXACT_SUFFIX.so \
  --output /workspace/gato-q02-native-raw-UNIQUE_TIMESTAMP
```

Keep configure/build/run stdout/stderr and compiler versions beside the raw
receipt. This module uses Release `-O3 -use_fast_math`, float32, N32 and normal
`PLANT_TIAGO_RIGHT`; neither multimodal nor double precision is enabled. Review
cache and compile commands before execution. CMake requires an explicit
architecture and Release. It writes modules only below the isolated build root.
The runner records module hash, cache and compile commands when present.

First-run estimates, **not measurements**: approximately 1–5 minutes for the
single CUDA translation unit and 50–500 MB build storage, then 1–2 minutes for
351 short accessor calls and less than 10 MB raw output on GTX1070. There are no
SQP solves. Root schedules all GPU access sequentially; no Jetson run is implied.
A substantially harder build or unexpected result is an escalation/stopping
point, not permission to add instrumentation or broaden the protocol.

Every native invocation, including one explicitly tagged warmup, retains input,
goal, full declared flat output, CUDA-event kernel time and wall time in its
own NPZ. Each call archives exact float32 input/goal/index/category before
invocation and flushes a started JSONL event. Exceptions retain a failed event,
error and measured wrapper elapsed, with null plus reasons for unavailable
outputs/native clocks. Import/constructor failures retain attempted setup and
the full frozen input cohort. Nonfinite/unavailable timing scalars use valid
JSON null plus reasons while original returned scalar bytes remain in NPZ.
Independent mutable `review.json` is excluded from source-manifest refresh.
Wall time includes call allocation and copies; initialization is excluded. All scalar/B4/FD arrays and checks remain
available. Warmup is excluded from scientific comparisons but never discarded.
Completed discrepancy/nonfinite checks are negative outcomes: execute the fixed
budget, preserve all results, hold qualification and report. Execution errors
stop and preserve partial outputs; no retry-until-green, sample filtering,
solver fix or threshold refitting is allowed. No native qualification is
claimed by CPU setup success or compilation alone.
