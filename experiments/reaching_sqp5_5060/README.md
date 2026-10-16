# SQP5 reaching benchmark on RTX5060 — 16 October 2026

This new cohort runs exactly 800 scientific episodes and four diagnostic
canonicals. It does not resume or modify historical 8 s or conditional 30 s
studies. No local GTX1070 GPU execution is part of preparation.

Each family has the same 100 inputs across these four B1 variants:

| Arm | Joint-position penalty | Running position / terminal weight |
| --- | --- | --- |
| NORMAL | Original C2 relaxed barrier, mode 0 | 4 / 80 |
| FLAT | Existing flat interior, mode 1 | 4 / 80 |
| TERMINAL10 | Original C2 relaxed barrier, mode 0 | 4 / 800 |
| TERMINAL10_FLAT | Existing flat interior, mode 1 | 4 / 800 |

The random family preserves the original frozen100 starts, FK goal poses and
order. The cone family has 100 new fixed Cartesian targets, each reached from
comfortable posture `[-.5,-1.4,-.3,-1.7,0,-.9,0] rad` and zero velocity. The
apex is the initial tool FK in `torso_lift_link`; forward is torso +X. Felix
confirmed 45 degrees **from the forward axis**, or a 90 degree full opening.
There is no chosen inner or outer radius. Valid targets initially inside the
50 mm arrival region are retained and flagged; zero-speed tasks already arrived
at the t=0 feedback boundary record arrival time 0 without a solve. This frozen
sample contains none.

All variants use N32, 30 ms knots and feedback, a .93 s prediction horizon,
maximum30 s, RK4 steps3 ms, mode A/unpaid compute, SQP allowance5 and PCG120.
Natural zero-PCG stopping remains enabled as a practical heuristic; no KKT
acceptance gate or stationarity certificate. Orientation and final approach are
off. Arrival means position error<=50 mm and joint-speedL1<=1 rad/s at a feedback
boundary, with zero dwell. Record sampled raw joint/speed/torque compliance and
first violations separately from arrival. There is no clipping or safety veto
in this internal simulation benchmark; it does not qualify physical motion or
collision clearance.

Other explicit weights are qd=.05, u=1e-5, joint=.02125, speed=1.275,
effort=.02, mu=10, rho=.01; orientation weights0/0. Cost bounds have .20 rad
joint inset, speed fraction .80 and effort fraction .85. The flat variant only
changes the joint-position term; speed and effort penalties stay the original
ones. Every task has a fresh solver/session, cold repeated actual state and
zero controls, one dual/rho reset, then ordinary retained state and warm shifts.
The IK witness is neither a solver seed nor a joint target.

The 30 ms feedback is common to all eight arms. The earlier selected flat100
used10 ms; this study does not silently inherit that setting or pool its results.
The changed SQP allowance and new cone population also define new evidence.

## Radius-free Cartesian sampling

The sampler derives a conservative ball from the actual URDF serial chain.
Its center is the first actuated joint origin in torso coordinates. Its radius
is the sum of all subsequent joint-origin translation norms, including the
fixed tool offset. Rotations preserve norms, so triangle inequality guarantees
the ball contains every FK endpoint. It is a containing proposal volume, not
a reach cutoff. All chain terms and the proof are saved with the targets.

Uniform Cartesian cube proposals are rejected outside this ball or the cone.
Each geometric candidate receives the historical position-only bounded IK
screen: comfortable start plus five independent random raw-joint-box seeds,
200 function evaluations per start, 1e-6 numerical tolerances, acceptance at
<=1 mm positional residual inside raw joint limits. Keep the first100
acceptances. No orientation, soft-margin, collision, gravity, dynamic,
distance/difficulty or controller-outcome selection is applied. Accepted goals
follow the volume proposal conditioned on bounded numerical IK acceptance;
this is not a certificate of uniform coverage of the entire reachable workspace.
Failure to find a witness is not proof of unreachability.

Sampler commit `dd2a311`, seed20261016 and separate target/IK RNG streams are
frozen. Limits were10,000 IK candidates,1,000,000 cube proposals and3,600 s CPU,
checked between bounded IK calls. The actual CPU preparation took12.67 s:
9,232 cube proposals,132 IK candidates and100 acceptances. Target distances
range88.83–719.35 mm. Independent CPU FK/raw-bound/cone checks pass all100;
maximum witness residual .8203 mm. Full proposal/IK logs stay under `_expr_results`;
the compact targets/arrays/CPU receipt and paired tasks are tracked in `inputs`.

## Source, build and readiness

Native solver/cost/controller source is unchanged from
`b0c1ce13c16a47cd47d8a9838df2350b12b05010`, including the existing experiment-only
mode0/1 setter. A single private N32/120-real build serves every arm. It uses
the explicit mode/inset defines printed in `launch.py`; no old GTX1070 binary
is copied into this build. All numerical sources and complete protocol values
are hashed. The ordinary execution loop comes from the frozen flat-boundary
worker; changes are input/configuration plumbing, t=0 arrival handling and
compact recording. Plant integration, solver call, applied first control,
warm-shift and stopping arithmetic are retained.

The runner/collector/native helper/machine profiles are exact copies from the
reviewed `experiment/simple-runner` source at `210d0de`, isolated on this branch.
The central runner checkout and its private worktrees/builds are untouched.
One plan pins the clean published study commit, all input hashes, eight arms,
800 cases, one shared build and four shared canonical groups.

Each variant first runs one30 s diagnostic canonical, shared across both
families. Historical baseline canonicals can time out; arrival/physical
negatives remain measured outcomes and do not hold this benchmark. Incomplete,
nonfinite or setup/import/module/configuration failures hold dependent work.
The raw worker `preflight_pass` flag is retained beside the explicit diagnostic
readiness flag. Individual task execution failures stay in the denominator and
independent cases continue. There is no automatic retry or tuning.

## Manual RTX5060 execution

After the reviewed branch is published, from the clean dedicated checkout in
the existing CUDA container:

```sh
git pull --ff-only origin experiment/reaching-sqp5-5060-20261016 && python3 experiments/reaching_sqp5_5060/launch.py
```

From an existing GATO checkout, this single command fetches the study and creates
a separate detached worktree, preserving the checkout's current branch/edits:

```sh
git fetch origin refs/heads/experiment/reaching-sqp5-5060-20261016 && git worktree add --detach ../GATO-reaching-sqp5-5060-run FETCH_HEAD && python3 ../GATO-reaching-sqp5-5060-run/experiments/reaching_sqp5_5060/launch.py
```

Choose a new worktree path if that directory already exists. To resume the
prepared worktree, run its launcher again; do not recreate or overwrite it.

Publication status and an exact-commit bootstrap command are supplied by the
root agent after review; this README does not claim a push has happened. The
launcher prints the full plan and dry-run before native work, chooses the
existing `/opt/gato-venv/bin/python3` when available, and builds only one module.
It does not install packages, change power/clocks, run Gazebo or move a robot.
Required environment: Git, CMake>=3.24, C++ compiler, CUDA supporting120-real,
Python headers, NumPy, Pinocchio and pybind11. `--python` selects another existing
interpreter. Same-command resume skips completed tasks and preserves failed
attempts; only explicit `--retry-failed` creates retries. `--collect-only` works
from saved data. `--write-plan-only` performs no device queries or native calls;
`--dry-run` also prints the runner's machine metadata/queue.

All new raw results stay beneath this checkout's
`_expr_results/reaching-sqp5-5060-20261016/rtx5060/`. Source/build caches use
`.experiment-cache/reaching-sqp5-5060-20261016/`. The launcher releases GPU work
before saved-only collection/export, then creates the separate raw backup at
`_expr_results/reaching-sqp5-5060-20261016-backup/results.tar.gz`. Return this
archive to the desktop `_expr_results`; final reports/figures/tables and compact
editable inputs belong in desktop `thesis_results`.

## Work and storage planning

Worst case is804,000 feedback solves if every task and canonical lasts30 s.
Each call allows at most5 SQP iterations using the existing stopping behavior;
actual reported work is recorded rather than assuming every call spends its cap.
Retained GTX1070 cone/terminal/flat records give roughly .32–.35 ms native time
per reported SQP iteration, with CPU/recording overhead about1.6–4.4 ms per
feedback. These are coarse cross-protocol planning references, not RTX5060
measurements. Allow roughly30–120 minutes including800 process startups, with
substantial uncertainty from the new trajectories, CPU and GPU environment.
Compilation is additional. There is no timeout-driven budget reduction.

Routine full solver horizons are omitted. Each case retains the initial cold
seed and goal, every executed raw state/control/timestamp, full typed native
PCG/SQP/work arrays, and per-feedback timing/work/error observations. Initial
inputs and the full controller/seed policy stay reproducible. This avoids
millions of tiny files; approximately10 files per episode are expected.
The dense executed arrays have an uncompressed worst case about1.75 GB across
804 episodes; typed work and compact solve logs add roughly1 GB before
compression. Plan for2–5 GB raw and a separate backup of similar or smaller size,
plus build/source caches. This is an allowance, not a measured storage gate.

Preparation checks use CPU only:

```sh
/opt/gato-venv/bin/python3 experiments/reaching_sqp5_5060/check_worker_cpu.py
python3 experiments/reaching_sqp5_5060/launch.py --write-plan-only
```

The bounded fake-solver contract checks all four mode/weight variants,
initial/ordinary arrival boundaries, one dual/rho reset, zero-control cold
seed, ten3 ms RK4 substeps, complete raw execution/typed work recording and
diagnostic-negative handling. Real RTX5060 build/native execution remains
pending until Felix manually invokes the reviewed launcher.
