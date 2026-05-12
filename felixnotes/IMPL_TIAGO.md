# Implementing Tiago

This file records the information that matters if Tiago support needs to be
recreated from scratch.

It does not try to mirror the full implementation. Use it as a guide and refer
to the checked-in Tiago files for the current state.

## High-Level Shape

The intended Tiago integration in this repo is:

- an arm-only model, not the full mobile manipulator
- rooted at `torso_lift_link`
- built around the right 7-DoF arm
- backed by GRiD-generated dynamics plus a handwritten wrapper under
  `gato/dynamics/tiago_right/`

The most relevant files are expected to be:

- `tools/generate_tiago_dynamics.py`
- `gato/dynamics/tiago_right/tiago_right_grid.cuh`
- `gato/dynamics/tiago_right/tiago_right_plant.cuh`
- `examples/tiago_arm_tracking_simple.py`
- `examples/tiago_arm_mpc_tracking_simple.py`
- `examples/tiago_arm_mpc_fig8_tracking.py`
- `examples/tiago_arm_pickplace_goals.py`
- `examples/plot_figure8_tracking.py`

## Local Inputs Not Stored In Git

The repository intentionally does not keep the local Tiago URDF workspace in
Git. `TiagoProURDF/` is ignored.

For a fresh Tiago implementation, expect to provide locally:

- the full Tiago URDF, typically as `TiagoProURDF/tiago_pro.urdf`

The GRiD-normalized arm-only URDF is no longer intended to be kept as a local
artifact. `tools/generate_tiago_dynamics.py` creates it in a temporary
directory, runs vanilla GRiD, and removes the temporary file after generation.
Python examples should also use the full URDF. The BSQP Python interface
reduces it to the seven right-arm joints with Pinocchio when
`plant_type="tiago_right"` is selected.

For closed-loop MPC simulation, use an arm-only Pinocchio model that matches
the generated GRiD dynamics. A model reduced from the full Tiago URDF can be
kinematically useful, but it retains enough locked-body structure that its
dynamics do not match the arm-only GRiD model closely enough for validating
rollout tracking. The MPC smoke example creates the native arm-only URDF in a
temporary directory and removes it after the run.

## Step 1: Extract An Arm-Only URDF

Use `tools/generate_tiago_dynamics.py` for dynamics generation.

The script:

- reads `TiagoProURDF/tiago_pro.urdf`
- extracts the selected arm rooted at `torso_lift_link`
- writes the extracted GRiD input only to a temporary directory
- normalizes negative principal axes only in that temporary GRiD input
- runs vanilla GRiD
- writes the generated header to
  `gato/dynamics/tiago_right/tiago_right_grid.cuh` by default

Typical command:

```sh
python tools/generate_tiago_dynamics.py
```

Important extraction behavior:

- keep only:
  - `torso_lift_link`
  - `arm_right_1_joint` through `arm_right_7_joint`
  - `arm_right_tool_joint`
  - corresponding arm links
- ensure every joint origin explicitly has both `xyz` and `rpy`
- normalize negative principal joint axes for GRiD

## Step 2: Account For GRiD Parser Constraints

Two parser constraints mattered:

- GRiD expected joint origins to contain both `xyz` and `rpy`
- GRiD only handled positive principal joint axes in the required path

The Tiago-specific consequence is important:

- `arm_right_6_joint` originally uses axis `0 0 -1`
- the GRiD-normalized export rewrites it to `0 0 1`
- the joint limits must be negated/swapped accordingly
- the handwritten wrapper must then undo this convention mismatch at runtime

That joint-6 sign handling is not optional.

## Step 3: Run GRiD On The Normalized Arm

GRiD is available as a `GRiD/` submodule. Its nested submodules may need an
HTTPS rewrite because upstream records them as SSH URLs:

```sh
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive GRiD
```

The active Python environment must also provide GRiD's Python dependencies:

```sh
python -m pip install -r GRiD/requirements.txt
```

When regenerating Tiago dynamics:

- run `python tools/generate_tiago_dynamics.py`
- do not keep the normalized arm URDF as a checked-in or local workflow
  artifact
- use the script output under `gato/dynamics/tiago_right/`

Important rule:

- do not hand-edit GRiD-generated outputs
- adapt behavior in the handwritten wrapper or regenerate the artifact instead

Important limitation:

- the current vanilla GRiD checkout generates the core dynamics helpers but does
  not generate the old trial's end-effector pose/gradient helper functions
- Tiago tracking wrapper work must account for this explicitly rather than
  assuming `end_effector_pose_*` helpers exist in the regenerated header

## Step 4: Build The Handwritten Tiago Wrapper

The wrapper is where most of the real integration work lives.

Responsibilities:

- expose Tiago joint/velocity/control limits
- map between the original Tiago convention and the GRiD-normalized convention
- include the joint-6 sign conversion in:
  - `q`
  - `qd`
  - `u`
  - `qdd`
  - forward-dynamics Jacobians
  - end-effector tracking gradients
- integrate the plant into:
  - `gato/settings.h`
  - `CMakeLists.txt`
  - `python/bindings.cu`
  - `python/bsqp/interface.py`

## Step 5: Avoid The Tiago-Specific Shared-Memory Traps

Two Tiago wrapper mistakes turned out to be real bugs.

### 1. Do Not Route Cooperative GRiD Outputs Through Thread-Local Arrays

The initial Tiago wrapper used local arrays like:

- `T q_grid[NUM_JOINTS]`
- `T qdd_grid[NUM_JOINTS]`

for paths that GRiD expected to cooperate on through shared memory. That caused
bad dynamics behavior in the solver.

The stable fix was to use wrapper-controlled shared storage instead.

### 2. Be Careful With Any End-Effector Helper Path

The old Tiago trial used generated fixed-target helpers such as:

- `end_effector_pose_device_arm_right_tool_joint`
- `end_effector_pose_gradient_device_arm_right_tool_joint`

grab `extern __shared__` internally. Calling them from inside solver kernels
caused shared-memory layout collisions.

The stable fix was:

- call the corresponding `_inner` helper variants
- pass explicitly managed scratch storage from the wrapper

The current vanilla GRiD checkout does not generate those helpers, but the same
shared-memory rule applies to any future generated or handwritten GPU
kinematics helper used from inside solver kernels: the wrapper must make scratch
ownership explicit.

## Step 6: Re-Debug With Finite Differences

The fastest way to validate a fresh Tiago integration is to reuse the debug
surface added during the original bring-up.

Use:

- `examples/tiago_debug_checks.py`

It compares:

- one-step rollout linearization against finite differences
- tracking gradients against finite differences
- then prints SQP/PCG/line-search diagnostics

The most useful interpretation:

- if rollout linearization is wrong, focus on the dynamics wrapper/sign mapping
- if rollout is fine but tracking terms are wrong, focus on Tiago
  end-effector/tracking derivatives
- if both are fine, the remaining bug is likely in the generic optimizer path

## Step 7: Separate Plant Bugs From Demo Tuning

Not every bad Tiago trajectory was a plant bug.

Two issues were only example/controller configuration:

- the original Tiago `ready` pose started too close to a joint limit
- weak limit/velocity penalties made the closed-loop motion look jittery

Useful repo references:

- `python/bsqp/config.py`
- `BUGS.md`

So when bringing Tiago up again:

1. verify kinematic feasibility first
2. validate rollout and tracking derivatives
3. only then tune the example/controller defaults

## Step 8: Repro Commands That Were Useful

Build:

```sh
source .venv/bin/activate
PLANT=tiago_right KNOTS=16 ./tools/build.sh
```

Smoke example:

```sh
source .venv/bin/activate
PYTHONPATH=python python examples/tiago_arm_simple.py
```

Tracking example:

```sh
source .venv/bin/activate
PYTHONPATH=python python examples/tiago_arm_tracking.py --ik-check
```

Derivative check:

```sh
source .venv/bin/activate
PYTHONPATH=python python examples/tiago_debug_checks.py --terminal
```
