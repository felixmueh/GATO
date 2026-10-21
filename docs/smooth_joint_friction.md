# Optional smooth joint friction

```python
gato.register_robot("my_arm", "arm.urdf", N=[32], ee_frame="tool_joint",
                    joint_friction_smoothing=1.0)  # rad/s; per-joint list also accepted
solver = gato.BSQP("arm.urdf", plant_type="my_arm", batch_size=1, N=32, dt=0.002,
                   joint_friction_smoothing=0.03)  # optional constructor override
solver.set_joint_friction_smoothing(0.06)  # no compilation; a per-joint vector also works
```

The opt-in reads `<dynamics friction="c" damping="b"/>` in independent joint
order. It enables both passive terms, using actuator torque minus
`b*v + c*tanh(v/width)`. Width uses rad/s for revolute joints and m/s for
prismatic joints. Registration with None (the default) preserves rigid-body dynamics and ignores
those URDF tags. GRiD's hard-sign passive option remains disabled, avoiding
any double counting. URDF coefficients and friction-enabled versus disabled
support participate in the model cache key. Numeric widths do not.

Registration stores default widths for newly constructed Python solvers; an
explicit constructor argument overrides them. Re-registering with different
widths updates that default while reusing generated headers and compiled modules.
Existing solvers retain their values. Each solver owns an independent device
buffer shared by its batch lanes and knots. Setters copy a small width vector
between calls and do not reset trajectory warm starts or duals. Concurrent
updates/solves on one instance are unsupported, as for other solver setters.

Widths must be positive and finite after conversion to float32. Changing width
cannot enable losses in a module built without them; register the opt-in and
build that module first. Earlier compile-time-width binaries require one rebuild.
Existing registration metadata with `[Coulomb, viscous, width]` rows retains
its width defaults when its module is rebuilt. Malformed defaults raise an error.
The native C++ solver defaults to width 1 in each joint; Python applies the
registration default during construction. Runtime setters also validate values
at the native boundary. Friction-disabled solvers allocate no width buffer.

Record `solver.joint_friction_smoothing` alongside the model identity: a module
hash alone no longer identifies the effective smoothing law. This array is a
snapshot; use the setter to change the device values.

The fixed-base adapter evaluates the modified dynamics and includes
`b + c/width*(1-tanh(v/width)**2)` on the inverse-dynamics velocity diagonal
before composing its analytical acceleration Jacobian. The inverse-dynamics
position derivative uses the resulting acceleration too. Existing integration
and KKT assembly consume these consistent values/derivatives unchanged.

This option supports the default first-order solver variant with independent
scalar joints. Floating base, mimic joints, exact-Hessian and contact-force
variants are rejected. No stiction or Stribeck effect is modeled. External
simulators may intentionally use hard Coulomb friction to study model mismatch.

Small widths increase the slope near rest and can make an explicit integrator
unstable. A smooth function alone does not solve discretization stiffness.
Choose width and knot duration together; the example is illustrative rather
than a universally qualified parameterization. Neither coefficients nor widths
are physical calibration. Offline application tests keep the hard-sign plant
and record their own timing, width and coefficient provenance.
