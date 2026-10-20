# Optional smooth joint friction

```python
gato.register_robot("my_arm", "arm.urdf", N=[32], ee_frame="tool_joint",
                    joint_friction_smoothing=1.0)  # rad/s; per-joint list also accepted
solver = gato.BSQP("arm.urdf", plant_type="my_arm", N=32, dt=0.002)
```

The opt-in reads `<dynamics friction="c" damping="b"/>` in independent joint
order. It enables both passive terms, using actuator torque minus
`b*v + c*tanh(v/width)`. Width uses rad/s for revolute joints and m/s for
prismatic joints. None (the default) preserves rigid-body dynamics and ignores
those URDF tags. GRiD's hard-sign passive option remains disabled, avoiding
any double counting. Coefficients and widths participate in the model cache key.

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
