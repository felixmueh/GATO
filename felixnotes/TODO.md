# Fix hadn written end-effector kinematics/Jacobian. (GRID EE Helpers)
# Move tiago urdf to examples/tiagopro_description
# Only compile for one architecture by default!
# Remove? Rework pendulum_config in MPC_GATO
- Why is this on there?
- **In general, split experiment and controller logic**
# Why is there still a special case build setup for tiago in bsqp/interface.py
# Bad interface of self.solver.sim_forward. This is implicitly batched over batches set in self.solver.set_f_ext_B.
