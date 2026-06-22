"""
THIS IS A COPY OF THE MPC_GATO CLASS FROM THE GATO MPC REPO, WITH MODIFICATIONS FOR THE TIAGO ROBOT.
"""


import sys
import time
import numpy as np
import pinocchio as pin
from bsqp.interface import BSQP
from bsqp.common import rk4, sample_reference_horizon, shift_packed_trajectory_warm_start
from bsqp.config import DEFAULT_SOLVER_PARAMS
from gato_tiago.tiago_controller_process import elapsed_sim_time_from_stamp

# Import force estimator if available
sys.path.append('./examples')
try:
    from force_estimator import ForceEstimator
except ImportError:
    ForceEstimator = None


class MPC_GATO:

    def __init__(
        self,
        model,
        model_path,
        N=32,
        dt=0.03125,
        batch_size=1,
        constant_f_ext=None,
        track_full_stats=False,
        plant_type='indy7',
        pendulum_config=None,
        solver_params=None,
    ):
        """
        Initialize MPC controller.
        
        Args:
            model: Pinocchio model
            model_path: Path to URDF file
            N: Prediction horizon (knot points)
            dt: Time step
            batch_size: Number of parallel trajectories
            constant_f_ext: Constant external force/torque (optional)
            track_full_stats: If True, track all stats; if False, only essential ones
            plant_type: Plant identifier used for selecting dynamics (e.g., 'indy7', 'iiwa14')
            pendulum_config: Optional dict with keys: mass, length, damping, initial_angle
        """
        # Store original model for solver (without pendulum)
        self.solver_model = model
        
        # Add pendulum to simulation model if configured
        if pendulum_config is not None:
            self.model = self._add_pendulum_to_model(model.copy(), pendulum_config)
            self.pendulum_config = pendulum_config
            self.has_pendulum = True
            # Store dimensions
            self.nq_robot = self.solver_model.nq
            self.nv_robot = self.solver_model.nv
        else:
            self.model = model
            self.pendulum_config = None
            self.has_pendulum = False
            self.nq_robot = model.nq
            self.nv_robot = model.nv
            
        self.model.gravity.linear = np.array([0, 0, -9.81])
        self.data = self.model.createData()
        
        # Initialize solver with configurable parameters
        solver_cfg = DEFAULT_SOLVER_PARAMS.copy()
        if solver_params is not None:
            solver_cfg.update(solver_params)

        self.solver = BSQP(
            model_path=model_path,
            batch_size=batch_size,
            N=N,
            dt=dt,
            plant_type=plant_type,
            max_sqp_iters=solver_cfg['max_sqp_iters'],
            kkt_tol=solver_cfg['kkt_tol'],
            max_pcg_iters=solver_cfg['max_pcg_iters'],
            pcg_tol=solver_cfg['pcg_tol'],
            solve_ratio=solver_cfg['solve_ratio'],
            mu=solver_cfg['mu'],
            q_cost=solver_cfg['q_cost'],
            qd_cost=solver_cfg['qd_cost'],
            u_cost=solver_cfg['u_cost'],
            N_cost=solver_cfg['N_cost'],
            ee_orient_cost=solver_cfg.get('ee_orient_cost', 0.0),
            ee_orient_N_cost=solver_cfg.get('ee_orient_N_cost', 0.0),
            q_lim_cost=solver_cfg['q_lim_cost'],
            vel_lim_cost=solver_cfg['vel_lim_cost'],
            ctrl_lim_cost=solver_cfg['ctrl_lim_cost'],
            rho=solver_cfg['rho'],
        )

        self.solver_params = solver_cfg
        
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq_robot + self.nv_robot  # Solver state dimension (robot only)
        self.nu = self.solver_model.nv  # Control dimension (robot only)
        self.N = N
        self.dt = dt
        self.batch_size = batch_size
        self.track_full_stats = track_full_stats
        self.plant_type = plant_type
        
        # Setup external forces if provided
        self.setup_external_forces(constant_f_ext)
        
        # Setup force estimator for batch > 1
        self.setup_force_estimator()
        
    def setup_external_forces(self, constant_f_ext):
        """Setup external forces for simulation."""
        self.constant_f_ext_world = constant_f_ext if constant_f_ext is not None else np.zeros(6)
        # @TODO: remove this again.
        if self.plant_type == "tiago_right" and np.any(self.constant_f_ext_world):
            raise NotImplementedError(
                "External wrench dynamics are not implemented for plant_type='tiago_right'. "
                "Use constant_f_ext=None or a zero wrench."
            )
        
        # Create force vector for Pinocchio simulation
        self.actual_f_ext = pin.StdVec_Force()
        for _ in range(self.model.njoints):
            self.actual_f_ext.append(pin.Force.Zero())
            
        if constant_f_ext is not None:
            self.actual_f_ext[-1] = pin.Force(constant_f_ext[:3], constant_f_ext[3:])
    
    def setup_force_estimator(self):
        """Initialize force estimator for batch processing."""
        if self.batch_size > 1 and ForceEstimator is not None:
            self.force_estimator = ForceEstimator(
                batch_size=self.batch_size,
                initial_radius=5.0,
                min_radius=2.0,
                max_radius=20.0,
                smoothing_factor=0.5
            )
        else:
            self.force_estimator = None
            
    def run_mpc_fig8(
        self,
        x_start,
        fig8_traj,
        sim_dt=0.001,
        sim_time=5.0,
        controller=None,
        controller_timeout=2.0,
        offline_timing="controller_dt",
    ):
        """
        Run MPC controller tracking figure-8 trajectory.
        
        Returns only essential statistics for visualization and analysis.
        """
        use_controller = controller is not None
        if offline_timing not in {"controller_dt", "solve_time"}:
            raise ValueError("offline_timing must be 'controller_dt' or 'solve_time'")
        sim_stamp_origin_sec = None
        if use_controller:
            controller.initialize()
            controller_period = 1.0 / controller.target_hz
            if self.dt < controller_period:
                print(
                    "WARNING: MPC dt "
                    f"{self.dt:.6f}s is shorter than controller period "
                    f"{controller_period:.6f}s. The controller cannot apply every "
                    "planned torque knot."
                )
            state = controller.read_state(timeout_sec=controller_timeout)
            if state is None:
                raise RuntimeError("controller did not provide an initial state")
            sim_stamp_origin_sec = state.stamp_sec
            x_start = np.concatenate([state.q, state.qd]).astype(np.float32)

        # Initialize essential statistics
        stats = {
            'timestamps': [],
            'solve_times': [],        # GPU solve time in ms
            'goal_distances': [],     # Tracking error in meters
            'ee_actual': [],          # Actual end-effector positions
            'joint_positions': [],    # Joint positions over time
            'joint_velocities': [],   # Joint velocities over time
            'applied_controls': [],   # First control applied over the last interval
            'planned_controls': [],   # First control planned for the next interval
        }
        if use_controller:
            stats['state_age_sec'] = []
            stats['controller_command_rate_hz'] = []
            stats['controller_max_period_sec'] = []
        
        # Add SQP iterations only if tracking full stats
        if self.track_full_stats:
            stats['sqp_iters'] = []
            stats['pcg_iters'] = []
            stats['pcg_times_us'] = []
            stats['kkt_converged'] = []
            
        # Initialize simulation
        total_sim_time = 0.0
        x_curr = x_start
        q = x_start[:self.nq]
        dq = x_start[self.nq:self.nx]
        
        # Prepare batch inputs
        x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
        ee_g = fig8_traj[:6*self.N]
        ee_g_batch = np.tile(ee_g, (self.batch_size, 1))
        
        # Initialize warm start
        XU = np.zeros(self.N*(self.nx+self.nu)-self.nu)
        for i in range(self.N):
            start_idx = i * (self.nx + self.nu)
            XU[start_idx:start_idx+self.nx] = x_curr
        XU_batch = np.tile(XU, (self.batch_size, 1))
        
        # Reset solver
        self.solver.reset_dual()
        
        # Warm up solve
        self.update_force_batch(q)
        XU_batch, _ = self.solver.solve(x_curr_batch, ee_g_batch, XU_batch)
        XU_best = XU_batch[0, :]
        if use_controller:
            controller.send_trajectory(self.control_horizon_from_plan(XU_best), self.dt)
        
        print(f"\nRunning MPC: N={self.N}, batch={self.batch_size}, time={sim_time}s")
        if np.any(self.constant_f_ext_world):
            print(f"External force: {self.constant_f_ext_world[:3]}")
            
        # Main control loop
        last_solver_wall_time = self.dt
        last_total_sim_time = 0.0
        while total_sim_time < sim_time:
            
            # Store state for force estimation
            x_last = x_curr
            u_last = XU_best[self.nx:self.nx+self.nu]
            applied_u = u_last.copy()

            if use_controller:
                state = controller.read_state(timeout_sec=controller_timeout)
                if state is None:
                    raise RuntimeError("controller did not provide a current state")
                q = state.q.astype(np.float32)
                dq = state.qd.astype(np.float32)
                x_curr = np.concatenate([q, dq]).astype(np.float32)
                total_sim_time = elapsed_sim_time_from_stamp(
                    state.stamp_sec,
                    sim_stamp_origin_sec,
                    last_total_sim_time,
                )
                timestep = max(total_sim_time - last_total_sim_time, sim_dt)
                last_total_sim_time = total_sim_time
            else:
                if offline_timing == "solve_time":
                    timestep = max(float(last_solver_wall_time), np.finfo(np.float64).eps)
                else:
                    # Default offline simulation follows the controller clock.
                    # Live ROS/Gazebo mode uses message stamps above.
                    timestep = self.dt

                interval_elapsed = 0.0
                while interval_elapsed + 1e-12 < timestep:
                    step_dt = min(sim_dt, timestep - interval_elapsed)
                    offset = int(interval_elapsed / self.dt)
                    u_idx = self.nx + (self.nx+self.nu)*min(offset, self.N-1)
                    u = XU_best[u_idx:u_idx+self.nu]
                    q, dq = rk4(self.model, self.data, q, dq, u, step_dt, self.actual_f_ext)
                    total_sim_time += step_dt
                    interval_elapsed += step_dt

                x_curr = np.concatenate([q, dq])
            
            # Check if trajectory is complete
            if total_sim_time >= (len(fig8_traj) / 6 - self.N) * self.dt:
                break
                
            # Prepare next optimization
            x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
            ee_g = sample_reference_horizon(
                fig8_traj,
                total_sim_time,
                self.dt,
                self.N,
            ).astype(np.float32)
            ee_g_batch[:, :] = ee_g
            XU_batch = shift_packed_trajectory_warm_start(
                XU_batch,
                x_curr,
                self.nx,
                self.nu,
                self.N,
                timestep,
                self.dt,
            )
            XU_batch[:, :self.nx] = x_curr
            
            # Update forces and solve
            self.update_force_batch(q)
            self.solver.reset_rho()
            
            start = time.time()
            XU_batch_new, gpu_solve_time = self.solver.solve(
                x_curr_batch,
                ee_g_batch,
                XU_batch,
            )
            last_solver_wall_time = time.time() - start
            
            # Select best trajectory
            best_id = self.evaluate_best_trajectory(x_last, u_last, x_curr, timestep)
            XU_best = XU_batch_new[best_id, :]
            XU_batch[:, :] = XU_best
            planned_u = XU_best[self.nx:self.nx+self.nu].copy()
            if use_controller:
                controller.send_trajectory(self.control_horizon_from_plan(XU_best), self.dt)
            
            # Collect essential statistics
            ee_pos = self.solver.ee_pos(q)
            goal_dist = np.linalg.norm(ee_pos[:3] - ee_g[6:9])
            
            stats['timestamps'].append(total_sim_time)
            stats['solve_times'].append(gpu_solve_time/1000.0)  # Convert to ms
            stats['goal_distances'].append(goal_dist)
            stats['ee_actual'].append(ee_pos.copy())
            stats['joint_positions'].append(q.copy())
            stats['joint_velocities'].append(dq.copy())
            stats['applied_controls'].append(applied_u)
            stats['planned_controls'].append(planned_u)
            if use_controller:
                stats['state_age_sec'].append(state.age_sec)
                stats['controller_command_rate_hz'].append(state.command_rate_hz)
                stats['controller_max_period_sec'].append(state.max_period_sec)
            
            if self.track_full_stats:
                solver_stats = self.solver.get_stats()
                # Get first element from batch for sqp_iters
                sqp_iters = solver_stats['sqp_iters']
                if isinstance(sqp_iters, np.ndarray):
                    stats['sqp_iters'].append(int(sqp_iters[0]))
                else:
                    stats['sqp_iters'].append(int(sqp_iters))

                pcg_iters = np.asarray(solver_stats.get('pcg_iters', []), dtype=np.int32)
                if pcg_iters.size:
                    if pcg_iters.ndim == 2:
                        stats['pcg_iters'].append(pcg_iters[:, best_id].copy())
                    else:
                        stats['pcg_iters'].append(pcg_iters.reshape(-1).copy())
                else:
                    stats['pcg_iters'].append(np.array([], dtype=np.int32))

                pcg_times = np.asarray(solver_stats.get('pcg_times_us', []), dtype=np.float32).reshape(-1)
                stats['pcg_times_us'].append(pcg_times.copy())

                kkt_converged = solver_stats.get('kkt_converged', [])
                if isinstance(kkt_converged, np.ndarray):
                    stats['kkt_converged'].append(int(kkt_converged[best_id]))
                else:
                    stats['kkt_converged'].append(int(kkt_converged) if kkt_converged != [] else 0)
                
        # Convert to numpy arrays
        for key in stats:
            if stats[key]:
                try:
                    stats[key] = np.array(stats[key])
                except (ValueError, TypeError):
                    # Keep as list if conversion fails
                    pass
                
        # Print summary
        print(f"Avg error: {np.mean(stats['goal_distances']):.4f}m")
        print(f"Avg solve time: {np.mean(stats['solve_times']):.3f}ms")
        
        return None, stats  # Return None for trajectory (not needed)

    def control_horizon_from_plan(self, XU_best):
        """Extract the torque horizon from a packed state/control trajectory."""
        controls = []
        for i in range(self.N - 1):
            u_idx = self.nx + i * (self.nx + self.nu)
            controls.append(XU_best[u_idx:u_idx+self.nu].copy())
        return np.asarray(controls, dtype=np.float64)
    
    def update_force_batch(self, q):
        """Update force hypotheses for batch solving."""
        if self.batch_size == 1 or self.force_estimator is None:
            return
            
        # Generate force batch
        force_batch = self.force_estimator.generate_batch()
        
        # Transform to GATO frame
        transformed_batch = np.zeros_like(force_batch)
        for i in range(self.batch_size):
            transformed_batch[i, :] = self.transform_force_to_gato_frame(q, force_batch[i, :])
            
        self.solver.set_f_ext_B(transformed_batch)
        
    def evaluate_best_trajectory(self, x_last, u_last, x_curr, dt):
        """Evaluate which trajectory best matches reality."""
        if self.batch_size == 1 or self.force_estimator is None:
            return 0
            
        # Simulate all hypotheses
        x_next_batch = self.solver.sim_forward(x_last, u_last, dt)
        
        # Calculate errors
        errors = np.linalg.norm(x_next_batch - x_curr[None, :], axis=1)
        best_id = np.argmin(errors)
        
        # Update estimator
        self.force_estimator.update(best_id, errors, alpha=0.6, beta=0.5)
        
        return best_id
        
    def transform_force_to_gato_frame(self, q, f_world):
        """Transform force from world frame to GATO frame."""
        data = self.solver_model.createData()
        q_robot = q[:self.nq_robot]
        
        pin.forwardKinematics(self.solver_model, data, q_robot)
        pin.updateFramePlacements(self.solver_model, data)
        
        # Joint indices - get end-effector parent joint
        jid_ee_fin = self.solver_model.getFrameId("EE")
        jid_ee_pin = self.solver_model.frames[jid_ee_fin].parentJoint
        jid_eep_pin = jid_ee_pin - 1  # End-effector parent joint
        
        # Get transformations
        transform_world_to_ee = data.oMi[jid_ee_pin]
        transform_world_to_jeep = data.oMi[jid_eep_pin]
        transform_jeep_to_ee = transform_world_to_jeep.inverse() * transform_world_to_ee
        
        # Transform force
        force_ee_world = pin.Force(f_world[:3], f_world[3:])
        force_ee_local = transform_world_to_ee.actInv(force_ee_world)
        wrench_jeep_local = transform_jeep_to_ee.actInv(force_ee_local)
        
        result = np.zeros(6)
        result[:3] = wrench_jeep_local.linear
        result[3:] = wrench_jeep_local.angular
        
        return result
    
    def _add_pendulum_to_model(self, model, config):
        """Add a 3D pendulum (spherical joint) to the end-effector."""
        mass = config.get('mass', 15.0)
        length = config.get('length', 0.3)
        
        ee_joint_id = model.njoints - 1  # Last joint is EE
        pendulum_joint_id = model.addJoint(
            ee_joint_id,
            pin.JointModelSpherical(),
            pin.SE3.Identity(),
            "pendulum_joint"
        )
        
        # Create inertia for pendulum bob (point mass at distance)
        com = np.array([0.0, 0.0, -length])  # Center of mass along -Z
        inertia_matrix = np.diag([0.001, 0.001, 0.001])  # Small inertia at COM
        pendulum_inertia = pin.Inertia(mass, com, inertia_matrix)
        model.appendBodyToJoint(pendulum_joint_id, pendulum_inertia, pin.SE3.Identity())
        
        return model
    
    def run_mpc_goals(
        self, 
        x_start, 
        goals, 
        sim_dt=0.001, 
        goal_timeout=5.0,
        goal_threshold=0.05,
        goal_axis_threshold=np.deg2rad(15.0),
        velocity_threshold=1.0,
        goal_dwell_time=0.0,
        controller=None,
        controller_timeout=2.0,
    ):
        """
        Run MPC controller tracking discrete goal positions.
        
        Args:
            x_start: Initial state (robot only, no pendulum)
            goals: List of 3D positions [x, y, z] or 6D poses [x, y, z, roll, pitch, yaw]
            sim_dt: Simulation timestep
            goal_timeout: Max time per goal before timeout
            goal_threshold: Distance threshold for goal reached (m)
            goal_axis_threshold: Tool local z-axis direction threshold for 6D goals (rad)
            velocity_threshold: Velocity threshold for goal reached (rad/s L1 norm)
            goal_dwell_time: Time to remain within the threshold before switching goals
            
        Returns:
            q_traj: Joint trajectory (robot only)
            stats: Dictionary with tracking statistics including:
                - goal_outcomes: ['reached', 'timeout', ...] per goal
                - goal_reached_times: [time1, time2, ...] or None
                - time_to_all_reached: total time if all succeeded, else None
                - standard tracking stats (solve_times, timestamps, etc.)
        """
        use_controller = controller is not None
        goals = np.asarray(goals, dtype=np.float32)
        goals = np.atleast_2d(goals)
        if goals.shape[1] == 3:
            pose_goals = np.hstack([goals, np.zeros((goals.shape[0], 3), dtype=np.float32)])
            check_goal_axis = False
        elif goals.shape[1] == 6:
            pose_goals = goals
            check_goal_axis = True
        else:
            raise ValueError(f"goals must have shape (n, 3) or (n, 6), got {goals.shape}")
        sim_stamp_origin_sec = None
        if use_controller:
            if self.has_pendulum:
                raise ValueError("live controller mode does not support pendulum simulation")
            controller.initialize()
            controller_period = 1.0 / controller.target_hz
            if self.dt < controller_period:
                print(
                    "WARNING: MPC dt "
                    f"{self.dt:.6f}s is shorter than controller period "
                    f"{controller_period:.6f}s. The controller cannot apply every "
                    "planned torque knot."
                )
            state = controller.read_state(timeout_sec=controller_timeout)
            if state is None:
                raise RuntimeError("controller did not provide an initial state")
            sim_stamp_origin_sec = state.stamp_sec
            x_start = np.concatenate([state.q, state.qd]).astype(np.float32)

        # Initialize statistics
        stats = {
            'timestamps': [],
            'solve_times': [],
            'goal_distances': [],
            'ee_actual': [],
            'joint_positions': [],
            'joint_velocities': [],
            'best_trajectory_id': [],
            'applied_controls': [],
            'planned_controls': [],
            'tool_axis_errors': [],
            'goal_indices': [],
        }
        if use_controller:
            stats['state_age_sec'] = []
            stats['controller_command_rate_hz'] = []
            stats['controller_max_period_sec'] = []
        
        if self.track_full_stats:
            stats['sqp_iters'] = []
            stats['pcg_iters'] = []
            
        stats['goal_outcomes'] = ['not_reached'] * len(pose_goals)
        stats['goal_reached_times'] = [None] * len(pose_goals)
        stats['time_to_all_reached'] = None
        
        # Initialize simulation state
        total_sim_time = 0.0
        accumulated_time = 0.0
        
        # Initialize augmented state with pendulum if configured
        if self.has_pendulum:
            x_start_aug = np.zeros(self.nq + self.nv)
            x_start_aug[:self.nx] = x_start  # Robot state
            # Pendulum initial angle
            pendulum_init = self.pendulum_config.get('initial_angle', np.array([0.3, 0.0, 0.0]))
            x_start_aug[self.nq_robot:self.nq_robot+3] = pendulum_init
            q = x_start_aug[:self.nq]
            dq = x_start_aug[self.nq:]
        else:
            q = x_start[:self.nq]
            dq = x_start[self.nq:]
        
        # Solver uses robot-only state
        x_curr = x_start
        x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
        
        # Initialize first goal
        current_goal_idx = 0
        current_goal = pose_goals[current_goal_idx]
        ee_g = np.tile(current_goal, self.N)
        ee_g_batch = np.tile(ee_g, (self.batch_size, 1))
        
        # Initialize warm start
        XU = np.zeros(self.N*(self.nx+self.nu)-self.nu)
        for i in range(self.N):
            start_idx = i * (self.nx + self.nu)
            XU[start_idx:start_idx+self.nx] = x_curr
        self.solver.reset_dual()
        XU_batch = np.tile(XU, (self.batch_size, 1))
        
        # Warm up solve
        self.update_force_batch(q[:self.nq_robot] if self.has_pendulum else q)
        XU_batch, _ = self.solver.solve(x_curr_batch, ee_g_batch, XU_batch)
        XU_best = XU_batch[0, :]
        if use_controller:
            controller.send_trajectory(self.control_horizon_from_plan(XU_best), self.dt)
        
        print(f"\nRunning MPC: N={self.N}, batch={self.batch_size}, {len(pose_goals)} goals")
        if self.has_pendulum:
            print(f"Pendulum: mass={self.pendulum_config['mass']}kg, length={self.pendulum_config['length']}m")
        
        # Start timing for the current goal
        goal_start_time = total_sim_time
        goal_dwell_start_time = None
        # Main control loop
        last_total_sim_time = 0.0
        while total_sim_time < goal_timeout * len(pose_goals):
            
            # Store state for force estimation
            x_last = x_curr
            u_last = XU_best[self.nx:self.nx+self.nu]
            applied_u = u_last.copy()

            if use_controller:
                state = controller.read_state(timeout_sec=controller_timeout)
                if state is None:
                    raise RuntimeError("controller did not provide a current state")
                q = state.q.astype(np.float32)
                dq = state.qd.astype(np.float32)
                x_curr = np.concatenate([q, dq]).astype(np.float32)
                total_sim_time = elapsed_sim_time_from_stamp(
                    state.stamp_sec,
                    sim_stamp_origin_sec,
                    last_total_sim_time,
                )
                timestep = max(total_sim_time - last_total_sim_time, sim_dt)
                last_total_sim_time = total_sim_time
                q_robot = q
                dq_robot = dq
            else:
                # Simulate forward with current control
                timestep = self.dt
                nsteps = int(timestep/sim_dt)

                for i in range(nsteps):
                    offset = int(i/(self.dt/sim_dt))
                    u_idx = self.nx + (self.nx+self.nu)*min(offset, self.N-1)
                    u = XU_best[u_idx:u_idx+self.nu]

                    # Augment control with robot torques and pendulum damping if needed
                    if self.has_pendulum:
                        damping = self.pendulum_config.get('damping', 0.4)
                        u_aug = np.zeros(self.nv)
                        u_aug[:self.nu] = u
                        u_aug[self.nu:] = -damping * dq[self.nv_robot:]
                    else:
                        u_aug = u

                    q, dq = rk4(self.model, self.data, q, dq, u_aug, sim_dt, self.actual_f_ext)
                    total_sim_time += sim_dt

                # Handle residual time
                if timestep % sim_dt > 1e-5:
                    accumulated_time += timestep % sim_dt
                    if accumulated_time >= sim_dt:
                        accumulated_time = 0.0
                        offset = int(nsteps/(self.dt/sim_dt))
                        u_idx = self.nx + (self.nx+self.nu)*min(offset, self.N-1)
                        u = XU_best[u_idx:u_idx+self.nu]

                        if self.has_pendulum:
                            damping = self.pendulum_config.get('damping', 0.4)
                            u_aug = np.zeros(self.nv)
                            u_aug[:self.nu] = u
                            u_aug[self.nv_robot:] = -damping * dq[self.nv_robot:]
                        else:
                            u_aug = u

                        q, dq = rk4(self.model, self.data, q, dq, u_aug, sim_dt, self.actual_f_ext)
                        total_sim_time += sim_dt

                # Update solver state (robot only)
                q_robot = q[:self.nq_robot] if self.has_pendulum else q
                dq_robot = dq[:self.nv_robot] if self.has_pendulum else dq
                x_curr = np.concatenate([q_robot, dq_robot])
            
            # Check goal reached or timeout
            measured_goal_idx = current_goal_idx
            ee_pos = self.solver.ee_pos(q_robot)
            current_dist = np.linalg.norm(ee_pos - current_goal[:3])
            current_axis_error = self.solver.ee_tool_axis_error(q_robot, current_goal[3:6])
            current_vel = np.linalg.norm(dq_robot, ord=1)
            within_goal = (
                (current_dist < goal_threshold)
                and (current_vel < velocity_threshold)
                and ((not check_goal_axis) or (current_axis_error < goal_axis_threshold))
            )
            if within_goal:
                if goal_dwell_start_time is None:
                    goal_dwell_start_time = total_sim_time
                reached = (total_sim_time - goal_dwell_start_time) >= goal_dwell_time
            else:
                goal_dwell_start_time = None
                reached = False
            timeout = (total_sim_time - goal_start_time) >= goal_timeout
            
            if reached or timeout:
                if reached:
                    stats['goal_outcomes'][current_goal_idx] = 'reached'
                    stats['goal_reached_times'][current_goal_idx] = total_sim_time
                else:
                    stats['goal_outcomes'][current_goal_idx] = 'timeout'
                    
                current_goal_idx += 1
                if current_goal_idx >= len(pose_goals):
                    break
                    
                current_goal = pose_goals[current_goal_idx]
                ee_g = np.tile(current_goal, self.N)
                goal_start_time = total_sim_time
                goal_dwell_start_time = None
                self.solver.reset_rho()
            
            # Prepare next optimization
            x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
            ee_g_batch[:, :] = ee_g
            XU_batch = shift_packed_trajectory_warm_start(
                XU_batch,
                x_curr,
                self.nx,
                self.nu,
                self.N,
                timestep,
                self.dt,
            )
            XU_batch[:, :self.nx] = x_curr
            
            # Update forces and solve
            self.update_force_batch(q_robot)
            self.solver.reset_rho()
            
            start = time.time()
            XU_batch_new, gpu_solve_time = self.solver.solve(x_curr_batch, ee_g_batch, XU_batch)
            wall_solve_time = time.time() - start
            
            # Select best trajectory
            best_id = self.evaluate_best_trajectory(x_last, u_last, x_curr, max(sim_dt, round(timestep / sim_dt) * sim_dt))
            XU_best = XU_batch_new[best_id, :]
            XU_batch[:, :] = XU_best
            planned_u = XU_best[self.nx:self.nx+self.nu].copy()
            if use_controller:
                controller.send_trajectory(self.control_horizon_from_plan(XU_best), self.dt)
            
            # Collect statistics
            stats['timestamps'].append(total_sim_time)
            stats['solve_times'].append(gpu_solve_time/1000.0)  # Convert to ms
            stats['goal_distances'].append(current_dist)
            stats['tool_axis_errors'].append(current_axis_error)
            stats['ee_actual'].append(ee_pos.copy())
            stats['joint_positions'].append(q_robot.copy())
            stats['joint_velocities'].append(dq_robot.copy())
            stats['best_trajectory_id'].append(best_id)
            stats['applied_controls'].append(applied_u)
            stats['planned_controls'].append(planned_u)
            stats['goal_indices'].append(measured_goal_idx)
            if use_controller:
                stats['state_age_sec'].append(state.age_sec)
                stats['controller_command_rate_hz'].append(state.command_rate_hz)
                stats['controller_max_period_sec'].append(state.max_period_sec)
            
            if self.track_full_stats:
                solver_stats = self.solver.get_stats()
                # Get first element from batch for sqp_iters and pcg_iters
                sqp_iters = solver_stats['sqp_iters']
                if isinstance(sqp_iters, np.ndarray):
                    stats['sqp_iters'].append(int(sqp_iters[0]))
                else:
                    stats['sqp_iters'].append(int(sqp_iters))
                    
                pcg_iters = solver_stats.get('pcg_iters', [])
                if len(pcg_iters) > 0:
                    if isinstance(pcg_iters, np.ndarray) and pcg_iters.ndim == 2:
                        stats['pcg_iters'].append(int(pcg_iters[0, 0]))
                    else:
                        stats['pcg_iters'].append(int(pcg_iters[0]))
                else:
                    stats['pcg_iters'].append(0)
        
        # Convert to numpy arrays
        for key in stats:
            if isinstance(stats[key], list) and len(stats[key]) > 0:
                if key not in ['goal_outcomes', 'goal_reached_times', 'time_to_all_reached']:
                    try:
                        stats[key] = np.array(stats[key])
                    except (ValueError, TypeError):
                        # Keep as list if conversion fails (e.g., inhomogeneous shapes)
                        pass
        
        # Compute time to all-goals-reached if all were reached
        if all([o == 'reached' for o in stats['goal_outcomes']]):
            reached_times = [t for t in stats['goal_reached_times'] if t is not None]
            if len(reached_times) == len(pose_goals):
                stats['time_to_all_reached'] = float(np.max(reached_times))
        
        # Print summary
        goals_reached = sum(1 for o in stats['goal_outcomes'] if o == 'reached')
        print(f"Goals reached: {goals_reached}/{len(pose_goals)}")
        if len(stats['solve_times']) > 0:
            print(f"Avg solve time: {np.mean(stats['solve_times']):.3f}ms")
        
        return None, stats  # Return None for trajectory (not needed)
