"""
THIS IS A COPY OF THE MPC_GATO CLASS FROM THE GATO MPC REPO, WITH MODIFICATIONS FOR THE TIAGO ROBOT.
"""


import sys
import time
import numpy as np
import pinocchio as pin
from bsqp.interface import BSQP
from bsqp.common import (
    rk4,
    sample_packed_trajectory_control,
    sample_reference_horizon,
    shift_packed_trajectory_warm_start,
)
from bsqp.config import DEFAULT_SOLVER_PARAMS
from gato_tiago.tiago_controller_process import elapsed_sim_time_from_stamp
from gato_tiago.config import TIAGO_DEFAULT_STARTUP_TIMEOUT_SEC
from gato_tiago.final_approach import FinalApproachSwitch

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
        final_approach=False,
        final_approach_params=None,
        final_approach_trigger=None,
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
            final_approach: Enable the per-goal, latched reaching parameter switch.
            final_approach_params: Overrides using the solver_params cost/SQP/mu/PCG names.
            final_approach_trigger: Distance in m and max joint speed in rad/s.
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

        self.final_approach = FinalApproachSwitch(
            solver_cfg, enabled=final_approach, params=final_approach_params,
            trigger=final_approach_trigger,
        )

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
        if self.final_approach.enabled:
            self.final_approach.check_solver(self.solver)
        
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
        controller_startup_timeout=TIAGO_DEFAULT_STARTUP_TIMEOUT_SEC,
        record_raw=False,
        record_callback=None,
        state_period_s=0.,
    ):
        """
        Run MPC controller tracking figure-8 trajectory.
        
        Current solve_time mode executes the old active plan during the current
        solver call wall duration, then activates the result on arrival.
        solve_time_legacy preserves the historical preceding-duration clock.
        record_raw/record_callback retain every solver input/output and simulated
        substep; callbacks are read-only observers outside the solve timer.
        Default controller_dt and the live ROS control policy are preserved.
        state_period_s>0 holds exact periodic plant samples for offline mode B;
        zero retains exact feedback. It does not throttle solves or control.
        """
        self.final_approach.reset(self.solver)
        use_controller = controller is not None
        if not np.isfinite(state_period_s) or state_period_s < 0:
            raise ValueError('state_period_s must be finite and nonnegative')
        if state_period_s > 0 and (use_controller or offline_timing != 'solve_time'):
            raise ValueError('state_period_s is only supported for offline solve_time tracking')
        if not np.isfinite(sim_dt) or sim_dt <= 0 or not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("sim_dt and dt must be finite and positive")
        if not np.isfinite(sim_time) or sim_time <= 0:
            raise ValueError("sim_time must be finite and positive")
        if offline_timing not in {"controller_dt", "solve_time", "solve_time_legacy"}:
            raise ValueError("offline_timing must be 'controller_dt', 'solve_time', or 'solve_time_legacy'")
        if offline_timing == "solve_time":
            if use_controller:
                raise ValueError("offline_timing='solve_time' is only valid without a live controller")
            from bsqp.offline_mpc import run_mode_b
            return run_mode_b(self, x_start, reference=fig8_traj, sim_time=sim_time,
                sim_dt=sim_dt, record_raw=record_raw, record_callback=record_callback,
                clock=time.perf_counter, **({'state_period_s': state_period_s} if state_period_s > 0 else {}),
                integrate=lambda q, dq, u, step_dt: rk4(self.model, self.data, q, dq, u, step_dt, self.actual_f_ext))

        sim_stamp_origin_sec = None
        if use_controller:
            controller.initialize(timeout_sec=controller_startup_timeout)
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
        
        from bsqp.offline_mpc import OfflineRecorder, finish_arrays
        recorder = OfflineRecorder(self, stats, offline_timing, record_raw, record_callback, time.perf_counter)
        if use_controller:
            stats['timing_mode'] = 'live'
        stats['timing_convention'] = 'historical fixed clock' if offline_timing == 'controller_dt' else 'legacy preceding returned-plan duration'

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
        XU_batch, native_us, startup_wall, bootstrap = recorder.solve(
            x_curr_batch, ee_g_batch, XU_batch, 0., 0, 0., 'startup/cold', 0.)
        stats['startup_wall_solve_time_s'] = float(startup_wall)
        stats['startup_native_solve_time_ms'] = float(native_us)/1000.
        stats['initial_state'] = x_curr.copy()
        plan_solve_index = 0
        optimizer_time = 0.
        recorder.select(bootstrap, 0)
        recorder.activation(bootstrap, 0.)
        XU_best = XU_batch[0, :]
        stats['initial_planned_control'] = XU_best[self.nx:self.nx+self.nu].copy()
        if use_controller:
            controller.send_trajectory(self.control_horizon_from_plan(XU_best), self.dt)
        
        print(f"\nRunning MPC: N={self.N}, batch={self.batch_size}, time={sim_time}s")
        if np.any(self.constant_f_ext_world):
            print(f"External force: {self.constant_f_ext_world[:3]}")
            
        # Main control loop
        last_solver_wall_time = self.dt
        last_total_sim_time = 0.0
        while total_sim_time < sim_time:
            interval_start = total_sim_time
            
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
                if offline_timing == "solve_time_legacy":
                    timestep = max(float(last_solver_wall_time), np.finfo(np.float64).eps)
                else:
                    # Default offline simulation follows the controller clock.
                    # Live ROS/Gazebo mode uses message stamps above.
                    timestep = self.dt

                interval_elapsed = 0.0
                while interval_elapsed + 1e-12 < timestep:
                    step_dt = min(sim_dt, timestep - interval_elapsed)
                    u = sample_packed_trajectory_control(
                        XU_best, self.nx, self.nu, self.N, interval_elapsed, self.dt
                    )
                    q, dq = recorder.step(
                        lambda q, dq, torque, step: rk4(self.model, self.data, q, dq, torque, step, self.actual_f_ext),
                        q, dq, u, step_dt, total_sim_time, plan_solve_index, 0, interval_elapsed)
                    total_sim_time += step_dt
                    interval_elapsed += step_dt

                x_curr = np.concatenate([q, dq])
            
            recorder.interval(interval_start, total_sim_time-interval_start, timestep,
                              plan_solve_index, 0, False, interval_start)
            stats['terminal_state'] = x_curr.copy()
            stats['terminal_time_s'] = float(total_sim_time)

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
            
            XU_batch_new, gpu_solve_time, last_solver_wall_time, raw = recorder.solve(
                x_curr_batch, ee_g_batch, XU_batch, total_sim_time, 0, timestep, 'warm', optimizer_time)
            optimizer_time = total_sim_time
            plan_solve_index += 1
            
            # Select best trajectory
            best_id = self.evaluate_best_trajectory(x_last, u_last, x_curr, timestep)
            XU_best = XU_batch_new[best_id, :]
            recorder.select(raw, best_id)
            recorder.activation(raw, total_sim_time)
            XU_batch[:, :] = XU_best
            planned_u = XU_best[self.nx:self.nx+self.nu].copy()
            if use_controller:
                controller.send_trajectory(self.control_horizon_from_plan(XU_best), self.dt)
            
            # Collect essential statistics
            ee_pos = self.solver.ee_pos(q)
            goal_dist = np.linalg.norm(ee_pos[:3] - ee_g[6:9])
            
            stats['timestamps'].append(total_sim_time)
            stats['solve_times'].append(gpu_solve_time/1000.0)  # Native us -> ms
            stats['native_solve_times_ms'].append(float(gpu_solve_time)/1000.)
            stats['wall_solve_times_s'].append(float(last_solver_wall_time))
            stats['feedback_timestamps_s'].append(float(total_sim_time))
            stats['control_activation_timestamps_s'].append(float(total_sim_time))
            stats['plan_activation_status'].append('activated')
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
                
        # Result conversion and output can outlast the final torque horizon.
        if use_controller:
            controller.finish_control(timeout_sec=controller_timeout)

        stats['terminal_state'] = x_curr.copy()
        stats['terminal_time_s'] = float(total_sim_time)
        finish_arrays(stats)

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
        controller_startup_timeout=TIAGO_DEFAULT_STARTUP_TIMEOUT_SEC,
        reset_rho_each_solve=False,
        initial_guess=None,
        offline_timing="controller_dt",
        record_raw=False,
        record_callback=None,
    ):
        """
        Run MPC controller tracking discrete goal positions.
        
        Args:
            x_start: Initial state (robot only, no pendulum)
            goals: List of 3D positions [x, y, z] or 6D poses [x, y, z, roll, pitch, yaw]
            sim_dt: Maximum simulation substep (seconds)
            offline_timing: "controller_dt" (A) or "solve_time" (B). B executes
                the old active plan during the measured solve call, then activates
                new output on arrival. solve_time_legacy retains the checkpoint
                preceding-duration approximation. Startup interval is nominal.
            record_raw: Retain every solve input/output and executed substep.
            record_callback: Optional (event, record) observer with record_raw=True;
                invoked outside solve timing. Records must be treated as read-only.
                In-progress/failure records also remain in self.last_run_stats.
            goal_timeout: Max time per goal before timeout
            goal_threshold: Distance threshold for goal reached (m)
            goal_axis_threshold: Tool local z-axis direction threshold for 6D goals (rad)
            velocity_threshold: Velocity threshold for goal reached (rad/s L1 norm)
            goal_dwell_time: Time to remain within the threshold before switching goals
            reset_rho_each_solve: Reset damping at every feedback solve. Default
                False retains adaptive damping, as in the tuned reaching study.
            initial_guess: Optional packed initial trajectory, not a tracking
                reference. Shape (packed_size,), (1, packed_size) or (batch, packed_size).
            
        Returns:
            q_traj: Joint trajectory (robot only)
            stats: Dictionary with tracking statistics including:
                - goal_outcomes: ['reached', 'timeout', ...] per goal
                - goal_reached_times: [time1, time2, ...] or None
                - time_to_all_reached: total time if all succeeded, else None
                - standard tracking stats (solve_times, timestamps, etc.)
        """
        use_controller = controller is not None
        from copy import deepcopy

        if offline_timing not in {"controller_dt", "solve_time", "solve_time_legacy"}:
            raise ValueError("offline_timing must be 'controller_dt', 'solve_time', or 'solve_time_legacy'")
        if record_callback is not None and not record_raw:
            raise ValueError("record_callback requires record_raw=True")
        if not np.isfinite(sim_dt) or sim_dt <= 0 or not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("sim_dt and dt must be finite and positive")
        if not np.isfinite(goal_timeout) or goal_timeout <= 0:
            raise ValueError("goal_timeout must be finite and positive")
        if use_controller and offline_timing != "controller_dt":
            raise ValueError("offline_timing='solve_time' is only valid without a live controller")
        goals = np.asarray(goals, dtype=np.float32)
        goals = np.atleast_2d(goals)
        if len(goals) == 0:
            raise ValueError("at least one goal is required")
        if goals.shape[1] == 3:
            pose_goals = np.hstack([goals, np.zeros((goals.shape[0], 3), dtype=np.float32)])
            check_goal_axis = False
        elif goals.shape[1] == 6:
            pose_goals = goals
            check_goal_axis = True
        else:
            raise ValueError(f"goals must have shape (n, 3) or (n, 6), got {goals.shape}")
        if (self.final_approach.enabled and not check_goal_axis
                and any(self.final_approach.after[key] > 0
                        for key in ('ee_orient_cost', 'ee_orient_N_cost'))):
            raise ValueError("Final-approach orientation costs require 6D pose goals; "
                             "set both final orientation costs to zero for position-only goals")
        if initial_guess is not None:
            initial_guess = np.atleast_2d(np.asarray(initial_guess, dtype=np.float32))
            packed_size = self.N * (self.nx + self.nu) - self.nu
            if (initial_guess.shape not in ((1, packed_size), (self.batch_size, packed_size))
                    or not np.isfinite(initial_guess).all()):
                raise ValueError("initial_guess must be a finite packed trajectory matching N and batch_size")
        if offline_timing == "solve_time":
            from bsqp.offline_mpc import run_mode_b
            return run_mode_b(self, x_start, goals=pose_goals, check_axis=check_goal_axis,
                goal_axis_threshold=goal_axis_threshold, initial_guess=initial_guess,
                reset_rho_each_solve=reset_rho_each_solve,
                goal_timeout=goal_timeout, goal_threshold=goal_threshold,
                velocity_threshold=velocity_threshold, goal_dwell_time=goal_dwell_time,
                sim_dt=sim_dt, record_raw=record_raw, record_callback=record_callback,
                clock=time.perf_counter,
                integrate=lambda q, dq, u, step_dt: rk4(self.model, self.data, q, dq, u, step_dt, self.actual_f_ext))

        self.final_approach.reset(self.solver)
        sim_stamp_origin_sec = None
        if use_controller:
            if self.has_pendulum:
                raise ValueError("live controller mode does not support pendulum simulation")
            controller.initialize(timeout_sec=controller_startup_timeout)
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
            # Activity refers to the next planned control, while goal_indices
            # labels the measured state from the previously executed interval.
            'planned_final_approach_active': [],
            'planned_goal_indices': [],
            'final_approach_switches': [],
        }
        stats.update({
            'offline_timing': offline_timing,
            'timing_mode': ('live' if use_controller else ('B-legacy' if offline_timing == 'solve_time_legacy' else 'A')),
            'raw_execution_scope': 'simulated plant substeps; live applied controls are unobservable here',
            'timing_scope': 'perf_counter wall seconds around solver.solve only; native microseconds converted to milliseconds',
            'nominal_feedback_s': float(self.dt),
            'wall_solve_times_s': [],
            'native_solve_times_ms': [],
            'update_intervals_s': [],
            'requested_update_intervals_s': [],
            'interval_start_times_s': [],
            'interval_goal_indices': [],
            'interval_plan_solve_indices': [],
            'interval_timeout_clipped': [],
        })
        if record_raw:
            stats['raw_solves'] = []
            stats['raw_execution'] = []
        self.last_run_stats = stats  # Preserve partial evidence on exceptions.

        def notify_record(event, record):
            if record_callback is not None:
                try:
                    record_callback(event, record)
                except BaseException as exc:
                    stats['recording_failure'] = {
                        'event': event, 'error_type': type(exc).__name__, 'error': str(exc)}
                    raise

        def solve_recorded(state, reference, warm_start, *, category, goal_index, shift_s):
            # Copies precede the timed call because native wrappers may mutate
            # their buffers. Recording/force updates are outside the B clock.
            if record_raw:
                raw = {'solve_index': len(stats['raw_solves']), 'category': category,
                       'time_s': float(total_sim_time), 'goal_index': int(goal_index),
                       'warm_start_shift_s': float(shift_s),
                       'state': state.copy(), 'reference': reference.copy(),
                       'warm_start': warm_start.copy(), 'status': 'started'}
                stats['raw_solves'].append(raw)
                notify_record('solve_started', raw)
            solve_start = time.perf_counter()
            try:
                result, native_us = self.solver.solve(state, reference, warm_start)
            except BaseException as exc:
                if record_raw:
                    raw.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                               wall_solve_time_s=float(time.perf_counter() - solve_start))
                    notify_record('solve_failed', raw)
                raise
            wall_s = time.perf_counter() - solve_start
            if record_raw:
                raw.update(status='completed', output=result.copy(), wall_solve_time_s=float(wall_s),
                           native_solve_time_ms=float(native_us) / 1000.0)
                if self.track_full_stats:
                    raw['solver_stats'] = deepcopy(self.solver.get_stats())
                notify_record('solve_completed', raw)
            return result, native_us, wall_s

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
        self.solver.reset_rho()
        XU_batch = np.tile(XU, (self.batch_size, 1))
        if initial_guess is not None:
            XU_batch = np.broadcast_to(initial_guess, XU_batch.shape).copy()
            XU_batch[:, :self.nx] = x_curr

        event = self.final_approach.update(
            self.solver, time_s=0.0, goal_index=0,
            distance_m=float(np.linalg.norm(self.solver.ee_pos(x_curr[:self.nq_robot]) - current_goal[:3])),
            joint_velocities=x_curr[self.nq_robot:self.nx],
        )
        if event is not None:
            stats['final_approach_switches'].append(event)
        
        # Startup task solve (cold input), not an unrelated CUDA warm-up.
        self.update_force_batch(q[:self.nq_robot] if self.has_pendulum else q)
        XU_batch, startup_native_us, startup_wall_s = solve_recorded(
            x_curr_batch, ee_g_batch, XU_batch, category='startup/cold',
            goal_index=0, shift_s=0.0)
        stats['startup_wall_solve_time_s'] = float(startup_wall_s)
        stats['startup_native_solve_time_ms'] = float(startup_native_us) / 1000.0
        last_solver_wall_time = self.dt  # Nominal startup; bootstrap is not paid.
        plan_solve_index = 0
        XU_best = XU_batch[0, :]
        if record_raw:
            stats['raw_solves'][-1]['best_trajectory_id'] = 0
        stats['initial_state'] = x_curr.copy()
        stats['initial_planned_control'] = XU_best[self.nx:self.nx+self.nu].copy()
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
            interval_start = total_sim_time
            interval_goal_index = current_goal_idx
            
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
                # Execute the current (previously returned) plan. B intentionally
                # uses the preceding solve duration, matching the tracking harness.
                requested_timestep = (max(float(last_solver_wall_time), np.finfo(np.float64).eps)
                                      if offline_timing == "solve_time_legacy" else self.dt)
                timestep = requested_timestep
                if offline_timing == "solve_time_legacy":
                    timestep = min(timestep, max(0.0, goal_timeout - (total_sim_time - goal_start_time)))
                interval_elapsed = 0.0
                while interval_elapsed + 1e-12 < timestep:
                    step_dt = min(sim_dt, timestep - interval_elapsed)
                    # Split at control knots so even a noncommensurate plant grid
                    # samples each piecewise-constant control at its valid time.
                    next_knot = (np.floor(interval_elapsed / self.dt + 1e-10) + 1) * self.dt
                    if next_knot < (self.N - 1) * self.dt:
                        step_dt = min(step_dt, next_knot - interval_elapsed)
                    u = sample_packed_trajectory_control(
                        XU_best, self.nx, self.nu, self.N, interval_elapsed + 1e-12, self.dt)
                    if self.has_pendulum:
                        damping = self.pendulum_config.get('damping', 0.4)
                        u_aug = np.zeros(self.nv)
                        u_aug[:self.nu] = u
                        u_aug[self.nu:] = -damping * dq[self.nv_robot:]
                    else:
                        u_aug = u
                    if record_raw:
                        raw_step = {
                            'time_s': float(interval_start + interval_elapsed),
                            'step_dt_s': float(step_dt), 'plan_solve_index': plan_solve_index,
                            'goal_index': interval_goal_index, 'control': u_aug.copy(),
                            'state_before': np.concatenate([q, dq]).copy(), 'status': 'started'}
                        stats['raw_execution'].append(raw_step)
                        notify_record('step_started', raw_step)
                    try:
                        q, dq = rk4(self.model, self.data, q, dq, u_aug, step_dt, self.actual_f_ext)
                    except BaseException as exc:
                        if record_raw:
                            raw_step.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                            notify_record('step_failed', raw_step)
                        raise
                    if record_raw:
                        raw_step.update(status='completed', state_after=np.concatenate([q, dq]).copy())
                        notify_record('step_completed', raw_step)
                    interval_elapsed += step_dt
                # No remainder accumulation: integrate fractional last substep now.
                total_sim_time = interval_start + timestep
                q_robot = q[:self.nq_robot] if self.has_pendulum else q
                dq_robot = dq[:self.nv_robot] if self.has_pendulum else dq
                x_curr = np.concatenate([q_robot, dq_robot])
            
            stats['interval_start_times_s'].append(float(interval_start))
            stats['update_intervals_s'].append(float(
                total_sim_time - interval_start if use_controller else timestep))
            stats['requested_update_intervals_s'].append(float(
                timestep if use_controller else requested_timestep))
            stats['interval_goal_indices'].append(interval_goal_index)
            stats['interval_plan_solve_indices'].append(plan_solve_index)
            stats['interval_timeout_clipped'].append(bool(
                (not use_controller) and offline_timing == "solve_time_legacy" and timestep < requested_timestep))

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
            if offline_timing == "solve_time_legacy":
                timeout = timeout or np.isclose(total_sim_time - goal_start_time, goal_timeout, rtol=0, atol=1e-12)
            
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
                self.final_approach.reset(self.solver)
                self.solver.reset_rho()

            event = self.final_approach.update(
                self.solver, time_s=total_sim_time, goal_index=current_goal_idx,
                distance_m=float(np.linalg.norm(ee_pos - current_goal[:3])),
                joint_velocities=dq_robot,
            )
            if event is not None:
                stats['final_approach_switches'].append(event)
            
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
            if reset_rho_each_solve:
                self.solver.reset_rho()
            
            XU_batch_new, gpu_solve_time, wall_solve_time = solve_recorded(
                x_curr_batch, ee_g_batch, XU_batch,
                category=('goal_switch' if current_goal_idx != interval_goal_index else 'warm'),
                goal_index=current_goal_idx, shift_s=timestep)
            last_solver_wall_time = wall_solve_time
            plan_solve_index += 1
            
            # Select best trajectory
            best_id = self.evaluate_best_trajectory(x_last, u_last, x_curr, timestep)
            XU_best = XU_batch_new[best_id, :]
            if record_raw:
                stats['raw_solves'][-1]['best_trajectory_id'] = int(best_id)
            XU_batch[:, :] = XU_best
            planned_u = XU_best[self.nx:self.nx+self.nu].copy()
            if use_controller:
                controller.send_trajectory(self.control_horizon_from_plan(XU_best), self.dt)
            
            # Collect statistics
            stats['timestamps'].append(total_sim_time)
            stats['solve_times'].append(gpu_solve_time/1000.0)  # Native us -> ms
            stats['wall_solve_times_s'].append(float(wall_solve_time))
            stats['native_solve_times_ms'].append(float(gpu_solve_time) / 1000.0)
            stats['goal_distances'].append(current_dist)
            stats['tool_axis_errors'].append(current_axis_error)
            stats['ee_actual'].append(ee_pos.copy())
            stats['joint_positions'].append(q_robot.copy())
            stats['joint_velocities'].append(dq_robot.copy())
            stats['best_trajectory_id'].append(best_id)
            stats['applied_controls'].append(applied_u)
            stats['planned_controls'].append(planned_u)
            stats['goal_indices'].append(measured_goal_idx)
            stats['planned_final_approach_active'].append(self.final_approach.active)
            stats['planned_goal_indices'].append(current_goal_idx)
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
        
        # Stop torque execution before statistics or caller-side result writing.
        if use_controller:
            controller.finish_control(timeout_sec=controller_timeout)

        stats['terminal_state'] = x_curr.copy()
        stats['terminal_time_s'] = float(total_sim_time)

        # Convert to numpy arrays
        for key in stats:
            if isinstance(stats[key], list) and len(stats[key]) > 0:
                if key not in ['raw_solves', 'raw_execution', 'goal_outcomes', 'goal_reached_times', 'time_to_all_reached', 'final_approach_switches']:
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
