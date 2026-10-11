import sys
import time
import numpy as np
import pinocchio as pin
from .wrench import WorldToolWrench, world_tool_to_joint
from .interface import BSQP
from .common import (
    rk4,
    sample_packed_trajectory_control,
    sample_reference_horizon,
    shift_packed_trajectory_warm_start,
)
from .config import DEFAULT_SOLVER_PARAMS

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
                "Constant world-wrench prediction is unsupported for plant_type='tiago_right'. "
                "Use zero here or the explicit body-fixed force experiment backend."
            )
        
        # Create force vector for Pinocchio simulation
        self.actual_f_ext = pin.StdVec_Force()
        for _ in range(self.model.njoints):
            self.actual_f_ext.append(pin.Force.Zero())
            
        if constant_f_ext is not None and np.any(constant_f_ext):
            # The simulation re-expresses world/tool-point force at each RK stage.
            self.actual_f_ext = WorldToolWrench(self.model, constant_f_ext)
    
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
        offline_timing="controller_dt",
        record_raw=False,
        record_callback=None,
    ):
        """
        Run MPC controller tracking figure-8 trajectory.
        
        Current solve_time mode executes the old active plan during the current
        solver call wall duration, then activates the result on arrival.
        solve_time_legacy preserves the historical preceding-duration clock.
        record_raw/record_callback retain every solver input/output and simulated
        substep; callbacks are read-only observers outside the solve timer.
        Default controller_dt and the live ROS control policy are preserved.
        """
        if not np.isfinite(sim_dt) or sim_dt <= 0 or not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("sim_dt and dt must be finite and positive")
        if not np.isfinite(sim_time) or sim_time <= 0:
            raise ValueError("sim_time must be finite and positive")
        if offline_timing not in {"controller_dt", "solve_time", "solve_time_legacy"}:
            raise ValueError("offline_timing must be 'controller_dt', 'solve_time', or 'solve_time_legacy'")

        if offline_timing == "solve_time":
            from bsqp.offline_mpc import run_mode_b
            return run_mode_b(self, x_start, reference=fig8_traj, sim_time=sim_time,
                sim_dt=sim_dt, record_raw=record_raw, record_callback=record_callback,
                clock=time.perf_counter,
                integrate=lambda q, dq, u, step_dt: rk4(self.model, self.data, q, dq, u, step_dt, self.actual_f_ext))

        # Initialize essential statistics
        stats = {
            'timestamps': [],
            'solve_times': [],        # GPU solve time in ms
            'goal_distances': [],     # Tracking error in meters
            'tool_axis_errors': [],   # EE local z-axis direction error in radians
            'ee_actual': [],          # Actual end-effector positions
            'joint_positions': [],    # Joint positions over time
            'joint_velocities': [],   # Joint velocities over time
            'applied_controls': [],   # First control applied over the last interval
            'planned_controls': [],   # First control planned for the next interval
        }
        
        from bsqp.offline_mpc import OfflineRecorder, finish_arrays
        recorder = OfflineRecorder(self, stats, offline_timing, record_raw, record_callback, time.perf_counter)
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
        
        print(f"\nRunning MPC: N={self.N}, batch={self.batch_size}, time={sim_time}s")
        if np.any(self.constant_f_ext_world):
            print(f"External force: {self.constant_f_ext_world[:3]}")
            
        # Main control loop
        last_solver_wall_time = self.dt
        while total_sim_time < sim_time:
            interval_start = total_sim_time
            
            # Store state for force estimation
            x_last = x_curr
            u_last = XU_best[self.nx:self.nx+self.nu]
            applied_u = u_last.copy()
            
            if offline_timing == "solve_time_legacy":
                timestep = max(float(last_solver_wall_time), np.finfo(np.float64).eps)
            else:
                # Default offline simulation follows the controller clock.
                # Wall-clock solve time is diagnostic output in this mode.
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
            
            # Collect essential statistics
            q_solver = q[:self.nq_robot]
            ee_pos = self.solver.ee_pos(q_solver)
            goal_dist = np.linalg.norm(ee_pos[:3] - ee_g[6:9])
            tool_axis_error = self.solver.ee_tool_axis_error(q_solver, ee_g[9:12])
            
            stats['timestamps'].append(total_sim_time)
            stats['solve_times'].append(gpu_solve_time/1000.0)  # Native us -> ms
            stats['native_solve_times_ms'].append(float(gpu_solve_time)/1000.)
            stats['wall_solve_times_s'].append(float(last_solver_wall_time))
            stats['feedback_timestamps_s'].append(float(total_sim_time))
            stats['control_activation_timestamps_s'].append(float(total_sim_time))
            stats['plan_activation_status'].append('activated')
            stats['goal_distances'].append(goal_dist)
            stats['tool_axis_errors'].append(tool_axis_error)
            stats['ee_actual'].append(ee_pos.copy())
            stats['joint_positions'].append(q.copy())
            stats['joint_velocities'].append(dq.copy())
            stats['applied_controls'].append(applied_u)
            stats['planned_controls'].append(planned_u)
            
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
                
        stats['terminal_state'] = x_curr.copy()
        stats['terminal_time_s'] = float(total_sim_time)
        finish_arrays(stats)

        # Print summary
        print(f"Avg error: {np.mean(stats['goal_distances']):.4f}m")
        print(f"Avg tool-axis direction error: {np.mean(stats['tool_axis_errors']):.4f}rad")
        print(f"Avg solve time: {np.mean(stats['solve_times']):.3f}ms")
        
        return None, stats  # Return None for trajectory (not needed)
    
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
        """World components about tool origin -> final-joint-local, linear-first.

        The returned value is an initial-state conversion. Native dynamics hold
        local components fixed; this is not full constant-world-force prediction.
        """
        _, value = world_tool_to_joint(self.solver_model, q[:self.nq_robot], f_world)
        return value
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
        velocity_threshold=1.0,
        goal_dwell_time=0.0,
        offline_timing="controller_dt",
        record_raw=False,
        record_callback=None,
    ):
        """
        Run MPC controller tracking discrete goal positions.
        
        Args:
            x_start: Initial state (robot only, no pendulum)
            goals: List of 3D goal positions np.array([x, y, z])
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
        from copy import deepcopy

        if offline_timing not in {"controller_dt", "solve_time", "solve_time_legacy"}:
            raise ValueError("offline_timing must be 'controller_dt', 'solve_time', or 'solve_time_legacy'")
        if record_callback is not None and not record_raw:
            raise ValueError("record_callback requires record_raw=True")
        if not np.isfinite(sim_dt) or sim_dt <= 0 or not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("sim_dt and dt must be finite and positive")
        if not np.isfinite(goal_timeout) or goal_timeout <= 0:
            raise ValueError("goal_timeout must be finite and positive")
        if len(goals) == 0:
            raise ValueError("at least one goal is required")
        if offline_timing == "solve_time":
            from bsqp.offline_mpc import run_mode_b
            poses = np.atleast_2d(np.asarray(goals))
            if poses.shape[1] != 3:
                raise ValueError("goals must contain three-dimensional positions")
            poses = np.hstack([poses, np.zeros((len(poses), 3))])
            return run_mode_b(self, x_start, goals=poses, reset_rho_each_solve=True,
                goal_timeout=goal_timeout, goal_threshold=goal_threshold,
                velocity_threshold=velocity_threshold, goal_dwell_time=goal_dwell_time,
                sim_dt=sim_dt, record_raw=record_raw, record_callback=record_callback,
                clock=time.perf_counter,
                integrate=lambda q, dq, u, step_dt: rk4(self.model, self.data, q, dq, u, step_dt, self.actual_f_ext))

        # Initialize statistics
        stats = {
            'timestamps': [],
            'solve_times': [],
            'goal_distances': [],
            'ee_actual': [],
            'joint_positions': [],
            'joint_velocities': [],
            'best_trajectory_id': []
        }
        
        stats.update({
            'offline_timing': offline_timing,
            'timing_mode': 'B-legacy' if offline_timing == 'solve_time_legacy' else 'A',
            'raw_execution_scope': 'simulated plant substeps',
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

        if self.track_full_stats:
            stats['sqp_iters'] = []
            stats['pcg_iters'] = []
            
        stats['goal_outcomes'] = ['not_reached'] * len(goals)
        stats['goal_reached_times'] = [None] * len(goals)
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
        current_goal = goals[current_goal_idx]
        ee_g = np.tile(np.concatenate([current_goal, np.zeros(3)]), self.N)
        ee_g_batch = np.tile(ee_g, (self.batch_size, 1))
        
        # Initialize warm start
        XU = np.zeros(self.N*(self.nx+self.nu)-self.nu)
        for i in range(self.N):
            start_idx = i * (self.nx + self.nu)
            XU[start_idx:start_idx+self.nx] = x_curr
        self.solver.reset_dual()
        XU_batch = np.tile(XU, (self.batch_size, 1))
        
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
        
        print(f"\nRunning MPC: N={self.N}, batch={self.batch_size}, {len(goals)} goals")
        if self.has_pendulum:
            print(f"Pendulum: mass={self.pendulum_config['mass']}kg, length={self.pendulum_config['length']}m")
        
        # Start timing for the current goal
        goal_start_time = total_sim_time
        goal_dwell_start_time = None
        # Main control loop
        while total_sim_time < goal_timeout * len(goals):
            interval_start = total_sim_time
            interval_goal_index = current_goal_idx
            
            # Store state for force estimation
            x_last = x_curr
            u_last = XU_best[self.nx:self.nx+self.nu]
            
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
            stats['update_intervals_s'].append(float(timestep))
            stats['requested_update_intervals_s'].append(float(requested_timestep))
            stats['interval_goal_indices'].append(interval_goal_index)
            stats['interval_plan_solve_indices'].append(plan_solve_index)
            stats['interval_timeout_clipped'].append(bool(
                offline_timing == "solve_time_legacy" and timestep < requested_timestep))

            # Check goal reached or timeout
            ee_pos = self.solver.ee_pos(q_robot)
            current_dist = np.linalg.norm(ee_pos - current_goal)
            current_vel = np.linalg.norm(dq_robot, ord=1)
            within_goal = (current_dist < goal_threshold) and (current_vel < velocity_threshold)
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
                if current_goal_idx >= len(goals):
                    break
                    
                current_goal = goals[current_goal_idx]
                ee_g = np.tile(np.concatenate([current_goal, np.zeros(3)]), self.N)
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
            
            # Collect statistics
            stats['timestamps'].append(total_sim_time)
            stats['solve_times'].append(gpu_solve_time/1000.0)  # Native us -> ms
            stats['wall_solve_times_s'].append(float(wall_solve_time))
            stats['native_solve_times_ms'].append(float(gpu_solve_time) / 1000.0)
            stats['goal_distances'].append(current_dist)
            stats['ee_actual'].append(ee_pos.copy())
            stats['joint_positions'].append(q_robot.copy())
            stats['joint_velocities'].append(dq_robot.copy())
            stats['best_trajectory_id'].append(best_id)
            
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
        
        stats['terminal_state'] = x_curr.copy()
        stats['terminal_time_s'] = float(total_sim_time)

        # Convert to numpy arrays
        for key in stats:
            if isinstance(stats[key], list) and len(stats[key]) > 0:
                if key not in ['raw_solves', 'raw_execution', 'goal_outcomes', 'goal_reached_times', 'time_to_all_reached']:
                    try:
                        stats[key] = np.array(stats[key])
                    except (ValueError, TypeError):
                        # Keep as list if conversion fails (e.g., inhomogeneous shapes)
                        pass
        
        # Compute time to all-goals-reached if all were reached
        if all([o == 'reached' for o in stats['goal_outcomes']]):
            reached_times = [t for t in stats['goal_reached_times'] if t is not None]
            if len(reached_times) == len(goals):
                stats['time_to_all_reached'] = float(np.max(reached_times))
        
        # Print summary
        goals_reached = sum(1 for o in stats['goal_outcomes'] if o == 'reached')
        print(f"Goals reached: {goals_reached}/{len(goals)}")
        if len(stats['solve_times']) > 0:
            print(f"Avg solve time: {np.mean(stats['solve_times']):.3f}ms")
        
        return None, stats  # Return None for trajectory (not needed)
