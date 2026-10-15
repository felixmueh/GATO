"""Controller-only offline clock and raw recording; never constructs a solver.

Current mode B replays the active control plan during each measured solve call,
then activates the result on arrival. Optimizer reference time and control
activation time are deliberately distinct. Legacy A/B loops live in their
original controllers for historical replays.
"""
from copy import deepcopy
import numpy as np

from .common import sample_packed_trajectory_control, sample_reference_horizon, shift_packed_trajectory_warm_start


class OfflineRecorder:
    """Raw observer bookkeeping outside the measured solver call."""
    def __init__(self, owner, stats, timing, record_raw, callback, clock):
        self.owner, self.stats, self.raw, self.callback, self.clock = owner, stats, record_raw, callback, clock
        if callback is not None and not record_raw:
            raise ValueError('record_callback requires record_raw=True')
        stats.update(offline_timing=timing,
                     timing_mode={'controller_dt': 'A', 'solve_time': 'B', 'solve_time_legacy': 'B-legacy'}[timing],
                     timing_scope='perf_counter wall seconds around solver.solve only; native microseconds converted to milliseconds',
                     nominal_feedback_s=float(owner.dt),
                     raw_execution_scope='simulated plant substeps; live applied controls are unobservable here',
                     wall_solve_times_s=[], native_solve_times_ms=[], update_intervals_s=[],
                     requested_update_intervals_s=[], interval_start_times_s=[], interval_goal_indices=[],
                     interval_plan_solve_indices=[], interval_timeout_clipped=[], feedback_timestamps_s=[],
                     control_activation_timestamps_s=[], plan_activation_status=[])
        if self.raw:
            stats.update(raw_solves=[], raw_execution=[])
        owner.last_run_stats = stats
        self.solve_count = 0

    def emit(self, event, record):
        if self.callback is not None:
            try:
                self.callback(event, record)
            except BaseException as exc:
                self.stats['recording_failure'] = dict(event=event, error_type=type(exc).__name__, error=str(exc))
                raise

    def solve(self, state, reference, warm, time_s, goal_index, shift_s, category, optimizer_origin_s):
        raw = dict(solve_index=self.solve_count, category=category, time_s=float(time_s),
                   feedback_time_s=float(time_s), optimizer_reference_time_s=float(time_s),
                   previous_optimizer_reference_time_s=float(optimizer_origin_s),
                   goal_index=int(goal_index), warm_start_shift_s=float(shift_s), status='started',
                   activation_status='pending', activation_time_s=None)
        if self.raw:
            raw.update(state=state.copy(), reference=reference.copy(), warm_start=warm.copy())
            self.stats['raw_solves'].append(raw)
            self.emit('solve_started', raw)
        started = self.clock()
        try:
            output, native_us = self.owner.solver.solve(state, reference, warm)
        except BaseException as exc:
            raw.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                       wall_solve_time_s=float(self.clock() - started))
            if self.raw:
                self.emit('solve_failed', raw)
            raise
        wall_s = self.clock() - started
        raw.update(status='completed', wall_solve_time_s=float(wall_s), native_solve_time_ms=float(native_us)/1000.,
                   solve_arrival_time_s=float(time_s + wall_s) if self.stats['offline_timing'] == 'solve_time' else float(time_s))
        if self.raw:
            raw['output'] = output.copy()
            if self.owner.track_full_stats:
                raw['solver_stats'] = deepcopy(self.owner.solver.get_stats())
            self.emit('solve_completed', raw)
        self.solve_count += 1
        return output, native_us, wall_s, raw

    def select(self, raw, lane):
        raw['best_trajectory_id'] = int(lane)
        if self.raw:
            self.emit('solve_selected', raw)

    def activation(self, raw, time_s, status='activated'):
        raw.update(activation_status=status, activation_time_s=None if time_s is None else float(time_s))
        if self.raw:
            self.emit('plan_activated' if status == 'activated' else 'plan_discarded', raw)

    def step(self, integrate, q, dq, control, dt, time_s, plan_index, goal_index, plan_age_s):
        raw = dict(time_s=float(time_s), step_dt_s=float(dt), plan_solve_index=int(plan_index),
                   goal_index=int(goal_index), active_control_age_s=float(plan_age_s), status='started')
        if self.raw:
            raw.update(control=control.copy(), state_before=np.concatenate([q, dq]).copy())
            self.stats['raw_execution'].append(raw)
            self.emit('step_started', raw)
        try:
            q, dq = integrate(q, dq, control, dt)
        except BaseException as exc:
            if self.raw:
                raw.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                self.emit('step_failed', raw)
            raise
        self.stats.update(terminal_state=np.concatenate([q[:self.owner.nq_robot], dq[:self.owner.nv_robot]]).copy(),
                          terminal_time_s=float(time_s + dt))
        if self.raw:
            raw.update(status='completed', state_after=np.concatenate([q, dq]).copy())
            self.emit('step_completed', raw)
        return q, dq

    def interval(self, start, actual, requested, plan_index, goal_index, clipped, activation_s):
        s = self.stats
        for key, value in (('interval_start_times_s', start), ('update_intervals_s', actual),
                           ('requested_update_intervals_s', requested), ('interval_goal_indices', goal_index),
                           ('interval_plan_solve_indices', plan_index), ('interval_timeout_clipped', clipped)):
            s[key].append(value)
        if self.raw:
            self.emit('interval_completed', dict(status='completed', time_s=float(start), update_interval_s=float(actual),
                requested_interval_s=float(requested), plan_solve_index=int(plan_index), goal_index=int(goal_index),
                timeout_clipped=bool(clipped), active_plan_activation_time_s=float(activation_s)))


def finish_arrays(stats):
    exclusions = {'raw_solves', 'raw_execution', 'goal_outcomes', 'goal_reached_times', 'final_approach_switches'}
    for key, value in stats.items():
        if key not in exclusions and isinstance(value, list) and value:
            try:
                stats[key] = np.asarray(value)
            except (ValueError, TypeError):
                pass


def run_mode_b(owner, x_start, *, integrate, clock, sim_dt, record_raw, record_callback,
               goals=None, reference=None, sim_time=None, goal_timeout=None, goal_threshold=.05,
               goal_axis_threshold=np.inf, check_axis=False, velocity_threshold=1., goal_dwell_time=0.,
               initial_guess=None, reset_rho_each_solve=False):
    """Elapsed solve execution with active-plan age independent of seed age.

    Bootstrap is activated at t=0 and executed for one nominal dt. At every
    feedback point, the seed shifts from the previous optimizer's reference
    origin to the measured state time. The OLD active plan plays during the
    just-measured call. New control u0 starts only when that solve arrives.
    """
    tracking = reference is not None
    for name, value in (('sim_dt', sim_dt), ('dt', owner.dt),
                        ('duration', sim_time if tracking else goal_timeout)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if owner.N < 2:
        raise ValueError('at least two knots are required')
    if tracking:
        reference = np.asarray(reference)
        if reference.ndim != 1 or reference.size % 6 or reference.size < owner.N*6:
            raise ValueError('reference must contain at least N flattened six-dimensional samples')
        end_time = min(float(sim_time), (reference.size//6 - owner.N) * owner.dt)
    else:
        goals = np.atleast_2d(np.asarray(goals))
        if not len(goals) or goals.shape[1] != 6:
            raise ValueError('at least one six-dimensional goal is required')
    stats = {key: [] for key in ('timestamps', 'solve_times', 'goal_distances', 'tool_axis_errors',
        'ee_actual', 'joint_positions', 'joint_velocities', 'best_trajectory_id', 'applied_controls',
        'planned_controls', 'goal_indices', 'planned_goal_indices', 'planned_final_approach_active',
        'final_approach_switches', 'sqp_iters', 'pcg_iters', 'pcg_times_us', 'kkt_converged')}
    recorder = OfflineRecorder(owner, stats, 'solve_time', record_raw, record_callback, clock)
    stats['timing_convention'] = 'old active plan executes during measured call; returned plan starts at arrival'
    if not tracking:
        stats.update(goal_outcomes=['not_reached']*len(goals), goal_reached_times=[None]*len(goals), time_to_all_reached=None,
                     goal_start_times_s=[None]*len(goals), goal_requested_deadline_times_s=[None]*len(goals),
                     goal_end_times_s=[None]*len(goals), goal_timeout_lateness_s=[None]*len(goals))
        stats['goal_start_times_s'][0] = 0.
        stats['goal_requested_deadline_times_s'][0] = float(goal_timeout)
    final = getattr(owner, 'final_approach', None)
    if final is not None:
        final.reset(owner.solver)
    x_start = np.asarray(x_start)
    if owner.has_pendulum:
        augmented = np.zeros(owner.nq + owner.nv)
        augmented[:owner.nx] = x_start
        augmented[owner.nq_robot:owner.nq_robot+3] = owner.pendulum_config.get('initial_angle', np.array([.3, 0., 0.]))
        q, dq = augmented[:owner.nq], augmented[owner.nq:]
    else:
        q, dq = x_start[:owner.nq].copy(), x_start[owner.nq:].copy()
    time_s, goal_index, goal_start, dwell_start = 0., 0, 0., None
    x_curr = x_start.copy()

    def robot_state():
        return np.concatenate([q[:owner.nq_robot], dq[:owner.nv_robot]])

    def target():
        return (sample_reference_horizon(reference, time_s, owner.dt, owner.N).astype(np.float32)
                if tracking else np.tile(goals[goal_index], owner.N))

    def update_final():
        if final is not None and not tracking:
            event = final.update(owner.solver, time_s=time_s, goal_index=goal_index,
                distance_m=float(np.linalg.norm(owner.solver.ee_pos(q[:owner.nq_robot]) - goals[goal_index, :3])),
                joint_velocities=dq[:owner.nv_robot])
            if event is not None:
                stats['final_approach_switches'].append(event)

    def check_task():
        nonlocal goal_index, goal_start, dwell_start
        if tracking:
            return time_s >= end_time - 1e-12
        pos = owner.solver.ee_pos(q[:owner.nq_robot])
        distance = np.linalg.norm(pos - goals[goal_index, :3])
        axis = owner.solver.ee_tool_axis_error(q[:owner.nq_robot], goals[goal_index, 3:]) if check_axis else 0.
        within = distance < goal_threshold and np.linalg.norm(dq[:owner.nv_robot], ord=1) < velocity_threshold and axis < goal_axis_threshold
        if within:
            if dwell_start is None:
                dwell_start = time_s
            reached = time_s - dwell_start >= goal_dwell_time
        else:
            dwell_start, reached = None, False
        timeout = time_s - goal_start >= goal_timeout - 1e-12
        if reached or timeout:
            deadline = goal_start + goal_timeout
            stats['goal_end_times_s'][goal_index] = float(time_s)
            stats['goal_timeout_lateness_s'][goal_index] = float(max(0., time_s - deadline))
            if record_raw:
                recorder.emit('goal_finished', dict(status='completed', goal_index=int(goal_index),
                    time_s=float(time_s), requested_deadline_s=float(deadline),
                    timeout_lateness_s=float(max(0., time_s-deadline)), outcome='reached' if reached else 'timeout'))
            stats['goal_outcomes'][goal_index] = 'reached' if reached else 'timeout'
            if reached:
                stats['goal_reached_times'][goal_index] = float(time_s)
            goal_index += 1
            if goal_index == len(goals):
                return True
            goal_start, dwell_start = time_s, None
            stats['goal_start_times_s'][goal_index] = float(goal_start)
            stats['goal_requested_deadline_times_s'][goal_index] = float(goal_start + goal_timeout)
            if final is not None:
                final.reset(owner.solver)
            owner.solver.reset_rho()
        return False

    warm = np.zeros((owner.batch_size, owner.N*(owner.nx+owner.nu)-owner.nu))
    for knot in range(owner.N):
        warm[:, knot*(owner.nx+owner.nu):knot*(owner.nx+owner.nu)+owner.nx] = x_curr
    if initial_guess is not None:
        warm = np.broadcast_to(initial_guess, warm.shape).copy()
        warm[:, :owner.nx] = x_curr
    owner.solver.reset_dual()
    if not tracking and final is not None:
        owner.solver.reset_rho()
    update_final()
    owner.update_force_batch(q[:owner.nq_robot])
    output, native, wall, bootstrap = recorder.solve(np.tile(x_curr, (owner.batch_size, 1)),
        np.tile(target(), (owner.batch_size, 1)), warm, 0., 0, 0., 'startup/cold', 0.)
    stats.update(startup_wall_solve_time_s=float(wall), startup_native_solve_time_ms=float(native)/1000.,
                 initial_state=x_curr.copy(), initial_planned_control=output[0, owner.nx:owner.nx+owner.nu].copy())
    active_plan, active_index, activation_time = output[0].copy(), 0, 0.
    # Selected previous optimizer solution has a solve-start reference origin,
    # independently of the eventual active-control origin.
    warm = output.copy()
    optimizer_time = 0.
    recorder.select(bootstrap, 0)
    recorder.activation(bootstrap, 0.)

    def execute(requested):
        nonlocal q, dq, time_s, x_curr
        start = time_s
        # A nonterminal goal's nominal timeout never cancels an in-flight
        # native call. Pay all of its measured computation before checking the
        # goal and starting another solve. Only the overall task's final goal
        # or tracking end may truncate the offline episode.
        remaining = (end_time-time_s if tracking else
                     goal_timeout-(time_s-goal_start) if goal_index == len(goals)-1 else np.inf)
        actual = min(float(requested), max(0., remaining))
        goal_during_interval = goal_index
        end = start + actual
        while time_s < end - 1e-12:
            age = max(0., time_s-activation_time)
            step_dt = min(sim_dt, end-time_s)
            knot = int(np.floor(age / owner.dt + 1e-10))
            if knot < owner.N-2:
                step_dt = min(step_dt, activation_time+(knot+1)*owner.dt-time_s)
            control = sample_packed_trajectory_control(active_plan, owner.nx, owner.nu,
                                                       owner.N, age+1e-12, owner.dt)
            if owner.has_pendulum:
                augmented_control = np.zeros(owner.nv)
                augmented_control[:owner.nu] = control
                augmented_control[owner.nu:] = -owner.pendulum_config.get('damping', .4)*dq[owner.nv_robot:]
            else:
                augmented_control = control
            q, dq = recorder.step(integrate, q, dq, augmented_control, step_dt, time_s,
                                  active_index, goal_during_interval, age)
            time_s += step_dt
        time_s = end
        x_curr = robot_state()
        recorder.interval(start, actual, float(requested), active_index, goal_during_interval,
                          actual < requested, activation_time)
        stats.update(terminal_state=x_curr.copy(), terminal_time_s=float(time_s))
        return actual, actual < requested

    # First execution remains nominal; bootstrap computation is uncharged.
    measured_last = x_curr.copy()
    measured_control = active_plan[owner.nx:owner.nx+owner.nu].copy()
    selection_elapsed, clipped = execute(owner.dt)
    done = check_task()
    previous_goal = 0
    while not done:
        update_final()
        feedback_time, planning_goal = time_s, goal_index
        feedback_state = x_curr.copy()
        goal_reference = target()
        shift_s = feedback_time - optimizer_time
        warm = shift_packed_trajectory_warm_start(warm, feedback_state, owner.nx, owner.nu,
                                                owner.N, shift_s, owner.dt)
        warm[:, :owner.nx] = feedback_state
        owner.update_force_batch(q[:owner.nq_robot])
        if tracking or reset_rho_each_solve:
            owner.solver.reset_rho()
        # Preserve selection's measured transition: use observations at feedback
        # start, never silently substitute post-computation plant state.
        output, native, wall, raw = recorder.solve(np.tile(feedback_state, (owner.batch_size, 1)),
            np.tile(goal_reference, (owner.batch_size, 1)), warm, feedback_time, planning_goal, shift_s,
            'goal_switch' if planning_goal != previous_goal else 'warm', optimizer_time)
        lane = owner.evaluate_best_trajectory(measured_last, measured_control, feedback_state, selection_elapsed)
        recorder.select(raw, lane)
        warm = np.broadcast_to(output[lane], output.shape).copy()
        optimizer_time = feedback_time
        measured_last = feedback_state
        measured_control = sample_packed_trajectory_control(active_plan, owner.nx, owner.nu, owner.N,
                                                           feedback_time-activation_time+1e-12, owner.dt).copy()
        # Record measurements at SOLVE START; arrival/activation gets its own
        # clock below, so stale initial state is never mislabeled as live state.
        q_feedback, dq_feedback = feedback_state[:owner.nq_robot], feedback_state[owner.nq_robot:]
        pos = owner.solver.ee_pos(q_feedback)
        ref_pose = goal_reference[:6]
        stats['timestamps'].append(float(feedback_time))
        stats['feedback_timestamps_s'].append(float(feedback_time))
        stats['solve_times'].append(float(native)/1000.)
        stats['native_solve_times_ms'].append(float(native)/1000.)
        stats['wall_solve_times_s'].append(float(wall))
        stats['goal_distances'].append(float(np.linalg.norm(pos-ref_pose[:3])))
        stats['tool_axis_errors'].append(float(owner.solver.ee_tool_axis_error(q_feedback, ref_pose[3:]))
                                        if hasattr(owner.solver, 'ee_tool_axis_error') else np.nan)
        stats['ee_actual'].append(pos.copy())
        stats['joint_positions'].append(q_feedback.copy())
        stats['joint_velocities'].append(dq_feedback.copy())
        stats['best_trajectory_id'].append(int(lane))
        stats['applied_controls'].append(measured_control.copy())
        stats['planned_controls'].append(output[lane, owner.nx:owner.nx+owner.nu].copy())
        stats['goal_indices'].append(int(planning_goal))
        stats['planned_goal_indices'].append(int(planning_goal))
        stats['planned_final_approach_active'].append(final.active if final is not None else False)
        if owner.track_full_stats:
            native_stats = raw.get('solver_stats') if record_raw else owner.solver.get_stats()
            sqp = np.asarray(native_stats.get('sqp_iters', [])).reshape(-1)
            stats['sqp_iters'].append(int(sqp[lane]) if sqp.size else 0)
            stats['pcg_iters'].append(np.asarray(native_stats.get('pcg_iters', [])).copy())
            stats['pcg_times_us'].append(np.asarray(native_stats.get('pcg_times_us', [])).copy())
            stats['kkt_converged'].append(np.asarray(native_stats.get('kkt_converged', [])).copy())
        selection_elapsed, clipped = execute(wall)
        done = check_task()
        previous_goal = planning_goal
        if done or clipped or goal_index != planning_goal:
            status = 'discarded_task_terminal' if done else 'discarded_goal_transition'
            recorder.activation(raw, None, status)
            stats['control_activation_timestamps_s'].append(None)
            stats['plan_activation_status'].append(status)
        else:
            active_plan, active_index, activation_time = output[lane].copy(), raw['solve_index'], time_s
            recorder.activation(raw, time_s)
            stats['control_activation_timestamps_s'].append(float(time_s))
            stats['plan_activation_status'].append('activated')
    stats.update(terminal_state=x_curr.copy(), terminal_time_s=float(time_s))
    if not tracking and all(o == 'reached' for o in stats['goal_outcomes']):
        stats['time_to_all_reached'] = float(max(stats['goal_reached_times']))
    finish_arrays(stats)
    return None, stats
