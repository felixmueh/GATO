"""Functional ModeB simulation with an assumed independent execution safety loop.
Every actual segment is filtered, including stale plans during charged solves.
Filter CPU wall is recorded separately; actual concurrent scheduling is unproved.
"""
import json
import hashlib
import math
import time
import traceback
import importlib.util
from pathlib import Path

_filter_spec=importlib.util.spec_from_file_location('execution_torque_filter',Path(__file__).with_name('filter.py'))
_filter_module=importlib.util.module_from_spec(_filter_spec)
_filter_spec.loader.exec_module(_filter_module)

class FilterStop(RuntimeError):
    """No accepted forecast; incomplete simulation, not infeasibility proof."""

def run_episode_filtered(api, runtime, args, protocol, pair):
    sha, write, safe = api.sha, api.write, api.safe
    task, metrics, arrival = api.task, api.metrics, api.arrival
    unpack, pose, limit_checks, work_summary = api.unpack, api.pose, api.limit_checks, api.work_summary
    MODEL = api.MODEL
    np, pin, MPC, common = runtime
    out = args.output / ("preflight" if pair == -1 else f"pair_{pair:05d}")
    out.mkdir()
    cfg = protocol["solver_params"]
    model = pin.buildModelFromUrdf(str(args.source / MODEL))
    model.gravity.linear = np.array([0., 0., -9.81])
    data = model.createData()
    x, witness, goal, reference, indices = task(np, pin, model, data, args, pair)
    states, controls, times, control_times = [x.copy()], [], [0.], []
    rows, observations = [], [metrics(np, pin, model, data, x, goal)]
    reached = position_touch = pose_touch = None
    failure = None
    failure_phase = None
    failure_class = None
    filter_reports=[]
    proposed_controls=[]
    filter_impl=None
    interrupted = False
    phase = "controller_setup"
    started = time.perf_counter()
    warm = common.initialize_warm_start(x, 32, 14, 7)[None].astype(np.float32)
    np.savez_compressed(out / "inputs.npz", initial_state=x, source_goal_q=witness,
                        target_xyz=goal.translation, target_rotation=goal.rotation,
                        reference=reference, first_seed=warm)
    write(out / "preparation.json", dict(pair_id=pair, **indices, protocol_id=protocol["id"],
          selected_input_sha256=sha(out / "inputs.npz"), initial_state_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
          goal_pose_sha256=hashlib.sha256(np.r_[goal.translation, goal.rotation.ravel()].tobytes()).hexdigest(),
          initial=observations[0], solver_params=cfg))
    try:
        controller = MPC(model, str(args.source / MODEL), N=32, dt=.03, batch_size=1,
                         plant_type="tiago_right", solver_params=cfg, track_full_stats=True)
        controller.force_estimator = None
        controller.solver.set_f_ext_B(np.zeros((1, 6), dtype=np.float32))
        controller.solver.reset_dual()
        controller.solver.reset_rho()
        try:
            external_force_values=np.asarray([force.vector for force in controller.actual_f_ext],dtype=float)
        except AttributeError:
            external_force_values=np.asarray(controller.actual_f_ext,dtype=float)
        if not np.isfinite(external_force_values).all() or np.any(external_force_values):
            raise ValueError('This filter simulation assumes actual external forces are zero')
        filter_impl=_filter_module.PredictiveLimitFilter(model,common.rk4,settings=protocol['predictive_filter']['settings'])
        phase = "episode"
        step = 0
        active_plan, active_step, activation_time = None, None, None
        optimizer_time = 0.
        seed, optimized, stats = warm.copy(), None, {}
        with (out / "solves.jsonl").open("w") as log:
            while times[-1] < protocol['timeout_s'] - 1e-10:
                now = times[-1]
                shift = 0. if step == 0 else now - optimizer_time
                if step:
                    warm = common.shift_packed_trajectory_warm_start(warm[0], x, 14, 7, 32, shift, .03)[None]
                seed = warm.copy()
                sx, su = unpack(np, seed[0])
                grid = now + np.arange(32) * .03
                seed_limits = limit_checks(np, model, sx, su, grid, grid[:-1])
                if getattr(args, 'save_plans', False):
                    np.savez_compressed(out / f"step_{step:04d}_input.npz", actual_state=x,
                                        candidates=seed, reference=reference)
                # Only this wrapper call is charged. Stats, IO and RK4 are outside.
                optimized, stats, native_us, wall = None, {}, None, None
                attempt_state = x.copy()
                phase = "solve"
                solve_started = time.perf_counter()
                try:
                    optimized, native_us = controller.solver.solve(x[None], reference, warm.copy())
                finally:
                    wall = time.perf_counter() - solve_started
                phase = 'stats'
                stats = controller.solver.get_stats()
                phase = 'recording'
                arrays = {name: np.asarray(stats[name]) for name in
                          ("pcg_iters", "sqp_iters", "step_size", "min_merit", "kkt_converged", "initial_merit", "final_merit")}
                if getattr(args, 'save_plans', False):
                    np.savez_compressed(out / f"step_{step:04d}_returned.npz", optimized=optimized,
                                        native_solve_us=native_us, solve_wall_s=wall, **arrays)
                predicted_x, predicted_u = unpack(np, optimized[0])
                bootstrap = step == 0
                charged = 0. if bootstrap else wall
                row = dict(step=step, solve_returned=True, work_available=True, feedback_time_s=now, feedback_state=x.copy(),
                           timing_category='startup/cold' if bootstrap else 'warm',
                           optimizer_reference_time_s=now, previous_optimizer_origin_s=optimizer_time,
                           warm_shift_s=shift, native_solve_us=float(native_us), solve_wall_s=wall,
                           completed_solve_returns=sum(r.get('solve_returned',True) for r in rows),
                      work_complete=all(r.get('work_available',True) for r in rows),
                      work_totals_scope='Observed returned stats only; unavailable failed-call work remains null in solve rows',
                      charged_update_count=sum(r.get('actual_charged_elapsed_s',0)>0 for r in rows),
                      charged_solve_s=charged, actual_charged_elapsed_s=0.,
                           seed_limits=seed_limits,
                           predicted_limits=limit_checks(np, model, predicted_x, predicted_u, grid, grid[:-1]),
                           predicted_first_future_state=predicted_x[1].copy(),
                           predicted_first_future_maximum_velocity_ratio=float(np.max(np.abs(predicted_x[1,7:])/model.velocityLimit)),
                           accepted_step_sizes=arrays['step_size'].copy(),
                           work=work_summary(np, stats, cfg['max_sqp_iters']))
                row['work']['cap_hit'] = row['work']['actual_sqp'] >= cfg['max_sqp_iters']
                if not np.isfinite(optimized).all() or not math.isfinite(row['work']['final_merit']) or not math.isfinite(wall) or wall <= 0:
                    row.update(executed_duration_s=0., returned_plan_status='rejected_numerical', control_activation_time_s=None)
                    rows.append(row); log.write(json.dumps(safe(row), allow_nan=False)+'\n'); log.flush()
                    raise RuntimeError('No finite optimized plan/merit/positive wrapper duration; invalid output retained')
                if bootstrap:
                    active_plan, active_step, activation_time = optimized[0].copy(), 0, 0.
                duration = min(.03 if bootstrap else charged, protocol['timeout_s'] - now)
                row.update(executed_duration_s=duration,
                           actual_charged_elapsed_s=0. if bootstrap else duration,
                           charged_interval_clipped=not bootstrap and duration < charged,
                           executed_prefix_plan_step=active_step,
                           executed_plan_activation_time_s=activation_time)
                dense, prefix_times, prefix_controls, prefix_control_times = [x.copy()], [now], [], []
                end = now + duration
                phase = "plant"
                while times[-1] < end - 1e-12:
                    age = max(0., times[-1]-activation_time)
                    knot = min(30, int(math.floor(age/.03+1e-10)))
                    h = min(.003, end-times[-1])
                    if knot < 30:
                        h = min(h, activation_time+(knot+1)*.03-times[-1])
                    u = np.asarray(active_plan[knot*21+14:knot*21+21], dtype=float)
                    phase='filter'
                    decision=filter_impl.filter(x,u)
                    decision.update(time_s=times[-1],requested_execution_step_s=h,active_plan_step=active_step,
                                    active_plan_knot=knot,filter_index=len(filter_reports))
                    filter_reports.append(decision)
                    with (out/'filter_samples.jsonl').open('a') as filter_log:
                        filter_log.write(json.dumps(safe(decision),allow_nan=False)+'\n')
                    if decision['selected_torque'] is None:
                        row.update(executed_duration_s=times[-1]-now,
                                   actual_charged_elapsed_s=0. if bootstrap else times[-1]-now,
                                   returned_plan_status='filter_stopped_incomplete',control_activation_time_s=None,
                                   executed_prefix_limits=limit_checks(np,model,dense,prefix_controls,prefix_times,prefix_control_times),
                                   final=observations[-1])
                        rows.append(row);log.write(json.dumps(safe(row),allow_nan=False)+'\n');log.flush()
                        raise FilterStop(decision['status'])
                    proposed_controls.append(u.copy())
                    u=np.asarray(decision['selected_torque'],dtype=float)
                    phase='plant'
                    q, v = common.rk4(model, controller.data, x[:7], x[7:], u, h, controller.actual_f_ext)
                    x = np.r_[q, v]
                    control_times.append(times[-1]); controls.append(u.copy())
                    prefix_control_times.append(times[-1]); prefix_controls.append(u.copy())
                    times.append(times[-1]+h); states.append(x.copy())
                    observations.append(metrics(np, pin, model, data, x, goal))
                    observation = observations[-1]
                    position = observation['position_error_m'] <= .05
                    touching = position and observation['tool_axis_error_deg'] <= 30.
                    if position and position_touch is None: position_touch = times[-1]
                    if touching and pose_touch is None: pose_touch = times[-1]
                    dense.append(x.copy()); prefix_times.append(times[-1])
                    if not np.isfinite(x).all(): break
                row['executed_prefix_limits'] = limit_checks(np, model, dense, prefix_controls, prefix_times, prefix_control_times)
                row['final'] = observations[-1]
                terminal = times[-1] >= protocol['timeout_s']-1e-10 or arrival(observations[-1], protocol)
                row['returned_plan_status'] = 'bootstrap_activated' if bootstrap else 'discarded_task_terminal' if terminal else 'activated'
                row['control_activation_time_s'] = 0. if bootstrap else None if terminal else times[-1]
                rows.append(row); log.write(json.dumps(safe(row), allow_nan=False)+'\n'); log.flush()
                if not np.isfinite(x).all(): raise RuntimeError('Nonfinite executed plant state retained')
                if arrival(observations[-1], protocol):
                    reached = times[-1]; break
                warm = optimized.copy()
                optimizer_time = now
                if not bootstrap and not terminal:
                    active_plan, active_step, activation_time = optimized[0].copy(), step, times[-1]
                step += 1
    except BaseException as error:
        failure = traceback.format_exc()
        # Full attempted input/output retained on failure even in compact mode.
        if phase != 'controller_setup':
            payload = dict(actual_state=attempt_state, seed_first_knot=seed[0,:14], plant_state_after=x, reference=reference, seed=seed)
            if optimized is not None: payload['optimized'] = optimized
            payload.update({'stats_'+k: np.asarray(v) for k,v in stats.items() if k in
                            ('pcg_iters','sqp_iters','step_size','min_merit','kkt_converged','initial_merit','final_merit')})
            np.savez_compressed(out/'failed_solve.npz', **payload)
            if not rows or rows[-1]['step'] != step:
                available = dict(actual_sqp=None,pcg_iteration_sum=None,zero_pcg_outer_exit=False,
                                 cap_hit=None,accepted_steps=None,line_search_count=None,stop_reason='call_failed_before_work_available')
                failed_row=dict(step=step,feedback_time_s=now,feedback_state=attempt_state,
                    optimizer_reference_time_s=now,previous_optimizer_origin_s=optimizer_time,warm_shift_s=shift,
                    timing_category='startup/cold' if step==0 else 'warm',solve_returned=optimized is not None,
                    work_available=False,native_solve_us=float(native_us) if native_us is not None else None,
                    solve_wall_s=wall,charged_solve_s=0. if step==0 else wall,
                    actual_charged_elapsed_s=0. if step==0 else times[-1]-now,
                    executed_duration_s=times[-1]-now,returned_plan_status='call_failed',
                    control_activation_time_s=None,failure_phase=phase,work=available)
                rows.append(failed_row)
                with (out/'solves.jsonl').open('a') as failure_log:
                    failure_log.write(json.dumps(safe(failed_row),allow_nan=False)+'\n')
        numerical = isinstance(error, (FloatingPointError, np.linalg.LinAlgError)) or any(marker in str(error) for marker in ('No finite optimized plan', 'Nonfinite executed plant state'))
        failure_class = 'filter_stop' if isinstance(error,FilterStop) else 'numerical' if numerical else 'infrastructure'
        failure_phase = phase
        interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
        write(out / "execution_failure.json", dict(error=failure, failure_phase=phase, interrupted=interrupted,
              simulated_seconds=times[-1], completed_solves=len(rows)))
    finally:
        obs = {key: np.asarray([row[key] for row in observations]) for key in observations[0]}
        np.savez_compressed(out / "executed.npz", proposed_controls=np.asarray(proposed_controls).reshape(-1,7), states=np.asarray(states), controls=np.asarray(controls).reshape(-1, 7),
                            times=np.asarray(times), control_times=np.asarray(control_times), **obs)
        checks = limit_checks(np, model, states, controls, times, control_times)
        indices_last = np.flatnonzero(np.asarray(times) >= max(0., times[-1]-1.))
        last_q = np.asarray(states)[indices_last, :7]
        errors = obs["position_error_m"][indices_last]
        hover = dict(observation_seconds=float(times[-1]-times[indices_last[0]]),
                     position_error_min_m=float(np.min(errors)), position_error_max_m=float(np.max(errors)),
                     net_position_error_reduction_m=float(errors[0]-errors[-1]),
                     joint_total_variation_L1_rad=float(np.abs(np.diff(last_q, axis=0)).sum()),
                     maximum_joint_speed_rad_s=float(np.max(np.abs(np.asarray(states)[indices_last, 7:]))),
                     maximum_speed_L1_rad_s=float(np.max(obs["velocity_l1_rad_s"][indices_last])),
                     limitation="Observation ends at rough arrival or timeout; no post-arrival settling inference")
        initial_xyz = pose(pin, model, data, np.asarray(states[0][:7], dtype=float)).translation
        final_xyz = pose(pin, model, data, np.asarray(states[-1][:7], dtype=float)).translation if np.isfinite(states[-1]).all() else np.full(3, np.nan)
        result = dict(pair_id=pair, outcome="execution_failed" if failure else "reached" if reached is not None else "timeout",
                      complete=failure_class!='filter_stop', task_success=reached is not None and failure is None,
                      physical_pass=bool(checks["physical_pass"] and failure is None), execution_error=failure,
                      prefix_physical_pass=checks["physical_pass"],
                      failure_phase=failure_phase, shared_setup_failure=bool(failure and failure_class == "infrastructure"),
                      failure_class=failure_class,
                      interrupted=interrupted,
                      simulated_seconds=times[-1], reached_seconds=reached,
                      arrival_policy=protocol['arrival_policy'], arrival_orientation_gate=False,
                      orientation_required_for_arrival=False,
                      tool_axis_tolerance_role='30degrees used only for descriptive first_pose_touch_seconds',
                      first_position_touch_seconds=position_touch, first_pose_touch_seconds=pose_touch,
                      touch_sampling="dense plant steps <=3ms; rough arrival checked only after first nominal prefix or charged solve interval",
                      initial=observations[0], final=observations[-1], physical=checks, last_second=hover,
                      executed_endpoint_displacement_m=float(np.linalg.norm(final_xyz-initial_xyz)),
                      feedback_solves=len(rows), actual_sqp_sum=sum(r["work"]["actual_sqp"] or 0 for r in rows),
                      actual_pcg_iteration_sum=sum(r["work"]["pcg_iteration_sum"] or 0 for r in rows),
                      zero_pcg_exit_solves=sum(r["work"]["zero_pcg_outer_exit"] for r in rows),
                      solve_wall_s=sum(r["solve_wall_s"] or 0 for r in rows), native_solve_us=sum(r["native_solve_us"] or 0 for r in rows),
                      episode_wall_s=time.perf_counter()-started, timing_mode="B", offline_timing="solve_time",
                      completed_solve_returns=sum(r.get('solve_returned',True) for r in rows),
                      work_complete=all(r.get('work_available',True) for r in rows),
                      work_totals_scope='Observed returned stats only; unavailable failed-call work remains null in solve rows',
                      charged_update_count=sum(r.get('actual_charged_elapsed_s',0)>0 for r in rows),
                      charged_solve_s=sum(r['charged_solve_s'] or 0 for r in rows),
                      actual_charged_elapsed_s=sum(r['actual_charged_elapsed_s'] for r in rows),
                      bootstrap_unpaid_solve_wall_s=rows[0]['solve_wall_s'] if rows else None,
                      recording_policy='All executed states/controls, per-solve timings/work/limits; full failed solve; optional all plans',
                      save_plans=bool(getattr(args,'save_plans',False)),
                      predictive_filter=dict(settings=protocol['predictive_filter']['settings'],
                          architecture=protocol['predictive_filter']['architecture'],time_policy=protocol['predictive_filter']['time_policy'],
                          evaluation_count=len(filter_reports),modified_segments=sum(r.get('modified',False) for r in filter_reports),
                          guard_certified_segments=sum(r['guard_certified'] for r in filter_reports),
                          recovery_only_segments=sum(r['status']=='recovery_only_not_guard_certified' for r in filter_reports),
                          no_candidate_certified=sum(r['status']=='no_candidate_certified' for r in filter_reports),
                          filter_stop=failure_class=='filter_stop',
                          CPU_wall_s=sum(r['filter_wall_s'] for r in filter_reports),
                          over_nominal3ms_count=sum(r['nominal_step_overrun'] for r in filter_reports),
                          max_filter_wall_s=max((r['filter_wall_s'] for r in filter_reports),default=None),
                          p95_filter_wall_s=float(np.percentile([r['filter_wall_s'] for r in filter_reports],95)) if filter_reports else None,
                          timing_fully_paid=False,actual_concurrent_scheduling_proven=False),
                      original_J1_outcome=protocol['original_J1_labels'].get(str(pair)),
                      median_solve_wall_s=float(np.median([r['solve_wall_s'] for r in rows if r['solve_wall_s'] is not None])) if rows else None,
                      p95_solve_wall_s=float(np.percentile([r['solve_wall_s'] for r in rows if r['solve_wall_s'] is not None],95)) if rows else None)
        if pair == -1:
            old = json.loads(args.preflight_input.read_text())
            result["canonical_geometry_check"] = observations[0]["position_error_m"] >= old.get("minimum_initial_distance_m", .1)
            result["canonical_motion_check"] = result["executed_endpoint_displacement_m"] >= old.get("minimum_executed_displacement_m", .05)
            result["preflight_pass"] = bool(result["task_success"] and result["physical_pass"] and result["canonical_geometry_check"] and result["canonical_motion_check"])
        write(out / "metrics.json", result)
    print(json.dumps(safe(result), allow_nan=False), flush=True)
    return result
