"""CPU-only parameter-study resolution; no native or ROS imports."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

DISABLED_STOPS = ['proposed_trajectory_torque_limit', 'joint_position_limit',
                  'joint_velocity_limit', 'collision_margin', 'collision_body_speed']


def safety_policy(backend, report_only=False, environment=None):
    environment = os.environ if environment is None else environment
    if report_only and (backend != 'gazebo'
            or environment.get('GATO_ROS_PROFILE') != 'simulation'
            or environment.get('ROS_DOMAIN_ID') != '1'
            or environment.get('RMW_IMPLEMENTATION') != 'rmw_cyclonedds_cpp'):
        raise ValueError('--dangerously-disable-safety-stops requires explicit Gazebo backend and simulation ROS environment')
    return dict(report_only_physical_stops=bool(report_only),
        explicit_cli_opt_in=bool(report_only),
        disabled_stops=DISABLED_STOPS if report_only else [],
        monitoring_retained=True, torque_clamping=False,
        retained_stops=['nonfinite_state_or_control', 'stale_state_or_trajectory',
                        'monitor_setup_or_worker_failure', 'execution_failure'],
        handback_required=True, physical_criteria_unchanged=True)


def periodic_reference(task, period, duration, N, dt):
    """Regrid the original sampled T01 period without changing its geometry/phase."""
    original = np.asarray(task['reference_packed'], dtype=task['reference_dtype']).reshape(-1, 6)
    if hashlib.sha256(original.tobytes()).hexdigest() != task['reference_sha256']:
        raise ValueError('Frozen T01 reference hash mismatch')
    base_period, base_dt = task['period_s'], task['reference_dt_s']
    endpoint = int(round(base_period / base_dt))
    if len(original) <= endpoint or abs(endpoint * base_dt - base_period) > 1e-9:
        raise ValueError('Frozen reference must contain a complete period and its endpoint')
    # Endpoint included: noninteger new period/dt never shortens the requested run.
    count = math.ceil((duration + (N - 1) * dt) / dt) + 1
    times = np.arange(count, dtype=np.float64) * dt
    phase = np.remainder(times, period) * (base_period / period)
    base_times = np.arange(endpoint + 1, dtype=np.float64) * base_dt
    reference = np.column_stack([np.interp(phase, base_times, original[:endpoint + 1, i])
                                 for i in range(6)]).astype('<f8')
    return reference.reshape(-1)


def resolve_protocol(inputs, case, backend, preflight=False, report_only=False):
    if backend not in ('internal', 'gazebo'):
        raise ValueError('Only internal or Gazebo simulation is supported')
    policy = safety_policy(backend, report_only)
    inputs = Path(inputs)
    canonical = json.loads((inputs / 'v01.json').read_text())
    cohort = json.loads((inputs / 't01.json').read_text())
    parameters = case.get('parameters', {})
    if set(parameters) - {'N', 'dt_s', 'sqp_max_iter', 'pcg_max_iter'}:
        raise ValueError('Unknown study parameter; unsafe policy cannot come from a case or plan')
    N = parameters.get('N', 32)
    dt = parameters.get('dt_s', .03)
    sqp = parameters.get('sqp_max_iter', 5)
    pcg = parameters.get('pcg_max_iter', 120)
    if type(N) is not int or N not in (16, 32, 64, 128, 256, 512):
        raise ValueError('N must be 16,32,64,128,256,512')
    if type(dt) not in (int, float) or dt not in (.01, .015, .02, .03, .045, .06):
        raise ValueError('dt_s must be a declared study knot spacing')
    if type(sqp) is not int or sqp not in (1, 2, 5) or type(pcg) is not int or pcg != 120:
        raise ValueError('Study requires SQP1/2/5 and PCG120')
    if preflight:
        N, dt, sqp = 32, .03, 5
        p = copy.deepcopy(canonical)
        p['case_id'] = 'canonical-preflight'
    else:
        task_id = case.get('id')
        if task_id not in ('goal-3', 'figure8-6s', 'figure8-12s'):
            raise ValueError('Unknown study task')
        template_id = 'figure8-12s' if task_id == 'figure8-6s' else task_id
        task = copy.deepcopy(next(t for t in cohort['tasks'] if t['id'] == template_id))
        p = {k: copy.deepcopy(v) for k, v in cohort.items() if k != 'tasks'}
        p.update(task)
        p['id'] = p['case_id'] = task_id
        p['solver_params'] = copy.deepcopy(cohort[p['kind'] + '_solver_params'])
        if p['kind'] == 'tracking':
            period = 6. if task_id == 'figure8-6s' else 12.
            reference = periodic_reference(task, period, period, N, dt)
            p.update(duration_s=period, period_s=period, reference_dt_s=dt,
                reference_packed=reference.tolist(), reference_dtype='<f8',
                reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                reference_scope='original frozen T01 sampled period, phase/start/orientation preserved; periodic linear regrid through task duration plus full preview',
                reference_coverage_s=(len(reference) // 6 - 1) * dt,
                reference_template_sha256=task['reference_sha256'],
                provisional_RMS_gate_m=.025 if period == 6. else .03,
                provisional_maximum_gate_m=.05 if period == 6. else .06)
            p['maximum_reference_speed_m_s'] = task['maximum_reference_speed_m_s'] * task['period_s'] / period
    p['solver_params'] = dict(p['solver_params'], max_sqp_iters=sqp, max_pcg_iters=120)
    p.update(N=N, dt_s=dt, nominal_feedback_s=dt,
        prediction_duration_s=(N - 1) * dt, execution_backend=backend,
        campaign_id='task-parameters-20261014',
        timing_mode='B' if backend == 'internal' else 'live',
        offline_timing='solve_time', safety_policy=policy,
        repeat_policy='frozen planned attempts; failures preserved; no automatic retries',
        release_status='prepared only; motion/build authorization separate')
    p['canonical_setup_sha256'] = hashlib.sha256(json.dumps(dict(canonical=canonical,
        backend=backend, pcg=120, safety_policy=policy), sort_keys=True).encode()).hexdigest()
    p['preflight_setup_sha256'] = p['canonical_setup_sha256']
    p['input_sha256'] = hashlib.sha256(
        (inputs / ('v01.json' if preflight else 't01.json')).read_bytes()).hexdigest()
    if backend == 'gazebo':
        p.update(controller_timeout_s=8., controller_startup_timeout_s=30.,
            simulation_guard_timeout_s=8., physical_scope='observed q/v and published effort; measured actuator torque unavailable',
            live_controller=dict(target_hz=125. if p['kind'] == 'tracking' else 100.,
                reset_q=p['initial_state'][:7], reset_duration_sec=2., stale_timeout_sec=.1,
                max_abs_torque=26., clamp_torque=False,
                disable_collision_safety_for_sim_debug=False,
                collision_min_distance_m=.04, collision_check_timeout_sec=.05,
                collision_max_monitored_geometry_speed_m_s=1.,
                joint_position_margin_rad=0., joint_velocity_scale=1.,
                history_max_records=20000, start_method='spawn',
                dangerously_disable_safety_stops=bool(report_only), execution_backend='gazebo'))
    return p
