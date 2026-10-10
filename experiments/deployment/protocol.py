"""Frozen D01 task selection; importable without ROS or a native module."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = Path(__file__).parent / 'inputs/t01.json'
INPUT_SHA256 = 'aec47cfa3819de5adbb8633b637f5bd9bcd327cb3afc5043a856b323af5aac41'
PREFLIGHT_ID = 'preflight-goal-1'
CASE_IDS = (PREFLIGHT_ID, 'goal-1', 'goal-2', 'goal-3',
            'figure8-24s', 'figure8-18s', 'figure8-12s')
LIVE_DEFAULTS = dict(target_hz=100.0, reset_duration_sec=2.0,
    stale_timeout_sec=0.1, max_abs_torque=26.0, clamp_torque=False,
    disable_collision_safety_for_sim_debug=False, collision_min_distance_m=0.04,
    collision_check_timeout_sec=0.05, collision_max_monitored_geometry_speed_m_s=1.0,
    joint_position_margin_rad=0.0, joint_velocity_scale=1.0,
    history_max_records=20000, start_method='spawn')
LIVE_DEFAULTS['collision_blacklist_path'] = str(ROOT / 'tiago_configs/comfortable_pose_collision_blacklist.json')


def resolve_case(case):
    if case.get('id') not in CASE_IDS or case.get('repeat', 0) != 0:
        raise ValueError('D01 has one pass of exactly the seven named cases')
    parameters = case.get('parameters', {})
    if parameters.get('backend') != 'gazebo':
        raise ValueError('Explicit backend=gazebo is required')
    unknown = set(parameters) - {'backend', 'preflight_output'}
    if unknown:
        raise ValueError('Unsupported task overrides: ' + ', '.join(sorted(unknown)))
    if hashlib.sha256(INPUT.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError('Frozen T01 input bytes changed')
    frozen = json.loads(INPUT.read_text())
    task_id = 'goal-1' if case['id'] == PREFLIGHT_ID else case['id']
    task = next(t for t in frozen['tasks'] if t['id'] == task_id)
    # The original internal clock/release metadata stays in the copied input;
    # live protocols explicitly resolve only scientifically relevant fields.
    keys = ('N', 'dt_s', 'sim_dt_s', 'initial_state', 'nominal_feedback_s',
            'position_threshold_m', 'axis_threshold_deg', 'speed_L1_threshold_rad_s',
            'goal_dwell_time_s', 'final_approach', 'batch_size', 'constant_f_ext',
            'gravity_m_s2', 'pendulum', 'initial_guess', 'seed_policy',
            'physical_validation', 'urdf', 'known_limitations')
    protocol = {key: frozen[key] for key in keys}
    protocol.update(task)
    protocol.update(case_id=case['id'], campaign_id='D01', execution_backend='gazebo',
        timing_mode='live', computation_paid=True, machine='rtx5060',
        prediction_duration_s=(frozen['N'] - 1) * frozen['dt_s'],
        input_sha256=INPUT_SHA256, solver_params=frozen[task['kind'] + '_solver_params'],
        live_controller={**LIVE_DEFAULTS, 'reset_q': frozen['initial_state'][:7]},
        controller_timeout_s=8.0, controller_startup_timeout_s=30.0,
        simulation_guard_timeout_s=8.0,
        ros_policy='unchanged public live state-stamp clock, latest trajectory queue, '
                   'runtime horizon expiry and handback',
        timing_scope='wrapper_wall_s measures only public BSQP.solve; input/output copies, '
                     'available stats and recorder IO excluded; live plant pays all elapsed time',
        sim_dt_argument_usage='0.003 s retained in public task call; unused by live ROS loops; '
                              'Gazebo physics integration is configured externally and not inferred from /clock sample spacing',
        reset_rho_each_solve=(task['kind'] == 'tracking'),
        physical_scope='observed joint positions/velocities and observed published effort '
                       'against model limits; actuator torque unavailable',
        repeat_policy='one pass per named case; preflight failure holds the family; no retry',
        release_status='proposal; no execution authorization conveyed by this protocol')
    if task['kind'] == 'tracking':
        protocol['live_controller']['target_hz'] = 125.0
    protocol['physical_validation'] = dict(protocol['physical_validation'],
                                          scope=protocol['physical_scope'])
    canonical_setup = {key: frozen[key] for key in ('N', 'dt_s', 'sim_dt_s',
        'initial_state', 'reaching_solver_params', 'position_threshold_m',
        'axis_threshold_deg', 'speed_L1_threshold_rad_s', 'goal_dwell_time_s',
        'final_approach', 'batch_size', 'initial_guess', 'urdf')}
    canonical_setup['live_controller'] = {**LIVE_DEFAULTS, 'reset_q': frozen['initial_state'][:7]}
    canonical_setup['live_controller']['collision_blacklist_path'] = 'tiago_configs/comfortable_pose_collision_blacklist.json'
    canonical_setup.update(controller_timeout_s=8., controller_startup_timeout_s=30.)
    protocol['preflight_setup_sha256'] = hashlib.sha256(
        json.dumps(canonical_setup, sort_keys=True).encode()).hexdigest()
    return protocol


def gazebo_environment_guard(environment=None):
    environment = os.environ if environment is None else environment
    if (environment.get('GATO_ROS_PROFILE') != 'simulation'
            or environment.get('ROS_DOMAIN_ID') != '1'
            or environment.get('RMW_IMPLEMENTATION') != 'rmw_cyclonedds_cpp'):
        raise RuntimeError('Source the simulation ROS profile/domain 1 before D01; '
                           'the worker never sources or changes ROS environment')
    if environment.get('GATO_MACHINE') != 'rtx5060':
        raise RuntimeError('D01 targets GATO_MACHINE=rtx5060/Gazebo')


def preflight_guard(protocol, case, module_sha256=None):
    if protocol['case_id'] == PREFLIGHT_ID:
        return
    location = case.get('parameters', {}).get('preflight_output')
    if not location:
        raise RuntimeError('A successful same-checkout D01 canonical preflight is required')
    location = Path(location).resolve()
    metrics = json.loads((location / 'metrics.json').read_text())
    if not (metrics.get('case_id') == PREFLIGHT_ID
            and metrics.get('execution_backend') == 'gazebo'
            and metrics.get('machine') == 'rtx5060'
            and metrics.get('timing_mode') == 'live'
            and metrics.get('input_sha256') == INPUT_SHA256
            and metrics.get('preflight_setup_sha256') == protocol['preflight_setup_sha256']
            and metrics.get('task_success') is True
            and metrics.get('physical_pass') is True
            and metrics.get('execution_error') is None
            and metrics.get('handback_confirmed') is True):
        raise RuntimeError('D01 canonical task/physical preflight failed; family held')
    if module_sha256 is not None:
        identity = json.loads((location / 'provenance.json').read_text())
        if identity.get('module', {}).get('sha256') != module_sha256:
            raise RuntimeError('D01 preflight module differs from loaded task module; family held')
