"""Own canonical mode-A preflight, separate N32 worker, not run_v01's B diagnostic."""
from pathlib import Path
import traceback
import numpy as np

# Load only pure local plumbing until guards have accepted concrete evidence.
import importlib.util
_spec = importlib.util.spec_from_file_location('f01_release', Path(__file__).with_name('release.py'))
R = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R)
F = R.load('factory'); Records = R.load('record_bank')


def execute(owner, emit, config):
    """Production controller path in A, then existing sampled-task evaluator."""
    stats = {}
    error = None
    try:
        _, stats = owner.run_mpc_goals(np.asarray(config['initial_state'], dtype=np.float32),
            [np.asarray(config['goal'], dtype=np.float32)], sim_dt=.003, goal_timeout=8.,
            goal_threshold=.05, goal_axis_threshold=np.deg2rad(30.), velocity_threshold=1.,
            goal_dwell_time=0., reset_rho_each_solve=False, initial_guess=None,
            offline_timing='controller_dt', record_raw=True, record_callback=emit)
    except BaseException:
        error = traceback.format_exc(); stats = getattr(owner, 'last_run_stats', {})
        last = stats.get('raw_solves', [{}])[-1] if stats.get('raw_solves') else {}
        emit('execution_failed', dict(last, error=error))
    raw = stats.get('raw_execution', [])
    x = [np.asarray(config['initial_state'], dtype=np.float32)]; u = []; t = [0.]
    for step in raw:
        if 'state_after' in step:
            x.append(step['state_after']); u.append(step['control']); t.append(step['time_s'] + step['step_dt_s'])
    emit('readiness_execution', dict(states=np.asarray(x), controls=np.asarray(u).reshape(-1, 7), times=np.asarray(t), stats=stats))
    spec = importlib.util.spec_from_file_location('f01_evaluate', R.ROOT / 'experiments/campaign/evaluate.py')
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    return evaluator.evaluate(config, owner.model, np.asarray(x), np.asarray(u).reshape(-1, 7), np.asarray(t), stats, error)


def run(output, manifest_path, manifest_sha256, machine, family_state, *, builder=None, execute_task=None):
    output = Path(output)
    recorder = Records.Recorder(output, R.P.preflight_protocol())
    context = None; owner = None
    result = dict(campaign_id='F01-A-preflight', machine=machine, timing_mode='A', execution_backend='internal',
                  task_success=None, physical_pass=None, execution_error=None, outcome='execution_held',
                  evidence_kind='synthetic_test' if any((builder, execute_task)) else 'measurement')
    try:
        context = R.validate(manifest_path, manifest_sha256, machine, 32)
        if Path(family_state).exists():
            previous = R.read(family_state)
            if previous['status'] == 'held': raise RuntimeError('Family already held; no automatic preflight retry')
        bank, _ = R.P.load_bank(); events = F.Events(recorder)
        factory = F.Factory(context, bank, events, builder)
        owner = factory('reaching', dict(N=32, SQP=5, PCG=120))
        config = R.P.preflight_protocol()
        if owner.solver_params != config['solver_params']: raise ValueError('Canonical public preset differs')
        result.update((execute_task or execute)(owner, events, config))
        result.update(canonical_config_sha256=R.canonical_hash(), source_commit=context['collector_source_commit'],
            runtime_source_sha256=context['modules']['32']['runtime_source_sha256'],
            module_sha256=context['modules']['32']['sha256'], observer_patch_sha256=context['observer_patch_sha256'],
            outcome='execution_failed' if result['execution_error'] else ('task_passed' if result['task_success'] else 'task_failed'))
    except BaseException as error:
        result.update(execution_error=str(error), outcome='execution_held' if context is None else 'execution_failed')
    finally:
        if owner is not None: owner.close()
        recorder.finish(result)
    if context is not None:
        passed = result['execution_error'] is None and result['task_success'] is True and result['physical_pass'] is True
        R.write_state(family_state, context, 'ready' if passed else 'held',
            readiness_metrics=dict(path=str(output / 'metrics.json'), sha256=R.sha(output / 'metrics.json')),
            reason=None if passed else 'Canonical A rough/physical/execute failure; retain and discuss before another family worker')
    return result
