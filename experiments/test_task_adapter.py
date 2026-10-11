"""CPU-only task wiring and partial-record preservation; no simulated solver."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from adapters import task
from recording import iter_events, sha


def canonical():
    return dict(kind='reaching', initial_state=[0.] * 14, goal=[.1, .2, .3, 0., 0., 0.],
                N=32, dt_s=.03, sim_dt_s=.003, timeout_s=8., position_threshold_m=.05,
                axis_threshold_deg=30., speed_L1_threshold_rad_s=1., gravity_m_s2=[0., 0., -9.81],
                constant_f_ext=[0.] * 6, solver_params={'qd_cost': .1, 'vel_lim_cost': .1},
                timing_mode='B', offline_timing='solve_time', timing_scope='solve wrapper wall/native',
                release_status='historical gate', family_hold_policy='historical hold')


def test_selection_keeps_frozen_task_parameters(tmp_path):
    inputs = tmp_path / 'experiments/campaign/inputs'
    inputs.mkdir(parents=True)
    original = canonical()
    (inputs / 'v01.json').write_text(json.dumps(original))
    selected = task.select_protocol(tmp_path, {}, 'A', True)
    for key in ('initial_state', 'goal', 'N', 'dt_s', 'sim_dt_s', 'solver_params', 'timeout_s'):
        assert selected[key] == original[key]
    assert selected['offline_timing'] == 'controller_dt' and selected['timing_mode'] == 'A'
    assert 'computation time ignored' in selected['approved_clock_convention']
    assert selected['historical_release_metadata']['release_status'] == original['release_status']
    assert 'family_hold_policy' not in selected
    assert json.loads((inputs / 'v01.json').read_text()) == original
    frozen_task = dict(id='track', kind='tracking', reference_packed=[1., 2., 3., 0., 0., 0.],
                       reference_dt_s=.03, duration_s=12., provisional_RMS_gate_m=.05)
    cohort = dict(original, reaching_solver_params=original['solver_params'],
                  tracking_solver_params={'qd_cost': .01}, tasks=[frozen_task])
    (inputs / 't01.json').write_text(json.dumps(cohort))
    selected = task.select_protocol(tmp_path, {'id': 'track'}, 'B', False)
    assert selected['solver_params'] == cohort['tracking_solver_params']
    assert selected['reference_packed'] == frozen_task['reference_packed']
    assert selected['duration_s'] == 12. and selected['offline_timing'] == 'solve_time'
    with pytest.raises(ValueError, match='mode B'):
        task.select_protocol(tmp_path, {'id': 'track'}, 'A', False)


@pytest.mark.parametrize('mode, offline', [('A', 'controller_dt'), ('B', 'solve_time')])
def test_reaching_arguments_and_tracking_helper(mode, offline):
    config = dict(canonical(), offline_timing=offline)
    calls = []
    controller = SimpleNamespace(run_mpc_goals=lambda *args, **kwargs: calls.append((args, kwargs)))
    emit = object()
    task.execute(controller, config, emit, object())
    args, kwargs = calls[0]
    assert args[0].dtype == np.float32 and args[1][0].dtype == np.float32
    np.testing.assert_array_equal(args[1][0], np.asarray(config['goal'], dtype=np.float32))
    assert kwargs == dict(sim_dt=.003, goal_timeout=8., goal_threshold=.05,
                          goal_axis_threshold=np.deg2rad(30.), velocity_threshold=1.,
                          goal_dwell_time=0., reset_rho_each_solve=False, initial_guess=None,
                          offline_timing=offline, record_raw=True, record_callback=emit)
    helpers = SimpleNamespace(execute_episode=lambda *args: ('delegated', args))
    tracking = dict(config, kind='tracking')
    assert task.execute(controller, tracking, emit, helpers) == ('delegated', (controller, tracking, emit))


def test_explicit_native_module_selection(tmp_path, monkeypatch):
    modules = tmp_path / 'build/modules'
    modules.mkdir(parents=True)
    path = modules / 'bsqpN32_tiago_right.test.so'
    path.write_bytes(b'never imported')
    native = object()
    calls = []
    monkeypatch.setattr(task, 'load', lambda name, selected: calls.append((name, selected)) or native)
    original_import = task.importlib.import_module
    monkeypatch.setattr(task.importlib, 'import_module',
                        lambda name: object() if name == 'bsqp' else original_import(name))
    monkeypatch.setitem(sys.modules, 'bsqp.bsqpN32_tiago_right', object())
    selected, selected_path = task.native_module(tmp_path / 'source', tmp_path / 'build')
    assert selected is native and selected_path == path
    assert calls == [('bsqp.bsqpN32_tiago_right', path)]
    assert sys.modules['bsqp.bsqpN32_tiago_right'] is native
    with pytest.raises(ValueError, match='Expected one'):
        task.native_module(tmp_path, tmp_path / 'build', 64)


def test_setup_failure_and_existing_evidence_preserved(tmp_path):
    output = tmp_path / 'result'
    assert task.run(tmp_path / 'missing-source', tmp_path / 'build', {}, output, 'A', True) == 1
    metrics_bytes = (output / 'metrics.json').read_bytes()
    assert json.loads(metrics_bytes)['outcome'] == 'execution_failed'
    assert next(iter_events(output))['event'] == 'execution_failed'
    with pytest.raises(FileExistsError):
        task.run(tmp_path, tmp_path, {}, output, 'A', True)
    assert (output / 'metrics.json').read_bytes() == metrics_bytes


@pytest.mark.parametrize('interrupt, preflight, expected_status', [(True, False, 1), (False, True, 1), (False, False, 0)])
def test_partial_execution_and_scientific_negative(tmp_path, monkeypatch, interrupt, preflight, expected_status):
    source = tmp_path / 'source'
    source.mkdir()
    urdf = source / 'robot.urdf'
    urdf.write_bytes(b'fixture urdf')
    module_path = tmp_path / 'module.so'
    module_path.write_bytes(b'fixture module')
    config = dict(canonical(), urdf={'path': 'robot.urdf', 'sha256': sha(urdf)})
    copied_inputs = tmp_path / 'copied-inputs'
    copied_inputs.mkdir()
    source_inputs = source / 'experiments/campaign/inputs'
    source_inputs.mkdir(parents=True)
    stale = dict(config, goal=[.9] * 6)
    (source_inputs / 'v01.json').write_text(json.dumps(stale))
    (copied_inputs / 'v01.json').write_text(json.dumps(config))
    (source_inputs / 't01.json').write_text(json.dumps(dict(stale, reaching_solver_params=config['solver_params'],
          tasks=[dict(id='goal', kind='reaching', goal=stale['goal'])])))
    (copied_inputs / 't01.json').write_text(json.dumps(dict(config, reaching_solver_params=config['solver_params'],
          tasks=[dict(id='goal', kind='reaching', goal=config['goal'])])))
    native = object()
    stats = {'raw_execution': [dict(state_after=np.full(14, np.nan), control=np.zeros(7), time_s=0., step_dt_s=.003)],
             'raw_solves': [dict(wall_solve_time_s=.001, native_solve_time_ms=.9,
                                selected_metadata=np.array([np.inf]))]}
    observed = []

    def controller_factory(model, path, **kwargs):
        assert kwargs['N'] == 32 and kwargs['dt'] == .03 and kwargs['final_approach'] is False
        assert kwargs['solver_params'] == config['solver_params']
        return SimpleNamespace(model=model, solver=SimpleNamespace(lib=native),
                               solver_params=kwargs['solver_params'], last_run_stats=stats)

    def execute(controller, protocol, emit, helpers):
        assert protocol['goal'] == config['goal'] != stale['goal']
        emit('solve_completed', {'state': np.full(14, np.nan)})
        if interrupt:
            raise KeyboardInterrupt('user interrupted')
        return None, stats

    def metrics(protocol, model, x, u, t, partial, error, output):
        assert np.isnan(x[-1]).all() and u.shape == (1, 7) and t[-1] == .003
        observed.append(error)
        return {'task_success': False, 'physical_pass': False, 'execution_error': error}

    helpers = SimpleNamespace(trace_metrics=metrics, distribution=lambda values: {'count': len(values)})
    monkeypatch.setattr(task, 'load_helpers', lambda *args: helpers)
    monkeypatch.setattr(task, 'native_module', lambda *args: (native, module_path))
    monkeypatch.setattr(task, 'execute', execute)
    monkeypatch.setattr(task.subprocess, 'check_output', lambda *args, **kwargs: 'source-commit\n')
    model = SimpleNamespace(gravity=SimpleNamespace(linear=None))
    monkeypatch.setitem(sys.modules, 'pinocchio', SimpleNamespace(buildModelFromUrdf=lambda path: model))
    production = SimpleNamespace(__file__=str(source / 'tiago_src/gato_tiago/tiago_mpc_controller.py'),
                                 MPC_GATO=controller_factory)
    original_import = task.importlib.import_module
    monkeypatch.setattr(task.importlib, 'import_module', lambda name: production if name ==
                        'gato_tiago.tiago_mpc_controller' else original_import(name))
    output = tmp_path / 'result'
    assert task.run(source, tmp_path / 'build', {'id': 'goal'}, output, 'B', preflight,
                    inputs=copied_inputs) == expected_status
    result = json.loads((output / 'metrics.json').read_text())
    assert result['task_success'] is False and result['physical_pass'] is False
    assert result['configuration']['goal'] == config['goal']
    assert json.loads((output / 'protocol.json').read_text())['goal'] == config['goal']
    assert result['outcome'] == ('execution_failed' if interrupt else 'task_failed')
    events = list(iter_events(output))
    assert np.isnan(events[0]['record']['state']).all()
    final = next(e for e in events if e['event'] == 'final_raw_solves')
    assert np.isposinf(final['record']['solves'][0]['selected_metadata'][0])
    assert observed and (observed[0] is not None) == interrupt
