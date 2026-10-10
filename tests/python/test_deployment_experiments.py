"""Representative CPU checks for deployment recording and lifecycle semantics."""
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import signal
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.deployment.protocol import resolve_case, gazebo_environment_guard, preflight_guard
from experiments.deployment.live import RecordedSolver, RecordedController, execute_with_handback
from experiments.deployment.raw import RawRecorder
from experiments.deployment.evaluate import physical_metrics, task_metrics


def case(name='goal-1'):
    return dict(id=name, parameters=dict(backend='gazebo'))


def test_dry_run_imports_no_ros_native_or_pinocchio(tmp_path):
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'rclpy', 'pinocchio', 'bsqp', 'gato_tiago'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, Block())
from experiments.deployment.run_d01 import main
raise SystemExit(main())
'''
    result = subprocess.run([sys.executable, '-c', code, '--case', json.dumps(case('figure8-18s')),
        '--output', str(tmp_path), '--dry-run'], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    protocol = json.loads(result.stdout)
    assert protocol['execution_backend'] == 'gazebo'
    assert protocol['timing_mode'] == 'live' and protocol['computation_paid']
    assert protocol['live_controller']['target_hz'] == 125
    assert protocol['provisional_RMS_gate_m'] == .025
    assert not list(tmp_path.iterdir())
    reaching = resolve_case(case('preflight-goal-1'))
    assert reaching['live_controller']['target_hz'] == 100
    assert reaching['solver_params']['qd_cost'] == .1
    assert reaching['solver_params']['ee_orient_cost'] == .1


def test_solver_preserves_exact_input_output_failure_and_timer_scope(tmp_path):
    ticks, events = [10.], []
    seed = np.arange(8, dtype=np.float64).reshape(1, -1)
    before = seed.copy()
    result = (np.arange(8, dtype=np.float32).reshape(1, -1), 123)
    class Solver:
        stats = {'native_flag': np.array([1], dtype=np.int32)}
        def solve(self, state, reference, warm):
            warm[:] = -99
            ticks[0] += .25
            return result
        def get_stats(self):
            return self.stats
    def recorder(kind, value):
        events.append((kind, value))
        ticks[0] += 20  # IO is outside measured solve wall.
    solver = RecordedSolver(Solver(), recorder, lambda: ticks[0])
    state, reference = np.zeros((1, 2)), np.ones((1, 6))
    assert solver.solve(state, reference, seed) is result
    assert solver.get_stats() is solver.delegate.stats
    raw = events[0][1]
    np.testing.assert_array_equal(raw['warm_start'], before)
    np.testing.assert_array_equal(raw['state'], state)
    np.testing.assert_array_equal(raw['reference'], reference)
    assert raw['wrapper_wall_s'] == .25
    assert raw['returned_native_time_us'] == 123
    result[0][:] = 7
    assert not np.all(raw['output'] == 7)
    exception = RuntimeError('retained failure')
    class Failing(Solver):
        def solve(self, state, reference, warm):
            warm[:] = np.nan
            raise exception
    failed = RecordedSolver(Failing(), RawRecorder(tmp_path))
    with pytest.raises(RuntimeError) as caught:
        failed.solve(state, reference, before)
    assert caught.value is exception
    event = json.loads((tmp_path / 'events.jsonl').read_text())
    assert event['event'] == 'solve_failure'
    saved = np.load(tmp_path / event['raw_file'])
    np.testing.assert_array_equal(saved['record__warm_start'], np.arange(8).reshape(1, -1))
    def broken_recorder(*args):
        raise OSError('record write failed')
    failed = RecordedSolver(Failing(), broken_recorder)
    with pytest.raises(RuntimeError) as caught:
        failed.solve(state, reference, before)
    assert caught.value is exception
    assert failed.recording_errors[0]['event'] == 'solve_failure'


@pytest.mark.parametrize('shutdown_failure', [False, True])
def test_task_failure_always_closes_and_preserves_both_histories(tmp_path, shutdown_failure):
    called = []
    class Fake:
        def close(self, **kwargs):
            called.append('close')
            assert kwargs['timeout_sec'] == 8.0
            if shutdown_failure:
                raise RuntimeError('restoration unconfirmed')
        def write_state_history_csv(self, path):
            called.append('csv'); path.write_text('raw state history')
        def write_full_state_history_jsonl(self, path):
            called.append('full'); path.write_text('raw full joint/mode history')
    live = RecordedController(Fake(), lambda *args: None)
    def task(_):
        raise ValueError('public task failed')
    stats, errors = execute_with_handback(live, task, tmp_path, lambda *args: None)
    assert called == ['close', 'csv', 'full']
    assert errors[0]['stage'] == 'task'
    assert live.handback_confirmed is (not shutdown_failure)
    assert [e['stage'] for e in errors] == (['task', 'handback'] if shutdown_failure else ['task'])
    assert (tmp_path / 'full_joint_state_history.jsonl').read_text()


def test_task_and_physical_outcomes_are_distinct():
    protocol = resolve_case(case())
    protocol['goal'] = [1., 0., 0., 0., 0., 0.]
    protocol['target_rotation'] = np.eye(3)
    task = task_metrics(protocol, [0., 1.], [[0., 0., 0.], [1., 0., 0.]],
        [np.eye(3), np.eye(3)], np.zeros((2, 7)), {'goal_outcomes': ['reached']})
    assert task['task_success'] is True
    model = SimpleNamespace(lowerPositionLimit=-np.ones(7), upperPositionLimit=np.ones(7),
                            velocityLimit=np.ones(7), effortLimit=np.ones(7))
    physical = physical_metrics(protocol, model, np.zeros((2, 7)), np.full((2, 7), 2.), np.zeros((2, 7)))
    assert physical['physical_pass'] is False
    assert physical['measured_actuator_torque_available'] is False
    assert physical['full_physical_compliance'] is None


def test_gazebo_and_canonical_preflight_guard(tmp_path):
    with pytest.raises(RuntimeError, match='simulation ROS'):
        gazebo_environment_guard({'GATO_ROS_PROFILE': 'robot', 'ROS_DOMAIN_ID': '2'})
    gazebo_environment_guard(dict(GATO_ROS_PROFILE='simulation', ROS_DOMAIN_ID='1',
        RMW_IMPLEMENTATION='rmw_cyclonedds_cpp', GATO_MACHINE='rtx5060'))
    with pytest.raises(RuntimeError, match='preflight'):
        preflight_guard(resolve_case(case()), case())
    saved = dict(case_id='preflight-goal-1', execution_backend='gazebo', machine='rtx5060',
        timing_mode='live', input_sha256=resolve_case(case())['input_sha256'], task_success=True,
        preflight_setup_sha256=resolve_case(case())['preflight_setup_sha256'],
        physical_pass=False, execution_error=None, handback_confirmed=True)
    (tmp_path / 'metrics.json').write_text(json.dumps(saved))
    following = case(); following['parameters']['preflight_output'] = str(tmp_path)
    with pytest.raises(RuntimeError, match='family held'):
        preflight_guard(resolve_case(following), following)
    saved['physical_pass'] = True
    (tmp_path / 'metrics.json').write_text(json.dumps(saved))
    preflight_guard(resolve_case(following), following)


def test_observer_failure_is_preserved_and_closed_with_sibling_output(tmp_path, monkeypatch):
    from experiments.deployment import live, run_d01
    calls = []
    class Observer:
        def __init__(self, output):
            calls.append('observe')
        def require_advancing_clock(self, timeout_s):
            raise RuntimeError('observer clock error')
        def close(self):
            calls.append('close')
    monkeypatch.setattr(live, 'GazeboObserver', Observer)
    for key, value in dict(GATO_ROS_PROFILE='simulation', ROS_DOMAIN_ID='1',
        RMW_IMPLEMENTATION='rmw_cyclonedds_cpp', GATO_MACHINE='rtx5060').items():
        monkeypatch.setenv(key, value)
    # Source and output are siblings rather than source-contained runner paths.
    output = tmp_path / 'sibling-output'
    assert run_d01.run_case(case('preflight-goal-1'), output) == 1
    assert calls == ['observe', 'close']
    metrics = json.loads((output / 'metrics.json').read_text())
    assert metrics['execution_error'][0]['message'] == 'observer clock error'
    assert metrics['physical_pass'] is None and metrics['family_held']
    assert metrics['task_success'] is None and metrics['task_unavailable']
    assert (output / 'events.jsonl').read_text()


def test_failed_recorder_cannot_prevent_controller_histories_or_observer_cleanup(tmp_path, monkeypatch):
    from experiments.deployment import live, run_d01, evaluate
    calls = []
    class Recorder:
        def __init__(self, output):
            pass
        def __call__(self, event, value):
            if event != 'gazebo_guard_passed':
                raise OSError('recorder unavailable')
        def manifest(self):
            raise OSError('manifest unavailable')
    class Observer:
        def __init__(self, output):
            pass
        def require_advancing_clock(self, timeout_s):
            pass
        def close(self):
            calls.append('observer close')
    class Controller:
        def __init__(self, **kwargs):
            pass
        def close(self, **kwargs):
            assert kwargs['timeout_sec'] == 8
            calls.append('controller close')
        def write_state_history_csv(self, path):
            calls.append('csv'); path.write_text('retained csv')
            return {'retained': True}
        def write_full_state_history_jsonl(self, path):
            calls.append('full'); path.write_text('retained full history')
    class Solver:
        pass
    class MPC:
        def __init__(self, *args, **kwargs):
            self.solver = Solver()
    pin = ModuleType('pinocchio')
    pin.buildModelFromUrdf = lambda _: object()
    package = ModuleType('gato_tiago'); package.__path__ = []
    mpc_module = ModuleType('gato_tiago.tiago_mpc_controller'); mpc_module.MPC_GATO = MPC
    controller_module = ModuleType('gato_tiago.tiago_controller_process')
    controller_module.TiagoControllerOrchestrator = Controller
    native = ModuleType('bsqp.bsqpN32_tiago_right')
    artifact = tmp_path / 'fake-module'; artifact.write_bytes(b'CPU fake module')
    native.__file__ = str(artifact)
    for name, module in {'pinocchio': pin, 'gato_tiago': package,
            mpc_module.__name__: mpc_module, controller_module.__name__: controller_module,
            native.__name__: native}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(run_d01, 'RawRecorder', Recorder)
    monkeypatch.setattr(live, 'GazeboObserver', Observer)
    def task(mpc, *args):
        return None, {'goal_outcomes': ['reached']}
    monkeypatch.setattr(run_d01, 'execute_task', task)
    # Independently available task evidence stays successful despite subsequent
    # recorder/handback-history finalization failures. No native evaluation.
    monkeypatch.setattr(evaluate, 'evaluate_live', lambda *args:
        {'task_success': True, 'physical_pass': True})
    for key, value in dict(GATO_ROS_PROFILE='simulation', ROS_DOMAIN_ID='1',
        RMW_IMPLEMENTATION='rmw_cyclonedds_cpp', GATO_MACHINE='rtx5060').items():
        monkeypatch.setenv(key, value)
    output = tmp_path / 'worker-output'
    handlers_before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    assert run_d01.run_case(case('preflight-goal-1'), output) == 1
    assert {sig: signal.getsignal(sig) for sig in handlers_before} == handlers_before
    assert calls == ['controller close', 'csv', 'full', 'observer close']
    assert (output / 'controller_state_history.csv').read_text() == 'retained csv'
    assert (output / 'full_joint_state_history.jsonl').read_text() == 'retained full history'
    metrics = json.loads((output / 'metrics.json').read_text())
    assert metrics['handback_confirmed'] is True
    assert metrics['task_success'] is True and metrics['physical_pass'] is True
    assert metrics['family_held'] is True
    errors = metrics['execution_error']
    assert errors[0]['event'] == 'close_request'
    assert any(e.get('event') == 'close_request' for e in errors)
    assert any(e.get('event') == 'close_confirmed_restored' for e in errors)
    assert any(e['stage'] == 'manifest_write' for e in errors)
    assert not any(e['stage'] == 'handback' for e in errors)
