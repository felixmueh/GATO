"""Focused CPU checks; no native load, build, ROS, or motion."""
import importlib.util
import json
from pathlib import Path
import queue
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tiago_src'))
sys.path.insert(0, str(Path(__file__).parent))
from protocol import periodic_reference, resolve_protocol, safety_policy
from worker import make_finite_recorder, run_native_readiness
from gato_tiago import safety_monitor as safety
from gato_tiago.tiago_controller_process import (
    _validate_trajectory, _require_gazebo_report_only_context, TorqueTrajectory)

SIM = dict(GATO_ROS_PROFILE='simulation', ROS_DOMAIN_ID='1',
           RMW_IMPLEMENTATION='rmw_cyclonedds_cpp')


@pytest.fixture
def inputs(tmp_path):
    for name in ('t01', 'v01'):
        data = subprocess.check_output(['git', '-C', str(ROOT), 'show',
            f'fa6f6be:experiments/campaign/inputs/{name}.json'])
        (tmp_path / f'{name}.json').write_bytes(data)
    return tmp_path


def test_reference_anchor_geometry_and_full_preview(inputs):
    p = resolve_protocol(inputs, dict(id='figure8-12s', parameters={}), 'internal')
    frozen = json.loads((inputs / 't01.json').read_text())
    original = next(t for t in frozen['tasks'] if t['id'] == 'figure8-12s')
    np.testing.assert_allclose(p['reference_packed'],
        original['reference_packed'][:len(p['reference_packed'])], atol=2e-15, rtol=0)
    p = resolve_protocol(inputs, dict(id='figure8-6s', parameters=dict(N=128, dt_s=.045,
        sqp_max_iter=5, pcg_max_iter=120)), 'internal')
    assert p['reference_dt_s'] == p['dt_s'] == .045
    assert p['duration_s'] == 6. and p['reference_coverage_s'] >= 6. + 127 * .045
    assert p['solver_params']['max_pcg_iters'] == 120
    np.testing.assert_array_equal(p['reference_packed'][:6], original['reference_packed'][:6])


def test_canonical_and_case_cannot_enable_unsafe(inputs):
    p = resolve_protocol(inputs, dict(id='goal-3', parameters=dict(N=128, dt_s=.06,
        sqp_max_iter=1, pcg_max_iter=120)), 'internal', preflight=True)
    assert (p['N'], p['dt_s'], p['solver_params']['max_sqp_iters']) == (32, .03, 5)
    assert p['preflight_setup_sha256'] == p['canonical_setup_sha256']
    assert len(p['input_sha256']) == 64
    with pytest.raises(ValueError, match='unsafe policy'):
        resolve_protocol(inputs, dict(id='goal-3', parameters=dict(report_only=True)), 'internal')


@pytest.mark.parametrize('backend,env', [('robot', SIM), ('internal', SIM),
    ('gazebo', {}), ('gazebo', dict(SIM, GATO_ROS_PROFILE='robot'))])
def test_unsafe_rejected_without_explicit_gazebo(backend, env):
    with pytest.raises(ValueError):
        safety_policy(backend, True, env)
    with pytest.raises(ValueError):
        _require_gazebo_report_only_context(True, backend, env)


def test_safe_default_and_unclamped_report_only():
    _require_gazebo_report_only_context(False, 'robot', {})
    _require_gazebo_report_only_context(True, 'gazebo', SIM)
    controls = np.zeros((31, 7)); controls[-1, 2] = 26.01
    trajectory = TorqueTrajectory(controls, .03)
    with pytest.raises(ValueError, match='torque exceeds'):
        _validate_trajectory(trajectory, 7, 26., False)
    np.testing.assert_array_equal(_validate_trajectory(trajectory, 7, 26., False, True), controls)
    controls[-1, 2] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        _validate_trajectory(trajectory, 7, 26., False, True)


def test_report_only_joint_violation_still_checks_collision_and_rejects_nan(monkeypatch):
    checker = safety._SafetyChecker.__new__(safety._SafetyChecker)
    checker.settings = safety.CollisionSafetySettings(report_only=True)
    checker.collision_model = SimpleNamespace(model=object())
    checker._last_checked_seq = -1
    state = safety.SafetyCheckState(np.zeros(7), np.zeros(7), 0., 0., 1, {}, {})
    monkeypatch.setattr(safety, 'state_to_qv', lambda *a: (np.zeros(7), np.zeros(7)))
    monkeypatch.setattr(safety, 'check_joint_limits', lambda *a, **k: [
        SimpleNamespace(joint='arm_right_1_joint', kind='position', value=4.)])
    monkeypatch.setattr(safety, 'compute_collision_margin_violation', lambda *a, **k:
        SimpleNamespace(geometry_a='a', geometry_b='b', approximate_distance_m=.02))
    monkeypatch.setattr(safety, 'compute_collision_body_speeds', lambda *a:
        [SimpleNamespace(speed_bound_m_s=.1, geometry='a')])
    reports = checker.check(state)
    assert [r['kind'] for r in reports] == ['joint_limit', 'collision_margin']
    assert checker._last_checked_seq == 1
    monkeypatch.setattr(safety, 'state_to_qv', lambda *a: (np.full(7, np.nan), np.zeros(7)))
    with pytest.raises(safety._SafetyFault, match='nonfinite'):
        checker.check(SimpleNamespace(**{**state.__dict__, 'seq': 2}))


def test_async_violation_advances_initial_sequence_and_continues():
    monitor = safety.AsyncSafetyMonitor.__new__(safety.AsyncSafetyMonitor)
    monitor._response_q = queue.Queue()
    monitor._pending_seq = 1
    monitor._pending_since_monotonic_sec = 0.
    monitor.physical_summary = dict(checked_states=0, violating_states=0,
        violations_by_kind={}, first_violation=None, last_violation=None,
        minimum_violating_pair_approximate_distance_m=None)
    monitor._response_q.put(safety._SafetyCheckResult(seq=1, ok=True,
        physical_reports=({'kind': 'collision_margin', 'approximate_distance_m': .02},)))
    monitor._poll_responses()
    assert monitor.last_checked_seq == 1 and monitor.safety_status == 'violations_report_only'
    monitor._response_q.put(safety._SafetyCheckResult(seq=2, ok=True))
    monitor._poll_responses()
    assert monitor.last_checked_seq == 2 and monitor.safety_status == 'ok'
    assert monitor.physical_summary['violating_states'] == 1


def test_finite_guard_records_before_error_and_does_not_add_limits():
    events = []
    class Parent:
        def __call__(self, kind, record):
            events.append((kind, record))
    recorder = make_finite_recorder(Parent)()
    # Finite physical violations are retained for reporting, not new internal gates.
    recorder('step_completed', dict(state_after=np.full(14, 100.), control=np.full(7, 40.)))
    for kind, record in [('solve_completed', dict(output=np.array([[np.nan]]))),
                         ('step_completed', dict(state_after=np.full(14, np.nan))),
                         ('feedback_state', dict(q=np.full(7, np.inf)))]:
        with pytest.raises(ValueError, match='nonfinite execution'):
            recorder(kind, record)
        assert events[-1] == (kind, record)


@pytest.mark.parametrize('nonfinite', [False, True])
def test_native_readiness_preserves_new_N_one_call_and_final_output_gate(inputs, tmp_path, monkeypatch, nonfinite):
    # Explicit CPU fixture for collector wiring; not native/GPU qualification.
    sys.path.insert(0, '/workspace/GATO-experiment-runner-simple/experiments')
    from adapters import task
    import worker
    module = tmp_path / 'fixture.so'; module.write_bytes(b'CPU fixture only')
    native = object(); calls = []
    monkeypatch.setattr(task, 'native_module', lambda source, build, N: (native, module))
    class Solver:
        def __init__(self, *args, **kwargs):
            assert args[1:4] == (1, 256, .03)
            assert kwargs['max_sqp_iters'] == 1 and kwargs['max_pcg_iters'] == 120
            self.lib = native; self.stats = {'kkt_converged': [False]}
        def solve(self, state, reference, seed):
            calls.append((state.copy(), reference.copy(), seed.copy()))
            result = seed.copy()
            if nonfinite:
                result[0, -1] = np.nan
            return result, np.array([10.])
    interface = SimpleNamespace(__file__=str(ROOT / 'python/bsqp/interface.py'), BSQP=Solver)
    original_import = worker.importlib.import_module
    monkeypatch.setattr(worker.importlib, 'import_module',
        lambda name: interface if name == 'bsqp.interface' else original_import(name))
    args = SimpleNamespace(backend='internal', preflight=False, dangerously_disable_safety_stops=False,
        inputs=inputs, dry_run=False, runner_root=Path('/workspace/GATO-experiment-runner-simple'),
        source=ROOT, build=tmp_path / 'build', output=tmp_path / 'result', collector_pin='0' * 40)
    case = dict(id='goal-3', parameters=dict(N=256, dt_s=.03, sqp_max_iter=1, pcg_max_iter=120))
    status = run_native_readiness(args, case)
    metrics = json.loads((args.output / 'metrics.json').read_text())
    p = json.loads((args.output / 'protocol.json').read_text())
    canonical = json.loads((inputs / 'v01.json').read_text())
    assert len(calls) == 1 and metrics['optimizer_calls'] == 1 and metrics['N'] == 256
    assert p['goal'] == canonical['goal'] and p['initial_state'] == canonical['initial_state']
    assert metrics['motion_executed'] is False and metrics['task_success'] is None
    assert metrics['finite_readiness_pass'] is (not nonfinite)
    assert status == int(nonfinite)
    assert (args.output / 'native_output.npz').is_file()
