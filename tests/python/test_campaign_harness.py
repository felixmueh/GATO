"""CPU contract checks; synthetic records are not scientific task evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT/'experiments/campaign'
sys.path.insert(0, str(CAMPAIGN))
from raw import RawRecorder, sha
from evaluate import distribution, evaluate
from run_v01 import load_protocol, run_case
import run_v01
from make_plan import make_plan
from build_isolated import prepare, build_native
import build_isolated
from propose_t01 import figure8_speed
from t01_protocol import diagnostic_protocol


def protocol():
    return json.loads((CAMPAIGN/'inputs/v01.json').read_text())


def test_frozen_canonical_protocol_preserves_fixture():
    p = protocol()
    fixture = ROOT/p['frozen_fixture']['path']
    original = json.loads(fixture.read_text())
    assert sha(fixture) == p['frozen_fixture']['sha256']
    assert original['sim_dt_s'] == .001 and p['sim_dt_s'] == .003
    assert p['initial_state'] == original['initial_state'] and p['goal'] == original['goal']
    assert p['N'] == 32 and p['prediction_duration_s'] == pytest.approx((32-1)*.03)
    assert p['offline_timing'] == 'solve_time' and not p['final_approach']
    assert p['solver_params']['qd_cost'] == .1 and p['solver_params']['max_pcg_iters'] == 120
    assert p['solver_params']['ee_orient_cost'] == .1 and p['solver_params']['ee_orient_N_cost'] == 2.
    assert p['initial_guess'] is None and p['solver_numerical_pass'] is None


def test_raw_recorder_retains_nonfinite_arrays_and_native_telemetry(tmp_path):
    record = {'state':np.array([[np.nan, np.inf, -np.inf, 1]], dtype=np.float32),
              'solver_stats':{'flag':np.array([False]), 'residual':float('nan')},
              'output':np.array([[2., np.nan]], dtype=np.float64)}
    original = record['state'].copy()
    recorder = RawRecorder(tmp_path)
    recorder('solve_completed', record)
    event = json.loads((tmp_path/'events.jsonl').read_text())
    assert event['record']['state']['nonfinite_count'] == 3
    assert sha(tmp_path/event['raw_file']) == event['sha256']
    with np.load(tmp_path/event['raw_file'], allow_pickle=False) as raw:
        np.testing.assert_equal(raw['record__state'], original)
        assert raw['record__state'].dtype == np.float32
        assert np.isnan(raw['record__solver_stats__residual'])
        assert raw['record__solver_stats__flag'].dtype == bool
        assert np.isnan(raw['record__output'][0, 1])
    np.testing.assert_equal(record['state'], original)
    (tmp_path/'stdout.log').write_text('still changing')
    recorder.manifest()
    assert 'stdout.log' not in json.loads((tmp_path/'SHA256SUMS.json').read_text())


def test_distribution_never_discards_bad_timing_samples():
    assert distribution([1., np.nan, 10.])['mean'] is None
    assert distribution([1., np.nan, 10.])['count'] == 3
    assert distribution([])['mean'] is None
    assert distribution([1., 3.])['mean'] == 2.


@pytest.mark.parametrize('event', ['solve_selected', 'plan_activated', 'plan_discarded', 'interval_completed'])
def test_true_B_observer_metadata_events_are_durable(tmp_path, event):
    recorder = RawRecorder(tmp_path)
    record = {'solve_index':1, 'plan_solve_index':0, 'feedback_time_s':.03,
              'activation_time_s':None if event == 'plan_discarded' else .04,
              'activation_status':'discarded_task_terminal' if event == 'plan_discarded' else 'activated',
              'output':np.array([[np.nan, 1.]], dtype=np.float32)}
    recorder(event, record)
    saved = json.loads((tmp_path/'events.jsonl').read_text())
    assert saved['event'] == event and saved['record']['plan_solve_index'] == 0
    assert saved['record']['activation_time_s'] == record['activation_time_s']
    with np.load(tmp_path/saved['raw_file']) as values:
        np.testing.assert_equal(values['record__output'], record['output'])


@pytest.fixture
def model():
    import pinocchio as pin
    return pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))


def synthetic_execution(p):
    initial = np.asarray(p['initial_state'])
    endpoint = initial.copy()
    endpoint[:7] += np.asarray(p['endpoint_delta_rad'])
    return np.array([initial, endpoint]), np.zeros((1, 7)), np.array([0., .003])


def test_task_and_physical_validation_separate_and_exception_preserved(model):
    p = protocol()
    x, u, t = synthetic_execution(p)
    result = evaluate(p, model, x, u, t, {'goal_outcomes':['reached']})
    assert result['task_success'] and result['physical_pass']
    assert result['solver_numerical_pass'] is None
    u[0, 0] = model.effortLimit[0]+1.
    result = evaluate(p, model, x, u, t, {'goal_outcomes':['reached']})
    assert result['task_success'] and result['physical_pass'] is False
    u[:] = 0.
    result = evaluate(p, model, x, u, t, {'goal_outcomes':['reached']}, error='synthetic failure')
    assert result['task_success'] is False and result['physical_pass'] is True


def test_bad_execution_clock_and_nonfinite_states_fail_without_drops(model):
    p = protocol()
    x, u, t = synthetic_execution(p)
    result = evaluate(p, model, x, u, [0., .004], {'goal_outcomes':['reached']})
    assert not result['task_success'] and not result['physical_pass']
    x[1, 0] = np.nan
    result = evaluate(p, model, x, u, t, {'goal_outcomes':['reached']})
    assert result['task_success'] is False and result['physical_pass'] is False
    assert np.isnan(x[1, 0])


def test_A_diagnostic_requires_completed_failed_B_same_machine_and_hash(tmp_path, monkeypatch):
    monkeypatch.setenv('GATO_MACHINE', 'gtx1070')
    evidence = tmp_path/'failed.json'
    failure = {'timing_mode':'B', 'execution_backend':'internal', 'outcome':'task_failed',
               'configuration':protocol(), 'machine':'gtx1070',
               'task_success':False, 'physical_pass':True, 'execution_error':None}
    evidence.write_text(json.dumps(failure))
    parameters = {'protocol':'V01', 'offline_timing':'controller_dt',
                  'diagnostic_after_failed_B':{'path':'failed.json', 'sha256':sha(evidence)}}
    assert load_protocol({'parameters':parameters}, tmp_path)['timing_mode'] == 'A'
    for field, invalid in [('timing_mode', 'A'), ('machine', 'rtx5060'),
                           ('execution_backend', 'gazebo'), ('outcome', 'execution_failed')]:
        bad = dict(failure, **{field:invalid})
        evidence.write_text(json.dumps(bad))
        parameters['diagnostic_after_failed_B']['sha256'] = sha(evidence)
        with pytest.raises(ValueError, match='completed same-machine'):
            load_protocol({'parameters':parameters}, tmp_path)
    with pytest.raises(ValueError):
        load_protocol({'parameters':{'protocol':'V01', 'offline_timing':'controller_dt',
                                      'diagnostic_after_failed_B':True}}, tmp_path)
    with pytest.raises(ValueError):
        load_protocol({'parameters':{'protocol':'T01', 'offline_timing':'solve_time'}})


@pytest.mark.parametrize('rough,physical,outcome,error,eligible', [
    (False, True, 'task_failed', None, True),
    (True, False, 'task_passed', None, True),
    (False, False, 'task_failed', None, True),
    (True, True, 'task_passed', None, False),
    (None, False, 'task_failed', None, False),
    (False, None, 'task_failed', None, False),
    (None, None, 'task_failed', None, False),
    (0, False, 'task_failed', None, False),
    (True, False, 'task_passed', '', False),
    (False, True, 'execution_failed', None, False),
    (False, True, 'task_failed', 'InterruptedError', False),
    (True, False, 'interrupted', None, False),
])
def test_A_diagnostic_scientific_negative_eligibility_includes_physical_failure(
        tmp_path, monkeypatch, rough, physical, outcome, error, eligible):
    monkeypatch.setenv('GATO_MACHINE', 'gtx1070')
    evidence = tmp_path/'failed.json'
    failure = {'timing_mode':'B', 'execution_backend':'internal', 'outcome':outcome,
               'configuration':protocol(), 'machine':'gtx1070', 'task_success':rough,
               'physical_pass':physical, 'execution_error':error}
    evidence.write_text(json.dumps(failure))
    case = {'parameters':{'protocol':'V01', 'offline_timing':'controller_dt',
            'diagnostic_after_failed_B':{'path':'failed.json', 'sha256':sha(evidence)}}}
    if eligible:
        assert load_protocol(case, tmp_path)['timing_mode'] == 'A'
    else:
        with pytest.raises(ValueError, match='completed same-machine'):
            load_protocol(case, tmp_path)


def git_repo(path):
    path.mkdir()
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'config', 'user.name', 'CPU Test'], check=True)
    subprocess.run(['git', '-C', str(path), 'config', 'user.email', 'cpu-test@example.invalid'], check=True)


def commit(path):
    subprocess.run(['git', '-C', str(path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(path), 'commit', '-qm', '[tiago] synthetic CPU fixture'], check=True)
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def test_build_archive_does_not_mutate_cached_sources_and_ignores_untracked(tmp_path):
    source = tmp_path/'source'
    git_repo(source)
    (source/'CMakeLists.txt').write_text('synthetic committed build input')
    head = commit(source)
    (source/'untracked.so').write_bytes(b'not an input')
    before = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'])
    isolated = prepare(source, tmp_path/'build')
    assert isolated != source
    assert (isolated/'CMakeLists.txt').read_bytes() == (source/'CMakeLists.txt').read_bytes()
    assert not (isolated/'untracked.so').exists()
    assert subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain']) == before
    assert json.loads((tmp_path/'build/build_source_receipt.json').read_text())['source_commit'] == head
    assert prepare(source, tmp_path/'build') == isolated
    assert subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain']) == before
    (isolated/'CMakeLists.txt').write_text('source drift must not be hidden')
    with pytest.raises(ValueError, match='source bytes differ'):
        prepare(source, tmp_path/'build')
    assert (isolated/'CMakeLists.txt').read_text() == 'source drift must not be hidden'


def test_interrupted_archive_extraction_is_preserved_then_retry_uses_same_commit(tmp_path):
    source = tmp_path/'source'
    git_repo(source)
    (source/'CMakeLists.txt').write_text('pinned source')
    commit(source)
    build = tmp_path/'build'
    partial = build/'campaign-source'
    partial.mkdir(parents=True)
    (partial/'half-extracted.txt').write_text('retained interrupted artifact')
    native = build/'campaign-native'
    native.mkdir()
    (native/'failure.log').write_text('retained failed configure')
    result = prepare(source, build)
    assert result == partial and (result/'CMakeLists.txt').read_text() == 'pinned source'
    preserved = list((build/'preserved-build-attempts').iterdir())
    assert len(preserved) == 1
    assert (preserved[0]/'campaign-source/half-extracted.txt').read_text() == 'retained interrupted artifact'
    assert (preserved[0]/'campaign-native/failure.log').read_text() == 'retained failed configure'
    assert prepare(source, build) == result


def test_failed_build_retry_preserves_evidence_and_forces_stale_artifact_rebuild(tmp_path, monkeypatch):
    source = tmp_path/'source'
    git_repo(source)
    (source/'CMakeLists.txt').write_text('synthetic CPU build')
    (source/'python/bsqp').mkdir(parents=True)
    (source/'python/bsqp/__init__.py').write_text('')
    commit(source)
    build = tmp_path/'build'
    real_run = subprocess.run
    fail_next = [True]

    def fake_build(command, **kwargs):
        if command[0] != 'cmake':
            return real_run(command, **kwargs)
        native = build/'campaign-native'
        native.mkdir(exist_ok=True)
        (native/'CMakeCache.txt').write_text('synthetic CPU cache')
        if fail_next[0]:
            fail_next[0] = False
            raise subprocess.CalledProcessError(1, command)
        if '--build' in command:
            assert '--clean-first' in command
            (build/'campaign-source/python/bsqp/bsqpN32_tiago_right.synthetic.so').write_bytes(b'fresh CPU synthetic artifact')
            (native/'cuda_access_smoke').write_bytes(b'synthetic CPU readiness file')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build_isolated.subprocess, 'run', fake_build)
    with pytest.raises(subprocess.CalledProcessError):
        build_native(source, build, '/synthetic/python', '61')
    statuses = [json.loads(p.read_text())['status'] for p in (build/'preserved-build-attempts').glob('*/attempt.json')]
    assert statuses == ['failed']
    module = build/'campaign-source/python/bsqp/bsqpN32_tiago_right.synthetic.so'
    module.write_bytes(b'stale newer artifact')
    build_native(source, build, '/synthetic/python', '61')
    assert module.read_bytes() == b'fresh CPU synthetic artifact'
    snapshots = list((build/'preserved-build-attempts').glob('*/campaign-source/python/bsqp/*.so'))
    assert any(p.read_bytes() == b'stale newer artifact' for p in snapshots)
    statuses = [json.loads(p.read_text())['status'] for p in (build/'preserved-build-attempts').glob('*/attempt.json')]
    assert sorted(statuses) == ['completed', 'failed']


def test_plan_pins_exact_bytes_default_disabled_and_forbids_legacy_and_jetson(tmp_path):
    source = tmp_path/'source'
    git_repo(source)
    from make_plan import REQUIRED
    paths = set(REQUIRED+['tests/python/tiago_task_cases/short_reach.json'])
    for relative in paths:
        target = source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'{}')
    head = commit(source)
    plan = make_plan(head, source=source)
    experiment = plan['experiments'][0]
    assert experiment['source_commit'] == head and experiment['repeats'] == 1
    assert experiment['enabled'] is False and experiment['machines'] == ['gtx1070', 'rtx5060']
    assert 'campaign-source' in experiment['command'][1]
    assert experiment['cases'][0]['parameters']['offline_timing'] == 'solve_time'
    with pytest.raises(ValueError, match='Jetson'):
        make_plan(head, machines=['jetson'], source=source)
    with pytest.raises(ValueError, match='legacy'):
        make_plan('7187cf8842d0cc33184ade757c15e69c60b69bca', source=source)
    (source/'experiments/campaign/run_v01.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes differ'):
        make_plan(head, source=source)


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('machine', ['gtx1070', 'rtx5060'])
@pytest.mark.parametrize('mode',['solve_time','controller_dt'])
def test_worker_finalize_partial_raw_and_selection_without_GPU(tmp_path, monkeypatch, fail, machine, mode):
    monkeypatch.delenv('GATO_BUILD_DIR', raising=False)
    released = protocol()
    released['release_status'] = 'released after reviewed clock decision'
    protocol_path = tmp_path/'synthetic_released_protocol.json'
    protocol_path.write_text(json.dumps(released))
    monkeypatch.setattr(run_v01, 'PROTOCOL', protocol_path)
    source = tmp_path/'source'
    git_repo(source)
    paths = ['python/bsqp/interface.py', 'python/bsqp/common.py', 'python/bsqp/offline_mpc.py',
             'python/bsqp/mpc_controller.py',
             'tiago_src/gato_tiago/tiago_mpc_controller.py', 'tiago_src/gato_tiago/config.py',
             'experiments/campaign/run_v01.py', 'experiments/campaign/t01_protocol.py', 'experiments/campaign/raw.py',
             'experiments/campaign/evaluate.py', protocol()['frozen_fixture']['path'],
             protocol()['urdf']['path']]
    for relative in paths:
        target = source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
    native_path = source/'python/bsqp/bsqpN32_tiago_right.synthetic.so'
    native_path.write_bytes(b'synthetic CPU file, never imported as a library')
    commit(source)
    fake_native = types.SimpleNamespace(__file__=str(native_path))

    class FakeController:
        def __init__(self, model, path, **kwargs):
            self.solver = types.SimpleNamespace(lib=fake_native)
            self.solver_params = kwargs['solver_params']

        def run_mpc_goals(self, initial, goals, **kwargs):
            assert kwargs['record_raw'] and kwargs['offline_timing'] == mode
            callback = kwargs['record_callback']
            solve = {'status':'started', 'category':'startup/cold', 'state':initial.copy(),
                     'warm_start':np.zeros((1, 665)), 'reference':np.tile(goals[0], 32), 'solve_index':0}
            self.last_run_stats = {'raw_solves':[solve], 'raw_execution':[],
                                   'startup_wall_solve_time_s':.001,'startup_native_solve_time_ms':.5,
                                   'goal_outcomes':['reached'], 'wall_solve_times_s':[],
                                   'native_solve_times_ms':[], 'update_intervals_s':[.003]}
            callback('solve_started', solve)
            solve.update(status='completed', output=np.array([[np.nan]]),
                         wall_solve_time_s=.001, native_solve_time_ms=.5)
            callback('solve_completed', solve)
            solve['best_trajectory_id'] = 0  # Assigned after observer, as production.
            endpoint = initial.copy()
            endpoint[:7] += np.asarray(protocol()['endpoint_delta_rad'])
            step = {'time_s':0., 'step_dt_s':.003, 'state_before':initial.copy(),
                    'control':np.zeros(7), 'status':'started'}
            self.last_run_stats['raw_execution'].append(step)
            callback('step_started', step)
            step.update(status='completed', state_after=endpoint)
            callback('step_completed', step)
            if fail:
                raise RuntimeError('synthetic controller failure after observed step')
            return endpoint, self.last_run_stats

    fake_package = types.ModuleType('gato_tiago')
    fake_package.tiago_mpc_controller = types.SimpleNamespace(MPC_GATO=FakeController)
    monkeypatch.setitem(sys.modules, 'gato_tiago', fake_package)
    monkeypatch.setitem(sys.modules, 'bsqp.bsqpN32_tiago_right', fake_native)
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(__version__='synthetic',
                        cuda=types.SimpleNamespace(is_available=lambda:False)))
    monkeypatch.setattr(run_v01, 'ROOT', source)
    monkeypatch.setenv('GATO_SOURCE_DIR', str(source))
    monkeypatch.setenv('GATO_MACHINE', machine)
    output = tmp_path/'output'
    parameters={'protocol':'V01','offline_timing':mode}
    if mode=='controller_dt':
        failed=tmp_path/'failed.json';failed.write_text(json.dumps(dict(timing_mode='B',execution_backend='internal',outcome='task_failed',execution_error=None,task_success=False,physical_pass=True,machine=machine,configuration=released)))
        parameters['diagnostic_after_failed_B']={'path':failed.name,'sha256':sha(failed)}
    assert run_case({'parameters':parameters},output,tmp_path)==int(fail)
    metrics = json.loads((output/'metrics.json').read_text())
    assert metrics['task_success'] == (not fail)
    assert metrics['physical_pass'] and metrics['solver_numerical_pass'] is None
    assert metrics['outcome'] == ('execution_failed' if fail else 'task_passed')
    if mode=='controller_dt':
        assert metrics['configuration']==diagnostic_protocol(released)
        assert 'unpaid' in metrics['interval_alignment'] and 'ignored' in metrics['timing_scope']
        assert metrics['startup_wall_solve_time_s']==.001 and metrics['startup_native_solve_time_ms']==.5
        assert metrics['configuration']['approved_clock_convention']=='mode A fixed feedback clock; computation time ignored'
    with np.load(output/'execution.npz') as execution:
        np.testing.assert_allclose(execution['t'], [0., .003])
        assert len(execution['x']) == 2 and len(execution['u']) == 1
    events = [json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    final = next(event for event in events if event['event'] == 'final_raw_solves')
    assert final['record']['solves'][0]['best_trajectory_id'] == 0
    with np.load(output/final['raw_file']) as values:
        assert np.isnan(values['record__solves__0__output'][0, 0])


def test_direct_worker_held_B_negative_control_precedes_imports_or_constructor(tmp_path, monkeypatch):
    held = protocol()
    held['release_status'] = 'held synthetic negative control'
    held_path = tmp_path/'held_protocol.json'
    held_path.write_text(json.dumps(held))
    monkeypatch.setattr(run_v01, 'PROTOCOL', held_path)
    def unexpected_import(name, *args, **kwargs):
        raise AssertionError('held worker attempted native import')
    monkeypatch.setattr(run_v01.importlib, 'import_module', unexpected_import)
    with pytest.raises(ValueError, match='execution held'):
        run_case({'parameters':{'protocol':'V01', 'offline_timing':'solve_time'}}, tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('machine', ['jetson', 'local', 'gazebo', None])
def test_desktop_release_rejects_unreleased_machine_before_native_import(tmp_path, monkeypatch, machine):
    if machine is None:
        monkeypatch.delenv('GATO_MACHINE', raising=False)
    else:
        monkeypatch.setenv('GATO_MACHINE', machine)
    def unexpected_import(name, *args, **kwargs):
        raise AssertionError('unreleased machine attempted native import')
    monkeypatch.setattr(run_v01.importlib, 'import_module', unexpected_import)
    with pytest.raises(ValueError, match='release_machines'):
        run_case({'parameters':{'protocol':'V01', 'offline_timing':'solve_time'}}, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_desktop_release_retains_canonical_protocol_evidence_and_unknown_certificate():
    p = protocol()
    assert p['release_status'] == 'released after reviewed clock decision'
    assert p['release_machines'] == ['gtx1070', 'rtx5060']
    assert p['Q02_status'].startswith('FAILED original frozen protocol')
    assert p['solver_numerical_pass'] is None
    assert p['repeat_policy'].startswith('one fixed attempt per machine')
    assert p['sim_dt_s'] == .003 and p['offline_timing'] == 'solve_time'
    for relative, expected in p['reviewed_controller_sha256'].items():
        assert sha(ROOT/relative) == expected
    for key in ['required_suite_source_receipt', 'required_suite_log', 'required_suite_summary', 'Q02_saved_data_audit']:
        proof = p['release_evidence'][key]
        # Receipts refer to root-retained evidence, not portable dependencies
        # required in every runner checkout. Root independently audits bytes.
        assert Path(proof['path']).is_absolute()
        assert len(proof['sha256']) == 64 and all(c in '0123456789abcdef' for c in proof['sha256'])


def test_release_source_guard_rejects_changed_controller_before_native_import(tmp_path, monkeypatch):
    monkeypatch.setenv('GATO_MACHINE', 'gtx1070')
    monkeypatch.setattr(run_v01, 'sha', lambda path:'0'*64)
    def unexpected_import(name, *args, **kwargs):
        raise AssertionError('changed source attempted native import')
    monkeypatch.setattr(run_v01.importlib, 'import_module', unexpected_import)
    with pytest.raises(ValueError, match='controller source changed'):
        run_case({'parameters':{'protocol':'V01', 'offline_timing':'solve_time'}}, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_T01_analytic_speed_and_retained_witness_screening():
    for period in [24., 18., 12.]:
        phase = np.linspace(0., 2*np.pi, 100001)
        omega = 2*np.pi/period
        numerical = np.max(np.hypot(.16*omega*np.sin(phase), .16*omega*np.cos(2*phase)))
        assert figure8_speed(period)['maximum_reference_speed_m_s'] == pytest.approx(numerical)
    p = json.loads((CAMPAIGN/'inputs/t01_proposal.json').read_text())
    assert sha(CAMPAIGN/'propose_t01.py') == p['generator_sha256']
    assert sha(CAMPAIGN/'inputs/t01_witness.npz') == p['witness_sha256']
    assert all(goal['sampled_inverse_dynamics_pass'] and goal['outside_5cm_start_region'] for goal in p['reaching'])
    assert p['proposed_repeats']['jetson'] is None
    assert p['approval_record']['exact_text'] == 'Yes but try this first. If this does not work also try with 1ms.'
    assert p['conditional_plant_variant']['qualifies_canonical_3ms'] is False
    assert p['provisional_tracking_gates_frozen']
