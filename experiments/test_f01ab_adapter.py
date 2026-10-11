"""CPU wiring checks: retained bytes, fresh episodes, and failure continuation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from adapters import f01ab as adapter
from recording import iter_events, sha


@pytest.fixture
def setup(tmp_path):
    source, inputs = tmp_path / 'source', tmp_path / 'inputs'
    inputs.mkdir()
    for name in ['experiments/frequency_campaign/episode.py',
                 'experiments/frequency_campaign/factory.py', 'python/bsqp/common.py']:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# scientific code placeholder for injected worker test\n')
    episodes = [dict(id=name, kind='reaching', start_index=0, goal_indices=[0, 1],
                     goal_switch_before_update=10, updates=20,
                     categories=['cold'] + ['warm'] * 9 + ['goal_switch'] + ['warm'] * 9)
                for name in ('first', 'second')]
    bank = dict(N=[16], SQP=[1], PCG=[100], batch_size=1, dt_s=.03, sim_dt_s=.003,
                offline_timing='controller_dt', variants={'test': {'episodes': episodes}},
                presets={'reaching': {'qd_cost': .1, 'max_sqp_iters': 5, 'max_pcg_iters': 120}},
                initial_guess_policy='constant-current-state', warm_start_shift_s=.03)
    (inputs / 'bank.json').write_text(json.dumps(bank))
    np.savez(inputs / 'arrays.npz', starts=np.zeros((1, 14), dtype=np.float32),
             goals=np.zeros((2, 6)), warmup_state=np.zeros(14, dtype=np.float32),
             warmup_goal=np.zeros(6))
    module = tmp_path / 'bsqpN16_tiago_right.fake.so'
    module.write_bytes(b'test module bytes, never imported')
    case = {'parameters': dict(variant='test', N=16, SQP=1, PCG=100,
                               timing_mode='A', backend='internal')}
    return source, inputs, module, case, bank


def fake_science(monkeypatch, calls, *, warmup_fail=False):
    class Owner:
        def __init__(self, cell, params):
            self.N, self.dt, self.batch_size = cell['N'], .03, 1
            self.solver_params = params
            self.solver = SimpleNamespace(reset_dual=lambda: calls.append('reset_dual'),
                                          reset_rho=lambda: calls.append('reset_rho'))
        def update_force_batch(self, q):
            calls.append(('force', q.copy()))
        def close(self):
            calls.append('close')

    class Events:
        def __init__(self, recorder):
            self.recorder, self.native_calls = recorder, 0
        def __call__(self, kind, record):
            self.recorder(kind, record)

    class Factory:
        def __init__(self, context, bank, events, builder):
            calls.append(('context', context))
            self.bank = bank
        def __call__(self, kind, cell):
            calls.append(('create', kind))
            params = dict(self.bank['presets'][kind], max_sqp_iters=cell['SQP'], max_pcg_iters=cell['PCG'])
            return Owner(cell, params)

    def timed_solve(owner, state, reference, warm, metadata, emit):
        calls.append(('warmup', state.copy(), reference.copy(), warm.copy(), metadata))
        emit('solve_started', dict(metadata, state=state))
        if warmup_fail:
            raise RuntimeError('warmup unavailable')

    def collect_episode(owner, episode, arrays, cell, emit, integrate, observe):
        calls.append(('episode', episode['id'], episode['categories']))
        emit('solve_returned', dict(episode_id=episode['id'], output=np.array([np.nan, np.inf])))
        return dict(episode_id=episode['id'], execution_error='native error' if episode['id'] == 'first' else None,
                    task_success=False, physical_pass=False)

    F = SimpleNamespace(Events=Events, Factory=Factory)
    E = SimpleNamespace(cold_seed=lambda state, N: np.zeros((1, N * 21 - 7)),
                        timed_solve=timed_solve, collect_episode=collect_episode)
    monkeypatch.setattr(adapter, 'load', lambda source, relative, name: F if name == 'f01_factory' else E)


def test_continues_after_failed_episode_and_preserves_records(setup, monkeypatch, tmp_path):
    source, inputs, module, case, bank = setup
    calls = []
    fake_science(monkeypatch, calls)
    output = tmp_path / 'result'
    bank_before = sha(inputs / 'arrays.npz')
    result = adapter.run_case(case, output, source, module, inputs, 'test-machine',
                              builder=object(), integrator=lambda owner: object(), observer=object())
    assert result['outcome'] == 'execution_failed'
    assert result['execution_error'] == 'native error'
    assert [r['episode_id'] for r in result['episodes']] == ['first', 'second']
    assert result['missing_episodes'] == []
    assert calls.count('close') == 3  # disposable warmup plus both fresh episodes
    warmup = next(c for c in calls if isinstance(c, tuple) and c[0] == 'warmup')
    assert warmup[1].dtype == np.float32 and warmup[3].shape == (1, 329)
    assert warmup[4]['excluded'] is True
    assert [c[2] for c in calls if isinstance(c, tuple) and c[0] == 'episode'] == [
        bank['variants']['test']['episodes'][0]['categories'],
        bank['variants']['test']['episodes'][1]['categories']]
    events = list(iter_events(output))
    returned = [e for e in events if e['event'] == 'solve_returned']
    assert len(returned) == 2 and np.isnan(returned[0]['record']['output'][0])
    assert np.isposinf(returned[0]['record']['output'][1])
    provenance = json.loads((output / 'provenance.json').read_text())
    assert provenance['module_sha256'] == sha(module)
    assert provenance['input_sha256']['arrays.npz'] == bank_before == sha(inputs / 'arrays.npz')
    assert result['evidence_kind'] == 'synthetic_test'


def test_shared_warmup_failure_retains_partial_data(setup, monkeypatch, tmp_path):
    source, inputs, module, case, _ = setup
    calls = []
    fake_science(monkeypatch, calls, warmup_fail=True)
    result = adapter.run_case(case, tmp_path / 'failed', source, module, inputs,
                              builder=object(), integrator=lambda owner: object(), observer=object())
    assert 'warmup unavailable' in result['execution_error']
    assert calls.count('close') == 1
    assert not any(c[0] == 'episode' for c in calls if isinstance(c, tuple))
    assert [e['event'] for e in iter_events(tmp_path / 'failed')] == ['solve_started', 'execution_failed']


def test_basic_shape_validation_and_module_discovery(setup, tmp_path):
    _, inputs, module, case, _ = setup
    bank, arrays = adapter.load_bank(inputs, case['parameters'])
    assert arrays['starts'].dtype == np.float32
    build = tmp_path / 'build'
    (build / 'modules').mkdir(parents=True)
    selected = build / 'modules/bsqpN16_tiago_right.cpython-test.so'
    selected.write_bytes(b'module')
    assert adapter.resolve_module(build, 16) == selected.resolve()
    assert adapter.resolve_module(None, 16, module) == module.resolve()
    with pytest.raises(ValueError, match='Expected one'):
        adapter.resolve_module(build, 32)
    np.savez(inputs / 'arrays.npz', starts=np.zeros((1, 13)), goals=np.zeros((2, 6)),
             warmup_state=np.zeros(14), warmup_goal=np.zeros(6))
    with pytest.raises(ValueError, match='shape'):
        adapter.load_bank(inputs, case['parameters'])
