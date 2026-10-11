"""Bounded F01c checks: scientific reuse, exact coverage and retained failures."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from adapters import f01c as adapter
from adapters import f01c_collect as science
from recording import iter_events, sha
import f01c_plan

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'experiments/inputs/f01c'


def frozen(commit, name):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', commit + ':' + name])


def case(B=16, SQP=5, PCG=100):
    return {'parameters':dict(N=32, B=B, SQP=SQP, PCG=PCG,
        diagnostics=(B, SQP, PCG)==(16, 5, 100), timing_mode='A', backend='internal')}


@pytest.fixture
def runtime(tmp_path):
    source = tmp_path/'source'
    for name in ('experiments/frequency_campaign/episode.py',
                 'experiments/frequency_campaign/factory.py', 'python/bsqp/common.py'):
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(frozen(adapter.NATIVE_SOURCE, name))
    module = tmp_path/'bsqpN32_tiago_right.fake.so'
    module.write_bytes(b'CPU injection only; never imported')
    return source, module


def builder(log, failure=None):
    def build(source, module, bank, events, F):
        class Native:
            def solve(self, xu, timestep, state, reference):
                log.append(('solve', xu.copy(), state.copy(), reference.copy()))
                output = xu.copy()
                if failure == 'nonfinite':output[0, 0] = np.nan
                if failure == 'malformed':output = output[:, :-1]
                return dict(XU=output, sqp_time_us=1000., diagnostic=np.array([np.nan, np.inf]))

        class Solver:
            def __init__(self):self.solver=Native()
            def reset_dual(self):log.append('dual')
            def reset_rho(self):log.append('rho')
            def set_f_ext_B(self, force):log.append(('force', force.copy()))
            def solve(self, state, reference, warm):
                reply = self.solver.solve(warm, .03, state, reference)
                return reply['XU'], reply['sqp_time_us']
            def get_stats(self):
                if failure == 'stats':raise RuntimeError('telemetry decode failed')
                return dict(sqp_iters=np.array([1]), pcg_iters=np.array([3]))

        def create(B, SQP, PCG):
            log.append(('create', B, SQP, PCG))
            if failure == 'constructor':raise RuntimeError('native setup unavailable')
            params=dict(bank['preset'], max_sqp_iters=SQP, max_pcg_iters=PCG)
            return F.Owned(SimpleNamespace(solver=Solver(), N=32, dt=.03,
                                          batch_size=B, solver_params=params), events)
        return create
    return build


def test_transplant_keeps_scientific_functions_and_frozen_bank():
    original = frozen(adapter.COLLECTOR_SOURCE, 'experiments/batch_timing_campaign/collect.py')
    parsed = ast.parse(original)
    expected = {v.name:ast.dump(v) for v in parsed.body if isinstance(v, ast.FunctionDef)}
    actual = {v.name:ast.dump(v) for v in ast.parse(Path(science.__file__).read_text()).body if isinstance(v, ast.FunctionDef)}
    for name in ('jobs','collect'):assert actual[name] == expected[name]
    diagnostic = Path(adapter.__file__).with_name('f01c_diagnostics.py')
    assert diagnostic.read_bytes() == frozen(adapter.COLLECTOR_SOURCE, 'experiments/batch_timing_campaign/diagnostics.py')
    for name in ('bank.json','arrays.npz','preparation.json','seed_diagnostics.json','SHA256SUMS.json'):
        assert (INPUTS/name).read_bytes() == frozen(adapter.COLLECTOR_SOURCE, 'experiments/batch_timing_campaign/inputs/'+name)


def test_plan_preserves_coverage_and_machine_exclusions(tmp_path):
    campaign = (ROOT/'experiments/campaign.json').read_bytes()
    plan = f01c_plan.prepare(tmp_path/'plan.json')
    experiment = plan['experiments'][0]
    bank, _ = adapter.load_bank(INPUTS)
    assert experiment['machines'] == ['rtx5060']
    assert len(experiment['cases']) == 15
    counts = [len(science.jobs(bank, **{key:p[key] for key in ('B','SQP','PCG','diagnostics')}))
              for c in experiment['cases'] for p in [adapter.validate_case(c)]]
    assert sum(counts) == 1695
    assert sum(c['parameters']['diagnostics'] for c in experiment['cases']) == 1
    assert (ROOT/'experiments/campaign.json').read_bytes() == campaign
    existing=json.loads(campaign)['experiments'][0]
    assert experiment['build'] == existing['build']
    assert experiment['source_commit'] == adapter.NATIVE_SOURCE


def test_completed_worker_retains_exact_candidate_partition_reset_and_raw(runtime, tmp_path):
    source,module=runtime
    log=[]
    result=adapter.run_case(case(B=8, SQP=1),tmp_path/'result',source,module,INPUTS,'cpu',builder=builder(log))
    assert result['outcome']=='timing_completed'
    assert result['execution_error'] is None
    assert result['native_calls']==result['attempted_calls']==37
    assert len(result['collections'])==18 and all(r['complete'] for r in result['collections'])
    assert log.count('dual')==log.count('rho')==37
    assert sum(isinstance(v,tuple) and v[0]=='create' for v in log)==37
    force=[v[1] for v in log if isinstance(v,tuple) and v[0]=='force']
    assert all(v.shape==(8,6) and v.dtype==np.float32 and not v.any() for v in force)
    bank,arrays=adapter.load_bank(INPUTS)
    solves=[e['record'] for e in iter_events(tmp_path/'result') if e['event']=='solve_started']
    for r in solves[1:]:
        task=next(t for t in bank['inputs'] if t['id']==r['input_id'])
        np.testing.assert_array_equal(r['warm_start'],arrays['pools'][task['start_index'],r['candidate_ids']])
    assert result['task_success'] is result['physical_pass'] is result['solver_numerical_pass'] is None
    provenance=json.loads((tmp_path/'result/provenance.json').read_text())
    assert provenance['module_sha256']==sha(module)
    assert provenance['original_collector_source_commit']==adapter.COLLECTOR_SOURCE
    assert result['evidence_kind']=='synthetic_test'


@pytest.mark.parametrize('failure',['constructor','nonfinite','malformed','stats'])
def test_failed_worker_preserves_output_status_and_missing_calls(runtime,tmp_path,failure):
    source,module=runtime
    log=[]
    output=tmp_path/failure
    result=adapter.run_case(case(),output,source,module,INPUTS,'cpu',builder=builder(log,failure))
    assert result['outcome']=='execution_failed' and result['execution_error']
    assert result['attempted_calls']==1 and result['calls'][0]['status']=='failed'
    assert len(result['missing_calls'])==24
    assert len(result['collections'])==18 and not any(c['complete'] for c in result['collections'])
    events=list(iter_events(output))
    returns=[e['record'] for e in events if e['event']=='native_return']
    if failure=='constructor':
        assert not returns and result['native_calls']==0
    else:
        assert len(returns)==1 and result['native_calls']==1
        payload=returns[0]['native_reply']
        assert np.isnan(payload['diagnostic'][0]) and np.isposinf(payload['diagnostic'][1])
        if failure=='nonfinite':assert np.isnan(payload['XU'][0,0])
        if failure=='malformed':assert payload['XU'].shape==(16,664)
    assert json.loads((output/'metrics.json').read_text())['outcome']=='execution_failed'
    assert (output/'SHA256SUMS.json').is_file()


def test_correspondence_schedule_and_invalid_cases():
    bank,_=adapter.load_bank(INPUTS)
    jobs=science.jobs(bank,16,5,100,True)
    assert len(jobs)==25
    assert [r['candidate_ids'] for r in jobs[-6:-3]]==[[0,0]+list(range(2,16))]*3
    assert [r['candidate_ids'] for r in jobs[-3:]]==[list(reversed(range(16)))]*3
    for changes in ({'timing_mode':'B'}, {'backend':'gazebo'}, {'SQP':2}, {'diagnostics':False}):
        invalid=case();invalid['parameters'].update(changes)
        with pytest.raises(ValueError):adapter.validate_case(invalid)
