"""CPU protocol, stage selection, query and partial-recording checks."""
import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from adapters import resource_profiles as adapter, resource_profiles_common as C
from recording import iter_events
import resource_profiles_plan

ROOT=Path(__file__).resolve().parents[1]
INPUTS=ROOT/'experiments/inputs/f01c'


def test_n128_extension_bank_and_exact_three_case_plan(runtime,tmp_path):
    source,_=runtime
    from adapters import f01ab
    C.E=f01ab.load(source,'experiments/frequency_campaign/episode.py','profile_bank_test')
    folder=ROOT/'experiments/inputs/f05_n128_f01ab'
    original=json.loads((ROOT/'experiments/inputs/f01ab/bank.json').read_text())
    assert (folder/'arrays.npz').read_bytes()==(ROOT/'experiments/inputs/f01ab/arrays.npz').read_bytes()
    with np.load(folder/'arrays.npz',allow_pickle=False) as data:
        for case in C.stage_extension_cases()[:2]:
            meta,arrays=C.bank_f01ab(folder,C.sha(folder/'SHA256SUMS.json'),case)
            assert meta['preset']==original['presets']['reaching']
            measured=C.request(meta,arrays,1);warmup=C.request(meta,arrays,1,warmup=True)
            assert measured['warm'].shape==warmup['warm'].shape==(1,128*21-7)
            assert measured['reference'].shape==warmup['reference'].shape==(1,128*6)
            np.testing.assert_array_equal(measured['state'][0],data['starts'][0])
            np.testing.assert_array_equal(measured['reference'][0],np.tile(data['goals'][0],128))
            np.testing.assert_array_equal(measured['warm'],C.E.cold_seed(data['starts'][0],128))
            np.testing.assert_array_equal(warmup['warm'],C.E.cold_seed(data['warmup_state'],128))
    plan=resource_profiles_plan.prepare_stage_extension(tmp_path/'extension.json')
    entry=plan['experiments'][0]
    assert entry['source_commit']=='852b0cbcfe6e56fe5337a3b6e5fe8e777f959b77'
    assert [adapter.validate_case(c) for c in entry['cases']]==C.stage_extension_cases()
    assert plan['preparation']['native_calls']==6
    assert len(C.cases())==38


def frozen(name):
    return subprocess.check_output(['git','-C',str(ROOT),'show',adapter.SOURCE+':'+name])


@pytest.fixture
def runtime(tmp_path):
    source=tmp_path/'source'
    for name in ('experiments/frequency_campaign/factory.py','experiments/frequency_campaign/episode.py','python/bsqp/common.py'):
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(frozen(name))
    module=tmp_path/'module.fake.so';module.write_bytes(b'CPU injected module; never imported')
    return source,module


def wrapped(parameters):return dict(parameters=dict(parameters,timing_mode='A',backend='internal'))


def build(log,failure=None):
    def create_factory(context,bank,events):
        caseflags=context['selected_module']['flags']
        class Solver:
            def __init__(self,SQP):
                self.cap=SQP;self.solver=SimpleNamespace(solve=self.native_solve)
            def native_solve(self,warm,dt,state,reference):
                output=warm.copy()
                if failure=='nonfinite':output[0,0]=np.nan
                reply=dict(XU=output,sqp_time_us=1000.)
                if caseflags['GATO_EXPERIMENT_STAGE_TIMING'] and failure!='profile_missing':
                    reply['experiment_profile_stages']=dict(schema_version=1,units='stream_elapsed_us',
                        pcg_times_us_is_placeholder=True,windows=[dict(stage=name,sqp_iteration=i,
                            elapsed_us=10.,unavailable_reason=None) for name,i in C.stage_layout(self.cap)])
                if failure=='unexpected_profile':reply['experiment_profile_stages']={}
                return reply
            def reset_dual(self):log.append('dual')
            def reset_rho(self):log.append('rho')
            def solve(self,state,reference,warm):
                log.append(('solve',state.copy(),reference.copy(),warm.copy()))
                reply=self.solver.solve(warm,.03,state,reference)
                return reply['XU'],reply['sqp_time_us']
            def get_stats(self):return dict(sqp_iters=np.array([self.cap]),pcg_iters=np.array([1]))
        def create(B,SQP,PCG):
            log.append(('create',B,SQP,PCG))
            if failure=='constructor':raise RuntimeError('setup unavailable')
            params=dict(bank['preset'],max_sqp_iters=SQP,max_pcg_iters=PCG)
            return C.F.Owned(SimpleNamespace(solver=Solver(SQP),N=32,dt=.03,batch_size=B,solver_params=params),events)
        return create
    return create_factory


def test_exact_shared_helpers_and_plan_coverage(tmp_path):
    before=(ROOT/'experiments/campaign.json').read_bytes()
    original=ast.parse(frozen('experiments/resource_profile_campaign/common.py'))
    actual=ast.parse(Path(C.__file__).read_text())
    original_funcs={v.name:ast.dump(v) for v in original.body if isinstance(v,ast.FunctionDef)}
    actual_funcs={v.name:ast.dump(v) for v in actual.body if isinstance(v,ast.FunctionDef)}
    for name in ('cases','schedule','stage_layout','validate_profile','request','bank'):
        assert original_funcs[name]==actual_funcs[name]
    for machine in ('rtx5060','jetson'):
        plan=resource_profiles_plan.prepare(tmp_path/(machine+'.json'),machine)
        experiments=plan['experiments']
        cases=[adapter.validate_case(c) for e in experiments for c in e['cases']]
        assert len(cases)==38
        assert sorted(map(lambda c:json.dumps(c,sort_keys=True),cases))==sorted(map(lambda c:json.dumps(c,sort_keys=True),C.cases()))
        assert experiments[0]['machines']==experiments[1]['machines']==[machine]
        assert experiments[2]['machines']==[machine] and experiments[2]['prepared_machines']==[machine]
        assert 'task_suite.py' in experiments[2]['preflight']['command'][1]
        assert len([c for c in cases if c['arm']=='query'])==8
        assert len([c for c in cases if c['arm']=='first_response'])==18
        assert len([c for c in cases if c['arm']=='profile'])==6
        assert [c['duration_s'] for c in cases if c['arm']=='sustained']==[120]*6
        assert experiments[0]['build']==experiments[1]['build']
        assert experiments[2]['build']!=experiments[0]['build']
    assert (ROOT/'experiments/campaign.json').read_bytes()==before


def test_query_is_zero_solve_retains_unsupported_resources(runtime,tmp_path):
    source,module=runtime
    case=C.cases()[0]
    status=dict(code=0)
    payload=dict(N=case['N'],batch_size=case['B'],**{k:status for k in
        ('device_status','properties_status','driver_status','runtime_status')},kernels=[
            dict(kernel=k,attributes_status=status,occupancy_status=dict(code=1),
                 fits_default_limits_prediction=False,observed_launch=None)
            for k in ('setup_kkt','merit','pcg','schur1','schur2','dz','line_search')])
    result=adapter.run_case(wrapped(case),tmp_path/'query',source,module,INPUTS,'cpu',
                            resource_query=lambda context,p:payload)
    assert result['status']=='completed' and result['solver_calls']==result['native_calls']==0
    returns=[e['record'] for e in iter_events(tmp_path/'query') if e['event']=='query_returned']
    assert returns[0]['payload']['kernels'][0]['fits_default_limits_prediction'] is False


@pytest.mark.parametrize('arm,count',[('first_response',1),('profile',2)])
def test_first_response_profile_warmups_reset_and_clocks(runtime,tmp_path,arm,count):
    source,module=runtime;log=[]
    case=next(c for c in C.cases() if c['arm']==arm and c['B']==16 and c['SQP']==5)
    result=adapter.run_case(wrapped(case),tmp_path/arm,source,module,INPUTS,'cpu',builder=build(log))
    assert result['status']=='completed'
    assert result['native_calls']==result['completed_calls']==count
    assert result['measured_calls']==1 and log.count('dual')==log.count('rho')==count
    started=[e['record'] for e in iter_events(tmp_path/arm) if e['event']=='solve_started']
    assert [r['category'] for r in started]==(['warmup','measured'] if arm=='profile' else ['measured'])
    native=[e['record']['native_reply'] for e in iter_events(tmp_path/arm) if e['event']=='native_return']
    assert ('experiment_profile_stages' in native[0])==(arm=='profile')
    assert 'XU' not in native[0] and native[0]['sqp_time_us']==1000.
    assert native[0]['omitted_array_quality']['XU']['finite'] is True
    assert all('warm_start' not in r and 'state' not in r and 'reference' not in r for r in started)


def test_sustained_checks_boundary_before_next_request_and_retains_overrun(runtime,tmp_path):
    source,module=runtime;log=[]
    case=next(c for c in C.cases() if c['arm']=='sustained' and c['B']==1)
    class Clock:
        def __init__(self):self.value=0
        def __call__(self):self.value+=1;return self.value
    result=adapter.run_case(wrapped(case),tmp_path/'sustained',source,module,INPUTS,'cpu',
                            builder=build(log),clock=Clock())
    assert result['status']=='completed'
    assert result['sustained_wall_s']>=120 and result['boundary_overrun_s']>=0
    assert result['native_calls']==result['measured_calls']+1
    assert result['coverage']==[dict(input_index=C.schedule(1,i)[0],candidate_ids=C.schedule(1,i)[1])
                                for i in range(result['measured_calls'])]


@pytest.mark.parametrize('failure',['constructor','nonfinite','profile_missing','unexpected_profile'])
def test_failures_retain_native_quality_and_metrics(runtime,tmp_path,failure):
    source,module=runtime;log=[]
    arm='profile' if failure=='profile_missing' else 'first_response'
    case=next(c for c in C.cases() if c['arm']==arm)
    result=adapter.run_case(wrapped(case),tmp_path/failure,source,module,INPUTS,'cpu',builder=build(log,failure))
    assert result['outcome']==result['status']=='execution_failed'
    assert result['execution_error'] and result['completed_calls']==0
    assert result['native_calls']==(0 if failure=='constructor' else 1)
    returns=[e['record']['native_reply'] for e in iter_events(tmp_path/failure) if e['event']=='native_return']
    if failure=='nonfinite':
        assert 'XU' not in returns[0]
        quality=returns[0]['omitted_array_quality']['XU']
        assert quality['finite'] is False and quality['nonfinite_count']==1
    assert (tmp_path/failure/'SHA256SUMS.json').exists()


def test_saved_analysis_reads_chunks_and_retains_failed_call(runtime,tmp_path):
    from analysis.resource_profiles import load
    source,module=runtime;log=[]
    case=next(c for c in C.cases() if c['arm']=='profile' and c['SQP']==5)
    adapter.run_case(wrapped(case),tmp_path/'profile',source,module,INPUTS,'cpu',builder=build(log))
    failing=next(c for c in C.cases() if c['arm']=='first_response')
    adapter.run_case(wrapped(failing),tmp_path/'failed',source,module,INPUTS,'cpu',builder=build([], 'nonfinite'))
    index=tmp_path/'index.json'
    index.write_text(json.dumps(dict(attempts=[dict(experiment='F05-stageON',attempt='profile'),
                                              dict(experiment='F05-stageOFF',attempt='failed')])))
    result=load(index)
    assert len(result['calls'])==3 and len(result['stages'])==74
    assert sum(r['status']=='failed' for r in result['calls'])==1
    assert sum(r['category']=='warmup' for r in result['calls'])==1
    assert result['attempts'][1]['metrics']['execution_error'] is not None
