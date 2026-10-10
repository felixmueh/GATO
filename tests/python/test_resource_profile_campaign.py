"""Injected CPU-only metadata and durable boundary tests; no native imports."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.resource_profile_campaign import common as C, run, release as R
from experiments.resource_profile_campaign.audit_scope import audit
from experiments.resource_profile_campaign.build_isolated import commands


def make_bank(tmp_path):
    bank=tmp_path/'bank';bank.mkdir()
    arrays=dict(starts=np.zeros((2,14),'<f4'),pools=np.zeros((2,16,665),'<f4'),references=np.zeros((6,192),'<f4'),
        warmup_state=np.zeros(14,'<f4'),warmup_pool=np.zeros((16,665),'<f4'),warmup_reference=np.zeros(192,'<f4'))
    np.savez(bank/'arrays.npz',**arrays)
    meta=dict(schema='F01c-six-input-N32-v1',inputs=[dict(id=f'start-{i//3}-goal-{i%3+1}',start_index=i//3,goal_index=i%3) for i in range(6)],
        preset=dict(qd_cost=.1,vel_lim_cost=.1),approved_budgets=[list(x) for x in C.BUDGETS])
    C.write_json(bank/'bank.json',meta)
    C.write_json(bank/'SHA256SUMS.json',{f:C.sha(bank/f) for f in ('bank.json','arrays.npz')})
    return bank,C.sha(bank/'SHA256SUMS.json')


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def advance(self,value):self.now+=value


def profile(cap, early=False):
    return dict(schema_version=1,units='stream_elapsed_us',pcg_times_us_is_placeholder=True,
        windows=[dict(stage=name,sqp_iteration=i,elapsed_us=None if early and i is not None and (i>0 or name in C.STAGES[5:]) else 0.,
            unavailable_reason='not_reached' if early and i is not None and (i>0 or name in C.STAGES[5:]) else None) for name,i in C.stage_layout(cap)])


def injected(tmp_path,monkeypatch,case,*,fail=None):
    bank,digest=make_bank(tmp_path);state=tmp_path/'family.json';C.write_json(state,dict(status='ready',machine='synthetic-machine',release_manifest_sha256='synthetic-manifest',canonical_receipt='preserve-me'))
    context=dict(machine='synthetic-machine',manifest_sha256='synthetic-manifest',source_commit='a'*40,
        selected_module=dict(path='synthetic-module',flags={'GATO_EXPERIMENT_STAGE_TIMING':case['stage_timing']}),
        bank=dict(path=str(bank),sha256=digest))
    def validate(*args):
        if json.loads(state.read_text())['status']!='ready':raise RuntimeError('Family held/interrupted')
        return context
    monkeypatch.setattr(R,'validate',validate)
    clock=Clock();owners=[]
    class Native:
        def solve(self,xu,dt,state,ref):
            clock.advance(11)
            if fail=='native':raise RuntimeError('injected-native-error')
            reply=dict(XU=xu.copy(),sqp_time_us=11000000,pcg_times_us=np.zeros(case['SQP'],np.float32),
                pcg_iters=np.zeros((case['SQP'],case['B']),np.int32))
            if case['stage_timing']:reply['experiment_profile_stages']=profile(case['SQP'],True)
            return reply
    class Public:
        def __init__(self):self.solver=Native()
        def reset_dual(self):pass
        def reset_rho(self):pass
        def solve(self,state,reference,warm):
            reply=self.solver.solve(warm,.03,state,reference)
            if fail=='normalize':raise RuntimeError('injected-normalization-error')
            self.reply=reply;return reply['XU'],reply['sqp_time_us']
        def get_stats(self):return self.reply
    def builder(context,meta,events):
        def factory(B,SQP,PCG):
            clock.advance(6)
            owner=C.F.Owned(SimpleNamespace(solver=Public(),solver_params=dict(meta['preset'],max_sqp_iters=SQP,max_pcg_iters=PCG)),events)
            owners.append(owner);return owner
        return factory
    args=dict(manifest_path='synthetic',manifest_sha256='synthetic',machine='synthetic-machine',family_state=state,inputs=bank,builder=builder,clock=clock)
    return args,owners,state


def records(output):return [json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]


def test_cases_flags_counts_and_held_template():
    cases=C.cases();assert len(cases)==38
    assert sum(c['arm']=='query' for c in cases)==8
    assert sum(c['arm']=='first_response' for c in cases)==18
    assert sum(c['arm']=='profile' for c in cases)==6
    assert sum(c['arm']=='sustained' for c in cases)==6
    assert all(c['stage_timing']==(c['arm']=='profile') for c in cases)
    template=json.loads((C.HERE/'held_plan_template.json').read_text())
    assert all(e['enabled'] is False and e['repeats']==1 for e in template['experiments'])
    assert json.loads((C.HERE/'release_manifest.json').read_text())['status']=='held'


def test_schedulers_full_pool_prefixes():
    assert [C.schedule(1,i) for i in (0,15,16,95,96)]==[(0,[0]),(0,[15]),(1,[0]),(5,[15]),(0,[0])]
    assert set((a,ids[0]) for a,ids in [C.schedule(1,i) for i in range(96)])==set((a,b) for a in range(6) for b in range(16))
    assert [C.schedule(16,i)[0] for i in range(7)]==[0,1,2,3,4,5,0]
    assert all(C.schedule(16,i)[1]==list(range(16)) for i in range(6))


def test_zero_cap_terminal_and_early_skip_rows():
    assert len(C.stage_layout(0))==2
    C.validate_profile(profile(0),0);C.validate_profile(profile(10,True),10)
    assert len(C.stage_layout(10))*2==144
    p=profile(1,True);p['windows'][1]['unavailable_reason']='not_reached'
    with pytest.raises(ValueError):C.validate_profile(p,1)
    p=profile(1);p['windows'][2]['sqp_iteration']=1
    with pytest.raises(ValueError):C.validate_profile(p,1)


def test_actual_bank_schema_integrity_and_precall_copy(tmp_path):
    bank,digest=make_bank(tmp_path);meta,arrays=C.bank(bank,digest)
    request=C.request(meta,arrays,16);request['warm'][0,0]=1
    assert arrays['pools'][0,0,0]==0
    (bank/'bank.json').write_text('{}')
    with pytest.raises(ValueError):C.bank(bank,digest)


def test_first_response_one_call_no_warmup(tmp_path,monkeypatch):
    case=next(c for c in C.cases() if c['arm']=='first_response' and c['B']==1)
    args,owners,state=injected(tmp_path,monkeypatch,case);out=tmp_path/'raw'
    result=run.run_case(case,out,**args)
    assert result['native_calls']==result['measured_calls']==result['completed_calls']==len(owners)==1
    assert result['evidence_kind']=='synthetic_test'
    saved=json.loads(state.read_text());assert saved['status']=='ready' and saved['canonical_receipt']=='preserve-me'
    assert result['coverage']==[dict(input_index=0,candidate_ids=[0])]
    assert any(e['event']=='native_return' for e in records(out))


def test_profile_warmup_and_raw_stage_payload(tmp_path,monkeypatch):
    case=next(c for c in C.cases() if c['arm']=='profile' and c['B']==16)
    args,owners,_=injected(tmp_path,monkeypatch,case);out=tmp_path/'raw';result=run.run_case(case,out,**args)
    assert result['native_calls']==2 and result['measured_calls']==1 and len(owners)==2
    native=[e for e in records(out) if e['event']=='native_return']
    assert len(native)==2 and all('experiment_profile_stages' in e['record']['native_reply'] for e in native)


def test_sustained_deadline_before_next_owner_keeps_overrun(tmp_path,monkeypatch):
    case=next(c for c in C.cases() if c['arm']=='sustained' and c['B']==1)
    args,owners,_=injected(tmp_path,monkeypatch,case);out=tmp_path/'raw';result=run.run_case(case,out,**args)
    # Each complete cycle17s, approved120s arm finishes8calls at136s; one priorwarmup.
    assert result['measured_calls']==8 and result['native_calls']==len(owners)==9
    assert result['sustained_wall_s']==136 and result['boundary_overrun_s']==16
    assert result['coverage']==[dict(input_index=0,candidate_ids=[i]) for i in range(8)]


@pytest.mark.parametrize('failure,event',[('normalize','native_return'),('native','native_call_failed')])
def test_return_and_native_exception_retained_hold_next_call(tmp_path,monkeypatch,failure,event):
    case=next(c for c in C.cases() if c['arm']=='first_response' and c['B']==1)
    args,owners,state=injected(tmp_path,monkeypatch,case,fail=failure);out=tmp_path/'failed'
    with pytest.raises(RuntimeError):run.run_case(case,out,**args)
    events=records(out);payload=next(e for e in events if e['event']==event)
    key='native_argument_arrays_at_return' if event=='native_return' else 'native_argument_arrays_at_failure'
    with np.load(out/payload['raw_file'],allow_pickle=False) as z:assert z[payload['record'][key]['XU']['npz_key']].shape==(1,665)
    assert json.loads(state.read_text())['status']=='held'
    assert json.loads((out/'metrics.json').read_text())['native_calls']==1
    with pytest.raises(RuntimeError):run.run_case(case,tmp_path/'next',**args)
    assert len(owners)==1


def test_real_held_guard_precedes_injected_import(tmp_path):
    case=C.cases()[0];path=C.HERE/'release_manifest.json';called=[]
    with pytest.raises(RuntimeError):run.run_case(case,tmp_path/'raw',manifest_path=path,manifest_sha256=C.sha(path),
        machine='synthetic-machine',family_state=tmp_path/'missing',query=lambda *a:called.append(a))
    assert not called
    assert json.loads((tmp_path/'raw/metrics.json').read_text())['status']=='held'


def test_scope_and_variant_build_commands():
    assert audit()['status']=='PASS_CPU_STATIC_ONLY'
    off=commands('source','native','python','arch',32,False)[0];on=commands('source2','native2','python','arch',32,True)[0]
    assert '-DGATO_EXPERIMENT_STAGE_TIMING=OFF' in off and '-DGATO_EXPERIMENT_STAGE_TIMING=ON' in on
    assert '-DGATO_EXPERIMENT_TELEMETRY=ON' in off and '-DGATO_EXPERIMENT_RESOURCE_QUERY=ON' in off


def resource_payload(case, attribute_failure=False):
    return dict(N=case['N'],batch_size=case['B'],device_status={'code':0},properties_status={'code':0},
        driver_status={'code':0},runtime_status={'code':0},kernels=[dict(kernel=name,
            attributes_status={'code':1 if attribute_failure and name=='pcg' else 0},
            occupancy_status={'code':1},theoretical_active_blocks_per_SM=None,fits_default_limits_prediction=False)
            for name in ('setup_kkt','merit','pcg','schur1','schur2','dz','line_search')])


@pytest.mark.parametrize('attribute_failure',[False,True])
def test_query_unsupported_occupancy_retained_attribute_failure_holds(tmp_path,monkeypatch,attribute_failure):
    case=C.cases()[0];args,_,state=injected(tmp_path,monkeypatch,case)
    args.pop('builder');args['query']=lambda context,c:resource_payload(c,attribute_failure)
    out=tmp_path/'query'
    if attribute_failure:
        with pytest.raises(RuntimeError):run.run_case(case,out,**args)
        assert json.loads(state.read_text())['status']=='held'
    else:
        result=run.run_case(case,out,**args)
        assert result['native_calls']==result['solver_calls']==0
    event=next(e for e in records(out) if e['event']=='query_returned')
    assert event['record']['payload']['kernels'][0]['occupancy_status']['code']==1
    assert event['record']['payload']['kernels'][0]['theoretical_active_blocks_per_SM'] is None


def test_wrong_manifest_hash_stops_before_import(tmp_path):
    called=[];case=C.cases()[0]
    with pytest.raises(RuntimeError):run.run_case(case,tmp_path/'raw',manifest_path=C.HERE/'release_manifest.json',
        manifest_sha256='0'*64,machine='synthetic-machine',family_state=tmp_path/'missing',query=lambda *args:called.append(args))
    assert not called


def test_compiled_flag_mismatch_rejected_before_native_import(tmp_path):
    # Pinned synthetic receipt exercises the guard without importing its fake .so.
    source=tmp_path/'archive';source.mkdir();mapping={}
    for name in ('python/bsqp/interface.py','gato/bsqp/bsqp.cuh','experiments/resource_profile_campaign/run.py'):
        dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((C.ROOT/name).read_bytes());mapping[name]=C.sha(dest)
    def item(path):return dict(path=str(path),sha256=C.sha(path))
    def saved(name,value):
        path=tmp_path/name;C.write_json(path,value);return item(path)
    module=source/'python/bsqp/bsqpN16_tiago_right-synthetic.so';module.write_text('synthetic_test_non_loadable')
    cache=tmp_path/'CMakeCache.txt';cache.write_text('synthetic')
    archive=saved('archive.json',dict(source_commit='a'*40,source_file_sha256=mapping))
    build=saved('build.json',dict(N=16,source_commit='a'*40,source_root=str(source),
        flags=dict(GATO_EXPERIMENT_STAGE_TIMING=True,GATO_EXPERIMENT_RESOURCE_QUERY=True,GATO_EXPERIMENT_TELEMETRY=True),
        module=item(module),cmake_cache=item(cache),source_receipt=archive))
    manifest=saved('release.json',dict(status='released_F04_F05',source_commit='a'*40,machine='synthetic-machine',
        implementation_review=saved('review.json',dict(verdict='approved',evidence_kind='synthetic_test')),
        observer_acceptance=saved('acceptance.json',dict(decision='accept_existing_first_instrumentation_without_equivalence_gate',evidence_kind='synthetic_test')),
        modules={R.key(16,False):dict(path=str(module),sha256=C.sha(module),build_receipt=build)}))
    called=[]
    with pytest.raises(RuntimeError,match='Compiled N/flags/source differ'):
        run.run_case(C.cases()[0],tmp_path/'raw',manifest_path=manifest['path'],manifest_sha256=manifest['sha256'],
            machine='synthetic-machine',family_state=tmp_path/'no-state',query=lambda *a:called.append(a))
    assert not called


def test_interrupted_running_marker_blocks_next_import(tmp_path,monkeypatch):
    case=C.cases()[0];args,_,state=injected(tmp_path,monkeypatch,case);args.pop('builder');calls=[]
    def interrupted(*a):
        calls.append(a)
        assert json.loads(state.read_text())['status']=='running'
        raise SystemExit('injected-interruption; no native imported')
    args['query']=interrupted
    with pytest.raises(SystemExit):run.run_case(case,tmp_path/'interrupted',**args)
    marker=json.loads(state.read_text())
    assert marker['status']=='running' and marker['canonical_receipt']=='preserve-me'
    assert marker['current_attempt']['current_raw_path']==str(tmp_path/'interrupted')
    assert (tmp_path/'interrupted/events.jsonl').exists()
    with pytest.raises(RuntimeError):run.run_case(case,tmp_path/'next',**args)
    assert len(calls)==1


def test_runner_metrics_mirror_failure_holds_after_retained_success(tmp_path,monkeypatch):
    case=next(c for c in C.cases() if c['arm']=='first_response' and c['B']==1)
    args,owners,state=injected(tmp_path,monkeypatch,case);out=tmp_path/'raw'
    def fail_mirror(*args):raise OSError('injected-runner-mirror-error')
    monkeypatch.setattr(run.shutil,'copyfile',fail_mirror)
    with pytest.raises(OSError,match='mirror-error'):
        run.run_case(case,out,metrics_mirror=tmp_path/'runner-metrics.json',**args)
    assert len(owners)==1 and json.loads(state.read_text())['status']=='held'
    assert any(e['event']=='native_return' for e in records(out))
    metrics=json.loads((out/'metrics.json').read_text())
    assert metrics['status']=='execution_failed' and metrics['failure_stage']=='runner_metrics_mirror' and metrics['native_calls']==1
    assert (out/'SHA256SUMS.json').exists()


def test_positive_actual_pinned_release_and_canonical_readiness_guard(tmp_path):
    # All qualification-looking fixtures are synthetic; this does not release real execution.
    case=next(c for c in C.cases() if c['arm']=='first_response' and c['B']==1)
    source=tmp_path/'synthetic-source';source.mkdir();mapping={}
    for name in ('python/bsqp/interface.py','gato/bsqp/bsqp.cuh','experiments/resource_profile_campaign/run.py'):
        dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((C.ROOT/name).read_bytes());mapping[name]=C.sha(dest)
    def item(path):return dict(path=str(path),sha256=C.sha(path))
    def saved(name,value):
        path=tmp_path/name;C.write_json(path,value);return item(path)
    module=source/'python/bsqp/bsqpN32_tiago_right-synthetic.so';module.write_text('synthetic_test_non_loadable')
    flags=dict(GATO_EXPERIMENT_STAGE_TIMING=False,GATO_EXPERIMENT_RESOURCE_QUERY=True,GATO_EXPERIMENT_TELEMETRY=True)
    cache=tmp_path/'CMakeCache.txt'
    cache.write_text('KNOTS:STRING=32\nPLANT:STRING=tiago_right\n'+''.join(f'{k}:BOOL={"ON" if v else "OFF"}\n' for k,v in flags.items()))
    build=saved('build.json',dict(N=32,source_commit='a'*40,source_root=str(source),flags=flags,
        module=item(module),cmake_cache=item(cache),source_receipt=saved('archive.json',dict(source_commit='a'*40,source_file_sha256=mapping))))
    readiness=saved('ready.json',dict(evidence_kind='synthetic_test',status='ready',task_success=True,physical_pass=True,execution_error=None,
        machine='synthetic-machine',source_commit='a'*40,module_sha256=C.sha(module),timing_mode='A',canonical_config_sha256=C.F01.canonical_hash()))
    manifest=saved('release.json',dict(status='released_F04_F05',evidence_kind='synthetic_test',source_commit='a'*40,machine='synthetic-machine',
        implementation_review=saved('review.json',dict(verdict='approved',evidence_kind='synthetic_test')),
        observer_acceptance=saved('acceptance.json',dict(decision='accept_existing_first_instrumentation_without_equivalence_gate',evidence_kind='synthetic_test')),
        modules={R.key(32,False):dict(path=str(module),sha256=C.sha(module),build_receipt=build)},canonical_A_readiness=readiness,
        duration_approval=item(C.HERE/'duration_approval.json'),sustained_duration_s=120))
    state=tmp_path/'family.json';C.write_json(state,dict(status='ready',machine='synthetic-machine',release_manifest_sha256=manifest['sha256']))
    # Actual guard, no monkeypatch; fake .so is only hashed, never imported.
    context=R.validate(manifest['path'],manifest['sha256'],'synthetic-machine',case,state)
    assert context['selected_module']['flags']==flags and context['selected_module']['sha256']==C.sha(module)
    assert json.loads(state.read_text())['status']=='ready'
