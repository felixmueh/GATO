"""CPU-only practical T01 contract tests; synthetic evidence is not qualification."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/campaign'),str(ROOT/'python')]
from raw import RawRecorder,sha,write_json
from t01_protocol import INPUTS,reference_bytes,episode_protocol,preflight,fallback,load_case,diagnostic_protocol
from run_t01 import execute_episode,trace_metrics,run_case
from make_task_plan import make_plan


def protocol(name='t01.json'): return json.loads((INPUTS/name).read_text())


def bundle(tmp_path,p,identity,passed=True,physical=True,error=None):
    n=len(list(tmp_path.glob('*metrics.json')))
    m=tmp_path/f'{n}metrics.json';v=tmp_path/f'{n}provenance.json'
    write_json(m,dict(machine=identity['machine'],execution_backend='internal',timing_mode=p['timing_mode'],configuration=p,execution_error=error,outcome='task_passed' if passed else 'task_failed',task_success=passed,physical_pass=physical))
    h=hashlib.sha256((json.dumps(p,indent=2)+'\n').encode()).hexdigest()
    write_json(v,dict(identity,task_config_sha256=h))
    return {'metrics':{'path':m.name,'sha256':sha(m)},'provenance':{'path':v.name,'sha256':sha(v)}}


def test_preflight_requires_physical_and_exact_native_provenance(tmp_path):
    identity={'machine':'gtx1070','source_commit':'a'*40,'module_sha256':'b'*64}
    p=protocol('v01.json')
    good=bundle(tmp_path,p,identity)
    preflight(good,tmp_path,identity,p)
    for altered in (dict(identity,machine='rtx5060'),dict(identity,module_sha256='c'*64),dict(identity,source_commit='d'*40)):
        with pytest.raises(ValueError): preflight(good,tmp_path,altered,p)
    with pytest.raises(ValueError): preflight(bundle(tmp_path,p,identity,physical=False),tmp_path,identity,p)
    with pytest.raises(ValueError): preflight(bundle(tmp_path,p,identity,error='CUDA import failure'),tmp_path,identity,p)


def test_fallback_full_matched_cohort_not_failed_only_or_errors(tmp_path):
    p=protocol();identity={'machine':'gtx1070','source_commit':'a'*40,'module_sha256':'b'*64}
    entries={t['id']:bundle(tmp_path,episode_protocol(p,t),identity) for t in p['tasks']}
    with pytest.raises(ValueError): fallback({'tasks':entries},tmp_path,identity,p)
    task=p['tasks'][0];entries[task['id']]=bundle(tmp_path,episode_protocol(p,task),identity,physical=False)
    fallback({'tasks':entries},tmp_path,identity,p)
    with pytest.raises(ValueError): fallback({'tasks':{task['id']:entries[task['id']]}},tmp_path,identity,p)
    entries[task['id']]=bundle(tmp_path,episode_protocol(p,task),identity,passed=False,error='worker crash')
    with pytest.raises(ValueError): fallback({'tasks':entries},tmp_path,identity,p)


def test_failed_preflight_requires_separate_completed_A_diagnostic(tmp_path):
    p=protocol();v=protocol('v01.json');identity={'machine':'gtx1070','source_commit':'a'*40,'module_sha256':'b'*64}
    failed=bundle(tmp_path,v,identity,passed=False)
    with pytest.raises(ValueError): fallback({'failed_preflight':failed},tmp_path,identity,p)
    a=diagnostic_protocol(v)
    diagnostic=bundle(tmp_path,a,identity)
    fallback({'failed_preflight':failed,'A_diagnostic':diagnostic},tmp_path,identity,p)
    old=dict(v,timing_mode='A',offline_timing='controller_dt',name='V01-A-diagnostic',campaign_id='V01-A-diagnostic',diagnostic_only=a['diagnostic_only'])
    with pytest.raises(ValueError,match='exact-protocol'):
        fallback({'failed_preflight':failed,'A_diagnostic':bundle(tmp_path,old,identity)},tmp_path,identity,p)
    for key in ('solver_params','goal','initial_state','N','dt_s','sim_dt_s','physical_validation'):
        assert a[key]==v[key]
    assert v['timing_mode']=='B' and 'old active plan' in v['implemented_clock_convention']
    with pytest.raises(ValueError): fallback({'failed_preflight':bundle(tmp_path,v,identity),'A_diagnostic':diagnostic},tmp_path,identity,p)


def test_frozen_tracking_reference_and_public_presets_no_mutation(tmp_path):
    p=protocol();task=p['tasks'][3];episode=episode_protocol(p,task)
    reference=reference_bytes(task).reshape(-1,6)
    assert reference.shape[0]*.03 > task['duration_s']+.93
    assert episode['solver_params']['qd_cost']==.01 and episode['solver_params']['vel_lim_cost']==.001
    assert episode['solver_params']['ee_orient_cost']==0 and p['reaching_solver_params']['qd_cost']==.1
    class Controller:
        def run_mpc_fig8(self,state,ref,**kwargs):
            assert np.array_equal(ref,reference.reshape(-1))
            assert kwargs['sim_time']==24 and kwargs['sim_dt']==.003 and kwargs['offline_timing']=='solve_time'
            assert kwargs['record_raw'] and kwargs['record_callback'] is recorder
            return [],{}
    recorder=RawRecorder(tmp_path)
    execute_episode(Controller(),episode,recorder)
    task['reference_packed'][0]+=.001
    with pytest.raises(ValueError): reference_bytes(task)


def test_tracking_actual_time_RMS_completion_and_physical_separation(tmp_path):
    import pinocchio as pin
    p=episode_protocol(protocol(),protocol()['tasks'][3]);p['duration_s']=.003;p['sim_dt_s']=.003
    model=pin.buildModelFromUrdf(str(ROOT/p['urdf']['path']))
    x=np.tile(p['initial_state'],(3,1));u=np.zeros((2,7));t=np.array([0.,.001,.003])
    result=trace_metrics(p,model,x,u,t,{},None,tmp_path)
    assert result['executed_duration_s']==.003 and result['task_gates']['full_period_completed']
    # Reference moves after t0. Compare actual-time sampled targets from emitted CSV.
    trace=np.genfromtxt(tmp_path/'executed_trace.csv',delimiter=',',names=True)
    expected=np.sqrt(np.trapezoid(trace['position_error_m']**2,t)/.003)
    assert result['tracking_RMS_m']==pytest.approx(expected)
    assert np.isnan(trace['applied_control_Nm_0'][0])
    assert result['physical_pass'] is True
    u[:]=1e6
    bad=trace_metrics(p,model,x,u,t,{},None,tmp_path)
    assert bad['task_success']==result['task_success'] and bad['physical_pass'] is False
    short=trace_metrics(p,model,x[:2],u[:1],t[:2],{},None,tmp_path)
    assert short['task_success'] is False and not short['task_gates']['full_period_completed']


def test_failed_warm_solve_preserves_bytes_and_schema(tmp_path):
    recorder=RawRecorder(tmp_path)
    warm=np.array([1.,np.nan,np.inf],dtype=np.float32)
    recorder('solve_started',{'category':'warm','warm_start':warm,'state':np.zeros(14)})
    recorder('solve_failed',{'category':'warm','warm_start':warm,'wall_solve_time_s':.01,'error':'synthetic failure'})
    with np.load(tmp_path/recorder.events[-1]['raw_file']) as raw:
        assert np.isnan(raw['record__warm_start'][1]) and np.isinf(raw['record__warm_start'][2])
    assert recorder.events[-1]['record']['warm_start']['nonfinite_count']==2


def test_release_hold_occurs_before_native_identity(tmp_path,monkeypatch):
    import run_t01
    held=dict(protocol(),release_status='held synthetic protocol')
    monkeypatch.setattr(run_t01,'load_case',lambda case:held)
    monkeypatch.setattr(run_t01,'current_identity',lambda:pytest.fail('native identity accessed under release hold'))
    with pytest.raises(ValueError,match='held'):
        run_case({'id':'goal-1','parameters':{'protocol':'T01','offline_timing':'solve_time','preflight':None,'fallback':None}},tmp_path)
    with pytest.raises(ValueError): make_plan('7187cf8'+'0'*33)
    with pytest.raises(ValueError): make_plan('a'*40,enabled=True)


def test_reaching_failure_observer_preserves_started_warm_and_partial_execution(tmp_path):
    p=episode_protocol(protocol(),protocol()['tasks'][1]);recorder=RawRecorder(tmp_path)
    warm=np.array([2.,np.nan],dtype=np.float32)
    class FailingController:
        def run_mpc_goals(self,initial,goals,**kwargs):
            assert np.array_equal(initial,np.asarray(p['initial_state'],dtype=np.float32))
            assert np.array_equal(goals[0],np.asarray(p['goal'],dtype=np.float32))
            assert kwargs['goal_timeout']==8 and kwargs['sim_dt']==.003
            assert kwargs['offline_timing']=='solve_time' and kwargs['record_raw']
            assert kwargs['initial_guess'] is None and kwargs['goal_dwell_time']==0
            started={'category':'warm','status':'started','warm_start':warm,'state':initial.copy()}
            self.last_run_stats={'raw_solves':[started],'raw_execution':[]}
            kwargs['record_callback']('solve_started',started)
            started.update(status='failed',wall_solve_time_s=.002,error='synthetic native failure')
            kwargs['record_callback']('solve_failed',started)
            raise RuntimeError('synthetic native failure')
    controller=FailingController()
    with pytest.raises(RuntimeError,match='synthetic native'):
        execute_episode(controller,p,recorder)
    assert controller.last_run_stats['raw_solves'][0]['status']=='failed'
    assert [e['event'] for e in recorder.events]==['solve_started','solve_failed']
    with np.load(tmp_path/recorder.events[0]['raw_file']) as raw:
        assert np.isnan(raw['record__warm_start'][1])


def test_all_six_fallback_tasks_keep_targets_presets_and_gates(tmp_path):
    canonical=protocol(); alternate=protocol('t01_1ms.json')
    assert len(canonical['tasks'])==len(alternate['tasks'])==6
    assert canonical['tasks']==alternate['tasks']
    for key in ('initial_state','N','dt_s','batch_size','reaching_solver_params',
                'tracking_solver_params','physical_validation','seed_policy'):
        assert canonical[key]==alternate[key]
    assert canonical['sim_dt_s']==.003 and alternate['sim_dt_s']==.001
    seen=[]
    class Controller:
        def run_mpc_goals(self,state,goals,**kwargs):
            seen.append(('reaching',goals[0].copy()))
            assert kwargs['sim_dt']==.001 and kwargs['offline_timing']=='solve_time'
            return [],{}
        def run_mpc_fig8(self,state,reference,**kwargs):
            seen.append(('tracking',reference.copy()))
            assert kwargs['sim_dt']==.001 and kwargs['offline_timing']=='solve_time'
            return [],{}
    for task in alternate['tasks']:
        directory=tmp_path/task['id'];directory.mkdir()
        recorder=RawRecorder(directory)
        execute_episode(Controller(),episode_protocol(alternate,task),recorder)
        if task['kind']=='tracking':
            event=recorder.events[0]
            assert event['event']=='frozen_reference'
            assert sha(directory/event['raw_file'])==event['sha256']
            with np.load(directory/event['raw_file']) as raw:
                assert raw['record__reference'].tobytes()==reference_bytes(task).tobytes()
    assert [kind for kind,_ in seen]==['reaching']*3+['tracking']*3
    for (_,value),task in zip(seen,alternate['tasks']):
        expected=np.asarray(task['goal'],dtype=np.float32) if task['kind']=='reaching' else reference_bytes(task)
        assert np.array_equal(value,expected)


@pytest.mark.parametrize('cohort',['T01','T01-1ms','V01-1ms'])
def test_every_cohort_holds_direct_worker_and_enabled_planner(tmp_path,monkeypatch,cohort):
    import run_t01
    from t01_protocol import PROTOCOL_FILES
    held=dict(protocol(PROTOCOL_FILES[cohort]),release_status='held synthetic protocol')
    monkeypatch.setattr(run_t01,'load_case',lambda case:held)
    monkeypatch.setattr(run_t01,'current_identity',lambda:pytest.fail('held worker accessed native identity'))
    parameters={'protocol':cohort,'offline_timing':'solve_time','fallback':None}
    if cohort!='V01-1ms': parameters['preflight']=None
    with pytest.raises(ValueError,match='held'):
        run_case({'id':'unused-under-hold','parameters':parameters},tmp_path)
    assert not tmp_path.exists() or not list(tmp_path.iterdir())
    held_source=tmp_path/'held-source'
    held_path=held_source/'experiments/campaign/inputs'/PROTOCOL_FILES[cohort]
    held_path.parent.mkdir(parents=True);write_json(held_path,held)
    with pytest.raises(ValueError,match='Held'):
        make_plan('a'*40,cohort=cohort,enabled=True,source=held_source)


def test_receipts_reject_changed_harness_bytes_and_evidence_hash(tmp_path):
    identity={'machine':'gtx1070','source_commit':'a'*40,'module_sha256':'b'*64,
              'source_sha256':{'python/bsqp/offline_mpc.py':'c'*64}}
    p=protocol('v01.json');good=bundle(tmp_path,p,identity)
    preflight(good,tmp_path,identity,p)
    changed=dict(identity,source_sha256={'python/bsqp/offline_mpc.py':'d'*64})
    with pytest.raises(ValueError,match='source bytes'):
        preflight(good,tmp_path,changed,p)
    metrics=tmp_path/good['metrics']['path'];metrics.write_text(metrics.read_text()+'\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        preflight(good,tmp_path,identity,p)


@pytest.mark.parametrize('cohort',['T01','T01-1ms','V01-1ms'])
def test_disabled_plan_pins_matched_desktops_and_every_case(monkeypatch,cohort):
    import make_task_plan
    # These inputs are still under implementation. Simulate the committed-file
    # read only; plan generation and every hash/case assertion remain real.
    def committed(command):
        assert command[:4]==['git','-C',str(ROOT),'show']
        return (ROOT/command[4].split(':',1)[1]).read_bytes()
    monkeypatch.setattr(make_task_plan.subprocess,'check_output',committed)
    plan=make_plan('a'*40,cohort=cohort)
    assert len(plan['experiments'])==2
    assert [e['machines'] for e in plan['experiments']]==[['gtx1070'],['rtx5060']]
    expected=['rough-reaching-1ms'] if cohort=='V01-1ms' else [t['id'] for t in protocol()['tasks']]
    for entry in plan['experiments']:
        assert entry['enabled'] is False and entry['repeats']==1
        assert entry['source_commit']=='a'*40
        assert [case['id'] for case in entry['cases']]==expected
        assert all(case['parameters']['protocol']==cohort for case in entry['cases'])
        assert all(case['parameters']['offline_timing']=='solve_time' for case in entry['cases'])
        for path,digest in entry['inputs'].items(): assert sha(ROOT/path)==digest
        assert '{build}/campaign-source/experiments/campaign/run_t01.py' in entry['command']
    monkeypatch.setattr(make_task_plan.subprocess,'check_output',lambda command:b'changed committed file')
    with pytest.raises(ValueError,match='differ from reviewed commit'):
        make_plan('a'*40,cohort=cohort)


@pytest.fixture
def released_worker(tmp_path,monkeypatch):
    import run_t01
    from types import SimpleNamespace
    root=tmp_path/'build/campaign-source';root.mkdir(parents=True)
    original=protocol();cohort=json.loads(json.dumps(original))
    cohort['release_status']='released after reviewed clock decision'
    paths=set(cohort['reviewed_controller_sha256'])|{'experiments/campaign/'+n for n in ('run_t01.py','t01_protocol.py','raw.py','evaluate.py','make_task_plan.py','build_isolated.py')}
    paths|={'python/bsqp/common.py','python/bsqp/interface.py','tiago_src/gato_tiago/config.py',cohort['urdf']['path']}
    for relative in paths:
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
    cohort['reviewed_controller_sha256']={p:sha(root/p) for p in cohort['reviewed_controller_sha256']}
    cohort['reviewed_harness_sha256']={p:sha(root/p) for p in paths if p.startswith('experiments/campaign/')}
    inputs=root/'experiments/campaign/inputs';inputs.mkdir()
    v=protocol('v01.json');write_json(inputs/'v01.json',v);write_json(inputs/'t01.json',cohort)
    cohort['preflight_protocol_sha256']=sha(inputs/'v01.json')
    native_path=root/'python/bsqp/bsqpN32_tiago_right.fake.so';native_path.write_bytes(b'synthetic module, never executable')
    shared=[p for p in paths if p!=cohort['urdf']['path']]
    identity={'machine':'gtx1070','source_commit':'a'*40,'module_sha256':sha(native_path),'module_path':str(native_path),'source_sha256':{p:sha(root/p) for p in shared}}
    source_path=root.parent/'build_source_receipt.json'
    write_json(source_path,dict(source_commit=identity['source_commit'],isolated_source=str(root),source_file_sha256=identity['source_sha256']))
    smoke=root.parent/'campaign-native/cuda_access_smoke';smoke.parent.mkdir();smoke.write_bytes(b'synthetic smoke, never executable')
    build_path=root.parent/'build_receipt.json'
    write_json(build_path,dict(source=str(root),module_path=str(native_path),module_sha256=sha(native_path),build_source_receipt_sha256=sha(source_path),cuda_access_smoke_sha256=sha(smoke)))
    evidence=tmp_path/'evidence';evidence.mkdir();proof=bundle(evidence,v,identity)
    case={'id':cohort['tasks'][0]['id'],'parameters':{'protocol':'T01','offline_timing':'solve_time','preflight':proof,'fallback':None}}
    calls=[];native=SimpleNamespace(__file__=str(native_path))
    class Controller:
        def __init__(self,*args,**kwargs):
            calls.append('constructor');self.solver_params=kwargs['solver_params'];self.solver=SimpleNamespace(lib=native)
        def run_mpc_goals(self,initial,goals,**kwargs):
            calls.append('episode')
            solve={'category':'warm','status':'completed','selected_lane':0,'warm_start':np.array([np.nan,2.],dtype=np.float32),'wall_solve_time_s':.002,'native_solve_time_ms':1.}
            kwargs['record_callback']('solve_completed',solve)
            self.last_run_stats={'raw_solves':[solve],'raw_execution':[{'state_after':initial.copy(),'control':np.zeros(7),'time_s':0.,'step_dt_s':.003}],'update_intervals_s':[.003]}
            if 'fail-native' in calls: raise RuntimeError('synthetic native call failed')
            return [],self.last_run_stats
    def imported(name):
        calls.append('import:'+name)
        if name=='bsqp.bsqpN32_tiago_right': return native
        prefix='python/' if name.startswith('bsqp.') else 'tiago_src/'
        m=SimpleNamespace(__file__=str(root/(prefix+name.replace('.','/')+'.py')))
        if name=='gato_tiago.tiago_mpc_controller':m.MPC_GATO=Controller
        return m
    monkeypatch.setenv('GATO_MACHINE','gtx1070');monkeypatch.delenv('GATO_CAMPAIGN_BUILD_RECEIPT',raising=False)
    monkeypatch.setattr(run_t01,'ROOT',root);monkeypatch.setattr(run_t01,'INPUTS',inputs)
    monkeypatch.setattr(run_t01,'load_case',lambda case:cohort)
    monkeypatch.setattr(run_t01,'current_identity',lambda:identity)
    monkeypatch.setattr(run_t01.importlib,'import_module',imported)
    monkeypatch.setattr(run_t01.subprocess,'run',lambda *a,**k:calls.append('smoke'))
    return SimpleNamespace(root=root,cohort=cohort,case=case,identity=identity,calls=calls,inputs=evidence,
        output=tmp_path/'output',native=native,build=build_path,source=source_path,smoke=smoke)


@pytest.mark.parametrize('machine',[None,'jetson','rtx5060-gazebo'])
def test_released_machine_rejected_before_identity(released_worker,monkeypatch,machine):
    import run_t01
    f=released_worker
    if machine is None:monkeypatch.delenv('GATO_MACHINE')
    else:monkeypatch.setenv('GATO_MACHINE',machine)
    monkeypatch.setattr(run_t01,'current_identity',lambda:pytest.fail('identity accessed for unsupported machine'))
    with pytest.raises(ValueError,match='desktop'):run_t01.run_case(f.case,f.output,f.inputs)
    assert f.calls==[]


@pytest.mark.parametrize('relative',['python/bsqp/offline_mpc.py','experiments/campaign/run_t01.py'])
def test_changed_released_source_rejected_before_identity(released_worker,monkeypatch,relative):
    import run_t01
    f=released_worker;(f.root/relative).write_text('changed source')
    monkeypatch.setattr(run_t01,'current_identity',lambda:pytest.fail('identity accessed for changed source'))
    with pytest.raises(ValueError,match='source bytes'):run_t01.run_case(f.case,f.output,f.inputs)
    assert f.calls==[]


@pytest.mark.parametrize('bad',['missing-build','module','module-path','source-hash','source-commit','shared-source','smoke'])
def test_build_failures_finalize_before_native_import(released_worker,bad):
    import run_t01
    f=released_worker
    if bad=='missing-build':f.build.unlink()
    elif bad=='module':Path(f.identity['module_path']).write_bytes(b'changed module')
    elif bad=='smoke':f.smoke.write_bytes(b'changed readiness')
    elif bad in {'module-path','source-hash'}:
        b=json.loads(f.build.read_text());b['module_path' if bad=='module-path' else 'build_source_receipt_sha256']='changed';write_json(f.build,b)
    else:
        s=json.loads(f.source.read_text())
        if bad=='source-commit':s['source_commit']='b'*40
        else:s['source_file_sha256']['python/bsqp/common.py']='c'*64
        write_json(f.source,s);b=json.loads(f.build.read_text());b['build_source_receipt_sha256']=sha(f.source);write_json(f.build,b)
    assert run_t01.run_case(f.case,f.output,f.inputs)==1
    assert not any(c=='constructor' or c.startswith('import:') for c in f.calls)
    metrics=json.loads((f.output/'metrics.json').read_text())
    assert metrics['outcome']=='execution_failed' and metrics['physical_pass'] is None
    assert json.loads((f.output/'SHA256SUMS.json').read_text())['execution.npz']==sha(f.output/'execution.npz')


@pytest.mark.parametrize('failed',[False,True])
def test_released_worker_preserves_success_and_partial_failure_raw(released_worker,failed):
    import run_t01
    f=released_worker
    if failed:f.calls.append('fail-native')
    assert run_t01.run_case(f.case,f.output,f.inputs)==int(failed)
    assert f.calls.index('smoke')<f.calls.index('constructor')<f.calls.index('episode')
    metrics=json.loads((f.output/'metrics.json').read_text())
    assert metrics['outcome']==('execution_failed' if failed else 'task_failed')
    events=[json.loads(line) for line in (f.output/'events.jsonl').read_text().splitlines()]
    event=next(e for e in events if e['event']=='solve_completed')
    assert event['record']['selected_lane']==0
    with np.load(f.output/event['raw_file']) as arrays:assert np.isnan(arrays['record__warm_start'][0])
    with np.load(f.output/'execution.npz') as arrays:assert arrays['x'].shape==(2,14) and arrays['u'].shape==(1,7)
    for path,digest in json.loads((f.output/'SHA256SUMS.json').read_text()).items():assert sha(f.output/path)==digest


def test_imported_native_path_mismatch_prevents_constructor(released_worker):
    import run_t01
    f=released_worker;f.native.__file__=str(f.root/'elsewhere.so')
    assert run_t01.run_case(f.case,f.output,f.inputs)==1
    assert 'constructor' not in f.calls


def test_enabled_plan_real_committed_source_selects_one_machine(tmp_path):
    import subprocess
    source=tmp_path/'source';source.mkdir()
    paths=['experiments/campaign/inputs/'+name for name in ('t01.json','v01.json','v01_1ms.json')]
    paths+=['experiments/campaign/'+name for name in ('mode_b_protocol.json','run_t01.py','t01_protocol.py','raw.py','evaluate.py','build_isolated.py','make_task_plan.py')]
    paths+=['tests/python/tiago_task_cases/short_reach.json','python/bsqp/offline_mpc.py']
    for relative in paths:
        target=source/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
    frozen=source/'experiments/campaign/inputs/t01.json'
    p=json.loads(frozen.read_text());p['release_status']='released after reviewed clock decision';write_json(frozen,p)
    def git(*args):return subprocess.check_output(['git','-C',str(source),*args],stderr=subprocess.DEVNULL,text=True).strip()
    git('init');git('add','.');git('-c','user.name=T01 CPU test','-c','user.email=t01-test@example.invalid','commit','-m','[tiago] temporary synthetic plan source')
    commit=git('rev-parse','HEAD')
    def saved(name,data):
        path=source/name;write_json(path,data);return {'path':name,'sha256':sha(path)}
    release={'review':saved('review.json',{'source_commit':commit,'verdict':'approved'}),
             'task_suite':saved('suite.json',{'source_commit':commit,'command':'make test-tiago-tasks','exit_code':0,'skips':0})}
    # Worker validates exact receipt contents; planner pins their declared bytes.
    proof={'metrics':saved('preflight-metrics.json',{'synthetic':'not qualification'}),
           'provenance':saved('preflight-provenance.json',{'synthetic':'not qualification'})}
    evidence={'release':release,'gtx1070':{'preflight':proof,'fallback':None}}
    result=make_plan(commit,evidence=evidence,enabled=True,source=source,enable_machines=['gtx1070'])
    assert [e['enabled'] for e in result['experiments']]==[True,False]
    assert len(result['experiments'][0]['cases'])==len(result['experiments'][1]['cases'])==6
    assert result['experiments'][1]['cases'][0]['parameters']['preflight'] is None
    assert result['experiments'][0]['inputs']['review.json']==release['review']['sha256']
    with pytest.raises(ValueError,match='Held'):make_plan(commit,evidence=evidence,enabled=True,source=source)
    write_json(source/'suite.json',{'source_commit':commit,'command':'make test-tiago-tasks','exit_code':0,'skips':1})
    evidence['release']['task_suite']['sha256']=sha(source/'suite.json')
    with pytest.raises(ValueError,match='without skips'):
        make_plan(commit,evidence=evidence,enabled=True,source=source,enable_machines=['gtx1070'])


def test_imported_shared_path_mismatch_prevents_constructor(released_worker,monkeypatch):
    import run_t01
    f=released_worker;original=run_t01.importlib.import_module
    def imported(name):
        result=original(name)
        if name=='bsqp.common':result.__file__=str(f.root/'stale-common.py')
        return result
    monkeypatch.setattr(run_t01.importlib,'import_module',imported)
    assert run_t01.run_case(f.case,f.output,f.inputs)==1
    assert 'constructor' not in f.calls
    assert 'Imported shared source path mismatch' in json.loads((f.output/'metrics.json').read_text())['execution_error']


def test_missing_pinocchio_is_failed_execution_without_skip(released_worker,monkeypatch):
    import builtins
    import run_t01
    f=released_worker;original=builtins.__import__
    def imported(name,*args,**kwargs):
        if name=='pinocchio':raise ModuleNotFoundError('synthetic missing required Pinocchio')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',imported)
    assert run_t01.run_case(f.case,f.output,f.inputs)==1
    assert 'constructor' not in f.calls
    result=json.loads((f.output/'metrics.json').read_text())
    assert result['outcome']=='execution_failed' and 'missing required Pinocchio' in result['execution_error']
