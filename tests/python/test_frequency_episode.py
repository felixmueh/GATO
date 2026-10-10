"""Mocked revealing plans/plant plus CPU guards; no native call/build."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
def load(name):
    spec=importlib.util.spec_from_file_location('frequency_'+name,ROOT/f'experiments/frequency_campaign/{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
E=load('episode');P=load('protocol');R=load('record_bank');W=load('run')

@pytest.fixture(scope='module')
def data():return P.load_bank()

class Solver:
    def __init__(self,owner,fail=None):self.owner=owner;self.calls=[];self.dual=0;self.rho=0;self.fail=fail
    def reset_dual(self):self.dual+=1
    def reset_rho(self):self.rho+=1
    def solve(self,state,reference,warm):
        self.calls.append((state.copy(),reference.copy(),warm.copy()))
        if len(self.calls)==self.fail:raise RuntimeError('fixed injected solve failure')
        plan=E.cold_seed(state[0],self.owner.N).astype(np.float32)  # native output dtype
        for k in range(self.owner.N-1):plan[0,k*21+14:k*21+21]=len(self.calls)+10*k
        return plan,1234.
    def get_stats(self):return {'sqp_iters':np.array([1]),'pcg_iters':np.array([[2]]),'pcg_stop_reasons':np.array([[0]])}

class Owner:
    def __init__(self,kind,cell,bank,fail=None):
        self.N=cell['N'];self.dt=.03;self.batch_size=1;self.closed=False
        self.solver_params=dict(bank['presets'][kind],max_sqp_iters=cell['SQP'],max_pcg_iters=cell['PCG'])
        self.solver=Solver(self,fail)
    def update_force_batch(self,q):pass
    def close(self):self.closed=True

def plant(q,v,u,dt):return q+u*dt,v

def emit_buffer():
    events=[]
    return events,lambda kind,record:events.append((kind,deepcopy(record)))

def cell(variant='short12'):return dict(variant=variant,N=16,SQP=2,PCG=100,timing_mode='A',backend='internal')

def test_prefix_public_modeA_sampling_and_warm_alignment(data):
    bank,arrays=data;owner=Owner('reaching',cell(),bank);events,emit=emit_buffer()
    episode=bank['variants']['short12']['episodes'][0]
    result=E.collect_episode(owner,episode,arrays,cell(),emit,plant)
    assert result['completed_updates']==20 and result['execution_error'] is None
    # Independent inherited A prefix: u0 constant over10substeps, then shift one knot.
    state=arrays['starts'][0].copy()
    for slot in range(4):
        before,reference,warm=owner.solver.calls[slot]
        np.testing.assert_array_equal(before[0],state)
        if slot==0:np.testing.assert_array_equal(warm,E.cold_seed(state,16))
        else:np.testing.assert_array_equal(warm[0,14:21],np.full(7,slot+10,dtype=np.float32))
        q=state[:7].copy();v=state[7:].copy()
        for _ in range(10):q,v=plant(q,v,np.full(7,slot+1,dtype=np.float32),.003)
        state=np.concatenate([q,v])
    assert owner.solver.dual==1 and owner.solver.rho==2
    started=[r for event,r in events if event=='solve_started']
    assert [r['category'] for r in started]==episode['categories']
    np.testing.assert_array_equal(started[10]['reference'][0],np.tile(arrays['goals'][1],16))
    assert any(event=='rho_reset' for event,_ in events)
    assert len([e for e,_ in events if e=='step_completed'])==200
    assert [r['time_s'] for e,r in events if e=='interval_completed'][-1]==.57
    assert started[1]['replay_input_hashes']['state']['sha256']!=started[0]['replay_input_hashes']['state']['sha256']


def test_tracking_keeps_original_global_reference(data):
    bank,arrays=data;episode=bank['variants']['short12']['episodes'][3];owner=Owner('tracking',cell(),bank)
    events,emit=emit_buffer();E.collect_episode(owner,episode,arrays,cell(),emit,plant)
    for i,(_,reference,_) in enumerate(owner.solver.calls):
        np.testing.assert_allclose(reference[0],arrays['reference_0'].reshape(-1,6)[i:i+16].reshape(-1),rtol=0,atol=1e-15)
    assert owner.solver.rho==0 and owner.solver.dual==1


def test_fixed_count_success_does_not_stop_and_physical_is_separate(data):
    bank,arrays=data;episode=bank['variants']['short12']['episodes'][0]
    def observer(state,goal,control):return dict(rough_reached=True,physical_pass=False,position_error_m=.01)
    events,emit=emit_buffer();report=E.collect_episode(Owner('reaching',cell(),bank),episode,arrays,cell(),emit,plant,observer)
    assert report['completed_updates']==20 and report['task_success'] is True
    assert report['physical_pass'] is False and report['solver_numerical_pass'] is None


def test_failure_has_partial_realized_inputs_and_missing_slots(data,tmp_path):
    bank,arrays=data;episode=bank['variants']['short12']['episodes'][0]
    recorder=R.Recorder(tmp_path/'raw',dict(test_only=True))
    report=E.collect_episode(Owner('reaching',cell(),bank,fail=3),episode,arrays,cell(),recorder,plant)
    recorder.finish(report)
    assert report['attempted_solves']==3 and report['completed_solves']==2
    assert report['missing_solve_slots']==list(range(3,20))
    assert report['missing_update_slots']==list(range(2,20))
    manifest=json.loads((tmp_path/'raw/SHA256SUMS.json').read_text())
    assert 'events.jsonl' in manifest and len([k for k in manifest if k.startswith('raw/')])>40
    events=[json.loads(line) for line in (tmp_path/'raw/events.jsonl').read_text().splitlines()]
    assert any(e['event']=='solve_failed' for e in events)


def test_nonfinite_step_preserved_without_next_solve(data):
    bank,arrays=data;events,emit=emit_buffer();calls=0
    def broken(q,v,u,dt):return q*np.nan,v
    report=E.collect_episode(Owner('reaching',cell(),bank),bank['variants']['short12']['episodes'][0],arrays,cell(),emit,broken)
    assert report['attempted_solves']==1 and report['completed_updates']==0
    assert any(e=='step_completed' and np.isnan(r['state_after']).any() for e,r in events)
    assert report['execution_error'] is not None

@pytest.mark.parametrize('variant,count,cold,warm,switch',[('full24',24,24,444,12),('short12',12,12,222,6)])
def test_cell_fresh_solver_warmup_and_category_counts(data,variant,count,cold,warm,switch):
    bank,arrays=data;owners=[];events,emit=emit_buffer()
    def factory(kind,c):
        owner=Owner(kind,c,bank);owners.append(owner);return owner
    report=E.collect_cell(factory,bank,arrays,cell(variant),emit,lambda owner:plant)
    assert report['execution_error'] is None and len(report['episodes'])==count
    assert len(owners)==count+1 and all(o.closed for o in owners)
    starts=[r for e,r in events if e=='solve_started']
    assert sum(r['category']=='process_cuda_warmup' for r in starts)==1
    for category,n in [('cold',cold),('warm',warm),('goal_switch',switch)]:assert sum(r['category']==category for r in starts)==n
    assert all(len(o.solver.calls)==20 for o in owners[1:])
    assert not np.array_equal(owners[0].solver.calls[0][0],owners[1].solver.calls[0][0])


def test_wrong_effective_preset_stops_before_call(data):
    bank,arrays=data;owners=[];events,emit=emit_buffer()
    def factory(kind,c):
        owner=Owner(kind,c,bank);owner.solver_params['max_pcg_iters']=120;owners.append(owner);return owner
    result=E.collect_cell(factory,bank,arrays,cell(),emit,lambda owner:plant)
    assert result['execution_error'] and len(result['missing_episodes'])==12
    assert not owners[0].solver.calls and owners[0].closed

@pytest.mark.parametrize('N,SQP,PCG',[(8,2,100),(16,3,100),(16,2,120),(True,2,100)])
def test_invalid_cell(data,N,SQP,PCG):
    bank,_=data;c=cell();c.update(N=N,SQP=SQP,PCG=PCG)
    with pytest.raises(ValueError):P.validate_case({'parameters':c},bank)


def test_held_worker_before_native_import_and_preserves_receipt(tmp_path):
    before={name for name in sys.modules if name.startswith('bsqp.bsqpN')}
    with pytest.raises(RuntimeError,match='held'):W.run_case({'parameters':cell()},tmp_path/'held')
    receipt=json.loads((tmp_path/'held/metrics.json').read_text())
    assert receipt['outcome']=='execution_held' and receipt['native_calls']==0
    assert {name for name in sys.modules if name.startswith('bsqp.bsqpN')}==before
    with pytest.raises(ValueError,match='Fresh'):W.run_case({'parameters':cell()},tmp_path/'held')

@pytest.fixture
def release_bundle(tmp_path,data):
    import subprocess
    bank,_=data;commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    hashes={p:P.sha(ROOT/p) for p in P.SOURCE_FILES}
    module=tmp_path/'test-only-module-bytes';module.write_bytes(b'not an executable module; guard fixture only')
    identity=dict(machine='gtx1070',source_commit=commit,source_sha256=hashes,
        module_path_by_N={'16':str(module),'32':str(module)},module_sha256_by_N={'16':P.sha(module),'32':P.sha(module)})
    expected=P.preflight_protocol()
    metrics=dict(configuration=expected,machine='gtx1070',execution_backend='internal',timing_mode='A',
        task_success=True,physical_pass=True,execution_error=None,outcome='task_passed')
    provenance=dict(source_commit=commit,machine='gtx1070',module_sha256=P.sha(module),source_sha256=hashes,
        task_config_sha256=P.hashlib.sha256((json.dumps(expected,indent=2,allow_nan=False)+'\n').encode()).hexdigest())
    build=dict(source_commit=commit,source_sha256=hashes,module_sha256_by_N=identity['module_sha256_by_N'])
    def saved(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return dict(path=name,sha256=P.sha(path))
    release=dict(source_commit=commit,source_sha256=hashes,bank_manifest_sha256=P.BANK_MANIFEST_SHA256,
        build=saved('build.json',build),preflight=dict(metrics=saved('metrics.json',metrics),provenance=saved('provenance.json',provenance)))
    return bank,release,identity,tmp_path


def test_fully_valid_synthetic_guards_still_hold_before_import(release_bundle):
    bank,release,identity,inputs=release_bundle
    with pytest.raises(RuntimeError,match='execution held'):P.guard({'parameters':cell()},bank,release,inputs,identity)

@pytest.mark.parametrize('mutation',['machine','modeB','physical','source','build','bank','path'])
def test_guard_rejects_wrong_readiness_and_identity(release_bundle,mutation):
    bank,release,identity,inputs=release_bundle
    if mutation in ['machine','modeB','physical']:
        path=inputs/'metrics.json';m=json.loads(path.read_text())
        if mutation=='machine':m['machine']='rtx5060'
        if mutation=='modeB':m['timing_mode']='B'
        if mutation=='physical':m['physical_pass']=False
        path.write_text(json.dumps(m));release['preflight']['metrics']['sha256']=P.sha(path)
    elif mutation=='source':identity['source_commit']='wrong'
    elif mutation=='build':
        path=inputs/'build.json';m=json.loads(path.read_text());m['module_sha256_by_N']['16']='wrong';path.write_text(json.dumps(m));release['build']['sha256']=P.sha(path)
    elif mutation=='bank':release['bank_manifest_sha256']='wrong'
    else:release['build']['path']='../escape.json'
    with pytest.raises(ValueError):P.guard({'parameters':cell()},bank,release,inputs,identity)


def test_public_float64_plant_precision_and_cold_seed_independent_oracle(data):
    """Reconstruct public A prefix without collector seed/step/cast helpers.

    Tiny f64 increments are below f32 resolution. Rounding every interval loses
    them, so exact comparisons detect the previously forced f32 feedback.
    """
    bank,arrays=data;episode=bank['variants']['short12']['episodes'][0]
    owner=Owner('reaching',cell(),bank);events,emit=emit_buffer()
    def precise_plant(q,v,u,dt):
        return (np.asarray(q,dtype=np.float64)+np.asarray(u,dtype=np.float64)*dt+1e-10,
                np.asarray(v,dtype=np.float64)+2e-11)
    report=E.collect_episode(owner,episode,arrays,cell(),emit,precise_plant)
    assert report['completed_updates']==20
    initial=arrays['starts'][0].copy()
    expected_cold=np.zeros(16*(14+7)-7)
    for k in range(16):expected_cold[k*21:k*21+14]=initial
    assert owner.solver.calls[0][0].dtype==np.dtype('float32')
    assert owner.solver.calls[0][2].dtype==np.dtype('float64')
    np.testing.assert_array_equal(owner.solver.calls[0][2][0],expected_cold)
    state=initial
    for slot in range(5):
        received,_,warm=owner.solver.calls[slot]
        assert received.dtype==state.dtype
        np.testing.assert_array_equal(received[0],state)
        if slot:
            # Public shift helper deliberately converts only primal/state anchor
            # to f32; the actual feedback/plant state remains f64 independently.
            assert warm.dtype==np.dtype('float32')
            np.testing.assert_array_equal(warm[0,:14],state.astype(np.float32))
        q=state[:7].copy();v=state[7:].copy()
        for substep in range(10):
            q=np.asarray(q,dtype=np.float64)+np.full(7,slot+1,dtype=np.float64)*.003+1e-10
            v=np.asarray(v,dtype=np.float64)+2e-11
            actual=[r for e,r in events if e=='step_completed' and r['slot']==slot and r['substep']==substep][0]
            np.testing.assert_array_equal(actual['state_after'],np.concatenate([q,v]))
        state=np.concatenate([q,v])
        actual_interval=[r for e,r in events if e=='interval_completed' and r['slot']==slot][0]
        assert actual_interval['new_state'].dtype==np.dtype('float64')
        np.testing.assert_array_equal(actual_interval['new_state'],state)
    assert not np.array_equal(owner.solver.calls[1][0][0],owner.solver.calls[1][0][0].astype(np.float32))
    first=[r for e,r in events if e=='solve_started'][0]
    assert first['warm_start'].dtype==np.dtype('float64')
    assert first['unmeasured_internal_state']['rho'] and first['unmeasured_internal_state']['duals']


def test_readiness_metadata_is_own_held_A_not_inherited_B_release():
    p=P.preflight_protocol()
    assert p['timing_mode']=='A' and p['offline_timing']=='controller_dt'
    assert p['release_status'].startswith('held:') and p['release_machines']==[]
    assert p['release_authorization'] is None and p['release_scope']=='none; preparation only'
    assert 'release_evidence' not in p and 'reviewed_controller_sha256' not in p
    assert 'prior B approval does not release A' in p['clock_decision_record']['scope']
    assert 'A qualification' in p['source_provenance']['previous_B_context']
    assert 'canonical A' in p['family_hold_policy']
    assert p['N']==32 and p['dt_s']==.03 and p['sim_dt_s']==.003
    assert p['solver_params']['qd_cost']==.1 and p['solver_params']['vel_lim_cost']==.1
    assert p['position_threshold_m']==.05 and p['axis_threshold_deg']==30


def test_initial_state_observation_affects_physical_scope(data):
    bank,arrays=data;events,emit=emit_buffer();episode=bank['variants']['short12']['episodes'][0]
    def observer(state,goal,control):
        return dict(physical_pass=control is not None,rough_reached=False,
                    position_error_m=.1,limit_excess=dict(q_rad=.2 if control is None else 0))
    result=E.collect_episode(Owner('reaching',cell(),bank),episode,arrays,cell(),emit,plant,observer)
    assert result['completed_updates']==20 and result['physical_pass'] is False
    initial=[r for e,r in events if e=='observation' and r.get('initial_state')]
    assert len(initial)==1 and initial[0]['time_s']==0 and initial[0]['control'] is None
    np.testing.assert_array_equal(initial[0]['state'],arrays['starts'][0])
    assert result['initial_observation']['limit_excess']['q_rad']==.2
    assert result['task_success'] is False


def test_returned_output_retained_when_stats_normalization_fails(data):
    bank,arrays=data;owner=Owner('reaching',cell(),bank);events,emit=emit_buffer()
    def broken_stats():raise ValueError('injected normalization failure')
    owner.solver.get_stats=broken_stats
    result=E.collect_episode(owner,bank['variants']['short12']['episodes'][0],arrays,cell(),emit,plant)
    returned=[r for e,r in events if e=='solve_returned']
    assert len(returned)==1 and returned[0]['output'].shape==(1,16*21-7)
    assert returned[0]['native_solve_time_ms']==1.234
    assert any(e=='solve_failed' and r['status']=='telemetry_failed' for e,r in events)
    assert result['attempted_solves']==1 and result['completed_updates']==0
