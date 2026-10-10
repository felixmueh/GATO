"""Actual BB pilot branch with a synthetic public/native solver, never CUDA."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pinocchio as pin
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'examples/randomized_multimodal'))
import tiago_reaching_batch_pilot as P
import validate_reaching_mpc as H
import tiago_reaching_campaign as M
import analyze_reaching_campaign as A
from tiago_batching_execution import execute_active_plan

BB=ROOT/'experiments/tiago_batching/bb_protocol.json'
PREFLIGHT=ROOT/'experiments/tiago_batching/bb_preflight.json'


class Clock:
    def __init__(self):self.t=0.
    def __call__(self):return self.t


@pytest.fixture
def harness(tmp_path,monkeypatch):
    clock=Clock();owners=[]
    settings=dict(walls=[.007,.0175,.04,.04],noeligible=set(),fail=None,reach=None,control_scale=1.)
    monkeypatch.setattr(H.time,'perf_counter',clock)
    initial=np.asarray(json.loads(PREFLIGHT.read_text())['initial_state'])
    def constructor(model,path,N,dt,batch_size,**kwargs):
        class Public:
            def __init__(self):
                self.calls=[];self.dual=0;self.rho=0;self.lib=SimpleNamespace(__file__=__file__)
                outer=self
                class Native:
                    def solve(self,warm,*args):
                        index=len(outer.calls);outer.calls.append(warm.copy())
                        clock.t+=settings['walls'][min(index,len(settings['walls'])-1)]
                        if settings['fail']==('solve',index):raise RuntimeError('injected native failure')
                        z=warm.copy()
                        for k in range(N):
                            z[:,k*21:k*21+7]=warm[:,:7]+k
                            if k<N-1:
                                z[:,k*21+14:k*21+21]=0.
                                z[:,k*21+14]=(index*10+k+1)*settings['control_scale']
                        return dict(XU=z,sqp_time_us=987654.,unfiltered=np.array([np.nan,np.inf]))
                self.solver=Native()
            def solve(self,state,reference,warm):
                reply=self.solver.solve(warm,dt,state,reference)
                if settings['fail']==('normalize',len(self.calls)-1):raise ValueError('injected normalization failure')
                return reply['XU'],reply['sqp_time_us']
            def reset_dual(self):self.dual+=1
            def reset_rho(self):self.rho+=1
            def set_f_ext_B(self,force):assert not np.count_nonzero(force)
            def sim_forward(self,x,u,dt):return x.copy()
            def get_stats(self):
                index=len(self.calls)-1
                if settings['fail']==('stats',index):raise RuntimeError('injected stats failure')
                merits=np.full(batch_size,np.nan) if index in settings['noeligible'] else np.arange(batch_size)[::-1].astype(float)
                return dict(final_merit=merits,initial_merit=np.ones(batch_size)*20,
                    step_size=np.ones((5,batch_size)),typed_nonfinite=np.array([np.nan,np.inf]))
        force=pin.StdVec_Force()
        for _ in range(model.njoints):force.append(pin.Force.Zero())
        owner=SimpleNamespace(N=N,model=model,data=model.createData(),actual_f_ext=force,solver=Public())
        owners.append(owner);return owner
    monkeypatch.setattr(P,'MPC_GATO',constructor)
    monkeypatch.setattr(P,'rk4',lambda model,data,q,v,u,h,force:(q+h*u,v.copy()))
    monkeypatch.setattr(P,'state_cost',lambda *a:np.zeros(3))
    def metrics(model,data,x,goal):
        reached=settings['reach'] is not None and x[0]-initial[0]>=settings['reach']
        return dict(xyz_error=0. if reached else 1.,tool_axis_error_deg=0.,velocity_l1=0.,orientation_error_deg=0.)
    monkeypatch.setattr(P,'metrics',metrics)
    args=SimpleNamespace(protocol=BB,preflight_input=PREFLIGHT,output=tmp_path,
        pool=ROOT/'experiments/tiago_batching/inputs',evidence_kind='synthetic_test',seed=20260925,
        feedback=.03,sim_dt=.007,seconds=.1,position_tolerance=.05,tool_axis_tolerance_deg=30.,
        velocity_l1_tolerance=1.,prefix_guard=False,qd_cost=None,vel_lim_cost=None)
    return args,settings,owners,clock


def folder(args,batch=1):return args.output/f'preflight_dt_0.03_B{batch}'


def load(path):return json.loads(path.read_text())


def test_actual_branch_old_controls_new_u0_and_independent_warm_clock(harness):
    args,settings,owners,clock=harness
    result=P.run_episode(args,-1,1,.03);root=folder(args)
    assert result['complete'] and result['timing_mode']=='B'
    rows=[json.loads(line) for line in (root/'progress.jsonl').read_text().splitlines()]
    assert rows[0]['activation_time_s']==0. and rows[0]['wall_seconds']==pytest.approx(.007)
    assert rows[1]['feedback_time_s']==pytest.approx(.03)
    assert rows[1]['activation_time_s']==pytest.approx(.0475)
    assert rows[2]['warm_start_shift_s']==pytest.approx(.0175)
    assert rows[2]['previous_optimizer_reference_time_s']==pytest.approx(.03)
    with np.load(root/'executed.npz') as a:
        controls,times,ids,ages=(a[k] for k in ('controls','times','plan_solve_indices','active_control_ages_s'))
        assert np.all(controls[times[:-1]<.03-1e-10,0]==1.)
        assert np.all(controls[(times[:-1]>=.03-1e-10)&(times[:-1]<.0475-1e-10),0]==2.)
        assert controls[np.where(ids==1)[0][0],0]==11. and ages[np.where(ids==1)[0][0]]==0.
        assert 12. in controls[ids==1,0]  # knot boundary inside40ms call
        assert times[-1]==pytest.approx(.1) and np.all(np.diff(times)>0.) and np.max(np.diff(times))<=.007+1e-12
    with np.load(root/'step_0001_returned.npz') as ret,np.load(root/'step_0002_input.npz') as inp:
        expected=H.shift_packed_trajectory_warm_start(ret['optimized'][0],inp['actual_state'][0],14,7,32,.0175,.03)
        np.testing.assert_allclose(inp['candidates'][0],expected,atol=3e-6)
    assert owners[0].solver.dual==1 and owners[0].solver.rho==4
    assert result['solve_wall_seconds']==pytest.approx(.1045) and result['simulated_seconds']==pytest.approx(.1)
    assert rows[-1]['activation_status']=='discarded_task_terminal'
    assert rows[-1]['solve_arrival_time_s']==pytest.approx(.1275)
    assert result['execution_intervals'][-1]['timeout_clipped']
    assert all(r['gpu_microseconds']==987654. for r in rows)
    with np.load(root/'step_0003_returned.npz') as ret:
        assert np.isnan(ret['native__unfiltered'][0]) and np.isinf(ret['native__unfiltered'][1])
    events=[load_record for load_record in map(json.loads,(root/'execution.jsonl').read_text().splitlines())]
    assert sum(e['event']=='step_started' for e in events)==sum(e['event']=='step_completed' for e in events)


def test_actual_branch_holds_last_control_past_plan_end(harness):
    args,settings,owners,clock=harness;args.seconds=1.06;settings['walls']=[.003,1.03]
    result=P.run_episode(args,-1,1,.03)
    with np.load(folder(args)/'executed.npz') as a:
        mask=a['times'][:-1]>=.93
        assert np.all(a['controls'][mask,0]==31.)
    assert result['simulated_seconds']==pytest.approx(1.06)
    assert result['execution_intervals'][-1]['requested_s']==pytest.approx(1.03)


@pytest.mark.parametrize('cold',[True,False])
def test_noeligible_preserves_task_success_separately_and_pays_old_plan(harness,cold):
    args,settings,owners,clock=harness
    settings['noeligible']={0 if cold else 1};settings['walls']=[.002,.02];settings['reach']=.06
    result=P.run_episode(args,-1,1,.03)
    assert result['controller_failure']=='no finite optimized plan, merit, and nominal prefix'
    assert result['task_success']==(not cold)
    assert result['outcome']==('failed' if cold else 'reached')
    assert result['simulated_seconds']==pytest.approx(0. if cold else .05)
    row=load(folder(args)/f'step_{0 if cold else 1:04d}_clock.json')
    assert row['winner'] is None and row['activation_time_s'] is None
    assert row['activation_status']==('discarded_no_eligible_bootstrap' if cold else 'discarded_task_terminal')
    if not cold:
        record=dict(pair_id=-1,batch=1,attempts=[dict(state='completed',summary_path=str((folder(args)/'summary.json').relative_to(args.output)))])
        analyzed,_,_=A.episode(args.output,record,A.limits(),{})
        assert analyzed['reached'] and analyzed['controller_failure']==result['controller_failure']
        assert analyzed['selection_failure']==result['selection_failure']


def test_reach_during_oldplan_discard_keeps_eligible_output(harness):
    args,settings,owners,clock=harness;settings['reach']=.06;settings['walls']=[.001,.02]
    result=P.run_episode(args,-1,1,.03)
    assert result['task_success'] and result['controller_failure'] is None
    row=load(folder(args)/'step_0001_clock.json');assert row['winner']==0
    assert row['activation_status']=='discarded_task_terminal' and row['activation_time_s'] is None
    assert (folder(args)/'step_0001_returned.npz').exists()


@pytest.mark.parametrize('wall',[0.,-1.,np.nan,np.inf])
def test_invalid_paid_clock_stops_with_saved_return_and_no_fiction(harness,wall):
    args,settings,owners,clock=harness;settings['walls']=[.001,wall]
    with pytest.raises(ValueError,match='finite and positive'):P.run_episode(args,-1,1,.03)
    root=folder(args);assert load(root/'summary.json')['simulated_seconds']==pytest.approx(.03)
    assert (root/'step_0001_returned.npz').exists() and (root/'technical_failure.json').exists()


@pytest.mark.parametrize('stage',['solve','normalize','stats'])
def test_solver_failures_preserve_raw_and_last_completed_plant(harness,stage):
    args,settings,owners,clock=harness;settings['fail']=(stage,1)
    with pytest.raises((RuntimeError,ValueError)):P.run_episode(args,-1,1,.03)
    root=folder(args);assert load(root/'summary.json')['simulated_seconds']==pytest.approx(.03)
    assert (root/'step_0001_input.npz').exists() and (root/'step_0001_failure.json').exists()
    if stage!='solve':assert (root/'step_0001_returned.npz').exists()
    assert (root/'receipt.json').exists()


def test_b16_keeps_original_rng_rho_slotduals_and_fresh_candidates(harness):
    args,settings,owners,clock=harness;args.seconds=.04;settings['walls']=[.001,.01];settings['control_scale']=100.
    result=P.run_episode(args,-1,16,.03);root=folder(args,16)
    calls=owners[0].solver.calls
    np.testing.assert_array_equal(calls[0],np.tile(calls[0][0],(16,1)))
    assert len(calls)==2 and owners[0].solver.dual==1 and owners[0].solver.rho==2
    row=load(root/'step_0001_clock.json');assert row['winner']==15 and len(row['proposal'])==15
    assert [m['amplitude_rad'] for m in row['proposal']]==[.02]*5+[.05]*5+[.1]*5
    with np.load(root/'step_0000_returned.npz') as r,np.load(root/'step_0001_input.npz') as i:
        nominal=H.shift_packed_trajectory_warm_start(r['optimized'][15],i['actual_state'][0],14,7,32,.03,.03)
        expected,metadata=P.perturbations(owners[0].model,nominal,16,.03,np.random.default_rng(np.random.SeedSequence([20260925,0,30000])))
        np.testing.assert_array_equal(i['candidates'],expected)
    assert (root/'step_0001_prefixes.npz').exists() and not result['physical']['physical_valid']


def test_bb_roster_preflight_and_manager_identity_watchdog(tmp_path,monkeypatch):
    protocol=H.load_protocol(BB);ba=H.load_protocol()
    assert protocol['task_ids']==ba['task_ids'] and len(protocol['roster']['pair_records'])==100
    pre=load(PREFLIGHT);original=load(ROOT/'experiments/tiago_batching/preflight.json')
    for key in original.keys()-{'name','campaign_id','timing_mode','offline_timing','status','repeat_policy','geometry_and_cost_provenance'}:
        assert pre[key]==original[key]
    for key in original['geometry_and_cost_provenance'].keys()-{'scope'}:
        assert pre['geometry_and_cost_provenance'][key]==original['geometry_and_cost_provenance'][key]
    assert pre['geometry_and_cost_provenance']['scope']==original['geometry_and_cost_provenance']['scope'].replace('new A cohort','new B cohort')
    assert pre['timing_mode']=='B' and pre['offline_timing']=='solve_time'
    args=SimpleNamespace(protocol=BB,pool=ROOT/'experiments/tiago_batching/inputs',pair_manifest=ROOT/'experiments/tiago_batching/historical_manifest.json',count=100,sample_seed=20260926,wall_timeout=None)
    with pytest.raises(ValueError,match='explicit calibrated'):M.prepare(args,tmp_path)
    args.wall_timeout=321.;monkeypatch.setattr(M,'provenance',lambda *a:dict(solver_params=protocol['solver_params']))
    manifest,status=M.prepare(args,tmp_path)
    assert manifest['timing_mode']=='B' and len(status['episodes'])==200 and manifest['wall_timeout_seconds']==321.
    args.protocol=H.PROTOCOL_PATH
    with pytest.raises(ValueError,match='cross-mode'):M.resume(args,tmp_path)
    fake=tmp_path/'changed.json';changed=dict(protocol,task_ids=list(reversed(protocol['task_ids'])))
    fake.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='identity'):H.load_protocol(fake)


def test_analyzer_rejects_cross_mode_and_keeps_noeligible_flag():
    with pytest.raises(ValueError,match='timing_mode'):
        A.validate_campaign_episode({'timing_mode':'A'},{'timing_mode':'B'})


def test_proposal_recording_and_statistics_delays_are_unpaid(harness,monkeypatch):
    args,settings,owners,clock=harness;args.seconds=.05;settings['walls']=[.001,.02]
    original_save=np.savez_compressed;original_proposals=P.perturbations
    def slow_save(*a,**kw):
        clock.t+=.005
        return original_save(*a,**kw)
    def slow_proposals(*a,**kw):
        clock.t+=.111
        return original_proposals(*a,**kw)
    monkeypatch.setattr(np,'savez_compressed',slow_save)
    monkeypatch.setattr(P,'perturbations',slow_proposals)
    result=P.run_episode(args,-1,1,.03)
    assert result['simulated_seconds']==pytest.approx(.05)
    assert result['wall_solve_times_s']==pytest.approx([.02])
    assert result['overheads']['proposal_generation_wall_s']==pytest.approx(.111)
    assert result['overheads']['raw_recording_wall_s']>=.03
    assert result['elapsed_wall_seconds']>.15


@pytest.mark.parametrize('nonfinite',[False,True])
def test_partial_plant_failures_keep_attempted_and_completed_substeps(harness,monkeypatch,nonfinite):
    args,settings,owners,clock=harness;args.sim_dt=.003
    original=P.rk4;calls=0
    # Prefix screening also uses RK4; identify actual replay through its helper.
    def rk4(model,data,q,v,u,h,force):return original(model,data,q,v,u,h,force)
    monkeypatch.setattr(P,'rk4',rk4)
    import tiago_batching_execution as E
    original_execute=E.execute_active_plan
    def faulty_execute(*a,**kw):
        integrate=kw['integrate']
        def faulty(state,u,h):
            nonlocal calls
            calls+=1
            if calls==3:
                if nonfinite:return np.full(14,np.nan)
                raise RuntimeError('injected plant failure')
            return integrate(state,u,h)
        kw['integrate']=faulty
        return original_execute(*a,**kw)
    monkeypatch.setattr(E,'execute_active_plan',faulty_execute)
    with pytest.raises((RuntimeError,FloatingPointError)):P.run_episode(args,-1,1,.03)
    root=folder(args)
    with np.load(root/'executed.npz') as a:
        assert len(a['controls'])==(3 if nonfinite else 2)
        assert a['times'][-1]==pytest.approx(.009 if nonfinite else .006)
        if nonfinite:assert np.isnan(a['states'][-1]).all()
    events=list(map(json.loads,(root/'execution.jsonl').read_text().splitlines()))
    assert sum(e['event']=='step_started' for e in events)==3
    assert events[-1]['event']=='step_failed'
    assert load(root/'summary.json')['controller_failure'].startswith('technical execution failure')


def test_recording_failure_stops_before_execution_and_keeps_solve_reply(harness,monkeypatch):
    args,settings,owners,clock=harness
    original=P.write_json
    def fail_once(path,value):
        if path.name=='step_0001_clock.json' and value.get('status')=='completed' and value.get('activation_status')=='pending':
            raise OSError('injected clock recording failure')
        return original(path,value)
    monkeypatch.setattr(P,'write_json',fail_once)
    with pytest.raises(OSError):P.run_episode(args,-1,1,.03)
    root=folder(args);assert (root/'step_0001_returned.npz').exists()
    assert load(root/'summary.json')['simulated_seconds']==pytest.approx(.03)
    assert load(root/'step_0001_clock.json')['activation_status']=='discarded_execution_failure'


def test_public_b_arguments_fail_before_constructor(harness):
    args,settings,owners,clock=harness;args.evidence_kind='measurement'
    with pytest.raises(ValueError,match='Frozen BB arguments'):P.run_episode(args,-1,1,.03)
    assert not owners
    args.evidence_kind='synthetic_test';args.prefix_guard=True
    with pytest.raises(ValueError,match='forbids'):P.run_episode(args,-1,1,.03)
    assert not owners


def test_failed_call_clock_does_not_relabel_previous_bootstrap(harness):
    args,settings,owners,clock=harness;settings['fail']=('solve',1)
    with pytest.raises(RuntimeError):P.run_episode(args,-1,1,.03)
    root=folder(args)
    assert load(root/'step_0000_clock.json')['status']=='completed'
    assert load(root/'step_0000_clock.json')['activation_status']=='activated_bootstrap'
    assert load(root/'step_0001_clock.json')['status']=='failed'


def test_assessment_failure_preserves_return_and_stops_before_paid_motion(harness,monkeypatch):
    args,settings,owners,clock=harness;original=P.assess;calls=0
    def fail(*a,**kw):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('injected assessment failure')
        return original(*a,**kw)
    monkeypatch.setattr(P,'assess',fail)
    with pytest.raises(RuntimeError):P.run_episode(args,-1,1,.03)
    root=folder(args);assert (root/'step_0001_returned.npz').exists()
    assert load(root/'summary.json')['simulated_seconds']==pytest.approx(.03)
    assert load(root/'step_0001_clock.json')['activation_status']=='discarded_execution_failure'


def test_changed_preflight_is_rejected_before_constructor(harness):
    args,settings,owners,clock=harness
    changed=load(PREFLIGHT);changed['goal'][0]+=.1
    substitute=args.output/'changed-preflight.json';substitute.write_text(json.dumps(changed))
    args.preflight_input=substitute
    with pytest.raises(ValueError,match='preflight input differs'):P.run_episode(args,-1,1,.03)
    assert not owners


def test_bb_manager_requires_explicit_watchdog_before_creating_campaign(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['campaign','--protocol',str(BB),'--output',str(tmp_path/'new')])
    with pytest.raises(SystemExit) as e:M.main()
    assert e.value.code==2 and not (tmp_path/'new').exists()


def test_bb_manager_holds_after_mock_child_setup_failure(tmp_path,monkeypatch):
    protocol=H.load_protocol(BB)
    args=SimpleNamespace(protocol=BB,pool=ROOT/'experiments/tiago_batching/inputs',pair_manifest=ROOT/'experiments/tiago_batching/historical_manifest.json',count=100,sample_seed=20260926,wall_timeout=321.)
    monkeypatch.setattr(M,'provenance',lambda *a:dict(solver_params=protocol['solver_params']))
    manifest,status=M.prepare(args,tmp_path)
    def fail(*a,**kw):raise OSError('injected child setup failure, no child launched')
    monkeypatch.setattr(M.subprocess,'Popen',fail)
    M.run_attempt(args,tmp_path,manifest,status,status['episodes'][0],args.pool)
    assert status['episodes'][0]['state']=='technical_failure' and status['family_hold']
    assert all(e['state']=='pending' for e in status['episodes'][1:])
    command=status['episodes'][0]['attempts'][0]['command']
    assert command[-2:]==['--protocol',str(BB)]
    assert (tmp_path/status['episodes'][0]['attempts'][0]['receipt_path']).exists()
