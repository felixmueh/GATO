"""CPU model/mock checks for the recovered current-default offline portfolio."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pinocchio as pin
import pytest

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/'examples/randomized_multimodal'
sys.path.insert(0,str(SCRIPTS))
import tiago_reaching_batch_pilot as P
import tiago_reaching_campaign as M
import validate_reaching_mpc as H


@pytest.fixture
def model():return pin.buildModelFromUrdf(str(H.MODEL))


def test_actual_model_pool_preserves_cold_lane_and_goalblind_envelope(model):
    task=json.loads((ROOT/'experiments/tiago_batching/preflight.json').read_text())
    state=np.asarray(task['initial_state']);nominal=H.initialize_warm_start(state,P.N,14,7).astype(np.float32)
    seeds,metadata=P.perturbations(model,nominal,16,.03,np.random.default_rng(7))
    assert seeds.shape==(16,665) and seeds.dtype==np.float32 and len(metadata)==15
    np.testing.assert_array_equal(seeds[0],nominal)
    x,u=P.unpack(seeds);np.testing.assert_array_equal(x[:,0],np.tile(state.astype(np.float32),(16,1)))
    for lane in range(1,16):
        assert metadata[lane-1]['amplitude_rad']==(.02,.05,.1)[(lane-1)//5]
        assert abs(float(np.max(abs(x[lane,:,:7]-x[0,:,:7])))-metadata[lane-1]['amplitude_rad'])<3e-7
        assert np.isfinite(u[lane]).all()
    single,meta=P.perturbations(model,nominal,1,.03,np.random.default_rng(7))
    np.testing.assert_array_equal(single[0],nominal);assert meta==[]


def controller_for_model(model):
    forces=pin.StdVec_Force()
    for _ in range(model.njoints):forces.append(pin.Force.Zero())
    return SimpleNamespace(model=model,data=model.createData(),actual_f_ext=forces,
        solver=SimpleNamespace(sim_forward=lambda x,u,dt:x.copy()))


def test_original_finite_prefix_merit_selection_without_bounds_veto(model):
    task=json.loads((ROOT/'experiments/tiago_batching/preflight.json').read_text());state=np.asarray(task['initial_state'])
    seed=H.initialize_warm_start(state,P.N,14,7).astype(np.float32);plans=np.tile(seed,(2,1));plans[0,14]=model.effortLimit[0]*2
    controller=controller_for_model(model)
    winner,lanes,_=P.assess(controller,plans,state,.03,.003,np.array([-1.,0.]),.03)
    assert winner==0 and lanes[0]['finite'] and not lanes[0]['prefix_physical_valid']
    plans[0,14]=np.nan
    winner,_,_=P.assess(controller,plans,state,.03,.003,np.array([-1.,0.]),.03);assert winner==1
    winner,_,_=P.assess(controller,plans,state,.03,.003,np.array([np.nan,np.nan]),.03);assert winner is None
    plans[0]=seed
    winner,_,_=P.assess(controller,plans,state,.03,.003,np.zeros(2),.03);assert winner==0


def test_nonfinite_prefix_is_ineligible_even_when_plan_and_merit_finite(model,monkeypatch):
    state=np.zeros(14);plans=np.zeros((2,665));plans[0,14]=1.
    def prefix(controller,x,u,*args):return (np.full((2,14),np.nan) if u[0]==1. else np.tile(x,(2,1))),np.array([.03])
    monkeypatch.setattr(P,'prefix',prefix)
    winner,rows,_=P.assess(controller_for_model(model),plans,state,.03,.003,np.array([-1.,0.]),.03)
    assert winner==1 and rows[0]['finite'] and not rows[0]['prefix_physical']['finite']


class FakeNative:
    def solve(self,warm,*args):return dict(XU=warm.copy(),sqp_time_us=123,initial_merit=np.array([np.nan]))


class FakePublic:
    def __init__(self,fail=None):self.solver=H.NativeCapture(FakeNative());self.fail=fail
    def solve(self,state,reference,warm):
        reply=self.solver.solve(warm,.03,state,reference)
        if self.fail=='decode':raise ValueError('injected normalization failure')
        return reply['XU'],reply['sqp_time_us']
    def get_stats(self):
        if self.fail=='stats':raise RuntimeError('injected stats failure')
        return dict(initial_merit=np.array([np.nan]),final_merit=np.array([np.inf]),sqp_iters=np.array([5]))


@pytest.mark.parametrize('failure',[None,'decode','stats'])
def test_raw_input_reply_and_numeric_stats_survive_failures(tmp_path,failure):
    controller=SimpleNamespace(N=32,solver=FakePublic(failure));state=np.ones((1,14),dtype=np.float64)
    reference=np.ones((1,192),dtype=np.float32);warm=np.ones((1,665),dtype=np.float32);cfg=H.load_protocol()['solver_params']
    if failure:
        with pytest.raises((ValueError,RuntimeError)):H.solve_recorded(controller,state,reference,warm,tmp_path,0,cfg)
    else:H.solve_recorded(controller,state,reference,warm,tmp_path,0,cfg)
    with np.load(tmp_path/'step_0000_input.npz',allow_pickle=False) as a:
        assert a['actual_state'].dtype==np.float64;np.testing.assert_array_equal(a['candidates'],warm);np.testing.assert_array_equal(a['caps'],[32,1,5,120])
    with np.load(tmp_path/'step_0000_returned.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['native__XU'],warm);assert np.isnan(a['native__initial_merit'][0])
    if failure is None:
        with np.load(tmp_path/'step_0000_stats.npz',allow_pickle=False) as a:assert np.isnan(a['stats__initial_merit'][0]) and np.isinf(a['stats__final_merit'][0])
    else:assert (tmp_path/'step_0000_failure.json').exists()


def test_exact100_frozen_manifest_new_defaults_and_no_old_resume(tmp_path,monkeypatch):
    args=SimpleNamespace(pool=ROOT/'experiments/tiago_batching/inputs',pair_manifest=ROOT/'experiments/tiago_batching/historical_manifest.json',count=100,sample_seed=20260926,wall_timeout=120.)
    monkeypatch.setattr(M,'provenance',lambda _:dict(solver_params=H.load_protocol()['solver_params']))
    manifest,status=M.prepare(args,tmp_path)
    original=json.loads(args.pair_manifest.read_text());assert manifest['pair_ids']==original['pair_ids'] and len(status['episodes'])==200
    assert manifest['N']==32 and manifest['simulation_step']==.003 and manifest['solver_params']['ctrl_lim_cost']==.05
    manifest.pop('cohort_id');M.atomic_json(tmp_path/'manifest.json',manifest)
    with pytest.raises(ValueError,match='Historical cohort'):M.resume(args,tmp_path)
    other=tmp_path/'other';other.mkdir();args.count=2
    with pytest.raises(ValueError,match='exact100'):M.prepare(args,other)


def test_preflight_hook_uses_same_pose_preset_and_model(tmp_path):
    args=SimpleNamespace(preflight_input=ROOT/'experiments/tiago_batching/preflight.json')
    model,data,state,endpoint,goal,reference,si,gi=P.inputs(args,-1)
    task=json.loads(args.preflight_input.read_text());assert reference.shape==(192,) and (si,gi)==(-1,-1)
    np.testing.assert_array_equal(reference[:6],np.asarray(task['goal'],dtype=np.float32))
    assert np.linalg.norm(H.pose(model,data,state[:7]).translation-goal.translation)>.05
    expected=dict(H.TIAGO_TRACKING_SOLVER_PARAMS,qd_cost=.1,vel_lim_cost=.1,ctrl_lim_cost=.05,ee_orient_cost=.1,ee_orient_N_cost=2.,max_sqp_iters=5,max_pcg_iters=120)
    assert H.load_protocol()['solver_params']==expected


def test_manager_holds_on_technical_failure_without_starting_remaining_cases(tmp_path,monkeypatch):
    status={'episodes':[dict(pair_id=3626,batch=1,state='pending',attempts=[]),dict(pair_id=3626,batch=16,state='pending',attempts=[])]}
    monkeypatch.setattr(M,'prepare',lambda *a:({'pair_ids':[3626]},status));started=[]
    def fail(args,root,manifest,state,episode,pool):
        started.append(episode['batch']);episode.update(state='technical_failure',attempts=[dict(error='injected worker failure')])
    monkeypatch.setattr(M,'run_attempt',fail)
    monkeypatch.setattr(sys,'argv',['campaign','--output',str(tmp_path)])
    assert M.main()==1 and started==[1] and status['episodes'][1]['state']=='pending' and status['family_hold']['pair_id']==3626


def test_two_mock_solves_keep_original_cold_warm_rho_and_dual_policy(tmp_path,monkeypatch):
    owners=[]
    class Public:
        def __init__(self,model,B):
            self.calls=[];self.dual=0;self.rho=0;self.B=B;self.lib=SimpleNamespace(__file__=__file__)
            outer=self
            class Native:
                def solve(self,warm,dt,state,reference):
                    outer.calls.append(warm.copy());reply=warm.copy()
                    for lane in range(B):
                        for k in range(P.N-1):
                            body=reply[lane,k*21:k*21+21]
                            body[14:]=pin.rnea(model,model.createData(),body[:7].astype(float),body[7:14].astype(float),np.zeros(7))
                    return dict(XU=reply,sqp_time_us=123)
            self.solver=Native()
        def solve(self,state,reference,warm):
            reply=self.solver.solve(warm,.03,state,reference);return reply['XU'],reply['sqp_time_us']
        def reset_dual(self):self.dual+=1
        def reset_rho(self):self.rho+=1
        def set_f_ext_B(self,wrench):assert not np.count_nonzero(wrench)
        def sim_forward(self,x,u,dt):return x.copy()
        def get_stats(self):return dict(final_merit=np.linspace(1.,0.,self.B),initial_merit=np.ones(self.B)*2,step_size=np.ones((5,self.B)))
    def constructor(model,path,N,dt,batch_size,**kwargs):
        owner=controller_for_model(model);owner.N=N;owner.dt=dt;owner.solver=Public(model,batch_size);owner.solver_params=kwargs['solver_params'];owners.append(owner);return owner
    monkeypatch.setattr(P,'MPC_GATO',constructor)
    args=SimpleNamespace(preflight_input=ROOT/'experiments/tiago_batching/preflight.json',output=tmp_path,pool=ROOT/'experiments/tiago_batching/inputs',
        evidence_kind="synthetic_test",seed=20260925,feedback=.03,sim_dt=.003,seconds=.06,position_tolerance=.05,tool_axis_tolerance_deg=30.,velocity_l1_tolerance=1.,prefix_guard=False,qd_cost=None,vel_lim_cost=None)
    result=P.run_episode(args,-1,16,.03);owner=owners[0];calls=owner.solver.calls
    assert result['feedback_solves']==2 and owner.solver.dual==1 and owner.solver.rho==2
    np.testing.assert_array_equal(calls[0],np.tile(calls[0][0],(16,1)))
    assert np.any(calls[1][1:]!=calls[1][0]) and owner.solver_params==H.load_protocol()['solver_params']
    saved=tmp_path/'preflight_dt_0.03_B16'
    assert all((saved/f'step_000{i}_{kind}.npz').exists() for i in [0,1] for kind in ['input','returned','stats'])
    with np.load(saved/'executed.npz',allow_pickle=False) as a:
        assert a['states'].dtype==np.float64 and a['controls'].shape==(20,7) and a['times'][-1]==pytest.approx(.06)


@pytest.mark.parametrize('orphan',[False,True])
def test_interrupted_or_dead_orphan_resume_never_starts_second_child(tmp_path,monkeypatch,orphan):
    args=SimpleNamespace(pool=ROOT/'experiments/tiago_batching/inputs',pair_manifest=ROOT/'experiments/tiago_batching/historical_manifest.json',count=100,sample_seed=20260926,wall_timeout=120.)
    monkeypatch.setattr(M,'provenance',lambda _:dict(solver_params=H.load_protocol()['solver_params']))
    manifest,status=M.prepare(args,tmp_path);episode=status['episodes'][0]
    directory=tmp_path/'attempts/synthetic';directory.mkdir(parents=True);(directory/'episode.log').write_text('synthetic interruption; no actual child launched\n')
    attempt=dict(relative_output='attempts/synthetic',state='running',pid=None);episode.update(state='running',attempts=[attempt])
    if orphan:M.save_status(tmp_path,status)
    else:M.finish_attempt(tmp_path,status,episode,attempt,'interrupted',error='injected interruption')
    started=[];monkeypatch.setattr(M,'run_attempt',lambda *a:started.append(True))
    monkeypatch.setattr(sys,'argv',['campaign','--output',str(tmp_path),'--resume'])
    with pytest.raises(SystemExit) as error:M.main()
    assert error.value.code==2 and not started
    saved=json.loads((tmp_path/'status.json').read_text())
    assert saved['family_hold'] and saved['episodes'][0]['state']=='interrupted' and saved['episodes'][1]['state']=='pending'
