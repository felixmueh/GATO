"""Focused CPU-only policy checks; fake native solver, no CUDA imports/calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

study = load('batch_study_cpu', HERE/'run_batching.py')
proposals = load('batch_proposals_cpu', HERE/'ba_trajectory_proposals.py')
common = load('batch_common_cpu', ROOT/'python/bsqp/common.py')


def test_historical_proposals_keep_nominal_start_and_declared_scales():
    model = pin.buildModelFromUrdf(str(ROOT/study.MODEL))
    x = np.r_[np.zeros(7), np.zeros(7)]
    nominal = common.initialize_warm_start(x,32,14,7).astype(np.float32)
    seeds, meta = proposals.perturbations(model,nominal,16,.03,np.random.default_rng(123))
    assert seeds.shape == (16,665)
    np.testing.assert_array_equal(seeds[0],nominal)
    np.testing.assert_array_equal(seeds[:,:14],np.repeat(x[None],16,axis=0))
    assert [m['amplitude_rad'] for m in meta] == [.02]*5+[.05]*5+[.1]*5
    np.testing.assert_allclose([m['peak_delta_q_rad'] for m in meta],[m['amplitude_rad'] for m in meta])
    one, metadata = proposals.perturbations(model,nominal,1,.03,np.random.default_rng(123))
    np.testing.assert_array_equal(one,nominal[None]);assert metadata == []


def test_merit_selection_keeps_finite_physical_violator_and_lane_tie_break():
    model = SimpleNamespace(lowerPositionLimit=-np.ones(7)*.1,upperPositionLimit=np.ones(7)*.1,
                            velocityLimit=np.ones(7),effortLimit=np.ones(7),names=['universe']+[f'j{i}' for i in range(7)])
    def rk4(model,data,q,v,u,h,f):return q+h*u,v.copy()
    controller = SimpleNamespace(model=model,data=None,actual_f_ext=np.zeros(6))
    fake_common = SimpleNamespace(rk4=rk4)
    packed = np.zeros((3,665),dtype=np.float32)
    packed[1,14:21]=10.;packed[2]=np.nan
    winner,rows,prefixes=study.assess_candidates(np,fake_common,controller,packed,np.zeros(14),0.,.03,np.array([2.,1.,0.]))
    assert winner==1 and rows[1]['eligible'] and not rows[1]['prefix_limits']['physical_pass']
    assert not rows[2]['eligible'] and len(prefixes[1][0])==11
    winner,_,_=study.assess_candidates(np,fake_common,controller,packed,np.zeros(14),0.,.03,np.array([1.,1.,0.]))
    assert winner==0
    winner,_,_=study.assess_candidates(np,fake_common,controller,packed,np.zeros(14),0.,.03,np.full(3,np.nan))
    assert winner is None


def test_coupled_work_uses_all_pcg_rows_and_separate_selected_counts():
    stats=dict(pcg_iters=np.array([[3,4],[0,1]]),sqp_iters=np.array([2,2]),ls_num_iters=1,
        step_size=np.ones((1,2)),kkt_converged=np.array([0,1]),initial_merit=np.array([2.,3.]),final_merit=np.array([1.,2.]))
    w=study.work_summary(np,stats,1000,2,0)
    assert w['pcg_iteration_sum']==8 and w['selected_lane_pcg_iteration_sum']==3
    assert w['all_lane_sqp_sum']==4 and w['pcg_count_rows']==2
    assert not w['zero_pcg_outer_exit'] and w['stop_reason']=='existing_mixed_outer_exit'
    stats['pcg_iters'][-1]=0
    assert study.work_summary(np,stats,1000,2,0)['zero_pcg_outer_exit']
    stats['ls_num_iters']=2;stats['step_size']=np.ones((2,2))
    assert study.work_summary(np,stats,2,2,0)['stop_reason']=='budget_exhausted'


def test_episode_records_first_solve_and_retains_slot_state(tmp_path):
    cfg=json.loads((HERE/'protocols/batching-B16.json').read_text())
    instances=[]
    class Solver:
        def __init__(self):self.calls=[];self.resets=[0,0]
        def reset_dual(self):self.resets[0]+=1
        def reset_rho(self):self.resets[1]+=1
        def set_f_ext_B(self,x):assert np.array_equal(x,np.zeros((16,6)))
        def solve(self,x,ref,seeds):
            self.calls.append((x.copy(),seeds.copy()))
            np.testing.assert_array_equal(x,np.repeat(x[:1],16,axis=0))
            out=seeds.copy()
            for lane in range(16):out[lane,14:21]=(lane+1)*.001
            self.out=out.copy()
            return out,1.
        def get_stats(self):
            merit=np.arange(16,dtype=float)+2;merit[1]=-1.
            return dict(pcg_iters=np.zeros((1,16),dtype=int),sqp_iters=np.ones(16,dtype=int),
                step_size=np.empty((0,16)),min_merit=np.empty((0,16)),kkt_converged=np.zeros(16,dtype=int),
                initial_merit=np.arange(16,dtype=float),final_merit=merit,ls_num_iters=0)
    class MPC:
        def __init__(self,model,*args,**kw):
            assert kw['batch_size']==16
            self.model=model;self.data=model.createData();self.actual_f_ext=np.zeros(6)
            self.solver=Solver();instances.append(self)
    fake_common=SimpleNamespace(initialize_warm_start=common.initialize_warm_start,
        shift_packed_trajectory_warm_start=common.shift_packed_trajectory_warm_start,
        rk4=lambda model,data,q,v,u,h,f:(q+h*u,v.copy()))
    counter=[0]
    def observation(*args):
        counter[0]+=1
        return dict(position_error_m=.04 if counter[0]>=21 else .06,tool_axis_error_deg=90.,
                    orientation_error_deg=100.,velocity_l1_rad_s=0.)
    args=SimpleNamespace(source=ROOT,pool=Path('/workspace/GATO-experiment-runner-simple/experiments/inputs/batching'),
        output=tmp_path,preflight_input=None)
    with patch.object(study,'metrics',observation):
        result=study.run_episode((np,pin,MPC,fake_common),args,cfg,9651)
    solver=instances[0].solver
    assert result['task_success'] and result['feedback_solves']==2 and len(solver.calls)==2
    assert solver.resets==[1,1] and instances[0].force_estimator is None
    np.testing.assert_array_equal(solver.calls[0][1],np.repeat(solver.calls[0][1][:1],16,axis=0))
    case=tmp_path/'pair_09651'
    with np.load(case/'executed.npz') as a:
        assert len(a['controls'])==20
        np.testing.assert_allclose(a['controls'],.002,atol=1e-9)
        actual=a['states'][10]
    with np.load(case/'step_0000_returned.npz') as a:previous=a['optimized'][1]
    expected=common.shift_packed_trajectory_warm_start(previous,actual,14,7,32,.03,.03)
    with np.load(case/'step_0001_input.npz') as a:
        np.testing.assert_array_equal(a['candidates'][0],expected)
        assert a['proposal_coefficients'].shape==(15,3,7)
    assert result['all_lane_sqp_sum']==32 and result['actual_sqp_sum']==2
