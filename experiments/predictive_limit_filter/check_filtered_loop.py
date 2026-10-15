#!/usr/bin/env python3
"""CPU fake-native loop contract: actual applied bootstrap/stale controls and stop."""
import importlib.util,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pinocchio as pin
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    api=load('original_filter_loop_api',HERE.parent/'random_reaching_baseline/run_frozen_baseline.py')
    mode=load('test_filtered_loop',HERE/'mode_b_filtered.py')
    prep=load('test_filtered_prepare',HERE/'prepare_filtered.py')
    p=prep.make_protocol(Path('/pending/module'),Path('/pending/receipt'),'Jetson')
    p['timeout_s']=.042
    applied=[];instances=[];filters=[];control_calls=[0]
    class Solver:
        def __init__(self):self.n=0;self.resets=[0,0]
        def set_f_ext_B(self,x):assert not np.any(x)
        def reset_dual(self):self.resets[0]+=1
        def reset_rho(self):self.resets[1]+=1
        def solve(self,x,ref,warm):
            self.n+=1;result=warm.copy()
            for k in range(31):result[0,k*21+14:k*21+21]=float(self.n)
            return result,1.
        def get_stats(self):return dict(pcg_iters=np.array([0]),sqp_iters=np.array([1]),step_size=np.array([0.]),min_merit=np.array([0.]),kkt_converged=np.array([0]),initial_merit=np.array([1.]),final_merit=np.array([.5]),ls_num_iters=0)
    class MPC:
        def __init__(self,model,*args,**kw):self.solver=Solver();self.data=model.createData();self.actual_f_ext=np.zeros(6);instances.append(self)
    def plant(model,data,q,v,u,h,fext):
        applied.append(np.array(u));return q.copy(),v.copy()
    common=SimpleNamespace(initialize_warm_start=lambda x,N,nx,nu:np.r_[np.tile(np.r_[x,np.zeros(7)],31),x],
        shift_packed_trajectory_warm_start=lambda z,x,*rest:z.copy(),rk4=plant)
    class StubFilter:
        stop_after=None
        def __init__(self,*args,**kwargs):self.calls=[];filters.append(self)
        def filter(self,x,u):
            self.calls.append(u.copy());control_calls[0]+=1
            stop=self.stop_after is not None and len(self.calls)>=self.stop_after
            return dict(status='no_candidate_certified' if stop else 'guard_forecast_pass',selected_torque=None if stop else (u+.125).tolist(),
                filter_wall_s=.005,nominal_step_overrun=True,guard_certified=not stop,modified=not stop)
    def observation(*args):return dict(position_error_m=.1,tool_axis_error_deg=0.,orientation_error_deg=0.,velocity_l1_rad_s=0.)
    clock=[0.]
    def perf():clock[0]+=.001;return clock[0]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);args=SimpleNamespace(source=SOURCE,pool=SOURCE/'experiments/jetson_random_reaching/inputs',output=root,preflight_input=None,save_plans=False)
        with patch.object(mode._filter_module,'PredictiveLimitFilter',StubFilter),patch.object(api,'metrics',observation),patch.object(mode.time,'perf_counter',perf),patch('builtins.print'):
            r=mode.run_episode_filtered(api,(np,pin,MPC,common),args,p,502)
        assert not r['execution_error'] and r['complete'] and r['physical_pass']
        assert instances[-1].solver.resets==[1,1]
        assert len(filters[-1].calls)==len(applied)>10
        assert np.allclose(filters[-1].calls[0],1.)
        assert all(np.allclose(v,u+.125) for u,v in zip(filters[-1].calls,applied))
        records=[json.loads(x) for x in (root/'pair_00502/solves.jsonl').read_text().splitlines()]
        assert len(records)>1 and records[0]['timing_category']=='startup/cold'
        assert records[1]['executed_prefix_plan_step']==0 and records[1]['charged_solve_s']>0
        assert np.allclose(filters[-1].calls[10],1.),'Oldplan must still supply proposed control during secondsolve'
        with np.load(root/'pair_00502/executed.npz',allow_pickle=False) as z:
            np.testing.assert_allclose(z['controls']-z['proposed_controls'],.125)
        # Filter rejection applies no corresponding plant step and is incomplete.
        args.output=root/'stop';args.output.mkdir();applied.clear();StubFilter.stop_after=2
        with patch.object(mode._filter_module,'PredictiveLimitFilter',StubFilter),patch.object(api,'metrics',observation),patch.object(mode.time,'perf_counter',perf),patch('builtins.print'):
            stopped=mode.run_episode_filtered(api,(np,pin,MPC,common),args,p,502)
        assert len(applied)==1 and len(filters[-1].calls)==2
        assert stopped['failure_class']=='filter_stop' and not stopped['complete'] and not stopped['physical_pass'] and not stopped['shared_setup_failure']
        assert stopped['prefix_physical_pass'] and stopped['predictive_filter']['no_candidate_certified']==1
        assert stopped['actual_sqp_sum']==1,'Returned native work must survive a filter stop'
    # Validator-only predecessor check: no runtime/native load, even though dry_run=False.
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);can=root/'canonical';can.mkdir()
        module=Path('/workspace/GATO-random-reaching-joint-margins/.experiment-modules/q020/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so')
        receipt=SOURCE/'experiments/jetson_random_reaching/inputs/desktop_native_build_receipt.json'
        protocol=prep.make_protocol(module,receipt,'GTX-smoke')
        prep.write(can/'protocol.json',protocol)
        prep.write(can/'cohort_status.json',dict(infrastructure_pass=True,assigned=1,recorded=1))
        row=dict(shared_setup_failure=False,interrupted=False,execution_error='filter stop',failure_class='filter_stop',physical={'finite':True})
        prep.write(can/'summary.json',[row])
        a=SimpleNamespace(source=SOURCE,module=module,module_receipt=receipt,pool=SOURCE/'experiments/jetson_random_reaching/inputs',
            output=root/'first1',stage='first1',pairs=[502],preflight_input=None,dry_run=False)
        prep.validate(a,protocol)
        row.update(execution_error='nonfinite plant',failure_class='numerical',physical={'finite':False})
        prep.write(can/'summary.json',[row])
        try:prep.validate(a,protocol)
        except ValueError as error:assert 'Unexpected canonical' in str(error)
        else:raise AssertionError('Unexpected canonical numerical failure must hold')
    print(json.dumps(dict(status='PASS',native_calls=0,bootstrap_and_stale_plan_filtered=True,only_filtered_controls_applied=True,
        once_only_resets=True,no_command_applies_no_step=True,filter_stop_incomplete_not_physical_pass=True,
        actual_native_work_retained=True,filter_cpu_overrun_descriptive_not_gate=True,canonical_filterstop_allowed_numerical_holds=True)))

if __name__=='__main__':main()
