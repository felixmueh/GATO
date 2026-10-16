#!/usr/bin/env python3
"""Bounded N256 fake-native episode and saved-input contracts; no CUDA import."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import pinocchio as pin

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

w=load('long_worker_contract',HERE/'run_case.py')
adapter=load('long_adapter_contract',HERE/'adapter.py')
launcher=load('long_launcher_contract',HERE/'launch.py')

def packed(state):
    return np.r_[np.tile(np.r_[state,np.zeros(7)],255),state]

class Contracts(unittest.TestCase):
    def test_all30_saved_inputs_dry_run_and_full_reference(self):
        rows=json.loads((HERE/'inputs/tasks.json').read_text())
        model=pin.buildModelFromUrdf(str(ROOT/w.MODEL));data=model.createData()
        with tempfile.TemporaryDirectory() as directory:
            for row in rows:
                args=SimpleNamespace(source=ROOT,module=Path(directory)/'no-native.so',tasks=HERE/'inputs/tasks.json',
                    protocol=HERE/'inputs/protocol.json',output=Path(directory)/row['case_id'],
                    case_id=row['case_id'],preflight_input=None)
                protocol,info=w.prepare(args)
                x,unused,goal,ref,indices=w.task(np,pin,model,data,args,row['case_id'])
                self.assertTrue(np.array_equal(x,np.asarray(row['initial_state'])))
                self.assertEqual(ref.shape,(1,1536));self.assertEqual(packed(x).shape,(5369,))
                self.assertTrue(np.array_equal(ref.reshape(256,6),np.tile(np.asarray(row['goal_reference_single'],dtype=np.float32),(256,1))))
                self.assertTrue(np.array_equal(unused,np.zeros(7)))
                self.assertFalse(args.output.exists())
            traps=[r for r in rows if r['start_type']=='trap']
            self.assertEqual(len(traps),15)
            self.assertTrue(any(np.linalg.norm(r['initial_state'][7:])>0 for r in traps))

    def test_plan_serial30_plus_one_canonical(self):
        plan=launcher.make_plan('CPU-only-source')
        self.assertEqual(plan['scientific_episodes'],30)
        self.assertEqual(plan['diagnostic_canonicals'],1)
        self.assertEqual(len(plan['experiments']),1)
        experiment=plan['experiments'][0]
        self.assertEqual(len(experiment['cases']),30)
        self.assertIn('-DKNOTS=256',experiment['build']['commands'][0])
        self.assertEqual(experiment['build']['artifacts'],['modules/bsqpN256_tiago_right*.so'])
        self.assertEqual([r['parameters']['start_type'] for r in experiment['cases']],['original','trap']*15)

    def episode(self,kind):
        events=[];steps=[];shifts=[]
        model=pin.buildModelFromUrdf(str(ROOT/w.MODEL));data=model.createData()
        initial=np.r_[[-.5,-1.4,-.3,-1.7,0.,-.9,0.],np.ones(7)*.05]
        endpoint=initial[:7].copy();endpoint[0]+=.3
        if kind=='t0':
            endpoint=initial[:7].copy()
        goal=w.pose(pin,model,data,endpoint)
        ref=np.tile(np.r_[goal.translation,pin.rpy.matrixToRpy(goal.rotation)],256).astype(np.float32)[None]
        prior_task=w.task
        w.task=lambda *unused:(initial.copy(),np.zeros(7),goal,ref,
            dict(case_id='test',pair_id=8048,start_type='trap',source_base_time_s=21.))
        class Native:
            def set_tiago_joint_cost_mode(self,mode,q):
                events.append(('mode',mode));self.mode,self.q=mode,q
            def get_tiago_joint_cost_mode(self):
                return dict(mode=self.mode,q_reference=self.q,joint_margin_rad=.2)
        class Solver:
            def set_f_ext_B(self,value):pass
            def reset_dual(self):events.append(('dual',))
            def reset_rho(self):events.append(('rho',))
            def solve(self,x,reference,seed):
                events.append(('solve',))
                self_state,self_u=w.unpack(np,seed[0])
                assert self_state.shape==(256,14) and self_u.shape==(255,7)
                assert np.array_equal(seed[0],packed(initial).astype(np.float32))
                assert np.array_equal(x[0],initial)
                assert reference.shape==(1,1536)
                out=seed.copy()
                if kind=='invalid':out[0,0]=np.nan
                return out,15.
            def get_stats(self):
                return dict(ls_num_iters=0,pcg_iters=np.array([0],dtype=np.int32),sqp_iters=np.array([1],dtype=np.int32),
                    step_size=np.array([0.],dtype=np.float32),min_merit=np.array([1.],dtype=np.float32),
                    kkt_converged=np.array([0],dtype=np.int32),initial_merit=np.array([1.]),final_merit=np.array([1.]))
        class MPC:
            def __init__(self,*args,**kwargs):
                assert kwargs['N']==256 and kwargs['dt']==.03 and kwargs['batch_size']==1
                events.append(('session',));self.solver=Solver();self.data=data;self.actual_f_ext=None
        def rk4(model,data,q,v,u,h,force):
            steps.append(h)
            if kind=='arrival' and len(steps)>=8:
                return endpoint.copy(),np.zeros(7)
            return initial[:7].copy(),initial[7:].copy()
        def shift(opt,x,nx,nu,N,duration,dt):
            shifts.append((N,duration,dt));return packed(x).astype(np.float32)
        common=SimpleNamespace(initialize_warm_start=lambda *args:packed(initial),rk4=rk4,
                               shift_packed_trajectory_warm_start=shift)
        protocol=json.loads((HERE/'inputs/protocol.json').read_text());protocol['timeout_s']=.02
        try:
            with tempfile.TemporaryDirectory() as directory:
                args=SimpleNamespace(source=ROOT,output=Path(directory),preflight_input=None)
                with contextlib.redirect_stdout(io.StringIO()):
                    result=w.run_episode((np,pin,MPC,common,Native()),args,protocol,'test')
                out=Path(directory)/'test'
                self.assertEqual(events.count(('session',)),1)
                self.assertEqual(events.count(('dual',)),1);self.assertEqual(events.count(('rho',)),1)
                self.assertEqual([e for e in events if e[0]=='mode'],[('mode',1)])
                with np.load(out/'inputs.npz') as inputs:
                    states,controls=w.unpack(np,inputs['first_seed'][0])
                    self.assertTrue(np.array_equal(states,np.tile(initial.astype(np.float32),(256,1))))
                    self.assertTrue(np.array_equal(controls,np.zeros((255,7))))
                    self.assertNotIn('source_goal_q',inputs.files)
                if kind=='invalid':
                    self.assertEqual(result['outcome'],'execution_failed')
                    self.assertTrue((out/'invalid_optimized.npz').exists())
                else:
                    count=0 if kind=='t0' else 2
                    self.assertEqual(result['feedback_solves'],count,result)
                    self.assertEqual(result['zero_pcg_exit_solves'],count)
                    if kind in ('arrival','t0'):
                        self.assertTrue(result['task_success'])
                        self.assertAlmostEqual(result['reached_seconds'],0. if kind=='t0' else .02)
                    else:self.assertEqual(result['outcome'],'timeout')
                    self.assertTrue(np.allclose(np.asarray(steps).reshape(-1,4),np.tile([.003,.003,.003,.001],(count,1))))
                    self.assertTrue(all(item==(256,.01,.03) for item in shifts))
                    self.assertTrue(result['physical_pass'])
                    with np.load(out/'work_stats.npz') as stats:
                        if count:self.assertEqual(stats['pcg_iters'].dtype,np.dtype('int32'))
                    with np.load(out/'executed.npz') as saved:
                        self.assertEqual(len(saved['states']),count*4+1)
                        self.assertTrue(np.array_equal(saved['states'][0],initial))
        finally:w.task=prior_task

    def test_nonzero_velocity_seed_and_fractional_shift_arrival(self):self.episode('arrival')
    def test_t0_arrival_no_native_solve(self):self.episode('t0')
    def test_timeout_saved_as_negative(self):self.episode('timeout')
    def test_invalid_plan_saved_as_failure(self):self.episode('invalid')
    def test_canonical_requires_rough_and_physical_pass(self):
        okay=dict(complete=True,preflight_pass=True,task_success=True,physical_pass=True,physical=dict(finite=True),
                  execution_error=None,interrupted=False)
        self.assertTrue(adapter.canonical_pass(okay))
        for key in ('complete','preflight_pass','task_success','physical_pass'):
            self.assertFalse(adapter.canonical_pass(dict(okay,**{key:False})))
        self.assertFalse(adapter.canonical_pass(dict(okay,execution_error='failed')))
    def test_strict_raw_first_breach_and_tolerance_physical(self):
        m=SimpleNamespace(lowerPositionLimit=-np.ones(7),upperPositionLimit=np.ones(7),velocityLimit=np.ones(7)*2,
                          effortLimit=np.ones(7)*10,names=['root']+[str(i) for i in range(7)])
        states=np.zeros((3,14));states[1,9]=2.+1e-7;states[2,0]=1.+1e-7
        result=w.limit_checks(np,m,states,np.zeros((2,7)),[0.,.003,.006],[0.,.003])
        self.assertTrue(result['physical_pass'])
        self.assertEqual(result['first_raw_violation']['kind'],'velocity')
        self.assertEqual(result['first_raw_violation']['joint_index'],2)
        self.assertEqual(result['first_raw_violation']['time_s'],.003)

if __name__=='__main__':unittest.main(verbosity=2)
