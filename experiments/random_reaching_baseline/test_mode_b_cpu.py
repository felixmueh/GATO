"""CPU clocks for the isolated frozen-reaching B path; no native imports."""
import ast
import contextlib
import io
import json
from pathlib import Path
import hashlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pinocchio as pin
import run_frozen_baseline as worker
import mode_b_episode as modeb


class Clock:
    now=0.
    def __call__(self):return self.now


class ModeBClock(unittest.TestCase):
    def execute(self, folder, failure=None):
        clock=Clock();shifts=[]
        class Solver:
            calls=0
            def set_f_ext_B(self,*unused):pass
            def reset_dual(self):pass
            def reset_rho(self):pass
            def solve(self,state,reference,warm):
                call=self.calls;self.calls+=1;clock.now+=.05
                if failure=='cuda' and call==1:raise RuntimeError('CUDA fatal access error')
                output=warm.copy()
                for k in range(31):output[0,k*21+14:k*21+21]=call*100+k+1
                if failure=='nonfinite' and call==1:output[0,-1]=np.nan
                return output,50_000.
            def get_stats(self):
                clock.now+=2.  # Deliberate recording/stat overhead, never charged.
                return dict(pcg_iters=np.array([0]),sqp_iters=np.array([1]),step_size=np.array([]),
                    min_merit=np.array([]),kkt_converged=np.array([0]),initial_merit=np.array([2.]),final_merit=np.array([1.]),ls_num_iters=0)
        class MPC:
            def __init__(self,model,*args,**kwargs):
                self.data=model.createData();self.actual_f_ext=np.zeros(6);self.solver=Solver()
        def shift(warm,state,nx,nu,n,duration,dt):
            shifts.append(duration);out=warm.copy();out[:14]=state;return out
        def rk4(model,data,q,v,u,h,force):return q+u*h,np.zeros(7)
        common=SimpleNamespace(initialize_warm_start=lambda x,*unused:np.tile(np.r_[x,np.zeros(7)],32)[:-7],
                               shift_packed_trajectory_warm_start=shift,rk4=rk4)
        root=Path(__file__).resolve().parents[2]
        args=SimpleNamespace(source=root,output=Path(folder),save_plans=False)
        cfg=dict(max_sqp_iters=1)
        protocol=dict(id='cpu-clock',solver_params=cfg,timeout_s=.135,position_tolerance_m=.05,
                      velocity_l1_tolerance_rad_s=1.,arrival_policy='position_and_speed_only')
        x=np.zeros(14);goal=SimpleNamespace(translation=np.zeros(3),rotation=np.eye(3))
        def metric(np_,pin_,model,data,state,goal):return dict(position_error_m=1.,tool_axis_error_deg=0.,orientation_error_deg=0.,velocity_l1_rad_s=0.)
        with patch.object(worker,'task',return_value=(x,np.zeros(7),goal,np.zeros((1,192)),{})), \
             patch.object(worker,'metrics',side_effect=metric),patch.object(modeb.time,'perf_counter',clock), \
             contextlib.redirect_stdout(io.StringIO()):
            result=modeb.run_episode_mode_b(worker,(np,pin,MPC,common),args,protocol,3546)
        out=Path(folder)/'pair_03546'
        rows=[json.loads(line) for line in (out/'solves.jsonl').read_text().splitlines()]
        return result,rows,np.load(out/'executed.npz'),shifts,out

    def test_original_default_a_episode_unchanged(self):
        # Captured from original e1bc38b runtime; no unpublished Git object needed.
        expected_ast_sha256='5a61819118c658aef13d258ecd48d9a91390130987c0455718d20cf95228cc96'
        tree=ast.parse(Path(worker.__file__).read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_episode')
        self.assertEqual(hashlib.sha256(ast.dump(function,include_attributes=False).encode()).hexdigest(),expected_ast_sha256)

    def test_old_plan_age_arrival_first_knot_warm_origin_clip_and_overhead(self):
        with tempfile.TemporaryDirectory() as folder:
            result,rows,trace,shifts,out=self.execute(folder)
            self.assertIsNone(result['execution_error'])
            self.assertAlmostEqual(result['simulated_seconds'],.135)
            self.assertEqual(len(rows),4)
            np.testing.assert_allclose(shifts,[.03,.05,.05],atol=1e-12)
            self.assertEqual(rows[0]['charged_solve_s'],0.)
            self.assertAlmostEqual(rows[1]['control_activation_time_s'],.08)
            self.assertEqual(rows[1]['executed_prefix_plan_step'],0)
            self.assertEqual(rows[2]['executed_prefix_plan_step'],1)
            self.assertEqual(rows[3]['returned_plan_status'],'discarded_task_terminal')
            self.assertAlmostEqual(rows[3]['actual_charged_elapsed_s'],.005)
            for row in rows:self.assertAlmostEqual(row['solve_wall_s'],.05)
            for row in rows:self.assertTrue(row['work']['cap_hit'] and row['work']['zero_pcg_outer_exit'])
            for t,u in zip(trace['control_times'],trace['controls']):
                expected=1 if t<.03-1e-10 else 2 if t<.06-1e-10 else 3 if t<.08-1e-10 else 101 if t<.11-1e-10 else 102 if t<.13-1e-10 else 201
                self.assertAlmostEqual(u[0],expected)
            self.assertFalse(list(out.glob('step_*_returned.npz')))
            trace.close()

    def test_numerical_failure_retained_infrastructure_failure_distinct(self):
        for failure,expected in [('nonfinite',False),('cuda',True)]:
            with tempfile.TemporaryDirectory() as folder:
                result,rows,trace,shifts,out=self.execute(folder,failure)
                self.assertIsNotNone(result['execution_error'])
                self.assertEqual(result['shared_setup_failure'],expected)
                self.assertFalse(result['physical_pass'])
                self.assertTrue((out/'failed_solve.npz').exists())
                self.assertGreaterEqual(len(rows),1)
                trace.close()


if __name__=='__main__':unittest.main()
