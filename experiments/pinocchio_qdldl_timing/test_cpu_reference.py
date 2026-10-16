"""Bounded CPU functional contracts; no benchmark cohort or GPU calls."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE), str(ROOT/'examples/solve_time_heatmap')]
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
import pinocchio as pin
from cpu_solver import CPUSolver, barrier
from protocol import solver_parameters
from experiment import Recorder, run_instrumented
from run_matrix import load_bridge, records


class CPUReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
        cls.bridge=load_bridge(Path(os.environ.get('CPU_QDLDL_CACHE',ROOT/'build/pinocchio-qdldl-cpu')))
        cls.frozen=np.load(HERE/'inputs.npz')
        cls.start=cls.frozen['x_start'].copy()
        cls.target=cls.frozen['reference'].reshape(-1,6)[0].copy()

    def solver(self,batch=1):
        params=solver_parameters(16,.03*15,'tiago_right')|{'max_sqp_iters':1}
        return CPUSolver(self.model,16,batch,params,self.bridge)

    def rest_plan(self,s):
        return s.pack(np.tile(self.start,(16,1)),np.zeros((15,7)))

    def test_exact_original_grid_and_frozen_inputs(self):
        original=json.loads((ROOT/'examples/solve_time_heatmap/nb_cells.json').read_text())
        current=json.loads((HERE/'matrix_proposal.json').read_text())
        self.assertEqual(current['cells'],original['cells'])
        self.assertEqual(len(current['cells']),30)
        self.assertEqual((current['updates'],current['repeats']),(300,2))
        self.assertEqual(self.frozen['reference'].shape,(14400,))

    def test_prediction_analytic_derivatives(self):
        s=self.solver();x=self.start.copy();x[7:]=np.arange(7)*.01
        u=np.arange(7)*.1
        prediction,a,b=s.dynamics(x,u,True)
        epsilon=1e-6
        for col in [0,5,7,12]:
            direction=np.eye(14)[col]*epsilon
            fd=(s.dynamics(x+direction,u)-s.dynamics(x-direction,u))/(2*epsilon)
            np.testing.assert_allclose(fd,a[:,col],rtol=2e-4,atol=2e-6)
        direction=np.eye(7)[3]*epsilon
        fd=(s.dynamics(x,u+direction)-s.dynamics(x,u-direction))/(2*epsilon)
        np.testing.assert_allclose(fd,b[:,3],rtol=2e-4,atol=2e-6)
        acceleration=pin.aba(self.model,self.model.createData(),x[:7],x[7:],u).copy()
        np.testing.assert_allclose(prediction[:7],x[:7]+.03*x[7:]+.5*.03**2*acceleration)

    def test_tracking_frame_jacobian(self):
        s=self.solver();_,j=s.position(self.start[:7],True)
        for col in [0,5,6]:
            d=np.eye(7)[col]*1e-6
            fd=(s.position(self.start[:7]+d)-s.position(self.start[:7]-d))/2e-6
            np.testing.assert_allclose(fd,j[:,col],atol=1e-7,rtol=1e-4)

    def test_cost_gradients_and_terminal_terms(self):
        s=self.solver();x=self.start.copy();x[7:]=.02
        u=np.arange(7)*.01
        for terminal in [False,True]:
            _,gx,h,gu,r=s.knot_cost(x,u,self.target,terminal,True)
            for col in [0,5,8]:
                d=np.eye(14)[col]*1e-6
                fd=(s.knot_cost(x+d,u,self.target,terminal)-s.knot_cost(x-d,u,self.target,terminal))/2e-6
                self.assertAlmostEqual(fd,gx[col],delta=2e-6)
            self.assertGreater(float(np.linalg.eigvalsh(h).min()),0.)
            if terminal:
                self.assertIsNone(gu);self.assertIsNone(r)
            else:
                for col in [0,6]:
                    d=np.eye(7)[col]*1e-6
                    fd=(s.knot_cost(x,u+d,self.target)-s.knot_cost(x,u-d,self.target))/2e-6
                    self.assertAlmostEqual(fd,gu[col],delta=2e-6)

    def test_historical_outside_limit_shape_is_not_repaired(self):
        values,g,h=barrier(np.array([-2.,0.,2.]),np.full(3,-1.),np.ones(3))
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(g[0],0.)
        self.assertLess(g[2],0.)
        self.assertTrue((h>0).all())

    def test_native_qdldl_known_block_system(self):
        n=16;s=np.zeros((n,14,42));rhs=np.zeros((n+2,14))
        dense=np.eye(n*14)*-3
        for k in range(n):
            s[k,:,14:28]=-3*np.eye(14)
            rhs[k+1]=np.linspace(-.1,.1,14)
            if k<n-1:
                s[k,:,28:]=.2*np.eye(14);s[k+1,:,:14]=.2*np.eye(14)
                dense[k*14:(k+1)*14,(k+1)*14:(k+2)*14]=.2*np.eye(14)
                dense[(k+1)*14:(k+2)*14,k*14:(k+1)*14]=.2*np.eye(14)
        actual=np.asarray(self.bridge.solve(n,s,rhs.ravel(),1e-3)).reshape(n+2,14)[1:-1].ravel()
        np.testing.assert_allclose(actual,np.linalg.solve(dense,rhs[1:-1].ravel()),rtol=1e-4,atol=1e-6)
        with self.assertRaises(RuntimeError):
            bad=s.copy();bad[1,1,15]=np.nan
            self.bridge.solve(n,bad,rhs.ravel(),1e-3)

    def test_dense_kkt_direction_contract(self):
        s=self.solver();x,u=s.unpack(self.rest_plan(s));ref=np.tile(self.target,(16,1))
        schur,rhs,terms=s.linearize(x,u,self.start,ref,.01)
        lam=self.bridge.solve(16,schur,rhs,1e-3)
        dx,du,_=s.direction(terms,lam)
        qi,ri,gx,gu,a,b,c=terms
        defects=np.r_[dx[0]+c[0],(dx[1:]-np.einsum('kij,kj->ki',a,dx[:-1])-np.einsum('kij,kj->ki',b,du)+c[1:]).ravel()]
        self.assertLess(float(np.linalg.norm(defects)),2e-3)
        multiplier=np.asarray(lam).reshape(18,14)[1:-1]
        sx=np.einsum('kij,kj->ki',np.linalg.inv(qi),dx)+gx-multiplier
        sx[:-1]+=np.einsum('kji,kj->ki',a,multiplier[1:])
        np.testing.assert_allclose(sx,0,atol=1e-6)

    def test_whole_batch_has_real_serial_lane_work(self):
        s=self.solver(2);plan=self.rest_plan(s)
        result,us=s.solve(np.tile(self.start,(2,1)),np.tile(self.target, (2,16)),np.tile(plan,(2,1)))
        self.assertGreater(us,0)
        np.testing.assert_allclose(result[0],result[1],atol=1e-8)
        self.assertEqual(s.get_stats()['direct_solve_work'],[1,1])
        self.assertEqual(s.get_stats()['sqp_iters'],[1,1])
        self.assertTrue(np.isfinite(result).all())

    def test_instrumented_two_update_clock_reset_and_recording(self):
        s=self.solver();calls={'rho':0,'dual':0}
        rho,dual=s.reset_rho,s.reset_dual
        def reset_rho():calls['rho']+=1;rho()
        def reset_dual():calls['dual']+=1;dual()
        s.reset_rho,s.reset_dual=reset_rho,reset_dual
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            record=Recorder(out,batch_size=1,max_sqp_iters=1,max_solve_ms=0,
                wall_time=5.,protocol='wall-time',meta={'linsys_solver':'QDLDL'},updates=2)
            end,status=run_instrumented(s,self.model,self.start,self.frozen['reference'],record,
                plant='tiago_right',updates=2,max_solve_ms=0,horizon_time=.45)
            result=record.save(status,end_time=end);record.stream.close()
            self.assertEqual((status,end),('completed',.02))
            self.assertEqual(calls,{'rho':3,'dual':1})
            self.assertEqual(result['samples'],2)
            self.assertEqual(result['direct_linear_solves'],2)
            self.assertEqual(result['initialization']['call_count'],1)
            np.testing.assert_allclose([r['timestamp_s'] for r in record.rows],[.01,.02])
            self.assertTrue(all(r['native_ms']>0 for r in record.rows))
        self.assertNotIn('torch',sys.modules)

    def test_partial_stream_stays_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'N16_B1_R0';out.mkdir()
            (out/'solves.jsonl').write_text('{"native_ms":1.0}\n{"unfinished":')
            result=records(root)
            self.assertEqual(result[0]['status'],'incomplete')
            self.assertEqual(len(result[0]['rows']),1)


if __name__=='__main__':unittest.main()
