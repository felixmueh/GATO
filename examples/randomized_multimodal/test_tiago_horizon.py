"""Analytic CPU checks for simulation timing and cost accounting (no CUDA)."""
import unittest
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from tiago_horizon import ROOT, fine_rollout
from tiago_horizon_mpc import integrated_objective, interpolate_coarse, shifted_controls


class ClockAndQuadratureTests(unittest.TestCase):
    def test_constant_state_and_torque_integral(self):
        states=np.zeros((2,14));controls=np.full((1,7),2.)
        xyz=np.array([[.1,0.,0.],[.1,0.,0.]])
        scene=dict(goal=np.zeros(3),center=np.array([10.,10.]),radius=.02)
        cost=integrated_objective(states,controls,np.array([0.,.06]),xyz,scene,SimpleNamespace(margin=.008))
        self.assertAlmostEqual(cost['position'],.02)
        self.assertAlmostEqual(cost['effort'],.0028)
        self.assertAlmostEqual(cost['terminal'],1.)

    def test_fractional_shift_preserves_zoh_interval_integral(self):
        shifted=shifted_controls(np.tile(np.arange(3.)[:,None],(1,7)),.04,.06,np.zeros(7))
        np.testing.assert_allclose(shifted[:,0],[1.5,1.,0.],atol=1e-14)
        self.assertAlmostEqual(.04*shifted[:,0].sum(),.1)

    def test_interpolation_of_constant_acceleration(self):
        times=np.arange(3)*.04
        states=np.c_[np.tile((.5*times**2)[:,None],(1,7)),np.tile(times[:,None],(1,7))]
        query=np.array([.03,.05,.06])
        got=interpolate_coarse(states,query,.04)
        np.testing.assert_allclose(got[:,:7],np.tile((.5*query**2)[:,None],(1,7)),atol=1e-14)
        np.testing.assert_allclose(got[:,7:],np.tile(query[:,None],(1,7)),atol=1e-14)

    def test_gravity_hold_at_nondividing_controller_clock(self):
        model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
        q=np.array([-.39,-1.73,-.38,-2.35,0.,-1.21,.04]);x0=np.r_[q,np.zeros(7)]
        control=pin.rnea(model,model.createData(),q,np.zeros(7),np.zeros(7)).copy()
        _,dense,times,failure=fine_rollout(model,x0,np.tile(control,(31,1)),.45/31,.001,.06)
        self.assertIsNone(failure)
        self.assertAlmostEqual(times[-1],.06,places=13)
        np.testing.assert_allclose(dense,np.tile(x0,(len(dense),1)),atol=1e-10)


if __name__=='__main__':
    unittest.main()
