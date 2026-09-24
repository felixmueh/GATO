"""Small numerical checks for the independent experiment model (CPU only)."""
import unittest
import numpy as np
from model import Config, scene, seeds, rollout, objective, certify, pack, unpack


class ModelTests(unittest.TestCase):
    def test_objective_gradient_matches_directional_difference(self):
        cfg=Config(); task=scene(24092400)
        _,u=seeds(task['x0'],task['goal'],cfg.knots,cfg.dt,8,91)
        rng=np.random.default_rng(33)
        for initial in (u[0],u[3]):
            z=initial.ravel(); direction=rng.normal(size=z.size); direction/=np.linalg.norm(direction)
            _,grad=objective(z,task,cfg,True)
            eps=1e-6
            difference=(objective(z+eps*direction,task,cfg)-objective(z-eps*direction,task,cfg))/(2*eps)
            self.assertAlmostEqual(difference,float(grad@direction),places=5)

    def test_endpoint_and_batch_prefix_are_exact(self):
        cfg=Config(); task=scene(24092400)
        x,u=seeds(task['x0'],task['goal'],cfg.knots,cfg.dt,16,91)
        x4,u4=seeds(task['x0'],task['goal'],cfg.knots,cfg.dt,4,91)
        np.testing.assert_array_equal(u[:4],u4)
        np.testing.assert_allclose(x[:,-1,:2],np.broadcast_to(task['goal'],(16,2)),atol=1e-12)
        np.testing.assert_allclose(x[:,-1,2:],0,atol=1e-12)
        np.testing.assert_allclose(unpack(pack(x,u),cfg)[0],x,atol=2e-7)

    def test_continuous_collision_detected_between_safe_knots(self):
        cfg=Config(knots=2,dt=1.,acceleration_limit=10.,velocity_limit=10.)
        task=dict(x0=np.array([-1.,0,2.,0]),goal=np.array([1.,0]),center=np.zeros(2),radius=.2,buffer=.01)
        row=certify(np.zeros((1,2)),task,cfg)
        self.assertAlmostEqual(row['min_clearance'],-.2)
        self.assertFalse(row['feasible'])

    def test_nonfinite_controls_are_retained_as_failure(self):
        cfg=Config(); task=scene(24092400)
        row=certify(np.full((cfg.knots-1,2),np.nan),task,cfg)
        self.assertFalse(row['feasible'])
        self.assertEqual(row['mode'],'invalid')


if __name__=='__main__': unittest.main()
