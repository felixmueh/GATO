"""CPU regression checks for TIAGo selection, model frames and seven-joint execution."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pinocchio as pin

from plants import plant_inputs, plant_metadata, end_effector_position
from protocol import solver_parameters
from experiment import Recorder, run_instrumented
from plotting import normalize, aggregate, report
import run
from test_experiment import Clock
from gato_tiago.config import TIAGO_TRACKING_SOLVER_PARAMS, TIAGO_RIGHT_START_CONFIGS


class TiagoTests(unittest.TestCase):
    def test_inputs_use_native_seven_joint_model_and_horizontal_tool_reference(self):
        urdf, start, reference = plant_inputs('tiago_right')
        model = pin.buildModelFromUrdf(str(urdf))
        self.assertEqual((model.nq, model.nv, start.shape), (7, 7, (14,)))
        np.testing.assert_array_equal(start[:7], TIAGO_RIGHT_START_CONFIGS['comfortable_high_clearance'])
        np.testing.assert_array_equal(start[7:], np.zeros(7))
        data = model.createData()
        tool = end_effector_position('tiago_right', model, data, start[:7])
        expected = (data.oMf[model.getFrameId('torso_lift_link')].inverse() *
                    data.oMf[model.getFrameId('arm_right_tool_link')]).translation
        np.testing.assert_allclose(tool, expected)
        self.assertGreater(np.linalg.norm(tool - data.oMi[6].translation), .01)
        points = reference.reshape(-1, 6)
        self.assertEqual(points.shape, (2400, 6))
        np.testing.assert_allclose(points[0, :3], tool + [.02, 0., 0.])
        np.testing.assert_allclose(points[1200, :3], tool + [.34, 0., 0.], atol=1e-12)
        np.testing.assert_allclose(points[:, 2], tool[2])
        np.testing.assert_array_equal(points[:, 3:], 0.)
        self.assertAlmostEqual(np.ptp(points[:, 1]), .16)

    def test_tiago_tuning_is_anchored_to_live_n64_dt008(self):
        params = solver_parameters(64, .504, 'tiago_right')
        for key, value in TIAGO_TRACKING_SOLVER_PARAMS.items():
            self.assertAlmostEqual(params[key], value)
        for n in (8, 16, 32, 128):
            other = solver_parameters(n, .504, 'tiago_right')
            for key in ('q_cost', 'qd_cost', 'u_cost', 'q_lim_cost', 'vel_lim_cost', 'ctrl_lim_cost'):
                self.assertAlmostEqual(other[key] * (n - 1), params[key] * 63)
            self.assertEqual(other['N_cost'], params['N_cost'])
        self.assertEqual(solver_parameters(8, plant='tiago_right'), TIAGO_TRACKING_SOLVER_PARAMS)

    def test_cli_and_worker_default_to_tiago_and_reject_indy_resources(self):
        args = run.parser().parse_args(['--gpu-name', 'GPU'])
        self.assertEqual(args.plant, 'tiago_right')
        self.assertEqual(run.configuration(args)['nq'], 7)
        command = run.cell_command(args, 8, 1, 0)
        self.assertEqual(command[command.index('--plant') + 1], 'tiago_right')
        self.assertTrue(command[command.index('--build-dir') + 1].endswith('solve-time-heatmap-tiago_right'))
        for resource in ({}, {'plant': 'indy7'}):
            with self.assertRaisesRegex(ValueError, 'does not match'):
                run.validate_resource_plant(resource, 'tiago_right')
        run.validate_resource_plant({'plant': 'tiago_right'}, 'tiago_right')
        with patch.object(run, 'match_torch_libraries') as setup, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run.main(['--gpu-name', 'GPU', '--protocol', 'reference'])
            setup.assert_not_called()

    def test_seven_joint_loop_preserves_shapes_and_uses_current_tool_error(self):
        urdf, start, reference = plant_inputs('tiago_right')
        model = pin.buildModelFromUrdf(str(urdf))
        clock = Clock()
        class GravitySolver:
            N, batch_size = 8, 1
            def set_f_ext_B(self, force):
                np.testing.assert_array_equal(force, np.zeros((1, 6)))
            def reset_dual(self): pass
            def reset_rho(self): pass
            def solve(self, state, goal, warmstart):
                assert state.shape == (1, 14)
                assert goal.shape == (1, 48)
                assert warmstart.shape == (1, 161)
                result = warmstart.copy()
                for knot in range(7):
                    result[:, knot * 21 + 14:knot * 21 + 21] = pin.computeGeneralizedGravity(
                        model, model.createData(), state[0, :7])
                clock.now += .001
                return result, 1000.
            def get_stats(self):
                return dict(sqp_iters=[1], kkt_converged=[True])
        for horizon in (None, .504):
            with self.subTest(horizon=horizon), tempfile.TemporaryDirectory() as folder:
                with patch('experiment.time.monotonic', clock), contextlib.redirect_stdout(io.StringIO()):
                    rec = Recorder(Path(folder), batch_size=1, max_sqp_iters=1,
                                   max_solve_ms=0, wall_time=.0025, protocol='wall-time', meta={})
                    try:
                        end, status = run_instrumented(GravitySolver(), model, start, reference, rec,
                            wall_time=.0025, horizon_time=horizon, plant='tiago_right')
                        self.assertEqual(status, 'completed')
                        self.assertGreater(len(rec.rows), 0)
                        for row, state, ee in zip(rec.rows, rec.states, rec.ee):
                            self.assertEqual(state.shape, (14,))
                            goal = reference.reshape(-1, 6)[row['reference_offset'], :3]
                            self.assertAlmostEqual(row['tracking_error_m'], np.linalg.norm(ee - goal))
                    finally:
                        rec.stream.close()

    def test_report_preserves_historical_plant_and_labels_new_tiago(self):
        args = run.parser().parse_args(['--gpu-name', 'GPU', '--knots', '8', '--batches', '1'])
        config = run.configuration(args)
        for plant in ('tiago_right', None):
            selected = dict(config)
            if plant is None:
                for key in plant_metadata('tiago_right'):
                    selected.pop(key)
            normalized, results = normalize({'config': selected, 'results': []})
            rendered = report(normalized, results, aggregate(normalized, results))
            self.assertIn('GATO / TIAGo right arm' if plant else 'GATO / Indy7', rendered)
            if plant:
                self.assertIn('arm_right_tool_link in torso_lift_link', rendered)
                self.assertIn('current-time reference', rendered)
                self.assertNotIn('joint-6 origin', rendered)


if __name__ == '__main__':
    unittest.main()
