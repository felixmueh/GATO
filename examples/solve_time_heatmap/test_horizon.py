"""Physical-time checks for the fixed-duration prediction grid."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiment import Recorder, run_instrumented
from protocol import advance_plan, periodic_reference, solver_parameters
from test_experiment import Clock, Model, Solver, fake_fk, fake_rk4
import run


class HorizonTest(unittest.TestCase):
    def test_costs_match_anchor_and_keep_running_integrals_constant(self):
        self.assertEqual(solver_parameters(32, .31), solver_parameters(32))
        for n in (8, 16, 32, 64, 128, 256):
            params = solver_parameters(n, .31)
            for name in ('q_cost', 'qd_cost', 'u_cost'):
                self.assertAlmostEqual(params[name] * (n-1), solver_parameters(32)[name] * 31)
            self.assertEqual(params['N_cost'], 20.)

    def test_reference_has_same_physical_endpoints_at_every_resolution(self):
        reference = np.tile(np.arange(600)[:, None] * .01, (1, 6))
        for n in (8, 16, 32, 64, 128, 256):
            times = .12 + np.arange(n) * .31/(n-1)
            values = periodic_reference(reference, times)
            np.testing.assert_allclose(values[:, 0], times, atol=1e-14)
            np.testing.assert_allclose(values[[0,-1], 0], [.12, .43])
        seam = periodic_reference(reference, [5.995, 6., 6.005])
        np.testing.assert_allclose(seam[:, 0], [2.995, 0., .005], atol=1e-12)

    def test_control_execution_crosses_prediction_knots(self):
        # Four controls occur in one update: 1 for 3ms, 2 for 3ms,
        # 3 for 3ms, then 4 for 1ms. Exact double-integrator solution.
        plan = np.zeros(8*18-6)
        for k in range(7):
            plan[k*18+12:k*18+18] = k+1
        with patch('protocol.rk4', side_effect=fake_rk4) as integration:
            q, dq = advance_plan(Model(), None, np.zeros(6), np.zeros(6), plan, 8, .003)
        np.testing.assert_allclose(dq, .022, atol=1e-14)
        np.testing.assert_allclose(q, .000083, atol=1e-14)
        self.assertAlmostEqual(sum(call.args[-1] for call in integration.call_args_list), .01)
        self.assertTrue(all(0 < call.args[-1] <= .001 for call in integration.call_args_list))

    def test_noninteger_knot_boundaries_and_terminal_control(self):
        n, duration = 128, .01
        plan = np.zeros(n*18-6)
        for k in range(n-1):
            plan[k*18+12:k*18+18] = 1.
        # dt smaller than integration step, ending exactly at terminal state.
        with patch('protocol.rk4', side_effect=fake_rk4):
            q, dq = advance_plan(Model(), None, np.zeros(6), np.zeros(6), plan,
                                 n, duration/(n-1), duration)
        np.testing.assert_allclose(dq, duration, atol=1e-13)
        np.testing.assert_allclose(q, .5*duration**2, atol=1e-13)

    def test_fixed_duration_preserves_control_clock_and_uses_current_goal(self):
        clock = Clock()
        reference = np.tile(np.arange(600)[:, None] * .01, (1, 6))
        with tempfile.TemporaryDirectory() as folder, \
                patch('experiment.time.monotonic', clock), \
                patch('protocol.rk4', fake_rk4), \
                patch('pinocchio.forwardKinematics', fake_fk), \
                contextlib.redirect_stdout(io.StringIO()):
            solver = Solver(clock, durations=(1000., 1000.))
            recorder = Recorder(Path(folder), batch_size=2, max_sqp_iters=3,
                max_solve_ms=1000., wall_time=.0015, protocol='wall-time', meta={})
            try:
                end, status = run_instrumented(solver, Model(), np.zeros(12), reference,
                    recorder, wall_time=.0015, horizon_time=.31)
                self.assertEqual(status, 'completed')
                self.assertAlmostEqual(end, .02)
                goals = solver.inputs[1][1][0].reshape(8, 6)
                np.testing.assert_allclose(goals[[0,-1], 0], [.01, .32])
                self.assertAlmostEqual(recorder.rows[0]['tracking_error_m'],
                    np.linalg.norm(recorder.ee[0] - [.01]*3))
            finally:
                recorder.stream.close()

    def test_cli_passes_modes_and_rejects_incompatible_duration(self):
        for extra in (['--horizon-time', '0.009'],
                      ['--horizon-time', '.31', '--protocol', 'reference'],
                      ['--horizon-time', 'nan'], ['--max-init-solves', '0']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.main(['--gpu-name', 'test', *extra])
        for sqp, initialization in [('1000', 'native-stop'), ('1', 'single')]:
            with patch.object(run, 'match_torch_libraries'), patch.object(run, 'sweep') as sweep:
                run.main(['--gpu-name', 'test', '--horizon-time', '.31', '--max-sqp-iters', sqp])
            args = sweep.call_args.args[0]
            self.assertEqual(args.initialization, initialization)
            command = run.cell_command(args, 32, 1, 0)
            child = run.parser().parse_args(command[2:])
            self.assertEqual(child.initialization, initialization)
            self.assertEqual(child.horizon_time, .31)
            self.assertEqual(run.configuration(args)['prediction_dt_by_n']['32'], .01)


if __name__ == '__main__':
    unittest.main()
