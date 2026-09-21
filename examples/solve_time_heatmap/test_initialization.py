"""Initialization convergence and measurement isolation checks with a CPU solver."""
import json
import unittest

import numpy as np

from experiment import NumericInstability
import test_experiment as fixtures


class Solver(fixtures.Solver):
    def __init__(self, clock, *, flags, durations, nonfinite_call=None):
        super().__init__(clock, durations, nonfinite_call)
        self.flags = flags

    def solve(self, state, goal, warmstart):
        result, duration = super().solve(state, goal, warmstart)
        # Distinct member plans expose accidental batch-0 broadcasting between
        # initialization solves. The measured policy may still broadcast.
        result[1, 12:18] += .25
        self.outputs[-1] = result.copy()
        return result, duration

    def get_stats(self):
        flags = self.flags[min(len(self.inputs) - 1, len(self.flags) - 1)]
        return dict(sqp_iters=np.where(flags, 1, 3), kkt_converged=flags,
                    initial_merit=[3.], final_merit=[1.])


class InitializationTest(unittest.TestCase):
    setUp = fixtures.ExperimentTest.setUp
    tearDown = fixtures.ExperimentTest.tearDown
    recorder = fixtures.ExperimentTest.recorder
    execute = fixtures.ExperimentTest.execute

    def test_waits_for_every_member_without_advancing_initial_problem(self):
        solver = Solver(self.clock, flags=[[True, False], [True, True]],
                        durations=[2_000_000., 3_000_000., 1_000.])
        recorder = self.recorder(wall_time=.0025)
        end, status = self.execute(solver, recorder, initialization='native-stop')
        result = recorder.save(status, end)
        self.assertEqual(status, 'completed')
        self.assertEqual(len(recorder.initialization_rows), 2)
        np.testing.assert_array_equal(solver.inputs[0][0], solver.inputs[1][0])
        np.testing.assert_array_equal(solver.inputs[0][1], solver.inputs[1][1])
        np.testing.assert_array_equal(solver.inputs[1][2], solver.outputs[0])
        self.assertEqual(solver.reset_dual_count, 1)
        self.assertEqual(solver.reset_rho_count, len(solver.inputs))
        initialization = result['initialization']
        self.assertEqual(initialization['status'], 'ready')
        self.assertEqual(initialization['call_count'], 2)
        self.assertEqual(initialization['total_native_ms'], 5000.)
        self.assertAlmostEqual(initialization['wall_time_s'], 5.)
        self.assertTrue(initialization['all_native_stop'])
        self.assertEqual(result['samples'], 3)
        self.assertEqual(result['avg_native_ms'], 1.)
        self.assertEqual(result['slow_solve_count'], 0)
        self.assertAlmostEqual(result['measured_wall_elapsed_s'], .003)
        self.assertAlmostEqual(end, .03)
        self.assertEqual(result['warmup']['native_ms'], 3000.)
        history = json.loads((self.output / 'initialization.json').read_text())
        self.assertEqual(history['calls'], recorder.initialization_rows)
        self.assertEqual(history['status'], 'ready')
        self.assertEqual(json.loads((self.output / 'warmup.json').read_text()), history['calls'][-1])
        with np.load(self.output / 'samples.npz') as samples:
            np.testing.assert_array_equal(samples['solve_times'], [1., 1., 1.])
        self.assertEqual(len((self.output / 'solves.jsonl').read_text().splitlines()), 3)

    def test_budget_exhaustion_has_no_measured_samples_or_plant_advance(self):
        solver = Solver(self.clock, flags=[[True, False]], durations=[2_000_000.])
        recorder = self.recorder()
        end, status = self.execute(solver, recorder, initialization='native-stop',
                                   max_init_solves=2)
        result = recorder.save(status, end)
        self.assertEqual((end, status), (0., 'initialization_incomplete'))
        self.assertEqual(len(solver.inputs), 2)
        self.assertEqual(recorder.rows, [])
        self.assertIsNone(recorder.measurement_started)
        self.assertIsNone(recorder.measurement_finished)
        self.assertEqual(result['initialization']['status'], 'incomplete')
        self.assertFalse(result['initialization']['all_native_stop'])
        self.assertEqual(result['samples'], 0)
        self.assertIsNone(result['avg_native_ms'])
        self.assertEqual(result['measured_wall_elapsed_s'], 0.)
        self.assertEqual((self.output / 'solves.jsonl').read_text(), '')
        for state, goal, _ in solver.inputs:
            np.testing.assert_array_equal(state, np.tile(self.start, (2, 1)))
            np.testing.assert_array_equal(goal, solver.inputs[0][1])

    def test_single_mode_preserves_one_call_even_without_native_stop(self):
        solver = Solver(self.clock, flags=[[False, False]], durations=[2_000_000., 1000.])
        recorder = self.recorder(wall_time=.0005)
        end, status = self.execute(solver, recorder, initialization='single')
        result = recorder.save(status, end)
        self.assertEqual(status, 'completed')
        self.assertEqual(result['initialization']['status'], 'single_solve')
        self.assertEqual(result['initialization']['call_count'], 1)
        self.assertEqual(result['samples'], 1)

    def test_nonfinite_initialization_records_failure_and_freezes_phase_clock(self):
        solver = Solver(self.clock, flags=[[True, False]], durations=[2_000_000.],
                        nonfinite_call=1)
        recorder = self.recorder()
        with self.assertRaisesRegex(NumericInstability, 'warm-up'):
            self.execute(solver, recorder, initialization='native-stop')
        self.clock.now += 50.
        result = recorder.save('unstable_nonfinite')
        self.assertEqual(result['samples'], 0)
        self.assertIsNone(recorder.measurement_started)
        self.assertEqual(result['initialization']['status'], 'incomplete')
        self.assertEqual(result['initialization']['call_count'], 2)
        self.assertAlmostEqual(result['initialization']['wall_time_s'], 4.)
        self.assertEqual(recorder.warmup['finite_plan_batch'], [True, False])


if __name__ == '__main__':
    unittest.main()
