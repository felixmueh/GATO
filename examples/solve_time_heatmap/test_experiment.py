"""Deterministic CPU checks for clock, cutoff and warm-up measurement semantics.

Run with the experiment's Python environment:
    python -m unittest discover -s examples/solve_time_heatmap -p 'test_*.py'
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiment import NumericInstability, Recorder, run_instrumented
import protocol
import pinocchio as pin


# Frozen reference loop from the previously validated historical reproduction.
# Production executes only experiment.run_instrumented; this is its regression oracle.
def run_reference_protocol(solver, model, x_start, fig8_traj, *,
                           dt=.01, sim_dt=.001, sim_time=10.):
    """Run the original zero-disturbance benchmark with an existing BSQP solver."""
    n, batch = solver.N, solver.batch_size
    nq, nv = model.nq, model.nv
    nx, nu = nq + nv, nv
    data = model.createData()
    model.gravity.linear = np.array([0., 0., -9.81])
    x_curr = x_start.copy()
    q, dq = x_curr[:nq], x_curr[nq:nx]
    x_curr_batch = np.tile(x_curr, (batch, 1))
    ee_g = fig8_traj[:6*n]
    ee_g_batch = np.tile(ee_g, (batch, 1))
    xu = np.zeros(n * (nx + nu) - nu)
    for i in range(n):
        xu[i * (nx + nu):i * (nx + nu) + nx] = x_start
    xu_batch = np.tile(xu, (batch, 1))
    solver.set_f_ext_B(np.zeros((batch, 6), dtype=np.float32))
    solver.reset_dual()
    # The historical interface reset rho within every solve, including warm-up.
    solver.reset_rho()
    xu_batch, _ = solver.solve(x_curr_batch, ee_g_batch, xu_batch)
    xu_best = xu_batch[0, :]
    stats = {key: [] for key in (
        'timestamps', 'solve_times', 'goal_distances', 'joint_positions',
        'joint_velocities', 'sqp_iters', 'planned_controls', 'ee_actual')}
    total_sim_time = accumulated_time = 0.
    while total_sim_time < sim_time:
        previous_solve_time = stats['solve_times'][-1] / 1000. if stats['solve_times'] else dt
        timestep = min(previous_solve_time, dt)
        nsteps = int(timestep / sim_dt)
        for i in range(nsteps):
            offset = int(i / (dt / sim_dt))
            u_idx = nx + (nx + nu) * min(offset, n - 1)
            u = xu_best[u_idx:u_idx + nu]
            q, dq = protocol.rk4(model, data, q, dq, u, sim_dt)
            total_sim_time += sim_dt
        if timestep % sim_dt > 1e-5:
            accumulated_time += timestep % sim_dt
        if accumulated_time >= sim_dt:
            accumulated_time -= sim_dt
            offset = int(nsteps / (dt / sim_dt))
            u_idx = nx + (nx + nu) * min(offset, n - 1)
            u = xu_best[u_idx:u_idx + nu]
            q, dq = protocol.rk4(model, data, q, dq, u, sim_dt)
            total_sim_time += sim_dt
        x_curr = np.concatenate([q, dq])
        eepos_offset = int(total_sim_time / dt)
        # Preserve the historical stopping expression, including its extra 6.
        if eepos_offset >= len(fig8_traj) / 6 - 6 * n:
            break
        x_curr_batch = np.tile(x_curr, (batch, 1))
        ee_g = fig8_traj[6*eepos_offset:6*(eepos_offset+n)]
        ee_g_batch[:, :] = ee_g
        xu_batch[:, :nx] = x_curr
        solver.reset_rho()
        xu_batch_new, solve_time_us = solver.solve(x_curr_batch, ee_g_batch, xu_batch)
        xu_best = xu_batch_new[0, :]
        xu_batch[:, :] = xu_best
        pin.forwardKinematics(model, data, q)
        ee_pos = data.oMi[6].translation.copy()
        stats['timestamps'].append(total_sim_time)
        stats['solve_times'].append(solve_time_us / 1000.)
        stats['goal_distances'].append(np.linalg.norm(ee_pos - ee_g[6:9]))
        stats['joint_positions'].append(q.copy())
        stats['joint_velocities'].append(dq.copy())
        stats['sqp_iters'].append(int(np.asarray(solver.get_stats()['sqp_iters']).reshape(-1)[0]))
        stats['planned_controls'].append(xu_best[nx:nx+nu].copy())
        stats['ee_actual'].append(ee_pos)
    return {key: np.asarray(values) for key, values in stats.items()}


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class Solver:
    N = 8
    batch_size = 2

    def __init__(self, clock, durations=(2_000_000., 1000.), nonfinite_call=None):
        self.clock, self.durations = clock, durations
        self.nonfinite_call = nonfinite_call
        self.inputs, self.outputs = [], []
        self.reset_dual_count = self.reset_rho_count = 0

    def set_f_ext_B(self, values):
        np.testing.assert_array_equal(values, np.zeros((self.batch_size, 6)))

    def reset_dual(self):
        self.reset_dual_count += 1

    def reset_rho(self):
        self.reset_rho_count += 1

    def solve(self, state, goal, warmstart):
        call = len(self.inputs)
        self.inputs.append(tuple(value.copy() for value in (state, goal, warmstart)))
        result = warmstart.copy()
        result[:, 12:18] = .01 * (call + 1)
        if call == self.nonfinite_call:
            result[1, -1] = np.nan
        duration = self.durations[min(call, len(self.durations) - 1)]
        self.clock.now += duration / 1e6
        self.outputs.append(result.copy())
        return result, duration

    def get_stats(self):
        return dict(sqp_iters=np.array([1, 3]), kkt_converged=np.array([True, False]),
                    initial_merit=[3.], final_merit=[1.])


class Model:
    nq = nv = 6

    def __init__(self):
        self.gravity = SimpleNamespace(linear=None)

    def createData(self):
        return SimpleNamespace(oMi=[SimpleNamespace(translation=np.zeros(3)) for _ in range(7)])


def fake_rk4(model, data, q, dq, u, dt):
    return q + dq * dt + .5 * u * dt**2, dq + u * dt


def fake_fk(model, data, q):
    data.oMi[6].translation = q[:3].copy()


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.clock = Clock()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch('experiment.time.monotonic', self.clock))
        self.stack.enter_context(patch('protocol.rk4', fake_rk4))
        self.stack.enter_context(patch('pinocchio.forwardKinematics', fake_fk))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.reference = np.arange(600 * 6, dtype=float) / 10000.
        self.start = np.r_[np.arange(6) / 10., np.zeros(6)]
        self.recorders = []

    def tearDown(self):
        for recorder in self.recorders:
            recorder.stream.close()
        self.stack.close()
        self.temp.cleanup()

    def recorder(self, *, wall_time=.003, cutoff=1000., protocol='wall-time'):
        recorder = Recorder(self.output, batch_size=2, max_sqp_iters=3,
            max_solve_ms=cutoff, wall_time=wall_time, protocol=protocol, meta={})
        self.recorders.append(recorder)
        return recorder

    def execute(self, solver, recorder, **kwargs):
        return run_instrumented(solver, Model(), self.start, self.reference, recorder,
                                wall_time=recorder.wall_time, max_solve_ms=recorder.max_solve_ms,
                                protocol=recorder.protocol, **kwargs)

    def test_warmup_excluded_from_budget_cutoff_arrays_and_aggregates(self):
        solver = Solver(self.clock)
        recorder = self.recorder(wall_time=.0025)
        end, status = self.execute(solver, recorder)
        result = recorder.save(status, end)
        self.assertEqual(status, 'completed')
        self.assertEqual(result['samples'], 3)
        self.assertAlmostEqual(result['avg_native_ms'], 1.)
        self.assertAlmostEqual(result['measured_wall_elapsed_s'], .003)
        self.assertAlmostEqual(result['budget_overrun_s'], .0005)
        self.assertEqual(result['slow_solve_count'], 0)
        self.assertEqual(result['warmup']['native_ms'], 2000.)
        self.assertFalse(result['warmup']['slow_solve'])
        self.assertEqual(result['batch_member_native_stop_count'], 3)
        self.assertEqual(result['batch_member_cap_count'], 3)
        self.assertEqual(result['any_cap_count'], 3)
        samples = np.load(self.output / 'samples.npz')
        np.testing.assert_allclose(samples['solve_times'], [1., 1., 1.])
        self.assertNotIn('plans', samples.files)
        self.assertEqual(len((self.output / 'solves.jsonl').read_text().splitlines()), 3)

    def test_fixed_step_and_soft_wall_budget(self):
        solver = Solver(self.clock, durations=(100., 700_000.))
        recorder = self.recorder(wall_time=1.)
        end, status = self.execute(solver, recorder)
        self.assertEqual(status, 'completed')
        self.assertAlmostEqual(end, .02)
        self.assertAlmostEqual(recorder.measured_elapsed(), 1.4)
        self.assertEqual([row['reference_offset'] for row in recorder.rows], [1, 2])
        np.testing.assert_allclose([row['timestamp_s'] for row in recorder.rows], [.01, .02])
        np.testing.assert_allclose(recorder.states[0][:6], self.start[:6] + .5 * .01 * .01**2)
        self.assertEqual(solver.reset_dual_count, 1)
        self.assertEqual(solver.reset_rho_count, 3)

    def test_cutoff_is_strict_and_slow_sample_retained(self):
        solver = Solver(self.clock, durations=(2_000_000., 1_000_000., 1_000_001.))
        recorder = self.recorder(wall_time=5.)
        end, status = self.execute(solver, recorder)
        self.assertEqual(status, 'slow_solve')
        self.assertEqual(len(recorder.rows), 2)
        self.assertFalse(recorder.rows[0]['slow_solve'])
        self.assertTrue(recorder.rows[1]['slow_solve'])
        self.assertAlmostEqual(end, .02)

    def test_zero_disables_cutoff(self):
        solver = Solver(self.clock, durations=(3_000_000., 1_500_000.))
        recorder = self.recorder(wall_time=2.5, cutoff=0.)
        end, status = self.execute(solver, recorder)
        self.assertEqual(status, 'completed')
        self.assertEqual(len(recorder.rows), 2)
        self.assertFalse(any(row['slow_solve'] for row in recorder.rows))

    def test_nonfinite_secondary_batch_member_retained(self):
        solver = Solver(self.clock, nonfinite_call=1)
        recorder = self.recorder()
        with self.assertRaisesRegex(NumericInstability, 'Nonfinite plan'):
            self.execute(solver, recorder)
        self.assertEqual(len(recorder.rows), 1)
        self.assertEqual(recorder.rows[0]['finite_plan_batch'], [True, False])
        self.assertTrue(recorder.rows[0]['finite_plan'])
        self.assertIsNotNone(recorder.measurement_finished)
        result = recorder.save('unstable_nonfinite')
        self.assertEqual(result['samples'], 1)
        self.assertEqual(json.loads((self.output / 'results.json').read_text())['status'], 'unstable_nonfinite')

    def test_nonfinite_warmup_has_no_measured_samples(self):
        solver = Solver(self.clock, nonfinite_call=0)
        recorder = self.recorder()
        with self.assertRaisesRegex(NumericInstability, 'warm-up'):
            self.execute(solver, recorder)
        self.assertEqual(recorder.rows, [])
        self.assertIsNone(recorder.measurement_started)
        self.assertEqual(recorder.warmup['finite_plan_batch'], [True, False])

    def test_reference_matches_historical_inputs_and_outputs(self):
        durations = (2_000_000., 400., 4_700., 25_000., 50_000., 400., 4_700.)
        old_solver = Solver(self.clock, durations=durations)
        old = run_reference_protocol(old_solver, Model(), self.start, self.reference, sim_time=.065)
        new_solver = Solver(self.clock, durations=durations)
        recorder = self.recorder(protocol='reference', cutoff=0.)
        end, status = self.execute(new_solver, recorder, sim_time=.065)
        self.assertEqual(status, 'completed')
        self.assertEqual(len(old_solver.inputs), len(new_solver.inputs))
        for old_input, new_input in zip(old_solver.inputs, new_solver.inputs):
            for old_array, new_array in zip(old_input, new_input):
                np.testing.assert_array_equal(old_array, new_array)
        for old_output, new_output in zip(old_solver.outputs, new_solver.outputs):
            np.testing.assert_array_equal(old_output, new_output)
        np.testing.assert_array_equal(old['timestamps'], [row['timestamp_s'] for row in recorder.rows])
        np.testing.assert_array_equal(old['solve_times'], [row['native_ms'] for row in recorder.rows])
        np.testing.assert_array_equal(old['goal_distances'], [row['tracking_error_m'] for row in recorder.rows])
        np.testing.assert_array_equal(old['joint_positions'], np.asarray(recorder.states)[:, :6])
        np.testing.assert_array_equal(old['joint_velocities'], np.asarray(recorder.states)[:, 6:])
        self.assertGreaterEqual(end, .065)


if __name__ == '__main__':
    unittest.main()
