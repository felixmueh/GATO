"""CPU integration checks for timestamp alignment in the public live MPC loops.

The native solver and ROS feedback are recording boundaries. The controller loop,
reference sampling and packed warm-start shifting are the production Python code.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from gato_tiago.config import TIAGO_REACHING_SOLVER_PARAMS
from gato_tiago.final_approach import FinalApproachSwitch
from gato_tiago.tiago_mpc_controller import ForceEstimator, MPC_GATO


class RecordingSolver:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.inputs = []
        self.outputs = []
        self.references = []
        self.prediction_intervals = []

    def reset_dual(self): pass
    def reset_rho(self): pass
    def ee_pos(self, q): return np.zeros(3)
    def ee_tool_axis_error(self, q, rpy): return 0.0

    def solve(self, state, reference, warm_start):
        self.inputs.append(warm_start.copy())
        self.references.append(reference.copy())
        self.current_state = state[0].copy()
        if len(self.inputs) == 1:
            # A nonconstant plan makes an artificial fractional shift observable.
            plan = np.asarray([0, 0, 1, 10, 20, 2, 20, 40, 3, 30, 60],
                              dtype=np.float32)
            output = np.tile(plan, (self.batch_size, 1))
        else:
            output = warm_start.copy()
        self.outputs.append(output.copy())
        return output, 100.0

    def sim_forward(self, state, control, interval):
        assert interval > 0, "native prediction must not receive zero elapsed time"
        self.prediction_intervals.append(interval)
        predictions = np.tile(self.current_state, (self.batch_size, 1))
        predictions[:, 0] += np.arange(self.batch_size) - 2
        return predictions


class RecordingEstimator:
    def __init__(self):
        self.updates = []

    def update(self, best_id, errors, **kwargs):
        self.updates.append((best_id, errors.copy(), kwargs))


class RecordingFeedback:
    target_hz = 100

    def __init__(self):
        # Two initial duplicates, a new sub-sim_dt sample, its duplicate,
        # another fresh sample and a final sample that terminates both tasks.
        self.stamps = iter([10.0, 10.0, 10.0, 10.001, 10.001, 10.011, 10.3])
        self.commands = []
        self.finished = False

    def initialize(self, **kwargs): pass
    def finish_control(self, **kwargs): self.finished = True

    def send_trajectory(self, controls, dt):
        self.commands.append((controls.copy(), dt))

    def read_state(self, **kwargs):
        return SimpleNamespace(q=np.zeros(1), qd=np.zeros(1),
                               stamp_sec=next(self.stamps), age_sec=0.0,
                               command_rate_hz=100.0, max_period_sec=.01)


def run_live_boundary(task, batch_size):
    mpc = MPC_GATO.__new__(MPC_GATO)
    mpc.N, mpc.dt = 4, .03
    mpc.nq = mpc.nv = mpc.nq_robot = mpc.nv_robot = mpc.nu = 1
    mpc.nx = 2
    mpc.batch_size = batch_size
    mpc.track_full_stats = mpc.has_pendulum = False
    mpc.constant_f_ext_world = np.zeros(6)
    mpc.final_approach = FinalApproachSwitch(TIAGO_REACHING_SOLVER_PARAMS)
    mpc.solver = RecordingSolver(batch_size)
    mpc.force_estimator = RecordingEstimator() if batch_size > 1 else None
    mpc.update_force_batch = lambda q: None
    feedback = RecordingFeedback()
    if task == 'tracking':
        # Twelve reference knots, with a varying position component.
        reference = np.zeros((12, 6), dtype=np.float32)
        reference[:, 0] = np.arange(12)
        _, stats = mpc.run_mpc_fig8(np.zeros(2), reference.ravel(),
                                  sim_time=.03, sim_dt=.003, controller=feedback)
    else:
        _, stats = mpc.run_mpc_goals(np.zeros(2), [[1, 0, 0]],
                                   goal_timeout=.03, sim_dt=.003,
                                   controller=feedback)
        assert feedback.finished
    return mpc, feedback, stats


@pytest.mark.parametrize('task', ['tracking', 'reaching'])
@pytest.mark.parametrize('batch_size', [1, 4])
def test_live_warm_start_uses_observation_elapsed_time(task, batch_size):
    mpc, feedback, stats = run_live_boundary(task, batch_size)
    solver = mpc.solver
    assert len(solver.inputs) == 6  # Warm-up plus five observations.
    np.testing.assert_allclose(stats['timestamps'], [0, 0, .001, .001, .011],
                               atol=1e-12)
    # Repeated observations must preserve every future state/control knot.
    for solve_index in [1, 2, 4]:
        np.testing.assert_array_equal(solver.inputs[solve_index],
                                      solver.outputs[solve_index - 1])
    # A fresh 1 ms sample shifts by 1 ms even though sim_dt is 3 ms.
    np.testing.assert_allclose(solver.inputs[3][:, 3:5],
                               np.tile([10 + 1 / 3, 20 + 2 / 3], (batch_size, 1)),
                               rtol=1e-6)
    assert len(feedback.commands) == len(solver.inputs)
    if task == 'tracking':
        for solve_index in [1, 2, 4]:
            np.testing.assert_array_equal(solver.references[solve_index],
                                          solver.references[solve_index - 1])
        assert solver.references[3][0, 0] == pytest.approx(1 / 30)
    if batch_size > 1:
        assert solver.prediction_intervals == pytest.approx([.001, .01])
        assert len(mpc.force_estimator.updates) == 2
        if task == 'reaching':
            np.testing.assert_array_equal(stats['best_trajectory_id'], [0, 0, 2, 2, 2])


def test_zero_elapsed_preserves_selected_force_lane_and_estimator():
    mpc = MPC_GATO.__new__(MPC_GATO)
    mpc.batch_size = 4
    mpc.solver = RecordingSolver(4)
    mpc.solver.current_state = np.zeros(2)
    mpc.force_estimator = RecordingEstimator()
    assert mpc.evaluate_best_trajectory(np.zeros(2), np.zeros(1), np.zeros(2), .001) == 2
    previous_errors = mpc.force_estimator.updates[0][1].copy()
    assert mpc.evaluate_best_trajectory(np.zeros(2), np.zeros(1), np.zeros(2), 0) == 2
    assert len(mpc.force_estimator.updates) == 1
    np.testing.assert_array_equal(mpc.force_estimator.updates[0][1], previous_errors)
    assert mpc.solver.prediction_intervals == [.001]


def test_zero_elapsed_leaves_real_force_hypotheses_unchanged():
    assert ForceEstimator is not None
    mpc = MPC_GATO.__new__(MPC_GATO)
    mpc.batch_size = 4
    mpc.solver = RecordingSolver(4)
    mpc.force_estimator = ForceEstimator(4)
    mpc._last_best_trajectory_id = 3
    mpc.force_estimator.smoothed_estimate[:] = [1, 2, 3, .1, .2, .3]
    mpc.force_estimator.momentum[:] = [.4, .5, .6, 0, 0, 0]
    before_hypotheses = mpc.force_estimator.generate_batch().copy()
    before_rotation = mpc.force_estimator.current_rotation.copy()
    before_radius = mpc.force_estimator.radius
    before_confidence = mpc.force_estimator.confidence
    assert mpc.evaluate_best_trajectory(np.zeros(2), np.zeros(1), np.zeros(2), 0) == 3
    np.testing.assert_array_equal(mpc.force_estimator.generate_batch(), before_hypotheses)
    np.testing.assert_array_equal(mpc.force_estimator.current_rotation, before_rotation)
    assert mpc.force_estimator.radius == before_radius
    assert mpc.force_estimator.confidence == before_confidence
    assert mpc.force_estimator.error_history == []
    assert mpc.solver.prediction_intervals == []
