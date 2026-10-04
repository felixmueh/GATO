"""Behavior of the optional, latched MPC parameter switch."""
from types import SimpleNamespace

import numpy as np
import pytest

from gato_tiago.config import TIAGO_REACHING_SOLVER_PARAMS
from gato_tiago.final_approach import FinalApproachSwitch


class SolverBoundary:
    """Record parameter updates at the external CUDA solver boundary."""
    batch_size = 1

    def __init__(self):
        self.costs = []
        self.budgets = []
        self.mus = []
        self.pcgs = []
        self.solver = SimpleNamespace(
            set_cost_weights=lambda *values: None,
            set_max_sqp_iters=lambda value: None,
            set_mu_batch=lambda values: self.mus.append(values.copy()),
            set_pcg_tol_batch=lambda values: self.pcgs.append(values.copy()),
        )

    def set_cost_weights(self, **weights):
        self.costs.append(weights)

    def set_max_sqp_iters(self, value):
        self.budgets.append(value)


def test_switch_requires_both_gates_and_stays_latched_without_replacing_solver():
    solver = SolverBoundary()
    switch = FinalApproachSwitch(TIAGO_REACHING_SOLVER_PARAMS, enabled=True)
    for distance, speed in [(0.076, 0.01), (0.05, 0.101)]:
        assert switch.update(solver, time_s=1.0, goal_index=0,
                             distance_m=distance, joint_velocities=[speed]) is None
    assert solver.costs == []
    event = switch.update(solver, time_s=2.0, goal_index=0,
                          distance_m=0.075, joint_velocities=[-0.1, 0.04])
    assert event['time_s'] == 2.0
    assert event['max_joint_speed_rad_s'] == 0.1
    assert solver.costs[-1]['q_cost'] == 400
    assert solver.costs[-1]['N_cost'] == 1200
    assert solver.costs[-1]['qd_cost'] == 1
    assert solver.budgets == [20]
    np.testing.assert_allclose(solver.mus[-1], [30])
    assert switch.update(solver, time_s=3.0, goal_index=0,
                         distance_m=0.2, joint_velocities=[1.0]) is None
    assert switch.active
    assert len(solver.costs) == 1


def test_new_goal_restores_approach_and_can_trigger_again_with_custom_settings():
    solver = SolverBoundary()
    switch = FinalApproachSwitch(
        TIAGO_REACHING_SOLVER_PARAMS, enabled=True,
        params={"qd_cost": 2.5, "mu": 15},
        trigger={"distance_m": 0.02, "max_joint_speed_rad_s": 0.05},
    )
    assert switch.update(solver, time_s=1, goal_index=0, distance_m=0.03,
                         joint_velocities=[0]) is None
    switch.update(solver, time_s=2, goal_index=0, distance_m=0.01, joint_velocities=[0])
    assert solver.costs[-1]['qd_cost'] == 2.5

    assert solver.costs[-1]['vel_lim_cost'] == TIAGO_REACHING_SOLVER_PARAMS['vel_lim_cost']
    switch.reset(solver)
    assert not switch.active
    assert solver.costs[-1]['qd_cost'] == TIAGO_REACHING_SOLVER_PARAMS['qd_cost']
    assert solver.budgets[-1] == TIAGO_REACHING_SOLVER_PARAMS['max_sqp_iters']
    event = switch.update(solver, time_s=3, goal_index=1, distance_m=0.01,
                          joint_velocities=[-0.05])
    assert event['goal_index'] == 1
    assert solver.costs[-1]['qd_cost'] == 2.5


def test_nonfinite_feedback_never_triggers_and_disabled_switch_is_inert():
    solver = SolverBoundary()
    switch = FinalApproachSwitch(TIAGO_REACHING_SOLVER_PARAMS, enabled=True)
    for distance, velocity in [(np.nan, [0]), (0.01, [np.nan]), (np.inf, [0])]:
        assert switch.update(solver, time_s=0, goal_index=0, distance_m=distance,
                             joint_velocities=velocity) is None
    disabled = FinalApproachSwitch(TIAGO_REACHING_SOLVER_PARAMS)
    assert disabled.update(solver, time_s=0, goal_index=0, distance_m=0,
                           joint_velocities=[0]) is None
    disabled.reset(solver)
    assert not switch.active
    assert solver.costs == []


@pytest.mark.parametrize('params,trigger', [
    ({'qd_cost': -1}, {}), ({'u_cost': np.nan}, {}),
    ({'max_sqp_iters': 0}, {}), ({'max_sqp_iters': 2.5}, {}),
    ({'mu': np.inf}, {}), ({'unknown_cost': 1}, {}),
    ({}, {'distance_m': -0.01}), ({}, {'max_joint_speed_rad_s': np.nan}),
    ({}, {'typo': 1}),
])
def test_bad_configuration_is_rejected_before_any_solve(params, trigger):
    with pytest.raises(ValueError):
        FinalApproachSwitch(TIAGO_REACHING_SOLVER_PARAMS, enabled=True,
                            params=params, trigger=trigger)


def test_old_native_extension_is_rejected_before_mutating_parameters():
    solver = SolverBoundary()
    del solver.solver.set_cost_weights
    switch = FinalApproachSwitch(TIAGO_REACHING_SOLVER_PARAMS, enabled=True)
    with pytest.raises(RuntimeError, match='rebuild'):
        switch.update(solver, time_s=0, goal_index=0, distance_m=0, joint_velocities=[0])
    assert solver.budgets == []
    assert solver.costs == []
    assert not switch.active


def make_reaching_boundary(monkeypatch, *, enabled=True):
    import pinocchio as pin
    from pathlib import Path
    from gato_tiago import tiago_mpc_controller as module
    from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS

    model_path = Path(__file__).resolve().parents[2] / 'gato/dynamics/tiago_right/tiago_right_arm.urdf'
    model = pin.buildModelFromUrdf(str(model_path))
    q = TIAGO_RIGHT_START_CONFIGS['comfortable_high_clearance'].copy()
    seen = []
    initial_plans = []

    class NativeBoundary(SolverBoundary):
        def __init__(self, **params):
            super().__init__()
            self.params = params
            self.model = model
            self.data = model.createData()

        def set_cost_weights(self, **weights):
            super().set_cost_weights(**weights)
            self.params.update(weights)

        def reset_dual(self): pass
        def reset_rho(self): pass

        def solve(self, x, ref, plan):
            seen.append(self.params['q_cost'])
            initial_plans.append(plan.copy())
            return plan.copy(), 100

        def ee_pos(self, q):
            pin.framesForwardKinematics(model, self.data, q)
            return self.data.oMf[model.getFrameId('arm_right_tool_link')].translation.copy()

        def ee_tool_axis_error(self, q, rpy): return 0.0

    monkeypatch.setattr(module, 'BSQP', NativeBoundary)
    mpc = module.MPC_GATO(model, str(model_path), N=16, dt=.03,
                          plant_type='tiago_right', solver_params=TIAGO_REACHING_SOLVER_PARAMS,
                          final_approach=enabled)
    return mpc, q, seen, initial_plans, module


@pytest.mark.parametrize('with_seed', [False, True])
def test_reaching_applies_switch_before_first_solve_and_restores_for_next_goal(monkeypatch, with_seed):
    mpc, q, seen, initial_plans, module = make_reaching_boundary(monkeypatch)
    goal = mpc.solver.ee_pos(q)

    class Feedback:
        target_hz = 100
        stamp = 0.0
        finished = False
        def initialize(self, **kwargs): pass
        def send_trajectory(self, *args): pass
        def finish_control(self, **kwargs): self.finished = True
        def read_state(self, **kwargs):
            self.stamp += .03
            return SimpleNamespace(q=q, qd=np.zeros(7), stamp_sec=self.stamp,
                                   age_sec=0, command_rate_hz=100, max_period_sec=.01)

    feedback = Feedback()
    seed = np.arange(16 * 21 - 7, dtype=np.float32)[None] * .00001
    seed[:, :14] = np.r_[q, np.zeros(7)]
    kwargs = {'initial_guess': seed} if with_seed else {}
    with pytest.raises(ValueError, match='6D'):
        mpc.run_mpc_goals(np.r_[q, np.zeros(7)], [goal], controller=feedback)
    assert seen == []
    assert feedback.stamp == 0
    pose_goals = [np.r_[goal, [0, 0, 0]], np.r_[goal+[.2, 0, 0], [0, 0, 0]]]
    _, stats = mpc.run_mpc_goals(np.r_[q, np.zeros(7)], pose_goals,
                                 goal_timeout=.06, controller=feedback, **kwargs)
    if with_seed:
        np.testing.assert_allclose(initial_plans[0], seed)
    assert seen[0] == 400
    assert all(value == 4 for value in seen[1:])
    assert len(stats['final_approach_switches']) == 1
    assert stats['final_approach_switches'][0]['time_s'] == 0
    assert stats['final_approach_switches'][0]['goal_index'] == 0
    assert not mpc.final_approach.active
    assert stats['planned_goal_indices'][0] == 1
    assert stats['goal_indices'][0] == 0
    assert not stats['planned_final_approach_active'][0]
    assert feedback.finished


def test_offline_reaching_executes_exact_30ms_prefixes_on_1ms_grid(monkeypatch):
    mpc, q, _, _, module = make_reaching_boundary(monkeypatch, enabled=False)
    steps = []
    def plant_boundary(model, data, q, velocity, torque, dt, forces):
        steps.append(dt)
        return q.copy(), velocity.copy()
    monkeypatch.setattr(module, 'rk4', plant_boundary)
    goal = mpc.solver.ee_pos(q) + [.2, 0, 0]
    _, stats = mpc.run_mpc_goals(np.r_[q, np.zeros(7)], [goal],
                                 sim_dt=.001, goal_timeout=.12, goal_threshold=0)
    assert stats['terminal_time_s'] == pytest.approx(.12, abs=1e-12)
    assert len(steps) == 120
    np.testing.assert_allclose(stats['timestamps'], [.03, .06, .09], atol=1e-12)
