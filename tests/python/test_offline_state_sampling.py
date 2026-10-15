"""CPU-only mode-B plant sampling and historical timestamp/activation semantics."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from bsqp.offline_mpc import run_mode_b


class Clock:
    now = 0.
    def __call__(self):
        return self.now


class Solver:
    def __init__(self, clock, durations):
        self.clock, self.durations = clock, durations
        self.inputs, self.outputs = [], []

    def reset_dual(self): pass
    def reset_rho(self): pass
    def ee_pos(self, q): return np.array([q[0], 0., 0.])
    def ee_tool_axis_error(self, q, rpy): return 0.

    def solve(self, state, reference, warm):
        index = len(self.inputs)
        self.inputs.append((state.copy(), reference.copy(), warm.copy()))
        plan = np.zeros_like(warm)
        for k in range(4):
            plan[:, 3*k:3*k+2] = [k, 0.]
            if k < 3:
                plan[:, 3*k+2] = 10*index+k+1
        self.outputs.append(plan.copy())
        self.clock.now += self.durations[min(index, len(self.durations)-1)]
        return plan, 100.


def run_sampling(period=None, durations=(.002, .0037), end=.05, sim_dt=.001):
    clock = Clock()
    solver = Solver(clock, durations)
    force_states = []
    owner = SimpleNamespace(N=4, dt=.008, batch_size=1, nx=2, nu=1,
        nq=1, nv=1, nq_robot=1, nv_robot=1, has_pendulum=False,
        track_full_stats=False, solver=solver, final_approach=None,
        update_force_batch=lambda q: force_states.append(q.copy()),
        evaluate_best_trajectory=lambda *args: 0)
    # Exact known continuous trajectory q(t)=t, v(t)=2t, independent of control.
    def integrate(q, dq, u, dt):
        return q+dt, dq+2*dt
    reference = np.zeros((100, 6))
    reference[:, 0] = np.arange(100)*owner.dt
    events = []
    arguments = {} if period is None else dict(state_period_s=period)
    _, stats = run_mode_b(owner, np.zeros(2), integrate=integrate, clock=clock,
        sim_dt=sim_dt, record_raw=True,
        record_callback=lambda kind, payload: events.append((kind, deepcopy(payload))),
        reference=reference.ravel(), sim_time=end, **arguments)
    return owner, stats, events


def test_exact_sample_boundaries_split_steps_and_keep_latest_crossed_sample():
    # A long in-flight solve crosses 20, 40 and 60 ms boundaries.
    _, stats, events = run_sampling(.02, durations=(.002, .057), end=.072)
    samples = [e for kind, e in events if kind == 'state_sampled']
    assert [e['sample_index'] for e in samples] == [0, 1, 2, 3]
    np.testing.assert_allclose([e['sample_timestamp_s'] for e in samples], [0, .02, .04, .06], atol=1e-14)
    for sample in samples:
        np.testing.assert_allclose(sample['state'], [sample['sample_timestamp_s'], 2*sample['sample_timestamp_s']], atol=1e-13)
    solves = stats['raw_solves']
    assert solves[1]['sample_timestamp_s'] == 0.
    assert solves[2]['sample_timestamp_s'] == pytest.approx(.06)
    assert solves[2]['sample_age_s'] == pytest.approx(.005)
    assert solves[2]['state'][0, 0] == pytest.approx(.06)
    assert solves[2]['actual_feedback_state'][0] == pytest.approx(.065)
    for _, step in [e for e in events if e[0] == 'step_completed']:
        start, stop = step['time_s'], step['time_s']+step['step_dt_s']
        assert not any(start+1e-13 < boundary < stop-1e-13 for boundary in [.02, .04, .06])


def test_repeated_samples_hold_state_reference_origin_but_match_historical_warm_floor():
    owner, stats, events = run_sampling(.02)
    solves = stats['raw_solves']
    repeats = [r for r in solves[1:] if r['repeated_sample']]
    assert len(repeats) > 1
    for raw in repeats:
        assert raw['sample_delta_s'] == 0.
        assert raw['effective_warm_start_shift_s'] == .001
        assert raw['warm_start_shift_s'] == .001
        assert raw['optimizer_reference_time_s'] == raw['sample_timestamp_s']
        assert raw['previous_optimizer_reference_time_s'] == raw['sample_timestamp_s']
        assert raw['state'][0, 0] == pytest.approx(raw['sample_timestamp_s'])
        assert raw['reference'][0, 0] == pytest.approx(raw['sample_timestamp_s'], abs=1e-8)
        # The historical 1 ms floor advances the next warm state knot, even
        # though observation/reference time is repeated.
        assert raw['warm_start'][0, 3] == pytest.approx(1.125)
    assert len([e for kind, e in events if kind == 'state_sampled']) < len(solves)
    # Solver arrivals make the integration phase fractional: the 20 ms sample
    # must split a nominal 1 ms step beginning at 19.1 ms.
    steps = [e for kind, e in events if kind == 'step_completed']
    split = next(e for e in steps if abs(e['time_s']-.0191) < 1e-12)
    assert split['step_dt_s'] == pytest.approx(.0009)
    assert split['time_s']+split['step_dt_s'] == pytest.approx(.02)
    assert stats['terminal_time_s'] == pytest.approx(.05)
    np.testing.assert_allclose(stats['terminal_state'], [.05, .1], atol=1e-13)


def test_returned_plan_starts_at_first_knot_on_actual_arrival_not_sample_stamp():
    owner, stats, events = run_sampling(.02)
    steps = [e for kind, e in events if kind == 'step_completed']
    for raw in stats['raw_solves']:
        if raw['activation_status'] != 'activated':
            continue
        activation = raw['activation_time_s']
        step = next(e for e in steps if e['plan_solve_index'] == raw['solve_index'])
        assert step['time_s'] == pytest.approx(activation)
        assert step['active_control_age_s'] == pytest.approx(0., abs=1e-13)
        assert step['control'][0] == owner.solver.outputs[raw['solve_index']][0, 2]
        if raw['solve_index'] > 0:
            assert activation == pytest.approx(raw['time_s']+raw['wall_solve_time_s'])
    assert any(raw['activation_time_s'] > raw['sample_timestamp_s']
               for raw in stats['raw_solves'][1:] if raw['activation_status'] == 'activated')


def test_zero_period_preserves_exact_mode_b_outputs_and_event_schema():
    owner_a, a, events_a = run_sampling()
    owner_b, b, events_b = run_sampling(0.)
    assert [kind for kind, _ in events_a] == [kind for kind, _ in events_b]
    assert not any(kind == 'state_sampled' for kind, _ in events_b)
    assert a.keys() == b.keys()
    for key in ['terminal_state', 'timestamps', 'joint_positions', 'update_intervals_s']:
        np.testing.assert_array_equal(a[key], b[key])
    for first, second in zip(owner_a.solver.inputs, owner_b.solver.inputs):
        for x, y in zip(first, second):
            np.testing.assert_array_equal(x, y)
    assert not any('sample_timestamp_s' in raw for raw in b['raw_solves'])


@pytest.mark.parametrize('period', [0., .02])
def test_public_frontend_forwards_only_positive_sampling_option(period):
    from gato_tiago.tiago_mpc_controller import MPC_GATO
    owner = MPC_GATO.__new__(MPC_GATO)
    owner.dt = .008
    owner.solver = object()
    owner.final_approach = SimpleNamespace(reset=lambda solver: None)
    with patch('bsqp.offline_mpc.run_mode_b', return_value='forwarded') as execute:
        result = owner.run_mpc_fig8(np.zeros(2), np.zeros(100), offline_timing='solve_time',
                                   state_period_s=period)
    assert result == 'forwarded'
    if period:
        assert execute.call_args.kwargs['state_period_s'] == period
    else:
        assert 'state_period_s' not in execute.call_args.kwargs


def test_public_frontend_rejects_sampling_on_live_or_other_clock():
    from gato_tiago.tiago_mpc_controller import MPC_GATO
    owner = MPC_GATO.__new__(MPC_GATO)
    owner.solver = object()
    owner.final_approach = SimpleNamespace(reset=lambda solver: None)
    for options in [dict(offline_timing='controller_dt'),
                    dict(offline_timing='solve_time', controller=object())]:
        with pytest.raises(ValueError, match='offline solve_time tracking'):
            owner.run_mpc_fig8(np.zeros(2), np.zeros(100), state_period_s=.02, **options)


@pytest.mark.parametrize('period', [-.01, float('nan'), float('inf')])
def test_invalid_sample_period_rejected_without_solver_calls(period):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        run_sampling(period)
