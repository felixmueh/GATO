"""CPU execution oracles for reaching timing; no solver construction or GPU calls."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gato_tiago.config import TIAGO_REACHING_SOLVER_PARAMS
from gato_tiago.final_approach import FinalApproachSwitch


@pytest.fixture(params=['generic', 'tiago'])
def boundary(request, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    path = root / ('python/bsqp/mpc_controller.py' if request.param == 'generic'
                   else 'tiago_src/gato_tiago/tiago_mpc_controller.py')
    name = 'bsqp._timing_oracle' if request.param == 'generic' else '_tiago_timing_oracle'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as isolated:
        isolated.setitem(sys.modules, 'bsqp.interface', SimpleNamespace(BSQP=object))
        spec.loader.exec_module(module)
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(module, 'time', SimpleNamespace(perf_counter=lambda: clock.now))
    mpc = module.MPC_GATO.__new__(module.MPC_GATO)
    mpc.N, mpc.nq, mpc.nv, mpc.nx, mpc.nu = 3, 1, 1, 2, 1
    mpc.nq_robot, mpc.nv_robot = 1, 1
    mpc.dt, mpc.batch_size, mpc.track_full_stats = .03, 1, True
    mpc.has_pendulum = False
    mpc.constant_f_ext_world = np.zeros(6)
    mpc.model, mpc.data, mpc.actual_f_ext = None, None, None
    mpc.final_approach = FinalApproachSwitch(TIAGO_REACHING_SOLVER_PARAMS)
    mpc.update_force_batch = lambda q: None
    selected = []
    mpc.evaluate_best_trajectory = lambda *args: selected.append(args[-1]) or 0
    # Each plan has different controls, enabling interval-to-solve alignment checks.
    plans = []
    for j in range(30):
        plans.append(np.array([[0., 0., 1. + j*10, 1., 2., 2. + j*10, 2., 4.]], dtype=np.float32))
    calls, steps = [], []
    delays = [.8, .075, .0075] + [.01]*30

    def solve(x, ref, warm):
        calls.append((x.copy(), ref.copy(), warm.copy()))
        j = len(calls)-1
        clock.now += delays[j]
        # Deliberately mutate input as a native wrapper may do.
        warm[:] = -900
        return plans[j].copy(), 2500. + j*100

    def plant(model, data, q, dq, u, dt, forces):
        assert u.shape == (1,), 'terminal state cannot supply an empty control'
        steps.append((float(u[0]), dt))
        # A simple integral gives independent expected results from torque×time.
        return q + u*dt, dq.copy()

    monkeypatch.setattr(module, 'rk4', plant)
    mpc.solver = SimpleNamespace(reset_dual=lambda: None, reset_rho=lambda: None,
        solve=solve, ee_pos=lambda q: np.array([q[0], 0., 0.]),
        ee_tool_axis_error=lambda q, goal: 0.,
        get_stats=lambda: {'sqp_iters': np.array([2]), 'pcg_iters': np.array([[3]])})
    return SimpleNamespace(mpc=mpc, module=module, delays=delays, calls=calls,
                           plans=plans, steps=steps, selected=selected, clock=clock, kind=request.param)


def run(boundary, **kwargs):
    return boundary.mpc.run_mpc_goals(np.zeros(2), [[100., 0., 0.]],
        sim_dt=.007, goal_timeout=.12, record_raw=True, **kwargs)[1]


def test_legacy_clock_controls_remainder_hold_and_warm_alignment(boundary):
    stats = run(boundary, offline_timing='solve_time_legacy')
    # Startup .8 seconds is retained but nominal .03 is executed first.
    np.testing.assert_allclose(stats['update_intervals_s'], [.03, .075, .0075, .0075])
    np.testing.assert_allclose(stats['requested_update_intervals_s'], [.03, .075, .0075, .01])
    assert stats['interval_timeout_clipped'].tolist() == [False, False, False, True]
    assert stats['interval_plan_solve_indices'].tolist() == [0, 1, 2, 3]
    assert stats['terminal_time_s'] == pytest.approx(.12)
    assert stats['goal_outcomes'] == ['timeout']
    assert stats['startup_wall_solve_time_s'] == pytest.approx(.8)
    np.testing.assert_allclose(stats['wall_solve_times_s'], [.075, .0075, .01])
    np.testing.assert_allclose(stats['native_solve_times_ms'], [2.6, 2.7, 2.8])
    assert stats['startup_native_solve_time_ms'] == pytest.approx(2.5)
    # Second interval uses plan1 controls 11 for 30ms, then 12 for 45ms,
    # including a 15ms hold beyond its 60ms prediction horizon.
    expected = .03*1 + .03*11 + .045*12 + .0075*21 + .0075*31
    assert stats['terminal_state'][0] == pytest.approx(expected)
    raw_steps = stats['raw_execution']
    assert sum(s['step_dt_s'] for s in raw_steps) == pytest.approx(.12)
    assert any(s['step_dt_s'] == pytest.approx(.002) for s in raw_steps)
    assert max(s['step_dt_s'] for s in raw_steps) <= .007
    for a, b in zip(raw_steps, raw_steps[1:]):
        assert b['time_s'] == pytest.approx(a['time_s'] + a['step_dt_s'])
        np.testing.assert_allclose(b['state_before'], a['state_after'])
    solve1 = stats['raw_solves'][1]
    assert solve1['warm_start_shift_s'] == pytest.approx(.03)
    # First shifted control is u1; later controls hold u1. Shifted states are
    # interpolated/clamped, and the measured current state is re-anchored.
    np.testing.assert_allclose(solve1['warm_start'][0], [0.03, 0., 2., 2., 4., 2., 2., 4.])
    assert solve1['state'][0, 0] == pytest.approx(.03)
    np.testing.assert_array_equal(stats['raw_solves'][0]['output'], boundary.plans[0])
    assert (stats['raw_solves'][0]['warm_start'] == 0).all()
    np.testing.assert_allclose(boundary.selected, [.03, .075, .0075])
    assert len(raw_steps) > len(stats['timestamps'])  # All substeps, no downsampling.


def test_mode_a_default_ignores_wall_duration_and_records_terminal_interval(boundary):
    stats = run(boundary)
    np.testing.assert_allclose(stats['update_intervals_s'], [.03]*4)
    assert not stats['interval_timeout_clipped'].any()
    assert stats['terminal_time_s'] == pytest.approx(.12)
    assert len(stats['raw_solves']) == 4
    assert stats['offline_timing'] == 'controller_dt'
    assert sum(dt for _, dt in boundary.steps) == pytest.approx(.12)
    np.testing.assert_allclose(stats['timestamps'], [.03, .06, .09])


def test_mode_b_multiple_timeout_goals_keep_goal_switch_separate(boundary):
    _, stats = boundary.mpc.run_mpc_goals(np.zeros(2), [[100.,0.,0.], [200.,0.,0.]],
        sim_dt=.007, goal_timeout=.04, offline_timing='solve_time', record_raw=True)
    assert stats['goal_outcomes'] == ['timeout', 'timeout']
    assert stats['terminal_time_s'] == pytest.approx(.145)
    assert stats['interval_goal_indices'][0] == 0
    assert stats['interval_goal_indices'][-1] == 1
    switch = [r for r in stats['raw_solves'] if r['category'] == 'goal_switch']
    assert len(switch) == 1
    assert switch[0]['time_s'] == pytest.approx(.105)
    assert switch[0]['goal_index'] == 1
    assert switch[0]['reference'][0, 0] == 200
    assert stats['raw_solves'][0]['category'] == 'startup/cold'


def test_terminal_reach_is_recorded_even_without_feedback_solve(boundary):
    _, stats = boundary.mpc.run_mpc_goals(np.zeros(2), [[.03,0.,0.]],
        sim_dt=.007, goal_timeout=1., offline_timing='solve_time', record_raw=True)
    assert stats['goal_outcomes'] == ['reached']
    assert stats['goal_reached_times'] == [pytest.approx(.03)]
    assert len(stats['raw_solves']) == 1
    assert len(stats['update_intervals_s']) == 1
    assert len(stats['wall_solve_times_s']) == 0
    assert stats['terminal_state'][0] == pytest.approx(.03)


@pytest.mark.parametrize('kwargs', [dict(offline_timing='invalid'), dict(sim_dt=0),
                                    dict(sim_dt=np.nan), dict(goal_timeout=-1)])
def test_invalid_timing_rejected_before_solver(boundary, kwargs):
    with pytest.raises(ValueError):
        boundary.mpc.run_mpc_goals(np.zeros(2), [[1.,0.,0.]], **kwargs)
    assert boundary.calls == []


def test_live_mode_b_is_rejected_without_changing_ros_policy(boundary):
    if boundary.kind != 'tiago':
        return
    with pytest.raises(ValueError, match='without a live controller'):
        boundary.mpc.run_mpc_goals(np.zeros(2), [[1.,0.,0.]],
                                   controller=object(), offline_timing='solve_time')
    assert boundary.calls == []


def test_record_callbacks_are_outside_paid_solve_time(boundary):
    events = []
    def observer(event, record):
        events.append((event, record['status']))
        # Expensive synchronous archival does not enter the declared solve-call
        # timer. Its artificial duration must not stretch execution intervals.
        boundary.clock.now += .2
    stats = run(boundary, offline_timing='solve_time', record_callback=observer)
    np.testing.assert_allclose(stats['wall_solve_times_s'], [.075, .0075, .01])
    assert stats['terminal_time_s'] == pytest.approx(.12)
    assert ('solve_started', 'started') in events
    assert ('solve_completed', 'completed') in events
    assert ('step_completed', 'completed') in events
    assert boundary.mpc.last_run_stats is stats


def test_startup_solve_failure_preserves_input_and_rethrows(boundary):
    def broken_solve(*args):
        raise RuntimeError('injected solver boundary failure')
    boundary.mpc.solver.solve = broken_solve
    with pytest.raises(RuntimeError, match='injected solver'):
        run(boundary, offline_timing='solve_time')
    partial = boundary.mpc.last_run_stats
    assert len(partial['raw_solves']) == 1
    record = partial['raw_solves'][0]
    assert record['status'] == 'failed'
    assert record['error_type'] == 'RuntimeError'
    assert record['category'] == 'startup/cold'
    assert (record['warm_start'] == 0).all()
    assert 'output' not in record
    assert partial['raw_execution'] == []


def test_plant_failure_preserves_attempted_control_state_and_rethrows(boundary, monkeypatch):
    monkeypatch.setattr(boundary.module, 'rk4', lambda *args: (_ for _ in ()).throw(
        RuntimeError('injected plant boundary failure')))
    with pytest.raises(RuntimeError, match='injected plant'):
        run(boundary, offline_timing='solve_time')
    partial = boundary.mpc.last_run_stats
    record = partial['raw_execution'][0]
    assert record['status'] == 'failed'
    np.testing.assert_array_equal(record['state_before'], [0., 0.])
    np.testing.assert_array_equal(record['control'], [1.])
    assert record['time_s'] == 0.
    assert record['step_dt_s'] == .007
    assert 'state_after' not in record


def test_recording_failure_preserved_and_propagated(boundary):
    def observer(event, record):
        raise OSError('injected archive failure')
    with pytest.raises(OSError, match='injected archive'):
        run(boundary, offline_timing='solve_time', record_callback=observer)
    partial = boundary.mpc.last_run_stats
    assert partial['recording_failure']['event'] == 'solve_started'
    assert partial['recording_failure']['error_type'] == 'OSError'
    assert partial['raw_solves'][0]['status'] == 'started'
    assert boundary.calls == []


def test_current_b_executes_old_plan_at_activation_age_then_new_plan_on_arrival(boundary):
    stats = run(boundary, offline_timing='solve_time')
    assert stats['timing_mode'] == 'B'
    np.testing.assert_allclose(stats['update_intervals_s'], [.03, .075, .0075, .0075])
    assert stats['interval_plan_solve_indices'].tolist() == [0, 0, 1, 2]
    # First 30ms consumes bootstrap u0=1; during first compute, bootstrap u1=2
    # continues from age30ms and holds beyond horizon60ms. Returned u0=11 starts
    # only at105ms; next u0=21 starts at112.5ms.
    assert stats['terminal_state'][0] == pytest.approx(.03*1 + .075*2 + .0075*11 + .0075*21)
    np.testing.assert_allclose(stats['feedback_timestamps_s'], [.03, .105, .1125])
    assert stats['control_activation_timestamps_s'].tolist()[:2] == [pytest.approx(.105), pytest.approx(.1125)]
    assert stats['control_activation_timestamps_s'][-1] is None
    assert stats['plan_activation_status'][-1] == 'discarded_task_terminal'
    solve2 = stats['raw_solves'][2]
    assert solve2['previous_optimizer_reference_time_s'] == pytest.approx(.03)
    assert solve2['warm_start_shift_s'] == pytest.approx(.075)
    assert solve2['feedback_time_s'] == pytest.approx(.105)
    assert solve2['warm_start'][0, 0] == pytest.approx(.18)
    assert solve2['warm_start'][0, 2] == pytest.approx(12.)  # Warm seed age75ms, not newly active control age0.
    initial_compute = [r for r in stats['raw_execution'] if .03 <= r['time_s'] < .105-1e-12]
    assert initial_compute
    assert all(r['plan_solve_index'] == 0 and r['control'][0] == 2 for r in initial_compute)
    assert initial_compute[0]['active_control_age_s'] == pytest.approx(.03)
    latest = stats['raw_solves'][-1]
    np.testing.assert_allclose(latest['warm_start'][0, 3:5], [1.25, 2.5])
    np.testing.assert_allclose(latest['warm_start'][0, [2, 5]], [21., 22.])
    assert latest['output'][0, 2] == 31.
    assert latest['activation_time_s'] is None
    assert latest['activation_status'] == 'discarded_task_terminal'
    assert latest['wall_solve_time_s'] == pytest.approx(.01)
    assert stats['update_intervals_s'][-1] == pytest.approx(.0075)


def test_current_b_goal_transition_keeps_active_old_plan_and_retains_unused_output(boundary):
    _, stats = boundary.mpc.run_mpc_goals(np.zeros(2), [[100.,0.,0.], [200.,0.,0.]],
        sim_dt=.007, goal_timeout=.04, offline_timing='solve_time', record_raw=True)
    raw = stats['raw_solves']
    assert raw[1]['activation_status'] == 'discarded_goal_transition'
    assert raw[2]['category'] == 'goal_switch'
    assert raw[1]['solve_arrival_time_s'] == pytest.approx(.105)
    assert raw[2]['feedback_time_s'] == pytest.approx(.105)
    assert raw[2]['previous_optimizer_reference_time_s'] == pytest.approx(.03)
    assert raw[2]['warm_start_shift_s'] == pytest.approx(.075)
    after_transition = [r for r in stats['raw_execution'] if .105-1e-12 <= r['time_s'] < .1125-1e-12]
    assert after_transition
    assert all(r['plan_solve_index'] == 0 and r['control'][0] == 2 for r in after_transition)
    assert after_transition[0]['active_control_age_s'] == pytest.approx(.105)
    assert stats['goal_outcomes'] == ['timeout', 'timeout']
    # This is the retained review failure: first call's 75ms must be paid in
    # full after feedback30ms, despite goal0's requested deadline40ms.
    # Bootstrap is exempt because its duration is explicitly uncharged.
    for previous, following in zip(raw[1:], raw[2:]):
        assert following['feedback_time_s'] >= previous['solve_arrival_time_s'] - 1e-12
    assert raw[1]['activation_time_s'] is None
    assert raw[1]['output'][0, 2] == 11.  # Solved output retained, never used.
    assert raw[2]['activation_time_s'] == pytest.approx(.1125)
    np.testing.assert_allclose(stats['goal_requested_deadline_times_s'], [.04, .145])
    np.testing.assert_allclose(stats['goal_end_times_s'], [.105, .145])
    np.testing.assert_allclose(stats['goal_timeout_lateness_s'], [.065, 0.], atol=1e-12)
    assert stats['interval_timeout_clipped'].tolist()[:3] == [False, False, False]
    assert stats['interval_timeout_clipped'][-1]
    # Distinct output2 controls begin only at112.5ms, after the old startup
    # plan pays both calls across the goal switch; discarded output1u=11
    # never appears in the executed controls.
    controls = [r['control'][0] for r in stats['raw_execution']]
    assert 11. not in controls
    assert 21. in controls
    assert stats['terminal_time_s'] == pytest.approx(.145)


def test_current_b_reach_during_compute_does_not_activate_completed_output(boundary):
    _, stats = boundary.mpc.run_mpc_goals(np.zeros(2), [[.18,0.,0.]],
        sim_dt=.007, goal_threshold=.001, goal_timeout=.5, offline_timing='solve_time', record_raw=True)
    assert stats['goal_outcomes'] == ['reached']
    assert stats['terminal_time_s'] == pytest.approx(.105)
    assert len(stats['raw_solves']) == 2
    assert stats['raw_solves'][1]['status'] == 'completed'
    assert stats['raw_solves'][1]['activation_status'] == 'discarded_task_terminal'
    assert all(r['plan_solve_index'] == 0 for r in stats['raw_execution'])
