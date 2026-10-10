"""Tracking clocks share the reaching elapsed-time implementation and CPU oracles."""
import numpy as np
import pytest

from test_reaching_timing import boundary  # Both controller implementations; no native constructors.


def reference(boundary):
    poses = np.zeros((30, 6), dtype=np.float32)
    poses[:, 0] = np.arange(30)*boundary.mpc.dt
    poses[:, 3] = np.arange(30)*boundary.mpc.dt*2
    return poses.reshape(-1)


def run(boundary, **kwargs):
    return boundary.mpc.run_mpc_fig8(np.zeros(2), reference(boundary), sim_dt=.007,
                                    sim_time=.12, record_raw=True, **kwargs)[1]


def test_tracking_current_b_replays_old_prefix_and_uses_actual_feedback_reference(boundary):
    stats = run(boundary, offline_timing='solve_time')
    assert stats['interval_plan_solve_indices'].tolist() == [0, 0, 1, 2]
    np.testing.assert_allclose(stats['feedback_timestamps_s'], [.03, .105, .1125])
    assert stats['terminal_time_s'] == pytest.approx(.12)
    assert stats['terminal_state'][0] == pytest.approx(.03*1+.075*2+.0075*11+.0075*21)
    assert stats['control_activation_timestamps_s'][-1] is None
    assert stats['plan_activation_status'][-1] == 'discarded_task_terminal'
    for solve in stats['raw_solves']:
        # Each reference window starts at actual measured solve-start time,
        # without anticipating arrival time or shifting reference by wall solve.
        assert solve['reference'][0, 0] == pytest.approx(solve['feedback_time_s'], abs=2e-8)
        assert solve['reference'][0, 3] == pytest.approx(solve['feedback_time_s']*2, abs=3e-8)
    solve2 = stats['raw_solves'][2]
    assert solve2['warm_start_shift_s'] == pytest.approx(.075)
    assert solve2['warm_start'][0, 0] == pytest.approx(.18)
    assert solve2['warm_start'][0, 2] == 12.
    assert sum(r['step_dt_s'] for r in stats['raw_execution']) == pytest.approx(.12)
    assert stats['raw_solves'][0]['category'] == 'startup/cold'


def test_tracking_mode_a_default_keeps_fixed_clock_and_full_raw_records(boundary):
    stats = run(boundary)
    assert stats['timing_mode'] == 'A'
    np.testing.assert_allclose(stats['timestamps'], [.03, .06, .09, .12])
    # Historical tracking tests duration at loop entry, solves after last
    # nominal prefix; retain that extra terminal solve for frozen replay.
    assert len(stats['raw_solves']) == 5
    np.testing.assert_allclose(stats['update_intervals_s'], [.03]*4)
    assert stats['terminal_time_s'] == pytest.approx(.12)
    assert len(stats['raw_execution']) == 20
    np.testing.assert_allclose(stats['wall_solve_times_s'], [.075, .0075, .01, .01])


def test_tracking_legacy_flag_preserves_returned_plan_preceding_duration(boundary):
    stats = run(boundary, offline_timing='solve_time_legacy')
    assert stats['timing_mode'] == 'B-legacy'
    assert stats['interval_plan_solve_indices'].tolist() == [0, 1, 2, 3]
    assert stats['terminal_state'][0] == pytest.approx(.03*1+.035*11+.040*12+.0075*21+.01*31)
    # Legacy sampling on the retained 7ms grid straddles the 30ms knot:
    # u0 lasts35ms. Preserve that discretization as historical evidence.
    # Legacy tracking does not clip end time; this is retained and labeled.
    assert stats['terminal_time_s'] == pytest.approx(.1225)


def test_tracking_current_b_control_knot_split_and_end_hold(boundary):
    boundary.delays[1] = .0475
    boundary.delays[2] = .1
    stats = run(boundary, offline_timing='solve_time')
    # New plan1 starts at .0775 and splits at its own knot .1075, independent
    # of plant grid and optimizer's earlier reference origin .03.
    assert stats['raw_solves'][1]['activation_time_s'] == pytest.approx(.0775)
    rows = [r for r in stats['raw_execution'] if r['plan_solve_index'] == 1]
    assert rows[0]['active_control_age_s'] == pytest.approx(0.)
    before = [r for r in rows if r['time_s'] < .1075-1e-12]
    after = [r for r in rows if r['time_s'] >= .1075-1e-12]
    assert before and after
    assert all(r['control'][0] == 11 for r in before)
    assert all(r['control'][0] == 12 for r in after)
    assert before[-1]['time_s'] + before[-1]['step_dt_s'] == pytest.approx(.1075)
    assert stats['terminal_time_s'] == pytest.approx(.12)
    assert stats['interval_timeout_clipped'][-1]


def test_tracking_callbacks_and_startup_failure_preserve_partial_evidence(boundary):
    events = []
    def observer(event, record):
        events.append(event)
        boundary.clock.now += .1
    stats = run(boundary, offline_timing='solve_time', record_callback=observer)
    np.testing.assert_allclose(stats['wall_solve_times_s'], [.075, .0075, .01])
    assert 'plan_activated' in events and 'plan_discarded' in events
    assert boundary.mpc.last_run_stats is stats
    boundary.mpc.solver.solve = lambda *args: (_ for _ in ()).throw(RuntimeError('tracking injected failure'))
    with pytest.raises(RuntimeError, match='tracking injected'):
        run(boundary, offline_timing='solve_time')
    failed = boundary.mpc.last_run_stats['raw_solves'][0]
    assert failed['status'] == 'failed'
    assert failed['warm_start'].shape == (1, 8)
    assert failed['reference'][0, 0] == 0
    assert 'output' not in failed


def test_tracking_live_elapsed_flag_rejected_without_ros_change(boundary):
    if boundary.kind != 'tiago':
        return
    with pytest.raises(ValueError, match='without a live controller'):
        boundary.mpc.run_mpc_fig8(np.zeros(2), reference(boundary), controller=object(), offline_timing='solve_time')
    assert boundary.calls == []


@pytest.mark.parametrize('mode', ['controller_dt', 'solve_time', 'solve_time_legacy'])
@pytest.mark.parametrize('kwargs', [dict(sim_dt=0), dict(sim_dt=np.nan), dict(sim_time=0), dict(sim_time=-1)])
def test_tracking_invalid_timing_rejected_before_solver(boundary, mode, kwargs):
    with pytest.raises(ValueError):
        boundary.mpc.run_mpc_fig8(np.zeros(2), reference(boundary), offline_timing=mode, **kwargs)
    assert boundary.calls == []


def test_tracking_plant_error_retains_completed_prefix_and_attempted_step(boundary, monkeypatch):
    original = boundary.module.rk4
    calls = []
    def broken(*args):
        calls.append(None)
        if len(calls) == 3:
            raise RuntimeError('tracking injected plant failure')
        return original(*args)
    monkeypatch.setattr(boundary.module, 'rk4', broken)
    with pytest.raises(RuntimeError, match='tracking injected plant'):
        run(boundary, offline_timing='solve_time')
    stats = boundary.mpc.last_run_stats
    assert [s['status'] for s in stats['raw_execution']] == ['completed', 'completed', 'failed']
    assert stats['terminal_time_s'] == pytest.approx(.014)
    assert stats['terminal_state'][0] == pytest.approx(.014)
    assert stats['raw_execution'][-1]['state_before'][0] == pytest.approx(.014)
    assert 'state_after' not in stats['raw_execution'][-1]
