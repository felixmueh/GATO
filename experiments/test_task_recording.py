"""Ordinary-task data contract checks, without solving or changing motion."""
from copy import deepcopy

import numpy as np

from adapters.task_recording import TaskRecorder, project_task_record
from recording import RawRecorder, iter_events, load_controller_stats, same_value


def test_native_quality_work_and_motion_survive_projection(tmp_path):
    native = dict(XU=np.array([np.nan, 3.], np.float32), q=np.array([np.inf, -.25]),
                  pcg_iters=np.array([[0, 120], [3, 2]], np.int32),
                  sqp_iters=np.array([1, 2]), kkt_converged=np.array([False, False]),
                  final_merit=np.array([1., 2.]), stage_times_us=np.array([4., 6.]))
    solve = dict(solve_index=0, category='startup/cold', status='completed',
                 warm_start=np.zeros((2, 665)), reference=np.zeros((2, 192)),
                 output=native['XU'], solver_stats=native, wall_solve_time_s=.003,
                 native_solve_time_ms=2.5, best_trajectory_id=1,
                 lane_quality={'finite': [False, True]})
    original = deepcopy(solve)
    step = dict(time_s=.003, step_dt_s=.003, state_before=np.zeros(14),
                state_after=np.arange(14.), control=np.arange(7.),
                plan_solve_index=0, goal_index=0, active_control_age_s=.001)
    with TaskRecorder(tmp_path) as recorder:
        recorder('solve_started', dict(solve, status='started'))
        recorder('solve_completed', solve)
        recorder('solve_selected', solve)
        recorder('step_started', step)
        recorder('step_completed', step)
        recorder('solve_failed', dict(solve_index=1, error='failed solve', wall_solve_time_s=.004))
        stats = dict(raw_solves=[solve], raw_execution=[step],
                     update_intervals_s=np.array([.006]), goal_outcomes=['timeout'])
        recorder.write_controller_stats(stats)
    events = list(iter_events(tmp_path))
    kept = events[1]['record']
    assert not {'output', 'warm_start', 'reference'} & kept.keys()
    np.testing.assert_array_equal(kept['solver_stats']['pcg_iters'], native['pcg_iters'])
    np.testing.assert_array_equal(kept['solver_stats']['kkt_converged'], native['kkt_converged'])
    assert kept['solver_stats']['omitted_array_quality']['q']['nonfinite_count'] == 1
    assert kept['solver_stats']['omitted_array_quality']['q']['finite_max_abs'] == .25
    assert kept['omitted_array_quality']['output']['finite'] is False
    assert kept['lane_quality'] == solve['lane_quality']
    assert same_value(events[4]['record'], step)
    assert events[5]['record']['error'] == 'failed solve'
    assert same_value(solve, original) and stats['raw_solves'][0] is solve
    assert same_value(load_controller_stats(tmp_path)['update_intervals_s'], stats['update_intervals_s'])


def test_context_keeps_observed_reference_control_and_full_study_io(tmp_path):
    observed = dict(q=np.ones(7), torques=np.ones(7), reference=np.ones(6),
                    lane_results={'passed': True})
    assert same_value(project_task_record('feedback_state', observed), observed)
    assert same_value(project_task_record('frozen_reference', observed), observed)
    publication = project_task_record('planned_published_horizon', dict(observed, dt_s=.03))
    assert 'torques' not in publication and publication['dt_s'] == .03
    with RawRecorder(tmp_path) as recorder:
        recorder('matched_input', observed)
    assert same_value(next(iter_events(tmp_path))['record'], observed)


def test_bb_omits_prefix_horizons_preserves_lane_assessments():
    payload = dict(path='step_0001_prefixes.npz', payload=dict(
        lane_0_states=np.zeros((10, 14)), lane_0_step_dt_s=np.full(10, .003),
        lane_0_quality={'finite': True}, merits=np.array([1., 2.])))
    kept = project_task_record('step_record', payload)['payload']
    assert set(kept) == {'lane_0_quality', 'merits'}


def test_small_finite_native_lists_preserve_values_shape_and_consumers(tmp_path):
    # Tiny vectors/matrices remain numerically lossless; large/nonfinite arrays
    # retain their typed NPZ representation rather than changing missing values.
    tiny = np.array([[0., .12345678], [3., -4.]], np.float32)
    nonfinite = np.array([np.nan, np.inf], np.float32)
    large = np.arange(33, dtype=np.int32)
    empty = np.zeros((0, 5), dtype=np.float32)
    record = dict(solver_stats=dict(min_merit=tiny, diagnostic=nonfinite, pcg_iters=large, empty=empty),
                  observed=tiny, stats__sqp_iters=np.array([2], np.int32))
    projected = project_task_record('solve_completed', record)
    assert isinstance(projected['solver_stats']['min_merit'], list)
    np.testing.assert_array_equal(np.asarray(projected['solver_stats']['min_merit'], dtype=tiny.dtype), tiny)
    assert isinstance(projected['solver_stats']['diagnostic'], np.ndarray)
    assert isinstance(projected['solver_stats']['pcg_iters'], np.ndarray)
    assert projected['solver_stats']['empty'].shape == (0, 5)
    assert projected['observed'] is tiny and projected['stats__sqp_iters'] == [2]


def test_json_and_production_feedback_summaries_decode_equivalently(tmp_path):
    feedback = dict(timestamps=np.array([.03, .06]),
                    joint_positions=np.arange(14., dtype=np.float64).reshape(2, 7),
                    goal_indices=np.array([0, 1]),
                    goal_outcomes=['reached', 'timeout'])
    historical = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in feedback.items()}
    historical['missing_observation'] = [1., None]
    historical['ragged_observation'] = [[1.], [2., 3.]]
    for name, stats in (('production', feedback), ('historical', historical)):
        with TaskRecorder(tmp_path / name) as recorder:
            recorder.write_controller_stats(stats)
    production = load_controller_stats(tmp_path / 'production')
    saved = load_controller_stats(tmp_path / 'historical')
    for key in feedback:
        if isinstance(feedback[key], np.ndarray):
            assert same_value(production[key], saved[key])
        else:
            assert production[key] == saved[key]
    assert saved['missing_observation'] == [1., None]
    assert saved['ragged_observation'] == [[1.], [2., 3.]]
    assert historical['timestamps'] == [.03, .06]  # Input untouched.


def test_object_histories_preserve_missing_and_ragged_observations(tmp_path):
    ragged = np.empty(2, dtype=object)
    ragged[0] = np.array([1, 2], dtype=np.int32)
    ragged[1] = np.array([3], dtype=np.int32)
    stats = dict(control_activation_timestamps_s=np.array([.03, None], dtype=object),
                 diagnostics={'ragged': ragged}, terminal_state=np.arange(14.))
    with TaskRecorder(tmp_path) as recorder:
        recorder.write_controller_stats(stats)
    saved = load_controller_stats(tmp_path)
    assert saved['control_activation_timestamps_s'] == [.03, None]
    for actual, expected in zip(saved['diagnostics']['ragged'], ragged):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(saved['terminal_state'], stats['terminal_state'])
    assert stats['control_activation_timestamps_s'].dtype == object
    assert list(iter_events(tmp_path))[0]['record']['control_activation_timestamps_s'] == [.03, None]
