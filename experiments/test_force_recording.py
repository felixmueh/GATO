"""Recording seams and preserved force measurements, without CUDA or tasks."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from adapters.force_recording import ForceRecorder, FullForceRecorder, project_force_record
from recording import iter_events


def force_payload():
    plan = np.ones((16, 665), dtype=np.float32)
    reply = dict(XU=plan, sqp_time_us=17, sqp_iters=np.array([2]),
                 pcg_iters=np.array([[100], [0]]), pcg_times_us=np.array([1., 2.]),
                 kkt_converged=np.array([False]), final_merit=np.array([3.]),
                 ls_num_iters=2, ls_step_size=np.array([[.5], [.25]]))
    return dict(mode='adapted', status='failed', error={'type': 'ValueError', 'message': 'retained'},
                plans=plan, selected_plan=plan[0], warm_seed=plan,
                inputs_submitted={'candidate_warm_seeds': plan},
                prediction_request={'inputs': {'candidate_warm_seeds': plan}},
                to_request={'warm_seeds': plan},
                inputs=dict(previous_time_s=0., observed_time_s=.035,
                            previous_state=np.zeros(14), observed_state=np.ones(14),
                            candidate_states=np.ones((16, 14)), candidate_references=plan,
                            candidate_warm_seeds=plan, applied_history=[{'control': np.ones(7)}]),
                hypotheses=np.ones((16, 6)), predictions=np.ones((16, 14)),
                errors=np.arange(16.), selected_index=2, selected_force=np.ones(6),
                estimated_force_error_N=2., estimated_torque_error_Nm=.1,
                truth_direction={'world_force_N': np.array([0., 0., -10.])},
                timings_s={'total': .035, 'to_backend': .03, 'candidate_forward': .004},
                backend_measurements=[{'native_solve_us': 17, 'public_solve_wall_s': .03}],
                native_records=[{'candidate_indices': [2], 'native_reply': reply}])


def test_force_projection_keeps_measurements_failures_and_worker_values(tmp_path):
    original = force_payload()
    smaller = project_force_record('force_tick', original)
    for field in ('plans', 'selected_plan', 'warm_seed', 'inputs_submitted', 'prediction_request', 'to_request'):
        assert field not in smaller and field in original
    for field in ('hypotheses', 'predictions', 'errors', 'selected_force'):
        np.testing.assert_array_equal(smaller[field], original[field])
    assert smaller['timings_s'] == original['timings_s']
    assert smaller['error'] == original['error']
    assert smaller['backend_measurements'] == original['backend_measurements']
    assert set(smaller['inputs']) == {'previous_time_s', 'observed_time_s', 'previous_state', 'observed_state'}
    reply = smaller['native_records'][0]['native_reply']
    assert 'XU' not in reply
    for key in ('sqp_iters', 'pcg_iters', 'pcg_times_us', 'kkt_converged'):
        np.testing.assert_array_equal(reply[key], original['native_records'][0]['native_reply'][key])
    assert original['native_records'][0]['native_reply']['XU'].shape == (16, 665)
    # Every executed substep and its applied control survives, at original dtype.
    with ForceRecorder(tmp_path) as recorder:
        for index in range(3):
            recorder('step_completed', dict(time_s=index * .003, step_dt_s=.003,
                     state_before=np.arange(14., dtype=np.float64),
                     state_after=np.arange(14., dtype=np.float64) + index,
                     control=np.arange(7., dtype=np.float32), plan_solve_index=0))
        recorder('force_tick', original)
        recorder('execution_failed', {'error': 'all failures retained'})
        recorder.manifest()
    events = list(iter_events(tmp_path))
    assert len(events) == 5
    assert events[-1]['record']['error'] == 'all failures retained'
    for index, event in enumerate(events[:3]):
        assert event['record']['step_dt_s'] == .003
        np.testing.assert_array_equal(event['record']['state_after'], np.arange(14.) + index)
        assert event['record']['control'].dtype == np.float32
    assert json.loads((tmp_path / 'recording.json').read_text())['full_solver_io'] is False


def test_force_comparison_retains_full_io(tmp_path):
    original = force_payload()
    with FullForceRecorder(tmp_path) as recorder:
        recorder('force_tick', original)
        recorder.manifest()
    saved = next(iter_events(tmp_path))['record']
    for field in ('plans', 'selected_plan', 'warm_seed'):
        np.testing.assert_array_equal(saved[field], original[field])
    np.testing.assert_array_equal(saved['native_records'][0]['native_reply']['XU'],
                                  original['native_records'][0]['native_reply']['XU'])
    assert json.loads((tmp_path / 'recording.json').read_text())['full_solver_io'] is True


@pytest.mark.parametrize('worker,full_io', [('run_task', False), ('run_native', True)])
def test_pinned_worker_import_gets_recording_projection(tmp_path, worker, full_io):
    """Exercise the deployed subprocess seam using a tiny source worker."""
    source = tmp_path / 'source'
    package = source / 'experiments' / 'force_adapter'
    package.mkdir(parents=True)
    (source / 'experiments' / '__init__.py').write_text('')
    (package / '__init__.py').write_text('')
    (package / (worker + '.py')).write_text(
        'import argparse\n'
        'from pathlib import Path\n'
        'def main():\n'
        '    from recording import RawRecorder\n'
        '    import numpy as np\n'
        '    p=argparse.ArgumentParser(); p.add_argument("--output"); a=p.parse_args()\n'
        '    with RawRecorder(Path(a.output)) as r:\n'
        '        r("force_tick", {"plans":np.ones((16,665)), "selected_index":3})\n'
        '        r.manifest()\n'
        '    return 0\n')
    output = tmp_path / 'output'
    wrapper = Path(__file__).parent / 'adapters' / 'force_worker.py'
    result = subprocess.run([sys.executable, str(wrapper), '--source', str(source),
                             '--worker', worker, '--output', str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = next(iter_events(output))['record']
    assert ('plans' in record) is full_io
    assert record['selected_index'] == 3
