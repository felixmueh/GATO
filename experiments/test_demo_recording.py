"""Recording-only CPU checks; no solver, plant, build or motion."""
from copy import deepcopy
import json

import numpy as np
import pytest

from adapters.demo import DemoRecorder, collector_source, load_collector
from recording import iter_events, load_controller_stats, same_value
from adapters.task_recording import project_task_record


def payload():
    return dict(solve_index=0, category='warm', status='started', activation_status='pending',
                state=np.zeros((1,14),np.float32), reference=np.ones((1,192),np.float32),
                warm_start=np.zeros((1,665),np.float32))


def test_lifecycle_deltas_cross_chunks_snapshot_and_failure(tmp_path):
    expected=[]
    with DemoRecorder(tmp_path) as recorder:
        record=payload()
        def emit(kind):
            expected.append((kind,deepcopy(project_task_record(kind,record))));recorder(kind,record)
        emit('solve_started')
        recorder.flush()
        assert len(list(iter_events(tmp_path)))==1  # Explicit flush retains observation.
        record.update(status='completed', output=np.ones((1,665),np.float32),
                      solver_stats={'pcg_iters':np.array([[0],[2]],np.int32)},wall_solve_time_s=.004)
        emit('solve_completed');recorder.flush()
        record['best_trajectory_id']=0;emit('solve_selected')
        record.update(activation_status='activated',activation_time_s=.004);emit('plan_activated')
        record['output'][:]=8  # Immutable earlier snapshots.
        record=payload();record['solve_index']=1;emit('solve_started')
        record.update(status='failed',error_type='RuntimeError',error='retained failure',wall_solve_time_s=.001)
        emit('solve_failed')
        stats={'raw_solves':[expected[3][1],expected[-1][1]],'raw_execution':[],
               'timestamps':np.array([.004]),'goal_outcomes':['timeout']}
        recorder.write_controller_stats(stats)
        assert 'raw_solves' in stats  # Task metrics still use original histories.
    decoded=list(iter_events(tmp_path))
    assert all(e['event']==k and same_value(e['record'],r) for e,(k,r) in zip(decoded,expected))
    summary=load_controller_stats(tmp_path)
    assert summary['history_references']['raw_solves']['count']==2
    assert decoded[-2]['record']['error']=='retained failure'
    assert 'raw_solves' not in summary
    envelopes=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert envelopes[2]['record_ref']==1 and list(envelopes[2]['record'])==['best_trajectory_id']


def test_summary_preserves_unique_feedback_and_metric_semantics(tmp_path):
    from pathlib import Path
    repository=Path(__file__).resolve().parents[1]
    collector,_=collector_source(repository,tmp_path/'build')
    worker=load_collector(collector/'experiments/deployment/demo/run_demo.py')
    recorder=DemoRecorder(tmp_path/'attempt');record=payload();recorder('solve_started',record)
    native=dict(sqp_iters=np.array([2],np.int32),pcg_iters=np.array([[0],[2]],np.int32),
                pcg_times_us=np.array([2.,3.],np.float32),kkt_converged=np.array([0],np.int32))
    record.update(status='completed',solver_stats=native,best_trajectory_id=0)
    recorder('solve_completed',record);recorder('solve_selected',record)
    stats=dict(raw_solves=[record],raw_execution=[],timestamps=np.array([.03]),
               goal_indices=np.array([0]),goal_distances=np.array([.02]),
               tool_axis_errors=np.array([.1]),joint_velocities=np.zeros((1,7)),
               ee_actual=np.array([[np.nan,2.,3.]]),goal_outcomes=['timeout'],
               sqp_iters=np.array([2]),pcg_iters=[native['pcg_iters']],
               pcg_times_us=[native['pcg_times_us']],kkt_converged=[native['kkt_converged']])
    protocol=dict(task_id='square',goals=[{}],position_threshold_m=.05,axis_threshold_deg=30.,speed_L1_threshold_rad_s=1.)
    before=worker.rough_reaching_metrics(protocol,stats)
    recorder.write_controller_stats(stats);recorder.manifest()
    restored=load_controller_stats(tmp_path/'attempt')
    assert worker.rough_reaching_metrics(protocol,restored)==before
    assert same_value(restored['ee_actual'],stats['ee_actual'])  # NaN not converted to null.
    for key in ('sqp_iters','pcg_iters','pcg_times_us','kkt_converged'):
        assert key not in restored and key in restored['history_references']
    restored_native=next(e for e in iter_events(tmp_path/'attempt') if e['event']=='solve_selected')['record']['solver_stats']
    for key in native:
        np.testing.assert_array_equal(restored_native[key],native[key])
    # A differently scoped solver-history value is unique and must be retained.
    stats['pcg_iters']=[np.array([[7]],np.int32)]
    recorder.write_controller_stats(stats)
    assert same_value(load_controller_stats(tmp_path/'attempt')['pcg_iters'][0],stats['pcg_iters'][0])


def test_handled_failure_flushes_current_chunk(tmp_path):
    with pytest.raises(RuntimeError):
        with DemoRecorder(tmp_path) as recorder:
            recorder('solve_started',payload())
            recorder('execution_failure',{'stage':'execution','message':'fixture failure'})
            raise RuntimeError('fixture failure')
    assert [e['event'] for e in iter_events(tmp_path)]==['solve_started','execution_failure']
