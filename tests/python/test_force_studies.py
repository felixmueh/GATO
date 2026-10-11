"""Bounded CPU wiring checks for the real force study workers and plant clock."""
from pathlib import Path
import sys
import json
import numpy as np
import pinocchio as pin
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'python'))
sys.path.insert(0,'/workspace/GATO-experiment-runner-simple/experiments')
from recording import RawRecorder,sha,write_json,iter_events
from experiments.force_adapter.run_native import run_case
from experiments.force_adapter.task_backend import ForceTask
from bsqp.offline_mpc import run_mode_b


class Optimizer:
    def __init__(self,*args):
        self.native_records=[];self.measurements=[];self.last_stats={};self.requests=[]
    def __call__(self,request):
        self.requests.append(request)
        self.native_records.append(dict(candidate_indices=request.candidate_indices,native_reply=dict(XU=request.warm_seeds.copy(),sqp_time_us=10.)))
        self.measurements.append(dict(native_solve_us=10.,memory=None,memory_error='CPU test has no telemetry'))
        self.last_stats=dict(sqp_iters=np.ones(len(request.candidate_indices)),kkt_converged=np.zeros(len(request.candidate_indices)))
        return request.warm_seeds.copy()


def test_native_pair_reuses_frozen_inputs_and_excludes_unrelated_warmup(tmp_path):
    outputs=[]
    module=tmp_path/'unused.so';module.write_bytes(b'cpu placeholder')
    for mode in ('original','adapted'):
        case=dict(parameters=dict(study='E03',mode=mode,input_index=0,truth_index=1,timing_mode='A',backend='internal'))
        out=tmp_path/mode
        metrics=run_case(case,out,ROOT/'experiments/force_adapter/native_inputs',module,RawRecorder,sha,write_json,optimizer_factory=Optimizer)
        assert metrics['execution_error'] is None
        records=[e['record'] for e in iter_events(out) if e['event']=='force_tick']
        assert [r['category'] for r in records]==['excluded_warmup','measured']
        assert len(records[-1]['native_records'])==1
        outputs.append(records[-1])
    a,b=outputs
    np.testing.assert_allclose(a['predictions'],b['predictions'],atol=1e-4)
    np.testing.assert_array_equal(a['hypotheses'],b['hypotheses'])
    assert a['selected_index']==b['selected_index']
    assert len(a['to_request']['candidate_indices'])==16 and len(b['to_request']['candidate_indices'])==1


def test_static_optimizer_failure_retains_raw_inputs_and_error(tmp_path):
    class Bad(Optimizer):
        def __call__(self,request):raise RuntimeError('retained backend failure')
    module=tmp_path/'unused.so';module.write_bytes(b'placeholder')
    metrics=run_case(dict(parameters=dict(study='E03',mode='adapted',input_index=0,truth_index=0,timing_mode='A',backend='internal')),
        tmp_path/'failure',ROOT/'experiments/force_adapter/native_inputs',module,RawRecorder,sha,write_json,optimizer_factory=Bad)
    assert 'retained backend failure' in metrics['execution_error']
    events=list(iter_events(tmp_path/'failure'))
    assert any(e['event']=='force_tick_inputs' for e in events)
    assert next(e['record'] for e in events if e['event']=='force_tick')['status']=='failed'


@pytest.mark.parametrize('mode,expected',[('no_estimator',1),('original',16),('adapted',1)])
def test_task_startup_has_no_truth_access_and_keeps_history_and_memory_bounded(mode,expected):
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    optimizer=Optimizer();events=[]
    task=ForceTask(model,'unused',{},mode,np.array([2.,0,0,0,0,0]),lambda e,r:events.append((e,r)),optimizer)
    state=np.r_[(model.lowerPositionLimit+model.upperPositionLimit)/2,np.zeros(7)]
    warm=np.zeros((1,665));reference=np.zeros((1,192))
    task.solve(state[None,:],reference,warm)
    np.testing.assert_array_equal(optimizer.requests[0].backend_wrenches,np.zeros((1,6)))
    task.record('solve_completed',{})
    assert not optimizer.measurements and not optimizer.native_records
    q,v=task.integrate(state[:7],state[7:],np.zeros(7),.003)
    task.solve(np.r_[q,v][None,:],reference,warm)
    assert len(optimizer.requests[-1].candidate_indices)==expected
    task.record('solve_completed',{})
    assert not optimizer.measurements and not optimizer.native_records
    if mode!='no_estimator':
        tick=next(r for e,r in events if e=='force_tick')
        assert tick['inputs'].applied_history[0].start_s==0.
        assert tick['inputs'].applied_history[-1].end_s==.003
        assert len(tick['backend_measurements'])==1


def test_force_facade_uses_unchanged_mode_B_bootstrap_and_old_plan_clock():
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    optimizer=Optimizer();events=[]
    task=ForceTask(model,'unused',{},'no_estimator',np.zeros(6),lambda e,r:events.append((e,r)),optimizer)
    state=np.r_[(model.lowerPositionLimit+model.upperPositionLimit)/2,np.zeros(7)]
    readings=iter([0.,.1,.1,.11,.11,.12])
    _,stats=run_mode_b(task,state,integrate=task.integrate,clock=lambda:next(readings),sim_dt=.003,
        record_raw=True,record_callback=task.record,reference=np.zeros(36*6),sim_time=.05)
    assert stats['startup_wall_solve_time_s']==pytest.approx(.1)
    np.testing.assert_allclose(stats['update_intervals_s'],[.03,.01,.01],atol=1e-8)
    assert stats['interval_plan_solve_indices'].tolist()==[0,0,1]
    assert stats['terminal_time_s']==pytest.approx(.05)
    assert len(optimizer.requests)==3


def test_task_worker_bootstrap_failure_preserves_partial_measurement(tmp_path):
    from experiments.force_adapter.run_task import run_case as task_case
    class Bad(Optimizer):
        def __call__(self,request):
            self.measurements.append(dict(native_solve_us=None,memory=None,memory_error='failed constructor'))
            raise RuntimeError('bootstrap failure')
    module=tmp_path/'unused.so';module.write_bytes(b'placeholder')
    metrics=task_case(dict(parameters=dict(mode='no_estimator',truth_index=1,timing_mode='B',backend='internal')),
        tmp_path/'task',ROOT/'experiments/force_adapter/task_inputs',module,RawRecorder,sha,write_json,preflight=True,optimizer=Bad())
    assert 'bootstrap failure' in metrics['execution_error']
    assert metrics['task_success'] is None and metrics['physical_pass'] is None
    events=list(iter_events(tmp_path/'task'))
    assert any(e['event']=='force_algorithm_inputs' for e in events)
    failed=next(e['record'] for e in events if e['event']=='force_algorithm_failed')
    assert failed['backend_measurements'][0]['memory_error']=='failed constructor'
