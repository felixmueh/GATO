"""Bounded control-choice preparation and stopping/seed semantics, CPU only."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import control_studies_plan as plans
from adapters import control_studies as adapter


def test_plan_is_bounded_and_preserves_shared_baselines(tmp_path):
    before=(plans.ROOT/'experiments/campaign.json').read_bytes()
    desktop=plans.prepare(tmp_path/'desktop.json');jetson=plans.prepare(tmp_path/'jetson.json',True)
    cases=desktop['experiments'][0]['cases']
    assert len(cases)==37 and len(jetson['experiments'][0]['cases'])==6
    assert len({c['id'] for c in cases})==37
    assert desktop['experiments'][0]['machines']==['gtx1070','rtx5060']
    assert sum(c['parameters']['variant']=='current-default' for c in cases)==3
    assert desktop['unsupported']['C06-model-gripper']
    assert 'cuda_access_smoke' in desktop['experiments'][0]['build']['artifacts']
    assert (plans.ROOT/'experiments/campaign.json').read_bytes()==before


def test_damping_limit_terminal_changes_are_isolated():
    base=dict(kind='reaching',solver_params=dict(qd_cost=.1,vel_lim_cost=.1,N_cost=80.,q_cost=4.),initial_state=[0.]*14)
    protocol=dict(id='test',observation_dwell_s=8.03,baseline_reuse='one shared baseline')
    low=next(v for v in plans.variants() if v['id']=='damping-low')
    changed=adapter.apply_protocol(base,low,protocol)
    assert changed['solver_params']==dict(base['solver_params'],qd_cost=.01)
    assert base['solver_params']['qd_cost']==.1 and changed['full_observation']
    assert not adapter.apply_protocol(base,low,protocol,True)['full_observation']


def test_cold_and_unshifted_are_reanchored_without_changed_inputs():
    previous=np.arange(665,dtype=np.float32)[None];state=np.arange(14,dtype=np.float32)+100
    kept=adapter.seed('unshifted',previous,state,14,7,32)
    np.testing.assert_array_equal(kept[0,:14],state);np.testing.assert_array_equal(kept[0,14:],previous[0,14:])
    cold=adapter.seed('cold',previous,state,14,7,32)
    for k in range(32):np.testing.assert_array_equal(cold[0,k*21:k*21+14],state)
    for k in range(31):assert not cold[0,k*21+14:k*21+21].any()
    assert previous[0,0]==0


def test_observation_timeout_and_rough_success_are_separate(tmp_path):
    (tmp_path/'metrics.json').write_text(json.dumps(dict(task_success=False,task_gates={'public_reached':False},execution_error=None)))
    (tmp_path/'controller_stats.json').write_text(json.dumps(dict(goal_outcomes=['timeout'])))
    fields=('time_s','position_error_m','tool_axis_error_deg','speed_L1_rad_s')
    with (tmp_path/'executed_trace.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        w.writerow(dict(zip(fields,(0.,.1,0.,0.))));w.writerow(dict(zip(fields,(3.,.05,30.,1.))));w.writerow(dict(zip(fields,(8.,.03,10.,.1))))
    metrics=adapter.window_metrics(tmp_path)
    assert metrics['observation_completed'] and metrics['task_success'] and metrics['first_rough_success_s']==3.
    assert metrics['public_controller_goal_outcomes']==['timeout']
    assert adapter.window_metrics(tmp_path,True)['task_success'] is None


def test_target_suite_failure_holds_canonical_case(tmp_path,monkeypatch):
    inputs=plans.ROOT/'experiments/inputs/control_studies'
    monkeypatch.setattr(adapter,'ensure_suite',lambda *a:1)
    monkeypatch.setattr(adapter.task,'run',lambda *a: (_ for _ in ()).throw(AssertionError('must not execute canonical after suitefailure')))
    assert adapter.run(tmp_path,tmp_path,inputs,None,tmp_path/'output',True)==1
    metrics=json.loads((tmp_path/'output/metrics.json').read_text())
    assert metrics['task_success'] is metrics['physical_pass'] is None
