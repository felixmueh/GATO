"""Bounded control-choice preparation and stopping/seed semantics, CPU only."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import control_studies_plan as plans
from adapters import control_studies as adapter


def test_plan_is_bounded_and_preserves_shared_baselines(tmp_path):
    before=(plans.ROOT/'experiments/campaign.json').read_bytes()
    desktop=plans.prepare(tmp_path/'desktop.json');jetson=plans.prepare(tmp_path/'jetson.json',True)
    cases=desktop['experiments'][0]['cases']
    assert len(cases)==38 and len(jetson['experiments'][0]['cases'])==6
    assert len({c['id'] for c in cases})==38
    assert desktop['experiments'][0]['machines']==['gtx1070','rtx5060']
    assert sum(c['parameters']['variant']=='current-default' for c in cases)==3
    assert not desktop['unsupported']
    assert sum(c['parameters']['variant']=='holding-gripper-upper' for c in cases)==1
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


@pytest.mark.parametrize('compact', [False, True])
def test_observation_timeout_and_rough_success_are_separate(tmp_path, compact):
    (tmp_path/'metrics.json').write_text(json.dumps(dict(task_success=False,task_gates={'public_reached':False},execution_error=None)))
    if compact:
        from adapters.task_recording import TaskRecorder
        with TaskRecorder(tmp_path) as recorder:
            recorder.write_controller_stats(dict(goal_outcomes=['timeout']))
    else:
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


def test_actual_gripper_reduction_preserves_arm_mapping_and_mimics():
    import pytest
    inputs=plans.ROOT/'experiments/inputs/control_studies'
    protocol=json.loads((inputs/'protocol.json').read_text())
    plant,identity=adapter.gripper_plant(inputs/protocol['C06_gripper']['asset'],.8028514559173916,protocol['C06_gripper']['asset_sha256'])
    assert plant.nq==plant.nv==7
    assert identity['arm_joint_names']==[f'arm_right_{i}_joint' for i in range(1,8)]
    positions=identity['locked_joint_positions_rad']
    assert positions['gripper_right_inner_finger_left_joint']==positions['gripper_right_finger_joint']
    assert positions['gripper_right_fingertip_left_joint']==-positions['gripper_right_finger_joint']
    assert identity['model_mismatch_intentional'] is True
    with pytest.raises(ValueError,match='outside asset limits'):
        adapter.gripper_plant(inputs/protocol['C06_gripper']['asset'],1.)


def test_failed_secondary_suite_gets_fresh_runner_attempt_and_pass_is_reused(tmp_path,monkeypatch):
    from recording import sha,write_json
    source=tmp_path/'source';source.mkdir();(source/'fixture.py').write_text('# synthetic fixture\n')
    build=tmp_path/'build';(build/'modules').mkdir(parents=True)
    module=build/'modules/bsqpN32_tiago_right.synthetic.so';module.write_bytes(b'synthetic native identity')
    (build/'cuda_access_smoke').write_bytes(b'synthetic smoke')
    monkeypatch.setattr(adapter.subprocess,'check_output',lambda *a,**k:'source-pin')
    group=tmp_path/'_preflight/ControlChoices-B'
    legacy=tmp_path/'_preflight/_task_suite/source';legacy.mkdir(parents=True)
    write_json(legacy/'metrics.json',dict(outcome='execution_failed',evidence_kind='synthetic_test',execution_error='retained original failure'))
    legacy_bytes=(legacy/'metrics.json').read_bytes()
    calls=[]
    def fake_suite(src,bld,out):
        out=Path(out);assert not out.exists();out.mkdir(parents=True);calls.append(out)
        passed=len(calls)>1
        write_json(out/'metrics.json',dict(outcome='task_suite_passed' if passed else 'execution_failed',exact_cases=passed,skipped=0,evidence_kind='synthetic_test'))
        write_json(out/'provenance.json',dict(source_commit='source-pin',smoke_sha256=sha(build/'cuda_access_smoke'),module_sha256=sha(module),adapter_sha256=sha(adapter.task_suite.__file__),
            bootstrap_sha256=sha(adapter.task_suite.EXPERIMENTS/'task_bootstrap/sitecustomize.py'),fixture_sha256={'fixture.py':sha(source/'fixture.py')}))
        return 0 if passed else 1
    monkeypatch.setattr(adapter.task_suite,'run',fake_suite)
    first=group/'attempt-001/task_suite';second=group/'attempt-002/task_suite';third=group/'attempt-003/task_suite'
    assert adapter.ensure_suite(source,build,first,legacy)==1
    failed_bytes=(first/'metrics.json').read_bytes()
    assert adapter.ensure_suite(source,build,second,legacy)==0
    assert adapter.ensure_suite(source,build,third,legacy)==0 and (third/'reuse.json').exists()
    assert calls==[first,second]
    assert (first/'metrics.json').read_bytes()==failed_bytes and (legacy/'metrics.json').read_bytes()==legacy_bytes
    # A passed result with changed fixture identity also needs a fresh invocation.
    (source/'fixture.py').write_text('# changed synthetic fixture\n')
    fourth=group/'attempt-004/task_suite'
    assert adapter.ensure_suite(source,build,fourth,legacy)==0 and calls[-1]==fourth


def test_same_machine_suite_reuse_across_families_is_source_scoped(tmp_path,monkeypatch):
    from recording import sha,write_json
    monkeypatch.setenv('GATO_MACHINE','gtx1070')
    source=tmp_path/'source';source.mkdir();(source/'fixture.py').write_text('# frozen')
    build=tmp_path/'build';(build/'modules').mkdir(parents=True)
    module=build/'modules/bsqpN32_tiago_right.test.so';module.write_bytes(b'native')
    (build/'cuda_access_smoke').write_bytes(b'smoke')
    pin=['source-pin'];monkeypatch.setattr(adapter.subprocess,'check_output',lambda *a,**k:pin[0])
    root=tmp_path/'gtx1070/_preflight';prior=root/'T01/attempt-001/task_suite';prior.mkdir(parents=True)
    write_json(prior.parent/'receipt.json',dict(machine='gtx1070',status='completed'))
    write_json(prior/'metrics.json',dict(outcome='task_suite_passed',exact_cases=True,skipped=0))
    write_json(prior/'provenance.json',dict(source_commit='source-pin',smoke_sha256=sha(build/'cuda_access_smoke'),module_sha256=sha(module),adapter_sha256=sha(adapter.task_suite.__file__),bootstrap_sha256=sha(adapter.task_suite.EXPERIMENTS/'task_bootstrap/sitecustomize.py'),fixture_sha256={'fixture.py':sha(source/'fixture.py')}))
    saved=(prior/'metrics.json').read_bytes();calls=[]
    monkeypatch.setattr(adapter.task_suite,'run',lambda src,bld,out:calls.append(out) or 1)
    def target(name):
        p=root/name/'attempt-001/task_suite';p.parent.mkdir(parents=True);write_json(p.parent/'receipt.json',dict(machine='gtx1070'));return p
    out=target('F01c');assert adapter.ensure_suite(source,build,out)==0 and not calls
    assert json.loads((out/'reuse.json').read_text())['evidence']==str(prior)
    pin[0]='different-source';assert adapter.ensure_suite(source,build,target('changed-source'))==1
    pin[0]='source-pin';monkeypatch.setenv('GATO_MACHINE','jetson');assert adapter.ensure_suite(source,build,target('wrong-machine'))==1
    assert (prior/'metrics.json').read_bytes()==saved
