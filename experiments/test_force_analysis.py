"""Saved force correspondence reuses E03 and keeps failed/foreign cohorts."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from recording import RawRecorder,sha,write_json
from analysis.force import load


def test_saved_pairs_failures_and_distinct_cohorts(tmp_path):
    attempts=[]
    for mode,bank,status in [('original','bankA','complete'),('adapted','bankA','complete'),('adapted','bankB','complete'),('original','bankA','failed')]:
        name=f'attempt-{len(attempts):03d}';raw=tmp_path/name
        with RawRecorder(raw) as recorder:
            write_json(raw/'protocol.json',dict(case=dict(mode=mode,input_index=0,truth_index=1)))
            write_json(raw/'provenance.json',dict(machine='rtx5060',source_commit='a'*40,module_sha256='module',bank_manifest_sha256=bank))
            write_json(raw/'metrics.json',dict(parameters=dict(mode=mode,input_index=0,truth_index=1),evidence_kind='measurement',outcome=status,execution_error='retained' if status=='failed' else None))
            recorder('force_tick',dict(category='measured',status=status,selected_index=3,hypotheses=np.zeros((16,6)),predictions=np.zeros((16,14)),
                timings_s=dict(total=2. if mode=='original' else 1.),backend_measurements=[dict(native_solve_us=1000.,memory=None,memory_error='unavailable')]))
            recorder.manifest()
        attempts.append(dict(experiment='E03',attempt=name,status='execution_failed' if status=='failed' else 'completed',manifest_sha256=sha(raw/'SHA256SUMS.json')))
    index=tmp_path/'index.json';write_json(index,dict(attempts=attempts))
    result=load(index)
    assert len(result['attempts'])==4 and len(result['ticks'])==4
    assert len(result['pairs'])==1 and result['pairs'][0]['E01_correspondence']
    assert result['pairs'][0]['original_over_adapted_wall']==2.
    assert result['ticks'][0]['native_reciprocal_Hz']==1000.
    assert result['ticks'][0]['observed_additional_device_bytes'] is None
    assert result['ticks'][-1]['status']=='failed'
    # collect.py's real current output is a list, unlike historical pinned indices.
    from collect import collect
    for entry in attempts:
        write_json(tmp_path/entry['attempt']/'receipt.json',dict(experiment='E03',machine='rtx5060',case=dict(id=entry['attempt']),status=entry['status']))
    blocked=tmp_path/'attempt-004';blocked.mkdir()
    write_json(blocked/'receipt.json',dict(experiment='E02-original-truth1',machine='rtx5060',case=dict(id='original-truth1',
        parameters=dict(mode='original',truth_index=1)),status='blocked',error='retained canonical failure'))
    preflight=tmp_path/'attempt-005';preflight.mkdir()
    write_json(preflight/'receipt.json',dict(experiment='E02-original-truth1',kind='preflight',machine='rtx5060',
        case=dict(id='preflight'),status='completed'))
    write_json(preflight/'metrics.json',dict(task_success=True,physical_pass=True))
    collected=collect(tmp_path)
    list_result=load(index)
    assert isinstance(collected,list) and len(list_result['attempts'])==5
    assert len(list_result['pairs'])==1 and list_result['ticks'][-1]['status']=='failed'
    assert list_result['tasks'][0]['mode']=='original' and list_result['tasks'][0]['truth_index']==1
    assert list_result['tasks'][0]['execution_error']=='retained canonical failure'
    assert list_result['tasks'][0]['task_success'] is None
    # Retain the pinned-dict integrity rejection independently of list support.
    write_json(index,dict(attempts=attempts))

    (tmp_path/'attempt-000'/'metrics.json').write_text('{}')
    import pytest
    with pytest.raises(ValueError,match='Pinned raw'):load(index)


def test_persistent_bank_ticks_memory_and_own_zero_baselines(tmp_path,monkeypatch):
    entries=[]
    def save(name,mode,*,study='E03',force=10.,rms=None):
        raw=tmp_path/name
        params=dict(study=study,mode=mode,input_index=2,truth_index=1,force_N=force,repeat=0,
                    stage='measurement',timing_mode='A' if study=='E03' else 'B',backend='internal')
        workspace=dict(device_bytes=4250952 if mode=='original' else 273796,pinned_host_bytes=71808 if mode=='original' else 4488)
        with RawRecorder(raw) as recorder:
            write_json(raw/'protocol.json',dict(protocol_id='force-persistent-grid-native-v2' if study=='E03' else None,
                protocol='force-persistent-grid-task-v2',case=params,params={'rho':.01}))
            write_json(raw/'provenance.json',dict(machine='gtx1070',source_commit='b'*40,module_sha256='module-v2',
                bank_manifest_sha256='bank-v2' if study=='E03' else None,input_sha256={'t01':'task-v2'}))
            write_json(raw/'metrics.json',dict(parameters=params,outcome='force_bank_completed' if study=='E03' else 'force_task_completed',
                evidence_kind='synthetic_test',execution_error=None,tracking_RMS_error_m=rms,task_success=False,physical_pass=True,
                maximum_sampled_effort_ratio=.8,executed_duration_s=24.,requested_duration_s=24.))
            if study=='E03':
                recorder('force_construction',dict(workspace=workspace,optimizer_wall_s=.1))
                for tick in (-1,*range(20)):
                    inputs=dict(previous_time_s=0.,observed_time_s=.047,previous_state=np.zeros(14),observed_state=np.ones(14),
                        applied_history=[dict(start_s=0.,end_s=.047,control=np.zeros(7))],
                        candidate_warm_seeds=np.ones((16,665))*(mode=='original'))
                    recorder('force_tick',dict(category='excluded_warmup' if tick<0 else 'measured',tick_number=tick,
                        status='complete',selected_index=3,hypotheses=np.zeros((16,6)),predictions=np.zeros((16,14)),inputs=inputs,
                        workspace=workspace,estimated_force_error_N=2.,estimated_torque_error_Nm=0.,
                        prediction_observation=dict(elapsed_interval_s=.047,average_applied_control=np.zeros(7)),
                        timings_s=dict(total=2. if mode=='original' else 1.,candidate_forward=.1,estimator_update=.01),
                        backend_measurements=[dict(native_solve_us=1000.)]))
            else:
                recorder('step_completed',dict(time_s=0.,step_dt_s=.003,sampled_effort_ratio=.8,maximum_applied_abs_torque_Nm=10.))
                recorder('attached_force_direction',dict(world_force_N=np.array([0.,0.,-force]),deviation_from_world_down_deg=0.))
            recorder.manifest()
        entries.append(dict(experiment=study+'-'+mode,attempt=name,status='completed',manifest_sha256=sha(raw/'SHA256SUMS.json')))
    save('bank-original','original');save('bank-adapted','adapted')
    for mode,zero,disturbed in [('no_estimator',.020,.040),('original',.025,.030),('adapted',.026,.029)]:
        save(mode+'-zero',mode,study='E02',force=0.,rms=zero)
        save(mode+'-force',mode,study='E02',rms=disturbed)
    index=tmp_path/'index.json';write_json(index,dict(attempts=entries))
    result=load(index)
    assert len(result['pairs'])==20 and all(row['E01_correspondence'] for row in result['pairs'])
    assert [row['tick_number'] for row in result['pairs']]==list(range(20))
    assert result['pairs'][0]['adapted_requested_device_bytes']==273796
    assert result['ticks'][1]['elapsed_observation_interval_s']==.047
    assert result['ticks'][1]['estimated_force_error_N']==2.
    assert len(result['constructions'])==2 and len(result['execution'])==12
    disturbed={row['mode']:row for row in result['tasks'] if row['force_N']==10.}
    for mode,delta in [('no_estimator',.020),('original',.005),('adapted',.003)]:
        assert np.isclose(disturbed[mode]['RMS_above_own_zero_m'],delta)
        assert disturbed[mode]['task_success'] is False # Preserve existing scientific gates.
    from analysis.force import main
    monkeypatch.setattr(sys,'argv',['force.py','--index',str(index),'--output',str(tmp_path/'derived')])
    assert main()==0
    import csv
    exported=list(csv.DictReader((tmp_path/'derived'/'pairs.csv').open()))
    assert len(exported)==20 and exported[0]['adapted_requested_device_bytes']=='273796'
    summary=json.loads((tmp_path/'derived'/'summary.json').read_text())
    assert summary['E01_pairs']==20 and 'persistent owners' in summary['limitations']
    save('bank-adapted-duplicate','adapted')
    write_json(index,dict(attempts=entries))
    duplicate_result=load(index)
    assert len(duplicate_result['pairs'])==0 and len(duplicate_result['attempts'])==9
    assert duplicate_result['unpaired'][0]['reason']=='missing method or duplicate completed attempts'
    save('original-zero-duplicate','original',study='E02',force=0.,rms=.027)
    write_json(index,dict(attempts=entries))
    ambiguous=next(row for row in load(index)['tasks'] if row['mode']=='original' and row['force_N']==10.)
    assert ambiguous['zero_baseline_attempt'] is None and ambiguous['RMS_above_own_zero_m'] is None
