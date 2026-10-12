"""Saved-only force screen and revised-plan checks; no GPU/process launches."""
import hashlib
import json
from pathlib import Path
import pytest
import force_screen as S
import force_successor_plan as P

def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
def smoke(root,crash=False):
    p=root/'smoke';c=root/'canonical'
    put(c/'metrics.json',dict(task_success=True,physical_pass=True))
    put(p/'smoke_receipt.json',dict(status='completed',canonical_output=str(c)))
    put(p/'metrics.json',dict(outcome='force_task_completed',parameters=dict(mode='original',force_N=0),
        execution_error='native crash' if crash else None,physical_pass=True,task_success=False,
        tracking_RMS_error_m=.026,tracking_max_error_m=.045,executed_duration_s=24.))
    return p

def test_screen_selects_smallest_physical_qualifier_and_retains_failure(tmp_path):
    original=smoke(tmp_path)
    for f,rms,physical,status in [(0,.025,True,'completed'),(5,.036,False,'completed'),(10,None,None,'failed'),(20,.04,True,'completed')]:
        p=tmp_path/'screen'/str(f)
        params=dict(mode='no_estimator',force_N=f,stage='screen')
        put(p/'receipt.json',dict(status=status,case=dict(parameters=params)))
        put(p/'metrics.json',dict(parameters=params,tracking_RMS_error_m=rms,physical_pass=physical,
            task_success=False,maximum_sampled_effort_ratio=.8,executed_duration_s=24.,execution_error='retained error' if status=='failed' else None))
        put(p/'provenance.json',dict(source_commit='same-source',module_sha256='same-module',input_sha256={'t01.json':'same-settings-and-reference'}))
    d=S.select(tmp_path/'screen',original,tmp_path/'selection.json')
    assert d['selected_force_N']==20 and len(d['all_attempts'])==4
    assert d['all_attempts'][1]['execution_error']=='retained error' # Sorted paths0,10,20,5.
    assert d['original_smoke']['task_success'] is False # Old tracking gate is not smoke gate.
    with pytest.raises(ValueError,match='crashed'):
        S.smoke_evidence(smoke(tmp_path/'crashed',True))
    partial=tmp_path/'screen/20/metrics.json';d=json.loads(partial.read_text());d['executed_duration_s']=12.;put(partial,d)
    assert S.select(tmp_path/'screen',original,tmp_path/'partial-selection.json')['selected_force_N'] is None
    baseline=tmp_path/'screen/0/provenance.json';d=json.loads(baseline.read_text());d['source_commit']=S.PRE_RECORDING_SOURCE;put(baseline,d)
    for f in (5,10,20):
        p=tmp_path/f'screen/{f}/provenance.json';d=json.loads(p.read_text());d['source_commit']='a'*40;put(p,d)
    with pytest.raises(ValueError,match='settings differ'):
        S.select(tmp_path/'screen',original,tmp_path/'strict-selection.json')
    compatible=S.select(tmp_path/'screen',original,tmp_path/'recording-selection.json',recording_only_source='a'*40)
    assert compatible['all_attempts'][2]['reviewed_recording_only_source'] is True
    p=tmp_path/'screen/20/provenance.json';d=json.loads(p.read_text());d['module_sha256']='different-module';put(p,d)
    with pytest.raises(ValueError,match='settings differ'):
        S.select(tmp_path/'screen',original,tmp_path/'different-module.json',recording_only_source='a'*40)
    put(tmp_path/'screen/retained/attempt-002/receipt.json',dict(status='failed',case=dict(parameters=dict(stage='screen',mode='no_estimator',force_N=20))))
    with pytest.raises(ValueError,match='exactly one retained'):
        S.select(tmp_path/'screen',original,tmp_path/'duplicate-attempt.json',recording_only_source='a'*40)

def test_speed_only_waiver_scoped_and_adapter_preserves_raw(tmp_path,monkeypatch):
    from adapters import force as A
    m=dict(campaign_id='E02',timing_mode='B',execution_backend='internal',parameters=dict(mode='original'),
        task_success=True,physical_pass=False,execution_error=None,outcome='force_task_completed',
        physical_limit_excess=dict(q_rad=0.,v_rad_s=.465362,u_Nm=0.))
    assert not S.physical_eligibility(m) and S.physical_eligibility(m,True)
    for changed in [dict(execution_backend='robot'),dict(physical_limit_excess=dict(q_rad=.001,v_rad_s=.4,u_Nm=0.)),dict(physical_limit_excess=dict(q_rad=0.,v_rad_s=.4,u_Nm=.01))]:
        assert not S.physical_eligibility(dict(m,**changed),True)
    source=tmp_path/'source';worker=source/'experiments/force_adapter/run_task.py';worker.parent.mkdir(parents=True);worker.write_text('# synthetic worker fixture\n')
    monkeypatch.setattr(A.subprocess,'check_output',lambda *a,**kw:'synthetic-source')
    def execute(command,**kwargs):
        path=Path(command[command.index('--output')+1])/'metrics.json';put(path,m)
        put(path.parent/'SHA256SUMS.json',{'metrics.json':hashlib.sha256(path.read_bytes()).hexdigest()})
        return type('Result',(),{'returncode':1})()
    output=tmp_path/'attempt'
    assert A.run(source,tmp_path,tmp_path,output,preflight=True,mode='original',execute=execute,allow_speed_only=True)==0
    assert json.loads((output/'metrics.json').read_text())==m
    assert json.loads((output/'eligibility.json').read_text())['physical_pass'] is False
    hashes=json.loads((output/'SHA256SUMS.json').read_text())
    assert hashes['metrics.json']==hashlib.sha256((output/'metrics.json').read_bytes()).hexdigest()

def test_final_shapes_and_selected_bank_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'ROOT',tmp_path)
    monkeypatch.setattr(P,'task_inputs',lambda source:{})
    put(tmp_path/'experiments/successor/inventory.json',dict(plan_files=['successor/gtx1070_internal.json']))
    selection=tmp_path/'selection.json';put(selection,dict(selected_force_N=10,outcome='selected_smallest_qualifying_force'))
    bank=tmp_path/'bank';bank.mkdir();put(bank/'protocol.json',dict(protocol_id='force-persistent-grid-native-v2',selected_force_N=10))
    (bank/'arrays.npz').write_bytes(b'synthetic-bank')
    put(bank/'SHA256SUMS.json',{name:hashlib.sha256((bank/name).read_bytes()).hexdigest() for name in ('arrays.npz','protocol.json')})
    files=P.prepare_final('a'*40,selection,bank)
    assert len(files)==3
    for filename in files:
        d=json.loads((tmp_path/'experiments'/filename).read_text());native,*tasks=d['experiments']
        assert len(native['cases'])==24 and sum(len(e['cases']) for e in tasks)==12
        assert native['preflight']['command']==['{build}/cuda_access_smoke']
        for e in tasks:
            assert len(e['cases'])==4 and '--mode' in e['preflight']['command']
            assert [c['parameters']['force_N'] for c in e['cases']]==[0,10,10,10]
            assert [c['parameters']['repeat'] for c in e['cases']]==[0,0,1,2]
    put(selection,dict(selected_force_N=None,outcome='no_qualifying_force_report_before_expansion'))
    with pytest.raises(ValueError,match='No bounded'):P.prepare_final('a'*40,selection,bank)


def test_reviewed_force_canonical_reuse_keeps_failure_and_module(tmp_path,monkeypatch):
    from adapters import force as A
    from recording import sha,write_json
    monkeypatch.setenv('GATO_MACHINE','gtx1070')
    current='61a8135f6b986ef1deca1de562ac184936698133';approved='bbbb047c9827beedff15b40d42aacc6389fa06ba'
    monkeypatch.setattr(A.subprocess,'check_output',lambda *a,**k:current)
    prior=tmp_path/'prior';prior.mkdir();failed=tmp_path/'failed';failed.mkdir()
    source=tmp_path/'source';worker=source/'experiments/force_adapter/run_task.py';worker.parent.mkdir(parents=True);worker.write_text('# worker')
    inputs=tmp_path/'inputs';inputs.mkdir();(inputs/'v01.json').write_text('{}')
    build=tmp_path/'build';(build/'modules').mkdir(parents=True);module=build/'modules/bsqpN32_tiago_right.test.so';module.write_bytes(b'current campaign module')
    params=dict(machine='gtx1070',mode='original',force_N=0,stage='canonical',timing_mode='B',backend='internal')
    m=dict(campaign_id='E02',timing_mode='B',execution_backend='internal',parameters=params,task_success=True,physical_pass=False,execution_error=None,outcome='force_task_completed',physical_limit_excess=dict(q_rad=0.,v_rad_s=.465,u_Nm=0.))
    write_json(prior/'metrics.json',m);write_json(prior/'receipt.json',dict(case=dict(parameters=params)))
    write_json(prior/'provenance.json',dict(source_commit=approved,module_sha256='approved module',input_sha256={'v01.json':sha(inputs/'v01.json')}))
    write_json(failed/'metrics.json',dict(execution_error='Nonfinite plant/control retained'))
    comparison=tmp_path/'comparison.json';write_json(comparison,dict(approved=dict(module_sha256='approved module'),campaign=dict(module_sha256=sha(module)),comparison=dict(force_B1_B16_function_code_matches=True)))
    assessment=tmp_path/'assessment.json';write_json(assessment,dict(scope='reviewed test fixture'))
    metadata=tmp_path/'reuse.json';write_json(metadata,dict(approved_canonical=str(prior),approved_source=approved,runtime_source=current,evidence_sha256={f:sha(prior/f) for f in ['receipt.json','metrics.json','provenance.json']},native_comparison=str(comparison),native_comparison_sha256=sha(comparison),reuse_assessment=str(assessment),reuse_assessment_sha256=sha(assessment),later_failed_canonical=str(failed),later_failed_metrics_sha256=sha(failed/'metrics.json')))
    original={p.name:p.read_bytes() for p in prior.iterdir()}
    def forbidden(*a,**kw):raise AssertionError('Repeated canonical execution')
    output=tmp_path/'new'
    assert A.run(source,build,inputs,output,preflight=True,mode='original',execute=forbidden,allow_speed_only=True,reuse_canonical_metadata=metadata)==0
    metrics=json.loads((output/'metrics.json').read_text());assert metrics['physical_pass'] is False and metrics['later_failed_canonical']==str(failed)
    assert metrics['actual_module_sha256']==sha(module) and metrics['reused_evidence_source_commit']==approved
    assert original=={p.name:p.read_bytes() for p in prior.iterdir()}
    module.write_bytes(b'changed module')
    assert A.run(source,build,inputs,tmp_path/'bad-module',preflight=True,mode='original',execute=forbidden,allow_speed_only=True,reuse_canonical_metadata=metadata)==1
    monkeypatch.setenv('GATO_MACHINE','jetson')
    assert A.run(source,build,inputs,tmp_path/'wrong-machine',preflight=True,mode='original',execute=forbidden,allow_speed_only=True,reuse_canonical_metadata=metadata)==1
