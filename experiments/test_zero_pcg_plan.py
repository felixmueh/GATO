"""CPU checks of successor coverage, identities and physically timed references."""
import hashlib
import json
from pathlib import Path
import numpy as np
import package_plans
import zero_pcg_plan as Z
from adapters import task

def test_full_successor_coverage_and_fresh_readiness(tmp_path):
    files=Z.prepare();package_plans.merge(tmp_path,files)
    for machine in ('gtx1070','rtx5060','jetson'):
        d=json.loads((Z.ROOT/'experiments/successor'/f'{machine}_internal.json').read_text())
        e={v['id']:v for v in d['experiments']}
        assert list(e)[:2]==['T01','T01-faster-proposal']
        assert len(e['T01']['cases'])==6 and len(e['F01ab']['cases'])==64 and len(e['F01c']['cases'])==15
        assert len(e['BB']['cases'])==200
        assert not any(k.startswith('F04') or k=='CylinderChecks' or k=='Q02' for k in e)
        if machine!='jetson':assert len(e['BA']['cases'])==300
        else:assert 'BA' not in e
        for v in e.values():
            assert v['source_commit']==Z.PINS[v['original_runtime_source_commit']]
            assert v['stopping_policy']==Z.POLICY
            assert not any('copy-module' in c for c in v['build']['commands'])
            assert '--reuse-gtx1070' not in v['preflight']['command']
        assert Z.ORDINARY in e['CylinderTasks']['command']
        assert len(e['CylinderTasks']['cases'])==(6 if machine=='jetson' else 8)
    assert len(json.loads((tmp_path/'jetson_proposed.json').read_text())['experiments'])>0
    desktop=json.loads((tmp_path/'gtx1070_desktop.json').read_text())
    assert not any(e['id'].startswith('F04') or e['id'] in ('M02','Q02') for e in desktop['experiments'])
    assert not any('robot' in e['backend'].lower() for e in json.loads((tmp_path/'rtx5060_gazebo.json').read_text())['experiments'])

def test_faster_proposal_preserves_geometry_clock_and_old_inputs():
    old=Z.ROOT/'experiments/inputs/t01/t01.json';before=old.read_bytes();Z.faster_inputs()
    folder=Z.ROOT/'experiments/inputs/t01_faster';p=json.loads((folder/'t01.json').read_text())
    original=json.loads(before);base=np.asarray(original['tasks'][3]['reference_packed']).reshape(-1,6)
    assert [v['period_s'] for v in p['tasks']]==[8,6,4]
    assert p['tracking_solver_params']==original['tracking_solver_params'] and old.read_bytes()==before
    for t in p['tasks']:
        selected=task.select_protocol(Z.ROOT,{'id':t['id']},'B',False,folder)
        ref=np.asarray(selected['reference_packed'],dtype='<f8').reshape(-1,6)
        assert hashlib.sha256(ref.tobytes()).hexdigest()==selected['reference_sha256']
        assert (len(ref)-1)*.03>=t['duration_s']+.93
        np.testing.assert_allclose(ref[0],base[0])
        assert selected['provisional_RMS_gate_m']==.025 and selected['provisional_maximum_gate_m']==.05
        assert selected['offline_timing']=='solve_time'

def test_wrapper_retains_failed_canonical_evidence(tmp_path,monkeypatch):
    from adapters import successor_preflight as P
    monkeypatch.setattr(P,'ensure_suite',lambda *a:0)
    def execute(command):
        target=Path(command[-1]);target.mkdir();(target/'metrics.json').write_text(json.dumps(dict(task_success=False,physical_pass=True)))
        return type('Result',(),{'returncode':1})()
    monkeypatch.setattr(P.subprocess,'run',execute)
    assert P.run(tmp_path,tmp_path,tmp_path/'attempt',['canonical','@canonical-output@'])==1
    d=json.loads((tmp_path/'attempt/metrics.json').read_text())
    assert d['task_success'] is False and d['physical_pass'] is True and d['execution_error']
    assert (tmp_path/'attempt/canonical/metrics.json').exists()
