"""Small real-collector saved-data smoke; retries/modes and setup failure remain."""
import json
from recording import write_json
from analysis.batching_tasks import analyze


def test_terminal_attempts_keep_failed_retry_and_mode(tmp_path):
    raw=tmp_path/'raw';output=tmp_path/'derived'
    for i,(study,mode,status,task) in enumerate([('BB','B','failed',None),('BB','B','completed',False),('BA','A','blocked',None),('BB','B','running',None)]):
        path=raw/f'gtx1070/{study}/pair1/attempt-{i:03d}';path.mkdir(parents=True)
        write_json(path/'receipt.json',dict(experiment=study,machine='gtx1070',status=status,protocol='frozen',timing_mode=mode,backend='internal',
            identity=dict(source_commit='synthetic-source'),case=dict(id='pair1',parameters=dict(pair_id=1,batch=1,arm='nominal16x' if study=='BA' else 'default')),
            error='saved setup failure' if status=='blocked' else None))
        if status!='blocked':write_json(path/'metrics.json',dict(task_success=task,physical_pass=False,outcome='timeout',campaign_id=study,timing_mode=mode))
    preflight=raw/'gtx1070/_preflight/BA/attempt-001';preflight.mkdir(parents=True)
    write_json(preflight/'receipt.json',dict(experiment='BA',kind='preflight',status='completed',case=dict(id='preflight')))
    write_json(preflight/'metrics.json',dict(task_success=True,physical_pass=True))
    rows=analyze(raw,output)
    assert len(rows)==3 and [r['status'] for r in rows].count('completed')==1
    assert {r['timing_mode'] for r in rows}=={'A','B'}
    assert next(r for r in rows if r['study']=='BA')['execution_error']=='saved setup failure'
    groups=json.loads((output/'summary.json').read_text())['counts']
    assert len(groups)==2 and next(r for r in groups if r['study']=='BB')['attempts']==2
    assert next(r for r in groups if r['study']=='BB')['task_fail']==1
