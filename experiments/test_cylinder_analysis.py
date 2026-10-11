"""Small saved-data smoke: failed attempts, own seed/output and no retry pooling."""
import csv
import json
from pathlib import Path
import numpy as np
from recording import RawRecorder,write_json
from analysis import cylinder


def saved(root,number,batch,status):
    path=root/f'gtx1070/CylinderTasks/easy30-B{batch}/attempt-{number:03d}';path.mkdir(parents=True)
    case=dict(id=f'easy30-B{batch}',parameters=dict(study='X03',batch=batch),repeat=0)
    write_json(path/'receipt.json',dict(experiment='CylinderTasks',machine='gtx1070',status=status,case=case,
        identity=dict(experiment='CylinderTasks',source_commit='synthetic-source',protocol='synthetic-protocol',timing_mode='B',backend='internal')))
    if status=='blocked':return path
    write_json(path/'provenance.json',dict(case=case,source_commit='synthetic-source',module_sha256='synthetic-module',adapter_sha256='synthetic-adapter',science_sha256={'run.py':'synthetic-worker'},inputs_sha256={'scene.json':'synthetic-input'}))
    write_json(path/'metrics.json',dict(campaign_id='X03',evidence_kind='synthetic_test',task_success=status=='completed',physical_pass=False,
        outcome='task_failed' if status=='failed' else 'task_passed',timing_mode='B',execution_backend='internal',execution_error='synthetic interruption' if status=='failed' else None,
        collision_samples_pass=False,min_executed_tool_clearance_m=-.01,wall_solve_times_s=[.01],native_solve_times_ms=[2.]))
    x=np.zeros((2,14));times=np.array([0,.03]);np.savez(path/'executed.npz',states=x,controls=np.zeros((1,7)),times=times)
    seeds=np.zeros((batch,665));outputs=seeds.copy()
    if number==2:outputs[:,1]=.1
    with RawRecorder(path) as recorder:
        recorder('initial_seeds',dict(scene=dict(goal_xyz=[1,0,0]),candidates=seeds))
        recorder('solve_input',dict(step=0,candidates=seeds))
        recorder('solve_returned',dict(step=0,optimized=outputs))
        recorder('solve_stats',dict(step=0,stats__initial_merit=np.ones(batch)*10,stats__final_merit=np.ones(batch)*8,stats__step_size=np.ones((1,batch))))
        recorder('executed_clearance',dict(tool_xyz=np.array([[0,0,0],[.02,0,0]]),clearance_m=np.array([.02,-.01])))
        recorder.manifest()
    return path


def test_saved_attempts_plots_and_seed_output_separation(tmp_path):
    raw=tmp_path/'raw';out=tmp_path/'derived'
    first=saved(raw,1,1,'failed');saved(raw,2,1,'completed');saved(raw,1,8,'blocked')
    original=(first/'events.jsonl').read_bytes()
    assert cylinder.analyze(raw,out)==0
    summary=json.loads((out/'summary.json').read_text())
    assert len(summary['attempts'])==3 and not summary['errors']
    assert {r['attempt_status'] for r in summary['attempts']}=={'failed','completed','blocked'}
    assert all(r['source_commit']=='synthetic-source' and r['protocol_id']=='synthetic-protocol' for r in summary['attempts'])
    with (out/'seed_to_own_output.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==2 and [r['own_output_array_unchanged'] for r in rows]==['True','False']
    assert all(r['seed_origin']=='supplied-route prefix' for r in rows)
    assert len(list(out.glob('attempt-*.png')))==2
    assert len(list(out.glob('comparison-*.png')))==2 # Incompatible missing native evidence remains separate.
    assert (first/'events.jsonl').read_bytes()==original
    with (out/'trajectory.csv').open() as f:trace=list(csv.DictReader(f))
    assert len(trace)==4 and trace[1]['tool_clearance_m']=='-0.01'


def test_native_progress_nonfinite_retained():
    seed=np.zeros(665);output=seed.copy();output[2]=np.nan
    row=cylinder.native_progress(seed,output,{},0)
    assert row['own_output_change_linf'] is None and row['output_finite'] is False
    assert row['own_output_array_unchanged'] is False and row['initial_merit'] is None
