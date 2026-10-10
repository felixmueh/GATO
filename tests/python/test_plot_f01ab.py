"""Tiny persisted RawRecorder-shaped synthetic fixtures, never measurement."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
P=module('plot_f01','experiments/analysis/plot_f01ab.py')
RAW=module('plot_raw','experiments/campaign/raw.py')


def fixture(root, machine='gtx1070', path='attempt-001'):
    bank=root/'bank.json'
    if not bank.exists():shutil.copyfile(ROOT/'experiments/frequency_campaign/inputs/bank.json',bank)
    definition=json.loads(bank.read_text());episode=definition['variants']['short12']['episodes'][0]['id']
    attempt=root/path;attempt.mkdir();recorder=RAW.RawRecorder(attempt)
    entry=dict(machine=machine,N=16,SQP=1,PCG=100,source_commit='synthetic-source',cohort_id='synthetic-only',
        attempt_id='001',path=path)
    protocol=dict(N=16,SQP=1,PCG=100,variant='short12',timing_mode='A',offline_timing='controller_dt',
        backend='internal',batch_size=1,source_commit='synthetic-source',cohort_id='synthetic-only')
    RAW.write_json(attempt/'protocol.json',protocol)
    RAW.write_json(attempt/'provenance.json',dict(machine=machine,source_commit='synthetic-source',cohort_id='synthetic-only',
        evidence_kind='synthetic_test',bank_sha256=P.sha(bank)))
    RAW.write_json(attempt/'metrics.json',dict(episodes=[dict(episode_id=episode,task_success=False,physical_pass=False,solver_numerical_pass=None,execution_error=None)],execution_error='synthetic partial fixture'))
    def event(category,slot,native=None,wall=None,failed=False):
        record=dict(episode_id=None if category=='process_cuda_warmup' else episode,slot=slot,
            category=category,N=16,caps=dict(SQP=1,PCG=100),timing_mode='A',state=np.zeros((1,14)),
            warm_start=np.zeros((1,329)),reference=np.zeros((1,96)),replay_input_hashes={'synthetic':'test-only'})
        if category=='process_cuda_warmup':record['excluded']=True
        recorder('solve_started',record)
        if native is not None:record['native_solve_time_ms']=native
        if wall is not None:record['wall_solve_time_s']=wall
        if failed:recorder('solve_failed',record)
        else:
            record['output']=np.zeros((1,329));recorder('solve_returned',record)
            record['solver_stats']={'sqp_iters':np.array([1])};recorder('solve_completed',record)
    event('process_cuda_warmup',None,100.,.1)
    event('cold',0,1.,.002)
    event('warm',1,3.,.006)
    event('goal_switch',10,2.,.004)
    event('warm',2,np.nan,np.inf,failed=True)
    event('warm',3,None,.008,failed=True)
    recorder.manifest();entry['manifest_sha256']=P.sha(attempt/'SHA256SUMS.json')
    return entry


def selection(root,entries):
    path=root/'selection.json';RAW.write_json(path,dict(evidence_kind='synthetic_test',bank=dict(path='bank.json',sha256=P.sha(root/'bank.json')),attempts=entries))
    return path,P.sha(path)


def test_count_categories_failures_negative_outcomes_and_reciprocals(tmp_path):
    entry=fixture(tmp_path);path,digest=selection(tmp_path,[entry])
    loaded=P.load(path,digest,synthetic_test=True);summaries,coverage,matrix=P.summarize(loaded)
    assert len(loaded['samples'])==241  # all scheduled missing slots plus excluded warmup
    rows=loaded['samples'];cold=[r for r in rows if r['category']=='cold' and r['slot']==0][0]
    assert cold['native_ms']==1 and cold['wrapper_ms']==2 and cold['task_success'] is False and cold['physical_pass'] is False
    assert sum(r['call_status']=='completed' for r in rows)==4  # three episode calls + warmup, not twelve stages
    assert sum(r['call_status']=='failed' for r in rows)==2
    warm=[s for s in summaries if s['category']=='warm' and s['scope']=='wrapper_ms'][0]
    assert warm['mean_ms']==7 and warm['reciprocal_mean_ms_Hz']==1000/7
    assert warm['nonfinite']==1
    native=[s for s in summaries if s['category']=='warm' and s['scope']=='native_ms'][0]
    assert native['mean_ms']==3 and native['missing']==220 and native['nonfinite']==1
    switch=[c for c in coverage if c['category']=='goal_switch'][0]
    assert switch['observed']==1 and switch['expected']==6
    assert len(matrix)==192 and sum(r['status']=='missing' for r in matrix)==191
    assert rows[0]['excluded'] is True


def test_fresh_labeled_render_uses_matched_bins_and_exports_all_samples(tmp_path):
    first=fixture(tmp_path);second=fixture(tmp_path,'rtx5060','attempt-5060')
    path,digest=selection(tmp_path,[first,second]);loaded=P.load(path,digest,synthetic_test=True)
    output=tmp_path/'derived-test-only';P.render(loaded,output)
    assert len(list(output.glob('*.png')))==8 and len(list(output.glob('*.pdf')))==8
    assert sum(1 for _ in (output/'all_samples.csv').open())==483
    receipt=json.loads((output/'provenance.json').read_text())
    assert receipt['evidence_kind']=='synthetic_test' and len(receipt['histogram_bins'])==2
    assert receipt['category_colors']['goal_switch']!=receipt['category_colors']['warm']
    with pytest.raises(FileExistsError):P.render(loaded,output)
    with pytest.raises(ValueError,match='immutable'):P.render(loaded,tmp_path/'attempt-001/plots')


@pytest.mark.parametrize('mutation',['raw_bytes','manifest_escape','category'])
def test_integrity_and_category_mutations_rejected_before_render(tmp_path,mutation):
    entry=fixture(tmp_path);attempt=tmp_path/'attempt-001';manifest=json.loads((attempt/'SHA256SUMS.json').read_text())
    if mutation=='raw_bytes':
        p=next((attempt/'raw').glob('*.npz'));p.write_bytes(p.read_bytes()+b'changed')
    elif mutation=='manifest_escape':manifest['../outside.npz']='wrong'
    else:
        events=[json.loads(x) for x in (attempt/'events.jsonl').read_text().splitlines()]
        for event in events:
            if event['record'].get('category')=='goal_switch':event['record']['category']='warm'
        (attempt/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events));manifest['events.jsonl']=P.sha(attempt/'events.jsonl')
    if mutation!='raw_bytes':
        RAW.write_json(attempt/'SHA256SUMS.json',manifest);entry['manifest_sha256']=P.sha(attempt/'SHA256SUMS.json')
    path,digest=selection(tmp_path,[entry]);output=tmp_path/'must-not-render'
    with pytest.raises(ValueError):P.render(P.load(path,digest,synthetic_test=True),output)
    assert not output.exists()


def test_duplicate_source_mode_and_synthetic_label_rejected(tmp_path):
    entry=fixture(tmp_path);path,digest=selection(tmp_path,[entry,entry])
    with pytest.raises(ValueError,match='Duplicate'):P.load(path,digest,synthetic_test=True)
    second=dict(entry,path='another',cohort_id='changed',attempt_id='002');path,digest=selection(tmp_path,[entry,second])
    with pytest.raises(ValueError,match='cohort'):P.load(path,digest,synthetic_test=True)
    path,digest=selection(tmp_path,[entry])
    with pytest.raises(ValueError,match='Synthetic'):P.load(path,digest)
    attempt=tmp_path/'attempt-001';protocol=json.loads((attempt/'protocol.json').read_text());protocol['timing_mode']='B'
    RAW.write_json(attempt/'protocol.json',protocol);manifest=json.loads((attempt/'SHA256SUMS.json').read_text());manifest['protocol.json']=P.sha(attempt/'protocol.json')
    RAW.write_json(attempt/'SHA256SUMS.json',manifest);entry['manifest_sha256']=P.sha(attempt/'SHA256SUMS.json');path,digest=selection(tmp_path,[entry])
    with pytest.raises(ValueError,match='mode'):P.load(path,digest,synthetic_test=True)
