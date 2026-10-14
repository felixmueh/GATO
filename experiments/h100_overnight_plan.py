#!/usr/bin/env python3
"""Select unchanged batching/timing definitions for one H100 queue; never run."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE/'successor/merged/gtx1070_desktop.json'
RESOURCE_BASE = HERE/'launch/rtx5060.json'
OUTPUT = HERE/'h100_overnight.json'


def prepare():
    original = {e['id']:e for e in json.loads(BASE.read_text())['experiments']}
    resource = next(e for e in json.loads(RESOURCE_BASE.read_text())['experiments']
                    if e['id']=='F04-stageOFF')
    entries = []
    for name in ['F01ab','F01c','F05-stageOFF','F05-stageON','F04-stageOFF','BB','BA']:
        entry = copy.deepcopy(resource if name=='F04-stageOFF' else original[name])
        entry['machines'] = ['h100']
        if 'prepared_machines' in entry:entry['prepared_machines'] = ['h100']
        # User waived failed-task qualification gates for this new GPU queue.
        # Timing cells already have their original warmups/native observations.
        entry.pop('preflight', None)
        entry['h100_readiness_policy'] = 'No new three-task suite or task-success gate; existing native observations/warmups retained'
        if name in ('BA','BB'):
            entry['preflight'] = dict(group=f'{name}-H100-execution-only-20261014',
                command=['{python}','{runner}/h100_batch_preflight.py',
                    '--source','{source}','--build','{build}',
                    '--inputs','{inputs}/experiments/inputs/batching',
                    '--output','{output}','--study',name])
            entry['h100_readiness_policy'] = 'Original canonical arms; only execution/setup/nonfinite failures hold family. Task and physical negatives stay recorded and continue.'
        if name=='F05-stageON':
            entry['cases'] = [case for case in entry['cases']
                              if case['parameters']['SQP']!=5]
        if name=='F05-stageOFF':
            # All OFF science cells and the unchanged canonical use N32 only.
            entry['build']['commands'][0] = [
                '-DKNOTS=32' if arg=='-DKNOTS=16;32;64;128' else arg
                for arg in entry['build']['commands'][0]]
            entry['build']['artifacts'] = ['modules/bsqpN32_tiago_right*.so','cuda_access_smoke']
        entries.append(entry)
    counts = {e['id']:len(e['cases'])*e.get('repeats',1) for e in entries}
    assert counts=={'F01ab':64,'F01c':15,'F05-stageOFF':24,'F05-stageON':4,
                    'F04-stageOFF':8,'BB':200,'BA':300}
    plan = dict(version=1,cohort_id='h100-batching-timing-20261014',experiments=entries,
        selection=dict(counts=counts,total_scientific_workers=sum(counts.values()),
            order='General timing first, then computation-paid BB, then BA',
            original_plan=str(BASE.relative_to(HERE)),
            original_plan_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
            resource_plan=str(RESOURCE_BASE.relative_to(HERE)),
            resource_plan_sha256=hashlib.sha256(RESOURCE_BASE.read_bytes()).hexdigest(),
            preserved='Scientific cases, input bytes, budgets, numerical source pins and original canonical-arm settings',
            readiness_waiver='Explicit user instruction: earlier failed-task preflights must not block other machines. No new frozen three-task/canonical task-success gate; BA/BB retain execution-only canonical arms, with actual task/physical labels unchanged.',
            execution_helper_sha256=hashlib.sha256((HERE/'h100_batch_preflight.py').read_bytes()).hexdigest(),
            build_change='F05 OFF builds only its used N32; numerical module flags unchanged. Other build recipes retained; h100 profile selects90-real.',
            reused_completed='H100 N32/N128 B1/B16 SQP5 stage profiles and September23 figure-eight remain in their original result roots; excluded here, never rerun or relabeled.',
            excluded='Task parameter grids, random/control/force/cylinder/Gazebo/robot studies, BA seeds-alone, B02',
            failures='Independent case failures retained and independent work continues; shared preflight/setup failures hold dependent cases. No automatic retries.',
            timing_scope='ModeA timing/BA and modeB BB retain their original clocks; actual work is descriptive, no KKT acceptance gate.'))
    OUTPUT.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    print(json.dumps(counts),sum(counts.values()),'workers',OUTPUT)
    return plan


if __name__=='__main__':prepare()
