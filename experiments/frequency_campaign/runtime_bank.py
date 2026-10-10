"""Check frozen F01 bytes and definitions; historical screening is not recomputed.

The actual machine's canonical task/physical preflight remains the execution gate.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BANK_MANIFEST_SHA256='29f8ccdff73d36c339e50ea7153a46fb6ecd22b32da786b74f16cb9e1a61f609'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(output,root=ROOT):
    output=Path(output)
    if sha(output/'SHA256SUMS.json')!=BANK_MANIFEST_SHA256:raise ValueError('Pinned bank manifest differs')
    spec=importlib.util.spec_from_file_location('frozen_f01_definitions',root/'experiments/frequency_campaign/freeze_inputs.py')
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    TASK,TASK_HASH=generator.TASK,generator.TASK_HASH
    ENDPOINTS,ENDPOINT_HASH=generator.ENDPOINTS,generator.ENDPOINT_HASH
    MODEL,MODEL_HASH=generator.MODEL,generator.MODEL_HASH
    source_inputs=generator.source_inputs
    arrays_and_schedules=generator.arrays_and_schedules
    output=Path(output); task,endpoints=source_inputs(root)
    manifest=json.loads((output/'SHA256SUMS.json').read_text())
    expected={'arrays.npz','bank.json','freeze_inputs.py','model.urdf','provenance.json','screening.json','source_endpoints.json','source_t01.json','witnesses.npz'}
    if set(manifest)!=expected:raise ValueError('Incomplete/failed or extra input artifacts')
    for name,digest in manifest.items():
        if sha(output/name)!=digest:raise ValueError('Bank artifact bytes changed: '+name)
    arrays,variants=arrays_and_schedules(task,endpoints)
    with np.load(output/'arrays.npz',allow_pickle=False) as saved:
        if set(saved.files)!=set(arrays):raise ValueError('Array keys changed')
        for name,a in arrays.items():
            b=saved[name]
            if b.dtype!=a.dtype or b.shape!=a.shape or b.tobytes()!=a.tobytes():raise ValueError('Array definition changed: '+name)
    bank=json.loads((output/'bank.json').read_text())
    if bank['variants']!=variants or bank['reference_sha256']!=[t['reference_sha256'] for t in task['tasks'][3:]]:
        raise ValueError('Episode/reference definition changed')
    fixed=dict(N=[16,32,64,128],SQP=[1,2,5,10],PCG=[100,200,400,800],batch_size=1,
        timing_mode='A',offline_timing='controller_dt',dt_s=.03,sim_dt_s=.003,
        warm_start_shift_s=.03,goal_switch_preserves_warm_start=True,
        presets=dict(reaching=task['reaching_solver_params'],tracking=task['tracking_solver_params']),
        reference_periods_s=[t['period_s'] for t in task['tasks'][3:]],
        warmup=dict(category='process_cuda_warmup',excluded_from_measured_categories=True,
            state_array='warmup_state',goal_array='warmup_goal',standard_initial_guess=True))
    if any(bank.get(key)!=value for key,value in fixed.items()):raise ValueError('Bank configuration changed')
    for name,digest in [('model.urdf',MODEL_HASH),('source_t01.json',TASK_HASH),
                        ('source_endpoints.json',ENDPOINT_HASH),('freeze_inputs.py',sha(root/'experiments/frequency_campaign/freeze_inputs.py'))]:
        if sha(output/name)!=digest:raise ValueError('Frozen source/model/generator changed: '+name)
    report=json.loads((output/'screening.json').read_text())
    expected_routes={f'start_{s}_goal_{g}' for s in range(4) for g in range(3)}|{f'switch_goal_{g}_to_{(g+1)%3}' for g in range(3)}
    if (report.get('all_routes_pass') is not True or report.get('all_starts_outside5cm') is not True
        or report.get('all_start_q_qd_limits_pass') is not True or {r['id'] for r in report['routes']}!=expected_routes
        or not all(r.get('positive_sampled_margins') is True for r in report['routes'])):
        raise ValueError('Historical saved screening incomplete/negative')
    with np.load(output/'witnesses.npz',allow_pickle=False) as saved:
        expected={name+'_'+suffix for name in expected_routes for suffix in ['q','v','a','u']}|{'time_s'}
        if set(saved.files)!=expected:raise ValueError('Saved witness metadata keys differ')
        for name in saved.files:
            value=saved[name]
            shape=(report['sample_count'],) if name=='time_s' else (report['sample_count'],7)
            if value.shape!=shape or value.dtype!=np.dtype('float64'):raise ValueError('Saved witness metadata shape/dtype differs: '+name)
    provenance=json.loads((output/'provenance.json').read_text())
    if provenance.get('generator_sha256')!=sha(root/'experiments/frequency_campaign/freeze_inputs.py') or provenance.get('source_sha256')!={TASK:TASK_HASH,ENDPOINTS:ENDPOINT_HASH,MODEL:MODEL_HASH}:
        raise ValueError('Provenance source identity changed')
    return bank
