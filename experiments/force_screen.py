#!/usr/bin/env python3
"""Select the frozen bounded force screen from terminal saved attempts only."""
import argparse
import json
import math
import os
import re
from pathlib import Path
from recording import sha,write_json

DEGRADATION_M=.010
FORCES=(5.,10.,20.)
PRE_RECORDING_SOURCE='bbbb047c9827beedff15b40d42aacc6389fa06ba'

def receipts(root):
    for directory,folders,files in os.walk(root):
        # Recorder raw directories contain NPZ chunks, not attempt receipts.
        folders[:]=sorted(name for name in folders if name!='raw')
        if 'receipt.json' in files:yield Path(directory)/'receipt.json'

def physical_eligibility(metrics,allow_speed_only=False):
    """Eligibility only; never change the source physical result."""
    if not allow_speed_only:return metrics.get('physical_pass') is True
    p=metrics.get('parameters',{});limits=metrics.get('physical_limit_excess',{})
    if (metrics.get('campaign_id')!='E02' or metrics.get('timing_mode')!='B' or
        metrics.get('execution_backend')!='internal' or p.get('mode') not in ('original','adapted','no_estimator')):return False
    if not all(isinstance(limits.get(k),(int,float)) and math.isfinite(limits[k]) for k in ('q_rad','v_rad_s','u_Nm')):return False
    return bool(limits['q_rad']<=1e-4 and limits['u_Nm']<=1e-3)

def smoke_evidence(attempt,allow_speed_only=False):
    attempt=Path(attempt).resolve();metrics=json.loads((attempt/'metrics.json').read_text())
    receipt=attempt/'receipt.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved.get('status')!='completed':raise ValueError('Original0N smoke is not terminal completed')
        if saved.get('preflight_path'):
            from adapters.deployment import preflight_path
            canonical=preflight_path(attempt)
        else:
            direct=json.loads((attempt/'smoke_receipt.json').read_text())
            if direct.get('status')!='completed':raise ValueError('Original0N direct smoke is not terminal completed')
            canonical=Path(direct['canonical_output']).resolve()
    else:
        saved=json.loads((attempt/'smoke_receipt.json').read_text())
        if saved.get('status')!='completed':raise ValueError('Original0N smoke is not terminal completed')
        canonical=Path(saved['canonical_output']).resolve()
    gate=json.loads((canonical/'metrics.json').read_text())
    if gate.get('execution_error') or gate.get('task_success') is not True or not physical_eligibility(gate,allow_speed_only):
        raise ValueError('Original0N canonical failed; force study held')
    if metrics.get('execution_error') or metrics.get('outcome')!='force_task_completed':raise ValueError('Original0N smoke crashed/incomplete; hold force study and notify root')
    p=metrics.get('parameters',{})
    if p.get('mode')!='original' or p.get('force_N')!=0:raise ValueError('Smoke must be original-arm0N')
    if not physical_eligibility(metrics,allow_speed_only):raise ValueError('Original0N smoke failed non-waived physical eligibility; hold and notify root')
    if not all(isinstance(metrics.get(k),(int,float)) and math.isfinite(metrics[k]) for k in ('tracking_RMS_error_m','tracking_max_error_m')):
        raise ValueError('Original0N smoke metrics nonfinite; hold force study')
    duration=metrics.get('executed_duration_s')
    if not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<24.-1e-6:
        raise ValueError('Original0N smoke did not complete the full24s tracking task')
    return dict(attempt=str(attempt),metrics_sha256=sha(attempt/'metrics.json'),canonical_output=str(canonical),
        canonical_metrics_sha256=sha(canonical/'metrics.json'),task_success=metrics.get('task_success'),physical_pass=metrics.get('physical_pass'),
        speed_only_waiver=allow_speed_only,physical_limit_excess=metrics.get('physical_limit_excess'),eligibility_pass=True,
        tracking_RMS_error_m=metrics['tracking_RMS_error_m'],note='Execution/finite/physical smoke gate; historical20mm task gate is reported, not a smoke execution gate.')

def select(root,smoke,output,allow_speed_only=False,recording_only_source=None):
    if recording_only_source is not None and not re.fullmatch('[0-9a-f]{40}',recording_only_source):
        raise ValueError('Explicit reviewed recording-only full source commit required')
    evidence=smoke_evidence(smoke,allow_speed_only);rows=[]
    for receipt_path in sorted(receipts(root)):
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('kind')=='preflight':continue
        path=receipt_path.parent/'metrics.json'
        metrics=json.loads(path.read_text()) if path.exists() else {}
        p=metrics.get('parameters',receipt.get('case',{}).get('parameters',{}))
        if p.get('stage')!='screen' or p.get('mode')!='no_estimator':continue
        provenance=receipt_path.parent/'provenance.json'
        identity=json.loads(provenance.read_text()) if provenance.exists() else {}
        rows.append(dict(force_N=p.get('force_N'),attempt=str(receipt_path.parent),status=receipt.get('status'),
            execution_error=metrics.get('execution_error') or receipt.get('error'),task_success=metrics.get('task_success'),
            physical_pass=metrics.get('physical_pass'),physical_limit_excess=metrics.get('physical_limit_excess'),
            physical_eligibility=physical_eligibility(metrics,allow_speed_only),RMS_m=metrics.get('tracking_RMS_error_m'),
            executed_duration_s=metrics.get('executed_duration_s'),
            sampled_effort_ratio=metrics.get('maximum_sampled_effort_ratio'),metrics_sha256=sha(path) if path.exists() else None,
            compatibility={k:identity.get(k) for k in ('source_commit','module_sha256','input_sha256')}))
    by_force={}
    for row in rows:by_force.setdefault(row['force_N'],[]).append(row)
    if any(len(by_force.get(f,[]))!=1 for f in (0.,*FORCES)):
        raise ValueError('Screen needs exactly one retained attempt per0/5/10/20N; do not silently pool or choose retries')
    baseline=by_force[0.][0]
    def finite(v):return isinstance(v,(int,float)) and math.isfinite(v)
    if (baseline['status']!='completed' or baseline['execution_error'] or not baseline['physical_eligibility'] or not finite(baseline['RMS_m']) or
        not finite(baseline['executed_duration_s']) or baseline['executed_duration_s']<24.-1e-6 or
        not finite(baseline['sampled_effort_ratio']) or baseline['sampled_effort_ratio']>1):
        raise ValueError('New0N screen baseline invalid; preserve outcomes and report before changing scope')
    if not all(baseline['compatibility'].values()):raise ValueError('New0N baseline source/module/task-input identity unavailable')
    selected=None
    for force in FORCES:
        row=by_force[force][0]
        row['same_source_module_settings']=row['compatibility']==baseline['compatibility']
        old=baseline['compatibility'];new=row['compatibility']
        row['reviewed_recording_only_source']=bool(recording_only_source and old['source_commit']==PRE_RECORDING_SOURCE and
            new['source_commit']==recording_only_source and old['module_sha256']==new['module_sha256'] and old['input_sha256']==new['input_sha256'])
        row['compatible_scientific_settings']=row['same_source_module_settings'] or row['reviewed_recording_only_source']
        if row['status']=='completed' and not row['execution_error'] and not row['compatible_scientific_settings']:
            raise ValueError('Completed screen source/module/task settings differ; report before F* freeze')
        row['degradation_m']=row['RMS_m']-baseline['RMS_m'] if finite(row['RMS_m']) else None
        row['qualifies']=bool(row['compatible_scientific_settings'] and row['status']=='completed' and not row['execution_error'] and row['physical_eligibility'] and
            finite(row['executed_duration_s']) and row['executed_duration_s']>=24.-1e-6 and
            finite(row['sampled_effort_ratio']) and row['sampled_effort_ratio']<=1 and finite(row['degradation_m']) and row['degradation_m']>=DEGRADATION_M)
        if selected is None and row['qualifies']:selected=force
    result=dict(protocol='force-screen5-10-20-v2',degradation_threshold_m=DEGRADATION_M,
        speed_only_waiver=allow_speed_only,waiver_scope='Internal force simulations only; speed descriptive, unchanged position/torque/finite/effort gates; raw physical_pass retained',
        reviewed_recording_only_source=recording_only_source,recording_compatibility_scope='Only the explicitly reviewed recording-only successor of bbbb047; identical native module and task/settings inputs; original source labels retained',
        original_smoke=evidence,baseline=baseline,all_attempts=rows,selected_force_N=selected,
        outcome='selected_smallest_qualifying_force' if selected else 'no_qualifying_force_report_before_expansion',
        note='No task-gate relabeling; all scientific/physical failures and execution errors remain explicit. No extended screen or tuning.')
    write_json(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--screen-root',type=Path,required=True)
    p.add_argument('--smoke-attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-speed-only',action='store_true')
    p.add_argument('--recording-only-source',help='Exact reviewed recording-only successor allowed against bbbb047 baseline; no other identity relaxation')
    a=p.parse_args()
    print(json.dumps(select(a.screen_root,a.smoke_attempt,a.output,a.allow_speed_only,a.recording_only_source),indent=2))
