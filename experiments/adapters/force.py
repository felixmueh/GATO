#!/usr/bin/env python3
"""Dispatch the revised pinned force workers without modifying their science."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import sha,write_json
from force_screen import physical_eligibility

def reuse_original_canonical(source,build,inputs,output,metadata):
    """Explicit reviewed GTX1070 evidence reuse; never executes another canonical."""
    raise ValueError('Original force canonical reuse is held pending explicit Felix decision; newer nonfinite execution gate remains unresolved')

def run(source,build,inputs,output,case=None,preflight=False,mode=None,execute=subprocess.run,allow_speed_only=False,reuse_canonical_metadata=None):
    source,build,inputs,output=(Path(p).resolve() for p in (source,build,inputs,output))
    output.mkdir(parents=True,exist_ok=True)
    params=case['parameters'] if case else {}
    bank=params.get('study')=='E03' and not preflight
    worker='run_native' if bank else 'run_task'
    command=[sys.executable,'-m','experiments.force_adapter.'+worker,'--build',str(build),
        '--runner',str(Path(__file__).resolve().parents[1]),'--inputs',str(inputs),'--output',str(output)]
    if preflight:command+=['--preflight','--mode',mode,'--force-N','0','--repeat','0']
    else:
        path=output/'worker_case.json';write_json(path,case);command+=['--case',str(path)]
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(
        [str(source/'python'),str(source),str(Path(__file__).resolve().parents[1])]))
    write_json(output/'runner_force_adapter.json',dict(source_commit=subprocess.check_output(
        ['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        adapter_sha256=sha(__file__),worker_sha256=sha(source/'experiments/force_adapter'/f'{worker}.py'),
        command=command,parameters=params,canonical_force_N=0 if preflight else None,
        note='Undisturbed canonical readiness; disturbance task physical outcomes are checked in actual runs.'))
    try:
        if reuse_canonical_metadata:
            if not (preflight and mode=='original' and allow_speed_only):raise ValueError('Canonical evidence reuse is only authorized for original internal-force preflight with explicit waiver')
            return reuse_original_canonical(source,build,inputs,output,reuse_canonical_metadata)
        code=execute(command,cwd=source,env=environment).returncode
        path=output/'metrics.json'
        metrics=json.loads(path.read_text()) if path.exists() else dict(task_success=None,physical_pass=None,
            execution_error=f'Force worker exited {code} without terminal metrics')
        # Source metrics/manifests remain authoritative. A nonzero canonical
        # gate can be a physical failure, not an execution error.
        if not path.exists():
            write_json(path,metrics)
        if allow_speed_only and not bank:
            eligible=bool(code in (0,1) and not metrics.get('execution_error') and metrics.get('outcome')=='force_task_completed'
                and (not preflight or metrics.get('task_success') is True) and physical_eligibility(metrics,True))
            eligibility=output/'eligibility.json'
            write_json(eligibility,dict(eligibility_pass=eligible,speed_only_waiver=True,
                scope='Internal force simulation only; raw physical flags unchanged',worker_returncode=code,
                physical_pass=metrics.get('physical_pass'),physical_limit_excess=metrics.get('physical_limit_excess')))
            manifest=output/'SHA256SUMS.json'
            if manifest.exists():
                hashes=json.loads(manifest.read_text());hashes['eligibility.json']=sha(eligibility);write_json(manifest,hashes)
            if preflight:return 0 if eligible else (code or 1)
        return code or int(bool(metrics.get('execution_error'))) or int(preflight and not(
            metrics.get('task_success') is True and metrics.get('physical_pass') is True))
    except BaseException:
        path=output/'metrics.json';metrics=json.loads(path.read_text()) if path.exists() else {}
        metrics.update(task_success=metrics.get('task_success'),physical_pass=metrics.get('physical_pass'),execution_error=traceback.format_exc())
        write_json(path,metrics);return 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','build','inputs','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--case',type=Path);p.add_argument('--preflight',action='store_true')
    p.add_argument('--mode',choices=('original','adapted','no_estimator'));p.add_argument('--allow-speed-only',action='store_true');p.add_argument('--reuse-canonical-metadata',type=Path);a=p.parse_args()
    if a.preflight and not a.mode:p.error('Preflight mode required')
    if not a.preflight and not a.case:p.error('Case required')
    raise SystemExit(run(a.source,a.build,a.inputs,a.output,json.loads(a.case.read_text()) if a.case else None,a.preflight,a.mode,allow_speed_only=a.allow_speed_only,reuse_canonical_metadata=a.reuse_canonical_metadata))
