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
    if os.environ.get('GATO_MACHINE')!='gtx1070':raise ValueError('Force canonical evidence reuse is GTX1070 only')
    d=json.loads(Path(metadata).read_text());prior=Path(d['approved_canonical'])
    for relative,expected in d['evidence_sha256'].items():
        if sha(prior/relative)!=expected:raise ValueError('Approved canonical evidence changed: '+relative)
    assessment=Path(d['reuse_assessment'])
    if sha(assessment)!=d['reuse_assessment_sha256']:raise ValueError('Approved reuse assessment changed')
    comparison=Path(d['native_comparison'])
    if sha(comparison)!=d['native_comparison_sha256']:raise ValueError('Reviewed native comparison changed')
    c=json.loads(comparison.read_text());provenance=json.loads((prior/'provenance.json').read_text())
    receipt=json.loads((prior/'receipt.json').read_text());metrics=json.loads((prior/'metrics.json').read_text())
    current=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    modules=list(Path(build).joinpath('modules').glob('bsqpN32_tiago_right*.so'))
    if (current!=d['runtime_source'] or provenance['source_commit']!=d['approved_source']
        or d['approved_source']!='bbbb047c9827beedff15b40d42aacc6389fa06ba'
        or current!='61a8135f6b986ef1deca1de562ac184936698133'
        or len(modules)!=1 or sha(modules[0])!=c['campaign']['module_sha256']
        or provenance['module_sha256']!=c['approved']['module_sha256']
        or c['comparison'].get('force_B1_B16_function_code_matches') is not True):
        raise ValueError('Reviewed force source/module compatibility does not match')
    for relative,expected in provenance['input_sha256'].items():
        if sha(Path(inputs)/relative)!=expected:raise ValueError('Approved canonical input differs: '+relative)
    p=receipt['case']['parameters']
    if (p.get('machine')!='gtx1070' or p.get('mode')!='original' or p.get('force_N')!=0
        or p.get('stage')!='canonical' or p.get('timing_mode')!='B' or p.get('backend')!='internal'
        or metrics.get('task_success') is not True or metrics.get('execution_error')
        or metrics.get('outcome')!='force_task_completed' or not physical_eligibility(metrics,True)):
        raise ValueError('Approved canonical is not eligible under scoped speed waiver')
    failed=Path(d['later_failed_canonical'])
    if sha(failed/'metrics.json')!=d['later_failed_metrics_sha256']:raise ValueError('Retained later failure changed')
    metrics=dict(metrics,reused_canonical_evidence=str(prior),later_failed_canonical=str(failed),
        reused_evidence_source_commit=provenance['source_commit'],actual_runtime_source_commit=current,actual_module_sha256=sha(modules[0]),
        reuse_scope='Approved prior evidence; later nonfinite failure preserved, not resolved; current campaign module retained')
    write_json(output/'metrics.json',metrics)
    write_json(output/'eligibility.json',dict(eligibility_pass=True,speed_only_waiver=True,physical_pass=metrics['physical_pass'],
        physical_limit_excess=metrics['physical_limit_excess'],scope='Reviewed internal-force GTX1070 canonical evidence reuse only'))
    write_json(output/'reuse.json',dict(metadata_sha256=sha(metadata),approved_evidence=d,
        actual_runtime_source=current,actual_module_sha256=sha(modules[0]),no_new_canonical_execution=True))
    write_json(output/'SHA256SUMS.json',{name:sha(output/name) for name in ('metrics.json','eligibility.json','reuse.json')})
    return 0

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
