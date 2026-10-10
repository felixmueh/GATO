"""Explicit desktop T01 deployment only; legacy queue never modified.

Preflight/fallback receipt bundles are copied hashed inputs, supplied per machine.
Generation defaults disabled. Enabling requires released protocols plus hashed
independent review and exact required-suite evidence for the reviewed commit.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
from raw import sha, write_json
from t01_protocol import ROOT, INPUTS, PROTOCOL_FILES


def evidence_paths(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            yield value
        else:
            for child in value.values(): yield from evidence_paths(child)
    elif isinstance(value,list):
        for child in value: yield from evidence_paths(child)


def make_plan(commit,cohort='T01',evidence=None,enabled=False,source=ROOT,enable_machines=None):
    if not re.fullmatch('[0-9a-f]{40}',commit) or commit.startswith('7187cf8'):
        raise ValueError('Full reviewed40-character source commit required; legacy queue excluded')
    if cohort not in PROTOCOL_FILES: raise ValueError('Unknown frozen task cohort')
    source=Path(source);protocol_path='experiments/campaign/inputs/'+PROTOCOL_FILES[cohort]
    protocol=json.loads((source/protocol_path).read_text());evidence=evidence or {}
    desktops={'gtx1070','rtx5060'}
    requested=set(enable_machines if enable_machines is not None else desktops if enabled else [])
    if not requested<=desktops or (requested and not enabled):
        raise ValueError('Requested enable machines must be desktop targets with enable-reviewed')
    if enabled:
        if (protocol['release_status']!='released after reviewed clock decision'
            or not requested or not requested<=set(evidence) or 'release' not in evidence
            or not set(evidence)<=desktops|{'release'}):
            raise ValueError('Held: require released protocol and requested desktop preflight receipts plus review/suite evidence')
        release=evidence['release']
        if set(release)!={'review','task_suite'}: raise ValueError('Require hashed review and suite receipts')
        for kind,entry in release.items():
            path=source/entry['path']
            if sha(path)!=entry['sha256']: raise ValueError('Release evidence hash mismatch')
            record=json.loads(path.read_text())
            if record.get('source_commit')!=commit: raise ValueError('Release receipt is not exact reviewed commit')
            if kind=='review' and record.get('verdict')!='approved': raise ValueError('TrueB independent review not approved')
            if kind=='task_suite' and (record.get('command')!='make test-tiago-tasks' or record.get('exit_code')!=0 or record.get('skips',0)!=0):
                raise ValueError('Exact required make suite has not passed without skips')
    inputs=[protocol_path,'experiments/campaign/inputs/v01.json','experiments/campaign/inputs/v01_1ms.json','experiments/campaign/inputs/t01.json','experiments/campaign/mode_b_protocol.json','tests/python/tiago_task_cases/short_reach.json']
    scripts=['experiments/campaign/'+name for name in ('run_t01.py','t01_protocol.py','raw.py','evaluate.py','build_isolated.py','make_task_plan.py')]+['python/bsqp/offline_mpc.py']
    for path in set(inputs+scripts):
        committed=subprocess.check_output(['git','-C',str(source),'show',f'{commit}:{path}'])
        if committed!=(source/path).read_bytes(): raise ValueError('Deployment bytes differ from reviewed commit: '+path)
    experiments=[]
    for machine in ('gtx1070','rtx5060'):
        expected={'fallback'} if cohort=='V01-1ms' else {'preflight','fallback'}
        receipts=evidence.get(machine,{key:None for key in expected})
        if set(receipts)!=expected:
            raise ValueError('Machine evidence keys differ from cohort')
        if machine in requested and (receipts.get('preflight') is None and cohort!='V01-1ms' or receipts.get('fallback') is None and cohort!='T01'):
            raise ValueError('Missing machine preflight/fallback evidence')
        hashed={path:sha(source/path) for path in set(inputs)}
        for item in evidence_paths({'machine':receipts,'release':evidence.get('release')}):
            relative=Path(item['path'])
            if relative.is_absolute() or '..' in relative.parts or sha(source/relative)!=item['sha256']:
                raise ValueError('Evidence must be exact hashed declared input under source')
            hashed[str(relative)]=item['sha256']
        parameters={'protocol':cohort,'offline_timing':'solve_time','fallback':receipts.get('fallback')}
        if cohort!='V01-1ms': parameters['preflight']=receipts.get('preflight')
        cases=[{'id':t['id'],'parameters':parameters} for t in protocol['tasks']] if cohort!='V01-1ms' else [{'id':'rough-reaching-1ms','parameters':parameters}]
        experiments.append({'id':f'campaign-{cohort}-internal-B-{machine}','source_commit':commit,'enabled':machine in requested,'machines':[machine],'repeats':1,'inputs':hashed,
            'build':{'commands':[['{python}','{source}/experiments/campaign/build_isolated.py','--source','{source}','--build','{build}','--python','{python}','--cuda-arch','{cuda_arch}']], 'artifacts':['campaign-source/python/bsqp/bsqpN32_tiago_right*.so','campaign-native/cuda_access_smoke','build_receipt.json','build_source_receipt.json']},
            'command':['{python}','{build}/campaign-source/experiments/campaign/run_t01.py','--case','{case}','--output','{output}','--inputs','{inputs}'],'cases':cases})
    return {'version':1,'experiments':experiments,'release_notes':{'fallback':'1ms requires all-six completed3ms scientific evidence or failed3ms preflight plus A diagnostic; errors hold, no automatic fallback','desktop_matching':'same six frozen tasks1070/5060 internal; no Jetson/Gazebo/robot','execution_hold':'disabled by default; exact trueB review, make suite and hashed same-source/module preflight required','tracking_gates':'provisional; Felix reviews simulation before D01/D02'}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit',required=True);parser.add_argument('--cohort',choices=PROTOCOL_FILES,default='T01');parser.add_argument('--evidence',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--enable-reviewed',action='store_true');parser.add_argument('--enable-machine',choices=['gtx1070','rtx5060'],action='append')
    args=parser.parse_args();write_json(args.output,make_plan(args.source_commit,args.cohort,json.loads(args.evidence.read_text()) if args.evidence else None,args.enable_reviewed,enable_machines=args.enable_machine))
