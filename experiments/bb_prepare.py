#!/usr/bin/env python3
"""BB-only module reuse and one sequential B1/B16 family preflight."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from recording import sha, write_json


def copy_module(module, expected, build):
    module, build=Path(module),Path(build)
    if sha(module)!=expected:raise ValueError('Retained BB native module differs')
    target=build/'modules'/module.name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(module,target)
    return target


def preflight(source,build,inputs,output,reuse=False,mode='B',nominal=False):
    source,build,inputs,output=map(Path,(source,build,inputs,output))
    output.mkdir(parents=True,exist_ok=True)
    modules=list((build/'modules').glob('bsqpN32_tiago_right*.so'))
    if len(modules)!=1:raise ValueError('Expected exactly one BB N32 module')
    metrics=dict(campaign_id='BB-readiness' if mode=='B' else 'BA-readiness',timing_mode=mode,
        execution_backend='internal',task_success=None,physical_pass=None,reused=reuse,arms=[1,16],
        nominal16x_preflight=nominal)
    def failed(code,arm):
        write_json(output/'metrics.json',dict(metrics,execution_error=f'{arm} preflight exited {code}',
            failed_arm=arm,returncode=code,task_success=None,physical_pass=None))
        return code
    if reuse:
        evidence=json.loads((inputs/'gtx1070_readiness.json').read_text())
        compatibility=json.loads((inputs/'source_reuse.json').read_text())
        for relative,expected in compatibility['unchanged_runtime_sha256'].items():
            if sha(source/relative)!=expected:raise ValueError('BB retained readiness runtime differs: '+relative)
        if (sha(modules[0])!=evidence['native_module']['sha256'] or
            not evidence['required_task_suite']['passed'] or not evidence['canonical_preflight']['passed'] or
            [a['batch_size'] for a in evidence['canonical_preflight']['arms']]!=[1,16] or
            not all(a['task_success'] and a['physical_pass'] for a in evidence['canonical_preflight']['arms'])):
            raise ValueError('Retained BB readiness/module does not match')
        original=evidence['canonical_preflight']
        if mode=='A':
            original=json.loads((inputs/'gtx1070_A_preflight.json').read_text())
            if original['returncode']!=0 or not all(s['task_success'] and s['physical']['physical_valid'] for s in original['summaries'].values()):
                raise ValueError('Retained canonical A preflight failed')
        metrics.update(machine='gtx1070',
            evidence=str(inputs/'gtx1070_readiness.json'),evidence_sha256=sha(inputs/'gtx1070_readiness.json'),
            module_sha256=sha(modules[0]),original_preflight=original,
            source_reuse=compatibility,
            limitation='Recording-only source hooks; unchanged native/controller/timing evidence reused; canonical reuse adds no GPU calls')
        write_json(output/'reused_canonical.json',dict(metrics,task_success=True,physical_pass=True))
        if not nominal:
            write_json(output/'metrics.json',dict(metrics,task_success=True,physical_pass=True))
            return 0
    for batch in (() if reuse else (1,16)):
        command=[sys.executable,str(Path(__file__).parent/('adapters/bb.py' if mode=='B' else 'adapters/ba.py')),'--source',str(source),'--build',str(build),
            '--pool',str(inputs),'--protocol',str(inputs/'bb_protocol.json'),'--preflight-input',str(inputs/'bb_preflight.json'),
            '--batch',str(batch),'--output',str(output/f'B{batch}')]
        if mode=='A':
            command=[sys.executable,str(Path(__file__).parent/'adapters/ba.py'),'--source',str(source),'--build',str(build),
                '--inputs',str(inputs),'--preflight','--batch',str(batch),'--output',str(output/f'B{batch}')]
        result=subprocess.run(command)
        if result.returncode:return failed(result.returncode,f'default-B{batch}')
    if nominal:
        command=[sys.executable,str(Path(__file__).parent/'adapters/ba.py'),'--source',str(source),'--build',str(build),
            '--inputs',str(inputs),'--preflight','--batch','1','--arm','nominal16x','--output',str(output/'nominal16x-B1')]
        result=subprocess.run(command)
        if result.returncode:return failed(result.returncode,'nominal16x-B1')
    write_json(output/'metrics.json',dict(metrics,task_success=True,physical_pass=True))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    copy=sub.add_parser('copy-module')
    copy.add_argument('--module',type=Path,required=True)
    copy.add_argument('--sha256',required=True)
    copy.add_argument('--build',type=Path,required=True)
    pf=sub.add_parser('preflight')
    for key in ('source','build','inputs','output'):pf.add_argument('--'+key,type=Path,required=True)
    pf.add_argument('--reuse-gtx1070',action='store_true')
    pf.add_argument('--mode',choices=('A','B'),default='B')
    pf.add_argument('--ba-nominal-preflight',action='store_true')
    args=parser.parse_args()
    if args.action=='copy-module':copy_module(args.module,args.sha256,args.build)
    else:sys.exit(preflight(args.source,args.build,args.inputs,args.output,args.reuse_gtx1070,args.mode,args.ba_nominal_preflight))
