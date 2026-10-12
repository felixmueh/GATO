#!/usr/bin/env python3
"""Reuse the existing frozen suite, then run a family's canonical command."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.control_studies import ensure_suite
from recording import write_json,sha
from adapters import task

def reuse_canonical(source,build,output,command,group):
    if not group or Path(group).name!=group: return None
    source=Path(source);build=Path(build)
    pin=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    modules=list((build/'modules').glob('bsqpN32_tiago_right*.so'))
    if len(modules)!=1:return None
    inputs=Path(command[command.index('--inputs')+1])
    mode=command[command.index('--timing-mode')+1]
    expected=task.select_protocol(source,{'id':'preflight'},mode,True,inputs)
    for receipt in sorted((output.parent.parent/group).glob('attempt-*/receipt.json'),reverse=True):
        prior=receipt.parent
        try:
            r=json.loads(receipt.read_text());m=json.loads((prior/'metrics.json').read_text())
            provenance=json.loads((prior/'canonical/provenance.json').read_text())
            protocol=json.loads((prior/'canonical/protocol.json').read_text())
            if (r.get('status')!='completed' or r.get('source_commit')!=pin
                or m.get('task_success') is not True or m.get('physical_pass') is not True
                or m.get('execution_error') or provenance.get('source_commit')!=pin
                or provenance.get('module_sha256')!=sha(modules[0])
                or provenance.get('collector_sha256')!=sha(task.__file__)
                or provenance.get('protocol_sha256')!=sha(prior/'canonical/protocol.json')
                or protocol!=expected):continue
            return prior,m
        except (OSError,ValueError,KeyError):continue
    return None

def run(source,build,output,command,reuse_canonical_group=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reused=reuse_canonical(source,build,output,command,reuse_canonical_group) if reuse_canonical_group else None
    if reused:
        prior,metrics=reused;metrics=dict(metrics,reused_preflight_evidence=str(prior))
        write_json(output/'metrics.json',metrics);write_json(output/'reuse.json',dict(evidence=str(prior),receipt_sha256=sha(prior/'receipt.json'),scope='Same source/module/collector/canonical settings passed suite and canonical'))
        return 0
    code=ensure_suite(Path(source),Path(build),output/'task_suite')
    if code:
        write_json(output/'metrics.json',dict(task_success=None,physical_pass=None,execution_error='Changed-module frozen task suite failed'))
        return code
    canonical=output/'canonical'
    command=[str(canonical) if v=='@canonical-output@' else v for v in command]
    code=subprocess.run(command).returncode
    path=canonical/'metrics.json'
    metrics=json.loads(path.read_text()) if path.exists() else dict(task_success=None,physical_pass=None,execution_error='Canonical command produced no metrics')
    metrics.update(suite_evidence=str(output/'task_suite'),canonical_evidence=str(canonical))
    if code and not metrics.get('execution_error'):metrics['execution_error']=f'Canonical preflight exited {code}'
    write_json(output/'metrics.json',metrics)
    return code or int(metrics.get('task_success') is not True or metrics.get('physical_pass') is not True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','build','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--reuse-canonical-group')
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('Canonical command required after --')
    raise SystemExit(run(a.source,a.build,a.output,command,a.reuse_canonical_group))
