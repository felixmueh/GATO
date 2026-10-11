#!/usr/bin/env python3
"""Reuse the existing frozen suite, then run a family's canonical command."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.control_studies import ensure_suite
from recording import write_json

def run(source,build,output,command):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
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
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('Canonical command required after --')
    raise SystemExit(run(a.source,a.build,a.output,command))
