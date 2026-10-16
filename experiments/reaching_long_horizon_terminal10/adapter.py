#!/usr/bin/env python3
"""One N256 task or shared canonical under the unchanged serial runner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

COHORT='reaching-long-horizon-terminal10-20261016'

def canonical_pass(m):
    return bool(m.get('complete') and m.get('preflight_pass') and m.get('task_success') and
                m.get('physical_pass') and not m.get('execution_error') and not m.get('interrupted') and
                m.get('physical',{}).get('finite'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','build','inputs','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--case',type=Path)
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    hold=args.build/'long_horizon_shared_setup_failure.json'
    inputs=args.inputs/'experiments/reaching_long_horizon_terminal10/inputs'
    case_id='canonical' if args.preflight else json.loads(args.case.read_text())['parameters']['case_id']
    base=dict(campaign_id=COHORT,case_id=case_id)
    try:
        if hold.exists():
            raise ValueError('Shared setup failure holds dependent GPU work: '+str(hold))
        modules=list((args.build/'modules').glob('bsqpN256_tiago_right*.so'))
        if len(modules)!=1:
            raise ValueError('Expected one reviewed shared N256 module')
        episode=args.output/'episode'
        argv=[sys.executable,str(args.source/'experiments/reaching_long_horizon_terminal10/run_case.py'),
              '--source',str(args.source),'--module',str(modules[0]),'--tasks',str(inputs/'tasks.json'),
              '--protocol',str(inputs/'protocol.json'),'--output',str(episode)]
        argv += ['--preflight-input',str(inputs/'canonical.json')] if args.preflight else ['--case-id',case_id]
        completed=subprocess.run(argv,check=False)
        metrics_path=episode/('preflight' if args.preflight else case_id)/'metrics.json'
        metrics=json.loads(metrics_path.read_text())
        metrics.update(base,execution_backend='internal',adapter_argv=argv)
        (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
        if metrics.get('shared_setup_failure'):
            hold.write_text(json.dumps(dict(case_id=case_id,metrics_path=str(metrics_path),
                reason=metrics.get('execution_error')),indent=2)+'\n')
        print(json.dumps(dict(case_id=case_id,outcome=metrics.get('outcome'),arrival_s=metrics.get('reached_seconds'),
            physical_pass=metrics.get('physical_pass'),wall_s=metrics.get('episode_wall_s'),
            feedback_solves=metrics.get('feedback_solves')),allow_nan=False),flush=True)
        return (0 if canonical_pass(metrics) else 1) if args.preflight else completed.returncode
    except BaseException:
        metrics=dict(base,execution_error=traceback.format_exc(),task_success=False,physical_pass=False,
                     shared_setup_failure=True,outcome='execution_failed')
        (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        if not hold.exists():
            hold.write_text(json.dumps(metrics,indent=2)+'\n')
        raise

if __name__=='__main__':
    raise SystemExit(main())
