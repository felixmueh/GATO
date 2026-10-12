#!/usr/bin/env python3
"""One unchanged live Gazebo fault trial after explicit simulator prerequisites.

Local simulator/PAL observations and assisted reset are supplied by the host.
No Docker orchestration or robot deployment is inferred by this adapter.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import write_json,sha
from adapters import task_suite

DEFAULTS={'arm_right_controller','arm_right_gravity_compensation_controller'}
LEGACY_CONTRACT='position-plus-gravity-v1'
POSITION_CONTRACT='position-only-v2'
RUNTIME_FILES=('tiago_src/gato_tiago/tiago_mpc_controller.py','tiago_src/gato_tiago/tiago_controller_process.py','tiago_src/gato_tiago/ros_tiago.py')


def prerequisites(config,source,now,control_mode_contract=LEGACY_CONTRACT):
    if control_mode_contract not in (LEGACY_CONTRACT,POSITION_CONTRACT):
        raise ValueError('Unknown fault-trial control-mode contract')
    if config.get('simulation_only') is not True or config.get('domain') not in (1,91):
        raise ValueError('Explicit local Gazebo-only simulator/domain configuration required')
    receipt_path=Path(config['canonical_live_preflight_receipt'])
    receipt=json.loads(receipt_path.read_text())
    p=json.loads((receipt_path.parent/'protocol.json').read_text())
    provenance=json.loads((receipt_path.parent/'provenance.json').read_text())
    if not (receipt['task_success'] is True and receipt['physical_pass'] is True and
            receipt['execution_backend']=='gazebo' and receipt['handback_confirmed'] is True and p['N']==32 and abs(p['dt_s']-.03)<1e-12 and
            abs(p['solver_params']['qd_cost']-.1)<1e-12 and abs(p['solver_params']['vel_lim_cost']-.1)<1e-12):
        raise ValueError('Required actual canonical live task/physical receipt is missing or incompatible')
    for name in RUNTIME_FILES:
        if provenance['source_sha256'].get(name)!=sha(Path(source)/name):
            raise ValueError('Canonical live receipt runtime differs: '+name)
    log=Path(config['pal_controller_log']);rows=[]
    for line in log.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:continue # original append log remains retained by host
    complete=[r for r in rows if r.get('controllers') and all(c.get('name') and c.get('state') for c in r['controllers'])]
    if not complete:raise ValueError('PAL controller observations unavailable; no motion authorized by this setup')
    latest=complete[-1];active={c['name'] for c in latest['controllers'] if c['state']=='active'}
    age=now-latest['wall']
    if control_mode_contract==POSITION_CONTRACT:
        restored='arm_right_controller' in active and not {'arm_right_gravity_compensation_controller','gato_arm_effort_forward_runtime'} & active
    else:
        restored=DEFAULTS<=active and 'gato_arm_effort_forward_runtime' not in active
    if age<0 or age>config.get('maximum_pal_observation_age_s',2.) or not restored:
        raise ValueError('Fresh observed default controllers required after separately logged simulator restoration')
    return dict(canonical_receipt_sha256=sha(receipt_path),canonical_protocol_sha256=sha(receipt_path.parent/'protocol.json'),
        canonical_provenance_sha256=sha(receipt_path.parent/'provenance.json'),pal_snapshot=latest,pal_age_s=age,
        control_mode_contract=control_mode_contract)


def run(source,build,config_path,case,output,preflight=False,control_mode_contract=LEGACY_CONTRACT):
    source,build,config_path,output=map(lambda p:Path(p).resolve(),(source,build,config_path,output))
    output.mkdir(parents=True,exist_ok=True)
    metrics=dict(campaign_id='M03',research_question='RQ-monitor',execution_backend='Gazebo',timing_mode='live',
        task_success=None,physical_pass=None,execution_error=None,simulation_only=True,control_mode_contract=control_mode_contract)
    try:
        config=json.loads(config_path.read_text())['simulator_faults'];proof=prerequisites(config,source,time.monotonic(),control_mode_contract)
        modules=list((build/'modules').glob('bsqpN64_tiago_right*.so'))
        if len(modules)!=1:raise ValueError('Expected exact isolated N64 native module')
        if preflight and task_suite.run(source,build,output/'native-task-suite'):
            raise RuntimeError('Required genuine target native task suite failed; no simulator trial started')
        # The suite can take minutes. PAL observations must still be fresh when
        # the original supervisor starts; never reuse a stale startup snapshot.
        proof=prerequisites(config,source,time.monotonic(),control_mode_contract)
        selected='baseline' if preflight else case['parameters']['fault']
        script=source/'tiago_examples/fault_campaign/run.py'
        command=[sys.executable,'-B',str(script),'trial','--case',selected,'--output',str(output/'trial'),
            '--domain',str(config['domain']),'--controller-log',config['pal_controller_log'],
            '--inject-after','2','--horizon-fault-duration','0']
        bootstrap=Path(__file__).resolve().parents[1]/'fault_bootstrap'
        environment=dict(os.environ,GATO_FAULT_REPO=str(source),GATO_FAULT_NATIVE_MODULE=str(modules[0]),
            PYTHONPATH=str(bootstrap)+os.pathsep+str(source/'python')+os.pathsep+str(source/'tiago_src'),PYTHONDONTWRITEBYTECODE='1')
        environment.pop('GATO_FAULT_SPEC',None)
        write_json(output/'provenance.json',dict(command=command,module_sha256=sha(modules[0]),config_sha256=sha(config_path),
            adapter_sha256=sha(__file__),bootstrap_sha256=sha(bootstrap/'sitecustomize.py'),prerequisites=proof,
            source_runtime_sha256={p:sha(source/p) for p in RUNTIME_FILES},
            native_qualification='Target build/live setup remains pending until actual execution; no task pass inferred from source review'))
        with (output/'trial.log').open('w') as log:result=subprocess.run(command,cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT,check=False)
        path=output/'trial/result.json';summary=json.loads(path.read_text()) if path.exists() else {}
        metrics.update(original_returncode=result.returncode,original_summary=summary,
            fault_response_observed=summary.get('outcome')=='default_control_observed',
            baseline_setup_pass=summary.get('outcome')=='baseline_completed' if preflight else None)
        if control_mode_contract==POSITION_CONTRACT and summary.get('control_mode_contract')!=POSITION_CONTRACT:
            raise RuntimeError('Fault-trial classifier differs from requested position-only-v2 contract')
        if result.returncode or not summary or (preflight and not metrics['baseline_setup_pass']):
            raise RuntimeError('Original simulator trial incomplete or baseline setup failed; retain all observations')
        metrics['outcome']='fault_trial_observed'
    except BaseException:metrics.update(outcome='execution_failed',execution_error=traceback.format_exc())
    write_json(output/'metrics.json',metrics)
    write_json(output/'SHA256SUMS.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'})
    return 1 if metrics['execution_error'] else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','build','simulator-config','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--case',type=Path);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--control-mode-contract',choices=(LEGACY_CONTRACT,POSITION_CONTRACT),default=LEGACY_CONTRACT)
    args=parser.parse_args()
    if not args.case and not args.preflight:parser.error('--case or --preflight required')
    sys.exit(run(args.source,args.build,args.simulator_config,json.loads(args.case.read_text()) if args.case else None,args.output,args.preflight,args.control_mode_contract))
