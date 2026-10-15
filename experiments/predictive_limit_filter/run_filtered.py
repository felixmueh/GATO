#!/usr/bin/env python3
"""Functional filtered20 SIMULATION ONLY; no real-time concurrency/robot claim."""
import argparse,importlib.util,json,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['source','module','module-receipt','pool','protocol','output']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--stage',choices=['canonical','first1','remaining19'],required=True)
    ap.add_argument('--pairs',type=int,nargs='+');ap.add_argument('--preflight-input',type=Path)
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--save-plans',action='store_true')
    args=ap.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    prep=load('filtered20_prepare',HERE/'prepare_filtered.py');p,info=prep.validate(args,prep.read(args.protocol))
    if args.dry_run:print(json.dumps(info,indent=2,allow_nan=False));return 0
    args.output.mkdir(parents=True,exist_ok=False);info['created_utc']=datetime.now(timezone.utc).isoformat()
    prep.write(args.output/'protocol.json',p);prep.write(args.output/'provenance.json',info)
    prep.write(args.output/'launch_receipt.json',dict(argv=sys.argv,cwd=str(Path.cwd()),created_utc=info['created_utc'],
        protocol_sha256=prep.sha(args.protocol),worker_sha256=prep.sha(__file__),module_sha256=info['module_sha256'],
        simulation_only=True,actual_concurrent_scheduling_proven=False))
    api=load('original_frozen_worker',HERE.parent/'random_reaching_baseline/run_frozen_baseline.py')
    try:runtime=api.load_runtime(args.source,args.module)
    except BaseException:
        prep.write(args.output/'setup_failure.json',dict(error=traceback.format_exc()));raise
    mode=load('filtered20_mode_b',HERE/'mode_b_filtered.py');rows=[]
    for pair in info['pairs']:
        row=mode.run_episode_filtered(api,runtime,args,p,pair);rows.append(row);prep.write(args.output/'summary.json',rows)
        if row['shared_setup_failure'] or row['interrupted']:break
    status=dict(assigned=len(info['pairs']),recorded=len(rows),completed_episodes=sum(r['complete'] for r in rows),
        incomplete_filter_stops=sum(r['failure_class']=='filter_stop' for r in rows),
        arrivals=sum(r['task_success'] for r in rows),physical_passes=sum(r['physical_pass'] for r in rows),
        execution_errors=sum(r['execution_error'] is not None for r in rows),
        failure_record_count_including_filter_stops=sum(r['execution_error'] is not None for r in rows),
        numerical_errors=sum(r['failure_class']=='numerical' for r in rows),
        infrastructure_errors=sum(r['failure_class']=='infrastructure' for r in rows),
        infrastructure_pass=len(rows)==len(info['pairs']) and not any(r['shared_setup_failure'] or r['interrupted'] for r in rows),
        scope='Conditional20 or explicitlynamedGTXsmoke; incompleteprefixes never fullphysicalpasses',
        timing='Nativewrapper timepaid; filterCPUwallseparate independent-loop assumption, actualconcurrentdeploymentunproved')
    if args.stage=='canonical':
        r=rows[0] if rows else None
        status['canonical_dependent_gate']=bool(r is not None and status['infrastructure_pass'] and r['physical']['finite'] and
            (not r['execution_error'] or r.get('failure_class')=='filter_stop'))
        status['infrastructure_pass']=status['canonical_dependent_gate']
    prep.write(args.output/'cohort_status.json',status)
    print(json.dumps(status,indent=2));return 0 if status['infrastructure_pass'] else 1

if __name__=='__main__':raise SystemExit(main())
