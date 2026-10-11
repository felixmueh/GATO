#!/usr/bin/env python3
"""Merge explicit prepared plans and optionally package runner files/source pins.

This does not execute studies. Missing definitions and scientific decisions stay
visible in the inventory. Physical robot scripts remain outside every queue.
"""
import argparse
import hashlib
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tarfile
import uuid

ROOT=Path(__file__).resolve().parents[1]
PLANS=('campaign.json','f01c.json','resource_profiles.json','model_checks.json','control_studies.json',
    'collision_monitor.json','bb_gtx1070.json','bb_rtx5060.json','ba_gtx1070.json','ba_rtx5060.json',
    'demo_internal.json','demo_gazebo.json','deployment.json','simulated_faults.json','cylinder_studies.json','cylinder_studies_jetson_proposed.json','t01_jetson_proposed.json','f01c_jetson_proposed.json',
    'resource_profiles-jetson-proposed.json','control_studies_jetson_proposed.json','collision_monitor_jetson_proposed.json',
    'bb_jetson_proposed.json','ba_jetson_proposed.json',
    'force-gtx1070-proposed.json','force-rtx5060-proposed.json','force-jetson-proposed.json')
OMISSIONS={
 'Jetson F01ab':'Completed according to Felix; manual collection/import pending; never rerun for migration.',
 'GTX1070 T01/F01ab/F01c/F04/F05':'Retained completed measurements; not selected again.',
  'D02 physical robot':'Standalone scripts/manual confirmation and index; never automatic runner execution.',
 'B02 QDLDL':'Not selected: Felix dropped this campaign comparison; historical results retained, no pending port.',
 'Q03 target task suite':'Separate readiness helper/relevant original fixture checks; not duplicated as research data.',
 'Historical M01/F02/F03/C02/C07/L00/X03-IIWA':'Record-only; exact result locations retained in tracker/evidence ledger.',
 'L01 learning':'Deferred.',
}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def sources(experiments):
    result=set()
    for experiment in experiments:
        for key in ('source_commit','collector_source_commit'):
            if experiment.get(key):result.add(experiment[key])
        dependencies=experiment.get('dependency_source_commits',[])
        if isinstance(dependencies,dict):dependencies=dependencies.values()
        result.update(dependencies)
    return sorted(result)


def merge(output,plan_files=None):
    output=Path(output);definitions=[];missing=[];provenance={};retained=[];source_notes={}
    for filename in plan_files or PLANS:
        path=ROOT/'experiments'/filename
        if not path.is_file():missing.append(filename);continue
        plan=json.loads(path.read_text())
        source_notes[filename]={k:v for k,v in plan.items() if k not in ('version','experiments','retained_results')}
        if plan.get('retained_results'):retained.append(dict(plan=filename,results=plan['retained_results']))
        for experiment in plan['experiments']:
            if not experiment.get('enabled',True):continue
            definitions.append((filename,experiment))
    targets={}
    for machine in ('gtx1070','rtx5060','jetson'):
        for backend_group in ('desktop','gazebo'):
            if machine=='gtx1070' and backend_group=='gazebo':continue
            if machine=='jetson' and backend_group=='gazebo':continue
            selected=[];seen=set();origins=[]
            for filename,original in definitions:
                if machine not in original.get('machines',[]):continue
                backend=original.get('backend',original.get('execution_backend','internal')).lower()
                gazebo='gazebo' in backend
                if gazebo!=(backend_group=='gazebo'):continue
                if 'robot' in backend or 'physical' in backend:raise ValueError('Physical robot cannot enter automatic merged plans')
                if machine=='jetson' and 'f01ab' in original['id'].lower():raise ValueError('Completed Jetson F01ab must never be queued')
                if original['id'] in seen:raise ValueError('Conflicting duplicate experiment ID: '+original['id'])
                seen.add(original['id']);experiment=deepcopy(original);experiment['machines']=[machine];selected.append(experiment)
                origins.append(dict(plan=filename,experiment=original['id']))
            name='jetson_proposed.json' if machine=='jetson' else f'{machine}_{backend_group}.json'
            plan=dict(version=1,status='proposed_duration_discussion_and_target_review_pending' if machine=='jetson' else 'prepared_not_executed',
                experiments=selected,source_plans=origins,source_plan_notes={name:source_notes[name] for name in dict.fromkeys(r['plan'] for r in origins)},retained_results=retained,
                omissions=OMISSIONS,missing_plan_files=missing,
                note='Explicit prepared coverage only. Canonical preflights and actual task/physical gates retained; GPU use sequential. No physical robot queue or completed-study reruns.')
            write(output/name,plan);targets[name]=plan
    all_experiments=[e for p in targets.values() for e in p['experiments']]
    inventory=dict(plans={name:dict(experiments=[e['id'] for e in p['experiments']],
        cases=sum(len(e.get('cases',[]))*e.get('repeats',1) for e in p['experiments']),status=p['status']) for name,p in targets.items()},
        source_commits=sources(all_experiments),source_plan_files=list(plan_files or PLANS),source_plan_notes=source_notes,missing_plan_files=missing,omissions=OMISSIONS,
        readiness='Preparation inventory, not a statement that all experiments or targets are execution-ready.')
    write(output/'inventory.json',inventory)
    return inventory


def package(output,inventory,plans_output=None):
    output=Path(output)
    if subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--','experiments']).returncode:
        raise ValueError('Checkpoint changed runner files before creating a source package')
    tracked_names=set(subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z','experiments'],text=True).split('\0'))
    required=['experiments/'+name for name in inventory.get('source_plan_files',[]) if (ROOT/'experiments'/name).is_file()]
    if Path(__file__).resolve().is_relative_to(ROOT):required.append(str(Path(__file__).resolve().relative_to(ROOT)))
    missing=[name for name in required if name not in tracked_names]
    if missing:raise ValueError('Checkpoint prepared package files first: '+', '.join(missing))
    output.mkdir(parents=True,exist_ok=False)
    # Full source history is needed to preserve exact commits. Tracked historical
    # Git objects remain; working-tree results, caches and private builds are excluded.
    runner=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    pins=sorted(set(inventory['source_commits']+[runner]));namespace='experiment-package-'+uuid.uuid4().hex[:10]
    refs=[];runner_ref=None
    try:
        for pin in pins:
            full=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',pin+'^{commit}'],text=True).strip()
            ref='refs/heads/'+namespace+('-runner' if full==runner else '-'+full[:12])
            subprocess.run(['git','-C',str(ROOT),'update-ref',ref,full,''],check=True);refs.append(ref)
            if full==runner:runner_ref=ref
        subprocess.run(['git','-C',str(ROOT),'bundle','create',str((output/'sources.bundle').resolve()),*refs],check=True)
    finally:
        for ref in refs:subprocess.run(['git','-C',str(ROOT),'update-ref','-d',ref],check=True)
    # Package tracked current runner assets plus generated explicit plans. Never
    # recursively copy this checkout or accidentally include local.json/raw/cache.
    tracked=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z','experiments'],text=True).split('\0')
    files={ROOT/name for name in tracked if name and name!='experiments/local.json'}
    generated=Path(plans_output or ROOT/'experiments/merged')
    files.update(generated.glob('*.json'))
    with tarfile.open(output/'runner-assets.tar.gz','w:gz') as archive:
        for path in sorted(files):
            if path.is_file():
                relative=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else 'experiments/merged/'+path.name
                archive.add(path,arcname=relative,recursive=False)
    write(output/'inventory.json',dict(inventory,runner_commit=runner,runner_bundle_branch=runner_ref.removeprefix('refs/heads/'),
        package_sha256={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('sources.bundle','runner-assets.tar.gz')},
        bundle_scope='Exact source Git objects, including tracked historical objects; no working-tree result/build/cache trees.',
        runner_scope='Tracked experiments code/assets and generated plans; local.json excluded.',
        setup='Clone sources.bundle using runner_bundle_branch, extract runner-assets.tar.gz there, then run an explicit merged plan. Runner owns private source checkouts/shared builds.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/merged')
    parser.add_argument('--plan-files',nargs='+',help='Explicit alternate source-plan filenames; omitted uses the declared inventory')
    parser.add_argument('--package',type=Path,help='Fresh package directory; invoke only after implementation/source pins are checkpointed')
    args=parser.parse_args();inventory=merge(args.output,args.plan_files)
    print(json.dumps(inventory,indent=2))
    if args.package:package(args.package,inventory,args.output)
