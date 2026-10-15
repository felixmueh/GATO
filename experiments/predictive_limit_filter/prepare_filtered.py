#!/usr/bin/env python3
"""Portable CPU preparation for functional filtered simulation of original J1 nonpasses."""
import argparse,copy,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]
INPUTS=SOURCE/'experiments/jetson_random_reaching/inputs'
spec=importlib.util.spec_from_file_location('portable_jetson_prep',INPUTS.parent/'prepare.py')
portable=importlib.util.module_from_spec(spec);spec.loader.exec_module(portable)
sha=portable.sha;read=portable.read;write=portable.write
ROLE='J1_nonphysical20_filtered_functional_simulation'
SETTINGS=dict(q_margin_rad=.02,velocity_margin_rad_s=.10,effort_fraction=.95,forecast_s=.03,forecast_step_s=.003,nominal_control_step_s=.003)
TARGET_SHA='3ba05f3cafe4038c88fbdc77acd891dbe99899cf1db6d2bcde9de6984312035a'
DESKTOP_SHA='d9bee23f90a1d7816b32db23749bd651db2a7537f94c3dfec15f1a448bb34cc8'
FILTER_FILES=['filter.py','mode_b_filtered.py','run_filtered.py','prepare_filtered.py','original_J1_selection.json']
ARCHITECTURE='Functional independent high-rate execution safety-loop assumption; filter every actual segment includingbootstrap andstaleplans duringnativewrapper; actual Python/GIL concurrent scheduling not implemented or proved'
TIME_POLICY='Native BSQP.solve wrapper wall advancesoldplan exactlyasoriginalModeB; filterCPUwall recordedseparately, notaddedto simulatedclock, assumedoverlap NOTfully computation-paid measurement;3msoverrunsdescriptiveonly'

def validate(args,p):
    _,assignment=portable.verify_bundle()
    selected=read(HERE/'original_J1_selection.json')
    if sha(HERE/'original_J1_selection.json')!=p['selection_sha256'] or len(selected['task_ids'])!=20:
        raise ValueError('Original J1 outcome-selected20 identity differs')
    if p['study_role']!=ROLE or p['run_task_ids']!=selected['task_ids'] or p['original_J1_labels']!=selected['labels']:
        raise ValueError('Retain all20 original nonpasses including original4errors')
    baseline=read(INPUTS/'desktop_protocol.json')
    exceptions={'id','study_role','run_task_ids','solver_params','native_module_path','native_module_sha256','native_build_receipt','research_baseline_freeze','reused_canonical','validation_assignment','timing_mode','offline_timing'}
    for key,value in baseline.items():
        if key not in exceptions and p.get(key)!=value:raise ValueError('Frozen J1 setting differs: '+key)
    if p['solver_params']!=dict(baseline['solver_params'],max_sqp_iters=5):raise ValueError('Use exact original J1 cap5/costs')
    if p['timing_mode']!='B' or p['offline_timing']!='solve_time':raise ValueError('Only original ModeB clock')
    if p['predictive_filter']!=dict(settings=SETTINGS,architecture=ARCHITECTURE,time_policy=TIME_POLICY):raise ValueError('Frozen filter policy differs')
    if p['task_ids']!=assignment['task_ids'] or p['validation_task_ids']!=assignment['task_ids']:raise ValueError('Original population context differs')
    if set(p['filter_source_sha256'])!=set(FILTER_FILES):raise ValueError('Complete reviewed filter source pins required')
    for name,digest in p['filter_source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Filter source changed: '+name)
    for key,name in [('research_baseline_freeze','research_baseline_freeze.json'),('reused_canonical','canonical_metrics.json'),('validation_assignment','assignment.json')]:
        if sha(p[key]['path'])!=sha(INPUTS/name):raise ValueError('Frozen portable context differs')
    captured=read(INPUTS/'native_source_sha256.json')['native_files']
    for name,digest in captured.items():
        if sha(args.source/name)!=digest:raise ValueError('Frozen numerical/controller source differs: '+name)
    if {n:sha(args.pool/n) for n in p['input_sha256']}!=p['input_sha256']:raise ValueError('Frozen pool differs')
    expected={'canonical':[-1],'first1':[502],'remaining19':[i for i in selected['task_ids'] if i!=502]}[args.stage]
    if args.stage=='canonical':
        if not args.preflight_input or sha(args.preflight_input)!=sha(INPUTS/'canonical_geometry.json'):raise ValueError('Exact canonical required')
    elif args.preflight_input or args.pairs!=expected:raise ValueError('Run exact stage IDs; do not repeat first1 in remaining19')
    if args.output.exists():raise ValueError('Fresh output required; preserve original/failed data')
    platform=p['target_platform']
    if platform not in ('GTX-smoke','Jetson'):raise ValueError('Only declared targets are supported')
    expected_sha=DESKTOP_SHA if platform=='GTX-smoke' else TARGET_SHA
    if p['native_module_sha256']!=expected_sha:raise ValueError('Retained module identity differs')
    module_sha=sha(args.module) if args.module.is_file() else None
    receipt=read(args.module_receipt) if args.module_receipt.is_file() else None
    if module_sha is not None and module_sha!=expected_sha:raise ValueError('Actual retained module differs')
    if not args.dry_run and (module_sha is None or receipt is None):raise ValueError('Actual module/receipt required')
    if receipt:
        if receipt.get('module_sha256')!=expected_sha or receipt.get('numerical_source_base')!=portable.BASE or receipt.get('cost_bound_defines')!=portable.BOUNDS:
            raise ValueError('Module receipt numerical source/bounds differs')
        if platform=='Jetson' and (receipt.get('cuda_arch') not in (87,'87') or receipt.get('host_machine') not in ('aarch64','arm64')):
            raise ValueError('Reuse retained ARM sm87 build')
    predecessor_status='not_applicable' if args.stage=='canonical' else 'pending actual execution; dryrun does not waive gate'
    if args.stage!='canonical' and not args.dry_run:
        root=args.output.parent
        stages=['canonical']+(['first1'] if args.stage=='remaining19' else [])
        for stage in stages:
            status=read(root/stage/'cohort_status.json')
            if not status['infrastructure_pass'] or status['assigned']!=1 or status['recorded']!=1:
                raise ValueError('Dependent stage held by incomplete/shared-failed predecessor: '+stage)
            if read(root/stage/'protocol.json')!=p:
                raise ValueError('Predecessor protocol differs')
            rows=read(root/stage/'summary.json')
            if len(rows)!=1 or rows[0]['shared_setup_failure'] or rows[0]['interrupted']:
                raise ValueError('Predecessor stage is not terminal/setup-valid')
            r=rows[0]
            if stage=='canonical' and ((r['execution_error'] and r.get('failure_class')!='filter_stop') or not r['physical']['finite']):
                raise ValueError('Unexpected canonical numerical/nonfinite failure holds dependent work')
        predecessor_status='verified terminal predecessors; canonical scientific timeout/filterstop labels preserved'
    return p,dict(protocol=p,source=str(args.source),source_commit=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip(),
        source_sha256=captured,frontend=str(SOURCE),frontend_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        pairs=expected,module=str(args.module),module_sha256=module_sha,module_receipt_sha256=sha(args.module_receipt) if receipt else None,
        module_status='pending target path: CPU dryrun only' if module_sha is None else 'retained exact module',
        predecessor_status=predecessor_status,wrapper_sha256=sha(HERE/'run_filtered.py'),filter_source_sha256=p['filter_source_sha256'],
        selection_sha256=p['selection_sha256'],architecture=ARCHITECTURE,time_policy=TIME_POLICY,native_calls_in_dryrun=0,
        host_power_policy='MODE40W userconfirmed; hostpower/clockchecks waived; no hostsettings queried/changed')

def make_protocol(module,receipt,platform):
    selected=read(HERE/'original_J1_selection.json');p=copy.deepcopy(read(INPUTS/'desktop_protocol.json'))
    p['solver_params']['max_sqp_iters']=5
    for key,name in [('research_baseline_freeze','research_baseline_freeze.json'),('reused_canonical','canonical_metrics.json'),('validation_assignment','assignment.json')]:
        p[key]=dict(p[key],path=str((INPUTS/name).resolve()),sha256=sha(INPUTS/name))
    p.update(id='J1-original20-predictive-filter-functional-'+platform,study_role=ROLE,
        timing_mode='B',offline_timing='solve_time',run_task_ids=selected['task_ids'],target_platform=platform,
        native_module_path=str(module),native_module_sha256=DESKTOP_SHA if platform=='GTX-smoke' else TARGET_SHA,native_build_receipt=str(receipt),
        execution_order=[502]+[i for i in selected['task_ids'] if i!=502],original_J1_labels=selected['labels'],selection_sha256=sha(HERE/'original_J1_selection.json'),
        predictive_filter=dict(settings=SETTINGS,architecture=ARCHITECTURE,time_policy=TIME_POLICY),
        filter_source_sha256={n:sha(HERE/n) for n in FILTER_FILES},
        cohort_scope='All20 originalJ1 physicalnonpasses,16finiteunsafe+4numericalerrors; outcome-selecteddiagnosticrerunsfromoriginalstart; no100populationclaim',
        canonical_policy='Known finite timeout/physical negatives and filter-search-stop remain diagnostic negatives; unexpected canonical numerical/nonfinite or shared setup holds dependent stages; remaining19 requires terminal first1 (individual numerical/filterstops permitted)')
    return p

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--module',type=Path,required=True);ap.add_argument('--module-receipt',type=Path,required=True)
    ap.add_argument('--platform',choices=['Jetson','GTX-smoke'],required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    p=make_protocol(args.module.resolve(),args.module_receipt.resolve(),args.platform)
    write(args.output/'protocol.json',p)
    commands=[]
    for stage in ['canonical','first1']+(['remaining19'] if args.platform=='Jetson' else []):
        extra=['--preflight-input',str(INPUTS/'canonical_geometry.json')] if stage=='canonical' else ['--pairs',*map(str,[502] if stage=='first1' else [i for i in p['run_task_ids'] if i!=502])]
        argv=[sys.executable,str(HERE/'run_filtered.py'),'--source',str(SOURCE),'--module',str(args.module.resolve()),'--module-receipt',str(args.module_receipt.resolve()),
            '--pool',str(INPUTS),'--protocol',str(args.output/'protocol.json'),'--stage',stage,'--output',str(args.output.parent/stage),*extra]
        dry=subprocess.run(argv+['--dry-run'],capture_output=True,text=True,check=True)
        (args.output/(stage+'.dryrun.json')).write_text(dry.stdout)
        commands.append(dict(stage=stage,argv=argv,cwd=str(SOURCE),protocol_sha256=sha(args.output/'protocol.json'),dryrun_sha256=sha(args.output/(stage+'.dryrun.json'))))
    write(args.output/'launch_commands.json',commands)
    write(args.output/'PREPARED_NOT_RUN.json',dict(platform=args.platform,protocol_sha256=sha(args.output/'protocol.json'),CPUdryruns=len(commands),native_calls=0,
        assignment=p['run_task_ids'],original_label_counts={'finite_unsafe':16,'execution_errors':4},filter_source_sha256=p['filter_source_sha256']))
    print(args.output)

if __name__=='__main__':main()
