"""Mechanical target qualification and guarded F01ab orchestration, no solver changes."""
import importlib.util,json,os,shutil,subprocess,time
from collections import Counter
import hashlib,re
import target
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CORE='16b33719b4f0fd463eec7262c99b16b0b324366f'
RUNTIME='502fa29c8b0d649dcb347f9534bf29c6935fa6dd'
SCIENCE='9cbf25721056e5b9dae1a5daf9e24982849b0573'

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def pin(path,sha):return dict(path=str(path),sha256=sha(path))

def qualify(source,work,sha,write,*,collector=None,prepared=None,parent=None):
    collector=collector or source;prepared=prepared or work
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=RUNTIME:raise ValueError('Prepared source Git identity differs')
    R=load(collector/'experiments/frequency_campaign/release.py','jetson_release')
    acceptance_path=HERE/'global-authority/telemetry-acceptance-decision.json'
    acceptance=json.loads(acceptance_path.read_text());modules={};checks=[]
    for N in [16,32,64,128]:
        original=prepared/'builds'/f'N{N}'/f'f01-build-N{N}.json';b=json.loads(original.read_text())
        if b['source_commit']!=RUNTIME or b['variant']!='ON':raise ValueError('Actual build source/observer differs')
        if b['flags']!=dict(GATO_EXPERIMENT_TELEMETRY=True,N=N,plant='tiago_right',cuda_arch='87-real'):raise ValueError('Actual build flags differ')
        archive=Path(b['source_root']);module=Path(b['module']['path']);cache=prepared/'builds'/f'N{N}'/'campaign-native/CMakeCache.txt'
        if sha(module)!=b['module']['sha256']:raise ValueError('Actual module changed')
        for name,digest in b['source_file_sha256'].items():
            if sha(archive/name)!=digest or sha(source/name)!=digest:raise ValueError('Actual archived source differs: '+name)
        cache_values={line.split('=',1)[0].split(':',1)[0]:line.split('=',1)[1] for line in cache.read_text().splitlines() if '=' in line and not line.startswith(('#','//'))}
        for key,value in [('GATO_EXPERIMENT_TELEMETRY','ON'),('CMAKE_CUDA_ARCHITECTURES','87-real'),('KNOTS',str(N)),('PLANT','tiago_right'),('CMAKE_BUILD_TYPE','Release')]:
            if cache_values.get(key)!=value:raise ValueError('Actual cache target differs: '+key)
        if N==32:
            # Preserve original pinned cache; new honest post-Make receipt points
            # to current cache, after the unchanged Make reconfiguration.
            if sha(prepared/'suite/CMakeCache.before.txt')!=b['cmake_cache']['sha256']:raise ValueError('Pre-Make cache identity differs')
            b=dict(b,cmake_cache=pin(cache,sha),original_build_receipt=pin(original,sha),post_make_cache_review='mechanical exact target checks passed; original cache snapshot retained')
            if prepared!=work:
                old=json.loads(Path(parent['release']['path']).read_text())
                reviewed=Path(old['modules']['32']['build_receipt']['path'])
                if sha(reviewed)!=old['modules']['32']['build_receipt']['sha256'] or json.loads(reviewed.read_text())['cmake_cache']!=pin(cache,sha):raise ValueError('Original qualified post-Make cache receipt changed')
            else:
                reviewed=work/'builds/N32/f01-build-N32-post-make.json';write(reviewed,b)
        else:
            if sha(cache)!=b['cmake_cache']['sha256']:raise ValueError('Non-suite cache changed')
            reviewed=original
        modules[str(N)]=dict(path=str(module),sha256=sha(module),build_receipt=pin(reviewed,sha))
        checks.append(dict(N=N,module_sha256=sha(module),cache_sha256=sha(cache),source_receipt=b['source_receipt']))
    prep=json.loads((prepared/'preparation.json').read_text())
    if prep['suite_exit_code']!=0 or prep['module_before_sha256']!=prep['module_after_sha256']:raise ValueError('Actual unchanged suite failed/changed module')
    for x in prep['suite_summaries']:
        if sha(Path(x['path']))!=x['sha256']:raise ValueError('Actual suite summary changed')
    summaries=[json.loads(Path(x['path']).read_text()) for x in prep['suite_summaries']]
    smoke=(prepared/'logs/make-test-tiago-tasks.log').read_text()
    if not re.search(r'cuda_access_smoke_ok=1[^\n]*\bsm=87\b',smoke):raise ValueError('Actual suite CUDA smoke did not confirm SM87')
    if {x.get('case') for x in summaries}!={'short_reach','long_reach','figure8_tracking'}:raise ValueError('Exact three suite case identities differ')
    for item,summary in zip(prep['suite_summaries'],summaries):
        saved=Path(item['path']).parent
        provenance=json.loads((saved/'receipt.json').read_text())
        if provenance['source_commit']!=RUNTIME or provenance['module_sha256']!=prep['module_before_sha256'] or Path(provenance['module_path']).resolve()!=Path(modules['32']['path']).resolve():raise ValueError('Actual suite source/module differs')
        archive=prepared/'builds/N32/campaign-source'
        if provenance['worker_sha256']!=sha(archive/'tests/python/tiago_task_worker.py') or provenance['case_sha256']!=sha(archive/'tests/python/tiago_task_cases'/f"{summary['case']}.json"):raise ValueError('Unchanged suite worker/fixture differs')
        for name,digest in provenance['source_sha256'].items():
            if sha(archive/name)!=digest:raise ValueError('Suite runtime source differs: '+name)
    if len(summaries)!=3 or not all(x.get('passed') is True for x in summaries):raise ValueError('Actual task suite not3passes')
    suite=work/'jetson-task-suite.json';write(suite,dict(status='passed',machine='jetson',exit_code=0,fixed_task_processes=3,error=None,
        restored_patch_sha256=acceptance['accepted_native_patch_sha256'],module_before_sha256=prep['module_before_sha256'],module_after_sha256=prep['module_after_sha256'],
        preparation=pin(prepared/'preparation.json',sha),summaries=prep['suite_summaries'],validation='Actual same-machine unchanged Make outputs, no imported GTX evidence'))
    review=work/'automated-target-validation.json';write(review,dict(verdict='approved',kind='automated_target_validation_under_reviewed_source_policy',validation_kind='automated',policy=[pin(HERE/name,sha) for name in ['full_run.py','prepare_jetson.py','target.py','reuse.py']],suite_log=pin(prepared/'logs/make-test-tiago-tasks.log',sha),actual_cuda_smoke='sm=87',checks=checks,
        authority='Accepted original observer source authority; target mechanical validation, not human review or numeric equivalence',task_suite=pin(suite,sha),suite_receipts=[pin(Path(x['path']).parent/'receipt.json',sha) for x in prep['suite_summaries']]))
    manifest=work/'jetson-release.json';write(manifest,dict(status='released_F01_A',machine='jetson',production_core_commit=CORE,compiled_source_commit=RUNTIME,collector_source_commit=subprocess.check_output(['git','-C',str(collector),'rev-parse','HEAD'],text=True).strip(),parent_failed_attempt=parent,
        collector_source_sha256={name:sha(collector/name) for name in R.COLLECTOR},bank_manifest_sha256=R.P.BANK_MANIFEST_SHA256,
        selection_sha256=sha(collector/'experiments/frequency_campaign/selected_variant.json'),observer_patch_sha256=acceptance['accepted_native_patch_sha256'],
        telemetry_acceptance=pin(acceptance_path,sha),reason_review=pin(HERE/'global-authority/telemetry-saved-review.json',sha),build_review=pin(review,sha),
        task_suite=dict(pin(suite,sha),machine='jetson'),modules=modules))
    for N in [16,32,64,128]:R.validate(manifest,sha(manifest),'jetson',N)
    return R,manifest

def completed_cell(results,eid):
    receipts=list((results/'jetson'/eid).glob('*/attempt-*/receipt.json'))
    if len(receipts)!=1:raise RuntimeError('Expected exactly one fresh cell attempt: '+eid)
    p=receipts[0];r=json.loads(p.read_text());attempt=p.parent
    if r.get('status')!='completed':raise RuntimeError('Runner cell failed/not finalized: '+eid)
    m=json.loads((attempt/'metrics.json').read_text())
    if m.get('execution_error') is not None or m.get('outcome')!='timing_cell_completed' or m.get('native_calls')!=241:raise RuntimeError('Worker cell failed/partial: '+eid)
    raw=attempt/'f01-raw';manifest=raw/'SHA256SUMS.json'
    if not manifest.is_file():raise RuntimeError('Raw finalization missing: '+eid)
    hashes=json.loads(manifest.read_text())
    required={'events.jsonl','protocol.json','provenance.json','metrics.json'}
    if not required.issubset(hashes):raise RuntimeError('Raw required files missing: '+eid)
    actual={str(x.relative_to(raw)) for x in raw.rglob('*') if x.is_file() and x!=manifest}
    if actual!=set(hashes):raise RuntimeError('Raw manifest coverage differs: '+eid)
    for name,digest in hashes.items():
        target=raw/name
        if not target.resolve().is_relative_to(raw.resolve()) or target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Raw file missing/changed: '+eid+' '+name)
    if json.loads((raw/'metrics.json').read_text())!=m:raise RuntimeError('Mirrored metrics differ: '+eid)
    events=[json.loads(line) for line in (raw/'events.jsonl').read_text().splitlines()]
    for index,event in enumerate(events):
        if event['event_index']!=index or hashes.get(event['raw_file'])!=event['sha256']:raise RuntimeError('Raw event identity differs: '+eid)
    solves=[x for x in events if x['event']=='solve_completed']
    expected=Counter(cold=12,warm=222,goal_switch=6,process_cuda_warmup=1)
    if Counter(x['record']['category'] for x in solves)!=expected:raise RuntimeError('Raw solve category coverage differs: '+eid)
    if sum(x['event']=='solve_started' for x in events)!=241 or sum(x['event']=='native_return' for x in events)!=241:
        raise RuntimeError('Raw native/input call coverage differs: '+eid)
    episodes=m.get('episodes',[])
    if len(episodes)!=12 or m.get('missing_episodes') or any(x.get('execution_error') or x.get('attempted_solves')!=20 or x.get('completed_updates')!=20 or x.get('missing_solve_slots') or x.get('missing_update_slots') for x in episodes):
        raise RuntimeError('Raw episode coverage differs: '+eid)
    return p,m

def run_campaign(source,work,python,sha,write,command,*,prepared=None,parent=None):
    state=work/'launch-state.json';family=work/'family-state.json';started=time.time()
    write(state,dict(status='running',stage='target_validation',started=time.time()))
    R=None;context=None
    try:
        collector=work/'collector-source'
        collector_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        command(['git','clone','--shared','--no-checkout',str(ROOT),str(collector)],work,work/'logs/collector-clone.log')
        command(['git','-C',str(collector),'checkout','--detach',collector_commit],work,work/'logs/collector-checkout.log')
        for name in ['protocol.py','release.py','runtime_bank.py']:
            relative=Path('experiments/frequency_campaign')/name
            if sha(ROOT/relative)!=sha(collector/relative):raise ValueError('Committed collector differs from launcher checkout; use the published reviewed commit')
        R,manifest=qualify(source,work,sha,write,collector=collector,prepared=prepared,parent=parent);digest=sha(manifest);context=R.validate(manifest,digest,'jetson',32)
        write(state,dict(status='running',stage='canonical_A_preflight',release=pin(manifest,sha)))
        command([python,'-B',str(collector/'experiments/frequency_campaign/run.py'),'--readiness','--output',str(work/'readiness'),
                 '--manifest',str(manifest),'--manifest-sha256',digest,'--machine','jetson','--family-state',str(family)],collector,work/'logs/readiness.log')
        R.readiness_guard(context,family) # Actual task_success AND physical_pass; exit0 alone insufficient.
        clone=work/'runner-repo'
        command(['git','clone','--shared','--no-checkout',str(collector),str(clone)],work,work/'logs/runner-clone.log')
        command(['git','-C',str(clone),'checkout','--detach',collector_commit],work,work/'logs/runner-checkout.log')
        results=work/'results';cache=work/'runner-cache'
        write(clone/'experiments/local.json',dict(machine='jetson',python=python,output=str(results),cache=str(cache)))
        cases=R.selected_cases();experiments=[]
        frozen={f'experiments/frequency_campaign/inputs/{name}':digest for name,digest in json.loads((collector/'experiments/frequency_campaign/inputs/SHA256SUMS.json').read_text()).items()}
        frozen.update({name:sha(collector/name) for name in R.COLLECTOR})
        frozen['experiments/frequency_campaign/selected_variant.json']=sha(collector/'experiments/frequency_campaign/selected_variant.json')
        for cell in cases:
            eid=f"F01ab-N{cell['N']}-SQP{cell['SQP']}-PCG{cell['PCG']}"
            experiments.append(dict(id=eid,enabled=True,source_commit=collector_commit,machines=['jetson'],repeats=1,build=dict(commands=[],artifacts=[]),
                command=[python,'-B',str(collector/'experiments/frequency_campaign/run.py'),'--case','{case}','--output','{output}',
                         '--manifest',str(manifest),'--manifest-sha256',digest,'--machine','jetson','--family-state',str(family)],
                cases=[dict(id=eid,parameters=cell)],inputs=frozen))
        if len(experiments)!=64 or {(c['N'],c['SQP'],c['PCG']) for c in cases}!={(n,s,p) for n in [16,32,64,128] for s in [1,2,5,10] for p in [100,200,400,800]}:raise ValueError('Frozen64matrix required')
        plan=work/'jetson-f01ab-plan.json';write(plan,dict(version=1,experiments=experiments))
        argv=[python,'-B',str(clone/'experiments/run.py'),'--plan',str(plan),'--machine','jetson','--output',str(results)]
        command(argv+['--dry-run'],clone,work/'logs/runner-dry-run.log')
        pending=[line.removeprefix('pending: ') for line in (work/'logs/runner-dry-run.log').read_text().splitlines() if line.startswith('pending: ')]
        if pending!=[f"{e['id']}/{e['id']}/0" for e in experiments]:raise RuntimeError('Actual runner dry-run differs from exact64 pending queue')
        write(work/'runner-dry-run.json',dict(status='exact64_pending',plan=pin(plan,sha),log=pin(work/'logs/runner-dry-run.log',sha),pending=pending))
        completed=[]
        for e in experiments:
            eid=e['id'];R.readiness_guard(context,family)
            # Durable marker BEFORE child/native import. Abrupt termination leaves
            # running marker; parent command refuses another attempt until reviewed.
            write(state,dict(status='running',stage='matrix_cell',cell=eid,completed_cells=completed,plan=pin(plan,sha)))
            command(argv+['--only',eid],clone,work/'logs'/f'{eid}.log')
            receipt,metrics=completed_cell(results,eid)
            completed.append(dict(id=eid,status='completed',receipt=pin(receipt,sha),native_calls=metrics['native_calls'],episodes=12,categories=dict(cold=12,warm=222,goal_switch=6,process_cuda_warmup=1),raw_manifest=pin(receipt.parent/'f01-raw/SHA256SUMS.json',sha),events=pin(receipt.parent/'f01-raw/events.jsonl',sha)))
        if len(completed)!=64 or sum(x['native_calls'] for x in completed)!=15424:raise RuntimeError('Matrix coverage incomplete')
        files=[p for p in work.rglob('*') if p.is_file() and not p.is_symlink()]
        write(state,dict(status='completed',elapsed_campaign_wall_s=time.time()-started,storage_bytes=sum(p.stat().st_size for p in files),storage_files=len(files),finished=time.time(),plan=pin(plan,sha),release=pin(manifest,sha),parent_failed_attempt=parent,cells=completed,native_calls=15424,automated_target_validation=pin(work/'automated-target-validation.json',sha) if (work/'automated-target-validation.json').exists() else None,canonical_A_readiness=pin(family,sha),dry_run=pin(work/'runner-dry-run.json',sha)))
        print('F01ab complete:64cells,15424calls.',flush=True)
    except BaseException as error:
        if context is not None:R.write_state(family,context,'held',reason=str(error),failed_attempt=str(work))
        previous=json.loads(state.read_text()) if state.exists() else {}
        write(state,dict(previous,status='held',error_type=type(error).__name__,error=str(error)))
        raise
    # Packaging cannot invalidate a completed measurement cohort. It can also
    # be repeated independently with export_results.py without running experiments.
    try:
        from export_results import export_results
        export_results(work, ROOT/'jetson-f01ab-results.tar.gz')
    except Exception as error:
        print('Measurements completed; results export failed:', error, flush=True)
        print('Run python3 experiments/jetson_preparation/export_results.py to retry packaging.', flush=True)


def continue_prepared(work,python,sha,write,command,device=None):
    prep=json.loads((work/'preparation.json').read_text())
    for marker in work.parent.glob('*/launch-state.json'):
        if json.loads(marker.read_text()).get('status') in ['running','held']:raise RuntimeError('Existing campaign held or interrupted; preserve and review: '+str(marker))
    if prep.get('status')!='prepared_TARGET_REVIEW_REQUIRED':raise RuntimeError('Only successful original preparation may continue; no failed resume')
    if prep.get('runtime_source')!=RUNTIME or prep.get('science_source')!=SCIENCE or prep.get('cuda_arch')!='87-real':raise ValueError('Prepared source/architecture differs')
    if any((work/name).exists() for name in ['launch-state.json','family-state.json','readiness','runner-repo','runner-cache','results','jetson-release.json','automated-target-validation.json']):raise RuntimeError('Campaign stage already started/held; no automatic repeat')
    for item in prep['build_receipts']+prep['suite_summaries']:
        if sha(Path(item['path']))!=item['sha256']:raise ValueError('Prepared receipt changed')
    python=str(Path(shutil.which(python) or python).expanduser().absolute())
    if python!=prep.get('python'):raise ValueError('Prepared Python environment differs; no changed-setup continuation')
    selection=target.select_target(device)
    write(work/'target-selection-continuation.json',selection)
    (work/'tmp').mkdir(exist_ok=True);(work/'environment-cache').mkdir(exist_ok=True)
    subprocess.run([python,'-B','-c','import numpy, pinocchio, pytest, pybind11, torch'],check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(work/'tmp'),XDG_CACHE_HOME=str(work/'environment-cache')))
    os.environ.update(TMPDIR=str(work/'tmp'),XDG_CACHE_HOME=str(work/'environment-cache'),PYTHONDONTWRITEBYTECODE='1')
    run_campaign(work/'source',work,python,sha,write,command)
