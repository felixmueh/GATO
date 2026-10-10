#!/usr/bin/env python3
"""Run approved F01ab short12 on Jetson with actual target guards.

Four sequential isolated87-real builds + ONE unchanged N32 ON task suite.
Actual target validation, canonical A preflight, then64sequentialcells. No automatic retries.
"""
import argparse,hashlib,json,os,platform,shutil,subprocess,sys,datetime,signal,re,time
import full_run
import target
import reuse
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parents[1]
RUNTIME='502fa29c8b0d649dcb347f9534bf29c6935fa6dd'
SCIENCE='9cbf25721056e5b9dae1a5daf9e24982849b0573'
BANK='29f8ccdff73d36c339e50ea7153a46fb6ecd22b32da786b74f16cb9e1a61f609'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def verify_package():
    for name,digest in json.loads((PACKAGE/'SHA256SUMS.json').read_text()).items():
        path=(PACKAGE/name).resolve()
        if not path.is_relative_to(PACKAGE.resolve()) or sha(path)!=digest:raise ValueError('Package changed: '+name)
    decision=json.loads((PACKAGE/'global-authority/telemetry-acceptance-decision.json').read_text())
    if decision['decision']!='accept_existing_first_instrumentation_without_equivalence_gate':raise ValueError('Observer authority differs')
    if decision['comparison_audit']['sha256']!=sha(PACKAGE/'global-authority/telemetry-saved-review.json'):raise ValueError('Accepted reason audit differs')

def descendant_groups(pid):
    """Retain native workers even when the runner gave them a new session."""
    rows={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rsplit(')',1)[1].split()
            rows[int(path.parent.name)]=(int(fields[1]),int(fields[2]))
        except (OSError,ValueError):pass
    children={pid}
    while True:
        added={p for p,(parent,_) in rows.items() if parent in children}
        if added.issubset(children):break
        children |= added
    return {rows[p][1] for p in children if p in rows and rows[p][1]!=os.getpgrp()}

def run(command,cwd,log,env=None):
    print('Running:',repr(command),'log:',str(log),flush=True)
    with log.open('w') as stream:
        child=subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=child.wait()
            if code:raise subprocess.CalledProcessError(code,command)
        except BaseException:
            if child.poll() is None:
                # SIGINT reaches the generic runner's own child cleanup handler.
                descendants=descendant_groups(child.pid)
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:pass
                for group in descendants | {child.pid}:
                    try:os.killpg(group,signal.SIGKILL)
                    except ProcessLookupError:pass
                child.wait()
            raise

def retain_suite(actual,retained):
    """Copy actual fixed CMake artifact directory, preserving its originals."""
    if actual.exists() and not retained.exists():
        retained.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(actual,retained)
    summaries=list(retained.glob("*/summary.json"))
    if len(summaries)!=3 or not all(json.loads(p.read_text()).get("passed") is True for p in summaries):
        raise RuntimeError("Required suite did not retain exactly3 passing task summaries")
    return summaries

def check_prerequisites(repo,work,python,allowed,device=None):
    prerequisites=allowed/'.prerequisites'
    for name in ['tmp','cache']:(prerequisites/name).mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(prerequisites/'tmp'),XDG_CACHE_HOME=str(prerequisites/'cache'),PYTHONDONTWRITEBYTECODE='1')
    tools={}
    for tool in ['cmake','make','git','nvcc','c++']:
        executable=shutil.which(tool)
        if not executable:raise RuntimeError('Missing required tool '+tool+'; no attempt directory created')
        tools[tool]=subprocess.check_output([executable,'--version'],text=True).splitlines()[0]
    version=re.search(r'(\d+)\.(\d+)',tools['cmake'])
    if not version or tuple(map(int,version.groups()))<(3,24):raise RuntimeError('CMake >=3.24 required; no attempt directory created')
    deps=subprocess.check_output([python,'-c','import numpy, pinocchio, pytest, pybind11, torch, sys, sysconfig; from pathlib import Path; assert Path(sysconfig.get_path("include"), "Python.h").is_file(), "Python development headers missing"; assert Path(pybind11.get_cmake_dir(), "pybind11Config.cmake").is_file(), "pybind11 CMake package missing"; print(sys.executable); print(numpy.__version__); print(pinocchio.__version__); print(pybind11.get_cmake_dir())'],text=True)
    selection=target.select_target(device)
    if work.exists():raise FileExistsError('Fresh preparation directory required; failed evidence retained: '+str(work))
    for commit in [RUNTIME,SCIENCE]:
        subprocess.run(['git','-C',str(repo),'cat-file','-e',commit+'^{commit}'],check=True)
        subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',commit,'HEAD'],check=True)
    return tools,deps,selection

def prepare(repo,work,python,device=None):
    verify_package()
    repo=repo.expanduser().resolve();work=work.expanduser().absolute()
    # Preserve venv identity: do not resolve its executable symlink.
    python=str(Path(shutil.which(python) or python).expanduser().absolute())
    allowed=ROOT/'example_artifacts/jetson_f01ab'
    if not work.resolve().is_relative_to(allowed.resolve()):raise ValueError('All artifacts must remain under '+str(allowed))
    for marker in allowed.glob('*/launch-state.json'):
        if json.loads(marker.read_text()).get('status') in ['running','held']:
            raise RuntimeError('Previous native preparation/campaign is held or interrupted; preserve and review: '+str(marker))
    tools,deps,selection=check_prerequisites(repo,work,python,allowed,device)
    work.mkdir(parents=True);logs=work/'logs';logs.mkdir();source=work/'source'
    (work/'tmp').mkdir();(work/'environment-cache').mkdir()
    os.environ.update(TMPDIR=str(work/'tmp'),XDG_CACHE_HOME=str(work/'environment-cache'),PYTHONDONTWRITEBYTECODE='1')
    write(work/'tool-environment.json',dict(tools=tools,python_dependencies=deps,target_selection=selection,disk_usage_bytes=dict(zip(['total','used','free'],shutil.disk_usage(work)))))
    write(work/'target-selection.json',selection)
    write(work/'launch-state.json',dict(status='running',stage='builds_and_suite'))
    started=time.time()
    record=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),started_unix_s=started,status='started',machine_claim='jetson',platform=platform.platform(),python=python,
                runtime_source=RUNTIME,science_source=SCIENCE,cuda_arch='87-real',commands=[],target_review='PENDING',canonical_A_readiness='NOT_RUN',matrix_calls=0)
    try:
        command=['git','clone','--shared','--no-checkout',str(repo),str(source)]
        record['commands'].append(command);run(command,repo,logs/'source-clone.log')
        command=['git','-C',str(source),'checkout','--detach',RUNTIME]
        record['commands'].append(command);run(command,repo,logs/'source-checkout.log')
        assert sha(source/'experiments/frequency_campaign/inputs/SHA256SUMS.json')==BANK
        for N in [16,32,64,128]:
            build=work/'builds'/f'N{N}'
            command=[python,'-B',str(source/'experiments/frequency_campaign/build_isolated.py'),'--source',str(source),'--build',str(build),'--python',python,'--cuda-arch','87-real','--N',str(N)]
            record['commands'].append(command);run(command,source,logs/f'build-N{N}.log')
        build=work/'builds/N32';archive=build/'campaign-source';native=build/'campaign-native'
        receipt=json.loads((build/'f01-build-N32.json').read_text());module=Path(receipt['module']['path'])
        before=sha(module)
        record['module_before_sha256']=before
        cache=native/'CMakeCache.txt';snapshots=work/'suite';snapshots.mkdir()
        cache_before=snapshots/'CMakeCache.before.txt';shutil.copy2(cache,cache_before)
        if sha(cache_before)!=receipt['cmake_cache']['sha256']:raise RuntimeError('N32 cache differs from pinned builder receipt before Make')
        actual=archive/'test-artifacts/tiago-tasks';retained=snapshots/'raw'
        record.update(suite_actual_artifact_root=str(actual),suite_retained_artifact_root=str(retained),N32_cache_before=dict(path=str(cache_before),sha256=sha(cache_before)))
        git_dir=subprocess.check_output(['git','-C',str(source),'rev-parse','--absolute-git-dir'],text=True).strip()
        # Genuine502 object/index, exact archived502 worktree: truthful external
        # Git metadata for unchanged task worker provenance, never a fabricated Git repo.
        env=dict(os.environ,GIT_DIR=git_dir,GIT_WORK_TREE=str(archive),GATO_BUILD_DIR=str(native),
                 GATO_TEST_ARTIFACT_DIR=str(work/'suite/raw'),PYTHONDONTWRITEBYTECODE='1')
        command=['make','test-tiago-tasks'];record['commands'].append(command)
        record['suite_environment']={k:env[k] for k in ['GIT_DIR','GIT_WORK_TREE','GATO_BUILD_DIR','GATO_TEST_ARTIFACT_DIR','PYTHONDONTWRITEBYTECODE']}
        try:
            run(command,archive,logs/'make-test-tiago-tasks.log',env)
            record['suite_exit_code']=0
        except subprocess.CalledProcessError as error:
            record['suite_exit_code']=error.returncode
            raise
        finally:
            record['module_after_sha256']=sha(module) if module.is_file() else None
            cache_after=snapshots/'CMakeCache.after.txt';shutil.copy2(cache,cache_after)
            record['N32_cache_after']=dict(path=str(cache_after),sha256=sha(cache_after))
            # CMake overrides the environment artifact path. Preserve failed
            # artifacts too, without replacing a Make failure with a retry.
            if actual.exists():shutil.copytree(actual,retained)
        summaries=retain_suite(actual,retained)
        after=sha(module)
        if before!=after:raise RuntimeError('Required suite changed N32 module; retain evidence and hold')
        record.update(status='prepared_FOR_AUTOMATED_TARGET_VALIDATION',suite_exit_code=0,suite_fixed_task_processes=3,
                      module_before_sha256=before,module_after_sha256=after,
                      suite_summaries=[dict(path=str(p),sha256=sha(p)) for p in summaries],
                      build_receipts=[dict(path=str(work/'builds'/f'N{N}'/f'f01-build-N{N}.json'),sha256=sha(work/'builds'/f'N{N}'/f'f01-build-N{N}.json')) for N in [16,32,64,128]],
                      N32_cache_after_suite_sha256=sha(native/'CMakeCache.txt'),
                      note='Actual suite/source/cache receipts retained; reviewed automated target checks follow before canonical preflight/matrix.')
    except BaseException as error:
        record.update(status='FAILED_HELD',error_type=type(error).__name__,error=str(error))
        write(work/'launch-state.json',dict(status='held',stage='builds_and_suite',error=str(error)))
        raise
    finally:
        record['finished_unix_s']=time.time();record['elapsed_wall_s']=record['finished_unix_s']-started
        record['logs']=[dict(path=str(p),sha256=sha(p)) for p in logs.iterdir() if p.is_file()]
        write(work/'preparation.json',record)
    full_run.run_campaign(source,work,python,sha,write,run)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['jetson'],default=None,help='Explicit Jetson target for containers hiding host device-tree; actual suite SM87 remains required')
    parser.add_argument('--check',action='store_true',help='CPU package integrity check only')
    parser.add_argument('--repo',type=Path,default=ROOT,help='Existing Git checkout (default: this repository)')
    parser.add_argument('--work',type=Path,default=ROOT/'example_artifacts/jetson_f01ab'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'),help='Fresh attempt under this worktree/example_artifacts/jetson_f01ab')
    resume=parser.add_mutually_exclusive_group()
    resume.add_argument('--reuse-qualified',type=Path,help='Fresh setup-correction cohort reusing only the qualified Jetson pre-owner screening failure')
    resume.add_argument('--continue-prepared',type=Path,help='Continue only a successful prior preparation under example_artifacts, without rebuilding/repeating Make')
    parser.add_argument('--python',default=sys.executable,help='Existing working GATO Python/venv executable; preserve its path')
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda signum,frame: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    if args.check:verify_package();print('Package hashes/source authority checked. No build/GPU performed.')
    else:
        if args.reuse_qualified:
            verify_package()
            reuse.reuse_qualified(args.reuse_qualified,args.work,args.python,sha,write,run,device=args.device)
        elif args.continue_prepared:
            work=args.continue_prepared.expanduser().resolve()
            if not work.is_relative_to((ROOT/'example_artifacts/jetson_f01ab').resolve()):parser.error('Prepared artifacts must be under this worktree/example_artifacts/jetson_f01ab')
            verify_package()
            full_run.continue_prepared(work,args.python,sha,write,run,device=args.device)
        else:prepare(args.repo,args.work,args.python,device=args.device)
