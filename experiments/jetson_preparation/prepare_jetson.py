#!/usr/bin/env python3
"""Prepare reviewed F01ab sources on Jetson; stop before campaign release.

Four sequential isolated87-real builds + ONE unchanged N32 ON task suite.
No target review/readiness is invented. No parameter changes or automatic retries.
"""
import argparse,hashlib,json,os,platform,shutil,subprocess,sys
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parents[1]
RUNTIME='502fa29c8b0d649dcb347f9534bf29c6935fa6dd'
SCIENCE='9cbf25721056e5b9dae1a5daf9e24982849b0573'
BANK='29f8ccdff73d36c339e50ea7153a46fb6ecd22b32da786b74f16cb9e1a61f609'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def verify_package():
    for name,digest in json.loads((PACKAGE/'SHA256SUMS.json').read_text()).items():
        path=(PACKAGE/name).resolve()
        if not path.is_relative_to(PACKAGE.resolve()) or sha(path)!=digest:raise ValueError('Package changed: '+name)
    decision=json.loads((PACKAGE/'global-authority/telemetry-acceptance-decision.json').read_text())
    if decision['decision']!='accept_existing_first_instrumentation_without_equivalence_gate':raise ValueError('Observer authority differs')
    if decision['comparison_audit']['sha256']!=sha(PACKAGE/'global-authority/telemetry-saved-review.json'):raise ValueError('Accepted reason audit differs')

def run(command,cwd,log,env=None):
    print('Running:',repr(command),'log:',str(log),flush=True)
    with log.open('w') as stream:
        subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def retain_suite(actual,retained):
    """Copy actual fixed CMake artifact directory, preserving its originals."""
    if actual.exists() and not retained.exists():
        retained.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(actual,retained)
    summaries=list(retained.glob("*/summary.json"))
    if len(summaries)!=3 or not all(json.loads(p.read_text()).get("passed") is True for p in summaries):
        raise RuntimeError("Required suite did not retain exactly3 passing task summaries")
    return summaries

def prepare(repo,work,python):
    verify_package()
    repo=repo.expanduser().resolve();work=work.expanduser().absolute()
    # Preserve venv identity: do not resolve its executable symlink.
    python=str(Path(shutil.which(python) or python).expanduser().absolute())
    if work.exists():raise FileExistsError('Fresh preparation directory required; failed evidence retained: '+str(work))
    for commit in [RUNTIME,SCIENCE]:subprocess.run(['git','-C',str(repo),'cat-file','-e',commit+'^{commit}'],check=True)
    work.mkdir(parents=True);logs=work/'logs';logs.mkdir();source=work/'source'
    record=dict(status='started',machine_claim='jetson',platform=platform.platform(),python=python,
                runtime_source=RUNTIME,science_source=SCIENCE,cuda_arch='87-real',commands=[],target_review='PENDING',canonical_A_readiness='NOT_RUN',matrix_calls=0)
    try:
        command=['git','-C',str(repo),'worktree','add','--detach',str(source),RUNTIME]
        record['commands'].append(command);run(command,repo,logs/'worktree.log')
        assert sha(source/'experiments/frequency_campaign/inputs/SHA256SUMS.json')==BANK
        for N in [16,32,64,128]:
            build=work/'builds'/f'N{N}'
            command=[python,'-B',str(source/'experiments/frequency_campaign/build_isolated.py'),'--source',str(source),'--build',str(build),'--python',python,'--cuda-arch','87-real','--N',str(N)]
            record['commands'].append(command);run(command,source,logs/f'build-N{N}.log')
        build=work/'builds/N32';archive=build/'campaign-source';native=build/'campaign-native'
        receipt=json.loads((build/'f01-build-N32.json').read_text());module=Path(receipt['module']['path'])
        before=sha(module)
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
        finally:
            cache_after=snapshots/'CMakeCache.after.txt';shutil.copy2(cache,cache_after)
            record['N32_cache_after']=dict(path=str(cache_after),sha256=sha(cache_after))
            # CMake overrides the environment artifact path. Preserve failed
            # artifacts too, without replacing a Make failure with a retry.
            if actual.exists():shutil.copytree(actual,retained)
        summaries=retain_suite(actual,retained)
        after=sha(module)
        if before!=after:raise RuntimeError('Required suite changed N32 module; retain evidence and hold')
        record.update(status='prepared_TARGET_REVIEW_REQUIRED',suite_exit_code=0,suite_fixed_task_processes=3,
                      module_before_sha256=before,module_after_sha256=after,
                      suite_summaries=[dict(path=str(p),sha256=sha(p)) for p in summaries],
                      build_receipts=[dict(path=str(work/'builds'/f'N{N}'/f'f01-build-N{N}.json'),sha256=sha(work/'builds'/f'N{N}'/f'f01-build-N{N}.json')) for N in [16,32,64,128]],
                      N32_cache_after_suite_sha256=sha(native/'CMakeCache.txt'),
                      note='Suite/source/cache receipts need target saved review. No preflight, release manifest, enabled plan or F01matrix execution is claimed.')
    except BaseException as error:
        record.update(status='FAILED_HELD',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        record['logs']=[dict(path=str(p),sha256=sha(p)) for p in logs.iterdir() if p.is_file()]
        write(work/'preparation.json',record)
    print('Preparation complete. Send preparation.json, logs and suite/build receipts for review; full matrix remains held.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='CPU package integrity check only')
    parser.add_argument('--repo',type=Path,default=ROOT,help='Existing Git checkout (default: this repository)')
    parser.add_argument('--work',type=Path,default=Path.home()/'gato-jetson-f01ab-preparation-20261010',help='Fresh persistent preparation directory')
    parser.add_argument('--python',default=sys.executable,help='Existing working GATO Python/venv executable; preserve its path')
    args=parser.parse_args()
    if args.check:verify_package();print('Package hashes/source authority checked. No build/GPU performed.')
    else:
        prepare(args.repo,args.work,args.python)
