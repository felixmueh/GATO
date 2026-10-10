"""CPU fake pipeline exercises launch order, evidence guards and permanent holds."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import prepare_jetson as P
import full_run as F


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def raw_cell(results,eid,damage=None):
    attempt=results/'jetson'/eid/'identity'/'attempt-001'
    raw=attempt/'f01-raw';raw.mkdir(parents=True)
    episodes=[dict(execution_error=None,attempted_solves=20,completed_updates=20,missing_solve_slots=[],missing_update_slots=[]) for _ in range(12)]
    metrics=dict(execution_error=None,outcome='timing_cell_completed',native_calls=241,episodes=episodes,missing_episodes=[])
    dump(attempt/'receipt.json',dict(status='completed'))
    dump(attempt/'metrics.json',metrics);dump(raw/'metrics.json',metrics)
    dump(raw/'protocol.json',{});dump(raw/'provenance.json',{})
    events=[]
    for category,count in [('cold',12),('warm',222),('goal_switch',6),('process_cuda_warmup',1)]:
        for _ in range(count):
            for event in ['solve_started','native_return','solve_completed']:
                name=f'raw/{len(events):07d}-{event}.npz';file=raw/name;file.parent.mkdir(exist_ok=True);file.write_bytes(b'fake retained arrays')
                events.append(dict(event_index=len(events),event=event,raw_file=name,sha256=P.sha(file),record=dict(category=category)))
    (raw/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in events)+'\n')
    dump(raw/'SHA256SUMS.json',{str(x.relative_to(raw)):P.sha(x) for x in raw.rglob('*') if x.is_file()})
    if damage=='missing':(raw/events[0]['raw_file']).unlink()
    if damage=='hash':(raw/events[0]['raw_file']).write_bytes(b'changed')
    if damage=='worker':dump(attempt/'receipt.json',dict(status='failed'))
    if damage=='abort':(attempt/'receipt.json').unlink()
    if damage=='category':
        events[0]['record']['category']='unknown'
        # Change one completed record, then rehash final manifest.
        events[2]['record']['category']='unknown'
        (raw/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in events)+'\n')
        dump(raw/'SHA256SUMS.json',{str(x.relative_to(raw)):P.sha(x) for x in raw.rglob('*') if x.is_file() and x.name!='SHA256SUMS.json'})


@pytest.fixture
def pipeline(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir();work=repo/'example_artifacts/jetson_f01ab/attempt'
    monkeypatch.setattr(P,'BANK',hashlib.sha256(b'{}').hexdigest());monkeypatch.setattr(P,'ROOT',repo);monkeypatch.setattr(P,'verify_package',lambda:None)
    monkeypatch.setattr(P,'check_prerequisites',lambda *a:({},'fake dependencies','Orin fake CPU pipeline'))
    monkeypatch.setattr(P.subprocess,'check_output',lambda *a,**k:str(work/'source/.git')+'\n')
    calls=[];failure={'stage':None}
    cases=[dict(variant='short12',N=n,SQP=s,PCG=p,timing_mode='A',backend='internal') for n in [16,32,64,128] for s in [1,2,5,10] for p in [100,200,400,800]]
    class Release:
        COLLECTOR=()
        def selected_cases(self):return cases
        def validate(self,*args):return {'machine':'jetson'}
        def readiness_guard(self,context,family):
            if failure['stage']=='preflight' or json.loads(family.read_text())['status']!='ready':raise RuntimeError('preflight task/physical failed')
        def write_state(self,path,context,status,**kw):dump(path,dict(status=status,**kw))
    release=Release()
    def qualify(source,work,sha,write):
        manifest=work/'jetson-release.json';dump(manifest,dict(fake=True));return release,manifest
    monkeypatch.setattr(F,'qualify',qualify)
    def command(argv,cwd,log,env=None):
        calls.append(argv);log.parent.mkdir(parents=True,exist_ok=True);log.write_text('')
        if '--no-checkout' in argv:
            source=Path(argv[-1]);source.mkdir(parents=True);(source/'experiments').mkdir()
            if source.name=='source':
                inputs=source/'experiments/frequency_campaign/inputs';dump(inputs/'SHA256SUMS.json',{})
                dump(inputs.parent/'selected_variant.json',{})
            return
        if '--N' in argv:
            n=int(argv[argv.index('--N')+1]);build=Path(argv[argv.index('--build')+1]);module=build/'campaign-source/python/bsqp/module.so';module.parent.mkdir(parents=True);module.write_bytes(b'fake module')
            cache=build/'campaign-native/CMakeCache.txt';cache.parent.mkdir();cache.write_text('fake cache')
            dump(build/f'f01-build-N{n}.json',dict(module=dict(path=str(module),sha256=P.sha(module)),cmake_cache=dict(path=str(cache),sha256=P.sha(cache))))
            if failure['stage']=='build':raise subprocess.CalledProcessError(1,argv)
        elif argv==['make','test-tiago-tasks']:
            for name in ['short_reach','long_reach','figure8_tracking']:dump(cwd/'test-artifacts/tiago-tasks'/name/'summary.json',dict(case=name,passed=failure['stage']!='suite'))
            if failure['stage']=='suite':raise subprocess.CalledProcessError(1,argv)
        elif '--readiness' in argv:dump(work/'family-state.json',dict(status='ready'))
        elif '--dry-run' in argv:log.write_text('\n'.join(f"pending: F01ab-N{c['N']}-SQP{c['SQP']}-PCG{c['PCG']}/F01ab-N{c['N']}-SQP{c['SQP']}-PCG{c['PCG']}/0" for c in cases)+'\n')
        elif '--only' in argv:
            eid=argv[-1];raw_cell(work/'results',eid,failure['stage'])
            if failure['stage']=='interrupt':raise KeyboardInterrupt()
    monkeypatch.setattr(P,'run',command)
    return repo,work,calls,failure


def test_complete64_no_second_build_or_suite(pipeline):
    repo,work,calls,failure=pipeline;P.prepare(repo,work,sys.executable)
    state=json.loads((work/'launch-state.json').read_text())
    assert state['status']=='completed' and state['native_calls']==15424 and len(state['cells'])==64
    assert len([x for x in calls if '--N' in x])==4
    assert calls.count(['make','test-tiago-tasks'])==1
    assert len([x for x in calls if '--readiness' in x])==1
    assert len([x for x in calls if '--only' in x])==64
    plan=json.loads((work/'jetson-f01ab-plan.json').read_text())
    assert all(x['build']==dict(commands=[],artifacts=[]) for x in plan['experiments'])
    assert len(json.loads((work/'runner-dry-run.json').read_text())['pending'])==64


@pytest.mark.parametrize('stage',['build','suite','preflight','worker','missing','hash','abort','category','interrupt'])
def test_stop_and_reentry_hold(pipeline,stage):
    repo,work,calls,failure=pipeline;failure['stage']=stage
    with pytest.raises(BaseException):P.prepare(repo,work,sys.executable)
    assert json.loads((work/'launch-state.json').read_text())['status']=='held'
    assert len([x for x in calls if '--only' in x])==(0 if stage in ['build','suite','preflight'] else 1)
    previous=len(calls)
    with pytest.raises(RuntimeError,match='held or interrupted'):P.prepare(repo,work.parent/'next',sys.executable)
    assert len(calls)==previous
    if stage=='suite':
        prep=json.loads((work/'preparation.json').read_text())
        assert prep['suite_exit_code']==1 and prep['module_before_sha256']==prep['module_after_sha256']
        assert len(list((work/'suite/raw').glob('*/summary.json')))==3


def test_continue_refuses_preflight_started(tmp_path):
    dump(tmp_path/'preparation.json',dict(status='prepared_TARGET_REVIEW_REQUIRED',runtime_source=F.RUNTIME,science_source=F.SCIENCE,cuda_arch='87-real'))
    (tmp_path/'readiness').mkdir()
    with pytest.raises(RuntimeError,match='already started'):F.continue_prepared(tmp_path,sys.executable,P.sha,P.write,lambda *a:pytest.fail('native rerun'))


@pytest.mark.parametrize('bad',[None,'sm','source','cache','hash'])
def test_actual_automated_qualification(tmp_path,monkeypatch,bad):
    source=tmp_path/'source';work=tmp_path/'work';source.mkdir();work.mkdir()
    dump(source/'experiments/frequency_campaign/selected_variant.json',{})
    (source/'input.txt').write_text('pinned source')
    monkeypatch.setattr(F.subprocess,'check_output',lambda *a,**k:F.RUNTIME+'\n')
    release=types.SimpleNamespace(COLLECTOR=(),P=types.SimpleNamespace(BANK_MANIFEST_SHA256=P.BANK),validate=lambda *a:None)
    monkeypatch.setattr(F,'load',lambda *a:release)
    for n in [16,32,64,128]:
        build=work/'builds'/f'N{n}';archive=build/'campaign-source';archive.mkdir(parents=True)
        (archive/'input.txt').write_text('pinned source')
        module=archive/'python/bsqp/module.so';module.parent.mkdir(parents=True);module.write_bytes(b'pinned fake native')
        cache=build/'campaign-native/CMakeCache.txt';cache.parent.mkdir();cache.write_text(f'GATO_EXPERIMENT_TELEMETRY:BOOL=ON\nCMAKE_CUDA_ARCHITECTURES:STRING=87-real\nKNOTS:STRING={n}\nPLANT:STRING=tiago_right\nCMAKE_BUILD_TYPE:STRING=Release\n')
        source_receipt=build/'build_source_receipt.json';dump(source_receipt,dict(source_commit=F.RUNTIME))
        dump(build/f'f01-build-N{n}.json',dict(source_commit=F.RUNTIME,variant='ON',flags=dict(GATO_EXPERIMENT_TELEMETRY=True,N=n,plant='tiago_right',cuda_arch='87-real'),source_root=str(archive),module=dict(path=str(module),sha256=P.sha(module)),source_file_sha256={'input.txt':P.sha(source/'input.txt')},cmake_cache=dict(path=str(cache),sha256=P.sha(cache)),source_receipt=dict(path=str(source_receipt),sha256=P.sha(source_receipt))))
        if n==32:
            snapshot=work/'suite/CMakeCache.before.txt';snapshot.parent.mkdir();snapshot.write_bytes(cache.read_bytes())
    archive=work/'builds/N32/campaign-source';worker=archive/'tests/python/tiago_task_worker.py';worker.parent.mkdir(parents=True);worker.write_text('unchanged fake worker')
    summaries=[]
    module=archive/'python/bsqp/module.so'
    for name in ['short_reach','long_reach','figure8_tracking']:
        fixture=archive/'tests/python/tiago_task_cases'/f'{name}.json';dump(fixture,dict(name=name))
        summary=work/'suite/raw'/name/'summary.json';dump(summary,dict(case=name,passed=True));summaries.append(dict(path=str(summary),sha256=P.sha(summary)))
        dump(summary.parent/'receipt.json',dict(source_commit=F.RUNTIME,module_sha256=P.sha(module),module_path=str(module),worker_sha256=P.sha(worker),case_sha256=P.sha(fixture),source_sha256={'input.txt':P.sha(archive/'input.txt')}))
    dump(work/'preparation.json',dict(suite_exit_code=0,module_before_sha256=P.sha(module),module_after_sha256=P.sha(module),suite_summaries=summaries))
    log=work/'logs/make-test-tiago-tasks.log';log.parent.mkdir();log.write_text('cuda_access_smoke_ok=1 sm=87\n')
    if bad=='sm':log.write_text('cuda_access_smoke_ok=1 sm=86\n')
    if bad=='source':
        path=work/'suite/raw/short_reach/receipt.json';data=json.loads(path.read_text());data['source_commit']='0'*40;dump(path,data)
    if bad=='cache':(work/'builds/N64/campaign-native/CMakeCache.txt').write_text('wrong target')
    if bad=='hash':(work/'suite/raw/short_reach/summary.json').write_text('{}')
    if bad:
        with pytest.raises(ValueError):F.qualify(source,work,P.sha,P.write)
    else:
        _,manifest=F.qualify(source,work,P.sha,P.write)
        assert json.loads(manifest.read_text())['machine']=='jetson'
        validation=json.loads((work/'automated-target-validation.json').read_text())
        assert validation['validation_kind']=='automated' and validation['actual_cuda_smoke']=='sm=87'


def test_nested_process_interruption_cleanup(tmp_path):
    """Interrupt the real runner, whose worker owns an independent session."""
    import threading
    import time
    marker=tmp_path/'worker.pid';interrupted=tmp_path/'interrupt.done'
    runner=HERE.parents[1]/'experiments/run.py'
    wrapper=tmp_path/'wrapper.py'
    wrapper.write_text(f"import importlib.util, os, sys\ns=importlib.util.spec_from_file_location('runner',{str(runner)!r});r=importlib.util.module_from_spec(s);s.loader.exec_module(r)\nr.execute([sys.executable,'-c',\"import os,time;open({str(marker)!r},'w').write(str(os.getpid()));time.sleep(60)\"],{str(tmp_path)!r},r.Path({str(tmp_path)!r}),dict(os.environ))\n")
    original_wait=subprocess.Popen.wait
    def interrupt_when_worker_started():
        deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
        # Deliver SIGINT to this pytest thread via CPython main-thread interrupt.
        import signal
        os.kill(os.getpid(),signal.SIGINT)
    thread=threading.Thread(target=interrupt_when_worker_started,daemon=True);thread.start()
    with pytest.raises(KeyboardInterrupt):P.run([sys.executable,'-B',str(wrapper)],tmp_path,tmp_path/'outer.log')
    pid=int(marker.read_text());stat=Path(f'/proc/{pid}/stat')
    assert not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z'
