"""CPU evidence tests for frozen-bank reuse and the compiled/collector split."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types

import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import full_run as F
import prepare_jetson as P
import reuse as U


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

B=load(ROOT/'experiments/frequency_campaign/runtime_bank.py','runtime_bank_test')


def dump(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_runtime_bank_uses_saved_screen_only(monkeypatch):
    # Pinocchio/RNEA is never imported; use of the unchanged generator's screen
    # function would fail even on this development machine.
    monkeypatch.setitem(sys.modules,'pinocchio',types.SimpleNamespace())
    bank=B.validate(ROOT/'experiments/frequency_campaign/inputs')
    assert bank['variants']['short12']['counts']==dict(cold=12,warm=222,goal_switch=6)
    source=(ROOT/'experiments/frequency_campaign/runtime_bank.py').read_text()
    assert 'screen(' not in source and 'rnea' not in source and 'pinocchio_version' not in source


@pytest.mark.parametrize('changed',['arrays.npz','screening.json','bank.json','SHA256SUMS.json'])
def test_runtime_bank_rejects_changed_frozen_bytes(tmp_path,changed):
    bank=tmp_path/'bank';shutil.copytree(ROOT/'experiments/frequency_campaign/inputs',bank)
    with (bank/changed).open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError):B.validate(bank)


@pytest.fixture
def parent(tmp_path,monkeypatch):
    old=tmp_path/'original';old.mkdir()
    release=dict(status='released_F01_A',collector_source_commit=F.RUNTIME,modules={},build_review=dict(path=str(old/'review.json'),sha256='checked by old validator'))
    dump(old/'jetson-release.json',release);monkeypatch.setattr(U,'OLD_RELEASE_SHA256',P.sha(old/'jetson-release.json'))
    dump(old/'launch-state.json',dict(status='held',stage='canonical_A_preflight'))
    metrics=old/'readiness/f01-raw/metrics.json'
    dump(metrics,dict(campaign_id='F01-A-preflight',machine='jetson',timing_mode='A',execution_backend='internal',task_success=None,physical_pass=None,execution_error='Screening data changed or failed',outcome='execution_failed',evidence_kind='measurement'))
    (metrics.parent/'raw').mkdir();dump(metrics.parent/'SHA256SUMS.json',{'metrics.json':P.sha(metrics)})
    dump(old/'family-state.json',dict(status='held',machine='jetson',release_manifest_sha256=U.OLD_RELEASE_SHA256,readiness_metrics=F.pin(metrics,P.sha)))
    dump(old/'preparation.json',dict(python=str(Path(sys.executable).absolute()),runtime_source=F.RUNTIME,science_source=F.SCIENCE,cuda_arch='87-real',build_receipts=[],suite_summaries=[]))
    dump(old/'review.json',dict(policy=[]))
    old_validator=types.SimpleNamespace(validate=lambda *a:None)
    monkeypatch.setattr(F,'load',lambda *a:old_validator)
    return old,old_validator


def test_selective_parent_acceptance_preserves_old_marks(parent):
    old,_=parent;before=(old/'launch-state.json').read_bytes()
    accepted=U.validate_parent(old,P.sha)
    assert accepted['release']['sha256']==U.OLD_RELEASE_SHA256
    assert (old/'launch-state.json').read_bytes()==before


@pytest.mark.parametrize('damage',['other_error','native_event','matrix','missing_receipt','changed_module'])
def test_selective_parent_refuses_other_failures(parent,damage):
    old,validator=parent;metrics=old/'readiness/f01-raw/metrics.json'
    if damage=='other_error':
        value=json.loads(metrics.read_text());value['execution_error']='native abort';dump(metrics,value)
        family=json.loads((old/'family-state.json').read_text());family['readiness_metrics']=F.pin(metrics,P.sha);dump(old/'family-state.json',family)
        dump(metrics.parent/'SHA256SUMS.json',{'metrics.json':P.sha(metrics)})
    if damage=='native_event':(metrics.parent/'events.jsonl').write_text('{"event":"solve_started"}\n')
    if damage=='matrix':(old/'results').mkdir()
    if damage=='missing_receipt':(old/'family-state.json').unlink()
    if damage=='changed_module':
        def changed(*a):raise ValueError('Actual native module differs')
        validator.validate=changed
    with pytest.raises((ValueError,OSError)):U.validate_parent(old,P.sha)


@pytest.fixture
def split_release(tmp_path):
    R=load(ROOT/'experiments/frequency_campaign/release.py','split_release_test')
    collector=tmp_path/'collector';collector.mkdir()
    for name in (*R.RUNTIME,*R.COLLECTOR,'experiments/frequency_campaign/selected_variant.json'):
        target=collector/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    R.ROOT=collector
    acceptance=HERE/'global-authority/telemetry-acceptance-decision.json';decision=json.loads(acceptance.read_text())
    review=tmp_path/'review.json';dump(review,dict(verdict='approved'))
    suite=tmp_path/'suite.json';dump(suite,dict(status='passed',exit_code=0,fixed_task_processes=3,error=None,restored_patch_sha256=decision['accepted_native_patch_sha256'],module_before_sha256='same',module_after_sha256='same'))
    modules={}
    for n in [16,32,64,128]:
        archive=tmp_path/f'N{n}/source';archive.mkdir(parents=True)
        for name in R.RUNTIME:
            target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        module=archive/f'python/bsqp/bsqpN{n}_tiago_right.so';module.write_bytes(b'unchanged fake native')
        cache=tmp_path/f'N{n}/CMakeCache.txt';cache.write_text(f'GATO_EXPERIMENT_TELEMETRY:BOOL=ON\nKNOTS:STRING={n}\nPLANT:STRING=tiago_right\n')
        source=tmp_path/f'N{n}/source.json';dump(source,dict(source_commit=F.RUNTIME,source_file_sha256={name:P.sha(archive/name) for name in R.RUNTIME}))
        receipt=tmp_path/f'N{n}/build.json';dump(receipt,dict(variant='ON',source_commit=F.RUNTIME,source_root=str(archive),flags=dict(GATO_EXPERIMENT_TELEMETRY=True,N=n,plant='tiago_right'),module=F.pin(module,P.sha),cmake_cache=F.pin(cache,P.sha),source_receipt=F.pin(source,P.sha),source_file_sha256={name:P.sha(archive/name) for name in R.RUNTIME}))
        modules[str(n)]=dict(path=str(module),sha256=P.sha(module),build_receipt=F.pin(receipt,P.sha))
    manifest=tmp_path/'release.json';dump(manifest,dict(status='released_F01_A',machine='jetson',production_core_commit=F.CORE,compiled_source_commit=F.RUNTIME,collector_source_commit='1'*40,collector_source_sha256={name:P.sha(collector/name) for name in R.COLLECTOR},bank_manifest_sha256=R.P.BANK_MANIFEST_SHA256,selection_sha256=P.sha(collector/'experiments/frequency_campaign/selected_variant.json'),observer_patch_sha256=decision['accepted_native_patch_sha256'],telemetry_acceptance=F.pin(acceptance,P.sha),reason_review=F.pin(HERE/'global-authority/telemetry-saved-review.json',P.sha),build_review=F.pin(review,P.sha),task_suite=dict(F.pin(suite,P.sha),machine='jetson'),modules=modules))
    return R,manifest,collector


def test_compiled_and_collector_source_split(split_release):
    R,manifest,collector=split_release
    context=R.validate(manifest,P.sha(manifest),'jetson',64)
    assert context['compiled_source_commit']==F.RUNTIME and context['collector_source_commit']=='1'*40
    assert len(context['collector_source_sha256'])==9


def test_compiled_runtime_must_match_collector(split_release):
    R,manifest,collector=split_release
    (collector/R.RUNTIME[0]).write_text('changed runtime')
    with pytest.raises(ValueError,match='Runtime source bytes differ'):R.validate(manifest,P.sha(manifest),'jetson',32)


def test_changed_native_module_rejected(split_release):
    R,manifest,collector=split_release
    value=json.loads(manifest.read_text());Path(value['modules']['32']['path']).write_bytes(b'changed native')
    with pytest.raises(ValueError,match='Actual native module differs'):R.validate(manifest,P.sha(manifest),'jetson',32)


def test_actual_overwritten_family_has_no_metrics_pointer(parent):
    old,_=parent
    family=json.loads((old/'family-state.json').read_text());family.pop('readiness_metrics');family.update(reason='Command returned non-zero exit status 1',failed_attempt=str(old));dump(old/'family-state.json',family)
    assert U.validate_parent(old,P.sha)['failed_readiness']['sha256']==P.sha(old/'readiness/f01-raw/metrics.json')
