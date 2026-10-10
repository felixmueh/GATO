"""CPU target-selection checks; no CUDA, native imports, builds or task runs."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import target as T
import prepare_jetson as P
import full_run as F


@pytest.fixture
def hidden_model(monkeypatch):
    original=Path.read_text
    def hidden(path,*args,**kwargs):
        if str(path)=='/proc/device-tree/model':raise FileNotFoundError('container hides device-tree')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',hidden)
    monkeypatch.setattr(T.platform,'machine',lambda:'aarch64')


def test_explicit_target_with_hidden_model(hidden_model):
    selection=T.select_target('jetson')
    assert selection['target']=='jetson' and selection['selection']=='explicit_user_selection'
    assert selection['host_architecture']=='aarch64' and selection['device_tree_model'] is None
    assert 'FileNotFoundError' in selection['device_tree_read_error']


def test_omitted_target_keeps_host_detection(hidden_model):
    with pytest.raises(RuntimeError,match='--device jetson'):T.select_target()


def test_auto_host_orin_detection(monkeypatch):
    monkeypatch.setattr(T.Path,'read_text',lambda *a,**k:'NVIDIA Jetson Orin\x00')
    assert T.select_target()['selection']=='host_Orin_model_detection'


def test_optional_model_permission_error(monkeypatch):
    def unreadable(*args,**kwargs):raise PermissionError('device-tree inaccessible')
    monkeypatch.setattr(T.Path,'read_text',unreadable)
    assert 'PermissionError' in T.select_target('jetson')['device_tree_read_error']
    with pytest.raises(RuntimeError,match='--device jetson'):T.select_target()


@pytest.mark.parametrize('device',['jetson',None])
def test_selection_before_attempt_or_build(tmp_path,monkeypatch,hidden_model,device):
    monkeypatch.setattr(P,'ROOT',tmp_path);monkeypatch.setattr(P,'verify_package',lambda:None)
    monkeypatch.setattr(P.shutil,'which',lambda name:'/fake/'+name)
    monkeypatch.setattr(P.subprocess,'check_output',lambda *a,**k:'cmake version 3.26.1\n')
    monkeypatch.setattr(P.subprocess,'run',lambda *a,**k:None)
    calls=[]
    def stop_after_selection(argv,cwd,log,env=None):
        calls.append(argv);raise RuntimeError('fake stop before build')
    monkeypatch.setattr(P,'run',stop_after_selection)
    work=tmp_path/'example_artifacts/jetson_f01ab/attempt'
    with pytest.raises(RuntimeError,match='fake stop before build' if device else '--device jetson'):
        P.prepare(tmp_path,work,'python3',device=device)
    if device:
        assert len(calls)==1 and '--no-checkout' in calls[0]
        assert json.loads((work/'target-selection.json').read_text())['selection']=='explicit_user_selection'
    else:assert not work.exists() and calls==[]


def test_continuation_explicit_target_no_native_rerun(tmp_path,monkeypatch,hidden_model):
    python=str(Path(sys.executable).absolute())
    (tmp_path/'preparation.json').write_text(json.dumps(dict(status='prepared_TARGET_REVIEW_REQUIRED',runtime_source=F.RUNTIME,science_source=F.SCIENCE,cuda_arch='87-real',python=python,build_receipts=[],suite_summaries=[])))
    monkeypatch.setattr(F.subprocess,'run',lambda *a,**k:None)
    observed=[]
    monkeypatch.setattr(F,'run_campaign',lambda *a:observed.append(a))
    for name in ['TMPDIR','XDG_CACHE_HOME','PYTHONDONTWRITEBYTECODE']:monkeypatch.setenv(name,'fake_original')
    F.continue_prepared(tmp_path,python,P.sha,P.write,lambda *a:pytest.fail('build/task rerun'),device='jetson')
    assert len(observed)==1
    assert json.loads((tmp_path/'target-selection-continuation.json').read_text())['selection']=='explicit_user_selection'
