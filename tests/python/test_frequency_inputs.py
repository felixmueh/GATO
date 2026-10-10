"""CPU provenance/schedule/witness tests; no native/controller import."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('frequency_freeze',ROOT/'experiments/frequency_campaign/freeze_inputs.py')
f=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(f)


@pytest.fixture(scope='module')
def frozen(tmp_path_factory):
    path=tmp_path_factory.mktemp('frequency')/'bank'
    f.freeze(path)
    return path


def rehash(path,name):
    p=path/'SHA256SUMS.json';manifest=json.loads(p.read_text());manifest[name]=f.sha(path/name);f.write_json(p,manifest)


def test_full_and_first_two_subset_counts(frozen):
    bank=f.validate(frozen)
    full,short=(bank['variants'][key] for key in ['full24','short12'])
    assert full['counts']=={'cold':24,'warm':444,'goal_switch':12}
    assert short['counts']=={'cold':12,'warm':222,'goal_switch':6}
    assert short['episodes']==full['episodes'][:12]
    assert short['start_indices']==[0,1]
    assert len(bank['N'])*len(bank['SQP'])*len(bank['PCG'])==64
    for episode in full['episodes']:
        assert episode['categories'][0]=='cold' and len(episode['categories'])==20
        if episode['kind']=='reaching':
            assert episode['categories'][10]=='goal_switch'
            assert episode['goal_switch_before_update']==10
        else:assert episode['categories'].count('goal_switch')==0


def test_exact_arrays_and_global_references(frozen):
    task,endpoints=f.source_inputs();expected,_=f.arrays_and_schedules(task,endpoints)
    with np.load(frozen/'arrays.npz',allow_pickle=False) as arrays:
        assert arrays['starts'].shape==(4,14) and arrays['starts'].dtype==np.dtype('<f4')
        assert np.count_nonzero(arrays['starts'][:,7:])==0
        for name,value in expected.items():assert arrays[name].tobytes()==value.tobytes()
        for i,t in enumerate(task['tasks'][3:]):
            assert arrays[f'reference_{i}'].dtype==np.dtype('<f8')
            assert f.hashlib.sha256(arrays[f'reference_{i}'].tobytes()).hexdigest()==t['reference_sha256']
        assert all(not np.array_equal(arrays['warmup_state'],s) for s in arrays['starts'])
    warmup=json.loads((frozen/'bank.json').read_text())['warmup']
    assert warmup['excluded_from_measured_categories'] is True


def test_sampled_witness_scope_and_positive_margins(frozen):
    report=json.loads((frozen/'screening.json').read_text())
    assert report['all_routes_pass'] and len(report['routes'])==15
    assert report['maximum_sample_step_s']<=.003+1e-12
    assert min(min(row) for row in report['start_goal_distance_m'])>.05
    assert report['start_goal_distance_m'][1][0]<.1
    assert report['qualifies_T01_or_canonical'] is False
    for route in report['routes']:
        assert min(route['margins'].values())>0
        assert 0<=route['necessary_velocity_time_bound_s']<8
    assert 'tracking transient/dynamic feasibility' in report['unqualified']
    with np.load(frozen/'witnesses.npz',allow_pickle=False) as w:
        assert w['time_s'][0]==0 and w['time_s'][-1]==8
        for name in ['start_0_goal_0','switch_goal_2_to_0']:
            assert np.max(abs(w[name+'_v'][[0,-1]]))<1e-12
            assert np.max(abs(w[name+'_a'][[0,-1]]))<1e-12


def test_refuse_overwrite_preserves_bytes(frozen):
    before=f.sha(frozen/'arrays.npz')
    with pytest.raises(FileExistsError):f.freeze(frozen)
    assert f.sha(frozen/'arrays.npz')==before


@pytest.mark.parametrize('mutation',['schedule','caps','reference','model','witness','false_screen'])
def test_mutation_rejected_even_with_rehashed_manifest(frozen,tmp_path,mutation):
    path=tmp_path/'altered';shutil.copytree(frozen,path)
    if mutation in ['schedule','caps']:
        p=path/'bank.json';bank=json.loads(p.read_text())
        if mutation=='schedule':bank['variants']['full24']['episodes'][0]['categories'][10]='warm'
        else:bank['PCG'][0]=120
        f.write_json(p,bank);name='bank.json'
    elif mutation=='reference':
        with np.load(path/'arrays.npz',allow_pickle=False) as saved: arrays={k:saved[k].copy() for k in saved.files}
        arrays['reference_0'][0]+=.001;np.savez_compressed(path/'arrays.npz',**arrays);name='arrays.npz'
    elif mutation=='model':
        p=path/'model.urdf';p.write_text(p.read_text().replace('name="tiago_right_arm"','name="changed"')+'\n');name='model.urdf'
    elif mutation=='witness':
        with np.load(path/'witnesses.npz',allow_pickle=False) as saved: arrays={k:saved[k].copy() for k in saved.files}
        arrays['start_0_goal_0_u'][0,0]+=.01;np.savez_compressed(path/'witnesses.npz',**arrays);name='witnesses.npz'
    else:
        p=path/'screening.json';report=json.loads(p.read_text());report['all_routes_pass']=False;f.write_json(p,report);name='screening.json'
    rehash(path,name)
    with pytest.raises(ValueError):f.validate(path)


def test_failed_input_preparation_is_preserved(tmp_path):
    root=tmp_path/'changed-source'
    for relative in [f.TASK,f.ENDPOINTS,f.MODEL]:
        p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,p)
    (root/f.TASK).write_text('{}\n');output=tmp_path/'failed'
    with pytest.raises(ValueError,match='Frozen source input/model changed'):f.freeze(output,root)
    receipt=json.loads((output/'preparation_failure.json').read_text())
    assert receipt['gpu_calls']==0 and receipt['type']=='ValueError'
    assert (output/'SHA256SUMS.json').exists()
    with pytest.raises(FileExistsError):f.freeze(output,root)


def test_no_controller_or_native_imports():
    # Repository pytest conftest imports bsqp; audit this utility in a fresh process.
    code = """
import importlib.util,sys
spec=importlib.util.spec_from_file_location('freeze',sys.argv[1])
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
t,e=f.source_inputs();a,_=f.arrays_and_schedules(t,e);f.screen(a,f.ROOT/f.MODEL)
assert 'bsqp.interface' not in sys.modules
assert 'gato_tiago.tiago_mpc_controller' not in sys.modules
assert not any(name.startswith('bsqp.bsqpN') for name in sys.modules)
"""
    subprocess.run([sys.executable,'-c',code,str(ROOT/'experiments/frequency_campaign/freeze_inputs.py')],check=True)
