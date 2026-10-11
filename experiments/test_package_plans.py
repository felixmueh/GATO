"""CPU plan composition; never build, run GPU/ROS, or create a source bundle."""
import json
from pathlib import Path

import pytest

import package_plans


def test_exact_pending_composition_and_explicit_dependencies(tmp_path):
    inventory=package_plans.merge(tmp_path)
    desktop=json.loads((tmp_path/'gtx1070_desktop.json').read_text())
    bb=next(e for e in desktop['experiments'] if e['id']=='BB')
    assert len(bb['cases'])==176
    assert not any('F01' in e['id'] or e['id']=='T01' for e in desktop['experiments'])
    jetson=json.loads((tmp_path/'jetson_proposed.json').read_text())
    assert not any('f01ab' in e['id'].lower() for e in jetson['experiments'])
    assert all(e['machines']==['jetson'] for e in jetson['experiments'])
    gazebo=json.loads((tmp_path/'rtx5060_gazebo.json').read_text())
    demo=next(e for e in gazebo['experiments'] if e['id']=='demo_visual_square_gazebo')
    assert len(demo['cases'])==1 and demo['collector_source_commit'].startswith('2c7e2b1')
    assert demo['collector_source_commit'] in inventory['source_commits']
    assert not any('robot' in e['backend'].lower() for p in (desktop,jetson,gazebo) for e in p['experiments'])
    assert 'Force E01/E02/E03' not in inventory['omissions'] and inventory['readiness'].startswith('Preparation')
    assert any(e['id']=='E03' for e in desktop['experiments'])
    assert any(e['id']=='E03' for e in jetson['experiments'])


def test_missing_definitions_visible_and_duplicate_ids_fail(tmp_path,monkeypatch):
    root=tmp_path/'repo';(root/'experiments').mkdir(parents=True)
    spec=dict(id='study',source_commit='runtime',collector_source_commit='collector',
              dependency_source_commits={'probe':'probe'},machines=['rtx5060'],backend='internal',cases=[dict(id='one')])
    for name in ('one.json','duplicate.json'):
        (root/'experiments'/name).write_text(json.dumps(dict(experiments=[spec])))
    monkeypatch.setattr(package_plans,'ROOT',root)
    inventory=package_plans.merge(tmp_path/'output',['one.json','missing.json'])
    assert inventory['missing_plan_files']==['missing.json']
    assert inventory['source_commits']==['collector','probe','runtime']
    with pytest.raises(ValueError,match='duplicate experiment ID'):
        package_plans.merge(tmp_path/'bad',['one.json','duplicate.json'])


def test_source_bundle_package_in_synthetic_repo_is_portable(tmp_path,monkeypatch):
    import subprocess
    import tarfile
    root=tmp_path/'synthetic-repo';root.mkdir();(root/'experiments').mkdir()
    (root/'experiments/run.py').write_text('# synthetic packaging fixture, no experiment\n')
    (root/'experiments/plan.json').write_text('{}\n')
    subprocess.run(['git','init',str(root)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(root),'add','experiments'],check=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Packaging Test','-c','user.email=packaging@example.invalid','commit','-m','synthetic fixture'],check=True,capture_output=True)
    pin=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    plans=tmp_path/'generated';plans.mkdir();(plans/'inventory.json').write_text('{}')
    (root/'experiments/local.json').write_text('{"private":"excluded"}')
    (root/'.experiment-results').mkdir();(root/'.experiment-results/raw.json').write_text('{"private":"excluded"}')
    monkeypatch.setattr(package_plans,'ROOT',root)
    output=tmp_path/'package'
    package_plans.package(output,dict(source_commits=[pin]),plans)
    inventory=json.loads((output/'inventory.json').read_text())
    clone=tmp_path/'clone'
    subprocess.run(['git','clone','--branch',inventory['runner_bundle_branch'],str(output/'sources.bundle'),str(clone)],check=True,capture_output=True)
    assert (clone/'experiments/run.py').read_text().startswith('# synthetic')
    assert subprocess.check_output(['git','-C',str(root),'for-each-ref','--format=%(refname)','refs/heads/experiment-package-'],text=True)==''
    with tarfile.open(output/'runner-assets.tar.gz') as archive:
        names=archive.getnames()
    assert 'experiments/merged/inventory.json' in names
    assert 'experiments/local.json' not in names and not any('experiment-results' in name for name in names)
