"""Pinned archive packaging checks, without builds, native imports or ROS."""
import json
from pathlib import Path
import subprocess

from adapters import demo


def test_pinned_collector_materialized_under_shared_build(tmp_path):
    repository=Path(__file__).resolve().parents[1]
    collector,pin=demo.collector_source(repository,tmp_path/'build')
    expected=subprocess.check_output(['git','-C',str(repository),'show',pin+':experiments/deployment/demo/proposal.json'])
    assert (collector/'experiments/deployment/demo/proposal.json').read_bytes()==expected
    assert collector.is_relative_to(tmp_path/'build')
    assert demo.collector_source(repository,tmp_path/'build')==(collector,pin)
    assert demo.sha(collector/'experiments/deployment/demo/proposal.json')==demo.sha(repository/'experiments/inputs/demo/proposal.json')


def test_explicit_backend_plans_share_native_recipe_and_exclude_robot():
    root=Path(__file__).resolve().parent
    internal=json.loads((root/'demo_internal.json').read_text())['experiments'][0]
    gazebo=json.loads((root/'demo_gazebo.json').read_text())['experiments'][0]
    campaign=json.loads((root/'campaign.json').read_text())['experiments'][0]
    assert internal['build']==gazebo['build']==campaign['build']
    assert internal['timing_mode']=='B' and gazebo['timing_mode']=='live'
    assert internal['source_commit'].startswith('502fa29') and gazebo['source_commit'].startswith('63940f7')
    assert internal['inputs']==gazebo['inputs']
    assert [c['id'] for c in internal['cases']]==['square','figure8-24s','figure8-18s','figure8-12s']
    assert internal['machines']==gazebo['machines']==['rtx5060']


def test_interrupted_archive_never_publishes_partial_cache(tmp_path,monkeypatch):
    import pytest
    repository=Path(__file__).resolve().parents[1]
    pin=subprocess.check_output(['git','-C',str(repository),'rev-parse',demo.COLLECTOR],text=True).strip()
    monkeypatch.setattr(demo.tarfile,'open',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('interrupted extraction')))
    with pytest.raises(RuntimeError,match='interrupted extraction'):
        demo.collector_source(repository,tmp_path/'build')
    assert not (tmp_path/'build/demo-collectors'/pin).exists()
    assert list((tmp_path/'build/demo-collectors').iterdir())==[]


def test_gazebo_runner_keeps_sequence_hold_policy(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();build=tmp_path/'build';(build/'modules').mkdir(parents=True)
    (build/'modules/bsqpN32_tiago_right.test.so').write_bytes(b'synthetic')
    collector=tmp_path/'collector';collector.mkdir();inputs=tmp_path/'inputs';inputs.mkdir()
    (inputs/'proposal.json').write_text('{}');input_file=collector/'proposal.json';input_file.write_text('{}')
    preflight=tmp_path/'rtx5060/_preflight/demo/attempt-001';preflight.mkdir(parents=True);(preflight/'metrics.json').write_text('{}')
    output=tmp_path/'rtx5060/demo/case/attempt-001';output.mkdir(parents=True)
    (output/'receipt.json').write_text(json.dumps(dict(preflight_path=str(preflight.relative_to(tmp_path)))))
    class Worker:
        INPUT=input_file
        def main(self):
            sequence=Path(demo.sys.argv[demo.sys.argv.index('--output')+1]);sequence.mkdir()
            rows=[dict(task_id='square',status='execution_failed',metrics=dict(execution_error='handbackfailed'))]
            rows += [dict(task_id=t,status='held',reason='handbackfailure') for t in ('figure8-24s','figure8-18s','figure8-12s')]
            (sequence/'index.json').write_text(json.dumps(rows));return 1
        def run_task(self,*args):raise AssertionError('Bridge bypassed heldsequence helper')
    worker=Worker()
    monkeypatch.setattr(demo,'collector_source',lambda *a:(collector,'synthetic-collector'))
    monkeypatch.setattr(demo,'load_collector',lambda *a:worker)
    monkeypatch.setattr(demo.importlib.util,'module_from_spec',lambda *a:worker)
    class Loader:
        def exec_module(self,*a):pass
    monkeypatch.setattr(demo.importlib.util,'spec_from_file_location',lambda *a:type('Spec',(),{'loader':Loader()})())
    assert demo.run(source,build,dict(id='all-demo-tasks'),output,'gazebo',False,inputs)==1
    metrics=json.loads((output/'metrics.json').read_text())
    assert [r['status'] for r in metrics['tasks']]==['execution_failed','held','held','held']
    assert metrics['task_success'] is None and metrics['execution_error']


def test_explicit_runtime_pin_preserves_frozen_demo_protocol(tmp_path,monkeypatch):
    from recording import write_json
    source=tmp_path/'source';source.mkdir();build=tmp_path/'build';(build/'modules').mkdir(parents=True)
    (build/'modules/bsqpN32_tiago_right.test.so').write_bytes(b'checked')
    output=tmp_path/'attempt';output.mkdir();write_json(output/'receipt.json',{})
    collector=tmp_path/'collector';collector.mkdir();input_file=collector/'proposal.json';input_file.write_text('{}')
    pin='f'*40;seen=[]
    (source/'python/bsqp').mkdir(parents=True);(source/'python/bsqp/offline_mpc.py').write_bytes(b'unchanged task clock')
    class Worker:
        INPUT=input_file
        RUNTIME_PINS={'internal':'old'}
        def resolve_preflight(self,backend):return dict(runtime_pin=self.RUNTIME_PINS[backend],duration=8)
        def run_task(self,protocol,*a):
            seen.append(protocol);write_json(output/'SHA256SUMS.json',{});return dict(task_success=True,physical_pass=True,execution_error=None)
    worker=Worker()
    monkeypatch.setattr(demo,'collector_source',lambda *a:(collector,'frozen-collector'))
    monkeypatch.setattr(demo,'load_collector',lambda *a:worker)
    monkeypatch.setattr(demo.importlib.util,'module_from_spec',lambda *a:worker)
    class Loader:
        def exec_module(self,*a):pass
    monkeypatch.setattr(demo.importlib.util,'spec_from_file_location',lambda *a:type('Spec',(),{'loader':Loader()})())
    monkeypatch.setattr(demo.subprocess,'check_output',lambda command,**k:pin+'\n' if 'rev-parse' in command else b'unchanged task clock')
    assert demo.run(source,build,{},output,'internal',True,None,pin)==0
    assert seen==[dict(runtime_pin=pin,duration=8)]


def test_collector_array_boundary_keeps_json_and_force_values(tmp_path):
    import numpy as np
    path=tmp_path/'frozen_collector.py'
    path.write_text("import numpy as np\ndef run_task(p,constructor):\n    return constructor(constant_f_ext=p['constant_f_ext'])\n")
    before=path.read_bytes();worker=demo.load_collector(path)
    protocol={'constant_f_ext':[0.,0.,0.,0.,0.,0.]};frozen=json.dumps(protocol)
    def constructor(constant_f_ext):
        assert isinstance(constant_f_ext,np.ndarray) and constant_f_ext.dtype==np.float32
        np.testing.assert_array_equal(constant_f_ext,protocol['constant_f_ext']);return True
    assert worker.run_task(protocol,constructor)
    assert path.read_bytes()==before and json.dumps(protocol)==frozen
