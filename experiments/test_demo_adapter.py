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


def test_finalization_cached_raw_is_exact_and_nonraw_foreign_still_hashed(tmp_path,monkeypatch):
    import hashlib
    from types import SimpleNamespace
    raw=tmp_path/'raw';raw.mkdir()
    own=raw/'0000000-solve_completed.npz';own.write_bytes(b'own immutable raw')
    foreign=raw/'foreign.npz';foreign.write_bytes(b'foreign raw')
    (tmp_path/'events.jsonl').write_text('recorded events\n')
    (tmp_path/'metrics.json').write_text('new final metrics\n')
    (tmp_path/'stderr.log').write_text('runner-owned mutable log')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    recorder=SimpleNamespace(output=tmp_path,events=[dict(raw_file='raw/'+own.name,sha256=digest(own))])
    expected={str(p.relative_to(tmp_path)):digest(p) for p in [own,foreign,tmp_path/'events.jsonl',tmp_path/'metrics.json']}
    calls=[]
    def checked_hash(path):
        calls.append(path)
        assert path!=own,'Own raw should not be reread at finalization'
        return digest(path)
    monkeypatch.setattr(demo,'sha',checked_hash)
    assert demo.manifest_hashes(recorder)==expected
    assert set(calls)=={foreign,tmp_path/'events.jsonl',tmp_path/'metrics.json'}
    (tmp_path/'metrics.json').write_text('changed final metrics\n')
    assert demo.manifest_hashes(recorder)['metrics.json']==digest(tmp_path/'metrics.json')


def test_gazebo_square_uses_explicit_collector_and_only_one_task(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();build=tmp_path/'build';(build/'modules').mkdir(parents=True)
    (build/'modules/bsqpN32_tiago_right.test.so').write_bytes(b'checked module')
    collector=tmp_path/'collector';collector.mkdir();inputs=tmp_path/'inputs';inputs.mkdir()
    (inputs/'proposal.json').write_text('{}');input_file=collector/'proposal.json';input_file.write_text('{}')
    preflight=tmp_path/'rtx5060/_preflight/square/attempt-001';preflight.mkdir(parents=True)
    (preflight/'metrics.json').write_text('{}')
    output=tmp_path/'rtx5060/square/case/attempt-001';output.mkdir(parents=True)
    (output/'receipt.json').write_text(json.dumps(dict(preflight_path=str(preflight.relative_to(tmp_path)))))
    pin='a'*40;seen=[]
    class Worker:
        INPUT=input_file
        def main(self):raise AssertionError('Standalone square must not launch the four-task sequence')
        def resolve(self,backend,task):return dict(backend=backend,task_id=task)
        def run_task(self,protocol,s,m,o,pf):
            seen.append((protocol,pf));(o/'SHA256SUMS.json').write_text('{}')
            return dict(task_id='square',task_success=True,physical_pass=True,execution_error=None,handback_confirmed=True)
    worker=Worker()
    def selected_collector(s,b,requested):
        assert requested==pin;return collector,pin
    monkeypatch.setattr(demo,'collector_source',selected_collector)
    monkeypatch.setattr(demo,'load_collector',lambda *a:worker)
    assert demo.run(source,build,dict(id='square'),output,'gazebo',False,inputs,None,pin)==0
    assert seen==[(dict(backend='gazebo',task_id='square'),preflight)]
    metrics=json.loads((output/'metrics.json').read_text())
    assert metrics['collector_source_commit']==pin and metrics['task_id']=='square'
    assert 'sequence_index' not in metrics


def test_corrected_collector_array_boundary_is_used_without_second_override(tmp_path):
    import numpy as np
    path=tmp_path/'corrected_collector.py'
    path.write_text("import numpy as np\ndef run_task(p,constructor):\n    return constructor(constant_f_ext=np.asarray(p['constant_f_ext'],dtype=np.float32))\n")
    before=path.read_bytes();worker=demo.load_collector(path)
    assert worker.GATO_FORCE_ARRAY_OVERRIDE_APPLIED is False
    result=worker.run_task({'constant_f_ext':[0.]*6},lambda constant_f_ext:constant_f_ext)
    assert isinstance(result,np.ndarray) and result.dtype==np.float32
    assert path.read_bytes()==before


def test_accepted_square_plan_pins_materialized_collector_and_n32_only(tmp_path):
    repository=Path(__file__).resolve().parents[1]
    plan=json.loads((repository/'experiments/launch/rtx5060_gazebo.json').read_text())
    assert sum(len(e['cases'])*e.get('repeats',1) for e in plan['experiments'])==23
    square=next(e for e in plan['experiments'] if e['id']=='SquareDemo-v5-Gazebo')
    assert [c['id'] for c in square['cases']]==['square']
    assert square['source_commit']=='8dbb8b8d6dc1d65e902373a9aa58693e33329c14'
    assert '-DKNOTS=32' in square['build']['commands'][0]
    assert len(next(e for e in plan['experiments'] if e['id']=='D01')['cases'])==2
    assert len(next(e for e in plan['experiments'] if e['id']=='M03')['cases'])==20
    pin=square['collector_source_commit']
    collector,actual=demo.collector_source(repository,tmp_path/'build',pin)
    assert actual==pin and '--collector-pin' in square['command']
    frozen=repository/next(iter(square['inputs']))
    archived=collector/'experiments/deployment/demo/proposal.json'
    assert frozen.read_bytes()==archived.read_bytes()
    proposal=json.loads(frozen.read_text())
    assert len(proposal['goals'])==5 and proposal['goals'][-1]['tool_axis']==[1.0,0.0,0.0]
    worker=demo.load_collector(collector/'experiments/deployment/demo/run_demo.py')
    selected=worker.resolve('gazebo','square')
    assert selected['goals']==proposal['goals'] and selected['N']==32
    assert selected['runtime_pin']==square['source_commit']
    assert worker.GATO_FORCE_ARRAY_OVERRIDE_APPLIED is False
