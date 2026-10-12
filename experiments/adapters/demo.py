#!/usr/bin/env python3
"""Run one pinned visual-demo collector with a separately pinned public runtime."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import io
import tempfile
import shutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import sha,write_json
from adapters.task_recording import TaskRecorder, TASK_RECORDING_FORMAT

COLLECTOR='2c7e2b11bd5bab8fa3a761d17cf55c1aa9cff55e'


def collector_source(repository,build,collector_pin=None):
    requested=collector_pin or COLLECTOR
    if len(requested)!=40:raise ValueError('Collector pin must be a full immutable Git commit')
    pin=subprocess.check_output(['git','-C',str(repository),'rev-parse',requested],text=True).strip()
    if pin!=requested:raise ValueError('Collector pin must resolve to its exact immutable commit')
    target=Path(build)/'demo-collectors'/pin
    if not target.exists():
        payload=subprocess.check_output(['git','-C',str(repository),'archive','--format=tar',pin,'experiments/deployment'])
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=Path(tempfile.mkdtemp(prefix=pin+'-',dir=target.parent))
        try:
            with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                for entry in archive.getmembers():
                    relative=Path(entry.name)
                    if relative.is_absolute() or '..' in relative.parts or entry.issym() or entry.islnk():raise ValueError('Unsafe collector archive path')
                    if entry.isdir():(temporary/relative).mkdir(parents=True,exist_ok=True)
                    elif entry.isfile():
                        path=temporary/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.extractfile(entry).read())
            temporary.rename(target)
        finally:
            if temporary.exists():shutil.rmtree(temporary)
    if not (target/'experiments/deployment/demo/run_demo.py').is_file():raise ValueError('Incomplete collector cache; retain failure and replace this generated cache')
    return target,pin


MANIFEST_OVERRIDE_VERSION=TASK_RECORDING_FORMAT
FINALIZED_FILES=('events.jsonl','case.json','protocol.json','provenance.json',
    'controller_stats.json','metrics.json','ros_observations.jsonl',
    'controller_state_history.csv','full_joint_state_history.jsonl')


def manifest_hashes(recorder):
    """Same frozen file set; reuse only hashes recorded for own immutable raw files."""
    cached={e['raw_file']:e['sha256'] for e in recorder.events}
    files=list((recorder.output/'raw').glob('*.npz'))
    files += [recorder.output/name for name in FINALIZED_FILES if (recorder.output/name).is_file()]
    result={}
    for path in sorted(files):
        relative=str(path.relative_to(recorder.output))
        result[relative]=cached[relative] if relative in cached and path.parent==recorder.output/'raw' else sha(path)
    return result


class DemoRecorder(TaskRecorder):
    """Ordinary demo observations, with collector-owned final file manifests."""
    def manifest(self):
        super().manifest()
        path = self.output/'SHA256SUMS.json'
        hashes = json.loads(path.read_text())
        for name in FINALIZED_FILES:
            target = self.output/name
            if target.is_file() and name not in hashes:
                hashes[name] = sha(target)
        write_json(path, hashes)


def load_collector(path):
    spec=importlib.util.spec_from_file_location('pinned_visual_demo',path);worker=importlib.util.module_from_spec(spec)
    # Frozen JSON force values stay unchanged; the native API requires arrays.
    text=path.read_text();original="constant_f_ext=p['constant_f_ext']"
    replacement="constant_f_ext=np.asarray(p['constant_f_ext'],dtype=np.float32)"
    if text.count(original)==1 and text.count(replacement)==0:
        text = text.replace(original,replacement);force_override=True
    elif text.count(original)==0 and text.count(replacement)==1:
        force_override=False  # New collector owns the reviewed array boundary.
    else:raise ValueError('Frozen collector force argument seam changed')
    summary_seam="        recorder('controller_stats',stats)\n        write_json(output/'controller_stats.json',stats)"
    if 'RawRecorder' in text:
        if text.count(summary_seam) != 1:
            raise ValueError('Frozen collector summary recording seam changed')
        text = text.replace(summary_seam, "        recorder.write_controller_stats(stats)")
    exec(compile(text,str(path),'exec'),worker.__dict__)
    worker.GATO_FORCE_ARRAY_OVERRIDE_APPLIED=force_override
    if hasattr(worker,'RawRecorder'):
        worker.RawRecorder = DemoRecorder
    return worker

def run(source,build,case,output,backend,preflight=False,inputs=None,runtime_pin=None,collector_pin=None):
    source,build,output=map(lambda p:Path(p).resolve(),(source,build,output))
    collector,pin=collector_source(source,build,collector_pin)
    sys.path.insert(0,str(collector))
    path=collector/'experiments/deployment/demo/run_demo.py'
    worker=load_collector(path)
    if inputs is not None and sha(Path(inputs)/'proposal.json')!=sha(worker.INPUT):raise ValueError('Pinned collector input differs from runner frozen input')
    if runtime_pin is not None:
        resolved=subprocess.check_output(['git','-C',str(source),'rev-parse',runtime_pin],text=True).strip()
        if resolved!=runtime_pin or len(runtime_pin)!=40:raise ValueError('Explicit reviewed runtime pin must be a full commit')
        if backend=='internal':
            relative='python/bsqp/offline_mpc.py'
            expected=subprocess.check_output(['git','-C',str(source),'show',runtime_pin+':'+relative])
            if (source/relative).read_bytes()!=expected:raise ValueError('Runtime bytes differ: '+relative)
        worker.RUNTIME_PINS=dict(worker.RUNTIME_PINS,**{backend:runtime_pin})
    worker.ROOT=collector  # collector-owned files are frozen archive bytes; runtime is supplied separately
    matches=list((build/'modules').glob('bsqpN32_tiago_right*.so'))
    if len(matches)!=1:raise ValueError('Expected one reviewed N32module in selected shared build')
    receipt=json.loads((output/'receipt.json').read_text())
    preflight_output=None
    if not preflight:
        relative=receipt.get('preflight_path')
        if not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('Missing relative runner preflight evidence')
        candidates=[root/relative for root in output.parents if (root/relative/'metrics.json').is_file()]
        if len(candidates)!=1:raise ValueError('Preflight path does not resolve uniquely')
        preflight_output=candidates[0]
    if backend=='gazebo' and not preflight and case['id'] not in ('square','all-demo-tasks'):
        raise ValueError('Gazebo demo selects standalone square or the historical held-on-failure sequence')
    if backend=='gazebo' and not preflight and case['id']=='all-demo-tasks':
        sequence=output/'sequence'
        sys.argv=[str(path),'--backend','gazebo','--source',str(source),'--module',str(matches[0]),
                  '--output',str(sequence),'--preflight-output',str(preflight_output)]
        code=worker.main()
        index=json.loads((sequence/'index.json').read_text())
        completed=[r for r in index if r['status']=='completed']
        metrics=dict(campaign_id='demo_visual_square',backend='gazebo',timing_mode='live',
            machine=__import__('os').environ.get('GATO_MACHINE'),sequence_index=str(sequence/'index.json'),tasks=index,
            task_success=all(r['metrics'].get('task_success') is True for r in completed) if len(completed)==4 else None,
            physical_pass=all(r['metrics'].get('physical_pass') is True for r in completed) if len(completed)==4 else None,
            handback_confirmed=all(r['metrics'].get('handback_confirmed') is True for r in completed) if len(completed)==4 else False,
            execution_error='Live sequence held; individual task/held outcomes retained' if code else None)
        write_json(output/'SHA256SUMS.json',{'sequence/index.json':sha(sequence/'index.json')})
    else:
        protocol=worker.resolve_preflight(backend) if preflight else worker.resolve(backend,case['id'])
        metrics=worker.run_task(protocol,source,matches[0],output,preflight_output)
    metrics.update(collector_source_commit=pin,collection_adapter_sha256=sha(__file__),
        collector_adapter_overrides=(['Force constructor argument list to float32 array; frozen protocol unchanged']
            if getattr(worker,'GATO_FORCE_ARRAY_OVERRIDE_APPLIED',True) else [])+
            ['Ordinary task observations with full executed motion/timing/work/quality; routine solver I/O omitted; task/solver settings unchanged'],
        manifest_override_version=MANIFEST_OVERRIDE_VERSION,
        recording_format=MANIFEST_OVERRIDE_VERSION)
    write_json(output/'metrics.json',metrics)
    manifest=json.loads((output/'SHA256SUMS.json').read_text());manifest['metrics.json']=sha(output/'metrics.json');write_json(output/'SHA256SUMS.json',manifest)
    if preflight:return int(bool(metrics['execution_error']) or metrics.get('task_success') is not True or metrics.get('physical_pass') is not True or (backend!='internal' and metrics.get('handback_confirmed') is not True))
    return int(bool(metrics['execution_error']) or (backend!='internal' and metrics.get('handback_confirmed') is not True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','build','case','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--backend',choices=('internal','gazebo'),required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--runtime-pin')
    parser.add_argument('--collector-pin',help='Optional full collector commit; historical collector remains the default')
    args=parser.parse_args()
    raise SystemExit(run(args.source,args.build,json.loads(args.case.read_text()),args.output,args.backend,args.preflight,args.inputs,args.runtime_pin,args.collector_pin))
