#!/usr/bin/env python3
"""Current positional cylinder cohorts; ordinary orientation preflight is separate."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import RawRecorder,sha,write_json
from adapters import task,task_suite,control_studies

ORDINARY_RUNTIME='502fa29c8b0d649dcb347f9534bf29c6935fa6dd'


def ordinary_preflight(source,build,inputs,output,runtime_commit=ORDINARY_RUNTIME):
    # The cylinder source retains its reviewed custom paid-plan worker. Its
    # ordinary controller predates mode B, so run the unchanged accepted frontend
    # in this separate process with the already-built ordinary native module.
    from run import worktree
    runtime=worktree(runtime_commit,Path(build)/'prerequisite-cache')
    original=task.native_module
    def ordinary(src,bld,N=32):
        path=exact_module(bld,'bsqpN32_tiago_right.')
        sys.path[:0]=[str(src/'python'),str(src/'tiago_src')]
        importlib.import_module('bsqp');name='bsqp.bsqpN32_tiago_right'
        native=task.load(name,path);sys.modules[name]=native;return native,path
    task.native_module=ordinary
    try:return task.run(runtime,build,None,output,'B',True,inputs)
    finally:task.native_module=original


def exact_module(build,name):
    paths=list((Path(build)/'modules').glob(name+'*.so'))
    if len(paths)!=1:raise ValueError('Expected one exact '+name)
    return paths[0]


def preload(source,build):
    sys.path[:0]=[str(source/'python'),str(source/'tiago_src'),str(source/'examples/randomized_multimodal'),str(source/'experiments/cylinder_current')]
    importlib.import_module('bsqp')
    path=exact_module(build,'bsqpN32_tiago_right_multimodal');name='bsqp.bsqpN32_tiago_right_multimodal'
    native=task.load(name,path);sys.modules[name]=native
    return native,path


def make_factory(source,inputs):
    params=json.loads((inputs/'v01.json').read_text())['solver_params']
    from bsqp.mpc_controller import MPC_GATO
    def factory(model,batch,scene):
        cfg=dict(params,ee_orient_cost=scene['obstacle_running_cost'],ee_orient_N_cost=scene['obstacle_terminal_cost'])
        return MPC_GATO(model,str(source/'gato/dynamics/tiago_right/tiago_right_arm.urdf'),N=32,dt=.03,batch_size=batch,
            plant_type='tiago_right_multimodal',solver_params=cfg,track_full_stats=True),cfg
    return factory


def scene_for(inputs,case):
    if case['parameters']['study']=='X03':
        info=json.loads((inputs/'easy30.json').read_text());inst=info['instance']
        scene=dict(initial_state=inst['start_q']+[0.]*7,goal_xyz=inst['goal_position'],seconds=8.,
            obstacle_running_cost=1.,obstacle_terminal_cost=1.,cylinder=dict(initial_center_xy=inst['pillar_xy'],
            final_center_xy=inst['pillar_xy'],radius=inst['clearance_radius_m'],motion_start_seconds=0.,motion_duration_seconds=1.))
        with np.load(inputs/'easy30_routes.npz',allow_pickle=False) as data:routes=data['prefix'].copy()
        return scene,routes
    return json.loads((inputs/(case['parameters']['scene']+'.json')).read_text()),None


def run(source,build,inputs,output,case=None,preflight=False,ordinary_runtime=ORDINARY_RUNTIME):
    source,build,inputs,output=map(lambda p:Path(p).resolve(),(source,build,inputs,output))
    if preflight:
        # Both modules live in this build; existing helpers select ordinary exact name.
        suite=output/'task_suite'
        old_resolve=task_suite.resolve_module
        task_suite.resolve_module=lambda b,N: exact_module(b,'bsqpN32_tiago_right.')
        try:code=control_studies.ensure_suite(source,build,suite)
        finally:task_suite.resolve_module=old_resolve
        if code:
            write_json(output/'metrics.json',dict(task_success=None,physical_pass=None,execution_error='Required ordinary task suite failed',suite_evidence=str(suite)))
            return code
        # Prerequisite imports must not enter the custom cylinder worker process.
        code=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--source',str(source),
            '--build',str(build),'--inputs',str(inputs),'--output',str(output/'ordinary-canonical'),'--ordinary-runtime',ordinary_runtime,
            '--ordinary-preflight']).returncode
        if code:
            write_json(output/'metrics.json',dict(task_success=None,physical_pass=None,execution_error='Ordinary canonical preflight failed',ordinary_evidence=str(output/'ordinary-canonical')))
            return code
        v=json.loads((inputs/'v01.json').read_text())
        scene=dict(initial_state=v['initial_state'],goal_xyz=v['goal'][:3],seconds=8.,obstacle_running_cost=1.,obstacle_terminal_cost=1.,
            cylinder=dict(initial_center_xy=[10.,10.],final_center_xy=[10.,10.],radius=.1,motion_start_seconds=0.,motion_duration_seconds=1.))
        code=collect(source,build,inputs,output/'cylinder-canonical',dict(id='cylinder-canonical',parameters=dict(study='X02',batch=1,scene_id=0)),scene,None,True)
        metrics=json.loads((output/'cylinder-canonical/metrics.json').read_text())
        native_source=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        write_json(output/'metrics.json',dict(task_success=metrics.get('task_success'),physical_pass=metrics.get('physical_pass'),execution_error=metrics.get('execution_error'),
            cylinder_evidence=str(output/'cylinder-canonical'),ordinary_evidence=str(output/'ordinary-canonical'),suite_evidence=str(suite),
            ordinary_runtime_source_commit=ORDINARY_RUNTIME,ordinary_native_source_commit=native_source,cylinder_runtime_source_commit=native_source,
            orientation_scope='Only ordinary canonical module qualifies orientation; cylinder check position-only'))
        return code
    scene,routes=scene_for(inputs,case) if case['parameters']['study']!='X02' else (None,None)
    return collect(source,build,inputs,output,case,scene,routes)


def collect(source,build,inputs,output,case,scene,routes,preflight=False):
    metrics=dict(campaign_id=case['parameters']['study'],case_id=case['id'],timing_mode='A' if scene is None else 'B',
        execution_backend='internal',task_success=None,physical_pass=None,solver_numerical_pass=None,execution_error=None)
    with RawRecorder(output) as recorder:
        try:
            native,module=preload(source,build)
            folder=source/'experiments/cylinder_current'
            write_json(output/'provenance.json',dict(source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                module_sha256=sha(module),adapter_sha256=sha(__file__),science_sha256={p.name:sha(p) for p in folder.glob('*.py')},
                inputs_sha256={p.name:sha(p) for p in inputs.iterdir() if p.is_file()},case=case,
                scope='New current-default position-only cylinder cohort; ordinary orientation qualification separate; no full-arm clearance certificate'))
            if scene is None:
                probe_path=exact_module(build,'cylinder_probe');probe=task.load('cylinder_probe',probe_path).Probe()
                checks=task.load('cylinder_checks',folder/'checks.py')
                state=json.loads((inputs/'v01.json').read_text())['initial_state']
                metrics.update(checks.run(probe,state,recorder),probe_module_sha256=sha(probe_path))
            else:
                # Retained helper sibling imports are frozen to this exact source folder.
                sys.modules['helpers']=task.load('helpers',folder/'helpers.py')
                worker=task.load('cylinder_worker',folder/'run_tasks.py')
                args=SimpleNamespace(output=str(output),scene_id=case['parameters'].get('scene_id',30))
                metrics.update(worker.run(args,scene,case['parameters']['batch'],recorder,make_factory(source,inputs),initial_candidates=routes))
                write_json(output/'protocol.json',dict(case=case,scene=scene,N=32,dt_s=.03,sim_dt_s=.003,timeout_s=8.,mode='B',offline_timing='solve_time',
                    params=json.loads((inputs/'v01.json').read_text())['solver_params'],orientation_scope='disabled objective; orientation slots encode cylinder weights'))
        except BaseException:
            metrics['execution_error']=traceback.format_exc();recorder('execution_failed',dict(error=metrics['execution_error']))
            partial=output/'summary.json'
            if partial.exists():metrics.update(json.loads(partial.read_text()),execution_error=metrics['execution_error'])
        metrics['outcome']='execution_failed' if metrics['execution_error'] else 'checks_completed' if scene is None else 'task_passed' if metrics['task_success'] else 'task_failed'
        write_json(output/'metrics.json',metrics);recorder.manifest()
    return 1 if metrics['execution_error'] or preflight and not (metrics['task_success'] and metrics['physical_pass']) else 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','build','inputs','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',type=Path);p.add_argument('--preflight',action='store_true')
    p.add_argument('--ordinary-preflight',action='store_true');p.add_argument('--ordinary-runtime',default=ORDINARY_RUNTIME);a=p.parse_args()
    if a.ordinary_preflight:raise SystemExit(ordinary_preflight(a.source,a.build,a.inputs,a.output,a.ordinary_runtime))
    raise SystemExit(run(a.source,a.build,a.inputs,a.output,json.loads(a.case.read_text()) if a.case else None,a.preflight,a.ordinary_runtime))
