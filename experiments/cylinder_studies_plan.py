#!/usr/bin/env python3
"""Freeze bounded X02-X04 current-B studies, keeping historical outcomes separate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
SOURCE_ROOT=Path('/workspace/GATO-eval-cylinder')

def prepare(source_commit,jetson=False):
    folder=ROOT/'experiments/inputs/cylinder_studies';folder.mkdir(parents=True,exist_ok=True)
    for p in (SOURCE_ROOT/'experiments/cylinder_current/inputs').iterdir():
        if p.is_file():(folder/p.name).write_bytes(p.read_bytes())
    for name in ('v01.json','t01.json'):
        (folder/name).write_bytes((ROOT/'experiments/inputs/control_studies'/name).read_bytes())
    protocol=dict(id='cylinder-current-default-B-v1',source_commit=source_commit,
        X02='Four center/interior/exterior/boundary stage/terminal cylinder checks. Gauss-Newton is not exact Hessian; boundary curvature not a precision gate.',
        X03='Easy seed30 eight supplied original route templates versus lane0 cold gravity-hold seed; physically timed first .93s of original1.86s, not compressed; own-lane optimized outputs subsequently warm-shift.',
        X04='Retained scene04_fast positive pair and scene02/03 failed transfers, B1/B16 once under current defaults.',
        defaults=dict(N=32,dt_s=.03,sim_dt_s=.003,timeout_s=8.,qd_cost=.1,vel_lim_cost=.1,max_sqp=5,max_pcg=120),
        obstacle_overrides='Retained scenes1/1; supplied-route current cohort1/1. Native orientation slots repurposed as cylinder weights: positional goal only, no orientation qualification.',
        gates='Ordinary N32 frozen task suite and orientation-on canonicalB preflight; separate position-only cylinder canonical. Scientific failures retain raw and independent cases continue.',
        historical_sources=['c084033b4147d267e094dd7699e0f31dd0b7ba6a','4f6c8fd4be428c135fc16564c5dd56eb61ce5660'],
        history='Historical modeA/N64 moving outcomes and TIAGo portfolio failures remain distinct; IIWA component record-only.',
        collision_scope='Tool-center sampled distance to actual-time infinite cylinder only; no full-arm or continuous-time certificate',
        qualifications='Prepared CPU only; exact target native task/cylinder validation pending; no convergence/global-optimum claim')
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    paths=['native/CMakeLists.txt','native/task_readiness/CMakeLists.txt','native/cylinder_studies/CMakeLists.txt']
    build=dict(inputs={p:hashlib.sha256((ROOT/'experiments'/p).read_bytes()).hexdigest() for p in paths},commands=[
        ['cmake','-S','{runner}/native/cylinder_studies','-B','{build}','-DGATO_SOURCE_DIR={source}','-DKNOTS=32','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}-real','-DPython3_EXECUTABLE={python}'],
        ['cmake','--build','{build}','--parallel','2']],artifacts=['modules/bsqpN32_tiago_right*.so','modules/cylinder_probe*.so','cuda_access_smoke'])
    cases=[]
    if not jetson:
        cases.append(dict(id='cylinder-checks',parameters=dict(study='X02')))
        for b in (1,8):cases.append(dict(id=f'supplied-easy30-B{b}',parameters=dict(study='X03',batch=b)))
    for name,scene_id in (('scene04_fast',4),('scene02',2),('scene03',3)):
        for b in (1,16):cases.append(dict(id=f'{name}-B{b}',parameters=dict(study='X04',scene=name,scene_id=scene_id,batch=b)))
    command=['{python}','{runner}/adapters/cylinder_studies.py','--source','{source}','--build','{build}','--inputs','{inputs}/experiments/inputs/cylinder_studies','--output','{output}']
    common=dict(research_question='RQ-cylinder-routes',protocol=protocol['id'],source_commit=source_commit,
        machines=['jetson'] if jetson else ['gtx1070','rtx5060'],backend='internal',inputs=inputs,build=build,
        command=command+['--case','{case}'],timeout_seconds=600)
    experiments=[]
    if not jetson:
        experiments.append(dict(common,id='CylinderChecks',timing_mode='A',cases=[c for c in cases if c['parameters']['study']=='X02']))
    experiments.append(dict(common,id='CylinderTasks',timing_mode='B',cases=[c for c in cases if c['parameters']['study']!='X02'],
        preflight=dict(group='CylinderStudies-B',command=command+['--preflight'])))
    plan=dict(version=1,status='proposed_target_duration_discussion_pending' if jetson else 'prepared_target_validation_pending',experiments=experiments)
    path=ROOT/'experiments'/('cylinder_studies_jetson_proposed.json' if jetson else 'cylinder_studies.json')
    path.write_text(json.dumps(plan,indent=2)+'\n');return plan

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-commit',required=True);p.add_argument('--jetson-proposed',action='store_true');a=p.parse_args()
    print(sum(len(e['cases']) for e in prepare(a.source_commit,a.jetson_proposed)['experiments']),'prepared cases')
