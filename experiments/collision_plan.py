#!/usr/bin/env python3
"""Bundle used collision geometry and freeze bounded CPU monitor comparisons."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
SOURCE='2db284ca8e11f42f8f3248403b879652966893ed'
FAULT_SOURCE=Path('/workspace/GATO-eval-collision')
ORIGINAL=Path('/workspace/GATO/TiagoProURDF')
WRAPPED=Path('/workspace/GATO/tiago_configs/collision_alpha_wrap_a3_o25_cap001')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundle(urdf,target,fallback=None):
    urdf,target=Path(urdf),Path(target)
    if not urdf.is_file():raise FileNotFoundError('Original collision asset unavailable: '+str(urdf))
    target.mkdir(parents=True,exist_ok=True)
    xml=ET.parse(urdf);used=[]
    for mesh in xml.findall('.//collision/geometry/mesh'):
        uri=mesh.attrib['filename'];relative=uri.removeprefix('package://')
        source=Path(relative) if Path(relative).is_absolute() else urdf.parent/relative
        if not source.is_file() and fallback:source=Path(fallback)/relative
        if not source.is_file():raise FileNotFoundError('Collision mesh unavailable: '+uri)
        digest=sha(source);name='meshes/'+digest[:12]+'-'+source.name
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and sha(destination)!=digest:raise ValueError('Bundled mesh differs')
        shutil.copyfile(source,destination);mesh.set('filename',name)
        used.append(dict(original_uri=uri,original_path=str(source),sha256=digest,bundled=name))
    xml.write(target/'model.urdf',encoding='utf-8',xml_declaration=True)
    return dict(original_urdf=str(urdf),original_sha256=sha(urdf),bundled_sha256=sha(target/'model.urdf'),
        used_collision_meshes=used,scope='Only collision-used meshes bundled; URDF mesh URI rewrite only, no geometry changes')


def corpus():
    comfortable=[-.36,-1.83,-.47,-2.35,0.,-1.20,.04]
    clearance=[-.39,-1.73,-.38,-2.35,0.,-1.21,.04]
    positions=[('comfortable',comfortable),('high-clearance',clearance),
        ('high-clearance-joint1-plus05',[clearance[0]+.05]+clearance[1:]),
        ('high-clearance-joint1-minus05',[clearance[0]-.05]+clearance[1:])]
    return [dict(id=name+('-stationary' if speed==0 else '-velocity05'),right_arm_q_rad=q,
        right_arm_qd_rad_s=[speed]*7,torso_lift_joint_m=.1,other_reduced_joints='Pinocchio neutral positions and zero velocities',
        scope='Synthetic named joint snapshots for CPU query workload; no executed robot motion')
        for name,q in positions for speed in (0.,.05)]


def prepare(output=None):
    inputs=ROOT/'experiments/inputs/collision';inputs.mkdir(parents=True,exist_ok=True)
    assets=dict(original=bundle(ORIGINAL/'tiago_pro.urdf',inputs/'original'),
        wrapped=bundle(WRAPPED/'tiago_pro_collision_alpha_wrap_a3_o25_cap001.urdf',inputs/'wrapped',ORIGINAL))
    blacklist=subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':tiago_configs/comfortable_pose_collision_blacklist.json'])
    (inputs/'blacklist.json').write_bytes(blacklist)
    protocol=dict(id='monitor-CPU-eight-state-v1',source_commit=SOURCE,states=corpus(),assets=assets,
        min_distance_m=.04,query_repeats=100,worker_repeats=20,stop_at_first=True,
        queries=['exact-all-pair-distance','runtime-margin-first-violation'],
        worker_scope='Complete synchronous _SafetyChecker.check CPU wall time (state mapping, limits, margin and geometry-speed checks); excludes async queue/polling/process startup.',
        exact_scope='Fresh collision-data allocation and Pinocchio exact distances for ALL retained pairs; compare whole query behavior, not interchangeable distance-kernel speedup',
        margin_scope='Existing compute_collision_margin_violation at4cm with stop_at_first=True; may short-circuit at first violation',
        duplicated_worker_scope='The identical runtime margin worker is measured under both query case labels. This is repeated worker evidence, not a distinct exact-worker implementation.',
        raw='All per-call times and violations/errors retained; safety rejection is completed evidence, setup failure holds dependent case',
        machine_scope='Current CPU model/package details retained; not historical2Hz reconstruction or live125Hz monitor guarantee')
    (inputs/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in inputs.rglob('*') if p.is_file()}
    cases=[dict(id=asset+'-'+query,parameters=dict(asset=asset,query=query)) for asset in assets for query in ('exact','margin')]
    plan=dict(version=1,experiments=[dict(id='M02',research_question='RQ-monitor',protocol=protocol['id'],source_commit=SOURCE,
        machines=['gtx1070','rtx5060'],backend='CPU-Pinocchio-Coal',timing_mode='A',inputs=hashes,
        command=['{python}','{runner}/adapters/collision_monitor.py','--source','{source}',
            '--inputs','{inputs}/experiments/inputs/collision','--case','{case}','--output','{output}'],cases=cases,timeout_seconds=600)],
        status='prepared_CPU_target_check_pending',jetson='Same four cells proposed after target duration discussion; no GPU required',
        M03=dict(source=str(FAULT_SOURCE),base_commit=SOURCE,status='unchanged fault harness transplanted; CPU review pending; runtime source checkpoint required',
            simulation_only=True,defaults='N64/.008s/125Hz;6s figure8 prefix;injectafter2s;4s restoration;one-shot malformed horizons; existing failure/cleanup semantics',
            historical_results='external robot_faults/2026-09-21 recovery pending'),
        M04='Saved fault response/observed braking illustration analyzer; actual state age and clearance unavailable without compatible co-temporal evidence. No safety certificate.')
    output=Path(output) if output else ROOT/'experiments/collision_monitor.json'
    output.write_text(json.dumps(plan,indent=2)+'\n')
    jetson=deepcopy(plan);jetson['experiments'][0]['machines']=['jetson']
    jetson['status']='proposed_target_duration_discussion_pending'
    (ROOT/'experiments/collision_monitor_jetson_proposed.json').write_text(json.dumps(jetson,indent=2)+'\n')
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();plan=prepare(args.output);print(len(plan['experiments'][0]['cases']),'CPU monitor cases prepared')
