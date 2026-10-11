#!/usr/bin/env python3
"""M02 existing exact/margin functions and complete synchronous worker timings."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import RawRecorder,sha,write_json


def named_state(S,model,record,seq):
    import pinocchio as pin
    q=pin.neutral(model)
    positions={str(model.names[j]):float(np.arctan2(q[model.joints[j].idx_q+1],q[model.joints[j].idx_q]))
        if model.joints[j].nq==2 and model.joints[j].nv==1 else float(q[model.joints[j].idx_q]) for j in range(1,model.njoints)}
    velocities={key:0. for key in positions}
    positions['torso_lift_joint']=record['torso_lift_joint_m']
    for i,joint in enumerate(S.RIGHT_ARM_JOINTS):
        positions[joint]=record['right_arm_q_rad'][i];velocities[joint]=record['right_arm_qd_rad_s'][i]
    return S.SafetyCheckState(q=np.asarray(record['right_arm_q_rad']),qd=np.asarray(record['right_arm_qd_rad_s']),
        stamp_sec=0.,received_monotonic_sec=time.monotonic(),seq=seq,joint_positions=positions,joint_velocities=velocities)


def run(source,inputs,case,output):
    source,inputs,output=map(Path,(source,inputs,output))
    metrics=dict(campaign_id='M02',research_question='RQ-monitor',execution_backend='CPU-Pinocchio-Coal',timing_mode='A',
        task_success=None,physical_pass=None,execution_error=None,query_calls=0,worker_calls=0,safety_rejections=0)
    with RawRecorder(output) as recorder:
        try:
            sys.path.insert(0,str(source/'tiago_src'))
            import pinocchio as pin
            from gato_tiago import safety_monitor as S
            if Path(S.__file__).resolve()!=source.resolve()/'tiago_src/gato_tiago/safety_monitor.py':raise ValueError('Monitor source import differs')
            protocol=json.loads((inputs/'protocol.json').read_text());parameters=case['parameters']
            if parameters['asset'] not in ('original','wrapped') or parameters['query'] not in ('exact','margin'):raise ValueError('Unknown monitor case')
            urdf=inputs/parameters['asset']/'model.urdf';blacklist=inputs/'blacklist.json'
            full=pin.buildModelFromUrdf(str(urdf))
            reference=named_state(S,full,protocol['states'][0],0)
            started=time.perf_counter()
            collision=S.build_tiago_collision_model(urdf_path=urdf,reference_positions=reference.joint_positions,
                blacklist_path=blacklist,controlled_joint_names=S.RIGHT_ARM_JOINTS)
            initialization=time.perf_counter()-started
            write_json(output/'provenance.json',dict(source=str(source),monitor_sha256=sha(S.__file__),
                adapter_sha256=sha(__file__),protocol_sha256=sha(inputs/'protocol.json'),urdf_sha256=sha(urdf),
                assets=protocol['assets'][parameters['asset']],geometry_objects=len(collision.geometry_model.geometryObjects),
                collision_pairs=len(collision.geometry_model.collisionPairs),pinocchio_version=pin.__version__,initialization_wall_s=initialization))
            for index,record in enumerate(protocol['states']):
                state=named_state(S,full,record,index+1)
                q,v=S.state_to_qv(collision.model,S.NamedJointState(state.joint_positions,state.joint_velocities))
                for repeat in range(protocol['query_repeats']):
                    started=time.perf_counter()
                    if parameters['query']=='margin':
                        result=S.compute_collision_margin_violation(collision,q,margin_m=.04,stop_at_first=True)
                        payload=None if result is None else asdict(result)
                    else:
                        data,geometry=collision.make_data()
                        pin.computeDistances(collision.model,data,collision.geometry_model,geometry,q)
                        distances=np.asarray([r.min_distance for r in geometry.distanceResults])
                        payload=dict(distances_m=distances,min_distance_m=float(np.min(distances)) if len(distances) else None)
                    elapsed=time.perf_counter()-started
                    recorder('query',dict(state=record,repeat=repeat,query=parameters['query'],wall_s=elapsed,result=payload));metrics['query_calls']+=1
                settings=S.CollisionSafetySettings(urdf_path=urdf,blacklist_path=blacklist,controlled_joint_names=tuple(S.RIGHT_ARM_JOINTS))
                checker=S._SafetyChecker(settings=settings,initial_state=state)
                for repeat in range(protocol['worker_repeats']):
                    next_state=named_state(S,full,record,index*protocol['worker_repeats']+repeat+1)
                    started=time.perf_counter();fault=None
                    try:checker.check(next_state)
                    except S._SafetyFault as error:fault=dict(kind=error.fault_kind,message=str(error));metrics['safety_rejections']+=1
                    elapsed=time.perf_counter()-started
                    recorder('synchronous_worker',dict(state=record,repeat=repeat,wall_s=elapsed,fault=fault));metrics['worker_calls']+=1
            metrics.update(outcome='monitor_checks_completed',worker_scope=protocol['worker_scope'])
        except BaseException:
            metrics.update(outcome='execution_failed',execution_error=traceback.format_exc());recorder('execution_failed',dict(error=metrics['execution_error']))
        write_json(output/'metrics.json',metrics);recorder.manifest()
    return 1 if metrics['execution_error'] else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','inputs','case','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();sys.exit(run(args.source,args.inputs,json.loads(args.case.read_text()),args.output))
