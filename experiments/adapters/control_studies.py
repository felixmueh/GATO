#!/usr/bin/env python3
"""Bounded control choices through the existing production modeB controller."""
import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters import task,task_suite
from recording import sha,write_json

PARAM_KEYS=('q_cost','qd_cost','u_cost','N_cost','ee_orient_cost','ee_orient_N_cost','q_lim_cost','vel_lim_cost','ctrl_lim_cost','max_sqp_iters','mu','pcg_tol')


def gripper_plant(asset,angle,expected_sha256=None):
    """Lock real URDF finger joints; preserve the seven arm coordinates."""
    import pinocchio as pin
    asset=Path(asset)
    if expected_sha256 and sha(asset)!=expected_sha256:raise ValueError('Frozen gripper model asset differs')
    root=ET.parse(asset).getroot();full=pin.buildModelFromUrdf(str(asset));q=pin.neutral(full)
    positions={};locked=[]
    for joint in root.findall('joint'):
        name=joint.attrib['name']
        if not name.startswith('gripper_right') or joint.attrib['type']=='fixed':continue
        mimic=joint.find('mimic')
        if mimic is not None and mimic.attrib['joint']!='gripper_right_finger_joint':raise ValueError('Unexpected gripper mimic driver')
        value=float(angle) if mimic is None else float(angle)*float(mimic.attrib.get('multiplier','1'))+float(mimic.attrib.get('offset','0'))
        limits=joint.find('limit')
        if value<float(limits.attrib['lower']) or value>float(limits.attrib['upper']):raise ValueError('Declared gripper coordinate outside asset limits')
        index=full.getJointId(name)
        if index>=full.njoints:raise ValueError('Gripper joint absent from Pinocchio model')
        positions[name]=value;q[full.joints[index].idx_q]=value;locked.append(index)
    if len(locked)!=6:raise ValueError('Expected six actual gripper finger joints')
    plant=pin.buildReducedModel(full,locked,q)
    names=[f'arm_right_{i}_joint' for i in range(1,8)]
    if plant.nq!=7 or plant.nv!=7 or list(plant.names)[1:]!=names:raise ValueError('Reduced gripper plant coordinate mapping differs')
    if plant.getFrameId('arm_right_tool_link')>=plant.nframes:raise ValueError('Reduced plant tool frame unavailable')
    return plant,dict(asset=str(asset),asset_sha256=sha(asset),finger_angle_rad=float(angle),locked_joint_positions_rad=positions,
        arm_joint_names=names,plant_dynamics='Pinocchio actual-URDF reduction at fixed gripper coordinates',
        native_optimizer='unchanged nominal-zero wider lumped model',model_mismatch_intentional=True,
        scope='No native regeneration, payload, physical gripper motion, friction, drive response or collision certificate')


def seed(policy,previous,state,nx,nu,knots):
    result=np.asarray(previous,dtype=np.float32).copy()
    if policy=='cold':
        result.fill(0)
        for k in range(knots):result[...,k*(nx+nu):k*(nx+nu)+nx]=state
    elif policy!='unshifted':raise ValueError('Unsupported control warm seed policy')
    result[...,:nx]=state
    return result


def apply_protocol(base,variant,protocol,preflight=False):
    result=dict(base);result['solver_params']=dict(base['solver_params'],**variant.get('weights',{}))
    if 'terminal_factor' in variant:result['solver_params']['N_cost']*=variant['terminal_factor']
    initial=np.asarray(base['initial_state']).copy()
    if not preflight and variant.get('initial_delta_rad'):initial[0]+=variant['initial_delta_rad']
    result['initial_state']=initial.tolist()
    result.update(control_study=variant,collection_protocol=protocol['id'],
        full_observation=base['kind']=='reaching' and not preflight,
        observation_dwell_s=protocol['observation_dwell_s'] if not preflight else 0.,
        baseline_reuse=protocol['baseline_reuse'])
    return result


def window_metrics(output,holding=False):
    output=Path(output);metrics=json.loads((output/'metrics.json').read_text())
    if not (output/'executed_trace.csv').exists():return metrics
    with (output/'executed_trace.csv').open() as stream:rows=list(csv.DictReader(stream))
    rough=[float(r['position_error_m'])<=.05 and float(r['tool_axis_error_deg'])<=30 and float(r['speed_L1_rad_s'])<=1 for r in rows]
    complete=bool(rows and abs(float(rows[-1]['time_s'])-8.)<1e-8 and metrics.get('execution_error') is None)
    stats=json.loads((output/'controller_stats.json').read_text())
    metrics.update(observation_completed=complete,public_controller_goal_outcomes=stats.get('goal_outcomes'),
        public_stop_semantics='timeout ends explicit8s observation; rough success uses zero dwell separately',
        rough_success_zero_dwell=bool(any(rough)),first_rough_success_s=next((float(r['time_s']) for r,good in zip(rows,rough) if good),None),
        final_position_error_m=float(rows[-1]['position_error_m']) if rows else None,
        final_tool_axis_error_deg=float(rows[-1]['tool_axis_error_deg']) if rows else None,
        final_speed_L1_rad_s=float(rows[-1]['speed_L1_rad_s']) if rows else None,
        public_termination_gates=metrics.pop('task_gates',None),
        task_success=None if holding else bool(any(rough)),
        task_gates=dict(rough_success_zero_dwell=bool(any(rough)),observation_completed=complete))
    if holding:
        metrics.update(holding_scope='Pose regulation from its initial FK; finite8s tail offset, no reaching or equilibrium certificate',
            initial_inside_goal=True)
    metrics['outcome']='execution_failed' if metrics.get('execution_error') else 'holding_observation' if holding else 'rough_task_passed' if metrics['task_success'] else 'rough_task_failed'
    write_json(output/'metrics.json',metrics)
    return metrics


def ensure_suite(source,build,output):
    output=Path(output)
    module=task_suite.resolve_module(Path(build),32)
    if (output/'metrics.json').exists():
        metrics=json.loads((output/'metrics.json').read_text())
        provenance=json.loads((output/'provenance.json').read_text())
        if (metrics.get('outcome')=='task_suite_passed' and metrics.get('exact_cases') and not metrics.get('skipped') and
            provenance['module_sha256']==sha(module) and provenance['adapter_sha256']==sha(task_suite.__file__) and
            provenance['bootstrap_sha256']==sha(task_suite.EXPERIMENTS/'task_bootstrap/sitecustomize.py') and
            all(sha(Path(source)/p)==h for p,h in provenance['fixture_sha256'].items())):
            return 0
        return 1 # Existing failed or incompatible evidence is retained; no implicit retry.
    return task_suite.run(source,build,output)


def run(source,build,inputs,case,output,preflight=False):
    source,build,inputs,output=map(lambda p:Path(p).resolve(),(source,build,inputs,output))
    protocol=json.loads((inputs/'protocol.json').read_text())
    if preflight:
        # All cases share the pinned source/module. The substantive adapter has
        # CPU checks; these are genuine unchanged target controller regressions.
        suite_output=output.parent.parent/'_task_suite'/protocol['source_commit']
        code=ensure_suite(source,build,suite_output)
        if code:
            output.mkdir(parents=True,exist_ok=True)
            write_json(output/'metrics.json',dict(campaign_id='ControlChoices-readiness',execution_error='Required target task suite failed or incompatible',
                task_success=None,physical_pass=None,suite_evidence=str(suite_output)))
            return code
        return task.run(source,build,None,output,'B',True,inputs)
    matches=[v for v in protocol['variants'] if v['id']==case['parameters']['variant']]
    if len(matches)!=1:raise ValueError('Unknown frozen control variant')
    variant=matches[0]
    original_select,original_native,original_execute=task.select_protocol,task.native_module,task.execute
    production=None;old_constructor=None;offline=None;original_shift=None
    current={}
    def select(src,selected,mode,pf,inp):
        base=original_select(src,dict(id=case['parameters']['base_task']),mode,pf,inp)
        current.update(apply_protocol(base,variant,protocol,pf));return current
    def preload(src,bld,N=32):
        nonlocal production,old_constructor,offline,original_shift
        result=original_native(src,bld,N)
        production=importlib.import_module('gato_tiago.tiago_mpc_controller')
        old_constructor=production.MPC_GATO
        def constructor(*args,**kwargs):
            gripper_identity=None
            if 'gripper_angle_rad' in variant:
                plant,gripper_identity=gripper_plant(inputs/protocol['C06_gripper']['asset'],variant['gripper_angle_rad'],protocol['C06_gripper']['asset_sha256'])
                plant.gravity.linear=np.asarray(current['gravity_m_s2'])
                args=(plant,)+args[1:]
                current['plant_model_variation']=gripper_identity
            if 'switch_distance_m' in variant:
                after={k:current['solver_params'][k] for k in PARAM_KEYS}
                for k in protocol['C03_policy']['scaled']:after[k]*=10
                after['N_cost']=after['q_cost'];after['ee_orient_N_cost']=after['ee_orient_cost']
                kwargs.update(final_approach=True,final_approach_params=after,
                    final_approach_trigger=dict(distance_m=variant['switch_distance_m'],max_joint_speed_rad_s=.1))
            elif variant.get('final_approach'):
                kwargs.update(final_approach=True,final_approach_params=protocol['C04_final_params'],final_approach_trigger=protocol['C04_trigger'])
            owner=old_constructor(*args,**kwargs)
            if variant.get('holding'):
                import pinocchio as pin
                state=np.asarray(current['initial_state']);data=owner.model.createData()
                pin.framesForwardKinematics(owner.model,data,state[:7])
                placement=data.oMf[owner.model.getFrameId('arm_right_tool_link')]
                current['goal']=np.r_[placement.translation,pin.rpy.matrixToRpy(placement.rotation)].tolist()
                current['resolved_holding_reference']='FK of approved initial state'
                write_json(output/'protocol.json',current)
            if variant.get('seed_policy'):
                offline=importlib.import_module('bsqp.offline_mpc');original_shift=offline.shift_packed_trajectory_warm_start
                offline.shift_packed_trajectory_warm_start=lambda previous,state,nx,nu,knots,elapsed,dt:seed(variant['seed_policy'],previous,state,nx,nu,knots)
            write_json(output/'control_choices.json',dict(variant=variant,resolved_solver_params=owner.solver_params,
                final_approach=dict(enabled=owner.final_approach.enabled,after=owner.final_approach.after,trigger=owner.final_approach.trigger),
                adapter_sha256=sha(__file__),protocol_sha256=sha(inputs/'protocol.json'),plant_model_variation=gripper_identity))
            return owner
        production.MPC_GATO=constructor
        return result
    def execute(owner,p,emit,helpers):
        if not p['full_observation']:return original_execute(owner,p,emit,helpers)
        def observed(event,record):
            if variant.get('seed_policy') and event.startswith('solve_'):
                record=dict(record,control_seed_policy=variant['seed_policy'],effective_seed_shift_s=0.,
                    parent_shift_semantics='Original warm_start_shift_s records elapsed optimizer age; this experiment overrides seed construction only')
            emit(event,record)
        return owner.run_mpc_goals(np.asarray(p['initial_state'],dtype=np.float32),[np.asarray(p['goal'],dtype=np.float32)],
            sim_dt=p['sim_dt_s'],goal_timeout=8.,goal_threshold=.05,goal_axis_threshold=np.deg2rad(30.),velocity_threshold=1.,
            goal_dwell_time=p['observation_dwell_s'],reset_rho_each_solve=False,initial_guess=None,
            offline_timing='solve_time',record_raw=True,record_callback=observed)
    try:
        task.select_protocol,task.native_module,task.execute=select,preload,execute
        code=task.run(source,build,dict(id=case['parameters']['base_task']),output,'B',False,inputs)
        if current.get('full_observation'):window_metrics(output,variant.get('holding',False))
        metrics=json.loads((output/'metrics.json').read_text());metrics.update(campaign_id=variant['study'],research_question='RQ-control-choices',
            control_variant=variant['id'],base_task=case['parameters']['base_task'],collection_protocol=protocol['id'],
            plant_model_variation=current.get('plant_model_variation'))
        write_json(output/'metrics.json',metrics)
        manifest=json.loads((output/'SHA256SUMS.json').read_text())
        for filename in ('metrics.json','protocol.json','control_choices.json'):
            if (output/filename).is_file():manifest[filename]=sha(output/filename)
        write_json(output/'SHA256SUMS.json',manifest)
        return code
    finally:
        task.select_protocol,task.native_module,task.execute=original_select,original_native,original_execute
        if production is not None and old_constructor is not None:production.MPC_GATO=old_constructor
        if offline is not None and original_shift is not None:offline.shift_packed_trajectory_warm_start=original_shift


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','build','inputs','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--case',type=Path);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if not args.case and not args.preflight:parser.error('--case or --preflight required')
    case=json.loads(args.case.read_text()) if args.case else None
    sys.exit(run(args.source,args.build,args.inputs,case,args.output,args.preflight))
