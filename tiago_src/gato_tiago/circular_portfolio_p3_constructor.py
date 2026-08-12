"""Capability-blocked direct computed-torque constructor for P3."""

from __future__ import annotations

import time
from typing import Mapping

import numpy as np

from gato_tiago.circular_portfolio import DENSE_SAMPLES,DENSE_SUBSTEPS,DT
from gato_tiago.circular_portfolio_p3 import KD,KP,planned_proxy,certify_planned_proxy


CONSTRUCTOR_EXECUTION_AUTHORIZATION=None


def computed_torque_rollout(proxy:Mapping,q0,rnea,aba,kinematics,*,deadline,
                            monotonic=time.monotonic,execution_counts=None,
                            phase="primary"):
    """One deterministic rollout; interval controls are canonical float32."""
    q0=np.asarray(q0,np.float64); substep=DT/DENSE_SUBSTEPS
    state=np.empty((DENSE_SAMPLES,14),np.float64)
    tool=np.empty((DENSE_SAMPLES,3),np.float64)
    control=np.empty((95,7),np.float32)
    state[0]=np.r_[q0,np.zeros(7)]; tool[0]=kinematics(q0)[0]
    if execution_counts is not None:execution_counts[f"{phase}_fk_calls"]+=1
    index=0; rnea_calls=0; aba_calls=0; fk_calls=1
    for knot in range(95):
        if monotonic()>=deadline: raise TimeoutError("P3 campaign wall limit")
        q,qd=state[index,:7],state[index,7:]
        acceleration=(proxy["planned_qdd_float64"][knot]
            +KP*(proxy["planned_q_float64"][knot]-q)
            +KD*(proxy["planned_qd_float64"][knot]-qd))
        control[knot]=np.asarray(rnea(q,qd,acceleration),np.float32); rnea_calls+=1
        if execution_counts is not None:execution_counts[f"{phase}_rnea_calls"]+=1
        applied=control[knot].astype(np.float64)
        for _ in range(DENSE_SUBSTEPS):
            q,qd=state[index,:7],state[index,7:]
            qdd=np.asarray(aba(q,qd,applied),np.float64); aba_calls+=1
            if execution_counts is not None:execution_counts[f"{phase}_aba_calls"]+=1
            state[index+1,:7]=q+substep*qd+.5*substep**2*qdd
            state[index+1,7:]=qd+substep*qdd
            index+=1; tool[index]=kinematics(state[index,:7])[0]; fk_calls+=1
            if execution_counts is not None:execution_counts[f"{phase}_fk_calls"]+=1
    if execution_counts is not None:execution_counts[f"{phase}_rollouts"]+=1
    return {"applied_controls_float32":control,"pin_dense_state_float64":state,
        "pin_dense_tool_float64":tool,"measured_call_counts":{
            "rnea_calls":rnea_calls,"aba_calls":aba_calls,"fk_calls":fk_calls}}


def independent_replay(q0,controls,aba,kinematics,*,deadline,
                       monotonic=time.monotonic,execution_counts=None,
                       phase="immediate"):
    q0=np.asarray(q0,np.float64); controls=np.asarray(controls)
    if controls.shape!=(95,7) or controls.dtype!=np.float32:
        raise ValueError("P3 replay controls must be canonical float32")
    substep=DT/DENSE_SUBSTEPS; state=np.empty((DENSE_SAMPLES,14),np.float64)
    tool=np.empty((DENSE_SAMPLES,3),np.float64); state[0]=np.r_[q0,np.zeros(7)]
    tool[0]=kinematics(q0)[0]; index=0; aba_calls=0;fk_calls=1
    if execution_counts is not None:execution_counts[f"{phase}_fk_calls"]+=1
    for row in controls:
        if monotonic()>=deadline: raise TimeoutError("P3 campaign wall limit")
        applied=row.astype(np.float64)
        for _ in range(DENSE_SUBSTEPS):
            q,qd=state[index,:7],state[index,7:];qdd=aba(q,qd,applied);aba_calls+=1
            if execution_counts is not None:execution_counts[f"{phase}_aba_calls"]+=1
            state[index+1,:7]=q+substep*qd+.5*substep**2*qdd
            state[index+1,7:]=qd+substep*qdd;index+=1
            tool[index]=kinematics(state[index,:7])[0];fk_calls+=1
            if execution_counts is not None:execution_counts[f"{phase}_fk_calls"]+=1
    if execution_counts is not None:execution_counts[f"{phase}_replays"]+=1
    return state,tool,{"aba_calls":aba_calls,"fk_calls":fk_calls}


def certify_rollout(row,q0,q_goal,direction,route,index,lower,upper,velocity,
                    effort,goal_tool,pillar,reference,kinematics,aba,deadline,
                    reconstruct_cost,open_turn,default_side,monotonic=time.monotonic,
                    execution_counts=None,replay_phase="immediate"):
    proxy=row["proxy"]; planned=certify_planned_proxy(proxy,q0,q_goal,direction,route,index)
    state=np.asarray(row["pin_dense_state_float64"]);tool=np.asarray(row["pin_dense_tool_float64"])
    controls=np.asarray(row["applied_controls_float32"])
    replay_state,replay_tool,replay_counts=independent_replay(
        q0,controls,aba,kinematics,deadline=deadline,monotonic=monotonic,
        execution_counts=execution_counts,phase=replay_phase)
    solver=np.arange(96)*DENSE_SUBSTEPS
    base=reconstruct_cost(state[solver],controls.astype(np.float64),tool[solver],reference,
                          lower,upper,velocity,effort,include_toll=False)
    full=reconstruct_cost(state[solver],controls.astype(np.float64),tool[solver],reference,
                          lower,upper,velocity,effort,include_toll=True)
    planned_tool=[]
    for q in proxy["planned_q_float64"]:
        planned_tool.append(kinematics(q)[0])
        if execution_counts is not None:execution_counts[f"{replay_phase}_semantic_fk_calls"]+=1
    planned_tool=np.asarray(planned_tool)
    deviation=np.linalg.norm(tool[solver]-planned_tool,axis=1)
    speed=np.linalg.norm(kinematics(state[-1,:7])[1]@state[-1,7:])
    if execution_counts is not None:execution_counts[f"{replay_phase}_semantic_fk_calls"]+=1
    turn=float(open_turn(tool,pillar))
    # Open-polar turn has the opposite sign of the midpoint displacement.
    expected_sign=-int(default_side) if route=="short" else int(default_side)
    gates={"proxy":planned["passes"],"primary_finite":np.isfinite(state).all()
        and np.isfinite(tool).all() and np.isfinite(controls).all(),
        "primary_shapes":state.shape==(DENSE_SAMPLES,14) and tool.shape==(DENSE_SAMPLES,3),
        "common_x0":np.array_equal(state[0],np.r_[q0,np.zeros(7)]),
        "independent_exact":np.array_equal(state,replay_state) and np.array_equal(tool,replay_tool),
        "q_limits":np.all(state[:,:7]>=lower) and np.all(state[:,:7]<=upper),
        "v_limits":np.all(np.abs(state[:,7:])<=velocity),
        "u_limits":np.all(np.abs(controls.astype(np.float64))<=effort),
        "terminal":np.linalg.norm(tool[-1]-goal_tool)<=.015,
        "speed":speed<=.05,
        "clearance":np.min(np.linalg.norm(tool[:,:2]-pillar[None],axis=1)-.03)>=.005,
        "topology":abs(turn)>2.4 and int(np.sign(turn))==expected_sign,
        "proxy_tracking_rms":np.sqrt(np.mean(deviation**2))<=.002,
        "proxy_tracking_max":np.max(deviation,initial=0.)<=.004,
        "primary_counts":row["measured_call_counts"]=={
            "rnea_calls":95,"aba_calls":6080,"fk_calls":6081},
        "replay_counts":replay_counts=={"aba_calls":6080,"fk_calls":6081}}
    detail={"gates":gates,"base_cost":base["cost"],"full_cost":full["cost"],
        "toll_contribution":full["toll_contribution"],
        "toll_residual_float64":full["toll_residual_float64"],
        "tool_length_m":float(np.sum(np.linalg.norm(np.diff(tool,axis=0),axis=1))),
        "turn":turn,"terminal_error_m":float(np.linalg.norm(tool[-1]-goal_tool)),
        "final_speed_mps":float(speed),"minimum_clearance_m":float(np.min(
            np.linalg.norm(tool[:,:2]-pillar[None],axis=1)-.03)),
        "proxy_tracking_rms_m":float(np.sqrt(np.mean(deviation**2))),
        "proxy_tracking_max_m":float(np.max(deviation,initial=0.)),
        "passes":bool(all(gates.values()))}
    return detail


def execute_direct_constructor(identity,q0,q_goal,direction,lower,upper,velocity,
    effort,goal_tool,pillar,reference,kinematics,rnea,aba,reconstruct_cost,open_turn,
    *,default_side,deadline,monotonic=time.monotonic,authorization=None,
    execution_counts=None):
    if CONSTRUCTOR_EXECUTION_AUTHORIZATION is None or authorization is not CONSTRUCTOR_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P3 direct constructor is blocked")
    route,index=identity[-2],identity[-1]; proxy=planned_proxy(q0,q_goal,direction,route,index)
    row=computed_torque_rollout(proxy,q0,rnea,aba,kinematics,
        deadline=deadline,monotonic=monotonic,execution_counts=execution_counts,
        phase="primary");row["identity"]=list(identity);row["proxy"]=proxy
    row["certificate"]=certify_rollout(row,q0,q_goal,direction,route,index,lower,upper,
        velocity,effort,goal_tool,pillar,reference,kinematics,aba,deadline,
        reconstruct_cost,open_turn,default_side,monotonic,
        execution_counts,"immediate")
    return row
