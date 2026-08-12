"""Frozen zero-SQP CUDA replay contract for the accepted P3 portfolio."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from gato_tiago.circular_portfolio import DENSE_SAMPLES, INTERVALS, KNOTS
from gato_tiago.circular_portfolio_constructor import _open_turn, reconstruct_portfolio_cost


PROTOCOL="tiago_tool_center_constructed_portfolio_cuda_replay_p4_1"
WORKER_PROTOCOL=PROTOCOL+"_worker"
OUTPUT=Path("/tmp/tiago-tool-center-circular-portfolio-p4-cuda-replay-authorized-once/p4.json")
P3_OUTPUT=Path("/tmp/tiago-tool-center-circular-portfolio-p3-cpu-screen-authorized-once/p3.json")
P3_PINS={
    "json":"5a63d1383f4de5d95b5d0af7556b99eb23f6b1f2b4ba5b0a7a7ee6fe5a69be47",
    "manifest":"9b6fb89fd8d741bad89fba283f24ef4cd498b57e625230f0a90d2836d6647880",
    "pointer":"77db605dfecbf1438cef67422cab8acb1093283cbf242b9b147003dddfc516f0",
    "gen194":"3d5e70d15aa2223e6c336efc98f449ced6a7614ee9e6c41253ea2a653b64fd25"}
EXTENSION={"module":"bsqp.bsqpN96_tiago_right_circular_portfolio_toll",
    "relative_path":"python/bsqp/bsqpN96_tiago_right_circular_portfolio_toll.cpython-310-x86_64-linux-gnu.so",
    "sha256":"b079410ade9e7de19ed3d4b7ed6f6ace27172cd442ccea0d5a46bb7277067a2f",
    "size":6690480,"build_head":"2f1011da2a240fe8eae9ff25b2b3ee991c11c6c2",
    "arch":"61-real","KNOT_POINTS":96,"REFERENCE_SIZE":10,
    "TOOL_POSITION_FRAME":"arm_right_tool_joint_origin","TOOL_POSITION_SIZE":3}
TASKS=12
LANES=16
SIM_FORWARD_CALLS=INTERVALS*64
TOOL_POSITION_CALLS=DENSE_SAMPLES


def array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(f"{value.dtype.str}|{value.shape}|".encode()+value.tobytes()).hexdigest()


def worker_paths(task_index):
    if not isinstance(task_index,int) or isinstance(task_index,bool) or not 0<=task_index<TASKS:
        raise ValueError("invalid task index")
    stem=f"p4.worker.{task_index:02d}";root=OUTPUT.parent
    return {"request":root/f"{stem}.request.json","input":root/f"{stem}.input.npz",
        "json":root/f"{stem}.json","npz":root/f"{stem}.npz"}


def input_schema(arrays:Mapping):
    specs={"x0_float32":((LANES,14),np.float32),
        "controls_float32":((LANES,INTERVALS,7),np.float32),
        "b1_seed_xu_float32":((2009,),np.float32),
        "b16_seed_xu_float32":((LANES,2009),np.float32)}
    return bool(set(arrays)==set(specs) and all(np.asarray(arrays[k]).shape==s
        and np.asarray(arrays[k]).dtype==np.dtype(d) and np.isfinite(arrays[k]).all()
        for k,(s,d) in specs.items()) and np.array_equal(arrays["b1_seed_xu_float32"],
        arrays["b16_seed_xu_float32"][0]) and np.array_equal(arrays["x0_float32"],
        np.repeat(arrays["x0_float32"][:1],LANES,axis=0)))


def output_schema(arrays:Mapping):
    specs={"captured_x0_float32":((LANES,14),np.float32),
        "captured_controls_float32":((LANES,INTERVALS,7),np.float32),
        "captured_b1_seed_xu_float32":((2009,),np.float32),
        "captured_b16_seed_xu_float32":((LANES,2009),np.float32),
        "cuda_dense_state_float32":((LANES,DENSE_SAMPLES,14),np.float32),
        "cuda_dense_tool_float32":((LANES,DENSE_SAMPLES,3),np.float32)}
    return bool(set(arrays)==set(specs) and all(np.asarray(arrays[k]).shape==s
        and np.asarray(arrays[k]).dtype==np.dtype(d) and np.isfinite(arrays[k]).all()
        for k,(s,d) in specs.items()))


def certify_lane(pin_state,pin_tool,cuda_state,cuda_tool,controls,goal,pillar,reference,
                 lower,upper,velocity,effort,tool_velocity,route,default_side,
                 accepted_p3_certificate):
    ps=np.asarray(pin_state,np.float64);pt=np.asarray(pin_tool,np.float64)
    cs=np.asarray(cuda_state,np.float64);ct=np.asarray(cuda_tool,np.float64)
    u=np.asarray(controls);ref=np.asarray(reference)
    if (ps.shape!=(DENSE_SAMPLES,14) or cs.shape!=(DENSE_SAMPLES,14)
            or pt.shape!=(DENSE_SAMPLES,3) or ct.shape!=(DENSE_SAMPLES,3)
            or u.shape!=(INTERVALS,7) or u.dtype!=np.float32 or ref.shape!=(KNOTS*10,)
            or ref.dtype!=np.float32):return {"passes":False}
    if route not in ("short","long"):return {"passes":False}
    lower=np.asarray(lower);upper=np.asarray(upper);velocity=np.asarray(velocity);effort=np.asarray(effort)
    speed=float(np.linalg.norm(tool_velocity(cs[-1,:7],cs[-1,7:])))
    clearance=np.linalg.norm(ct[:,:2]-np.asarray(pillar)[None],axis=1)-.03
    disagreement=np.linalg.norm(ct-pt,axis=1)
    indices=np.arange(KNOTS)*64
    pin_costs={name:reconstruct_portfolio_cost(ps[indices],u,pt[indices],ref,lower,upper,
        velocity,effort,include_toll=flag) for name,flag in (("base",False),("full",True))}
    cuda_costs={name:reconstruct_portfolio_cost(cs[indices],u,ct[indices],ref,lower,upper,
        velocity,effort,include_toll=flag) for name,flag in (("base",False),("full",True))}
    pin_turn=_open_turn(pt,pillar);cuda_turn=_open_turn(ct,pillar)
    turn_threshold=2.4 if route=="short" else 3.0
    q_violation=float(np.max(np.maximum(lower-cs[:,:7],cs[:,:7]-upper),initial=0.))
    velocity_ratio=float(np.max(np.abs(cs[:,7:])/velocity,initial=0.))
    effort_ratio=float(np.max(np.abs(u)/effort,initial=0.))
    terminal_error=float(np.linalg.norm(ct[-1]-goal))
    expected_sign=-int(default_side) if route=="short" else int(default_side)
    accepted=accepted_p3_certificate
    gates={"finite":all(np.isfinite(x).all() for x in (ps,pt,cs,ct,u)),
        "initial":np.array_equal(cs[0].astype(np.float32),ps[0].astype(np.float32)),
        "q_limits":q_violation<=0,"velocity_limits":velocity_ratio<=1,
        "effort_limits":effort_ratio<=1,
        "terminal":terminal_error<=.015,"speed":speed<=.05,
        "clearance":np.min(clearance)>=.005,
        "model_tool_agreement":np.max(disagreement)<=.001,
        "turn_agreement":abs(pin_turn)>=turn_threshold and abs(cuda_turn)>=turn_threshold
            and int(np.sign(pin_turn))==int(np.sign(cuda_turn))==expected_sign,
        "cost_agreement":all(abs(pin_costs[k]["cost"]-cuda_costs[k]["cost"])
            <=max(.001,.01*abs(pin_costs[k]["cost"])) for k in ("base","full")),
        "accepted_p3_binding":isinstance(accepted,Mapping) and accepted.get("passes") is True
            and accepted.get("base_cost")==pin_costs["base"]["cost"]
            and accepted.get("full_cost")==pin_costs["full"]["cost"]
            and accepted.get("turn")==pin_turn}
    return {"gates":gates,"speed_m_s":speed,"minimum_clearance_m":float(np.min(clearance)),
        "maximum_tool_disagreement_m":float(np.max(disagreement)),
        "maximum_q_violation":q_violation,"maximum_velocity_ratio":velocity_ratio,
        "maximum_effort_ratio":effort_ratio,"terminal_error_m":terminal_error,
        "pin_base_cost":pin_costs["base"]["cost"],"pin_full_cost":pin_costs["full"]["cost"],
        "cuda_base_cost":cuda_costs["base"]["cost"],"cuda_full_cost":cuda_costs["full"]["cost"],
        "pin_turn":pin_turn,"cuda_turn":cuda_turn,
        "passes":bool(all(gates.values()))}


def certify_task(lanes,controls,tools):
    controls=np.asarray(controls);tools=np.asarray(tools)
    if (len(lanes)!=16 or controls.shape!=(16,95,7) or controls.dtype!=np.float32
            or tools.shape!=(16,DENSE_SAMPLES,3) or not np.isfinite(tools).all()
            or not all(row.get("passes") is True for row in lanes)):return {"passes":False}
    pairs=[]
    for i in range(8):
        short,long=lanes[i],lanes[8+i]
        gates={"pin_base":long["pin_base_cost"]-short["pin_base_cost"]
                >=max(.01,.02*abs(long["pin_base_cost"])),
            "cuda_base":long["cuda_base_cost"]-short["cuda_base_cost"]
                >=max(.01,.02*abs(long["cuda_base_cost"])),
            "pin_reversal":short["pin_full_cost"]-long["pin_full_cost"]
                >=max(.01,.05*abs(long["pin_full_cost"])),
            "cuda_reversal":short["cuda_full_cost"]-long["cuda_full_cost"]
                >=max(.01,.05*abs(long["cuda_full_cost"])),
            "topology":short["pin_turn"]*long["pin_turn"]<0
                and short["cuda_turn"]*long["cuda_turn"]<0}
        pairs.append({"gates":gates,"passes":bool(all(gates.values()))})
    within=[]
    for family in (range(8),range(8,16)):
        for first in family:
            for second in family:
                if first>=second:continue
                within.append(float(np.sqrt(np.mean(np.sum(
                    (tools[first].astype(np.float64)-tools[second])**2,axis=1)))))
    cross=[]
    for first in range(8):
        for second in range(8,16):
            cross.append(float(np.max(np.linalg.norm(
                tools[first,:,:2].astype(np.float64)-tools[second,:,:2],axis=1))))
    portfolio={"unique_control_bytes":len({row.tobytes() for row in controls})==16,
        "unique_cuda_path_bytes":len({row.tobytes() for row in tools})==16,
        "within_family_path_separation":all(value>=.001 for value in within),
        "cross_family_separation":all(value>=.075 for value in cross),
        "cross_family_topology":all(lanes[a]["cuda_turn"]*lanes[b]["cuda_turn"]<0
            for a in range(8) for b in range(8,16))}
    return {"pairs":pairs,"portfolio_gates":portfolio,
        "minimum_within_family_tool_rms_m":min(within),
        "minimum_cross_family_max_separation_m":min(cross),
        "passes":bool(all(row["passes"] for row in pairs) and all(portfolio.values()))}
