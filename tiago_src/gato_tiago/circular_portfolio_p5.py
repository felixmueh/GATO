"""Frozen N260 cross-model pilot contract for the constructed route portfolio."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from gato_tiago.circular_portfolio import (CONTROL_LIMIT_COST, CYLINDER_WEIGHT,
    N_COST, Q_COST, QD_COST, Q_LIMIT_COST, TOLL_WEIGHT, U_COST,
    VELOCITY_LIMIT_COST, smoothstep_toll_residual_gradient, validate_reference_row)


PROTOCOL="tiago_tool_center_constructed_route_portfolio_p5_n260_pilot_1"
WORKER_PROTOCOL=PROTOCOL+"_worker"
OUTPUT=Path("/tmp/tiago-tool-center-constructed-route-portfolio-p5-n260-pilot-authorized-once/p5.json")
KNOTS=260
INTERVALS=259
DENSE_SUBSTEPS=64
DENSE_SAMPLES=INTERVALS*DENSE_SUBSTEPS+1
DT=.0125
KP=100.0
KD=20.0
PILOT_IDENTITY=("development",12600)
PILOT_ROUTES=(("short",0,.15),("long",0,.20))
LANES=16
ACTIVE_LANES=2
PILLAR_RADIUS=.03
CLEARANCE_MARGIN=.005
TOLL_LENGTH=.01
CAMPAIGN_WALL_LIMIT_S=600.0
WORKER_WALL_LIMIT_S=300.0
BUILD_COMMAND=("./tools/build.sh","--plant","tiago_right_constructed_route_portfolio_toll",
    "--knots","260","--target","bsqpN260_tiago_right_constructed_route_portfolio_toll",
    "--native-cuda-arch")
EXTENSION={"module":"bsqp.bsqpN260_tiago_right_constructed_route_portfolio_toll",
    "relative_path":"python/bsqp/bsqpN260_tiago_right_constructed_route_portfolio_toll.cpython-310-x86_64-linux-gnu.so",
    "sha256":None,"size":None,"build_head":None,"arch":"61-real",
    "KNOT_POINTS":260,"REFERENCE_SIZE":10,
    "TOOL_POSITION_FRAME":"arm_right_tool_joint_origin","TOOL_POSITION_SIZE":3}

P4_V2_REJECTED_REPORT={
    "protocol":"tiago_tool_center_constructed_portfolio_cuda_replay_p4_v2_1",
    "classification":"scientific_replay_rejected_and_closed","stage":"replay_failed",
    "completed":0,"pending":12,"error_type":"RuntimeError",
    "error_message":"P4 task replay failed","rejected_p4_v2_artifact_loads":0,
    "trigger_elapsed_s":77.53576374595286,"observed_finish_elapsed_s":77.55036743998062,
    "call_counts":{"p3_artifact_loads":1,"p3_disk_recertifications":1,
        "p3_independent_pin_replays":192,"prerequisite_artifact_loads":2,
        "pin_model_contexts":2,"worker_subprocess_attempts":1,
        "worker_subprocess_successes":1,"b16_constructors":1,
        "sim_forward_calls":6080,"tool_position_calls":6081,"solve_calls":0,
        "sqp_calls":0,"optimizer_calls":0,"rng_calls":0,"task_construction_calls":0},
    "hashes":{"gen0_json":"4cc839641def4514e3089adfc8b3a36af52c2b47d307ed33af85d0317f74c712",
        "gen1_json":"a7571642cfb6e4b10d015bc7851b10fc065e0ee1d9a4322b29eb72da9cfcf73c",
        "latest":"34adc48451cb46090cec3d16e2fff15f2de9fd14962547d6d3197e9003e473a6",
        "rejection_json":"1e95d4ac086d43dc9c0b27563d5ca6cf8bdb5bacabe1c22d6243108fb3daa1f6",
        "rejection_pointer":"9bcce9293a6690ed1baf8154cbf6392bdaf0127de86dca56950364ebcdc539b0",
        "worker_request":"9bdc235004dac88f04f113611799d41ecd77ef920937bacbdb49921196b49c75",
        "worker_input":"d8bf6c730a689049976ba7dcce907be4029d7d3f813ec0ad4735495a00811d4c",
        "worker_json":"cd396a3abe629bd6d4ef19bdb814940aea6ca3a70706db349f262747e06bf49c",
        "worker_npz":"d304d8703d472534a5f5d2003f35071617732c4c79acd42818a852d8b5a8fbed"},
    "oracle_evidence":False,"benchmark_evidence":False,"sqp_evidence":False}


def array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(f"{value.dtype.str}|{value.shape}|".encode()+value.tobytes()).hexdigest()


def endpoint_map():
    scalar=np.empty((2,INTERVALS),np.float64)
    for impulse in range(INTERVALS):
        q=0.;v=0.
        for knot in range(INTERVALS):
            a=1. if knot==impulse else 0.
            q=q+DT*v+.5*DT**2*a;v=v+DT*a
        scalar[:,impulse]=(q,v)
    return scalar


def endpoint_exact_scalar_bases():
    tau=(np.arange(INTERVALS,dtype=np.float64)+.5)/INTERVALS
    duration=INTERVALS*DT
    progress=10*tau**3-15*tau**4+6*tau**5
    derivative=30*tau**2-60*tau**3+30*tau**4
    second=60*tau-180*tau**2+120*tau**3
    base=second/duration**2
    perturb=(-np.pi**2*np.sin(np.pi*progress)*derivative**2
             +np.pi*np.cos(np.pi*progress)*second)/duration**2
    endpoint=endpoint_map();right=endpoint.T@np.linalg.solve(endpoint@endpoint.T,np.eye(2))
    base=base+right@(np.asarray((1.,0.))-endpoint@base)
    perturb=perturb-right@(endpoint@perturb)
    return {"progress_float64":progress,"scalar_endpoint_float64":endpoint,
        "scalar_right_inverse_float64":right,"base_acceleration_float64":base,
        "perturbation_acceleration_float64":perturb}


def integrate_acceleration(q0,qdd):
    q=np.empty((KNOTS,7),np.float64);qd=np.empty_like(q)
    q[0]=np.asarray(q0,np.float64);qd[0]=0.
    for knot in range(INTERVALS):
        q[knot+1]=q[knot]+DT*qd[knot]+.5*DT**2*qdd[knot]
        qd[knot+1]=qd[knot]+DT*qdd[knot]
    return q,qd


def planned_proxy(q0,q_goal,direction,route):
    if route not in ("short","long"):raise ValueError("invalid P5 route")
    amplitude=.15 if route=="short" else .20
    sign=1. if route=="short" else -1.
    basis=endpoint_exact_scalar_bases()
    qdd=(basis["base_acceleration_float64"][:,None]
        *(np.asarray(q_goal)-np.asarray(q0))[None]
        +sign*amplitude*basis["perturbation_acceleration_float64"][:,None]
        *np.asarray(direction)[None])
    q,qd=integrate_acceleration(q0,qdd)
    return {**basis,"amplitude_float64":np.asarray(amplitude),
        "route_sign_int8":np.asarray(int(sign),np.int8),"planned_qdd_float64":qdd,
        "planned_q_float64":q,"planned_qd_float64":qd}


def certify_proxy(value,q0,q_goal,direction,route):
    fresh=planned_proxy(q0,q_goal,direction,route)
    exact=set(value)==set(fresh) and all(np.array_equal(np.asarray(value[k]),v)
                                         for k,v in fresh.items())
    endpoint=np.r_[fresh["planned_q_float64"][-1]-q_goal,
                   fresh["planned_qd_float64"][-1]]
    gates={"exact":exact,"finite":all(np.isfinite(v).all() for v in fresh.values()),
        "endpoint":np.max(np.abs(endpoint),initial=0.)<=1e-12,
        "direction":np.asarray(direction).shape==(7,) and abs(np.linalg.norm(direction)-1)<=1e-12,
        "endpoint_rank":np.linalg.matrix_rank(fresh["scalar_endpoint_float64"])==2}
    return {"gates":gates,"passes":bool(all(gates.values()))}


def pack_seed(q,qd,controls):
    q=np.asarray(q,np.float32);qd=np.asarray(qd,np.float32);controls=np.asarray(controls,np.float32)
    if q.shape!=(KNOTS,7) or qd.shape!=(KNOTS,7) or controls.shape!=(INTERVALS,7):
        raise ValueError("P5 seed arrays invalid")
    seed=np.empty(KNOTS*21-7,np.float32)
    for knot in range(KNOTS):
        offset=knot*21;seed[offset:offset+7]=q[knot];seed[offset+7:offset+14]=qd[knot]
        if knot<INTERVALS:seed[offset+14:offset+21]=controls[knot]
    return seed


def worker_input_schema(arrays:Mapping):
    specs={"x0_float32":((LANES,14),np.float32),
        "controls_float32":((LANES,INTERVALS,7),np.float32),
        "b1_seed_xu_float32":((KNOTS*21-7,),np.float32),
        "b16_seed_xu_float32":((LANES,KNOTS*21-7),np.float32)}
    return bool(set(arrays)==set(specs) and all(np.asarray(arrays[k]).shape==shape
        and np.asarray(arrays[k]).dtype==np.dtype(dtype) and np.isfinite(arrays[k]).all()
        for k,(shape,dtype) in specs.items())
        and np.array_equal(arrays["b1_seed_xu_float32"],arrays["b16_seed_xu_float32"][0])
        and np.array_equal(arrays["x0_float32"],np.repeat(arrays["x0_float32"][:1],LANES,0))
        and not np.array_equal(arrays["controls_float32"][0],arrays["controls_float32"][1]))


def worker_output_schema(arrays:Mapping):
    specs={"captured_x0_float32":((LANES,14),np.float32),
        "captured_controls_float32":((LANES,INTERVALS,7),np.float32),
        "captured_b1_seed_xu_float32":((KNOTS*21-7,),np.float32),
        "captured_b16_seed_xu_float32":((LANES,KNOTS*21-7),np.float32),
        "b1_dense_state_float32":((DENSE_SAMPLES,14),np.float32),
        "b1_dense_tool_float32":((DENSE_SAMPLES,3),np.float32),
        "b16_dense_state_float32":((LANES,DENSE_SAMPLES,14),np.float32),
        "b16_dense_tool_float32":((LANES,DENSE_SAMPLES,3),np.float32)}
    return bool(set(arrays)==set(specs) and all(np.asarray(arrays[k]).shape==shape
        and np.asarray(arrays[k]).dtype==np.dtype(dtype) and np.isfinite(arrays[k]).all()
        for k,(shape,dtype) in specs.items()))


def reconstruct_cost(state,control,tool,reference,lower,upper,velocity,effort,include_toll):
    state=np.asarray(state,np.float64);control=np.asarray(control,np.float64)
    tool=np.asarray(tool,np.float64);reference=np.asarray(reference)
    if state.shape!=(KNOTS,14) or control.shape!=(INTERVALS,7) or tool.shape!=(KNOTS,3) \
            or reference.shape!=(KNOTS*10,) or reference.dtype!=np.float32:
        raise ValueError("P5 objective arrays invalid")
    ref=reference.reshape(KNOTS,10);total=0.;toll=0.;residual=np.zeros(KNOTS)
    barrier=lambda x,lo,hi:float(np.sum(-np.log(np.maximum(x-lo,1e-10))
                                             -np.log(np.maximum(hi-x,1e-10))))
    for k in range(KNOTS):
        row=validate_reference_row(ref[k]);error=tool[k]-row[:3]
        total+=.5*(N_COST if k==KNOTS-1 else Q_COST)*float(error@error)
        delta=tool[k,:2]-row[3:5];keepout=float(row[5]+row[9])
        total+=.5*CYLINDER_WEIGHT*max(1.-float(delta@delta)/keepout**2,0.)**2
        if k<INTERVALS:
            residual[k]=smoothstep_toll_residual_gradient(tool[k,:2],row)[0]
            contribution=.5*TOLL_WEIGHT*residual[k]**2;toll+=contribution
            if include_toll:total+=contribution
        total+=.5*QD_COST*float(state[k,7:]@state[k,7:])
        total+=Q_LIMIT_COST*barrier(state[k,:7],lower,upper)
        total+=VELOCITY_LIMIT_COST*barrier(state[k,7:],-velocity,velocity)
        if k<INTERVALS:
            total+=.5*U_COST*float(control[k]@control[k])
            total+=CONTROL_LIMIT_COST*barrier(control[k],-effort,effort)
    return {"cost":float(total),"toll_contribution":float(toll),
        "toll_residual_float64":residual}


def open_turn(tool,pillar):
    delta=np.asarray(tool)[:,:2]-np.asarray(pillar)[None]
    angle=np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
    return float(angle[-1]-angle[0])


def certify_route(pin_state,pin_tool,cuda_state,cuda_tool,controls,goal,pillar,reference,
                  lower,upper,velocity,effort,tool_velocity,route,default_side):
    ps=np.asarray(pin_state,np.float64);pt=np.asarray(pin_tool,np.float64)
    cs=np.asarray(cuda_state,np.float64);ct=np.asarray(cuda_tool,np.float64)
    controls=np.asarray(controls)
    if any(x.shape!=shape for x,shape in ((ps,(DENSE_SAMPLES,14)),(pt,(DENSE_SAMPLES,3)),
            (cs,(DENSE_SAMPLES,14)),(ct,(DENSE_SAMPLES,3)),(controls,(INTERVALS,7)))):
        return {"passes":False}
    knots=np.arange(KNOTS)*DENSE_SUBSTEPS;delta=np.linalg.norm(pt-ct,axis=1)
    pin_base=reconstruct_cost(ps[knots],controls,pt[knots],reference,lower,upper,velocity,effort,False)
    pin_full=reconstruct_cost(ps[knots],controls,pt[knots],reference,lower,upper,velocity,effort,True)
    cuda_base=reconstruct_cost(cs[knots],controls,ct[knots],reference,lower,upper,velocity,effort,False)
    cuda_full=reconstruct_cost(cs[knots],controls,ct[knots],reference,lower,upper,velocity,effort,True)
    pin_turn=open_turn(pt,pillar);cuda_turn=open_turn(ct,pillar)
    expected=-int(default_side) if route=="short" else int(default_side)
    metrics={"pin_terminal_m":float(np.linalg.norm(pt[-1]-goal)),
        "cuda_terminal_m":float(np.linalg.norm(ct[-1]-goal)),
        "pin_speed_mps":float(np.linalg.norm(tool_velocity(ps[-1,:7],ps[-1,7:]))),
        "cuda_speed_mps":float(np.linalg.norm(tool_velocity(cs[-1,:7],cs[-1,7:]))),
        "tool_disagreement_rms_m":float(np.sqrt(np.mean(delta**2))),
        "tool_disagreement_max_m":float(np.max(delta)),
        "pin_clearance_m":float(np.min(np.linalg.norm(pt[:,:2]-pillar,axis=1)-PILLAR_RADIUS)),
        "cuda_clearance_m":float(np.min(np.linalg.norm(ct[:,:2]-pillar,axis=1)-PILLAR_RADIUS)),
        "pin_turn_rad":pin_turn,"cuda_turn_rad":cuda_turn,
        "pin_base_cost":pin_base["cost"],"pin_full_cost":pin_full["cost"],
        "cuda_base_cost":cuda_base["cost"],"cuda_full_cost":cuda_full["cost"]}
    gates={"finite":all(np.isfinite(x).all() for x in (ps,pt,cs,ct,controls)),
        "q_limits":np.all(ps[:,:7]>=lower) and np.all(ps[:,:7]<=upper)
            and np.all(cs[:,:7]>=lower) and np.all(cs[:,:7]<=upper),
        "v_limits":np.all(np.abs(ps[:,7:])<=velocity) and np.all(np.abs(cs[:,7:])<=velocity),
        "u_limits":np.all(np.abs(controls.astype(np.float64))<=effort),
        "terminal":metrics["pin_terminal_m"]<=.015 and metrics["cuda_terminal_m"]<=.015,
        "speed":metrics["pin_speed_mps"]<=.05 and metrics["cuda_speed_mps"]<=.05,
        "clearance":metrics["pin_clearance_m"]>=CLEARANCE_MARGIN
            and metrics["cuda_clearance_m"]>=CLEARANCE_MARGIN,
        "agreement":metrics["tool_disagreement_max_m"]<=.001,
        "topology":abs(pin_turn)>=(2.4 if route=="short" else 3.0)
            and abs(cuda_turn)>=(2.4 if route=="short" else 3.0)
            and int(np.sign(pin_turn))==int(np.sign(cuda_turn))==expected,
        "model_cost_agreement":all(abs(a-b)<=max(.001,.01*abs(a)) for a,b in
            ((pin_base["cost"],cuda_base["cost"]),(pin_full["cost"],cuda_full["cost"]))) }
    return {"route":route,"metrics":metrics,"gates":gates,"passes":bool(all(gates.values()))}


def certify_pair(short,long):
    if not short.get("passes") or not long.get("passes"):return {"passes":False}
    sm=short["metrics"];lm=long["metrics"]
    gates={"opposite_topology":int(np.sign(sm["pin_turn_rad"]))==-int(np.sign(lm["pin_turn_rad"]))
            and int(np.sign(sm["cuda_turn_rad"]))==-int(np.sign(lm["cuda_turn_rad"])),
        "pin_reversal":lm["pin_base_cost"]-sm["pin_base_cost"]>=max(.01,.02*abs(lm["pin_base_cost"]))
            and sm["pin_full_cost"]-lm["pin_full_cost"]>=max(.01,.05*abs(lm["pin_full_cost"])),
        "cuda_reversal":lm["cuda_base_cost"]-sm["cuda_base_cost"]>=max(.01,.02*abs(lm["cuda_base_cost"]))
            and sm["cuda_full_cost"]-lm["cuda_full_cost"]>=max(.01,.05*abs(lm["cuda_full_cost"]))}
    return {"gates":gates,"passes":bool(all(gates.values()))}


def static_extension_declaration():
    return bool(EXTENSION=={"module":"bsqp.bsqpN260_tiago_right_constructed_route_portfolio_toll",
        "relative_path":"python/bsqp/bsqpN260_tiago_right_constructed_route_portfolio_toll.cpython-310-x86_64-linux-gnu.so",
        "sha256":None,"size":None,"build_head":None,"arch":"61-real",
        "KNOT_POINTS":260,"REFERENCE_SIZE":10,
        "TOOL_POSITION_FRAME":"arm_right_tool_joint_origin","TOOL_POSITION_SIZE":3})
