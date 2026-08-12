"""Frozen CUDA-authoritative P6 two-route seed pilot contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from gato_tiago.circular_portfolio import (CONTROL_LIMIT_COST,CYLINDER_WEIGHT,N_COST,
    Q_COST,QD_COST,Q_LIMIT_COST,TOLL_WEIGHT,U_COST,VELOCITY_LIMIT_COST,
    smoothstep_toll_residual_gradient,validate_reference_row)

from gato_tiago.circular_portfolio_p5 import (ACTIVE_LANES,BUILD_COMMAND,
    CAMPAIGN_WALL_LIMIT_S,CLEARANCE_MARGIN,DENSE_SAMPLES,DENSE_SUBSTEPS,DT,
    EXTENSION,INTERVALS,KNOTS,LANES,PILLAR_RADIUS,PILOT_IDENTITY,PILOT_ROUTES,
    TOLL_LENGTH,WORKER_WALL_LIMIT_S,array_hash,pack_seed,planned_proxy,
    reconstruct_cost,open_turn)
from gato_tiago.circular_portfolio_p5_v2 import P5_V1_REJECTED_REPORT


PROTOCOL="tiago_tool_center_cuda_authoritative_route_seed_p6_1"
WORKER_PROTOCOL=PROTOCOL+"_worker"
OUTPUT=Path("/tmp/tiago-tool-center-cuda-authoritative-route-seed-p6-authorized-once/p6.json")
KP=25.0
KD=10.0
SOLVER_TRANSITION={"solver_knot_dt":DT,"dense_substeps":DENSE_SUBSTEPS,
    "dense_substep_dt":DT/DENSE_SUBSTEPS,"held_control":True,
    "integrator_source":"gato/dynamics/integrator.cuh","integration":"trapezoidal"}
P5_V2_REJECTED_REPORT={
    "protocol":"tiago_tool_center_constructed_route_portfolio_p5_n260_pilot_v2_1",
    "classification":["parent_child_worker_provenance_adapter_mismatch",
        "scientific_cuda_open_loop_infeasible"],
    "head":"a174b4591e43396a95821b28b17af0e08c7fbea2","exit":1,
    "runner_trigger_elapsed_s":7.956335727008991,"runner_finish_elapsed_s":7.956335727008991,
    "worker_elapsed_s":2.037281984987203,"completed":0,"pending":2,
    "hashes":{"gen0_json":"94b532a3b9236dc62ea0b31ef8680499c2ed2b1a59171729fafa9dd169963385",
        "gen1_json":"3c69a63f6ef4d939dbe018771289cddf47b5ed02db1de8fbeac6e7bbd606d3fc",
        "gen2_json":"233823c92efb8c11633baea9a1fa5aa180ad5af1af78cd3ef87912b0712a49ee",
        "latest_pointer":"f71f6e8336c8a2d10bd8ef6c8ba505c0912df73f3af0d28955478fd2f1f61414",
        "rejection_json":"6cca62b4f2109978b924a62b1a888efe64cac85d13926dc87bbf37ff5071a39b",
        "route0_json":"910ac4b2c2c1dd419a360aadff13c332b6ca9b0bc2cdcba837dd5c888543a7c2",
        "route0_npz":"64305385b598a2146a7841b2715680805957ca18e869f897e11c46c90540da7b",
        "route1_json":"367261e6b1e6e2fadaa93a966f259d428ddd54f80913eac42c30b8073859854d",
        "route1_npz":"cbb5e60df0429103fe2946cd6b95de6a6bfba0a3a0e26d4ba1760db2b2f8b6ad",
        "worker_request":"a806d8796c374ecdc2c4085da3b292e25d6eda313cddc527e5d2e9c9941f4864",
        "worker_input_npz":"c0f367dac4d6692bd9e451a56b7b73c003abf42aed2e09781cb708f0dd16881e",
        "worker_json":"ba76df5346e36f530645d7136bd333c80db5218a90e66e4bb1acfde4c9a4f03b",
        "worker_npz":"37790ddeda47fab47cb3fd265e1bd27279064cee225e307034e39506875fe3ce"},
    "runner_nonzero_counts":{"prerequisite_artifact_loads":1,"pin_model_contexts":1,
        "primary_pin_rollouts":2,"independent_pin_replays":2,"rnea_calls":518,
        "primary_aba_calls":33152,"independent_aba_calls":33152,
        "primary_fk_calls":33154,"independent_fk_calls":33154,
        "task_endpoint_fk_calls":1,"direction_fk_calls":1,"worker_subprocess_attempts":1},
    "actual_worker_counts":{"b1_constructors":1,"b16_constructors":1,
        "b1_sim_forward_calls":16576,"b16_sim_forward_calls":16576,
        "b1_tool_position_calls":16577,"b16_tool_position_calls":16577,
        "solve_calls":0,"sqp_calls":0},
    "b1_lane0_bitwise_equal":True,"b16_distinct_alternating_streams":2,
    "rejected_p5_v2_artifact_loads":0,"oracle_evidence":False,
    "benchmark_evidence":False,"sqp_evidence":False}

SCREEN_RESULTS={
    25:{"short":{"terminal_m":7.31041445674679e-05,"speed_mps":1.972387969111591e-04,
        "q_ratio":.9490805001879538,"v_ratio":.10652093887329102,"u_ratio":.3334095294658954,
        "clearance_m":.015330019231673035,"turn_rad":-3.022348572738704,
        "fk_rms_m":6.487364344823148e-08,"fk_max_m":1.8132213233359923e-07,
        "base_cost":-58.124634283855485,"full_cost":-29.29485069331373},
        "long":{"terminal_m":7.119223928036494e-05,"speed_mps":1.7540905328060588e-04,
        "q_ratio":.9490805001879538,"v_ratio":.1086572289466858,"u_ratio":.3306588392991286,
        "clearance_m":.015258875209717783,"turn_rad":3.2596274624317867,
        "fk_rms_m":6.494739711127643e-08,"fk_max_m":2.02358145891751e-07,
        "base_cost":-57.93026498470906,"full_cost":-57.93026498470906}},
    100:{"short":{"terminal_m":7.267217820546869e-06},"long":{"terminal_m":6.5676394660685e-06}},
    400:{"short":{"terminal_m":1.802318226413288e-06},"long":{"terminal_m":1.7413769422984448e-06}}}


def worker_paths():
    root=OUTPUT.parent
    return {"request":root/"p6.worker.request.json","input":root/"p6.worker.input.npz",
        "npz":root/"p6.worker.npz","json":root/"p6.worker.json",
        "rejected":root/"p6.worker.rejected.json"}


def route_paths(index):
    if index not in (0,1):raise ValueError("invalid P6 route index")
    return {"json":OUTPUT.with_name(f"p6.route.{index}.json"),
        "npz":OUTPUT.with_name(f"p6.route.{index}.npz")}


WORKER_INPUT_SPECS={"x0_float32":((14,),np.float32),
    "q_goal_float64":((7,),np.float64),"direction_float64":((7,),np.float64),
    "short_proxy_q_float64":((KNOTS,7),np.float64),
    "short_proxy_qd_float64":((KNOTS,7),np.float64),
    "short_proxy_qdd_float64":((INTERVALS,7),np.float64),
    "long_proxy_q_float64":((KNOTS,7),np.float64),
    "long_proxy_qd_float64":((KNOTS,7),np.float64),
    "long_proxy_qdd_float64":((INTERVALS,7),np.float64)}
WORKER_OUTPUT_SPECS={"captured_x0_float32":((14,),np.float32),
    "generated_knot_state_float32":((2,KNOTS,14),np.float32),
    "generated_knot_tool_float32":((2,KNOTS,3),np.float32),
    "rnea_q_float64":((2,INTERVALS,7),np.float64),
    "rnea_qd_float64":((2,INTERVALS,7),np.float64),
    "rnea_qdd_float64":((2,INTERVALS,7),np.float64),
    "rnea_u_float64":((2,INTERVALS,7),np.float64),
    "recorded_controls_float32":((2,INTERVALS,7),np.float32),
    "independent_b1_knot_state_float32":((2,KNOTS,14),np.float32),
    "independent_b1_knot_tool_float32":((2,KNOTS,3),np.float32),
    "b1_knot_transition_defect_float32":((2,INTERVALS,14),np.float32),
    "b1_dense_state_float32":((2,DENSE_SAMPLES,14),np.float32),
    "b1_dense_tool_float32":((2,DENSE_SAMPLES,3),np.float32),
    "fresh_b16_knot_state_float32":((LANES,KNOTS,14),np.float32),
    "fresh_b16_knot_tool_float32":((LANES,KNOTS,3),np.float32),
    "b16_knot_transition_defect_float32":((LANES,INTERVALS,14),np.float32),
    "fresh_b16_state_float32":((LANES,DENSE_SAMPLES,14),np.float32),
    "fresh_b16_tool_float32":((LANES,DENSE_SAMPLES,3),np.float32),
    "b1_seed_xu_float32":((KNOTS*21-7,),np.float32),
    "b16_seed_xu_float32":((LANES,KNOTS*21-7,),np.float32)}


def exact_arrays(arrays,specs):
    return bool(set(arrays)==set(specs) and all(np.asarray(arrays[k]).shape==shape
        and np.asarray(arrays[k]).dtype==np.dtype(dtype) and np.isfinite(arrays[k]).all()
        for k,(shape,dtype) in specs.items()))


def feedback_acceleration(proxy,knot,state):
    state=np.asarray(state,np.float64)
    return np.asarray(proxy["planned_qdd_float64"])[knot] \
        +KP*(np.asarray(proxy["planned_q_float64"])[knot]-state[:7]) \
        +KD*(np.asarray(proxy["planned_qd_float64"])[knot]-state[7:])


def reconstruct_cost_components(state,control,tool,reference,lower,upper,velocity,effort):
    state=np.asarray(state,np.float64);control=np.asarray(control,np.float64)
    tool=np.asarray(tool,np.float64);reference=np.asarray(reference)
    if (state.shape!=(KNOTS,14) or control.shape!=(INTERVALS,7)
            or tool.shape!=(KNOTS,3) or reference.shape!=(KNOTS*10,)
            or reference.dtype!=np.float32):raise ValueError("P6 objective arrays invalid")
    ref=reference.reshape(KNOTS,10);residual=np.zeros(KNOTS,np.float64)
    names=("tool_tracking","cylinder","joint_velocity","position_barrier",
        "velocity_barrier","control","control_barrier")
    components={name:0. for name in names}
    barrier=lambda x,lo,hi:float(np.sum(-np.log(np.maximum(x-lo,1e-10))
                                             -np.log(np.maximum(hi-x,1e-10))))
    for knot in range(KNOTS):
        row=validate_reference_row(ref[knot]);error=tool[knot]-row[:3]
        components["tool_tracking"]+=.5*(N_COST if knot==INTERVALS else Q_COST)*float(error@error)
        delta=tool[knot,:2]-row[3:5];keepout=float(row[5]+row[9])
        components["cylinder"]+=.5*CYLINDER_WEIGHT*max(
            1.-float(delta@delta)/keepout**2,0.)**2
        components["joint_velocity"]+=.5*QD_COST*float(state[knot,7:]@state[knot,7:])
        components["position_barrier"]+=Q_LIMIT_COST*barrier(state[knot,:7],lower,upper)
        components["velocity_barrier"]+=VELOCITY_LIMIT_COST*barrier(
            state[knot,7:],-velocity,velocity)
        if knot<INTERVALS:
            residual[knot]=smoothstep_toll_residual_gradient(tool[knot,:2],row)[0]
            components["control"]+=.5*U_COST*float(control[knot]@control[knot])
            components["control_barrier"]+=CONTROL_LIMIT_COST*barrier(
                control[knot],-effort,effort)
    base=float(sum(components.values()));toll=float(.5*TOLL_WEIGHT*(residual@residual))
    return {"non_toll_components":{k:float(v) for k,v in components.items()},
        "base_cost":base,"toll_contribution":toll,"full_cost":base+toll,
        "toll_residual_float64":residual}


def authoritative_metrics(state,cuda_tool,pin_tool,controls,goal,pillar,reference,
        lower,upper,velocity,effort,final_jacobian,route,default_side,solver_knots):
    state=np.asarray(state,np.float64);cuda_tool=np.asarray(cuda_tool,np.float64)
    pin_tool=np.asarray(pin_tool,np.float64);controls=np.asarray(controls)
    samples=KNOTS if solver_knots else DENSE_SAMPLES
    if (state.shape!=(samples,14) or cuda_tool.shape!=(samples,3)
            or pin_tool.shape!=(samples,3)
            or controls.shape!=(INTERVALS,7)):
        return {"passes":False}
    difference=np.linalg.norm(cuda_tool-pin_tool,axis=1)
    cuda_cost=(reconstruct_cost_components(state,controls,cuda_tool,reference,
        lower,upper,velocity,effort) if solver_knots else None)
    final_jacobian=np.asarray(final_jacobian,np.float64)
    if final_jacobian.shape!=(3,7) or not np.isfinite(final_jacobian).all():return {"passes":False}
    midpoint=.5*(lower+upper);half=.5*(upper-lower)
    cuda_turn=open_turn(cuda_tool,pillar);pin_turn=open_turn(pin_tool,pillar)
    expected=-int(default_side) if route=="short" else int(default_side)
    metrics={"q_ratio":float(np.max(np.abs((state[:,:7]-midpoint)/half))),
        "v_ratio":float(np.max(np.abs(state[:,7:])/velocity)),
        "u_ratio":float(np.max(np.abs(controls.astype(np.float64))/effort)),
        "terminal_m":float(np.linalg.norm(cuda_tool[-1]-goal)),
        "speed_mps":float(np.linalg.norm(final_jacobian@state[-1,7:])),
        "clearance_m":float(np.min(np.linalg.norm(cuda_tool[:,:2]-pillar,axis=1)-PILLAR_RADIUS)),
        "pin_clearance_m":float(np.min(np.linalg.norm(pin_tool[:,:2]-pillar,axis=1)-PILLAR_RADIUS)),
        "turn_rad":cuda_turn,"pin_turn_rad":pin_turn,
        "fk_rms_m":float(np.sqrt(np.mean(difference**2))),"fk_max_m":float(np.max(difference)),
        "tool_path_length_m":float(np.sum(np.linalg.norm(np.diff(cuda_tool,axis=0),axis=1))),
        "cuda_base_cost":cuda_cost["base_cost"] if solver_knots else None,
        "cuda_full_cost":cuda_cost["full_cost"] if solver_knots else None,
        "cuda_toll_contribution":cuda_cost["toll_contribution"] if solver_knots else None}
    gates={"finite":all(np.isfinite(value).all() for value in (state,cuda_tool,pin_tool,controls)),
        "limits":max(metrics[k] for k in ("q_ratio","v_ratio","u_ratio"))<=1.,
        "terminal":metrics["terminal_m"]<=.015,"speed":metrics["speed_mps"]<=.05,
        "clearance":min(metrics["clearance_m"],metrics["pin_clearance_m"])>=CLEARANCE_MARGIN,
        "topology":abs(cuda_turn)>=(2.4 if route=="short" else 3.)
            and int(np.sign(cuda_turn))==int(np.sign(pin_turn))==expected,
        "identical_state_fk":metrics["fk_max_m"]<=.001,
        "cost_decomposition":not solver_knots or (cuda_cost["toll_contribution"]>=0.
            and abs(cuda_cost["full_cost"]-cuda_cost["base_cost"]
                    -cuda_cost["toll_contribution"])<=1e-10
            and all(np.isfinite(list(cuda_cost["non_toll_components"].values())))) }
    details={"path_segments_float64":np.linalg.norm(np.diff(cuda_tool,axis=0),axis=1),
        "cuda_cost":cuda_cost,"pin_final_jacobian_float64":final_jacobian,
        "pin_final_jv_float64":final_jacobian@state[-1,7:],
        "pin_dense_tool_float64":pin_tool}
    return {"route":route,"solver_knots":solver_knots,"metrics":metrics,"details":details,"gates":gates,
        "passes":bool(all(gates.values()))}


def static_design():
    return {"protocol":PROTOCOL,"output":str(OUTPUT),"identity":list(PILOT_IDENTITY),
        "routes":[list(x) for x in PILOT_ROUTES],"kp":KP,"kd":KD,
        "solver_transition":SOLVER_TRANSITION,
        "prior_dense_feedback_screen_non_evidence":SCREEN_RESULTS,
        "full_dt_solver_knot_pilot_pending":True,
        "cuda_authoritative_dynamics":True,"pin_dynamic_replay":False,
        "pin_identical_state_fk_jacobian":True,
        "pin_rnea_quarantined_nominal_constructor":True,"pin_rnea_calls":518,
        "pin_rnea_feasibility_evidence":False,"attempts":2,"retries":0,
        "optimizer_calls":0,"solve_calls":0,"sqp_calls":0,"rng_calls":0,
        "extension":EXTENSION,"p5_v1_rejected_report":P5_V1_REJECTED_REPORT,
        "p5_v2_rejected_report":P5_V2_REJECTED_REPORT}
