"""Pure frozen schema for the compact P8 circular-route feasibility pilot."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from gato_tiago.circular_portfolio import (CLEARANCE_MARGIN_M,CONTROL_LIMIT_COST,
    CYLINDER_WEIGHT,LONG_ANGLE_RAD,N_COST,PHYSICAL_RADIUS_M,Q_COST,QD_COST,
    Q_LIMIT_COST,SHORT_ANGLE_RAD,TOLL_WEIGHT,U_COST,VELOCITY_LIMIT_COST,
    smoothstep_toll_residual_gradient,validate_reference_row)
from gato_tiago.circular_portfolio_p5 import EXTENSION


PROTOCOL="tiago_tool_center_circular_route_feasibility_p8_1"
WORKER_PROTOCOL=PROTOCOL+"_worker"
OUTPUT=Path("/tmp/tiago-tool-center-circular-route-feasibility-p8-authorized-once/p8.json")
SPLIT="development";SEED=12600;ROUTES=("short","long")
KNOTS=260;INTERVALS=259;DT=.05;LANES=16
IK_ITERATIONS=8;DLS_DAMPING=.03;KP=25.;KD=10.
AFFINE_SUBSTEPS=16;AFFINE_SAMPLES=INTERVALS*AFFINE_SUBSTEPS+1
RUNNER_WALL_LIMIT_S=600.;WORKER_WALL_LIMIT_S=300.
PILLAR_RADIUS=PHYSICAL_RADIUS_M;CLEARANCE=CLEARANCE_MARGIN_M
RUNNER_EXECUTION_AUTHORIZATION=None
WORKER_EXECUTION_AUTHORIZATION=None

P6_REJECTED_REPORT={
    "protocol":"tiago_tool_center_cuda_authoritative_route_seed_p6_1",
    "classification":"scientific_full_dt_and_dense_infeasible",
    "public_failure_recertification_defect":"required_fresh_pass_instead_of_exact_failure_binding",
    "numerical_metrics_precision":"rounded_report_only",
    "exact_fields_scope":"file_hashes_sizes_call_ledger_and_retained_timings",
    "head":"c3052e0548dd3919d988886ffda7bffc4ef4354f","completed":0,"pending":2,
    "rejected_p6_artifact_loads":0,
    "hashes":{"gen00_json":"906d0f3290e4a976f1994c5d244e46d7d4a6707b49d9096df00d8459960050c0",
        "gen01_json":"b9d4303a9c8ce74b1e99e696fc5e17cb96dbb6a5e36fd4e723f595de578fe8ae",
        "latest":"4ed09779d43cffa855f402fc3b5d80bdc5d463c5f8b2bc4997b9e7d33e1f34d5",
        "rejection":"e57fe3109e2d37ecd421acbc158285f9b970b622a0835299af7dbbbc8b2a3a05",
        "worker_request":"a8220a48f405249b57e9d2419a174e03569c76028be73a44f0cb3bfe8a74eb5c",
        "worker_input":"677bf18d1c7f335ccf463a906e9e7ea174411c303ff7e66423c175eface026c3",
        "worker_json":"f867eb40ba5e811a16216fc77af26cad26dff16d6e3a025598cc3cdecec8d13f",
        "worker_npz":"49f784d37cbff1d124d9eda30c1ea9f90c176027dab17d46686fe7e33a962a51"},
    "sizes":{"gen00_json":11996,"gen01_json":152600,"latest":279,"rejection":22072,
        "worker_request":1008,"worker_input":89866,"worker_json":11225,"worker_npz":21411918},
    "timing":{"trigger_s":5.283755609008949,"cleanup_s":5.352931618981529,
        "worker_s":3.153907111962326,"rnea_s":.015685668971855193},
    "counts":{"prerequisite_loads":1,"worker_attempts":1,"worker_successes":1,
        "b1_constructors":1,"b16_constructors":1,"b1_sim_forward_calls":34188,
        "b1_tool_position_calls":34192,"b16_sim_forward_calls":16835,
        "b16_tool_position_calls":16837,"parent_pin_contexts":1,"worker_pin_contexts":1,
        "total_pin_contexts":2,"constructor_rnea_calls":518,"recert_rnea_calls":518,
        "total_rnea_calls":1036,"parent_start_fk_calls":1,"parent_direction_fk_calls":1,
        "parent_identical_state_fk_calls":33674,"total_parent_kinematics_calls":33676,
        "optimizer_calls":0,"solve_calls":0,"sqp_calls":0,"rng_calls":0},
    "full_dt":{"turn_rad":[.172023693,.172023582],"interior_toll_saturated":[258,258],
        "base_short_advantage":.2023125257,"path_short_advantage_m":.01020029263,
        "q_ratio":[.9490805002,.9490805002],"v_ratio":[.1071876049,.1091134787],
        "u_ratio":[.3330061252,.3306588393],"terminal_m":[2.0495e-7,2.6206e-7],
        "speed_mps":[3.6602e-7,2.0301e-7],"clearance_m":[.303538787,.303538787],
        "path_m":[.1754495015,.1856497941],"base":[-58.12535855,-57.92304602],
        "full":[6.624641454,6.826953979],"toll":[64.75,64.75],
        "topology_pass":False,"reversal_pass":False},
    "dense":{"short":{"q_ratio":15.6483,"v_ratio":41.4629,"terminal_m":.561510,
            "speed_mps":4.34639,"turn_rad":-.838096,"path_m":4.49677},
        "long":{"q_ratio":20.9094,"v_ratio":72.4233,"terminal_m":.429303,
            "speed_mps":3.90843,"turn_rad":-6.05644,"path_m":4.83316}},
    "pin_cuda_fk_submicron":True,"oracle_evidence":False,"benchmark_evidence":False,
    "sqp_evidence":False}

_P7_CONFIGS=((.03,25.,10.,2.1584,2.2429,3.2636,3.1032),
    (.03,100.,20.,2.1584,2.2559,3.2636,3.1218),
    (.03,400.,40.,2.1584,2.3080,3.2636,3.1960),
    (.1,25.,10.,2.1582,2.2427,3.2638,3.1036),
    (.1,100.,20.,2.1582,2.2557,3.2638,3.1222),
    (.1,400.,40.,2.1582,2.3077,3.2638,3.1964))
P7_EXPLORATORY_GRID=tuple({"ik_iterations":iterations,"damping":damping,
    "kp":kp,"kd":kd,"short_v_ratio":sv,"short_u_ratio":su,
    "long_v_ratio":lv,"long_u_ratio":lu,"geometry_topology_reversal_pass":True,
    "dynamics_limits_pass":False,"overall_pass":False}
    for iterations in (8,16,32) for damping,kp,kd,sv,su,lv,lu in _P7_CONFIGS)


def array_hash(value):
    a=np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode()+a.tobytes()).hexdigest()


def worker_paths():
    root=OUTPUT.parent
    return {"request":root/"p8.worker.request.json","input":root/"p8.worker.input.npz",
        "json":root/"p8.worker.json","npz":root/"p8.worker.npz",
        "rejected":root/"p8.worker.rejected.json"}


def checkpoint_path(generation):return OUTPUT.with_name(f"p8.gen{generation}.json")
def latest_path():return OUTPUT.with_name("p8.partial.latest.json")
def rejection_path():return OUTPUT.with_name("p8.rejected.json")
def manifest_path():return OUTPUT.with_name("p8.manifest.json")


CONSTRUCTION_SPECS={
    "x0_float32":((14,),np.float32),"q_goal_float64":((7,),np.float64),
    "goal_tool_float64":((3,),np.float64),"pillar_float64":((2,),np.float64),
    "reference_float32":((KNOTS*10,),np.float32),
    "target_tool_float64":((2,KNOTS,3),np.float64),
    "proxy_q_float64":((2,KNOTS,7),np.float64),
    "proxy_qd_float64":((2,KNOTS,7),np.float64),
    "proxy_qdd_float64":((2,INTERVALS,7),np.float64),
    "dls_q_before_float64":((2,KNOTS-2,IK_ITERATIONS,7),np.float64),
    "dls_position_float64":((2,KNOTS-2,IK_ITERATIONS,3),np.float64),
    "dls_residual_float64":((2,KNOTS-2,IK_ITERATIONS,3),np.float64),
    "dls_jacobian_float64":((2,KNOTS-2,IK_ITERATIONS,3,7),np.float64),
    "dls_dq_float64":((2,KNOTS-2,IK_ITERATIONS,7),np.float64),
    "dls_selected_face_int8":((2,KNOTS-2,IK_ITERATIONS,7),np.int8)}

WORKER_OUTPUT_SPECS={
    "captured_x0_float32":((14,),np.float32),
    "generated_state_float32":((2,KNOTS,14),np.float32),
    "generated_tool_float32":((2,KNOTS,3),np.float32),
    "rnea_q_float64":((2,INTERVALS,7),np.float64),
    "rnea_qd_float64":((2,INTERVALS,7),np.float64),
    "rnea_qdd_float64":((2,INTERVALS,7),np.float64),
    "rnea_u_float64":((2,INTERVALS,7),np.float64),
    "controls_float32":((2,INTERVALS,7),np.float32),
    "fresh_b1_state_float32":((2,KNOTS,14),np.float32),
    "fresh_b1_tool_float32":((2,KNOTS,3),np.float32),
    "b1_defect_float32":((2,INTERVALS,14),np.float32),
    "fresh_b16_state_float32":((LANES,KNOTS,14),np.float32),
    "fresh_b16_tool_float32":((LANES,KNOTS,3),np.float32),
    "b16_defect_float32":((LANES,INTERVALS,14),np.float32),
    "b1_seed_float32":((KNOTS*21-7,),np.float32),
    "b16_seed_float32":((LANES,KNOTS*21-7,),np.float32),
    "affine_cuda_tool_float32":((2,AFFINE_SAMPLES,3),np.float32)}


def exact_arrays(arrays:Mapping,specs:Mapping):
    return bool(set(arrays)==set(specs) and all(np.asarray(arrays[k]).shape==shape
        and np.asarray(arrays[k]).dtype==np.dtype(dtype) and np.isfinite(arrays[k]).all()
        for k,(shape,dtype) in specs.items()))


def pack_seed(q,qd,u):
    q=np.asarray(q,np.float32);qd=np.asarray(qd,np.float32);u=np.asarray(u,np.float32)
    if q.shape!=(KNOTS,7) or qd.shape!=(KNOTS,7) or u.shape!=(INTERVALS,7):
        raise ValueError("P8 seed shapes invalid")
    out=np.empty(KNOTS*21-7,np.float32)
    for k in range(KNOTS):
        offset=21*k;out[offset:offset+7]=q[k];out[offset+7:offset+14]=qd[k]
        if k<INTERVALS:out[offset+14:offset+21]=u[k]
    return out


def reconstruct_cost(state,control,tool,reference,lower,upper,velocity,effort):
    state=np.asarray(state,np.float64);control=np.asarray(control,np.float64)
    tool=np.asarray(tool,np.float64);reference=np.asarray(reference)
    if state.shape!=(KNOTS,14) or control.shape!=(INTERVALS,7) or tool.shape!=(KNOTS,3) \
            or reference.shape!=(KNOTS*10,) or reference.dtype!=np.float32:
        raise ValueError("P8 objective arrays invalid")
    ref=reference.reshape(KNOTS,10);residual=np.zeros(KNOTS);parts={name:0. for name in
        ("tool","cylinder","velocity","q_barrier","v_barrier","control","u_barrier")}
    barrier=lambda x,lo,hi:float(np.sum(-np.log(np.maximum(x-lo,1e-10))
                                             -np.log(np.maximum(hi-x,1e-10))))
    for k in range(KNOTS):
        row=validate_reference_row(ref[k]);error=tool[k]-row[:3]
        parts["tool"]+=.5*(N_COST if k==INTERVALS else Q_COST)*float(error@error)
        delta=tool[k,:2]-row[3:5];keepout=float(row[5]+row[9])
        parts["cylinder"]+=.5*CYLINDER_WEIGHT*max(1.-float(delta@delta)/keepout**2,0.)**2
        parts["velocity"]+=.5*QD_COST*float(state[k,7:]@state[k,7:])
        parts["q_barrier"]+=Q_LIMIT_COST*barrier(state[k,:7],lower,upper)
        parts["v_barrier"]+=VELOCITY_LIMIT_COST*barrier(state[k,7:],-velocity,velocity)
        if k<INTERVALS:
            residual[k]=smoothstep_toll_residual_gradient(tool[k,:2],row)[0]
            parts["control"]+=.5*U_COST*float(control[k]@control[k])
            parts["u_barrier"]+=CONTROL_LIMIT_COST*barrier(control[k],-effort,effort)
    base=float(sum(parts.values()));toll=float(.5*TOLL_WEIGHT*(residual@residual))
    return {"components":parts,"base":base,"toll":toll,"full":base+toll,
        "residual_float64":residual}


def open_turn(tool,pillar):
    delta=np.asarray(tool)[:,:2]-np.asarray(pillar)[None]
    return float(np.unwrap(np.arctan2(delta[:,1],delta[:,0]))[-1]
                 -np.unwrap(np.arctan2(delta[:,1],delta[:,0]))[0])


def static_design():
    return {"protocol":PROTOCOL,"identity":[SPLIT,SEED],"routes":list(ROUTES),
        "ik_iterations":IK_ITERATIONS,"dls_damping":DLS_DAMPING,
        "kp":KP,"kd":KD,"knots":KNOTS,"dt":DT,"horizon_s":INTERVALS*DT,
        "attempts":2,"retries":0,"optimizer_calls":0,"solve_calls":0,"sqp_calls":0,
        "rng_calls":0,"dense_dynamics":False,"affine_fk_samples_only":True,
        "p6_report":P6_REJECTED_REPORT,"p7_grid":P7_EXPLORATORY_GRID,
        "p7_grid_tuning_nonbenchmark":True,"p7_grid_used_only_to_predeclare_simplest":True}
