"""Frozen direct computed-torque construction contract for portfolio P3."""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np

from gato_tiago.circular_portfolio import DT, DENSE_SUBSTEPS, INTERVALS, KNOTS
from gato_tiago.circular_portfolio_constructor import integrate_acceleration
from gato_tiago.circular_portfolio_p2 import endpoint_map


PROTOCOL="tiago_tool_center_constructed_opposite_side_portfolio_p3_1"
BENCHMARK_CLASS="constructed_opposite_side_route_portfolio"
CAMPAIGN_WALL_LIMIT_S=600.0
KP=400.0
KD=40.0
DLS_LAMBDA=0.03
TOLL_LENGTH_M=0.01
SHORT_AMPLITUDES=np.asarray([.15+.01*i for i in range(8)],np.float64)
LONG_AMPLITUDES=np.asarray([.20+.01*i for i in range(8)],np.float64)
P2_REJECTED_REPORT={
    "protocol":"tiago_tool_center_circular_portfolio_p2_1",
    "source_head":"8d770f6fa60eede5629637283c433ad18f68011f",
    "exit_code":1,"identity":["development",12600,"short",0],
    "stage":"runtime_watchdog_rejected","elapsed_s":30.015096527989954,
    "completed":0,"pending":1,
    "evaluation_counts":{"objective":235,"gradient":235,"inequality":234,
        "inequality_jacobian":235,"kinematics":90866,"rnea":89205,
        "rnea_derivatives":89110,"aba":0},
    "phase_timings_s":{"proxy_s":.047100216,"affine_s":.060659846,
        "least_squares_s":.110033873,"optimizer_s":0.0},
    "primary_pin_replay_calls":0,"independent_pin_recert_replay_calls":0,
    "worker_calls":0,"cuda_calls":0,"sqp_calls":0,
    "rejected_p2_artifact_loads":0,
    "hashes":{"rejection_json":"4dbe44ae9f720f49c97632801b30be810ba84af6a9226493cec48c2e90dfba4b",
        "rejection_npz":"8739c76e681f900923b900c9df0ef75cf421d39cabb54650c4b9ad19b6a76d85",
        "latest":"52e71ec32db58bf83158a494b9d86c20cf87cbb2dd0d3bdcb7c820b9c7f18aac",
        "gen0_json":"2821014a7c6549123bcd7c6aecfd568bd556b442b693d61812c988fe73d1b3b7",
        "gen0_npz":"8739c76e681f900923b900c9df0ef75cf421d39cabb54650c4b9ad19b6a76d85",
        "gen1_json":"bc2d92654edefeca264cbba519c92fc7378a1854a09fb2820b80c79dc7e29d43",
        "gen1_npz":"8739c76e681f900923b900c9df0ef75cf421d39cabb54650c4b9ad19b6a76d85"}}


def producer_array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(f"{value.dtype.str}|{value.shape}|".encode()+value.tobytes()).hexdigest()


def endpoint_exact_scalar_bases():
    """Midpoint-sampled quintic/endpoint-zero perturbation, endpoint repaired."""
    tau=(np.arange(INTERVALS,dtype=np.float64)+.5)/INTERVALS
    duration=INTERVALS*DT
    progress=10*tau**3-15*tau**4+6*tau**5
    derivative=30*tau**2-60*tau**3+30*tau**4
    second=60*tau-180*tau**2+120*tau**3
    base=second/duration**2
    perturb=(-np.pi**2*np.sin(np.pi*progress)*derivative**2
             +np.pi*np.cos(np.pi*progress)*second)/duration**2
    full=endpoint_map()
    scalar=full[np.ix_((0,7),np.arange(0,INTERVALS*7,7))]
    right=scalar.T@np.linalg.solve(scalar@scalar.T,np.eye(2))
    base=base+right@(np.asarray((1.,0.))-scalar@base)
    perturb=perturb-right@(scalar@perturb)
    return {"progress_float64":progress,"scalar_endpoint_float64":scalar,
        "scalar_right_inverse_float64":right,"base_acceleration_float64":base,
        "perturbation_acceleration_float64":perturb}


def perturbation_direction(q0,q_goal,start_tool,goal_tool,kinematics,default_side):
    q0=np.asarray(q0,np.float64);q_goal=np.asarray(q_goal,np.float64)
    _position,jacobian=kinematics(.5*(q0+q_goal))
    chord=np.asarray(goal_tool[:2])-np.asarray(start_tool[:2])
    perpendicular=np.asarray((-chord[1],chord[0],0.0),np.float64)
    perpendicular/=np.linalg.norm(perpendicular)
    vector=jacobian.T@np.linalg.solve(jacobian@jacobian.T+DLS_LAMBDA**2*np.eye(3),
                                      perpendicular)
    vector/=np.linalg.norm(vector)
    tool_direction=(jacobian@vector)[:2]
    orientation=int(np.sign(chord[0]*tool_direction[1]-chord[1]*tool_direction[0]))
    if orientation!=int(default_side): vector=-vector
    return vector


def planned_proxy(q0,q_goal,direction,route,profile_index):
    if route not in ("short","long") or not 0<=profile_index<8:
        raise ValueError("invalid P3 lane identity")
    amplitude=(SHORT_AMPLITUDES if route=="short" else LONG_AMPLITUDES)[profile_index]
    # direction is signed to the public default side.  Therefore short uses
    # +direction and long uses -direction.
    sign=1.0 if route=="short" else -1.0
    basis=endpoint_exact_scalar_bases()
    qdd=(basis["base_acceleration_float64"][:,None]*(np.asarray(q_goal)-np.asarray(q0))[None]
          +sign*amplitude*basis["perturbation_acceleration_float64"][:,None]
          *np.asarray(direction)[None])
    q,qd=integrate_acceleration(q0,np.zeros(7),qdd)
    return {**basis,"amplitude_float64":np.asarray(amplitude,np.float64),
        "route_sign_int8":np.asarray(int(sign),np.int8),"planned_qdd_float64":qdd,
        "planned_q_float64":q,"planned_qd_float64":qd}


def certify_planned_proxy(proxy:Mapping,q0,q_goal,direction,route,profile_index):
    fresh=planned_proxy(q0,q_goal,direction,route,profile_index)
    exact=set(proxy)==set(fresh) and all(np.array_equal(np.asarray(proxy[k]),v)
                                         for k,v in fresh.items())
    endpoint=np.r_[fresh["planned_q_float64"][-1]-q_goal,
                   fresh["planned_qd_float64"][-1]]
    gates={"exact":exact,"finite":all(np.isfinite(v).all() for v in fresh.values()),
        "endpoint":np.max(np.abs(endpoint),initial=0.)<=1e-12,
        "direction_unit":abs(np.linalg.norm(direction)-1.)<=1e-12,
        "rank":np.linalg.matrix_rank(fresh["scalar_endpoint_float64"])==2}
    return {"gates":gates,"passes":bool(all(gates.values()))}


def expected_ledger(task_identities):
    return tuple((split,int(seed),route,index) for split,seed in task_identities
                 for route in ("short","long") for index in range(8))
