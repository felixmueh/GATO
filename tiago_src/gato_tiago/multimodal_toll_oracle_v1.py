"""Frozen static contract and pure math for the quarantined Tiago oracle V1."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from gato_tiago import multimodal_toll as toll


ORACLE_PROTOCOL_VERSION = "tiago_tool_center_toll_oracle_v1_1"
ORACLE_EXECUTION_AUTHORIZATION = None
ORACLE_OUTPUT_PATH = Path(
    "/tmp/tiago-tool-center-toll-oracle-v1-authorized-once/oracle.json"
)
AUTHORIZED_CWD = Path("/workspace/GATO")
AUTHORIZED_ORIG_ARGV = (
    "python", "-B", "-m", "gato_tiago.multimodal_toll_oracle_v1_runner",
    "--execute", "--output", str(ORACLE_OUTPUT_PATH),
)

TASK_ARTIFACT_ROOT = Path(
    "/tmp/tiago-tool-center-toll-v4-task-construction-authorized-once"
)
TASK_ARTIFACT_PINS = {
    "final_json": (TASK_ARTIFACT_ROOT / "v4.json", "6695c10a048ed07284422871201c458ed8f3a37fbd965591a3897d90ef682aac"),
    "final_npz": (TASK_ARTIFACT_ROOT / "v4.npz", "4d343d9dc75bf51987c9a18f61239207d1959f59caab64e492ab711ff9bfc7e8"),
    "final_manifest": (TASK_ARTIFACT_ROOT / "v4.manifest.json", "72cef090d49c4f76efe3398b2dfcdd5b72c91cecfe28efc910f4d67a18a01007"),
    "latest_pointer": (TASK_ARTIFACT_ROOT / "v4.partial.latest.json", "3d37c6851f0e92a8b541585aa0f6e61b4f9b4f402c7bd9b038cac2dbdde6d460"),
}
MODEL_ARTIFACT_ROOT = Path(
    "/tmp/tiago-tool-center-toll-v4-model-preflight-v4-authorized-once"
)
MODEL_ARTIFACT_PINS = {
    "final_json": (MODEL_ARTIFACT_ROOT / "model.json", "3493feae03b7b6368a0dc506aa1f0b5067c63c16f90e08b3455c2314f6e7cc16"),
    "final_npz": (MODEL_ARTIFACT_ROOT / "model.npz", "fe29896b3c5cacbfb15be2a66ddc222a88f8e2e5c2646e183cdbaac34bd1fb6e"),
    "final_manifest": (MODEL_ARTIFACT_ROOT / "model.manifest.json", "438df87e388352ebe5762ca6e9cc3163019d1d9a9fdb616d5d4b027e8bc1a92c"),
    "latest_pointer": (MODEL_ARTIFACT_ROOT / "model.partial.latest.json", "d0eddf81f3a7072fda795d92ad7f0e75d869ff62cb282c3eb1b453b1af98539e"),
    "worker_input": (MODEL_ARTIFACT_ROOT / "model.worker-input.npz", "2e0953180dc9661a4740e57a6f94d9a8c4e438e7f4e3b34273bf5c07a154f426"),
}
CUDA_EXTENSION_MODULE="bsqp.bsqpN64_tiago_right_multimodal_toll"
CUDA_EXTENSION_PATH=Path("python/bsqp/bsqpN64_tiago_right_multimodal_toll.cpython-310-x86_64-linux-gnu.so")
CUDA_EXTENSION_SHA256="dbf7a016b644d75a36ebaddb6c25a456dec51814d5f794f5f2c5cfaa2b184ccb"
CUDA_EXTENSION_SIZE_BYTES=6686384
CUDA_BUILD_HEAD="154556ab7b45a5137a6d9c107c1e6e433b9cc057"
CUDA_ARCH="61-real"
CUDA_MODULE_ATTRIBUTES={"KNOT_POINTS":64,"REFERENCE_SIZE":10,"TOOL_POSITION_FRAME":"arm_right_tool_joint_origin","TOOL_POSITION_SIZE":3}

DEVELOPMENT_TASK_SEEDS = tuple(range(12600, 12604))
HELDOUT_TASK_SEEDS = tuple(range(12700, 12708))
EXPECTED_TASK_IDENTITIES = tuple(
    [("development", seed) for seed in DEVELOPMENT_TASK_SEEDS]
    + [("heldout", seed) for seed in HELDOUT_TASK_SEEDS]
)
SIDES = ("default", "alternate")
TEMPLATE_MARGIN_M = (0.003, 0.006, 0.009, 0.012, 0.015)
EXPECTED_LEDGER = tuple(
    (phase, seed, side, margin_index)
    for phase, seed in EXPECTED_TASK_IDENTITIES
    for side in SIDES
    for margin_index in range(5)
)
EXPECTED_PAIR_COUNT = 120
EXPECTED_SIDE_ARTIFACT_COUNT=124*2+120*2+120*2+120*4
KNOTS = 64
INTERVALS = 63
NQ = NV = NU = 7
NX = 14
Z_WIDTH = KNOTS * NX + INTERVALS * NU
DT = 0.0125
DENSE_SUBSTEPS = 64
DENSE_SAMPLES = INTERVALS * DENSE_SUBSTEPS + 1
KEEP_OUT_RADIUS_M = 0.035
PHYSICAL_CLEARANCE_M = 0.005
TEMPLATE_RADII_M = tuple(KEEP_OUT_RADIUS_M + value for value in TEMPLATE_MARGIN_M)
TEMPLATE_ANCHOR_WEIGHT = 200.0
ACQUISITION_MAXITER = 1200
POLISH_MAXITER = 2000
ACQUISITION_WALL_LIMIT_S = 900.0
POLISH_WALL_LIMIT_S = 1800.0
CAMPAIGN_WALL_LIMIT_S = 21600.0
TRUST_CONSTR_OPTIONS = {
    "method": "trust-constr", "gtol": 1e-10, "xtol": 1e-12,
    "barrier_tol": 1e-12, "sparse_jacobian": True,
    "curvature": "BFGS",
}
DLS_ITERATIONS_PER_KNOT = 12
DLS_DAMPING = 0.03
DLS_MARGIN_RAD = 0.01
FACE_STATUS_ORDER = (-1, 0, 1)
FACE_COUNT = 3**7
FACE_FEASIBILITY_TOLERANCE = 1e-12
FREE_NORMAL_RESIDUAL_TOLERANCE = 1e-10
OBJECTIVE_TIE_ABSOLUTE_TOLERANCE = 1e-12
DERIVATIVE_REL_TOLERANCE = 1e-5
DERIVATIVE_ABS_TOLERANCE = 1e-7
PRIMAL_TOLERANCE = 1e-6
ACQUISITION_PRIMAL_TOLERANCE = 1e-4
KKT_TOLERANCE = 1e-5
ACQUISITION_TEMPLATE_RMS_M = 0.015
STABILITY_COST_REL = 0.01
STABILITY_TOOL_RMS_M = 0.01
MIN_INTENDED_MATCHES = 4
MIN_ACTUAL_MODE_ROWS = 4
WITHIN_MODE_COST_SPREAD_REL = 0.01
WITHIN_MODE_TOOL_RMS_M = 0.01
TERMINAL_CUDA_M = 0.015
TERMINAL_PIN_M = 0.020
FINAL_TOOL_SPEED_MPS = 0.05
DENSE_TOOL_MODEL_TOLERANCE_M = 0.001
OPEN_TURN_RAD = 2.0
WINDING_INTEGER_RESIDUAL = 0.10
DISTINCT_CONTROL_RMS = 1e-4
DISTINCT_TOOL_RMS_M = 0.001
FULL_GAP_REL = 0.05
FULL_GAP_ABS = 0.01
BASE_LENGTH_ADVANTAGE_M = 0.005
BASE_COST_TOLERANCE = 1e-6
MODEL_COST_REL = 0.01
MODEL_COST_ABS = 0.001
BOOTSTRAP_SEED = 20260814
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_LOWER_REL = 0.02
COLLISION_SCOPE = "Tiago tool-center homotopy"
EQUALITY_COUNT=NX+INTERVALS*NX
INEQUALITY_COUNT=KNOTS+2

OBJECTIVE_WEIGHTS = {
    "q": 2.0, "N": 260.0, "qd": 0.15, "u": 7.5e-5,
    "cylinder": 800.0, "toll": 0.5,
    "q_barrier": 0.01, "v_barrier": 0.001, "u_barrier": 0.003,
}

REQUIRED_SOURCE_PATHS = {
    "oracle_schema": "tiago_src/gato_tiago/multimodal_toll_oracle_v1.py",
    "oracle_runner": "tiago_src/gato_tiago/multimodal_toll_oracle_v1_runner.py",
    "oracle_worker": "tiago_src/gato_tiago/multimodal_toll_oracle_v1_worker.py",
    "oracle_tests": "tests/python/test_tiago_multimodal_toll_oracle_v1.py",
    "public_schema": "tiago_src/gato_tiago/multimodal_toll.py",
    "task_certifier": "tiago_src/gato_tiago/multimodal_toll_v4_runner.py",
    "model_certifier": "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight_v4_runner.py",
    "construction_oracle": "tiago_src/gato_tiago/multimodal_toll_v4_oracle_schema.py",
    "plant": "gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "model": "gato/dynamics/tiago_right/tiago_right_arm.urdf",
    "bindings": "python/bindings.cu",
    "tool_position_kernel": "gato/bsqp/kernels/tool_position.cuh",
    "integrator": "gato/dynamics/integrator.cuh",
    "simulation": "gato/bsqp/kernels/sim.cuh",
    "solver": "gato/bsqp/bsqp.cuh",
    "tiago_grid": "gato/dynamics/tiago_right/tiago_right_grid.cuh",
    "cuda_helper": "gato/utils/cuda.cuh",
    "cmake": "CMakeLists.txt",
    "build_script": "tools/build.sh",
}

BASE_ARRAY_SPECS={
    "public_x0_float32":((12,14),np.float32),"public_reference_float32":((12,10),np.float32),
    "public_default_side_int8":((12,),np.int8),"quarantined_q8_float64":((12,7),np.float64),
    "joint_lower_float64":((7,),np.float64),"joint_upper_float64":((7,),np.float64),
    "velocity_limit_float64":((7,),np.float64),"effort_limit_float64":((7,),np.float64),
    "bootstrap_input_gaps_float64":((2,8),np.float64),"bootstrap_samples_float64":((2,BOOTSTRAP_RESAMPLES),np.float64),
}
ROW_ARRAY_SPECS={
    "template_float64":((KNOTS,3),np.float64),"seed_q_float64":((KNOTS,7),np.float64),
    "seed_qd_float64":((KNOTS,7),np.float64),"seed_qdd_float64":((INTERVALS,7),np.float64),
    "seed_u_float64":((INTERVALS,7),np.float64),"seed_primal_float64":((Z_WIDTH,),np.float64),
    "dls_q_before":((744,7),np.float64),"dls_position":((744,3),np.float64),
    "dls_jacobian":((744,3,7),np.float64),"dls_residual":((744,3),np.float64),
    "dls_lower":((744,7),np.float64),"dls_upper":((744,7),np.float64),
    "dls_selected_dq":((744,7),np.float64),"dls_selected_status":((744,7),np.int8),
    "dls_selected_objective":((744,),np.float64),"dls_selected_primal":((744,),np.float64),
    "dls_selected_free_residual":((744,),np.float64),"dls_selected_face_index":((744,),np.int64),
    "acquisition_primal_float64":((Z_WIDTH,),np.float64),
    "acquisition_tool_float64":((KNOTS,3),np.float64),
    "acquisition_dense_state_float64":((DENSE_SAMPLES,NX),np.float64),
    "acquisition_dense_tool_float64":((DENSE_SAMPLES,3),np.float64),
    "polish_primal_float64":((Z_WIDTH,),np.float64),"polish_tool_float64":((KNOTS,3),np.float64),
    "pin_dense_state_float64":((DENSE_SAMPLES,NX),np.float64),"pin_dense_tool_float64":((DENSE_SAMPLES,3),np.float64),
    "cuda_dense_state_float32":((DENSE_SAMPLES,NX),np.float32),"cuda_dense_tool_float32":((DENSE_SAMPLES,3),np.float32),
    "dense_time_float64":((DENSE_SAMPLES,),np.float64),"controls_float64":((INTERVALS,NU),np.float64),
    "raw_equality_multipliers_float64":((EQUALITY_COUNT,),np.float64),
    "raw_inequality_multipliers_float64":((INEQUALITY_COUNT,),np.float64),
    "raw_bound_multipliers_float64":((Z_WIDTH,),np.float64),
    "canonical_equality_multipliers_float64":((EQUALITY_COUNT,),np.float64),
    "canonical_inequality_multipliers_float64":((INEQUALITY_COUNT,),np.float64),
    "canonical_lower_multipliers_float64":((Z_WIDTH,),np.float64),
    "canonical_upper_multipliers_float64":((Z_WIDTH,),np.float64),
}
ACQUISITION_ARRAY_NAMES=frozenset(name for name in ROW_ARRAY_SPECS if name.startswith(("template_","seed_","dls_","acquisition_")))
POLISH_ARRAY_NAMES=frozenset(name for name in ROW_ARRAY_SPECS if name.startswith(("polish_","pin_","controls_","dense_time_","raw_","canonical_")))
REPLAY_ARRAY_NAMES=frozenset({"cuda_dense_state_float32","cuda_dense_tool_float32"})


def array_hash(value) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(a.dtype.str.encode() + repr(a.shape).encode() + a.tobytes()).hexdigest()


def quintic_progress(count=KNOTS):
    s = np.linspace(0.0, 1.0, int(count), dtype=np.float64)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def tangent_arc_tangent_template(start_xyz, goal_xyz, center_xy, radius, side):
    """Analytic side template sampled by XY arclength and quintic progress."""
    start = np.asarray(start_xyz, dtype=np.float64)
    goal = np.asarray(goal_xyz, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    if start.shape != (3,) or goal.shape != (3,) or center.shape != (2,):
        raise ValueError("template geometry shapes are fixed")
    if side not in (-1, 1) or radius <= 0:
        raise ValueError("template side/radius invalid")

    def tangent(point):
        delta = point - center
        distance = float(np.linalg.norm(delta))
        if distance <= radius:
            raise ValueError("template endpoint is inside the route circle")
        base = np.arctan2(delta[1], delta[0])
        offset = np.arccos(radius / distance)
        angle = base + side * offset
        return center + radius * np.array([np.cos(angle), np.sin(angle)]), angle

    first, a0 = tangent(start[:2])
    last, a1 = tangent(goal[:2])
    if side > 0:
        while a1 <= a0:
            a1 += 2 * np.pi
    else:
        while a1 >= a0:
            a1 -= 2 * np.pi
    lengths = np.array([
        np.linalg.norm(first - start[:2]), radius * abs(a1 - a0),
        np.linalg.norm(goal[:2] - last),
    ])
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    distance = quintic_progress() * cumulative[-1]
    xy = np.empty((KNOTS, 2), dtype=np.float64)
    for i, value in enumerate(distance):
        if value <= cumulative[1]:
            alpha = value / lengths[0]
            xy[i] = start[:2] + alpha * (first - start[:2])
        elif value <= cumulative[2]:
            alpha = (value - cumulative[1]) / lengths[1]
            angle = a0 + alpha * (a1 - a0)
            xy[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            alpha = (value - cumulative[2]) / lengths[2]
            xy[i] = last + alpha * (goal[:2] - last)
    z = start[2] + quintic_progress() * (goal[2] - start[2])
    result = np.c_[xy, z]
    result[0], result[-1] = start, goal
    return result


def enumerate_box_dls(jacobian, residual, lower, upper):
    """Exhaust all 3^7 faces in lower/free/upper lexicographic order."""
    J = np.asarray(jacobian, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if not (J.shape == (3, 7) and r.shape == (3,) and lo.shape == hi.shape == (7,)):
        raise ValueError("box-DLS fixed shapes violated")
    H = J.T @ J + DLS_DAMPING**2 * np.eye(7)
    rhs = J.T @ r
    status = np.asarray(list(itertools.product(FACE_STATUS_ORDER, repeat=7)), dtype=np.int8)
    dq = np.empty((FACE_COUNT, 7)); objective = np.empty(FACE_COUNT)
    primal = np.empty(FACE_COUNT); normal = np.empty(FACE_COUNT)
    feasible = np.zeros(FACE_COUNT, dtype=np.bool_)
    for index, face in enumerate(status):
        value = np.empty(7); free = face == 0; active = ~free
        value[face == -1] = lo[face == -1]; value[face == 1] = hi[face == 1]
        if np.any(free):
            value[free] = np.linalg.solve(
                H[np.ix_(free, free)], rhs[free] - H[np.ix_(free, active)] @ value[active]
            )
        gradient = H @ value - rhs
        defect = J @ value - r
        dq[index] = value
        objective[index] = 0.5 * defect @ defect + 0.5 * DLS_DAMPING**2 * value @ value
        primal[index] = max(0.0, float(np.max(lo-value)), float(np.max(value-hi)))
        normal[index] = float(np.max(np.abs(gradient[free]))) if np.any(free) else 0.0
        feasible[index] = np.isfinite(value).all() and primal[index] <= FACE_FEASIBILITY_TOLERANCE and normal[index] <= FREE_NORMAL_RESIDUAL_TOLERANCE
    candidates = np.flatnonzero(feasible)
    if not candidates.size:
        raise RuntimeError("no feasible box-DLS face")
    minimum = float(np.min(objective[candidates]))
    selected = int(candidates[objective[candidates] <= minimum + OBJECTIVE_TIE_ABSOLUTE_TOLERANCE][0])
    return {"status": status, "dq": dq, "objective": objective, "primal": primal,
            "free_residual": normal, "feasible": feasible, "selected": selected}


def independently_enumerate_box_dls(jacobian, residual, lower, upper):
    """Separate code-index verifier; never calls the constructor enumerator."""
    J=np.asarray(jacobian,dtype=np.float64); r=np.asarray(residual,dtype=np.float64)
    lo=np.asarray(lower,dtype=np.float64); hi=np.asarray(upper,dtype=np.float64)
    H=J.T@J+DLS_DAMPING**2*np.eye(7); rhs=J.T@r
    status=np.empty((FACE_COUNT,7),dtype=np.int8); dq=np.empty((FACE_COUNT,7))
    objective=np.empty(FACE_COUNT); primal=np.empty(FACE_COUNT); normal=np.empty(FACE_COUNT)
    feasible=np.zeros(FACE_COUNT,dtype=np.bool_)
    for code in range(FACE_COUNT):
        remaining=code
        for joint in range(7):
            divisor=3**(6-joint); digit=remaining//divisor; remaining%=divisor
            status[code,joint]=FACE_STATUS_ORDER[digit]
        face=status[code]; value=np.empty(7); free=np.flatnonzero(face==0); fixed=np.flatnonzero(face!=0)
        for joint in fixed: value[joint]=lo[joint] if face[joint]==-1 else hi[joint]
        if free.size:
            value[free]=np.linalg.solve(H[np.ix_(free,free)],rhs[free]-H[np.ix_(free,fixed)]@value[fixed])
        gradient=H@value-rhs; defect=J@value-r
        dq[code]=value; objective[code]=.5*defect@defect+.5*DLS_DAMPING**2*value@value
        primal[code]=max(0.,float(np.max(lo-value)),float(np.max(value-hi)))
        normal[code]=float(np.max(np.abs(gradient[free]))) if free.size else 0.
        feasible[code]=np.isfinite(value).all() and primal[code]<=FACE_FEASIBILITY_TOLERANCE and normal[code]<=FREE_NORMAL_RESIDUAL_TOLERANCE
    candidates=np.flatnonzero(feasible)
    if not candidates.size: raise RuntimeError("independent box-DLS found no feasible face")
    minimum=float(np.min(objective[candidates])); selected=int(candidates[objective[candidates]<=minimum+OBJECTIVE_TIE_ABSOLUTE_TOLERANCE][0])
    return {"status":status,"dq":dq,"objective":objective,"primal":primal,
            "free_residual":normal,"feasible":feasible,"selected":selected}


def certify_box_dls_independently(jacobian,residual,lower,upper,retained):
    verified=independently_enumerate_box_dls(jacobian,residual,lower,upper)
    selected=int(retained["selected"]); other=int(verified["selected"])
    gates={
        "all_faces_identical":all(np.array_equal(np.asarray(retained[name]),verified[name])
                                   for name in ("status","feasible")),
        "selected_dq_maxabs":float(np.max(np.abs(np.asarray(retained["dq"])[selected]-verified["dq"][other])))<=1e-10,
        "selected_objective":abs(float(np.asarray(retained["objective"])[selected])-float(verified["objective"][other]))<=1e-12,
        "selected_index":selected==other,
    }
    return {**gates,"passes":bool(all(gates.values())),"verifier_calls_constructor_enumerator":False}


def acquisition_joint_path(template, q0, q8, lower, upper, position_jacobian):
    """Fixed continuation IK; callback is the sole model dependency."""
    template = np.asarray(template, dtype=np.float64)
    q = np.asarray(q0, dtype=np.float64).copy()
    path = [q.copy()]; histories = []
    for knot in range(1, KNOTS - 1):
        steps = []
        for _ in range(DLS_ITERATIONS_PER_KNOT):
            position, jacobian = position_jacobian(q)
            lo = np.asarray(lower, dtype=np.float64) + DLS_MARGIN_RAD - q
            hi = np.asarray(upper, dtype=np.float64) - DLS_MARGIN_RAD - q
            residual=template[knot]-position
            table = enumerate_box_dls(jacobian, residual, lo, hi); selected=int(table["selected"])
            selected_dq=np.asarray(table["dq"][selected]).copy()
            steps.append({"q_before":q.copy(),"position":np.asarray(position).copy(),
                          "jacobian":np.asarray(jacobian).copy(),"residual":residual.copy(),
                          "lower":lo.copy(),"upper":hi.copy(),"selected_dq":selected_dq,
                          "selected_status":np.asarray(table["status"][selected]).copy(),
                          "selected_objective":float(table["objective"][selected]),
                          "selected_primal":float(table["primal"][selected]),
                          "selected_free_residual":float(table["free_residual"][selected]),
                          "selected_face_index":selected})
            q = q + selected_dq
        histories.append(steps); path.append(q.copy())
    path.append(np.asarray(q8, dtype=np.float64).copy())
    return np.stack(path), histories


def compact_dls_history_arrays(histories):
    flat=[step for knot in histories for step in knot]
    if len(flat)!=(KNOTS-2)*DLS_ITERATIONS_PER_KNOT:
        raise ValueError("compact DLS history must retain exactly 744 steps")
    fields=("q_before","position","jacobian","residual","lower","upper","selected_dq",
            "selected_status","selected_objective","selected_primal","selected_free_residual","selected_face_index")
    return {f"dls_{name}":np.stack([np.asarray(step[name]) for step in flat]) for name in fields}


def certify_compact_dls_history(arrays):
    expected={"dls_q_before":((744,7),np.float64),"dls_position":((744,3),np.float64),
              "dls_jacobian":((744,3,7),np.float64),"dls_residual":((744,3),np.float64),
              "dls_lower":((744,7),np.float64),"dls_upper":((744,7),np.float64),
              "dls_selected_dq":((744,7),np.float64),"dls_selected_status":((744,7),np.int8),
              "dls_selected_objective":((744,),np.float64),"dls_selected_primal":((744,),np.float64),
              "dls_selected_free_residual":((744,),np.float64),"dls_selected_face_index":((744,),np.int64)}
    exact=set(arrays)==set(expected) and all(np.asarray(arrays[name]).shape==shape and np.asarray(arrays[name]).dtype==np.dtype(dtype)
                                                    for name,(shape,dtype) in expected.items())
    if not exact or not all(np.isfinite(np.asarray(v)).all() for v in arrays.values()):
        return {"exact_schema":exact,"passes":False}
    max_dq=max_objective=max_primal=max_free=0.; index_match=True; status_match=True
    for k in range(744):
        verified=independently_enumerate_box_dls(arrays["dls_jacobian"][k],arrays["dls_residual"][k],
                                                 arrays["dls_lower"][k],arrays["dls_upper"][k])
        index=int(arrays["dls_selected_face_index"][k]); other=int(verified["selected"])
        index_match &= index==other
        status_match &= np.array_equal(arrays["dls_selected_status"][k],verified["status"][other])
        max_dq=max(max_dq,float(np.max(np.abs(arrays["dls_selected_dq"][k]-verified["dq"][other]))))
        max_objective=max(max_objective,abs(float(arrays["dls_selected_objective"][k])-float(verified["objective"][other])))
        max_primal=max(max_primal,abs(float(arrays["dls_selected_primal"][k])-float(verified["primal"][other])))
        max_free=max(max_free,abs(float(arrays["dls_selected_free_residual"][k])-float(verified["free_residual"][other])))
    gates={"exact_schema":True,"selected_indices_match":index_match,
           "selected_status_match":status_match,
           "selected_dq_maxabs":max_dq<=1e-10,"selected_objective_maxabs":max_objective<=1e-12,
           "selected_primal_maxabs":max_primal<=1e-12,"selected_free_residual_maxabs":max_free<=1e-10}
    return {**gates,"max_dq":max_dq,"max_objective":max_objective,"max_primal":max_primal,
            "max_free_residual":max_free,"independent_reenumeration_calls_constructor":False,
            "passes":bool(all(gates.values()))}


def build_acquisition_initial(public_x0, public_reference, public_default_side,
                              q8, lower, upper, requested_side, margin_index,
                              position_jacobian, rnea):
    """Quarantined q8/template-aware acquisition initializer."""
    if margin_index not in range(5): raise ValueError("template margin index")
    x0=np.asarray(public_x0,dtype=np.float64); q8=np.asarray(q8,dtype=np.float64)
    if x0.shape!=(14,) or q8.shape!=(7,): raise ValueError("acquisition endpoint shapes")
    ref=toll.TollReference.from_solver_bytes(np.asarray(public_reference,dtype=np.float32))
    start,_=position_jacobian(x0[:7]); sign=canonical_mode_sign(public_default_side,requested_side)
    template=tangent_arc_tangent_template(start,ref.goal_xyz,ref.cylinder_xy,
                                          TEMPLATE_RADII_M[margin_index],sign)
    q,history=acquisition_joint_path(template,x0[:7],q8,lower,upper,position_jacobian)
    qd,qdd,u=knot_derivatives_and_rnea(q,rnea); x=np.c_[q,qd]
    return {"template_float64":template,"q_float64":q,"qd_float64":qd,
            "qdd_float64":qdd,"u_float64":u,"primal_float64":pack_z(x,u),
            **compact_dls_history_arrays(history),"canonical_mode_sign":sign,
            "q8_oracle_only":True,"benchmark_seed_eligible":False}


def knot_derivatives_and_rnea(q_path, rnea):
    q = np.asarray(q_path, dtype=np.float64)
    if q.shape != (KNOTS, 7):
        raise ValueError("q path must be 64x7")
    qd = np.zeros_like(q); qdd = np.empty((INTERVALS, 7)); u = np.empty((INTERVALS, 7))
    for k in range(INTERVALS):
        qd[k+1] = 2 * (q[k+1]-q[k]) / DT - qd[k]
        qdd[k] = (qd[k+1]-qd[k]) / DT
        u[k] = rnea(q[k], qd[k], qdd[k])
    return qd, qdd, u


def pack_z(x, u):
    x = np.asarray(x, dtype=np.float64); u = np.asarray(u, dtype=np.float64)
    if x.shape != (KNOTS, NX) or u.shape != (INTERVALS, NU):
        raise ValueError("oracle primal shapes invalid")
    return np.r_[x.ravel(), u.ravel()]


def unpack_z(z):
    value = np.asarray(z, dtype=np.float64)
    if value.shape != (Z_WIDTH,):
        raise ValueError("oracle primal width must be 1337")
    return value[:KNOTS*NX].reshape(KNOTS, NX), value[KNOTS*NX:].reshape(INTERVALS, NU)


POLISH_INPUT_FIELDS = frozenset({
    "acquisition_primal_float64", "public_x0_float32", "public_reference_float32",
    "joint_lower_float64", "joint_upper_float64", "velocity_limit_float64",
    "effort_limit_float64", "solver_options",
})


def validate_unanchored_polish_input(payload: Mapping):
    """The certifying boundary excludes q8, side, templates, and oracle metadata."""
    if set(payload) != POLISH_INPUT_FIELDS:
        raise ValueError("unanchored polish input has tainted or missing fields")
    if np.asarray(payload["acquisition_primal_float64"]).shape != (Z_WIDTH,):
        raise ValueError("unanchored acquisition primal shape")
    if np.asarray(payload["public_x0_float32"]).shape != (14,):
        raise ValueError("unanchored x0 shape")
    if np.asarray(payload["public_reference_float32"]).shape != (10,):
        raise ValueError("unanchored reference shape")
    if payload["solver_options"] != {**TRUST_CONSTR_OPTIONS, "maxiter":POLISH_MAXITER}:
        raise ValueError("unanchored solver options changed")
    return True


def certify_polish_problem_binding(problem,payload):
    if type(problem) is not OriginalOracleNLP or problem.anchored:
        return False
    validate_unanchored_polish_input(payload)
    return bool(np.array_equal(problem.x0,np.asarray(payload["public_x0_float32"],dtype=np.float64))
                and np.array_equal(problem.reference,np.asarray(payload["public_reference_float32"],dtype=np.float32))
                and np.array_equal(problem.lower,np.asarray(payload["joint_lower_float64"]))
                and np.array_equal(problem.upper,np.asarray(payload["joint_upper_float64"]))
                and np.array_equal(problem.velocity,np.asarray(payload["velocity_limit_float64"]))
                and np.array_equal(problem.effort,np.asarray(payload["effort_limit_float64"])))


def joint_barrier(value, lower, upper):
    left = np.maximum(np.asarray(value)-np.asarray(lower), 1e-10)
    right = np.maximum(np.asarray(upper)-np.asarray(value), 1e-10)
    return -np.log(left)-np.log(right)


def reconstruct_objective(x, u, positions, reference, lower, upper, velocity, effort, *, include_toll=True):
    x = np.asarray(x, dtype=np.float64); u = np.asarray(u, dtype=np.float64)
    p = np.asarray(positions, dtype=np.float64); lo=np.asarray(lower,dtype=np.float64); hi=np.asarray(upper,dtype=np.float64)
    vel=np.asarray(velocity,dtype=np.float64); eff=np.asarray(effort,dtype=np.float64)
    if (x.shape!=(KNOTS,NX) or u.shape!=(INTERVALS,NU) or p.shape!=(KNOTS,3)
            or lo.shape!=(NQ,) or hi.shape!=(NQ,) or vel.shape!=(NV,) or eff.shape!=(NU,)):
        raise ValueError("objective array shapes invalid")
    ref = toll.TollReference.from_solver_bytes(np.asarray(reference, dtype=np.float32))
    total = 0.0
    for k in range(KNOTS):
        work = toll.workspace_cost_gradient_gn(p[k], ref, terminal=k == KNOTS-1)
        total += work["cost"] - (0.0 if include_toll else work["toll_cost"])
        total += 0.5*OBJECTIVE_WEIGHTS["qd"]*float(x[k,7:]@x[k,7:])
        total += OBJECTIVE_WEIGHTS["q_barrier"]*float(np.sum(joint_barrier(x[k,:7], lo, hi)))
        total += OBJECTIVE_WEIGHTS["v_barrier"]*float(np.sum(joint_barrier(x[k,7:], -vel, vel)))
        if k < INTERVALS:
            total += 0.5*OBJECTIVE_WEIGHTS["u"]*float(u[k]@u[k])
            total += OBJECTIVE_WEIGHTS["u_barrier"]*float(np.sum(joint_barrier(u[k], -eff, eff)))
    return float(total)


def barrier_gradient(value, lower, upper):
    value=np.asarray(value,dtype=np.float64); lower=np.asarray(lower,dtype=np.float64); upper=np.asarray(upper,dtype=np.float64)
    return -1.0/np.maximum(value-lower,1e-10)+1.0/np.maximum(upper-value,1e-10)


class OriginalOracleNLP:
    """Exact original-task value/Jacobian using independently supplied Pin callbacks.

    ``aba_derivatives`` returns ``(qdd,dqdd_dq,dqdd_dqd,dqdd_du)`` and
    ``tool_velocity_derivatives`` returns ``(vtool,dvtool_dq,dvtool_dqd)``.
    Those callback outputs are retained at the runner boundary in production.
    """
    def __init__(self, public_x0, reference, lower, upper, velocity, effort,
                 kinematics, aba_derivatives, tool_velocity_derivatives,
                 *, template=None, rnea=None, dense_replay=None):
        self.x0=np.asarray(public_x0,dtype=np.float64); self.reference=np.asarray(reference,dtype=np.float32)
        self.lower=np.asarray(lower,dtype=np.float64); self.upper=np.asarray(upper,dtype=np.float64)
        self.velocity=np.asarray(velocity,dtype=np.float64); self.effort=np.asarray(effort,dtype=np.float64)
        if self.x0.shape!=(NX,) or self.reference.shape!=(10,) or any(v.shape!=(7,) for v in (self.lower,self.upper,self.velocity,self.effort)):
            raise ValueError("original NLP public input shapes invalid")
        self.kinematics=kinematics; self.aba_derivatives=aba_derivatives
        self.tool_velocity_derivatives=tool_velocity_derivatives
        self.rnea=rnea; self.dense_replay=dense_replay
        self.template=None if template is None else np.asarray(template,dtype=np.float64)
        if self.template is not None and self.template.shape!=(KNOTS,3): raise ValueError("anchor template shape")

    @property
    def anchored(self): return self.template is not None

    def positions_and_jacobians(self,x):
        positions=[]; jacobians=[]
        for q in x[:,:7]:
            p,J=self.kinematics(q); positions.append(np.asarray(p,dtype=np.float64)); jacobians.append(np.asarray(J,dtype=np.float64))
        return np.stack(positions),np.stack(jacobians)

    def objective(self,z):
        x,u=unpack_z(z); positions,_=self.positions_and_jacobians(x)
        value=reconstruct_objective(x,u,positions,self.reference,self.lower,self.upper,self.velocity,self.effort,include_toll=True)
        if self.template is not None:
            error=positions-self.template; value+=.5*TEMPLATE_ANCHOR_WEIGHT*float(np.sum(error*error))
        return value

    def objective_gradient(self,z):
        x,u=unpack_z(z); positions,J=self.positions_and_jacobians(x); gradient_x=np.zeros_like(x); gradient_u=np.zeros_like(u)
        ref=toll.TollReference.from_solver_bytes(self.reference)
        for k in range(KNOTS):
            workspace=toll.workspace_cost_gradient_gn(positions[k],ref,terminal=k==KNOTS-1)
            p_gradient=np.asarray(workspace["gradient"])
            if self.template is not None: p_gradient=p_gradient+TEMPLATE_ANCHOR_WEIGHT*(positions[k]-self.template[k])
            gradient_x[k,:7]=J[k].T@p_gradient+OBJECTIVE_WEIGHTS["q_barrier"]*barrier_gradient(x[k,:7],self.lower,self.upper)
            gradient_x[k,7:]=OBJECTIVE_WEIGHTS["qd"]*x[k,7:]+OBJECTIVE_WEIGHTS["v_barrier"]*barrier_gradient(x[k,7:],-self.velocity,self.velocity)
            if k<INTERVALS:
                gradient_u[k]=OBJECTIVE_WEIGHTS["u"]*u[k]+OBJECTIVE_WEIGHTS["u_barrier"]*barrier_gradient(u[k],-self.effort,self.effort)
        return pack_z(gradient_x,gradient_u)

    def equality(self,z):
        x,u=unpack_z(z); rows=[x[0]-self.x0]
        for k in range(INTERVALS):
            qdd,*_=self.aba_derivatives(x[k,:7],x[k,7:],u[k]); qdd=np.asarray(qdd,dtype=np.float64)
            rows.append(x[k+1,:7]-x[k,:7]-DT*x[k,7:]-.5*DT**2*qdd)
            rows.append(x[k+1,7:]-x[k,7:]-DT*qdd)
        return np.concatenate(rows)

    def equality_jacobian(self,z):
        x,u=unpack_z(z); rows=NX+INTERVALS*NX; A=np.zeros((rows,Z_WIDTH)); A[:NX,:NX]=np.eye(NX)
        cursor=NX
        for k in range(INTERVALS):
            _qdd,Dq,Dv,Du=self.aba_derivatives(x[k,:7],x[k,7:],u[k])
            Dq=np.asarray(Dq); Dv=np.asarray(Dv); Du=np.asarray(Du)
            current=k*NX; following=(k+1)*NX; control=KNOTS*NX+k*NU
            A[cursor:cursor+7,current:current+7]=-np.eye(7)-.5*DT**2*Dq
            A[cursor:cursor+7,current+7:current+14]=-DT*np.eye(7)-.5*DT**2*Dv
            A[cursor:cursor+7,following:following+7]=np.eye(7)
            A[cursor:cursor+7,control:control+7]=-.5*DT**2*Du; cursor+=7
            A[cursor:cursor+7,current:current+7]=-DT*Dq
            A[cursor:cursor+7,current+7:current+14]=-np.eye(7)-DT*Dv
            A[cursor:cursor+7,following+7:following+14]=np.eye(7)
            A[cursor:cursor+7,control:control+7]=-DT*Du; cursor+=7
        return A

    def inequality(self,z):
        x,_=unpack_z(z); positions,_=self.positions_and_jacobians(x)
        ref=toll.TollReference.from_solver_bytes(self.reference); error=positions[-1]-np.asarray(ref.goal_xyz)
        vtool,*_=self.tool_velocity_derivatives(x[-1,:7],x[-1,7:]); vtool=np.asarray(vtool)
        clearance=np.sum((positions[:,:2]-np.asarray(ref.cylinder_xy))**2,axis=1)-KEEP_OUT_RADIUS_M**2
        return np.r_[TERMINAL_CUDA_M**2-error@error,FINAL_TOOL_SPEED_MPS**2-vtool@vtool,clearance]

    def inequality_jacobian(self,z):
        x,_=unpack_z(z); positions,J=self.positions_and_jacobians(x); ref=toll.TollReference.from_solver_bytes(self.reference)
        G=np.zeros((KNOTS+2,Z_WIDTH)); terminal=(KNOTS-1)*NX
        error=positions[-1]-np.asarray(ref.goal_xyz); G[0,terminal:terminal+7]=-2*error@J[-1]
        vtool,Dq,Dv=self.tool_velocity_derivatives(x[-1,:7],x[-1,7:]); vtool=np.asarray(vtool)
        G[1,terminal:terminal+7]=-2*vtool@np.asarray(Dq); G[1,terminal+7:terminal+14]=-2*vtool@np.asarray(Dv)
        delta=positions[:,:2]-np.asarray(ref.cylinder_xy)
        for k in range(KNOTS): G[k+2,k*NX:k*NX+7]=2*delta[k]@J[k,:2,:]
        return G

    def bounds_arrays(self):
        lower_x=np.tile(np.r_[self.lower,-self.velocity],(KNOTS,1)); upper_x=np.tile(np.r_[self.upper,self.velocity],(KNOTS,1))
        return pack_z(lower_x,np.tile(-self.effort,(INTERVALS,1))),pack_z(upper_x,np.tile(self.effort,(INTERVALS,1)))


def original_kkt(gradient, equality_jacobian, equality, inequality_jacobian,
                 inequality, equality_multipliers, inequality_multipliers,
                 lower_slack, upper_slack, lower_multipliers, upper_multipliers):
    """Independent h>=0, L=f+lambda*c-nu*h KKT reconstruction."""
    g = np.asarray(gradient); A = np.asarray(equality_jacobian); G = np.asarray(inequality_jacobian)
    c = np.asarray(equality); h = np.asarray(inequality); lam = np.asarray(equality_multipliers); nu = np.asarray(inequality_multipliers)
    lo_s = np.asarray(lower_slack); hi_s = np.asarray(upper_slack); lo_nu = np.asarray(lower_multipliers); hi_nu = np.asarray(upper_multipliers)
    stationarity = g + A.T@lam - G.T@nu - lo_nu + hi_nu
    dual = np.r_[np.maximum(-nu, 0), np.maximum(-lo_nu, 0), np.maximum(-hi_nu, 0)]
    complementarity = np.r_[nu*h, lo_nu*lo_s, hi_nu*hi_s]
    result = {"equality_inf": float(np.max(np.abs(c), initial=0)), "inequality_violation_inf": float(np.max(np.maximum(-h,0), initial=0)),
              "stationarity_inf": float(np.max(np.abs(stationarity), initial=0)), "dual_sign_inf": float(np.max(dual, initial=0)),
              "complementarity_inf": float(np.max(np.abs(complementarity), initial=0)),
              "stationarity": stationarity, "dual_sign": dual, "complementarity": complementarity}
    result["passes"] = result["equality_inf"] <= PRIMAL_TOLERANCE and result["inequality_violation_inf"] <= PRIMAL_TOLERANCE and max(result["stationarity_inf"], result["dual_sign_inf"], result["complementarity_inf"]) <= KKT_TOLERANCE
    return result


KKT_ARRAY_FIELDS=frozenset({
    "objective_gradient","equality_jacobian","equality_residual","inequality_jacobian",
    "inequality_slack","equality_multipliers","inequality_multipliers",
    "lower_slack","upper_slack","lower_multipliers","upper_multipliers",
    "scipy_raw_constraint_multipliers","scipy_raw_bound_multipliers",
})


def certify_retained_kkt(arrays:Mapping):
    if set(arrays)!=KKT_ARRAY_FIELDS: return {"passes":False,"exact_schema":False}
    a={name:np.asarray(value,dtype=np.float64) for name,value in arrays.items()}
    m=a["equality_residual"].size; n=a["inequality_slack"].size
    shapes=bool(a["objective_gradient"].shape==(Z_WIDTH,)
                and a["equality_jacobian"].shape==(m,Z_WIDTH)
                and a["inequality_jacobian"].shape==(n,Z_WIDTH)
                and a["equality_multipliers"].shape==(m,)
                and a["inequality_multipliers"].shape==(n,)
                and all(a[name].shape==(Z_WIDTH,) for name in
                        ("lower_slack","upper_slack","lower_multipliers","upper_multipliers")))
    finite=all(np.isfinite(value).all() for value in a.values())
    if not shapes or not finite: return {"passes":False,"exact_schema":True,"shapes":shapes,"finite":finite}
    result=original_kkt(a["objective_gradient"],a["equality_jacobian"],a["equality_residual"],
                        a["inequality_jacobian"],a["inequality_slack"],a["equality_multipliers"],
                        a["inequality_multipliers"],a["lower_slack"],a["upper_slack"],
                        a["lower_multipliers"],a["upper_multipliers"])
    return {"exact_schema":True,"shapes":True,"finite":True,**result}


def map_scipy_multipliers(raw_equality,raw_inequality,raw_bounds):
    raw_eq=np.asarray(raw_equality,dtype=np.float64); raw_ineq=np.asarray(raw_inequality,dtype=np.float64)
    raw_bound=np.asarray(raw_bounds,dtype=np.float64)
    if raw_eq.shape!=(EQUALITY_COUNT,) or raw_ineq.shape!=(INEQUALITY_COUNT,) or raw_bound.shape!=(Z_WIDTH,):
        raise ValueError("raw SciPy multiplier shapes invalid")
    return {"equality":raw_eq.copy(),"inequality":-raw_ineq,
            "lower":np.maximum(-raw_bound,0.),"upper":np.maximum(raw_bound,0.)}


def certify_primal_kkt(problem,primal,retained):
    if type(problem) is not OriginalOracleNLP or problem.anchored:
        return {"exact_unanchored_problem":False,"passes":False}
    required={"raw_equality_multipliers_float64","raw_inequality_multipliers_float64",
              "raw_bound_multipliers_float64","canonical_equality_multipliers_float64",
              "canonical_inequality_multipliers_float64","canonical_lower_multipliers_float64",
              "canonical_upper_multipliers_float64"}
    if set(retained)!=required: return {"exact_unanchored_problem":True,"exact_schema":False,"passes":False}
    try:
        mapped=map_scipy_multipliers(retained["raw_equality_multipliers_float64"],
                                     retained["raw_inequality_multipliers_float64"],
                                     retained["raw_bound_multipliers_float64"])
        canonical_exact=all(np.array_equal(np.asarray(retained[f"canonical_{name}_multipliers_float64"]),mapped[name])
                            for name in ("equality","inequality","lower","upper"))
        z=np.asarray(primal,dtype=np.float64); lower,upper=problem.bounds_arrays()
        result=original_kkt(problem.objective_gradient(z),problem.equality_jacobian(z),problem.equality(z),
                            problem.inequality_jacobian(z),problem.inequality(z),mapped["equality"],mapped["inequality"],
                            z-lower,upper-z,mapped["lower"],mapped["upper"])
    except (ValueError,KeyError,IndexError):
        return {"exact_unanchored_problem":True,"exact_schema":True,"passes":False}
    return {"exact_unanchored_problem":True,"exact_schema":True,"raw_to_canonical_exact":canonical_exact,
            **result,"passes":bool(canonical_exact and result["passes"])}


def path_length(path):
    value = np.asarray(path, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(value, axis=0), axis=1)))


def mode_label(path, center):
    label, turn = toll.signed_open_turn(np.asarray(path)[:,:2], center)
    return label, turn


def canonical_mode_sign(public_default_side, requested_side):
    sign=int(public_default_side)
    if sign not in (-1,1) or requested_side not in SIDES:
        raise ValueError("canonical mode identity invalid")
    return sign if requested_side=="default" else -sign


def aggregate_task(rows: Sequence[Mapping], default_label: str):
    """Fail-closed pure task aggregation; topology rows are never deduplicated."""
    expected = {(side, i) for side in SIDES for i in range(5)}
    identities = {(row.get("requested_side"), row.get("margin_index")) for row in rows}
    certified = [row for row in rows if row.get("certified") is True]
    intended = {side: [row for row in certified if row.get("requested_side") == side and row.get("actual_side") == side] for side in SIDES}
    actual = {side: [row for row in certified if row.get("actual_side") == side] for side in SIDES}
    gates = {
        "exact_ten_rows": len(rows)==10 and identities==expected,
        "all_calls_attempted": all(row.get("acquisition_attempted") is True and row.get("polish_attempted") is True for row in rows),
        "intended_four_each": all(len(intended[side]) >= MIN_INTENDED_MATCHES for side in SIDES),
        "actual_four_each": all(len(actual[side]) >= MIN_ACTUAL_MODE_ROWS for side in SIDES),
        "default_label_not_reversed": default_label == "default",
        "all_rows_have_canonical_certificate": all(type(row.get("certificate_pass")) is bool for row in rows),
        "at_least_eight_certified": len(certified)>=8,
    }
    return {**gates, "counts": {side: len(actual[side]) for side in SIDES}, "passes": bool(all(gates.values()))}


def frozen_metadata():
    return {
        "protocol": ORACLE_PROTOCOL_VERSION, "execution_authorized": False,
        "output": str(ORACLE_OUTPUT_PATH), "ledger_count": len(EXPECTED_LEDGER),
        "task_pins": {k:(str(p),h) for k,(p,h) in TASK_ARTIFACT_PINS.items()},
        "model_pins": {k:(str(p),h) for k,(p,h) in MODEL_ARTIFACT_PINS.items()},
        "oracle_only": True, "benchmark_seed_eligible": False,
        "collision_scope": COLLISION_SCOPE, "optimizer_calls": {"acquisition":120,"polish":120},
        "bootstrap": {"seed":BOOTSTRAP_SEED,"resamples":BOOTSTRAP_RESAMPLES,"only_random_call":True},
    }
