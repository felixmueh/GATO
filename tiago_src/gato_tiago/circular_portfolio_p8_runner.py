"""Disabled compact P8 pilot orchestration and independent disk recertification."""

from __future__ import annotations

import argparse,hashlib,importlib.metadata,json,os,shlex,subprocess,sys,time
from pathlib import Path

import numpy as np

from gato_tiago.circular_portfolio import (LONG_ANGLE_RAD,SHORT_ANGLE_RAD,
    construct_geometry,certify_geometry,quintic_progress,reference_from_geometry)
from gato_tiago.circular_portfolio_p8 import (AFFINE_SAMPLES,AFFINE_SUBSTEPS,
    CONSTRUCTION_SPECS,DLS_DAMPING,DT,EXTENSION,IK_ITERATIONS,INTERVALS,KNOTS,KD,KP,
    LANES,OUTPUT,P6_REJECTED_REPORT,P7_EXPLORATORY_GRID,PROTOCOL,WORKER_OUTPUT_SPECS,
    RUNNER_WALL_LIMIT_S,SEED,WORKER_PROTOCOL,array_hash,checkpoint_path,exact_arrays,
    latest_path,manifest_path,open_turn,reconstruct_cost,rejection_path,worker_paths)
from gato_tiago.circular_portfolio_p8_worker import (SUCCESS_COUNTS as WORKER_COUNTS,
    certify_counts as worker_certify_counts,certify_output,
    certify_summary as certify_worker_summary,expected_argv as worker_argv,
    certify_rejection as certify_worker_rejection,snapshot as worker_snapshot)
from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite,_production_pin_context
from gato_tiago.multimodal_toll_oracle_v1 import enumerate_box_dls
from gato_tiago.circular_portfolio_p5_v2_runner import canonical_authentication

RUNNER_EXECUTION_AUTHORIZATION=None

THREAD_ENV={"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
    "NUMEXPR_NUM_THREADS":"1"}
AUTHORIZED_CWD="/workspace/GATO"
AUTHORIZED_ORIG_ARGV=("python","-B","-m","gato_tiago.circular_portfolio_p8_runner",
    "--execute","--output",str(OUTPUT))
SOURCE_PATHS=("tiago_src/gato_tiago/circular_portfolio.py",
    "tiago_src/gato_tiago/circular_portfolio_p8.py",
    "tiago_src/gato_tiago/circular_portfolio_p8_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p8_worker.py",
    "tiago_src/gato_tiago/circular_portfolio_p5.py",
    "tiago_src/gato_tiago/circular_portfolio_p5_v2_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_prerequisite_runner.py",
    "tiago_src/gato_tiago/multimodal_toll_oracle_v1.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_oracle_schema.py",
    "tiago_src/gato_tiago/multimodal_pillar.py",
    "gato/dynamics/tiago_right/tiago_right_arm.urdf")
COUNT_KEYS=("prerequisite_loads","parent_pin_contexts","primary_cpu_endpoint_fk_calls",
    "primary_dls_kinematics_calls","primary_cpu_proxy_fk_calls",
    "independent_cpu_endpoint_fk_calls","independent_dls_kinematics_calls",
    "independent_cpu_proxy_fk_calls","parent_rnea_recert_calls",
    "parent_identical_state_fk_calls","parent_affine_fk_calls","worker_attempts",
    "worker_successes","worker_pin_contexts","b1_constructors","b16_constructors",
    "worker_rnea_calls","b1_sim_forward_calls","b16_sim_forward_calls",
    "b1_tool_position_calls","b16_tool_position_calls","optimizer_calls","solve_calls",
    "sqp_calls","rng_calls","rejected_p6_artifact_loads","science_routes_attempted",
    "science_routes_completed","completed")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def zero_counts():return {key:0 for key in COUNT_KEYS}


def snapshot():
    root=Path(__file__).resolve().parents[2]
    run=lambda *args:subprocess.run(["git",*args],cwd=root,check=True,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
    extension=(root/EXTENSION["relative_path"]).resolve()
    return {"head":run("rev-parse","HEAD"),
        "clean":run("status","--porcelain","--untracked-files=no")=="",
        "sources":{name:sha(root/name) for name in SOURCE_PATHS},
        "extension":{"path":str(extension),"sha256":sha(extension),
            "size":extension.stat().st_size,"build_head":EXTENSION["build_head"],
            "arch":EXTENSION["arch"],"attributes":{key:EXTENSION[key] for key in
                ("KNOT_POINTS","REFERENCE_SIZE","TOOL_POSITION_FRAME","TOOL_POSITION_SIZE")}}}


def frozen_extension():
    root=Path(__file__).resolve().parents[2]
    return {"path":str((root/EXTENSION["relative_path"]).resolve()),
        "sha256":EXTENSION["sha256"],"size":EXTENSION["size"],
        "build_head":EXTENSION["build_head"],"arch":EXTENSION["arch"],
        "attributes":{key:EXTENSION[key] for key in
            ("KNOT_POINTS","REFERENCE_SIZE","TOOL_POSITION_FRAME","TOOL_POSITION_SIZE")}}


def provenance(start,end=None):
    return {"cwd":str(Path.cwd().resolve()),"orig_argv":list(sys.orig_argv),
        "command":shlex.join(sys.orig_argv),"thread_environment":{k:os.environ.get(k) for k in THREAD_ENV},
        "runtime_versions":{"python":sys.version,"numpy":np.__version__,
            "pinocchio":importlib.metadata.version("pin")},"start":start,"end":end,
        "p6_report":P6_REJECTED_REPORT,"p7_grid":P7_EXPLORATORY_GRID}


def certify_provenance(value,final):
    try:
        current=snapshot();argv=tuple(value["orig_argv"])
        argv_ok=argv==AUTHORIZED_ORIG_ARGV or (OUTPUT!=Path(AUTHORIZED_ORIG_ARGV[-1])
            and argv[:-1]==AUTHORIZED_ORIG_ARGV[:-1] and argv[-1]==str(OUTPUT))
        return bool(set(value)=={"cwd","orig_argv","command","thread_environment",
                "runtime_versions","start","end","p6_report","p7_grid"}
            and value["cwd"]==AUTHORIZED_CWD and argv_ok and value["command"]==shlex.join(argv)
            and value["thread_environment"]==THREAD_ENV
            and value["runtime_versions"]=={"python":sys.version,"numpy":np.__version__,
                "pinocchio":importlib.metadata.version("pin")}
            and value["start"]==current and value["start"]["clean"] is True
            and value["start"]["extension"]==frozen_extension()
            and value["p6_report"]==P6_REJECTED_REPORT and tuple(value["p7_grid"])==P7_EXPLORATORY_GRID
            and ((not final and value["end"] is None) or final and value["end"]==value["start"]))
    except Exception:return False


def atomic_json(path,value,pointer=False):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if candidate.exists() or path.exists() and not pointer:raise FileExistsError("P8 artifact exists")
    try:candidate.write_text(json.dumps(value,sort_keys=True,separators=(",",":")));candidate.replace(path)
    finally:
        if candidate.exists():candidate.unlink()


def atomic_npz(path,arrays):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError("P8 artifact exists")
    try:
        with candidate.open("wb") as stream:np.savez(stream,**arrays)
        candidate.replace(path)
    finally:
        if candidate.exists():candidate.unlink()


def publish_checkpoint(generation,stage,counts,prov,detail):
    doc={"protocol":PROTOCOL,"generation":generation,"stage":stage,"incomplete":True,
        "counts":counts,"detail":detail,"provenance":prov,"evidence":False}
    path=checkpoint_path(generation);atomic_json(path,doc)
    atomic_json(latest_path(),{"protocol":PROTOCOL,"generation":generation,"path":str(path),
        "sha256":sha(path),"incomplete":True},True)


def exact_box_dls(jacobian,residual,q,lower,upper):
    retained=enumerate_box_dls(jacobian,residual,lower+.01-q,upper-.01-q)
    index=int(retained["selected"])
    return retained["dq"][index],retained["status"][index]


def analytic_target(geometry,route):
    progress=quintic_progress(np.arange(KNOTS,dtype=np.float64)/INTERVALS)
    delta=(-geometry.orientation_sign*SHORT_ANGLE_RAD if route=="short"
           else geometry.orientation_sign*LONG_ANGLE_RAD)
    radial=geometry.start_xyz[:2]-geometry.pillar_xy
    start_angle=np.arctan2(radial[1],radial[0]);angles=start_angle+delta*progress
    xy=geometry.pillar_xy+geometry.circle_radius_m*np.c_[np.cos(angles),np.sin(angles)]
    z=geometry.start_xyz[2]+(geometry.goal_xyz[2]-geometry.start_xyz[2])*progress
    value=np.c_[xy,z];value[0]=geometry.start_xyz;value[-1]=geometry.goal_xyz
    return value


def time_scale_proxy(q,dt=DT):
    q=np.asarray(q,np.float64)
    if q.shape!=(2,KNOTS,7) or dt<=0:raise ValueError("P8 proxy path invalid")
    qd=np.empty_like(q);qd[:,0]=0.;qd[:,-1]=0.;qd[:,1:-1]=(q[:,2:]-q[:,:-2])/(2*dt)
    return qd,np.diff(qd,axis=1)/dt


def construct_cpu(prerequisite,kinematics,deadline,monotonic=time.monotonic,
                  counts=None,phase="primary"):
    q0=prerequisite["public_x0_float32"][0,:7].astype(np.float64)
    qgoal=prerequisite["quarantined_q8_float64"][0];default=int(prerequisite[
        "public_default_side_int8"][0]);start=kinematics(q0)[0];goal=kinematics(qgoal)[0]
    if counts is not None:counts[f"{phase}_cpu_endpoint_fk_calls"]+=2
    geometry=construct_geometry(start,goal,default)
    if not certify_geometry(geometry)["passes"]:raise RuntimeError("P8 geometry invalid")
    reference=np.tile(reference_from_geometry(geometry).as_float32(),KNOTS)
    targets=np.stack([analytic_target(geometry,route) for route in ("short","long")])
    q=np.empty((2,KNOTS,7));before=np.empty((2,KNOTS-2,IK_ITERATIONS,7))
    position=np.empty((2,KNOTS-2,IK_ITERATIONS,3));residual=np.empty_like(position)
    jacobian=np.empty((2,KNOTS-2,IK_ITERATIONS,3,7));dq=np.empty_like(before)
    face=np.empty((2,KNOTS-2,IK_ITERATIONS,7),np.int8)
    lower=prerequisite["joint_lower_float64"];upper=prerequisite["joint_upper_float64"]
    for route in range(2):
        q[route,0]=q0
        for knot in range(1,KNOTS-1):
            current=q[route,knot-1].copy()
            for step in range(IK_ITERATIONS):
                if monotonic()>deadline:raise TimeoutError("P8 CPU preflight wall limit")
                p,J=kinematics(current);r=targets[route,knot]-p
                if counts is not None:counts[f"{phase}_dls_kinematics_calls"]+=1
                delta,status=exact_box_dls(J,r,current,lower,upper)
                before[route,knot-1,step]=current;position[route,knot-1,step]=p
                residual[route,knot-1,step]=r;jacobian[route,knot-1,step]=J
                dq[route,knot-1,step]=delta;face[route,knot-1,step]=status;current+=delta
            q[route,knot]=current
        q[route,-1]=qgoal
    qd,qdd=time_scale_proxy(q)
    arrays={"x0_float32":prerequisite["public_x0_float32"][0],"q_goal_float64":qgoal,
        "goal_tool_float64":goal,"pillar_float64":geometry.pillar_xy,"reference_float32":reference,
        "target_tool_float64":targets,"proxy_q_float64":q,"proxy_qd_float64":qd,
        "proxy_qdd_float64":qdd,"dls_q_before_float64":before,"dls_position_float64":position,
        "dls_residual_float64":residual,"dls_jacobian_float64":jacobian,
        "dls_dq_float64":dq,"dls_selected_face_int8":face}
    if not exact_arrays(arrays,CONSTRUCTION_SPECS):raise RuntimeError("P8 construction schema invalid")
    actual=np.asarray([[kinematics(row)[0] for row in route] for route in q])
    if counts is not None:counts[f"{phase}_cpu_proxy_fk_calls"]+=2*KNOTS
    error=np.linalg.norm(actual-targets,axis=2)
    endpoint=np.r_[q[:,-1]-qgoal,qd[:,-1]]
    detail={"geometry":certify_geometry(geometry),"dls_iterations":IK_ITERATIONS,
        "dls_damping":DLS_DAMPING,"target_hash":array_hash(targets),
        "history_hashes":{k:array_hash(v) for k,v in arrays.items() if k.startswith("dls_")},
        "ik_rms_m":[float(x) for x in np.sqrt(np.mean(error**2,axis=1))],
        "ik_max_m":[float(x) for x in np.max(error,axis=1)],
        "endpoint_residual_max":float(np.max(np.abs(endpoint))),
        "passes":bool(np.max(error)<=.004 and np.max(np.sqrt(np.mean(error**2,axis=1)))<=.002
            and np.max(np.abs(endpoint))<=1e-12)}
    return arrays,detail


def certify_construction(arrays,prerequisite,kinematics,deadline=float("inf"),
                         monotonic=time.monotonic,counts=None,phase="independent"):
    try:
        fresh,detail=construct_cpu(prerequisite,kinematics,deadline,monotonic,counts,phase)
        return canonical_authentication({"detail":detail,"passes":bool(detail["passes"] and all(
            np.array_equal(arrays[k],fresh[k]) for k in fresh))})
    except Exception:return {"passes":False}


def certify_science(worker_arrays,construction,prerequisite,kinematics,rnea,
                    deadline=float("inf"),monotonic=time.monotonic,counts=None):
    try:
        if not certify_output(worker_arrays,construction):return {"passes":False}
        state=worker_arrays["generated_state_float32"];tool=worker_arrays["generated_tool_float32"]
        controls=worker_arrays["controls_float32"];pin_tool=[];final_j=[]
        for route in range(2):
            if counts is not None:counts["science_routes_attempted"]+=1
            if monotonic()>deadline:raise TimeoutError("P8 campaign wall limit")
            values=[]
            for q in state[route,:,:7]:
                if monotonic()>deadline:raise TimeoutError("P8 campaign wall limit")
                values.append(kinematics(q.astype(np.float64)))
                if counts is not None:counts["parent_identical_state_fk_calls"]+=1
            pin_tool.append([x[0] for x in values]);final_j.append(values[-1][1])
            for k in range(INTERVALS):
                if monotonic()>deadline:raise TimeoutError("P8 campaign wall limit")
                retained=worker_arrays["rnea_u_float64"][route,k];fresh=rnea(
                    worker_arrays["rnea_q_float64"][route,k],
                    worker_arrays["rnea_qd_float64"][route,k],
                    worker_arrays["rnea_qdd_float64"][route,k])
                if counts is not None:counts["parent_rnea_recert_calls"]+=1
                if not np.array_equal(retained,fresh):return {"passes":False}
        pin_tool=np.asarray(pin_tool);affine_pin=np.empty_like(worker_arrays["affine_cuda_tool_float32"],
            dtype=np.float64)
        for route in range(2):
            index=0
            for k in range(INTERVALS):
                for substep in range(AFFINE_SUBSTEPS):
                    if monotonic()>deadline:raise TimeoutError("P8 campaign wall limit")
                    alpha=substep/AFFINE_SUBSTEPS
                    q=(1-alpha)*state[route,k,:7]+alpha*state[route,k+1,:7]
                    affine_pin[route,index]=kinematics(q.astype(np.float64))[0];index+=1
                    if counts is not None:counts["parent_affine_fk_calls"]+=1
            if monotonic()>deadline:raise TimeoutError("P8 campaign wall limit")
            affine_pin[route,-1]=kinematics(state[route,-1,:7].astype(np.float64))[0]
            if counts is not None:counts["parent_affine_fk_calls"]+=1
        lower=prerequisite["joint_lower_float64"];upper=prerequisite["joint_upper_float64"]
        velocity=prerequisite["velocity_limit_float64"];effort=prerequisite["effort_limit_float64"]
        midpoint=.5*(lower+upper);half=.5*(upper-lower);goal=construction["goal_tool_float64"]
        pillar=construction["pillar_float64"];rows=[]
        for index,route in enumerate(("short","long")):
            target=construction["target_tool_float64"][index]
            difference=np.linalg.norm(tool[index].astype(np.float64)-target,axis=1)
            same_q=np.linalg.norm(tool[index].astype(np.float64)-pin_tool[index],axis=1)
            affine_difference=np.linalg.norm(worker_arrays["affine_cuda_tool_float32"][index]
                .astype(np.float64)-affine_pin[index],axis=1)
            cost=reconstruct_cost(state[index],controls[index],tool[index],
                construction["reference_float32"],lower,upper,velocity,effort)
            turn=open_turn(tool[index],pillar);expected=-int(prerequisite[
                "public_default_side_int8"][0]) if route=="short" else int(prerequisite[
                "public_default_side_int8"][0])
            metrics={"q_ratio":float(np.max(np.abs((state[index,:,:7]-midpoint)/half))),
                "v_ratio":float(np.max(np.abs(state[index,:,7:])/velocity)),
                "u_ratio":float(np.max(np.abs(controls[index])/effort)),
                "terminal_m":float(np.linalg.norm(tool[index,-1]-goal)),
                "speed_mps":float(np.linalg.norm(final_j[index]@state[index,-1,7:])),
                "clearance_m":float(np.min(np.linalg.norm(
                    worker_arrays["affine_cuda_tool_float32"][index,:,:2]-pillar,axis=1)-.03)),
                "pin_sampled_clearance_m":float(np.min(np.linalg.norm(
                    affine_pin[index,:,:2]-pillar,axis=1)-.03)),
                "turn_rad":turn,"arc_rms_m":float(np.sqrt(np.mean(difference**2))),
                "arc_max_m":float(np.max(difference)),"same_q_fk_max_m":float(np.max(same_q)),
                "sampled_affine_fk_max_m":float(np.max(affine_difference)),
                "path_length_m":float(np.sum(np.linalg.norm(np.diff(tool[index],axis=0),axis=1))),
                "base_cost":cost["base"],"full_cost":cost["full"],"toll":cost["toll"],
                "toll_exposure":int(np.count_nonzero(cost["residual_float64"][1:-1]>0)),
                "toll_saturation":int(np.count_nonzero(cost["residual_float64"][1:-1]>=.95))}
            gates={"finite":all(np.isfinite(v) for v in metrics.values()),
                "limits":max(metrics[k] for k in ("q_ratio","v_ratio","u_ratio"))<=1.,
                "terminal":metrics["terminal_m"]<=.015,"speed":metrics["speed_mps"]<=.05,
                "clearance":min(metrics["clearance_m"],metrics["pin_sampled_clearance_m"])>=.005,
                "same_q_fk":metrics["same_q_fk_max_m"]<=.001
                    and metrics["sampled_affine_fk_max_m"]<=.001,
                "arc":metrics["arc_rms_m"]<=.002 and metrics["arc_max_m"]<=.004,
                "topology":abs(turn)>=(2.4 if route=="short" else 3.) and int(np.sign(turn))==expected,
                "cost_decomposition":abs(cost["full"]-cost["base"]-cost["toll"])<=1e-10}
            rows.append({"route":route,"metrics":metrics,"cost":cost,"gates":gates,
                "passes":bool(all(gates.values()))})
        if counts is not None:
            counts["science_routes_completed"]=2;counts["completed"]=2
        short,long=rows;path=long["metrics"]["path_length_m"]-short["metrics"]["path_length_m"]
        base=long["metrics"]["base_cost"]-short["metrics"]["base_cost"]
        full=short["metrics"]["full_cost"]-long["metrics"]["full_cost"]
        threshold=max(1.,.05*max(1.,abs(long["metrics"]["full_cost"])))
        pair={"distinct_controls":not np.array_equal(controls[0],controls[1]),
            "distinct_tools":not np.array_equal(tool[0],tool[1]),"opposite":np.sign(
                short["metrics"]["turn_rad"])==-np.sign(long["metrics"]["turn_rad"]),
            "short_path":path>=.005,"base_advantage":base>=.10,"full_reversal":full>=threshold}
        return canonical_authentication({"routes":rows,"pair":pair,"differences":{"path_m":path,"base":base,
            "full":full,"threshold":threshold},"affine_sampling":{"subintervals":16,
            "continuous_dynamics_evidence":False},"passes":bool(all(r["passes"] for r in rows)
                and all(pair.values()))})
    except TimeoutError:raise
    except Exception:return {"passes":False,"operational_incomplete":True}


def certify_science_prefix(worker_arrays,construction,kinematics,rnea,counts):
    """Recompute only the parent semantic prefix completed before interruption."""
    try:
        if not certify_counts(counts) or not certify_output(worker_arrays,construction):return False
        state=worker_arrays["generated_state_float32"]
        remaining_fk=counts["parent_identical_state_fk_calls"]
        remaining_rnea=counts["parent_rnea_recert_calls"]
        for route in range(2):
            take=min(KNOTS,remaining_fk)
            for knot in range(take):kinematics(state[route,knot,:7].astype(np.float64))
            remaining_fk-=take
            take=min(INTERVALS,remaining_rnea)
            for knot in range(take):
                fresh=rnea(worker_arrays["rnea_q_float64"][route,knot],
                    worker_arrays["rnea_qd_float64"][route,knot],
                    worker_arrays["rnea_qdd_float64"][route,knot])
                if not np.array_equal(fresh,worker_arrays["rnea_u_float64"][route,knot]):return False
            remaining_rnea-=take
        remaining_affine=counts["parent_affine_fk_calls"]
        for route in range(2):
            for knot in range(INTERVALS):
                for substep in range(AFFINE_SUBSTEPS):
                    if remaining_affine==0:return remaining_fk==remaining_rnea==0
                    alpha=substep/AFFINE_SUBSTEPS
                    q=(1-alpha)*state[route,knot,:7]+alpha*state[route,knot+1,:7]
                    kinematics(q.astype(np.float64));remaining_affine-=1
            if remaining_affine:
                kinematics(state[route,-1,:7].astype(np.float64));remaining_affine-=1
        return remaining_fk==remaining_rnea==remaining_affine==0
    except Exception:return False


def success_counts():
    value=zero_counts();value.update({"prerequisite_loads":1,"parent_pin_contexts":1,
        "primary_cpu_endpoint_fk_calls":2,
        "primary_dls_kinematics_calls":2*(KNOTS-2)*IK_ITERATIONS,
        "primary_cpu_proxy_fk_calls":2*KNOTS,
        "independent_cpu_endpoint_fk_calls":2,
        "independent_dls_kinematics_calls":2*(KNOTS-2)*IK_ITERATIONS,
        "independent_cpu_proxy_fk_calls":2*KNOTS,"parent_rnea_recert_calls":518,
        "parent_identical_state_fk_calls":2*KNOTS,"parent_affine_fk_calls":2*AFFINE_SAMPLES,
        "worker_attempts":1,"worker_successes":1,"worker_pin_contexts":1,
        "b1_constructors":1,"b16_constructors":1,"worker_rnea_calls":518,
        "b1_sim_forward_calls":1036,"b16_sim_forward_calls":259,
        "b1_tool_position_calls":9329,"b16_tool_position_calls":260,
        "science_routes_attempted":2,"science_routes_completed":2,"completed":2})
    return value


def cpu_counts():
    value=zero_counts();value.update({"prerequisite_loads":1,"parent_pin_contexts":1,
        "primary_cpu_endpoint_fk_calls":2,
        "primary_dls_kinematics_calls":2*(KNOTS-2)*IK_ITERATIONS,
        "primary_cpu_proxy_fk_calls":2*KNOTS,
        "independent_cpu_endpoint_fk_calls":2,
        "independent_dls_kinematics_calls":2*(KNOTS-2)*IK_ITERATIONS,
        "independent_cpu_proxy_fk_calls":2*KNOTS})
    return value


def certify_counts(value,final=False):
    if not isinstance(value,dict) or set(value)!=set(COUNT_KEYS) \
            or any(type(v) is not int or v<0 for v in value.values()):return False
    expected=success_counts()
    if final:return value==expected
    if not all(value[k]<=expected[k] for k in expected):return False
    primary=("primary_cpu_endpoint_fk_calls","primary_dls_kinematics_calls",
        "primary_cpu_proxy_fk_calls")
    independent=("independent_cpu_endpoint_fk_calls","independent_dls_kinematics_calls",
        "independent_cpu_proxy_fk_calls")
    if any(value[k] for k in primary) and value["parent_pin_contexts"]!=1:return False
    if any(value[k] for k in independent) and any(value[k]!=cpu_counts()[k] for k in primary):
        return False
    if value["worker_attempts"] and any(value[k]!=cpu_counts()[k] for k in (*primary,*independent)):
        return False
    worker_keys=("worker_pin_contexts","b1_constructors","b16_constructors",
        "worker_rnea_calls","b1_sim_forward_calls","b16_sim_forward_calls",
        "b1_tool_position_calls","b16_tool_position_calls")
    if any(value[k] for k in worker_keys) and value["worker_attempts"]!=1:return False
    if value["worker_successes"]:
        if value["worker_attempts"]!=1 or any(value[k]!=expected[k] for k in worker_keys):return False
    parent_recert=("parent_rnea_recert_calls","parent_identical_state_fk_calls",
        "parent_affine_fk_calls")
    if any(value[k] for k in parent_recert) and value["worker_successes"]!=1:return False
    attempted=value["science_routes_attempted"];completed=value["science_routes_completed"]
    fk=value["parent_identical_state_fk_calls"];rnea=value["parent_rnea_recert_calls"]
    affine=value["parent_affine_fk_calls"]
    if attempted not in (0,1,2) or completed not in (0,2) or completed>attempted:return False
    if attempted and value["worker_successes"]!=1:return False
    if attempted==0 and (fk or rnea or affine):return False
    if attempted==1 and (fk>260 or rnea>259 or affine):return False
    if fk<260 and rnea:return False
    if rnea<259 and fk>260:return False
    if attempted==2 and fk<260:return False
    if 260<fk<520 and rnea!=259:return False
    if rnea>259 and fk!=520:return False
    if affine and (fk!=520 or rnea!=518 or attempted!=2):return False
    if completed==2 and (fk,rnea,affine)!=(520,518,2*AFFINE_SAMPLES):return False
    if value["completed"]!=completed:return False
    if value["completed"]==2 and value!=expected:return False
    return True


def certify_worker_execution(value,counts):
    keys={"status","returncode","internal_counts_known","counts"}
    if not isinstance(value,dict) or set(value)!=keys:return False
    status=value["status"]
    if status=="not_started":return value=={"status":"not_started","returncode":None,
        "internal_counts_known":False,"counts":None} and counts["worker_attempts"]==0
    if status=="internal_unknown":return value["counts"] is None \
        and value["internal_counts_known"] is False and counts["worker_attempts"]==1 \
        and counts["worker_successes"]==0
    if status in ("known_partial","known_complete"):
        if value["internal_counts_known"] is not True or not worker_certify_counts(
                value["counts"],status=="known_complete"):return False
        mapping={"pin_model_contexts":"worker_pin_contexts","rnea_calls":"worker_rnea_calls"}
        if any(counts[mapping.get(k,k)]!=v for k,v in value["counts"].items()):return False
        return counts["worker_attempts"]==1 and counts["worker_successes"]==(status=="known_complete")
    return False


def refuse_existing():
    paths=[OUTPUT,manifest_path(),latest_path(),rejection_path(),
        *(checkpoint_path(i) for i in range(4)),*worker_paths().values()]
    return not any(path.exists() or Path(str(path)+".candidate").exists() for path in paths)


def namespace_paths():return set(OUTPUT.parent.glob("p8*")) if OUTPUT.parent.is_dir() else set()


def active_artifacts():
    paths=[OUTPUT,manifest_path(),latest_path(),
        *(checkpoint_path(i) for i in range(4)),*worker_paths().values()]
    result={}
    for path in paths:
        for candidate in (path,Path(str(path)+".candidate")):
            result[str(candidate)]={"present":candidate.is_file(),
                "sha256":sha(candidate) if candidate.is_file() else None,
                "size":candidate.stat().st_size if candidate.is_file() else None}
    return result


def classify_artifacts(construction=None):
    """Classify partial/candidate files as non-evidence while binding exact bytes."""
    result={};paths=worker_paths()
    for path in (*paths.values(),OUTPUT,manifest_path(),latest_path(),
                 *(Path(str(path)+".candidate") for path in
                   (*paths.values(),OUTPUT,manifest_path(),latest_path()))):
        if not path.is_file():continue
        row={"sha256":sha(path),"size":path.stat().st_size,"classification":None}
        try:
            if path.suffix==".npz" or ".npz." in path.name:
                with np.load(path,allow_pickle=False) as archive:
                    arrays={key:archive[key] for key in archive.files}
                if path==paths["input"] or "worker.input.npz" in path.name:
                    valid=construction is not None and exact_arrays(arrays,CONSTRUCTION_SPECS) \
                        and all(np.array_equal(arrays[key],construction[key]) for key in construction)
                    row["classification"]="exact_worker_input" if valid else "invalid_npz_non_evidence"
                else:
                    valid=construction is not None and certify_output(arrays,construction)
                    row["classification"]="exact_worker_output" if valid else "invalid_npz_non_evidence"
            else:
                json.loads(path.read_text())
                row["classification"]="complete_json_non_evidence"
        except Exception:row["classification"]="incomplete_non_evidence"
        result[str(path)]=row
    return result


def certify_active_artifacts(value):
    try:
        current=active_artifacts()
        return set(value)==set(current) and all(item==current[path] for path,item in value.items())
    except Exception:return False


def recertify_retained_p8(output=OUTPUT):
    try:
        output=Path(output);summary=json.loads(output.read_text());pointer=json.loads(latest_path().read_text())
        manifest=json.loads(manifest_path().read_text());paths=worker_paths()
        with np.load(paths["input"],allow_pickle=False) as archive:construction={k:archive[k] for k in archive.files}
        with np.load(paths["npz"],allow_pickle=False) as archive:worker_arrays={k:archive[k] for k in archive.files}
        worker_summary=json.loads(paths["json"].read_text())
        expected_request={"protocol":WORKER_PROTOCOL,"identity":["development",12600],
            "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
            "output_json_path":str(paths["json"]),"output_npz_path":str(paths["npz"]),
            "extension":EXTENSION,"wall_limit_s":300.}
        authentication,prerequisite=authenticate_cpu_prerequisite()
        authentication=canonical_authentication(authentication);model,kin,rnea,*_=_production_pin_context()
        construction_cert=certify_construction(construction,prerequisite,kin)
        science=certify_science(worker_arrays,construction,prerequisite,kin,rnea)
        checkpoints=[json.loads(checkpoint_path(i).read_text()) for i in range(4)]
        stages=("generation_zero","cpu_authenticated","cuda_certified","honest_end")
        expected_checkpoint_counts=(zero_counts(),cpu_counts(),success_counts(),success_counts())
        expected_details=({}, {"authentication":authentication,"construction":construction_cert},
            {"certificate":science},{"certificate":science})
        checkpoint_ok=all(set(doc)=={"protocol","generation","stage","incomplete","counts",
                "detail","provenance","evidence"} and doc["protocol"]==PROTOCOL
            and doc["generation"]==i and doc["stage"]==stages[i] and doc["incomplete"] is True
            and doc["counts"]==expected_checkpoint_counts[i] and doc["detail"]==expected_details[i]
            and doc["evidence"] is False and certify_provenance(doc["provenance"],i==3)
            for i,doc in enumerate(checkpoints))
        side=[*map(checkpoint_path,range(4)),paths["request"],paths["input"],paths["json"],paths["npz"]]
        expected_manifest={"protocol":PROTOCOL,"json_path":str(output),"json_sha256":sha(output),
            "side_artifacts":[{"path":str(p),"sha256":sha(p),"size":p.stat().st_size} for p in side],
            "overall_pass":True}
        expected_pointer={"protocol":PROTOCOL,"incomplete":False,"json_path":str(output),
            "json_sha256":sha(output),"manifest_path":str(manifest_path()),
            "manifest_sha256":sha(manifest_path()),"publication_finish_elapsed_s":pointer[
                "publication_finish_elapsed_s"]}
        summary_keys={"protocol","incomplete","completed","pending","counts","authentication",
            "construction_certificate","worker","worker_execution","certificate","provenance",
            "semantic_finish_elapsed_s","evidence","oracle_evidence","benchmark_evidence",
            "sqp_evidence"}
        summary_ok=bool(set(summary)==summary_keys and summary["protocol"]==PROTOCOL
            and summary["incomplete"] is False and summary["completed"]==2 and summary["pending"]==0
            and summary["counts"]==success_counts() and summary["authentication"]==authentication
            and summary["construction_certificate"]==construction_cert and summary["worker"]==worker_summary
            and certify_worker_execution(summary["worker_execution"],summary["counts"])
            and summary["certificate"]==science and summary["provenance"]==checkpoints[3]["provenance"]
            and np.isfinite(summary["semantic_finish_elapsed_s"])
            and 0<=summary["semantic_finish_elapsed_s"]<=pointer["publication_finish_elapsed_s"]<=600.
            and summary["evidence"] is True and summary["oracle_evidence"] is False
            and summary["benchmark_evidence"] is False and summary["sqp_evidence"] is False)
        gates={"request":json.loads(paths["request"].read_text())==expected_request,
            "worker":certify_worker_summary(worker_summary,worker_arrays,construction),
            "construction":construction_cert["passes"],"science":science["passes"],
            "summary":summary_ok,
            "checkpoints":checkpoint_ok,"manifest":manifest==expected_manifest,
            "pointer":pointer==expected_pointer and 0<=pointer["publication_finish_elapsed_s"]<=600.,
            "namespace":set(OUTPUT.parent.glob("p8*"))==set((*side,output,manifest_path(),latest_path()))}
        return {"gates":gates,"passes":bool(all(gates.values()))}
    except Exception:return {"passes":False}


def recertify_retained_failure(output=OUTPUT):
    try:
        rejected=json.loads(rejection_path().read_text());existing=[i for i in range(4)
            if checkpoint_path(i).is_file()]
        boundary=bool(set(rejected)=={"protocol","stage","error_type","error_message",
                "incomplete","completed","pending","counts","trigger_elapsed_s",
                "cleanup_finish_elapsed_s","wall_limit_s","active_artifacts",
                "artifact_classifications","diagnostic_certificate","worker_execution","evidence"}
            and rejected["protocol"]==PROTOCOL and rejected["incomplete"] is True
            and rejected["evidence"] is False and certify_counts(rejected["counts"])
            and certify_worker_execution(rejected["worker_execution"],rejected["counts"])
            and rejected["stage"] in ("pilot_failed","runtime_watchdog_rejected")
            and rejected["pending"]==2-rejected["completed"]
            and rejected["active_artifacts"]==active_artifacts()
            and 0<=rejected["trigger_elapsed_s"]<=rejected["cleanup_finish_elapsed_s"])
        stages=("generation_zero","cpu_authenticated","cuda_certified","honest_end")
        authentication=prerequisite=construction=construction_cert=science=None
        paths=worker_paths();worker_summary=worker_arrays=None
        if len(existing)>=2:
            authentication,prerequisite=authenticate_cpu_prerequisite()
            authentication=canonical_authentication(authentication)
            _model,kin,rnea,*_=_production_pin_context()
            with np.load(paths["input"],allow_pickle=False) as archive:
                construction={k:archive[k] for k in archive.files}
            construction_cert=certify_construction(construction,prerequisite,kin)
            if not construction_cert["passes"]:return {"passes":False}
            expected_request={"protocol":WORKER_PROTOCOL,"identity":["development",12600],
                "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
                "output_json_path":str(paths["json"]),"output_npz_path":str(paths["npz"]),
                "extension":EXTENSION,"wall_limit_s":300.}
            if json.loads(paths["request"].read_text())!=expected_request:return {"passes":False}
        complete_worker=paths["json"].is_file() and paths["npz"].is_file()
        known_complete=rejected["worker_execution"]["status"]=="known_complete"
        if complete_worker and known_complete:
            if not all(paths[key].is_file() for key in ("request","input","json","npz")) \
                    or prerequisite is None:return {"passes":False}
            worker_summary=json.loads(paths["json"].read_text())
            with np.load(paths["npz"],allow_pickle=False) as archive:
                worker_arrays={k:archive[k] for k in archive.files}
            if not certify_worker_summary(worker_summary,worker_arrays,construction):return {"passes":False}
            if rejected["counts"]["science_routes_completed"]==2:
                science=certify_science(worker_arrays,construction,prerequisite,kin,rnea)
            else:
                if not certify_science_prefix(worker_arrays,construction,kin,rnea,
                        rejected["counts"]):return {"passes":False}
                science=(None if rejected["stage"]=="runtime_watchdog_rejected"
                    or rejected["counts"]["science_routes_attempted"]==0
                    else {"passes":False,"operational_incomplete":True})
        elif paths["npz"].is_file():
            with np.load(paths["npz"],allow_pickle=False) as archive:
                orphan_arrays={key:archive[key] for key in archive.files}
            # Exact outputs are bound even if summary parsing failed. Malformed output bytes
            # remain explicitly classified and cannot become evidence.
            if known_complete and (construction is None
                    or not certify_output(orphan_arrays,construction)):return {"passes":False}
        if rejected["diagnostic_certificate"]!=science:return {"passes":False}
        if rejected["artifact_classifications"]!=classify_artifacts(construction):
            return {"passes":False}
        expected_counts=(zero_counts(),cpu_counts(),success_counts(),success_counts())
        expected_details=({}, {"authentication":authentication,"construction":construction_cert},
            {"certificate":science},{"certificate":science})
        chain=existing==list(range(len(existing)))
        for generation in existing:
            doc=json.loads(checkpoint_path(generation).read_text())
            if (set(doc)!={"protocol","generation","stage","incomplete","counts","detail",
                    "provenance","evidence"} or doc["protocol"]!=PROTOCOL
                    or doc["generation"]!=generation or doc["stage"]!=stages[generation]
                    or doc["incomplete"] is not True or doc["counts"]!=expected_counts[generation]
                    or doc["detail"]!=expected_details[generation] or doc["evidence"] is not False
                    or not certify_provenance(doc["provenance"],generation==3)):
                chain=False
        if paths["rejected"].is_file() and not certify_worker_rejection(json.loads(
                paths["rejected"].read_text())):return {"passes":False}
        # A late publication may leave complete non-authoritative final documents.
        final_keys={"protocol","incomplete","completed","pending","counts","authentication",
            "construction_certificate","worker","worker_execution","certificate","provenance",
            "semantic_finish_elapsed_s","evidence","oracle_evidence","benchmark_evidence","sqp_evidence"}
        for candidate in (OUTPUT,Path(str(OUTPUT)+".candidate")):
            if not candidate.is_file():continue
            orphan=json.loads(candidate.read_text())
            if (science is None or set(orphan)!=final_keys or orphan.get("protocol")!=PROTOCOL
                    or orphan.get("certificate")!=science or orphan.get("counts")!=success_counts()
                    or orphan.get("authentication")!=authentication
                    or orphan.get("construction_certificate")!=construction_cert
                    or orphan.get("worker")!=worker_summary or orphan.get("incomplete") is not False
                    or orphan.get("completed")!=2 or orphan.get("pending")!=0
                    or not certify_worker_execution(orphan.get("worker_execution"),orphan["counts"])
                    or not certify_provenance(orphan.get("provenance"),True)
                    or not np.isfinite(orphan.get("semantic_finish_elapsed_s",np.nan))
                    or not 0<=orphan["semantic_finish_elapsed_s"]<=600.
                    or orphan.get("evidence") is not True
                    or orphan.get("oracle_evidence") is not False
                    or orphan.get("benchmark_evidence") is not False
                    or orphan.get("sqp_evidence") is not False):
                return {"passes":False}
        if manifest_path().is_file() or Path(str(manifest_path())+".candidate").is_file():
            if not OUTPUT.is_file():return {"passes":False}
            side=[*map(checkpoint_path,range(4)),paths["request"],paths["input"],paths["json"],paths["npz"]]
            expected_manifest={"protocol":PROTOCOL,"json_path":str(OUTPUT),
                "json_sha256":sha(OUTPUT),"side_artifacts":[{"path":str(path),
                "sha256":sha(path),"size":path.stat().st_size} for path in side],"overall_pass":True}
            for candidate in (manifest_path(),Path(str(manifest_path())+".candidate")):
                if candidate.is_file() and json.loads(candidate.read_text())!=expected_manifest:
                    return {"passes":False}
        if latest_path().is_file() and json.loads(latest_path().read_text()).get("incomplete") is False:
            return {"passes":False}
        latest=json.loads(latest_path().read_text());last=checkpoint_path(existing[-1])
        allowed={rejection_path(),latest_path(),*(checkpoint_path(i) for i in existing)}
        allowed.update(path for path in worker_paths().values() if path.is_file())
        if OUTPUT.is_file():allowed.add(OUTPUT)
        if manifest_path().is_file():allowed.add(manifest_path())
        allowed.update(Path(path) for path in rejected["artifact_classifications"] if Path(path).is_file())
        return {"passes":bool(boundary and chain and latest=={"protocol":PROTOCOL,
            "generation":existing[-1],"path":str(last),"sha256":sha(last),"incomplete":True}
            and namespace_paths()==allowed)}
    except Exception:return {"passes":False}


def execute(output=OUTPUT,authorization=None,monotonic=time.monotonic): # pragma: no cover
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P8 runner blocked")
    if Path(output).resolve()!=OUTPUT:raise RuntimeError("wrong P8 output")
    if OUTPUT.parent.exists():raise FileExistsError("P8 namespace exists")
    if Path.cwd().resolve()!=Path(AUTHORIZED_CWD) or tuple(sys.orig_argv)!=AUTHORIZED_ORIG_ARGV \
            or {k:os.environ.get(k) for k in THREAD_ENV}!=THREAD_ENV:
        raise RuntimeError("P8 invocation invalid")
    initial=snapshot()
    if not initial["clean"] or initial["extension"]!=frozen_extension():
        raise RuntimeError("P8 provenance invalid")
    start=monotonic();deadline=start+600.;cpu_deadline=start+30.;counts=zero_counts();science=None
    worker_execution={"status":"not_started","returncode":None,
        "internal_counts_known":False,"counts":None}
    OUTPUT.parent.mkdir();prov=provenance(initial);publish_checkpoint(0,"generation_zero",counts,prov,{})
    try:
        authentication,prerequisite=authenticate_cpu_prerequisite()
        authentication=canonical_authentication(authentication);counts["prerequisite_loads"]=1
        _model,kin,rnea,*_=_production_pin_context();counts["parent_pin_contexts"]=1
        construction,construction_detail=construct_cpu(prerequisite,kin,cpu_deadline,monotonic,
            counts,"primary")
        construction_cert=certify_construction(construction,prerequisite,kin,cpu_deadline,
            monotonic,counts,"independent")
        if monotonic()>cpu_deadline:raise TimeoutError("P8 CPU preflight wall limit")
        if not construction_cert["passes"]:raise RuntimeError("P8 CPU construction recert failed")
        publish_checkpoint(1,"cpu_authenticated",counts,prov,{"authentication":authentication,
            "construction":construction_cert})
        paths=worker_paths();atomic_npz(paths["input"],construction)
        request={"protocol":WORKER_PROTOCOL,"identity":["development",12600],
            "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
            "output_json_path":str(paths["json"]),"output_npz_path":str(paths["npz"]),
            "extension":EXTENSION,"wall_limit_s":300.};atomic_json(paths["request"],request)
        remaining=deadline-monotonic()
        if remaining<=0:raise TimeoutError("P8 campaign wall limit")
        env={**os.environ,**THREAD_ENV};counts["worker_attempts"]+=1
        worker_execution={"status":"internal_unknown","returncode":None,
            "internal_counts_known":False,"counts":None}
        try:result=subprocess.run(worker_argv(),cwd=AUTHORIZED_CWD,env=env,
            timeout=min(300.,remaining))
        except subprocess.TimeoutExpired as error:raise TimeoutError("P8 worker wall limit") from error
        worker_execution["returncode"]=result.returncode
        if result.returncode:
            if paths["rejected"].is_file():
                worker_rejection=json.loads(paths["rejected"].read_text())
                if not certify_worker_rejection(worker_rejection):
                    raise RuntimeError("P8 worker rejection invalid")
                mapping={"pin_model_contexts":"worker_pin_contexts","rnea_calls":"worker_rnea_calls"}
                for key,value in worker_rejection["counts"].items():counts[mapping.get(key,key)]=value
                worker_execution={"status":"known_partial","returncode":result.returncode,
                    "internal_counts_known":True,"counts":worker_rejection["counts"]}
                if worker_rejection["error_type"]=="TimeoutError":raise TimeoutError(
                    "P8 worker wall limit")
            raise RuntimeError("P8 worker failed")
        worker_summary=json.loads(paths["json"].read_text())
        with np.load(paths["npz"],allow_pickle=False) as archive:worker_arrays={k:archive[k] for k in archive.files}
        if not certify_worker_summary(worker_summary,worker_arrays,construction):
            raise RuntimeError("P8 worker invalid")
        worker_execution={"status":"known_complete","returncode":0,"internal_counts_known":True,
            "counts":worker_summary["counts"]}
        mapping={"pin_model_contexts":"worker_pin_contexts","rnea_calls":"worker_rnea_calls"}
        for key,value in worker_summary["counts"].items():counts[mapping.get(key,key)]=value
        counts["worker_successes"]+=1
        science=certify_science(worker_arrays,construction,prerequisite,kin,rnea,deadline,
            monotonic,counts)
        if monotonic()>deadline:raise TimeoutError("P8 campaign wall limit")
        if not science["passes"]:raise RuntimeError("P8 scientific certificate failed")
        publish_checkpoint(2,"cuda_certified",counts,prov,{"certificate":science})
        end=snapshot();prov=provenance(initial,end)
        publish_checkpoint(3,"honest_end",counts,prov,{"certificate":science})
        semantic_elapsed=float(monotonic()-start)
        if semantic_elapsed>600.:raise TimeoutError("P8 campaign wall limit")
        final={"protocol":PROTOCOL,"incomplete":False,"completed":2,"pending":0,
            "counts":counts,"authentication":authentication,"construction_certificate":construction_cert,
            "worker":worker_summary,"worker_execution":worker_execution,"certificate":science,"provenance":prov,
            "semantic_finish_elapsed_s":semantic_elapsed,"evidence":True,
            "oracle_evidence":False,"benchmark_evidence":False,"sqp_evidence":False}
        atomic_json(OUTPUT,final)
        side=[*map(checkpoint_path,range(4)),paths["request"],paths["input"],paths["json"],paths["npz"]]
        manifest={"protocol":PROTOCOL,"json_path":str(OUTPUT),"json_sha256":sha(OUTPUT),
            "side_artifacts":[{"path":str(p),"sha256":sha(p),"size":p.stat().st_size} for p in side],
            "overall_pass":True};atomic_json(manifest_path(),manifest)
        pointer={"protocol":PROTOCOL,"incomplete":False,"json_path":str(OUTPUT),
            "json_sha256":sha(OUTPUT),"manifest_path":str(manifest_path()),
            "manifest_sha256":sha(manifest_path()),"publication_finish_elapsed_s":None}
        finish=float(monotonic()-start)
        if finish>600.:raise TimeoutError("P8 publication wall limit")
        pointer["publication_finish_elapsed_s"]=finish;atomic_json(latest_path(),pointer,True)
    except Exception as error:
        trigger=float(monotonic()-start);stage="runtime_watchdog_rejected" if isinstance(error,
            (TimeoutError,subprocess.TimeoutExpired)) else "pilot_failed"
        classifications=classify_artifacts(construction if 'construction' in locals() else None)
        artifacts=active_artifacts();cleanup=float(monotonic()-start)
        rejection={"protocol":PROTOCOL,"stage":stage,"error_type":type(error).__name__,
            "error_message":str(error),"incomplete":True,"completed":counts["completed"],
            "pending":2-counts["completed"],"counts":counts,"trigger_elapsed_s":trigger,
            "cleanup_finish_elapsed_s":cleanup,"wall_limit_s":600.,
            "active_artifacts":artifacts,"diagnostic_certificate":science,
            "artifact_classifications":classifications,
            "worker_execution":worker_execution,"evidence":False}
        if not rejection_path().exists():atomic_json(rejection_path(),rejection)
        raise


def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    execute(args.output,RUNNER_EXECUTION_AUTHORIZATION)


if __name__=="__main__":main()
