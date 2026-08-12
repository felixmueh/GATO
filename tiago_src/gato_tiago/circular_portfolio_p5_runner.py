"""Transactional two-route orchestration boundary for the N260 P5 pilot."""

from __future__ import annotations

import argparse,hashlib,importlib.metadata,json,os,shlex,subprocess,sys,time
from pathlib import Path

import numpy as np

from gato_tiago.circular_portfolio_p3 import perturbation_direction
from gato_tiago.circular_portfolio_p3_runner import geometry_from_lane_zero,json_safe
from gato_tiago.circular_portfolio_p5 import (BUILD_COMMAND,CAMPAIGN_WALL_LIMIT_S,
    DENSE_SAMPLES,DENSE_SUBSTEPS,DT,EXTENSION,INTERVALS,KNOTS,LANES,OUTPUT,
    P4_V2_REJECTED_REPORT,PILOT_IDENTITY,PILOT_ROUTES,PROTOCOL,WORKER_PROTOCOL,
    array_hash,certify_pair,certify_pair_detail,certify_proxy,certify_route,
    certify_route_detail,pack_seed,planned_proxy,
    route_paths,worker_input_schema,worker_paths)
from gato_tiago.circular_portfolio_p5_worker import (certify_output as certify_worker_output,
    certify_partial_counts as certify_worker_partial_counts)


RUNNER_EXECUTION_AUTHORIZATION=object()
AUTHORIZED_CWD="/workspace/GATO"
AUTHORIZED_ORIG_ARGV=("python","-B","-m","gato_tiago.circular_portfolio_p5_runner",
    "--execute","--output",str(OUTPUT))
THREAD_ENV={"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
    "NUMEXPR_NUM_THREADS":"1"}
SOURCE_PATHS=("CMakeLists.txt","tools/build.sh","python/bindings.cu","python/bsqp/interface.py",
    "gato/bsqp/bsqp.cuh","gato/bsqp/kernels/tool_position.cuh",
    "gato/dynamics/integrator.cuh",
    "gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "gato/dynamics/tiago_right/tiago_right_grid.cuh",
    "gato/dynamics/tiago_right/tiago_right_arm.urdf","gato/utils/cuda.cuh",
    "tiago_src/gato_tiago/circular_portfolio.py",
    "tiago_src/gato_tiago/circular_portfolio_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_prerequisite_runner.py",
    "tiago_src/gato_tiago/config.py",
    "tiago_src/gato_tiago/circular_portfolio_p2.py",
    "tiago_src/gato_tiago/circular_portfolio_p2_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_p2_preflight_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p3.py",
    "tiago_src/gato_tiago/circular_portfolio_p3_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p5.py",
    "tiago_src/gato_tiago/circular_portfolio_p5_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p5_worker.py",
    "tiago_src/gato_tiago/multimodal_pillar.py",
    "tiago_src/gato_tiago/multimodal_toll.py",
    "tiago_src/gato_tiago/multimodal_toll_v4.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_oracle_schema.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_runner.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight_v4.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight_v4_runner.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight_v4_worker.py",
    "tiago_src/gato_tiago/multimodal_toll_oracle_v2.py",
    "tiago_src/gato_tiago/multimodal_toll_oracle_v2_prerequisite_runner.py")

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError(f"P5 artifact exists: {path}")
    candidate.write_text(json.dumps(value,sort_keys=True,separators=(",",":")))
    candidate.replace(path)
def atomic_npz(path,arrays):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError(f"P5 artifact exists: {path}")
    with candidate.open("xb") as stream:np.savez(stream,**arrays)
    candidate.replace(path)
def atomic_pointer(path,value):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if candidate.exists():raise FileExistsError(f"P5 pointer candidate exists: {candidate}")
    candidate.write_text(json.dumps(value,sort_keys=True,separators=(",",":")))
    candidate.replace(path)
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
            "arch":EXTENSION["arch"],"attributes":{k:EXTENSION[k] for k in
                ("KNOT_POINTS","REFERENCE_SIZE","TOOL_POSITION_FRAME","TOOL_POSITION_SIZE")}}}
def frozen_extension():
    return {"path":str((Path(__file__).resolve().parents[2]/EXTENSION["relative_path"]).resolve()),
        "sha256":EXTENSION["sha256"],"size":EXTENSION["size"],
        "build_head":EXTENSION["build_head"],"arch":"61-real",
        "attributes":{k:EXTENSION[k] for k in ("KNOT_POINTS","REFERENCE_SIZE",
            "TOOL_POSITION_FRAME","TOOL_POSITION_SIZE")}}
def start_provenance():
    current=snapshot();environment={k:os.environ.get(k) for k in THREAD_ENV}
    if environment!=THREAD_ENV:raise RuntimeError("P5 requires single-thread environment")
    if current["extension"]!=frozen_extension():raise RuntimeError("P5 extension pin mismatch")
    return {"cwd":str(Path.cwd().resolve()),"orig_argv":list(sys.orig_argv),
        "command":shlex.join(sys.orig_argv),"head_start":current["head"],"head_end":None,
        "clean_start":current["clean"],"clean_end":None,"sources_start":current["sources"],
        "sources_end":None,"extension_start":current["extension"],"extension_end":None,
        "thread_environment":environment,"runtime_versions":{"python":sys.version,
            "numpy":np.__version__,"pinocchio":importlib.metadata.version("pin")},
        "p4_v2_rejected_report":P4_V2_REJECTED_REPORT}
def finish_provenance(value):
    current=snapshot();value.update({"head_end":current["head"],"clean_end":current["clean"],
        "sources_end":current["sources"],"extension_end":current["extension"]})
def certify_provenance(value,final):
    current=snapshot()
    keys={"cwd","orig_argv","command","head_start","head_end","clean_start","clean_end",
        "sources_start","sources_end","extension_start","extension_end","thread_environment",
        "runtime_versions","p4_v2_rejected_report"}
    return bool(set(value)==keys and value.get("cwd")==AUTHORIZED_CWD
        and tuple(value.get("orig_argv",()))==AUTHORIZED_ORIG_ARGV
        and value.get("command")==shlex.join(AUTHORIZED_ORIG_ARGV)
        and value.get("head_start")==current["head"]
        and value.get("clean_start") is current["clean"] is True
        and value.get("sources_start")==current["sources"]
        and value.get("extension_start")==frozen_extension()
        and value.get("runtime_versions")=={"python":sys.version,"numpy":np.__version__,
            "pinocchio":importlib.metadata.version("pin")}
        and value.get("p4_v2_rejected_report")==P4_V2_REJECTED_REPORT
        and ((not final and value.get("head_end") is value.get("clean_end")
            is value.get("sources_end") is value.get("extension_end") is None)
          or (final and value.get("head_end")==value["head_start"]
            and value.get("clean_end") is True and value.get("sources_end")==value["sources_start"]
            and value.get("extension_end")==value["extension_start"])))

def static_design():
    return {"protocol":PROTOCOL,"output":str(OUTPUT),"pilot_identity":list(PILOT_IDENTITY),
        "routes":[list(value) for value in PILOT_ROUTES],"kp":100.0,"kd":20.0,
        "knots":260,"dt":.0125,"dense_substeps":64,"attempts":2,"retries":0,
        "optimizer_calls":0,"solve_calls":0,"sqp_calls":0,"rng_calls":0,
        "build_command":list(BUILD_COMMAND),"extension":EXTENSION,
        "campaign_wall_limit_s":CAMPAIGN_WALL_LIMIT_S,
        "p4_v2_rejected_report":P4_V2_REJECTED_REPORT,"p4_v2_artifact_loads":0,
        "cuda_evidence":False,"benchmark_evidence":False,"sqp_evidence":False}
def certify_static_design(value):
    return bool(value==static_design() and value["attempts"]==2 and value["retries"]==0
        and value["optimizer_calls"]==value["solve_calls"]==value["sqp_calls"]==0
        and value["p4_v2_artifact_loads"]==0 and value["extension"]==EXTENSION)

def initial_counts():
    return {"prerequisite_artifact_loads":0,"prerequisite_independent_recertifications":0,
        "pin_model_contexts":0,
        "primary_pin_rollouts":0,"independent_pin_replays":0,"rnea_calls":0,
        "primary_aba_calls":0,"independent_aba_calls":0,"primary_fk_calls":0,
        "independent_fk_calls":0,"task_endpoint_fk_calls":0,"direction_fk_calls":0,
        "owned_task_endpoint_fk_calls":0,"owned_direction_fk_calls":0,
        "owned_pin_replays":0,"owned_rnea_calls":0,
        "owned_aba_calls":0,"owned_fk_calls":0,"worker_subprocess_attempts":0,
        "worker_subprocess_successes":0,"b1_constructors":0,"b16_constructors":0,
        "b1_sim_forward_calls":0,"b16_sim_forward_calls":0,
        "b1_tool_position_calls":0,"b16_tool_position_calls":0,"optimizer_calls":0,
        "solve_calls":0,"sqp_calls":0,"rng_calls":0,"task_construction_calls":0}
def final_counts():
    value=initial_counts();value.update({"prerequisite_artifact_loads":2,
        "prerequisite_independent_recertifications":1,"pin_model_contexts":1,
        "primary_pin_rollouts":2,"independent_pin_replays":2,"rnea_calls":2*INTERVALS,
        "primary_aba_calls":2*INTERVALS*DENSE_SUBSTEPS,
        "independent_aba_calls":2*INTERVALS*DENSE_SUBSTEPS,
        "primary_fk_calls":2*DENSE_SAMPLES,"independent_fk_calls":2*DENSE_SAMPLES,
        "task_endpoint_fk_calls":1,"direction_fk_calls":1,
        "owned_task_endpoint_fk_calls":1,"owned_direction_fk_calls":1,
        "owned_pin_replays":2,"owned_rnea_calls":2*INTERVALS,
        "owned_aba_calls":2*INTERVALS*DENSE_SUBSTEPS,"owned_fk_calls":2*DENSE_SAMPLES,
        "worker_subprocess_attempts":1,"worker_subprocess_successes":1,
        "b1_constructors":1,"b16_constructors":1,
        "b1_sim_forward_calls":INTERVALS*DENSE_SUBSTEPS,
        "b16_sim_forward_calls":INTERVALS*DENSE_SUBSTEPS,
        "b1_tool_position_calls":DENSE_SAMPLES,"b16_tool_position_calls":DENSE_SAMPLES})
    return value
def pin_routes_counts():
    value=initial_counts();value.update({"prerequisite_artifact_loads":1,"pin_model_contexts":1,
        "primary_pin_rollouts":2,"independent_pin_replays":2,"rnea_calls":2*INTERVALS,
        "primary_aba_calls":2*INTERVALS*DENSE_SUBSTEPS,
        "independent_aba_calls":2*INTERVALS*DENSE_SUBSTEPS,
        "primary_fk_calls":2*DENSE_SAMPLES,"independent_fk_calls":2*DENSE_SAMPLES,
        "task_endpoint_fk_calls":1,"direction_fk_calls":1})
    return value

def _guard(deadline,monotonic):
    if monotonic()>=deadline:raise TimeoutError("P5 campaign wall limit")
def _publication_finish_elapsed(started,deadline,monotonic):
    elapsed=monotonic()-started
    _guard(deadline,monotonic)
    return elapsed
def _publish_final_pointer(payload,started,deadline,monotonic):
    document=dict(payload)
    document["publication_finish_elapsed_s"]=_publication_finish_elapsed(
        started,deadline,monotonic)
    atomic_pointer(OUTPUT.with_name("p5.final.latest.json"),document)
    return document

def _pin_rollout(proxy,q0,rnea,aba,kinematics,counts,phase,deadline,monotonic):
    state=np.empty((DENSE_SAMPLES,14),np.float64);tool=np.empty((DENSE_SAMPLES,3),np.float64)
    controls=np.empty((INTERVALS,7),np.float32);state[0]=np.r_[q0,np.zeros(7)]
    _guard(deadline,monotonic);tool[0]=kinematics(q0)[0];counts[f"{phase}_fk_calls"]+=1;index=0
    for knot in range(INTERVALS):
        _guard(deadline,monotonic)
        q,qd=state[index,:7],state[index,7:]
        acceleration=proxy["planned_qdd_float64"][knot]+100.*(proxy["planned_q_float64"][knot]-q) \
            +20.*(proxy["planned_qd_float64"][knot]-qd)
        controls[knot]=np.asarray(rnea(q,qd,acceleration),np.float32);counts["rnea_calls"]+=1
        for _ in range(DENSE_SUBSTEPS):
            _guard(deadline,monotonic)
            q,qd=state[index,:7],state[index,7:];qdd=np.asarray(aba(q,qd,controls[knot].astype(np.float64)))
            if phase=="primary":counts["primary_aba_calls"]+=1
            else:counts["independent_aba_calls"]+=1
            state[index+1,:7]=q+DT/DENSE_SUBSTEPS*qd+.5*(DT/DENSE_SUBSTEPS)**2*qdd
            state[index+1,7:]=qd+DT/DENSE_SUBSTEPS*qdd;index+=1
            tool[index]=kinematics(state[index,:7])[0];counts[f"{phase}_fk_calls"]+=1
    counts["primary_pin_rollouts" if phase=="primary" else "independent_pin_replays"]+=1
    return state,tool,controls

def _pin_replay(q0,controls,aba,kinematics,counts,deadline,monotonic):
    state=np.empty((DENSE_SAMPLES,14),np.float64);tool=np.empty((DENSE_SAMPLES,3),np.float64)
    state[0]=np.r_[q0,np.zeros(7)];_guard(deadline,monotonic)
    tool[0]=kinematics(q0)[0];counts["independent_fk_calls"]+=1;index=0
    for control in controls:
        applied=np.asarray(control,np.float32).astype(np.float64)
        for _ in range(DENSE_SUBSTEPS):
            _guard(deadline,monotonic)
            q,qd=state[index,:7],state[index,7:];qdd=np.asarray(aba(q,qd,applied))
            counts["independent_aba_calls"]+=1
            state[index+1,:7]=q+DT/DENSE_SUBSTEPS*qd+.5*(DT/DENSE_SUBSTEPS)**2*qdd
            state[index+1,7:]=qd+DT/DENSE_SUBSTEPS*qdd;index+=1
            tool[index]=kinematics(state[index,:7])[0];counts["independent_fk_calls"]+=1
    counts["independent_pin_replays"]+=1
    return state,tool

ROUTE_ARRAY_SPECS={"direction_float64":((7,),np.float64),
    "planned_q_float64":((KNOTS,7),np.float64),"planned_qd_float64":((KNOTS,7),np.float64),
    "planned_qdd_float64":((INTERVALS,7),np.float64),
    "applied_controls_float32":((INTERVALS,7),np.float32),
    "primary_pin_state_float64":((DENSE_SAMPLES,14),np.float64),
    "primary_pin_tool_float64":((DENSE_SAMPLES,3),np.float64),
    "independent_pin_state_float64":((DENSE_SAMPLES,14),np.float64),
    "independent_pin_tool_float64":((DENSE_SAMPLES,3),np.float64)}
FINAL_ARRAY_SPECS={"public_x0_float32":((14,),np.float32),
    "accepted_task_reference_float32":((960,),np.float32),
    "quarantined_q8_float64":((7,),np.float64),"q8_tool_float64":((3,),np.float64),
    "default_side_int8":((),np.int8),"joint_lower_float64":((7,),np.float64),
    "joint_upper_float64":((7,),np.float64),"velocity_limit_float64":((7,),np.float64),
    "effort_limit_float64":((7,),np.float64),"reference_float32":((KNOTS*10,),np.float32),
    "pillar_float64":((2,),np.float64),"short_unit_float64":((2,),np.float64)}
def exact_arrays(arrays,specs):
    return bool(set(arrays)==set(specs) and all(np.asarray(arrays[k]).shape==shape
        and np.asarray(arrays[k]).dtype==np.dtype(dtype) and np.isfinite(arrays[k]).all()
        for k,(shape,dtype) in specs.items()))
def route_arrays(direction,proxy,state,tool,replay_state,replay_tool,controls):
    return {"direction_float64":np.asarray(direction,np.float64),
        "planned_q_float64":proxy["planned_q_float64"],
        "planned_qd_float64":proxy["planned_qd_float64"],
        "planned_qdd_float64":proxy["planned_qdd_float64"],
        "applied_controls_float32":controls,"primary_pin_state_float64":state,
        "primary_pin_tool_float64":tool,"independent_pin_state_float64":replay_state,
        "independent_pin_tool_float64":replay_tool}
def route_summary(index,route,arrays):
    paths=route_paths(index)
    return {"protocol":PROTOCOL,"identity":[*PILOT_IDENTITY,route,0],"index":index,
        "npz_path":str(paths["npz"]),"npz_sha256":sha(paths["npz"]),
        "array_names":sorted(arrays),"array_hashes":{k:array_hash(arrays[k]) for k in sorted(arrays)},
        "primary_independent_exact":np.array_equal(arrays["primary_pin_state_float64"],
            arrays["independent_pin_state_float64"])
            and np.array_equal(arrays["primary_pin_tool_float64"],
                arrays["independent_pin_tool_float64"]),"incomplete":True}
def certify_route_artifact(summary,arrays,index,route):
    paths=route_paths(index);keys={"protocol","identity","index","npz_path","npz_sha256",
        "array_names","array_hashes","primary_independent_exact","incomplete"}
    return bool(set(summary)==keys and summary["protocol"]==PROTOCOL
        and summary["identity"]==[*PILOT_IDENTITY,route,0] and summary["index"]==index
        and summary["npz_path"]==str(paths["npz"])
        and summary["npz_sha256"]==sha(paths["npz"]) and exact_arrays(arrays,ROUTE_ARRAY_SPECS)
        and summary["array_names"]==sorted(ROUTE_ARRAY_SPECS)
        and summary["array_hashes"]=={k:array_hash(arrays[k]) for k in sorted(arrays)}
        and summary["primary_independent_exact"] is True and summary["incomplete"] is True)

def certify_route_artifact_prefix(values):
    if not isinstance(values,list) or len(values)>2:return False
    for index,summary in enumerate(values):
        route=PILOT_ROUTES[index][0];paths=route_paths(index)
        if not paths["json"].is_file() or not paths["npz"].is_file():return False
        if json.loads(paths["json"].read_text())!=summary:return False
        arrays=load_npz(paths["npz"])
        if not certify_route_artifact(summary,arrays,index,route):return False
    return True

def certify_active_worker_artifacts(prerequisite,counts):
    if counts["worker_subprocess_attempts"]==0:return True
    paths=worker_paths()
    if not paths["request"].is_file() or not paths["input"].is_file():return False
    request=json.loads(paths["request"].read_text())
    expected={"protocol":WORKER_PROTOCOL,"input_path":str(paths["input"]),
        "input_sha256":sha(paths["input"]),"output_path":str(paths["json"]),"extension":EXTENSION}
    if request!=expected:return False
    inputs=load_npz(paths["input"])
    if not worker_input_schema(inputs):return False
    route_arrays_by_index=[load_npz(route_paths(i)["npz"]) for i in (0,1)]
    expected_controls=np.stack([route_arrays_by_index[lane%2]["applied_controls_float32"]
                                for lane in range(LANES)])
    expected_seeds=np.stack([pack_seed(
        route_arrays_by_index[lane%2]["primary_pin_state_float64"][::DENSE_SUBSTEPS,:7],
        route_arrays_by_index[lane%2]["primary_pin_state_float64"][::DENSE_SUBSTEPS,7:],
        route_arrays_by_index[lane%2]["applied_controls_float32"]) for lane in range(LANES)])
    expected_x0=np.repeat(prerequisite["public_x0_float32"][0][None],LANES,0)
    if not np.array_equal(inputs["x0_float32"],expected_x0) \
            or not np.array_equal(inputs["controls_float32"],expected_controls) \
            or not np.array_equal(inputs["b16_seed_xu_float32"],expected_seeds) \
            or not np.array_equal(inputs["b1_seed_xu_float32"],expected_seeds[0]):return False
    if paths["npz"].is_file():
        output=load_npz(paths["npz"])
        from gato_tiago.circular_portfolio_p5 import worker_output_schema
        if not worker_output_schema(output):return False
        if (not np.array_equal(output["captured_x0_float32"],inputs["x0_float32"])
                or not np.array_equal(output["captured_controls_float32"],inputs["controls_float32"])):
            return False
    if paths["json"].is_file():
        summary=json.loads(paths["json"].read_text());output=load_npz(paths["npz"])
        if not certify_worker_output(summary,output,inputs)["passes"]:return False
    return True

def certify_final_arrays_task(arrays,prerequisite):
    if not exact_arrays(arrays,FINAL_ARRAY_SPECS):return False
    expected={"public_x0_float32":prerequisite["public_x0_float32"][0],
        "accepted_task_reference_float32":prerequisite["accepted_task_reference_float32"][0],
        "quarantined_q8_float64":prerequisite["quarantined_q8_float64"][0],
        "q8_tool_float64":prerequisite["q8_tool_float64"][0],
        "default_side_int8":prerequisite["public_default_side_int8"][0],
        "joint_lower_float64":prerequisite["joint_lower_float64"],
        "joint_upper_float64":prerequisite["joint_upper_float64"],
        "velocity_limit_float64":prerequisite["velocity_limit_float64"],
        "effort_limit_float64":prerequisite["effort_limit_float64"]}
    if any(not np.array_equal(arrays[k],v) for k,v in expected.items()):return False
    if len([k for k in arrays if k not in expected])!=3:return False
    return True

def certify_failure_side_artifacts(prerequisite):
    # Complete candidates are still non-evidence, but their bytes must be
    # semantically valid for the exact canonical destination.
    retained_routes={}
    for index,(route,_profile,_amplitude) in enumerate(PILOT_ROUTES):
        paths=route_paths(index);npz_candidates=[paths["npz"],Path(str(paths["npz"])+".candidate")]
        json_candidates=[paths["json"],Path(str(paths["json"])+".candidate")]
        arrays=None;npz_hash=None
        for path in npz_candidates:
            if path.is_file():
                try:current=load_npz(path)
                except Exception:
                    if path.name.endswith(".candidate"):continue
                    return False
                if not exact_arrays(current,ROUTE_ARRAY_SPECS):return False
                if arrays is not None and any(not np.array_equal(current[k],arrays[k])
                                              for k in ROUTE_ARRAY_SPECS):return False
                arrays=current
                if path==paths["npz"] or npz_hash is None:npz_hash=sha(path)
        if arrays is not None:retained_routes[index]=arrays
        for path in json_candidates:
            if path.is_file():
                try:summary=json.loads(path.read_text())
                except Exception:
                    if path.name.endswith(".candidate"):continue
                    return False
                if arrays is None or set(summary)!={"protocol","identity","index","npz_path",
                        "npz_sha256","array_names","array_hashes","primary_independent_exact",
                        "incomplete"}:return False
                if (summary["protocol"]!=PROTOCOL or summary["identity"]!=[*PILOT_IDENTITY,route,0]
                        or summary["index"]!=index or summary["npz_path"]!=str(paths["npz"])
                        or summary["npz_sha256"]!=npz_hash
                        or summary["array_names"]!=sorted(ROUTE_ARRAY_SPECS)
                        or summary["array_hashes"]!={k:array_hash(arrays[k]) for k in sorted(arrays)}
                        or summary["primary_independent_exact"] is not True
                        or summary["incomplete"] is not True):return False
    final_arrays=[]
    for path in (OUTPUT.with_suffix(".npz"),Path(str(OUTPUT.with_suffix(".npz"))+".candidate")):
        if path.is_file():
            try:arrays=load_npz(path)
            except Exception:
                if path.name.endswith(".candidate"):continue
                return False
            if not certify_final_arrays_task(arrays,prerequisite):return False
            final_arrays.append(arrays)
    if len(retained_routes)==2 and final_arrays:
        geometry=geometry_from_lane_zero(
            {"pin_dense_tool_float64":retained_routes[0]["primary_pin_tool_float64"]},
            {"pin_dense_tool_float64":retained_routes[1]["primary_pin_tool_float64"]},
            prerequisite["q8_tool_float64"][0],int(prerequisite["public_default_side_int8"][0]))
        if not geometry["passes"]:return False
        expected={"reference_float32":np.tile(geometry["reference_float32"],KNOTS),
            "pillar_float64":geometry["pillar_float64"],
            "short_unit_float64":geometry["short_unit_float64"]}
        if any(any(not np.array_equal(arrays[k],value) for k,value in expected.items())
               for arrays in final_arrays):return False
    paths=worker_paths()
    expected_worker_inputs=None
    if len(retained_routes)==2:
        controls=np.stack([retained_routes[lane%2]["applied_controls_float32"]
                           for lane in range(LANES)])
        seeds=np.stack([pack_seed(
            retained_routes[lane%2]["primary_pin_state_float64"][::DENSE_SUBSTEPS,:7],
            retained_routes[lane%2]["primary_pin_state_float64"][::DENSE_SUBSTEPS,7:],
            retained_routes[lane%2]["applied_controls_float32"]) for lane in range(LANES)])
        expected_worker_inputs={"x0_float32":np.repeat(
            prerequisite["public_x0_float32"][0][None],LANES,0),
            "controls_float32":controls,"b1_seed_xu_float32":seeds[0],
            "b16_seed_xu_float32":seeds}
    for path in (paths["npz"],Path(str(paths["npz"])+".candidate")):
        if path.is_file():
            try:arrays=load_npz(path)
            except Exception:
                if path.name.endswith(".candidate"):continue
                return False
            from gato_tiago.circular_portfolio_p5 import worker_output_schema
            if not worker_output_schema(arrays):return False
            if expected_worker_inputs is not None and (
                    not np.array_equal(arrays["captured_x0_float32"],expected_worker_inputs["x0_float32"])
                    or not np.array_equal(arrays["captured_controls_float32"],
                        expected_worker_inputs["controls_float32"])
                    or not np.array_equal(arrays["captured_b1_seed_xu_float32"],
                        expected_worker_inputs["b1_seed_xu_float32"])
                    or not np.array_equal(arrays["captured_b16_seed_xu_float32"],
                        expected_worker_inputs["b16_seed_xu_float32"])):return False
    return True

def recertify_failure_route_science(prerequisite,kinematics,rnea,aba,deadline,monotonic):
    """Own every complete canonical/orphan route without trusting its summary."""
    q0=np.asarray(prerequisite["public_x0_float32"][0,:7],np.float64)
    qgoal=np.asarray(prerequisite["quarantined_q8_float64"][0],np.float64)
    _guard(deadline,monotonic);start_tool=kinematics(q0)[0]
    direction=perturbation_direction(q0,qgoal,start_tool,
        prerequisite["q8_tool_float64"][0],kinematics,
        int(prerequisite["public_default_side_int8"][0]))
    complete=0
    for index,(route,_profile,_amplitude) in enumerate(PILOT_ROUTES):
        paths=route_paths(index)
        for path in (paths["npz"],Path(str(paths["npz"])+".candidate")):
            if not path.is_file():continue
            try:arrays=load_npz(path)
            except Exception:
                if path.name.endswith(".candidate"):continue
                return False
            if not exact_arrays(arrays,ROUTE_ARRAY_SPECS):return False
            if not np.array_equal(arrays["direction_float64"],direction):return False
            proxy=planned_proxy(q0,qgoal,direction,route)
            if any(not np.array_equal(arrays[name],proxy[name]) for name in
                   ("planned_q_float64","planned_qd_float64","planned_qdd_float64")):
                return False
            controls=np.empty((INTERVALS,7),np.float32)
            retained=arrays["primary_pin_state_float64"]
            for knot in range(INTERVALS):
                _guard(deadline,monotonic);dense=knot*DENSE_SUBSTEPS
                q,qd=retained[dense,:7],retained[dense,7:]
                acceleration=proxy["planned_qdd_float64"][knot] \
                    +100.*(proxy["planned_q_float64"][knot]-q) \
                    +20.*(proxy["planned_qd_float64"][knot]-qd)
                controls[knot]=np.asarray(rnea(q,qd,acceleration),np.float32)
            local=initial_counts()
            state,tool=_pin_replay(q0,controls,aba,kinematics,local,deadline,monotonic)
            if (not np.array_equal(controls,arrays["applied_controls_float32"])
                    or not np.array_equal(state,arrays["primary_pin_state_float64"])
                    or not np.array_equal(tool,arrays["primary_pin_tool_float64"])
                    or not np.array_equal(state,arrays["independent_pin_state_float64"])
                    or not np.array_equal(tool,arrays["independent_pin_tool_float64"])):
                return False
            complete+=1
    return complete>=sum(route_paths(i)["npz"].is_file() for i in (0,1))

def active_artifact_map():
    paths={**{f"worker_{k}":v for k,v in worker_paths().items()},
        **{f"route_{i}_{k}":v for i in (0,1) for k,v in route_paths(i).items()},
        "final_json":OUTPUT,"final_npz":OUTPUT.with_suffix(".npz"),
        "manifest":OUTPUT.with_name("p5.manifest.json"),
        "partial_pointer":OUTPUT.with_name("p5.partial.latest.json"),
        "final_pointer":OUTPUT.with_name("p5.final.latest.json")}
    paths.update({key+"_candidate":Path(str(path)+".candidate") for key,path in list(paths.items())})
    result={}
    for key,path in paths.items():
        classification="absent"
        if path.is_file():
            try:
                if path.name.endswith(".npz") or ".npz.candidate" in path.name:
                    load_npz(path);classification="complete_npz"
                else:json.loads(path.read_text());classification="complete_json"
                if key.endswith("_candidate"):classification="complete_candidate_non_evidence"
            except Exception:classification="incomplete_candidate_non_evidence" \
                if key.endswith("_candidate") else "malformed_evidence"
        result[key]={"path":str(path),"present":path.is_file(),
            "sha256":sha(path) if path.is_file() else None,
            "size":path.stat().st_size if path.is_file() else None,
            "classification":classification}
    return result

def certify_execution_counts(value,final=False):
    if not isinstance(value,dict) or set(value)!=set(initial_counts()) \
            or any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in value.values()):return False
    maximum=final_counts()
    if any(value[k]>maximum[k] for k in value):return False
    if any(value[k] for k in ("optimizer_calls","solve_calls","sqp_calls","rng_calls","task_construction_calls")):
        return False
    if value["pin_model_contexts"]>min(value["prerequisite_artifact_loads"],1):return False
    if value["prerequisite_independent_recertifications"]>1:return False
    if value["independent_pin_replays"]>value["primary_pin_rollouts"]:return False
    if value["worker_subprocess_attempts"] and not (value["prerequisite_artifact_loads"] in (1,2)
            and value["pin_model_contexts"]==1 and value["primary_pin_rollouts"]
            ==value["independent_pin_replays"]==2):return False
    if value["worker_subprocess_successes"]>value["worker_subprocess_attempts"]:return False
    if value["primary_pin_rollouts"]>2 or value["independent_pin_replays"]>2:return False
    if value["rnea_calls"]>min(2,(value["primary_pin_rollouts"]+1))*INTERVALS:return False
    if value["primary_aba_calls"]>min(2,value["primary_pin_rollouts"]+1)*INTERVALS*DENSE_SUBSTEPS:return False
    if value["primary_fk_calls"]>min(2,value["primary_pin_rollouts"]+1)*DENSE_SAMPLES:return False
    if value["independent_aba_calls"]>min(2,value["independent_pin_replays"]+1)*INTERVALS*DENSE_SUBSTEPS:return False
    if value["independent_fk_calls"]>min(2,value["independent_pin_replays"]+1)*DENSE_SAMPLES:return False
    if value["prerequisite_independent_recertifications"]>value["worker_subprocess_successes"]:return False
    if value["owned_pin_replays"]>2 or value["owned_rnea_calls"]>2*INTERVALS \
            or value["owned_aba_calls"]>2*INTERVALS*DENSE_SUBSTEPS \
            or value["owned_fk_calls"]>2*DENSE_SAMPLES:return False
    if any(value[k]>1 for k in ("task_endpoint_fk_calls","direction_fk_calls",
            "owned_task_endpoint_fk_calls","owned_direction_fk_calls")):return False
    worker_keys=("b1_constructors","b16_constructors","b1_sim_forward_calls",
        "b16_sim_forward_calls","b1_tool_position_calls","b16_tool_position_calls")
    if value["worker_subprocess_attempts"]==0 and any(value[k] for k in worker_keys):return False
    if value["worker_subprocess_successes"]==1 and any(value[k]!=maximum[k] for k in worker_keys):return False
    if final:return value==maximum
    return True

def owned_semantic_recert(prerequisite_authentication,prerequisite,final_arrays,
    worker_summary,worker_arrays,model,kinematics,rnea,aba,tool_velocity,
    deadline,monotonic,counts=None):
    """Own all scientific evidence by reopening every retained side artifact."""
    if not exact_arrays(final_arrays,FINAL_ARRAY_SPECS):return {"passes":False}
    from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite
    fresh_auth,fresh_prerequisite=authenticate_cpu_prerequisite()
    _guard(deadline,monotonic)
    if counts is not None:
        counts["prerequisite_artifact_loads"]+=1
        counts["prerequisite_independent_recertifications"]+=1
    if prerequisite_authentication!=fresh_auth:return {"passes":False}
    required={"public_x0_float32":fresh_prerequisite["public_x0_float32"][0],
        "accepted_task_reference_float32":fresh_prerequisite["accepted_task_reference_float32"][0],
        "quarantined_q8_float64":fresh_prerequisite["quarantined_q8_float64"][0],
        "q8_tool_float64":fresh_prerequisite["q8_tool_float64"][0],
        "default_side_int8":fresh_prerequisite["public_default_side_int8"][0],
        "joint_lower_float64":fresh_prerequisite["joint_lower_float64"],
        "joint_upper_float64":fresh_prerequisite["joint_upper_float64"],
        "velocity_limit_float64":fresh_prerequisite["velocity_limit_float64"],
        "effort_limit_float64":fresh_prerequisite["effort_limit_float64"]}
    if any(not np.array_equal(final_arrays[k],v) for k,v in required.items()):return {"passes":False}
    q0=final_arrays["public_x0_float32"][:7].astype(np.float64)
    _guard(deadline,monotonic);start_tool=kinematics(q0)[0]
    fresh_direction=perturbation_direction(q0,final_arrays["quarantined_q8_float64"],
        start_tool,final_arrays["q8_tool_float64"],kinematics,int(final_arrays["default_side_int8"]))
    if counts is not None:
        counts["owned_task_endpoint_fk_calls"]+=1;counts["owned_direction_fk_calls"]+=1
    route_records=[]
    for index,(route,_profile,_amplitude) in enumerate(PILOT_ROUTES):
        paths=route_paths(index)
        summary=json.loads(paths["json"].read_text())
        with np.load(paths["npz"],allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
        if not certify_route_artifact(summary,arrays,index,route):return {"passes":False}
        qgoal=final_arrays["quarantined_q8_float64"];direction=arrays["direction_float64"]
        if not np.array_equal(direction,fresh_direction):return {"passes":False}
        proxy=planned_proxy(q0,qgoal,direction,route)
        if not all(np.array_equal(arrays[k],proxy[k]) for k in
                ("planned_q_float64","planned_qd_float64","planned_qdd_float64")):
            return {"passes":False}
        controls=np.empty((INTERVALS,7),np.float32);state=arrays["primary_pin_state_float64"]
        for knot in range(INTERVALS):
            _guard(deadline,monotonic);dense=knot*DENSE_SUBSTEPS;q,qd=state[dense,:7],state[dense,7:]
            acceleration=proxy["planned_qdd_float64"][knot]+100.*(proxy["planned_q_float64"][knot]-q) \
                +20.*(proxy["planned_qd_float64"][knot]-qd)
            controls[knot]=np.asarray(rnea(q,qd,acceleration),np.float32)
            if counts is not None:counts["owned_rnea_calls"]+=1
        local=initial_counts()
        replay_state,replay_tool=_pin_replay(q0,controls,aba,kinematics,local,deadline,monotonic)
        if counts is not None:
            counts["owned_aba_calls"]+=local["independent_aba_calls"]
            counts["owned_fk_calls"]+=local["independent_fk_calls"]
            counts["owned_pin_replays"]+=1
        if (not np.array_equal(controls,arrays["applied_controls_float32"])
                or not np.array_equal(replay_state,arrays["primary_pin_state_float64"])
                or not np.array_equal(replay_tool,arrays["primary_pin_tool_float64"])
                or not np.array_equal(replay_state,arrays["independent_pin_state_float64"])
                or not np.array_equal(replay_tool,arrays["independent_pin_tool_float64"])):
            return {"passes":False}
        route_records.append((summary,arrays))
    geometry=geometry_from_lane_zero({"pin_dense_tool_float64":route_records[0][1]["primary_pin_tool_float64"]},
        {"pin_dense_tool_float64":route_records[1][1]["primary_pin_tool_float64"]},
        final_arrays["q8_tool_float64"],int(final_arrays["default_side_int8"]))
    if (not geometry["passes"] or not np.array_equal(final_arrays["pillar_float64"],geometry["pillar_float64"])
            or not np.array_equal(final_arrays["short_unit_float64"],geometry["short_unit_float64"])
            or not np.array_equal(final_arrays["reference_float32"],
                np.tile(np.asarray(geometry["reference_float32"],np.float32),KNOTS))):return {"passes":False}
    paths=worker_paths()
    with np.load(paths["input"],allow_pickle=False) as archive:worker_inputs={k:archive[k] for k in archive.files}
    request=json.loads(paths["request"].read_text())
    expected_request={"protocol":WORKER_PROTOCOL,"input_path":str(paths["input"]),
        "input_sha256":sha(paths["input"]),"output_path":str(paths["json"]),"extension":EXTENSION}
    if request!=expected_request or not certify_worker_output(worker_summary,worker_arrays,worker_inputs)["passes"]:
        return {"passes":False}
    expected_controls=np.stack([route_records[lane%2][1]["applied_controls_float32"] for lane in range(LANES)])
    expected_seeds=np.stack([pack_seed(
        route_records[lane%2][1]["primary_pin_state_float64"][::DENSE_SUBSTEPS,:7],
        route_records[lane%2][1]["primary_pin_state_float64"][::DENSE_SUBSTEPS,7:],
        route_records[lane%2][1]["applied_controls_float32"]) for lane in range(LANES)])
    if (not np.array_equal(worker_inputs["x0_float32"],
            np.repeat(final_arrays["public_x0_float32"][None],LANES,0))
            or not np.array_equal(worker_inputs["controls_float32"],expected_controls)
            or not np.array_equal(worker_inputs["b16_seed_xu_float32"],expected_seeds)
            or not np.array_equal(worker_inputs["b1_seed_xu_float32"],expected_seeds[0])):
        return {"passes":False}
    limits=tuple(final_arrays[k] for k in ("joint_lower_float64","joint_upper_float64",
        "velocity_limit_float64","effort_limit_float64"))
    routes=[]
    for lane,(summary,arrays) in enumerate(route_records):
        route=summary["identity"][2]
        detail=certify_route(arrays["primary_pin_state_float64"],arrays["primary_pin_tool_float64"],
            worker_arrays["b16_dense_state_float32"][lane],worker_arrays["b16_dense_tool_float32"][lane],
            arrays["applied_controls_float32"],final_arrays["q8_tool_float64"],
            final_arrays["pillar_float64"],final_arrays["reference_float32"],*limits,
            lambda q,qd:tool_velocity(model,q,qd),route,int(final_arrays["default_side_int8"]))
        if not certify_route_detail(detail):return {"passes":False}
        routes.append(detail)
    pair=certify_pair(*routes)
    gates={"prerequisite":True,"route_artifacts":True,"proxy":True,"pin_replay":True,
        "geometry":True,"worker":True,"routes":all(row["passes"] for row in routes),
        "pair":certify_pair_detail(pair)}
    return {"routes":routes,"pair":pair,"geometry":json_safe(geometry),
        "route_artifacts":[row[0] for row in route_records],
        "worker_artifact":{"request_path":str(paths["request"]),"request_sha256":sha(paths["request"]),
            "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
            "json_path":str(paths["json"]),"json_sha256":sha(paths["json"]),
            "npz_path":str(paths["npz"]),"npz_sha256":sha(paths["npz"])},
        "gates":gates,"passes":bool(all(gates.values()))}

def certify_owned_detail(value):
    return bool(isinstance(value,dict) and set(value)=={"routes","pair","geometry",
        "route_artifacts","worker_artifact","gates","passes"}
        and len(value["routes"])==2 and all(certify_route_detail(row) for row in value["routes"])
        and certify_pair_detail(value["pair"]) and isinstance(value["geometry"],dict)
        and value["geometry"].get("passes") is True and len(value["route_artifacts"])==2
        and set(value["worker_artifact"])=={"request_path","request_sha256","input_path",
            "input_sha256","json_path","json_sha256","npz_path","npz_sha256"}
        and set(value["gates"])=={"prerequisite","route_artifacts","proxy","pin_replay",
            "geometry","worker","routes","pair"}
        and all(item is True for item in value["gates"].values()) and value["passes"] is True)

def checkpoint_path(generation):return OUTPUT.with_name(f"p5.partial.gen{generation:02d}.json")
def publish_checkpoint(generation,stage,counts,provenance,detail=None):
    doc={"protocol":PROTOCOL,"generation":generation,"stage":stage,"incomplete":True,
        "counts":dict(counts),"provenance":provenance,"detail":detail or {}}
    atomic_json(checkpoint_path(generation),doc)
    pointer=OUTPUT.with_name("p5.partial.latest.json")
    atomic_pointer(pointer,{"protocol":PROTOCOL,"generation":generation,"path":str(checkpoint_path(generation)),
        "sha256":sha(checkpoint_path(generation)),"incomplete":True})
    return doc

def certify_checkpoint_document(value):
    if not isinstance(value,dict) or set(value)!={"protocol","generation","stage","incomplete",
            "counts","provenance","detail"} or value["protocol"]!=PROTOCOL \
            or value["incomplete"] is not True or not certify_execution_counts(value["counts"]):
        return False
    generation=value["generation"];stage=value["stage"]
    expected={0:"started",1:"prerequisite_authenticated",2:"pin_routes_constructed",
        3:"owned_semantic_certified",4:"honest_end"}
    if generation not in expected or stage!=expected[generation]:return False
    detail=value["detail"]
    if generation==0:return detail=={} and value["counts"]==initial_counts() and certify_provenance(value["provenance"],False)
    if generation==1:
        counts=initial_counts();counts["prerequisite_artifact_loads"]=1
        return set(detail)=={"authentication"} and isinstance(detail["authentication"],dict) \
            and value["counts"]==counts and certify_provenance(value["provenance"],False)
    if generation==2:return set(detail)=={"geometry","route_artifacts"} \
        and len(detail["route_artifacts"])==2 and value["counts"]==pin_routes_counts() \
        and certify_provenance(value["provenance"],False)
    if generation==3:return set(detail)=={"routes","pair","route_artifacts","worker_artifact"} \
        and all(certify_route_detail(row) for row in detail["routes"]) \
        and certify_pair_detail(detail["pair"]) and value["counts"]==final_counts() \
        and certify_provenance(value["provenance"],False)
    return set(detail)=={"elapsed_s"} and np.isfinite(detail["elapsed_s"]) \
        and value["counts"]==final_counts() and certify_provenance(value["provenance"],True)

def certify_final_document(value):
    keys={"protocol","incomplete","overall_pass","identity","routes","pair","counts",
        "worker","provenance","elapsed_s","oracle_evidence","benchmark_evidence","sqp_evidence",
        "prerequisite_authentication","owned_semantic_recertification","route_artifacts",
        "worker_artifact","npz_path","npz_sha256","array_names","array_hashes"}
    return bool(set(value)==keys and value["protocol"]==PROTOCOL and value["incomplete"] is False
        and value["overall_pass"] is True and value["identity"]==list(PILOT_IDENTITY)
        and [row.get("route") for row in value["routes"]]==["short","long"]
        and all(certify_route_detail(row) for row in value["routes"])
        and certify_pair_detail(value["pair"]) and certify_execution_counts(value["counts"],True)
        and value["worker"].get("certificate",{}).get("passes") is True
        and isinstance(value["prerequisite_authentication"],dict)
        and certify_owned_detail(value["owned_semantic_recertification"])
        and value["routes"]==value["owned_semantic_recertification"]["routes"]
        and value["pair"]==value["owned_semantic_recertification"]["pair"]
        and value["route_artifacts"]==value["owned_semantic_recertification"]["route_artifacts"]
        and value["worker_artifact"]==value["owned_semantic_recertification"]["worker_artifact"]
        and value["npz_path"]==str(OUTPUT.with_suffix(".npz"))
        and value["npz_sha256"]==sha(OUTPUT.with_suffix(".npz"))
        and value["array_names"]==sorted(FINAL_ARRAY_SPECS)
        and exact_arrays(load_npz(OUTPUT.with_suffix(".npz")),FINAL_ARRAY_SPECS)
        and value["array_hashes"]=={k:array_hash(v) for k,v in load_npz(OUTPUT.with_suffix(".npz")).items()}
        and certify_provenance(value["provenance"],True)
        and np.isfinite(value["elapsed_s"]) and 0<=value["elapsed_s"]<=CAMPAIGN_WALL_LIMIT_S
        and value["oracle_evidence"] is value["benchmark_evidence"] is value["sqp_evidence"] is False)

def load_npz(path):
    with np.load(path,allow_pickle=False) as archive:return {k:archive[k] for k in archive.files}

def certify_rejection(value):
    keys={"protocol","stage","incomplete","error_type","error_message","counts",
        "provenance","trigger_elapsed_s","finish_elapsed_s","campaign_wall_limit_s",
        "completed","pending","oracle_evidence",
        "benchmark_evidence","sqp_evidence","active_artifacts","worker_execution",
        "prerequisite_authentication","route_artifacts"}
    return bool(set(value)==keys and value["protocol"]==PROTOCOL and value["incomplete"] is True
        and value["stage"] in {"pilot_failed","runtime_watchdog_rejected","final_certification_failed"}
        and isinstance(value["error_type"],str) and isinstance(value["error_message"],str)
        and set(value["counts"])==set(initial_counts())
        and all(isinstance(x,int) and not isinstance(x,bool) and x>=0 for x in value["counts"].values())
        and value["counts"]["optimizer_calls"]==value["counts"]["solve_calls"] \
            ==value["counts"]["sqp_calls"]==value["counts"]["rng_calls"] \
            ==value["counts"]["task_construction_calls"]==0
        and certify_execution_counts(value["counts"])
        and value["active_artifacts"]==active_artifact_map()
        and certify_worker_execution(value["worker_execution"],value["counts"])
        and certify_worker_failure_binding(value)
        and ((value["counts"]["prerequisite_artifact_loads"]==0
              and value["prerequisite_authentication"] is None)
             or (value["counts"]["prerequisite_artifact_loads"]>=1
              and isinstance(value["prerequisite_authentication"],dict)))
        and isinstance(value["route_artifacts"],list)
        and certify_route_artifact_prefix(value["route_artifacts"])
        and len(value["route_artifacts"])<=value["counts"]["independent_pin_replays"]
        and certify_provenance(value["provenance"],True)
        and value["campaign_wall_limit_s"]==CAMPAIGN_WALL_LIMIT_S
        and np.isfinite(value["trigger_elapsed_s"])
        and 0<=value["trigger_elapsed_s"]<=CAMPAIGN_WALL_LIMIT_S
        and np.isfinite(value["finish_elapsed_s"])
        and value["finish_elapsed_s"]>=value["trigger_elapsed_s"]
        and value["completed"] in (0,1,2) and value["pending"]==2-value["completed"]
        and value["oracle_evidence"] is value["benchmark_evidence"] is value["sqp_evidence"] is False)

def certify_worker_execution(value,counts):
    if value is None:return counts["worker_subprocess_attempts"]==0
    if not isinstance(value,dict) or set(value)!={"status","internal_counts_known",
            "counts","lower_bounds","returncode","killed"}:return False
    status=value["status"]
    worker_keys=("b1_constructors","b16_constructors","b1_sim_forward_calls",
        "b16_sim_forward_calls","b1_tool_position_calls","b16_tool_position_calls",
        "solve_calls","sqp_calls")
    if status=="known_complete":return bool(counts["worker_subprocess_attempts"]==1
        and counts["worker_subprocess_successes"]==1 and value["internal_counts_known"] is True
        and value["counts"]=={k:final_counts()[k] for k in ("b1_constructors","b16_constructors",
            "b1_sim_forward_calls","b16_sim_forward_calls","b1_tool_position_calls",
            "b16_tool_position_calls","solve_calls","sqp_calls")}
        and all(counts[k]==value["counts"][k] for k in worker_keys)
        and value["lower_bounds"]==value["counts"]
        and value["returncode"]==0 and value["killed"] is False)
    if status=="known_partial":return bool(counts["worker_subprocess_attempts"]==1
        and counts["worker_subprocess_successes"]==0 and value["internal_counts_known"] is True
        and certify_worker_partial_counts(value["counts"])
        and all(counts[k]==value["counts"][k] for k in worker_keys)
        and value["lower_bounds"]==value["counts"]
        and value["returncode"]!=0 and value["killed"] is False)
    return bool(counts["worker_subprocess_attempts"]==1 and counts["worker_subprocess_successes"]==0
        and all(counts[k]==0 for k in worker_keys)
        and status in {"killed_internal_unknown","abnormal_exit_internal_unknown",
        "exit0_internal_unknown"} and value["internal_counts_known"] is False
        and value["counts"] is None and value["lower_bounds"]=={k:0 for k in worker_keys}
        and isinstance(value["returncode"],int)
        and value["killed"] is (status=="killed_internal_unknown"))

def certify_worker_failure_binding(rejection):
    execution=rejection["worker_execution"]
    if execution is None:return rejection["counts"]["worker_subprocess_attempts"]==0
    paths=worker_paths();status=execution["status"]
    if status=="known_partial":
        if not paths["rejected"].is_file():return False
        from gato_tiago.circular_portfolio_p5_worker import certify_rejection as worker_recert
        detail=json.loads(paths["rejected"].read_text())
        worker_keys=set(detail["counts"])
        expected_error="TimeoutError" if detail["error_type"]=="TimeoutError" else "RuntimeError"
        return bool(worker_recert(detail) and execution["counts"]==detail["counts"]
            and execution["lower_bounds"]==detail["counts"]
            and all(rejection["counts"][k]==detail["counts"][k] for k in worker_keys)
            and rejection["error_type"]==expected_error
            and rejection["error_message"]==("P5 worker wall limit" if expected_error=="TimeoutError"
                else f"P5 worker exited {execution['returncode']}")
            and rejection["active_artifacts"]["worker_rejected"]["sha256"]==sha(paths["rejected"])
            and execution["returncode"]!=0)
    if status=="known_complete":return paths["json"].is_file() and paths["npz"].is_file()
    return bool(not paths["rejected"].exists() and execution["counts"] is None
        and all(x==0 for x in execution["lower_bounds"].values()))

def execute(output=OUTPUT,authorization=None,monotonic=time.monotonic): # pragma: no cover
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P5 runner blocked")
    if Path(output).resolve()!=OUTPUT:raise RuntimeError("wrong P5 output")
    if OUTPUT.parent.exists():raise FileExistsError("P5 namespace exists")
    OUTPUT.parent.mkdir(parents=True);started=monotonic();deadline=started+CAMPAIGN_WALL_LIMIT_S
    counts=initial_counts();provenance=start_provenance();completed=0
    authentication=None;route_artifacts=[];worker_execution=None
    try:
        publish_checkpoint(0,"started",counts,provenance)
        from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite,_production_pin_context
        authentication,prerequisite=authenticate_cpu_prerequisite();counts["prerequisite_artifact_loads"]+=1
        _guard(deadline,monotonic)
        publish_checkpoint(1,"prerequisite_authenticated",counts,provenance,
            {"authentication":authentication})
        model,kinematics,rnea,_rnea_derivatives,aba,tool_velocity,_dense=_production_pin_context()
        counts["pin_model_contexts"]+=1
        _guard(deadline,monotonic)
        q0=prerequisite["public_x0_float32"][0,:7].astype(np.float64)
        qgoal=prerequisite["quarantined_q8_float64"][0]
        goal=prerequisite["q8_tool_float64"][0]
        default_side=int(prerequisite["public_default_side_int8"][0])
        _guard(deadline,monotonic);start_tool=kinematics(q0)[0];counts["task_endpoint_fk_calls"]+=1
        direction=perturbation_direction(q0,qgoal,start_tool,goal,kinematics,default_side)
        counts["direction_fk_calls"]+=1
        rows=[]
        for route_index,(route,_index,_amplitude) in enumerate(PILOT_ROUTES):
            _guard(deadline,monotonic)
            proxy=planned_proxy(q0,qgoal,direction,route)
            if not certify_proxy(proxy,q0,qgoal,direction,route)["passes"]:raise RuntimeError("P5 proxy failed")
            state,tool,controls=_pin_rollout(proxy,q0,rnea,aba,kinematics,counts,"primary",deadline,monotonic)
            replay_state,replay_tool=_pin_replay(q0,controls,aba,kinematics,counts,deadline,monotonic)
            if not np.array_equal(state,replay_state) or not np.array_equal(tool,replay_tool):
                raise RuntimeError("P5 independent Pin replay mismatch")
            arrays=route_arrays(direction,proxy,state,tool,replay_state,replay_tool,controls)
            paths=route_paths(route_index);atomic_npz(paths["npz"],arrays)
            summary=route_summary(route_index,route,arrays)
            if not certify_route_artifact(summary,arrays,route_index,route):
                raise RuntimeError("P5 route artifact failed")
            atomic_json(paths["json"],summary);route_artifacts.append(summary)
            rows.append({"route":route,"proxy":proxy,"state":state,"tool":tool,"controls":controls})
        geometry=geometry_from_lane_zero({"pin_dense_tool_float64":rows[0]["tool"]},
            {"pin_dense_tool_float64":rows[1]["tool"]},goal,default_side)
        if not geometry["passes"]:raise RuntimeError("P5 task geometry failed")
        _guard(deadline,monotonic)
        reference=np.tile(np.asarray(geometry["reference_float32"],np.float32),KNOTS)
        publish_checkpoint(2,"pin_routes_constructed",counts,provenance,
            {"geometry":json_safe(geometry),"route_artifacts":route_artifacts})
        inputs={"x0_float32":np.repeat(np.r_[q0,np.zeros(7)].astype(np.float32)[None],LANES,0),
            "controls_float32":np.stack([rows[lane%2]["controls"] for lane in range(LANES)]),
            "b16_seed_xu_float32":np.stack([pack_seed(rows[lane%2]["state"][::DENSE_SUBSTEPS,:7],
                rows[lane%2]["state"][::DENSE_SUBSTEPS,7:],rows[lane%2]["controls"])
                for lane in range(LANES)])}
        inputs["b1_seed_xu_float32"]=inputs["b16_seed_xu_float32"][0].copy()
        if not worker_input_schema(inputs):raise RuntimeError("P5 worker input failed")
        paths=worker_paths();atomic_npz(paths["input"],inputs)
        request={"protocol":WORKER_PROTOCOL,"input_path":str(paths["input"]),
            "input_sha256":sha(paths["input"]),"output_path":str(paths["json"]),"extension":EXTENSION}
        atomic_json(paths["request"],request);counts["worker_subprocess_attempts"]+=1
        remaining=CAMPAIGN_WALL_LIMIT_S-(monotonic()-started)
        if remaining<=0:raise TimeoutError("P5 campaign wall limit before worker")
        command=["python","-B","-m","gato_tiago.circular_portfolio_p5_worker","--execute",
            "--request",str(paths["request"]),"--output",str(paths["json"])]
        try:result=subprocess.run(command,cwd=AUTHORIZED_CWD,env=os.environ.copy(),timeout=min(300.,remaining))
        except subprocess.TimeoutExpired as source:
            error=TimeoutError("P5 worker wall limit")
            error.worker_execution={"status":"killed_internal_unknown","internal_counts_known":False,
                "counts":None,"lower_bounds":{k:0 for k in ("b1_constructors","b16_constructors",
                    "b1_sim_forward_calls","b16_sim_forward_calls","b1_tool_position_calls",
                    "b16_tool_position_calls","solve_calls","sqp_calls")},
                "returncode":-9,"killed":True};raise error from source
        if result.returncode:
            detail=json.loads(paths["rejected"].read_text()) if paths["rejected"].is_file() else None
            if detail is not None:
                from gato_tiago.circular_portfolio_p5_worker import certify_rejection as worker_recert
                if not worker_recert(detail):raise RuntimeError("P5 worker rejection invalid")
                for key,value in detail["counts"].items():counts[key]=value
                status={"status":"known_partial","internal_counts_known":True,
                    "counts":detail["counts"],"lower_bounds":detail["counts"],
                    "returncode":result.returncode,"killed":False}
            else:status={"status":"abnormal_exit_internal_unknown","internal_counts_known":False,
                "counts":None,"lower_bounds":{k:0 for k in ("b1_constructors","b16_constructors",
                    "b1_sim_forward_calls","b16_sim_forward_calls","b1_tool_position_calls",
                    "b16_tool_position_calls","solve_calls","sqp_calls")},
                "returncode":result.returncode,"killed":False}
            error=(TimeoutError("P5 worker wall limit") if detail is not None
                and detail["error_type"]=="TimeoutError"
                else RuntimeError(f"P5 worker exited {result.returncode}"))
            error.worker_execution=status;raise error
        try:
            summary=json.loads(paths["json"].read_text())
            with np.load(paths["npz"],allow_pickle=False) as archive:
                worker_arrays={k:archive[k] for k in archive.files}
            worker_cert=certify_worker_output(summary,worker_arrays,inputs)
            if not worker_cert["passes"]:raise RuntimeError("P5 worker certificate failed")
        except Exception as source:
            error=RuntimeError("P5 exit-zero worker output unavailable or invalid")
            error.worker_execution={"status":"exit0_internal_unknown","internal_counts_known":False,
                "counts":None,"lower_bounds":{k:0 for k in ("b1_constructors","b16_constructors",
                    "b1_sim_forward_calls","b16_sim_forward_calls","b1_tool_position_calls",
                    "b16_tool_position_calls","solve_calls","sqp_calls")},
                "returncode":0,"killed":False};raise error from source
        counts.update({"worker_subprocess_successes":1,"b1_constructors":1,"b16_constructors":1,
            "b1_sim_forward_calls":INTERVALS*DENSE_SUBSTEPS,
            "b16_sim_forward_calls":INTERVALS*DENSE_SUBSTEPS,
            "b1_tool_position_calls":DENSE_SAMPLES,"b16_tool_position_calls":DENSE_SAMPLES})
        worker_execution={"status":"known_complete","internal_counts_known":True,
            "counts":summary["counts"],"lower_bounds":summary["counts"],
            "returncode":0,"killed":False}
        final_arrays={"public_x0_float32":prerequisite["public_x0_float32"][0],
            "accepted_task_reference_float32":prerequisite["accepted_task_reference_float32"][0],
            "quarantined_q8_float64":prerequisite["quarantined_q8_float64"][0],
            "q8_tool_float64":prerequisite["q8_tool_float64"][0],
            "default_side_int8":np.asarray(default_side,np.int8),
            "joint_lower_float64":prerequisite["joint_lower_float64"],
            "joint_upper_float64":prerequisite["joint_upper_float64"],
            "velocity_limit_float64":prerequisite["velocity_limit_float64"],
            "effort_limit_float64":prerequisite["effort_limit_float64"],
            "reference_float32":reference,"pillar_float64":np.asarray(geometry["pillar_float64"]),
            "short_unit_float64":np.asarray(geometry["short_unit_float64"])}
        if not exact_arrays(final_arrays,FINAL_ARRAY_SPECS):raise RuntimeError("P5 final arrays invalid")
        final_npz=OUTPUT.with_suffix(".npz");atomic_npz(final_npz,final_arrays)
        owned=owned_semantic_recert(authentication,prerequisite,final_arrays,summary,worker_arrays,
            model,kinematics,rnea,aba,tool_velocity,deadline,monotonic,counts)
        if not owned["passes"]:raise RuntimeError("P5 owned semantic recert failed")
        _guard(deadline,monotonic)
        routes=owned["routes"];pair=owned["pair"];completed=2
        publish_checkpoint(3,"owned_semantic_certified",counts,provenance,
            {"routes":routes,"pair":pair,"route_artifacts":route_artifacts,
                "worker_artifact":owned["worker_artifact"]})
        finish_provenance(provenance);elapsed=monotonic()-started
        _guard(deadline,monotonic)
        publish_checkpoint(4,"honest_end",counts,provenance,{"elapsed_s":elapsed})
        document={"protocol":PROTOCOL,"incomplete":False,"overall_pass":True,
            "identity":list(PILOT_IDENTITY),"routes":routes,"pair":pair,"counts":counts,
            "worker":summary,"provenance":provenance,"elapsed_s":elapsed,
            "prerequisite_authentication":authentication,
            "owned_semantic_recertification":owned,"route_artifacts":route_artifacts,
            "worker_artifact":owned["worker_artifact"],"npz_path":str(final_npz),
            "npz_sha256":sha(final_npz),"array_names":sorted(final_arrays),
            "array_hashes":{k:array_hash(final_arrays[k]) for k in sorted(final_arrays)},
            "oracle_evidence":False,"benchmark_evidence":False,"sqp_evidence":False}
        if not certify_final_document(document):raise RuntimeError("P5 final certificate failed")
        _guard(deadline,monotonic)
        atomic_json(OUTPUT,document)
        side=[{"path":str(checkpoint_path(i)),"sha256":sha(checkpoint_path(i))} for i in range(5)]
        side += [{"path":str(route_paths(i)[k]),"sha256":sha(route_paths(i)[k])}
                 for i in (0,1) for k in ("json","npz")]
        side += [{"path":str(paths[k]),"sha256":sha(paths[k])} for k in
                 ("request","input","json","npz")]
        manifest={"protocol":PROTOCOL,"json_path":str(OUTPUT),"json_sha256":sha(OUTPUT),
            "npz_path":str(final_npz),"npz_sha256":sha(final_npz),"side_artifacts":side,
            "side_artifact_count":len(side),"overall_pass":True}
        manifest_path=OUTPUT.with_name("p5.manifest.json");atomic_json(manifest_path,manifest)
        pointer_document={"protocol":PROTOCOL,"json_path":str(OUTPUT),
            "json_sha256":sha(OUTPUT),"npz_path":str(final_npz),"npz_sha256":sha(final_npz),
            "manifest_path":str(manifest_path),"manifest_sha256":sha(manifest_path),
            "overall_pass":True}
        _publish_final_pointer(pointer_document,started,deadline,monotonic)
        return document
    except Exception as error:
        if provenance.get("head_end") is None:finish_provenance(provenance)
        elapsed=monotonic()-started;trigger_elapsed=min(elapsed,CAMPAIGN_WALL_LIMIT_S)
        stage="runtime_watchdog_rejected" if isinstance(error,TimeoutError) else \
            "final_certification_failed" if completed==2 else "pilot_failed"
        worker_execution=getattr(error,"worker_execution",worker_execution)
        rejected={"protocol":PROTOCOL,"stage":stage,"incomplete":True,
            "error_type":type(error).__name__,"error_message":str(error),"counts":counts,
            "provenance":provenance,"trigger_elapsed_s":trigger_elapsed,
            "finish_elapsed_s":elapsed,"campaign_wall_limit_s":CAMPAIGN_WALL_LIMIT_S,
            "completed":completed,"pending":2-completed,
            "active_artifacts":active_artifact_map(),"worker_execution":worker_execution,
            "prerequisite_authentication":authentication,"route_artifacts":route_artifacts,
            "oracle_evidence":False,"benchmark_evidence":False,"sqp_evidence":False}
        if not certify_rejection(rejected):raise RuntimeError("P5 rejection certificate failed") from error
        atomic_json(OUTPUT.with_name("p5.rejected.json"),rejected);raise

def recertify_retained_failure(path=OUTPUT.with_name("p5.rejected.json"),monotonic=time.monotonic):
    try:
        value=json.loads(Path(path).read_text())
        prerequisite=None
        if value["prerequisite_authentication"] is not None:
            from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite
            authentication,prerequisite=authenticate_cpu_prerequisite()
            if value["prerequisite_authentication"]!=authentication:return {"passes":False}
        elif any(item["present"] for key,item in value["active_artifacts"].items()
                 if key.startswith(("route_","worker_","final_npz"))):
            return {"passes":False}
        if prerequisite is not None and not certify_active_worker_artifacts(prerequisite,value["counts"]):
            return {"passes":False}
        if prerequisite is not None and not certify_failure_side_artifacts(prerequisite):
            return {"passes":False}
        if prerequisite is not None and any(route_paths(i)["npz"].is_file()
                or Path(str(route_paths(i)["npz"])+".candidate").is_file() for i in (0,1)):
            from gato_tiago.circular_portfolio_runner import _production_pin_context
            model,kinematics,rnea,_rd,aba,_tool_velocity,_dense=_production_pin_context()
            if not recertify_failure_route_science(prerequisite,kinematics,rnea,aba,
                    monotonic()+CAMPAIGN_WALL_LIMIT_S,monotonic):return {"passes":False}
        publication=value["active_artifacts"]
        if publication["final_json"]["present"]:
            if publication["final_json"]["classification"]!="complete_json" \
                    or not certify_final_document(json.loads(OUTPUT.read_text())):return {"passes":False}
        manifest_path=OUTPUT.with_name("p5.manifest.json")
        if publication["manifest"]["present"]:
            if publication["manifest"]["classification"]!="complete_json":return {"passes":False}
            manifest=json.loads(manifest_path.read_text())
            side=[{"path":str(item),"sha256":sha(item)} for item in
                  sorted(OUTPUT.parent.glob("p5.partial.gen*.json"))]
            side += [{"path":str(route_paths(i)[k]),"sha256":sha(route_paths(i)[k])}
                     for i in (0,1) for k in ("json","npz")]
            paths=worker_paths();side += [{"path":str(paths[k]),"sha256":sha(paths[k])}
                                          for k in ("request","input","json","npz")]
            if (manifest.get("protocol")!=PROTOCOL or manifest.get("json_path")!=str(OUTPUT)
                    or manifest.get("json_sha256")!=sha(OUTPUT)
                    or manifest.get("npz_path")!=str(OUTPUT.with_suffix(".npz"))
                    or manifest.get("npz_sha256")!=sha(OUTPUT.with_suffix(".npz"))
                    or manifest.get("side_artifacts")!=side
                    or manifest.get("side_artifact_count")!=len(side)
                    or manifest.get("overall_pass") is not True):return {"passes":False}
        checkpoints=sorted(OUTPUT.parent.glob("p5.partial.gen*.json"))
        late_owned=None
        if checkpoint_path(3).is_file():
            from gato_tiago.circular_portfolio_runner import _production_pin_context
            final_arrays=load_npz(OUTPUT.with_suffix(".npz"));paths=worker_paths()
            worker_summary=json.loads(paths["json"].read_text());worker_arrays=load_npz(paths["npz"])
            model,kinematics,rnea,_rd,aba,tool_velocity,_dense=_production_pin_context()
            late_owned=owned_semantic_recert(authentication,prerequisite,final_arrays,
                worker_summary,worker_arrays,model,kinematics,rnea,aba,tool_velocity,
                monotonic()+CAMPAIGN_WALL_LIMIT_S,monotonic,None)
            if not certify_owned_detail(late_owned):return {"passes":False}
        chain=[]
        for generation,item in enumerate(checkpoints):
            document=json.loads(item.read_text())
            if not certify_checkpoint_document(document):return {"passes":False}
            if generation==1 and document["detail"]["authentication"]!=authentication:
                return {"passes":False}
            if generation==2 and document["detail"]["route_artifacts"]!=value["route_artifacts"]:
                return {"passes":False}
            if generation==2:
                arrays=[load_npz(route_paths(i)["npz"]) for i in (0,1)]
                fresh_geometry=geometry_from_lane_zero(
                    {"pin_dense_tool_float64":arrays[0]["primary_pin_tool_float64"]},
                    {"pin_dense_tool_float64":arrays[1]["primary_pin_tool_float64"]},
                    prerequisite["q8_tool_float64"][0],
                    int(prerequisite["public_default_side_int8"][0]))
                if document["detail"]["geometry"]!=json_safe(fresh_geometry):return {"passes":False}
            if generation==3 and document["detail"]["route_artifacts"]!=value["route_artifacts"]:
                return {"passes":False}
            if generation==3 and document["detail"]!={"routes":late_owned["routes"],
                    "pair":late_owned["pair"],"route_artifacts":late_owned["route_artifacts"],
                    "worker_artifact":late_owned["worker_artifact"]}:return {"passes":False}
            chain.append({"path":str(item),"sha256":sha(item),"stage":document["stage"]})
        latest=OUTPUT.with_name("p5.partial.latest.json")
        pointer=json.loads(latest.read_text()) if latest.is_file() else None
        pointer_gate=bool(chain and pointer=={"protocol":PROTOCOL,"generation":len(chain)-1,
            "path":chain[-1]["path"],"sha256":chain[-1]["sha256"],"incomplete":True})
        return {"checkpoint_chain":chain,"passes":bool(certify_rejection(value) and pointer_gate)}
    except Exception:return {"passes":False}

def recertify_retained_p5(output=OUTPUT,monotonic=time.monotonic): # pragma: no cover
    """Independent public disk recertification; never trusts stored pass booleans."""
    try:
        output=Path(output)
        if output.resolve()!=OUTPUT or not output.is_file():return {"passes":False}
        value=json.loads(output.read_text());final_arrays=load_npz(output.with_suffix(".npz"))
        if not exact_arrays(final_arrays,FINAL_ARRAY_SPECS):return {"passes":False}
        from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite,_production_pin_context
        authentication,prerequisite=authenticate_cpu_prerequisite()
        if value.get("prerequisite_authentication")!=authentication:return {"passes":False}
        if not certify_final_arrays_task(final_arrays,prerequisite):return {"passes":False}
        model,kinematics,rnea,_rd,aba,tool_velocity,_dense=_production_pin_context()
        paths=worker_paths();worker_summary=json.loads(paths["json"].read_text())
        worker_arrays=load_npz(paths["npz"])
        deadline=monotonic()+CAMPAIGN_WALL_LIMIT_S
        owned=owned_semantic_recert(authentication,prerequisite,final_arrays,worker_summary,
            worker_arrays,model,kinematics,rnea,aba,tool_velocity,deadline,monotonic,None)
        if not owned["passes"] or value.get("owned_semantic_recertification")!=owned:
            return {"passes":False}
        if not certify_final_document(value):return {"passes":False}
        manifest_path=OUTPUT.with_name("p5.manifest.json")
        manifest=json.loads(manifest_path.read_text())
        expected_side=[{"path":str(checkpoint_path(i)),"sha256":sha(checkpoint_path(i))}
                       for i in range(5)]
        expected_side += [{"path":str(route_paths(i)[k]),"sha256":sha(route_paths(i)[k])}
                          for i in (0,1) for k in ("json","npz")]
        expected_side += [{"path":str(paths[k]),"sha256":sha(paths[k])}
                          for k in ("request","input","json","npz")]
        expected_manifest={"protocol":PROTOCOL,"json_path":str(OUTPUT),
            "json_sha256":sha(OUTPUT),"npz_path":str(OUTPUT.with_suffix(".npz")),
            "npz_sha256":sha(OUTPUT.with_suffix(".npz")),"side_artifacts":expected_side,
            "side_artifact_count":len(expected_side),"overall_pass":True}
        if manifest!=expected_manifest:return {"passes":False}
        pointer=json.loads(OUTPUT.with_name("p5.final.latest.json").read_text())
        finish=pointer.get("publication_finish_elapsed_s")
        expected_pointer={"protocol":PROTOCOL,"json_path":str(OUTPUT),
            "json_sha256":sha(OUTPUT),"npz_path":str(OUTPUT.with_suffix(".npz")),
            "npz_sha256":sha(OUTPUT.with_suffix(".npz")),"manifest_path":str(manifest_path),
            "manifest_sha256":sha(manifest_path),"publication_finish_elapsed_s":finish,
            "overall_pass":True}
        if (pointer!=expected_pointer or not isinstance(finish,(int,float)) or not np.isfinite(finish)
                or not value["elapsed_s"]<=finish<=CAMPAIGN_WALL_LIMIT_S):return {"passes":False}
        checkpoint_chain=[]
        for generation in range(5):
            path=checkpoint_path(generation);document=json.loads(path.read_text())
            if not certify_checkpoint_document(document):return {"passes":False}
            if generation==1 and document["detail"]["authentication"]!=authentication:
                return {"passes":False}
            if generation==2 and document["detail"]!={"geometry":owned["geometry"],
                    "route_artifacts":owned["route_artifacts"]}:
                return {"passes":False}
            if generation==3 and document["detail"]!={"routes":owned["routes"],
                    "pair":owned["pair"],"route_artifacts":owned["route_artifacts"],
                    "worker_artifact":owned["worker_artifact"]}:return {"passes":False}
            if generation==4 and document["detail"]!={"elapsed_s":value["elapsed_s"]}:
                return {"passes":False}
            checkpoint_chain.append({"path":str(path),"sha256":sha(path),"stage":document["stage"]})
        return {"checkpoint_chain":checkpoint_chain,"owned_semantic_recertification":owned,
            "manifest":True,"pointer":True,"passes":True}
    except Exception:return {"passes":False}

def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    execute(args.output,RUNNER_EXECUTION_AUTHORIZATION)

if __name__=="__main__":main()
