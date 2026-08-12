"""Capability-blocked orchestration and certification boundary for P6."""

from __future__ import annotations

import argparse,hashlib,importlib.metadata,json,os,shlex,subprocess,sys,time
from pathlib import Path

import numpy as np

from gato_tiago.circular_portfolio_p3 import perturbation_direction
from gato_tiago.circular_portfolio_p3_runner import geometry_from_lane_zero
from gato_tiago.circular_portfolio_p6 import (CAMPAIGN_WALL_LIMIT_S,EXTENSION,KNOTS,
    OUTPUT,P5_V1_REJECTED_REPORT,P5_V2_REJECTED_REPORT,PILOT_IDENTITY,PILOT_ROUTES,PROTOCOL,
    SCREEN_RESULTS,WORKER_INPUT_SPECS,authoritative_metrics,exact_arrays,planned_proxy,
    array_hash,feedback_acceleration,worker_paths)
from gato_tiago.circular_portfolio_p6_worker import (certify_output,certify_cuda_diagnostics,
    certify_provenance as certify_worker_provenance,
    certify_rejection as certify_worker_rejection,snapshot as worker_snapshot)
from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite,_production_pin_context
from gato_tiago.circular_portfolio_p5_runner import SOURCE_PATHS as P5_SOURCE_PATHS
from gato_tiago.circular_portfolio_p5_v2_runner import canonical_authentication


RUNNER_EXECUTION_AUTHORIZATION=None
AUTHORIZED_CWD="/workspace/GATO"
AUTHORIZED_ORIG_ARGV=("python","-B","-m","gato_tiago.circular_portfolio_p6_runner",
    "--execute","--output",str(OUTPUT))
THREAD_ENV={"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
    "NUMEXPR_NUM_THREADS":"1"}
SOURCE_PATHS=tuple(dict.fromkeys((*P5_SOURCE_PATHS,
    "tiago_src/gato_tiago/circular_portfolio_p5_v2.py",
    "tiago_src/gato_tiago/circular_portfolio_p5_v2_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p5_v2_worker.py",
    "tiago_src/gato_tiago/circular_portfolio_p6.py",
    "tiago_src/gato_tiago/circular_portfolio_p6_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p6_worker.py")))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    root=Path(__file__).resolve().parents[2]
    run=lambda *args:subprocess.run(["git",*args],cwd=root,check=True,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
    extension=(root/EXTENSION["relative_path"]).resolve()
    return {"head":run("rev-parse","HEAD"),
        "clean":run("status","--porcelain","--untracked-files=no")=="",
        "sources":{name:sha(root/name) for name in SOURCE_PATHS},
        "extension":{"path":str(extension),"sha256":sha(extension),"size":extension.stat().st_size,
            "build_head":EXTENSION["build_head"],"arch":EXTENSION["arch"],
            "attributes":{k:EXTENSION[k] for k in ("KNOT_POINTS","REFERENCE_SIZE",
                "TOOL_POSITION_FRAME","TOOL_POSITION_SIZE")}}}


def atomic_json(path,value,pointer=False):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if candidate.exists() or (path.exists() and not pointer):raise FileExistsError("P6 artifact exists")
    try:
        candidate.write_text(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False))
        candidate.replace(path)
    except Exception:
        if candidate.exists():candidate.unlink()
        raise


def atomic_npz(path,arrays):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError("P6 artifact exists")
    try:
        with candidate.open("xb") as stream:np.savez(stream,**arrays)
        candidate.replace(path)
    except Exception:
        if candidate.exists():candidate.unlink()
        raise


def checkpoint_path(generation):return OUTPUT.with_name(f"p6.partial.gen{generation:02d}.json")
def route_path(index):return OUTPUT.with_name(f"p6.route.{index}.json")
def manifest_path():return OUTPUT.with_name("p6.manifest.json")
def latest_path():return OUTPUT.with_name("p6.partial.latest.json")
def rejection_path():return OUTPUT.with_name("p6.rejected.json")


def expected_artifact_paths():
    paths=[checkpoint_path(i) for i in range(4)]+[latest_path(),OUTPUT,
        manifest_path(),route_path(0),route_path(1),*worker_paths().values()]
    return tuple(dict.fromkeys((*paths,*(Path(str(path)+".candidate") for path in paths))))


def active_artifact_map():
    return {str(path):{"present":path.is_file(),"sha256":sha(path) if path.is_file() else None,
        "size":path.stat().st_size if path.is_file() else None} for path in expected_artifact_paths()}


def certify_active_artifact_map(value):
    try:return value==active_artifact_map()
    except Exception:return False


def namespace_paths():return set(OUTPUT.parent.glob("p6*")) if OUTPUT.parent.is_dir() else set()


def certify_namespace(allowed):
    return namespace_paths()=={Path(path) for path in allowed if Path(path).is_file()}


def classify_candidates():
    result={}
    for path in expected_artifact_paths():
        if not str(path).endswith(".candidate") or not path.is_file():continue
        row={"sha256":sha(path),"size":path.stat().st_size,"classification":None}
        try:
            if ".npz." in path.name:
                with np.load(path,allow_pickle=False) as archive:
                    arrays={k:archive[k] for k in archive.files}
                if set(arrays)==set(WORKER_INPUT_SPECS) and exact_arrays(arrays,WORKER_INPUT_SPECS):
                    row["classification"]="complete_worker_input_non_evidence"
                elif certify_output(arrays):row["classification"]="complete_worker_output_non_evidence"
                else:row["classification"]="invalid_npz_non_evidence"
            else:
                json.loads(path.read_text());row["classification"]="complete_json_non_evidence"
        except Exception:row["classification"]="incomplete_non_evidence"
        result[str(path)]=row
    return result


def certify_worker_execution(value,counts):
    keys={"status","returncode","counts","killed","internal_counts_known"}
    if not isinstance(value,dict) or not keys.issubset(value):return False
    status=value["status"]
    if status=="not_started":
        return set(value)==keys and value=={"status":"not_started","returncode":None,
            "counts":None,"killed":False,"internal_counts_known":False} \
            and counts["worker_subprocess_attempts"]==0
    if status in ("unknown_timeout","unknown_abnormal_exit","exit0_internal_unknown"):
        return (set(value)==keys and value["counts"] is None
            and value["internal_counts_known"] is False
            and counts["worker_subprocess_attempts"]==1
            and counts["worker_subprocess_successes"]==0
            and value["killed"] is (status=="unknown_timeout")
            and ((status=="unknown_timeout" and value["returncode"] is None)
              or (status=="unknown_abnormal_exit" and type(value["returncode"]) is int
                  and value["returncode"]!=0)
              or (status=="exit0_internal_unknown" and value["returncode"]==0)))
    if status=="known_partial":
        return (set(value)==keys|{"rejection"} and value["internal_counts_known"] is True
            and value["killed"] is False and type(value["returncode"]) is int
            and value["returncode"]!=0 and value["counts"]==value["rejection"].get("counts")
            and certify_worker_rejection(value["rejection"])
            and all(counts[{"pin_model_contexts":"worker_pin_model_contexts",
                "rnea_calls":"constructor_rnea_calls"}.get(k,k)]==v
                for k,v in value["counts"].items() if k not in ("solve_calls","sqp_calls")))
    if status=="known_complete":
        return (set(value)==keys and value["returncode"]==0 and value["killed"] is False
            and value["internal_counts_known"] is True and value["counts"] is not None
            and counts["worker_subprocess_successes"]==1
            and all(counts[{"pin_model_contexts":"worker_pin_model_contexts",
                "rnea_calls":"constructor_rnea_calls"}.get(k,k)]==v
                for k,v in value["counts"].items() if k not in ("solve_calls","sqp_calls")))
    return False


SUCCESS_COUNTS={"completed":2,"prerequisite_artifact_loads":1,"parent_pin_model_contexts":1,
    "worker_pin_model_contexts":1,"total_pin_model_contexts":2,
    "worker_subprocess_attempts":1,"worker_subprocess_successes":1,
    "b1_constructors":1,"b16_constructors":1,"constructor_rnea_calls":518,
    "independent_rnea_recert_calls":518,"total_rnea_calls":1036,
    "b1_sim_forward_calls":34188,"b16_sim_forward_calls":16835,
    "b1_tool_position_calls":34192,"b16_tool_position_calls":16837,
    "parent_input_start_kinematics_calls":1,"parent_input_direction_kinematics_calls":1,
    "total_parent_input_kinematics_calls":2,"pin_identical_state_fk_calls":33674,
    "total_parent_kinematics_calls":33676,"optimizer_calls":0,"solve_calls":0,
    "sqp_calls":0,"rng_calls":0,"rejected_p5_v1_artifact_loads":0,
    "rejected_p5_v2_artifact_loads":0}


def zero_counts():return {key:0 for key in SUCCESS_COUNTS}


def generation_one_counts():
    value=zero_counts();value["prerequisite_artifact_loads"]=1;return value


def certify_counts(value,final=False):
    if not isinstance(value,dict) or set(value)!=set(SUCCESS_COUNTS):return False
    if any(type(item) is not int or item<0 for item in value.values()):return False
    if final:return value==SUCCESS_COUNTS
    if value["completed"] not in (0,2):return False
    if value["rejected_p5_v1_artifact_loads"] or value["rejected_p5_v2_artifact_loads"] \
            or value["optimizer_calls"] or value["solve_calls"] or value["sqp_calls"] \
            or value["rng_calls"]:return False
    if value["parent_pin_model_contexts"]>value["prerequisite_artifact_loads"] or value[
            "prerequisite_artifact_loads"]>1:return False
    if value["worker_pin_model_contexts"]>value["worker_subprocess_attempts"]:return False
    if value["total_pin_model_contexts"]!=value["parent_pin_model_contexts"]+value[
            "worker_pin_model_contexts"]:return False
    if value["worker_subprocess_attempts"]>value["parent_pin_model_contexts"] \
            or value["worker_subprocess_successes"]>value["worker_subprocess_attempts"]:return False
    for key,maximum in SUCCESS_COUNTS.items():
        if value[key]>maximum:return False
    if value["worker_subprocess_successes"]==1:
        variable=("completed","independent_rnea_recert_calls","pin_identical_state_fk_calls",
            "total_rnea_calls","total_parent_kinematics_calls")
        fixed=all(value[key]==SUCCESS_COUNTS[key] for key in SUCCESS_COUNTS if key not in variable)
        recert_pair=(value["independent_rnea_recert_calls"],value["pin_identical_state_fk_calls"])
        recert_ok=recert_pair in ((0,0),(SUCCESS_COUNTS["independent_rnea_recert_calls"],
            SUCCESS_COUNTS["pin_identical_state_fk_calls"]))
        return (fixed and recert_ok
            and value["total_rnea_calls"]==value["constructor_rnea_calls"]
                +value["independent_rnea_recert_calls"]
            and value["total_parent_kinematics_calls"]==value[
                "total_parent_input_kinematics_calls"]+value["pin_identical_state_fk_calls"]
            and value["completed"] in (0,2)
            and (value["completed"]==0 or recert_ok and value[
                "independent_rnea_recert_calls"]==SUCCESS_COUNTS["independent_rnea_recert_calls"]))
    if value["worker_subprocess_attempts"]==0:
        return not any(value[k] for k in ("b1_constructors","b16_constructors",
            "constructor_rnea_calls","independent_rnea_recert_calls","total_rnea_calls",
            "b1_sim_forward_calls","b16_sim_forward_calls","b1_tool_position_calls",
            "b16_tool_position_calls","parent_input_start_kinematics_calls",
            "parent_input_direction_kinematics_calls","total_parent_input_kinematics_calls",
            "pin_identical_state_fk_calls","total_parent_kinematics_calls","completed"))
    return (value["b1_constructors"] in (0,1) and value["b16_constructors"] in (0,1)
        and value["constructor_rnea_calls"]<=518
        and value["independent_rnea_recert_calls"]==0
        and value["total_rnea_calls"]==value["constructor_rnea_calls"]
        and value["pin_identical_state_fk_calls"]==0
        and value["completed"]==0)


def deadline_guard(deadline,monotonic):
    if monotonic()>deadline:raise TimeoutError("P6 campaign wall limit")


def publish_final_pointer(pointer,start,deadline,monotonic):
    finish=float(monotonic()-start)
    if monotonic()>deadline:raise TimeoutError("P6 campaign wall limit")
    pointer["publication_finish_elapsed_s"]=finish
    atomic_json(latest_path(),pointer,True)


def publish_checkpoint(generation,stage,counts,provenance,detail):
    if (generation,stage) not in ((0,"generation_zero"),(1,"prerequisite_authenticated"),
            (2,"routes_certified"),(3,"honest_end")):raise ValueError("P6 checkpoint stage")
    if not certify_counts(counts,final=generation>=2):raise ValueError("P6 checkpoint counts")
    path=checkpoint_path(generation)
    value={"protocol":PROTOCOL,"generation":generation,"stage":stage,"incomplete":True,
        "completed":counts["completed"],"pending":2-counts["completed"],
        "counts":counts,"provenance":provenance,"detail":detail,
        "oracle_evidence":False,"benchmark_evidence":False,"sqp_evidence":False}
    atomic_json(path,value)
    atomic_json(latest_path(),{"protocol":PROTOCOL,
        "generation":generation,"path":str(path),"sha256":sha(path),"incomplete":True},True)
    return value


def certify_provenance(value,final):
    keys={"cwd","orig_argv","command","head_start","head_end","clean_start","clean_end",
        "sources_start","sources_end","extension_start","extension_end","runtime_versions",
        "thread_environment","p5_v1_report","p5_v2_report"}
    try:
        current=snapshot();expected_argv=tuple(value.get("orig_argv",()))
        argv_ok=expected_argv==AUTHORIZED_ORIG_ARGV or (Path(OUTPUT)!=Path(
            "/tmp/tiago-tool-center-cuda-authoritative-route-seed-p6-authorized-once/p6.json")
            and expected_argv[:-1]==AUTHORIZED_ORIG_ARGV[:-1] and expected_argv[-1]==str(OUTPUT))
        return bool(set(value)==keys and value["cwd"]==AUTHORIZED_CWD
            and argv_ok and value["command"]==shlex.join(expected_argv)
            and value["head_start"]==current["head"] and value["clean_start"] is True
            and value["sources_start"]==current["sources"]
            and value["extension_start"]==current["extension"]
            and value["thread_environment"]==THREAD_ENV
            and value["runtime_versions"]=={"python":sys.version,"numpy":np.__version__,
                "pinocchio":importlib.metadata.version("pin")}
            and value["p5_v1_report"]==canonical_authentication(P5_V1_REJECTED_REPORT)
            and value["p5_v2_report"]==canonical_authentication(P5_V2_REJECTED_REPORT)
            and ((not final and value["head_end"] is value["clean_end"]
                is value["sources_end"] is value["extension_end"] is None)
              or (final and value["head_end"]==value["head_start"]
                and value["clean_end"] is True and value["sources_end"]==value["sources_start"]
                and value["extension_end"]==value["extension_start"])))
    except Exception:return False


def worker_document(summary,arrays):
    paths=worker_paths()
    expected_keys={"protocol","identity","request_path","request_sha256","input_path",
        "input_sha256","npz_path","npz_sha256","array_names","array_hashes","elapsed_s",
        "certificate","pin_model_contexts","b1_constructors","b16_constructors","b1_sim_forward_calls",
        "b16_sim_forward_calls","b1_tool_position_calls","b16_tool_position_calls",
        "rnea_calls","solve_calls","sqp_calls","phase_timings","provenance","imported_extension",
        "cuda_diagnostics"}
    return bool(set(summary)==expected_keys and summary["protocol"]==PROTOCOL+"_worker"
        and summary["identity"]==["development",12600]
        and summary["request_path"]==str(paths["request"])
        and summary["request_sha256"]==sha(paths["request"])
        and summary["input_path"]==str(paths["input"])
        and summary["input_sha256"]==sha(paths["input"])
        and summary["npz_path"]==str(paths["npz"])
        and summary["npz_sha256"]==sha(paths["npz"])
        and summary["array_names"]==sorted(arrays)
        and summary["array_hashes"]=={k:array_hash(arrays[k]) for k in sorted(arrays)}
        and summary["certificate"] is True and certify_output(arrays)
        and np.isfinite(summary["elapsed_s"]) and 0<=summary["elapsed_s"]<=300.
        and certify_worker_provenance(summary["provenance"],True)
        and summary["provenance"]["start_snapshot"]==worker_snapshot()
        and summary["provenance"]["start_snapshot"]["clean"] is True
        and summary["imported_extension"]=={
            "path":summary["provenance"]["start_snapshot"]["extension"]["path"],
            "sha256":EXTENSION["sha256"],"size":EXTENSION["size"]}
        and certify_cuda_diagnostics(summary["cuda_diagnostics"])
        and summary["pin_model_contexts"]==1
        and set(summary["phase_timings"])=={"constructor_rnea_elapsed_s"}
        and np.isfinite(summary["phase_timings"]["constructor_rnea_elapsed_s"])
        and summary["phase_timings"]["constructor_rnea_elapsed_s"]>=0
        and summary["b1_constructors"]==1 and summary["b16_constructors"]==1
        and summary["rnea_calls"]==2*259
        and summary["b1_sim_forward_calls"]==2*259*64+4*259
        and summary["b16_sim_forward_calls"]==259*64+259
        and summary["b1_tool_position_calls"]==2*(259*64)+4*259+4
        and summary["b16_tool_position_calls"]==259*64+259+2
        and summary["solve_calls"]==summary["sqp_calls"]==0)


def worker_input(prerequisite,kinematics):
    q0=prerequisite["public_x0_float32"][0,:7].astype(np.float64)
    qgoal=prerequisite["quarantined_q8_float64"][0]
    direction=perturbation_direction(q0,qgoal,kinematics(q0)[0],
        prerequisite["q8_tool_float64"][0],kinematics,
        int(prerequisite["public_default_side_int8"][0]))
    value={"x0_float32":prerequisite["public_x0_float32"][0],
        "q_goal_float64":qgoal,"direction_float64":direction}
    for route in ("short","long"):
        proxy=planned_proxy(q0,qgoal,direction,route)
        value.update({f"{route}_proxy_q_float64":proxy["planned_q_float64"],
            f"{route}_proxy_qd_float64":proxy["planned_qd_float64"],
            f"{route}_proxy_qdd_float64":proxy["planned_qdd_float64"]})
    if not exact_arrays(value,WORKER_INPUT_SPECS):raise RuntimeError("P6 worker input invalid")
    return value


def certify_worker_input_chain(prerequisite,kinematics):
    try:
        paths=worker_paths();expected=worker_input(prerequisite,kinematics)
        with np.load(paths["input"],allow_pickle=False) as archive:
            retained={k:archive[k] for k in archive.files}
        request=json.loads(paths["request"].read_text())
        expected_request={"protocol":PROTOCOL+"_worker","identity":["development",12600],
            "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
            "output_json_path":str(paths["json"]),"output_npz_path":str(paths["npz"]),
            "extension":EXTENSION,"wall_limit_s":300.0}
        return bool(exact_arrays(retained,WORKER_INPUT_SPECS)
            and all(np.array_equal(retained[k],expected[k]) for k in expected)
            and request==expected_request)
    except Exception:return False


def certify_quarantined_rnea(arrays,inputs,rnea):
    try:
        for index,route in enumerate(("short","long")):
            proxy={"planned_q_float64":inputs[f"{route}_proxy_q_float64"],
                "planned_qd_float64":inputs[f"{route}_proxy_qd_float64"],
                "planned_qdd_float64":inputs[f"{route}_proxy_qdd_float64"]}
            for knot in range(259):
                state=arrays["generated_knot_state_float32"][index,knot]
                acceleration=feedback_acceleration(proxy,knot,state)
                if (not np.array_equal(arrays["rnea_q_float64"][index,knot],state[:7].astype(np.float64))
                    or not np.array_equal(arrays["rnea_qd_float64"][index,knot],state[7:].astype(np.float64))
                    or not np.array_equal(arrays["rnea_qdd_float64"][index,knot],acceleration)
                    or not np.array_equal(arrays["rnea_u_float64"][index,knot],
                        np.asarray(rnea(state[:7].astype(np.float64),state[7:].astype(np.float64),
                            acceleration),np.float64))):return False
        return np.array_equal(arrays["rnea_u_float64"].astype(np.float32),
                              arrays["recorded_controls_float32"])
    except Exception:return False


def certify_campaign(arrays,prerequisite,model,kinematics,tool_velocity=None,rnea=None,inputs=None):
    if not certify_output(arrays):return {"passes":False}
    if rnea is None or inputs is None or not certify_quarantined_rnea(arrays,inputs,rnea):
        return {"passes":False}
    pin_dense=[];dense_jacobians=[];pin_knots=[];knot_jacobians=[]
    for dense_row,knot_row in zip(arrays["b1_dense_state_float32"],
                                  arrays["generated_knot_state_float32"]):
        dense_values=[kinematics(q.astype(np.float64)) for q in dense_row[:,:7]]
        knot_values=[kinematics(q.astype(np.float64)) for q in knot_row[:,:7]]
        pin_dense.append([value[0] for value in dense_values]);dense_jacobians.append(dense_values[-1][1])
        pin_knots.append([value[0] for value in knot_values]);knot_jacobians.append(knot_values[-1][1])
    pin_dense=np.asarray(pin_dense);pin_knots=np.asarray(pin_knots)
    goal=prerequisite["q8_tool_float64"][0];default=int(prerequisite["public_default_side_int8"][0])
    geometry=geometry_from_lane_zero({"pin_dense_tool_float64":arrays["b1_dense_tool_float32"][0]},
        {"pin_dense_tool_float64":arrays["b1_dense_tool_float32"][1]},goal,default)
    if not geometry["passes"]:return {"passes":False}
    reference=np.tile(np.asarray(geometry["reference_float32"],np.float32),KNOTS)
    limits=tuple(prerequisite[k] for k in ("joint_lower_float64","joint_upper_float64",
        "velocity_limit_float64","effort_limit_float64"))
    seed_details=[];dense_details=[]
    for index,route in enumerate(("short","long")):
        seed_details.append(authoritative_metrics(arrays["generated_knot_state_float32"][index],
            arrays["generated_knot_tool_float32"][index],pin_knots[index],
            arrays["recorded_controls_float32"][index],goal,geometry["pillar_float64"],
            reference,*limits,knot_jacobians[index],route,default,True))
        dense_details.append(authoritative_metrics(arrays["b1_dense_state_float32"][index],
            arrays["b1_dense_tool_float32"][index],pin_dense[index],
            arrays["recorded_controls_float32"][index],goal,geometry["pillar_float64"],
            reference,*limits,dense_jacobians[index],route,default,False))
    short,long=(row["metrics"] for row in seed_details)
    short_residual=np.asarray(seed_details[0]["details"]["cuda_cost"]["toll_residual_float64"])
    long_residual=np.asarray(seed_details[1]["details"]["cuda_cost"]["toll_residual_float64"])
    path_difference=long["tool_path_length_m"]-short["tool_path_length_m"]
    cuda_base_difference=long["cuda_base_cost"]-short["cuda_base_cost"]
    cuda_full_difference=short["cuda_full_cost"]-long["cuda_full_cost"]
    full_threshold=max(1.,.05*max(1.,abs(long["cuda_full_cost"])))
    pair={"opposite_topology":int(np.sign(short["turn_rad"]))==-int(np.sign(long["turn_rad"])),
        "finite":bool(np.isfinite([path_difference,cuda_base_difference,cuda_full_difference,
            full_threshold]).all()),
        "short_path_advantage":path_difference>=.005,
        "cuda_base_shorter":cuda_base_difference>=.10,
        "cuda_full_reversal":cuda_full_difference>=full_threshold,
        "short_exposure":int(np.count_nonzero(short_residual[1:-1]>0.))>=8,
        "short_saturation":int(np.count_nonzero(short_residual[1:-1]>=.95))>=8,
        "long_exposure_cap":float(np.max(long_residual))<=.05,
        "long_toll_cap":long["cuda_toll_contribution"]<=.01*short["cuda_toll_contribution"]}
    differences={"path_length_m":path_difference,"cuda_base_cost":cuda_base_difference,
        "cuda_full_cost":cuda_full_difference,"cuda_full_threshold":full_threshold}
    return {"seed_routes":seed_details,"dense_physical_routes":dense_details,"pair":pair,
        "pair_differences":differences,"geometry":geometry,
        "passes":bool(all(row["passes"] for row in (*seed_details,*dense_details))
            and all(pair.values()))}


def static_design():
    return {"protocol":PROTOCOL,"output":str(OUTPUT),"identity":list(PILOT_IDENTITY),
        "routes":[list(x) for x in PILOT_ROUTES],"gain":{"kp":25.,"kd":10.},
        "attempts":2,"retries":0,"cuda_authoritative":True,"pin_dynamic_replay":False,
        "pin_rnea_quarantined_nominal_constructor":True,"pin_rnea_calls":518,
        "pin_rnea_feasibility_evidence":False,
        "b1_seed_equals_b16_lane0":True,"b16_distinct_streams":2,
        "optimizer_calls":0,"solve_calls":0,"sqp_calls":0,"rng_calls":0,
        "campaign_wall_limit_s":CAMPAIGN_WALL_LIMIT_S,"extension":EXTENSION,
        "prior_dense_feedback_screen_non_evidence":SCREEN_RESULTS,
        "full_dt_solver_knot_pilot_pending":True,"p5_v2_report":P5_V2_REJECTED_REPORT}


def execute(output=OUTPUT,authorization=None,monotonic=time.monotonic): # pragma: no cover
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P6 runner blocked")
    if Path(output).resolve()!=OUTPUT:raise RuntimeError("wrong P6 output")
    if OUTPUT.parent.exists():raise FileExistsError("P6 namespace exists")
    if {k:os.environ.get(k) for k in THREAD_ENV}!=THREAD_ENV:
        raise RuntimeError("P6 requires single-thread environment")
    if Path.cwd().resolve()!=Path(AUTHORIZED_CWD) or tuple(sys.orig_argv)!=AUTHORIZED_ORIG_ARGV:
        raise RuntimeError("P6 invocation mismatch")
    current=snapshot()
    if current["clean"] is not True or current["extension"]["sha256"]!=EXTENSION["sha256"] \
            or current["extension"]["size"]!=EXTENSION["size"]:
        raise RuntimeError("P6 start provenance invalid")
    start=monotonic();deadline=start+600.;OUTPUT.parent.mkdir(parents=False)
    counts=zero_counts()
    provenance={"cwd":str(Path.cwd().resolve()),"orig_argv":list(sys.orig_argv),
        "command":shlex.join(sys.orig_argv),"head_start":current["head"],"head_end":None,
        "clean_start":current["clean"],"clean_end":None,
        "sources_start":current["sources"],"sources_end":None,
        "extension_start":current["extension"],"extension_end":None,
        "runtime_versions":{"python":sys.version,"numpy":np.__version__,
            "pinocchio":importlib.metadata.version("pin")},
        "thread_environment":THREAD_ENV,
        "p5_v1_report":canonical_authentication(P5_V1_REJECTED_REPORT),
        "p5_v2_report":canonical_authentication(P5_V2_REJECTED_REPORT)}
    worker_execution={"status":"not_started","returncode":None,"counts":None,
        "killed":False,"internal_counts_known":False}
    publish_checkpoint(0,"generation_zero",counts,provenance,{})
    try:
        deadline_guard(deadline,monotonic)
        authentication,prerequisite=authenticate_cpu_prerequisite()
        authentication=canonical_authentication(authentication);counts["prerequisite_artifact_loads"]=1
        deadline_guard(deadline,monotonic)
        publish_checkpoint(1,"prerequisite_authenticated",counts,provenance,
            {"authentication":authentication})
        model,kinematics,_rnea,_derivatives,_aba,tool_velocity,_dense=_production_pin_context()
        counts["parent_pin_model_contexts"]=1
        deadline_guard(deadline,monotonic)
        inputs=worker_input(prerequisite,kinematics);paths=worker_paths()
        counts["parent_input_start_kinematics_calls"]=1
        counts["parent_input_direction_kinematics_calls"]=1
        counts["total_parent_input_kinematics_calls"]=2
        counts["total_parent_kinematics_calls"]=2
        if any(path.exists() or Path(str(path)+".candidate").exists() for path in paths.values()):
            raise FileExistsError("P6 worker artifact exists")
        atomic_npz(paths["input"],inputs)
        request={"protocol":PROTOCOL+"_worker","identity":["development",12600],
            "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
            "output_json_path":str(paths["json"]),"output_npz_path":str(paths["npz"]),
            "extension":EXTENSION,"wall_limit_s":300.0}
        atomic_json(paths["request"],request);counts["worker_subprocess_attempts"]=1
        remaining=min(300.,deadline-monotonic())
        if remaining<=0:raise TimeoutError("P6 campaign wall limit")
        command=["python","-B","-m","gato_tiago.circular_portfolio_p6_worker",
            "--execute","--request",str(paths["request"]),"--output",str(paths["json"])]
        try:
            completed=subprocess.run(command,cwd="/workspace/GATO",env=os.environ.copy(),
                timeout=remaining,check=False)
        except subprocess.TimeoutExpired:
            worker_execution={"status":"unknown_timeout","returncode":None,"counts":None,
                "killed":True,"internal_counts_known":False};raise
        if completed.returncode:
            if paths["rejected"].is_file():
                worker_rejection=json.loads(paths["rejected"].read_text())
                if not certify_worker_rejection(worker_rejection):
                    raise RuntimeError("P6 worker rejection invalid")
                worker_map={"pin_model_contexts":"worker_pin_model_contexts",
                    "rnea_calls":"constructor_rnea_calls"}
                for key,value in worker_rejection["counts"].items():
                    if key in ("solve_calls","sqp_calls"):continue
                    counts[worker_map.get(key,key)]=value
                counts["total_pin_model_contexts"]=counts["parent_pin_model_contexts"]+counts[
                    "worker_pin_model_contexts"]
                counts["total_rnea_calls"]=counts["constructor_rnea_calls"]
                worker_execution={"status":"known_partial","returncode":completed.returncode,
                    "counts":worker_rejection["counts"],"killed":False,
                    "internal_counts_known":True,"rejection":worker_rejection}
                if worker_rejection["error_type"]=="TimeoutError":
                    raise TimeoutError("P6 worker wall limit")
            else:
                worker_execution={"status":"unknown_abnormal_exit",
                    "returncode":completed.returncode,"counts":None,"killed":False,
                    "internal_counts_known":False}
            raise RuntimeError("P6 worker failed")
        worker_execution={"status":"exit0_internal_unknown","returncode":0,"counts":None,
            "killed":False,"internal_counts_known":False}
        summary=json.loads(paths["json"].read_text())
        with np.load(paths["npz"],allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        if not worker_document(summary,arrays):raise RuntimeError("P6 worker artifact invalid")
        counts.update({"worker_subprocess_successes":1,"b1_constructors":1,
            "worker_pin_model_contexts":1,"total_pin_model_contexts":2,
            "b16_constructors":1,"constructor_rnea_calls":2*259,
            "b1_sim_forward_calls":2*259*64+4*259,
            "b16_sim_forward_calls":259*64+259,
            "b1_tool_position_calls":2*(259*64)+4*259+4,
            "b16_tool_position_calls":259*64+259+2})
        worker_execution={"status":"known_complete","returncode":0,
            "counts":{key:summary[key] for key in ("pin_model_contexts","b1_constructors","b16_constructors",
                "rnea_calls","b1_sim_forward_calls","b16_sim_forward_calls",
                "b1_tool_position_calls","b16_tool_position_calls","solve_calls","sqp_calls")},
            "killed":False,"internal_counts_known":True}
        deadline_guard(deadline,monotonic)
        measured_kinematics=0;measured_rnea=0
        def counted_kinematics(q):
            nonlocal measured_kinematics;measured_kinematics+=1;return kinematics(q)
        def counted_rnea(q,qd,qdd):
            nonlocal measured_rnea;measured_rnea+=1;return _rnea(q,qd,qdd)
        certificate=canonical_authentication(certify_campaign(arrays,prerequisite,model,
            counted_kinematics,tool_velocity,counted_rnea,inputs))
        counts["independent_rnea_recert_calls"]=measured_rnea
        counts["total_rnea_calls"]=counts["constructor_rnea_calls"]+measured_rnea
        counts["pin_identical_state_fk_calls"]=measured_kinematics
        counts["total_parent_kinematics_calls"]=counts[
            "total_parent_input_kinematics_calls"]+measured_kinematics
        if not certificate["passes"]:raise RuntimeError("P6 scientific certificate failed")
        for index,route in enumerate(("short","long")):
            atomic_json(route_path(index),{"protocol":PROTOCOL,"identity":["development",12600,
                route,0],"worker_npz_path":str(paths["npz"]),"worker_npz_sha256":sha(paths["npz"]),
                "route_index":index,"seed_certificate":certificate["seed_routes"][index],
                "dense_certificate":certificate["dense_physical_routes"][index],
                "passes":certificate["seed_routes"][index]["passes"]
                    and certificate["dense_physical_routes"][index]["passes"]})
        counts["completed"]=2
        publish_checkpoint(2,"routes_certified",counts,provenance,{"certificate":certificate})
        deadline_guard(deadline,monotonic)
        end=snapshot();provenance.update({"head_end":end["head"],"clean_end":end["clean"],
            "sources_end":end["sources"],"extension_end":end["extension"]})
        if (provenance["head_end"]!=provenance["head_start"] or provenance["clean_end"] is not True
                or provenance["sources_end"]!=provenance["sources_start"]
                or provenance["extension_end"]!=provenance["extension_start"]):
            raise RuntimeError("P6 end provenance invalid")
        publish_checkpoint(3,"honest_end",counts,provenance,{"certificate":certificate})
        deadline_guard(deadline,monotonic)
        semantic_finish=float(monotonic()-start)
        deadline_guard(deadline,monotonic)
        final={"protocol":PROTOCOL,"incomplete":False,"completed":2,"pending":0,
            "counts":counts,"authentication":authentication,"worker":summary,
            "worker_execution":worker_execution,"certificate":certificate,"provenance":provenance,
            "semantic_finish_elapsed_s":semantic_finish,"oracle_evidence":False,
            "benchmark_evidence":False,"sqp_evidence":False,"seed_evidence":True}
        atomic_json(OUTPUT,final)
        deadline_guard(deadline,monotonic)
        manifest=manifest_path()
        side_paths=[checkpoint_path(i) for i in range(4)]+[paths["request"],paths["input"],
            paths["json"],paths["npz"],route_path(0),route_path(1)]
        atomic_json(manifest,{"protocol":PROTOCOL,"json_path":str(OUTPUT),
            "json_sha256":sha(OUTPUT),"worker_json_path":str(paths["json"]),
            "worker_json_sha256":sha(paths["json"]),"worker_npz_path":str(paths["npz"]),
            "worker_npz_sha256":sha(paths["npz"]),"side_artifacts":[{"path":str(path),
                "sha256":sha(path),"size":path.stat().st_size} for path in side_paths],
            "overall_pass":True})
        # Precompute every filesystem-derived pointer field, then sample once immediately
        # before the sole authoritative pointer replacement.
        pointer={"protocol":PROTOCOL,"incomplete":False,"json_path":str(OUTPUT),
            "json_sha256":sha(OUTPUT),"manifest_path":str(manifest),
            "manifest_sha256":sha(manifest),"publication_finish_elapsed_s":None}
        publish_final_pointer(pointer,start,deadline,monotonic)
        return final
    except Exception as error:
        trigger_elapsed=float(monotonic()-start)
        try:
            end=snapshot();provenance.update({"head_end":end["head"],"clean_end":end["clean"],
                "sources_end":end["sources"],"extension_end":end["extension"]})
        except Exception:
            pass
        artifacts=active_artifact_map();cleanup_finish=float(monotonic()-start)
        rejection={"protocol":PROTOCOL,"stage":"runtime_watchdog_rejected"
                if isinstance(error,(TimeoutError,subprocess.TimeoutExpired)) else "pilot_failed",
            "error_type":type(error).__name__,"error_message":str(error),"incomplete":True,
            "completed":counts["completed"],"pending":2-counts["completed"],"counts":counts,
            "worker_execution":worker_execution,"provenance":provenance,
            "active_artifacts":artifacts,"candidate_classifications":classify_candidates(),
            "trigger_elapsed_s":trigger_elapsed,
            "cleanup_finish_elapsed_s":cleanup_finish,"campaign_wall_limit_s":600.,
            "oracle_evidence":False,
            "benchmark_evidence":False,"sqp_evidence":False,"seed_evidence":False}
        atomic_json(OUTPUT.with_name("p6.rejected.json"),rejection);raise


def recertify_retained_p6(output=OUTPUT): # pragma: no cover - external audit boundary
    try:
        output=Path(output);manifest=output.with_name("p6.manifest.json")
        pointer=output.with_name("p6.partial.latest.json");paths=worker_paths()
        summary=json.loads(output.read_text());manifest_doc=json.loads(manifest.read_text())
        pointer_doc=json.loads(pointer.read_text())
        side_paths=[checkpoint_path(i) for i in range(4)]+[paths["request"],paths["input"],
            paths["json"],paths["npz"],route_path(0),route_path(1)]
        expected_manifest={"protocol":PROTOCOL,"json_path":str(output),
            "json_sha256":sha(output),"worker_json_path":str(paths["json"]),
            "worker_json_sha256":sha(paths["json"]),"worker_npz_path":str(paths["npz"]),
            "worker_npz_sha256":sha(paths["npz"]),"side_artifacts":[{"path":str(path),
                "sha256":sha(path),"size":path.stat().st_size} for path in side_paths],
            "overall_pass":True}
        if manifest_doc!=expected_manifest:return {"passes":False}
        pointer_base={"protocol":PROTOCOL,"incomplete":False,"json_path":str(output),
            "json_sha256":sha(output),"manifest_path":str(manifest),
            "manifest_sha256":sha(manifest)}
        if (set(pointer_doc)!=set(pointer_base)|{"publication_finish_elapsed_s"}
                or any(pointer_doc[k]!=v for k,v in pointer_base.items())
                or not np.isfinite(pointer_doc["publication_finish_elapsed_s"])
                or not 0<=pointer_doc["publication_finish_elapsed_s"]<=600.):return {"passes":False}
        worker_summary=json.loads(paths["json"].read_text())
        with np.load(paths["npz"],allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        if not worker_document(worker_summary,arrays):return {"passes":False}
        authentication,prerequisite=authenticate_cpu_prerequisite()
        authentication=canonical_authentication(authentication)
        if summary.get("authentication")!=authentication or summary.get("worker")!=worker_summary:
            return {"passes":False}
        model,kinematics,_rnea,_derivatives,_aba,tool_velocity,_dense=_production_pin_context()
        if not certify_worker_input_chain(prerequisite,kinematics):return {"passes":False}
        with np.load(paths["input"],allow_pickle=False) as archive:
            retained_inputs={k:archive[k] for k in archive.files}
        fresh=canonical_authentication(certify_campaign(arrays,prerequisite,model,kinematics,
            tool_velocity,_rnea,retained_inputs))
        checkpoints=[json.loads(checkpoint_path(i).read_text()) for i in range(4)]
        stages=("generation_zero","prerequisite_authenticated","routes_certified","honest_end")
        expected_counts=(zero_counts(),generation_one_counts(),SUCCESS_COUNTS,SUCCESS_COUNTS)
        for generation,(doc,stage,expected) in enumerate(zip(checkpoints,stages,expected_counts)):
            if (set(doc)!={"protocol","generation","stage","incomplete","completed","pending",
                    "counts","provenance","detail","oracle_evidence","benchmark_evidence",
                    "sqp_evidence"} or doc.get("protocol")!=PROTOCOL or doc.get("generation")!=generation
                    or doc.get("stage")!=stage or doc.get("incomplete") is not True
                    or doc.get("counts")!=expected or doc.get("completed")!=expected["completed"]
                    or doc.get("pending")!=2-expected["completed"]
                    or not certify_provenance(doc.get("provenance"),generation==3)
                    or doc.get("oracle_evidence") is not False
                    or doc.get("benchmark_evidence") is not False
                    or doc.get("sqp_evidence") is not False):return {"passes":False}
        if (checkpoints[0]["detail"]!={} or checkpoints[1]["detail"]!={"authentication":authentication}
                or checkpoints[2]["detail"]!={"certificate":fresh}
                or checkpoints[3]["detail"]!={"certificate":fresh}):return {"passes":False}
        routes=[]
        for index,route in enumerate(("short","long")):
            route_doc=json.loads(route_path(index).read_text());routes.append(route_doc)
            if route_doc!={"protocol":PROTOCOL,"identity":["development",12600,route,0],
                    "worker_npz_path":str(paths["npz"]),"worker_npz_sha256":sha(paths["npz"]),
                    "route_index":index,"seed_certificate":fresh["seed_routes"][index],
                    "dense_certificate":fresh["dense_physical_routes"][index],"passes":True}:
                return {"passes":False}
        final_keys={"protocol","incomplete","completed","pending","counts","authentication",
            "worker","worker_execution","certificate","provenance","semantic_finish_elapsed_s","oracle_evidence",
            "benchmark_evidence","sqp_evidence","seed_evidence"}
        expected_execution={"status":"known_complete","returncode":0,
            "counts":{key:worker_summary[key] for key in ("pin_model_contexts",
                "b1_constructors","b16_constructors",
                "rnea_calls","b1_sim_forward_calls","b16_sim_forward_calls",
                "b1_tool_position_calls","b16_tool_position_calls","solve_calls","sqp_calls")},
            "killed":False,"internal_counts_known":True}
        final_boundary=(set(summary)==final_keys and summary["protocol"]==PROTOCOL
            and summary["incomplete"] is False and summary["completed"]==2 and summary["pending"]==0
            and summary["counts"]==SUCCESS_COUNTS and summary["worker_execution"]==expected_execution
            and summary["provenance"]==checkpoints[3]["provenance"]
            and np.isfinite(summary["semantic_finish_elapsed_s"])
            and 0<=summary["semantic_finish_elapsed_s"]
                <=pointer_doc["publication_finish_elapsed_s"]<=600.
            and summary["seed_evidence"] is True and summary["oracle_evidence"] is False
            and summary["benchmark_evidence"] is False and summary["sqp_evidence"] is False)
        gates={"documents":True,"worker":True,"authentication":True,"checkpoints":True,
            "routes":True,
            "semantic":fresh.get("passes") is True and summary.get("certificate")==fresh,
            "boundary":final_boundary,
            "namespace":certify_namespace((*side_paths,output,manifest,pointer))}
        return {"gates":gates,"passes":bool(all(gates.values()))}
    except Exception:
        return {"passes":False}


def recertify_retained_failure(output=OUTPUT):
    try:
        value=json.loads(Path(output).with_name("p6.rejected.json").read_text())
        keys={"protocol","stage","error_type","error_message","incomplete","completed",
            "pending","counts","worker_execution","provenance","active_artifacts",
            "candidate_classifications",
            "trigger_elapsed_s","cleanup_finish_elapsed_s","campaign_wall_limit_s",
            "oracle_evidence","benchmark_evidence","sqp_evidence","seed_evidence"}
        boundary=bool(set(value)==keys and value["protocol"]==PROTOCOL
            and value["stage"] in ("pilot_failed","runtime_watchdog_rejected")
            and value["incomplete"] is True and value["completed"] in (0,2)
            and value["pending"]==2-value["completed"]
            and certify_counts(value["counts"])
            and certify_worker_execution(value["worker_execution"],value["counts"])
            and certify_active_artifact_map(value["active_artifacts"])
            and value["candidate_classifications"]==classify_candidates()
            and certify_provenance(value["provenance"],value["provenance"].get("head_end") is not None)
            and value["oracle_evidence"] is value["benchmark_evidence"]
                is value["sqp_evidence"] is value["seed_evidence"] is False
            and value["campaign_wall_limit_s"]==600.
            and np.isfinite(value["trigger_elapsed_s"])
            and np.isfinite(value["cleanup_finish_elapsed_s"])
            and 0<=value["trigger_elapsed_s"]<=value["cleanup_finish_elapsed_s"])
        if not boundary:return {"passes":False}
        existing=[i for i in range(4) if checkpoint_path(i).is_file()]
        if existing!=list(range(len(existing))) or not existing:return {"passes":False}
        stages=("generation_zero","prerequisite_authenticated","routes_certified","honest_end")
        expected_counts=(zero_counts(),generation_one_counts(),SUCCESS_COUNTS,SUCCESS_COUNTS)
        prerequisite=None;authentication=None;model=None;kinematics=None;tool_velocity=None
        if len(existing)>=2:
            authentication,prerequisite=authenticate_cpu_prerequisite()
            authentication=canonical_authentication(authentication)
        for generation in existing:
            doc=json.loads(checkpoint_path(generation).read_text())
            if (set(doc)!={"protocol","generation","stage","incomplete","completed","pending",
                    "counts","provenance","detail","oracle_evidence","benchmark_evidence",
                    "sqp_evidence"} or doc.get("protocol")!=PROTOCOL or doc.get("generation")!=generation
                    or doc.get("stage")!=stages[generation]
                    or doc.get("incomplete") is not True
                    or doc.get("counts")!=expected_counts[generation]
                    or doc.get("completed")!=doc["counts"]["completed"]
                    or doc.get("pending")!=2-doc["completed"]
                    or not certify_provenance(doc.get("provenance"),generation==3)
                    or doc.get("oracle_evidence") is not False
                    or doc.get("benchmark_evidence") is not False
                    or doc.get("sqp_evidence") is not False):return {"passes":False}
            if generation==0 and doc.get("detail")!={}:return {"passes":False}
            if generation==1 and doc.get("detail")!={"authentication":authentication}:
                return {"passes":False}
        latest=json.loads(latest_path().read_text());last=existing[-1];last_path=checkpoint_path(last)
        if latest!={"protocol":PROTOCOL,"generation":last,"path":str(last_path),
                "sha256":sha(last_path),"incomplete":True}:return {"passes":False}
        # Any completed scientific route is independently regenerated from the exact worker bytes.
        route_presence=[route_path(i).is_file() for i in range(2)]
        fresh=None;worker_summary=None
        if value["completed"]==2 and route_presence!=[True,True]:return {"passes":False}
        paths=worker_paths()
        known_complete=value["worker_execution"]["status"]=="known_complete"
        if known_complete:
            if not all(paths[name].is_file() for name in ("request","input","json","npz")):
                return {"passes":False}
            worker_summary=json.loads(paths["json"].read_text())
            with np.load(paths["npz"],allow_pickle=False) as archive:
                arrays={k:archive[k] for k in archive.files}
            if not worker_document(worker_summary,arrays):return {"passes":False}
            if value["worker_execution"]["counts"]!={key:worker_summary[key] for key in (
                    "pin_model_contexts","b1_constructors","b16_constructors","rnea_calls",
                    "b1_sim_forward_calls","b16_sim_forward_calls","b1_tool_position_calls",
                    "b16_tool_position_calls","solve_calls","sqp_calls")}:
                return {"passes":False}
            if prerequisite is None:return {"passes":False}
            model,kinematics,_rnea,_derivatives,_aba,tool_velocity,_dense=_production_pin_context()
            if not certify_worker_input_chain(prerequisite,kinematics):return {"passes":False}
            with np.load(paths["input"],allow_pickle=False) as archive:
                retained_inputs={k:archive[k] for k in archive.files}
            fresh=canonical_authentication(certify_campaign(arrays,prerequisite,model,kinematics,
                tool_velocity,_rnea,retained_inputs))
            if fresh.get("passes") is not True:return {"passes":False}
            for generation in existing:
                if generation>=2 and json.loads(checkpoint_path(generation).read_text()).get(
                        "detail")!={"certificate":fresh}:return {"passes":False}
        if any(route_presence):
            if route_presence!=[True,True] or not known_complete or fresh is None:
                return {"passes":False}
            for index,route in enumerate(("short","long")):
                retained=json.loads(route_path(index).read_text())
                expected={"protocol":PROTOCOL,"identity":["development",12600,route,0],
                    "worker_npz_path":str(paths["npz"]),"worker_npz_sha256":sha(paths["npz"]),
                    "route_index":index,"seed_certificate":fresh["seed_routes"][index],
                    "dense_certificate":fresh["dense_physical_routes"][index],"passes":True}
                if retained!=expected:
                    return {"passes":False}
            if value["completed"] not in (0,2):return {"passes":False}
            final_candidates=[path for path in (Path(output),Path(str(output)+".candidate"))
                              if path.is_file()]
            for final_candidate in final_candidates:
                orphan=json.loads(final_candidate.read_text())
                if (orphan.get("protocol")!=PROTOCOL or orphan.get("incomplete") is not False
                        or orphan.get("completed")!=2 or orphan.get("pending")!=0
                        or orphan.get("counts")!=SUCCESS_COUNTS
                        or orphan.get("certificate")!=fresh or orphan.get("authentication")!=authentication
                        or orphan.get("worker")!=worker_summary or orphan.get("seed_evidence") is not True
                        or orphan.get("oracle_evidence") is not False
                        or orphan.get("benchmark_evidence") is not False
                        or orphan.get("sqp_evidence") is not False
                        or not np.isfinite(orphan.get("semantic_finish_elapsed_s",np.nan))):
                    return {"passes":False}
            if manifest_path().is_file() or Path(str(manifest_path())+".candidate").is_file():
                if not Path(output).is_file():return {"passes":False}
                side_paths=[checkpoint_path(i) for i in range(4)]+[paths["request"],paths["input"],
                    paths["json"],paths["npz"],route_path(0),route_path(1)]
                expected_manifest={"protocol":PROTOCOL,"json_path":str(output),
                    "json_sha256":sha(output),"worker_json_path":str(paths["json"]),
                    "worker_json_sha256":sha(paths["json"]),"worker_npz_path":str(paths["npz"]),
                    "worker_npz_sha256":sha(paths["npz"]),"side_artifacts":[{"path":str(path),
                        "sha256":sha(path),"size":path.stat().st_size} for path in side_paths],
                    "overall_pass":True}
                for candidate in (manifest_path(),Path(str(manifest_path())+".candidate")):
                    if candidate.is_file() and json.loads(candidate.read_text())!=expected_manifest:
                        return {"passes":False}
        allowed=[rejection_path(),*map(checkpoint_path,existing),latest_path()]
        allowed.extend(path for path in paths.values() if path.is_file())
        allowed.extend(route_path(i) for i in range(2) if route_path(i).is_file())
        if Path(output).is_file():allowed.append(Path(output))
        if manifest_path().is_file():allowed.append(manifest_path())
        allowed.extend(Path(path) for path in value["candidate_classifications"])
        return {"passes":certify_namespace(allowed)}
    except Exception:return {"passes":False}


def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    execute(args.output,RUNNER_EXECUTION_AUTHORIZATION)


if __name__=="__main__":main()
