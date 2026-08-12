"""Transactional all-task CUDA replay of immutable P3 portfolio rows."""

from __future__ import annotations

import argparse,copy,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
import numpy as np

from gato_tiago.circular_portfolio import TASK_IDENTITIES
from gato_tiago.circular_portfolio_constructor import pack_solver_seed
from gato_tiago.circular_portfolio_p3_runner import LEDGER,ROW_ARRAY_SPECS
from gato_tiago.circular_portfolio_p4 import (EXTENSION,OUTPUT,P3_OUTPUT,P3_PINS,PROTOCOL,
    TASKS,WORKER_PROTOCOL,array_hash,certify_lane,certify_task,input_schema,output_schema,
    worker_paths)
from gato_tiago.circular_portfolio_p4_worker import certify_output,certify_rejection as certify_worker_rejection


RUNNER_EXECUTION_AUTHORIZATION=object()
AUTHORIZED_CWD="/workspace/GATO"
AUTHORIZED_ORIG_ARGV=("python","-B","-m","gato_tiago.circular_portfolio_p4_runner",
    "--execute","--output",str(OUTPUT))
SOURCE_PATHS=("tiago_src/gato_tiago/circular_portfolio.py",
    "tiago_src/gato_tiago/circular_portfolio_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_p3.py",
    "tiago_src/gato_tiago/circular_portfolio_p3_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_p3_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p4.py",
    "tiago_src/gato_tiago/circular_portfolio_p4_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p4_worker.py",
    "tiago_src/gato_tiago/circular_portfolio_prerequisite_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_runner.py",
    "tiago_src/gato_tiago/config.py","tiago_src/gato_tiago/multimodal_pillar.py",
    "tiago_src/gato_tiago/multimodal_toll.py",
    "python/bindings.cu","python/bsqp/interface.py","CMakeLists.txt",
    "gato/bsqp/bsqp.cuh","gato/bsqp/kernels/tool_position.cuh","gato/utils/cuda.cuh",
    "gato/dynamics/integrator.cuh","gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "gato/dynamics/tiago_right/tiago_right_grid.cuh",
    "gato/dynamics/tiago_right/tiago_right_arm.urdf",
    "tests/python/test_tiago_circular_portfolio_p4_static.py")
THREAD_ENV={"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
    "NUMEXPR_NUM_THREADS":"1"}
CAMPAIGN_WALL_LIMIT_S=21600.0
WORKER_WALL_LIMIT_S=900.0

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value
def atomic_json(path,value):
    candidate=Path(str(path)+".candidate")
    with candidate.open("x") as stream:json.dump(value,stream,sort_keys=True,separators=(",",":"));stream.write("\n")
    os.replace(candidate,path)
def atomic_npz(path,arrays):
    candidate=Path(str(path)+".candidate")
    with candidate.open("xb") as stream:np.savez(stream,**arrays)
    os.replace(candidate,path)
def write_json_candidate(path,value):
    candidate=Path(str(path)+".candidate")
    with candidate.open("x") as stream:
        json.dump(value,stream,sort_keys=True,separators=(",",":"));stream.write("\n")
    return candidate

def p3_paths():return {"json":P3_OUTPUT,"manifest":P3_OUTPUT.with_name("p3.manifest.json"),
    "pointer":P3_OUTPUT.with_name("p3.partial.latest.json"),
    "gen194":P3_OUTPUT.with_name("p3.partial.gen194.json")}

def authenticate_p3(): # pragma: no cover - data-bearing only under runner authorization
    paths=p3_paths();measured={k:sha(v) for k,v in paths.items()}
    if measured!=P3_PINS:raise RuntimeError("P3 artifact pin mismatch")
    from gato_tiago.circular_portfolio_p3_runner import recertify_retained_p3
    recert=recertify_retained_p3(P3_OUTPUT)
    if recert.get("passes") is not True:raise RuntimeError("P3 disk recertification failed")
    document=json.loads(P3_OUTPUT.read_text())
    return {"pins":measured,"recertification":recert,"passes":True},document

def certify_p3_authentication(value):
    return bool(isinstance(value,dict) and set(value)=={"pins","recertification","passes"}
        and value["pins"]==P3_PINS and value["recertification"]=={"passes":True}
        and value["passes"] is True)

def snapshot():
    root=Path(__file__).resolve().parents[2]
    run=lambda *a:subprocess.run(["git",*a],cwd=root,check=True,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
    return {"head":run("rev-parse","HEAD"),"clean":run("status","--porcelain","--untracked-files=no")=="",
        "sources":{name:sha(root/name) for name in SOURCE_PATHS}}
def start_provenance():
    value=snapshot();environment={k:os.environ.get(k) for k in THREAD_ENV}
    if environment!=THREAD_ENV:raise RuntimeError("P4 requires single-thread host environment")
    return {"cwd":str(Path.cwd().resolve()),"orig_argv":list(sys.orig_argv),
        "command":shlex.join(sys.orig_argv),"head_start":value["head"],"head_end":None,
        "clean_start":value["clean"],"clean_end":None,"sources_start":value["sources"],
        "sources_end":None,"p3_pins":P3_PINS,"extension":EXTENSION,
        "thread_environment":environment}
def finish_provenance(value):
    end=snapshot();value.update({"head_end":end["head"],"clean_end":end["clean"],
        "sources_end":end["sources"]})
def certify_provenance(value,final):
    current=snapshot();keys={"cwd","orig_argv","command","head_start","head_end","clean_start",
        "clean_end","sources_start","sources_end","p3_pins","extension","thread_environment"}
    return bool(set(value)==keys and value["cwd"]==AUTHORIZED_CWD
        and tuple(value["orig_argv"])==AUTHORIZED_ORIG_ARGV
        and value["command"]==shlex.join(AUTHORIZED_ORIG_ARGV)
        and value["head_start"]==current["head"] and value["clean_start"] is True
        and value["sources_start"]==current["sources"] and current["clean"] is True
        and value["p3_pins"]==P3_PINS and value["extension"]==EXTENSION
        and value["thread_environment"]==THREAD_ENV
        and ((not final and value["head_end"] is value["clean_end"] is value["sources_end"] is None)
        or (final and value["head_end"]==current["head"] and value["clean_end"] is True
            and value["sources_end"]==current["sources"])))

def new_counts():return {"p3_artifact_loads":0,"p3_disk_recertifications":0,
    "p3_independent_pin_replays":0,"prerequisite_artifact_loads":0,"pin_model_contexts":0,
    "worker_subprocess_attempts":0,"worker_subprocess_successes":0,"b16_constructors":0,
    "sim_forward_calls":0,"tool_position_calls":0,"solve_calls":0,"sqp_calls":0,
    "optimizer_calls":0,"rng_calls":0,"task_construction_calls":0}
def checkpoint_counts(completed,stage):
    value=new_counts()
    if stage!="gen0":value.update({"p3_artifact_loads":1,"p3_disk_recertifications":1,
        "p3_independent_pin_replays":192,"prerequisite_artifact_loads":1,
        "pin_model_contexts":1})
    if completed or stage in ("task_completed","honest_end"):
        value["prerequisite_artifact_loads"]=2;value["pin_model_contexts"]=2
        value.update({"worker_subprocess_attempts":completed,
            "worker_subprocess_successes":completed,"b16_constructors":completed,
            "sim_forward_calls":completed*6080,"tool_position_calls":completed*6081})
    return value
def certify_counts(value,completed,stage,partial=False):
    if set(value)!=set(new_counts()) or any(not isinstance(x,int) or x<0 for x in value.values()):return False
    if any(value[k] for k in ("solve_calls","sqp_calls","optimizer_calls","rng_calls",
                              "task_construction_calls")):return False
    upstream=(value["p3_artifact_loads"],value["p3_disk_recertifications"],
        value["p3_independent_pin_replays"])
    if upstream not in ((0,0,0),(1,1,192)):return False
    if value["pin_model_contexts"]>value["prerequisite_artifact_loads"]:return False
    if upstream==(0,0,0) and (value["prerequisite_artifact_loads"]
            or value["pin_model_contexts"]):return False
    if upstream==(1,1,192) and (value["prerequisite_artifact_loads"],
            value["pin_model_contexts"]) not in ((1,1),(2,1),(2,2)):return False
    if value["worker_subprocess_attempts"] and not (upstream==(1,1,192)
            and value["prerequisite_artifact_loads"]==value["pin_model_contexts"]==2):return False
    if not partial:return value==checkpoint_counts(completed,stage)
    expected=checkpoint_counts(completed,"task_completed" if completed else "p3_authenticated")
    if value["p3_artifact_loads"] not in (0,1) or value["p3_disk_recertifications"] not in (0,1):return False
    if value["p3_independent_pin_replays"] not in (0,192):return False
    if not 0<=value["prerequisite_artifact_loads"]<=2 or not 0<=value["pin_model_contexts"]<=2:return False
    if not completed<=value["worker_subprocess_attempts"]<=min(12,completed+1):return False
    if not completed<=value["worker_subprocess_successes"]<=min(12,completed+1):return False
    if value["worker_subprocess_successes"]>value["worker_subprocess_attempts"]:return False
    successes=value["worker_subprocess_successes"]
    constructor_extra=value["b16_constructors"]-completed
    sim_extra=value["sim_forward_calls"]-completed*6080
    tool_extra=value["tool_position_calls"]-completed*6081
    if constructor_extra not in (0,1) or not 0<=sim_extra<=6080 or not 0<=tool_extra<=6081:return False
    if constructor_extra==0 and (sim_extra or tool_extra):return False
    if sim_extra<6080 and tool_extra:return False
    if successes==completed+1 and not (constructor_extra==1 and sim_extra==6080 and tool_extra==6081):return False
    return all(value[k]>=expected[k] for k in ("p3_artifact_loads","p3_disk_recertifications",
        "p3_independent_pin_replays")) if completed else True

def operational(trigger,finish=None):
    if finish is None:finish=trigger
    return {"trigger_elapsed_s":float(trigger),"observed_finish_elapsed_s":float(finish),
    "campaign_wall_limit_s":CAMPAIGN_WALL_LIMIT_S,"worker_wall_limit_s":WORKER_WALL_LIMIT_S,
    "timing_evidence":False}
def certify_operational(value):return bool(set(value)=={"trigger_elapsed_s",
    "observed_finish_elapsed_s","campaign_wall_limit_s","worker_wall_limit_s","timing_evidence"}
    and all(isinstance(value[k],(int,float)) and np.isfinite(value[k])
            for k in ("trigger_elapsed_s","observed_finish_elapsed_s"))
    and 0<=value["trigger_elapsed_s"]<=CAMPAIGN_WALL_LIMIT_S
    and value["observed_finish_elapsed_s"]>=value["trigger_elapsed_s"]
    and value["campaign_wall_limit_s"]==CAMPAIGN_WALL_LIMIT_S
    and value["worker_wall_limit_s"]==WORKER_WALL_LIMIT_S and value["timing_evidence"] is False)

def checkpoint(generation,stage,completed,artifacts,provenance,p3_authentication=None,
               call_counts=None,elapsed_s=0.):
    return {"protocol":PROTOCOL,"generation":generation,"stage":stage,"incomplete":True,
        "completed":completed,"pending":TASKS-completed,"worker_artifacts":copy.deepcopy(artifacts),
        "call_counts":copy.deepcopy(call_counts if call_counts is not None
            else checkpoint_counts(completed,stage)),"provenance":copy.deepcopy(provenance),
        "p3_authentication":copy.deepcopy(p3_authentication),
        "operational":operational(elapsed_s),
        "sqp_evidence":False,"benchmark_evidence":False}
def certify_artifact(value,index):
    paths=worker_paths(index)
    return bool(set(value)=={"task_index","request_path","request_sha256","input_path",
        "input_sha256","json_path","json_sha256","npz_path","npz_sha256"}
        and value["task_index"]==index
        and all(value[f"{key}_path"]==str(paths[key]) and isinstance(value[f"{key}_sha256"],str)
            and len(value[f"{key}_sha256"])==64 for key in ("request","input","json","npz")))
def certify_checkpoint(value,final=False):
    c=value.get("completed");stage=value.get("stage");g=value.get("generation")
    expected=(stage=="gen0" and g==0 and c==0) or (stage=="p3_authenticated" and g==1 and c==0) \
        or (stage=="task_completed" and g==c+1 and isinstance(c,int) and 1<=c<=12) \
        or (stage=="honest_end" and g==14 and c==12)
    return bool(set(value)=={"protocol","generation","stage","incomplete","completed","pending",
        "worker_artifacts","call_counts","provenance","p3_authentication",
        "operational","sqp_evidence","benchmark_evidence"}
        and value.get("protocol")==PROTOCOL and value.get("incomplete") is True and expected
        and value.get("pending")==TASKS-c and len(value.get("worker_artifacts",[]))==c
        and all(certify_artifact(a,i) for i,a in enumerate(value.get("worker_artifacts",[])))
        and certify_counts(value.get("call_counts",{}),c,stage) and value.get("sqp_evidence") is False
        and ((stage=="gen0" and value.get("p3_authentication") is None)
            or (stage!="gen0" and certify_p3_authentication(
                value.get("p3_authentication",{}))))
        and value.get("benchmark_evidence") is False and certify_provenance(value["provenance"],final)
        and certify_operational(value.get("operational",{})))
def publish_checkpoint(output,value):
    if not certify_checkpoint(value,value["stage"]=="honest_end"):raise RuntimeError("P4 checkpoint invalid")
    path=output.with_name(f"p4.partial.gen{value['generation']:02d}.json");atomic_json(path,value)
    atomic_json(output.with_name("p4.partial.latest.json"),{"protocol":PROTOCOL,"incomplete":True,
        "generation":value["generation"],"json_path":str(path),"json_sha256":sha(path)});return path

def row_files(task_index,p3_document):
    from gato_tiago.circular_portfolio_p3_runner import certify_row_summary
    manifest=json.loads(p3_paths()["manifest"].read_text())
    side={item["path"]:item["sha256"] for item in manifest["side_artifacts"]}
    rows=[]
    for lane in range(16):
        idx=task_index*16+lane;summary_path=P3_OUTPUT.with_name(f"p3.row.{idx:03d}.json")
        summary=json.loads(summary_path.read_text());npz=Path(summary["npz_path"])
        with np.load(npz,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
        if (summary!=p3_document["rows"][idx] or summary.get("identity")!=list(LEDGER[idx])
                or summary.get("npz_path")!=str(npz)
                or side.get(str(summary_path))!=sha(summary_path) or side.get(str(npz))!=sha(npz)
                or not certify_row_summary(summary,arrays,LEDGER[idx],npz)):
            raise RuntimeError("P3 row identity/hash/schema/certificate mismatch")
        rows.append((summary,arrays))
    return rows

def worker_inputs(rows):
    x0=np.stack([a["pin_dense_state_float64"][0].astype(np.float32) for _s,a in rows])
    controls=np.stack([a["applied_controls_float32"] for _s,a in rows])
    seeds=[]
    for _summary,a in rows:seeds.append(pack_solver_seed(a["planned_q_float64"],
        a["planned_qd_float64"],a["applied_controls_float32"]))
    seeds=np.stack(seeds)
    result={"x0_float32":x0,"controls_float32":controls,
        "b1_seed_xu_float32":seeds[0],"b16_seed_xu_float32":seeds}
    if not input_schema(result):raise RuntimeError("P4 input construction failed")
    return result

def run_worker(task_index,arrays,deadline,monotonic=time.monotonic,execution_counts=None): # pragma: no cover
    paths=worker_paths(task_index)
    if any(v.exists() for v in paths.values()):raise FileExistsError("P4 worker artifact exists")
    atomic_npz(paths["input"],arrays)
    request={"protocol":WORKER_PROTOCOL,"task_index":task_index,"input_path":str(paths["input"]),
        "input_sha256":sha(paths["input"]),"output_path":str(paths["json"]),"extension":EXTENSION}
    atomic_json(paths["request"],request)
    command=["python","-B","-m","gato_tiago.circular_portfolio_p4_worker","--execute",
        "--request",str(paths["request"]),"--output",str(paths["json"])]
    remaining=deadline-monotonic()
    if remaining<=0:raise TimeoutError("P4 campaign wall limit")
    if execution_counts is not None:execution_counts["worker_subprocess_attempts"]+=1
    try:
        result=subprocess.run(command,cwd=AUTHORIZED_CWD,env=os.environ.copy(),
            timeout=min(WORKER_WALL_LIMIT_S,remaining))
    except subprocess.TimeoutExpired as source:
        error=TimeoutError("P4 worker wall limit")
        error.worker_execution={"task_index":task_index,"status":"killed_internal_unknown",
            "internal_counts_known":False,"constructor_calls_lower_bound":0,
            "sim_forward_calls_lower_bound":0,"tool_position_calls_lower_bound":0}
        error.trigger_elapsed=min(WORKER_WALL_LIMIT_S,remaining)
        raise error from source
    if result.returncode:
        rejected=paths["json"].with_name(paths["json"].stem+".rejected.json")
        detail=None
        if rejected.is_file():
            detail=json.loads(rejected.read_text())
            if not certify_worker_rejection(detail,task_index):raise RuntimeError("worker rejection invalid")
            if execution_counts is not None:execution_counts.update({
                "b16_constructors":execution_counts["b16_constructors"]+detail["constructor_calls"],
                "sim_forward_calls":execution_counts["sim_forward_calls"]+detail["sim_forward_calls"],
                "tool_position_calls":execution_counts["tool_position_calls"]+detail["tool_position_calls"]})
        error=RuntimeError(f"P4 worker exited {result.returncode}")
        error.worker_execution={"task_index":task_index,
            "status":"known_partial" if detail is not None else "abnormal_exit_internal_unknown",
            "internal_counts_known":detail is not None,
            "constructor_calls_lower_bound":detail["constructor_calls"] if detail is not None else 0,
            "sim_forward_calls_lower_bound":detail["sim_forward_calls"] if detail is not None else 0,
            "tool_position_calls_lower_bound":detail["tool_position_calls"] if detail is not None else 0}
        raise error
    try:
        summary=json.loads(paths["json"].read_text())
        with np.load(paths["npz"],allow_pickle=False) as archive:
            output={k:archive[k] for k in archive.files}
        if not certify_output(summary,output,request,arrays,task_index)["passes"]:
            raise RuntimeError("P4 worker boundary failed")
    except Exception as source:
        error=RuntimeError("P4 exit-zero worker output unavailable or invalid")
        error.worker_execution={"task_index":task_index,"status":"exit0_internal_unknown",
            "internal_counts_known":False,"constructor_calls_lower_bound":0,
            "sim_forward_calls_lower_bound":0,"tool_position_calls_lower_bound":0}
        raise error from source
    if execution_counts is not None:execution_counts.update({
        "worker_subprocess_successes":execution_counts["worker_subprocess_successes"]+1,
        "b16_constructors":execution_counts["b16_constructors"]+1,
        "sim_forward_calls":execution_counts["sim_forward_calls"]+6080,
        "tool_position_calls":execution_counts["tool_position_calls"]+6081})
    return summary,output,request

def file_status(path):
    path=Path(path);present=path.is_file()
    return {"path":str(path),"present":present,
        "sha256":sha(path) if present else None,"size_bytes":path.stat().st_size if present else None}

def active_artifact_map(task_index):
    if task_index>=12:return None
    paths=worker_paths(task_index);rejected=paths["json"].with_name(paths["json"].stem+".rejected.json")
    result={key:file_status(paths[key]) for key in ("request","input","json","npz")}
    result["rejected"]=file_status(rejected)
    for key in ("request","input","json","npz"):
        result[key+"_candidate"]=file_status(Path(str(paths[key])+".candidate"))
    return result

def publication_artifact_map(output):
    paths={"json":output,"manifest":output.with_name("p4.manifest.json"),
        "pointer":output.with_name("p4.partial.latest.json"),
        "json_candidate":Path(str(output)+".candidate"),
        "manifest_candidate":Path(str(output.with_name("p4.manifest.json"))+".candidate"),
        "pointer_candidate":Path(str(output.with_name("p4.partial.latest.json"))+".candidate")}
    result={key:file_status(path) for key,path in paths.items()}
    for key,path in paths.items():
        if not path.is_file():result[key]["classification"]="absent"
        else:
            try:
                json.loads(path.read_text());result[key]["classification"]="complete_json"
            except (UnicodeDecodeError,json.JSONDecodeError):
                result[key]["classification"]=("incomplete_candidate_non_evidence"
                    if key.endswith("candidate") else "malformed_evidence")
    return result

def rejection_document(stage,completed,artifacts,tasks,provenance,call_counts,p3_auth,error,
                       checkpoint_count,trigger_elapsed_s,finish_elapsed_s,active_status,
                       publication_status):
    return {"protocol":PROTOCOL,"stage":stage,"incomplete":True,"completed":completed,
        "pending":12-completed,"worker_artifacts":copy.deepcopy(artifacts),
        "tasks":copy.deepcopy(tasks),"p3_authentication":copy.deepcopy(p3_auth),
        "call_counts":copy.deepcopy(call_counts),"error_type":type(error).__name__,
        "error_message":str(error),"provenance":copy.deepcopy(provenance),
        "checkpoint_count":checkpoint_count,
        "operational":operational(trigger_elapsed_s,finish_elapsed_s),
        "active_worker":active_status,"publication_artifacts":publication_status,
        "active_worker_execution":copy.deepcopy(getattr(error,"worker_execution",None)),
        "sqp_evidence":False,"benchmark_evidence":False}
def certify_rejection(value):
    keys={"protocol","stage","incomplete","completed","pending","worker_artifacts","tasks",
        "p3_authentication","call_counts","error_type","error_message","provenance",
        "checkpoint_count","operational","active_worker","publication_artifacts",
        "active_worker_execution",
        "sqp_evidence","benchmark_evidence"}
    completed=value.get("completed")
    phase=tuple(value.get("call_counts",{}).get(k) for k in ("p3_artifact_loads",
        "p3_disk_recertifications","p3_independent_pin_replays"))
    return bool(set(value)==keys and value.get("protocol")==PROTOCOL
        and value.get("stage") in ("replay_failed","runtime_watchdog_rejected",
            "final_certification_failed","publication_failed")
        and value.get("incomplete") is True and isinstance(completed,int) and 0<=completed<=12
        and value.get("pending")==12-completed and len(value.get("worker_artifacts",[]))==completed
        and len(value.get("tasks",[]))==completed
        and phase in ((0,0,0),(1,1,192))
        and ((phase==(0,0,0) and value.get("p3_authentication") is None)
            or (phase==(1,1,192) and certify_p3_authentication(value.get("p3_authentication",{}))))
        and all(certify_artifact(a,i) for i,a in enumerate(value.get("worker_artifacts",[])))
        and certify_counts(value.get("call_counts",{}),completed,value["stage"],True)
        and isinstance(value.get("error_type"),str) and isinstance(value.get("error_message"),str)
        and certify_provenance(value.get("provenance",{}),True)
        and isinstance(value.get("checkpoint_count"),int) and 1<=value["checkpoint_count"]<=15
        and certify_operational(value.get("operational",{}))
        and value.get("active_worker")==active_artifact_map(completed)
        and value.get("publication_artifacts")==publication_artifact_map(OUTPUT)
        and certify_active_worker_execution(value.get("active_worker_execution"),completed,
            value["call_counts"])
        and value.get("sqp_evidence") is False and value.get("benchmark_evidence") is False)

def certify_active_worker_execution(value,completed,counts_value):
    attempts=counts_value["worker_subprocess_attempts"]
    if attempts==completed:return value is None
    if attempts!=completed+1 or not isinstance(value,dict):return False
    keys={"task_index","status","internal_counts_known","constructor_calls_lower_bound",
        "sim_forward_calls_lower_bound","tool_position_calls_lower_bound"}
    return bool(set(value)==keys and value["task_index"]==completed
        and value["status"] in {"killed_internal_unknown","abnormal_exit_internal_unknown",
            "exit0_internal_unknown","known_partial","known_complete"}
        and isinstance(value["internal_counts_known"],bool)
        and all(isinstance(value[k],int) and value[k]>=0 for k in keys-{"task_index","status",
            "internal_counts_known"}) and ((value["status"] in {
            "killed_internal_unknown","abnormal_exit_internal_unknown",
            "exit0_internal_unknown"} and value["internal_counts_known"] is False
            and value["constructor_calls_lower_bound"]==value["sim_forward_calls_lower_bound"]
                ==value["tool_position_calls_lower_bound"]==0)
        or (value["status"] in {"known_partial","known_complete"}
            and value["internal_counts_known"] is True
            and value["constructor_calls_lower_bound"]==counts_value["b16_constructors"]-completed
            and value["sim_forward_calls_lower_bound"]==counts_value["sim_forward_calls"]-completed*6080
            and value["tool_position_calls_lower_bound"]==counts_value["tool_position_calls"]-completed*6081)))

def recertify_retained_failure(output=OUTPUT):
    try:
        output=Path(output);rejected=output.with_name("p4.rejected.json")
        pointer=output.with_name("p4.rejected.latest.json")
        value=json.loads(rejected.read_text());latest=json.loads(pointer.read_text())
        expected={"protocol":PROTOCOL,"incomplete":True,"stage":value["stage"],
            "json_path":str(rejected),"json_sha256":sha(rejected)}
        if not certify_rejection(value) or latest!=expected:return {"passes":False}
        publication=value["publication_artifacts"]
        if publication["json"]["classification"]=="complete_json":
            if not certify_final(json.loads(Path(publication["json"]["path"]).read_text())):
                return {"passes":False}
        elif publication["json"]["classification"]=="malformed_evidence":return {"passes":False}
        if publication["json_candidate"]["classification"]=="complete_json":
            if not certify_final(json.loads(Path(publication["json_candidate"]["path"]).read_text())):
                return {"passes":False}
        p3_document=None
        if value["p3_authentication"] is not None:
            fresh_auth,p3_document=authenticate_p3()
            if value["p3_authentication"]!=fresh_auth:return {"passes":False}
        if not all(certify_task_document(task,index) for index,task in enumerate(value["tasks"])):
            return {"passes":False}
        if value["tasks"]:
            from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite,_production_pin_context
            prereq_auth,prerequisite=authenticate_cpu_prerequisite()
            if prereq_auth["recertification"]["passes"] is not True:return {"passes":False}
            _model,kinematics,_rnea,_rd,_aba,_velocity,_dense=_production_pin_context()
            tool_velocity=lambda q,qd:kinematics(q)[1]@qd
        for index,artifact in enumerate(value["worker_artifacts"]):
            paths=worker_paths(index)
            if artifact!={"task_index":index,"request_path":str(paths["request"]),
                "request_sha256":sha(paths["request"]),"input_path":str(paths["input"]),
                "input_sha256":sha(paths["input"]),"json_path":str(paths["json"]),
                "json_sha256":sha(paths["json"]),"npz_path":str(paths["npz"]),
                "npz_sha256":sha(paths["npz"])}:return {"passes":False}
            request=json.loads(paths["request"].read_text());summary=json.loads(paths["json"].read_text())
            with np.load(paths["input"],allow_pickle=False) as archive:inputs={k:archive[k] for k in archive.files}
            with np.load(paths["npz"],allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
            if not certify_output(summary,arrays,request,inputs,index)["passes"]:return {"passes":False}
            rows=row_files(index,p3_document);expected_inputs=worker_inputs(rows)
            if not all(np.array_equal(inputs[k],expected_inputs[k]) for k in inputs):return {"passes":False}
            fresh=certify_task_output(index,p3_document,prerequisite,rows,inputs,arrays,tool_velocity)
            if json_safe(fresh)!=value["tasks"][index]:return {"passes":False}
        side=[]
        for generation in range(value["checkpoint_count"]):
            path=output.with_name(f"p4.partial.gen{generation:02d}.json")
            retained=json.loads(path.read_text())
            if not certify_checkpoint(retained,generation==14):return {"passes":False}
            expected_completed=0 if generation<2 else generation-1
            if retained["worker_artifacts"]!=value["worker_artifacts"][:expected_completed]:
                return {"passes":False}
            if generation==0:
                if retained["p3_authentication"] is not None:return {"passes":False}
            elif retained["p3_authentication"]!=value["p3_authentication"]:return {"passes":False}
            side.append({"path":str(path),"sha256":sha(path)})
        for artifact in value["worker_artifacts"]:
            for key in ("request","input","json","npz"):
                side.append({"path":artifact[f"{key}_path"],"sha256":artifact[f"{key}_sha256"]})
        if value["publication_artifacts"]!=publication_artifact_map(output):return {"passes":False}
        publication=value["publication_artifacts"]
        json_status=publication["json"] if publication["json"]["present"] \
            else publication["json_candidate"]
        expected_manifest={"protocol":PROTOCOL,"json_path":str(output),
            "json_sha256":json_status["sha256"],"side_artifacts":side}
        for key in ("manifest","manifest_candidate"):
            status=publication[key]
            if status["classification"]=="complete_json":
                if json.loads(Path(status["path"]).read_text())!=expected_manifest:return {"passes":False}
            elif status["classification"]=="malformed_evidence":return {"passes":False}
        latest_generation=value["checkpoint_count"]-1
        latest_checkpoint=output.with_name(f"p4.partial.gen{latest_generation:02d}.json")
        checkpoint_pointer={"protocol":PROTOCOL,"incomplete":True,
            "generation":latest_generation,"json_path":str(latest_checkpoint),
            "json_sha256":sha(latest_checkpoint)}
        manifest_status=publication["manifest"] if publication["manifest"]["present"] \
            else publication["manifest_candidate"]
        success_pointer={"protocol":PROTOCOL,"incomplete":False,"json_path":str(output),
            "json_sha256":json_status["sha256"],"manifest_path":str(output.with_name("p4.manifest.json")),
            "manifest_sha256":manifest_status["sha256"]}
        if publication["pointer"]["classification"]=="complete_json":
            retained_pointer=json.loads(Path(publication["pointer"]["path"]).read_text())
            if retained_pointer not in (checkpoint_pointer,success_pointer):return {"passes":False}
        elif publication["pointer"]["classification"]=="malformed_evidence":return {"passes":False}
        if publication["pointer_candidate"]["classification"]=="complete_json":
            if json.loads(Path(publication["pointer_candidate"]["path"]).read_text())!=success_pointer:
                return {"passes":False}
        # A failed active worker may retain only its identity-derived request,
        # input, rejection JSON, or unpublished NPZ; these are non-evidence.
        active=worker_paths(value["completed"] ) if value["completed"]<12 else None
        if active:
            if value["active_worker"]!=active_artifact_map(value["completed"]):return {"passes":False}
            allowed=set(active.values())|{active["json"].with_name(active["json"].stem+".rejected.json")}
            allowed|={Path(str(path)+".candidate") for path in active.values()}
            if any(path.name.startswith(f"p4.worker.{value['completed']:02d}") and path not in allowed
                   for path in output.parent.iterdir()):return {"passes":False}
            worker_rejected=active["json"].with_name(active["json"].stem+".rejected.json")
            if worker_rejected.is_file():
                detail=json.loads(worker_rejected.read_text())
                if not certify_worker_rejection(detail,value["completed"]):return {"passes":False}
            if active["request"].is_file():
                request=json.loads(active["request"].read_text())
                expected_request={"protocol":WORKER_PROTOCOL,"task_index":value["completed"],
                    "input_path":str(active["input"]),"input_sha256":sha(active["input"]),
                    "output_path":str(active["json"]),"extension":EXTENSION}
                if request!=expected_request:return {"passes":False}
            if active["input"].is_file():
                with np.load(active["input"],allow_pickle=False) as archive:
                    inputs={k:archive[k] for k in archive.files}
                if not input_schema(inputs):return {"passes":False}
                expected_inputs=worker_inputs(row_files(value["completed"],p3_document))
                if set(inputs)!=set(expected_inputs) or not all(
                        np.array_equal(inputs[k],expected_inputs[k]) for k in inputs):return {"passes":False}
            if active["npz"].is_file():
                with np.load(active["npz"],allow_pickle=False) as archive:
                    arrays={k:archive[k] for k in archive.files}
                if not output_schema(arrays) or not np.array_equal(
                        arrays["captured_x0_float32"],inputs["x0_float32"]) or not np.array_equal(
                        arrays["captured_controls_float32"],inputs["controls_float32"]):
                    return {"passes":False}
            if active["json"].is_file() and active["npz"].is_file():
                summary=json.loads(active["json"].read_text())
                if not certify_output(summary,arrays,request,inputs,value["completed"])["passes"]:
                    return {"passes":False}
            for key in ("request","json"):
                candidate=Path(str(active[key])+".candidate")
                if candidate.is_file():
                    try:json.loads(candidate.read_text())
                    except json.JSONDecodeError:pass
            for key in ("input","npz"):
                candidate=Path(str(active[key])+".candidate")
                if candidate.is_file():
                    try:
                        with np.load(candidate,allow_pickle=False) as archive:
                            candidate_arrays={k:archive[k] for k in archive.files}
                        if key=="input" and not input_schema(candidate_arrays):return {"passes":False}
                        if key=="npz" and not output_schema(candidate_arrays):return {"passes":False}
                    except (OSError,ValueError):pass
        return {"passes":True}
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):return {"passes":False}

def certify_task_output(task_index,p3_document,prerequisite,rows,inputs,worker_arrays,tool_velocity):
    geometry=p3_document["task_geometries"][task_index];ref=np.tile(
        np.asarray(geometry["reference_float32"],np.float32),(96,1)).ravel()
    lanes=[]
    for lane,(_summary,a) in enumerate(rows):
        lanes.append(certify_lane(a["pin_dense_state_float64"],a["pin_dense_tool_float64"],
            worker_arrays["cuda_dense_state_float32"][lane],worker_arrays["cuda_dense_tool_float32"][lane],
            inputs["controls_float32"][lane],prerequisite["q8_tool_float64"][task_index],
            geometry["pillar_float64"],ref,prerequisite["joint_lower_float64"],
            prerequisite["joint_upper_float64"],prerequisite["velocity_limit_float64"],
            prerequisite["effort_limit_float64"],tool_velocity,"short" if lane<8 else "long",
            int(prerequisite["public_default_side_int8"][task_index]),
            _summary["certificate"]))
    aggregate=certify_task(lanes,inputs["controls_float32"],
        worker_arrays["cuda_dense_tool_float32"])
    return {"task_index":task_index,"identity":list(TASK_IDENTITIES[task_index]),
        "lanes":json_safe(lanes),"aggregate":json_safe(aggregate),"passes":aggregate["passes"]}

def certify_task_document(value,task_index):
    lane_keys={"gates","speed_m_s","minimum_clearance_m","maximum_tool_disagreement_m",
        "maximum_q_violation","maximum_velocity_ratio","maximum_effort_ratio","terminal_error_m",
        "pin_base_cost","pin_full_cost","cuda_base_cost","cuda_full_cost","pin_turn","cuda_turn",
        "passes"}
    gate_keys={"finite","initial","q_limits","velocity_limits","effort_limits","terminal",
        "speed","clearance","model_tool_agreement","turn_agreement","cost_agreement",
        "accepted_p3_binding"}
    aggregate=value.get("aggregate",{});portfolio=aggregate.get("portfolio_gates",{})
    return bool(set(value)=={"task_index","identity","lanes","aggregate","passes"}
        and value["task_index"]==task_index and value["identity"]==list(TASK_IDENTITIES[task_index])
        and value["passes"] is True and len(value["lanes"])==16
        and all(set(lane)==lane_keys and set(lane["gates"])==gate_keys
            and lane["passes"] is True and all(lane["gates"].values())
            and all(isinstance(lane[k],(int,float)) and np.isfinite(lane[k])
                for k in lane_keys-{"gates","passes"}) for lane in value["lanes"])
        and set(aggregate)=={"pairs","portfolio_gates","minimum_within_family_tool_rms_m",
            "minimum_cross_family_max_separation_m","passes"}
        and len(aggregate["pairs"])==8 and all(pair.get("passes") is True
            and set(pair)=={"gates","passes"}
            and set(pair["gates"])=={"pin_base","cuda_base","pin_reversal","cuda_reversal",
                "topology"} and all(pair["gates"].values())
            for pair in aggregate["pairs"])
        and set(portfolio)=={"unique_control_bytes","unique_cuda_path_bytes",
            "within_family_path_separation","cross_family_separation","cross_family_topology"}
        and all(portfolio.values())
        and isinstance(aggregate["minimum_within_family_tool_rms_m"],(int,float))
        and np.isfinite(aggregate["minimum_within_family_tool_rms_m"])
        and aggregate["minimum_within_family_tool_rms_m"]>=.001
        and isinstance(aggregate["minimum_cross_family_max_separation_m"],(int,float))
        and np.isfinite(aggregate["minimum_cross_family_max_separation_m"])
        and aggregate["minimum_cross_family_max_separation_m"]>=.075
        and aggregate["passes"] is True)

def owned_semantic_recert(p3_document,prerequisite,tool_velocity):
    fresh=[];artifacts=[]
    for task_index in range(12):
        paths=worker_paths(task_index);request=json.loads(paths["request"].read_text())
        summary=json.loads(paths["json"].read_text())
        with np.load(paths["input"],allow_pickle=False) as archive:inputs={k:archive[k] for k in archive.files}
        with np.load(paths["npz"],allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
        if not certify_output(summary,arrays,request,inputs,task_index)["passes"]:
            return {"passes":False}
        rows=row_files(task_index,p3_document);expected=worker_inputs(rows)
        if set(inputs)!=set(expected) or not all(np.array_equal(inputs[k],expected[k]) for k in inputs):
            return {"passes":False}
        task=certify_task_output(task_index,p3_document,prerequisite,rows,inputs,arrays,tool_velocity)
        if not certify_task_document(task,task_index):return {"passes":False}
        fresh.append(json_safe(task));artifacts.append({"task_index":task_index,
            "request_path":str(paths["request"]),"request_sha256":sha(paths["request"]),
            "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
            "json_path":str(paths["json"]),"json_sha256":sha(paths["json"]),
            "npz_path":str(paths["npz"]),"npz_sha256":sha(paths["npz"])})
    return {"tasks":fresh,"worker_artifacts":artifacts,"passes":True}

def certify_final(value):
    return bool(set(value)=={"protocol","incomplete","overall_pass","p3_authentication",
        "tasks","worker_artifacts","owned_semantic_recertification","call_counts",
        "checkpoint_count","provenance","operational","sqp_evidence","benchmark_evidence"}
        and value.get("protocol")==PROTOCOL
        and value.get("incomplete") is False and value.get("overall_pass") is True
        and certify_p3_authentication(value.get("p3_authentication",{}))
        and len(value.get("tasks",[]))==12
        and all(certify_task_document(x,i) for i,x in enumerate(value["tasks"]))
        and len(value.get("worker_artifacts",[]))==12
        and set(value.get("owned_semantic_recertification",{}))=={"tasks","worker_artifacts","passes"}
        and value["owned_semantic_recertification"].get("passes") is True
        and value["owned_semantic_recertification"].get("tasks")==value["tasks"]
        and value["owned_semantic_recertification"].get("worker_artifacts")==value["worker_artifacts"]
        and value.get("call_counts")==checkpoint_counts(12,"honest_end")
        and value.get("checkpoint_count")==15 and certify_provenance(value["provenance"],True)
        and certify_operational(value.get("operational",{}))
        and value.get("sqp_evidence") is False and value.get("benchmark_evidence") is False)

def recertify_retained_p4(output=OUTPUT): # pragma: no cover - independent post-run audit
    try:
        output=Path(output).resolve();document=json.loads(output.read_text())
        if not certify_final(document):return {"passes":False}
        fresh_auth,p3_document=authenticate_p3()
        if document["p3_authentication"]!=fresh_auth:return {"passes":False}
        from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite,_production_pin_context
        prereq_auth,prerequisite=authenticate_cpu_prerequisite()
        if prereq_auth["recertification"]["passes"] is not True:return {"passes":False}
        _model,kinematics,_rnea,_rd,_aba,_velocity,_dense=_production_pin_context()
        tool_velocity=lambda q,qd:kinematics(q)[1]@qd
        fresh_tasks=[];fresh_artifacts=[]
        for task_index in range(12):
            paths=worker_paths(task_index);request=json.loads(paths["request"].read_text())
            summary=json.loads(paths["json"].read_text())
            with np.load(paths["input"],allow_pickle=False) as archive:
                inputs={k:archive[k] for k in archive.files}
            with np.load(paths["npz"],allow_pickle=False) as archive:
                arrays={k:archive[k] for k in archive.files}
            if not certify_output(summary,arrays,request,inputs,task_index)["passes"]:
                return {"passes":False}
            rows=row_files(task_index,p3_document);expected_inputs=worker_inputs(rows)
            if set(inputs)!=set(expected_inputs) or not all(
                    np.array_equal(inputs[k],expected_inputs[k]) for k in inputs):return {"passes":False}
            task=certify_task_output(task_index,p3_document,prerequisite,rows,inputs,arrays,tool_velocity)
            if json_safe(task)!=document["tasks"][task_index]:return {"passes":False}
            fresh_tasks.append(task);fresh_artifacts.append({"task_index":task_index,
                "request_path":str(paths["request"]),"request_sha256":sha(paths["request"]),
                "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
                "json_path":str(paths["json"]),"json_sha256":sha(paths["json"]),
                "npz_path":str(paths["npz"]),"npz_sha256":sha(paths["npz"])})
        if document["worker_artifacts"]!=fresh_artifacts:return {"passes":False}
        owned={"tasks":fresh_tasks,"worker_artifacts":fresh_artifacts,"passes":True}
        if document["owned_semantic_recertification"]!=owned:return {"passes":False}
        side=[]
        for generation in range(15):
            path=output.with_name(f"p4.partial.gen{generation:02d}.json")
            retained=json.loads(path.read_text())
            if not certify_checkpoint(retained,generation==14):return {"passes":False}
            expected_completed=0 if generation<2 else min(generation-1,12)
            if retained["worker_artifacts"]!=fresh_artifacts[:expected_completed]:return {"passes":False}
            if generation==0:
                if retained["p3_authentication"] is not None:return {"passes":False}
            elif retained["p3_authentication"]!=fresh_auth:return {"passes":False}
            side.append({"path":str(path),"sha256":sha(path)})
        for artifact in fresh_artifacts:
            for key in ("request","input","json","npz"):
                side.append({"path":artifact[f"{key}_path"],"sha256":artifact[f"{key}_sha256"]})
        manifest_path=output.with_name("p4.manifest.json")
        manifest=json.loads(manifest_path.read_text())
        expected_manifest={"protocol":PROTOCOL,"json_path":str(output),"json_sha256":sha(output),
            "side_artifacts":side}
        pointer_path=output.with_name("p4.partial.latest.json")
        pointer=json.loads(pointer_path.read_text())
        expected_pointer={"protocol":PROTOCOL,"incomplete":False,"json_path":str(output),
            "json_sha256":sha(output),"manifest_path":str(manifest_path),
            "manifest_sha256":sha(manifest_path)}
        return {"passes":manifest==expected_manifest and pointer==expected_pointer}
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError,RuntimeError):
        return {"passes":False}

def refuse_existing(output):
    paths=[output,output.with_name("p4.manifest.json"),output.with_name("p4.partial.latest.json"),
        output.with_name("p4.rejected.latest.json")]
    paths.extend(path for i in range(12) for path in worker_paths(i).values())
    if output.parent.exists() and (any(path.exists() for path in paths)
            or any(path.name.startswith("p4.") or ".candidate" in path.name
                   for path in output.parent.iterdir())):
        raise FileExistsError("P4 no-overwrite boundary")

def execute(output=OUTPUT,authorization=None,monotonic=time.monotonic): # pragma: no cover
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P4 CUDA replay blocked")
    output=Path(output).resolve()
    if output!=OUTPUT:raise RuntimeError("wrong P4 output")
    refuse_existing(output);output.parent.mkdir(parents=True);started=monotonic()
    deadline=started+CAMPAIGN_WALL_LIMIT_S;provenance=start_provenance()
    checkpoints=[];artifacts=[];tasks=[];execution_counts=new_counts();p3_auth=None
    failure_stage="replay_failed"
    try:
        checkpoints.append(publish_checkpoint(output,checkpoint(0,"gen0",0,artifacts,provenance,
            call_counts=execution_counts,elapsed_s=monotonic()-started)))
        p3_auth,p3_document=authenticate_p3();execution_counts.update({"p3_artifact_loads":1,
            "p3_disk_recertifications":1,"p3_independent_pin_replays":192,
            "prerequisite_artifact_loads":1,"pin_model_contexts":1})
        checkpoints.append(publish_checkpoint(output,checkpoint(1,"p3_authenticated",0,artifacts,
            provenance,p3_auth,execution_counts,monotonic()-started)))
        from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite,_production_pin_context
        prereq_auth,prerequisite=authenticate_cpu_prerequisite();execution_counts["prerequisite_artifact_loads"]+=1
        if prereq_auth["recertification"]["passes"] is not True:raise RuntimeError("prerequisite failed")
        _model,kinematics,_rnea,_rd,_aba,_velocity,_dense=_production_pin_context();execution_counts["pin_model_contexts"]+=1
        tool_velocity=lambda q,qd:kinematics(q)[1]@qd
        for task_index in range(12):
            if monotonic()>=deadline:raise TimeoutError("P4 campaign wall limit")
            rows=row_files(task_index,p3_document);inputs=worker_inputs(rows)
            summary,worker_arrays,_request=run_worker(task_index,inputs,deadline,monotonic,execution_counts)
            known_complete={"task_index":task_index,"status":"known_complete",
                "internal_counts_known":True,
                "constructor_calls_lower_bound":1,"sim_forward_calls_lower_bound":6080,
                "tool_position_calls_lower_bound":6081}
            try:
                if monotonic()>=deadline:raise TimeoutError("P4 campaign wall limit")
                result=certify_task_output(task_index,p3_document,prerequisite,rows,inputs,
                    worker_arrays,tool_velocity)
                if not result["passes"]:raise RuntimeError("P4 task replay failed")
                paths=worker_paths(task_index);artifact={"task_index":task_index,
                    "request_path":str(paths["request"]),"request_sha256":sha(paths["request"]),
                    "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
                    "json_path":str(paths["json"]),"json_sha256":sha(paths["json"]),
                    "npz_path":str(paths["npz"]),"npz_sha256":sha(paths["npz"])}
            except Exception as error:
                error.worker_execution=known_complete
                if isinstance(error,TimeoutError):error.trigger_elapsed=CAMPAIGN_WALL_LIMIT_S
                raise
            tasks.append(result);artifacts.append(artifact)
            checkpoints.append(publish_checkpoint(output,checkpoint(task_index+2,"task_completed",
                task_index+1,artifacts,provenance,p3_auth,execution_counts,monotonic()-started)))
        failure_stage="final_certification_failed"
        owned=owned_semantic_recert(p3_document,prerequisite,tool_velocity)
        if not owned.get("passes") or owned["tasks"]!=tasks or owned["worker_artifacts"]!=artifacts:
            raise RuntimeError("P4 owned semantic recertification failed")
        if monotonic()>=deadline:raise TimeoutError("P4 campaign wall limit")
        finish_provenance(provenance)
        checkpoints.append(publish_checkpoint(output,checkpoint(14,"honest_end",12,artifacts,
            provenance,p3_auth,execution_counts,monotonic()-started)))
        final={"protocol":PROTOCOL,"incomplete":False,"overall_pass":True,
        "p3_authentication":p3_auth,"tasks":tasks,"worker_artifacts":artifacts,
        "owned_semantic_recertification":owned,
        "call_counts":execution_counts,"checkpoint_count":len(checkpoints),"provenance":provenance,
        "operational":operational(monotonic()-started),
        "sqp_evidence":False,"benchmark_evidence":False}
        if not certify_final(final):raise RuntimeError("P4 final certification failed")
        failure_stage="publication_failed";manifest=output.with_name("p4.manifest.json")
        side=[{"path":str(p),"sha256":sha(p)} for p in checkpoints]
        for a in artifacts:
            for key in ("request","input","json","npz"):side.append({"path":a[f"{key}_path"],
                "sha256":a[f"{key}_sha256"]})
        output_candidate=write_json_candidate(output,final)
        manifest_value={"protocol":PROTOCOL,"json_path":str(output),
            "json_sha256":sha(output_candidate),"side_artifacts":side}
        manifest_candidate=write_json_candidate(manifest,manifest_value)
        os.replace(output_candidate,output);os.replace(manifest_candidate,manifest)
        atomic_json(output.with_name("p4.partial.latest.json"),{"protocol":PROTOCOL,
        "incomplete":False,"json_path":str(output),"json_sha256":sha(output),
        "manifest_path":str(manifest),"manifest_sha256":sha(manifest)})
        return final
    except Exception as error:
        trigger=min(getattr(error,"trigger_elapsed",monotonic()-started),CAMPAIGN_WALL_LIMIT_S)
        finish_provenance(provenance);stage=("runtime_watchdog_rejected"
            if isinstance(error,(TimeoutError,subprocess.TimeoutExpired)) else failure_stage)
        finish_elapsed=monotonic()-started
        rejected=rejection_document(stage,len(artifacts),artifacts,tasks,provenance,
            execution_counts,p3_auth,error,len(checkpoints),trigger,finish_elapsed,
            active_artifact_map(len(artifacts)),publication_artifact_map(output))
        if not certify_rejection(rejected):raise RuntimeError("P4 rejection certificate failed") from error
        path=output.with_name("p4.rejected.json");atomic_json(path,rejected)
        atomic_json(output.with_name("p4.rejected.latest.json"),{"protocol":PROTOCOL,
            "incomplete":True,"stage":stage,"json_path":str(path),"json_sha256":sha(path)})
        raise

def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    execute(args.output,RUNNER_EXECUTION_AUTHORIZATION)
if __name__=="__main__":main()
