"""Isolated batch-16 CUDA replay worker for portfolio P4."""

from __future__ import annotations

import argparse,ctypes,hashlib,importlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
import numpy as np

from gato_tiago.circular_portfolio import (CONTROL_LIMIT_COST,CYLINDER_WEIGHT,DT,KKT_TOL,
    MAX_PCG_ITERS,MU,N_COST,PCG_TOL,Q_COST,QD_COST,Q_LIMIT_COST,RHO,SOLVE_RATIO,U_COST,
    VELOCITY_LIMIT_COST)
from gato_tiago.circular_portfolio_p4 import (EXTENSION,LANES,SIM_FORWARD_CALLS,
    TOOL_POSITION_CALLS,WORKER_PROTOCOL,array_hash,input_schema,output_schema,worker_paths)

WORKER_EXECUTION_AUTHORIZATION=None
AUTHORIZED_CWD="/workspace/GATO"
THREAD_ENV={"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
    "NUMEXPR_NUM_THREADS":"1"}
SOURCE_PATHS=("tiago_src/gato_tiago/circular_portfolio.py",
    "tiago_src/gato_tiago/circular_portfolio_p4.py",
    "tiago_src/gato_tiago/circular_portfolio_p4_worker.py","python/bindings.cu",
    "python/bsqp/interface.py","gato/bsqp/bsqp.cuh","gato/bsqp/kernels/tool_position.cuh",
    "gato/utils/cuda.cuh","gato/dynamics/integrator.cuh",
    "gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "gato/dynamics/tiago_right/tiago_right_grid.cuh")

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def cuda_diagnostics():
    runtime=ctypes.c_int();driver=ctypes.c_int()
    library=ctypes.CDLL("libcudart.so")
    if library.cudaRuntimeGetVersion(ctypes.byref(runtime)) or library.cudaDriverGetVersion(ctypes.byref(driver)):
        raise RuntimeError("CUDA version diagnostics failed")
    line=subprocess.run(["nvidia-smi","--query-gpu=uuid,name,driver_version,compute_cap",
        "--format=csv,noheader"],check=True,text=True,stdout=subprocess.PIPE).stdout.strip().splitlines()
    if len(line)!=1:raise RuntimeError("P4 worker requires exactly one visible CUDA device")
    fields=[value.strip() for value in line[0].split(",")]
    if len(fields)!=4:raise RuntimeError("CUDA device diagnostics malformed")
    return {"uuid":fields[0],"name":fields[1],"driver_version":fields[2],
        "compute_capability":fields[3],"cuda_runtime_version":runtime.value,
        "cuda_driver_api_version":driver.value}
def snapshot():
    root=Path(__file__).resolve().parents[2]
    run=lambda *a:subprocess.run(["git",*a],cwd=root,check=True,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
    extension=(root/EXTENSION["relative_path"]).resolve()
    return {"head":run("rev-parse","HEAD"),"clean":run("status","--porcelain","--untracked-files=no")=="",
        "sources":{name:sha(root/name) for name in SOURCE_PATHS},
        "extension":{"path":str(extension),"sha256":sha(extension),"size":extension.stat().st_size}}
def start_provenance():
    current=snapshot();environment={k:os.environ.get(k) for k in THREAD_ENV}
    if environment!=THREAD_ENV:raise RuntimeError("P4 worker requires single-thread environment")
    return {"cwd":str(Path.cwd().resolve()),"orig_argv":list(sys.orig_argv),
        "command":shlex.join(sys.orig_argv),"head_start":current["head"],"head_end":None,
        "clean_start":current["clean"],"clean_end":None,"sources_start":current["sources"],
        "sources_end":None,"extension_start":current["extension"],"extension_end":None,
        "thread_environment":environment}
def finish_provenance(value):
    current=snapshot();value.update({"head_end":current["head"],"clean_end":current["clean"],
        "sources_end":current["sources"],"extension_end":current["extension"]})
def expected_argv(task_index):
    paths=worker_paths(task_index)
    return ("python","-B","-m","gato_tiago.circular_portfolio_p4_worker","--execute",
        "--request",str(paths["request"]),"--output",str(paths["json"]))
def certify_provenance(value,task_index):
    current=snapshot();keys={"cwd","orig_argv","command","head_start","head_end","clean_start",
        "clean_end","sources_start","sources_end","extension_start","extension_end",
        "thread_environment"}
    frozen={"path":str((Path(__file__).resolve().parents[2]/EXTENSION["relative_path"]).resolve()),
        "sha256":EXTENSION["sha256"],"size":EXTENSION["size"]}
    return bool(set(value)==keys and value["cwd"]==AUTHORIZED_CWD
        and tuple(value["orig_argv"])==expected_argv(task_index)
        and value["command"]==shlex.join(expected_argv(task_index))
        and value["head_start"]==value["head_end"]==current["head"]
        and value["clean_start"] is value["clean_end"] is current["clean"] is True
        and value["sources_start"]==value["sources_end"]==current["sources"]
        and value["extension_start"]==value["extension_end"]==frozen
        and value["thread_environment"]==THREAD_ENV)
def atomic_npz(path,arrays):
    candidate=Path(str(path)+".candidate")
    with candidate.open("xb") as stream:np.savez(stream,**arrays)
    os.replace(candidate,path)
def atomic_json(path,value):
    candidate=Path(str(path)+".candidate")
    with candidate.open("x") as stream:json.dump(value,stream,sort_keys=True,separators=(",",":"));stream.write("\n")
    os.replace(candidate,path)

def certify_output(summary,arrays,request,inputs,task_index):
    paths=worker_paths(task_index)
    expected_keys={"protocol","task_index","request_path","request_sha256","input_path",
        "input_sha256","output_npz_path","output_npz_sha256","module","extension",
        "constructor_calls","sim_forward_calls","tool_position_calls","solve_calls",
        "sqp_calls","array_names","array_hashes","provenance","cuda_diagnostics",
        "operational","certificate"}
    expected_request={"protocol","task_index","input_path","input_sha256","output_path","extension"}
    try:
        root=Path(__file__).resolve().parents[2]
        extension_path=(root/EXTENSION["relative_path"]).resolve()
        files=(Path(summary.get("request_path","")).is_file()
            and sha(summary["request_path"])==summary.get("request_sha256")
            and Path(summary.get("input_path","")).is_file()
            and sha(summary["input_path"])==summary.get("input_sha256")
            and Path(summary.get("output_npz_path","")).is_file()
            and sha(summary["output_npz_path"])==summary.get("output_npz_sha256"))
        extension_live=(extension_path.is_file() and sha(extension_path)==EXTENSION["sha256"]
            and extension_path.stat().st_size==EXTENSION["size"])
    except OSError:
        files=extension_live=False
    gates={"keys":set(summary)==expected_keys,"request_keys":set(request)==expected_request,
        "protocol":summary.get("protocol")==WORKER_PROTOCOL and request.get("protocol")==WORKER_PROTOCOL,
        "identity":summary.get("task_index")==task_index and request.get("task_index")==task_index,
        "paths":Path(summary.get("request_path","")).resolve()==paths["request"]
            and Path(summary.get("input_path","")).resolve()==paths["input"]
            and Path(summary.get("output_npz_path","")).resolve()==paths["npz"],
        "input":input_schema(inputs),"output":output_schema(arrays),
        "request":request.get("input_path")==summary.get("input_path")
            and request.get("input_sha256")==summary.get("input_sha256")
            and request.get("output_path")==str(paths["json"])
            and request.get("extension")==EXTENSION,
        "capture":np.array_equal(arrays.get("captured_x0_float32"),inputs.get("x0_float32"))
            and np.array_equal(arrays.get("captured_controls_float32"),inputs.get("controls_float32"))
            and np.array_equal(arrays.get("captured_b1_seed_xu_float32"),inputs.get("b1_seed_xu_float32"))
            and np.array_equal(arrays.get("captured_b16_seed_xu_float32"),inputs.get("b16_seed_xu_float32")),
        "module":extension_live and summary.get("module")==EXTENSION["module"]
            and summary.get("extension")==EXTENSION,
        "counts":summary.get("constructor_calls")==1
            and summary.get("sim_forward_calls")==SIM_FORWARD_CALLS
            and summary.get("tool_position_calls")==TOOL_POSITION_CALLS
            and summary.get("solve_calls")==summary.get("sqp_calls")==0,
        "provenance":certify_provenance(summary.get("provenance",{}),task_index),
        "cuda_diagnostics":certify_cuda_diagnostics(summary.get("cuda_diagnostics",{})),
        "elapsed":certify_operational(summary.get("operational",{})),
        "hashes":files and summary.get("array_names")==sorted(arrays)
            and summary.get("array_hashes")=={k:array_hash(arrays[k]) for k in sorted(arrays)}}
    fresh={"gates":gates,"passes":bool(all(gates.values()))}
    return {**fresh,"stored_equal":summary.get("certificate")==fresh,
        "passes":bool(fresh["passes"] and summary.get("certificate")==fresh)}

def _execute_worker_body(request_path,output_path,authorization,provenance,counters,diagnostic_holder,
                         started,monotonic=time.monotonic): # pragma: no cover
    if WORKER_EXECUTION_AUTHORIZATION is None or authorization is not WORKER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P4 CUDA worker blocked")
    request_path=Path(request_path).resolve();output_path=Path(output_path).resolve()
    token=output_path.stem.removeprefix("p4.worker.")
    if len(token)!=2 or not token.isdecimal():raise RuntimeError("invalid worker identity")
    task_index=int(token);paths=worker_paths(task_index)
    if request_path!=paths["request"] or output_path!=paths["json"]:raise RuntimeError("wrong worker path")
    if any(path.exists() for path in (paths["json"],paths["npz"])):raise FileExistsError("worker overwrite")
    request=json.loads(request_path.read_text())
    if set(request)!={"protocol","task_index","input_path","input_sha256","output_path","extension"} \
            or request["protocol"]!=WORKER_PROTOCOL or request["task_index"]!=task_index \
            or Path(request["input_path"]).resolve()!=paths["input"] \
            or Path(request["output_path"]).resolve()!=paths["json"] or request["extension"]!=EXTENSION:
        raise ValueError("worker request invalid")
    if sha(paths["input"])!=request["input_sha256"]:raise RuntimeError("input hash mismatch")
    with np.load(paths["input"],allow_pickle=False) as archive:inputs={k:archive[k] for k in archive.files}
    if not input_schema(inputs):raise ValueError("worker input invalid")
    diagnostics=cuda_diagnostics();diagnostic_holder["value"]=diagnostics
    module=importlib.import_module(EXTENSION["module"]);module_path=Path(module.__file__).resolve()
    if sha(module_path)!=EXTENSION["sha256"] or module_path.stat().st_size!=EXTENSION["size"]:
        raise RuntimeError("extension pin mismatch")
    if (int(module.KNOT_POINTS),int(module.REFERENCE_SIZE),str(module.TOOL_POSITION_FRAME),
            int(module.TOOL_POSITION_SIZE))!=(96,10,"arm_right_tool_joint_origin",3):
        raise RuntimeError("extension attributes mismatch")
    solver=module.BSQP_16_float(DT,0,KKT_TOL,MAX_PCG_ITERS,PCG_TOL,SOLVE_RATIO,MU,
        Q_COST,QD_COST,U_COST,N_COST,CYLINDER_WEIGHT,CYLINDER_WEIGHT,Q_LIMIT_COST,
        VELOCITY_LIMIT_COST,CONTROL_LIMIT_COST,RHO)
    counters["constructor_calls"]+=1
    state=np.empty((LANES,6081,14),np.float32);state[:,0]=inputs["x0_float32"]
    for interval in range(95):
        for substep in range(64):
            k=interval*64+substep
            state[:,k+1]=np.asarray(solver.sim_forward(state[:,k],inputs["controls_float32"][:,interval],DT/64))[0]
            counters["sim_forward_calls"]+=1
    tool=np.empty((LANES,6081,3),np.float32)
    for k in range(6081):
        tool[:,k]=solver.tool_position(state[:,k,:7]);counters["tool_position_calls"]+=1
    arrays={"captured_x0_float32":inputs["x0_float32"],
        "captured_controls_float32":inputs["controls_float32"],
        "captured_b1_seed_xu_float32":inputs["b1_seed_xu_float32"],
        "captured_b16_seed_xu_float32":inputs["b16_seed_xu_float32"],
        "cuda_dense_state_float32":state,"cuda_dense_tool_float32":tool}
    atomic_npz(paths["npz"],arrays);finish_provenance(provenance);elapsed=monotonic()-started
    summary={"protocol":WORKER_PROTOCOL,"task_index":task_index,"request_path":str(paths["request"]),
        "request_sha256":sha(paths["request"]),"input_path":str(paths["input"]),
        "input_sha256":sha(paths["input"]),"output_npz_path":str(paths["npz"]),
        "output_npz_sha256":sha(paths["npz"]),"module":EXTENSION["module"],"extension":EXTENSION,
        **counters,"solve_calls":0,"sqp_calls":0,
        "array_names":sorted(arrays),"array_hashes":{k:array_hash(arrays[k]) for k in sorted(arrays)},
        "provenance":provenance,"cuda_diagnostics":diagnostics,
        "operational":operational(elapsed,elapsed),
        "certificate":{}}
    first=certify_output(summary,arrays,request,inputs,task_index)
    summary["certificate"]={"gates":first["gates"],
        "passes":bool(all(first["gates"].values()))}
    if not certify_output(summary,arrays,request,inputs,task_index)["passes"]:raise RuntimeError("worker cert failed")
    atomic_json(paths["json"],summary);return summary

def certify_rejection(value,task_index):
    keys={"protocol","task_index","stage","incomplete","error_type","error_message",
        "request_path","request_sha256","input_path","input_sha256","npz_path","npz_sha256",
        "constructor_calls","sim_forward_calls","tool_position_calls","solve_calls","sqp_calls",
        "provenance","cuda_diagnostics","operational"}
    paths=worker_paths(task_index);constructor=value.get("constructor_calls")
    sim=value.get("sim_forward_calls");tool=value.get("tool_position_calls")
    return bool(set(value)==keys and value.get("protocol")==WORKER_PROTOCOL
        and value.get("task_index")==task_index and value.get("stage")=="worker_failed"
        and value.get("incomplete") is True and value.get("request_path")==str(paths["request"])
        and value.get("input_path")==str(paths["input"]) and value.get("npz_path")==str(paths["npz"])
        and value.get("request_sha256")==sha(paths["request"])
        and value.get("input_sha256")==sha(paths["input"])
        and value.get("npz_sha256")== (sha(paths["npz"]) if paths["npz"].is_file() else None)
        and constructor in (0,1) and isinstance(sim,int) and 0<=sim<=6080
        and isinstance(tool,int) and 0<=tool<=6081 and (constructor==1 or sim==tool==0)
        and (sim==6080 or tool==0) and value.get("solve_calls")==value.get("sqp_calls")==0
        and certify_provenance(value.get("provenance",{}),task_index)
        and (value.get("cuda_diagnostics") is None or certify_cuda_diagnostics(
            value["cuda_diagnostics"]))
        and certify_operational(value.get("operational",{})))

def certify_cuda_diagnostics(value):
    return bool(isinstance(value,dict) and set(value)=={"uuid","name","driver_version",
        "compute_capability","cuda_runtime_version","cuda_driver_api_version"}
        and value["compute_capability"]=="6.1"
        and all(isinstance(value[key],str) and value[key] for key in
            ("uuid","name","driver_version"))
        and all(isinstance(value[key],int) and value[key]>0 for key in
            ("cuda_runtime_version","cuda_driver_api_version")))

def operational(trigger,finish):return {"trigger_elapsed_s":float(trigger),
    "observed_finish_elapsed_s":float(finish),"worker_wall_limit_s":900.,
    "timing_evidence":False}
def certify_operational(value):return bool(set(value)=={"trigger_elapsed_s",
    "observed_finish_elapsed_s","worker_wall_limit_s","timing_evidence"}
    and all(isinstance(value[k],(int,float)) and np.isfinite(value[k])
            for k in ("trigger_elapsed_s","observed_finish_elapsed_s"))
    and 0<=value["trigger_elapsed_s"]<=900.
    and value["observed_finish_elapsed_s"]>=value["trigger_elapsed_s"]
    and value["worker_wall_limit_s"]==900. and value["timing_evidence"] is False)

def execute_worker(request_path,output_path,authorization=None,monotonic=time.monotonic): # pragma: no cover
    if WORKER_EXECUTION_AUTHORIZATION is None or authorization is not WORKER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P4 CUDA worker blocked")
    output_path=Path(output_path).resolve();token=output_path.stem.removeprefix("p4.worker.")
    if len(token)!=2 or not token.isdecimal():raise RuntimeError("invalid worker identity")
    task_index=int(token);started=monotonic();provenance=start_provenance()
    counters={"constructor_calls":0,"sim_forward_calls":0,"tool_position_calls":0}
    diagnostic_holder={"value":None}
    try:return _execute_worker_body(request_path,output_path,authorization,provenance,counters,
                                    diagnostic_holder,started,monotonic)
    except Exception as error:
        trigger=min(monotonic()-started,900.);finish_provenance(provenance);paths=worker_paths(task_index)
        rejected=paths["json"].with_name(paths["json"].stem+".rejected.json")
        value={"protocol":WORKER_PROTOCOL,"task_index":task_index,
            "stage":"worker_failed","incomplete":True,"error_type":type(error).__name__,
            "error_message":str(error),"request_path":str(paths["request"]),
            "request_sha256":sha(paths["request"]) if paths["request"].is_file() else None,
            "input_path":str(paths["input"]),
            "input_sha256":sha(paths["input"]) if paths["input"].is_file() else None,
            "npz_path":str(paths["npz"]),
            "npz_sha256":sha(paths["npz"]) if paths["npz"].is_file() else None,
            **counters,"solve_calls":0,"sqp_calls":0,"provenance":provenance,
            "cuda_diagnostics":diagnostic_holder["value"],
            "operational":operational(trigger,monotonic()-started)}
        if not certify_rejection(value,task_index):raise RuntimeError("worker rejection invalid") from error
        atomic_json(rejected,value)
        raise

def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--request",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    execute_worker(args.request,args.output,WORKER_EXECUTION_AUTHORIZATION)
if __name__=="__main__":main()
