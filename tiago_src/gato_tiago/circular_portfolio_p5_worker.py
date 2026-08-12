"""Capability-blocked B1/B16 CUDA worker for the N260 P5 pilot."""

from __future__ import annotations

import argparse,ctypes,hashlib,importlib,importlib.metadata,json,os,shlex,subprocess,sys,time
from pathlib import Path

import numpy as np

from gato_tiago.circular_portfolio import (CONTROL_LIMIT_COST,CYLINDER_WEIGHT,KKT_TOL,
    MAX_PCG_ITERS,MU,N_COST,PCG_TOL,Q_COST,QD_COST,Q_LIMIT_COST,RHO,SOLVE_RATIO,U_COST,
    VELOCITY_LIMIT_COST)
from gato_tiago.circular_portfolio_p5 import (DENSE_SAMPLES,DENSE_SUBSTEPS,DT,EXTENSION,
    INTERVALS,KNOTS,LANES,WORKER_PROTOCOL,array_hash,worker_input_schema,worker_output_schema,
    worker_paths)


WORKER_EXECUTION_AUTHORIZATION=None
AUTHORIZED_CWD="/workspace/GATO"
THREAD_ENV={"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
    "NUMEXPR_NUM_THREADS":"1"}
SOURCE_PATHS=("CMakeLists.txt","tools/build.sh","python/bindings.cu","gato/dynamics/integrator.cuh",
    "python/bsqp/interface.py","gato/bsqp/bsqp.cuh",
    "gato/bsqp/kernels/tool_position.cuh",
    "gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "gato/dynamics/tiago_right/tiago_right_grid.cuh",
    "gato/dynamics/tiago_right/tiago_right_arm.urdf","gato/utils/cuda.cuh",
    "tiago_src/gato_tiago/circular_portfolio.py",
    "tiago_src/gato_tiago/circular_portfolio_p5.py",
    "tiago_src/gato_tiago/circular_portfolio_p5_worker.py")

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError(f"P5 worker artifact exists: {path}")
    candidate.write_text(json.dumps(value,sort_keys=True,separators=(",",":")));candidate.replace(path)
def atomic_npz(path,arrays):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError(f"P5 worker artifact exists: {path}")
    with candidate.open("xb") as stream:np.savez(stream,**arrays)
    candidate.replace(path)
def cuda_diagnostics():
    runtime=ctypes.c_int();driver=ctypes.c_int();library=ctypes.CDLL("libcudart.so")
    if library.cudaRuntimeGetVersion(ctypes.byref(runtime)) or library.cudaDriverGetVersion(ctypes.byref(driver)):
        raise RuntimeError("P5 CUDA diagnostics failed")
    line=subprocess.run(["nvidia-smi","--query-gpu=uuid,name,driver_version,compute_cap",
        "--format=csv,noheader"],check=True,text=True,stdout=subprocess.PIPE).stdout.strip().splitlines()
    if len(line)!=1:raise RuntimeError("P5 requires exactly one CUDA device")
    fields=[item.strip() for item in line[0].split(",")]
    if len(fields)!=4 or fields[3]!="6.1":raise RuntimeError("P5 CUDA device unsupported")
    return {"uuid":fields[0],"name":fields[1],"driver_version":fields[2],
        "compute_capability":fields[3],"cuda_runtime_version":runtime.value,
        "cuda_driver_api_version":driver.value}
def certify_cuda_diagnostics(value):
    return bool(isinstance(value,dict) and set(value)=={"uuid","name","driver_version",
        "compute_capability","cuda_runtime_version","cuda_driver_api_version"}
        and all(isinstance(value[k],str) and value[k] for k in ("uuid","name","driver_version"))
        and value["compute_capability"]=="6.1"
        and all(isinstance(value[k],int) and value[k]>0 for k in
            ("cuda_runtime_version","cuda_driver_api_version")))
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
def expected_argv():
    paths=worker_paths()
    return ("python","-B","-m","gato_tiago.circular_portfolio_p5_worker","--execute",
        "--request",str(paths["request"]),"--output",str(paths["json"]))
def start_provenance():
    current=snapshot();environment={k:os.environ.get(k) for k in THREAD_ENV}
    if environment!=THREAD_ENV:raise RuntimeError("P5 worker requires single-thread environment")
    frozen={"path":str((Path(__file__).resolve().parents[2]/EXTENSION["relative_path"]).resolve()),
        "sha256":EXTENSION["sha256"],"size":EXTENSION["size"],
        "build_head":EXTENSION["build_head"],"arch":EXTENSION["arch"],
        "attributes":{k:EXTENSION[k] for k in ("KNOT_POINTS","REFERENCE_SIZE",
            "TOOL_POSITION_FRAME","TOOL_POSITION_SIZE")}}
    if current["extension"]!=frozen:raise RuntimeError("P5 extension pin mismatch")
    return {"cwd":str(Path.cwd().resolve()),"orig_argv":list(sys.orig_argv),
        "command":shlex.join(sys.orig_argv),"head_start":current["head"],"head_end":None,
        "clean_start":current["clean"],"clean_end":None,"sources_start":current["sources"],
        "sources_end":None,"extension_start":current["extension"],"extension_end":None,
        "thread_environment":environment,"runtime_versions":{"python":sys.version,
            "numpy":np.__version__,"pybind11":importlib.metadata.version("pybind11")}}
def finish_provenance(value):
    current=snapshot();value.update({"head_end":current["head"],"clean_end":current["clean"],
        "sources_end":current["sources"],"extension_end":current["extension"]})
def certify_provenance(value):
    current=snapshot();frozen={"path":str((Path(__file__).resolve().parents[2]/EXTENSION["relative_path"]).resolve()),
        "sha256":EXTENSION["sha256"],"size":EXTENSION["size"],
        "build_head":EXTENSION["build_head"],"arch":EXTENSION["arch"],
        "attributes":{k:EXTENSION[k] for k in ("KNOT_POINTS","REFERENCE_SIZE",
            "TOOL_POSITION_FRAME","TOOL_POSITION_SIZE")}}
    keys={"cwd","orig_argv","command","head_start","head_end","clean_start","clean_end",
        "sources_start","sources_end","extension_start","extension_end","thread_environment",
        "runtime_versions"}
    return bool(set(value)==keys and value.get("cwd")==AUTHORIZED_CWD and tuple(value.get("orig_argv",()))==expected_argv()
        and value.get("command")==shlex.join(expected_argv())
        and value.get("head_start")==value.get("head_end")==current["head"]
        and value.get("clean_start") is value.get("clean_end") is current["clean"] is True
        and value.get("sources_start")==value.get("sources_end")==current["sources"]
        and value.get("extension_start")==value.get("extension_end")==frozen
        and value.get("thread_environment")==THREAD_ENV
        and value.get("runtime_versions")=={"python":sys.version,"numpy":np.__version__,
            "pybind11":importlib.metadata.version("pybind11")})


def expected_counts():
    return {"b1_constructors":1,"b16_constructors":1,
        "b1_sim_forward_calls":INTERVALS*DENSE_SUBSTEPS,
        "b16_sim_forward_calls":INTERVALS*DENSE_SUBSTEPS,
        "b1_tool_position_calls":DENSE_SAMPLES,"b16_tool_position_calls":DENSE_SAMPLES,
        "solve_calls":0,"sqp_calls":0}

def partial_counts():
    return {key:0 for key in expected_counts()}
def certify_partial_counts(value):
    if not isinstance(value,dict) or set(value)!=set(expected_counts()):return False
    if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in value.values()):return False
    maximum=expected_counts()
    if any(value[k]>maximum[k] for k in value):return False
    if any(value[k] for k in ("solve_calls","sqp_calls")):return False
    return bool(value["b1_constructors"]>=bool(value["b1_sim_forward_calls"] or value["b1_tool_position_calls"])
        and value["b16_constructors"]>=bool(value["b16_sim_forward_calls"] or value["b16_tool_position_calls"])
        and value["b1_constructors"]>=value["b16_constructors"]
        and value["b1_sim_forward_calls"] in (value["b16_sim_forward_calls"],
            value["b16_sim_forward_calls"]+1)
        and value["b1_tool_position_calls"] in (value["b16_tool_position_calls"],
            value["b16_tool_position_calls"]+1)
        and (value["b1_sim_forward_calls"]==maximum["b1_sim_forward_calls"]
             or value["b1_tool_position_calls"]==0)
        and (value["b16_sim_forward_calls"]==maximum["b16_sim_forward_calls"]
             or value["b16_tool_position_calls"]==0))


def certify_output(summary,arrays,inputs):
    keys={"protocol","module","module_path","module_sha256","module_size","extension",
        "request_path","request_sha256","input_path","input_sha256","output_npz_path",
        "output_npz_sha256","counts","array_names","array_hashes","provenance","cuda_diagnostics",
        "elapsed_s","certificate"}
    gates={"summary_keys":set(summary)==keys,"protocol":summary.get("protocol")==WORKER_PROTOCOL,
        "module":summary.get("module")==EXTENSION["module"],
        "module_path":summary.get("module_path")==str((Path(__file__).resolve().parents[2]
            /EXTENSION["relative_path"]).resolve()),
        "module_hash":summary.get("module_sha256")==EXTENSION["sha256"]
            and summary.get("module_size")==EXTENSION["size"],
        "paths":summary.get("request_path")==str(worker_paths()["request"])
            and summary.get("input_path")==str(worker_paths()["input"])
            and summary.get("output_npz_path")==str(worker_paths()["npz"]),
        "file_hashes":summary.get("request_sha256")==sha(worker_paths()["request"])
            and summary.get("input_sha256")==sha(worker_paths()["input"])
            and summary.get("output_npz_sha256")==sha(worker_paths()["npz"]),
        "extension":summary.get("extension")==EXTENSION,
        "provenance":certify_provenance(summary.get("provenance",{})),
        "cuda_diagnostics":certify_cuda_diagnostics(summary.get("cuda_diagnostics")),
        "input":worker_input_schema(inputs),"output":worker_output_schema(arrays),
        "counts":summary.get("counts")==expected_counts(),
        "capture":worker_output_schema(arrays) and worker_input_schema(inputs)
            and np.array_equal(arrays["captured_x0_float32"],inputs["x0_float32"])
            and np.array_equal(arrays["captured_controls_float32"],inputs["controls_float32"])
            and np.array_equal(arrays["captured_b1_seed_xu_float32"],inputs["b1_seed_xu_float32"])
            and np.array_equal(arrays["captured_b16_seed_xu_float32"],inputs["b16_seed_xu_float32"]),
        "b1_b16_lane0":worker_output_schema(arrays)
            and np.array_equal(arrays["b1_dense_state_float32"],arrays["b16_dense_state_float32"][0])
            and np.array_equal(arrays["b1_dense_tool_float32"],arrays["b16_dense_tool_float32"][0]),
        "lane_outputs_distinct":worker_output_schema(arrays)
            and not np.array_equal(arrays["b16_dense_state_float32"][0],arrays["b16_dense_state_float32"][1])
            and not np.array_equal(arrays["b16_dense_tool_float32"][0],arrays["b16_dense_tool_float32"][1]),
        "arrays":summary.get("array_names")==sorted(arrays)
            and summary.get("array_hashes")=={k:array_hash(arrays[k]) for k in sorted(arrays)},
        "elapsed":isinstance(summary.get("elapsed_s"),(int,float))
            and np.isfinite(summary["elapsed_s"]) and 0<=summary["elapsed_s"]<=300.}
    certificate=summary.get("certificate")
    gates["stored_certificate"]=isinstance(certificate,dict) \
        and certificate.get("gates")=={k:v for k,v in gates.items() if k!="stored_certificate"} \
        and certificate.get("passes") is all(v is True for k,v in gates.items() if k!="stored_certificate")
    return {"gates":gates,"passes":bool(all(gates.values()))}


def execute_worker(request_path,authorization=None,monotonic=time.monotonic,
                   execution_state=None): # pragma: no cover
    if WORKER_EXECUTION_AUTHORIZATION is None or authorization is not WORKER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P5 worker blocked")
    paths=worker_paths()
    if Path(request_path).resolve()!=paths["request"]:raise RuntimeError("wrong P5 request path")
    request=json.loads(paths["request"].read_text())
    exact={"protocol":WORKER_PROTOCOL,"input_path":str(paths["input"]),
        "input_sha256":sha(paths["input"]),"output_path":str(paths["json"]),"extension":EXTENSION}
    if request!=exact:raise RuntimeError("P5 worker request invalid")
    started=monotonic();deadline=started+300.;provenance=start_provenance();counts=partial_counts()
    if execution_state is not None:execution_state.update({"provenance":provenance,
        "counts":counts,"diagnostics":None,"started":started})
    with np.load(paths["input"],allow_pickle=False) as archive:inputs={k:archive[k] for k in archive.files}
    if not worker_input_schema(inputs):raise ValueError("P5 worker input invalid")
    if monotonic()>=deadline:raise TimeoutError("P5 worker wall limit")
    diagnostics=cuda_diagnostics()
    if execution_state is not None:execution_state["diagnostics"]=diagnostics
    module=importlib.import_module(EXTENSION["module"])
    module_path=Path(module.__file__).resolve()
    frozen=(Path(__file__).resolve().parents[2]/EXTENSION["relative_path"]).resolve()
    if module_path!=frozen or sha(module_path)!=EXTENSION["sha256"] or module_path.stat().st_size!=EXTENSION["size"]:
        raise RuntimeError("P5 imported extension pin mismatch")
    if (int(module.KNOT_POINTS),int(module.REFERENCE_SIZE),str(module.TOOL_POSITION_FRAME),
            int(module.TOOL_POSITION_SIZE))!=(260,10,"arm_right_tool_joint_origin",3):
        raise RuntimeError("P5 extension attributes invalid")
    args=(DT,0,KKT_TOL,MAX_PCG_ITERS,PCG_TOL,SOLVE_RATIO,MU,Q_COST,QD_COST,U_COST,
        N_COST,CYLINDER_WEIGHT,CYLINDER_WEIGHT,Q_LIMIT_COST,VELOCITY_LIMIT_COST,
        CONTROL_LIMIT_COST,RHO)
    b1=module.BSQP_1_float(*args);counts["b1_constructors"]+=1
    b16=module.BSQP_16_float(*args);counts["b16_constructors"]+=1
    if monotonic()>=deadline:raise TimeoutError("P5 worker wall limit")
    b1_state=np.empty((DENSE_SAMPLES,14),np.float32);b1_state[0]=inputs["x0_float32"][0]
    b16_state=np.empty((LANES,DENSE_SAMPLES,14),np.float32);b16_state[:,0]=inputs["x0_float32"]
    for interval in range(INTERVALS):
        for substep in range(DENSE_SUBSTEPS):
            if monotonic()>=deadline:raise TimeoutError("P5 worker wall limit")
            k=interval*DENSE_SUBSTEPS+substep
            b1_next=np.asarray(b1.sim_forward(b1_state[k],inputs["controls_float32"][0,interval],DT/DENSE_SUBSTEPS))
            b1_state[k+1]=b1_next[0]
            counts["b1_sim_forward_calls"]+=1
            b16_next=np.asarray(b16.sim_forward(b16_state[:,k],inputs["controls_float32"][:,interval],DT/DENSE_SUBSTEPS))
            if b16_next.shape!=(LANES,14):raise RuntimeError("P5 B16 sim_forward layout invalid")
            b16_state[:,k+1]=b16_next
            counts["b16_sim_forward_calls"]+=1
    b1_tool=np.empty((DENSE_SAMPLES,3),np.float32)
    b16_tool=np.empty((LANES,DENSE_SAMPLES,3),np.float32)
    for k in range(DENSE_SAMPLES):
        if monotonic()>=deadline:raise TimeoutError("P5 worker wall limit")
        b1_tool[k]=np.asarray(b1.tool_position(b1_state[k,:7]))[0]
        counts["b1_tool_position_calls"]+=1
        value=np.asarray(b16.tool_position(b16_state[:,k,:7]))
        if value.shape!=(LANES,3):raise RuntimeError("P5 B16 tool layout invalid")
        b16_tool[:,k]=value
        counts["b16_tool_position_calls"]+=1
    arrays={"captured_x0_float32":inputs["x0_float32"],
        "captured_controls_float32":inputs["controls_float32"],
        "captured_b1_seed_xu_float32":inputs["b1_seed_xu_float32"],
        "captured_b16_seed_xu_float32":inputs["b16_seed_xu_float32"],
        "b1_dense_state_float32":b1_state,"b1_dense_tool_float32":b1_tool,
        "b16_dense_state_float32":b16_state,"b16_dense_tool_float32":b16_tool}
    finish_provenance(provenance)
    return arrays,monotonic()-started,provenance,counts,diagnostics,module_path

def certify_rejection(value):
    keys={"protocol","stage","incomplete","error_type","error_message","counts",
        "internal_counts_known","provenance","cuda_diagnostics","trigger_elapsed_s",
        "finish_elapsed_s","worker_wall_limit_s",
        "request_path","request_sha256","input_path","input_sha256","npz_path","npz_sha256"}
    paths=worker_paths();known=value.get("internal_counts_known")
    return bool(set(value)==keys and value.get("protocol")==WORKER_PROTOCOL
        and value.get("stage")=="worker_failed" and value.get("incomplete") is True
        and known is True and certify_partial_counts(value.get("counts"))
        and certify_provenance(value.get("provenance",{}))
        and (value.get("cuda_diagnostics") is None
             or certify_cuda_diagnostics(value["cuda_diagnostics"]))
        and value.get("request_path")==str(paths["request"])
        and value.get("request_sha256")==sha(paths["request"])
        and value.get("input_path")==str(paths["input"])
        and value.get("input_sha256")==sha(paths["input"])
        and value.get("npz_path")==str(paths["npz"])
        and value.get("npz_sha256")== (sha(paths["npz"]) if paths["npz"].is_file() else None)
        and value.get("worker_wall_limit_s")==300.0
        and np.isfinite(value.get("trigger_elapsed_s")) and 0<=value["trigger_elapsed_s"]<=300.
        and np.isfinite(value.get("finish_elapsed_s"))
        and value["finish_elapsed_s"]>=value["trigger_elapsed_s"])


def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--request",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    paths=worker_paths()
    if args.request.resolve()!=paths["request"] or args.output.resolve()!=paths["json"]:
        raise RuntimeError("wrong P5 worker paths")
    if any(path.exists() for key,path in paths.items() if key not in ("request","input")):
        raise FileExistsError("P5 worker output exists")
    state={};provenance=None;counts=partial_counts();diagnostics=None;started=time.monotonic()
    try:
        arrays,elapsed,provenance,counts,diagnostics,module_path=execute_worker(
            args.request,WORKER_EXECUTION_AUTHORIZATION,execution_state=state)
        atomic_npz(paths["npz"],arrays)
        summary={"protocol":WORKER_PROTOCOL,"module":EXTENSION["module"],
            "module_path":str(module_path),"module_sha256":sha(module_path),
            "module_size":module_path.stat().st_size,"extension":EXTENSION,
            "request_path":str(paths["request"]),"request_sha256":sha(paths["request"]),
            "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
            "output_npz_path":str(paths["npz"]),"output_npz_sha256":sha(paths["npz"]),
            "counts":counts,"array_names":sorted(arrays),
            "array_hashes":{k:array_hash(arrays[k]) for k in sorted(arrays)},
            "elapsed_s":elapsed,"provenance":provenance,"cuda_diagnostics":diagnostics,
            "certificate":{}}
        with np.load(paths["input"],allow_pickle=False) as archive:inputs={k:archive[k] for k in archive.files}
        first=certify_output(summary,arrays,inputs)
        stored={k:v for k,v in first["gates"].items() if k!="stored_certificate"}
        summary["certificate"]={"gates":stored,"passes":bool(all(stored.values()))}
        if not certify_output(summary,arrays,inputs)["passes"]:raise RuntimeError("P5 worker cert failed")
        atomic_json(paths["json"],summary)
    except Exception as error:
        provenance=state.get("provenance",provenance);counts=state.get("counts",counts)
        diagnostics=state.get("diagnostics",diagnostics);started=state.get("started",started)
        if provenance is not None and provenance.get("head_end") is None:finish_provenance(provenance)
        if provenance is not None:
            finish_elapsed=time.monotonic()-started;trigger_elapsed=min(finish_elapsed,300.)
            rejected={"protocol":WORKER_PROTOCOL,"stage":"worker_failed","incomplete":True,
                "error_type":type(error).__name__,"error_message":str(error),"counts":counts,
                "internal_counts_known":True,"provenance":provenance,
                "cuda_diagnostics":diagnostics,"trigger_elapsed_s":trigger_elapsed,
                "finish_elapsed_s":finish_elapsed,"worker_wall_limit_s":300.0,
                "request_path":str(paths["request"]),"request_sha256":sha(paths["request"]),
                "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
                "npz_path":str(paths["npz"]),"npz_sha256":sha(paths["npz"]) if paths["npz"].is_file() else None}
            if not certify_rejection(rejected):raise RuntimeError("P5 worker rejection cert failed") from error
            atomic_json(paths["rejected"],rejected)
        raise

if __name__=="__main__":main()
