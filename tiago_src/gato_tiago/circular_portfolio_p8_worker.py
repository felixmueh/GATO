"""Disabled P8 CUDA worker and pure evidence certificates."""

from __future__ import annotations

import argparse,hashlib,importlib,importlib.metadata,json,os,subprocess,sys,time
from pathlib import Path

import numpy as np

from gato_tiago.circular_portfolio_p5_worker import cuda_diagnostics,certify_cuda_diagnostics
from gato_tiago.circular_portfolio_p8 import (AFFINE_SAMPLES,AFFINE_SUBSTEPS,CONSTRUCTION_SPECS,
    DT,EXTENSION,INTERVALS,KD,KNOTS,KP,LANES,OUTPUT,PROTOCOL,
    WORKER_OUTPUT_SPECS,WORKER_PROTOCOL,array_hash,exact_arrays,pack_seed,worker_paths)

WORKER_EXECUTION_AUTHORIZATION=None


THREAD_ENV={"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
    "NUMEXPR_NUM_THREADS":"1"}
SOURCE_PATHS=("CMakeLists.txt","tools/build.sh","python/bindings.cu",
    "tiago_src/gato_tiago/circular_portfolio.py",
    "tiago_src/gato_tiago/circular_portfolio_p5.py",
    "tiago_src/gato_tiago/circular_portfolio_p5_worker.py",
    "tiago_src/gato_tiago/circular_portfolio_p6.py",
    "tiago_src/gato_tiago/circular_portfolio_p6_worker.py",
    "tiago_src/gato_tiago/circular_portfolio_p8.py",
    "tiago_src/gato_tiago/circular_portfolio_p8_worker.py",
    "tiago_src/gato_tiago/circular_portfolio_runner.py",
    "tiago_src/gato_tiago/multimodal_pillar.py",
    "gato/dynamics/tiago_right/tiago_right_arm.urdf","python/bsqp/interface.py",
    "gato/bsqp/bsqp.cuh","gato/bsqp/kernels/tool_position.cuh","gato/utils/cuda.cuh",
    "gato/dynamics/integrator.cuh","gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "gato/dynamics/tiago_right/tiago_right_grid.cuh")
SUCCESS_COUNTS={"pin_model_contexts":1,"b1_constructors":1,"b16_constructors":1,
    "rnea_calls":518,"b1_sim_forward_calls":1036,"b16_sim_forward_calls":259,
    "b1_tool_position_calls":9329,"b16_tool_position_calls":260,
    "solve_calls":0,"sqp_calls":0}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_argv():
    paths=worker_paths()
    return ("python","-B","-m","gato_tiago.circular_portfolio_p8_worker","--execute",
        "--request",str(paths["request"]),"--output",str(paths["json"]))


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


def certify_module(module):
    try:
        resolved=str(Path(module.__file__).resolve())
        attrs={key:getattr(module,key) for key in
            ("KNOT_POINTS","REFERENCE_SIZE","TOOL_POSITION_FRAME","TOOL_POSITION_SIZE")}
        return bool(resolved==frozen_extension()["path"] and sha(resolved)==EXTENSION["sha256"]
            and Path(resolved).stat().st_size==EXTENSION["size"]
            and attrs==frozen_extension()["attributes"]
            and hasattr(module,"BSQP_1_float") and hasattr(module,"BSQP_16_float"))
    except Exception:return False


def module_measurement(module):
    path=Path(module.__file__).resolve()
    return {"path":str(path),"sha256":sha(path),"size":path.stat().st_size,
        "build_head":EXTENSION["build_head"],"arch":EXTENSION["arch"],
        "attributes":{key:getattr(module,key) for key in
            ("KNOT_POINTS","REFERENCE_SIZE","TOOL_POSITION_FRAME","TOOL_POSITION_SIZE")}}


def provenance(start,end=None):
    return {"cwd":str(Path.cwd().resolve()),"orig_argv":list(sys.orig_argv),
        "thread_environment":{k:os.environ.get(k) for k in THREAD_ENV},
        "runtime_versions":{"python":sys.version,"numpy":np.__version__,
            "pinocchio":importlib.metadata.version("pin")},"start":start,"end":end}


def certify_provenance(value,final):
    try:
        current=snapshot()
        return bool(set(value)=={"cwd","orig_argv","thread_environment","runtime_versions",
                "start","end"} and value["cwd"]=="/workspace/GATO"
            and tuple(value["orig_argv"])==expected_argv() and value["thread_environment"]==THREAD_ENV
            and value["runtime_versions"]=={"python":sys.version,"numpy":np.__version__,
                "pinocchio":importlib.metadata.version("pin")}
            and value["start"]==current and value["start"]["clean"] is True
            and value["start"]["extension"]==frozen_extension()
            and ((not final and value["end"] is None) or final and value["end"]==value["start"]))
    except Exception:return False


def constructor_args():
    from gato_tiago.circular_portfolio_p6_worker import constructor_args as inherited
    values=list(inherited());values[0]=DT;return tuple(values)


def initial_counts():return {key:0 for key in SUCCESS_COUNTS}


def certify_counts(value,final):
    if not isinstance(value,dict) or set(value)!=set(SUCCESS_COUNTS) \
            or any(type(v) is not int or v<0 for v in value.values()):return False
    if final:return value==SUCCESS_COUNTS
    if not all(value[k]<=SUCCESS_COUNTS[k] for k in value):return False
    work=("b1_constructors","b16_constructors","rnea_calls","b1_sim_forward_calls",
        "b16_sim_forward_calls","b1_tool_position_calls","b16_tool_position_calls")
    if any(value[key] for key in work) and value["pin_model_contexts"]!=1:return False
    if value["b16_constructors"]>value["b1_constructors"]:return False
    calls=work[2:]
    if any(value[key] for key in calls) and (value["b1_constructors"],
            value["b16_constructors"])!=(1,1):return False
    if value["b16_sim_forward_calls"] or value["b16_tool_position_calls"]:
        if (value["rnea_calls"]!=518 or value["b1_sim_forward_calls"]!=1036
                or value["b1_tool_position_calls"]<1039):return False
    if value["b1_tool_position_calls"]>1039 and (value["b16_sim_forward_calls"],
            value["b16_tool_position_calls"])!=(259,260):return False
    return True


def atomic_json(path,value):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError("P8 worker artifact exists")
    try:candidate.write_text(json.dumps(value,sort_keys=True,separators=(",",":")));candidate.replace(path)
    finally:
        if candidate.exists():candidate.unlink()


def atomic_npz(path,arrays):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError("P8 worker artifact exists")
    try:
        with candidate.open("wb") as stream:np.savez(stream,**arrays)
        candidate.replace(path)
    finally:
        if candidate.exists():candidate.unlink()


def feedback(proxy,k,state):
    return (proxy["qdd"][k]+25.*(proxy["q"][k]-state[:7])
            +10.*(proxy["qd"][k]-state[7:]))


def generate(module,construction,rnea,deadline,monotonic=time.monotonic,counters=None):
    counters=initial_counts() if counters is None else counters
    if not exact_arrays(construction,CONSTRUCTION_SPECS):raise ValueError("P8 construction invalid")
    if monotonic()>=deadline:raise TimeoutError("P8 worker wall limit")
    b1=module.BSQP_1_float(*constructor_args());counters["b1_constructors"]+=1
    b16=module.BSQP_16_float(*constructor_args());counters["b16_constructors"]+=1
    def sim1(x,u):counters["b1_sim_forward_calls"]+=1;return np.asarray(b1.sim_forward(x,u,DT))
    def sim16(x,u):counters["b16_sim_forward_calls"]+=1;return np.asarray(b16.sim_forward(x,u,DT))
    def tool1(q):counters["b1_tool_position_calls"]+=1;return np.asarray(b1.tool_position(q))
    def tool16(q):counters["b16_tool_position_calls"]+=1;return np.asarray(b16.tool_position(q))
    x0=construction["x0_float32"];state=np.empty((2,KNOTS,14),np.float32)
    tool=np.empty((2,KNOTS,3),np.float32);u=np.empty((2,INTERVALS,7),np.float32)
    rq=np.empty((2,INTERVALS,7));rv=np.empty_like(rq);ra=np.empty_like(rq);ru=np.empty_like(rq)
    for route in range(2):
        proxy={"q":construction["proxy_q_float64"][route],
            "qd":construction["proxy_qd_float64"][route],
            "qdd":construction["proxy_qdd_float64"][route]}
        state[route,0]=x0;tool[route,0]=tool1(x0[:7])[0]
        for k in range(INTERVALS):
            if monotonic()>=deadline:raise TimeoutError("P8 worker wall limit")
            z=state[route,k].astype(np.float64);acc=feedback(proxy,k,z)
            rq[route,k]=z[:7];rv[route,k]=z[7:];ra[route,k]=acc
            ru[route,k]=rnea(z[:7],z[7:],acc);counters["rnea_calls"]+=1
            u[route,k]=ru[route,k].astype(np.float32)
            state[route,k+1]=sim1(state[route,k],u[route,k])[0]
            tool[route,k+1]=tool1(state[route,k+1,:7])[0]
    fresh=np.empty_like(state);fresh_tool=np.empty_like(tool);fresh[:,0]=x0
    fresh_tool[:,0]=np.stack([tool1(x0[:7])[0]]*2)
    for route in range(2):
        for k in range(INTERVALS):
            if monotonic()>=deadline:raise TimeoutError("P8 worker wall limit")
            fresh[route,k+1]=sim1(fresh[route,k],u[route,k])[0]
            fresh_tool[route,k+1]=tool1(fresh[route,k+1,:7])[0]
    lane_u=np.stack([u[0 if lane<8 else 1] for lane in range(LANES)])
    state16=np.empty((LANES,KNOTS,14),np.float32);tool_16=np.empty((LANES,KNOTS,3),np.float32)
    state16[:,0]=x0;tool_16[:,0]=tool16(state16[:,0,:7])
    for k in range(INTERVALS):
        if monotonic()>=deadline:raise TimeoutError("P8 worker wall limit")
        state16[:,k+1]=sim16(state16[:,k],lane_u[:,k]);tool_16[:,k+1]=tool16(state16[:,k+1,:7])
    affine=np.empty((2,AFFINE_SAMPLES,3),np.float32)
    for route in range(2):
        index=0
        for k in range(INTERVALS):
            for substep in range(AFFINE_SUBSTEPS):
                if monotonic()>=deadline:raise TimeoutError("P8 worker wall limit")
                alpha=substep/AFFINE_SUBSTEPS
                q=(1-alpha)*state[route,k,:7]+alpha*state[route,k+1,:7]
                affine[route,index]=tool1(q)[0];index+=1
        if monotonic()>=deadline:raise TimeoutError("P8 worker wall limit")
        affine[route,-1]=tool1(state[route,-1,:7])[0]
    seeds=np.stack([pack_seed(state[0 if lane<8 else 1,:,:7],
        state[0 if lane<8 else 1,:,7:],u[0 if lane<8 else 1]) for lane in range(LANES)])
    arrays={"captured_x0_float32":x0,"generated_state_float32":state,
        "generated_tool_float32":tool,"rnea_q_float64":rq,"rnea_qd_float64":rv,
        "rnea_qdd_float64":ra,"rnea_u_float64":ru,"controls_float32":u,
        "fresh_b1_state_float32":fresh,"fresh_b1_tool_float32":fresh_tool,
        "b1_defect_float32":fresh[:,1:]-state[:,1:],
        "fresh_b16_state_float32":state16,"fresh_b16_tool_float32":tool_16,
        "b16_defect_float32":state16[:,1:]-np.stack([state[0 if l<8 else 1,1:] for l in range(LANES)]),
        "b1_seed_float32":seeds[0],"b16_seed_float32":seeds,
        "affine_cuda_tool_float32":affine}
    if not exact_arrays(arrays,WORKER_OUTPUT_SPECS):raise RuntimeError("P8 worker output invalid")
    return arrays


def certify_output(arrays,construction):
    if not exact_arrays(arrays,WORKER_OUTPUT_SPECS) or not exact_arrays(
            construction,CONSTRUCTION_SPECS):return False
    state=arrays["generated_state_float32"];u=arrays["controls_float32"]
    promoted=state[:,:-1].astype(np.float64)
    expected_qdd=construction["proxy_qdd_float64"] \
        +KP*(construction["proxy_q_float64"][:,:-1]-promoted[:,:,:7]) \
        +KD*(construction["proxy_qd_float64"][:,:-1]-promoted[:,:,7:])
    seeds=np.stack([pack_seed(state[0 if l<8 else 1,:,:7],state[0 if l<8 else 1,:,7:],
        u[0 if l<8 else 1]) for l in range(LANES)])
    return bool(np.array_equal(arrays["captured_x0_float32"],construction["x0_float32"])
        and all(np.array_equal(route[0],construction["x0_float32"]) for route in state)
        and np.array_equal(arrays["rnea_q_float64"],promoted[:,:,:7])
        and np.array_equal(arrays["rnea_qd_float64"],promoted[:,:,7:])
        and np.array_equal(arrays["rnea_qdd_float64"],expected_qdd)
        and np.array_equal(arrays["fresh_b1_state_float32"],state)
        and np.array_equal(arrays["fresh_b1_tool_float32"],arrays["generated_tool_float32"])
        and not np.any(arrays["b1_defect_float32"]) and not np.any(arrays["b16_defect_float32"])
        and np.array_equal(arrays["rnea_u_float64"].astype(np.float32),u)
        and np.array_equal(arrays["b1_seed_float32"],seeds[0])
        and np.array_equal(arrays["b16_seed_float32"],seeds)
        and all(np.array_equal(arrays["fresh_b16_state_float32"][l],state[0 if l<8 else 1])
            and np.array_equal(arrays["fresh_b16_tool_float32"][l],
                               arrays["generated_tool_float32"][0 if l<8 else 1]) for l in range(LANES))
        and not np.array_equal(state[0],state[1]))


def certify_summary(summary,arrays,construction=None):
    paths=worker_paths();keys={"protocol","identity","request_path","request_sha256",
        "input_path","input_sha256","npz_path","npz_sha256","array_names","array_hashes",
        "elapsed_s","counts","provenance","extension","resolved_module",
        "cuda_diagnostics","certificate"}
    return bool(set(summary)==keys and summary["protocol"]==WORKER_PROTOCOL
        and summary["identity"]==["development",12600] and summary["request_path"]==str(paths["request"])
        and summary["request_sha256"]==sha(paths["request"])
        and summary["input_path"]==str(paths["input"]) and summary["input_sha256"]==sha(paths["input"])
        and summary["npz_path"]==str(paths["npz"]) and summary["npz_sha256"]==sha(paths["npz"])
        and summary["array_names"]==sorted(arrays)
        and summary["array_hashes"]=={k:array_hash(arrays[k]) for k in sorted(arrays)}
        and summary["certificate"] is True and construction is not None
        and certify_output(arrays,construction)
        and certify_counts(summary["counts"],True) and certify_provenance(summary["provenance"],True)
        and summary["extension"]==summary["provenance"]["start"]["extension"]
        and summary["resolved_module"]==frozen_extension()
        and certify_cuda_diagnostics(summary["cuda_diagnostics"])
        and np.isfinite(summary["elapsed_s"]) and 0<=summary["elapsed_s"]<=300.)


def certify_rejection(value):
    try:
        keys={"protocol","stage","error_type","error_message","counts",
            "internal_counts_known","trigger_elapsed_s","cleanup_finish_elapsed_s",
            "wall_limit_s","provenance","cuda_diagnostics","incomplete","evidence"}
        return bool(set(value)==keys and value["protocol"]==WORKER_PROTOCOL
            and value["stage"]=="worker_failed" and value["incomplete"] is True
            and value["evidence"] is False and value["internal_counts_known"] is True
            and certify_counts(value["counts"],False)
            and value["wall_limit_s"]==300. and np.isfinite(value["trigger_elapsed_s"])
            and np.isfinite(value["cleanup_finish_elapsed_s"])
            and 0<=value["trigger_elapsed_s"]<=value["cleanup_finish_elapsed_s"]
            and certify_provenance(value["provenance"],True)
            and (value["cuda_diagnostics"] is None
                 or certify_cuda_diagnostics(value["cuda_diagnostics"])))
    except Exception:return False


def execute(request,authorization=None,monotonic=time.monotonic): # pragma: no cover
    if WORKER_EXECUTION_AUTHORIZATION is None or authorization is not WORKER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P8 worker blocked")
    paths=worker_paths()
    if Path(request).resolve()!=paths["request"]:raise RuntimeError("wrong P8 worker request")
    if any(path.exists() or Path(str(path)+".candidate").exists()
            for key,path in paths.items() if key in ("json","npz","rejected")):
        raise FileExistsError("P8 worker output exists")
    start_time=monotonic();deadline=start_time+300.;start_snapshot=snapshot()
    counters=initial_counts();diagnostics=None
    try:
        if (Path.cwd().resolve()!=Path("/workspace/GATO") or tuple(sys.orig_argv)!=expected_argv()
                or {key:os.environ.get(key) for key in THREAD_ENV}!=THREAD_ENV
                or start_snapshot["clean"] is not True
                or start_snapshot["extension"]!=frozen_extension()):
            raise RuntimeError("P8 worker provenance invalid")
        request_doc=json.loads(paths["request"].read_text())
        expected={"protocol":WORKER_PROTOCOL,"identity":["development",12600],
            "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
            "output_json_path":str(paths["json"]),"output_npz_path":str(paths["npz"]),
            "extension":EXTENSION,"wall_limit_s":300.}
        if request_doc!=expected:raise RuntimeError("P8 worker request invalid")
        module=importlib.import_module(EXTENSION["module"])
        if not certify_module(module):raise RuntimeError("P8 resolved extension invalid")
        resolved_module=module_measurement(module)
        diagnostics=cuda_diagnostics(module)
        with np.load(paths["input"],allow_pickle=False) as archive:
            construction={key:archive[key] for key in archive.files}
        from gato_tiago.circular_portfolio_runner import _production_pin_context
        _model,_kin,rnea,_rd,_aba,_velocity,_dense=_production_pin_context();counters["pin_model_contexts"]+=1
        arrays=generate(module,construction,rnea,deadline,monotonic,counters)
        if monotonic()>deadline or not certify_output(arrays,construction):
            raise TimeoutError("P8 worker certificate deadline")
        atomic_npz(paths["npz"],arrays);end=snapshot()
        summary={"protocol":WORKER_PROTOCOL,"identity":["development",12600],
            "request_path":str(paths["request"]),"request_sha256":sha(paths["request"]),
            "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
            "npz_path":str(paths["npz"]),"npz_sha256":sha(paths["npz"]),
            "array_names":sorted(arrays),"array_hashes":{k:array_hash(arrays[k]) for k in sorted(arrays)},
            "elapsed_s":0.,"counts":counters,
            "provenance":provenance(start_snapshot,end),"extension":end["extension"],
            "resolved_module":resolved_module,
            "cuda_diagnostics":diagnostics,"certificate":True}
        if not certify_summary(summary,arrays,construction):raise RuntimeError("P8 worker summary invalid")
        finish=float(monotonic()-start_time)
        if finish>300.:raise TimeoutError("P8 worker publication wall limit")
        summary["elapsed_s"]=finish
        atomic_json(paths["json"],summary);return arrays
    except Exception as error:
        end_snapshot=snapshot()
        trigger=float(monotonic()-start_time)
        rejection={"protocol":WORKER_PROTOCOL,"stage":"worker_failed",
            "error_type":type(error).__name__,"error_message":str(error),"counts":counters,
            "internal_counts_known":True,"trigger_elapsed_s":trigger,
            "cleanup_finish_elapsed_s":float(monotonic()-start_time),"wall_limit_s":300.,
            "provenance":provenance(start_snapshot,end_snapshot),"cuda_diagnostics":diagnostics,
            "incomplete":True,"evidence":False}
        if not paths["rejected"].exists():atomic_json(paths["rejected"],rejection)
        raise


def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--request",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    if not args.execute or args.output.resolve()!=worker_paths()["json"]:raise RuntimeError("wrong P8 worker path")
    execute(args.request,WORKER_EXECUTION_AUTHORIZATION)


if __name__=="__main__":main()
