"""Capability-blocked CUDA-authoritative trajectory generator for P6."""

from __future__ import annotations

import argparse,hashlib,importlib,importlib.metadata,json,os,subprocess,sys,time
from pathlib import Path

import numpy as np

from gato_tiago.circular_portfolio_p5_worker import (CONTROL_LIMIT_COST,CYLINDER_WEIGHT,
    KKT_TOL,MAX_PCG_ITERS,MU,N_COST,PCG_TOL,Q_COST,QD_COST,Q_LIMIT_COST,RHO,
    SOLVE_RATIO,U_COST,VELOCITY_LIMIT_COST,cuda_diagnostics,certify_cuda_diagnostics)
from gato_tiago.circular_portfolio_p6 import (DENSE_SAMPLES,DENSE_SUBSTEPS,DT,EXTENSION,
    INTERVALS,KNOTS,LANES,WORKER_INPUT_SPECS,WORKER_OUTPUT_SPECS,WORKER_PROTOCOL,
    array_hash,exact_arrays,feedback_acceleration,pack_seed,worker_paths)


WORKER_EXECUTION_AUTHORIZATION=None
THREAD_ENV={"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
    "NUMEXPR_NUM_THREADS":"1"}
SOURCE_PATHS=("CMakeLists.txt","tools/build.sh","python/bindings.cu",
    "tiago_src/gato_tiago/circular_portfolio.py",
    "tiago_src/gato_tiago/circular_portfolio_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_p2.py",
    "tiago_src/gato_tiago/circular_portfolio_p2_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_p2_preflight_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p3.py",
    "tiago_src/gato_tiago/circular_portfolio_p3_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_p3_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p5.py",
    "tiago_src/gato_tiago/circular_portfolio_p5_v2.py",
    "tiago_src/gato_tiago/circular_portfolio_p5_worker.py",
    "tiago_src/gato_tiago/circular_portfolio_p6.py",
    "tiago_src/gato_tiago/circular_portfolio_p6_worker.py",
    "tiago_src/gato_tiago/circular_portfolio_runner.py",
    "tiago_src/gato_tiago/multimodal_pillar.py",
    "tiago_src/gato_tiago/config.py","gato/dynamics/tiago_right/tiago_right_arm.urdf",
    "python/bsqp/interface.py","gato/bsqp/bsqp.cuh","gato/bsqp/kernels/tool_position.cuh",
    "gato/utils/cuda.cuh",
    "gato/dynamics/integrator.cuh","gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "gato/dynamics/tiago_right/tiago_right_grid.cuh")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
            "arch":EXTENSION["arch"]}}


def expected_argv():
    paths=worker_paths()
    return ("python","-B","-m","gato_tiago.circular_portfolio_p6_worker","--execute",
        "--request",str(paths["request"]),"--output",str(paths["json"]))


def provenance(start_snapshot,end_snapshot=None):
    return {"cwd":str(Path.cwd().resolve()),"orig_argv":list(sys.orig_argv),
        "thread_environment":{k:os.environ.get(k) for k in THREAD_ENV},
        "runtime_versions":{"python":sys.version,"numpy":np.__version__,
            "pinocchio":importlib.metadata.version("pin")},
        "start_snapshot":start_snapshot,"end_snapshot":end_snapshot}


def certify_provenance(value,final):
    try:
        expected_keys={"cwd","orig_argv","thread_environment",
            "runtime_versions","start_snapshot","end_snapshot"}
        if set(value)!=expected_keys or value["cwd"]!="/workspace/GATO" \
                or tuple(value["orig_argv"])!=expected_argv() \
                or set(value["thread_environment"])!=set(THREAD_ENV) \
                or value["runtime_versions"]!={"python":sys.version,"numpy":np.__version__,
                    "pinocchio":importlib.metadata.version("pin")}:
            return False
        if value["start_snapshot"] is None:
            return not final and value["end_snapshot"] is None
        current=snapshot()
        return bool(value["thread_environment"]==THREAD_ENV
            and value["start_snapshot"]==current
            and ((not final and value["end_snapshot"] is None)
              or (final and value["end_snapshot"]==value["start_snapshot"])))
    except Exception:return False


def atomic_json(path,value):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError("P6 worker artifact exists")
    try:
        candidate.write_text(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False))
        candidate.replace(path)
    except Exception:
        if candidate.exists():candidate.unlink()
        raise


def atomic_npz(path,arrays):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError("P6 worker artifact exists")
    try:
        with candidate.open("xb") as stream:np.savez(stream,**arrays)
        candidate.replace(path)
    except Exception:
        if candidate.exists():candidate.unlink()
        raise


def constructor_args():
    return (DT,0,KKT_TOL,MAX_PCG_ITERS,PCG_TOL,SOLVE_RATIO,MU,Q_COST,QD_COST,U_COST,
        N_COST,CYLINDER_WEIGHT,CYLINDER_WEIGHT,Q_LIMIT_COST,VELOCITY_LIMIT_COST,
        CONTROL_LIMIT_COST,RHO)


def initial_counts():return {"pin_model_contexts":0,"b1_constructors":0,"b16_constructors":0,"rnea_calls":0,
    "b1_sim_forward_calls":0,"b16_sim_forward_calls":0,
    "b1_tool_position_calls":0,"b16_tool_position_calls":0,"solve_calls":0,"sqp_calls":0}


SUCCESS_COUNTS={"pin_model_contexts":1,"b1_constructors":1,"b16_constructors":1,"rnea_calls":518,
    "b1_sim_forward_calls":34188,"b16_sim_forward_calls":16835,
    "b1_tool_position_calls":34192,"b16_tool_position_calls":16837,
    "solve_calls":0,"sqp_calls":0}


def certify_counts(value,complete=False):
    return bool(isinstance(value,dict) and set(value)==set(SUCCESS_COUNTS)
        and all(type(v) is int and 0<=v<=SUCCESS_COUNTS[k] for k,v in value.items())
        and value["solve_calls"]==value["sqp_calls"]==0
        and (not complete or value==SUCCESS_COUNTS))


def certify_rejection(value):
    keys={"protocol","stage","error_type","error_message","trigger_elapsed_s",
        "cleanup_finish_elapsed_s","worker_wall_limit_s","incomplete",
        "counts","internal_counts_known","request_artifact","input_artifact",
        "provenance","cuda_diagnostics",
        "phase_timings","oracle_evidence","benchmark_evidence","sqp_evidence"}
    try:
        paths=worker_paths()
        return bool(set(value)==keys and value["protocol"]==WORKER_PROTOCOL
            and value["stage"]=="worker_failed" and value["incomplete"] is True
            and value["internal_counts_known"] is True and certify_counts(value["counts"])
            and set(value["phase_timings"])=={"constructor_rnea_elapsed_s"}
            and np.isfinite(value["phase_timings"]["constructor_rnea_elapsed_s"])
            and value["phase_timings"]["constructor_rnea_elapsed_s"]>=0
            and value["request_artifact"]=={"path":str(paths["request"]),
                "present":paths["request"].is_file(),
                "sha256":sha(paths["request"]) if paths["request"].is_file() else None,
                "size":paths["request"].stat().st_size if paths["request"].is_file() else None}
            and value["input_artifact"]=={"path":str(paths["input"]),
                "present":paths["input"].is_file(),
                "sha256":sha(paths["input"]) if paths["input"].is_file() else None,
                "size":paths["input"].stat().st_size if paths["input"].is_file() else None}
            and certify_provenance(value["provenance"],value["provenance"].get("end_snapshot") is not None)
            and (value["cuda_diagnostics"] is None
                or certify_cuda_diagnostics(value["cuda_diagnostics"]))
            and value["oracle_evidence"] is value["benchmark_evidence"]
                is value["sqp_evidence"] is False
            and value["worker_wall_limit_s"]==300.
            and np.isfinite(value["trigger_elapsed_s"])
            and np.isfinite(value["cleanup_finish_elapsed_s"])
            and 0<=value["trigger_elapsed_s"]<=value["cleanup_finish_elapsed_s"])
    except Exception:return False


def generate(module,inputs,rnea,deadline,monotonic=time.monotonic,counters=None,timings=None):
    if not exact_arrays(inputs,WORKER_INPUT_SPECS):raise ValueError("P6 worker input invalid")
    if monotonic()>=deadline:raise TimeoutError("P6 worker wall limit")
    counters=initial_counts() if counters is None else counters
    timings={"constructor_rnea_elapsed_s":0.} if timings is None else timings
    b1=module.BSQP_1_float(*constructor_args());counters["b1_constructors"]+=1
    b16=module.BSQP_16_float(*constructor_args());counters["b16_constructors"]+=1
    def tool1(q):
        value=np.asarray(b1.tool_position(q));counters["b1_tool_position_calls"]+=1;return value
    def sim1(x,u,dt):
        value=np.asarray(b1.sim_forward(x,u,dt));counters["b1_sim_forward_calls"]+=1;return value
    def tool16(q):
        value=np.asarray(b16.tool_position(q));counters["b16_tool_position_calls"]+=1;return value
    def sim16(x,u,dt):
        value=np.asarray(b16.sim_forward(x,u,dt));counters["b16_sim_forward_calls"]+=1;return value
    states=np.empty((2,KNOTS,14),np.float32);tools=np.empty((2,KNOTS,3),np.float32)
    controls=np.empty((2,INTERVALS,7),np.float32)
    rq=np.empty((2,INTERVALS,7),np.float64);rv=np.empty_like(rq)
    ra=np.empty_like(rq);ru=np.empty_like(rq)
    for route in range(2):
        proxy={"planned_q_float64":inputs[("short" if route==0 else "long")+"_proxy_q_float64"],
            "planned_qd_float64":inputs[("short" if route==0 else "long")+"_proxy_qd_float64"],
            "planned_qdd_float64":inputs[("short" if route==0 else "long")+"_proxy_qdd_float64"]}
        states[route,0]=inputs["x0_float32"]
        tools[route,0]=tool1(states[route,0,:7])[0]
        for knot in range(INTERVALS):
            if monotonic()>=deadline:raise TimeoutError("P6 worker wall limit")
            acceleration=feedback_acceleration(proxy,knot,states[route,knot])
            rq[route,knot]=states[route,knot,:7];rv[route,knot]=states[route,knot,7:]
            ra[route,knot]=acceleration;rnea_start=monotonic()
            ru[route,knot]=np.asarray(rnea(rq[route,knot],rv[route,knot],ra[route,knot]),np.float64)
            timings["constructor_rnea_elapsed_s"]+=float(monotonic()-rnea_start)
            counters["rnea_calls"]+=1
            controls[route,knot]=ru[route,knot].astype(np.float32)
            value=sim1(states[route,knot],controls[route,knot],DT)
            if value.shape!=(1,14):raise RuntimeError("P6 B1 layout invalid")
            states[route,knot+1]=value[0]
            tools[route,knot+1]=tool1(states[route,knot+1,:7])[0]
    # Independent full-DT B1 transition replay proves exact solver-knot feasibility.
    independent=np.empty_like(states);independent_tool=np.empty_like(tools)
    independent[:,0]=inputs["x0_float32"]
    independent_tool[:,0]=np.stack([tool1(inputs["x0_float32"][:7])[0]]*2)
    for route in range(2):
        for knot in range(INTERVALS):
            if monotonic()>=deadline:raise TimeoutError("P6 worker wall limit")
            independent[route,knot+1]=sim1(
                independent[route,knot],controls[route,knot],DT)[0]
            independent_tool[route,knot+1]=tool1(independent[route,knot+1,:7])[0]
    # Dense held-control replay is a separate physical-feasibility witness.
    dense=np.empty((2,DENSE_SAMPLES,14),np.float32);dense_tool=np.empty((2,DENSE_SAMPLES,3),np.float32)
    dense[:,0]=inputs["x0_float32"]
    dense_tool[:,0]=np.stack([tool1(inputs["x0_float32"][:7])[0]]*2)
    for route in range(2):
        index=0
        for knot in range(INTERVALS):
            for _ in range(DENSE_SUBSTEPS):
                if monotonic()>=deadline:raise TimeoutError("P6 worker wall limit")
                dense[route,index+1]=sim1(
                    dense[route,index],controls[route,knot],DT/DENSE_SUBSTEPS)[0]
                index+=1;dense_tool[route,index]=tool1(dense[route,index,:7])[0]
    lane_controls=np.stack([controls[lane%2] for lane in range(LANES)])
    knot16=np.empty((LANES,KNOTS,14),np.float32);knot16[:,0]=inputs["x0_float32"]
    knot_tool16=np.empty((LANES,KNOTS,3),np.float32)
    knot_tool16[:,0]=tool16(knot16[:,0,:7])
    for knot in range(INTERVALS):
        if monotonic()>=deadline:raise TimeoutError("P6 worker wall limit")
        knot16[:,knot+1]=sim16(knot16[:,knot],lane_controls[:,knot],DT)
        knot_tool16[:,knot+1]=tool16(knot16[:,knot+1,:7])
    fresh=np.empty((LANES,DENSE_SAMPLES,14),np.float32);fresh[:,0]=inputs["x0_float32"]
    fresh_tool=np.empty((LANES,DENSE_SAMPLES,3),np.float32);index=0
    fresh_tool[:,0]=tool16(fresh[:,0,:7])
    for knot in range(INTERVALS):
        if monotonic()>=deadline:raise TimeoutError("P6 worker wall limit")
        for _ in range(DENSE_SUBSTEPS):
            if monotonic()>=deadline:raise TimeoutError("P6 worker wall limit")
            value=sim16(fresh[:,index],lane_controls[:,knot],DT/DENSE_SUBSTEPS)
            if value.shape!=(LANES,14):raise RuntimeError("P6 B16 layout invalid")
            fresh[:,index+1]=value;index+=1
            fresh_tool[:,index]=tool16(fresh[:,index,:7])
    seeds=np.stack([pack_seed(states[lane%2,:,:7],states[lane%2,:,7:],controls[lane%2])
                    for lane in range(LANES)])
    result={"captured_x0_float32":inputs["x0_float32"],"generated_state_float32":states,
        "generated_knot_state_float32":states,"generated_knot_tool_float32":tools,
        "rnea_q_float64":rq,"rnea_qd_float64":rv,"rnea_qdd_float64":ra,
        "rnea_u_float64":ru,"recorded_controls_float32":controls,
        "independent_b1_knot_state_float32":independent,
        "independent_b1_knot_tool_float32":independent_tool,
        "b1_knot_transition_defect_float32":independent[:,1:]-states[:,1:],
        "b1_dense_state_float32":dense,"b1_dense_tool_float32":dense_tool,
        "fresh_b16_knot_state_float32":knot16,"fresh_b16_knot_tool_float32":knot_tool16,
        "b16_knot_transition_defect_float32":knot16[:,1:]
            -np.stack([states[lane%2,1:] for lane in range(LANES)]),
        "fresh_b16_state_float32":fresh,"fresh_b16_tool_float32":fresh_tool,
        "b1_seed_xu_float32":seeds[0],"b16_seed_xu_float32":seeds}
    result.pop("generated_state_float32")
    if not exact_arrays(result,WORKER_OUTPUT_SPECS):raise RuntimeError("P6 output schema invalid")
    return result


def certify_output(arrays):
    if not exact_arrays(arrays,WORKER_OUTPUT_SPECS):return False
    states=arrays["generated_knot_state_float32"];controls=arrays["recorded_controls_float32"]
    expected=np.stack([pack_seed(states[lane%2,:,:7],states[lane%2,:,7:],controls[lane%2])
                       for lane in range(LANES)])
    mapping=all(np.array_equal(arrays["fresh_b16_knot_state_float32"][lane],states[lane%2])
        and np.array_equal(arrays["fresh_b16_knot_tool_float32"][lane],
                           arrays["generated_knot_tool_float32"][lane%2])
        and np.array_equal(arrays["fresh_b16_state_float32"][lane],
                           arrays["b1_dense_state_float32"][lane%2])
        and np.array_equal(arrays["fresh_b16_tool_float32"][lane],
                           arrays["b1_dense_tool_float32"][lane%2])
        and np.array_equal(arrays["b16_seed_xu_float32"][lane],expected[lane])
        for lane in range(LANES))
    initial=arrays["captured_x0_float32"]
    return bool(np.array_equal(arrays["b16_seed_xu_float32"],expected)
        and all(np.array_equal(value,initial) for value in (
            states[0,0],states[1,0],arrays["independent_b1_knot_state_float32"][0,0],
            arrays["independent_b1_knot_state_float32"][1,0],
            arrays["b1_dense_state_float32"][0,0],arrays["b1_dense_state_float32"][1,0]))
        and all(np.array_equal(value,initial) for value in arrays["fresh_b16_knot_state_float32"][:,0])
        and all(np.array_equal(value,initial) for value in arrays["fresh_b16_state_float32"][:,0])
        and np.array_equal(arrays["b1_seed_xu_float32"],expected[0])
        and np.array_equal(arrays["independent_b1_knot_state_float32"],states)
        and np.array_equal(arrays["independent_b1_knot_tool_float32"],
                           arrays["generated_knot_tool_float32"])
        and not np.any(arrays["b1_knot_transition_defect_float32"])
        and not np.any(arrays["b16_knot_transition_defect_float32"])
        and np.array_equal(arrays["rnea_u_float64"].astype(np.float32),controls)
        and np.array_equal(arrays["rnea_q_float64"],states[:,:-1,:7].astype(np.float64))
        and np.array_equal(arrays["rnea_qd_float64"],states[:,:-1,7:].astype(np.float64))
        and mapping
        and not np.array_equal(states[0],states[1])
        and len({arrays["fresh_b16_state_float32"][lane].tobytes() for lane in range(LANES)})==2)


def execute(request,authorization=None,monotonic=time.monotonic,counters=None,timings=None): # pragma: no cover
    deadline=monotonic()+300.
    if WORKER_EXECUTION_AUTHORIZATION is None or authorization is not WORKER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P6 worker blocked")
    paths=worker_paths()
    if Path(request).resolve()!=paths["request"]:raise RuntimeError("wrong P6 worker request")
    value=json.loads(paths["request"].read_text())
    expected={"protocol":WORKER_PROTOCOL,"identity":["development",12600],
        "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
        "output_json_path":str(paths["json"]),"output_npz_path":str(paths["npz"]),
        "extension":EXTENSION,"wall_limit_s":300.0}
    if value!=expected:
        raise RuntimeError("P6 worker request invalid")
    from gato_tiago.circular_portfolio_runner import _production_pin_context
    _model,_kinematics,rnea,_rd,_aba,_velocity,_dense=_production_pin_context()
    counters["pin_model_contexts"]+=1
    module=importlib.import_module(EXTENSION["module"])
    with np.load(paths["input"],allow_pickle=False) as archive:inputs={k:archive[k] for k in archive.files}
    arrays=generate(module,inputs,rnea,deadline,monotonic,counters,timings)
    if not certify_output(arrays):raise RuntimeError("P6 worker certificate failed")
    return arrays


def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--request",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    paths=worker_paths()
    if args.request.resolve()!=paths["request"] or args.output.resolve()!=paths["json"]:
        raise RuntimeError("wrong P6 worker path")
    start=time.monotonic();deadline=start+300.;start_snapshot=None;counts=initial_counts();diagnostics=None
    timings={"constructor_rnea_elapsed_s":0.}
    try:
        if {k:os.environ.get(k) for k in THREAD_ENV}!=THREAD_ENV:
            raise RuntimeError("P6 worker requires single-thread environment")
        if Path.cwd().resolve()!=Path("/workspace/GATO") or tuple(sys.orig_argv)!=expected_argv():
            raise RuntimeError("P6 worker invocation mismatch")
        start_snapshot=snapshot()
        if start_snapshot["clean"] is not True or start_snapshot["extension"]["sha256"] \
                !=EXTENSION["sha256"] or start_snapshot["extension"]["size"]!=EXTENSION["size"]:
            raise RuntimeError("P6 worker start provenance invalid")
        arrays=execute(args.request,WORKER_EXECUTION_AUTHORIZATION,counters=counts,timings=timings)
        if time.monotonic()>deadline:raise TimeoutError("P6 worker wall limit")
        atomic_npz(paths["npz"],arrays)
        module=importlib.import_module(EXTENSION["module"]);module_path=Path(module.__file__).resolve()
        imported={"path":str(module_path),"sha256":sha(module_path),"size":module_path.stat().st_size}
        expected_path=(Path(__file__).resolve().parents[2]/EXTENSION["relative_path"]).resolve()
        if imported!={"path":str(expected_path),"sha256":EXTENSION["sha256"],"size":EXTENSION["size"]}:
            raise RuntimeError("P6 imported extension pin mismatch")
        diagnostics=cuda_diagnostics()
        end_snapshot=snapshot()
        if end_snapshot!=start_snapshot:raise RuntimeError("P6 worker provenance changed")
        if time.monotonic()>deadline:raise TimeoutError("P6 worker wall limit")
        summary={"protocol":WORKER_PROTOCOL,"identity":["development",12600],
            "request_path":str(paths["request"]),"request_sha256":sha(paths["request"]),
            "input_path":str(paths["input"]),"input_sha256":sha(paths["input"]),
            "npz_path":str(paths["npz"]),"npz_sha256":sha(paths["npz"]),
            "array_names":sorted(arrays),"array_hashes":{k:array_hash(arrays[k]) for k in sorted(arrays)},
            "elapsed_s":None,"phase_timings":timings,"certificate":True,
            "provenance":provenance(start_snapshot,end_snapshot),
            "imported_extension":imported,"cuda_diagnostics":diagnostics,
            **counts}
        if not certify_counts(counts,True):raise RuntimeError("P6 worker counts invalid")
        summary["elapsed_s"]=float(time.monotonic()-start)
        if time.monotonic()>deadline:raise TimeoutError("P6 worker wall limit")
        atomic_json(paths["json"],summary);return summary
    except Exception as error:
        trigger=float(time.monotonic()-start)
        end=snapshot() if start_snapshot is not None else None
        cleanup=float(time.monotonic()-start)
        rejection={"protocol":WORKER_PROTOCOL,"stage":"worker_failed",
            "error_type":type(error).__name__,"error_message":str(error),
            "trigger_elapsed_s":trigger,"cleanup_finish_elapsed_s":cleanup,
            "worker_wall_limit_s":300.,"incomplete":True,
            "counts":counts,"internal_counts_known":True,
            "phase_timings":timings,
            "request_artifact":{"path":str(paths["request"]),"present":paths["request"].is_file(),
                "sha256":sha(paths["request"]) if paths["request"].is_file() else None,
                "size":paths["request"].stat().st_size if paths["request"].is_file() else None},
            "input_artifact":{"path":str(paths["input"]),"present":paths["input"].is_file(),
                "sha256":sha(paths["input"]) if paths["input"].is_file() else None,
                "size":paths["input"].stat().st_size if paths["input"].is_file() else None},
            "provenance":provenance(start_snapshot,end),
            "cuda_diagnostics":diagnostics,
            "oracle_evidence":False,"benchmark_evidence":False,"sqp_evidence":False}
        if not paths["rejected"].exists():atomic_json(paths["rejected"],rejection)
        raise


if __name__=="__main__":main()
