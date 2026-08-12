"""One-shot B1 versus B16 solve comparison from the accepted cached route seeds."""

from __future__ import annotations

import argparse,copy,hashlib,importlib,json,os,subprocess,sys,time
from pathlib import Path

import numpy as np

from gato_tiago import circular_portfolio_cached_replay as cached
from gato_tiago import circular_portfolio_p8 as schema
from gato_tiago import circular_portfolio_p8_worker as worker
from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite,_production_pin_context


ROOT=Path("/tmp/tiago-circular-solve-compare-v3")
OUTPUT_JSON=ROOT/"result.json";OUTPUT_NPZ=ROOT/"result.npz";FAILURE_JSON=ROOT/"failure.json"
CACHED_JSON=Path("/tmp/tiago-circular-cached-replay/result.json")
CACHED_NPZ=Path("/tmp/tiago-circular-cached-replay/result.npz")
CACHED_JSON_SHA256="fee0f770ff8f44202d75f97ca1ca3c152a8e46c2afbb97a8c86c0a948f56566a"
CACHED_NPZ_SHA256="791755205f3d0c66fa6388d80525322cfa27dc61d3bf57cd4a3b23e2c33ab7f2"
CACHED_JSON_SIZE=13005;CACHED_NPZ_SIZE=1220148
AUTHORIZED_CWD="/workspace/GATO"
THREAD_ENV={"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
    "NUMEXPR_NUM_THREADS":"1"}
AUTHORIZED_ORIG_ARGV=("python","-B","-m","gato_tiago.circular_portfolio_solve_compare",
    "--output-root",str(ROOT))
SOLVER_OPTIONS={"dt":.05,"max_sqp_iters":60,"kkt_tol":1e-3,"max_pcg_iters":500,
    "pcg_tol":8e-4,"solve_ratio":1.,"mu":20.,"q_cost":2.,"qd_cost":.15,
    "u_cost":7.5e-5,"n_cost":260.,"cylinder_running":800.,"cylinder_terminal":800.,
    "q_barrier":.01,"v_barrier":.001,"u_barrier":.003,"rho":.01}
DEFECT_TOLERANCE=1e-4
SOLVER_EXTENSION={
    "module":"bsqp.bsqpN260_tiago_right_constructed_route_portfolio_toll_pcg_compact",
    "relative_path":"python/bsqp/bsqpN260_tiago_right_constructed_route_portfolio_toll_pcg_compact.cpython-310-x86_64-linux-gnu.so",
    "sha256":"464a72cf57440f948d6b40adf0a61caaa27071657721a8b7a488e55dd3e05432",
    "size":6678192,"build_head":"3aaf757dd2e0e0534bd1498aa57b3a4c5324a0de",
    "arch":"61-real","KNOT_POINTS":260,"REFERENCE_SIZE":10,
    "TOOL_POSITION_FRAME":"arm_right_tool_joint_origin","TOOL_POSITION_SIZE":3,
    "pcg_source_sha256":"38631d3716b0ea96c6bade40ecc659c83e51d73d5649ae05549df6a240c1291f"}
SOURCE_PATHS=tuple(dict.fromkeys((*worker.SOURCE_PATHS,
    "gato/bsqp/kernels/pcg.cuh",
    "tiago_src/gato_tiago/circular_portfolio_solve_compare.py",
    "tiago_src/gato_tiago/circular_portfolio_cached_replay.py",
    "tiago_src/gato_tiago/circular_portfolio_p8_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p5_v2_runner.py")))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError("solve comparison artifact exists")
    try:
        candidate.write_text(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False))
        candidate.replace(path)
    finally:
        if candidate.exists():candidate.unlink()


def atomic_npz(path,arrays):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError("solve comparison artifact exists")
    try:
        with candidate.open("xb") as stream:np.savez(stream,**arrays)
        candidate.replace(path)
    finally:
        if candidate.exists():candidate.unlink()


def builtin(value):
    if isinstance(value,dict):return {str(key):builtin(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [builtin(item) for item in value]
    if isinstance(value,np.ndarray):return builtin(value.tolist())
    if isinstance(value,np.generic):return builtin(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def snapshot():
    repo=Path(__file__).resolve().parents[2]
    run=lambda *args:subprocess.run(["git",*args],cwd=repo,check=True,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
    extension=frozen_extension();path=Path(extension["path"])
    measured={**extension,"sha256":sha(path),"size":path.stat().st_size}
    return {"head":run("rev-parse","HEAD"),
        "clean":run("status","--porcelain","--untracked-files=no")=="",
        "cwd":str(Path.cwd().resolve()),"orig_argv":list(sys.orig_argv),
        "thread_environment":{key:os.environ.get(key) for key in THREAD_ENV},
        "sources":{name:sha(repo/name) for name in SOURCE_PATHS},"extension":measured}


def certify_module(module):
    try:return bool(module_measurement(module)==frozen_extension()
        and hasattr(module,"BSQP_1_float") and hasattr(module,"BSQP_16_float"))
    except Exception:return False


def frozen_extension():
    root=Path(__file__).resolve().parents[2]
    return {"path":str((root/SOLVER_EXTENSION["relative_path"]).resolve()),
        **{key:SOLVER_EXTENSION[key] for key in ("sha256","size","build_head","arch",
            "pcg_source_sha256")},"attributes":{key:SOLVER_EXTENSION[key] for key in
            ("KNOT_POINTS","REFERENCE_SIZE","TOOL_POSITION_FRAME","TOOL_POSITION_SIZE")}}


def module_measurement(module):
    root=Path(__file__).resolve().parents[2];path=Path(module.__file__).resolve()
    return {"path":str(path),"sha256":sha(path),"size":path.stat().st_size,
        **{key:SOLVER_EXTENSION[key] for key in ("build_head","arch")},
        "pcg_source_sha256":sha(root/"gato/bsqp/kernels/pcg.cuh"),
        "attributes":{key:getattr(module,key) for key in
            ("KNOT_POINTS","REFERENCE_SIZE","TOOL_POSITION_FRAME","TOOL_POSITION_SIZE")}}


def certify_source_history(stored,current):
    try:
        keys={"head","clean","cwd","orig_argv","thread_environment","sources","extension"}
        start=stored["start"];end=stored["end"]
        stable=keys-{"orig_argv"}
        return bool(set(stored)=={"start","end"} and set(start)==set(end)==set(current)==keys
            and start==end and tuple(start["orig_argv"])==AUTHORIZED_ORIG_ARGV
            and all(current[key]==start[key] for key in stable)
            and current["clean"] is True and current["cwd"]==AUTHORIZED_CWD
            and current["thread_environment"]==THREAD_ENV)
    except Exception:return False


def load_inputs(cached_json=CACHED_JSON,cached_npz=CACHED_NPZ):
    cached_json=Path(cached_json);cached_npz=Path(cached_npz)
    if (cached_json.resolve()!=CACHED_JSON or cached_npz.resolve()!=CACHED_NPZ
            or not cached_json.is_file() or not cached_npz.is_file()
            or cached_json.stat().st_size!=CACHED_JSON_SIZE
            or cached_npz.stat().st_size!=CACHED_NPZ_SIZE
            or sha(cached_json)!=CACHED_JSON_SHA256 or sha(cached_npz)!=CACHED_NPZ_SHA256):
        raise RuntimeError("cached replay pin mismatch")
    document=json.loads(cached_json.read_text())
    with np.load(cached_npz,allow_pickle=False) as archive:replay={key:archive[key] for key in archive.files}
    construction=cached.load_cached_construction()
    if (document.get("passes") is not True or document.get("output_certificate") is not True
            or document.get("campaign_certificate",{}).get("passes") is not True
            or document.get("output_npz_sha256")!=CACHED_NPZ_SHA256
            or not schema.exact_arrays(replay,schema.WORKER_OUTPUT_SPECS)):
        raise RuntimeError("cached replay evidence invalid")
    return document,replay,construction


def solver_args():
    o=SOLVER_OPTIONS
    return (o["dt"],o["max_sqp_iters"],o["kkt_tol"],o["max_pcg_iters"],o["pcg_tol"],
        o["solve_ratio"],o["mu"],o["q_cost"],o["qd_cost"],o["u_cost"],o["n_cost"],
        o["cylinder_running"],o["cylinder_terminal"],o["q_barrier"],o["v_barrier"],
        o["u_barrier"],o["rho"])


def unpack_xu(value,batch):
    value=np.asarray(value)
    if value.shape!=(batch,schema.KNOTS*21-7) or value.dtype!=np.float32 or not np.isfinite(value).all():
        raise ValueError("solver XU schema invalid")
    q=np.empty((batch,schema.KNOTS,7),np.float32)
    qd=np.empty_like(q);u=np.empty((batch,schema.INTERVALS,7),np.float32)
    for knot in range(schema.KNOTS):
        offset=21*knot;q[:,knot]=value[:,offset:offset+7];qd[:,knot]=value[:,offset+7:offset+14]
        if knot<schema.INTERVALS:u[:,knot]=value[:,offset+14:offset+21]
    return q,qd,u


STAT_KEYS={"XU","sqp_time_us","sqp_iters","kkt_converged","final_merit","initial_merit",
    "ls_num_iters","pcg_times_us","pcg_iters","ls_min_merit","ls_step_size"}


def normalize_solve(raw,batch):
    if not isinstance(raw,dict) or set(raw)!=STAT_KEYS:raise ValueError("solver result schema invalid")
    out={"XU":np.asarray(raw["XU"],np.float32).copy(),
        "sqp_time_us":np.asarray(raw["sqp_time_us"],np.int64),
        "sqp_iters":np.asarray(raw["sqp_iters"],np.int32).reshape(batch).copy(),
        "kkt_converged":np.asarray(raw["kkt_converged"],np.int32).reshape(batch).copy(),
        "initial_merit":np.asarray(raw["initial_merit"],np.float32).reshape(batch).copy(),
        "final_merit":np.asarray(raw["final_merit"],np.float32).reshape(batch).copy(),
        "ls_num_iters":np.asarray(raw["ls_num_iters"],np.int64),
        "pcg_times_us":np.asarray(raw["pcg_times_us"],np.float32).copy(),
        "pcg_iters":np.asarray(raw["pcg_iters"],np.int32).copy(),
        "ls_min_merit":np.asarray(raw["ls_min_merit"],np.float32).copy(),
        "ls_step_size":np.asarray(raw["ls_step_size"],np.float32).copy()}
    if out["XU"].shape!=(batch,schema.KNOTS*21-7):raise ValueError("solver XU shape invalid")
    iterations=int(out["ls_num_iters"])
    pcg_rows=out["pcg_iters"].shape[0] if out["pcg_iters"].ndim==2 else -1
    if (not 0<=iterations<=SOLVER_OPTIONS["max_sqp_iters"]
            or pcg_rows not in (iterations,iterations+1)
            or out["pcg_iters"].shape!=(pcg_rows,batch)
            or out["ls_min_merit"].shape!=(iterations,batch)
            or out["ls_step_size"].shape!=(iterations,batch)
            or out["pcg_times_us"].shape!=(pcg_rows,)):
        raise ValueError("solver telemetry shape invalid")
    if any(not np.isfinite(value).all() for value in out.values()):raise ValueError("solver result nonfinite")
    return out


def replay_solution(solver,x0,controls,batch):
    state=np.empty((batch,schema.KNOTS,14),np.float32);state[:,0]=x0
    for knot in range(schema.INTERVALS):
        value=np.asarray(solver.sim_forward(state[:,knot],controls[:,knot],schema.DT),np.float32)
        if value.shape!=(batch,14):raise RuntimeError("solver replay layout invalid")
        state[:,knot+1]=value
    tool=np.asarray([solver.tool_position(state[:,knot,:7]) for knot in range(schema.KNOTS)],
        np.float32).transpose(1,0,2)
    if tool.shape!=(batch,schema.KNOTS,3):raise RuntimeError("solver tool layout invalid")
    return state,tool


def evaluate_lanes(planned,replayed,tool,controls,construction,prerequisite,kinematics):
    lower=prerequisite["joint_lower_float64"];upper=prerequisite["joint_upper_float64"]
    velocity=prerequisite["velocity_limit_float64"];effort=prerequisite["effort_limit_float64"]
    midpoint=.5*(lower+upper);half=.5*(upper-lower);pillar=construction["pillar_float64"]
    goal=construction["goal_tool_float64"];default=int(prerequisite["public_default_side_int8"][0])
    rows=[];affine=np.empty((len(replayed),schema.AFFINE_SAMPLES,3),np.float32)
    pin_tool=np.empty_like(tool,dtype=np.float64);terminal_j=np.empty((len(replayed),3,7),np.float64)
    for lane in range(len(replayed)):
        family="short" if lane<8 else "long";route=0 if family=="short" else 1
        for knot,q in enumerate(replayed[lane,:,:7]):
            pin_tool[lane,knot],jac=kinematics(q.astype(np.float64))
            if knot==schema.INTERVALS:terminal_j[lane]=jac
        index=0
        for knot in range(schema.INTERVALS):
            for substep in range(schema.AFFINE_SUBSTEPS):
                alpha=substep/schema.AFFINE_SUBSTEPS
                q=(1-alpha)*replayed[lane,knot,:7]+alpha*replayed[lane,knot+1,:7]
                affine[lane,index]=kinematics(q.astype(np.float64))[0];index+=1
        affine[lane,-1]=pin_tool[lane,-1]
        difference=np.linalg.norm(tool[lane].astype(np.float64)-construction["target_tool_float64"][route],axis=1)
        cost=schema.reconstruct_cost(replayed[lane],controls[lane],tool[lane],
            construction["reference_float32"],lower,upper,velocity,effort)
        turn=schema.open_turn(tool[lane],pillar);expected=-default if family=="short" else default
        metrics={"q_ratio":float(np.max(np.abs((replayed[lane,:,:7]-midpoint)/half))),
            "v_ratio":float(np.max(np.abs(replayed[lane,:,7:])/velocity)),
            "u_ratio":float(np.max(np.abs(controls[lane])/effort)),
            "terminal_m":float(np.linalg.norm(tool[lane,-1]-goal)),
            "speed_mps":float(np.linalg.norm(terminal_j[lane]@replayed[lane,-1,7:])),
            "clearance_m":float(np.min(np.linalg.norm(affine[lane,:,:2]-pillar,axis=1)-.03)),
            "same_q_fk_max_m":float(np.max(np.linalg.norm(tool[lane]-pin_tool[lane],axis=1))),
            "arc_rms_m":float(np.sqrt(np.mean(difference**2))),"arc_max_m":float(np.max(difference)),
            "turn_rad":turn,"path_length_m":float(np.sum(np.linalg.norm(np.diff(tool[lane],axis=0),axis=1))),
            "planned_replay_defect_max":float(np.max(np.abs(planned[lane]-replayed[lane]))),
            "base_cost":cost["base"],"toll":cost["toll"],"full_cost":cost["full"]}
        gates={"finite":all(np.isfinite(value) for value in metrics.values()),
            "limits":max(metrics[key] for key in ("q_ratio","v_ratio","u_ratio"))<=1.,
            "terminal":metrics["terminal_m"]<=.015,"speed":metrics["speed_mps"]<=.05,
            "clearance":metrics["clearance_m"]>=.005,"same_q_fk":metrics["same_q_fk_max_m"]<=.001,
            "arc":metrics["arc_rms_m"]<=.002 and metrics["arc_max_m"]<=.004,
            "topology":abs(turn)>=(2.4 if family=="short" else 3.) and int(np.sign(turn))==expected,
            "dynamics":metrics["planned_replay_defect_max"]<=DEFECT_TOLERANCE,
            "cost_decomposition":abs(cost["full"]-cost["base"]-cost["toll"])<=1e-10}
        rows.append({"lane":lane,"seed_family":family,"metrics":metrics,"gates":gates,
            "topology_preserved":gates["topology"],"cost":cost,"passes":bool(all(gates.values()))})
    return rows,affine,pin_tool,terminal_j


def solver_certificate(stats):
    batch=len(stats["sqp_iters"]);iterations=int(stats["ls_num_iters"])
    pcg_rows=int(stats["pcg_iters"].shape[0])
    telemetry={"sqp_time_nonnegative":int(stats["sqp_time_us"])>=0,
        "pcg_counts_in_range":bool(np.all((stats["pcg_iters"]>=0)
            &(stats["pcg_iters"]<=SOLVER_OPTIONS["max_pcg_iters"]))),
        "pcg_times_nonnegative":bool(np.all(stats["pcg_times_us"]>=0)),
        "step_sizes_in_range":bool(np.all((stats["ls_step_size"]==-1)
            |((stats["ls_step_size"]>0)&(stats["ls_step_size"]<=1)))),
        "sqp_counts_consistent":bool(np.all((stats["sqp_iters"]>=0)
            &(stats["sqp_iters"]<=pcg_rows))),
        "kkt_binary":bool(np.all((stats["kkt_converged"]==0)|(stats["kkt_converged"]==1)))}
    lane=[]
    for index in range(batch):
        initial=float(stats["initial_merit"][index]);final=float(stats["final_merit"][index])
        row={"lane":index,"initial_merit":initial,"final_merit":final,
            "sqp_iters":int(stats["sqp_iters"][index]),"kkt_converged":int(stats["kkt_converged"][index]),
            "max_pcg_iters":int(np.max(stats["pcg_iters"][:,index])) if stats["pcg_iters"].size else 0}
        row["passes"]=bool(np.isfinite(initial) and np.isfinite(final) and final<=initial
            and 0<=row["sqp_iters"]<=pcg_rows and row["kkt_converged"]==1
            and 0<=row["max_pcg_iters"]<=500);lane.append(row)
    return {"lanes":lane,"ls_num_iters":iterations,"sqp_time_us":int(stats["sqp_time_us"]),
        "telemetry_gates":telemetry,
        "passes":bool(all(row["passes"] for row in lane) and all(telemetry.values()))}


def comparison_certificate(b1_rows,b16_rows,b1_stats,b16_stats):
    b1_schema=bool(len(b1_rows)==1 and b1_rows[0].get("lane")==0
        and b1_rows[0].get("seed_family")=="short")
    b16_schema=bool(len(b16_rows)==16 and all(row.get("lane")==lane
        and row.get("seed_family")== ("short" if lane<8 else "long")
        for lane,row in enumerate(b16_rows)))
    feasible_short=[row for row in b16_rows[:8] if row["passes"]]
    feasible_long=[row for row in b16_rows[8:] if row["passes"]]
    feasible=[*feasible_short,*feasible_long]
    best=min(feasible,key=lambda row:row["metrics"]["full_cost"]) if feasible else None
    b1=b1_rows[0];improvement=(b1["metrics"]["full_cost"]-best["metrics"]["full_cost"] if best else None)
    threshold=max(1.,.05*max(1.,abs(b1["metrics"]["full_cost"])))
    gates={"b1_short_feasible":b1["seed_family"]=="short" and b1["passes"],
        "b1_lane_schema":b1_schema,"b16_lane_schema":b16_schema,
        "all_b16_lanes_pass":len(feasible)==16,
        "portfolio_has_short":bool(feasible_short),"portfolio_has_long":bool(feasible_long),
        "best_is_long":best is not None and best["seed_family"]=="long",
        "portfolio_beats_b1":best is not None and improvement>=threshold,
        "b1_solver":b1_stats["passes"],"b16_solver":b16_stats["passes"]}
    return {"gates":gates,"best_feasible_lane":best["lane"] if best else None,
        "best_feasible_family":best["seed_family"] if best else None,"improvement":improvement,
        "required_improvement":threshold,"passes":bool(all(gates.values()))}


def result_arrays(seed1,seed16,out1,out16,replay1,tool1,replay16,tool16,affine1,affine16,
                  pin1,pin16,j1,j16,x0,reference):
    arrays={"input_seed_b1_float32":seed1,"input_seed_b16_float32":seed16,
        "input_x0_b1_float32":x0[None].copy(),"input_x0_b16_float32":np.tile(x0,(16,1)),
        "input_reference_b1_float32":reference[None].copy(),
        "input_reference_b16_float32":np.tile(reference,(16,1)),
        "output_xu_b1_float32":out1["XU"],"output_xu_b16_float32":out16["XU"],
        "replay_state_b1_float32":replay1,"replay_tool_b1_float32":tool1,
        "replay_state_b16_float32":replay16,"replay_tool_b16_float32":tool16,
        "affine_pin_tool_b1_float32":affine1,"affine_pin_tool_b16_float32":affine16,
        "identical_state_pin_tool_b1_float64":pin1,"identical_state_pin_tool_b16_float64":pin16,
        "terminal_pin_jacobian_b1_float64":j1,"terminal_pin_jacobian_b16_float64":j16}
    for prefix,value in (("b1",out1),("b16",out16)):
        for key in STAT_KEYS-{"XU"}:arrays[f"raw_{key}_{prefix}"]=value[key]
    return arrays


def execute(root=ROOT,module_loader=importlib.import_module,monotonic=time.monotonic,
            prerequisite_loader=authenticate_cpu_prerequisite,pin_loader=_production_pin_context,
            cuda_probe=worker.cuda_diagnostics): # pragma: no cover - native boundary
    root=Path(root)
    if root.exists():raise FileExistsError("solve comparison root exists")
    root.mkdir();start=monotonic();counts={"b1_constructors":0,"b16_constructors":0,
        "b1_solve_calls":0,"b16_solve_calls":0};provenance_start=None
    try:
        document,cached_arrays,construction=load_inputs()
        provenance_start=snapshot()
        if (provenance_start["clean"] is not True or provenance_start["cwd"]!=AUTHORIZED_CWD
                or provenance_start["thread_environment"]!=THREAD_ENV
                or tuple(provenance_start["orig_argv"])!=AUTHORIZED_ORIG_ARGV
                or provenance_start["extension"]!=frozen_extension()):
            raise RuntimeError("solve comparison provenance invalid")
        module=module_loader(SOLVER_EXTENSION["module"])
        if not certify_module(module):raise RuntimeError("solve comparison module invalid")
        diagnostics=cuda_probe();_auth,prerequisite=prerequisite_loader()
        _model,kinematics,*_=pin_loader()
        seed1=np.ascontiguousarray(cached_arrays["b1_seed_float32"][None].copy())
        seed16=np.ascontiguousarray(cached_arrays["b16_seed_float32"].copy())
        if (not np.array_equal(seed1[0],seed16[0])
                or not all(np.array_equal(seed16[lane],seed1[0]) for lane in range(8))
                or not all(np.array_equal(seed16[lane],cached_arrays["b16_seed_float32"][8])
                           for lane in range(8,16))
                or not all(np.array_equal(seed16[lane],cached_arrays["b16_seed_float32"][lane])
                           for lane in range(16))):raise RuntimeError("solve seed mapping invalid")
        b1=module.BSQP_1_float(*solver_args());counts["b1_constructors"]+=1
        b16=module.BSQP_16_float(*solver_args());counts["b16_constructors"]+=1
        x0=construction["x0_float32"];reference=construction["reference_float32"]
        t=monotonic();raw1=b1.solve(seed1.copy(),schema.DT,x0[None].copy(),reference[None].copy())
        wall1=float(monotonic()-t);counts["b1_solve_calls"]+=1;out1=normalize_solve(raw1,1)
        t=monotonic();raw16=b16.solve(seed16.copy(),schema.DT,np.tile(x0,(16,1)),np.tile(reference,(16,1)))
        wall16=float(monotonic()-t);counts["b16_solve_calls"]+=1;out16=normalize_solve(raw16,16)
        q1,qd1,u1=unpack_xu(out1["XU"],1);q16,qd16,u16=unpack_xu(out16["XU"],16)
        planned1=np.concatenate((q1,qd1),axis=2);planned16=np.concatenate((q16,qd16),axis=2)
        replay1,tool1=replay_solution(b1,x0[None],u1,1)
        replay16,tool16=replay_solution(b16,np.tile(x0,(16,1)),u16,16)
        rows1,affine1,pin1,j1=evaluate_lanes(planned1,replay1,tool1,u1,construction,prerequisite,kinematics)
        rows16,affine16,pin16,j16=evaluate_lanes(planned16,replay16,tool16,u16,construction,prerequisite,kinematics)
        stat1=solver_certificate(out1);stat16=solver_certificate(out16)
        comparison=comparison_certificate(rows1,rows16,stat1,stat16)
        comparison["b1_vs_b16_lane0"]={"state_max_abs":float(np.max(np.abs(replay1[0]-replay16[0]))),
            "tool_max_abs":float(np.max(np.abs(tool1[0]-tool16[0]))),
            "control_max_abs":float(np.max(np.abs(u1[0]-u16[0]))),
            "full_cost_difference":float(rows1[0]["metrics"]["full_cost"]
                -rows16[0]["metrics"]["full_cost"]),"equality_required":False}
        arrays=result_arrays(seed1,seed16,out1,out16,replay1,tool1,replay16,tool16,
            affine1,affine16,pin1,pin16,j1,j16,x0,reference)
        atomic_npz(root/OUTPUT_NPZ.name,arrays)
        provenance_end=snapshot()
        if provenance_end!=provenance_start:raise RuntimeError("solve comparison provenance changed")
        result={"kind":"b1-b16-cached-seed-solve-comparison","cached_json_path":str(CACHED_JSON),
            "cached_json_sha256":CACHED_JSON_SHA256,"cached_json_size":CACHED_JSON_SIZE,
            "cached_npz_path":str(CACHED_NPZ),"cached_npz_sha256":CACHED_NPZ_SHA256,
            "cached_npz_size":CACHED_NPZ_SIZE,"p9_input_path":str(cached.P9_INPUT),
            "p9_input_sha256":cached.P9_INPUT_SHA256,"p9_input_size":cached.P9_INPUT_SIZE,
            "solver_options":SOLVER_OPTIONS,
            "lane_mapping":["short"]*8+["long"]*8,"counts":counts,
            "wall_elapsed_s":{"b1":wall1,"b16":wall16,"quality_run_observation_only":True},
            "solver_b1":stat1,"solver_b16":stat16,"b1_lanes":rows1,"b16_lanes":rows16,
            "comparison":comparison,"cuda_diagnostics":diagnostics,"extension":frozen_extension(),
            "provenance":{"start":provenance_start,"end":provenance_end},
            "npz_path":str(root/OUTPUT_NPZ.name),"npz_sha256":sha(root/OUTPUT_NPZ.name),
            "array_names":sorted(arrays),"array_hashes":{key:schema.array_hash(arrays[key]) for key in sorted(arrays)},
            "passes":comparison["passes"],"elapsed_s":float(monotonic()-start)}
        result=builtin(result);atomic_json(root/OUTPUT_JSON.name,result);return result
    except Exception as error:
        atomic_json(root/FAILURE_JSON.name,{"kind":"b1-b16-cached-seed-solve-failure",
            "error_type":type(error).__name__,"error_message":str(error),"counts":counts,
            "elapsed_s":float(monotonic()-start),"provenance_start":provenance_start,
            "evidence":False});raise


def recertify(root=ROOT,module_loader=importlib.import_module,
              prerequisite_loader=authenticate_cpu_prerequisite,pin_loader=_production_pin_context):
    try:
        root=Path(root);document=json.loads((root/OUTPUT_JSON.name).read_text())
        _cached,cached_arrays,construction=load_inputs()
        with np.load(root/OUTPUT_NPZ.name,allow_pickle=False) as archive:arrays={key:archive[key] for key in archive.files}
        _auth,prerequisite=prerequisite_loader();_model,kinematics,*_=pin_loader()
        module=module_loader(SOLVER_EXTENSION["module"])
        if not certify_module(module):return {"passes":False}
        b1=module.BSQP_1_float(*solver_args());b16=module.BSQP_16_float(*solver_args())
        out1={key:(arrays["output_xu_b1_float32"] if key=="XU" else arrays[f"raw_{key}_b1"]) for key in STAT_KEYS}
        out16={key:(arrays["output_xu_b16_float32"] if key=="XU" else arrays[f"raw_{key}_b16"]) for key in STAT_KEYS}
        out1=normalize_solve(out1,1);out16=normalize_solve(out16,16)
        q1,qd1,u1=unpack_xu(out1["XU"],1);q16,qd16,u16=unpack_xu(out16["XU"],16)
        replay1,tool1=replay_solution(b1,construction["x0_float32"][None],u1,1)
        replay16,tool16=replay_solution(b16,np.tile(construction["x0_float32"],(16,1)),u16,16)
        rows1,affine1,pin1,j1=evaluate_lanes(np.concatenate((q1,qd1),2),replay1,tool1,u1,
            construction,prerequisite,kinematics)
        rows16,affine16,pin16,j16=evaluate_lanes(np.concatenate((q16,qd16),2),replay16,tool16,u16,
            construction,prerequisite,kinematics)
        stat1=solver_certificate(out1);stat16=solver_certificate(out16)
        comparison=comparison_certificate(rows1,rows16,stat1,stat16)
        comparison["b1_vs_b16_lane0"]={"state_max_abs":float(np.max(np.abs(replay1[0]-replay16[0]))),
            "tool_max_abs":float(np.max(np.abs(tool1[0]-tool16[0]))),
            "control_max_abs":float(np.max(np.abs(u1[0]-u16[0]))),
            "full_cost_difference":float(rows1[0]["metrics"]["full_cost"]
                -rows16[0]["metrics"]["full_cost"]),"equality_required":False}
        rebuilt=result_arrays(arrays["input_seed_b1_float32"],arrays["input_seed_b16_float32"],
            out1,out16,replay1,tool1,replay16,tool16,affine1,affine16,pin1,pin16,j1,j16,
            construction["x0_float32"],construction["reference_float32"])
        current=snapshot();keys={"kind","cached_json_path","cached_json_sha256","cached_json_size",
            "cached_npz_path","cached_npz_sha256","cached_npz_size","p9_input_path",
            "p9_input_sha256","p9_input_size","solver_options","lane_mapping",
            "counts","wall_elapsed_s","solver_b1","solver_b16","b1_lanes","b16_lanes",
            "comparison","cuda_diagnostics","extension","provenance","npz_path","npz_sha256",
            "array_names","array_hashes","passes","elapsed_s"}
        exact=bool(set(arrays)==set(rebuilt) and all(np.array_equal(arrays[k],rebuilt[k]) for k in rebuilt)
            and np.array_equal(arrays["input_seed_b1_float32"],cached_arrays["b1_seed_float32"][None])
            and np.array_equal(arrays["input_seed_b16_float32"],cached_arrays["b16_seed_float32"])
            and document["npz_path"]==str(root/OUTPUT_NPZ.name)
            and document["npz_sha256"]==sha(root/OUTPUT_NPZ.name)
            and document["array_names"]==sorted(arrays)
            and document["array_hashes"]=={key:schema.array_hash(arrays[key]) for key in sorted(arrays)}
            and set(document)==keys and document["kind"]=="b1-b16-cached-seed-solve-comparison"
            and document["cached_json_path"]==str(CACHED_JSON)
            and document["cached_json_sha256"]==CACHED_JSON_SHA256
            and document["cached_json_size"]==CACHED_JSON_SIZE
            and document["cached_npz_path"]==str(CACHED_NPZ)
            and document["cached_npz_sha256"]==CACHED_NPZ_SHA256
            and document["cached_npz_size"]==CACHED_NPZ_SIZE
            and document["p9_input_path"]==str(cached.P9_INPUT)
            and document["p9_input_sha256"]==cached.P9_INPUT_SHA256
            and document["p9_input_size"]==cached.P9_INPUT_SIZE
            and document["solver_options"]==SOLVER_OPTIONS
            and document["lane_mapping"]==["short"]*8+["long"]*8
            and document["counts"]=={"b1_constructors":1,"b16_constructors":1,
                "b1_solve_calls":1,"b16_solve_calls":1}
            and set(document["wall_elapsed_s"])=={"b1","b16","quality_run_observation_only"}
            and document["wall_elapsed_s"]["quality_run_observation_only"] is True
            and all(np.isfinite(document["wall_elapsed_s"][key])
                    and document["wall_elapsed_s"][key]>=0 for key in ("b1","b16"))
            and document["solver_b1"]==builtin(stat1) and document["solver_b16"]==builtin(stat16)
            and document["b1_lanes"]==builtin(rows1) and document["b16_lanes"]==builtin(rows16)
            and document["comparison"]==builtin(comparison)
            and document["extension"]==frozen_extension()
            and worker.certify_cuda_diagnostics(document["cuda_diagnostics"])
            and certify_source_history(document["provenance"],current)
            and document["passes"] is comparison["passes"]
            and np.isfinite(document["elapsed_s"]) and document["elapsed_s"]>=0)
        return {"documents":exact,"science":comparison["passes"],"passes":bool(exact and comparison["passes"])}
    except Exception:return {"passes":False}


def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--output-root",type=Path,default=ROOT)
    args=parser.parse_args(argv)
    if tuple(sys.orig_argv)!=AUTHORIZED_ORIG_ARGV or args.output_root.resolve()!=ROOT:
        raise RuntimeError("solve comparison CLI boundary mismatch")
    if execute(args.output_root)["passes"] is not True:raise SystemExit(1)


if __name__=="__main__":main()
