"""Transactional CPU-only 192-lane screen for direct portfolio P3."""

from __future__ import annotations

import argparse,copy,hashlib,importlib.metadata,json,os,shlex,subprocess,sys,time
from pathlib import Path

import numpy as np

from gato_tiago.circular_portfolio import TASK_IDENTITIES,validate_reference_row
from gato_tiago.circular_portfolio_constructor import reconstruct_portfolio_cost,_open_turn
from gato_tiago.circular_portfolio_p3 import (
    BENCHMARK_CLASS,CAMPAIGN_WALL_LIMIT_S,KD,KP,LONG_AMPLITUDES,P2_REJECTED_REPORT,PROTOCOL,
    SHORT_AMPLITUDES,TOLL_LENGTH_M,expected_ledger,perturbation_direction,
    producer_array_hash,
)


RUNNER_EXECUTION_AUTHORIZATION=None
OUTPUT=Path("/tmp/tiago-tool-center-circular-portfolio-p3-cpu-screen-authorized-once/p3.json")
AUTHORIZED_CWD="/workspace/GATO"
AUTHORIZED_ORIG_ARGV=("python","-B","-m","gato_tiago.circular_portfolio_p3_runner",
    "--execute","--output",str(OUTPUT))
THREAD_ENV={"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
    "MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}
LEDGER=expected_ledger(TASK_IDENTITIES)
SOURCE_PATHS=("CMakeLists.txt","python/bindings.cu",
    "gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "gato/dynamics/tiago_right/tiago_right_grid.cuh",
    "gato/dynamics/tiago_right/tiago_right_arm.urdf",
    "gato/dynamics/integrator.cuh","gato/bsqp/bsqp.cuh",
    "gato/bsqp/kernels/tool_position.cuh","gato/utils/cuda.cuh",
    "python/bsqp/interface.py","tools/build.sh",
    "tiago_src/gato_tiago/config.py",
    "tiago_src/gato_tiago/circular_portfolio.py",
    "tiago_src/gato_tiago/circular_portfolio_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_prerequisite_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p2.py",
    "tiago_src/gato_tiago/circular_portfolio_p2_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_p2_preflight_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p3.py",
    "tiago_src/gato_tiago/circular_portfolio_p3_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_p3_runner.py",
    "tiago_src/gato_tiago/multimodal_pillar.py",
    "tiago_src/gato_tiago/multimodal_toll.py",
    "tiago_src/gato_tiago/multimodal_toll_v4.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_oracle_schema.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_runner.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight_v4.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight_v4_runner.py",
    "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight_v4_worker.py",
    "tiago_src/gato_tiago/multimodal_toll_oracle_v2.py",
    "tiago_src/gato_tiago/multimodal_toll_oracle_v2_prerequisite_runner.py",
    "tests/python/test_tiago_circular_portfolio_p3_static.py")
ROW_ARRAY_SPECS={"direction_float64":((7,),np.dtype(np.float64)),
    "planned_qdd_float64":((95,7),np.dtype(np.float64)),
    "planned_q_float64":((96,7),np.dtype(np.float64)),
    "planned_qd_float64":((96,7),np.dtype(np.float64)),
    "applied_controls_float32":((95,7),np.dtype(np.float32)),
    "pin_dense_state_float64":((6081,14),np.dtype(np.float64)),
    "pin_dense_tool_float64":((6081,3),np.dtype(np.float64)),
    "toll_residual_float64":((96,),np.dtype(np.float64))}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_environment():
    actual={key:os.environ.get(key) for key in THREAD_ENV}
    if actual!=THREAD_ENV: raise RuntimeError("P3 requires deterministic single-thread environment")
    return actual


def snapshot():
    root=Path(__file__).resolve().parents[2]
    run=lambda *args:subprocess.run(["git",*args],cwd=root,check=True,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
    return {"head":run("rev-parse","HEAD"),
        "clean":run("status","--porcelain","--untracked-files=no")=="",
        "sources":{name:sha(root/name) for name in SOURCE_PATHS}}


def start_provenance():
    current=snapshot()
    return {"cwd":str(Path.cwd().resolve()),"orig_argv":list(sys.orig_argv),
        "command":shlex.join(sys.orig_argv),"head_start":current["head"],
        "head_end":None,"clean_start":current["clean"],"clean_end":None,
        "sources_start":current["sources"],"sources_end":None,
        "thread_environment":require_environment(),"runtime_versions":{
            "python":sys.version,"numpy":np.__version__,
            "pinocchio":importlib.metadata.version("pin")},
        "prerequisite_pins":prerequisite_pins(),"benchmark_class":BENCHMARK_CLASS,
        "task_geometry_derived_from_predeclared_canonical_lane0_paths_before_sqp":True,
        "solver_outcomes_accessed_for_task_geometry":False}


def finish_provenance(value):
    current=snapshot();value["head_end"]=current["head"];value["clean_end"]=current["clean"]
    value["sources_end"]=current["sources"]


def prerequisite_pins():
    from gato_tiago.circular_portfolio_runner import PREREQUISITE_ARTIFACT_PINS
    return PREREQUISITE_ARTIFACT_PINS


def certify_provenance(value,final):
    current=snapshot()
    keys={"cwd","orig_argv","command","head_start","head_end","clean_start","clean_end",
        "sources_start","sources_end","thread_environment","runtime_versions",
        "prerequisite_pins","benchmark_class",
        "task_geometry_derived_from_predeclared_canonical_lane0_paths_before_sqp",
        "solver_outcomes_accessed_for_task_geometry"}
    return bool(set(value)==keys and value["cwd"]==AUTHORIZED_CWD
        and tuple(value["orig_argv"])==AUTHORIZED_ORIG_ARGV
        and value["command"]==shlex.join(AUTHORIZED_ORIG_ARGV)
        and value["head_start"]==current["head"] and value["clean_start"] is True
        and current["clean"] is True and value["sources_start"]==current["sources"]
        and value["thread_environment"]==THREAD_ENV and value["prerequisite_pins"]==prerequisite_pins()
        and value["runtime_versions"]=={"python":sys.version,"numpy":np.__version__,
            "pinocchio":importlib.metadata.version("pin")}
        and value["benchmark_class"]==BENCHMARK_CLASS
        and value["task_geometry_derived_from_predeclared_canonical_lane0_paths_before_sqp"] is True
        and value["solver_outcomes_accessed_for_task_geometry"] is False
        and ((not final and value["head_end"] is None and value["clean_end"] is None
              and value["sources_end"] is None) or (final and value["head_end"]==current["head"]
              and value["clean_end"] is True and value["sources_end"]==current["sources"])))


def row_arrays(row,direction):
    return {"direction_float64":np.asarray(direction,np.float64),
        "planned_qdd_float64":row["proxy"]["planned_qdd_float64"],
        "planned_q_float64":row["proxy"]["planned_q_float64"],
        "planned_qd_float64":row["proxy"]["planned_qd_float64"],
        "applied_controls_float32":row["applied_controls_float32"],
        "pin_dense_state_float64":row["pin_dense_state_float64"],
        "pin_dense_tool_float64":row["pin_dense_tool_float64"],
        "toll_residual_float64":row["certificate"]["toll_residual_float64"]}


def json_safe(value):
    if isinstance(value,dict):return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(item) for item in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def exact_row_schema(arrays):
    return bool(set(arrays)==set(ROW_ARRAY_SPECS) and all(
        np.asarray(arrays[name]).shape==shape and np.asarray(arrays[name]).dtype==dtype
        and np.isfinite(arrays[name]).all() for name,(shape,dtype) in ROW_ARRAY_SPECS.items()))


def row_summary(identity,row,arrays,path):
    certificate=json_safe({key:value for key,value in row["certificate"].items()
                 if key!="toll_residual_float64"}
    )
    return {"protocol":PROTOCOL,"identity":list(identity),"passes":certificate["passes"],
        "certificate":certificate,"measured_call_counts":row["measured_call_counts"],
        "npz_path":str(path),"array_names":sorted(arrays),
        "array_hashes":{name:producer_array_hash(arrays[name]) for name in sorted(arrays)}}


def certify_row_summary(summary,arrays,identity,path):
    keys={"protocol","identity","passes","certificate","measured_call_counts",
        "npz_path","array_names","array_hashes"}
    return bool(set(summary)==keys and summary["protocol"]==PROTOCOL
        and summary["identity"]==list(identity)
        and summary["passes"] is True and summary["certificate"]["passes"] is True
        and summary["measured_call_counts"]=={"rnea_calls":95,"aba_calls":6080,"fk_calls":6081}
        and summary["npz_path"]==str(path) and exact_row_schema(arrays)
        and summary["array_names"]==sorted(ROW_ARRAY_SPECS)
        and summary["array_hashes"]=={name:producer_array_hash(arrays[name])
                                      for name in sorted(arrays)})


def pair_certificate(short,long):
    short_c=short["certificate"];long_c=long["certificate"]
    base_advantage=long_c["base_cost"]-short_c["base_cost"]
    full_disadvantage=short_c["full_cost"]-long_c["full_cost"]
    gates={"both":short["passes"] is True and long["passes"] is True,
        "shorter":short_c["tool_length_m"]+0.005<=long_c["tool_length_m"],
        "base_advantage":base_advantage>=max(.01,.02*abs(long_c["base_cost"])),
        "full_reversal":full_disadvantage>=max(.01,.05*abs(long_c["full_cost"])),
        "saturation":int(np.count_nonzero(short["toll_residual_float64"][1:-1]>=.95))>=8,
        "long_toll":long_c["toll_contribution"]<=.01*short_c["toll_contribution"],
        "turns":abs(short_c["turn"])>=2.4 and abs(long_c["turn"])>=3.0
            and int(np.sign(short_c["turn"]))==-int(np.sign(long_c["turn"]))}
    return {"gates":gates,"base_advantage":base_advantage,
        "full_disadvantage":full_disadvantage,"passes":bool(all(gates.values()))}


def task_portfolio_certificate(rows,arrays):
    if len(rows)!=16 or len(arrays)!=16:return {"passes":False}
    controls=[np.asarray(item["applied_controls_float32"]) for item in arrays]
    tools=[np.asarray(item["pin_dense_tool_float64"]) for item in arrays]
    gates={"identities":len({tuple(row["identity"]) for row in rows})==16,
        "unique_control_bytes":len({value.tobytes() for value in controls})==16,
        "unique_realized_path_bytes":len({value.tobytes() for value in tools})==16}
    within=[]
    for family in (range(8),range(8,16)):
        for first in family:
            for second in family:
                if first>=second:continue
                control_rms=float(np.sqrt(np.mean((controls[first].astype(np.float64)
                                                    -controls[second])**2)))
                tool_rms=float(np.sqrt(np.mean(np.sum((tools[first]-tools[second])**2,axis=1))))
                within.append((control_rms,tool_rms))
    gates["within_family_control_distinct"] = all(c>=1e-4 for c,_p in within)
    gates["within_family_path_distinct"] = all(p>=.001 for _c,p in within)
    cross=[]
    for first in range(8):
        for second in range(8,16):
            maximum=float(np.max(np.linalg.norm(tools[first][:,:2]-tools[second][:,:2],axis=1)))
            turns=(rows[first]["certificate"]["turn"],rows[second]["certificate"]["turn"])
            cross.append((maximum,turns))
    gates["all_cross_family_separated"] = all(value>=.075 for value,_turns in cross)
    gates["all_cross_family_opposite_topology"] = all(
        int(np.sign(turns[0]))==-int(np.sign(turns[1])) for _value,turns in cross)
    return {"gates":gates,"minimum_within_control_rms":min(c for c,_p in within),
        "minimum_within_tool_rms_m":min(p for _c,p in within),
        "minimum_cross_family_max_separation_m":min(value for value,_turns in cross),
        "passes":bool(all(gates.values()))}


def geometry_from_lane_zero(short_row,long_row,goal,default_side):
    short_tool=short_row["pin_dense_tool_float64"];long_tool=long_row["pin_dense_tool_float64"]
    separation=np.linalg.norm(short_tool[:,:2]-long_tool[:,:2],axis=1)
    index=int(np.argmax(separation));pillar=.5*(short_tool[index,:2]+long_tool[index,:2])
    short_unit=short_tool[index,:2]-pillar;short_unit/=np.linalg.norm(short_unit)
    reference=validate_reference_row(np.asarray([*goal,*pillar,.03,*short_unit,
                                                  TOLL_LENGTH_M,.005],np.float32))
    gates={"separation":separation[index]>=.075,"default_side":int(default_side) in (-1,1),
        "reference":reference.shape==(10,) and reference.dtype==np.float32}
    return {"pillar_float64":pillar,"short_unit_float64":short_unit,
        "reference_float32":reference,"separation_m":float(separation[index]),
        "dense_index":index,"gates":gates,"passes":bool(all(gates.values()))}


def owned_campaign_semantic_recert(row_artifacts,task_geometries,prerequisite,
    kinematics,aba,*,deadline,monotonic,execution_counts):
    """Rebuild every retained P3 claim once before success publication."""
    from gato_tiago.circular_portfolio_p3_constructor import certify_rollout
    from gato_tiago.circular_portfolio_p3 import planned_proxy
    if len(row_artifacts)!=192 or len(task_geometries)!=12:
        return {"passes":False}
    lower=prerequisite["joint_lower_float64"];upper=prerequisite["joint_upper_float64"]
    velocity=prerequisite["velocity_limit_float64"];effort=prerequisite["effort_limit_float64"]
    directions=[]
    for task_index in range(12):
        directions.append(measured_direction(
            prerequisite["public_x0_float32"][task_index,:7].astype(np.float64),
            prerequisite["quarantined_q8_float64"][task_index],
            prerequisite["q0_tool_float64"][task_index],
            prerequisite["q8_tool_float64"][task_index],kinematics,
            int(prerequisite["public_default_side_int8"][task_index]),
            execution_counts,"owned"))
    fresh_rows=[];arrays_by_row=[]
    for row_index,identity in enumerate(LEDGER):
        task_index=row_index//16;split,seed,route,index=identity;artifact=row_artifacts[row_index]
        expected_json=Path(OUTPUT).with_name(f"{OUTPUT.stem}.row.{row_index:03d}.json")
        expected_npz=Path(OUTPUT).with_name(f"{OUTPUT.stem}.row.{row_index:03d}.npz")
        if artifact!={"identity":list(identity),"json_path":str(expected_json),
            "json_sha256":sha(expected_json),"npz_path":str(expected_npz),
            "npz_sha256":sha(expected_npz)}:return {"passes":False}
        summary=json.loads(expected_json.read_text())
        with np.load(expected_npz,allow_pickle=False) as archive:
            arrays={name:archive[name] for name in archive.files}
        if not certify_row_summary(summary,arrays,identity,expected_npz):return {"passes":False}
        q0=prerequisite["public_x0_float32"][task_index,:7].astype(np.float64)
        q_goal=prerequisite["quarantined_q8_float64"][task_index]
        start=prerequisite["q0_tool_float64"][task_index]
        goal=prerequisite["q8_tool_float64"][task_index]
        default=int(prerequisite["public_default_side_int8"][task_index])
        expected_direction=directions[task_index]
        if not np.array_equal(arrays["direction_float64"],expected_direction):return {"passes":False}
        geometry=task_geometries[task_index]
        pillar=np.asarray(geometry["pillar_float64"],np.float64)
        reference=np.tile(np.asarray(geometry["reference_float32"],np.float32),(96,1)).ravel()
        proxy=planned_proxy(q0,q_goal,expected_direction,route,index)
        retained={"identity":list(identity),"proxy":proxy,
            "applied_controls_float32":arrays["applied_controls_float32"],
            "pin_dense_state_float64":arrays["pin_dense_state_float64"],
            "pin_dense_tool_float64":arrays["pin_dense_tool_float64"],
            "measured_call_counts":summary["measured_call_counts"]}
        fresh=certify_rollout(retained,q0,q_goal,expected_direction,route,index,lower,upper,
            velocity,effort,goal,pillar,reference,kinematics,aba,deadline,
            reconstruct_portfolio_cost,_open_turn,default,monotonic,
            execution_counts,"owned")
        expected=json_safe({key:value for key,value in fresh.items()
                            if key!="toll_residual_float64"})
        if expected!=summary["certificate"] or not np.array_equal(
                fresh["toll_residual_float64"],arrays["toll_residual_float64"]):
            return {"passes":False}
        fresh_rows.append({**summary,
                           "toll_residual_float64":arrays["toll_residual_float64"]})
        arrays_by_row.append(arrays)
    portfolio=[];pairs=[]
    for task_index in range(12):
        base=task_index*16
        fresh_geometry=geometry_from_lane_zero(
            {"pin_dense_tool_float64":arrays_by_row[base]["pin_dense_tool_float64"]},
            {"pin_dense_tool_float64":arrays_by_row[base+8]["pin_dense_tool_float64"]},
            prerequisite["q8_tool_float64"][task_index],
            int(prerequisite["public_default_side_int8"][task_index]))
        if json_safe(fresh_geometry)!=task_geometries[task_index]:return {"passes":False}
        portfolio.append(task_portfolio_certificate(fresh_rows[base:base+16],
                                                     arrays_by_row[base:base+16]))
        for index in range(8):pairs.append(pair_certificate(
            fresh_rows[base+index],fresh_rows[base+8+index]))
    gates={"artifact_rows":len(fresh_rows)==192,
        "pairs":all(item["passes"] for item in pairs),
        "portfolios":all(item["passes"] for item in portfolio)}
    return {"pair_certificates":json_safe(pairs),
        "task_portfolio_certificates":json_safe(portfolio),"gates":gates,
        "passes":bool(all(gates.values()))}


def transaction_schema():
    return {"ledger_count":192,"profile_attempts":192,"computed_torque_rollouts":192,
        "immediate_independent_pin_replays":192,
        "owned_artifact_independent_pin_replays":192,"total_independent_pin_replays":384,
        "rnea_calls":192*95,
        "primary_rollout_aba_calls":192*6080,
        "immediate_recert_rollout_aba_calls":192*6080,
        "owned_artifact_recert_rollout_aba_calls":192*6080,
        "primary_rollout_fk_calls":192*6081,
        "immediate_recert_rollout_fk_calls":192*6081,
        "owned_artifact_recert_rollout_fk_calls":192*6081,
        "direction_fk_jacobian_calls":24,
        "immediate_semantic_certificate_fk_calls":192*97,
        "owned_artifact_semantic_certificate_fk_calls":192*97,
        "optimizer_calls":0,"scipy_calls":0,"rng_calls":0,"cuda_calls":0,
        "worker_calls":0,"sqp_calls":0,"retries":0,"p1_artifact_loads":0,
        "p2_artifact_loads":0,"prerequisite_artifact_loads":1,
        "wall_limit_s":CAMPAIGN_WALL_LIMIT_S,"resume_allowed":False}


def new_execution_counts():
    return {"primary_rollouts":0,"immediate_replays":0,"owned_replays":0,
        "primary_rnea_calls":0,"primary_aba_calls":0,"primary_fk_calls":0,
        "immediate_aba_calls":0,"immediate_fk_calls":0,
        "immediate_semantic_fk_calls":0,"owned_aba_calls":0,"owned_fk_calls":0,
        "owned_semantic_fk_calls":0,"primary_direction_calls":0,
        "owned_direction_calls":0,
        "optimizer_calls":0,"rng_calls":0,"cuda_calls":0,"worker_calls":0,
        "sqp_calls":0}


def final_execution_counts():
    value=new_execution_counts();value.update({"primary_rollouts":192,
        "immediate_replays":192,"owned_replays":192,"primary_rnea_calls":192*95,
        "primary_aba_calls":192*6080,"primary_fk_calls":192*6081,
        "immediate_aba_calls":192*6080,"immediate_fk_calls":192*6081,
        "immediate_semantic_fk_calls":192*97,"owned_aba_calls":192*6080,
        "owned_fk_calls":192*6081,"owned_semantic_fk_calls":192*97,
        "primary_direction_calls":12,"owned_direction_calls":12})
    return value


def _primary_rollouts_for_completed(completed):
    tasks,remainder=divmod(completed,16)
    return tasks*16+(0 if remainder==0 else remainder+1 if remainder<=8 else remainder)


def _primary_failure_rollout_allowance(completed):
    """Maximum completed primaries reachable beyond an accepted-row prefix."""
    remainder=completed%16
    # At a task boundary both lane-zero routes are constructed before geometry.
    # Thereafter only the next lane may complete, except after short7 because
    # long0 is precisely the already-counted second lane-zero construction.
    return 2 if remainder==0 else 0 if remainder==8 else 1


def expected_checkpoint_counts(completed):
    value=new_execution_counts();primary=_primary_rollouts_for_completed(completed)
    value.update({"primary_rollouts":primary,"immediate_replays":completed,
        "primary_rnea_calls":primary*95,"primary_aba_calls":primary*6080,
        "primary_fk_calls":primary*6081,"immediate_aba_calls":completed*6080,
        "immediate_fk_calls":completed*6081,"immediate_semantic_fk_calls":completed*97,
        "primary_direction_calls":int(np.ceil(completed/16)) if completed else 0})
    return value


def _bounded_phase(value,prefix,unit,maximum):
    return prefix*unit<=value<=min(maximum,(prefix+1)*unit)


def certify_execution_counts(value,final=False,completed=None,checkpoint=False):
    if set(value)!=set(new_execution_counts()) or any(
            not isinstance(item,int) or item<0 for item in value.values()):return False
    if any(value[key]!=0 for key in ("optimizer_calls","rng_calls","cuda_calls",
                                      "worker_calls","sqp_calls")):return False
    if final:return value==final_execution_counts()
    if completed is None or not isinstance(completed,int) or not 0<=completed<=192:return False
    if checkpoint:return value==expected_checkpoint_counts(completed)
    primary_min=_primary_rollouts_for_completed(completed)
    primary=value["primary_rollouts"];immediate=value["immediate_replays"]
    owned=value["owned_replays"]
    # A failure may occur after the next primary completes.  The exception is
    # task-local lane-zero geometry construction: at remainder zero both short0
    # and long0 can finish; at remainder eight long0 is already counted.
    primary_max=min(192,primary_min+_primary_failure_rollout_allowance(completed))
    if not primary_min<=primary<=primary_max:return False
    if not (completed<=immediate<=min(192,completed+1)):return False
    if completed<192 and (owned or value["owned_direction_calls"]):return False
    if not 0<=owned<=192:return False
    primary_direction_min=int(np.ceil(completed/16)) if completed else 0
    primary_direction_max=min(12,int(np.ceil((completed+1)/16)))
    if not primary_direction_min<=value["primary_direction_calls"]<=primary_direction_max:return False
    if not (_bounded_phase(value["primary_rnea_calls"],primary,95,192*95)
        and _bounded_phase(value["primary_aba_calls"],primary,6080,192*6080)
        and _bounded_phase(value["primary_fk_calls"],primary,6081,192*6081)
        and _bounded_phase(value["immediate_aba_calls"],immediate,6080,192*6080)
        and _bounded_phase(value["immediate_fk_calls"],immediate,6081,192*6081)):
        return False
    if not completed*97<=value["immediate_semantic_fk_calls"]<=immediate*97:return False
    if completed==192:
        # The owned recertifier eagerly authenticates all twelve task
        # directions before opening the first row artifact.  A failure during
        # that precompute may retain 0..12; any row work requires all twelve.
        owned_work=owned or value["owned_aba_calls"] or value["owned_fk_calls"] \
            or value["owned_semantic_fk_calls"]
        if not 0<=value["owned_direction_calls"]<=12:return False
        if owned_work and value["owned_direction_calls"]!=12:return False
        if not (_bounded_phase(value["owned_aba_calls"],owned,6080,192*6080)
            and _bounded_phase(value["owned_fk_calls"],owned,6081,192*6081)):
            return False
        semantic_min=max(0,owned-1)*97
        if not semantic_min<=value["owned_semantic_fk_calls"]<=owned*97:return False
    return True


def measured_direction(q0,q_goal,start,goal,kinematics,default_side,counts,phase):
    value=perturbation_direction(q0,q_goal,start,goal,kinematics,default_side)
    counts[f"{phase}_direction_calls"]+=1
    return value


def refuse_existing(output):
    output=Path(output);candidates=[output,output.with_suffix(".npz"),
        output.with_name(output.stem+".manifest.json"),
        output.with_name(output.stem+".partial.latest.json"),
        output.with_name(output.stem+".rejected.json")]
    candidates.extend(output.parent.glob(output.stem+".row.*"))
    candidates.extend(output.parent.glob(output.stem+".partial.gen*"))
    if any(path.exists() for path in candidates):raise FileExistsError("P3 forbids overwrite/resume")


def _write_json(path,value):
    with Path(path).open("x") as stream:json.dump(value,stream,sort_keys=True,separators=(",",":"));stream.write("\n")


def _atomic_npz(path,arrays):
    candidate=Path(str(path)+".candidate")
    with candidate.open("xb") as stream:np.savez(stream,**arrays)
    os.replace(candidate,path)


def _atomic_json(path,value):
    candidate=Path(str(path)+".candidate");_write_json(candidate,value);os.replace(candidate,path)


def checkpoint_document(generation,stage,completed,row_artifacts,provenance,call_counts=None):
    return {"protocol":PROTOCOL,"generation":generation,"stage":stage,
        "incomplete":True,"completed":completed,"pending":192-completed,
        "row_artifacts":copy.deepcopy(row_artifacts),"p2_rejected_report":P2_REJECTED_REPORT,
        "call_counts":copy.deepcopy(call_counts or new_execution_counts()),
        "oracle_evidence":False,"benchmark_evidence":False,
        "provenance":copy.deepcopy(provenance)}


def certify_checkpoint(document,final_provenance=False):
    completed=document.get("completed");generation=document.get("generation")
    stage=document.get("stage")
    generation_ok=((stage=="gen0" and generation==0 and completed==0)
        or (stage=="prerequisite_authenticated" and generation==1 and completed==0)
        or (stage=="profile_completed" and generation==completed+1 and completed>0)
        or (stage=="honest_end" and generation==194 and completed==192))
    keys={"protocol","generation","stage","incomplete","completed","pending",
        "row_artifacts","p2_rejected_report","call_counts","oracle_evidence",
        "benchmark_evidence","provenance"}
    return bool(set(document)==keys and document.get("protocol")==PROTOCOL
        and document.get("incomplete") is True and isinstance(generation,int)
        and isinstance(completed,int) and 0<=completed<=192
        and generation_ok
        and document.get("pending")==192-completed
        and len(document.get("row_artifacts",[]))==completed
        and [tuple(row["identity"]) for row in document.get("row_artifacts",[])]
            ==list(LEDGER[:completed])
        and all(set(row)=={"identity","json_path","json_sha256","npz_path","npz_sha256"}
                for row in document.get("row_artifacts",[]))
        and all(len(row["json_sha256"])==64 and len(row["npz_sha256"])==64
                for row in document.get("row_artifacts",[]))
        and document.get("p2_rejected_report")==P2_REJECTED_REPORT
        and certify_execution_counts(document.get("call_counts",{}),stage=="honest_end",
            completed,stage!="honest_end")
        and document.get("oracle_evidence") is False and document.get("benchmark_evidence") is False
        and certify_provenance(document.get("provenance",{}),final_provenance))


def publish_checkpoint(output,document):
    if not certify_checkpoint(document,document.get("stage")=="honest_end"):
        raise RuntimeError("P3 checkpoint certificate failed")
    path=Path(output).with_name(f"{Path(output).stem}.partial.gen{document['generation']:03d}.json")
    _write_json(path,document);pointer=Path(output).with_name(Path(output).stem+".partial.latest.json")
    candidate=Path(str(pointer)+".candidate")
    _write_json(candidate,{"protocol":PROTOCOL,"incomplete":True,
        "generation":document["generation"],"json_path":str(path),"json_sha256":sha(path)})
    os.replace(candidate,pointer);return path


def certify_final_document(document):
    keys={"protocol","incomplete","overall_pass","rows","pair_certificates",
        "task_portfolio_certificates","owned_semantic_recertification",
        "task_geometries","elapsed_s","transaction","provenance","checkpoint_count",
        "execution_counts","p2_rejected_report","oracle_evidence","benchmark_evidence"}
    gates={"keys":set(document)==keys,"protocol":document.get("protocol")==PROTOCOL,
        "complete":document.get("incomplete") is False and document.get("overall_pass") is True,
        "rows":len(document.get("rows",[]))==192
            and [tuple(row.get("identity",())) for row in document.get("rows",[])]==list(LEDGER)
            and all(row.get("passes") is True and row.get("certificate",{}).get("passes") is True
                    for row in document.get("rows",[])),
        "pairs":len(document.get("pair_certificates",[]))==96
            and all(pair.get("passes") is True for pair in document.get("pair_certificates",[])),
        "portfolio":len(document.get("task_portfolio_certificates",[]))==12
            and all(row.get("passes") is True
                    for row in document.get("task_portfolio_certificates",[])),
        "owned_semantic_recert":set(document.get("owned_semantic_recertification",{}))
            =={"pair_certificates","task_portfolio_certificates","gates","passes"}
            and set(document["owned_semantic_recertification"].get("gates",{}))
            =={"artifact_rows","pairs","portfolios"}
            and document["owned_semantic_recertification"].get("passes") is True,
        "owned_aggregate_identity":document.get("pair_certificates")
            ==document.get("owned_semantic_recertification",{}).get("pair_certificates")
            and document.get("task_portfolio_certificates")
            ==document.get("owned_semantic_recertification",{}).get("task_portfolio_certificates"),
        "geometries":len(document.get("task_geometries",[]))==12
            and all(row.get("passes") is True for row in document.get("task_geometries",[])),
        "elapsed":isinstance(document.get("elapsed_s"),float)
            and 0<=document["elapsed_s"]<=CAMPAIGN_WALL_LIMIT_S,
        "transaction":document.get("transaction")==transaction_schema(),
        "execution_counts":certify_execution_counts(document.get("execution_counts",{}),True),
        "checkpoints":document.get("checkpoint_count")==195,
        "closure":document.get("p2_rejected_report")==P2_REJECTED_REPORT,
        "non_evidence":document.get("oracle_evidence") is False
            and document.get("benchmark_evidence") is False,
        "provenance":certify_provenance(document.get("provenance",{}),True)}
    return {"gates":gates,"passes":bool(all(gates.values()))}


def certify_rejection(document):
    keys={"protocol","stage","incomplete","completed","pending","error_type",
        "error_message","elapsed_s","checkpoint_count","call_counts","provenance",
        "p2_rejected_report","oracle_evidence","benchmark_evidence"}
    return bool(set(document)==keys and document.get("protocol")==PROTOCOL
        and document.get("stage") in {"runtime_watchdog_rejected","screen_failed"}
        and document.get("incomplete") is True
        and isinstance(document.get("completed"),int) and 0<=document["completed"]<=192
        and document.get("pending")==192-document["completed"]
        and isinstance(document.get("checkpoint_count"),int)
        and 1<=document["checkpoint_count"]<=document["completed"]+2
        and isinstance(document.get("elapsed_s"),(int,float))
        and np.isfinite(document["elapsed_s"]) and document["elapsed_s"]>=0
        and certify_execution_counts(document.get("call_counts",{}),False,
                                      document["completed"],False)
        and document.get("p2_rejected_report")==P2_REJECTED_REPORT
        and document.get("oracle_evidence") is False
        and document.get("benchmark_evidence") is False
        and certify_provenance(document.get("provenance",{}),True))


def recertify_retained_failure(output=OUTPUT):
    try:
        output=Path(output);rejection=output.with_name(output.stem+".rejected.json")
        pointer=output.with_name(output.stem+".partial.latest.json")
        document=json.loads(rejection.read_text());latest=json.loads(pointer.read_text())
        expected={"protocol":PROTOCOL,"incomplete":True,"stage":document["stage"],
            "json_path":str(rejection),"json_sha256":sha(rejection)}
        if not certify_rejection(document) or latest!=expected or output.exists():
            return {"passes":False}
        # gen0/auth plus one checkpoint for every successfully retained row.
        count=document["checkpoint_count"]
        for generation in range(count):
            checkpoint=json.loads(output.with_name(
                f"{output.stem}.partial.gen{generation:03d}.json").read_text())
            if not certify_checkpoint(checkpoint,False):return {"passes":False}
        return {"passes":True}
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):
        return {"passes":False}


def recertify_retained_p3(output=OUTPUT):  # pragma: no cover - independent audit
    try:
        output=Path(output).resolve();document=json.loads(output.read_text())
        if not certify_final_document(document)["passes"]:return {"passes":False}
        from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite,_production_pin_context
        from gato_tiago.circular_portfolio_p3_constructor import certify_rollout
        from gato_tiago.circular_portfolio_p3 import planned_proxy
        auth,prerequisite=authenticate_cpu_prerequisite()
        if auth["recertification"]["passes"] is not True:return {"passes":False}
        _model,kinematics,_rnea,_rd,aba,_velocity,_dense=_production_pin_context()
        lower=prerequisite["joint_lower_float64"];upper=prerequisite["joint_upper_float64"]
        velocity=prerequisite["velocity_limit_float64"];effort=prerequisite["effort_limit_float64"]
        rows=[];loaded_arrays=[]
        deadline=float("inf")
        for row_index,identity in enumerate(LEDGER):
            task_index=row_index//16;split,seed,route,index=identity
            summary=document["rows"][row_index];npz=Path(summary["npz_path"])
            with np.load(npz,allow_pickle=False) as archive:
                arrays={name:archive[name] for name in archive.files}
            if not certify_row_summary(summary,arrays,identity,npz):return {"passes":False}
            q0=prerequisite["public_x0_float32"][task_index,:7].astype(np.float64)
            q_goal=prerequisite["quarantined_q8_float64"][task_index]
            goal=prerequisite["q8_tool_float64"][task_index]
            default=int(prerequisite["public_default_side_int8"][task_index])
            start=prerequisite["q0_tool_float64"][task_index]
            expected_direction=perturbation_direction(q0,q_goal,start,goal,kinematics,default)
            if not np.array_equal(arrays["direction_float64"],expected_direction):
                return {"passes":False}
            geometry=document["task_geometries"][task_index]
            pillar=np.asarray(geometry["pillar_float64"],np.float64)
            reference=np.tile(np.asarray(geometry["reference_float32"],np.float32),(96,1)).ravel()
            proxy=planned_proxy(q0,q_goal,arrays["direction_float64"],route,index)
            retained={"identity":list(identity),"proxy":proxy,
                "applied_controls_float32":arrays["applied_controls_float32"],
                "pin_dense_state_float64":arrays["pin_dense_state_float64"],
                "pin_dense_tool_float64":arrays["pin_dense_tool_float64"],
                "measured_call_counts":summary["measured_call_counts"]}
            fresh=certify_rollout(retained,q0,q_goal,arrays["direction_float64"],route,index,
                lower,upper,velocity,effort,goal,pillar,reference,kinematics,aba,deadline,
                reconstruct_portfolio_cost,_open_turn,default,lambda:0.0)
            fresh_json=json_safe({key:value for key,value in fresh.items()
                                  if key!="toll_residual_float64"})
            if fresh_json!=summary["certificate"] or not np.array_equal(
                    fresh["toll_residual_float64"],arrays["toll_residual_float64"]):
                return {"passes":False}
            rows.append({**summary,"toll_residual_float64":arrays["toll_residual_float64"]})
            loaded_arrays.append(arrays)
        for task_index in range(12):
            short_arrays=loaded_arrays[task_index*16]
            long_arrays=loaded_arrays[task_index*16+8]
            fresh_geometry=geometry_from_lane_zero(
                {"pin_dense_tool_float64":short_arrays["pin_dense_tool_float64"]},
                {"pin_dense_tool_float64":long_arrays["pin_dense_tool_float64"]},
                prerequisite["q8_tool_float64"][task_index],
                int(prerequisite["public_default_side_int8"][task_index]))
            if json_safe(fresh_geometry)!=document["task_geometries"][task_index]:
                return {"passes":False}
        pairs=[]
        for task in range(12):
            for index in range(8):pairs.append(pair_certificate(
                rows[task*16+index],rows[task*16+8+index]))
        if json_safe(pairs)!=document["pair_certificates"]:return {"passes":False}
        portfolios=[]
        for task in range(12):
            base=task*16;portfolios.append(task_portfolio_certificate(
                rows[base:base+16],loaded_arrays[base:base+16]))
        if json_safe(portfolios)!=document["task_portfolio_certificates"]:
            return {"passes":False}
        owned={"pair_certificates":json_safe(pairs),
            "task_portfolio_certificates":json_safe(portfolios),
            "gates":{"artifact_rows":True,"pairs":all(item["passes"] for item in pairs),
                     "portfolios":all(item["passes"] for item in portfolios)},
            "passes":bool(all(item["passes"] for item in pairs)
                           and all(item["passes"] for item in portfolios))}
        if document["owned_semantic_recertification"]!=owned:return {"passes":False}
        # Authenticate all immutable checkpoints and document hashes.
        expected_row_artifacts=[];expected_side=[]
        for row_index,row in enumerate(document["rows"]):
            json_path=Path(row["npz_path"]).with_suffix(".json");npz_path=Path(row["npz_path"])
            expected_row_artifacts.append({"identity":list(LEDGER[row_index]),
                "json_path":str(json_path),"json_sha256":sha(json_path),
                "npz_path":str(npz_path),"npz_sha256":sha(npz_path)})
        for generation in range(195):
            path=output.with_name(f"{output.stem}.partial.gen{generation:03d}.json")
            checkpoint=json.loads(path.read_text())
            expected_count=0 if generation<2 else min(generation-1,192)
            if (not certify_checkpoint(checkpoint,generation==194)
                    or checkpoint["row_artifacts"]!=expected_row_artifacts[:expected_count]):
                return {"passes":False}
            expected_side.append({"path":str(path),"sha256":sha(path)})
        for row in expected_row_artifacts:
            expected_side.extend(({"path":row["json_path"],"sha256":row["json_sha256"]},
                                  {"path":row["npz_path"],"sha256":row["npz_sha256"]}))
        manifest_path=output.with_name(output.stem+".manifest.json")
        pointer_path=output.with_name(output.stem+".partial.latest.json")
        manifest=json.loads(manifest_path.read_text());pointer=json.loads(pointer_path.read_text())
        expected_manifest={"protocol":PROTOCOL,"json_path":str(output),
            "json_sha256":sha(output),"side_artifacts":expected_side}
        if manifest!=expected_manifest or pointer!={"protocol":PROTOCOL,
            "incomplete":False,"json_path":str(output),"json_sha256":sha(output),
            "manifest_path":str(manifest_path),"manifest_sha256":sha(manifest_path)}:
            return {"passes":False}
        return {"passes":True}
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError,RuntimeError):
        return {"passes":False}


def execute(output=OUTPUT,authorization=None,monotonic=time.monotonic):  # pragma: no cover
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P3 CPU screen is blocked")
    output=Path(output).resolve()
    if output!=OUTPUT:raise RuntimeError("P3 output path not authorized")
    require_environment();refuse_existing(output);output.parent.mkdir(parents=True)
    from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite,_production_pin_context
    from gato_tiago.circular_portfolio_p3_constructor import (
        CONSTRUCTOR_EXECUTION_AUTHORIZATION,certify_rollout,computed_torque_rollout,
        execute_direct_constructor)
    started=monotonic();deadline=started+CAMPAIGN_WALL_LIMIT_S;provenance=start_provenance()
    rows=[];artifacts=[];row_artifacts=[];checkpoints=[];task_geometries=[]
    execution_counts=new_execution_counts()
    try:
        checkpoints.append(publish_checkpoint(output,checkpoint_document(
            0,"gen0",0,row_artifacts,provenance,execution_counts)))
        auth,prerequisite=authenticate_cpu_prerequisite()
        if auth["recertification"]["passes"] is not True:raise RuntimeError("P3 prerequisite failed")
        checkpoints.append(publish_checkpoint(output,checkpoint_document(
            1,"prerequisite_authenticated",0,row_artifacts,provenance,execution_counts)))
        _model,kinematics,rnea,_rd,aba,_velocity,_dense=_production_pin_context()
        lower=prerequisite["joint_lower_float64"];upper=prerequisite["joint_upper_float64"]
        velocity=prerequisite["velocity_limit_float64"];effort=prerequisite["effort_limit_float64"]
        for task_index,(split,seed) in enumerate(TASK_IDENTITIES):
            if monotonic()>=deadline:raise TimeoutError("P3 campaign wall limit")
            q0=prerequisite["public_x0_float32"][task_index,:7].astype(np.float64)
            q_goal=prerequisite["quarantined_q8_float64"][task_index]
            start=prerequisite["q0_tool_float64"][task_index];goal=prerequisite["q8_tool_float64"][task_index]
            default=int(prerequisite["public_default_side_int8"][task_index])
            direction=measured_direction(q0,q_goal,start,goal,kinematics,default,
                                         execution_counts,"primary")
            # Geometry is defined from actual canonical lane-zero rollouts before
            # any science row is accepted; no selection/filtering occurs.
            proxies=[]
            from gato_tiago.circular_portfolio_p3 import planned_proxy
            for route in ("short","long"):
                proxy=planned_proxy(q0,q_goal,direction,route,0)
                raw=computed_torque_rollout(proxy,q0,rnea,aba,kinematics,
                    deadline=deadline,monotonic=monotonic,
                    execution_counts=execution_counts,phase="primary");raw["proxy"]=proxy
                proxies.append(raw)
            geometry=geometry_from_lane_zero(proxies[0],proxies[1],goal,default)
            if not geometry["passes"]:raise RuntimeError("P3 geometry failed")
            task_geometries.append(json_safe(geometry))
            reference=np.tile(geometry["reference_float32"],(96,1)).ravel()
            for route in ("short","long"):
                for index in range(8):
                    identity=(split,seed,route,index)
                    if index==0:
                        row=proxies[0 if route=="short" else 1]
                        row["identity"]=list(identity)
                        row["certificate"]=certify_rollout(row,q0,q_goal,direction,route,index,
                            lower,upper,velocity,effort,goal,geometry["pillar_float64"],
                            reference,kinematics,aba,deadline,reconstruct_portfolio_cost,
                            _open_turn,default,monotonic,execution_counts,"immediate")
                    else:
                        row=execute_direct_constructor(identity,q0,q_goal,direction,lower,upper,
                            velocity,effort,goal,geometry["pillar_float64"],reference,kinematics,
                            rnea,aba,reconstruct_portfolio_cost,_open_turn,default_side=default,
                            deadline=deadline,monotonic=monotonic,
                            authorization=CONSTRUCTOR_EXECUTION_AUTHORIZATION,
                            execution_counts=execution_counts)
                    arrays=row_arrays(row,direction);npz=output.with_name(
                        f"{output.stem}.row.{len(rows):03d}.npz")
                    summary=row_summary(identity,row,arrays,npz)
                    if not certify_row_summary(summary,arrays,identity,npz):raise RuntimeError("P3 row failed")
                    _atomic_npz(npz,arrays);js=npz.with_suffix(".json");_write_json(js,summary)
                    row_artifacts.append({"identity":list(identity),"json_path":str(js),
                        "json_sha256":sha(js),"npz_path":str(npz),"npz_sha256":sha(npz)})
                    summary["toll_residual_float64"]=arrays["toll_residual_float64"]
                    rows.append(summary);artifacts.extend((js,npz))
                    checkpoints.append(publish_checkpoint(output,checkpoint_document(
                        len(rows)+1,"profile_completed",len(rows),row_artifacts,provenance,
                        execution_counts)))
        semantic=owned_campaign_semantic_recert(row_artifacts,task_geometries,
            prerequisite,kinematics,aba,deadline=deadline,monotonic=monotonic,
            execution_counts=execution_counts)
        if not semantic["passes"]:raise RuntimeError("P3 owned semantic recertification failed")
        finish_provenance(provenance);elapsed=monotonic()-started
        if elapsed>CAMPAIGN_WALL_LIMIT_S:raise TimeoutError("P3 campaign wall limit")
        checkpoints.append(publish_checkpoint(output,checkpoint_document(
            194,"honest_end",192,row_artifacts,provenance,execution_counts)))
        clean_rows=[]
        for row in rows:
            row=dict(row);row.pop("toll_residual_float64");clean_rows.append(row)
        final={"protocol":PROTOCOL,"incomplete":False,"overall_pass":True,
            "rows":clean_rows,"pair_certificates":semantic["pair_certificates"],
            "task_portfolio_certificates":semantic["task_portfolio_certificates"],
            "owned_semantic_recertification":semantic,
            "task_geometries":task_geometries,"elapsed_s":elapsed,
            "transaction":transaction_schema(),"provenance":provenance,
            "execution_counts":copy.deepcopy(execution_counts),
            "checkpoint_count":len(checkpoints),
            "p2_rejected_report":P2_REJECTED_REPORT,"oracle_evidence":False,
            "benchmark_evidence":False}
        if not certify_final_document(final)["passes"]:raise RuntimeError("P3 final certificate failed")
        _atomic_json(output,final);manifest=output.with_name(output.stem+".manifest.json")
        _atomic_json(manifest,{"protocol":PROTOCOL,"json_path":str(output),"json_sha256":sha(output),
            "side_artifacts":[{"path":str(p),"sha256":sha(p)}
                for p in [*checkpoints,*artifacts]]})
        pointer=output.with_name(output.stem+".partial.latest.json")
        _atomic_json(pointer,{"protocol":PROTOCOL,"incomplete":False,"json_path":str(output),
            "json_sha256":sha(output),"manifest_path":str(manifest),"manifest_sha256":sha(manifest)})
        return final
    except Exception as error:
        finish_provenance(provenance);stage=("runtime_watchdog_rejected"
            if isinstance(error,TimeoutError) else "screen_failed")
        rejected={"protocol":PROTOCOL,"stage":stage,"incomplete":True,
            "completed":len(rows),"pending":192-len(rows),"error_type":type(error).__name__,
            "error_message":str(error),"elapsed_s":monotonic()-started,
            "checkpoint_count":len(checkpoints),
            "call_counts":copy.deepcopy(execution_counts),
            "provenance":provenance,"p2_rejected_report":P2_REJECTED_REPORT,
            "oracle_evidence":False,"benchmark_evidence":False}
        if not certify_rejection(rejected):raise RuntimeError("P3 rejection certificate failed") from error
        rejection_path=output.with_name(output.stem+".rejected.json")
        _atomic_json(rejection_path,rejected)
        pointer=output.with_name(output.stem+".partial.latest.json")
        _atomic_json(pointer,{"protocol":PROTOCOL,"incomplete":True,"stage":stage,
            "json_path":str(rejection_path),"json_sha256":sha(rejection_path)})
        raise


def main(argv=None):  # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    execute(args.output,RUNNER_EXECUTION_AUTHORIZATION)


if __name__=="__main__":main()
