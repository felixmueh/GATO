"""One-shot CPU-only operational preflight for reduced circular portfolio P2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Mapping

import numpy as np

from gato_tiago.circular_portfolio import (
    DENSE_SAMPLES, EXPECTED_LEDGER, FINAL_TOOL_SPEED_TOLERANCE_MPS,
    PHYSICAL_RADIUS_M, TERMINAL_PIN_TOLERANCE_M,
)
from gato_tiago.circular_portfolio_constructor import _dense_reference, _open_turn
from gato_tiago.circular_portfolio_constructor import (
    ShootingResult,certify_proxy_construction,certify_shooting_result,
    linear_proxy_initial_acceleration,
)
from gato_tiago.circular_portfolio_p2 import (
    P1_REJECTED_REPORT, PREFLIGHT_WALL_LIMIT_S, PRECHECK_IDENTITY, PROTOCOL,
    certify_affine_map, certify_expansion,least_squares_initial_coefficients,
)


RUNNER_EXECUTION_AUTHORIZATION = None
OUTPUT = Path(
    "/tmp/tiago-tool-center-circular-portfolio-p2-preflight-authorized-once/preflight.json"
)
AUTHORIZED_CWD = "/workspace/GATO"
AUTHORIZED_ORIG_ARGV = (
    "python", "-B", "-m", "gato_tiago.circular_portfolio_p2_preflight_runner",
    "--execute", "--output", str(OUTPUT),
)
STAGES = ("gen0", "prerequisite_authenticated", "profile_attempted", "honest_end")
PHASE_KEYS={"proxy_s","affine_s","least_squares_s","optimizer_s","postcheck_s"}
EVALUATION_KEYS={"objective","gradient","inequality","inequality_jacobian",
                 "kinematics","rnea","rnea_derivatives","aba"}
REPLAY_COUNT_KEYS={"primary_pin_replay_calls",
    "independent_pin_recert_replay_calls","total_pin_replay_calls"}
THREAD_ENV={"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
            "MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}
SOURCE_PATHS=("CMakeLists.txt","python/bindings.cu",
    "gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "gato/dynamics/tiago_right/tiago_right_grid.cuh",
    "gato/dynamics/tiago_right/tiago_right_arm.urdf",
    "gato/dynamics/integrator.cuh","gato/bsqp/bsqp.cuh",
    "gato/bsqp/kernels/tool_position.cuh","gato/utils/cuda.cuh",
    "python/bsqp/interface.py","tools/build.sh",
    "tiago_src/gato_tiago/config.py",
    "tiago_src/gato_tiago/circular_portfolio_p2.py",
    "tiago_src/gato_tiago/circular_portfolio_p2_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_p2_preflight_runner.py",
    "tiago_src/gato_tiago/circular_portfolio.py",
    "tiago_src/gato_tiago/circular_portfolio_constructor.py",
    "tiago_src/gato_tiago/circular_portfolio_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_prerequisite_runner.py",
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
    "tests/python/test_tiago_circular_portfolio_p2_static.py")
ARRAY_SPECS = {
    "coefficients_float64": ((70,), np.dtype(np.float64)),
    "basis_float64": ((95,12), np.dtype(np.float64)),
    "scalar_endpoint_map_float64":((2,95),np.dtype(np.float64)),
    "temporal_projection_float64":((95,95),np.dtype(np.float64)),
    "temporal_reduced_float64":((95,12),np.dtype(np.float64)),
    "temporal_independent_float64":((95,10),np.dtype(np.float64)),
    "temporal_singular_values_float64":((12,),np.dtype(np.float64)),
    "raw_map_float64": ((665,84), np.dtype(np.float64)),
    "endpoint_map_float64": ((14,665), np.dtype(np.float64)),
    "endpoint_right_inverse_float64": ((665,14), np.dtype(np.float64)),
    "projection_float64": ((665,665), np.dtype(np.float64)),
    "particular_acceleration_float64": ((95,7), np.dtype(np.float64)),
    "reduced_map_float64": ((665,84), np.dtype(np.float64)),
    "independent_map_float64": ((665,70), np.dtype(np.float64)),
    "raw_to_independent_float64": ((70,84), np.dtype(np.float64)),
    "reduced_singular_values_float64": ((84,), np.dtype(np.float64)),
    "acceleration_float64": ((95,7), np.dtype(np.float64)),
    "q_float64": ((96,7), np.dtype(np.float64)),
    "qd_float64": ((96,7), np.dtype(np.float64)),
    "controls_float64": ((95,7), np.dtype(np.float64)),
    "positions_float64": ((96,3), np.dtype(np.float64)),
    "pin_dense_state_float64": ((DENSE_SAMPLES,14), np.dtype(np.float64)),
    "pin_dense_tool_float64": ((DENSE_SAMPLES,3), np.dtype(np.float64)),
    "endpoint_residual_float64": ((14,), np.dtype(np.float64)),
    "inequalities_float64": ((4114,), np.dtype(np.float64)),
    "shooting_objective_float64": ((), np.dtype(np.float64)),
    "optimizer_success_bool": ((), np.dtype(np.bool_)),
    "optimizer_status_int64": ((), np.dtype(np.int64)),
    "optimizer_iterations_int64": ((), np.dtype(np.int64)),
    "proxy_q_before_float64":((94,8,7),np.dtype(np.float64)),
    "proxy_position_float64":((94,8,3),np.dtype(np.float64)),
    "proxy_residual_float64":((94,8,3),np.dtype(np.float64)),
    "proxy_jacobian_float64":((94,8,3,7),np.dtype(np.float64)),
    "proxy_dq_float64":((94,8,7),np.dtype(np.float64)),
    "proxy_q_float64":((96,7),np.dtype(np.float64)),
    "proxy_initial_acceleration_float64":((95,7),np.dtype(np.float64)),
    "initial_coefficients_float64":((70,),np.dtype(np.float64)),
    "least_squares_residual_float64":((),np.dtype(np.float64)),
    "least_squares_rank_int64":((),np.dtype(np.int64)),
    "least_squares_singular_float64":((70,),np.dtype(np.float64)),
    "least_squares_reported_residual_float64":((1,),np.dtype(np.float64)),
}


def array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(
        f"{value.dtype.str}|{value.shape}|".encode()+value.tobytes()
    ).hexdigest()


def _file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_thread_environment():
    actual={name:os.environ.get(name) for name in THREAD_ENV}
    if actual!=THREAD_ENV: raise RuntimeError("P2 preflight requires four single-thread variables")
    return actual


def runtime_versions():
    return {"python":sys.version,"numpy":np.__version__,
            "scipy":importlib.metadata.version("scipy"),
            "pinocchio":importlib.metadata.version("pin")}


def repository_snapshot():
    root=Path(__file__).resolve().parents[2]
    run=lambda *args:subprocess.run(["git",*args],cwd=root,check=True,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
    return {"head":run("rev-parse","HEAD"),
        "tracked_clean":run("status","--porcelain","--untracked-files=no")=="",
        "source_hashes":{name:_file_hash(root/name) for name in SOURCE_PATHS}}


def start_provenance():
    snapshot=repository_snapshot()
    return {"cwd":str(Path.cwd().resolve()),"orig_argv":list(sys.orig_argv),
        "exact_command":shlex.join(sys.orig_argv),"git_head_at_start":snapshot["head"],
        "git_head_at_end":None,"tracked_clean_at_start":snapshot["tracked_clean"],
        "tracked_clean_at_end":None,
        "source_hashes_at_start":snapshot["source_hashes"],
        "source_hashes_at_end":None,"thread_environment":require_thread_environment(),
        "runtime_versions":runtime_versions(),
        "prerequisite_pins":p1_prerequisite_pins()}


def p1_prerequisite_pins():
    from gato_tiago.circular_portfolio_runner import PREREQUISITE_ARTIFACT_PINS
    return PREREQUISITE_ARTIFACT_PINS


def finish_provenance(value):
    snapshot=repository_snapshot()
    value["git_head_at_end"]=snapshot["head"]
    value["tracked_clean_at_end"]=snapshot["tracked_clean"]
    value["source_hashes_at_end"]=snapshot["source_hashes"]


def certify_provenance(value,final):
    required={"cwd","orig_argv","exact_command","git_head_at_start","git_head_at_end",
        "tracked_clean_at_start","tracked_clean_at_end","source_hashes_at_start",
        "source_hashes_at_end","thread_environment","runtime_versions","prerequisite_pins"}
    current=repository_snapshot()
    return bool(set(value)==required and value["cwd"]==AUTHORIZED_CWD
        and tuple(value["orig_argv"])==AUTHORIZED_ORIG_ARGV
        and value["exact_command"]==shlex.join(AUTHORIZED_ORIG_ARGV)
        and value["git_head_at_start"]==current["head"]
        and value["tracked_clean_at_start"] is True and current["tracked_clean"] is True
        and set(value["source_hashes_at_start"])==set(SOURCE_PATHS)
        and value["source_hashes_at_start"]==current["source_hashes"]
        and value["thread_environment"]==THREAD_ENV
        and value["runtime_versions"]==runtime_versions()
        and value["prerequisite_pins"]==p1_prerequisite_pins()
        and ((not final and value["git_head_at_end"] is None and value["tracked_clean_at_end"] is None
              and value["source_hashes_at_end"] is None)
             or (final and value["git_head_at_end"]==value["git_head_at_start"]
                 and value["tracked_clean_at_end"] is True
                 and value["source_hashes_at_end"]==value["source_hashes_at_start"])))


def exact_array_schema(arrays: Mapping):
    return bool(
        set(arrays)==set(ARRAY_SPECS)
        and all(np.asarray(arrays[name]).shape==shape
                and np.asarray(arrays[name]).dtype==dtype
                and np.isfinite(np.asarray(arrays[name])).all()
                for name,(shape,dtype) in ARRAY_SPECS.items())
    )


def certify_pin_preflight(
    retained: Mapping, arrays: Mapping, q0, q_goal, tool_reference,
    reference, lower, upper, velocity, effort, pillar, tool_velocity,
    dense_replay,
):
    affine = retained["affine"]
    expansion = certify_expansion(
        arrays["coefficients_float64"],affine,q0,q_goal
    )
    replay_state,replay_tool=dense_replay(
        np.r_[q0,np.zeros(7)],arrays["controls_float64"]
    )
    state=arrays["pin_dense_state_float64"]; tool=arrays["pin_dense_tool_float64"]
    controls=arrays["controls_float64"]
    dense_reference=_dense_reference(tool_reference)
    path_error=np.linalg.norm(tool-dense_reference,axis=1)
    reference_turn=_open_turn(tool_reference,pillar); actual_turn=_open_turn(tool,pillar)
    speed=float(np.linalg.norm(tool_velocity("pin",state[-1,:7],state[-1,7:])))
    pin_gates={
        "finite":all(np.isfinite(value).all() for value in (state,tool,controls)),
        "shape":state.shape==(DENSE_SAMPLES,14) and tool.shape==(DENSE_SAMPLES,3)
        and controls.shape==(95,7),
        "initial":np.array_equal(state[0],np.r_[q0,np.zeros(7)]),
        "q_limits":bool(np.all(state[:,:7]>=lower) and np.all(state[:,:7]<=upper)),
        "v_limits":bool(np.all(np.abs(state[:,7:])<=velocity)),
        "u_limits":bool(np.all(np.abs(controls)<=effort)),
        "clearance":np.min(np.linalg.norm(tool[:,:2]-pillar[None],axis=1)-PHYSICAL_RADIUS_M)>=.005,
        "terminal":np.linalg.norm(tool[-1]-tool_reference[-1])<=TERMINAL_PIN_TOLERANCE_M,
        "speed_jqd":speed<=FINAL_TOOL_SPEED_TOLERANCE_MPS,
        "circular_rms":float(np.sqrt(np.mean(path_error**2)))<=.002,
        "circular_max":np.max(path_error,initial=0.0)<=.004,
        "route_turn":abs(actual_turn)>=2.4 and int(np.sign(actual_turn))==int(np.sign(reference_turn)),
        "canonical_reference":np.asarray(reference).shape==(960,)
        and np.asarray(reference).dtype==np.float32,
    }
    gates={
        "identity": retained["identity"]==list(PRECHECK_IDENTITY),
        "schema": exact_array_schema(arrays),
        "affine": certify_affine_map(affine,q0,q_goal)["passes"],
        "proxy":certify_proxy_construction({name:arrays[name] for name in (
            "proxy_q_before_float64","proxy_position_float64","proxy_residual_float64",
            "proxy_jacobian_float64","proxy_dq_float64","proxy_q_float64")},
            q0,q_goal,tool_reference,retained["kinematics"])["passes"],
        "expansion": expansion["passes"]
        and np.array_equal(expansion["acceleration_float64"],arrays["acceleration_float64"])
        and np.array_equal(expansion["q_float64"],arrays["q_float64"])
        and np.array_equal(expansion["qd_float64"],arrays["qd_float64"]),
        "shooting": retained["shooting_certificate"]["passes"] is True,
        "pin_replay_owned": np.array_equal(replay_state,arrays["pin_dense_state_float64"])
        and np.array_equal(replay_tool,arrays["pin_dense_tool_float64"]),
        "pin_feasibility": bool(all(pin_gates.values())),
        "elapsed": 0.0 <= retained["elapsed_s"] <= PREFLIGHT_WALL_LIMIT_S,
        "no_cuda": True,
    }
    fresh_proxy_initial=linear_proxy_initial_acceleration(q0,q_goal,arrays["proxy_q_float64"])
    fresh_ls=least_squares_initial_coefficients(
        fresh_proxy_initial["initial_acceleration_float64"],affine
    )
    gates["proxy_initial"]=np.array_equal(
        arrays["proxy_initial_acceleration_float64"],fresh_proxy_initial["initial_acceleration_float64"])
    gates["least_squares"]=all(np.array_equal(arrays[name],fresh_ls[name]) for name in (
        "initial_coefficients_float64","least_squares_residual_float64",
        "least_squares_rank_int64","least_squares_singular_float64",
        "least_squares_reported_residual_float64"))
    return {"gates":gates,"pin_replay_gates":pin_gates,
            "phase_timings_s":retained["phase_timings_s"],
            "evaluation_counts":retained["evaluation_counts"],
            "elapsed_s":retained["elapsed_s"],
            "passes":bool(all(gates.values()))}


def bind_operational_elapsed(certificate,elapsed):
    """Bind the post-certificate clock sample without repeating semantics."""
    result=copy.deepcopy(certificate); elapsed=float(elapsed)
    result["elapsed_s"]=elapsed
    result["gates"]["elapsed"]=bool(np.isfinite(elapsed)
        and 0.0<=elapsed<=PREFLIGHT_WALL_LIMIT_S)
    result["passes"]=bool(all(result["gates"].values()))
    return result


def transaction_schema():
    return {
        "protocol":PROTOCOL,"identity":list(PRECHECK_IDENTITY),
        "stages":list(STAGES),"gen0_before_prerequisite_or_pin":True,
        "profile_attempts":1,"optimizer_calls":1,
        "primary_pin_replay_calls":1,"independent_pin_recert_replay_calls":1,
        "total_pin_replay_calls":2,
        "prerequisite_artifact_loads":1,"prerequisite_independent_recert_calls":1,
        "p1_artifact_loads":0,
        "cuda_calls":0,"worker_calls":0,"task_rng_calls":0,
        "task_construction_calls":0,"sqp_calls":0,"initializer_calls":0,
        "wall_limit_s":PREFLIGHT_WALL_LIMIT_S,"resume_allowed":False,
        "overwrite_allowed":False,"oracle_evidence":False,
        "benchmark_evidence":False,"p1_rejected_report":P1_REJECTED_REPORT,
    }


def certify_checkpoint(document: Mapping, arrays: Mapping, generation: int):
    expected_names=[] if generation<2 else sorted(ARRAY_SPECS)
    expected_replays=({"primary_pin_replay_calls":0,
        "independent_pin_recert_replay_calls":0,"total_pin_replay_calls":0}
        if generation<2 else {"primary_pin_replay_calls":1,
        "independent_pin_recert_replay_calls":int(generation==3),
        "total_pin_replay_calls":1+int(generation==3)})
    return bool(
        set(document)=={"protocol","generation","stage","incomplete","identity",
                        "array_names","array_hashes","p1_rejected_report",
                        "replay_call_counts","oracle_evidence","benchmark_evidence",
                        "provenance"}
        and 0<=generation<4 and document["protocol"]==PROTOCOL
        and document["generation"]==generation and document["stage"]==STAGES[generation]
        and document["incomplete"] is True
        and document["identity"]==list(PRECHECK_IDENTITY)
        and document["array_names"]==expected_names
        and document["array_hashes"]=={name:array_hash(arrays[name]) for name in expected_names}
        and set(arrays)==set(expected_names)
        and (generation<2 or exact_array_schema(arrays))
        and document["p1_rejected_report"]==P1_REJECTED_REPORT
        and document["replay_call_counts"]==expected_replays
        and document["oracle_evidence"] is False
        and document["benchmark_evidence"] is False
        and certify_provenance(document["provenance"],generation==3)
    )


def refuse_existing(output):
    output=Path(output).resolve()
    candidates=[output,output.with_suffix(".npz"),
                output.with_name(output.stem+".manifest.json"),
                output.with_name(output.stem+".partial.latest.json"),
                output.with_name(output.stem+".rejected.json"),
                output.with_name(output.stem+".rejected.npz")]
    candidates.extend(output.parent.glob(output.stem+".partial.gen*"))
    candidates.extend(output.parent.glob(output.stem+"*.candidate"))
    if any(path.exists() for path in candidates):
        raise FileExistsError("P2 preflight forbids overwrite or resume")


def _write_json(path,document):
    with Path(path).open("x") as stream:
        json.dump(document,stream,sort_keys=True,separators=(",",":"))
        stream.write("\n")


def publish_checkpoint(output,document,arrays):
    generation=document["generation"]
    if not certify_checkpoint(document,arrays,generation):
        raise ValueError("P2 preflight checkpoint certificate failed")
    output=Path(output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    json_path=output.with_name(f"{output.stem}.partial.gen{generation}.json")
    npz_path=output.with_name(f"{output.stem}.partial.gen{generation}.npz")
    pointer=output.with_name(output.stem+".partial.latest.json")
    if json_path.exists() or npz_path.exists(): raise FileExistsError("checkpoint exists")
    json_tmp=json_path.with_suffix(json_path.suffix+".candidate")
    npz_tmp=npz_path.with_suffix(npz_path.suffix+".candidate")
    with npz_tmp.open("xb") as stream: np.savez(stream,**arrays)
    _write_json(json_tmp,document)
    os.replace(npz_tmp,npz_path); os.replace(json_tmp,json_path)
    pointer_tmp=pointer.with_suffix(pointer.suffix+".candidate")
    _write_json(pointer_tmp,{"protocol":PROTOCOL,"incomplete":True,
        "generation":generation,"json_path":str(json_path),
        "json_sha256":hashlib.sha256(json_path.read_bytes()).hexdigest(),
        "npz_path":str(npz_path),"npz_sha256":hashlib.sha256(npz_path.read_bytes()).hexdigest()})
    os.replace(pointer_tmp,pointer)
    return json_path,npz_path


def checkpoint_document(generation, arrays, provenance, replay_call_counts):
    names=sorted(arrays)
    return {"protocol":PROTOCOL,"generation":generation,"stage":STAGES[generation],
            "incomplete":True,"identity":list(PRECHECK_IDENTITY),
            "array_names":names,"array_hashes":{name:array_hash(arrays[name]) for name in names},
            "p1_rejected_report":P1_REJECTED_REPORT,
            "replay_call_counts":dict(replay_call_counts),"oracle_evidence":False,
            "benchmark_evidence":False,"provenance":copy.deepcopy(provenance)}


def _operational_schema(phases,counts):
    return bool(set(phases)==PHASE_KEYS
        and all(isinstance(value,(int,float)) and not isinstance(value,bool)
                and np.isfinite(value) and value>=0 for value in phases.values())
        and set(counts)==EVALUATION_KEYS
        and all(isinstance(value,int) and not isinstance(value,bool) and value>=0
                for value in counts.values()))


def measured_pin_replay(callback,ledger,key,x0,controls):
    if key not in {"primary_pin_replay_calls","independent_pin_recert_replay_calls"}:
        raise ValueError("unknown P2 Pin replay ledger key")
    value=callback(x0,controls)
    ledger[key]+=1; ledger["total_pin_replay_calls"]+=1
    return value


def rejection_document(stage,error,elapsed,provenance,phases,counts,
                       checkpoint_generations=(),replay_call_counts=None):
    replay_call_counts=(replay_call_counts or {"primary_pin_replay_calls":0,
        "independent_pin_recert_replay_calls":0,"total_pin_replay_calls":0})
    return {"protocol":PROTOCOL,"stage":stage,"incomplete":True,
        "permanently_rejected":True,"attempted":1,"completed":0,"pending":1,
        "identity":list(PRECHECK_IDENTITY),"error_type":type(error).__name__,
        "error_message":str(error),"elapsed_s":float(elapsed),
        "wall_limit_s":PREFLIGHT_WALL_LIMIT_S,"phase_timings_s":dict(phases),
        "evaluation_counts":dict(counts),"p1_rejected_report":P1_REJECTED_REPORT,
        "checkpoint_generations":list(checkpoint_generations),
        "replay_call_counts":dict(replay_call_counts),
        "oracle_evidence":False,"benchmark_evidence":False,
        "resume_allowed":False,"provenance":copy.deepcopy(provenance)}


def certify_rejection(document):
    keys={"protocol","stage","incomplete","permanently_rejected","attempted",
        "completed","pending","identity","error_type","error_message","elapsed_s",
        "wall_limit_s","phase_timings_s","evaluation_counts","p1_rejected_report",
        "checkpoint_generations","oracle_evidence","benchmark_evidence",
        "replay_call_counts","resume_allowed","provenance"}
    return bool(set(document)==keys and document["protocol"]==PROTOCOL
        and document["stage"] in {"preflight_failed","runtime_watchdog_rejected"}
        and document["incomplete"] is True and document["permanently_rejected"] is True
        and document["attempted"]==1 and document["completed"]==0
        and document["pending"]==1 and document["identity"]==list(PRECHECK_IDENTITY)
        and isinstance(document["error_type"],str) and isinstance(document["error_message"],str)
        and isinstance(document["elapsed_s"],(int,float)) and np.isfinite(document["elapsed_s"])
        and document["elapsed_s"]>=0 and document["wall_limit_s"]==PREFLIGHT_WALL_LIMIT_S
        and _operational_schema(document["phase_timings_s"],document["evaluation_counts"])
        and document["checkpoint_generations"]==list(range(len(document["checkpoint_generations"])))
        and all(isinstance(value,int) and 0<=value<4
                for value in document["checkpoint_generations"])
        and set(document["replay_call_counts"])==REPLAY_COUNT_KEYS
        and (document["replay_call_counts"]["primary_pin_replay_calls"],
             document["replay_call_counts"]["independent_pin_recert_replay_calls"])
            in {(0,0),(1,0),(1,1)}
        and document["replay_call_counts"]["total_pin_replay_calls"]
            ==document["replay_call_counts"]["primary_pin_replay_calls"]
             +document["replay_call_counts"]["independent_pin_recert_replay_calls"]
        and document["p1_rejected_report"]==P1_REJECTED_REPORT
        and document["oracle_evidence"] is False and document["benchmark_evidence"] is False
        and document["resume_allowed"] is False
        and certify_provenance(document["provenance"],True))


def publish_rejection(output,document):
    if not certify_rejection(document): raise ValueError("P2 rejection certificate failed")
    output=Path(output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    json_path=output.with_name(output.stem+".rejected.json")
    npz_path=output.with_name(output.stem+".rejected.npz")
    json_tmp=json_path.with_suffix(json_path.suffix+".candidate")
    npz_tmp=npz_path.with_suffix(npz_path.suffix+".candidate")
    with npz_tmp.open("xb") as stream: np.savez(stream)
    _write_json(json_tmp,document); os.replace(npz_tmp,npz_path); os.replace(json_tmp,json_path)
    pointer=output.with_name(output.stem+".partial.latest.json")
    pointer_tmp=pointer.with_suffix(pointer.suffix+".candidate")
    _write_json(pointer_tmp,{"protocol":PROTOCOL,"incomplete":True,
        "stage":document["stage"],"json_path":str(json_path),
        "json_sha256":_file_hash(json_path),"npz_path":str(npz_path),
        "npz_sha256":_file_hash(npz_path)})
    os.replace(pointer_tmp,pointer)
    return json_path,npz_path


def certify_final(summary: Mapping, arrays: Mapping, certificate: Mapping):
    names=sorted(ARRAY_SPECS)
    required={"protocol","incomplete","overall_pass","identity","array_names",
              "array_hashes","certificate","phase_timings_s","evaluation_counts",
              "elapsed_s","wall_limit_s","checkpoint_count","transaction",
              "p1_rejected_report","replay_call_counts","oracle_evidence",
              "benchmark_evidence","provenance"}
    gates={
        "keys":set(summary)==required,"protocol":summary.get("protocol")==PROTOCOL,
        "complete":summary.get("incomplete") is False and summary.get("overall_pass") is True,
        "identity":summary.get("identity")==list(PRECHECK_IDENTITY),
        "arrays":exact_array_schema(arrays) and summary.get("array_names")==names
        and summary.get("array_hashes")=={name:array_hash(arrays[name]) for name in names},
        "certificate":summary.get("certificate")==certificate and certificate.get("passes") is True,
        "operational_detail":certificate.get("phase_timings_s")==summary.get("phase_timings_s")
        and certificate.get("evaluation_counts")==summary.get("evaluation_counts")
        and certificate.get("elapsed_s")==summary.get("elapsed_s"),
        "timing":summary.get("elapsed_s",np.inf)<=PREFLIGHT_WALL_LIMIT_S
        and summary.get("wall_limit_s")==PREFLIGHT_WALL_LIMIT_S,
        "operational_schema":_operational_schema(summary.get("phase_timings_s",{}),
                                                   summary.get("evaluation_counts",{})),
        "checkpoints":summary.get("checkpoint_count")==4,
        "transaction":summary.get("transaction")==transaction_schema(),
        "replay_calls":summary.get("replay_call_counts")=={
            "primary_pin_replay_calls":1,"independent_pin_recert_replay_calls":1,
            "total_pin_replay_calls":2},
        "p1_report":summary.get("p1_rejected_report")==P1_REJECTED_REPORT,
        "non_evidence":summary.get("oracle_evidence") is False
        and summary.get("benchmark_evidence") is False,
        "provenance":certify_provenance(summary.get("provenance",{}),True),
    }
    return {"gates":gates,"passes":bool(all(gates.values()))}


def recertify_retained_preflight(output=OUTPUT):  # pragma: no cover - independent audit
    try:
        output=Path(output).resolve()
        if not output.exists():
            rejected=output.with_name(output.stem+".rejected.json")
            rejected_npz=output.with_name(output.stem+".rejected.npz")
            pointer_path=output.with_name(output.stem+".partial.latest.json")
            document=json.loads(rejected.read_text()); pointer=json.loads(pointer_path.read_text())
            with np.load(rejected_npz,allow_pickle=False) as archive:
                empty=archive.files==[]
            expected={"protocol":PROTOCOL,"incomplete":True,"stage":document["stage"],
                "json_path":str(rejected),"json_sha256":_file_hash(rejected),
                "npz_path":str(rejected_npz),"npz_sha256":_file_hash(rejected_npz)}
            checkpoints=True
            for generation in document.get("checkpoint_generations",[]):
                checkpoint_json=output.with_name(f"{output.stem}.partial.gen{generation}.json")
                checkpoint_npz=output.with_name(f"{output.stem}.partial.gen{generation}.npz")
                checkpoint_document_value=json.loads(checkpoint_json.read_text())
                with np.load(checkpoint_npz,allow_pickle=False) as checkpoint_archive:
                    checkpoint_arrays={name:checkpoint_archive[name] for name in checkpoint_archive.files}
                checkpoints &= certify_checkpoint(checkpoint_document_value,checkpoint_arrays,generation)
            passed=certify_rejection(document) and empty and pointer==expected and checkpoints
            return {"failure":bool(passed),"passes":bool(passed)}
        summary=json.loads(output.read_text())
        with np.load(output.with_suffix(".npz"),allow_pickle=False) as archive:
            arrays={name:archive[name] for name in archive.files}
        from gato_tiago.circular_portfolio import construct_geometry
        from gato_tiago.circular_portfolio_runner import (
            _production_pin_context,authenticate_cpu_prerequisite,
        )
        authentication,prerequisite=authenticate_cpu_prerequisite()
        if not authentication["recertification"]["passes"]: return {"passes":False}
        _model,kinematics,rnea,_rnea_derivatives,aba,tool_velocity,dense_replay=(
            _production_pin_context()
        )
        q0=prerequisite["public_x0_float32"][0,:7].astype(np.float64)
        q_goal=prerequisite["quarantined_q8_float64"][0]
        geometry=construct_geometry(prerequisite["q0_tool_float64"][0],
            prerequisite["q8_tool_float64"][0],int(prerequisite["public_default_side_int8"][0]))
        tool_reference=prerequisite["route_tool_reference_float64"][0]
        reference=np.tile(prerequisite["portfolio_reference_float32"][0],(96,1)).ravel()
        shooting=ShootingResult(PRECHECK_IDENTITY,arrays["acceleration_float64"],
            arrays["q_float64"],arrays["qd_float64"],arrays["controls_float64"],
            arrays["positions_float64"],float(arrays["shooting_objective_float64"]),
            arrays["endpoint_residual_float64"],arrays["inequalities_float64"],
            bool(arrays["optimizer_success_bool"]),int(arrays["optimizer_status_int64"]),
            int(arrays["optimizer_iterations_int64"]),float(summary["elapsed_s"]))
        shooting_certificate=certify_shooting_result(shooting,q0,q_goal,tool_reference,
            prerequisite["joint_lower_float64"],prerequisite["joint_upper_float64"],
            prerequisite["velocity_limit_float64"],prerequisite["effort_limit_float64"],
            geometry.pillar_xy,kinematics,rnea,aba)
        affine={name:arrays[name] for name in (
            "basis_float64","scalar_endpoint_map_float64","temporal_projection_float64",
            "temporal_reduced_float64","temporal_independent_float64",
            "temporal_singular_values_float64","raw_map_float64","endpoint_map_float64",
            "endpoint_right_inverse_float64","projection_float64",
            "particular_acceleration_float64","reduced_map_float64",
            "independent_map_float64","raw_to_independent_float64",
            "reduced_singular_values_float64")}
        retained={"identity":list(PRECHECK_IDENTITY),"affine":affine,
            "shooting_certificate":shooting_certificate,
            "kinematics":kinematics,
            "phase_timings_s":summary["phase_timings_s"],
            "evaluation_counts":summary["evaluation_counts"],
            "elapsed_s":summary["elapsed_s"]}
        certificate=certify_pin_preflight(retained,arrays,q0,q_goal,tool_reference,
            reference,prerequisite["joint_lower_float64"],prerequisite["joint_upper_float64"],
            prerequisite["velocity_limit_float64"],prerequisite["effort_limit_float64"],
            geometry.pillar_xy,tool_velocity,dense_replay)
        final=certify_final(summary,arrays,certificate)
        checkpoints=True; sides=[]
        for generation in range(4):
            json_path=output.with_name(f"{output.stem}.partial.gen{generation}.json")
            npz_path=output.with_name(f"{output.stem}.partial.gen{generation}.npz")
            document=json.loads(json_path.read_text())
            with np.load(npz_path,allow_pickle=False) as archive:
                retained={name:archive[name] for name in archive.files}
            checkpoints &= certify_checkpoint(document,retained,generation)
            sides.extend(({"path":str(json_path),"sha256":hashlib.sha256(json_path.read_bytes()).hexdigest()},
                          {"path":str(npz_path),"sha256":hashlib.sha256(npz_path.read_bytes()).hexdigest()}))
        manifest_path=output.with_name(output.stem+".manifest.json")
        pointer_path=output.with_name(output.stem+".partial.latest.json")
        manifest=json.loads(manifest_path.read_text()); pointer=json.loads(pointer_path.read_text())
        documents=manifest=={"protocol":PROTOCOL,"json_path":str(output),
            "json_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
            "npz_path":str(output.with_suffix('.npz')),
            "npz_sha256":hashlib.sha256(output.with_suffix('.npz').read_bytes()).hexdigest(),
            "side_artifacts":sides}
        documents &= pointer=={"protocol":PROTOCOL,"incomplete":False,
            "json_path":str(output),"json_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
            "npz_path":str(output.with_suffix('.npz')),
            "npz_sha256":hashlib.sha256(output.with_suffix('.npz').read_bytes()).hexdigest(),
            "manifest_path":str(manifest_path),
            "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
        return {"final":final,"checkpoints":bool(checkpoints),"documents":bool(documents),
                "passes":bool(final["passes"] and checkpoints and documents)}
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError,RuntimeError):
        return {"passes":False}


def execute(output=OUTPUT,authorization=None,monotonic=time.monotonic):  # pragma: no cover
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("circular portfolio P2 preflight is blocked")
    output=Path(output).resolve()
    if output!=OUTPUT: raise RuntimeError("P2 preflight output path is not authorized")
    require_thread_environment()
    refuse_existing(output)
    from gato_tiago.circular_portfolio import construct_geometry
    from gato_tiago.circular_portfolio_p2_constructor import (
        CONSTRUCTOR_EXECUTION_AUTHORIZATION,execute_reduced_constructor,
    )
    from gato_tiago.circular_portfolio_runner import (
        _accepted_prerequisite_bundle,_production_pin_context,
        authenticate_cpu_prerequisite,
    )
    started=monotonic(); checkpoints=[]; provenance=start_provenance()
    phases={name:0.0 for name in PHASE_KEYS}; counts={name:0 for name in EVALUATION_KEYS}
    operational_state={"phase_timings_s":phases,"evaluation_counts":counts}
    replay_call_counts={"primary_pin_replay_calls":0,
        "independent_pin_recert_replay_calls":0,"total_pin_replay_calls":0}
    try:
        checkpoints.append(publish_checkpoint(output,checkpoint_document(
            0,{},provenance,replay_call_counts),{}))
        authentication,prerequisite_arrays=authenticate_cpu_prerequisite()
        detail,*_= _accepted_prerequisite_bundle(authentication,prerequisite_arrays)
        if not detail["passes"]: raise RuntimeError("P2 prerequisite authentication failed")
        checkpoints.append(publish_checkpoint(output,checkpoint_document(
            1,{},provenance,replay_call_counts),{}))
        model,kinematics,rnea,rnea_derivatives,aba,tool_velocity,dense_replay=_production_pin_context()
        q0=prerequisite_arrays["public_x0_float32"][0,:7].astype(np.float64)
        q_goal=prerequisite_arrays["quarantined_q8_float64"][0]
        default=int(prerequisite_arrays["public_default_side_int8"][0])
        geometry=construct_geometry(prerequisite_arrays["q0_tool_float64"][0],
                                    prerequisite_arrays["q8_tool_float64"][0],default)
        tool_reference=prerequisite_arrays["route_tool_reference_float64"][0]
        reference=np.tile(prerequisite_arrays["portfolio_reference_float32"][0],(96,1)).ravel()
        lower=prerequisite_arrays["joint_lower_float64"]
        upper=prerequisite_arrays["joint_upper_float64"]
        velocity=prerequisite_arrays["velocity_limit_float64"]
        effort=prerequisite_arrays["effort_limit_float64"]
        retained=execute_reduced_constructor(
            PRECHECK_IDENTITY,q0,q_goal,tool_reference,lower,upper,velocity,effort,
            geometry.pillar_xy,kinematics,rnea,rnea_derivatives,aba,
            wall_limit_s=PREFLIGHT_WALL_LIMIT_S,campaign_deadline=started+PREFLIGHT_WALL_LIMIT_S,
            monotonic=monotonic,authorization=CONSTRUCTOR_EXECUTION_AUTHORIZATION,
            operational_state=operational_state,
        )
        phases=dict(retained["phase_timings_s"]); counts=dict(retained["evaluation_counts"])
        if not _operational_schema(phases,counts): raise RuntimeError("P2 operational schema failed")
        pin_state,pin_tool=measured_pin_replay(dense_replay,replay_call_counts,
            "primary_pin_replay_calls",np.r_[q0,np.zeros(7)],retained["controls_float64"])
        retained["elapsed_s"]=monotonic()-started
        retained["kinematics"]=kinematics
        affine=retained["affine"]
        arrays={"coefficients_float64":retained["coefficients_float64"],
            "basis_float64":affine["basis_float64"],
            "scalar_endpoint_map_float64":affine["scalar_endpoint_map_float64"],
            "temporal_projection_float64":affine["temporal_projection_float64"],
            "temporal_reduced_float64":affine["temporal_reduced_float64"],
            "temporal_independent_float64":affine["temporal_independent_float64"],
            "temporal_singular_values_float64":affine["temporal_singular_values_float64"],
            "raw_map_float64":affine["raw_map_float64"],
            "endpoint_map_float64":affine["endpoint_map_float64"],
            "endpoint_right_inverse_float64":affine["endpoint_right_inverse_float64"],
            "projection_float64":affine["projection_float64"],
            "particular_acceleration_float64":affine["particular_acceleration_float64"],
            "reduced_map_float64":affine["reduced_map_float64"],
            "independent_map_float64":affine["independent_map_float64"],
            "raw_to_independent_float64":affine["raw_to_independent_float64"],
            "reduced_singular_values_float64":affine["reduced_singular_values_float64"],
            "acceleration_float64":retained["acceleration_float64"],
            "q_float64":retained["q_float64"],"qd_float64":retained["qd_float64"],
            "controls_float64":retained["controls_float64"],
            "positions_float64":retained["positions_float64"],
            "pin_dense_state_float64":pin_state,"pin_dense_tool_float64":pin_tool,
            "endpoint_residual_float64":retained["endpoint_residual_float64"],
            "inequalities_float64":retained["inequalities_float64"],
            "shooting_objective_float64":retained["shooting_objective_float64"],
            "optimizer_success_bool":np.asarray(retained["optimizer_success_bool"],np.bool_),
            "optimizer_status_int64":np.asarray(retained["optimizer_status_int64"],np.int64),
            "optimizer_iterations_int64":np.asarray(retained["optimizer_iterations_int64"],np.int64)}
        arrays.update({name:retained["proxy"][name] for name in (
        "proxy_q_before_float64","proxy_position_float64","proxy_residual_float64",
        "proxy_jacobian_float64","proxy_dq_float64","proxy_q_float64")})
        arrays["proxy_initial_acceleration_float64"]=retained["proxy_initial"]["initial_acceleration_float64"]
        arrays.update({name:retained["least_squares"][name] for name in (
        "initial_coefficients_float64","least_squares_residual_float64",
        "least_squares_rank_int64","least_squares_singular_float64",
        "least_squares_reported_residual_float64")})
        checkpoints.append(publish_checkpoint(output,checkpoint_document(
            2,arrays,provenance,replay_call_counts),arrays))
        # This is the sole in-process semantic certificate.  Its independent
        # proxy/map/LS/shooting/replay work is inside the operational clock.
        def independent_recert_replay(x0_value,controls_value):
            return measured_pin_replay(dense_replay,replay_call_counts,
                "independent_pin_recert_replay_calls",x0_value,controls_value)
        certificate=certify_pin_preflight(retained,arrays,q0,q_goal,tool_reference,reference,
            lower,upper,velocity,effort,geometry.pillar_xy,tool_velocity,
            independent_recert_replay)
        elapsed=monotonic()-started
        retained["elapsed_s"]=elapsed
        certificate=bind_operational_elapsed(certificate,elapsed)
        if elapsed>PREFLIGHT_WALL_LIMIT_S:
            raise TimeoutError("circular portfolio P2 preflight wall limit")
        if not certificate["passes"]: raise RuntimeError("P2 Pin preflight certificate failed")
        # Honest provenance capture and publication I/O are explicitly outside
        # the operational tractability measurement above.
        finish_provenance(provenance)
        checkpoints.append(publish_checkpoint(output,checkpoint_document(
            3,arrays,provenance,replay_call_counts),arrays))
        summary={"protocol":PROTOCOL,"incomplete":False,"overall_pass":True,
        "identity":list(PRECHECK_IDENTITY),"array_names":sorted(arrays),
        "array_hashes":{name:array_hash(arrays[name]) for name in sorted(arrays)},
        "certificate":certificate,"phase_timings_s":retained["phase_timings_s"],
        "evaluation_counts":retained["evaluation_counts"],"elapsed_s":elapsed,
        "wall_limit_s":PREFLIGHT_WALL_LIMIT_S,"checkpoint_count":4,
        "transaction":transaction_schema(),"p1_rejected_report":P1_REJECTED_REPORT,
        "replay_call_counts":dict(replay_call_counts),
        "oracle_evidence":False,"benchmark_evidence":False,
        "provenance":copy.deepcopy(provenance)}
        if not certify_final(summary,arrays,certificate)["passes"]:
            raise RuntimeError("P2 preflight final certificate failed")
        npz_path=output.with_suffix(".npz")
        npz_tmp=npz_path.with_suffix(npz_path.suffix+".candidate")
        json_tmp=output.with_suffix(output.suffix+".candidate")
        with npz_tmp.open("xb") as stream: np.savez(stream,**arrays)
        _write_json(json_tmp,summary)
        os.replace(npz_tmp,npz_path); os.replace(json_tmp,output)
        sides=[{"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
           for pair in checkpoints for path in pair]
        manifest_path=output.with_name(output.stem+".manifest.json")
        manifest_tmp=manifest_path.with_suffix(manifest_path.suffix+".candidate")
        _write_json(manifest_tmp,{"protocol":PROTOCOL,"json_path":str(output),
        "json_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "npz_path":str(npz_path),"npz_sha256":hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "side_artifacts":sides})
        os.replace(manifest_tmp,manifest_path)
        pointer=output.with_name(output.stem+".partial.latest.json")
        pointer_tmp=pointer.with_suffix(pointer.suffix+".candidate")
        _write_json(pointer_tmp,{"protocol":PROTOCOL,"incomplete":False,
        "json_path":str(output),"json_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "npz_path":str(npz_path),"npz_sha256":hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "manifest_path":str(manifest_path),
        "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
        os.replace(pointer_tmp,pointer)
        return summary
    except Exception as error:
        phases=dict(operational_state["phase_timings_s"])
        counts=dict(operational_state["evaluation_counts"])
        if provenance.get("git_head_at_end") is None: finish_provenance(provenance)
        stage=("runtime_watchdog_rejected" if isinstance(error,TimeoutError)
               or monotonic()-started>=PREFLIGHT_WALL_LIMIT_S else "preflight_failed")
        publish_rejection(output,rejection_document(stage,error,monotonic()-started,
            provenance,phases,counts,range(len(checkpoints)),replay_call_counts))
        raise


def main(argv=None):  # pragma: no cover
    parser=argparse.ArgumentParser(); parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(argv)
    if not args.execute: raise SystemExit("--execute is required")
    execute(args.output,RUNNER_EXECUTION_AUTHORIZATION)


if __name__=="__main__": main()
