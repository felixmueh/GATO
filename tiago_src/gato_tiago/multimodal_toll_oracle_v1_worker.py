"""Isolated CUDA replay worker contract for quarantined Tiago oracle V1."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from gato_tiago.multimodal_toll_oracle_v1 import (
    DENSE_SAMPLES, DENSE_TOOL_MODEL_TOLERANCE_M, ORACLE_OUTPUT_PATH,
    TERMINAL_CUDA_M, FINAL_TOOL_SPEED_MPS, PHYSICAL_CLEARANCE_M,
    CUDA_EXTENSION_MODULE, CUDA_EXTENSION_PATH, CUDA_EXTENSION_SHA256,
    EXPECTED_LEDGER, DT, DENSE_SUBSTEPS, CUDA_MODULE_ATTRIBUTES,
)
from gato_tiago import multimodal_toll as toll


WORKER_EXECUTION_AUTHORIZATION = None
WORKER_PROTOCOL_VERSION = "tiago_tool_center_toll_oracle_v1_worker_1"
AUTHORIZED_ROOT = ORACLE_OUTPUT_PATH.parent
EXPECTED_REPLAY_ARRAY_NAMES = frozenset({
    "captured_x0_float32", "captured_reference_float32", "captured_controls_float32",
    "captured_joint_lower_float64","captured_joint_upper_float64",
    "captured_velocity_limit_float64","captured_effort_limit_float64",
    "cuda_knot_states_float32", "cuda_dense_states_float32",
    "cuda_dense_tool_float32", "dense_time_float64",
})
EXPECTED_WORKER_SUMMARY_KEYS = frozenset({"protocol","identity","request_path","request_sha256","input_npz",
    "input_npz_sha256","extension_path","extension_sha256","module_attributes","constructor_calls",
    "sim_forward_calls","tool_position_calls","solve_calls","sqp_calls","certificate","array_names",
    "array_hashes","npz_sha256"})
SIM_FORWARD_CALLS=DENSE_SAMPLES-1
TOOL_POSITION_CALLS=(DENSE_SAMPLES+15)//16


def array_hash(value):
    a=np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(a.dtype.str.encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()


def validate_request(request:Mapping):
    required={"identity","request_json","input_npz","input_npz_sha256","extension_module","extension_path","extension_sha256",
              "worker_protocol","output_json","output_npz"}
    identity=tuple(request.get("identity",()))
    try: index=EXPECTED_LEDGER.index(identity)
    except ValueError: return False
    expected_input=AUTHORIZED_ROOT/f"oracle.worker.{index:04d}.input.npz"
    expected_request=AUTHORIZED_ROOT/f"oracle.worker.{index:04d}.request.json"
    expected_json=AUTHORIZED_ROOT/f"oracle.worker.{index:04d}.json"
    expected_npz=AUTHORIZED_ROOT/f"oracle.worker.{index:04d}.npz"
    return bool(set(request)==required and request["worker_protocol"]==WORKER_PROTOCOL_VERSION
                and request["extension_module"]==CUDA_EXTENSION_MODULE
                and Path(request["extension_path"])==CUDA_EXTENSION_PATH
                and request["extension_sha256"]==CUDA_EXTENSION_SHA256
                and Path(request["request_json"])==expected_request
                and Path(request["input_npz"])==expected_input
                and Path(request["output_json"])==expected_json
                and Path(request["output_npz"])==expected_npz
                and Path(request["output_json"]).parent.resolve()==AUTHORIZED_ROOT.resolve()
                and Path(request["output_npz"]).parent.resolve()==AUTHORIZED_ROOT.resolve()
                and all(isinstance(request[key],str) and len(request[key])==64
                        and all(character in "0123456789abcdef" for character in request[key])
                        for key in ("input_npz_sha256","extension_sha256")))


def certify_replay(summary, arrays):
    if set(arrays)!=EXPECTED_REPLAY_ARRAY_NAMES: return {"passes":False}
    x0=np.asarray(arrays["captured_x0_float32"]); reference=np.asarray(arrays["captured_reference_float32"])
    controls=np.asarray(arrays["captured_controls_float32"])
    lower=np.asarray(arrays["captured_joint_lower_float64"]); upper=np.asarray(arrays["captured_joint_upper_float64"])
    velocity=np.asarray(arrays["captured_velocity_limit_float64"]); effort=np.asarray(arrays["captured_effort_limit_float64"])
    knots=np.asarray(arrays["cuda_knot_states_float32"]); dense=np.asarray(arrays["cuda_dense_states_float32"])
    tool=np.asarray(arrays["cuda_dense_tool_float32"]); times=np.asarray(arrays["dense_time_float64"])
    exact=bool(x0.shape==(14,) and reference.shape==(10,) and controls.shape==(63,7) and knots.shape==(64,14)
               and dense.shape==(DENSE_SAMPLES,14) and tool.shape==(DENSE_SAMPLES,3)
               and times.shape==(DENSE_SAMPLES,) and all(value.dtype==dtype for value,dtype in (
                   (x0,np.float32),(reference,np.float32),(controls,np.float32),(knots,np.float32),(dense,np.float32),(tool,np.float32),(times,np.float64),
                   (lower,np.float64),(upper,np.float64),(velocity,np.float64),(effort,np.float64)))
               and all(value.shape==(7,) for value in (lower,upper,velocity,effort)))
    finite=all(np.isfinite(np.asarray(value)).all() for value in arrays.values())
    ref=toll.TollReference.from_solver_bytes(reference) if exact else None
    midpoint=.5*(lower+upper); half=.5*(upper-lower)
    derived={
        "terminal_error_m":float(np.linalg.norm(tool[-1]-np.asarray(ref.goal_xyz))) if exact else np.inf,
        "physical_clearance_m":float(np.min(np.linalg.norm(tool[:,:2]-np.asarray(ref.cylinder_xy),axis=1)-ref.physical_radius_m)) if exact else -np.inf,
        "q_ratio":float(np.max(np.abs((dense[:,:7]-midpoint)/half))) if exact else np.inf,
        "v_ratio":float(np.max(np.abs(dense[:,7:])/velocity)) if exact else np.inf,
        "u_ratio":float(np.max(np.abs(controls)/effort)) if exact else np.inf,
    }
    gates={"exact_array_schema":exact,"finite":finite,
           "protocol_and_module":summary.get("protocol")==WORKER_PROTOCOL_VERSION
                and summary.get("module_attributes")==CUDA_MODULE_ATTRIBUTES,
           "common_x0_exact":exact and np.array_equal(knots[0],x0),
           "knot_dense_identity":exact and np.array_equal(knots,dense[::DENSE_SUBSTEPS]),
           "dense_time_exact":exact and np.array_equal(times,np.arange(DENSE_SAMPLES,dtype=np.float64)*DT/DENSE_SUBSTEPS),
           "terminal_error":derived["terminal_error_m"]<=TERMINAL_CUDA_M,
           "physical_clearance":derived["physical_clearance_m"]>=PHYSICAL_CLEARANCE_M,
           "limits":max(derived["q_ratio"],derived["v_ratio"],derived["u_ratio"])<=1,
           "sim_forward_calls_exact":summary.get("sim_forward_calls")==SIM_FORWARD_CALLS,
           "tool_position_calls_exact":summary.get("tool_position_calls")==TOOL_POSITION_CALLS,
           "constructor_calls_exact":summary.get("constructor_calls")==2,
           "solve_calls_zero":summary.get("solve_calls")==summary.get("sqp_calls")==0}
    return {**gates,"derived_metrics":derived,"passes":bool(all(gates.values()))}


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path,value):
    path=Path(path)
    if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
    descriptor,temporary=tempfile.mkstemp(dir=path.parent,prefix=f".{path.name}.",suffix=".tmp")
    try:
        with os.fdopen(descriptor,"w") as stream: json.dump(value,stream,sort_keys=True,indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _atomic_npz(path,arrays):
    path=Path(path)
    if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
    descriptor,temporary=tempfile.mkstemp(dir=path.parent,prefix=f".{path.name}.",suffix=".tmp")
    try:
        with os.fdopen(descriptor,"wb") as stream: np.savez_compressed(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _run_authorized_worker(request_path,output_path):  # pragma: no cover - disabled data boundary
    request=json.loads(Path(request_path).read_text())
    if (not validate_request(request) or Path(request_path)!=Path(request["request_json"])
            or Path(output_path)!=Path(request["output_json"])): raise RuntimeError("oracle worker request invalid")
    if Path(request["output_json"]).exists() or Path(request["output_npz"]).exists():
        raise RuntimeError("oracle worker output already exists")
    if _sha256_file(request["input_npz"])!=request["input_npz_sha256"] or _sha256_file(request["extension_path"])!=request["extension_sha256"]:
        raise RuntimeError("oracle worker input/extension hash mismatch")
    with np.load(request["input_npz"],allow_pickle=False) as archive: inputs={name:archive[name] for name in archive.files}
    expected={"x0_float32","reference_float32","controls_float32","joint_lower_float64","joint_upper_float64","velocity_limit_float64","effort_limit_float64"}
    if set(inputs)!=expected: raise RuntimeError("oracle worker input schema")
    module=importlib.import_module(request["extension_module"])
    actual_attributes={name:getattr(module,name) for name in CUDA_MODULE_ATTRIBUTES}
    if actual_attributes!=CUDA_MODULE_ATTRIBUTES: raise RuntimeError("oracle worker module attributes mismatch")
    b1=module.BSQP_1_float(); b16=module.BSQP_16_float()
    dense=np.empty((DENSE_SAMPLES,14),np.float32); dense[0]=inputs["x0_float32"]
    for step in range(SIM_FORWARD_CALLS):
        interval=step//DENSE_SUBSTEPS
        dense[step+1]=np.asarray(b1.sim_forward(dense[step],inputs["controls_float32"][interval],DT/DENSE_SUBSTEPS))[0]
    tool=np.empty((DENSE_SAMPLES,3),np.float32)
    for start in range(0,DENSE_SAMPLES,16):
        chunk=dense[start:min(start+16,DENSE_SAMPLES),:7]; padded=np.concatenate([chunk,np.repeat(chunk[-1:],16-len(chunk),axis=0)])
        tool[start:min(start+16,DENSE_SAMPLES)]=np.asarray(b16.tool_position(padded))[:len(chunk)]
    arrays={"captured_x0_float32":inputs["x0_float32"],"captured_reference_float32":inputs["reference_float32"],
            "captured_controls_float32":inputs["controls_float32"],"captured_joint_lower_float64":inputs["joint_lower_float64"],
            "captured_joint_upper_float64":inputs["joint_upper_float64"],"captured_velocity_limit_float64":inputs["velocity_limit_float64"],
            "captured_effort_limit_float64":inputs["effort_limit_float64"],"cuda_knot_states_float32":dense[::DENSE_SUBSTEPS],
            "cuda_dense_states_float32":dense,"cuda_dense_tool_float32":tool,
            "dense_time_float64":np.arange(DENSE_SAMPLES,dtype=np.float64)*DT/DENSE_SUBSTEPS}
    summary={"protocol":WORKER_PROTOCOL_VERSION,"identity":request["identity"],"request_path":str(request_path),
             "request_sha256":_sha256_file(request_path),"input_npz":request["input_npz"],"input_npz_sha256":request["input_npz_sha256"],
             "extension_path":request["extension_path"],"extension_sha256":request["extension_sha256"],
             "module_attributes":actual_attributes,
             "constructor_calls":2,"sim_forward_calls":SIM_FORWARD_CALLS,"tool_position_calls":TOOL_POSITION_CALLS,"solve_calls":0,"sqp_calls":0}
    certificate=certify_replay(summary,arrays)
    if not certificate["passes"]: raise RuntimeError("oracle CUDA replay failed")
    _atomic_npz(request["output_npz"],arrays); summary.update({"certificate":certificate,"array_names":sorted(arrays),
        "array_hashes":{name:array_hash(value) for name,value in arrays.items()},"npz_sha256":_sha256_file(request["output_npz"])})
    _atomic_json(output_path,summary); return summary


def execute_worker(request,output,*,authorization=None):
    if WORKER_EXECUTION_AUTHORIZATION is None or authorization is not WORKER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("oracle replay worker is blocked")
    if Path(output).parent.resolve()!=AUTHORIZED_ROOT.resolve(): raise RuntimeError("worker output root mismatch")
    return _run_authorized_worker(request,output)


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--request"); parser.add_argument("--output")
    args=parser.parse_args(argv)
    if WORKER_EXECUTION_AUTHORIZATION is None: raise SystemExit("oracle replay worker is blocked")
    execute_worker(args.request,args.output,authorization=WORKER_EXECUTION_AUTHORIZATION)


if __name__=="__main__": main()
