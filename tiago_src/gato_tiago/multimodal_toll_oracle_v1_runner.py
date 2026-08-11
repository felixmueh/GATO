"""Transactional static runner for the quarantined Tiago oracle V1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Callable, Mapping, Sequence

import numpy as np

from gato_tiago import multimodal_toll as toll
from gato_tiago.multimodal_toll_oracle_v1 import *
from gato_tiago.multimodal_toll_oracle_v1_worker import (
    EXPECTED_REPLAY_ARRAY_NAMES, EXPECTED_WORKER_SUMMARY_KEYS, WORKER_EXECUTION_AUTHORIZATION,
    WORKER_PROTOCOL_VERSION, certify_replay, validate_request,
)


RUNNER_EXECUTION_AUTHORIZATION = None
RUNNER_PROTOCOL_VERSION = ORACLE_PROTOCOL_VERSION + "_runner_1"
_PRODUCTION_TOKEN = object()


class PinOracleModel:  # pragma: no cover - disabled production dependency
    """Exact Pinocchio callbacks for the accepted Tiago tool-frame model."""
    def __init__(self):
        import pinocchio as pin
        from gato_tiago.multimodal_pillar import load_model
        self.pin=pin; self.model=load_model(); self.data=self.model.createData()
        self.tool_id=self.model.getFrameId(toll.TOOL_FRAME)
        self.torso_id=self.model.getFrameId(toll.TORSO_FRAME)
        if self.tool_id>=self.model.nframes or self.torso_id>=self.model.nframes: raise RuntimeError("Tiago tool frame missing")

    def kinematics(self,q):
        p=self.pin; q=np.asarray(q,dtype=np.float64)
        p.forwardKinematics(self.model,self.data,q); p.updateFramePlacements(self.model,self.data)
        p.computeJointJacobians(self.model,self.data,q)
        world_J=p.getFrameJacobian(self.model,self.data,self.tool_id,p.LOCAL_WORLD_ALIGNED)[:3]
        torso=self.data.oMf[self.torso_id]; relative=torso.inverse()*self.data.oMf[self.tool_id]
        return relative.translation.copy(),torso.rotation.T@world_J

    def aba_derivatives(self,q,qd,u):
        p=self.pin; q=np.asarray(q,dtype=np.float64); qd=np.asarray(qd,dtype=np.float64); u=np.asarray(u,dtype=np.float64)
        qdd=p.aba(self.model,self.data,q,qd,u).copy()
        derivatives=p.computeABADerivatives(self.model,self.data,q,qd,u)
        return qdd,np.asarray(derivatives[0]).copy(),np.asarray(derivatives[1]).copy(),np.asarray(derivatives[2]).copy()

    def tool_velocity_derivatives(self,q,qd):
        p=self.pin; q=np.asarray(q,dtype=np.float64); qd=np.asarray(qd,dtype=np.float64)
        p.computeForwardKinematicsDerivatives(self.model,self.data,q,qd,np.zeros(7))
        p.updateFramePlacements(self.model,self.data)
        torso=self.data.oMf[self.torso_id]
        velocity=torso.rotation.T@p.getFrameVelocity(self.model,self.data,self.tool_id,p.LOCAL_WORLD_ALIGNED).linear
        Dq,Dv=p.getFrameVelocityDerivatives(self.model,self.data,self.tool_id,p.LOCAL_WORLD_ALIGNED)
        return velocity,torso.rotation.T@np.asarray(Dq)[:3],torso.rotation.T@np.asarray(Dv)[:3]

    def rnea(self,q,qd,qdd):
        return self.pin.rnea(self.model,self.data,np.asarray(q),np.asarray(qd),np.asarray(qdd)).copy()

    def dense_replay(self,x0,controls):
        state=np.empty((DENSE_SAMPLES,NX),dtype=np.float64); tool=np.empty((DENSE_SAMPLES,3),dtype=np.float64)
        state[0]=np.asarray(x0,dtype=np.float64); tool[0]=self.kinematics(state[0,:7])[0]
        for step in range(DENSE_SAMPLES-1):
            interval=step//DENSE_SUBSTEPS; dt=DT/DENSE_SUBSTEPS
            qdd=self.pin.aba(self.model,self.data,state[step,:7],state[step,7:],controls[interval]).copy()
            state[step+1,:7]=state[step,:7]+dt*state[step,7:]+.5*dt**2*qdd
            state[step+1,7:]=state[step,7:]+dt*qdd
            tool[step+1]=self.kinematics(state[step+1,:7])[0]
        return state,tool


def repository_root(module_path=Path(__file__)):
    root = Path(module_path).resolve().parents[2]
    if not (root/".git").exists() or not (root/"CMakeLists.txt").is_file():
        raise RuntimeError("oracle repository root mismatch")
    return root


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def source_hashes(root):
    return {
        label:{"path":path,"sha256":sha256_file(Path(root)/path)}
        for label,path in REQUIRED_SOURCE_PATHS.items()
    }


def start_provenance(root=None, *, orig_argv=None):
    root=repository_root() if root is None else Path(root).resolve()
    argv=tuple(sys.orig_argv if orig_argv is None else orig_argv)
    status=_git(root,"status","--porcelain","--untracked-files=no")
    extension=(root/CUDA_EXTENSION_PATH).resolve()
    return {
        "git_head_at_start":_git(root,"rev-parse","HEAD"),"git_head_at_end":None,
        "tracked_tree_clean_at_start":status=="","tracked_tree_clean_at_end":None,
        "full_git_status_at_start":_git(root,"status","--porcelain=v1"),
        "full_git_status_at_end":None,"cwd":str(Path.cwd().resolve()),
        "orig_argv":list(argv),"exact_command":shlex.join(argv),
        "source_hashes_at_start":source_hashes(root),"source_hashes_at_end":None,
        "python_version":sys.version,"numpy_version":np.__version__,
        "scipy_version":importlib.metadata.version("scipy"),"pinocchio_version":importlib.metadata.version("pin"),
        "extension_path":str(CUDA_EXTENSION_PATH),"extension_sha256":sha256_file(extension),
        "extension_size_bytes":extension.stat().st_size,"extension_build_head":CUDA_BUILD_HEAD,
        "cuda_arch":CUDA_ARCH,"expected_module_attributes":CUDA_MODULE_ATTRIBUTES,
        "task_artifact_pins":{k:{"path":str(p),"sha256":h} for k,(p,h) in TASK_ARTIFACT_PINS.items()},
        "model_artifact_pins":{k:{"path":str(p),"sha256":h} for k,(p,h) in MODEL_ARTIFACT_PINS.items()},
        "task_artifact_loads":0,"model_artifact_loads":0,"task_rng_calls":0,
        "task_construction_calls":0,"initializer_calls":0,"acquisition_optimizer_calls":0,
        "polish_optimizer_calls":0,"worker_subprocess_calls":0,"cuda_replay_calls":0,
        "sqp_optimization_calls":0,"bootstrap_rng_calls":0,
    }


def finish_runner_provenance(provenance, root=None):
    root=repository_root() if root is None else Path(root).resolve()
    provenance["git_head_at_end"]=_git(root,"rev-parse","HEAD")
    tracked=_git(root,"status","--porcelain","--untracked-files=no")
    provenance["tracked_tree_clean_at_end"]=tracked==""
    provenance["full_git_status_at_end"]=_git(root,"status","--porcelain=v1")
    provenance["source_hashes_at_end"]=source_hashes(root)
    extension=(root/CUDA_EXTENSION_PATH).resolve()
    provenance["extension_sha256_at_end"]=sha256_file(extension)
    provenance["extension_size_bytes_at_end"]=extension.stat().st_size


def certify_provenance(provenance):
    start=provenance.get("git_head_at_start"); end=provenance.get("git_head_at_end")
    sources=provenance.get("source_hashes_at_start",{})
    source_gate=bool(set(sources)==set(REQUIRED_SOURCE_PATHS) and all(
        row.get("path")==REQUIRED_SOURCE_PATHS[label]
        and isinstance(row.get("sha256"),str) and len(row["sha256"])==64
        and all(c in "0123456789abcdef" for c in row["sha256"])
        for label,row in sources.items()))
    return bool(
        isinstance(start,str) and len(start)==40 and all(c in "0123456789abcdef" for c in start)
        and start==end and provenance.get("tracked_tree_clean_at_start") is True
        and provenance.get("tracked_tree_clean_at_end") is True
        and provenance.get("cwd")==str(AUTHORIZED_CWD)
        and tuple(provenance.get("orig_argv",()))==AUTHORIZED_ORIG_ARGV
        and provenance.get("exact_command")==shlex.join(AUTHORIZED_ORIG_ARGV)
        and sources==provenance.get("source_hashes_at_end") and source_gate
        and provenance.get("extension_path")==str(CUDA_EXTENSION_PATH)
        and provenance.get("extension_sha256")==CUDA_EXTENSION_SHA256
        and provenance.get("extension_size_bytes")==CUDA_EXTENSION_SIZE_BYTES
        and provenance.get("extension_sha256_at_end")==CUDA_EXTENSION_SHA256
        and provenance.get("extension_size_bytes_at_end")==CUDA_EXTENSION_SIZE_BYTES
        and provenance.get("extension_build_head")==CUDA_BUILD_HEAD and provenance.get("cuda_arch")==CUDA_ARCH
        and provenance.get("expected_module_attributes")==CUDA_MODULE_ATTRIBUTES
        and provenance.get("task_artifact_pins")=={k:{"path":str(p),"sha256":h} for k,(p,h) in TASK_ARTIFACT_PINS.items()}
        and provenance.get("model_artifact_pins")=={k:{"path":str(p),"sha256":h} for k,(p,h) in MODEL_ARTIFACT_PINS.items()}
        and all(isinstance(provenance.get(name),str) and provenance.get(name) for name in
                ("python_version","numpy_version","scipy_version","pinocchio_version"))
        and provenance.get("task_artifact_loads")==1
        and provenance.get("model_artifact_loads")==1
        and provenance.get("task_rng_calls")==provenance.get("task_construction_calls")==0
        and provenance.get("initializer_calls")==0
        and provenance.get("acquisition_optimizer_calls")==EXPECTED_PAIR_COUNT
        and provenance.get("polish_optimizer_calls")==EXPECTED_PAIR_COUNT
        and provenance.get("worker_subprocess_calls")==EXPECTED_PAIR_COUNT
        and provenance.get("cuda_replay_calls")==EXPECTED_PAIR_COUNT
        and provenance.get("sqp_optimization_calls")==0
        and provenance.get("bootstrap_rng_calls")==1
        and provenance.get("all_compact_dls_binding_precheck_pass") is True
        and provenance.get("pin_derivative_diagnostics",{}).get("all_pin_derivative_diagnostics_pass") is True
    )


def _load_pinned_artifact(pins):  # pragma: no cover - data execution boundary
    for path,digest in pins.values():
        if not Path(path).is_file() or sha256_file(path)!=digest:
            raise RuntimeError("oracle prerequisite path/hash mismatch")
    summary=json.loads(Path(pins["final_json"][0]).read_text())
    with np.load(pins["final_npz"][0],allow_pickle=False) as archive:
        arrays={name:archive[name] for name in archive.files}
    return summary,arrays


def load_authenticated_task_artifact():  # pragma: no cover
    from gato_tiago.multimodal_toll_v4_runner import certify_final
    summary,arrays=_load_pinned_artifact(TASK_ARTIFACT_PINS)
    certificate=certify_final(summary.get("rows",()),summary.get("provenance",{}),arrays,
                              test_override=False)
    if certificate.get("all_v4_gates_pass") is not True or certificate!=summary.get("certificate"):
        raise RuntimeError("task artifact pure recertification failed")
    return summary,summary["rows"],arrays


def load_authenticated_model_artifact():  # pragma: no cover
    from gato_tiago.multimodal_toll_v4_model_preflight_v4_runner import recertify_retained_model_preflight
    summary,arrays=_load_pinned_artifact(MODEL_ARTIFACT_PINS)
    recert=recertify_retained_model_preflight(MODEL_ARTIFACT_PINS["final_json"][0],
                                               MODEL_ARTIFACT_PINS["final_npz"][0])
    if recert.get("all_roundtrip_recertification_gates_pass") is not True:
        raise RuntimeError("model artifact pure disk recertification failed")
    return summary,arrays


class ProductionOracleDependencies:  # pragma: no cover - all capabilities disabled
    def __init__(self,output,provenance):
        self.output=Path(output); self.provenance=provenance; self.pin=None
        self.model_arrays=None; self.derivatives_checked=False; self.campaign_deadline=None

    def set_campaign_deadline(self,deadline): self.campaign_deadline=float(deadline)

    def task_loader(self):
        summary,rows,arrays=load_authenticated_task_artifact(); self.provenance["task_artifact_loads"]+=1
        return summary,rows,arrays

    def model_loader(self):
        summary,arrays=load_authenticated_model_artifact(); self.provenance["model_artifact_loads"]+=1
        self.model_arrays=arrays
        return summary,arrays

    def _ensure_pin(self):
        if self.pin is None: self.pin=PinOracleModel()

    def _problem(self,task_index,template=None):
        self._ensure_pin(); a=self.model_arrays
        return OriginalOracleNLP(a["public_solver_x0_float32"][task_index],a["public_reference_float32"][task_index],
            a["model_lower_float64"],a["model_upper_float64"],a["model_velocity_float64"],a["model_effort_float64"],
            self.pin.kinematics,self.pin.aba_derivatives,self.pin.tool_velocity_derivatives,template=template,
            rnea=self.pin.rnea,dense_replay=self.pin.dense_replay)

    def problem_factory(self,index):
        task_index=EXPECTED_TASK_IDENTITIES.index(EXPECTED_LEDGER[index][:2]); return self._problem(task_index)

    def acquisition(self,identity,_task_rows,_task_arrays,model_arrays):
        self._ensure_pin()
        task_index=EXPECTED_TASK_IDENTITIES.index(tuple(identity[:2])); requested=identity[2]; margin_index=identity[3]
        seed=build_acquisition_initial(model_arrays["public_solver_x0_float32"][task_index],
            model_arrays["public_reference_float32"][task_index],model_arrays["public_default_side_int8"][task_index],
            model_arrays["quarantined_q8_float64"][task_index],model_arrays["model_lower_float64"],
            model_arrays["model_upper_float64"],requested,margin_index,self.pin.kinematics,self.pin.rnea)
        if not self.derivatives_checked:
            diagnostic=production_pin_derivative_diagnostics(self._problem(task_index),
                model_arrays["public_solver_x0_float32"][task_index,:7],model_arrays["quarantined_q8_float64"][task_index],self.pin.rnea)
            if not diagnostic["all_pin_derivative_diagnostics_pass"]: raise RuntimeError("production Pin derivatives failed")
            self.provenance["pin_derivative_diagnostics"]=_json_value(diagnostic); self.derivatives_checked=True
        anchored=self._problem(task_index,seed["template_float64"])
        result=solve_anchored_acquisition(anchored,seed["primal_float64"],campaign_deadline=self.campaign_deadline); self.provenance["acquisition_optimizer_calls"]+=1
        primal=np.asarray(result.x,dtype=np.float64); x,u=unpack_z(primal); tool,_=anchored.positions_and_jacobians(x)
        dense_state,dense_tool=self.pin.dense_replay(model_arrays["public_solver_x0_float32"][task_index],u)
        arrays={"template_float64":seed["template_float64"],"seed_q_float64":seed["q_float64"],
                "seed_qd_float64":seed["qd_float64"],"seed_qdd_float64":seed["qdd_float64"],
                "seed_u_float64":seed["u_float64"],"seed_primal_float64":seed["primal_float64"],
                **{name:value for name,value in seed.items() if name.startswith("dls_")},
                "acquisition_primal_float64":primal,"acquisition_tool_float64":tool,
                "acquisition_dense_state_float64":dense_state,"acquisition_dense_tool_float64":dense_tool}
        summary={"success":bool(result.success),"finite":bool(np.isfinite(primal).all()),
                 "status":int(result.status),"message":str(result.message),"iterations":int(result.nit),
                 "limit_reached":bool(result.nit>=ACQUISITION_MAXITER),"passes":bool(result.success and np.isfinite(primal).all())}
        return summary,arrays

    def polish(self,payload):
        task_index=next(i for i in range(12) if np.array_equal(self.model_arrays["public_solver_x0_float32"][i],payload["public_x0_float32"])
                        and np.array_equal(self.model_arrays["public_reference_float32"][i],payload["public_reference_float32"]))
        problem=self._problem(task_index)
        result=solve_unanchored_polish(problem,payload,campaign_deadline=self.campaign_deadline); self.provenance["polish_optimizer_calls"]+=1
        primal=np.asarray(result.x,dtype=np.float64); x,u=unpack_z(primal); tool,_=problem.positions_and_jacobians(x)
        pin_state,pin_tool=self.pin.dense_replay(problem.x0,u)
        arrays={"polish_primal_float64":primal,"polish_tool_float64":tool,
                "pin_dense_state_float64":pin_state,"pin_dense_tool_float64":pin_tool,
                "controls_float64":u,"dense_time_float64":np.arange(DENSE_SAMPLES,dtype=np.float64)*DT/DENSE_SUBSTEPS,
                **retained_scipy_multipliers(result)}
        summary={"success":bool(result.success),"finite":bool(np.isfinite(primal).all()),"status":int(result.status),
                 "message":str(result.message),"iterations":int(result.nit),"limit_reached":bool(result.nit>=POLISH_MAXITER),
                 "passes":bool(result.success and np.isfinite(primal).all())}
        return summary,arrays

    def replay(self,identity,polish_arrays,task_index,base_arrays):
        index=EXPECTED_LEDGER.index(tuple(identity)); stem=f"{self.output.stem}.worker.{index:04d}"
        request_path=self.output.with_name(stem+".request.json"); input_path=self.output.with_name(stem+".input.npz")
        output_json=self.output.with_name(stem+".json"); output_npz=self.output.with_name(stem+".npz")
        if any(path.exists() for path in (request_path,input_path,output_json,output_npz)): raise RuntimeError("worker side artifact exists")
        worker_input={"x0_float32":np.asarray(base_arrays["public_x0_float32"][task_index],np.float32),
                      "reference_float32":np.asarray(base_arrays["public_reference_float32"][task_index],np.float32),
                      "controls_float32":np.asarray(polish_arrays["controls_float64"],np.float32),
                      "joint_lower_float64":base_arrays["joint_lower_float64"],"joint_upper_float64":base_arrays["joint_upper_float64"],
                      "velocity_limit_float64":base_arrays["velocity_limit_float64"],"effort_limit_float64":base_arrays["effort_limit_float64"]}
        _atomic_npz(input_path,worker_input)
        request={"identity":list(identity),"request_json":str(request_path),
                 "input_npz":str(input_path),"input_npz_sha256":sha256_file(input_path),
                 "extension_module":CUDA_EXTENSION_MODULE,"extension_path":str(CUDA_EXTENSION_PATH),
                 "extension_sha256":CUDA_EXTENSION_SHA256,"worker_protocol":WORKER_PROTOCOL_VERSION,
                 "output_json":str(output_json),"output_npz":str(output_npz)}
        _atomic_json(request_path,request)
        command=[sys.executable,"-B","-m","gato_tiago.multimodal_toll_oracle_v1_worker","--request",str(request_path),"--output",str(output_json)]
        completed=subprocess.run(command,cwd=AUTHORIZED_CWD,env={**os.environ,"PYTHONPATH":"tiago_src:python"},text=True,capture_output=True)
        self.provenance["worker_subprocess_calls"]+=1
        if completed.returncode!=0: raise RuntimeError(f"oracle worker failed: {completed.returncode}: {completed.stderr}")
        summary=json.loads(output_json.read_text())
        with np.load(output_npz,allow_pickle=False) as archive: worker_arrays={name:archive[name] for name in archive.files}
        if not validate_request(request) or not certify_replay(summary,worker_arrays)["passes"]: raise RuntimeError("worker boundary recertification failed")
        self.provenance["cuda_replay_calls"]+=1
        artifact={"request_path":str(request_path),"request_sha256":sha256_file(request_path),"input_path":str(input_path),
                  "input_sha256":sha256_file(input_path),"json_path":str(output_json),"json_sha256":sha256_file(output_json),
                  "npz_path":str(output_npz),"npz_sha256":sha256_file(output_npz)}
        summary["artifact"]=artifact
        summary.update({"command":command,"cwd":str(AUTHORIZED_CWD),"exit_code":completed.returncode,
                        "stdout":completed.stdout,"stderr":completed.stderr})
        return summary,{"cuda_dense_state_float32":worker_arrays["cuda_dense_states_float32"],
                        "cuda_dense_tool_float32":worker_arrays["cuda_dense_tool_float32"]}

    def before_finish(self,rows,arrays,provenance):
        canonical=[]; all_dls=True
        try:
            if len(rows)!=EXPECTED_PAIR_COUNT: raise ValueError("incomplete pair ledger")
            for index,row in enumerate(rows):
                certificate=certify_retained_row(row,index,arrays,self.problem_factory(index),reenumerate=False)
                canonical.append(certificate["derived"]); all_dls &= certificate["dls"].get("passes") is True
            pre=aggregate_campaign(canonical,None,effort_limit=arrays["effort_limit_float64"])
            if not all(len(pre["heldout_gaps"][model])==8 for model in ("cuda","pin")): raise ValueError("incomplete heldout gaps")
            retained=bootstrap_heldout_gaps(pre["heldout_gaps"]); provenance["bootstrap_rng_calls"]+=1
            arrays["bootstrap_input_gaps_float64"]=np.stack([retained["input_gaps"][m] for m in ("cuda","pin")])
            arrays["bootstrap_samples_float64"]=retained["samples"]
        except (KeyError,ValueError,IndexError):
            all_dls=False
        provenance["all_compact_dls_binding_precheck_pass"]=bool(all_dls)


def _atomic_bytes(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _atomic_json(path, value):
    _atomic_bytes(path, (json.dumps(_json_value(value), sort_keys=True, indent=2)+"\n").encode())


def _atomic_npz(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **{k:np.asarray(v) for k,v in arrays.items()})
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _json_value(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Mapping): return {str(k):_json_value(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_json_value(v) for v in value]
    return value


def _no_existing(output):
    output = Path(output)
    candidates = [output, output.with_suffix(".npz"), output.with_suffix(".manifest.json"),
                  output.with_name(f"{output.stem}.partial.latest.json"),
                  *output.parent.glob(f"{output.stem}.partial.*"),
                  *output.parent.glob(f"{output.stem}.acquisition.*"),
                  *output.parent.glob(f"{output.stem}.pair.*"),
                  *output.parent.glob(f"{output.stem}.worker.*")]
    if any(path.exists() for path in candidates):
        raise RuntimeError("oracle refuses resume, overwrite, or rerun")


def _checkpoint(output, generation, stage, rows, arrays, provenance, *, watchdog=None):
    output = Path(output)
    npz = output.with_name(f"{output.stem}.partial.{generation:04d}.npz")
    summary = output.with_name(f"{output.stem}.partial.{generation:04d}.json")
    pointer = output.with_name(f"{output.stem}.partial.latest.json")
    if npz.exists() or summary.exists(): raise RuntimeError("checkpoint generation exists")
    retained = {"completed_count_int64":np.asarray(len(rows),dtype=np.int64)}
    _atomic_npz(npz, retained)
    payload = {"incomplete":True,"generation":generation,"stage":stage,
               "expected_identities":[list(v) for v in EXPECTED_LEDGER],
               "completed_identities":[row["identity"] for row in rows],
               "completed_count":len(rows),"pending_identities":[list(v) for v in EXPECTED_LEDGER[len(rows):]],
               "rows":rows,"provenance":provenance,"all_oracle_gates_pass":False,
               "array_names":sorted(retained),"array_hashes":{k:array_hash(v) for k,v in retained.items()}}
    if watchdog is not None:
        payload["runtime_watchdog"]={**watchdog,"operational_timing_only":True,
            "oracle_evidence":False,"benchmark_timing_evidence":False,"permanently_rejects_protocol":True}
    payload["npz_path"] = str(npz); payload["npz_sha256"] = sha256_file(npz)
    _atomic_json(summary,payload)
    _atomic_json(pointer,{"generation":generation,"json_path":str(summary),"json_sha256":sha256_file(summary),
                          "npz_path":str(npz),"npz_sha256":payload["npz_sha256"],"incomplete":True})
    return payload


def publish_pair(output,row_index,identity,arrays):
    output=Path(output); stem=f"{output.stem}.pair.{row_index:04d}"; npz=output.with_name(stem+".npz"); js=output.with_name(stem+".json")
    if npz.exists() or js.exists(): raise RuntimeError("pair overwrite")
    if set(arrays)!=set(ROW_ARRAY_SPECS) or any(np.asarray(arrays[name]).shape!=ROW_ARRAY_SPECS[name][0]
        or np.asarray(arrays[name]).dtype!=np.dtype(ROW_ARRAY_SPECS[name][1]) for name in arrays):
        raise RuntimeError("pair retained array schema invalid")
    _atomic_npz(npz,arrays); payload={"identity":list(identity),"row_index":row_index,"protocol":ORACLE_PROTOCOL_VERSION,
        "array_names":sorted(arrays),"array_hashes":{name:array_hash(value) for name,value in arrays.items()},
        "npz_path":str(npz),"npz_sha256":sha256_file(npz),"oracle_evidence":False,"ledger_pointer_published":False}
    _atomic_json(js,payload); return {"json_path":str(js),"json_sha256":sha256_file(js),"npz_path":str(npz),"npz_sha256":payload["npz_sha256"]}


def publish_acquisition(output, row_index, identity, summary, arrays):
    """Publish acquisition before polish; unmatched files remain non-evidence."""
    output = Path(output); stem = f"{output.stem}.acquisition.{row_index:04d}"
    npz = output.with_name(stem+".npz"); js = output.with_name(stem+".json")
    exact=set(arrays)==ACQUISITION_ARRAY_NAMES and all(np.asarray(arrays[name]).shape==ROW_ARRAY_SPECS[name][0]
        and np.asarray(arrays[name]).dtype==np.dtype(ROW_ARRAY_SPECS[name][1]) and np.isfinite(arrays[name]).all() for name in arrays)
    if not exact: raise RuntimeError("acquisition retained array schema invalid")
    if npz.exists() or js.exists(): raise RuntimeError("acquisition overwrite")
    _atomic_npz(npz, arrays)
    payload = {**summary,"identity":list(identity),"row_index":row_index,"protocol":ORACLE_PROTOCOL_VERSION,
               "npz_path":str(npz),"npz_sha256":sha256_file(npz),
               "array_names":sorted(arrays),"array_hashes":{k:array_hash(v) for k,v in arrays.items()},
               "ledger_pointer_published":False,"oracle_evidence":False}
    _atomic_json(js,payload)
    return {"json_path":str(js),"json_sha256":sha256_file(js),"npz_path":str(npz),"npz_sha256":payload["npz_sha256"]}


def authenticate_summary(summary, arrays, *, kind):
    expected_count = 603 if kind == "task" else None
    certificate_key = "all_v4_gates_pass" if kind == "task" else "all_model_preflight_gates_pass"
    return bool(
        summary.get("incomplete") is False
        and summary.get("certificate",{}).get(certificate_key) is True
        and summary.get(certificate_key) is True
        and set(summary.get("array_names",())) == set(arrays)
        and set(summary.get("array_hashes",())) == set(arrays)
        and all(summary["array_hashes"][k] == array_hash(v) for k,v in arrays.items())
        and (expected_count is None or len(arrays)==expected_count)
    )


def cross_bind_prerequisites(task_rows, task_arrays, model_arrays):
    identities = [tuple(row.get("identity",())) for row in task_rows]
    return bool(
        identities == list(EXPECTED_TASK_IDENTITIES)
        and np.array_equal(model_arrays.get("public_solver_x0_float32"),
                           np.stack([task_arrays[f"task_{seed}_x0_float32"] for _,seed in EXPECTED_TASK_IDENTITIES]))
        and np.array_equal(model_arrays.get("public_reference_float32"),
                           np.stack([task_arrays[f"task_{seed}_reference_float32"] for _,seed in EXPECTED_TASK_IDENTITIES]))
        and np.array_equal(model_arrays.get("public_default_side_int8"),
                           np.asarray([row["public_task"]["default_side"] for row in task_rows],dtype=np.int8))
        and np.array_equal(model_arrays.get("quarantined_q8_float64"),
                           np.stack([task_arrays[f"task_{seed}_history_q_float64"][8] for _,seed in EXPECTED_TASK_IDENTITIES]))
        and np.all(model_arrays.get("quarantined_q8_oracle_only_bool",False))
        and not np.any(model_arrays.get("quarantined_q8_initializer_eligible_bool",True))
        and not np.any(model_arrays.get("quarantined_q8_solver_seed_eligible_bool",True))
        and np.asarray(model_arrays.get("v4_artifact_authentication_gate_bool",False)).shape==()
        and bool(np.asarray(model_arrays.get("v4_artifact_authentication_gate_bool",False)).item())
    )


def directional_derivative_gate(value, gradient, z, direction, epsilon=1e-6):
    z=np.asarray(z,dtype=np.float64); d=np.asarray(direction,dtype=np.float64)
    analytic=float(np.asarray(gradient(z))@d)
    numeric=float((value(z+epsilon*d)-value(z-epsilon*d))/(2*epsilon))
    error=abs(analytic-numeric); scale=max(abs(analytic),abs(numeric))
    return {"analytic":analytic,"numeric":numeric,"absolute_error":error,
            "relative_error":error/max(scale,1e-30),"passes":error<=DERIVATIVE_ABS_TOLERANCE if scale<=DERIVATIVE_ABS_TOLERANCE else error/scale<=DERIVATIVE_REL_TOLERANCE}


def directional_jacobian_gate(value,jacobian,z,direction,epsilon=1e-6):
    z=np.asarray(z,dtype=np.float64); d=np.asarray(direction,dtype=np.float64)
    analytic=np.asarray(jacobian(z))@d
    numeric=(np.asarray(value(z+epsilon*d))-np.asarray(value(z-epsilon*d)))/(2*epsilon)
    error=float(np.max(np.abs(analytic-numeric),initial=0)); scale=float(max(np.max(np.abs(analytic),initial=0),np.max(np.abs(numeric),initial=0)))
    return {"absolute_error":error,"relative_error":error/max(scale,1e-30),
            "passes":error<=DERIVATIVE_ABS_TOLERANCE if scale<=DERIVATIVE_ABS_TOLERANCE else error/scale<=DERIVATIVE_REL_TOLERANCE}


def production_pin_derivative_diagnostics(problem,q0,q8,rnea):
    if type(problem) is not OriginalOracleNLP or problem.anchored: raise ValueError("Pin diagnostic problem must be original/unanchored")
    q0=np.asarray(q0,dtype=np.float64); q8=np.asarray(q8,dtype=np.float64)
    ref=toll.TollReference.from_solver_bytes(problem.reference)
    linear_q=q0[None,:]+np.linspace(0.,1.,65)[:,None]*(q8-q0)[None,:]
    distances=np.asarray([np.linalg.norm(problem.kinematics(q)[0][:2]-np.asarray(ref.cylinder_xy)) for q in linear_q])
    near_index=int(np.argmin(distances)); configurations=(q0,linear_q[near_index],q8)
    labels=("interior","near_cylinder","terminal"); rows=[]
    direction=np.sin(np.arange(1,Z_WIDTH+1,dtype=np.float64))*.01
    for label,q in zip(labels,configurations):
        x=np.zeros((KNOTS,NX)); x[:,:7]=q; u=np.tile(rnea(q,np.zeros(7),np.zeros(7)),(INTERVALS,1)); z=pack_z(x,u)
        objective=directional_derivative_gate(problem.objective,problem.objective_gradient,z,direction)
        equality=directional_jacobian_gate(problem.equality,problem.equality_jacobian,z,direction)
        inequality=directional_jacobian_gate(problem.inequality,problem.inequality_jacobian,z,direction)
        position,_=problem.kinematics(q)
        rows.append({"label":label,"q_float64":q,"primal_float64":z,
                     "linear_grid_index":near_index if label=="near_cylinder" else None,
                     "cylinder_distance_m":float(np.linalg.norm(np.asarray(position)[:2]-np.asarray(ref.cylinder_xy))),
                     "objective":objective,"equality":equality,"inequality":inequality,
                     "passes":objective["passes"] and equality["passes"] and inequality["passes"]})
    return {"rows":rows,"exact_labels":[r["label"] for r in rows]==list(labels),
            "all_pin_derivative_diagnostics_pass":all(r["passes"] for r in rows)}


def _trust_constr(problem, initial, *, maxiter, wall_limit, campaign_deadline):  # pragma: no cover
    from scipy.optimize import BFGS, Bounds, NonlinearConstraint, minimize
    from scipy.sparse import csr_matrix
    start=time.monotonic()
    def callback(_x,_state=None):
        now=time.monotonic()
        if now>=campaign_deadline: raise TimeoutError("frozen oracle campaign wall limit")
        if now-start > wall_limit: raise TimeoutError("frozen oracle per-call wall limit")
    constraints=(NonlinearConstraint(problem.equality,0.,0.,jac=lambda z:csr_matrix(problem.equality_jacobian(z))),
                 NonlinearConstraint(problem.inequality,0.,np.inf,jac=lambda z:csr_matrix(problem.inequality_jacobian(z))))
    lower,upper=problem.bounds_arrays()
    return minimize(problem.objective,initial,jac=problem.objective_gradient,hess=BFGS(),
                    method="trust-constr",constraints=constraints,bounds=Bounds(lower,upper),
                    callback=callback,options={"maxiter":maxiter,"gtol":1e-10,"xtol":1e-12,
                                               "barrier_tol":1e-12,"sparse_jacobian":True})


def solve_anchored_acquisition(problem, initial, *, campaign_deadline):  # pragma: no cover
    """Acquisition-only solve; ``problem`` may contain template/anchor data."""
    return _trust_constr(problem, initial, maxiter=ACQUISITION_MAXITER,
                         wall_limit=ACQUISITION_WALL_LIMIT_S,campaign_deadline=campaign_deadline)


def solve_unanchored_polish(problem, polish_input, *, campaign_deadline):  # pragma: no cover
    """Certifying solve whose boundary is deliberately free of oracle hints."""
    validate_unanchored_polish_input(polish_input)
    if not certify_polish_problem_binding(problem,polish_input):
        raise ValueError("polish problem is not exact public unanchored OriginalOracleNLP")
    return _trust_constr(problem, polish_input["acquisition_primal_float64"],
                         maxiter=POLISH_MAXITER, wall_limit=POLISH_WALL_LIMIT_S,
                         campaign_deadline=campaign_deadline)


def retained_scipy_multipliers(result):
    raw=getattr(result,"v",None)
    if not isinstance(raw,(list,tuple)) or len(raw)!=3: raise ValueError("trust-constr multiplier groups missing")
    raw_eq=np.asarray(raw[0],dtype=np.float64); raw_ineq=np.asarray(raw[1],dtype=np.float64); raw_bounds=np.asarray(raw[2],dtype=np.float64)
    mapped=map_scipy_multipliers(raw_eq,raw_ineq,raw_bounds)
    return {"raw_equality_multipliers_float64":raw_eq,"raw_inequality_multipliers_float64":raw_ineq,
            "raw_bound_multipliers_float64":raw_bounds,
            **{f"canonical_{name}_multipliers_float64":value for name,value in mapped.items()}}


def certify_replay_row(row, arrays):
    required = ("common_x0","common_reference","joint_lower","joint_upper","velocity_limit","effort_limit",
                "cuda_dense_tool","pin_dense_tool","cuda_dense_state","pin_dense_state","controls")
    if any(name not in arrays for name in required): return False
    cuda=np.asarray(arrays["cuda_dense_tool"]); pin=np.asarray(arrays["pin_dense_tool"])
    cuda_state=np.asarray(arrays["cuda_dense_state"]); pin_state=np.asarray(arrays["pin_dense_state"])
    x0=np.asarray(arrays["common_x0"]); controls=np.asarray(arrays["controls"])
    finite=all(np.isfinite(np.asarray(v)).all() for v in arrays.values())
    metrics=("cuda_terminal_error_m","pin_terminal_error_m","cuda_final_tool_speed_mps",
             "pin_final_tool_speed_mps","cuda_q_ratio","cuda_v_ratio","cuda_u_ratio",
             "pin_q_ratio","pin_v_ratio","pin_u_ratio")
    domains=all(np.isfinite(float(row.get(name,np.nan))) and float(row[name])>=0 for name in metrics)
    return bool(finite and cuda.shape==pin.shape==(DENSE_SAMPLES,3)
                and cuda_state.shape==pin_state.shape==(DENSE_SAMPLES,NX)
                and x0.shape==(NX,) and controls.shape==(INTERVALS,NU)
                and np.array_equal(cuda_state[0],x0) and np.array_equal(pin_state[0],x0)
                and domains
                and np.max(np.linalg.norm(cuda-pin,axis=1))<=DENSE_TOOL_MODEL_TOLERANCE_M
                and row.get("cuda_terminal_error_m",np.inf)<=TERMINAL_CUDA_M
                and row.get("pin_terminal_error_m",np.inf)<=TERMINAL_PIN_M
                and row.get("cuda_final_tool_speed_mps",np.inf)<=FINAL_TOOL_SPEED_MPS
                and row.get("pin_final_tool_speed_mps",np.inf)<=FINAL_TOOL_SPEED_MPS
                and row.get("cuda_physical_clearance_m",-np.inf)>=PHYSICAL_CLEARANCE_M
                and row.get("pin_physical_clearance_m",-np.inf)>=PHYSICAL_CLEARANCE_M
                and max(row.get("cuda_q_ratio",np.inf),row.get("cuda_v_ratio",np.inf),row.get("cuda_u_ratio",np.inf),
                        row.get("pin_q_ratio",np.inf),row.get("pin_v_ratio",np.inf),row.get("pin_u_ratio",np.inf))<=1)


def replay_state_l2_report(arrays):
    cuda=np.asarray(arrays["cuda_dense_state"],dtype=np.float64); pin=np.asarray(arrays["pin_dense_state"],dtype=np.float64)
    return {"max_state_l2_report_only":float(np.max(np.linalg.norm(cuda-pin,axis=1))),
            "state_l2_acceptance_gate":False}


def expected_final_array_names():
    return set(BASE_ARRAY_SPECS)|{f"row_{index:03d}_{name}" for index in range(EXPECTED_PAIR_COUNT) for name in ROW_ARRAY_SPECS}


def exact_final_array_schema(arrays):
    if set(arrays)!=expected_final_array_names(): return False
    for name,(shape,dtype) in BASE_ARRAY_SPECS.items():
        value=np.asarray(arrays[name])
        if value.shape!=shape or value.dtype!=np.dtype(dtype) or not np.isfinite(value).all(): return False
    for index in range(EXPECTED_PAIR_COUNT):
        for name,(shape,dtype) in ROW_ARRAY_SPECS.items():
            value=np.asarray(arrays[f"row_{index:03d}_{name}"])
            if value.shape!=shape or value.dtype!=np.dtype(dtype) or not np.isfinite(value).all(): return False
    return True


def row_arrays_from_final(arrays,index):
    prefix=f"row_{index:03d}_"
    return {name:np.asarray(arrays[prefix+name]) for name in ROW_ARRAY_SPECS}


def derive_replay_metrics(states,tools,controls,reference,lower,upper,velocity,effort,tool_speed_mps):
    state=np.asarray(states,dtype=np.float64); tool=np.asarray(tools,dtype=np.float64); controls=np.asarray(controls,dtype=np.float64)
    ref=toll.TollReference.from_solver_bytes(np.asarray(reference,dtype=np.float32)); lo=np.asarray(lower); hi=np.asarray(upper)
    midpoint=.5*(lo+hi); half=.5*(hi-lo)
    return {"terminal_error_m":float(np.linalg.norm(tool[-1]-np.asarray(ref.goal_xyz))),
            "final_tool_speed_mps":float(tool_speed_mps),
            "physical_clearance_m":float(np.min(np.linalg.norm(tool[:,:2]-np.asarray(ref.cylinder_xy),axis=1)-ref.physical_radius_m)),
            "q_ratio":float(np.max(np.abs((state[:,:7]-midpoint)/half))),
            "v_ratio":float(np.max(np.abs(state[:,7:])/np.asarray(velocity))),
            "u_ratio":float(np.max(np.abs(controls)/np.asarray(effort)))}


def certify_bound_acquisition_history(retained, identity, q8, default_side, problem, *, reenumerate):
    """Bind compact DLS diagnostics to the exact task, callbacks, and seed path."""
    dls={name:np.asarray(retained[name]) for name in ROW_ARRAY_SPECS if name.startswith("dls_")}
    exact_schema=all(dls[name].shape==ROW_ARRAY_SPECS[name][0]
                     and dls[name].dtype==np.dtype(ROW_ARRAY_SPECS[name][1])
                     and np.isfinite(dls[name]).all() for name in dls)
    if not exact_schema or problem.rnea is None:
        return {"exact_schema":exact_schema,"passes":False}
    ref=toll.TollReference.from_solver_bytes(problem.reference)
    requested=identity[2]; margin_index=int(identity[3])
    default_side=int(default_side)
    sign=canonical_mode_sign(default_side,requested)
    start=np.asarray(problem.kinematics(problem.x0[:7])[0],dtype=np.float64)
    template=tangent_arc_tangent_template(start,ref.goal_xyz,ref.cylinder_xy,
                                          TEMPLATE_RADII_M[margin_index],sign)
    q=np.asarray(problem.x0[:7],dtype=np.float64).copy(); seed_q=[q.copy()]
    maxima={name:0. for name in ("q_before","position","jacobian","residual","bounds")}
    recurrence=True
    for knot in range(1,KNOTS-1):
        for iteration in range(DLS_ITERATIONS_PER_KNOT):
            step=(knot-1)*DLS_ITERATIONS_PER_KNOT+iteration
            position,jacobian=problem.kinematics(q); position=np.asarray(position,dtype=np.float64); jacobian=np.asarray(jacobian,dtype=np.float64)
            residual=template[knot]-position
            lower=np.asarray(problem.lower)+DLS_MARGIN_RAD-q; upper=np.asarray(problem.upper)-DLS_MARGIN_RAD-q
            maxima["q_before"]=max(maxima["q_before"],float(np.max(np.abs(dls["dls_q_before"][step]-q))))
            maxima["position"]=max(maxima["position"],float(np.max(np.abs(dls["dls_position"][step]-position))))
            maxima["jacobian"]=max(maxima["jacobian"],float(np.max(np.abs(dls["dls_jacobian"][step]-jacobian))))
            maxima["residual"]=max(maxima["residual"],float(np.max(np.abs(dls["dls_residual"][step]-residual))))
            maxima["bounds"]=max(maxima["bounds"],float(np.max(np.abs(dls["dls_lower"][step]-lower))),
                                 float(np.max(np.abs(dls["dls_upper"][step]-upper))))
            q=q+dls["dls_selected_dq"][step]
        seed_q.append(q.copy())
    seed_q.append(np.asarray(q8,dtype=np.float64).copy()); seed_q=np.stack(seed_q)
    qd,qdd,u=knot_derivatives_and_rnea(seed_q,problem.rnea)
    recurrence=bool(np.array_equal(retained["seed_q_float64"],seed_q)
                    and np.array_equal(retained["seed_qd_float64"],qd)
                    and np.array_equal(retained["seed_qdd_float64"],qdd)
                    and np.array_equal(retained["seed_u_float64"],u)
                    and np.array_equal(retained["seed_primal_float64"],pack_z(np.c_[seed_q,qd],u)))
    template_gate=np.array_equal(retained["template_float64"],template)
    callback_gate=max(maxima.values())==0.
    enumeration=certify_compact_dls_history(dls) if reenumerate else {
        "passes":True,"independent_reenumeration_deferred_to_authoritative_final":True}
    gates={"exact_schema":True,"exact_template":template_gate,"callback_history_bound":callback_gate,
           "seed_recurrence_and_rnea_bound":recurrence,"independent_face_enumeration":enumeration.get("passes") is True}
    return {**gates,"max_callback_differences":maxima,"enumeration":enumeration,
            "passes":bool(all(gates.values()))}


def certify_independent_pin_replays(problem, acquisition_controls, polish_controls,
                                    acquisition_states, acquisition_tools, polish_states, polish_tools):
    if type(problem) is not OriginalOracleNLP or problem.anchored or problem.dense_replay is None:
        return {"exact_unanchored_model":False,"passes":False}
    try:
        acq_state,acq_tool=problem.dense_replay(problem.x0,np.asarray(acquisition_controls,dtype=np.float64))
        pin_state,pin_tool=problem.dense_replay(problem.x0,np.asarray(polish_controls,dtype=np.float64))
        maxima={"acquisition_state":float(np.max(np.abs(np.asarray(acquisition_states)-acq_state))),
                "acquisition_tool":float(np.max(np.abs(np.asarray(acquisition_tools)-acq_tool))),
                "polish_state":float(np.max(np.abs(np.asarray(polish_states)-pin_state))),
                "polish_tool":float(np.max(np.abs(np.asarray(polish_tools)-pin_tool)))}
    except (ValueError,IndexError,TypeError):
        return {"exact_unanchored_model":True,"passes":False}
    finite=all(np.isfinite(value) for value in maxima.values())
    gates={"exact_unanchored_model":True,"finite":finite,
           "acquisition_replay_exact":max(maxima["acquisition_state"],maxima["acquisition_tool"])<=1e-12,
           "polish_replay_exact":max(maxima["polish_state"],maxima["polish_tool"])<=1e-12}
    return {**gates,"maxabs":maxima,"passes":bool(all(gates.values()))}


def kinematic_tool_speed(problem,state):
    value=np.asarray(state,dtype=np.float64)
    if type(problem) is not OriginalOracleNLP or problem.anchored or value.shape!=(NX,): return np.inf
    return float(np.linalg.norm(problem.tool_velocity_derivatives(value[:7],value[7:])[0]))


def certify_worker_row_boundary(worker,retained,public,identity=None):
    expected={
        "captured_x0_float32":np.asarray(public["public_x0_float32"],np.float32),
        "captured_reference_float32":np.asarray(public["public_reference_float32"],np.float32),
        "captured_controls_float32":np.asarray(retained["controls_float64"],np.float32),
        "captured_joint_lower_float64":np.asarray(public["joint_lower_float64"],np.float64),
        "captured_joint_upper_float64":np.asarray(public["joint_upper_float64"],np.float64),
        "captured_velocity_limit_float64":np.asarray(public["velocity_limit_float64"],np.float64),
        "captured_effort_limit_float64":np.asarray(public["effort_limit_float64"],np.float64),
        "cuda_knot_states_float32":np.asarray(retained["cuda_dense_state_float32"])[::DENSE_SUBSTEPS],
        "cuda_dense_states_float32":np.asarray(retained["cuda_dense_state_float32"]),
        "cuda_dense_tool_float32":np.asarray(retained["cuda_dense_tool_float32"]),
        "dense_time_float64":np.asarray(retained["dense_time_float64"]),
    }
    hashes=worker.get("array_hashes",{})
    artifact=worker.get("artifact",{})
    command=worker.get("command",())
    return bool(worker.get("protocol")==WORKER_PROTOCOL_VERSION
                and isinstance(command,list) and command[1:4]==["-B","-m","gato_tiago.multimodal_toll_oracle_v1_worker"]
                and command[4:]==["--request",artifact.get("request_path"),"--output",artifact.get("json_path")]
                and (identity is None or worker.get("identity")==list(identity))
                and worker.get("cwd")==str(AUTHORIZED_CWD) and worker.get("exit_code")==0
                and worker.get("constructor_calls")==2 and worker.get("sim_forward_calls")==DENSE_SAMPLES-1
                and worker.get("tool_position_calls")==((DENSE_SAMPLES+15)//16)
                and worker.get("solve_calls")==worker.get("sqp_calls")==0
                and set(hashes)==set(expected) and all(hashes[name]==array_hash(value) for name,value in expected.items())
                and set(artifact)>={"request_path","request_sha256","input_path","input_sha256","json_path","json_sha256","npz_path","npz_sha256"})


def certify_retained_row(row,index,arrays,problem,*,reenumerate=True):
    """Single canonical row certificate derived from the final NPZ map."""
    identity=tuple(row.get("identity",())); expected=EXPECTED_LEDGER[index]
    task_index=EXPECTED_TASK_IDENTITIES.index(expected[:2]); retained=row_arrays_from_final(arrays,index)
    public={"public_x0_float32":arrays["public_x0_float32"][task_index],
            "public_reference_float32":arrays["public_reference_float32"][task_index],
            "joint_lower_float64":arrays["joint_lower_float64"],"joint_upper_float64":arrays["joint_upper_float64"],
            "velocity_limit_float64":arrays["velocity_limit_float64"],"effort_limit_float64":arrays["effort_limit_float64"],
            "acquisition_primal_float64":retained["acquisition_primal_float64"],
            "solver_options":{**TRUST_CONSTR_OPTIONS,"maxiter":POLISH_MAXITER}}
    binding=certify_polish_problem_binding(problem,public)
    requested=expected[2]; default_side=int(arrays["public_default_side_int8"][task_index])
    dls=certify_bound_acquisition_history(retained,expected,arrays["quarantined_q8_float64"][task_index],
                                          default_side,problem,reenumerate=reenumerate)
    kkt_arrays={name:retained[name] for name in retained if name.startswith("raw_") or name.startswith("canonical_")}
    kkt=certify_primal_kkt(problem,retained["polish_primal_float64"],kkt_arrays)
    reference=public["public_reference_float32"]; ref=toll.TollReference.from_solver_bytes(reference)
    lower=public["joint_lower_float64"]; upper=public["joint_upper_float64"]
    velocity=public["velocity_limit_float64"]; effort=public["effort_limit_float64"]
    acq_x,acq_u=unpack_z(retained["acquisition_primal_float64"]); polish_x,polish_u=unpack_z(retained["polish_primal_float64"])
    computed_acq_tool,_=problem.positions_and_jacobians(acq_x); computed_polish_tool,_=problem.positions_and_jacobians(polish_x)
    acq_eq=problem.equality(retained["acquisition_primal_float64"]); acq_h=problem.inequality(retained["acquisition_primal_float64"])
    bound_lo,bound_hi=problem.bounds_arrays(); acq_primal=max(float(np.max(np.abs(acq_eq))),float(np.max(np.maximum(-acq_h,0))),
                                                               float(np.max(np.maximum(bound_lo-retained["acquisition_primal_float64"],0))),
                                                               float(np.max(np.maximum(retained["acquisition_primal_float64"]-bound_hi,0))))
    template_rms=tool_rms(retained["acquisition_tool_float64"],retained["template_float64"])
    requested_sign=canonical_mode_sign(default_side,requested)
    _acq_label,acq_turn=mode_label(retained["acquisition_tool_float64"],ref.cylinder_xy)
    independent_replays=certify_independent_pin_replays(problem,acq_u,polish_u,
        retained["acquisition_dense_state_float64"],retained["acquisition_dense_tool_float64"],
        retained["pin_dense_state_float64"],retained["pin_dense_tool_float64"])
    pin_speed=kinematic_tool_speed(problem,retained["pin_dense_state_float64"][-1])
    cuda_speed=kinematic_tool_speed(problem,retained["cuda_dense_state_float32"][-1])
    pin_metrics=derive_replay_metrics(retained["pin_dense_state_float64"],retained["pin_dense_tool_float64"],retained["controls_float64"],reference,lower,upper,velocity,effort,pin_speed)
    cuda_metrics=derive_replay_metrics(retained["cuda_dense_state_float32"],retained["cuda_dense_tool_float32"],retained["controls_float64"],reference,lower,upper,velocity,effort,cuda_speed)
    pin_label,pin_turn=mode_label(retained["pin_dense_tool_float64"],ref.cylinder_xy)
    cuda_label,cuda_turn=mode_label(retained["cuda_dense_tool_float32"],ref.cylinder_xy)
    actual_sign=int(np.sign(pin_turn)) if abs(pin_turn)>=OPEN_TURN_RAD and np.sign(pin_turn)==np.sign(cuda_turn) else 0
    actual_side="default" if actual_sign==default_side else "alternate" if actual_sign==-default_side else "invalid"
    acq_cost=problem.objective(retained["acquisition_primal_float64"]); polish_cost=problem.objective(retained["polish_primal_float64"])
    stability_cost=abs(polish_cost-acq_cost)/max(abs(acq_cost),1e-30)
    stability_rms=tool_rms(retained["acquisition_dense_tool_float64"],retained["pin_dense_tool_float64"])
    costs={}; lengths={}
    for model,state,tool in (("pin",retained["pin_dense_state_float64"],retained["pin_dense_tool_float64"]),
                             ("cuda",retained["cuda_dense_state_float32"],retained["cuda_dense_tool_float32"])):
        knot_state=np.asarray(state,dtype=np.float64)[::DENSE_SUBSTEPS]; knot_tool=np.asarray(tool,dtype=np.float64)[::DENSE_SUBSTEPS]
        costs[f"{model}_full_cost"]=reconstruct_objective(knot_state,retained["controls_float64"],knot_tool,reference,lower,upper,velocity,effort,include_toll=True)
        costs[f"{model}_base_cost"]=reconstruct_objective(knot_state,retained["controls_float64"],knot_tool,reference,lower,upper,velocity,effort,include_toll=False)
        lengths[f"{model}_path_length"]=path_length(tool)
    finite=all(np.isfinite(value).all() for value in retained.values())
    raw_success=bool(row.get("acquisition",{}).get("success") is True and row.get("polish",{}).get("success") is True
                     and row.get("acquisition",{}).get("limit_reached") is False and row.get("polish",{}).get("limit_reached") is False)
    gates={"identity":identity==expected,"exact_public_problem_binding":binding,"compact_dls":dls.get("passes") is True,
           "finite":finite,"raw_optimizer_success":raw_success,"acquisition_original_primal":acq_primal<=ACQUISITION_PRIMAL_TOLERANCE,
           "acquisition_intended_mode":abs(acq_turn)>=OPEN_TURN_RAD and np.sign(acq_turn)==requested_sign,
           "acquisition_template_rms":template_rms<=ACQUISITION_TEMPLATE_RMS_M,
           "unanchored_kkt":kkt.get("passes") is True,"intended_stability":actual_side!=requested or (stability_cost<=STABILITY_COST_REL and stability_rms<=STABILITY_TOOL_RMS_M),
           "acquisition_pin_replay_independently_recomputed":independent_replays.get("acquisition_replay_exact") is True,
           "pin_replay_independently_recomputed":independent_replays.get("polish_replay_exact") is True,
           "pin_replay":pin_metrics["terminal_error_m"]<=TERMINAL_PIN_M and pin_metrics["final_tool_speed_mps"]<=FINAL_TOOL_SPEED_MPS and pin_metrics["physical_clearance_m"]>=PHYSICAL_CLEARANCE_M and max(pin_metrics["q_ratio"],pin_metrics["v_ratio"],pin_metrics["u_ratio"])<=1,
           "cuda_replay":cuda_metrics["terminal_error_m"]<=TERMINAL_CUDA_M and cuda_metrics["final_tool_speed_mps"]<=FINAL_TOOL_SPEED_MPS and cuda_metrics["physical_clearance_m"]>=PHYSICAL_CLEARANCE_M and max(cuda_metrics["q_ratio"],cuda_metrics["v_ratio"],cuda_metrics["u_ratio"])<=1,
           "model_tool_agreement":float(np.max(np.linalg.norm(retained["pin_dense_tool_float64"]-retained["cuda_dense_tool_float32"],axis=1)))<=DENSE_TOOL_MODEL_TOLERANCE_M,
           "model_mode_agreement":pin_label==cuda_label and actual_side in SIDES,
           "model_cost_agreement":all(abs(costs[f"pin_{kind}_cost"]-costs[f"cuda_{kind}_cost"])<=MODEL_COST_ABS+MODEL_COST_REL*max(abs(costs[f"pin_{kind}_cost"]),1e-30) for kind in ("full","base")),
           "tool_paths_recomputed":np.allclose(retained["acquisition_tool_float64"],computed_acq_tool,rtol=0,atol=1e-12)
                and np.allclose(retained["polish_tool_float64"],computed_polish_tool,rtol=0,atol=1e-12),
           "seed_path_bound":np.array_equal(retained["seed_q_float64"][0],problem.x0[:7])
                and np.array_equal(retained["seed_q_float64"][-1],arrays["quarantined_q8_float64"][task_index])
                and np.array_equal(retained["seed_primal_float64"],pack_z(np.c_[retained["seed_q_float64"],retained["seed_qd_float64"]],retained["seed_u_float64"])),
           "controls_public_and_common":np.array_equal(retained["controls_float64"],polish_u)
                and np.array_equal(retained["pin_dense_state_float64"][0],problem.x0)
                and np.array_equal(retained["cuda_dense_state_float32"][0],np.asarray(problem.x0,np.float32)),
           "dense_time":np.array_equal(retained["dense_time_float64"],np.arange(DENSE_SAMPLES,dtype=np.float64)*DT/DENSE_SUBSTEPS)}
    gates["worker_request_output_boundary"]=certify_worker_row_boundary(row.get("worker",{}),retained,public,identity)
    derived={"identity":list(identity),"requested_side":requested,"actual_side":actual_side,"margin_index":expected[3],
             "public_default_side":default_side,"cylinder_xy":list(ref.cylinder_xy),"acquisition_primal_inf":acq_primal,
             "acquisition_turn_rad":acq_turn,"template_rms_m":template_rms,"stability_cost_relative":stability_cost,
             "stability_dense_tool_rms_m":stability_rms,"pin_turn_rad":pin_turn,"cuda_turn_rad":cuda_turn,
             "pin_metrics":pin_metrics,"cuda_metrics":cuda_metrics,**costs,**lengths,
             "pin_dense_tool":retained["pin_dense_tool_float64"],"cuda_dense_tool":retained["cuda_dense_tool_float32"],
             "pin_controls":retained["controls_float64"],"cuda_controls":retained["controls_float64"],
             "certificate_pass":bool(all(gates.values())),"certified":bool(all(gates.values()))}
    return {"gates":gates,"dls":dls,"kkt":kkt,"independent_pin_replays":independent_replays,
            "derived":derived,"passes":bool(all(gates.values()))}


def normalized_rms(left,right,scale):
    a=np.asarray(left,dtype=np.float64); b=np.asarray(right,dtype=np.float64)
    s=np.asarray(scale,dtype=np.float64)
    if a.shape!=b.shape or not a.size or s.shape!=(a.shape[-1],) or np.any(s<=0):
        return np.inf
    return float(np.sqrt(np.mean(((a-b)/s)**2)))


def tool_rms(left,right):
    a=np.asarray(left,dtype=np.float64); b=np.asarray(right,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or not a.size:
        return np.inf
    return float(np.sqrt(np.mean(np.sum((a-b)**2,axis=1))))


def certify_oracle_row(row, arrays):
    """Pure per-row certificate; no retained success boolean is trusted."""
    numeric=("acquisition_original_primal_inf","acquisition_template_tool_rms_m",
             "polish_equality_inf","polish_inequality_violation_inf","polish_stationarity_inf",
             "polish_dual_sign_inf","polish_complementarity_inf","stability_cost_relative",
             "stability_dense_tool_rms_m","cuda_path_length","pin_path_length")
    domains=all(np.isfinite(float(row.get(name,np.nan))) and float(row[name])>=0 for name in numeric)
    signed_costs=("cuda_full_cost","pin_full_cost","cuda_base_cost","pin_base_cost",
                  "acquisition_original_full_cost","polish_full_cost")
    domains &= all(np.isfinite(float(row.get(name,np.nan))) for name in signed_costs)
    requested=row.get("requested_side"); actual=row.get("actual_side")
    acq_tool=np.asarray(arrays.get("acquisition_tool",())); template=np.asarray(arrays.get("acquisition_template",()))
    acq_dense=np.asarray(arrays.get("acquisition_dense_tool",())); polish_dense=np.asarray(arrays.get("polish_dense_tool",()))
    acq_shape=acq_tool.shape==template.shape==(KNOTS,3)
    dense_stability_shape=acq_dense.shape==polish_dense.shape==(DENSE_SAMPLES,3)
    computed_template_rms=tool_rms(acq_tool,template)
    try: requested_sign=canonical_mode_sign(row.get("public_default_side"),requested)
    except (TypeError,ValueError): requested_sign=0
    acq_label,acq_turn=mode_label(acq_tool,row.get("cylinder_xy",())) if acq_shape else ("invalid",0.)
    acquisition=bool(row.get("acquisition_finite") is True and row.get("acquisition_success") is True
                     and row.get("acquisition_original_primal_inf",np.inf)<=ACQUISITION_PRIMAL_TOLERANCE
                     and row.get("acquisition_intended_mode") is True and np.sign(acq_turn)==requested_sign
                     and computed_template_rms<=ACQUISITION_TEMPLATE_RMS_M
                     and abs(computed_template_rms-row.get("acquisition_template_tool_rms_m",np.inf))<=1e-12)
    kkt=bool(row.get("polish_finite") is True and row.get("polish_success") is True
             and row.get("polish_limit_reached") is False
             and row.get("polish_equality_inf",np.inf)<=PRIMAL_TOLERANCE
             and row.get("polish_inequality_violation_inf",np.inf)<=PRIMAL_TOLERANCE
             and max(row.get("polish_stationarity_inf",np.inf),row.get("polish_dual_sign_inf",np.inf),
                     row.get("polish_complementarity_inf",np.inf))<=KKT_TOLERANCE)
    computed_stability_rms=tool_rms(acq_dense,polish_dense) if dense_stability_shape else np.inf
    acq_cost=float(row.get("acquisition_original_full_cost",np.nan)); polish_cost=float(row.get("polish_full_cost",np.nan))
    computed_cost_change=abs(polish_cost-acq_cost)/max(abs(acq_cost),1e-30)
    stability=bool(actual!=requested or
                   (computed_cost_change<=STABILITY_COST_REL and computed_stability_rms<=STABILITY_TOOL_RMS_M
                    and abs(computed_cost_change-row.get("stability_cost_relative",np.inf))<=1e-12
                    and abs(computed_stability_rms-row.get("stability_dense_tool_rms_m",np.inf))<=1e-12))
    replay=certify_replay_row(row,arrays)
    cuda_tool=np.asarray(arrays.get("cuda_dense_tool",()))
    pin_tool=np.asarray(arrays.get("pin_dense_tool",()))
    center=np.asarray(row.get("cylinder_xy",()))
    shape_gate=cuda_tool.shape==pin_tool.shape==(DENSE_SAMPLES,3) and center.shape==(2,)
    cuda_label,cuda_turn=mode_label(cuda_tool,center) if shape_gate else ("invalid",0.)
    pin_label,pin_turn=mode_label(pin_tool,center) if shape_gate else ("invalid",0.)
    try: expected_sign=canonical_mode_sign(row.get("public_default_side"),actual)
    except (TypeError,ValueError): expected_sign=0
    topology=bool(expected_sign in (-1,1) and np.sign(cuda_turn)==np.sign(pin_turn)==expected_sign
                  and abs(cuda_turn)>=OPEN_TURN_RAD and abs(pin_turn)>=OPEN_TURN_RAD
                  and cuda_label==pin_label and actual in SIDES)
    try:
        computed_costs={}
        for model in ("cuda","pin"):
            state=np.asarray(arrays[f"{model}_dense_state"])[::DENSE_SUBSTEPS]
            tool=np.asarray(arrays[f"{model}_dense_tool"])[::DENSE_SUBSTEPS]
            for flavor,include in (("full",True),("base",False)):
                computed_costs[f"{model}_{flavor}"]=reconstruct_objective(
                    state,arrays["controls"],tool,arrays["common_reference"],arrays["joint_lower"],
                    arrays["joint_upper"],arrays["velocity_limit"],arrays["effort_limit"],include_toll=include)
    except (KeyError,ValueError,IndexError): computed_costs={}
    reconstructed=bool(len(computed_costs)==4 and all(
        abs(computed_costs[f"{model}_{flavor}"]-row.get(f"{model}_{flavor}_cost",np.inf))<=1e-9
        for model in ("cuda","pin") for flavor in ("full","base")))
    computed_lengths={model:path_length(arrays[f"{model}_dense_tool"]) for model in ("cuda","pin")} if shape_gate else {}
    length_bound=bool(len(computed_lengths)==2 and all(abs(computed_lengths[m]-row.get(f"{m}_path_length",np.inf))<=1e-12 for m in ("cuda","pin")))
    costs=bool(reconstructed and length_bound and abs(row.get("cuda_full_cost",np.inf)-row.get("pin_full_cost",-np.inf))
               <=MODEL_COST_ABS+MODEL_COST_REL*max(abs(row.get("pin_full_cost",np.inf)),1e-30)
               and abs(row.get("cuda_base_cost",np.inf)-row.get("pin_base_cost",-np.inf))
               <=MODEL_COST_ABS+MODEL_COST_REL*max(abs(row.get("pin_base_cost",np.inf)),1e-30))
    gates={"numeric_domains":domains,"acquisition":acquisition,"original_kkt":kkt,
           "intended_match_stability":stability,"dense_replay":replay,
           "open_turn_and_model_label":topology,"independent_objective_reconstruction":reconstructed,
           "path_length_reconstruction":length_bound,"model_cost_agreement":costs}
    return {**gates,"acquisition_turn_rad":acq_turn,"acquisition_label":acq_label,
            "computed_template_tool_rms_m":computed_template_rms,
            "computed_stability_cost_relative":computed_cost_change,
            "computed_stability_dense_tool_rms_m":computed_stability_rms,
            "computed_model_costs":computed_costs,
            "computed_path_lengths":computed_lengths,
            "cuda_turn_rad":cuda_turn,"pin_turn_rad":pin_turn,
            "cuda_label":cuda_label,"pin_label":pin_label,"passes":bool(all(gates.values()))}


def pair_topology(rows, model):
    pairs=[]; passed=True
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            same=left["actual_side"]==right["actual_side"]
            left_path=np.asarray(left.get(f"{model}_dense_tool",()))
            right_path=np.asarray(right.get(f"{model}_dense_tool",()))
            if left_path.shape!=right_path.shape or left_path.ndim!=2 or left_path.shape[1]!=3 or not left_path.size:
                result={"winding":np.nan,"nearest_integer":0,"integer_residual":np.inf,"valid_integer":False}
            else:
                result=toll.closed_loop_winding(left_path,right_path,left["cylinder_xy"])
            expected=0 if same else 1
            gate=result["valid_integer"] and abs(result["nearest_integer"])==expected
            pairs.append({"left":left["identity"],"right":right["identity"],"expected_abs":expected,**result,"passes":gate})
            passed &= gate
    return pairs,bool(passed)


def distinct_mode_gate(rows, model, effort_limit):
    groups={side:[row for row in rows if row.get("actual_side")==side] for side in SIDES}
    detail={}; passed=True
    for side,group in groups.items():
        found=False; pairs=[]
        for i,left in enumerate(group):
            for right in group[i+1:]:
                control=normalized_rms(left[f"{model}_controls"],right[f"{model}_controls"],effort_limit)
                tool=tool_rms(left[f"{model}_dense_tool"],right[f"{model}_dense_tool"])
                gate=control>=DISTINCT_CONTROL_RMS or tool>=DISTINCT_TOOL_RMS_M
                pairs.append({"left":left["identity"],"right":right["identity"],
                              "control_rms":control,"tool_rms_m":tool,"distinct":gate})
                found |= gate
        detail[side]={"pairs":pairs,"passes":len(group)>=2 and found}; passed &= detail[side]["passes"]
    return detail,bool(passed)


def representative(rows, model, side):
    candidates=[row for row in rows if row.get("certified") and row.get("actual_side")==side]
    return min(candidates,key=lambda row:(row[f"{model}_full_cost"],EXPECTED_LEDGER.index(tuple(row["identity"])))) if candidates else None


def certify_task_rows(task_rows, effort_limit):
    expected={(side,index) for side in SIDES for index in range(5)}
    keyed={(row.get("requested_side"),row.get("margin_index")) for row in task_rows}
    certified=[row for row in task_rows if row.get("certificate_pass") is True]
    intended={side:[r for r in certified if r.get("requested_side")==r.get("actual_side")==side] for side in SIDES}
    actual={side:[r for r in certified if r.get("actual_side")==side] for side in SIDES}
    topology={}; within={}; distinct={}; gates={
        "exact_ten_keyed_rows":len(task_rows)==10 and keyed==expected,
        "common_task_identity_and_geometry":len(task_rows)==10
            and len({tuple(r["identity"][:2]) for r in task_rows})==1
            and len({tuple(r["cylinder_xy"]) for r in task_rows})==1
            and len({r["public_default_side"] for r in task_rows})==1,
        "all_calls_attempted":all(r.get("acquisition_attempted") is True and r.get("polish_attempted") is True for r in task_rows),
        "at_least_eight_rows_certified":len(certified)>=8,
        "intended_four_each":all(len(intended[s])>=MIN_INTENDED_MATCHES for s in SIDES),
        "actual_four_each":all(len(actual[s])>=MIN_ACTUAL_MODE_ROWS for s in SIDES),
    }
    for model in ("cuda","pin"):
        pairs,pair_gate=pair_topology(certified,model); topology[model]=pairs
        dist,dist_gate=distinct_mode_gate(certified,model,effort_limit); distinct[model]=dist
        within_model={}; within_gate=True
        for side,group in actual.items():
            costs=np.asarray([r[f"{model}_full_cost"] for r in group],dtype=np.float64)
            spread=(float(np.max(costs)-np.min(costs))/max(float(np.min(np.abs(costs))),1e-30)) if costs.size else np.inf
            pair_rms=[tool_rms(a[f"{model}_dense_tool"],b[f"{model}_dense_tool"])
                      for i,a in enumerate(group) for b in group[i+1:]]
            max_rms=max(pair_rms,default=np.inf)
            passed=len(group)>=MIN_ACTUAL_MODE_ROWS and spread<=WITHIN_MODE_COST_SPREAD_REL and max_rms<=WITHIN_MODE_TOOL_RMS_M
            within_model[side]={"cost_spread_relative":spread,"max_tool_rms_m":max_rms,"passes":passed}
            within_gate &= passed
        within[model]=within_model
        gates[f"all_{model}_topology_pairs"]=pair_gate
        gates[f"{model}_distinct_support"]=dist_gate
        gates[f"{model}_within_mode_stability"]=within_gate
    cuda_classes={(tuple(row["left"]),tuple(row["right"])):(row["expected_abs"],abs(row["nearest_integer"]))
                  for row in topology.get("cuda",())}
    pin_classes={(tuple(row["left"]),tuple(row["right"])):(row["expected_abs"],abs(row["nearest_integer"]))
                 for row in topology.get("pin",())}
    gates["exact_cuda_pin_keyed_topology_classes"]=cuda_classes==pin_classes and len(cuda_classes)==len(certified)*(len(certified)-1)//2
    return {"gates":gates,"topology":topology,"within_mode":within,"distinctness":distinct,
            "passes":bool(all(gates.values()))}


def aggregate_campaign(rows, bootstrap=None, effort_limit=None):
    identities=[tuple(row.get("identity",())) for row in rows]
    task_reports={}; heldout_gaps={"cuda":[],"pin":[]}; all_pass=len(rows)==EXPECTED_PAIR_COUNT and identities==list(EXPECTED_LEDGER)
    for phase,seed in EXPECTED_TASK_IDENTITIES:
        task_rows=[row for row in rows if tuple(row["identity"][:2])==(phase,seed)]
        report=aggregate_task(task_rows,"default")
        structural=None
        if effort_limit is not None:
            structural=certify_task_rows(task_rows,np.asarray(effort_limit,dtype=np.float64))
            report={**report,"structural":structural,"passes":bool(report["passes"] and structural["passes"])}
        model_results={}
        for model in ("cuda","pin"):
            default=representative(task_rows,model,"default"); alternate=representative(task_rows,model,"alternate")
            valid=default is not None and alternate is not None
            if valid:
                relative=(default[f"{model}_full_cost"]-alternate[f"{model}_full_cost"])/abs(alternate[f"{model}_full_cost"])
                valid &= default[f"{model}_path_length"]+BASE_LENGTH_ADVANTAGE_M<=alternate[f"{model}_path_length"]
                valid &= default[f"{model}_base_cost"]<=alternate[f"{model}_base_cost"]+BASE_COST_TOLERANCE
                valid &= relative>=FULL_GAP_REL and default[f"{model}_full_cost"]-alternate[f"{model}_full_cost"]>=FULL_GAP_ABS
                model_results[model]={"default":default["identity"],"alternate":alternate["identity"],"relative_gap":relative,"passes":bool(valid)}
                if phase=="heldout": heldout_gaps[model].append(relative)
            else: model_results[model]={"passes":False}
        agreement=model_results["cuda"].get("default")==model_results["pin"].get("default") and model_results["cuda"].get("alternate")==model_results["pin"].get("alternate")
        task_pass=report["passes"] and all(v["passes"] for v in model_results.values()) and agreement
        task_reports[str(seed)]={"basins":report,"models":model_results,"model_agreement":agreement,"passes":task_pass}
        all_pass &= task_pass
    bootstrap_gate=bool(bootstrap is not None and bootstrap.get("seed")==BOOTSTRAP_SEED
                        and bootstrap.get("resamples")==BOOTSTRAP_RESAMPLES
                        and bootstrap.get("task_order")==list(HELDOUT_TASK_SEEDS)
                        and all(np.array_equal(np.asarray(bootstrap.get("input_gaps",{}).get(model,())),
                                               np.asarray(heldout_gaps[model])) for model in ("cuda","pin"))
                        and all(bootstrap.get("input_gap_hashes",{}).get(model)==array_hash(np.asarray(heldout_gaps[model],dtype=np.float64))
                                for model in ("cuda","pin"))
                        and all(bootstrap.get("lower95_relative",{}).get(model,-np.inf)>BOOTSTRAP_LOWER_REL for model in ("cuda","pin")))
    return {"tasks":task_reports,"heldout_gaps":heldout_gaps,"bootstrap":bootstrap,"bootstrap_pass":bootstrap_gate,
            "all_oracle_gates_pass":bool(all_pass and bootstrap_gate)}


def bootstrap_heldout_gaps(gaps):  # only authorized random call after rows freeze
    paired=np.stack([np.asarray(gaps[model],dtype=np.float64) for model in ("cuda","pin")])
    if paired.shape!=(2,len(HELDOUT_TASK_SEEDS)): raise ValueError("heldout CUDA+Pin task gap shape")
    rng=np.random.default_rng(BOOTSTRAP_SEED); samples=np.empty((2,BOOTSTRAP_RESAMPLES))
    for i in range(BOOTSTRAP_RESAMPLES):
        indices=rng.integers(0,len(HELDOUT_TASK_SEEDS),size=len(HELDOUT_TASK_SEEDS))
        samples[:,i]=np.mean(paired[:,indices],axis=1)
    return {"seed":BOOTSTRAP_SEED,"resamples":BOOTSTRAP_RESAMPLES,
            "task_order":list(HELDOUT_TASK_SEEDS),
            "input_gaps":{model:paired[index].copy() for index,model in enumerate(("cuda","pin"))},
            "input_gap_hashes":{model:array_hash(paired[index]) for index,model in enumerate(("cuda","pin"))},
            "lower95_relative":{model:float(np.quantile(samples[index],0.05)) for index,model in enumerate(("cuda","pin"))},
            "samples":samples}


def certify_retained_bootstrap(heldout_gaps,input_gaps,samples):
    expected_input=np.stack([np.asarray(heldout_gaps[m],dtype=np.float64) for m in ("cuda","pin")])
    supplied_input=np.asarray(input_gaps); supplied_samples=np.asarray(samples)
    if supplied_input.shape!=(2,8) or supplied_samples.shape!=(2,BOOTSTRAP_RESAMPLES) or not np.array_equal(supplied_input,expected_input):
        return {"passes":False}
    regenerated=bootstrap_heldout_gaps({m:expected_input[i] for i,m in enumerate(("cuda","pin"))})
    exact=np.array_equal(supplied_samples,regenerated["samples"])
    lower=regenerated["lower95_relative"]
    return {"samples_exact":exact,"lower95_relative":lower,
            "lower95_gate":all(lower[m]>BOOTSTRAP_LOWER_REL for m in ("cuda","pin")),
            "passes":bool(exact and all(lower[m]>BOOTSTRAP_LOWER_REL for m in ("cuda","pin")))}


def certify_final_oracle(rows, arrays, provenance, problem_factory, *, reenumerate=True):
    identities=[tuple(row.get("identity",())) for row in rows]; schema=exact_final_array_schema(arrays)
    row_certificates=[]; canonical_rows=[]
    if schema and len(rows)==EXPECTED_PAIR_COUNT:
        for index,row in enumerate(rows):
            certificate=certify_retained_row(row,index,arrays,problem_factory(index),reenumerate=reenumerate)
            row_certificates.append(certificate); canonical_rows.append(certificate["derived"])
    pre=aggregate_campaign(canonical_rows,None,effort_limit=arrays.get("effort_limit_float64") if schema else None)
    bootstrap=certify_retained_bootstrap(pre.get("heldout_gaps",{}),arrays.get("bootstrap_input_gaps_float64",()),
                                         arrays.get("bootstrap_samples_float64",())) if schema and len(canonical_rows)==120 else {"passes":False}
    bootstrap_summary=None
    if bootstrap.get("passes"):
        gaps={m:np.asarray(pre["heldout_gaps"][m],dtype=np.float64) for m in ("cuda","pin")}
        bootstrap_summary={"seed":BOOTSTRAP_SEED,"resamples":BOOTSTRAP_RESAMPLES,"task_order":list(HELDOUT_TASK_SEEDS),
                           "input_gaps":gaps,"input_gap_hashes":{m:array_hash(v) for m,v in gaps.items()},
                           "lower95_relative":bootstrap["lower95_relative"]}
    campaign=aggregate_campaign(canonical_rows,bootstrap_summary,effort_limit=arrays.get("effort_limit_float64") if schema else None)
    gates={
        "exact_ordered_120_rows":len(rows)==EXPECTED_PAIR_COUNT and identities==list(EXPECTED_LEDGER),
        "all_acquisition_side_artifacts_bound":all(
            set(row.get("acquisition_side_artifact",()))>={"json_path","json_sha256","npz_path","npz_sha256"}
            for row in rows),
        "all_pair_side_artifacts_bound":all(set(row.get("pair_side_artifact",()))>={"json_path","json_sha256","npz_path","npz_sha256"} for row in rows),
        "all_rows_attempted":all(row.get("acquisition_attempted") is True and row.get("polish_attempted") is True for row in rows),
        "exact_final_array_schema":schema,
        "all_rows_canonically_reconstructed":len(row_certificates)==120 and all(c["passes"]==c["derived"]["certificate_pass"] for c in row_certificates),
        "bootstrap_independently_regenerated":bootstrap.get("passes") is True,
        "campaign":campaign.get("all_oracle_gates_pass") is True,
        "provenance":certify_provenance(provenance),
    }
    public_rows=[{key:value for key,value in row.items() if not isinstance(value,np.ndarray)} for row in canonical_rows]
    public_certificates=[]
    for certificate in row_certificates:
        public_certificates.append({"gates":certificate["gates"],
            "dls":{k:v for k,v in certificate["dls"].items() if not isinstance(v,np.ndarray)},
            "kkt":{k:v for k,v in certificate["kkt"].items() if not isinstance(v,np.ndarray)},
            "independent_pin_replays":certificate["independent_pin_replays"],
            "passes":certificate["passes"]})
    return {"gates":gates,"row_certificates":public_certificates,"canonical_rows":public_rows,
            "bootstrap":bootstrap,"campaign":campaign,"all_oracle_gates_pass":bool(all(gates.values()))}


def _finalize(output, rows, arrays, provenance, certificate):
    output=Path(output); npz=output.with_suffix(".npz"); manifest=output.with_suffix(".manifest.json")
    if certificate.get("all_oracle_gates_pass") is not True:
        raise RuntimeError("oracle certificate failed before final publication")
    _atomic_npz(npz,arrays)
    summary={"incomplete":False,"protocol":ORACLE_PROTOCOL_VERSION,"rows":rows,
             "canonical_rows":certificate["canonical_rows"],
             "provenance":provenance,"certificate":certificate,"all_oracle_gates_pass":True,
             "array_names":sorted(arrays),"array_hashes":{k:array_hash(v) for k,v in arrays.items()},
             "npz_path":str(npz),"npz_sha256":sha256_file(npz)}
    _atomic_json(output,summary)
    side_paths=sorted([*output.parent.glob(f"{output.stem}.partial.[0-9][0-9][0-9][0-9].json"),
                       *output.parent.glob(f"{output.stem}.partial.[0-9][0-9][0-9][0-9].npz"),
                       *output.parent.glob(f"{output.stem}.acquisition.*"),*output.parent.glob(f"{output.stem}.pair.*"),
                       *output.parent.glob(f"{output.stem}.worker.*")])
    side_artifacts=[{"path":str(path),"sha256":sha256_file(path)} for path in side_paths]
    manifest_value={"protocol":ORACLE_PROTOCOL_VERSION,"json_path":str(output),
                    "json_sha256":sha256_file(output),"npz_path":str(npz),
                    "npz_sha256":summary["npz_sha256"],"incomplete":False,
                    "all_oracle_gates_pass":True,
                    "acquisition_side_artifacts":[row["acquisition_side_artifact"] for row in rows],
                    "all_side_artifacts":side_artifacts,"side_artifact_count":len(side_artifacts)}
    _atomic_json(manifest,manifest_value)
    pointer=output.with_name(f"{output.stem}.partial.latest.json")
    _atomic_json(pointer,{"generation":123,"incomplete":False,"superseded_by":str(output),
                          "json_path":str(output),"json_sha256":manifest_value["json_sha256"],
                          "npz_path":str(npz),"npz_sha256":summary["npz_sha256"],
                          "manifest_path":str(manifest),"manifest_sha256":sha256_file(manifest)})
    return summary


def _load_npz_map(path):
    with np.load(path,allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def _recertify_side_chain(summary_path,summary,final_arrays,manifest):
    root=Path(summary_path).parent; stem=Path(summary_path).stem
    side_map={item.get("path"):item.get("sha256") for item in manifest.get("all_side_artifacts",())}
    acquisition_ok=pair_ok=worker_ok=checkpoint_ok=True; worker_diagnostics=[]
    for index,identity in enumerate(EXPECTED_LEDGER):
        prefix=f"row_{index:03d}_"; retained={name:final_arrays[prefix+name] for name in ROW_ARRAY_SPECS}
        row=summary["rows"][index]
        acq_json=root/f"{stem}.acquisition.{index:04d}.json"; acq_npz=root/f"{stem}.acquisition.{index:04d}.npz"
        pair_json=root/f"{stem}.pair.{index:04d}.json"; pair_npz=root/f"{stem}.pair.{index:04d}.npz"
        request_json=root/f"{stem}.worker.{index:04d}.request.json"; input_npz=root/f"{stem}.worker.{index:04d}.input.npz"
        worker_json=root/f"{stem}.worker.{index:04d}.json"; worker_npz=root/f"{stem}.worker.{index:04d}.npz"
        paths=(acq_json,acq_npz,pair_json,pair_npz,request_json,input_npz,worker_json,worker_npz)
        if not all(path.is_file() and side_map.get(str(path))==sha256_file(path) for path in paths):
            acquisition_ok=pair_ok=worker_ok=False; continue
        try:
            acq_summary=json.loads(acq_json.read_text()); acq_arrays=_load_npz_map(acq_npz)
            acq_metadata={"identity","row_index","protocol","npz_path","npz_sha256","array_names","array_hashes",
                          "ledger_pointer_published","oracle_evidence"}
            exact_acq_artifact={"json_path":str(acq_json),"json_sha256":sha256_file(acq_json),
                                "npz_path":str(acq_npz),"npz_sha256":sha256_file(acq_npz)}
            acquisition_ok &= bool(row.get("acquisition_side_artifact")==exact_acq_artifact
                and acq_summary.get("identity")==list(identity) and acq_summary.get("row_index")==index
                and acq_summary.get("protocol")==ORACLE_PROTOCOL_VERSION
                and acq_summary.get("npz_path")==str(acq_npz) and acq_summary.get("npz_sha256")==sha256_file(acq_npz)
                and set(acq_arrays)==ACQUISITION_ARRAY_NAMES
                and set(acq_summary.get("array_names",()))==ACQUISITION_ARRAY_NAMES
                and all(acq_summary.get("array_hashes",{}).get(name)==array_hash(value)
                        and np.array_equal(value,retained[name]) for name,value in acq_arrays.items())
                and all(row.get("acquisition",{}).get(key)==value for key,value in acq_summary.items() if key not in acq_metadata))
            pair_summary=json.loads(pair_json.read_text()); pair_arrays=_load_npz_map(pair_npz)
            exact_pair_artifact={"json_path":str(pair_json),"json_sha256":sha256_file(pair_json),
                                 "npz_path":str(pair_npz),"npz_sha256":sha256_file(pair_npz)}
            pair_ok &= bool(row.get("pair_side_artifact")==exact_pair_artifact
                and pair_summary.get("identity")==list(identity) and pair_summary.get("row_index")==index
                and pair_summary.get("protocol")==ORACLE_PROTOCOL_VERSION
                and pair_summary.get("npz_path")==str(pair_npz) and pair_summary.get("npz_sha256")==sha256_file(pair_npz)
                and set(pair_arrays)==set(ROW_ARRAY_SPECS) and set(pair_summary.get("array_names",()))==set(ROW_ARRAY_SPECS)
                and all(pair_summary.get("array_hashes",{}).get(name)==array_hash(value)
                        and np.array_equal(value,retained[name]) for name,value in pair_arrays.items()))
            request=json.loads(request_json.read_text()); worker_input=_load_npz_map(input_npz)
            worker_summary=json.loads(worker_json.read_text()); worker_arrays=_load_npz_map(worker_npz)
            task_index=EXPECTED_TASK_IDENTITIES.index(identity[:2])
            public={"public_x0_float32":final_arrays["public_x0_float32"][task_index],
                    "public_reference_float32":final_arrays["public_reference_float32"][task_index],
                    "joint_lower_float64":final_arrays["joint_lower_float64"],"joint_upper_float64":final_arrays["joint_upper_float64"],
                    "velocity_limit_float64":final_arrays["velocity_limit_float64"],"effort_limit_float64":final_arrays["effort_limit_float64"]}
            expected_input={"x0_float32":np.asarray(public["public_x0_float32"],np.float32),
                "reference_float32":np.asarray(public["public_reference_float32"],np.float32),
                "controls_float32":np.asarray(retained["controls_float64"],np.float32),
                "joint_lower_float64":public["joint_lower_float64"],"joint_upper_float64":public["joint_upper_float64"],
                "velocity_limit_float64":public["velocity_limit_float64"],"effort_limit_float64":public["effort_limit_float64"]}
            expected_worker={"captured_x0_float32":expected_input["x0_float32"],
                "captured_reference_float32":expected_input["reference_float32"],"captured_controls_float32":expected_input["controls_float32"],
                "captured_joint_lower_float64":expected_input["joint_lower_float64"],"captured_joint_upper_float64":expected_input["joint_upper_float64"],
                "captured_velocity_limit_float64":expected_input["velocity_limit_float64"],"captured_effort_limit_float64":expected_input["effort_limit_float64"],
                "cuda_knot_states_float32":retained["cuda_dense_state_float32"][::DENSE_SUBSTEPS],
                "cuda_dense_states_float32":retained["cuda_dense_state_float32"],"cuda_dense_tool_float32":retained["cuda_dense_tool_float32"],
                "dense_time_float64":retained["dense_time_float64"]}
            row_worker=row.get("worker",{}); artifact=row_worker.get("artifact",{})
            worker_json_bound=all(row_worker.get(key)==value for key,value in worker_summary.items())
            recomputed_worker_certificate=certify_replay(worker_summary,worker_arrays)
            expected_worker_hashes={name:array_hash(value) for name,value in worker_arrays.items()}
            worker_gates={"request":validate_request(request) and request.get("request_json")==str(request_json)
                    and request.get("input_npz_sha256")==sha256_file(input_npz),
                "input":set(worker_input)==set(expected_input) and all(np.array_equal(worker_input[k],v) for k,v in expected_input.items()),
                "summary_schema":set(worker_summary)==EXPECTED_WORKER_SUMMARY_KEYS,
                "summary_paths_hashes":worker_summary.get("request_path")==str(request_json)
                    and worker_summary.get("request_sha256")==sha256_file(request_json)
                    and worker_summary.get("input_npz")==str(input_npz)
                    and worker_summary.get("input_npz_sha256")==sha256_file(input_npz)
                    and worker_summary.get("extension_path")==str(CUDA_EXTENSION_PATH)
                    and worker_summary.get("extension_sha256")==CUDA_EXTENSION_SHA256
                    and worker_summary.get("npz_sha256")==sha256_file(worker_npz),
                "summary_arrays":worker_summary.get("array_names")==sorted(worker_arrays)
                    and worker_summary.get("array_hashes")==expected_worker_hashes,
                "replay":recomputed_worker_certificate.get("passes") is True
                    and worker_summary.get("certificate")==_json_value(recomputed_worker_certificate),
                "output":set(worker_arrays)==set(expected_worker) and all(np.array_equal(worker_arrays[k],v) for k,v in expected_worker.items()),
                "json":worker_json_bound,
                "artifact":artifact==row.get("worker_artifact")=={"request_path":str(request_json),"request_sha256":sha256_file(request_json),
                    "input_path":str(input_npz),"input_sha256":sha256_file(input_npz),"json_path":str(worker_json),
                    "json_sha256":sha256_file(worker_json),"npz_path":str(worker_npz),"npz_sha256":sha256_file(worker_npz)}}
            worker_diagnostics.append({"identity":list(identity),**worker_gates,
                "output_mismatches":[name for name,value in expected_worker.items()
                                     if name not in worker_arrays or not np.array_equal(worker_arrays[name],value)]})
            worker_ok &= all(worker_gates.values())
        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):
            acquisition_ok=pair_ok=worker_ok=False
    end_generation=len(EXPECTED_LEDGER)+3
    final_provenance=summary.get("provenance",{})
    end_provenance_keys=("git_head_at_end","tracked_tree_clean_at_end","full_git_status_at_end",
                         "source_hashes_at_end","extension_sha256_at_end","extension_size_bytes_at_end")
    for generation in range(end_generation+1):
        js=root/f"{stem}.partial.{generation:04d}.json"; npz=root/f"{stem}.partial.{generation:04d}.npz"
        try:
            payload=json.loads(js.read_text()); retained=_load_npz_map(npz)
            completed=0 if generation<3 else min(generation-2,len(EXPECTED_LEDGER))
            stage=("before_prerequisite_loads" if generation==0 else "task_authenticated" if generation==1
                   else "model_authenticated" if generation==2 else "end_provenance" if generation==end_generation else "pair_completed")
            p=payload.get("provenance",{}); expected_counts={"task_artifact_loads":int(generation>=1),
                "model_artifact_loads":int(generation>=2),"acquisition_optimizer_calls":completed,
                "polish_optimizer_calls":completed,"worker_subprocess_calls":completed,"cuda_replay_calls":completed,
                "bootstrap_rng_calls":int(generation==end_generation)}
            expected_provenance=dict(final_provenance)
            expected_provenance.update(expected_counts)
            if generation<end_generation:
                for key in end_provenance_keys: expected_provenance[key]=None
                expected_provenance.pop("all_compact_dls_binding_precheck_pass",None)
            if generation<3: expected_provenance.pop("pin_derivative_diagnostics",None)
            if generation<2: expected_provenance.pop("model_artifact_authentication_pass",None)
            if generation<1: expected_provenance.pop("task_artifact_authentication_pass",None)
            checkpoint_ok &= bool(side_map.get(str(js))==sha256_file(js) and side_map.get(str(npz))==sha256_file(npz)
                and payload.get("incomplete") is True and payload.get("all_oracle_gates_pass") is False
                and payload.get("generation")==generation and payload.get("stage")==stage
                and payload.get("expected_identities")==[list(v) for v in EXPECTED_LEDGER]
                and payload.get("completed_identities")==[list(v) for v in EXPECTED_LEDGER[:completed]]
                and payload.get("completed_count")==completed and payload.get("pending_identities")==[list(v) for v in EXPECTED_LEDGER[completed:]]
                and payload.get("rows")==summary.get("rows",())[:completed]
                and set(retained)=={"completed_count_int64"} and retained["completed_count_int64"].shape==()
                and int(retained["completed_count_int64"].item())==completed
                and p==expected_provenance)
        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError): checkpoint_ok=False
    return {"acquisition_chain":bool(acquisition_ok),"pair_chain":bool(pair_ok),
            "worker_chain":bool(worker_ok),"checkpoint_chain":bool(checkpoint_ok),
            "worker_diagnostics":worker_diagnostics,
            "passes":bool(acquisition_ok and pair_ok and worker_ok and checkpoint_ok)}


def certify_final_document_boundary(summary,manifest,pointer,summary_path,npz_path,manifest_path,recomputed):
    summary_path=Path(summary_path); npz_path=Path(npz_path); manifest_path=Path(manifest_path)
    summary_gate=bool(summary.get("protocol")==ORACLE_PROTOCOL_VERSION and summary.get("incomplete") is False
        and summary.get("all_oracle_gates_pass") is True and summary.get("canonical_rows")==recomputed.get("canonical_rows")
        and summary.get("npz_path")==str(npz_path) and summary.get("npz_sha256")==sha256_file(npz_path))
    manifest_gate=bool(manifest.get("protocol")==ORACLE_PROTOCOL_VERSION and manifest.get("incomplete") is False
        and manifest.get("all_oracle_gates_pass") is True and manifest.get("json_path")==str(summary_path)
        and manifest.get("npz_path")==str(npz_path) and manifest.get("json_sha256")==sha256_file(summary_path)
        and manifest.get("npz_sha256")==sha256_file(npz_path))
    pointer_gate=bool(pointer.get("generation")==123 and pointer.get("incomplete") is False
        and pointer.get("superseded_by")==str(summary_path) and pointer.get("json_path")==str(summary_path)
        and pointer.get("npz_path")==str(npz_path) and pointer.get("manifest_path")==str(manifest_path)
        and pointer.get("json_sha256")==sha256_file(summary_path) and pointer.get("npz_sha256")==sha256_file(npz_path)
        and pointer.get("manifest_sha256")==sha256_file(manifest_path))
    return {"summary":summary_gate,"manifest":manifest_gate,"pointer":pointer_gate,
            "passes":bool(summary_gate and manifest_gate and pointer_gate)}


def _authenticated_recertification_factory(final_arrays):  # pragma: no cover - retained-evidence boundary
    task_summary,task_rows,task_arrays=load_authenticated_task_artifact()
    model_summary,model_arrays=load_authenticated_model_artifact()
    if not cross_bind_prerequisites(task_rows,task_arrays,model_arrays):
        raise RuntimeError("oracle recertification prerequisite cross-bind failed")
    final_bound=bool(np.array_equal(final_arrays["public_x0_float32"],model_arrays["public_solver_x0_float32"])
        and np.array_equal(final_arrays["public_reference_float32"],model_arrays["public_reference_float32"])
        and np.array_equal(final_arrays["public_default_side_int8"],model_arrays["public_default_side_int8"])
        and np.array_equal(final_arrays["quarantined_q8_float64"],model_arrays["quarantined_q8_float64"])
        and np.array_equal(final_arrays["joint_lower_float64"],model_arrays["model_lower_float64"])
        and np.array_equal(final_arrays["joint_upper_float64"],model_arrays["model_upper_float64"])
        and np.array_equal(final_arrays["velocity_limit_float64"],model_arrays["model_velocity_float64"])
        and np.array_equal(final_arrays["effort_limit_float64"],model_arrays["model_effort_float64"]))
    if not final_bound: raise RuntimeError("oracle final arrays do not match authenticated prerequisites")
    pin=PinOracleModel()
    def problem_factory(index):
        task_index=EXPECTED_TASK_IDENTITIES.index(EXPECTED_LEDGER[index][:2])
        return OriginalOracleNLP(model_arrays["public_solver_x0_float32"][task_index],
            model_arrays["public_reference_float32"][task_index],model_arrays["model_lower_float64"],
            model_arrays["model_upper_float64"],model_arrays["model_velocity_float64"],model_arrays["model_effort_float64"],
            pin.kinematics,pin.aba_derivatives,pin.tool_velocity_derivatives,rnea=pin.rnea,dense_replay=pin.dense_replay)
    return problem_factory


def recertify_retained_oracle(summary_path,npz_path,manifest_path,pointer_path):
    try:
        summary=json.loads(Path(summary_path).read_text()); manifest=json.loads(Path(manifest_path).read_text())
        pointer=json.loads(Path(pointer_path).read_text())
        with np.load(npz_path,allow_pickle=False) as archive: arrays={name:archive[name] for name in archive.files}
        problem_factory=_authenticated_recertification_factory(arrays)
        names=bool(exact_final_array_schema(arrays) and type(summary.get("array_names")) is list
                   and summary.get("array_names")==sorted(arrays))
        hashes=bool(names and type(summary.get("array_hashes")) is dict
                    and set(summary["array_hashes"])==set(arrays)
                    and all(summary["array_hashes"][name]==array_hash(value) for name,value in arrays.items()))
        recomputed=certify_final_oracle(summary.get("rows",()),arrays,summary.get("provenance",{}),problem_factory,reenumerate=True)
        detail=_json_value(recomputed)==summary.get("certificate")
        side_items=manifest.get("all_side_artifacts",())
        side=len(side_items)==manifest.get("side_artifact_count")==EXPECTED_SIDE_ARTIFACT_COUNT and all(
            Path(item["path"]).is_file() and sha256_file(item["path"])==item["sha256"] for item in side_items)
        side_map={item["path"]:item["sha256"] for item in side_items}
        row_side_bound=True
        for row in summary.get("rows",()):
            for field in ("acquisition_side_artifact","pair_side_artifact"):
                item=row.get(field,{})
                row_side_bound &= side_map.get(item.get("json_path"))==item.get("json_sha256") and side_map.get(item.get("npz_path"))==item.get("npz_sha256")
            item=row.get("worker",{}).get("artifact",{})
            for stem in ("request","input","json","npz"):
                row_side_bound &= side_map.get(item.get(f"{stem}_path"))==item.get(f"{stem}_sha256")
        row_side_bound &= manifest.get("acquisition_side_artifacts")==[row.get("acquisition_side_artifact") for row in summary.get("rows",())]
        semantic_side=_recertify_side_chain(summary_path,summary,arrays,manifest)
        boundary_detail=certify_final_document_boundary(summary,manifest,pointer,summary_path,npz_path,manifest_path,recomputed)
        boundary=boundary_detail["passes"]
        overall=bool(names and hashes and detail and side and row_side_bound and semantic_side["passes"]
                     and boundary and recomputed.get("all_oracle_gates_pass") is True)
        return {"exact_array_schema":names,"array_hashes":hashes,"certificate_exact":detail,
                "all_side_artifacts":side,"all_row_side_artifacts_bound":row_side_bound,"manifest_pointer_boundary":boundary,
                "document_boundary":boundary_detail,
                "semantic_side_chain":semantic_side,
                "all_disk_recertification_gates_pass":overall}
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,json.JSONDecodeError):
        return {"all_disk_recertification_gates_pass":False}


def _run_pipeline(output, *, provenance, task_loader:Callable, model_loader:Callable,
                  acquisition_solver:Callable, polish_solver:Callable, replay_solver:Callable,
                  finish_provenance:Callable, test_override:bool,
                  final_certifier:Callable|None=None,before_finish:Callable|None=None,
                  monotonic:Callable=time.monotonic,campaign_deadline_setter:Callable|None=None):
    output=Path(output); _no_existing(output); rows=[]; arrays={}
    campaign_start=float(monotonic()); campaign_deadline=campaign_start+CAMPAIGN_WALL_LIMIT_S
    if campaign_deadline_setter is not None: campaign_deadline_setter(campaign_deadline)
    _checkpoint(output,0,"before_prerequisite_loads",rows,arrays,provenance)
    task_summary,task_rows,task_arrays=task_loader();
    if not authenticate_summary(task_summary,task_arrays,kind="task"): raise RuntimeError("task prerequisite failed")
    provenance["task_artifact_authentication_pass"]=True; _checkpoint(output,1,"task_authenticated",rows,arrays,provenance)
    model_summary,model_arrays=model_loader()
    if not authenticate_summary(model_summary,model_arrays,kind="model") or not cross_bind_prerequisites(task_rows,task_arrays,model_arrays): raise RuntimeError("model prerequisite/cross-bind failed")
    provenance["model_artifact_authentication_pass"]=True; _checkpoint(output,2,"model_authenticated",rows,arrays,provenance)
    arrays.update({"public_x0_float32":np.asarray(model_arrays["public_solver_x0_float32"]),
                   "public_reference_float32":np.asarray(model_arrays["public_reference_float32"]),
                   "public_default_side_int8":np.asarray(model_arrays["public_default_side_int8"]),
                   "quarantined_q8_float64":np.asarray(model_arrays["quarantined_q8_float64"]),
                   "joint_lower_float64":np.asarray(model_arrays["model_lower_float64"]),
                   "joint_upper_float64":np.asarray(model_arrays["model_upper_float64"]),
                   "velocity_limit_float64":np.asarray(model_arrays["model_velocity_float64"]),
                   "effort_limit_float64":np.asarray(model_arrays["model_effort_float64"])})
    for index,identity in enumerate(EXPECTED_LEDGER):
        row={"identity":list(identity),"acquisition_attempted":True,"polish_attempted":False,"passes":False}
        try:
            acquisition,acq_arrays=acquisition_solver(identity,task_rows,task_arrays,model_arrays)
            side=publish_acquisition(output,index,identity,acquisition,acq_arrays)
            row.update({"acquisition":acquisition,"acquisition_side_artifact":side})
            prefix=f"row_{index:03d}_"; arrays.update({prefix+k:v for k,v in acq_arrays.items()})
            if acquisition.get("finite") is True:
                public_index=EXPECTED_TASK_IDENTITIES.index(tuple(identity[:2]))
                polish_input={
                    "acquisition_primal_float64":np.asarray(acq_arrays["acquisition_primal_float64"]),
                    "public_x0_float32":np.asarray(model_arrays["public_solver_x0_float32"])[public_index],
                    "public_reference_float32":np.asarray(model_arrays["public_reference_float32"])[public_index],
                    "joint_lower_float64":np.asarray(model_arrays["model_lower_float64"]),
                    "joint_upper_float64":np.asarray(model_arrays["model_upper_float64"]),
                    "velocity_limit_float64":np.asarray(model_arrays["model_velocity_float64"]),
                    "effort_limit_float64":np.asarray(model_arrays["model_effort_float64"]),
                    "solver_options":{**TRUST_CONSTR_OPTIONS,"maxiter":POLISH_MAXITER},
                }
                validate_unanchored_polish_input(polish_input)
                row["polish_attempted"]=True
                polish,polish_arrays=polish_solver(polish_input)
                replay,replay_arrays=replay_solver(identity,polish_arrays,public_index,arrays)
                row.update({"polish":polish,"passes":bool(acquisition.get("passes") and polish.get("passes")),
                            "worker":replay,"worker_artifact":replay.get("artifact"),**polish.get("row_metrics",{})})
                arrays.update({prefix+k:v for k,v in polish_arrays.items()})
                arrays.update({prefix+k:v for k,v in replay_arrays.items()})
                retained_row={name:arrays[prefix+name] for name in ROW_ARRAY_SPECS}
                row["pair_side_artifact"]=publish_pair(output,index,identity,retained_row)
        except BaseException as error:
            row.update({"error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()})
        rows.append(row)
        elapsed=max(0.,float(monotonic())-campaign_start); completed=index+1
        projected=elapsed/completed*EXPECTED_PAIR_COUNT
        rejected=elapsed>=CAMPAIGN_WALL_LIMIT_S or projected>CAMPAIGN_WALL_LIMIT_S
        if rejected:
            watchdog={"completed_count":completed,"elapsed_seconds":elapsed,"projected_total_seconds":projected,
                      "threshold_seconds":CAMPAIGN_WALL_LIMIT_S,
                      "trigger":"total_deadline" if elapsed>=CAMPAIGN_WALL_LIMIT_S else "projected_total"}
            _checkpoint(output,index+3,"runtime_watchdog_rejected",rows,arrays,provenance,watchdog=watchdog)
            raise RuntimeError("oracle V1 runtime watchdog permanently rejected campaign")
        _checkpoint(output,index+3,"pair_completed",rows,arrays,provenance)
    if before_finish is not None: before_finish(rows,arrays,provenance)
    finish_provenance(provenance); _checkpoint(output,123,"end_provenance",rows,arrays,provenance)
    if test_override: return {"rows":rows,"incomplete":True,"all_oracle_gates_pass":False}
    if final_certifier is None: raise RuntimeError("oracle final certifier missing")
    certificate=final_certifier(rows,arrays,provenance)
    return _finalize(output,rows,arrays,provenance,certificate)


def _production_pipeline(output, *, token=None):  # pragma: no cover
    if token is not _PRODUCTION_TOKEN: raise RuntimeError("private oracle pipeline")
    provenance=start_provenance(); dependencies=ProductionOracleDependencies(output,provenance)
    return _run_pipeline(output,provenance=provenance,task_loader=dependencies.task_loader,
        model_loader=dependencies.model_loader,acquisition_solver=dependencies.acquisition,
        polish_solver=dependencies.polish,replay_solver=dependencies.replay,
        before_finish=dependencies.before_finish,finish_provenance=finish_runner_provenance,
        campaign_deadline_setter=dependencies.set_campaign_deadline,
        final_certifier=lambda rows,arrays,p:certify_final_oracle(rows,arrays,p,dependencies.problem_factory,reenumerate=True),
        test_override=False)


def execute_oracle(output, *, authorization=None):
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("oracle V1 execution is blocked")
    if Path(output).resolve()!=ORACLE_OUTPUT_PATH.resolve(): raise RuntimeError("oracle output path not authorized")
    return _production_pipeline(output,token=_PRODUCTION_TOKEN)


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--execute",action="store_true"); parser.add_argument("--output")
    args=parser.parse_args(argv)
    if not args.execute or RUNNER_EXECUTION_AUTHORIZATION is None: raise SystemExit("oracle V1 execution is blocked")
    execute_oracle(args.output,authorization=RUNNER_EXECUTION_AUTHORIZATION)


if __name__=="__main__": main()
