"""One-shot cached P9 construction replay; no optimizer, SQP, or protocol campaign."""

from __future__ import annotations

import argparse,hashlib,importlib,json,subprocess,sys,time
from pathlib import Path

import numpy as np

from gato_tiago import circular_portfolio_p8_runner as science
from gato_tiago import circular_portfolio_p8_worker as worker
from gato_tiago.circular_portfolio_p5_v2_runner import canonical_authentication
from gato_tiago.circular_portfolio_runner import authenticate_cpu_prerequisite,_production_pin_context
from gato_tiago.circular_portfolio_p9 import CONSTRUCTION_SPECS,EXTENSION,WORKER_OUTPUT_SPECS


ROOT=Path("/tmp/tiago-circular-cached-replay")
OUTPUT_JSON=ROOT/"result.json"
OUTPUT_NPZ=ROOT/"result.npz"
FAILURE_JSON=ROOT/"failure.json"
P9_INPUT=Path("/tmp/tiago-tool-center-circular-route-feasibility-p9-authorized-once/p9.worker.input.npz")
P9_INPUT_SHA256="ed8927684aae5ccd02ef496678ff218373d36b7bddd6114c72ba70e75b432405"
P9_INPUT_SIZE=1497172
P9_GEN1=Path("/tmp/tiago-tool-center-circular-route-feasibility-p9-authorized-once/p9.gen1.json")
P9_GEN1_SHA256="830022b7685e359e7bf74b1bfbbec0665a35f37a6d0fea60c5f05f670bf8cde7"
P9_GEN1_SIZE=154641
WALL_LIMIT_S=300.
AUTHORIZED_CWD="/workspace/GATO"
AUTHORIZED_ORIG_ARGV=("python","-B","-m","gato_tiago.circular_portfolio_cached_replay",
    "--input",str(P9_INPUT),"--gen1",str(P9_GEN1),"--output-root",str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError("cached replay artifact exists")
    try:
        candidate.write_text(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False))
        candidate.replace(path)
    finally:
        if candidate.exists():candidate.unlink()


def atomic_npz(path,arrays):
    path=Path(path);candidate=Path(str(path)+".candidate")
    if path.exists() or candidate.exists():raise FileExistsError("cached replay artifact exists")
    try:
        with candidate.open("xb") as stream:np.savez(stream,**arrays)
        candidate.replace(path)
    finally:
        if candidate.exists():candidate.unlink()


def source_snapshot():
    repo=Path(__file__).resolve().parents[2]
    run=lambda *args:subprocess.run(["git",*args],cwd=repo,check=True,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
    sources=("tiago_src/gato_tiago/circular_portfolio_cached_replay.py",
        "tiago_src/gato_tiago/circular_portfolio_p5_worker.py",
        "tiago_src/gato_tiago/circular_portfolio_p8.py",
        "tiago_src/gato_tiago/circular_portfolio_p8_runner.py",
        "tiago_src/gato_tiago/circular_portfolio_p8_worker.py")
    extension=worker.frozen_extension();path=Path(extension["path"])
    measured={**extension,"sha256":sha(path),"size":path.stat().st_size}
    return {"head":run("rev-parse","HEAD"),
        "clean":run("status","--porcelain","--untracked-files=no")=="",
        "cwd":str(Path.cwd().resolve()),"argv":list(sys.orig_argv),
        "sources":{name:sha(repo/name) for name in sources},"extension":measured}


def certify_replay_argv(argv):
    return isinstance(argv,list) and tuple(argv)==AUTHORIZED_ORIG_ARGV


def certify_source_history(stored,current):
    """Bind immutable run provenance while allowing the auditor's argv to differ."""
    keys={"head","clean","cwd","argv","sources","extension"}
    try:
        start=stored["start"];end=stored["end"]
        stable_keys=keys-{"argv"}
        return bool(set(stored)=={"start","end"} and set(start)==set(end)==set(current)==keys
            and start==end and certify_replay_argv(start["argv"])
            and start["clean"] is True and start["cwd"]==AUTHORIZED_CWD
            and all(current[key]==start[key] for key in stable_keys)
            and current["clean"] is True and current["cwd"]==AUTHORIZED_CWD)
    except Exception:return False


def load_cached_construction(input_path=P9_INPUT,gen1_path=P9_GEN1):
    input_path=Path(input_path)
    if input_path.resolve()!=P9_INPUT or not input_path.is_file() \
            or input_path.stat().st_size!=P9_INPUT_SIZE or sha(input_path)!=P9_INPUT_SHA256:
        raise RuntimeError("cached P9 worker input pin mismatch")
    if gen1_path is None:raise RuntimeError("cached P9 gen1 pin required")
    gen1_path=Path(gen1_path)
    if gen1_path.resolve()!=P9_GEN1 or not gen1_path.is_file() \
            or gen1_path.stat().st_size!=P9_GEN1_SIZE or sha(gen1_path)!=P9_GEN1_SHA256:
        raise RuntimeError("cached P9 gen1 pin mismatch")
    gen1=json.loads(gen1_path.read_text())
    required={"protocol","generation","stage","incomplete","counts","detail",
        "provenance","operational_limits","evidence"}
    if (set(gen1)!=required or gen1["protocol"]!="tiago_tool_center_circular_route_feasibility_p9_1"
            or gen1["generation"]!=1 or gen1["stage"]!="cpu_authenticated"
            or gen1["incomplete"] is not True or gen1["evidence"] is not False
            or not isinstance(gen1["detail"],dict)
            or gen1["detail"].get("construction",{}).get("passes") is not True):
        raise RuntimeError("cached P9 gen1 retained construction evidence invalid")
    with np.load(input_path,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    if not science.exact_arrays(arrays,CONSTRUCTION_SPECS):
        raise RuntimeError("cached P9 construction schema mismatch")
    return arrays


def run_cached_replay(input_path=P9_INPUT,gen1_path=P9_GEN1,root=ROOT,
                      monotonic=time.monotonic,module_loader=importlib.import_module,
                      cuda_probe=worker.cuda_diagnostics): # pragma: no cover - GPU boundary
    root=Path(root)
    if root.exists():raise FileExistsError("cached replay root exists")
    root.mkdir();start=monotonic();counts=worker.initial_counts();snapshot=None;details={}
    try:
        construction=load_cached_construction(input_path,gen1_path)
        gen1=json.loads(Path(gen1_path).read_text())
        snapshot=source_snapshot()
        if (snapshot["clean"] is not True or snapshot["cwd"]!=AUTHORIZED_CWD
                or not certify_replay_argv(snapshot["argv"])
                or snapshot["extension"]!=worker.frozen_extension()):
            raise RuntimeError("cached replay start provenance invalid")
        module=module_loader(EXTENSION["module"])
        if not worker.certify_module(module):raise RuntimeError("cached replay extension mismatch")
        module_pin=worker.module_measurement(module);cuda=cuda_probe()
        authentication,prerequisite=authenticate_cpu_prerequisite()
        authentication=canonical_authentication(authentication)
        _model,kinematics,rnea,*_=_production_pin_context();counts["pin_model_contexts"]+=1
        arrays=worker.generate(module,construction,rnea,start+WALL_LIMIT_S,monotonic,counts)
        output_certificate=worker.certify_output(arrays,construction)
        campaign=science.certify_science(arrays,construction,prerequisite,kinematics,rnea,
            start+WALL_LIMIT_S,monotonic)
        if monotonic()>start+WALL_LIMIT_S:raise TimeoutError("cached replay wall limit")
        atomic_npz(root/OUTPUT_NPZ.name,arrays)
        details={"output_certificate":output_certificate,"campaign_certificate":campaign,
            "output_npz_path":str(root/OUTPUT_NPZ.name),
            "output_npz_sha256":sha(root/OUTPUT_NPZ.name),"array_names":sorted(arrays),
            "array_hashes":{key:science.array_hash(arrays[key]) for key in sorted(arrays)},
            "extension":module_pin,"cuda_diagnostics":cuda}
        end_snapshot=source_snapshot()
        if end_snapshot!=snapshot:raise RuntimeError("cached replay provenance changed")
        result={"kind":"cached-p9-cuda-replay","input_path":str(P9_INPUT),
            "input_sha256":P9_INPUT_SHA256,"input_size":P9_INPUT_SIZE,
            "gen1_path":str(P9_GEN1),"gen1_sha256":P9_GEN1_SHA256,
            "gen1_size":P9_GEN1_SIZE,
            "retained_construction_acceptance":{"protocol":gen1["protocol"],
                "generation":gen1["generation"],"stage":gen1["stage"],
                "incomplete":gen1["incomplete"],"evidence":gen1["evidence"],
                "construction_passes":gen1["detail"]["construction"]["passes"]},**details,
            "source":{"start":snapshot,"end":end_snapshot},
            "elapsed_s":float(monotonic()-start),"counts":counts,
            "passes":bool(output_certificate and campaign.get("passes") is True),
            "optimizer_calls":0,"solve_calls":0,"sqp_calls":0,"rng_calls":0}
        atomic_json(root/OUTPUT_JSON.name,result);return result
    except Exception as error:
        failure={"kind":"cached-p9-cuda-replay-failure","error_type":type(error).__name__,
            "error_message":str(error),"input_path":str(input_path),
            "input_sha256":sha(input_path) if Path(input_path).is_file() else None,
            "elapsed_s":float(monotonic()-start),"counts":counts,"source":snapshot,
            "details":details,
            "optimizer_calls":0,"solve_calls":0,"sqp_calls":0,"rng_calls":0,
            "evidence":False}
        if not (root/FAILURE_JSON.name).exists():atomic_json(root/FAILURE_JSON.name,failure)
        raise


def recertify_cached_replay(root=ROOT):
    """Read-only independent replay of retained cached output and both certificates."""
    try:
        root=Path(root);result=json.loads((root/OUTPUT_JSON.name).read_text())
        construction=load_cached_construction(P9_INPUT,P9_GEN1)
        with np.load(root/OUTPUT_NPZ.name,allow_pickle=False) as archive:
            arrays={key:archive[key] for key in archive.files}
        authentication,prerequisite=authenticate_cpu_prerequisite()
        canonical_authentication(authentication)
        _model,kinematics,rnea,*_=_production_pin_context()
        output_certificate=worker.certify_output(arrays,construction)
        campaign=science.certify_science(arrays,construction,prerequisite,kinematics,rnea)
        current=source_snapshot()
        gen1=json.loads(P9_GEN1.read_text())
        keys={"kind","input_path","input_sha256","input_size","gen1_path","gen1_sha256","gen1_size",
            "retained_construction_acceptance",
            "output_npz_path","output_npz_sha256","array_names","array_hashes","source",
            "extension","cuda_diagnostics","elapsed_s","counts","output_certificate",
            "campaign_certificate","passes","optimizer_calls","solve_calls","sqp_calls","rng_calls"}
        exact=bool(set(result)==keys and result["kind"]=="cached-p9-cuda-replay"
            and result["input_path"]==str(P9_INPUT) and result["input_sha256"]==P9_INPUT_SHA256
            and result["input_size"]==P9_INPUT_SIZE
            and result["gen1_path"]==str(P9_GEN1) and result["gen1_sha256"]==P9_GEN1_SHA256
            and result["gen1_size"]==P9_GEN1_SIZE
            and result["retained_construction_acceptance"]=={"protocol":gen1["protocol"],
                "generation":1,"stage":"cpu_authenticated","incomplete":True,"evidence":False,
                "construction_passes":True}
            and result["output_npz_path"]==str(root/OUTPUT_NPZ.name)
            and result["output_npz_sha256"]==sha(root/OUTPUT_NPZ.name)
            and result["array_names"]==sorted(arrays)
            and result["array_hashes"]=={key:science.array_hash(arrays[key]) for key in sorted(arrays)}
            and certify_source_history(result["source"],current)
            and result["extension"]==current["extension"]
            and worker.certify_cuda_diagnostics(result["cuda_diagnostics"])
            and result["output_certificate"] is output_certificate
            and result["campaign_certificate"]==campaign
            and result["counts"]==worker.SUCCESS_COUNTS
            and result["passes"] is bool(output_certificate and campaign.get("passes") is True)
            and result["optimizer_calls"]==result["solve_calls"]==result["sqp_calls"]
                ==result["rng_calls"]==0 and np.isfinite(result["elapsed_s"])
            and result["elapsed_s"]>=0)
        return {"output":output_certificate,"campaign":campaign.get("passes") is True,
            "documents":exact,"passes":bool(exact and result["passes"])}
    except Exception:return {"passes":False}


def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--input",type=Path,default=P9_INPUT)
    parser.add_argument("--gen1",type=Path,default=P9_GEN1)
    parser.add_argument("--output-root",type=Path,default=ROOT);args=parser.parse_args(argv)
    if tuple(sys.orig_argv)!=AUTHORIZED_ORIG_ARGV \
            or args.input.resolve()!=P9_INPUT or args.gen1.resolve()!=P9_GEN1 \
            or args.output_root.resolve()!=ROOT:
        raise RuntimeError("cached replay CLI boundary mismatch")
    result=run_cached_replay(args.input,args.gen1,args.output_root)
    if result["passes"] is not True:raise SystemExit(1)


if __name__=="__main__":main()
