"""Isolated provenance-corrected P4 V2 contract; all science is V1 parity."""

from __future__ import annotations

import hashlib,json,re,subprocess
from pathlib import Path

from gato_tiago.circular_portfolio_p4 import EXTENSION,P3_OUTPUT,P3_PINS

PROTOCOL="tiago_tool_center_constructed_portfolio_cuda_replay_p4_v2_1"
WORKER_PROTOCOL=PROTOCOL+"_worker"
OUTPUT=Path("/tmp/tiago-tool-center-circular-portfolio-p4-v2-cuda-replay-authorized-once/p4.json")
P3_PRODUCER_HEAD="bc7c89da2ebdc3dcf38d38bf0f0380590a022641"
P4_V1_REJECTED_REPORT={"protocol":"tiago_tool_center_constructed_portfolio_cuda_replay_p4_1",
    "classification":"launch_invalid_provenance_only","exit_code":1,
    "completed":0,"pending":12,"trigger_elapsed_s_approx":.04821,
    "observed_finish_elapsed_s_approx":.06231,
    "retained_counts":{"p3_artifact_loads":0,"p3_disk_recertifications":0,
        "p3_independent_pin_replays":0,"prerequisite_artifact_loads":0,
        "pin_model_contexts":0,"worker_subprocess_attempts":0,
        "worker_subprocess_successes":0,"b16_constructors":0,
        "sim_forward_calls":0,"tool_position_calls":0,"solve_calls":0,"sqp_calls":0,
        "optimizer_calls":0,"rng_calls":0,"task_construction_calls":0},
    "rejected_v1_artifact_loads":0,"oracle_evidence":False,"benchmark_evidence":False,
    "hashes":{"gen0_json":"8d1913a2402b5a80ab434b6ce8c555c968df6aa363d69c068648df9ed4122b5e",
        "gen0_pointer":"99439e162bbbb7d823be7b1b894600d7f266b0d432d877bd17348326597842a0",
        "rejection_json":"96b9841a3d42606201463585a9f8b9116178de0e841655ad9da749275524d632",
        "rejection_pointer":"7e1242e1f4d7a636e157b81ade8097c2be55767952ae1fcb254756ff7650a49d"}}

def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())

def worker_paths(task_index):
    if not isinstance(task_index,int) or isinstance(task_index,bool) or not 0<=task_index<12:
        raise ValueError("invalid V2 task index")
    stem=f"p4.worker.{task_index:02d}";root=OUTPUT.parent
    return {"request":root/f"{stem}.request.json","input":root/f"{stem}.input.npz",
        "json":root/f"{stem}.json","npz":root/f"{stem}.npz"}

def producer_bytes(path):
    result=subprocess.run(["git","show",f"{P3_PRODUCER_HEAD}:{path}"],cwd="/workspace/GATO",
        check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return result.stdout

AUTHORIZATION_TRANSITIONS={
    "tiago_src/gato_tiago/circular_portfolio_p3_runner.py":{
        "name":"RUNNER_EXECUTION_AUTHORIZATION","from":b"RUNNER_EXECUTION_AUTHORIZATION=object()",
        "to":b"RUNNER_EXECUTION_AUTHORIZATION=None"},
    "tiago_src/gato_tiago/circular_portfolio_p3_constructor.py":{
        "name":"CONSTRUCTOR_EXECUTION_AUTHORIZATION",
        "from":b"CONSTRUCTOR_EXECUTION_AUTHORIZATION=object()",
        "to":b"CONSTRUCTOR_EXECUTION_AUTHORIZATION=None"}}

def certify_authorization_transition(path,current,producer):
    transition=AUTHORIZATION_TRANSITIONS.get(path)
    if transition is None:return {"path":path,"authorization_transition":False,
        "producer_from_count":0,"current_to_count":0,"passes":current==producer}
    source=transition["from"];target=transition["to"]
    producer_count=producer.count(source);current_count=current.count(target)
    producer_wrong=producer.count(target);current_wrong=current.count(source)
    exact=producer_count==current_count==1 and producer_wrong==current_wrong==0 \
        and producer.replace(source,target,1)==current
    return {"path":path,"authorization_transition":True,"name":transition["name"],
        "from":source.decode(),"to":target.decode(),"producer_from_count":producer_count,
        "producer_to_count":producer_wrong,"current_from_count":current_wrong,
        "current_to_count":current_count,"passes":exact}

def scientific_source_parity():
    from gato_tiago import circular_portfolio_p3_runner as p3
    excluded={"tests/python/test_tiago_circular_portfolio_p3_static.py"}
    token_paths={"tiago_src/gato_tiago/circular_portfolio_p3_runner.py",
        "tiago_src/gato_tiago/circular_portfolio_p3_constructor.py"}
    detail={}
    root=Path("/workspace/GATO")
    for path in p3.SOURCE_PATHS:
        if path in excluded:continue
        current=(root/path).read_bytes();producer=producer_bytes(path)
        allowed=path in token_paths;transition=certify_authorization_transition(path,current,producer)
        detail[path]={"current_sha256":sha_bytes(current),"producer_sha256":sha_bytes(producer),
            "authorization_normalization_allowed":allowed,
            "authorization_transition":transition}
    passes=bool(re.fullmatch(r"[0-9a-f]{40}",P3_PRODUCER_HEAD)) and all(
        item["authorization_transition"]["passes"] and (item["authorization_normalization_allowed"]
        or item["current_sha256"]==item["producer_sha256"]) for item in detail.values())
    return {"producer_head":P3_PRODUCER_HEAD,"excluded_test_only_paths":sorted(excluded),
        "sources":detail,"passes":passes}

def historical_snapshot():
    from gato_tiago import circular_portfolio_p3_runner as p3
    full=subprocess.run(["git","rev-parse",P3_PRODUCER_HEAD],cwd="/workspace/GATO",check=True,
        text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
    return {"head":full,"clean":True,
        "sources":{path:sha_bytes(producer_bytes(path)) for path in p3.SOURCE_PATHS}}

def certify_scientific_source_parity(value):
    return bool(value==scientific_source_parity() and isinstance(value,dict)
        and value.get("passes") is True)

def authenticate_p3_historical(): # pragma: no cover - artifact-bearing V2 runtime only
    paths={"json":P3_OUTPUT,"manifest":P3_OUTPUT.with_name("p3.manifest.json"),
        "pointer":P3_OUTPUT.with_name("p3.partial.latest.json"),
        "gen194":P3_OUTPUT.with_name("p3.partial.gen194.json")}
    measured={key:sha(path) for key,path in paths.items()}
    if measured!=P3_PINS:raise RuntimeError("P3 pin mismatch")
    parity=scientific_source_parity()
    if not parity["passes"]:raise RuntimeError("P3 current scientific source parity failed")
    retained=json.loads(P3_OUTPUT.read_text());snapshot=historical_snapshot()
    provenance=retained.get("provenance",{})
    provenance_gate=(provenance.get("head_start")==provenance.get("head_end")==snapshot["head"]
        and provenance.get("clean_start") is provenance.get("clean_end") is True
        and provenance.get("sources_start")==provenance.get("sources_end")==snapshot["sources"])
    if not provenance_gate:raise RuntimeError("P3 retained producer provenance mismatch")
    from gato_tiago import circular_portfolio_p3_runner as p3
    original=p3.snapshot
    try:
        p3.snapshot=lambda: snapshot
        recert=p3.recertify_retained_p3(P3_OUTPUT)
    finally:p3.snapshot=original
    if recert!={"passes":True}:raise RuntimeError("P3 historical semantic recert failed")
    return {"pins":measured,"producer_snapshot":snapshot,"scientific_source_parity":parity,
        "retained_provenance_exact":True,"recertification":recert,"passes":True},retained

def certify_p3_authentication(value):
    return bool(isinstance(value,dict) and set(value)=={"pins","producer_snapshot",
        "scientific_source_parity","retained_provenance_exact","recertification","passes"}
        and value["pins"]==P3_PINS and value["producer_snapshot"]==historical_snapshot()
        and certify_scientific_source_parity(value["scientific_source_parity"])
        and value["retained_provenance_exact"] is True
        and value["recertification"]=={"passes":True} and value["passes"] is True)
