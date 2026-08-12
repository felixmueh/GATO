"""P4 V2 runner: V1 science with consumer-owned historical P3 authentication."""

from __future__ import annotations

import argparse,contextlib,copy,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np

from gato_tiago import circular_portfolio_p4_runner as _v1
from gato_tiago.circular_portfolio_p4_v2 import (EXTENSION,OUTPUT,P4_V1_REJECTED_REPORT,
    PROTOCOL,WORKER_PROTOCOL,authenticate_p3_historical,certify_p3_authentication,worker_paths)
from gato_tiago.circular_portfolio_p4_v2_worker import (certify_output,
    certify_rejection as certify_worker_rejection)

RUNNER_EXECUTION_AUTHORIZATION=object()
AUTHORIZED_ORIG_ARGV=("python","-B","-m","gato_tiago.circular_portfolio_p4_v2_runner",
    "--execute","--output",str(OUTPUT))
SOURCE_PATHS=tuple(path for path in _v1.SOURCE_PATHS
    if path!="tests/python/test_tiago_circular_portfolio_p4_static.py")+(
    "tiago_src/gato_tiago/circular_portfolio_p4_v2.py",
    "tiago_src/gato_tiago/circular_portfolio_p4_v2_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p4_v2_worker.py",
    "tests/python/test_tiago_circular_portfolio_p4_v2_static.py")
_BASE_START_PROVENANCE=_v1.start_provenance
_BASE_CERTIFY_PROVENANCE=_v1.certify_provenance

def start_provenance():
    value=_BASE_START_PROVENANCE()
    value["p4_v1_rejected_report"]=copy.deepcopy(P4_V1_REJECTED_REPORT)
    return value

def certify_provenance(value,final):
    if not isinstance(value,dict) or value.get("p4_v1_rejected_report")!=P4_V1_REJECTED_REPORT:
        return False
    base=copy.deepcopy(value);base.pop("p4_v1_rejected_report")
    return _BASE_CERTIFY_PROVENANCE(base,final)

def run_worker(task_index,arrays,deadline,monotonic=time.monotonic,execution_counts=None): # pragma: no cover
    paths=worker_paths(task_index)
    if any(path.exists() for path in paths.values()):raise FileExistsError("P4 V2 worker artifact exists")
    _v1.atomic_npz(paths["input"],arrays)
    request={"protocol":WORKER_PROTOCOL,"task_index":task_index,"input_path":str(paths["input"]),
        "input_sha256":_v1.sha(paths["input"]),"output_path":str(paths["json"]),"extension":EXTENSION}
    _v1.atomic_json(paths["request"],request)
    command=["python","-B","-m","gato_tiago.circular_portfolio_p4_v2_worker","--execute",
        "--request",str(paths["request"]),"--output",str(paths["json"])]
    remaining=deadline-monotonic()
    if remaining<=0:raise TimeoutError("P4 V2 campaign wall limit")
    if execution_counts is not None:execution_counts["worker_subprocess_attempts"]+=1
    try:
        result=subprocess.run(command,cwd=_v1.AUTHORIZED_CWD,env=os.environ.copy(),
            timeout=min(_v1.WORKER_WALL_LIMIT_S,remaining))
    except subprocess.TimeoutExpired as source:
        error=TimeoutError("P4 V2 worker wall limit")
        error.worker_execution={"task_index":task_index,"status":"killed_internal_unknown",
            "internal_counts_known":False,"constructor_calls_lower_bound":0,
            "sim_forward_calls_lower_bound":0,"tool_position_calls_lower_bound":0}
        error.trigger_elapsed=min(_v1.WORKER_WALL_LIMIT_S,remaining);raise error from source
    if result.returncode:
        rejected=paths["json"].with_name(paths["json"].stem+".rejected.json");detail=None
        if rejected.is_file():
            detail=json.loads(rejected.read_text())
            if not certify_worker_rejection(detail,task_index):raise RuntimeError("P4 V2 worker rejection invalid")
            if execution_counts is not None:execution_counts.update({
                "b16_constructors":execution_counts["b16_constructors"]+detail["constructor_calls"],
                "sim_forward_calls":execution_counts["sim_forward_calls"]+detail["sim_forward_calls"],
                "tool_position_calls":execution_counts["tool_position_calls"]+detail["tool_position_calls"]})
        error=RuntimeError(f"P4 V2 worker exited {result.returncode}")
        error.worker_execution={"task_index":task_index,
            "status":"known_partial" if detail is not None else "abnormal_exit_internal_unknown",
            "internal_counts_known":detail is not None,
            "constructor_calls_lower_bound":detail["constructor_calls"] if detail is not None else 0,
            "sim_forward_calls_lower_bound":detail["sim_forward_calls"] if detail is not None else 0,
            "tool_position_calls_lower_bound":detail["tool_position_calls"] if detail is not None else 0}
        raise error
    try:
        summary=json.loads(paths["json"].read_text())
        with np.load(paths["npz"],allow_pickle=False) as archive:output={k:archive[k] for k in archive.files}
        if not certify_output(summary,output,request,arrays,task_index)["passes"]:
            raise RuntimeError("P4 V2 worker boundary failed")
    except Exception as source:
        error=RuntimeError("P4 V2 exit-zero worker output unavailable or invalid")
        error.worker_execution={"task_index":task_index,"status":"exit0_internal_unknown",
            "internal_counts_known":False,"constructor_calls_lower_bound":0,
            "sim_forward_calls_lower_bound":0,"tool_position_calls_lower_bound":0}
        raise error from source
    if execution_counts is not None:execution_counts.update({
        "worker_subprocess_successes":execution_counts["worker_subprocess_successes"]+1,
        "b16_constructors":execution_counts["b16_constructors"]+1,
        "sim_forward_calls":execution_counts["sim_forward_calls"]+6080,
        "tool_position_calls":execution_counts["tool_position_calls"]+6081})
    return summary,output,request

@contextlib.contextmanager
def v2_runner_scope():
    names={"OUTPUT":OUTPUT,"PROTOCOL":PROTOCOL,"WORKER_PROTOCOL":WORKER_PROTOCOL,
        "worker_paths":worker_paths,"authenticate_p3":authenticate_p3_historical,
        "certify_p3_authentication":certify_p3_authentication,"SOURCE_PATHS":SOURCE_PATHS,
        "AUTHORIZED_ORIG_ARGV":AUTHORIZED_ORIG_ARGV,
        "RUNNER_EXECUTION_AUTHORIZATION":RUNNER_EXECUTION_AUTHORIZATION,
        "run_worker":run_worker,"certify_output":certify_output,
        "certify_worker_rejection":certify_worker_rejection,
        "start_provenance":start_provenance,"certify_provenance":certify_provenance}
    original={key:getattr(_v1,key) for key in names}
    try:
        for key,value in names.items():setattr(_v1,key,value)
        yield
    finally:
        for key,value in original.items():setattr(_v1,key,value)

def execute(output=OUTPUT,authorization=None,monotonic=time.monotonic): # pragma: no cover
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P4 V2 runner blocked")
    with v2_runner_scope():
        return _v1.execute(output,RUNNER_EXECUTION_AUTHORIZATION,monotonic)

def recertify_retained_p4_v2(output=OUTPUT): # pragma: no cover
    with v2_runner_scope():return _v1.recertify_retained_p4(output)

def recertify_retained_failure(output=OUTPUT): # pragma: no cover
    with v2_runner_scope():return _v1.recertify_retained_failure(output)

def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    execute(args.output,RUNNER_EXECUTION_AUTHORIZATION)
if __name__=="__main__":main()
