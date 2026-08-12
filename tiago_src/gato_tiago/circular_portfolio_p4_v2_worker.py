"""Isolated P4 V2 worker adapter; byte-identical replay science to accepted V1."""

from __future__ import annotations

import argparse,contextlib
from pathlib import Path

from gato_tiago import circular_portfolio_p4_worker as _v1
from gato_tiago.circular_portfolio_p4_v2 import WORKER_PROTOCOL,worker_paths

WORKER_EXECUTION_AUTHORIZATION=None
SOURCE_PATHS=tuple(path for path in _v1.SOURCE_PATHS
    if path!="tiago_src/gato_tiago/circular_portfolio_p4_worker.py")+(
    "tiago_src/gato_tiago/circular_portfolio_p4_worker.py",
    "tiago_src/gato_tiago/circular_portfolio_p4_v2.py",
    "tiago_src/gato_tiago/circular_portfolio_p4_v2_worker.py")

def expected_argv(task_index):
    paths=worker_paths(task_index)
    return ("python","-B","-m","gato_tiago.circular_portfolio_p4_v2_worker","--execute",
        "--request",str(paths["request"]),"--output",str(paths["json"]))

@contextlib.contextmanager
def v2_worker_scope():
    names={"WORKER_PROTOCOL":WORKER_PROTOCOL,"worker_paths":worker_paths,
        "expected_argv":expected_argv,"SOURCE_PATHS":SOURCE_PATHS,
        "WORKER_EXECUTION_AUTHORIZATION":WORKER_EXECUTION_AUTHORIZATION}
    original={key:getattr(_v1,key) for key in names}
    try:
        for key,value in names.items():setattr(_v1,key,value)
        yield
    finally:
        for key,value in original.items():setattr(_v1,key,value)

def certify_output(summary,arrays,request,inputs,task_index):
    with v2_worker_scope():return _v1.certify_output(summary,arrays,request,inputs,task_index)

def certify_rejection(value,task_index):
    with v2_worker_scope():return _v1.certify_rejection(value,task_index)

def execute_worker(request,output,authorization=None,monotonic=None): # pragma: no cover
    if WORKER_EXECUTION_AUTHORIZATION is None or authorization is not WORKER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P4 V2 worker blocked")
    with v2_worker_scope():
        kwargs={} if monotonic is None else {"monotonic":monotonic}
        return _v1.execute_worker(request,output,WORKER_EXECUTION_AUTHORIZATION,**kwargs)

def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--request",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    execute_worker(args.request,args.output,WORKER_EXECUTION_AUTHORIZATION)
if __name__=="__main__":main()
