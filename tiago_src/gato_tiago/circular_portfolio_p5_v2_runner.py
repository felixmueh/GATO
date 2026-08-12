"""P5 V2 adapter: identical pilot with canonical prerequisite-authentication JSON."""

from __future__ import annotations

import argparse,json,math
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from gato_tiago import circular_portfolio_p5 as v1_schema
from gato_tiago import circular_portfolio_p5_runner as v1
from gato_tiago import circular_portfolio_p5_v2 as schema


RUNNER_EXECUTION_AUTHORIZATION=None
AUTHORIZED_CWD="/workspace/GATO"
AUTHORIZED_ORIG_ARGV=("python","-B","-m","gato_tiago.circular_portfolio_p5_v2_runner",
    "--execute","--output",str(schema.OUTPUT))
CANONICALIZATION="recursive-builtins-v1"


def canonical_json(value,path="authentication"):
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError(f"nonfinite canonical value at {path}")
        return value
    if isinstance(value,np.generic):return canonical_json(value.item(),path)
    if isinstance(value,np.ndarray):
        return canonical_json(value.tolist(),path)
    if isinstance(value,(list,tuple)):
        return [canonical_json(item,f"{path}[{index}]") for index,item in enumerate(value)]
    if isinstance(value,dict):
        if any(not isinstance(key,str) for key in value):
            raise TypeError(f"non-string canonical key at {path}")
        return {key:canonical_json(item,f"{path}.{key}") for key,item in value.items()}
    raise TypeError(f"unsupported canonical value at {path}: {type(value).__name__}")


def canonical_authentication(value):
    result=canonical_json(value)
    json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)
    return result


@contextmanager
def v2_context():
    from gato_tiago import circular_portfolio_runner as predecessor
    from gato_tiago import circular_portfolio_p5_worker as v1_worker
    old={"schema_output":v1_schema.OUTPUT,"schema_protocol":v1_schema.PROTOCOL,
        "schema_worker_protocol":v1_schema.WORKER_PROTOCOL,"runner_output":v1.OUTPUT,
        "runner_protocol":v1.PROTOCOL,"runner_worker_protocol":v1.WORKER_PROTOCOL,
        "runner_argv":v1.AUTHORIZED_ORIG_ARGV,"runner_auth":v1.RUNNER_EXECUTION_AUTHORIZATION,
        "authenticate":predecessor.authenticate_cpu_prerequisite,"subprocess":v1.subprocess.run,
        "publish_checkpoint":v1.publish_checkpoint,"start_provenance":v1.start_provenance,
        "certify_provenance":v1.certify_provenance,"source_paths":v1.SOURCE_PATHS,
        "worker_protocol":v1_worker.WORKER_PROTOCOL}
    def authenticate():
        detail,arrays=old["authenticate"]()
        return canonical_authentication(detail),arrays
    def translated_run(command,*args,**kwargs):
        command=list(command)
        if "gato_tiago.circular_portfolio_p5_worker" in command:
            command[command.index("gato_tiago.circular_portfolio_p5_worker")]= \
                "gato_tiago.circular_portfolio_p5_v2_worker"
        return old["subprocess"](command,*args,**kwargs)
    def publish_checkpoint(generation,stage,counts,provenance,detail=None):
        if generation==1:
            candidate={"protocol":schema.PROTOCOL,"generation":generation,"stage":stage,
                "incomplete":True,"counts":dict(counts),"provenance":provenance,
                "detail":detail or {}}
            json.dumps(candidate,sort_keys=True,separators=(",",":"),allow_nan=False)
        return old["publish_checkpoint"](generation,stage,counts,provenance,detail)
    def start_provenance():
        value=old["start_provenance"]()
        value["p5_v1_rejected_report"]=schema.P5_V1_REJECTED_REPORT
        return value
    def certify_provenance(value,final):
        if value.get("p5_v1_rejected_report")!=schema.P5_V1_REJECTED_REPORT:return False
        base=dict(value);base.pop("p5_v1_rejected_report",None)
        return old["certify_provenance"](base,final)
    try:
        v1_schema.OUTPUT=schema.OUTPUT;v1_schema.PROTOCOL=schema.PROTOCOL
        v1_schema.WORKER_PROTOCOL=schema.WORKER_PROTOCOL
        v1.OUTPUT=schema.OUTPUT;v1.PROTOCOL=schema.PROTOCOL
        v1.WORKER_PROTOCOL=schema.WORKER_PROTOCOL;v1.AUTHORIZED_ORIG_ARGV=AUTHORIZED_ORIG_ARGV
        v1_worker.WORKER_PROTOCOL=schema.WORKER_PROTOCOL
        v1.RUNNER_EXECUTION_AUTHORIZATION=RUNNER_EXECUTION_AUTHORIZATION
        v1.SOURCE_PATHS=old["source_paths"]+("tiago_src/gato_tiago/circular_portfolio_p5_v2.py",
            "tiago_src/gato_tiago/circular_portfolio_p5_v2_runner.py",
            "tiago_src/gato_tiago/circular_portfolio_p5_v2_worker.py")
        predecessor.authenticate_cpu_prerequisite=authenticate;v1.subprocess.run=translated_run
        v1.publish_checkpoint=publish_checkpoint;v1.start_provenance=start_provenance
        v1.certify_provenance=certify_provenance
        yield
    finally:
        v1_schema.OUTPUT=old["schema_output"];v1_schema.PROTOCOL=old["schema_protocol"]
        v1_schema.WORKER_PROTOCOL=old["schema_worker_protocol"]
        v1.OUTPUT=old["runner_output"];v1.PROTOCOL=old["runner_protocol"]
        v1.WORKER_PROTOCOL=old["runner_worker_protocol"];v1.AUTHORIZED_ORIG_ARGV=old["runner_argv"]
        v1_worker.WORKER_PROTOCOL=old["worker_protocol"]
        v1.RUNNER_EXECUTION_AUTHORIZATION=old["runner_auth"]
        v1.SOURCE_PATHS=old["source_paths"]
        predecessor.authenticate_cpu_prerequisite=old["authenticate"];v1.subprocess.run=old["subprocess"]
        v1.publish_checkpoint=old["publish_checkpoint"];v1.start_provenance=old["start_provenance"]
        v1.certify_provenance=old["certify_provenance"]


def fallback_rejection(error):
    return {"protocol":schema.PROTOCOL,"stage":"launch_serialization_failed",
        "incomplete":True,"error_type":type(error).__name__,"error_message":str(error),
        "classification":"minimal_builtin_terminal_diagnostic","oracle_evidence":False,
        "benchmark_evidence":False,"sqp_evidence":False}


def publish_fallback(error):
    value=canonical_json(fallback_rejection(error),"rejection")
    path=schema.OUTPUT.with_name("p5.rejected.json")
    if not path.exists():
        candidate=Path(str(path)+".candidate")
        candidate.write_text(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False))
        candidate.replace(path)
    return value


def execute(output=schema.OUTPUT,authorization=None,monotonic=None): # pragma: no cover
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P5 V2 runner blocked")
    if Path(output).resolve()!=schema.OUTPUT:raise RuntimeError("wrong P5 V2 output")
    try:
        with v2_context():
            kwargs={} if monotonic is None else {"monotonic":monotonic}
            return v1.execute(schema.OUTPUT,RUNNER_EXECUTION_AUTHORIZATION,**kwargs)
    except Exception as error:
        publish_fallback(error);raise


def recertify_retained_p5(output=schema.OUTPUT,monotonic=None):
    with v2_context():
        kwargs={} if monotonic is None else {"monotonic":monotonic}
        return v1.recertify_retained_p5(schema.OUTPUT,**kwargs)


def recertify_retained_failure(path=None,monotonic=None):
    path=path or schema.OUTPUT.with_name("p5.rejected.json")
    try:
        value=json.loads(Path(path).read_text())
        if value.get("classification")=="minimal_builtin_terminal_diagnostic":
            return {"passes":value==canonical_json(value,"rejection") and set(value)=={
                "protocol","stage","incomplete","error_type","error_message","classification",
                "oracle_evidence","benchmark_evidence","sqp_evidence"}
                and value["protocol"]==schema.PROTOCOL
                and value["stage"]=="launch_serialization_failed"
                and value["incomplete"] is True
                and isinstance(value["error_type"],str) and isinstance(value["error_message"],str)
                and value["oracle_evidence"] is value["benchmark_evidence"] \
                    is value["sqp_evidence"] is False}
    except Exception:return {"passes":False}
    with v2_context():
        kwargs={} if monotonic is None else {"monotonic":monotonic}
        return v1.recertify_retained_failure(path,**kwargs)


def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    execute(args.output,RUNNER_EXECUTION_AUTHORIZATION)


if __name__=="__main__":main()
