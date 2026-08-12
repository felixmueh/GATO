"""Disabled P9 deadline-only adapter over the audited compact P8 pilot."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

from gato_tiago import circular_portfolio_p8 as p8_schema
from gato_tiago import circular_portfolio_p8_runner as p8
from gato_tiago import circular_portfolio_p9 as schema
from gato_tiago import circular_portfolio_p9_worker as worker


RUNNER_EXECUTION_AUTHORIZATION=None
AUTHORIZED_CWD="/workspace/GATO"
AUTHORIZED_ORIG_ARGV=("python","-B","-m","gato_tiago.circular_portfolio_p9_runner",
    "--execute","--output",str(schema.OUTPUT))
SOURCE_PATHS=tuple(dict.fromkeys((*p8.SOURCE_PATHS,
    "tiago_src/gato_tiago/circular_portfolio_p9.py",
    "tiago_src/gato_tiago/circular_portfolio_p9_runner.py",
    "tiago_src/gato_tiago/circular_portfolio_p9_worker.py")))


@contextmanager
def p9_context():
    old_provenance=p8.provenance;old_certify_provenance=p8.certify_provenance
    authorized_argv=(*AUTHORIZED_ORIG_ARGV[:-1],str(schema.OUTPUT))
    def versioned_provenance(start,end=None):
        value=old_provenance(start,end);value["p8_report"]=schema.P8_REJECTED_REPORT;return value
    def versioned_certify_provenance(value,final):
        if not isinstance(value,dict) or value.get("p8_report")!=schema.P8_REJECTED_REPORT:return False
        retained=dict(value);retained.pop("p8_report")
        return old_certify_provenance(retained,final)
    bindings=((p8_schema,"PROTOCOL",schema.PROTOCOL),(p8_schema,"WORKER_PROTOCOL",schema.WORKER_PROTOCOL),
        (p8_schema,"OUTPUT",schema.OUTPUT),(p8_schema,"CPU_WALL_LIMIT_S",schema.CPU_WALL_LIMIT_S),
        (p8_schema,"RUNNER_WALL_LIMIT_S",schema.RUNNER_WALL_LIMIT_S),
        (p8,"PROTOCOL",schema.PROTOCOL),(p8,"WORKER_PROTOCOL",schema.WORKER_PROTOCOL),
        (p8,"OUTPUT",schema.OUTPUT),(p8,"CPU_WALL_LIMIT_S",schema.CPU_WALL_LIMIT_S),
        (p8,"RUNNER_WALL_LIMIT_S",schema.RUNNER_WALL_LIMIT_S),(p8,"ARTIFACT_PREFIX","p9"),
        (p8,"RUNNER_EXECUTION_AUTHORIZATION",RUNNER_EXECUTION_AUTHORIZATION),
        (p8,"AUTHORIZED_ORIG_ARGV",authorized_argv),(p8,"SOURCE_PATHS",SOURCE_PATHS),
        (p8,"worker_argv",worker.expected_argv),(p8,"worker_paths",schema.worker_paths),
        (p8,"checkpoint_path",schema.checkpoint_path),(p8,"latest_path",schema.latest_path),
        (p8,"rejection_path",schema.rejection_path),(p8,"manifest_path",schema.manifest_path),
        (p8,"provenance",versioned_provenance),(p8,"certify_provenance",versioned_certify_provenance))
    old=[(owner,name,getattr(owner,name)) for owner,name,_value in bindings]
    try:
        with worker.p9_worker_context():
            for owner,name,value in bindings:setattr(owner,name,value)
            yield
    finally:
        for owner,name,value in reversed(old):setattr(owner,name,value)


def execute(output=schema.OUTPUT,authorization=None,monotonic=None): # pragma: no cover
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P9 runner blocked")
    kwargs={} if monotonic is None else {"monotonic":monotonic}
    with p9_context():return p8.execute(output,authorization,**kwargs)


def recertify_retained_p9(output=schema.OUTPUT):
    with p9_context():return p8.recertify_retained_p8(output)


def recertify_retained_failure(output=schema.OUTPUT):
    with p9_context():return p8.recertify_retained_failure(output)


def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    execute(args.output,RUNNER_EXECUTION_AUTHORIZATION)


if __name__=="__main__":main()
