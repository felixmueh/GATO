"""Disabled P9 worker adapter; CUDA science is byte-identical to P8."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from gato_tiago import circular_portfolio_p8 as p8_schema
from gato_tiago import circular_portfolio_p8_worker as p8
from gato_tiago import circular_portfolio_p9 as schema


WORKER_EXECUTION_AUTHORIZATION=None
SOURCE_PATHS=tuple(dict.fromkeys((*p8.SOURCE_PATHS,
    "tiago_src/gato_tiago/circular_portfolio_p9.py",
    "tiago_src/gato_tiago/circular_portfolio_p9_worker.py")))
SUCCESS_COUNTS=p8.SUCCESS_COUNTS
generate=p8.generate
certify_output=p8.certify_output


def expected_argv():
    paths=schema.worker_paths()
    return ("python","-B","-m","gato_tiago.circular_portfolio_p9_worker","--execute",
        "--request",str(paths["request"]),"--output",str(paths["json"]))


@contextmanager
def p9_worker_context():
    bindings=((p8_schema,"PROTOCOL",schema.PROTOCOL),(p8_schema,"WORKER_PROTOCOL",schema.WORKER_PROTOCOL),
        (p8_schema,"OUTPUT",schema.OUTPUT),(p8,"PROTOCOL",schema.PROTOCOL),
        (p8,"WORKER_PROTOCOL",schema.WORKER_PROTOCOL),(p8,"OUTPUT",schema.OUTPUT),
        (p8,"WORKER_EXECUTION_AUTHORIZATION",WORKER_EXECUTION_AUTHORIZATION),
        (p8,"SOURCE_PATHS",SOURCE_PATHS),(p8,"worker_paths",schema.worker_paths),
        (p8,"expected_argv",expected_argv))
    old=[(owner,name,getattr(owner,name)) for owner,name,_value in bindings]
    try:
        for owner,name,value in bindings:setattr(owner,name,value)
        yield
    finally:
        for owner,name,value in reversed(old):setattr(owner,name,value)


def execute(request,authorization=None,monotonic=None): # pragma: no cover
    if WORKER_EXECUTION_AUTHORIZATION is None or authorization is not WORKER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P9 worker blocked")
    kwargs={} if monotonic is None else {"monotonic":monotonic}
    with p9_worker_context():return p8.execute(request,authorization,**kwargs)


def certify_summary(summary,arrays,construction):
    with p9_worker_context():return p8.certify_summary(summary,arrays,construction)


def certify_rejection(value):
    with p9_worker_context():return p8.certify_rejection(value)


def main(argv=None): # pragma: no cover
    with p9_worker_context():return p8.main(argv)


if __name__=="__main__":main()
