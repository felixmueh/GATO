"""Isolated capability-blocked worker adapter for P5 V2."""

from __future__ import annotations

from contextlib import contextmanager

from gato_tiago import circular_portfolio_p5 as v1_schema
from gato_tiago import circular_portfolio_p5_worker as v1
from gato_tiago import circular_portfolio_p5_v2 as schema


WORKER_EXECUTION_AUTHORIZATION=None


@contextmanager
def v2_context():
    old=(v1_schema.OUTPUT,v1_schema.PROTOCOL,v1_schema.WORKER_PROTOCOL,
         v1.WORKER_PROTOCOL,v1.WORKER_EXECUTION_AUTHORIZATION,v1.expected_argv,v1.SOURCE_PATHS)
    def expected_argv():
        paths=schema.worker_paths()
        return ("python","-B","-m","gato_tiago.circular_portfolio_p5_v2_worker","--execute",
            "--request",str(paths["request"]),"--output",str(paths["json"]))
    try:
        v1_schema.OUTPUT=schema.OUTPUT;v1_schema.PROTOCOL=schema.PROTOCOL
        v1_schema.WORKER_PROTOCOL=schema.WORKER_PROTOCOL;v1.WORKER_PROTOCOL=schema.WORKER_PROTOCOL
        v1.WORKER_EXECUTION_AUTHORIZATION=WORKER_EXECUTION_AUTHORIZATION
        v1.expected_argv=expected_argv;v1.SOURCE_PATHS=old[6]+(
            "tiago_src/gato_tiago/circular_portfolio_p5_v2.py",
            "tiago_src/gato_tiago/circular_portfolio_p5_v2_worker.py")
        yield
    finally:
        (v1_schema.OUTPUT,v1_schema.PROTOCOL,v1_schema.WORKER_PROTOCOL,
         v1.WORKER_PROTOCOL,v1.WORKER_EXECUTION_AUTHORIZATION,
         v1.expected_argv,v1.SOURCE_PATHS)=old


def execute_worker(request,authorization=None,monotonic=None,execution_state=None): # pragma: no cover
    if WORKER_EXECUTION_AUTHORIZATION is None or authorization is not WORKER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P5 V2 worker blocked")
    with v2_context():
        kwargs={"execution_state":execution_state}
        if monotonic is not None:kwargs["monotonic"]=monotonic
        return v1.execute_worker(request,WORKER_EXECUTION_AUTHORIZATION,**kwargs)


def main(argv=None): # pragma: no cover
    with v2_context():
        return v1.main(argv)


if __name__=="__main__":main()
