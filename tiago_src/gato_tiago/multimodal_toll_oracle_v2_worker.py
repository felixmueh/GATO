"""Isolated CUDA replay worker boundary for full Oracle V2."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

from gato_tiago import multimodal_toll_oracle_v1_worker as _v1
from gato_tiago.multimodal_toll_oracle_v2_full import ORACLE_OUTPUT_PATH


WORKER_EXECUTION_AUTHORIZATION = None
WORKER_PROTOCOL_VERSION = "tiago_tool_center_toll_oracle_v2_worker_1"
AUTHORIZED_ROOT = ORACLE_OUTPUT_PATH.parent
EXPECTED_REPLAY_ARRAY_NAMES = _v1.EXPECTED_REPLAY_ARRAY_NAMES
EXPECTED_WORKER_SUMMARY_KEYS = _v1.EXPECTED_WORKER_SUMMARY_KEYS
SIM_FORWARD_CALLS = _v1.SIM_FORWARD_CALLS
TOOL_POSITION_CALLS = _v1.TOOL_POSITION_CALLS
array_hash = _v1.array_hash


@contextmanager
def _v2_worker_context():
    replacements = {
        "ORACLE_OUTPUT_PATH": ORACLE_OUTPUT_PATH,
        "AUTHORIZED_ROOT": AUTHORIZED_ROOT,
        "WORKER_PROTOCOL_VERSION": WORKER_PROTOCOL_VERSION,
    }
    previous = {name: getattr(_v1, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(_v1, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_v1, name, value)


def validate_request(request):
    with _v2_worker_context():
        return _v1.validate_request(request)


def certify_replay(summary, arrays):
    with _v2_worker_context():
        return _v1.certify_replay(summary, arrays)


def _run_authorized_worker(request_path, output):  # pragma: no cover
    with _v2_worker_context():
        return _v1._run_authorized_worker(request_path, output)


def execute_worker(request, output, authorization=None):  # pragma: no cover
    if (
        WORKER_EXECUTION_AUTHORIZATION is None
        or authorization is not WORKER_EXECUTION_AUTHORIZATION
    ):
        raise RuntimeError("Oracle V2 replay worker execution is blocked")
    if Path(output).parent.resolve() != AUTHORIZED_ROOT.resolve():
        raise RuntimeError("Oracle V2 worker output root mismatch")
    return _run_authorized_worker(request, output)


def main(argv=None):  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if WORKER_EXECUTION_AUTHORIZATION is None:
        raise SystemExit("Oracle V2 replay worker execution is blocked")
    execute_worker(
        args.request,
        args.output,
        authorization=WORKER_EXECUTION_AUTHORIZATION,
    )


if __name__ == "__main__":
    main()
