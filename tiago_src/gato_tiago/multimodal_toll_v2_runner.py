"""Fail-closed declaration for the Tiago toll V2 construction preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gato_tiago.multimodal_toll_v2 import (
    EXPECTED_TASK_IDENTITIES,
    REQUIRED_SOURCE_PATHS,
    V2_OUTPUT_PATH,
    frozen_v2_metadata,
)


RUNNER_EXECUTION_AUTHORIZATION = None
AUTHORIZED_OUTPUT_PATH = Path(V2_OUTPUT_PATH)
RUNNER_PROTOCOL_VERSION = "tiago_tool_center_toll_v2_runner_declaration_1"


def describe_v2_runner() -> dict:
    return {
        "protocol_version": RUNNER_PROTOCOL_VERSION,
        "authorization": RUNNER_EXECUTION_AUTHORIZATION,
        "authorized_output_path": str(AUTHORIZED_OUTPUT_PATH),
        "expected_identities": [list(row) for row in EXPECTED_TASK_IDENTITIES],
        "schema": frozen_v2_metadata(),
        "required_source_paths": dict(REQUIRED_SOURCE_PATHS),
        "task_instantiation_calls": 0,
        "model_calls": 0,
        "rng_calls": 0,
        "cuda_calls": 0,
        "sqp_calls": 0,
        "execution_implemented": False,
    }


def execute_v2_runner(output, *, authorization=None):
    del output, authorization
    raise RuntimeError("Stage V2 runner execution is blocked")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.describe and not args.execute:
        print(json.dumps(describe_v2_runner(), indent=2, sort_keys=True))
        return 0
    raise SystemExit("Stage V2 task/model execution is blocked")


if __name__ == "__main__":
    main()
