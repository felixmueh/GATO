#!/usr/bin/env python3
"""Fail-closed entry point for Tiago multimodal-toll SQP and timing stages."""

from __future__ import annotations

import argparse
import json

from gato_tiago.multimodal_toll import frozen_protocol_metadata


BENCHMARK_EXECUTION_AUTHORIZATION = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="print frozen static metadata")
    parser.add_argument("--execute", action="store_true", help="reserved for a later audited stage")
    parser.add_argument("--output", help="reserved output path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.describe and not args.execute:
        print(json.dumps(frozen_protocol_metadata(), indent=2, sort_keys=True))
        return 0
    if BENCHMARK_EXECUTION_AUTHORIZATION is None:
        raise SystemExit("GATO SQP, quality, and timing are blocked pending verifier authorization")
    raise SystemExit("benchmark runner is not implemented in the static checkpoint")


if __name__ == "__main__":
    main()
