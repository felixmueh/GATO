"""Capability-blocked orchestration boundary for the two-route N260 P5 pilot."""

from __future__ import annotations

import argparse,time
from pathlib import Path

from gato_tiago.circular_portfolio_p5 import (BUILD_COMMAND,CAMPAIGN_WALL_LIMIT_S,
    EXTENSION,OUTPUT,P4_V2_REJECTED_REPORT,PILOT_IDENTITY,PILOT_ROUTES,PROTOCOL)


RUNNER_EXECUTION_AUTHORIZATION=None


def static_design():
    return {"protocol":PROTOCOL,"output":str(OUTPUT),"pilot_identity":list(PILOT_IDENTITY),
        "routes":[list(value) for value in PILOT_ROUTES],"kp":100.0,"kd":20.0,
        "knots":260,"dt":.0125,"dense_substeps":64,"attempts":2,"retries":0,
        "optimizer_calls":0,"solve_calls":0,"sqp_calls":0,"rng_calls":0,
        "build_command":list(BUILD_COMMAND),"extension":EXTENSION,
        "campaign_wall_limit_s":CAMPAIGN_WALL_LIMIT_S,
        "p4_v2_rejected_report":P4_V2_REJECTED_REPORT,
        "p4_v2_artifact_loads":0,"cuda_evidence":False,"benchmark_evidence":False,
        "sqp_evidence":False}


def certify_static_design(value):
    return bool(value==static_design() and value["attempts"]==2 and value["retries"]==0
        and value["optimizer_calls"]==value["solve_calls"]==value["sqp_calls"]==0
        and value["p4_v2_artifact_loads"]==0 and value["extension"]["sha256"] is None)


def execute(output=OUTPUT,authorization=None,monotonic=time.monotonic): # pragma: no cover
    if RUNNER_EXECUTION_AUTHORIZATION is None or authorization is not RUNNER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P5 runner blocked")
    raise RuntimeError("P5 campaign remains blocked pending build/import-only authentication")


def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    execute(args.output,RUNNER_EXECUTION_AUTHORIZATION)

if __name__=="__main__":main()
