"""Frozen schema for the isolated P9 operational-deadline recovery."""

from __future__ import annotations

from pathlib import Path

from gato_tiago import circular_portfolio_p8 as p8


PROTOCOL="tiago_tool_center_circular_route_feasibility_p9_1"
WORKER_PROTOCOL=PROTOCOL+"_worker"
OUTPUT=Path("/tmp/tiago-tool-center-circular-route-feasibility-p9-authorized-once/p9.json")
CPU_WALL_LIMIT_S=1500.
RUNNER_WALL_LIMIT_S=2100.
WORKER_WALL_LIMIT_S=300.
RUNNER_EXECUTION_AUTHORIZATION=None
WORKER_EXECUTION_AUTHORIZATION=None

_P8_COUNT_KEYS=("prerequisite_loads","parent_pin_contexts","primary_cpu_endpoint_fk_calls",
    "primary_dls_kinematics_calls","primary_cpu_proxy_fk_calls",
    "independent_cpu_endpoint_fk_calls","independent_dls_kinematics_calls",
    "independent_cpu_proxy_fk_calls","parent_rnea_recert_calls",
    "parent_identical_state_fk_calls","parent_affine_fk_calls","worker_attempts",
    "worker_successes","worker_pin_contexts","b1_constructors","b16_constructors",
    "worker_rnea_calls","b1_sim_forward_calls","b16_sim_forward_calls",
    "b1_tool_position_calls","b16_tool_position_calls","optimizer_calls","solve_calls",
    "sqp_calls","rng_calls","rejected_p6_artifact_loads","science_routes_attempted",
    "science_routes_completed","completed")
P8_RETAINED_COUNTS={key:0 for key in _P8_COUNT_KEYS}
P8_RETAINED_COUNTS.update(prerequisite_loads=1,parent_pin_contexts=1,
    primary_cpu_endpoint_fk_calls=2,primary_dls_kinematics_calls=214)

# P9 never opens the closed P8 namespace.  Hashes, sizes, and retained timing
# are exact; this is operational-only evidence and contains no CUDA result.
P8_REJECTED_REPORT={
    "protocol":p8.PROTOCOL,"classification":"operational_cpu_preflight_timeout_only",
    "stage":"runtime_watchdog_rejected","completed":0,"pending":2,
    "trigger_elapsed_s":30.06871920998674,"cleanup_finish_elapsed_s":30.069459119986277,
    "hashes":{"gen0":"c25c29a8534e0e5e7a6e198fa48a7a2b166d85eb3cc3c23d70d02c641fa04bc1",
        "latest":"8c9c349cf4d56c7eea688c5dc28bf8543243a20d23062ef31fac907b9b76ea64",
        "rejection":"44e4d7bea1ac60dc5ac561bab838293505bebfc3b15fb3403e535d0a4b2d849e"},
    "sizes":{"gen0":10635,"latest":264,"rejection":4910},
    "nonzero_counts":{"prerequisite_loads":1,"parent_pin_contexts":1,
        "primary_cpu_endpoint_fk_calls":2,"primary_dls_kinematics_calls":214},
    "retained_counts":P8_RETAINED_COUNTS,"all_unlisted_counts_zero":True,
    "worker_execution":{"status":"not_started","returncode":None,
        "internal_counts_known":False,"counts":None},
    "worker_cuda_calls":0,"optimizer_calls":0,"rng_calls":0,"sqp_calls":0,
    "rejected_p8_artifact_loads":0,"oracle_evidence":False,
    "benchmark_evidence":False,"sqp_evidence":False}

# Science and array schemas are references to the frozen P8 implementation.
CONSTRUCTION_SPECS=p8.CONSTRUCTION_SPECS
WORKER_OUTPUT_SPECS=p8.WORKER_OUTPUT_SPECS
EXTENSION=p8.EXTENSION
KNOTS=p8.KNOTS;INTERVALS=p8.INTERVALS;DT=p8.DT;LANES=p8.LANES
AFFINE_SUBSTEPS=p8.AFFINE_SUBSTEPS;AFFINE_SAMPLES=p8.AFFINE_SAMPLES
IK_ITERATIONS=p8.IK_ITERATIONS;DLS_DAMPING=p8.DLS_DAMPING;KP=p8.KP;KD=p8.KD
array_hash=p8.array_hash;exact_arrays=p8.exact_arrays;pack_seed=p8.pack_seed
reconstruct_cost=p8.reconstruct_cost;open_turn=p8.open_turn


def worker_paths():
    root=OUTPUT.parent
    return {"request":root/"p9.worker.request.json","input":root/"p9.worker.input.npz",
        "json":root/"p9.worker.json","npz":root/"p9.worker.npz",
        "rejected":root/"p9.worker.rejected.json"}


def checkpoint_path(generation):return OUTPUT.with_name(f"p9.gen{generation}.json")
def latest_path():return OUTPUT.with_name("p9.partial.latest.json")
def rejection_path():return OUTPUT.with_name("p9.rejected.json")
def manifest_path():return OUTPUT.with_name("p9.manifest.json")


def static_design():
    value=dict(p8.static_design())
    value.update(protocol=PROTOCOL,cpu_wall_limit_s=CPU_WALL_LIMIT_S,
        campaign_wall_limit_s=RUNNER_WALL_LIMIT_S,worker_wall_limit_s=WORKER_WALL_LIMIT_S,
        p8_report=P8_REJECTED_REPORT,rejected_p8_artifact_loads=0)
    return value
