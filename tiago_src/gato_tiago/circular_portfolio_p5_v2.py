"""Isolated P5 V2 recovery contract for prerequisite-auth JSON canonicalization."""

from pathlib import Path

from gato_tiago.circular_portfolio_p5 import *  # noqa: F401,F403


PROTOCOL="tiago_tool_center_constructed_route_portfolio_p5_n260_pilot_v2_1"
WORKER_PROTOCOL=PROTOCOL+"_worker"
OUTPUT=Path("/tmp/tiago-tool-center-constructed-route-portfolio-p5-v2-n260-pilot-authorized-once/p5.json")
P5_V1_REJECTED_REPORT={
    "protocol":"tiago_tool_center_constructed_route_portfolio_p5_n260_pilot_1",
    "classification":"launch_invalid_serialization_failure",
    "stage":"prerequisite_authenticated_before_gen1_serialization",
    "hashes":{"gen0_json":"24acf0830bcc434811c72dbaa3f1ad410c9750c684c6bb33e87249e70be81e5b",
        "latest":"6c8b9e96193b84c04cb9f19b5b7a0db5c9a1da0a94f0cf21c9feac141bf388"},
    "retained_counts":{
        "prerequisite_artifact_loads":0,"prerequisite_independent_recertifications":0,
        "pin_model_contexts":0,"primary_pin_rollouts":0,"independent_pin_replays":0,
        "rnea_calls":0,"primary_aba_calls":0,"independent_aba_calls":0,
        "primary_fk_calls":0,"independent_fk_calls":0,"task_endpoint_fk_calls":0,
        "direction_fk_calls":0,"owned_task_endpoint_fk_calls":0,
        "owned_direction_fk_calls":0,"owned_pin_replays":0,"owned_rnea_calls":0,
        "owned_aba_calls":0,"owned_fk_calls":0,"worker_subprocess_attempts":0,
        "worker_subprocess_successes":0,"b1_constructors":0,"b16_constructors":0,
        "b1_sim_forward_calls":0,"b16_sim_forward_calls":0,"b1_tool_position_calls":0,
        "b16_tool_position_calls":0,"optimizer_calls":0,"solve_calls":0,"sqp_calls":0,
        "rng_calls":0,"task_construction_calls":0},
    "independently_reconstructed_execution":{"prerequisite_authentication_completed":True,
        "prerequisite_artifact_loads":1,"prerequisite_authentication_calls":1,
        "prerequisite_internal_fk_calls":24,"production_pin_contexts":0,"production_routes":0,
        "n260_imports":0,"worker_subprocesses":0,"cuda_calls":0,"optimizer_calls":0,
        "rng_calls":0,"sqp_calls":0},
    "bad_paths":[f"authentication.recertification.final.semantic.geometry_details[{index}].certificate.gates.circle_endpoint_identity"
                 for index in range(12)],
    "rejected_p5_v1_artifact_loads":0,"oracle_evidence":False,
    "benchmark_evidence":False,"sqp_evidence":False}


def worker_paths():
    root=OUTPUT.parent
    return {"request":root/"p5.worker.request.json","input":root/"p5.worker.input.npz",
        "npz":root/"p5.worker.npz","json":root/"p5.worker.json",
        "rejected":root/"p5.worker.rejected.json"}


def route_paths(index):
    if index not in (0,1):raise ValueError("invalid P5 V2 route index")
    return {"json":OUTPUT.with_name(f"p5.route.{index}.json"),
        "npz":OUTPUT.with_name(f"p5.route.{index}.npz")}
