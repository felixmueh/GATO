import copy,inspect,re
from pathlib import Path

import numpy as np
import pytest

from gato_tiago import circular_portfolio_p4_v2_runner as p4_runner
from gato_tiago import circular_portfolio_p4_v2_worker as p4_worker
from gato_tiago import circular_portfolio_p5 as schema
from gato_tiago import circular_portfolio_p5_runner as runner
from gato_tiago import circular_portfolio_p5_worker as worker


def _inputs():
    q=np.zeros((schema.KNOTS,7),np.float32);qd=np.zeros_like(q)
    controls=np.zeros((schema.LANES,schema.INTERVALS,7),np.float32)
    controls[1,:,0]=.01
    seeds=np.stack([schema.pack_seed(q,qd,controls[lane]) for lane in range(schema.LANES)])
    return {"x0_float32":np.zeros((schema.LANES,14),np.float32),
        "controls_float32":controls,"b1_seed_xu_float32":seeds[0],
        "b16_seed_xu_float32":seeds}


def test_p5_isolated_n260_build_wiring_and_all_tokens_closed():
    root=Path(__file__).resolve().parents[2]
    cmake=(root/"CMakeLists.txt").read_text();binding=(root/"python/bindings.cu").read_text()
    assert 'plant STREQUAL "tiago_right_constructed_route_portfolio_toll"' in cmake
    assert "TIAGO_CONSTRUCTED_ROUTE_PORTFOLIO_TOLL=1" in cmake
    assert "PLANT_SUFFIX tiago_right_constructed_route_portfolio_toll" in binding
    assert "PLANT_SUFFIX tiago_right_circular_portfolio_toll" in binding
    assert schema.BUILD_COMMAND==("./tools/build.sh","--plant",
        "tiago_right_constructed_route_portfolio_toll","--knots","260","--target",
        "bsqpN260_tiago_right_constructed_route_portfolio_toll","--native-cuda-arch")
    assert schema.static_extension_declaration()
    assert p4_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert p4_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is None
    pattern=re.compile(r"^[A-Z][A-Z0-9_]*AUTHORIZATION\s*=\s*object\(\)$")
    assert not [(path.name,line) for path in (root/"tiago_src/gato_tiago").glob("*.py")
        for line in path.read_text().splitlines() if pattern.fullmatch(line)]


def test_p4_v2_closure_is_exact_report_only_and_zero_load():
    report=schema.P4_V2_REJECTED_REPORT
    assert report["classification"]=="scientific_replay_rejected_and_closed"
    assert report["completed"]==0 and report["pending"]==12
    assert report["rejected_p4_v2_artifact_loads"]==0
    assert report["call_counts"]=={"p3_artifact_loads":1,"p3_disk_recertifications":1,
        "p3_independent_pin_replays":192,"prerequisite_artifact_loads":2,
        "pin_model_contexts":2,"worker_subprocess_attempts":1,
        "worker_subprocess_successes":1,"b16_constructors":1,
        "sim_forward_calls":6080,"tool_position_calls":6081,"solve_calls":0,
        "sqp_calls":0,"optimizer_calls":0,"rng_calls":0,"task_construction_calls":0}
    assert all(len(value)==64 for value in report["hashes"].values())


@pytest.mark.parametrize("route",("short","long"))
def test_endpoint_exact_n260_proxy_and_mutations(route):
    q0=np.linspace(-.2,.2,7);qg=np.linspace(.1,.4,7);direction=np.ones(7)/np.sqrt(7)
    value=schema.planned_proxy(q0,qg,direction,route)
    assert schema.certify_proxy(value,q0,qg,direction,route)["passes"]
    assert value["planned_q_float64"].shape==(260,7)
    assert value["planned_qdd_float64"].shape==(259,7)
    assert np.max(np.abs(value["planned_q_float64"][-1]-qg))<=1e-12
    assert np.max(np.abs(value["planned_qd_float64"][-1]))<=1e-12
    bad=copy.deepcopy(value);bad["planned_qdd_float64"][100,2]+=1e-7
    assert not schema.certify_proxy(bad,q0,qg,direction,route)["passes"]


def test_seed_pack_is_exact_interleaved_n260_and_inputs_are_float32_distinct():
    inputs=_inputs();assert schema.worker_input_schema(inputs)
    seed=inputs["b1_seed_xu_float32"]
    assert seed.shape==(5453,) and seed.dtype==np.float32
    assert np.array_equal(seed[:7],np.zeros(7,np.float32))
    assert np.array_equal(seed[7:14],np.zeros(7,np.float32))
    assert np.array_equal(seed[14:21],inputs["controls_float32"][0,0])
    bad=copy.deepcopy(inputs);bad["controls_float32"]=bad["controls_float32"].astype(np.float64)
    assert not schema.worker_input_schema(bad)
    bad=copy.deepcopy(inputs);bad["controls_float32"][1]=bad["controls_float32"][0]
    assert not schema.worker_input_schema(bad)


def test_worker_source_assigns_entire_b16_result_and_keeps_b1_scalar_index_only():
    source=inspect.getsource(worker.execute_worker)
    assert "b16_state[:,k+1]=b16_next" in source
    assert "b16_next[0]" not in source
    assert "b1_state[k+1]=b1_next[0]" in source
    assert "b16_next.shape!=(LANES,14)" in source


def test_worker_certificate_gates_lane0_parity_and_nonidentical_routes():
    inputs=_inputs();state=np.zeros((schema.LANES,schema.DENSE_SAMPLES,14),np.float32)
    tool=np.zeros((schema.LANES,schema.DENSE_SAMPLES,3),np.float32)
    state[1,:,0]=.01;tool[1,:,0]=.01
    arrays={"captured_x0_float32":inputs["x0_float32"],
        "captured_controls_float32":inputs["controls_float32"],
        "captured_b1_seed_xu_float32":inputs["b1_seed_xu_float32"],
        "captured_b16_seed_xu_float32":inputs["b16_seed_xu_float32"],
        "b1_dense_state_float32":state[0],"b1_dense_tool_float32":tool[0],
        "b16_dense_state_float32":state,"b16_dense_tool_float32":tool}
    summary={"protocol":schema.WORKER_PROTOCOL,"module":schema.EXTENSION["module"],
        "extension":schema.EXTENSION,"counts":worker.expected_counts(),
        "array_names":sorted(arrays),"array_hashes":{k:schema.array_hash(arrays[k]) for k in sorted(arrays)},
        "elapsed_s":1.,"certificate":{}}
    first=worker.certify_output(summary,arrays,inputs)
    stored={k:v for k,v in first["gates"].items() if k!="stored_certificate"}
    summary["certificate"]={"gates":stored,"passes":all(stored.values())}
    assert worker.certify_output(summary,arrays,inputs)["passes"]
    bad=copy.deepcopy(arrays);bad["b16_dense_state_float32"][1]=bad["b16_dense_state_float32"][0]
    bad["b16_dense_tool_float32"][1]=bad["b16_dense_tool_float32"][0]
    assert not worker.certify_output(summary,bad,inputs)["passes"]


def test_runner_is_static_build_stage_only_and_execution_blocked():
    value=runner.static_design();assert runner.certify_static_design(value)
    assert value["attempts"]==2 and value["retries"]==0
    assert value["optimizer_calls"]==value["solve_calls"]==value["sqp_calls"]==0
    assert value["p4_v2_artifact_loads"]==0 and value["cuda_evidence"] is False
    with pytest.raises(RuntimeError,match="blocked"):runner.execute(schema.OUTPUT)
    with pytest.raises(RuntimeError,match="blocked"):worker.execute_worker(Path("missing"))
    bad=copy.deepcopy(value);bad["kp"]=400.
    assert not runner.certify_static_design(bad)


def test_scientific_gate_thresholds_reject_known_n96_failure_values():
    source=inspect.getsource(schema.certify_route)
    for fragment in ('<=.015','<=.05','<=.001','>=CLEARANCE_MARGIN','model_cost_agreement'):
        assert fragment in source
    pair=inspect.getsource(schema.certify_pair)
    assert 'pin_reversal' in pair and 'cuda_reversal' in pair
