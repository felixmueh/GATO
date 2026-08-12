import copy,inspect,json,re
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


def test_p5_isolated_n260_build_wiring_and_exact_pilot_tokens():
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
    assert schema.EXTENSION["sha256"]=="8ed0f549d35d99790ee3756c4e9b1c792f8c40f6a2cccd17a2b8432693d04c49"
    assert schema.EXTENSION["size"]==6694576
    assert schema.EXTENSION["build_head"]=="b5bf3b1d68dd4d1cc91d3634ae5b62b1c9371579"
    assert p4_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert p4_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is not None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is not None
    pattern=re.compile(r"^[A-Z][A-Z0-9_]*AUTHORIZATION\s*=\s*object\(\)$")
    enabled=sorted((path.name,line) for path in (root/"tiago_src/gato_tiago").glob("*.py")
        for line in path.read_text().splitlines() if pattern.fullmatch(line))
    assert enabled==sorted([
        ("circular_portfolio_p5_runner.py","RUNNER_EXECUTION_AUTHORIZATION=object()"),
        ("circular_portfolio_p5_worker.py","WORKER_EXECUTION_AUTHORIZATION=object()")])
    assert schema.OUTPUT==Path(
        "/tmp/tiago-tool-center-constructed-route-portfolio-p5-n260-pilot-authorized-once/p5.json")
    assert runner.AUTHORIZED_ORIG_ARGV==("python","-B","-m",
        "gato_tiago.circular_portfolio_p5_runner","--execute","--output",str(schema.OUTPUT))
    assert not schema.OUTPUT.parent.exists()
    with pytest.raises(RuntimeError,match="wrong P5 output"):
        runner.execute(schema.OUTPUT.with_name("wrong.json"),runner.RUNNER_EXECUTION_AUTHORIZATION)
    with pytest.raises(RuntimeError,match="wrong P5 request path"):
        worker.execute_worker(Path("/tmp/wrong-p5.request.json"),worker.WORKER_EXECUTION_AUTHORIZATION)
    source=inspect.getsource(runner.execute)
    assert source.index("if Path(output).resolve()!=OUTPUT") \
        <source.index("if OUTPUT.parent.exists()")<source.index("OUTPUT.parent.mkdir")
    assert "if path.exists() or candidate.exists()" in inspect.getsource(runner.atomic_json)
    assert "if path.exists() or candidate.exists()" in inspect.getsource(runner.atomic_npz)


def test_p4_v2_closure_is_exact_report_only_and_zero_load():
    report=schema.P4_V2_REJECTED_REPORT
    assert report["classification"]=="scientific_replay_rejected_and_closed"
    assert report["completed"]==0 and report["pending"]==12
    assert report["rejected_p4_v2_artifact_loads"]==0
    assert report["lanes_1_through_15_evidence"] is False
    assert "broadcast" in report["broadcast_defect"]
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
    assert schema.route_paths(0)["npz"]!=schema.route_paths(1)["npz"]


def test_worker_source_assigns_entire_b16_result_and_keeps_b1_scalar_index_only():
    source=inspect.getsource(worker.execute_worker)
    assert "b16_state[:,k+1]=b16_next" in source
    assert "b16_next[0]" not in source
    assert "b1_state[k+1]=b1_next[0]" in source
    assert "b16_next.shape!=(LANES,14)" in source


def test_worker_certificate_gates_lane0_parity_and_nonidentical_routes(monkeypatch):
    inputs=_inputs();state=np.zeros((schema.LANES,schema.DENSE_SAMPLES,14),np.float32)
    tool=np.zeros((schema.LANES,schema.DENSE_SAMPLES,3),np.float32)
    state[1,:,0]=.01;tool[1,:,0]=.01
    arrays={"captured_x0_float32":inputs["x0_float32"],
        "captured_controls_float32":inputs["controls_float32"],
        "captured_b1_seed_xu_float32":inputs["b1_seed_xu_float32"],
        "captured_b16_seed_xu_float32":inputs["b16_seed_xu_float32"],
        "b1_dense_state_float32":state[0],"b1_dense_tool_float32":tool[0],
        "b16_dense_state_float32":state,"b16_dense_tool_float32":tool}
    provenance={"ok":True}
    paths=schema.worker_paths();module_path=str((Path(__file__).resolve().parents[2]
        /schema.EXTENSION["relative_path"]).resolve())
    summary={"protocol":schema.WORKER_PROTOCOL,"module":schema.EXTENSION["module"],
        "module_path":module_path,"module_sha256":schema.EXTENSION["sha256"],
        "module_size":schema.EXTENSION["size"],"request_path":str(paths["request"]),
        "request_sha256":"fixture","input_path":str(paths["input"]),"input_sha256":"fixture",
        "output_npz_path":str(paths["npz"]),"output_npz_sha256":"fixture",
        "extension":schema.EXTENSION,"counts":worker.expected_counts(),
        "array_names":sorted(arrays),"array_hashes":{k:schema.array_hash(arrays[k]) for k in sorted(arrays)},
        "elapsed_s":1.,"provenance":provenance,"cuda_diagnostics":{
            "uuid":"gpu","name":"device","driver_version":"1","compute_capability":"6.1",
            "cuda_runtime_version":12090,"cuda_driver_api_version":13000},"certificate":{}}
    monkeypatch.setattr(worker,"certify_provenance",lambda value:value==provenance)
    monkeypatch.setattr(worker,"sha",lambda _path:"fixture")
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


def test_runner_extension_measurement_and_provenance_fail_closed(monkeypatch):
    current=runner.snapshot();assert current["extension"]==runner.frozen_extension()
    wrong=copy.deepcopy(current);wrong["extension"]["sha256"]="0"*64
    monkeypatch.setattr(runner,"snapshot",lambda:wrong)
    for key,value in runner.THREAD_ENV.items():monkeypatch.setenv(key,value)
    with pytest.raises(RuntimeError,match="pin mismatch"):runner.start_provenance()


def test_exact_transaction_counts_rejection_and_checkpoint_surface(monkeypatch):
    assert runner.final_counts()=={
        **runner.initial_counts(),"prerequisite_artifact_loads":2,
        "prerequisite_independent_recertifications":1,"pin_model_contexts":1,
        "primary_pin_rollouts":2,"independent_pin_replays":2,"rnea_calls":518,
        "primary_aba_calls":33152,"independent_aba_calls":33152,
        "primary_fk_calls":33154,"independent_fk_calls":33154,
        "task_endpoint_fk_calls":1,"direction_fk_calls":1,
        "owned_task_endpoint_fk_calls":1,"owned_direction_fk_calls":1,
        "owned_pin_replays":2,"owned_rnea_calls":518,"owned_aba_calls":33152,
        "owned_fk_calls":33154,
        "worker_subprocess_attempts":1,"worker_subprocess_successes":1,
        "b1_constructors":1,"b16_constructors":1,"b1_sim_forward_calls":16576,
        "b16_sim_forward_calls":16576,"b1_tool_position_calls":16577,
        "b16_tool_position_calls":16577}
    provenance={"ok":True};monkeypatch.setattr(runner,"certify_provenance",
        lambda value,final:value==provenance and final)
    rejected={"protocol":schema.PROTOCOL,"stage":"pilot_failed","incomplete":True,
        "error_type":"RuntimeError","error_message":"injected","counts":runner.initial_counts(),
        "provenance":provenance,"trigger_elapsed_s":1.,"finish_elapsed_s":1.1,
        "campaign_wall_limit_s":schema.CAMPAIGN_WALL_LIMIT_S,"completed":0,"pending":2,
        "active_artifacts":runner.active_artifact_map(),"worker_execution":None,
        "prerequisite_authentication":None,"route_artifacts":[],
        "oracle_evidence":False,"benchmark_evidence":False,"sqp_evidence":False}
    assert runner.certify_rejection(rejected)
    bad=copy.deepcopy(rejected);bad["counts"]["solve_calls"]=1
    assert not runner.certify_rejection(bad)
    bad=copy.deepcopy(rejected);bad["pending"]=1
    assert not runner.certify_rejection(bad)


def test_scientific_gate_thresholds_reject_known_n96_failure_values():
    source=inspect.getsource(schema.certify_route)
    for fragment in ('<=.015','<=.05','<=.001','>=CLEARANCE_MARGIN','model_cost_agreement'):
        assert fragment in source
    pair=inspect.getsource(schema.certify_pair)
    assert 'pin_reversal' in pair and 'cuda_reversal' in pair


def test_exact_route_and_pair_detail_schemas_reject_nested_pass_lies():
    metrics={key:1. for key in ("pin_terminal_m","cuda_terminal_m","pin_speed_mps",
        "cuda_speed_mps","tool_disagreement_rms_m","tool_disagreement_max_m",
        "pin_clearance_m","cuda_clearance_m","pin_turn_rad","cuda_turn_rad",
        "pin_base_cost","pin_full_cost","cuda_base_cost","cuda_full_cost")}
    gates={key:True for key in ("finite","q_limits","v_limits","u_limits","terminal",
        "speed","clearance","agreement","topology","model_cost_agreement")}
    route={"route":"short","metrics":metrics,"gates":gates,"passes":True}
    assert schema.certify_route_detail(route)
    bad=copy.deepcopy(route);bad["metrics"]["cuda_terminal_m"]=float("nan")
    assert not schema.certify_route_detail(bad)
    bad=copy.deepcopy(route);bad["gates"]["agreement"]=False
    assert not schema.certify_route_detail(bad)
    pair={"gates":{"opposite_topology":True,"pin_reversal":True,"cuda_reversal":True},
        "passes":True}
    assert schema.certify_pair_detail(pair)
    bad=copy.deepcopy(pair);bad["gates"]["extra"]=True
    assert not schema.certify_pair_detail(bad)


def test_partial_counter_causality_and_worker_execution_states_reject_huge_counts():
    counts=runner.initial_counts();assert runner.certify_execution_counts(counts)
    bad=copy.deepcopy(counts);bad["primary_aba_calls"]=10**12
    assert not runner.certify_execution_counts(bad)
    bad=copy.deepcopy(counts);bad["worker_subprocess_attempts"]=1
    assert not runner.certify_execution_counts(bad)
    assert runner.certify_worker_execution(None,counts)
    killed={"status":"killed_internal_unknown","internal_counts_known":False,
        "counts":None,"lower_bounds":{k:0 for k in ("b1_constructors","b16_constructors",
            "b1_sim_forward_calls","b16_sim_forward_calls","b1_tool_position_calls",
            "b16_tool_position_calls","solve_calls","sqp_calls")},"returncode":-9,"killed":True}
    attempted=runner.pin_routes_counts();attempted["worker_subprocess_attempts"]=1
    assert runner.certify_worker_execution(killed,attempted)
    bad=copy.deepcopy(killed);bad["counts"]={}
    assert not runner.certify_worker_execution(bad,attempted)


def test_cpu_deadline_guards_before_dynamics_callbacks():
    calls=[];counts=runner.initial_counts()
    controls=np.zeros((schema.INTERVALS,7),np.float32)
    with pytest.raises(TimeoutError,match="campaign wall"):
        runner._pin_replay(np.zeros(7),controls,
            lambda *_args:calls.append("aba") or np.zeros(7),
            lambda _q:(calls.append("fk") or np.zeros(3),np.zeros((3,7))),
            counts,deadline=0.,monotonic=lambda:0.)
    assert calls==[] and counts==runner.initial_counts()


def test_publication_hash_delay_is_measured_and_guarded_before_pointer():
    clock=iter((599.9,599.9))
    assert runner._publication_finish_elapsed(0.,600.,lambda:next(clock))==599.9
    clock=iter((600.01,600.01))
    with pytest.raises(TimeoutError,match="campaign wall"):
        runner._publication_finish_elapsed(0.,600.,lambda:next(clock))
    source=inspect.getsource(runner.execute)
    assert source.index("atomic_json(manifest_path,manifest)") \
        <source.index("pointer_document={")<source.index("_publish_final_pointer(")
    boundary=inspect.getsource(runner._publish_final_pointer)
    assert boundary.index("_publication_finish_elapsed")<boundary.index("atomic_pointer(")


def test_delayed_precommit_never_writes_authoritative_pointer(tmp_path,monkeypatch):
    _patch_output(monkeypatch,tmp_path);writes=[]
    monkeypatch.setattr(runner,"atomic_pointer",lambda *args:writes.append(args))
    payload={"json_sha256":"already-computed","npz_sha256":"already-computed",
        "manifest_sha256":"already-computed"}
    clock=iter((600.01,600.01))
    with pytest.raises(TimeoutError,match="campaign wall"):
        runner._publish_final_pointer(payload,0.,600.,lambda:next(clock))
    assert writes==[] and not runner.OUTPUT.with_name("p5.final.latest.json").exists()


def test_failure_route_science_rejects_coherent_refreshed_alternate_direction(tmp_path,monkeypatch):
    _patch_output(monkeypatch,tmp_path)
    prerequisite={"public_x0_float32":np.zeros((1,14),np.float32),
        "quarantined_q8_float64":np.zeros((1,7)),"q8_tool_float64":np.zeros((1,3)),
        "public_default_side_int8":np.asarray([1],np.int8)}
    expected=np.ones(7,np.float64)/np.sqrt(7.)
    monkeypatch.setattr(runner,"perturbation_direction",lambda *_args:expected.copy())
    kinematics=lambda q:(np.asarray(q)[:3].copy(),np.zeros((3,7)))
    rnea=lambda _q,_qd,_qdd:np.zeros(7)
    aba=lambda _q,_qd,_u:np.zeros(7)
    q0=np.zeros(7);proxy=schema.planned_proxy(q0,q0,expected,"short")
    controls=np.zeros((schema.INTERVALS,7),np.float32);local=runner.initial_counts()
    state,tool=runner._pin_replay(q0,controls,aba,kinematics,local,10.,lambda:0.)
    arrays=runner.route_arrays(expected,proxy,state,tool,state.copy(),tool.copy(),controls)
    paths=schema.route_paths(0);np.savez(paths["npz"],**arrays)
    summary=runner.route_summary(0,"short",arrays)
    paths["json"].write_text(json.dumps(summary,sort_keys=True,separators=(",",":")))
    assert runner.recertify_failure_route_science(
        prerequisite,kinematics,rnea,aba,10.,lambda:0.)
    authentication={"synthetic":True};provenance={"final":True}
    import gato_tiago.circular_portfolio_runner as predecessor
    monkeypatch.setattr(predecessor,"authenticate_cpu_prerequisite",
        lambda:(authentication,prerequisite))
    monkeypatch.setattr(predecessor,"_production_pin_context",
        lambda:(object(),kinematics,rnea,object(),aba,object(),object()))
    monkeypatch.setattr(runner,"certify_provenance",lambda value,final:value==provenance)
    monkeypatch.setattr(runner,"certify_checkpoint_document",lambda _value:True)
    counts=runner.initial_counts();counts.update({"prerequisite_artifact_loads":1,
        "pin_model_contexts":1,"primary_pin_rollouts":1,"independent_pin_replays":1,
        "rnea_calls":schema.INTERVALS,"primary_aba_calls":schema.INTERVALS*schema.DENSE_SUBSTEPS,
        "independent_aba_calls":schema.INTERVALS*schema.DENSE_SUBSTEPS,
        "primary_fk_calls":schema.DENSE_SAMPLES,"independent_fk_calls":schema.DENSE_SAMPLES,
        "task_endpoint_fk_calls":1,"direction_fk_calls":1})
    for generation,detail in ((0,{}),(1,{"authentication":authentication})):
        runner.checkpoint_path(generation).write_text(json.dumps(
            {"stage":("started" if generation==0 else "prerequisite_authenticated"),
             "detail":detail},sort_keys=True,separators=(",",":")))
    latest={"protocol":schema.PROTOCOL,"generation":1,"path":str(runner.checkpoint_path(1)),
        "sha256":runner.sha(runner.checkpoint_path(1)),"incomplete":True}
    runner.OUTPUT.with_name("p5.partial.latest.json").write_text(json.dumps(latest,
        sort_keys=True,separators=(",",":")))
    rejection={"protocol":schema.PROTOCOL,"stage":"pilot_failed","incomplete":True,
        "error_type":"RuntimeError","error_message":"after route","counts":counts,
        "provenance":provenance,"trigger_elapsed_s":1.,"finish_elapsed_s":1.1,
        "campaign_wall_limit_s":schema.CAMPAIGN_WALL_LIMIT_S,"completed":1,"pending":1,
        "active_artifacts":runner.active_artifact_map(),"worker_execution":None,
        "prerequisite_authentication":authentication,"route_artifacts":[summary],
        "oracle_evidence":False,"benchmark_evidence":False,"sqp_evidence":False}
    rejected_path=runner.OUTPUT.with_name("p5.rejected.json")
    rejected_path.write_text(json.dumps(rejection,sort_keys=True,separators=(",",":")))
    assert runner.recertify_retained_failure(rejected_path)["passes"]
    # This is internally coherent for an alternate predeclared direction and all
    # outer array hashes are refreshed, but it is not the accepted task-derived route.
    alternate=-expected;alternate_proxy=schema.planned_proxy(q0,q0,alternate,"short")
    arrays.update({"direction_float64":alternate,
        "planned_q_float64":alternate_proxy["planned_q_float64"],
        "planned_qd_float64":alternate_proxy["planned_qd_float64"],
        "planned_qdd_float64":alternate_proxy["planned_qdd_float64"]})
    np.savez(paths["npz"],**arrays);summary=runner.route_summary(0,"short",arrays)
    paths["json"].write_text(json.dumps(summary,sort_keys=True,separators=(",",":")))
    rejection["route_artifacts"]=[summary];rejection["active_artifacts"]=runner.active_artifact_map()
    rejected_path.write_text(json.dumps(rejection,sort_keys=True,separators=(",",":")))
    assert not runner.recertify_retained_failure(rejected_path)["passes"]
    assert not runner.recertify_failure_route_science(
        prerequisite,kinematics,rnea,aba,10.,lambda:0.)


def test_worker_and_runner_sources_bind_runtime_failures_and_owned_disk_recert():
    worker_source=inspect.getsource(worker.execute_worker)
    assert "module.__file__" in worker_source and "sha(module_path)!=EXTENSION" in worker_source
    assert "cuda_diagnostics()" in worker_source and "monotonic()>=deadline" in worker_source
    assert "b16_state[:,k+1]=b16_next" in worker_source and "b16_next[0]" not in worker_source
    execute_source=inspect.getsource(runner.execute)
    assert "timeout=min(300.,remaining)" in execute_source
    assert "except subprocess.TimeoutExpired" in execute_source
    assert "owned_semantic_recert" in execute_source and "_publish_final_pointer" in execute_source
    owned_source=inspect.getsource(runner.owned_semantic_recert)
    assert "authenticate_cpu_prerequisite()" in owned_source
    assert 'json.loads(paths["json"].read_text())' in owned_source
    assert 'np.load(paths["npz"]' in owned_source
    public_source=inspect.getsource(runner.recertify_retained_p5)
    assert "owned_semantic_recert" in public_source and "expected_manifest" in public_source
    assert "expected_pointer" in public_source


def test_atomic_pointer_is_candidate_first_and_active_map_includes_all_candidates(tmp_path,monkeypatch):
    path=tmp_path/"latest.json";runner.atomic_pointer(path,{"generation":0})
    assert path.exists() and not Path(str(path)+".candidate").exists()
    runner.atomic_pointer(path,{"generation":1})
    assert json.loads(path.read_text())=={"generation":1}
    active=runner.active_artifact_map()
    assert "final_pointer" in active and "final_pointer_candidate" in active
    assert "worker_request_candidate" in active and "route_0_npz_candidate" in active


def test_checkpoint_stage_values_are_exact_and_provenance_bound(monkeypatch):
    monkeypatch.setattr(runner,"certify_provenance",lambda value,final:value=={"final":final})
    doc={"protocol":schema.PROTOCOL,"generation":0,"stage":"started","incomplete":True,
        "counts":runner.initial_counts(),"provenance":{"final":False},"detail":{}}
    assert runner.certify_checkpoint_document(doc)
    bad=copy.deepcopy(doc);bad["stage"]="prerequisite_authenticated"
    assert not runner.certify_checkpoint_document(bad)
    final={"protocol":schema.PROTOCOL,"generation":4,"stage":"honest_end","incomplete":True,
        "counts":runner.final_counts(),"provenance":{"final":True},"detail":{"elapsed_s":1.}}
    assert runner.certify_checkpoint_document(final)
    bad=copy.deepcopy(final);bad["counts"]["owned_aba_calls"]-=1
    assert not runner.certify_checkpoint_document(bad)


def test_worker_partial_counts_are_measured_and_causal():
    counts=worker.partial_counts();assert worker.certify_partial_counts(counts)
    counts["b1_constructors"]=counts["b16_constructors"]=1
    counts["b1_sim_forward_calls"]=1;counts["b16_sim_forward_calls"]=1
    assert worker.certify_partial_counts(counts)
    bad=copy.deepcopy(counts);bad["b1_tool_position_calls"]=1
    assert not worker.certify_partial_counts(bad)
    bad=copy.deepcopy(counts);bad["b16_sim_forward_calls"]=10**9
    assert not worker.certify_partial_counts(bad)


def _patch_output(monkeypatch,tmp_path):
    output=tmp_path/"p5.json"
    monkeypatch.setattr(schema,"OUTPUT",output);monkeypatch.setattr(runner,"OUTPUT",output)
    return output


def test_public_failure_disk_roundtrip_and_refreshed_checkpoint_mutation(tmp_path,monkeypatch):
    output=_patch_output(monkeypatch,tmp_path);provenance={"final":True}
    monkeypatch.setattr(runner,"certify_provenance",lambda value,final:value==provenance)
    gen={"protocol":schema.PROTOCOL,"generation":0,"stage":"started","incomplete":True,
        "counts":runner.initial_counts(),"provenance":provenance,"detail":{}}
    runner.atomic_json(runner.checkpoint_path(0),gen)
    pointer={"protocol":schema.PROTOCOL,"generation":0,"path":str(runner.checkpoint_path(0)),
        "sha256":runner.sha(runner.checkpoint_path(0)),"incomplete":True}
    runner.atomic_json(output.with_name("p5.partial.latest.json"),pointer)
    rejected={"protocol":schema.PROTOCOL,"stage":"pilot_failed","incomplete":True,
        "error_type":"RuntimeError","error_message":"synthetic","counts":runner.initial_counts(),
        "provenance":provenance,"trigger_elapsed_s":1.,"finish_elapsed_s":1.1,
        "campaign_wall_limit_s":schema.CAMPAIGN_WALL_LIMIT_S,"completed":0,"pending":2,
        "active_artifacts":runner.active_artifact_map(),"worker_execution":None,
        "prerequisite_authentication":None,"route_artifacts":[],"oracle_evidence":False,
        "benchmark_evidence":False,"sqp_evidence":False}
    output.with_name("p5.rejected.json").write_text(json.dumps(rejected,
        sort_keys=True,separators=(",",":")))
    assert runner.recertify_retained_failure(output.with_name("p5.rejected.json"))["passes"]
    bad_rejected=copy.deepcopy(rejected);bad_rejected["counts"]["primary_fk_calls"]=10**12
    output.with_name("p5.rejected.json").write_text(json.dumps(bad_rejected,
        sort_keys=True,separators=(",",":")))
    assert not runner.recertify_retained_failure(output.with_name("p5.rejected.json"))["passes"]
    output.with_name("p5.rejected.json").write_text(json.dumps(rejected,
        sort_keys=True,separators=(",",":")))
    gen["counts"]["optimizer_calls"]=1
    runner.checkpoint_path(0).write_text(json.dumps(gen,sort_keys=True,separators=(",",":")))
    pointer["sha256"]=runner.sha(runner.checkpoint_path(0))
    output.with_name("p5.partial.latest.json").write_text(json.dumps(pointer,sort_keys=True,separators=(",",":")))
    assert not runner.recertify_retained_failure(output.with_name("p5.rejected.json"))["passes"]


def test_worker_rejection_disk_binding_rejects_refreshed_semantic_lie(tmp_path,monkeypatch):
    output=_patch_output(monkeypatch,tmp_path);paths=schema.worker_paths()
    paths["request"].write_text("{}");np.savez(paths["input"],placeholder=np.zeros(1))
    counts=worker.partial_counts();counts["b1_constructors"]=counts["b16_constructors"]=1
    detail={"protocol":schema.WORKER_PROTOCOL,"stage":"worker_failed","incomplete":True,
        "error_type":"TimeoutError","error_message":"timeout","counts":counts,
        "internal_counts_known":True,"provenance":{},"cuda_diagnostics":None,
        "trigger_elapsed_s":300.,"finish_elapsed_s":300.1,"worker_wall_limit_s":300.,
        "request_path":str(paths["request"]),"request_sha256":runner.sha(paths["request"]),
        "input_path":str(paths["input"]),"input_sha256":runner.sha(paths["input"]),
        "npz_path":str(paths["npz"]),"npz_sha256":None}
    paths["rejected"].write_text(json.dumps(detail,sort_keys=True,separators=(",",":")))
    monkeypatch.setattr(worker,"certify_rejection",lambda value:value==detail)
    top=runner.pin_routes_counts();top["worker_subprocess_attempts"]=1
    for key,value in counts.items():top[key]=value
    execution={"status":"known_partial","internal_counts_known":True,"counts":counts,
        "lower_bounds":counts,"returncode":1,"killed":False}
    rejected={"worker_execution":execution,"counts":top,"error_type":"TimeoutError",
        "error_message":"P5 worker wall limit",
        "active_artifacts":runner.active_artifact_map()}
    assert runner.certify_worker_failure_binding(rejected)
    provenance={"final":True};authentication={"synthetic":True};prerequisite={}
    monkeypatch.setattr(runner,"certify_provenance",lambda value,final:value==provenance)
    monkeypatch.setattr(runner,"certify_route_artifact_prefix",lambda value:len(value)==2)
    monkeypatch.setattr(runner,"certify_active_worker_artifacts",lambda *_args:True)
    monkeypatch.setattr(runner,"certify_failure_side_artifacts",lambda *_args:True)
    import gato_tiago.circular_portfolio_runner as predecessor
    monkeypatch.setattr(predecessor,"authenticate_cpu_prerequisite",
        lambda:(authentication,prerequisite))
    monkeypatch.setattr(runner,"certify_checkpoint_document",lambda _value:True)
    runner.checkpoint_path(0).write_text(json.dumps({"detail":{},"stage":"started"}))
    latest={"protocol":schema.PROTOCOL,"generation":0,"path":str(runner.checkpoint_path(0)),
        "sha256":runner.sha(runner.checkpoint_path(0)),"incomplete":True}
    output.with_name("p5.partial.latest.json").write_text(json.dumps(latest,
        sort_keys=True,separators=(",",":")))
    public_rejection={"protocol":schema.PROTOCOL,"stage":"runtime_watchdog_rejected",
        "incomplete":True,"error_type":"TimeoutError","error_message":"P5 worker wall limit",
        "counts":top,"provenance":provenance,"trigger_elapsed_s":300.,
        "finish_elapsed_s":300.1,"campaign_wall_limit_s":schema.CAMPAIGN_WALL_LIMIT_S,
        "completed":2,"pending":0,"active_artifacts":runner.active_artifact_map(),
        "worker_execution":execution,"prerequisite_authentication":authentication,
        "route_artifacts":[{"index":0},{"index":1}],"oracle_evidence":False,
        "benchmark_evidence":False,"sqp_evidence":False}
    rejected_path=output.with_name("p5.rejected.json")
    rejected_path.write_text(json.dumps(public_rejection,sort_keys=True,separators=(",",":")))
    assert runner.recertify_retained_failure(rejected_path)["passes"]
    valid_counts=copy.deepcopy(counts)
    detail["counts"]["b1_constructors"]=0
    paths["rejected"].write_text(json.dumps(detail,sort_keys=True,separators=(",",":")))
    rejected["active_artifacts"]=runner.active_artifact_map()
    assert not runner.certify_worker_failure_binding(rejected)
    public_rejection["active_artifacts"]=runner.active_artifact_map()
    rejected_path.write_text(json.dumps(public_rejection,sort_keys=True,separators=(",",":")))
    assert not runner.recertify_retained_failure(rejected_path)["passes"]
    detail["counts"]=valid_counts;execution["counts"]=valid_counts
    execution["lower_bounds"]=valid_counts
    for key,value in valid_counts.items():top[key]=value
    paths["rejected"].write_text(json.dumps(detail,
        sort_keys=True,separators=(",",":")))
    rejected["active_artifacts"]=runner.active_artifact_map()
    rejected["error_message"]="fabricated translation"
    assert not runner.certify_worker_failure_binding(rejected)


def test_public_success_disk_roundtrip_rejects_refreshed_final_npz_task_mutation(tmp_path,monkeypatch):
    output=_patch_output(monkeypatch,tmp_path);provenance={"final":True}
    monkeypatch.setattr(runner,"certify_provenance",lambda value,final:value==provenance)
    prerequisite={"public_x0_float32":np.zeros((1,14),np.float32),
        "accepted_task_reference_float32":np.zeros((1,960),np.float32),
        "quarantined_q8_float64":np.zeros((1,7)),"q8_tool_float64":np.zeros((1,3)),
        "public_default_side_int8":np.asarray([1],np.int8),"joint_lower_float64":-np.ones(7),
        "joint_upper_float64":np.ones(7),"velocity_limit_float64":np.ones(7),
        "effort_limit_float64":np.ones(7)}
    authentication={"pins":{"synthetic":True}}
    final_arrays={"public_x0_float32":prerequisite["public_x0_float32"][0].copy(),
        "accepted_task_reference_float32":prerequisite["accepted_task_reference_float32"][0].copy(),
        "quarantined_q8_float64":prerequisite["quarantined_q8_float64"][0].copy(),
        "q8_tool_float64":prerequisite["q8_tool_float64"][0].copy(),
        "default_side_int8":prerequisite["public_default_side_int8"][0].copy(),
        "joint_lower_float64":prerequisite["joint_lower_float64"].copy(),
        "joint_upper_float64":prerequisite["joint_upper_float64"].copy(),
        "velocity_limit_float64":prerequisite["velocity_limit_float64"].copy(),
        "effort_limit_float64":prerequisite["effort_limit_float64"].copy(),
        "reference_float32":np.zeros(schema.KNOTS*10,np.float32),"pillar_float64":np.zeros(2),
        "short_unit_float64":np.asarray([1.,0.])}
    np.savez(output.with_suffix(".npz"),**final_arrays)
    metrics={k:1. for k in ("pin_terminal_m","cuda_terminal_m","pin_speed_mps","cuda_speed_mps",
        "tool_disagreement_rms_m","tool_disagreement_max_m","pin_clearance_m","cuda_clearance_m",
        "pin_turn_rad","cuda_turn_rad","pin_base_cost","pin_full_cost","cuda_base_cost","cuda_full_cost")}
    gates={k:True for k in ("finite","q_limits","v_limits","u_limits","terminal","speed",
        "clearance","agreement","topology","model_cost_agreement")}
    routes=[{"route":r,"metrics":copy.deepcopy(metrics),"gates":copy.deepcopy(gates),"passes":True}
            for r in ("short","long")]
    pair={"gates":{"opposite_topology":True,"pin_reversal":True,"cuda_reversal":True},"passes":True}
    route_artifacts=[{"index":i} for i in (0,1)]
    worker_artifact={k:"x" for k in ("request_path","request_sha256","input_path","input_sha256",
        "json_path","json_sha256","npz_path","npz_sha256")}
    owned={"routes":routes,"pair":pair,"geometry":{"passes":True},
        "route_artifacts":route_artifacts,"worker_artifact":worker_artifact,
        "gates":{k:True for k in ("prerequisite","route_artifacts","proxy","pin_replay",
            "geometry","worker","routes","pair")},"passes":True}
    for i in (0,1):
        schema.route_paths(i)["json"].write_text("{}")
        np.savez(schema.route_paths(i)["npz"],placeholder=np.zeros(1))
    paths=schema.worker_paths();paths["request"].write_text("{}")
    np.savez(paths["input"],placeholder=np.zeros(1));paths["json"].write_text(
        json.dumps({"certificate":{"passes":True}}));np.savez(paths["npz"],placeholder=np.zeros(1))
    monkeypatch.setattr(runner,"owned_semantic_recert",lambda *a,**k:copy.deepcopy(owned))
    import gato_tiago.circular_portfolio_runner as predecessor
    monkeypatch.setattr(predecessor,"authenticate_cpu_prerequisite",lambda:(authentication,prerequisite))
    monkeypatch.setattr(predecessor,"_production_pin_context",lambda:(object(),)*7)
    document={"protocol":schema.PROTOCOL,"incomplete":False,"overall_pass":True,
        "identity":list(schema.PILOT_IDENTITY),"routes":routes,"pair":pair,
        "counts":runner.final_counts(),"worker":{"certificate":{"passes":True}},
        "provenance":provenance,"elapsed_s":1.,"prerequisite_authentication":authentication,
        "owned_semantic_recertification":owned,"route_artifacts":route_artifacts,
        "worker_artifact":worker_artifact,"npz_path":str(output.with_suffix(".npz")),
        "npz_sha256":runner.sha(output.with_suffix(".npz")),"array_names":sorted(final_arrays),
        "array_hashes":{k:schema.array_hash(v) for k,v in sorted(final_arrays.items())},
        "oracle_evidence":False,"benchmark_evidence":False,"sqp_evidence":False}
    output.write_text(json.dumps(document,sort_keys=True,separators=(",",":")))
    monkeypatch.setattr(runner,"certify_checkpoint_document",lambda _value:True)
    stages=("started","prerequisite_authenticated","pin_routes_constructed",
        "owned_semantic_certified","honest_end")
    for i in range(5):runner.checkpoint_path(i).write_text(json.dumps({"stage":stages[i],"detail":({"authentication":authentication}
        if i==1 else {"geometry":owned["geometry"],"route_artifacts":route_artifacts} if i==2
        else {"routes":routes,"pair":pair,"route_artifacts":route_artifacts,"worker_artifact":worker_artifact}
        if i==3 else {"elapsed_s":1.} if i==4 else {})}))
    side=[{"path":str(runner.checkpoint_path(i)),"sha256":runner.sha(runner.checkpoint_path(i))} for i in range(5)]
    side += [{"path":str(schema.route_paths(i)[k]),"sha256":runner.sha(schema.route_paths(i)[k])}
             for i in (0,1) for k in ("json","npz")]
    side += [{"path":str(paths[k]),"sha256":runner.sha(paths[k])} for k in ("request","input","json","npz")]
    manifest={"protocol":schema.PROTOCOL,"json_path":str(output),"json_sha256":runner.sha(output),
        "npz_path":str(output.with_suffix(".npz")),"npz_sha256":runner.sha(output.with_suffix(".npz")),
        "side_artifacts":side,"side_artifact_count":len(side),"overall_pass":True}
    manifest_path=output.with_name("p5.manifest.json");manifest_path.write_text(json.dumps(manifest,sort_keys=True,separators=(",",":")))
    pointer={"protocol":schema.PROTOCOL,"json_path":str(output),"json_sha256":runner.sha(output),
        "npz_path":str(output.with_suffix(".npz")),"npz_sha256":runner.sha(output.with_suffix(".npz")),
        "manifest_path":str(manifest_path),"manifest_sha256":runner.sha(manifest_path),
        "publication_finish_elapsed_s":1.1,"overall_pass":True}
    output.with_name("p5.final.latest.json").write_text(json.dumps(pointer,sort_keys=True,separators=(",",":")))
    assert runner.certify_final_arrays_task(runner.load_npz(output.with_suffix(".npz")),prerequisite)
    assert runner.certify_final_document(document)
    assert runner.recertify_retained_p5(output)["passes"]
    late_pointer=copy.deepcopy(pointer);late_pointer["publication_finish_elapsed_s"]=600.01
    output.with_name("p5.final.latest.json").write_text(json.dumps(late_pointer,
        sort_keys=True,separators=(",",":")))
    assert not runner.recertify_retained_p5(output)["passes"]
    output.with_name("p5.final.latest.json").write_text(json.dumps(pointer,
        sort_keys=True,separators=(",",":")))
    final_arrays["public_x0_float32"][0]=1.;np.savez(output.with_suffix(".npz"),**final_arrays)
    document["npz_sha256"]=runner.sha(output.with_suffix(".npz"));document["array_hashes"]={k:schema.array_hash(v) for k,v in sorted(final_arrays.items())}
    output.write_text(json.dumps(document,sort_keys=True,separators=(",",":")))
    manifest.update({"json_sha256":runner.sha(output),"npz_sha256":runner.sha(output.with_suffix(".npz"))})
    manifest_path.write_text(json.dumps(manifest,sort_keys=True,separators=(",",":")))
    pointer.update({"json_sha256":runner.sha(output),"npz_sha256":runner.sha(output.with_suffix(".npz")),
        "manifest_sha256":runner.sha(manifest_path)})
    output.with_name("p5.final.latest.json").write_text(json.dumps(pointer,sort_keys=True,separators=(",",":")))
    assert not runner.recertify_retained_p5(output)["passes"]
