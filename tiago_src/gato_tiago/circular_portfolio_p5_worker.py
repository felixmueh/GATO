"""Capability-blocked B1/B16 CUDA worker for the N260 P5 pilot."""

from __future__ import annotations

import argparse,importlib,json,time
from pathlib import Path

import numpy as np

from gato_tiago.circular_portfolio import (CONTROL_LIMIT_COST,CYLINDER_WEIGHT,KKT_TOL,
    MAX_PCG_ITERS,MU,N_COST,PCG_TOL,Q_COST,QD_COST,Q_LIMIT_COST,RHO,SOLVE_RATIO,U_COST,
    VELOCITY_LIMIT_COST)
from gato_tiago.circular_portfolio_p5 import (DENSE_SAMPLES,DENSE_SUBSTEPS,DT,EXTENSION,
    INTERVALS,KNOTS,LANES,WORKER_PROTOCOL,array_hash,worker_input_schema,worker_output_schema)


WORKER_EXECUTION_AUTHORIZATION=None


def expected_counts():
    return {"b1_constructors":1,"b16_constructors":1,
        "b1_sim_forward_calls":INTERVALS*DENSE_SUBSTEPS,
        "b16_sim_forward_calls":INTERVALS*DENSE_SUBSTEPS,
        "b1_tool_position_calls":DENSE_SAMPLES,"b16_tool_position_calls":DENSE_SAMPLES,
        "solve_calls":0,"sqp_calls":0}


def certify_output(summary,arrays,inputs):
    keys={"protocol","module","extension","counts","array_names","array_hashes",
        "elapsed_s","certificate"}
    gates={"summary_keys":set(summary)==keys,"protocol":summary.get("protocol")==WORKER_PROTOCOL,
        "module":summary.get("module")==EXTENSION["module"],
        "extension":summary.get("extension")==EXTENSION,
        "input":worker_input_schema(inputs),"output":worker_output_schema(arrays),
        "counts":summary.get("counts")==expected_counts(),
        "capture":worker_output_schema(arrays) and worker_input_schema(inputs)
            and np.array_equal(arrays["captured_x0_float32"],inputs["x0_float32"])
            and np.array_equal(arrays["captured_controls_float32"],inputs["controls_float32"])
            and np.array_equal(arrays["captured_b1_seed_xu_float32"],inputs["b1_seed_xu_float32"])
            and np.array_equal(arrays["captured_b16_seed_xu_float32"],inputs["b16_seed_xu_float32"]),
        "b1_b16_lane0":worker_output_schema(arrays)
            and np.array_equal(arrays["b1_dense_state_float32"],arrays["b16_dense_state_float32"][0])
            and np.array_equal(arrays["b1_dense_tool_float32"],arrays["b16_dense_tool_float32"][0]),
        "lane_outputs_distinct":worker_output_schema(arrays)
            and not np.array_equal(arrays["b16_dense_state_float32"][0],arrays["b16_dense_state_float32"][1])
            and not np.array_equal(arrays["b16_dense_tool_float32"][0],arrays["b16_dense_tool_float32"][1]),
        "arrays":summary.get("array_names")==sorted(arrays)
            and summary.get("array_hashes")=={k:array_hash(arrays[k]) for k in sorted(arrays)},
        "elapsed":isinstance(summary.get("elapsed_s"),(int,float))
            and np.isfinite(summary["elapsed_s"]) and summary["elapsed_s"]>=0}
    certificate=summary.get("certificate")
    gates["stored_certificate"]=isinstance(certificate,dict) \
        and certificate.get("gates")=={k:v for k,v in gates.items() if k!="stored_certificate"} \
        and certificate.get("passes") is all(v is True for k,v in gates.items() if k!="stored_certificate")
    return {"gates":gates,"passes":bool(all(gates.values()))}


def execute_worker(input_path,authorization=None,monotonic=time.monotonic): # pragma: no cover
    if WORKER_EXECUTION_AUTHORIZATION is None or authorization is not WORKER_EXECUTION_AUTHORIZATION:
        raise RuntimeError("P5 worker blocked")
    if EXTENSION["sha256"] is None or EXTENSION["size"] is None or EXTENSION["build_head"] is None:
        raise RuntimeError("P5 extension has not completed build-only authentication")
    with np.load(input_path,allow_pickle=False) as archive:inputs={k:archive[k] for k in archive.files}
    if not worker_input_schema(inputs):raise ValueError("P5 worker input invalid")
    module=importlib.import_module(EXTENSION["module"])
    if (int(module.KNOT_POINTS),int(module.REFERENCE_SIZE),str(module.TOOL_POSITION_FRAME),
            int(module.TOOL_POSITION_SIZE))!=(260,10,"arm_right_tool_joint_origin",3):
        raise RuntimeError("P5 extension attributes invalid")
    args=(DT,0,KKT_TOL,MAX_PCG_ITERS,PCG_TOL,SOLVE_RATIO,MU,Q_COST,QD_COST,U_COST,
        N_COST,CYLINDER_WEIGHT,CYLINDER_WEIGHT,Q_LIMIT_COST,VELOCITY_LIMIT_COST,
        CONTROL_LIMIT_COST,RHO)
    b1=module.BSQP_1_float(*args);b16=module.BSQP_16_float(*args);started=monotonic()
    b1_state=np.empty((DENSE_SAMPLES,14),np.float32);b1_state[0]=inputs["x0_float32"][0]
    b16_state=np.empty((LANES,DENSE_SAMPLES,14),np.float32);b16_state[:,0]=inputs["x0_float32"]
    for interval in range(INTERVALS):
        for substep in range(DENSE_SUBSTEPS):
            k=interval*DENSE_SUBSTEPS+substep
            b1_next=np.asarray(b1.sim_forward(b1_state[k],inputs["controls_float32"][0,interval],DT/DENSE_SUBSTEPS))
            b1_state[k+1]=b1_next[0]
            b16_next=np.asarray(b16.sim_forward(b16_state[:,k],inputs["controls_float32"][:,interval],DT/DENSE_SUBSTEPS))
            if b16_next.shape!=(LANES,14):raise RuntimeError("P5 B16 sim_forward layout invalid")
            b16_state[:,k+1]=b16_next
    b1_tool=np.empty((DENSE_SAMPLES,3),np.float32)
    b16_tool=np.empty((LANES,DENSE_SAMPLES,3),np.float32)
    for k in range(DENSE_SAMPLES):
        b1_tool[k]=np.asarray(b1.tool_position(b1_state[k,:7]))[0]
        value=np.asarray(b16.tool_position(b16_state[:,k,:7]))
        if value.shape!=(LANES,3):raise RuntimeError("P5 B16 tool layout invalid")
        b16_tool[:,k]=value
    return {"captured_x0_float32":inputs["x0_float32"],
        "captured_controls_float32":inputs["controls_float32"],
        "captured_b1_seed_xu_float32":inputs["b1_seed_xu_float32"],
        "captured_b16_seed_xu_float32":inputs["b16_seed_xu_float32"],
        "b1_dense_state_float32":b1_state,"b1_dense_tool_float32":b1_tool,
        "b16_dense_state_float32":b16_state,"b16_dense_tool_float32":b16_tool},monotonic()-started


def main(argv=None): # pragma: no cover
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--input",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    if not args.execute:raise SystemExit("--execute required")
    arrays,elapsed=execute_worker(args.input,WORKER_EXECUTION_AUTHORIZATION)
    if args.output.exists():raise FileExistsError("P5 worker output exists")
    with args.output.open("xb") as stream:np.savez(stream,**arrays)
    summary={"protocol":WORKER_PROTOCOL,"module":EXTENSION["module"],"extension":EXTENSION,
        "counts":expected_counts(),"array_names":sorted(arrays),
        "array_hashes":{k:array_hash(arrays[k]) for k in sorted(arrays)},
        "elapsed_s":elapsed,"certificate":{}}
    first=certify_output(summary,arrays,{k:arrays["captured_"+k] for k in
        ("x0_float32","controls_float32","b1_seed_xu_float32","b16_seed_xu_float32")})
    stored={k:v for k,v in first["gates"].items() if k!="stored_certificate"}
    summary["certificate"]={"gates":stored,"passes":bool(all(stored.values()))}
    args.output.with_suffix(".json").write_text(json.dumps(summary,sort_keys=True,separators=(",",":")))

if __name__=="__main__":main()
