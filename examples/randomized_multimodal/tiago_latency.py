"""Matched-work throughput: identical ordered 16 starts, serial or chunked batches.

This measures solve calls including Python transfers after a warm-up, with a
fixed SQP iteration budget. It is not trajectory-quality or real-time evidence.
"""
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import pinocchio as pin
from batch_equivalence import ROOT, inputs, make_solver, sha, array_hash, dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--knots',type=int,required=True);p.add_argument('--duration',type=float,required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[1,2,4,8,16])
    p.add_argument('--iterations',type=int,default=20);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=1401);p.add_argument('--task',type=int,default=0)
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if any(b not in (1,2,4,8,16) for b in args.batches):p.error('Batches must divide the fixed16-start workload.')
    if args.iterations<1 or args.repeats<1:p.error('Positive iteration/repetition counts required.')
    if (args.output/'summary.json').exists():p.error('Completed output exists; preserve it and use a new directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    pool,meta=inputs(args,model);cfg=SimpleNamespace(dt=meta['dt'])
    module=importlib.import_module(f'bsqp.bsqpN{args.knots}_tiago_right_multimodal')
    records=[]
    # Alternate shape order so temporal drift is not systematically favorable
    # to larger batches. Every shape gets one untimed full-workload warm-up.
    solvers={b:make_solver(module,b,cfg,args.iterations,0.) for b in args.batches}
    def workload(b):
        solver=solvers[b];wall=0.;internal=0.;outputs=[];sqp=[];pcg_caps=0;pcg_breakdowns=0
        for offset in range(0,16,b):
            fed={k:np.ascontiguousarray(v[offset:offset+b]) for k,v in pool.items()}
            solver.reset_dual();solver.reset_rho();solver.set_f_ext_batch(fed['f_ext'])
            start=time.perf_counter()
            result=solver.solve(fed['XU'],cfg.dt,fed['x0'],fed['reference'])
            wall+=time.perf_counter()-start;internal+=float(result['sqp_time_us'])/1e6
            outputs.append(np.asarray(result['XU']).copy());sqp.extend(np.asarray(result['sqp_iters']).tolist())
            pcg_caps+=int(np.count_nonzero(np.asarray(result['pcg_status'])==0))
            pcg_breakdowns+=int(np.count_nonzero(np.asarray(result['pcg_status'])==2))
        values=np.concatenate(outputs)
        return dict(batch=b,starts=16,calls=16//b,solve_wall_seconds=wall,
                    solver_seconds=internal,finite_outputs=bool(np.isfinite(values).all()),
                    sqp_iterations=sqp,pcg_cap_events=pcg_caps,pcg_breakdown_events=pcg_breakdowns,
                    budget_counts_valid=bool(len(sqp)==16 and all(value==args.iterations for value in sqp)),
                    output_sha256=array_hash({'XU':values}))
    for b in args.batches:workload(b)
    for repeat in range(args.repeats):
        order=args.batches if repeat%2==0 else list(reversed(args.batches))
        for b in order:
            row=dict(repeat=repeat,**workload(b));records.append(row);print(json.dumps(row),flush=True)
    summary=[]
    for row in records:
        row['work_valid']=row['finite_outputs'] and row['budget_counts_valid'] and row['pcg_breakdown_events']==0
    serial=[r['solve_wall_seconds'] for r in records if r['batch']==1 and r['work_valid']]
    for b in args.batches:
        times=[r['solve_wall_seconds'] for r in records if r['batch']==b and r['work_valid']]
        median=float(np.median(times)) if times else None
        summary.append(dict(batch=b,valid_samples=len(times),total_samples=args.repeats,
                            median_seconds_for16_starts=median,min_seconds=min(times) if times else None,
                            max_seconds=max(times) if times else None,
                            median_speedup_vs_serial=float(np.median(serial))/median if serial and median else None))
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    dump(args.output/'summary.json',dict(config=config,records=records,summary=summary,
        input_sha256=meta['content_sha256'],binary=str(Path(module.__file__).resolve()),binary_sha256=sha(module.__file__),
        source_sha256=sha(__file__),all_finite=all(row['finite_outputs'] for row in records),
        all_budget_counts_valid=all(row['budget_counts_valid'] for row in records),
        verification='Throughput interpretation requires the separate matching batch-equivalence gate; this helper alone does not verify correctness.',
        scope='Matched16-start fixed-iteration solve throughput after warm-up. Transfers included; proposal generation, resets, and certification excluded. GPU contention must be assessed separately.'))


if __name__=='__main__':main()
