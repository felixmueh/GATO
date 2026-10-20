#!/usr/bin/env python3
"""Prepare/run existing public PCG preconditioners at 128/256/512 knots on Jetson."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from jetson_experiment import ROOT, base_metadata, environment, locks, monitor
from upstream_reach import sha256, write_json
from verification_worker import native_path

METHODS='normal,gpu_polynomial2,gpu_polynomial3,cpu_native_control,block_sgs,schwarz,incomplete_cholesky,block_jacobi,polynomial2,polynomial3,block_ldlt'
CASES=['reach-seed'+str(n) for n in range(2,8)]+['tracking-seed'+str(n) for n in (11,12,13)]

def plan(args):
    methods=args.methods.split(','); knots=[int(n) for n in args.knots.split(',')]
    if not methods or len(set(methods))!=len(methods) or set(methods)-set(METHODS.split(',')+['dummy']):
        raise ValueError('Choose distinct existing explicit-copy PCG methods; shared/direct/AMG methods are outside this cohort')
    if not knots or len(set(knots))!=len(knots) or set(knots)-{128,256,512}: raise ValueError('Choose distinct knot counts from128,256,512')
    cases=args.cases.split(',') if args.cases else CASES if args.full else ['reach-seed2','tracking-seed11']
    if not cases or set(cases)-set(CASES) or args.repeats<1: raise ValueError('Invalid cases/repeats')
    if args.preflight:
        cases=[next(c for c in cases if c.startswith(prefix)) for prefix in ('reach-','tracking-') if any(c.startswith(prefix) for c in cases)]
    repeats=1 if args.preflight else args.repeats
    jobs=[dict(knots=n,case_id=c,method=m,repeat=r) for n in knots for c in cases for r in range(repeats) for m in (methods if r%2==0 else methods[::-1])]
    # InitialN128 retained successful runs took0.4-0.8s compute. These planning
    # allowances include startup/copies/largerN; target preflight supersedes them.
    return dict(protocol='public-jetson-preconditioner-scaling-v1', knots=knots, methods=methods,cases=cases,repeats=repeats,
        process_count=len(jobs),jobs=jobs,primary_result_field='sqp_iteration_compute_mean_ms',primary_timing='sqp_iteration_compute_mean_ms: full native BSQP B1/SQP1 wall per attempt minus only in-interval native/candidate H2D/D2H copy walls',
        secondary_timing='per-SQP wall, per-SQP compute median/p95 and zero/nonzero-PCG strata; whole-task compute/wall retained',
        dt_s=.01,prediction_duration_s={str(n):(n-1)*.01 for n in knots},
        estimate=dict(status='rough planning allowance, not measured Jetson runtime; failed cases can consume full task budget',
            minutes_range=[len(jobs)*5/60,len(jobs)*30/60],raw_MiB_range=[len(jobs)*2,len(jobs)*20],
            basis='retained GTX1070N128 successful episode compute0.4-0.8s plus startup/copies, conservative largerN allowance; builds separate'),
        preflight=args.preflight)

def build(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    metadata=base_metadata(args.checkout);metadata.update(state='preparing',knots=args.knots.split(','))
    write_json(output/'build.json',metadata)
    cmake=shutil.which('cmake')
    if not cmake: raise RuntimeError('CMake unavailable in active environment')
    pybind=subprocess.check_output([sys.executable,'-m','pybind11','--cmakedir'],text=True).strip()
    modules='indy7:'+args.knots
    build_dir=args.build_dir.resolve() if args.build_dir else args.checkout/'build_precond_scaling'
    commands=[[cmake,'-S',str(args.checkout),'-B',str(build_dir),'-DCMAKE_BUILD_TYPE=Release','-DGATO_BUILD_DEMO=OFF',
        '-DMODULES='+modules,'-DPRECOND_MANAGED_S=OFF','-DPython3_EXECUTABLE='+sys.executable,'-Dpybind11_DIR='+pybind,
        '-DCMAKE_CUDA_ARCHITECTURES='+args.arch],[cmake,'--build',str(build_dir),'--parallel',str(args.jobs)]]
    metadata['commands']=commands; held=locks(args.checkout)
    try:
        previous=[]
        for n in args.knots.split(','):
            for path in sorted((args.checkout/'python/gato').glob(f'bsqpN{n}_indy7*.so')):
                saved=output/'previous_modules'/path.name;saved.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,saved)
                digest=sha256(saved)
                if digest!=sha256(path):raise RuntimeError(f'Previous-module archive differs from source: {path}')
                previous.append(dict(path=str(path),saved_path=str(saved),sha256=digest))
        metadata['previous_modules']=previous;write_json(output/'build.json',metadata)
        with (output/'stdout.log').open('w') as out,(output/'stderr.log').open('w') as err:
            for cmd in commands: subprocess.run(cmd,cwd=args.checkout,env=environment(args.checkout),stdout=out,stderr=err,check=True)
        metadata.update(state='complete',modules={n:dict(path=str(native_path(args.checkout,int(n))),sha256=sha256(native_path(args.checkout,int(n)))) for n in args.knots.split(',')})
    except subprocess.CalledProcessError as exc:
        metadata.update(state='failed',error=repr(exc),failed_command=exc.cmd,returncode=exc.returncode)
        print(f"Build failed (exit {exc.returncode}); full logs: {output}",file=sys.stderr)
        for name in ('stdout.log','stderr.log'):
            path=output/name
            print(f"\n{name} (last 40 lines):",file=sys.stderr)
            print('\n'.join(path.read_text(errors='replace').splitlines()[-40:]),file=sys.stderr)
        return exc.returncode if exc.returncode>0 else 1
    except BaseException as exc: metadata.update(state='failed',error=repr(exc));raise
    finally:
        write_json(output/'build.json',metadata)
        for handle in held:handle.close()

def run(args):
    planned=plan(args)
    if args.dry_run:
        print(json.dumps(planned,indent=2));return 0
    if not args.tasks:raise ValueError('--tasks is required unless --dry-run')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    metadata=base_metadata(args.checkout);metadata.update(state='preparing',plan=planned)
    write_json(output/'launch.json',metadata)
    held=locks(args.checkout);process=thread=error=None;statuses=[]
    try:
        process,thread,error=monitor(output)
        for n in planned['knots']:
            cmd=[sys.executable,str(Path(__file__).with_name('verification_worker.py')),'campaign','--checkout',str(args.checkout),
                '--tasks',str(args.tasks.resolve()),'--output',str(output/f'N{n}'),'--knots',str(n),'--primary-timing','compute_excluding_transfers',
                '--cases',','.join(planned['cases']),'--methods',','.join(planned['methods']),'--repeats',str(planned['repeats'])]
            if args.preflight: cmd.append('--preflight')
            with (output/f'N{n}.stdout').open('w') as out,(output/f'N{n}.stderr').open('w') as err:
                start=time.monotonic();result=subprocess.run(cmd,cwd=args.checkout,env=environment(args.checkout),stdout=out,stderr=err)
            statuses.append(dict(knots=n,returncode=result.returncode,wall_s=time.monotonic()-start,command=cmd));write_json(output/'status.json',statuses)
            print(f"N{n}: campaign exit {result.returncode}; logs {output/f'N{n}.stdout'} and {output/f'N{n}.stderr'}",flush=True)
            if result.returncode:
                status_file=output/f'N{n}'/'status.json'
                attempts=json.loads(status_file.read_text()) if status_file.exists() else []
                logs=[output/f'N{n}.stderr',output/f'N{n}.stdout']
                for attempt in attempts:
                    if attempt['returncode']:
                        name=f"{attempt['case_id']}-{attempt['method']}-r{attempt['repeat']}"
                        logs.extend(output/f'N{n}'/(name+suffix) for suffix in ('.stderr','.stdout'))
                for path in logs:
                    lines=path.read_text(errors='replace').splitlines()
                    if lines: print(f"\n{path} (last 20 lines):\n"+'\n'.join(lines[-20:]),file=sys.stderr,flush=True)
        metadata.update(state='complete' if all(s['returncode']==0 for s in statuses) else 'completed_with_failures')
        return int(any(s['returncode'] for s in statuses))
    except BaseException as exc:metadata.update(state='failed',error=repr(exc));raise
    finally:
        if process:
            process.terminate();process.wait();thread.join(timeout=2);error.close()
            metadata['clock_monitor_returncode']=process.returncode
            sample_file=output/'clock-samples.jsonl'
            metadata['clock_monitor_raw_line_count']=sum(1 for _ in sample_file.open()) if sample_file.exists() else 0
        metadata['clock_monitor_attempted']=process is not None
        write_json(output/'launch.json',metadata)
        for handle in held:handle.close()

def summarize(args):
    """One CSV row per independent process; retain failures and zero-PCG strata."""
    root=args.input.resolve();rows=[]
    launch=root/'launch.json'
    jobs=json.loads(launch.read_text()).get('plan',{}).get('jobs',[]) if launch.exists() else []
    if not jobs:
        for path in sorted(root.glob('N*/*/summary.json')):
            config=json.loads(path.with_name('config.json').read_text())
            jobs.append(dict(knots=config['knots'],method=config['method'],case_id=config.get('task_id'),repeat=int(path.parent.name.rsplit('-r',1)[1])))
    status=root/'status.json'
    horizon_status={r['knots']:r['returncode'] for r in json.loads(status.read_text())} if status.exists() else {}
    for job in jobs:
        name=f"{job['case_id']}-{job['method']}-r{job['repeat']}"
        horizon=root/f"N{job['knots']}";path=horizon/name/'summary.json'
        summary=json.loads(path.read_text()) if path.exists() else {}
        status=horizon/'status.json'
        attempts=json.loads(status.read_text()) if status.exists() else []
        attempt=next((r for r in attempts if (r['case_id'],r['method'],r['repeat'])==(job['case_id'],job['method'],job['repeat'])),{})
        error=summary.get('error') or summary.get('sqp_iteration_timing_error')
        if not path.exists():error='No summary produced; inspect saved horizon/process stderr logs'
        elif attempt.get('returncode'):error=error or f"Process exited with code {attempt['returncode']}"
        row=dict(knots=job['knots'],method=job['method'],case_id=job['case_id'],repeat=job['repeat'],
            process_id=name,error=error,summary_present=path.exists(),
            process_returncode=attempt.get('returncode'),horizon_returncode=horizon_status.get(job['knots']),
            stderr_path=str(horizon/(name+'.stderr')) if (horizon/(name+'.stderr')).exists() else str(root/f"N{job['knots']}.stderr"),
            all_goals_reached=summary.get("all_goals_reached"),
            tracking_completed=(summary.get("tracking") or {}).get("completed_requested_duration"))
        for key in ("sqp_iteration_count","sqp_iteration_compute_mean_ms","sqp_iteration_compute_median_ms",
                    "sqp_iteration_compute_p95_ms","sqp_iteration_wall_mean_ms","sqp_iteration_wall_median_ms",
                    "sqp_iteration_wall_p95_ms","sqp_iteration_transfer_mean_ms","pcg_iterations_total",
                    "whole_task_compute_us","whole_task_wall_us","realized_physical_duration_s"):
            row[key]=summary.get(key)
        for kind in ("zero_pcg","nonzero_pcg"):
            group=summary.get("sqp_iteration_"+kind,{})
            for key in ("count","compute_mean_ms","wall_mean_ms"):
                row[kind+"_"+key]=group.get(key)
        rows.append(row)
    if not rows:raise ValueError("No planned jobs or N*/<process>/summary.json files found; no table written")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(output=str(args.output),process_rows=len(rows),missing_summaries=sum(not r['summary_present'] for r in rows),scope="One row per scheduled process, including missing summaries; do not estimate repeat uncertainty from serial SQP samples")))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('build','freeze','run','summarize'):
        s=sub.add_parser(action);s.add_argument('--checkout',type=Path,default=ROOT);s.add_argument('--output',type=Path,required=True)
        s.add_argument('--knots',default='128,256,512')
        if action=='build':
            s.add_argument('--jobs',type=int,default=1);s.add_argument('--arch',default='native');s.add_argument('--build-dir',type=Path)
        elif action=='summarize':s.add_argument('--input',type=Path,required=True)
        elif action=='run':
            s.add_argument('--tasks',type=Path);s.add_argument('--methods',default=METHODS);s.add_argument('--cases');s.add_argument('--full',action='store_true')
            s.add_argument('--repeats',type=int,default=3);s.add_argument('--preflight',action='store_true');s.add_argument('--dry-run',action='store_true')
    args=p.parse_args();args.checkout=args.checkout.resolve()
    knot_counts=[int(n) for n in args.knots.split(',')]
    if not knot_counts or len(set(knot_counts))!=len(knot_counts) or set(knot_counts)-{128,256,512}:
        p.error('--knots must be distinct values from128,256,512')
    if args.action=='summarize':return summarize(args)
    if args.action=='freeze':
        return subprocess.call([sys.executable,str(Path(__file__).with_name('verification_worker.py')),'freeze','--checkout',str(args.checkout),
            '--output',str(args.output.resolve()),'--knots',args.knots],env=environment(args.checkout))
    return (build if args.action=='build' else run)(args) or 0
if __name__=='__main__':raise SystemExit(main())
