#!/usr/bin/env python3
"""Prepare/run existing public PCG preconditioners at 128/256/512 knots on Jetson."""
from __future__ import annotations
import argparse
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
        process_count=len(jobs),jobs=jobs,primary_timing='whole_task_wall_us - all measured explicit H2D/D2H copy wall',
        secondary_timing='actual whole_task_wall_us; instrumented and includes measurement fences',
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
        with (output/'stdout.log').open('w') as out,(output/'stderr.log').open('w') as err:
            for cmd in commands: subprocess.run(cmd,cwd=args.checkout,env=environment(args.checkout),stdout=out,stderr=err,check=True)
        metadata.update(state='complete',modules={n:dict(path=str(native_path(args.checkout,int(n))),sha256=sha256(native_path(args.checkout,int(n)))) for n in args.knots.split(',')})
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

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('build','freeze','run'):
        s=sub.add_parser(action);s.add_argument('--checkout',type=Path,default=ROOT);s.add_argument('--output',type=Path,required=True)
        s.add_argument('--knots',default='128,256,512')
        if action=='build':
            s.add_argument('--jobs',type=int,default=1);s.add_argument('--arch',default='native');s.add_argument('--build-dir',type=Path)
        elif action=='run':
            s.add_argument('--tasks',type=Path);s.add_argument('--methods',default=METHODS);s.add_argument('--cases');s.add_argument('--full',action='store_true')
            s.add_argument('--repeats',type=int,default=3);s.add_argument('--preflight',action='store_true');s.add_argument('--dry-run',action='store_true')
    args=p.parse_args();args.checkout=args.checkout.resolve()
    knot_counts=[int(n) for n in args.knots.split(',')]
    if not knot_counts or len(set(knot_counts))!=len(knot_counts) or set(knot_counts)-{128,256,512}:
        p.error('--knots must be distinct values from128,256,512')
    if args.action=='freeze':
        return subprocess.call([sys.executable,str(Path(__file__).with_name('verification_worker.py')),'freeze','--checkout',str(args.checkout),
            '--output',str(args.output.resolve()),'--knots',args.knots],env=environment(args.checkout))
    return (build if args.action=='build' else run)(args) or 0
if __name__=='__main__':raise SystemExit(main())
