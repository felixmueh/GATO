#!/usr/bin/env python3
"""Matched CPU-reference timing matrix. Build once, preserve every fixed attempt.

This is Python-orchestrated CPU wall time, not synchronized native CUDA time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback

# Set before importing NumPy/Pinocchio in this process or its child workers.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOL = ROOT/'examples/solve_time_heatmap'
sys.path[:0] = [str(HERE), str(TOOL)]
BACKEND = 'CPU Pinocchio + QDLDL (Python-orchestrated, serial lanes)'
TIMING = 'Whole-batch CPU wall: Python, Pinocchio, NumPy and QDLDL included; initialization, plant integration, recorder and input/output binding copies excluded'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    from experiment import save_json
    save_json(Path(path), value)


def identity(cache):
    modules = list(cache.glob('cpu_qdldl*.so'))
    if len(modules) != 1:
        raise RuntimeError('Expected one CPU QDLDL extension')
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    wanted = [name for name in tracked if name.startswith(('experiments/pinocchio_qdldl_timing/',
              'examples/solve_time_heatmap/', 'tiago_src/gato_tiago/config.py',
              'tiago_src/gato_tiago/references.py', 'gato/dynamics/tiago_right/',
              'gato/bsqp/qdldl_host.h', 'cmake/qdldl.cmake'))]
    qdldl = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cache/'_deps/qdldl-src', text=True).strip()
    if qdldl != 'df481005111f2864770a3eccb05ed4120727eff6':
        raise RuntimeError('QDLDL source is not the pinned dependency')
    return dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_files={name:digest(ROOT/name) for name in wanted},
                module_path=str(modules[0].resolve()), module_sha256=digest(modules[0]),
                qdldl_source_commit=qdldl,
                cmake_cache_sha256=digest(cache/'CMakeCache.txt'),
                frozen_inputs_sha256=digest(HERE/'inputs.npz'))


def build(cache, python):
    cache.mkdir(parents=True, exist_ok=True)
    with (cache/'build.log').open('a', buffering=1) as log:
        commands = [['cmake', '-S', str(HERE), '-B', str(cache),
                     '-DCMAKE_BUILD_TYPE=Release', f'-DPython3_EXECUTABLE={python}'],
                    ['cmake', '--build', str(cache), '--parallel', '2']]
        for command in commands:
            log.write('COMMAND '+json.dumps(command)+'\n')
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def load_bridge(cache):
    sys.path.insert(0, str(cache))
    import cpu_qdldl
    return cpu_qdldl


def worker(args):
    import numpy as np
    import pinocchio as pin
    from cpu_solver import CPUSolver
    from experiment import Recorder, run_instrumented, NumericInstability
    from plants import plant_metadata
    from protocol import solver_parameters
    n, batch, repeat = args.cell
    out = args.raw/f'N{n}_B{batch}_R{repeat}'
    out.mkdir(parents=True, exist_ok=False)
    ident = identity(args.cache)
    bridge = load_bridge(args.cache)
    source = np.load(HERE/'inputs.npz', allow_pickle=False)
    start, reference = source['x_start'].copy(), source['reference'].copy()
    if hashlib.sha256(start.tobytes()).hexdigest() != '6c54a01a71e18d67edae8fda665c28704bc423e820eb559e0e1415901e7ad5dd':
        raise RuntimeError('Frozen initial state hash differs')
    if hashlib.sha256(reference.tobytes()).hexdigest() != '476e4225a3b3789279bcecfe83522ff5b47877fbb6998bf6c3a8223bbe6dcb90':
        raise RuntimeError('Frozen reference hash differs')
    shutil.copyfile(HERE/'inputs.npz', out/'inputs.npz')
    horizon = .03*(n-1)
    params = solver_parameters(n, horizon, 'tiago_right') | {'max_sqp_iters':1}
    model = pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    solver = CPUSolver(model, n, batch, params, bridge)
    meta = dict(**plant_metadata('tiago_right'), **ident, N=n, batch_size=batch,
        repeat=repeat, gpu_name='Jetson Orin CPU reference', backend=BACKEND,
        linsys_solver='QDLDL', protocol='wall-time', sim_step_s=.01,
        horizon_time=horizon, prediction_dt_s=.03, solver_params=params,
        precision=bridge.PRECISION, timing_scope=TIMING,
        power_mode='MODE_40W user-confirmed; no fresh query or changes',
        platform=platform.platform(), cpu_machine=platform.machine(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        cpu_model=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else None,
        thread_environment={k:v for k,v in os.environ.items() if k.endswith('NUM_THREADS') or k=='VECLIB_MAXIMUM_THREADS'},
        numpy_version=np.__version__, pinocchio_version=pin.__version__)
    record = Recorder(out, batch_size=batch, max_sqp_iters=1, max_solve_ms=0,
        wall_time=5., protocol='wall-time', meta=meta, updates=300)
    def interrupted(signum, frame):
        raise TimeoutError(f'Parent bounded worker: signal {signum}; partial measurements retained')
    signal.signal(signal.SIGTERM, interrupted)
    status, error, end = 'incomplete', None, None
    try:
        end, status = run_instrumented(solver, model, start, reference, record,
            plant='tiago_right', updates=300, max_solve_ms=0,
            initialization='single', horizon_time=horizon)
    except (NumericInstability, FloatingPointError):
        status, error = 'nonfinite', traceback.format_exc()
    except RuntimeError as exc:
        status = 'nonfinite' if str(exc).startswith('QDLDL nonfinite') else 'incomplete'
        error = traceback.format_exc()
    except (KeyboardInterrupt, TimeoutError):
        status, error = 'incomplete', traceback.format_exc()
    except Exception:
        status, error = 'incomplete', traceback.format_exc()
    finally:
        record.save(status, end_time=end, error=error)
        record.stream.close()
    return 0 if status == 'completed' else 1


def records(raw):
    result = []
    for path in sorted(raw.glob('N*_B*_R*/solves.jsonl')):
        n,b,r = [int(v[1:]) for v in path.parent.name.split('_')]
        rows=[]
        for line in path.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass  # Last incomplete flushed row stays raw.
        saved=path.parent/'results.json'
        data=json.loads(saved.read_text()) if saved.exists() else {}
        status='completed' if data.get('status')=='completed' and len(rows)==300 else 'nonfinite' if data.get('status')=='nonfinite' else 'incomplete'
        result.append(dict(N=n,B=b,repeat=r,status=status,rows=rows,error=data.get('error')))
    return result


def export(raw, output, manifest):
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap
    output.mkdir(parents=True, exist_ok=True)
    repeats=records(raw)
    grid=json.loads((HERE/'matrix_proposal.json').read_text())
    rows=[]
    for cell in grid['cells']:
        n,b=cell['N'],cell['B']
        group=[r for r in repeats if (r['N'],r['B'])==(n,b)]
        samples=[sample for r in group for sample in r['rows']]
        times=np.asarray([sample['native_ms'] for sample in samples],float)
        status='completed' if len(group)==2 and all(r['status']=='completed' for r in group) else 'nonfinite' if any(r['status']=='nonfinite' for r in group) else 'incomplete'
        stats=dict(mean_ms=float(times.mean()),median_ms=float(np.median(times)),
            p95_ms=float(np.percentile(times,95)),sd_ms=float(times.std())) if len(times) else dict.fromkeys(['mean_ms','median_ms','p95_ms','sd_ms'],'')
        rows.append(dict(gpu='Jetson Orin CPU reference',gpu_uuid='',sm_count='',N=n,B=b,knot_spacing_s=.03,
            status=status,repeats=len(group),samples=len(samples),**stats,
            mean_sqp_iters=float(np.mean([s['sqp_iters'] for s in samples])) if samples else '',
            pcg_cap_hit_share='',pcg_iters_mean='',shmem_optin_applied='',
            source_commit=manifest['identity']['source_commit'],module_sha256=manifest['identity']['module_sha256']))
    with (output/'cells.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=grid['output_columns']);w.writeheader();w.writerows(rows)
    write_json(output/'repeat_status.json',[{k:v for k,v in r.items() if k!='rows'} for r in repeats])
    write_json(output/'metadata.json',manifest)
    shutil.copyfile(HERE/'matrix_proposal.json',output/'protocol.json')
    fig,ax=plt.subplots(figsize=(8,5))
    image=np.full((6,5),np.nan)
    for row in rows:
        if row['status']=='completed': image[grid['knots'].index(row['N']),grid['batches'].index(row['B'])]=row['median_ms']
    finite=image[np.isfinite(image)]
    if finite.size:
        from matplotlib.colors import LogNorm
        im=ax.imshow(image,norm=LogNorm(vmin=max(float(finite.min()),1e-9),vmax=max(float(finite.max()),float(finite.min())*1.01)))
        fig.colorbar(im,ax=ax,label='Whole-batch CPU wall median (ms)')
    else: ax.imshow(np.zeros((6,5)),cmap='Greys',vmin=0,vmax=1)
    for row in rows:
        i,j=grid['knots'].index(row['N']),grid['batches'].index(row['B'])
        label=f"{row['median_ms']:.2f}" if row['status']=='completed' else row['status']
        ax.text(j,i,label,ha='center',va='center',fontsize=8)
    ax.set(xticks=range(5),xticklabels=grid['batches'],yticks=range(6),yticklabels=grid['knots'],
           xlabel='Serial CPU lanes B',ylabel='State knots N',title='Pinocchio + QDLDL Python CPU reference')
    fig.tight_layout();fig.savefig(output/'quicklook.png',dpi=150);fig.savefig(output/'quicklook.pdf')
    counts={key:sum(r['status']==key for r in rows) for key in ['completed','nonfinite','incomplete']}
    report=f'''# Jetson Pinocchio + QDLDL CPU-reference timing matrix

Status: {counts}. This is a Python-orchestrated CPU reference. Whole-batch CPU
wall includes Python, Pinocchio, NumPy, Schur assembly, QDLDL and merit search.
It is not synchronized native GATO timing or an optimized CPU implementation.

All 30 cells request 300 updates per repeat, two repeats, one excluded initial
solve, SQP cap 1, 30 ms prediction knots, 10 ms control and at most 1 ms RK4
plant steps. Lanes are solved independently and serially, even with identical
warm starts. The frozen initial-state/reference bytes match the hybrid matrix.
Only first-state reanchoring and unshifted batch-0 warm broadcasting are used.
Mode A does not charge solve time to the plant. Each repeat has a 180 s bound.
Incomplete/nonfinite measurements and partial samples are retained, never
retried or represented as completed heatmap cells.

Pinocchio/NumPy assembly uses float64, then the historical native QDLDL bridge
casts to float32 and retains the full-row residual check. Prediction uses the
native type-2 trapezoidal map; the executed plant uses RK4. Historical clipped
log penalties, Gauss-Newton tracking, Q-only rho regularization and eight
global-minimum merit candidates are retained. No C2 continuation/interior
bounds, altered tracking preset or new KKT acceptance gate is introduced.

MODE_40W is user-confirmed, not freshly measured; no mode/clock changes.
CPU thread-library environment is fixed to one; effective affinity and package
identities are in raw per-repeat results. GPU-only fields are blank.
Source: {manifest['identity']['source_commit']}.
CPU QDLDL extension SHA256: {manifest['identity']['module_sha256']}.

![Completed-cell median CPU wall times; failures retain their labels.](quicklook.png)

| N | B | Status | Samples | Mean ms | Median ms | p95 ms |
|---:|---:|---|---:|---:|---:|---:|
'''
    for row in rows:
        report+='| '+ ' | '.join(str(row[k]) for k in ['N','B','status','samples','mean_ms','median_ms','p95_ms'])+' |\n'
    report+='''
Comparisons to the saved GPU PCG and hybrid QDLDL cohorts are descriptive:
precision, implementation overhead and evolving closed-loop inputs differ.
No equal-work, best-CPU-performance, task-success or physical-safety claim is
made. Cells.csv retains partial summary statistics for diagnostics; only
completed cells are plotted. All raw samples/logs remain outside this compact
export. Re-export with the pinned run_matrix.py --export-only command against
the saved raw directory; no measurements are performed by that command.
'''
    (output/'REPORT.md').write_text(report)
    # Self-contained PDF: prose, full 30-cell table and quicklook, no local links.
    with PdfPages(output/'REPORT.pdf') as pdf:
        prose=report.split('![')[0]
        page=plt.figure(figsize=(8.27,11.69));page.text(.06,.95,'\n'.join(textwrap.fill(line,92) if line else '' for line in prose.splitlines()),va='top',fontsize=9);pdf.savefig(page);plt.close(page)
        pdf.savefig(fig)
        page=plt.figure(figsize=(8.27,11.69));axis=page.add_axes([.04,.08,.92,.84]);axis.axis('off')
        table=axis.table(cellText=[[str(row[k]) if not isinstance(row[k],float) else f'{row[k]:.3f}' for k in ['N','B','status','samples','mean_ms','median_ms','p95_ms']] for row in rows],
            colLabels=['N','B','Status','Samples','Mean ms','Median ms','p95 ms'],loc='center');table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.35)
        page.text(.06,.97,'All cells; partial statistics do not imply completion',va='top',fontsize=12)
        page.text(.06,.045,'Python CPU reference; different arithmetic and orchestration from native CUDA GATO.\nNo task, physical-safety, equal-work or best-CPU-performance qualification.',fontsize=9)
        pdf.savefig(page);plt.close(page)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only',choices=['cpu-pinocchio-qdldl'],required=True)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--export-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--cell',type=int,nargs=3,help=argparse.SUPPRESS)
    args=p.parse_args()
    for key in ['raw','cache','export']:setattr(args,key,getattr(args,key).resolve())
    if args.cell:return worker(args)
    grid=json.loads((HERE/'matrix_proposal.json').read_text())
    cells=[(c['N'],c['B'],r) for c in grid['cells'] for r in range(2)]
    if args.dry_run:
        print(json.dumps(dict(backend=BACKEND,timing=TIMING,only='cpu-pinocchio-qdldl',cells=cells,
            updates=300,timeout_s=180,initialization='single excluded',GPU_operations=0),indent=2));return 0
    manifest_path=args.raw/'manifest.json'
    if args.export_only:
        export(args.raw,args.export,json.loads(manifest_path.read_text()));return 0
    build(args.cache,sys.executable)
    with (args.cache/'functional-tests.log').open('w') as log:
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_cpu_reference.py'],
            cwd=ROOT,check=True,env=os.environ|{'CPU_QDLDL_CACHE':str(args.cache)},
            stdout=log,stderr=subprocess.STDOUT)
    ident=identity(args.cache)
    if args.prepare_only:
        print(json.dumps(dict(status='PREPARED_NOT_RUN',identity=ident,CPU_tests='passed',GPU_operations=0),indent=2));return 0
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
    if dirty.strip():raise RuntimeError('Refuse measurements with dirty tracked sources')
    if manifest_path.exists():
        if not args.resume:raise RuntimeError('Existing cohort requires --resume; failures are never retried')
        manifest=json.loads(manifest_path.read_text())
        if manifest['identity']!=ident or manifest['cache']!=str(args.cache):raise RuntimeError('Source/module/cache changed; preserve old cohort')
        if manifest['status']=='shared_setup_failed':raise RuntimeError('Shared setup failure retained; review before any new launch')
    else:
        if args.raw.exists():raise RuntimeError('Raw directory exists without manifest; preserve it')
        args.raw.mkdir(parents=True)
        manifest=dict(status='running',identity=ident,cache=str(args.cache),backend=BACKEND,timing_scope=TIMING,
            power_mode='MODE_40W user-confirmed; no fresh query',started=time.time(),results=[],
            worker_timeout_s=180,updates=300,repeats=2,thread_environment={k:v for k,v in os.environ.items() if k.endswith('NUM_THREADS') or k=='VECLIB_MAXIMUM_THREADS'})
        shutil.copyfile(HERE/'inputs.npz',args.raw/'frozen_inputs.npz')
        shutil.copyfile(args.cache/'CMakeCache.txt',args.raw/'cpu-build-CMakeCache.txt')
        shutil.copyfile(args.cache/'build.log',args.raw/'cpu-build.log')
        write_json(manifest_path,manifest)
    try:
        setup_failed=False
        for n,b,r in cells:
            if any((v['N'],v['B'],v['repeat'])==(n,b,r) for v in manifest['results']):continue
            destination=args.raw/f'N{n}_B{b}_R{r}'
            status,code='incomplete',None
            if not destination.exists():
                command=[sys.executable,str(Path(__file__).resolve()),'--only',args.only,
                    '--raw',str(args.raw),'--cache',str(args.cache),'--export',str(args.export),
                    '--cell',str(n),str(b),str(r)]
                with (args.raw/f'N{n}_B{b}_R{r}.log').open('w') as log:
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    try:code=process.wait(timeout=180)
                    except (subprocess.TimeoutExpired,KeyboardInterrupt):
                        process.terminate()
                        try:code=process.wait(timeout=10)
                        except subprocess.TimeoutExpired:process.kill();code=process.wait()
                        if sys.exc_info()[0] is KeyboardInterrupt:raise
                saved=destination/'results.json'
                if saved.exists():status=json.loads(saved.read_text()).get('status','incomplete')
                elif code not in (0,-signal.SIGTERM,-signal.SIGKILL):setup_failed=True
            manifest['results'].append(dict(N=n,B=b,repeat=r,status=status,returncode=code))
            write_json(manifest_path,manifest)
            print(f'N{n}/B{b}/repeat{r}: {status}',flush=True)
            if setup_failed:
                manifest['status']='shared_setup_failed'
                break
        if not setup_failed:manifest['status']='finished'
    except KeyboardInterrupt:
        manifest['status']='interrupted'
    finally:
        manifest['finished']=time.time();write_json(manifest_path,manifest)
        export(args.raw,args.export,manifest)
    return 0 if manifest['status']=='finished' and all(v['status']=='completed' for v in manifest['results']) else 1


if __name__=='__main__':
    raise SystemExit(main())
