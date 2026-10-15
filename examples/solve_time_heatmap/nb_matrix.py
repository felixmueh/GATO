"""Only the reviewed fixed-300-update TIAGo N x B matrix; no legacy queue."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CELLS=HERE/'nb_cells.json'
ARCH={'gtx1070':'61','rtx5060':'120','jetson':'87','h100':'90'}
LABEL={'gtx1070':'GTX1070','rtx5060':'RTX5060','jetson':'Jetson Orin','h100':'H100'}

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def spec():
    value=json.loads(CELLS.read_text())
    assert [(c['N'],c['B']) for c in value['cells']]==[(n,b) for n in [16,32,64,128,256,512] for b in [1,2,4,8,16]]
    assert (value['updates'],value['repeats'],value['SQP'],value['PCG'],value['worker_timeout_s'])==(300,2,1,120,180)
    return value

def ordered_cells(value):
    # These are original grid cells, never additional smoke attempts.
    return sorted(value['cells'],key=lambda c:(0 if (c['N'],c['B'])==(32,1) else 1 if (c['N'],c['B'])==(512,1) else 2,c['N'],c['B']))

def tool():
    module_spec=importlib.util.spec_from_file_location('nb_heatmap_tool',HERE/'run.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return module

def arguments(t,cell,args):
    ns=t.parser().parse_args(cell['args']+['--backend',getattr(args,'backend','PCG'),'--gpu-name',LABEL[args.gpu],
        '--build-dir',str(args.cache),'--output',str(args.raw)])
    ns.output=args.raw;ns.build_dir=args.cache
    return ns

def command(cmd,log):
    with log.open('a') as f:
        f.write('COMMAND '+json.dumps(cmd)+'\n');f.flush()
        subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def prepare_build(args,value):
    args.cache.mkdir(parents=True,exist_ok=True);log=args.raw/'build.log'
    configure=['cmake','-S',str(HERE),'-B',str(args.cache),'-DPLANT=tiago_right',
        '-DKNOTS='+(';'.join(map(str,value['knots']))),'-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CUDA_ARCHITECTURES='+ARCH[args.gpu]+'-real','-DPython3_EXECUTABLE='+sys.executable,'-DGATO_LINSYS_SOLVER='+getattr(args,'backend','PCG')]
    command(configure,log)
    command(['cmake','--build',str(args.cache),'--parallel',str(args.jobs),'--target']+
            ['resources_N'+str(n) for n in value['knots']],log)
    resources={}
    for n in value['knots']:
        result=subprocess.run([str(args.cache/f'resources_N{n}')],capture_output=True,text=True)
        (args.raw/f'resources_N{n}.stdout.json').write_text(result.stdout)
        (args.raw/f'resources_N{n}.stderr.log').write_text(result.stderr)
        result.check_returncode();r=json.loads(result.stdout)
        if r['plant']!='tiago_right' or r['N']!=n:raise ValueError('Wrong resource probe identity')
        tool().validate_backend_identity(r.get('linsys_solver','PCG'),getattr(args,'backend','PCG'))
        if r.get('kernel_launches')!=0:raise ValueError('Resource probe must launch zero kernels')
        if r.get('compute_capability','').replace('.','')!=ARCH[args.gpu]:raise ValueError('Target GPU architecture differs')
        resources[str(n)]=r
    write(args.raw/'resources.json',resources)
    supported=[n for n in value['knots'] if all(k['fits'] for k in resources[str(n)]['kernels'])]
    if supported:
        command(['cmake','--build',str(args.cache),'--parallel',str(args.jobs),'--target']+
                [f'bsqpN{n}_tiago_right' for n in supported],log)
    modules={}
    for n in supported:
        paths=list((args.cache/'modules').glob(f'bsqpN{n}_tiago_right*.so'))
        if len(paths)!=1:raise ValueError(f'N{n} needs exactly one matching module')
        modules[str(n)]={'path':str(paths[0].resolve()),'sha256':sha(paths[0])}
    return resources,modules

def run(args):
    value=spec()
    if args.dry_run:
        print('Backend:',getattr(args,'backend','PCG'),'CPU QDLDL Schur/GPU remainder' if getattr(args,'backend','PCG')=='QDLDL' else 'GPU PCG')
        print('Retained N32/B1:',getattr(args,'reuse_n32_b1',None),'(two workers skipped when supplied)')
        print('Common cells SHA256:',sha(CELLS))
        print('Resource probes first:',value['knots'],'architecture',ARCH[args.gpu])
        print('Build numerical modules only where every resource row fits')
        for c in ordered_cells(value):print('N',c['N'],'B',c['B'],'T',c['horizon_time_s'],'args',json.dumps(c['args']))
        return 0
    if getattr(args,'reuse_n32_b1',None) and getattr(args,'backend','PCG')!='QDLDL':raise ValueError('Retained N32/B1 is only for QDLDL')
    if os.environ.get('CUDA_LAUNCH_BLOCKING','0')!='0':raise ValueError('Unset CUDA_LAUNCH_BLOCKING')
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=all','--','gato','python','examples/solve_time_heatmap','tiago_src/gato_tiago'],cwd=ROOT,text=True)
    if dirty:raise ValueError('Commit all measurement sources before execution: '+dirty)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    args.raw.mkdir(parents=True,exist_ok=True)
    if args.power_mode_file:
        dest=args.raw/'_host_conditions'/args.power_mode_file.name;dest.parent.mkdir(exist_ok=True)
        if dest.resolve()!=args.power_mode_file.resolve():shutil.copyfile(args.power_mode_file,dest)
    if args.export.is_relative_to(args.raw):raise ValueError('Compact export must stay outside raw')
    path=args.raw/'nb-results.json';t=tool()
    if path.exists():
        m=json.loads(path.read_text())
        if not args.resume:raise ValueError('Existing matrix requires --resume; failed attempts are never retried')
        if m['source_commit']!=head or m['cells_sha256']!=sha(CELLS) or m['gpu']!=args.gpu or m['cache']!=str(args.cache):raise ValueError('Resume source/config/cache identity differs')
        if m.get('backend','PCG')!=getattr(args,'backend','PCG'):raise ValueError('Resume backend differs')
        if m.get('retained_n32_b1'):
            from qdldl_retained import validate_copy
            validate_copy(args.raw,m['retained_n32_b1'])
        if len(m['resources'])!=6:raise ValueError('Setup/probe failure is retained; review it before any new launch')
        for mod in m['modules'].values():
            if sha(mod['path'])!=mod['sha256']:raise ValueError('Resume native module differs')
    else:
        if args.resume:raise ValueError('--resume requires existing nb-results.json')
        unexpected=[p for p in args.raw.iterdir() if p.name!='_host_conditions']
        if unexpected:raise ValueError('New raw root may contain only prior host conditions')
        m=dict(schema='tiago-nb-matrix-fixed300-v1',backend=getattr(args,'backend','PCG'),gpu=args.gpu,source_commit=head,cells_sha256=sha(CELLS),
            cells=value,cache=str(args.cache),started_at=time.time(),results=[],resources={},modules={},status='building')
        if getattr(args,'reuse_n32_b1',None):
            from qdldl_retained import inspect, RELATIVE_ROOT
            record=inspect(args.reuse_n32_b1)
            shutil.copytree(args.reuse_n32_b1,args.raw/RELATIVE_ROOT)
            # Archive includes original bytes; provenance remains the original
            # source/module, never the newly built N32 module.
            record['original_path']=str(args.reuse_n32_b1)
            m['retained_n32_b1']=record
        m['planned_new_repeat_slots']=58 if m.get('retained_n32_b1') else 60
        m['retained_repeat_count']=2 if m.get('retained_n32_b1') else 0
        write(path,m)
        snapshot=args.raw/'source_snapshot'
        for p in [*t.source_paths(),CELLS]:
            dest=snapshot/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        try:m['resources'],m['modules']=prepare_build(args,value)
        except Exception as exc:
            m.update(status='build_or_probe_failed',error=str(exc));write(path,m);raise
        m['gpu_metadata']=t.gpu_snapshot(resource=m['resources']['16'])
        m['build_cache']=(args.cache/'CMakeCache.txt').read_text();write(path,m)
    done={(r['N'],r['batch_size'],r['repeat']) for r in m['results']}
    if m.get('retained_n32_b1'):done.update({(32,1,0),(32,1,1)})
    for c in ordered_cells(value):
        if args.smoke_only and (c['N'],c['B']) not in [(32,1),(512,1)]:continue
        for repeat in range(value['repeats']):
            key=c['N'],c['B'],repeat
            if key in done:continue
            resource=m['resources'][str(c['N'])]
            if not all(k['fits'] for k in resource['kernels']):
                row=dict(N=c['N'],batch_size=c['B'],repeat=repeat,status='unsupported_shared_memory',samples=0)
            else:
                folder=args.raw/f'N{c["N"]}_B{c["B"]}_R{repeat}'
                # Never recycle an interrupted or failed attempt automatically.
                if folder.exists():
                    receipt=folder/'results.json'
                    row=json.loads(receipt.read_text()) if receipt.exists() else dict(N=c['N'],batch_size=c['B'],repeat=repeat,samples=0,status='interrupted',error='Pre-existing partial worker retained; no retry')
                else:row=t.run_cell_process(arguments(t,c,args),c['N'],c['B'],repeat)
            m['results'].append(row);done.add(key);m['status']='running';write(path,m)
            print(f'N{c["N"]} B{c["B"]} R{repeat}: {row["status"]}, samples={row.get("samples",0)}',flush=True)
            from nb_export import export
            export(args.raw,args.export)
    m.update(status='completed' if len(done)==60 else 'partial',finished_at=time.time())
    write(path,m)
    from nb_export import export
    export(args.raw,args.export,final=True)
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only',choices=['nb-matrix'],required=True)
    p.add_argument('--gpu',choices=ARCH,required=True)
    p.add_argument('--backend',choices=['PCG','QDLDL'],default='PCG')
    p.add_argument('--reuse-n32-b1',type=Path,help='Original completed QDLDL run.py output root; archive unchanged and skip its two workers')
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--power-mode-file',type=Path)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    for key in ['raw','cache','export']:setattr(args,key,getattr(args,key).resolve())
    if args.reuse_n32_b1:args.reuse_n32_b1=args.reuse_n32_b1.resolve()
    if args.jobs<1:p.error('--jobs must be positive')
    try:return run(args)
    except (ValueError,subprocess.CalledProcessError) as exc:p.error(str(exc))

if __name__=='__main__':raise SystemExit(main())
