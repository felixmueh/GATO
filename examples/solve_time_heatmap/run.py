"""Run the Indy7 solve-time matrix with one SQP step or native stopping.

The default uses fixed 10 ms simulation updates and a wall-time budget per cell.
Use --protocol reference to reproduce the original capped simulation clock.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
KNOTS = (8, 16, 32, 64, 128, 256)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_snapshot():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,uuid,driver_version,temperature.gpu,'
        'utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw',
        '--format=csv'], text=True).strip()


def match_torch_libraries():
    """Use the active environment's Torch libraries when loading the binding."""
    spec = importlib.util.find_spec('torch')
    if spec is None:
        return
    directory = str(Path(spec.origin).parent / 'lib')
    paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    if Path(directory).is_dir() and paths[0] != directory:
        env = dict(os.environ, LD_LIBRARY_PATH=':'.join(
            [directory] + [p for p in paths if p and p != directory]))
        os.execvpe(sys.executable, [sys.executable, str(HERE / 'run.py'), *sys.argv[1:]], env)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-name', help='Required plot label for new runs; saved for plot-only')
    p.add_argument('--build-dir', type=Path, default=ROOT / 'build/solve-time-heatmap-felix-devel')
    p.add_argument('--output', type=Path, default=ROOT / 'test-artifacts/solve-time-heatmap')
    p.add_argument('--protocol', choices=['wall-time', 'reference'], default='wall-time')
    p.add_argument('--max-sqp-iters', type=int, default=1000,
                   help='Use 1 for one SQP step; otherwise native stopping up to this cap (default: 1000)')
    p.add_argument('--wall-time', type=float, default=10.,
                   help='Measured wall-time budget per cell, excluding warm-up (wall-time protocol)')
    p.add_argument('--sim-time', type=float, default=10.,
                   help='Simulated duration per cell (reference protocol)')
    p.add_argument('--max-solve-ms', type=float, default=1000.,
                   help='Stop after a measured solve exceeds this duration; 0 disables (default: 1000)')
    p.add_argument('--knots', type=int, nargs='+', choices=KNOTS, default=list(KNOTS))
    p.add_argument('--batches', type=int, nargs='+', choices=BATCHES, default=list(BATCHES))
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--seed', type=int, default=17092026)
    p.add_argument('--save-plans', action='store_true', help='Also retain full batch-0 plans in raw samples')
    p.add_argument('--timeout', type=float,
                   help='Optional hard worker timeout including setup and warm-up (seconds)')
    p.add_argument('--resume', action='store_true', help='Resume an interrupted manifest with matching settings/sources')
    p.add_argument('--plot-only', action='store_true', help='Render saved results; no CUDA or solver run')
    p.add_argument('--cell', type=int, nargs=3, metavar=('N', 'BATCH', 'REPEAT'), help=argparse.SUPPRESS)
    return p


def configuration(args):
    return dict(gpu_name=args.gpu_name, protocol=args.protocol, knots=args.knots,
                batches=args.batches, repeats=args.repeats, max_sqp_iters=args.max_sqp_iters,
                wall_time=args.wall_time, sim_time=args.sim_time, max_solve_ms=args.max_solve_ms,
                seed=args.seed, save_plans=args.save_plans, timeout=args.timeout,
                dt=.01, sim_dt=.001, sim_step=.01, warmup_solves=1,
                timing='Full-batch synchronized native host solve time; initial warm-up and Python work excluded',
                frequency='1000 / mean native milliseconds; not batch throughput or end-to-end control rate',
                stopping='Unchanged native stopping rule up to the selected SQP cap',
                clock=('Fixed 10 ms updates, wall budget after warm-up; active solves finish'
                       if args.protocol == 'wall-time' else
                       'Original min(previous native solve time, 10 ms), fixed simulated duration'))


def source_paths():
    paths = list((ROOT / 'gato').rglob('*.cuh')) + list((ROOT / 'gato').rglob('*.h'))
    paths += list((ROOT / 'python').rglob('*.py')) + [ROOT / 'python/bindings.cu']
    paths += list(HERE.glob('*.py'))
    paths += [HERE / 'resources.cu', HERE / 'CMakeLists.txt', HERE / 'requirements.txt',
              ROOT / 'examples/indy7_description/indy7.urdf']
    return sorted(set(paths))


def prepare_manifest(args):
    manifest_path = args.output / 'results.json'
    config = configuration(args)
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths()}
    if manifest_path.exists():
        if not args.resume:
            raise ValueError('Output contains results; use a fresh --output, --resume or --plot-only')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema_version') != 2:
            raise ValueError('Legacy results can be plotted, but cannot be resumed by this runner')
        if manifest['config'] != config or manifest['source_sha256'] != hashes:
            raise ValueError('Resume settings or sources differ from the saved experiment')
        if manifest['build_dir'] != str(args.build_dir):
            raise ValueError('Resume build directory differs from the saved experiment')
        for filename, expected in manifest['module_sha256'].items():
            if digest(filename) != expected:
                raise ValueError(f'Resume module changed: {filename}')
        return manifest
    if args.resume:
        raise ValueError('--resume requires an existing results.json')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('New experiment output must be empty; choose a fresh --output')
    import numpy
    import pinocchio
    import matplotlib
    import torch
    manifest = dict(schema_version=2, config=config, status='running', results=[], resources={},
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    started_at=time.time(), executable=sys.executable, python=sys.version,
                    build_dir=str(args.build_dir), gpu_before=gpu_snapshot(), source_sha256=hashes,
                    module_sha256={}, library_path=os.environ.get('LD_LIBRARY_PATH', ''),
                    packages=dict(numpy=numpy.__version__, pinocchio=pinocchio.__version__,
                                  matplotlib=matplotlib.__version__, torch=torch.__version__),
                    build_cache=(args.build_dir / 'CMakeCache.txt').read_text(),
                    nvcc=subprocess.check_output(['nvcc', '--version'], text=True))
    for n in args.knots:
        probe = subprocess.run([str(args.build_dir / f'resources_N{n}')],
                               check=True, capture_output=True, text=True)
        resource = manifest['resources'][str(n)] = json.loads(probe.stdout)
        if all(k['fits'] for k in resource['kernels']):
            modules = list((args.build_dir / 'modules').glob(f'bsqpN{n}_indy7*.so'))
            if len(modules) != 1:
                raise ValueError(f'Build exactly one bsqpN{n}_indy7 module in {args.build_dir / "modules"}')
            manifest['module_sha256'][str(modules[0].resolve())] = digest(modules[0])
    args.output.mkdir(parents=True, exist_ok=True)
    for path in source_paths():
        destination = args.output / 'source_snapshot' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    (args.output / 'source.diff').write_text(subprocess.check_output(
        ['git', 'diff', 'HEAD', '--', 'gato', 'python', 'examples/solve_time_heatmap'], cwd=ROOT, text=True))
    save_json(manifest_path, manifest)
    return manifest


def cell_command(args, n, batch, repeat):
    command = [sys.executable, str(HERE / 'run.py'), '--cell', str(n), str(batch), str(repeat),
               '--gpu-name', args.gpu_name, '--build-dir', str(args.build_dir),
               '--output', str(args.output), '--protocol', args.protocol,
               '--max-sqp-iters', str(args.max_sqp_iters), '--wall-time', str(args.wall_time),
               '--sim-time', str(args.sim_time), '--max-solve-ms', str(args.max_solve_ms),
               '--seed', str(args.seed)]
    if args.save_plans:
        command.append('--save-plans')
    return command


def run_cell_process(args, n, batch, repeat):
    folder = args.output / f'N{n}_B{batch}_R{repeat}'
    receipt = folder / 'results.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['status'] not in ('running', 'interrupted'):
            return previous  # Worker finished before an interrupted parent saved its manifest.
    if folder.exists():
        archive = args.output / 'interrupted' / f'{folder.name}_{time.time_ns()}'
        archive.parent.mkdir(exist_ok=True)
        folder.rename(archive)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    log_path = args.output / f'{folder.name}.log'
    with log_path.open('a') as log:
        try:
            result = subprocess.run(cell_command(args, n, batch, repeat), cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            failure = f'Worker exited {result.returncode}; see {log_path.name}' if result.returncode else None
        except subprocess.TimeoutExpired:
            failure = f'Worker exceeded --timeout {args.timeout:g}s; see {log_path.name}'
    row = json.loads(receipt.read_text()) if receipt.exists() else dict(N=n, batch_size=batch, repeat=repeat, samples=0)
    if failure or row.get('status') in (None, 'running'):
        row.update(status='error', error=failure or 'Worker did not finish its result receipt')
    return row


def sweep(args):
    if os.environ.get('CUDA_LAUNCH_BLOCKING', '0') != '0':
        raise ValueError('Unset CUDA_LAUNCH_BLOCKING before benchmarking')
    dirty = subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'gato', 'python'],
                           cwd=ROOT, capture_output=True)
    if dirty.returncode:
        raise ValueError('Solver/Python sources differ from HEAD; use matched clean sources and rebuilt modules')
    manifest = prepare_manifest(args)
    cells = [(n, b, r) for r in range(args.repeats) for n in args.knots for b in args.batches]
    random.Random(args.seed).shuffle(cells)
    done = {(r['N'], r['batch_size'], r.get('repeat', 0)) for r in manifest['results']}
    for index, (n, batch, repeat) in enumerate(cells, 1):
        if (n, batch, repeat) in done:
            continue
        manifest.update(status='running', active_cell=dict(N=n, batch_size=batch, repeat=repeat, index=index))
        save_json(args.output / 'results.json', manifest)
        print(f'[{index}/{len(cells)}] N={n} B={batch} repeat={repeat}', flush=True)
        bad = [k for k in manifest['resources'][str(n)]['kernels'] if not k['fits']]
        if bad:
            row = dict(N=n, batch_size=batch, repeat=repeat,
                       status='unsupported_shared_memory', samples=0, kernels=bad)
        else:
            row = run_cell_process(args, n, batch, repeat)
        manifest['results'].append(row)
        manifest.pop('active_cell', None)
        save_json(args.output / 'results.json', manifest)
        print(f'  {row["status"]}: {row.get("avg_native_ms", "—")} ms, {row.get("samples", 0)} samples', flush=True)
    manifest.update(status='completed', finished_at=time.time(), gpu_after=gpu_snapshot())
    save_json(args.output / 'results.json', manifest)
    from plotting import plot
    plot(args.output)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.output, args.build_dir = args.output.resolve(), args.build_dir.resolve()
    if args.gpu_name is not None:
        args.gpu_name = args.gpu_name.strip()
        if not args.gpu_name:
            p.error('--gpu-name must not be empty')
    if args.plot_only:
        path = args.output / 'results.json'
        if not path.exists():
            p.error('--plot-only requires results.json in --output')
        data = json.loads(path.read_text())
        name = args.gpu_name or data['config'].get('gpu_name')
        if not name:
            p.error('Saved results need a GPU label; provide --gpu-name')
        if name != data['config'].get('gpu_name'):
            data['config']['gpu_name'] = name
            save_json(path, data)
        from plotting import plot
        plot(args.output)
        return 0
    if args.gpu_name is None:
        p.error('--gpu-name is required for a new or resumed run')
    values = [args.wall_time, args.sim_time, args.max_solve_ms]
    if args.timeout is not None:
        values.append(args.timeout)
    if not all(math.isfinite(value) for value in values):
        p.error('Durations and cutoff must be finite')
    if min(args.wall_time, args.sim_time, args.max_sqp_iters, args.repeats) <= 0 or args.max_solve_ms < 0:
        p.error('Durations, SQP cap and repeats must be positive; cutoff must be nonnegative')
    if args.timeout is not None and args.timeout <= 0:
        p.error('--timeout must be positive')
    if len(set(args.knots)) != len(args.knots) or len(set(args.batches)) != len(args.batches):
        p.error('Horizons and batch sizes must be unique')
    if args.seed < 0 or args.seed + args.repeats > 2**32:
        p.error('Seed plus repeat index must fit an unsigned 32-bit integer')
    if args.cell and (args.cell[0] not in KNOTS or args.cell[1] not in BATCHES or args.cell[2] < 0):
        p.error('Invalid worker cell')
    match_torch_libraries()
    if args.cell:
        from experiment import run_cell
        return run_cell(args)
    try:
        sweep(args)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
