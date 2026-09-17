"""Reproduce plots/fig8_benchmark_heatmap.ipynb with fresh, isolated modules.

See --help for the sweep and plot-only commands. Each cell runs in a separate
process so CUDA errors cannot contaminate subsequent measurements.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_snapshot():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,uuid,driver_version,temperature.gpu,'
        'utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw',
        '--format=csv'], text=True).strip()


def match_torch_libraries():
    """Avoid a container's global Torch .so files overriding the active venv."""
    spec = importlib.util.find_spec('torch')
    if spec is None:
        return  # Normal dependency validation will report the missing package.
    directory = str(Path(spec.origin).parent / 'lib')
    paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    if Path(directory).is_dir() and paths[0] != directory:
        env = dict(os.environ, LD_LIBRARY_PATH=':'.join([directory] + [p for p in paths if p and p != directory]))
        os.execvpe(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)


def worker(args):
    import numpy as np
    import pinocchio as pin

    sys.path.insert(0, str(ROOT / 'python'))
    sys.path.insert(0, str(ROOT / 'examples'))
    import bsqp
    bsqp.__path__ = [str(args.build_dir / 'modules'), str(ROOT / 'python/bsqp')]
    from bsqp.mpc_controller import MPC_GATO
    from bsqp.common import figure8
    from bsqp.config import DEFAULT_SOLVER_PARAMS, FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS

    class BenchmarkMPC(MPC_GATO):
        def setup_force_estimator(self):
            # The original estimator requires >3 samples. Use the zero-force
            # problem at B=2, as at B=1, while retaining the original B>=4 path.
            if self.batch_size == 2:
                self.force_estimator = None
            else:
                super().setup_force_estimator()

    n, batch, repeat = args.cell
    np.random.seed(args.seed + repeat)
    urdf = str(ROOT / 'examples/indy7_description/indy7.urdf')
    model = pin.buildModelFromUrdf(urdf)
    controller = BenchmarkMPC(model=model, model_path=urdf, N=n, dt=.01,
                             batch_size=batch, track_full_stats=True)
    module_path = Path(controller.solver.lib.__file__).resolve()
    if module_path.parent != (args.build_dir / 'modules').resolve():
        raise RuntimeError(f'Unexpected module: {module_path}')
    traj = figure8(.01, **FIG8_DEFAULT_PARAMS)
    start = np.r_[INDY7_START_CONFIGS['ready'], np.zeros(6)]
    before = gpu_snapshot()
    t0 = time.monotonic()
    # The original routine performs one warm-up solve before collecting stats.
    _, stats = controller.run_mpc_fig8(start, traj, sim_dt=.001, sim_time=args.sim_time,
                                     offline_timing='solve_time')
    elapsed = time.monotonic() - t0
    samples = np.asarray(stats['solve_times'])
    if not samples.size or not np.all(np.isfinite(samples)) or np.any(samples <= 0):
        raise RuntimeError('Missing/nonfinite/nonpositive solve-time samples')
    for key in ['goal_distances', 'joint_positions', 'joint_velocities']:
        if not np.all(np.isfinite(stats[key])):
            raise RuntimeError(f'Nonfinite {key}; reject invalid simulation timings')
    stem = f'N{n}_B{batch}_R{repeat}'
    np.savez_compressed(args.output / f'{stem}.npz', **stats)
    result = dict(N=n, batch_size=batch, repeat=repeat, status='ok',
                  samples=int(samples.size), avg_gpu_time_ms=float(samples.mean()),
                  std_gpu_time_ms=float(samples.std()), median_gpu_time_ms=float(np.median(samples)),
                  p95_gpu_time_ms=float(np.percentile(samples, 95)),
                  avg_goal_distance=float(np.mean(stats['goal_distances'])),
                  max_goal_distance=float(np.max(stats['goal_distances'])),
                  avg_sqp_iters=float(np.mean(stats['sqp_iters'])),
                  simulation_end_s=float(stats['timestamps'][-1]), wall_time_s=elapsed,
                  gpu_before=before, gpu_after=gpu_snapshot(), solver_params=DEFAULT_SOLVER_PARAMS,
                  module=str(module_path), module_sha256=digest(module_path),
                  seed=args.seed + repeat, raw_samples=f'{stem}.npz')
    save_json(args.output / f'{stem}.json', result)


def plot(output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    data = json.loads((output / 'results.json').read_text())
    ns, batches = data['config']['knots'], data['config']['batches']
    z = np.full((len(ns), len(batches)), np.nan)
    labels = {}
    summary = []
    for i, n in enumerate(ns):
        for j, b in enumerate(batches):
            cells = [r for r in data['results'] if r['N'] == n and r['batch_size'] == b]
            good = [r for r in cells if r['status'] == 'ok']
            if good:
                # Match the reference notebook: equally weight repeat means.
                z[i, j] = np.mean([r['avg_gpu_time_ms'] for r in good])
                if len(good) < data['config']['repeats']:
                    labels[i, j] = '*'
            else:
                labels[i, j] = ('SMEM' if cells and all(r['status'] == 'unsupported_shared_memory'
                                                      for r in cells) else 'FAIL' if cells else '—')
            summary.append(dict(N=n, batch_size=b, successful_repeats=len(good),
                                status='ok' if good else labels[i, j],
                                mean_ms=float(z[i, j]) if good else '',
                                samples=sum(r['samples'] for r in good),
                                tracking_mean_m=float(np.mean([r['avg_goal_distance'] for r in good])) if good else ''))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = plt.get_cmap('RdYlGn_r').copy()
    cmap.set_bad('#dedede')
    finite = z[np.isfinite(z)]
    vmax = max(20., float(finite.max())) if finite.size else 20.
    im = ax.imshow(np.ma.masked_invalid(z), origin='lower', aspect='auto',
                   cmap=cmap, norm=LogNorm(.09, vmax), interpolation='nearest')
    for i in range(len(ns)):
        for j in range(len(batches)):
            if np.isfinite(z[i, j]):
                rgba = cmap(im.norm(z[i, j]))
                luminance = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
                ax.text(j, i, f'{z[i,j]:.2f}{labels.get((i,j), "")}', ha='center', va='center',
                        fontsize=12, weight='bold', color='black' if luminance > .55 else 'white')
            else:
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, hatch='///',
                                       edgecolor='#aaaaaa', linewidth=0))
                ax.text(j, i, labels[i, j], ha='center', va='center', fontsize=10,
                        bbox=dict(facecolor='#dedede', edgecolor='none', pad=1))
    if finite.size and min(z.shape) >= 2:
        levels = [v for v in [.1, .2, 1., 4., 10., 20.] if finite.min() < v < finite.max()]
        if levels:
            cs = ax.contour(np.ma.masked_invalid(z), levels=levels, colors='blue', linewidths=2)
            ax.clabel(cs, fmt=lambda ms: f'{1/ms:g}kHz' if ms <= 1 else f'{1000/ms:g}Hz', fontsize=13)
    ax.set_xticks(range(len(batches)), labels=batches)
    ax.set_yticks(range(len(ns)), labels=ns)
    ax.set_xlabel('Batch Size', fontsize=19)
    ax.set_ylabel('Trajectory Length (N)', fontsize=19)
    ax.set_title('GATO / Indy7 — GeForce GTX 1070', fontsize=19, pad=16)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label('GPU Solve Time (ms)', fontsize=16)
    fig.text(.5, .025, f'felix-devel @ {data["git_head"][:7]} · Mean synchronized solver time; 1 SQP iteration; warm-up excluded\n'
             f'{data["config"]["sim_time"]:g} s figure-eight simulation × {data["config"]["repeats"]} repeats'
             ' · SMEM: per-block shared-memory limit · *: incomplete repeats', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, 1))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'gato_solve_time_heatmap_gtx1070.{ext}', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, default=ROOT / 'build/solve-time-heatmap-felix-devel')
    p.add_argument('--output', type=Path, default=ROOT / 'test-artifacts/solve-time-heatmap-gtx1070-felix-devel')
    p.add_argument('--knots', type=int, nargs='+', default=[8, 16, 32, 64, 128, 256])
    p.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    p.add_argument('--sim-time', type=float, default=10.)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--seed', type=int, default=17092026)
    p.add_argument('--timeout', type=float, default=300.)
    p.add_argument('--plot-only', action='store_true')
    p.add_argument('--cell', type=int, nargs=3, metavar=('N', 'BATCH', 'REPEAT'), help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.build_dir = args.build_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(args.output)
        return
    match_torch_libraries()
    if args.cell:
        worker(args)
        return
    if args.repeats < 1 or args.sim_time <= 0:
        p.error('repeats and simulation time must be positive')
    if len(set(args.knots)) != len(args.knots) or len(set(args.batches)) != len(args.batches):
        p.error('horizons and batches must be unique')
    # Fail before launching any cells if required experiment packages are absent.
    import numpy
    import pinocchio
    import matplotlib
    import torch
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'gato', 'python'],
                   cwd=ROOT, check=True, capture_output=True)
    # Do not silently mix runs with different settings or overwrite raw receipts.
    if (args.output / 'results.json').exists():
        p.error('output already contains results; choose a fresh --output or use --plot-only')
    config = dict(knots=args.knots, batches=args.batches, repeats=args.repeats,
                  sim_time=args.sim_time, seed=args.seed, timeout=args.timeout,
                  dt=.01, sim_dt=.001, warmup_solves=1, precision='float32',
                  offline_timing='solve_time',
                  timing='BSQP synchronized host wall time in microseconds / 1000',
                  batch2='zero force; source force estimator requires batch > 3')
    source_paths = list((ROOT / 'gato').rglob('*.cuh')) + list((ROOT / 'gato').rglob('*.h'))
    source_paths += list((ROOT / 'python').rglob('*.py'))
    source_paths += [ROOT / 'python/bindings.cu', Path(__file__), Path(__file__).with_name('resources.cu'),
                     Path(__file__).with_name('CMakeLists.txt'), Path(__file__).with_name('requirements.txt')]
    manifest = dict(config=config, git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    python=sys.version, executable=sys.executable, gpu_before=gpu_snapshot(),
                    library_path=os.environ.get('LD_LIBRARY_PATH', ''),
                    packages=dict(numpy=numpy.__version__, pinocchio=pinocchio.__version__, matplotlib=matplotlib.__version__, torch=torch.__version__),
                    nvcc=subprocess.check_output(['nvcc', '--version'], text=True),
                    build_cache=(args.build_dir / 'CMakeCache.txt').read_text(),
                    source_sha256={str(f.relative_to(ROOT)): digest(f) for f in source_paths},
                    resources={}, results=[])
    (args.output / 'source.diff').write_text(subprocess.check_output(['git', 'diff'], cwd=ROOT, text=True))
    for n in args.knots:
        proc = subprocess.run([str(args.build_dir / f'resources_N{n}')], capture_output=True, text=True, check=True)
        manifest['resources'][str(n)] = json.loads(proc.stdout)
    save_json(args.output / 'results.json', manifest)
    cells = [(n, b, r) for r in range(args.repeats) for n in args.knots for b in args.batches]
    random.Random(args.seed).shuffle(cells)
    for index, (n, b, r) in enumerate(cells):
        print(f'[{index+1}/{len(cells)}] N={n} B={b} repeat={r}', flush=True)
        bad = [k for k in manifest['resources'][str(n)]['kernels'] if not k['fits']]
        stem = f'N{n}_B{b}_R{r}'
        if bad:
            result = dict(N=n, batch_size=b, repeat=r, status='unsupported_shared_memory', kernels=bad)
        else:
            command = [sys.executable, str(Path(__file__).resolve()), '--cell', str(n), str(b), str(r),
                       '--build-dir', str(args.build_dir), '--output', str(args.output),
                       '--sim-time', str(args.sim_time), '--seed', str(args.seed)]
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            # An inherited debug launch mode would invalidate timings.
            if env.get('CUDA_LAUNCH_BLOCKING', '0') != '0':
                raise RuntimeError('Unset CUDA_LAUNCH_BLOCKING before benchmarking')
            with (args.output / f'{stem}.log').open('w') as log:
                try:
                    proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=args.timeout)
                    if proc.returncode:
                        raise RuntimeError(f'Worker exit {proc.returncode}; see {stem}.log')
                    result = json.loads((args.output / f'{stem}.json').read_text())
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    result = dict(N=n, batch_size=b, repeat=r, status='failed', error=str(exc))
        manifest['results'].append(result)
        save_json(args.output / 'results.json', manifest)
        print(f'  {result["status"]} {result.get("avg_gpu_time_ms", "")} ms', flush=True)
    manifest['gpu_after'] = gpu_snapshot()
    save_json(args.output / 'results.json', manifest)
    plot(args.output)


if __name__ == '__main__':
    main()
