#!/usr/bin/env python3
"""Portable, manually invoked Jetson build/freeze/run entry point; no robot motion."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from upstream_reach import ROOT, resolve_algorithm, sha256, write_json
from verification_worker import native_path


def environment(checkout):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(checkout / 'python'), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    return env


def base_metadata(checkout):
    return dict(checkout=str(checkout), python=sys.executable, python_version=sys.version,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip(),
        dirty_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout, text=True),
        launcher_sha256=sha256(__file__), power_mode='MODE_40W',
        power_mode_provenance='user-confirmed; not freshly measured',
        scope='simulated public Indy7; no ROS/robot motion; no power/clock setting changes')


def locks(checkout):
    handles = [open(checkout / '.gpu.lock', 'a'), open('/tmp/a2rlab-timing.lock', 'a')]
    deadline = time.monotonic() + 45
    for handle in handles:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Quiet window unavailable after 45 seconds; no work started')
                time.sleep(.2)
    return handles


def build(args):
    checkout, output = args.checkout.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = base_metadata(checkout)
    metadata.update(state='preparing', jobs=args.jobs, arch=args.arch, managed_path_requested=True)
    write_json(output / 'build.json', metadata)
    if 'PRECOND_MANAGED_S' not in (checkout / 'CMakeLists.txt').read_text():
        raise RuntimeError('This checkout lacks the explicit managed-memory experiment; integrate it before building')
    cmake = shutil.which('cmake')
    if cmake is None:
        raise RuntimeError('cmake is not available in the active environment')
    pybind = subprocess.check_output([sys.executable, '-m', 'pybind11', '--cmakedir'], text=True).strip()
    build_dir = args.build_dir.resolve() if args.build_dir else checkout / 'build_precond_jetson'
    commands = [[cmake, '-S', str(checkout), '-B', str(build_dir), '-DCMAKE_BUILD_TYPE=Release',
        '-DGATO_BUILD_DEMO=OFF', '-DMODULES=indy7:128', '-DPRECOND_MANAGED_S=ON',
        '-DPython3_EXECUTABLE=' + sys.executable, '-Dpybind11_DIR=' + pybind,
        '-DCMAKE_CUDA_ARCHITECTURES=' + args.arch],
        [cmake, '--build', str(build_dir), '--parallel', str(args.jobs)]]
    metadata['commands'] = commands
    held = locks(checkout)
    try:
        with (output / 'stdout.log').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
            for command in commands:
                subprocess.run(command, cwd=checkout, env=environment(checkout), stdout=stdout, stderr=stderr, check=True)
        module = native_path(checkout)
        metadata.update(state='complete', module=str(module), module_sha256=sha256(module))
    except BaseException as exc:
        metadata.update(state='failed', error=repr(exc))
        raise
    finally:
        write_json(output / 'build.json', metadata)
        for handle in held:
            handle.close()


def monitor(output):
    if shutil.which('tegrastats'):
        command = ['tegrastats', '--interval', '100']
    elif shutil.which('nvidia-smi'):
        command = ['nvidia-smi', '--query-gpu=timestamp,name,uuid,pstate,clocks.sm,clocks.mem,temperature.gpu',
                   '--format=csv', '-lms', '100']
    else:
        return None, None, None
    error = (output / 'clock-monitor.stderr').open('w')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error, text=True)
    def collect():
        with (output / 'clock-samples.jsonl').open('w') as stream:
            for line in process.stdout:
                stream.write(json.dumps(dict(observed_unix_ns=time.time_ns(), raw=line.rstrip())) + '\n')
                stream.flush()
    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    return process, thread, error


def run(args):
    checkout, output = args.checkout.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = base_metadata(checkout)
    metadata.update(state='preparing', primary_timing='actual whole-task wall, no transfer subtraction',
        methods=args.methods.split(','), preflight=args.preflight, repeats=args.repeats,
        timing_boundary='construction/prepare excluded from task wall and recorded separately; campaign also records full subprocess wall')
    write_json(output / 'launch.json', metadata)
    held = locks(checkout)
    process = thread = monitor_error = None
    try:
        # Capability/API inspection is outside every episode; shared memory is
        # explicitly selected by method, never inferred from the device name.
        sys.path.insert(0, str(checkout / 'python'))
        from upstream_reach import load_native
        native = load_native(None, 'indy7', 128)
        if not hasattr(native, 'shared_memory_info') or not hasattr(native, 'set_end_to_end_mode'):
            raise RuntimeError('Prebuilt module lacks the explicit shared-memory experiment API')
        infos = {}
        for method in metadata['methods']:
            native_method, linsys, algorithm = resolve_algorithm(method)
            native.set_method(native_method)
            native.set_end_to_end_mode(True)
            infos[method] = native.shared_memory_info()
            infos[method].update(algorithm=algorithm, linsys=linsys, native_method=native_method)
            if method.endswith('_shared') and not (infos[method]['managedMemory'] and infos[method]['managed_s_compiled']):
                raise RuntimeError('Explicit managed-memory path unavailable; no fallback')
        metadata['shared_memory_info'] = infos
        metadata['integrated_gpu_reported'] = all(info['integrated'] for info in infos.values())
        command = [sys.executable, str(Path(__file__).with_name('verification_worker.py')), 'campaign',
            '--checkout', str(checkout), '--tasks', str(args.tasks.resolve()), '--output', str(output / 'data'),
            '--methods', args.methods, '--repeats', str(args.repeats), '--primary-timing', 'wall']
        if args.cases:
            command += ['--cases', args.cases]
        if args.preflight:
            command.append('--preflight')
        metadata['command'] = command
        metadata['state'] = 'running'
        write_json(output / 'launch.json', metadata)
        process, thread, monitor_error = monitor(output)
        metadata['clock_monitor_attempted'] = process is not None
        with (output / 'stdout.log').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
            result = subprocess.run(command, cwd=checkout, env=environment(checkout), stdout=stdout, stderr=stderr)
        metadata.update(state='complete' if result.returncode == 0 else 'process_failure', returncode=result.returncode)
        return result.returncode
    except BaseException as exc:
        metadata.update(state='failed', error=repr(exc))
        raise
    finally:
        if process is not None:
            process.terminate(); process.wait(); thread.join(timeout=2); monitor_error.close()
            metadata['clock_monitor_returncode'] = process.returncode
            sample_file = output / 'clock-samples.jsonl'
            metadata['clock_monitor_raw_line_count'] = sum(1 for _ in sample_file.open()) if sample_file.exists() else 0
        metadata['end_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'launch.json', metadata)
        for handle in held:
            handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    for action in ('build', 'freeze', 'run'):
        s = sub.add_parser(action)
        s.add_argument('--checkout', type=Path, default=ROOT)
        s.add_argument('--output', type=Path, required=True)
        if action == 'build':
            s.add_argument('--build-dir', type=Path)
            s.add_argument('--jobs', type=int, default=1)
            s.add_argument('--arch', default='native')
        elif action == 'run':
            s.add_argument('--tasks', type=Path, required=True)
            s.add_argument('--methods', default='normal,block_ldlt_shared,gpu_bdsv')
            s.add_argument('--cases')
            s.add_argument('--repeats', type=int, default=3)
            s.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    if args.action == 'freeze':
        return subprocess.call([sys.executable, str(Path(__file__).with_name('verification_worker.py')), 'freeze',
                                '--checkout', str(args.checkout.resolve()), '--output', str(args.output.resolve())],
                               env=environment(args.checkout.resolve()))
    return {'build': build, 'run': run}[args.action](args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
