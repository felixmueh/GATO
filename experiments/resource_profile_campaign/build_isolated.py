"""Normal isolated archive builder; no execution until a reviewed plan is released."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/campaign'))
import importlib.util
spec = importlib.util.spec_from_file_location('profiles_archive', ROOT / 'experiments/campaign/build_isolated.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)


def commands(source, native, python, arch, N, profile):
    if N not in (16,32,64,128) or (profile and N != 32): raise ValueError('Frozen resource/profile horizon required')
    return [['cmake', '-S', str(source), '-B', str(native), '-DPLANT=tiago_right', f'-DKNOTS={N}',
        '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=ON', '-DGATO_EXPERIMENT_TELEMETRY=ON',
        '-DGATO_EXPERIMENT_RESOURCE_QUERY=ON', f'-DGATO_EXPERIMENT_STAGE_TIMING={"ON" if profile else "OFF"}',
        f'-DCMAKE_CUDA_ARCHITECTURES={arch}', f'-DPython3_EXECUTABLE={python}'],
        ['cmake', '--build', str(native), '--parallel', '2', '--target', f'bsqpN{N}_tiago_right']]


def build(source, output, python, arch, N, profile):
    output = Path(output).resolve()
    # One owning archive/build per N/flag variant, never overwrite timing with profile.
    if (output / 'resource-profile-build-attempt.json').exists(): raise ValueError('Fresh variant build root required; retain earlier attempt')
    output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    cmds = commands(output/'campaign-source', output / 'campaign-native', python, arch, N, profile)
    base.write_json(output / 'resource-profile-build-attempt.json', dict(status='started', commands=cmds, stage_timing=profile))
    try:
        isolated = base.prepare(source, output)
        for command in cmds: subprocess.run(command, check=True)
        modules = list((isolated / 'python/bsqp').glob(f'bsqpN{N}_tiago_right*.so'))
        if len(modules) != 1: raise RuntimeError('Exactly one archived module required')
        source_receipt = output / 'build_source_receipt.json'; archive = json.loads(source_receipt.read_text())
        cache = output / 'campaign-native/CMakeCache.txt'
        receipt = dict(source_commit=archive['source_commit'], source_root=str(isolated), N=N,
            flags=dict(GATO_EXPERIMENT_STAGE_TIMING=profile, GATO_EXPERIMENT_RESOURCE_QUERY=True, GATO_EXPERIMENT_TELEMETRY=True),
            module=dict(path=str(modules[0]), sha256=base.sha(modules[0])),
            cmake_cache=dict(path=str(cache), sha256=base.sha(cache)),
            source_receipt=dict(path=str(source_receipt), sha256=base.sha(source_receipt)),
            commands=cmds, build_wall_s=time.perf_counter()-started, timing_scope='archive/configure/compile host wall; not solve time')
        base.write_json(output / 'resource-profile-build.json', receipt)
        base.write_json(output / 'resource-profile-build-attempt.json', dict(status='completed', commands=cmds))
        return receipt
    except BaseException as error:
        base.write_json(output / 'resource-profile-build-attempt.json', dict(status='failed', commands=cmds, error=str(error)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', required=True); p.add_argument('--build', required=True)
    p.add_argument('--python', required=True); p.add_argument('--cuda-arch', required=True); p.add_argument('--N', type=int, required=True)
    p.add_argument('--profile', action='store_true'); p.add_argument('--dry-run', action='store_true'); a=p.parse_args()
    if a.dry_run: print(json.dumps(commands(Path(a.build)/'campaign-source',Path(a.build)/'campaign-native',a.python,a.cuda_arch,a.N,a.profile),indent=2))
    else: build(a.source,a.build,a.python,a.cuda_arch,a.N,a.profile)
