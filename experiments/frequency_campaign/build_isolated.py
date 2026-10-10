"""Normal runner archive/build adapter; root executes only after reviewed release."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/campaign'))
spec = importlib.util.spec_from_file_location('campaign_archive_builder', ROOT / 'experiments/campaign/build_isolated.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)


def build(source, build_root, python, cuda_arch, N):
    if N not in (16, 32, 64, 128): raise ValueError('Frozen F01 horizon required')
    source = base.prepare(source, build_root)
    build_root = Path(build_root).resolve(); native = build_root / 'campaign-native'
    commands = [
        ['cmake', '-S', str(source), '-B', str(native), '-DPLANT=tiago_right', f'-DKNOTS={N}',
         '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=ON', '-DGATO_EXPERIMENT_TELEMETRY=ON',
         f'-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}', f'-DPython3_EXECUTABLE={python}'],
        ['cmake', '--build', str(native), '--parallel', '2', '--target', f'bsqpN{N}_tiago_right'],
    ]
    base.write_json(build_root / 'f01-build-attempt.json', dict(status='started', N=N, commands=commands))
    try:
        for command in commands: subprocess.run(command, check=True)
        modules = list((source / 'python/bsqp').glob(f'bsqpN{N}_tiago_right*.so'))
        if len(modules) != 1: raise RuntimeError('Exactly one source-owned module required')
        source_receipt = build_root / 'build_source_receipt.json'
        archive = json.loads(source_receipt.read_text())
        cache = native / 'CMakeCache.txt'
        receipt = dict(variant='ON', source_commit=archive['source_commit'], source_root=str(source),
            flags=dict(GATO_EXPERIMENT_TELEMETRY=True, N=N, plant='tiago_right', cuda_arch=cuda_arch),
            module=dict(path=str(modules[0]), sha256=base.sha(modules[0])),
            cmake_cache=dict(path=str(cache), sha256=base.sha(cache)),
            source_receipt=dict(path=str(source_receipt), sha256=base.sha(source_receipt)),
            source_file_sha256=archive['source_file_sha256'], commands=commands)
        base.write_json(build_root / f'f01-build-N{N}.json', receipt)
        base.write_json(build_root / 'f01-build-attempt.json', dict(status='completed', N=N, commands=commands))
        return receipt
    except BaseException as error:
        base.write_json(build_root / 'f01-build-attempt.json', dict(status='failed', N=N, commands=commands, error=str(error)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True); p.add_argument('--python', required=True)
    p.add_argument('--cuda-arch', required=True); p.add_argument('--N', type=int, required=True)
    a = p.parse_args(); build(a.source, a.build, a.python, a.cuda_arch, a.N)
