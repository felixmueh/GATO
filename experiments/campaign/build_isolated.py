"""Build a committed source archive under the runner build root, never its cache checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import uuid

from raw import sha, write_json


def prepare(source, build):
    source, build = Path(source).resolve(), Path(build).resolve()
    destination = build/'campaign-source'
    build.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    archive = build/'campaign-source.tar'
    expected_archive = build/f'.prepare-{uuid.uuid4().hex}.tar'
    with expected_archive.open('wb') as stream:
        subprocess.run(['git', '-C', str(source), 'archive', '--format=tar', commit], stdout=stream, check=True)
    with tarfile.open(expected_archive) as stream:
        expected_files = {}
        for member in stream.getmembers():
            target = (destination/member.name).resolve()
            if not target.is_relative_to(destination) or member.issym() or member.islnk():
                raise ValueError(f'Archive path/link rejected: {member.name}')
            if member.isfile():
                expected_files[member.name] = hashlib.sha256(stream.extractfile(member).read()).hexdigest()
    receipt_path = build/'build_source_receipt.json'
    saved = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if saved is not None and (saved['source_commit'] != commit
        or saved['isolated_source'] != str(destination) or saved['cached_source'] != str(source)
        or saved['archive_sha256'] != sha(expected_archive)):
        raise ValueError('Pinned archive/source identity mismatch; existing evidence retained')
    if destination.exists() and saved is not None:
        for relative, expected in expected_files.items():
            current = destination/relative
            if current.is_symlink() or not current.is_file() or sha(current) != expected:
                raise ValueError(f'Pinned isolated source bytes differ: {relative}; existing evidence retained')
    elif destination.exists():
        # Interrupted extraction has no atomic receipt. Preserve it and any
        # CMake state, then reconstruct the same pinned bytes and path.
        failed = build/'preserved-build-attempts'/uuid.uuid4().hex
        failed.mkdir(parents=True)
        destination.rename(failed/'campaign-source')
        native = build/'campaign-native'
        if native.exists():
            native.rename(failed/'campaign-native')
    if archive.exists() and sha(archive) != sha(expected_archive):
        failed = build/'preserved-build-attempts'/uuid.uuid4().hex
        failed.mkdir(parents=True)
        archive.rename(failed/'campaign-source.tar')
    if archive.exists():
        expected_archive.unlink()
    else:
        expected_archive.replace(archive)
    if not destination.exists():
        destination.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(destination)
    # No submodule generation is needed: checked-in generated dynamics are the
    # immutable build inputs. Do not initialize/update the private runner source.
    write_json(build/'build_source_receipt.json', {'source_commit':commit,
               'archive_sha256':sha(archive), 'isolated_source':str(destination),
               'source_file_sha256':expected_files,
               'cached_source':str(source), 'submodules':'not needed for checked-in generated model build'})
    return destination


def build_native(source, build, python, cuda_arch):
    destination = prepare(source, build)
    build = Path(build).resolve()
    native = build/'campaign-native'
    attempt = build/'preserved-build-attempts'/uuid.uuid4().hex
    attempt.mkdir(parents=True)
    # Preserve receipts, previous modules and configure diagnostics before a
    # clean target rebuild. A newer corrupted .so can look up-to-date to CMake.
    previous = [build/'build_receipt.json', native/'CMakeCache.txt', native/'cuda_access_smoke',
                *native.glob('CMakeFiles/*.log'),
                *(destination/'python/bsqp').glob('bsqpN32_tiago_right*.so')]
    for path in previous:
        if path.is_file():
            target = attempt/path.relative_to(build)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    commands = [['cmake', '-S', str(destination), '-B', str(native), '-DPLANT=tiago_right',
                 '-DKNOTS=32', '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=ON',
                 f'-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}', f'-DPython3_EXECUTABLE={python}'],
                ['cmake', '--build', str(native), '--clean-first', '--parallel', '2', '--target',
                 'bsqpN32_tiago_right', 'cuda_access_smoke']]
    write_json(attempt/'attempt.json', {'status':'started', 'commands':commands})
    try:
        for command in commands:
            subprocess.run(command, check=True)
        modules = list((destination/'python/bsqp').glob('bsqpN32_tiago_right*.so'))
        if len(modules) != 1:
            raise RuntimeError('Expected exactly one freshly built N32 TIAGo module')
        receipt = {'source':str(destination), 'commands':commands,
                   'cmake_cache_sha256':sha(native/'CMakeCache.txt'),
                   'module_path':str(modules[0]), 'module_sha256':sha(modules[0]),
                   'cuda_access_smoke_sha256':sha(native/'cuda_access_smoke'),
                   'build_source_receipt_sha256':sha(build/'build_source_receipt.json')}
        write_json(build/'build_receipt.json', receipt)
    except BaseException as exc:
        write_json(attempt/'attempt.json', {'status':'failed', 'commands':commands,
                   'error_type':type(exc).__name__, 'error':str(exc)})
        raise
    write_json(attempt/'attempt.json', {'status':'completed', 'commands':commands,
                                       'build_receipt_sha256':sha(build/'build_receipt.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--cuda-arch', required=True)
    args = parser.parse_args()
    build_native(args.source, args.build, args.python, args.cuda_arch)
