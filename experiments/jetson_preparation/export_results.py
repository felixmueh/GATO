"""Export measurements and provenance, without build trees or execution."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def latest_completed(root):
    runs = sorted((root / 'example_artifacts/jetson_f01ab').glob('*/launch-state.json'))
    for state in reversed(runs):
        if json.loads(state.read_text()).get('status') == 'completed':
            return state.parent
    raise ValueError('No completed run found; use --run PATH to export a partial run')


def export_results(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    state = json.loads((run / 'launch-state.json').read_text())
    files = {}

    def include(path, name):
        path = Path(path)
        if path.is_symlink():
            raise ValueError('Refusing symlink in result export: ' + str(path))
        if path.is_file():
            files[name] = path
        elif path.is_dir():
            for child in sorted(path.rglob('*')):
                if child.is_symlink():
                    raise ValueError('Refusing symlink in result export: ' + str(child))
                if child.is_file():
                    files[name + '/' + child.relative_to(path).as_posix()] = child

    for path in sorted(run.glob('*.json')):
        include(path, path.name)
    for name in ('results', 'readiness', 'logs'):
        include(run / name, name)
    include(run / 'collector-source/experiments/frequency_campaign/inputs', 'inputs')
    # Copy compact suite/build evidence referenced by the release. Native binaries
    # and build/source/cache trees stay on the measurement machine.
    release_path = run / 'jetson-release.json'
    if release_path.exists():
        release = json.loads(release_path.read_text())
        for key, module in release.get('modules', {}).items():
            include(module['build_receipt']['path'], 'evidence/build-N' + key + '.json')
        parent = state.get('parent_failed_attempt') or {}
        prepared = Path(parent.get('release', {}).get('path', release_path)).parent
        include(prepared / 'logs/make-test-tiago-tasks.log', 'evidence/task-suite.log')
        preparation = prepared / 'preparation.json'
        if preparation.exists():
            include(preparation, 'evidence/preparation.json')
            for index, item in enumerate(json.loads(preparation.read_text()).get('suite_summaries', [])):
                include(item['path'], 'evidence/suite-' + str(index) + '.json')

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    prefix = 'jetson-f01ab-results'
    manifest = {}
    try:
        with tarfile.open(temporary, 'w:gz', compresslevel=1) as archive:
            for name, path in sorted(files.items()):
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                manifest[name] = dict(sha256=digest.hexdigest(), original_path=str(path))
                archive.add(path, arcname=prefix + '/' + name, recursive=False)
            description = dict(run=run.name, status=state['status'], cells=len(state.get('cells', state.get('completed_cells', []))), native_calls=state.get('native_calls'), files=manifest,
                               note='Paths in original receipts refer to the Jetson. This index maps exported files to relative archive paths. Builds, binaries, caches and source checkouts remain on the Jetson.')
            payload = (json.dumps(description, indent=2) + '\n').encode()
            member = tarfile.TarInfo(prefix + '/EXPORT.json')
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print('Results archive:', output, flush=True)
    print('Archive bytes:', output.stat().st_size, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Specific run, including a partial/failed run')
    parser.add_argument('--output', type=Path, default=ROOT / 'jetson-f01ab-results.tar.gz')
    args = parser.parse_args()
    export_results(args.run or latest_completed(ROOT), args.output)
