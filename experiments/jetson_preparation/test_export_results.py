"""Result export preserves evidence and excludes execution workspace bulk."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_results import export_results, latest_completed


def test_export_preserves_raw_and_status_without_builds(tmp_path):
    run = tmp_path / 'run'
    for name, data in {'launch-state.json': b'{"status":"completed","cells":[{}],"native_calls":241}',
                       'results/cell/raw/input.npz': b'raw-array-bytes',
                       'readiness/metrics.json': b'{}', 'logs/cell.log': b'log',
                       'builds/huge.o': b'build', 'runner-cache/cache': b'cache',
                       'collector-source/experiments/frequency_campaign/inputs/bank.json': b'{}'}.items():
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    before = (run / 'launch-state.json').read_bytes()
    output = export_results(run, tmp_path / 'result.tar.gz')
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert not any('builds/' in name or 'runner-cache/' in name for name in names)
        assert archive.extractfile('jetson-f01ab-results/results/cell/raw/input.npz').read() == b'raw-array-bytes'
        manifest = json.load(archive.extractfile('jetson-f01ab-results/EXPORT.json'))
        assert manifest['status'] == 'completed'
        for name, info in manifest['files'].items():
            assert hashlib.sha256(archive.extractfile('jetson-f01ab-results/' + name).read()).hexdigest() == info['sha256']
    assert (run / 'launch-state.json').read_bytes() == before
    export_results(run, output)  # Packaging is safely repeatable.


def test_latest_completed_skips_newer_failed_run(tmp_path):
    base = tmp_path / 'example_artifacts/jetson_f01ab'
    for name, status in [('001', 'completed'), ('002', 'held')]:
        (base / name).mkdir(parents=True)
        (base / name / 'launch-state.json').write_text(json.dumps({'status': status}))
    assert latest_completed(tmp_path) == base / '001'
    output = export_results(base / '002', tmp_path / 'failed.tar.gz')
    with tarfile.open(output) as archive:
        assert json.load(archive.extractfile('jetson-f01ab-results/EXPORT.json'))['status'] == 'held'
