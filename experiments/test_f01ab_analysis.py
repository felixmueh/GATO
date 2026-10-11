"""Saved-only mixed-format checks, including lazy scalar reads and reciprocal."""
import json

import numpy as np
import pytest

from analysis import f01ab
from recording import RawRecorder, write_json


def metadata(path, source):
    write_json(path / 'protocol.json', dict(N=32, SQP=5, PCG=100, variant='short12', timing_mode='A', backend='internal'))
    write_json(path / 'provenance.json', dict(source_commit=source, cohort_id='frozen', evidence_kind='synthetic_test'))
    write_json(path / 'metrics.json', dict(machine='test', timing_mode='A', execution_backend='internal',
                                        evidence_kind='synthetic_test', outcome='execution_failed', execution_error='retained error'))


def test_mixed_formats_no_array_bulk_reads_and_reciprocal(tmp_path, monkeypatch):
    current = tmp_path / 'new/attempt-001'
    with RawRecorder(current) as recorder:
        metadata(current, 'new-source')
        for slot, wall, native in [(0, .001, .5), (1, .003, 1.5)]:
            record = dict(episode_id='reach', slot=slot, category='cold', output=np.arange(200))
            recorder('solve_started', record)
            recorder('solve_returned', dict(record, wall_solve_time_s=wall, native_solve_time_ms=np.float32(native)))
            recorder('solve_completed', dict(record, wall_solve_time_s=wall, native_solve_time_ms=np.float32(native)))
        recorder('solve_failed', dict(episode_id='reach', slot=2, category='warm', error='failed call',
                                     wall_solve_time_s=float('nan')))
        recorder('solve_completed', dict(episode_id=None, slot=None, category='process_cuda_warmup',
                                        excluded=True, wall_solve_time_s=.0001, native_solve_time_ms=.1))
    old = tmp_path / 'old/attempt-001/f01-raw'
    (old / 'raw').mkdir(parents=True)
    metadata(old, 'old-source')
    np.savez(old / 'raw/legacy.npz', bad_time=np.asarray(np.inf), poison=np.array([object()], dtype=object))
    event = dict(event_index=0, event='solve_failed', raw_file='raw/legacy.npz', record=dict(
        episode_id='reach', slot=0, category='cold', wall_solve_time_s={'npz_key': 'bad_time', 'nonfinite_scalar': 'inf'}))
    (old / 'events.jsonl').write_text(json.dumps(event) + '\n')
    entries = [dict(experiment='F01ab', attempt=str(path), machine='test') for path in ('new/attempt-001', 'old/attempt-001')]
    write_json(tmp_path / 'index.json', entries)
    real_load, accessed = np.load, []

    class ScalarsOnly:
        def __init__(self, path, **kwargs):
            self.archive = real_load(path, **kwargs)
        def __getitem__(self, key):
            accessed.append(key)
            value = self.archive[key]
            assert value.shape == (), 'Analysis read a full plan/plant array'
            return value
        def close(self):
            self.archive.close()

    monkeypatch.setattr(f01ab.np, 'load', ScalarsOnly)
    rows, summaries, errors = f01ab.analyze(tmp_path / 'index.json', tmp_path / 'derived', plots=False)
    assert errors == [] and len(rows) == 5
    assert len([r for r in rows if r['excluded']]) == 1
    new_cold = next(s for s in summaries if s['attempt'].startswith('new') and s['category'] == 'cold' and s['scope'] == 'wrapper_ms')
    assert new_cold['calls'] == 2 and new_cold['mean_ms'] == 2.
    assert new_cold['reciprocal_mean_ms_Hz'] == 500.  # not mean(1000/1,1000/3)
    old_cold = next(s for s in summaries if s['attempt'].startswith('old') and s['scope'] == 'wrapper_ms')
    assert old_cold['mean_ms'] is None and old_cold['unavailable'] == 1 and old_cold['failed'] == 1
    assert np.isposinf(next(r for r in rows if r['attempt'].startswith('old'))['wrapper_ms'])
    failed = next(r for r in rows if r['category'] == 'warm')
    assert failed['call_status'] == 'failed' and failed['wrapper_ms_status'] == 'nonfinite'
    assert failed['native_ms_status'] == 'missing' and np.isnan(failed['wrapper_ms'])
    assert 'poison' not in accessed
    assert new_cold['source_commit'] != old_cold['source_commit']
    assert (tmp_path / 'derived/all_samples.csv').is_file()


def test_missing_attempt_and_output_safety(tmp_path):
    missing = tmp_path / 'attempt-001'
    missing.mkdir()
    metadata(missing, 'source')
    entries = [dict(experiment='F01ab', attempt='attempt-001')]
    write_json(tmp_path / 'index.json', entries)
    rows, summaries, errors = f01ab.analyze(tmp_path / 'index.json', tmp_path / 'derived', plots=False)
    assert errors == [] and rows[0]['call_status'] == 'no_solve'
    assert rows[0]['execution_error'] == 'retained error'
    assert all(s['execution_error'] == 'retained error' and s['attempt_status'] == 'execution_failed' for s in summaries)
    assert rows[0]['attempt_status'] == 'execution_failed'
    with pytest.raises(ValueError, match='outside raw'):
        f01ab.analyze(tmp_path / 'index.json', missing / 'derived', plots=False)
    assert not (missing / 'derived').exists()


def test_plot_generation(tmp_path):
    attempt = tmp_path / 'attempt-001'
    with RawRecorder(attempt) as recorder:
        metadata(attempt, 'source')
        recorder('solve_completed', dict(episode_id='reach', slot=0, category='cold',
                                        wall_solve_time_s=.002, native_solve_time_ms=1.5))
    write_json(tmp_path / 'index.json', [dict(experiment='F01ab', attempt='attempt-001')])
    f01ab.analyze(tmp_path / 'index.json', tmp_path / 'plots')
    assert (tmp_path / 'plots/attempt-000.png').stat().st_size > 0
