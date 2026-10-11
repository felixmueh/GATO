"""CPU checks for evidence preservation and interruption boundaries."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from recording import RawRecorder, iter_events, sha


def test_roundtrip_and_snapshot(tmp_path):
    array = np.array([[1, np.nan], [np.inf, -np.inf]], dtype='>f8')
    expected = array.copy()
    with RawRecorder(tmp_path, max_events=2) as recorder:
        recorder('input', {'array': array, 'empty': np.empty((0, 3), dtype=np.int16),
                           'nested': [np.float32(1.25), float('nan'), np.uint64(2**63)],
                           'a__b': np.array([3]), 'a': {'b': np.array([4])}})
        array[:] = 99
        recorder('output', {'flags': np.array([True, False]), 'count': 7})
    events = list(iter_events(tmp_path))
    assert [e['event'] for e in events] == ['input', 'output']
    result = events[0]['record']
    np.testing.assert_array_equal(result['array'], expected)
    assert result['array'].dtype == expected.dtype
    assert result['empty'].shape == (0, 3)
    assert result['empty'].dtype == np.dtype('int16')
    assert type(result['nested'][0]) is np.float32
    assert np.isnan(result['nested'][1])
    assert type(result['nested'][2]) is np.uint64
    assert result['nested'][2] == 2**63
    assert result['a__b'][0] == 3 and result['a']['b'][0] == 4
    assert len(list((tmp_path / 'raw').glob('*.npz'))) == 1


def test_partial_chunk_flushed_on_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with RawRecorder(tmp_path) as recorder:
            recorder('input', {'x': np.arange(3)})
            raise RuntimeError('worker failed')
    assert len(list(iter_events(tmp_path))) == 1
    recorder.manifest()
    manifest = json.loads((tmp_path / 'SHA256SUMS.json').read_text())
    assert manifest['events.jsonl'] == sha(tmp_path / 'events.jsonl')
    assert len(manifest) == 2
    with pytest.raises(ValueError, match='closed'):
        recorder('late', {})


def test_byte_and_count_limits(tmp_path):
    recorder = RawRecorder(tmp_path, max_events=3, max_bytes=1500)
    for i in range(10):
        recorder('solve', {'x': np.ones(100), 'i': i})
        assert recorder.buffer_bytes < 1500
        assert len(recorder._events) < 3
    recorder.close()
    assert len(list(iter_events(tmp_path))) == 10
    # An individual event larger than the limit is flushed immediately.
    other = tmp_path / 'large'
    with RawRecorder(other, max_bytes=8) as recorder:
        recorder('large', {'x': np.arange(1000)})
        assert not recorder._events and recorder.buffer_bytes == 0
    assert len(list(iter_events(other))) == 1


def test_hard_exit_retains_completed_chunks(tmp_path):
    source = Path(__file__).with_name('recording.py').parent
    script = """
import os, sys
sys.path.insert(0, sys.argv[1])
from recording import RawRecorder
r = RawRecorder(sys.argv[2], max_events=2)
for i in range(3): r('solve', {'i': i})
os._exit(9)
"""
    completed = subprocess.run([sys.executable, '-c', script, str(source), str(tmp_path)])
    assert completed.returncode == 9
    assert [e['record']['i'] for e in iter_events(tmp_path)] == [0, 1]
    with (tmp_path / 'events.jsonl').open('a') as stream:
        stream.write('{"interrupted":')
    assert len(list(iter_events(tmp_path))) == 2


def test_legacy_reader(tmp_path):
    (tmp_path / 'raw').mkdir()
    np.savez_compressed(tmp_path / 'raw/event.npz', record_x=np.array([1, 2], dtype=np.int32),
                        record_bad=np.asarray(np.inf))
    metadata = {'event_index': 0, 'event': 'solve', 'raw_file': 'raw/event.npz',
                'record': {'x': {'npz_key': 'record_x', 'shape': [2], 'dtype': 'int32'},
                           'bad': {'npz_key': 'record_bad', 'nonfinite_scalar': 'inf'}}}
    (tmp_path / 'events.jsonl').write_text(json.dumps(metadata) + '\n')
    record = next(iter_events(tmp_path))['record']
    np.testing.assert_array_equal(record['x'], [1, 2])
    assert record['x'].dtype == np.int32 and np.isposinf(record['bad'])
    with pytest.raises(FileExistsError):
        RawRecorder(tmp_path)


def test_manifest_reuses_own_chunk_hashes_and_rehashes_summaries(tmp_path,monkeypatch):
    import recording
    recorder=RawRecorder(tmp_path,max_events=1)
    recorder('input',{'x':np.arange(10,dtype=np.float32)})
    recorder('output',{'x':np.arange(5,dtype=np.int64)})
    recorder.close()
    expected={str(p.relative_to(tmp_path)):sha(p) for p in (tmp_path/'raw').glob('*.npz')}
    (tmp_path/'metrics.json').write_text('{"result":1}')
    # A foreign/recovered file has no cached digest and must still be hashed.
    external=tmp_path/'raw/external.npz';np.savez(external,x=np.array([9]))
    reads=[];original=recording.sha
    def count_reads(path):reads.append(str(Path(path).relative_to(tmp_path)));return original(path)
    monkeypatch.setattr(recording,'sha',count_reads)
    recorder.manifest()
    manifest=json.loads((tmp_path/'SHA256SUMS.json').read_text())
    assert all(manifest[name]==digest for name,digest in expected.items())
    assert not set(expected)&set(reads)
    assert 'raw/external.npz' in reads and 'metrics.json' in reads
    (tmp_path/'metrics.json').write_text('{"result":2}')
    recorder.manifest()
    assert json.loads((tmp_path/'SHA256SUMS.json').read_text())['metrics.json']==original(tmp_path/'metrics.json')
    # Resume continues through a new attempt; old attempts are never reopened.
    with pytest.raises(FileExistsError):RawRecorder(tmp_path)
