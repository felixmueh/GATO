"""CPU-only synthetic timings, never scientific measurement evidence."""
import json

import numpy as np
import pytest

from analysis import f01c
from recording import RawRecorder, write_json


def attempt(root, name, B, fail=False, legacy=False, cohort='synthetic'):
    path = root / name
    protocol = dict(parameters=dict(N=32, B=B, SQP=5, PCG=100), timing_mode='A',
        execution_backend='internal', native_source_commit='native', module_sha256='synthetic-module', collector_source_commit='collector',
        bank_manifest_sha256='bank')
    records = []
    calls = []
    for i in range(16 // B):
        row = dict(call_index=i, category='primary', input_id='input', repeat=0, B=B,
                   candidate_ids=list(range(i*B, (i+1)*B)), status='failed' if fail and i == 0 else 'completed',
                   native_ms=float(B), wrapper_ms=float(B*2))
        records.append(('solve_completed', dict(row, native_solve_time_ms=np.float32(B), wall_solve_time_s=B*.002)))
        calls.append(row)
    records.append(('solve_completed', dict(call_index=100, category='process_cuda_warmup', excluded=True,
        native_solve_time_ms=.5, wall_solve_time_s=.001)))
    if legacy:
        (path / 'raw').mkdir(parents=True)
        events = []
        for i, (kind, record) in enumerate(records):
            value = record.pop('native_solve_time_ms')
            filename = f'raw/{i}.npz'
            np.savez(path / filename, time=np.asarray(value))
            record['native_solve_time_ms'] = dict(npz_key='time', shape=[], dtype='float32')
            events.append(dict(event=kind, record=record, raw_file=filename))
        (path / 'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
    else:
        with RawRecorder(path, max_events=3) as recorder:
            for kind, record in records:
                recorder(kind, record)
    write_json(path / 'protocol.json', protocol)
    write_json(path / 'provenance.json', dict(cohort_id=cohort))
    write_json(path / 'metrics.json', dict(machine='synthetic-machine', evidence_kind='synthetic_test', calls=calls,
        outcome='execution_failed' if fail else 'timing_completed', execution_error='retained failure' if fail else None,
        collections=[dict(input_id='input', repeat=0, B=B, complete=not fail, completed_calls=len(calls),
                          expected_calls=16//B, native_ms=16., wrapper_ms=32.)]))
    return dict(experiment='F01c', attempt=name)


def test_mixed_formats_and_same16(tmp_path):
    entries = [attempt(tmp_path, 'serial', 1, legacy=True), attempt(tmp_path, 'batch', 16)]
    write_json(tmp_path / 'index.json', entries)
    rows, summaries, collections, pairs, errors = f01c.analyze(tmp_path/'index.json', tmp_path/'derived', False)
    assert not errors
    assert len(rows) == 19 and len(collections) == 2
    assert len([r for r in rows if r['category'] == 'process_cuda_warmup']) == 2
    assert all(r['evidence_kind'] == 'synthetic_test' for r in rows)
    assert all(r['status'] == 'matched_complete_same16' and r['serial_over_batch'] == 1. for r in pairs)
    scalar = next(r for r in summaries if r['B'] == 1 and r['scope'] == 'native_ms' and r['category'] == 'primary')
    assert scalar['mean_ms'] == 1. and scalar['reciprocal_mean_ms_Hz'] == 1000.


def test_failures_retries_and_output_safety(tmp_path):
    entries = [attempt(tmp_path, 'serial', 1), attempt(tmp_path, 'failed', 16, fail=True)]
    write_json(tmp_path/'index.json', entries)
    rows, summaries, _, pairs, errors = f01c.analyze(tmp_path/'index.json', tmp_path/'derived', False)
    assert not errors and all(p['status'] == 'unavailable' for p in pairs)
    failed = next(r for r in summaries if r['attempt'].endswith('failed') and r['category'] == 'primary')
    assert failed['failed'] == 1 and failed['timed'] == 1 and failed['execution_error'] == 'retained failure'
    entries.append(attempt(tmp_path, 'retry', 1))
    write_json(tmp_path/'index.json', entries)
    assert all(p['status'] == 'unavailable' for p in f01c.analyze(tmp_path/'index.json', tmp_path/'retry-derived', False)[3])
    with pytest.raises(ValueError, match='outside raw'):
        f01c.analyze(tmp_path/'index.json', tmp_path/'serial/derived', False)


def test_historical_absolute_index_pin_and_plot(tmp_path):
    entry = attempt(tmp_path, 'old', 16, legacy=True)
    path = tmp_path/'old'
    write_json(path/'SHA256SUMS.json', {p.name: f01c.sha(p) for p in path.glob('*.json')})
    historical = dict(machine='synthetic-machine', attempts=[dict(path=str(path), parameters=dict(N=32,B=16,SQP=5,PCG=100),
        manifest_sha256=f01c.sha(path/'SHA256SUMS.json'))])
    write_json(tmp_path/'selection.json', historical)
    assert not f01c.analyze(tmp_path/'selection.json', tmp_path/'plots')[-1]
    assert (tmp_path/'plots/attempt-000.png').is_file()
    historical['attempts'][0]['manifest_sha256'] = 'wrong'
    write_json(tmp_path/'selection.json', historical)
    assert 'manifest differs' in f01c.analyze(tmp_path/'selection.json', tmp_path/'bad', False)[-1][0]['error']


def test_simple_adapter_aliases_and_changed_helper(tmp_path):
    entries = [attempt(tmp_path, 'serial', 1), attempt(tmp_path, 'batch', 16)]
    for entry in entries:
        path = tmp_path / entry['attempt']
        protocol = json.loads((path/'protocol.json').read_text())
        protocol.pop('collector_source_commit')
        protocol.pop('bank_manifest_sha256')
        write_json(path/'protocol.json', protocol)
        write_json(path/'provenance.json', dict(original_collector_source_commit='collector',
            input_sha256={'SHA256SUMS.json': 'bank'}, adapter_sha256={'f01c.py': 'same'},
            scientific_source_sha256={'episode.py': 'same'}))
    write_json(tmp_path/'index.json', entries)
    pairs = f01c.analyze(tmp_path/'index.json', tmp_path/'derived', False)[3]
    assert all(p['status'] == 'matched_complete_same16' for p in pairs)
    provenance = json.loads((tmp_path/'batch/provenance.json').read_text())
    provenance['adapter_sha256']['f01c.py'] = 'changed'
    write_json(tmp_path/'batch/provenance.json', provenance)
    assert all(p['status'] == 'unavailable' for p in f01c.analyze(tmp_path/'index.json', tmp_path/'changed', False)[3])


def test_changed_native_module_does_not_pair(tmp_path):
    entries = [attempt(tmp_path, 'serial', 1), attempt(tmp_path, 'batch', 16)]
    protocol = json.loads((tmp_path/'batch/protocol.json').read_text())
    protocol['module_sha256'] = 'other-module'
    write_json(tmp_path/'batch/protocol.json', protocol)
    write_json(tmp_path/'index.json', entries)
    assert all(p['status'] == 'unavailable' for p in f01c.analyze(tmp_path/'index.json', tmp_path/'derived', False)[3])
