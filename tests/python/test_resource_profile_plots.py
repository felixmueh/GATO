"""CPU saved-event fixtures only; no CUDA module imports or queries."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.resource_profile_campaign import common as C, plot


def fixture(tmp_path):
    index = dict(machine='synthetic-machine', source_commit='synthetic-source', evidence_kind='synthetic_test', attempts=[])
    for arm in ('first_response', 'sustained', 'profile', 'query'):
        case = dict(study='F04' if arm == 'query' else 'F05', arm=arm, N=32, B=1, stage_timing=arm == 'profile')
        if arm != 'query': case.update(SQP=1, PCG=100)
        raw = tmp_path / arm; recorder = C.RAW.RawRecorder(raw)
        C.write_json(raw / 'protocol.json', dict(case, timing_mode='A', backend='internal'))
        C.write_json(raw / 'provenance.json', dict(index, module=dict(flags=dict(GATO_EXPERIMENT_STAGE_TIMING=case['stage_timing']))))
        C.write_json(raw / 'metrics.json', dict(evidence_kind='synthetic_test', status='completed', measured_calls=1, sustained_wall_s=1.2 if arm == 'sustained' else None))
        if arm == 'query':
            recorder('query_returned', dict(payload=dict(device_status=dict(code=0), kernels=[dict(kernel='pcg', occupancy_status=dict(code=1), theoretical_active_blocks_per_SM=None)])))
        else:
            metadata = dict(slot=0, category='measured')
            recorder('solve_started', metadata)
            reply = dict(sqp_time_us=2000., pcg_times_us=[0.])
            if arm == 'profile':
                reply['experiment_profile_stages'] = dict(schema_version=1, units='stream_elapsed_us', pcg_times_us_is_placeholder=True,
                    windows=[dict(stage=name, sqp_iteration=i, elapsed_us=None if name == 'pcg' else 10., unavailable_reason='not_reached' if name == 'pcg' else None) for name, i in C.stage_layout(1)])
            recorder('native_return', dict(native_reply=reply))
            recorder('solve_completed', dict(metadata, native_solve_time_ms=2., wall_solve_time_s=.003))
            recorder('call_finished', dict(metadata, native_solve_time_ms=2., wall_solve_time_s=.003))
        recorder.manifest()
        index['attempts'].append(dict(path=str(raw), manifest_sha256=C.sha(raw / 'SHA256SUMS.json'), case=case))
    path = tmp_path / 'index.json'; C.write_json(path, index)
    return path


def test_actual_fields_render_nulls_and_resource_status(tmp_path):
    path = fixture(tmp_path); loaded = plot.load(path, C.sha(path))
    assert len(loaded['calls']) == 3 and loaded['calls'][0]['native_ms'] == 2
    skipped = next(r for r in loaded['stages'] if r['stage'] == 'pcg')
    assert skipped['elapsed_us'] is None and skipped['unavailable_reason'] == 'not_reached'
    assert loaded['resources'][0]['theoretical_active_blocks_per_SM'] is None
    plot.render(loaded, tmp_path / 'plots')
    assert len(list((tmp_path / 'plots').glob('*.png'))) == 5
    provenance = json.loads((tmp_path / 'plots/provenance.json').read_text())
    assert provenance['pcg_times_us'] is None and 'UNAVAILABLE' in provenance['pcg_times_us_reason']
    assert provenance['index']['evidence_kind'] == 'synthetic_test'


def test_source_hash_and_immutable_guards(tmp_path):
    path = fixture(tmp_path)
    with pytest.raises(ValueError, match='index hash'):
        plot.load(path, 'wrong')
    index = json.loads(path.read_text()); index['source_commit'] = 'wrong'; C.write_json(path, index)
    with pytest.raises(ValueError, match='source/evidence'):
        plot.load(path, C.sha(path))
    index['source_commit'] = 'synthetic-source'; C.write_json(path, index)
    with pytest.raises(ValueError, match='immutable raw'):
        plot.render(plot.load(path, C.sha(path)), tmp_path / 'profile/plots')
    index['attempts'].append(dict(index['attempts'][0], path=None))
    C.write_json(path, index)
    with pytest.raises(ValueError, match='Duplicate selected case'):
        plot.load(path, C.sha(path))
    # A different claimed replicate cannot reuse the same resolved artifact.
    index['attempts'][-1] = dict(index['attempts'][0],
        path=str(tmp_path / 'first_response/../first_response'),
        case=dict(index['attempts'][0]['case'], replicate=99))
    C.write_json(path, index)
    with pytest.raises(ValueError, match='Duplicate selected resolved raw path'):
        plot.load(path, C.sha(path))
