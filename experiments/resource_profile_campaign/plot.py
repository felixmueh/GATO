"""CPU saved F04/F05 tables and views; no module import/query/solve.

Pinned index: machine, source_commit, evidence_kind, attempts. Each attempt has
path, manifest_sha256 and case matching the saved protocol (path=null is missing).
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.analysis.plot_f01ab import json_read, sha, verify_manifest, write_csv
from experiments.resource_profile_campaign.common import validate_profile


def decode(value, arrays):
    if isinstance(value, dict):
        if 'npz_key' in value:
            array = arrays[value['npz_key']]
            return array.item() if array.shape == () else array.tolist()
        return {k: decode(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v, arrays) for v in value]
    return value


def load(index_path, pin):
    if sha(index_path) != pin:
        raise ValueError('Selection index hash differs')
    index = json_read(index_path); attempts = []; calls = []; stages = []; resources = []
    seen_cases = set(); seen_paths = set()
    for entry in index['attempts']:
        case = entry['case']; identity = dict(case, attempt_path=entry['path'])
        case_key = json.dumps(case, sort_keys=True, allow_nan=False)
        if case_key in seen_cases:
            raise ValueError('Duplicate selected case')
        seen_cases.add(case_key)
        if entry['path'] is None:
            attempts.append(dict(identity, status='missing')); continue
        attempts_payload = None
        raw = Path(entry['path']).resolve()
        if raw in seen_paths:
            raise ValueError('Duplicate selected resolved raw path')
        seen_paths.add(raw)
        manifest = verify_manifest(raw, entry['manifest_sha256'])
        protocol = json_read(raw / 'protocol.json'); provenance = json_read(raw / 'provenance.json'); metrics = json_read(raw / 'metrics.json')
        if any(protocol.get(k) != v for k, v in case.items()) or protocol.get('timing_mode') != 'A' or protocol.get('backend') != 'internal':
            raise ValueError('Selected case/caps/mode/backend differs')
        if any(provenance.get(k) != index[k] for k in ('machine', 'source_commit', 'evidence_kind')) or metrics['evidence_kind'] != index['evidence_kind']:
            raise ValueError('Selected machine/source/evidence differs')
        if provenance['module']['flags']['GATO_EXPERIMENT_STAGE_TIMING'] != case['stage_timing']:
            raise ValueError('Selected stage build differs')
        grouped = {}; current = None
        for line in (raw / 'events.jsonl').read_text().splitlines():
            event = json.loads(line); record = event['record']; kind = event['event']
            if event['raw_file'] not in manifest or event['sha256'] != manifest[event['raw_file']]:
                raise ValueError('Unpinned event')
            if kind == 'query_returned':
                with np.load(raw / event['raw_file'], allow_pickle=False) as arrays:
                    payload = decode(record['payload'], arrays)
                resources.extend(dict(identity, **row) for row in payload['kernels'])
                attempts_payload = payload
            if 'slot' in record:
                current = record['slot']
                row = grouped.setdefault(current, dict(identity, slot=current, category=record['category'], status='started', native_ms=None, wrapper_ms=None))
                if kind in ('solve_returned', 'solve_completed', 'solve_failed', 'call_finished'):
                    with np.load(raw / event['raw_file'], allow_pickle=False) as arrays:
                        for key, target, scale in (('native_solve_time_ms', 'native_ms', 1), ('wall_solve_time_s', 'wrapper_ms', 1000)):
                            if key in record:
                                value = decode(record[key], arrays)
                                row[target] = value * scale if value is not None and np.isfinite(value) and value > 0 else None
                if kind in ('solve_failed', 'execution_failed'):
                    row.update(status='failed', error=record.get('error'))
                elif kind == 'call_finished':
                    row['status'] = 'completed'
            if kind == 'native_return' and current is not None:
                with np.load(raw / event['raw_file'], allow_pickle=False) as arrays:
                    reply = record['native_reply']; value = decode(reply.get('sqp_time_us'), arrays)
                    if value is not None and np.isfinite(value) and value > 0:
                        grouped[current]['native_ms'] = value / 1000
                    profile = decode(reply.get('experiment_profile_stages'), arrays)
                if profile is not None:
                    if not case['stage_timing']:
                        raise ValueError('Profile present in OFF timing arm')
                    validate_profile(profile, case['SQP'])
                    stages.extend(dict(identity, slot=current, category=grouped[current]['category'], **window) for window in profile['windows'])
                elif case['stage_timing']:
                    stages.append(dict(identity, slot=current, category=grouped[current]['category'], stage='profile', elapsed_us=None, unavailable_reason='profile_missing'))
        calls.extend(grouped.values())
        attempts.append(dict(identity, protocol=protocol, provenance=provenance, metrics=metrics,
                             resources=attempts_payload,
                             manifest_sha256=entry['manifest_sha256']))
    return dict(index=index, index_sha256=pin, attempts=attempts, calls=calls, stages=stages, resources=resources)


def render(loaded, output):
    output = Path(output).resolve()
    if any(entry['path'] is not None and output.is_relative_to(Path(entry['path']).resolve()) for entry in loaded['index']['attempts']):
        raise ValueError('Derived output in immutable raw')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('calls', 'stages', 'resources', 'attempts'):
        write_csv(output / (name + '.csv'), loaded[name])
    (output / 'provenance.json').write_text(json.dumps(dict(loaded, plot_sha256=sha(__file__),
        pcg_times_us=None, pcg_times_us_reason='UNAVAILABLE: native literal-zero placeholder; never a measured stage',
        limitations='Stage ON overhead separate; stream-elapsed windows include idle gaps. Short loop is not thermal steady-state. No actuation/task-quality claim.'), indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    index = loaded['index']; label = 'SYNTHETIC TEST DATA — NOT MEASUREMENT' if index['evidence_kind'] == 'synthetic_test' else 'F04/F05 saved measurements'
    timing_present = any(r['study'] == 'F05' for r in loaded['attempts'])
    for scope in (('native', 'wrapper') if timing_present else ()):
        for reciprocal in (False, True):
            fig, axes = plt.subplots(1, 2, figsize=(13, 6))
            for axis, arm in zip(axes, ('first_response', 'sustained')):
                selected = [r for r in loaded['calls'] if r['arm'] == arm and r['category'] == 'measured']
                for B in (1, 16):
                    for S, P in ((1, 100), (5, 100), (10, 800)):
                        series = [r for r in selected if (r['B'], r['SQP'], r['PCG']) == (B, S, P)]
                        values = [r[scope + '_ms'] for r in series]
                        axis.plot(range(len(values)), [np.nan if v is None else 1000/v if reciprocal else v for v in values], 'o', markersize=3, label=f'B{B} S{S}/P{P}, n={len(values)}')
                axis.set_title(arm.replace('_', ' ') + ' / stage OFF')
                axis.set_xlabel('recorded call index (each series)')
                axis.set_ylabel('1/solve time (calls/s)' if reciprocal else 'solve time (ms)'); axis.legend(fontsize=7); axis.grid(alpha=.2)
            durations = [r.get('metrics', {}).get('sustained_wall_s') for r in loaded['attempts'] if r['arm'] == 'sustained']
            fig.suptitle(f"{label}\n{index['machine']} / A / internal / N32 / {scope} / source {index['source_commit']}\nactual saved sustained duration(s): {durations}\nSolve-only reciprocals; constructor/reset/record loop duration separately retained", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, .82)); fig.savefig(output / f"{scope}-{'reciprocal' if reciprocal else 'solve-time'}.png", dpi=130); plt.close(fig)
    if loaded['stages']:
        measured = [r for r in loaded['stages'] if r['category'] == 'measured']
        grouped = defaultdict(list)
        for row in measured:
            grouped[(row['B'], row['SQP'], row['PCG'], row['stage'])].append(row)
        stage_summary = []
        for (B, S, P, name), windows in grouped.items():
            values = [r['elapsed_us'] for r in windows if r['elapsed_us'] is not None]
            stage_summary.append(dict(B=B, SQP=S, PCG=P, stage=name, reached=len(values),
                unavailable=len(windows)-len(values),
                mean_stream_elapsed_us=float(np.mean(values)) if values else None))
        write_csv(output / 'stage_summary.csv', stage_summary)
        names = list(dict.fromkeys(r['stage'] for r in measured))
        fig, axis = plt.subplots(figsize=(13, 6))
        for B, S, P in dict.fromkeys((r['B'], r['SQP'], r['PCG']) for r in measured):
            series = [r for r in stage_summary if (r['B'], r['SQP'], r['PCG']) == (B, S, P)]
            axis.plot([names.index(r['stage']) for r in series],
                [np.nan if r['mean_stream_elapsed_us'] is None else r['mean_stream_elapsed_us'] for r in series],
                'o', markersize=4, label=f'B{B} S{S}/P{P}')
            for r in series:
                axis.annotate(f"{r['reached']}/{r['reached']+r['unavailable']}",
                    (names.index(r['stage']), r['mean_stream_elapsed_us'] or 0), fontsize=6)
        axis.set_xticks(range(len(names)), names, rotation=35, ha='right', fontsize=8)
        axis.legend(fontsize=8)
        axis.set_ylabel('mean reached window stream elapsed (µs), includes idle gaps'); axis.set_title(f"{label}\n{index['machine']} / A / internal / N32 / source {index['source_commit']}\nStage ON separate cohort; reached/all window counts; null/skipped retained; pcg_times_us UNAVAILABLE", fontsize=9)
        fig.tight_layout(); fig.savefig(output / 'stage-windows.png', dpi=130); plt.close(fig)
    return loaded


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True); parser.add_argument('--index-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    render(load(args.index, args.index_sha256), args.output)
