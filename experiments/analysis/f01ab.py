#!/usr/bin/env python3
"""Saved-data F01ab CSVs and plots; supports chunked and legacy recordings.

Summaries are per attempt/category: retries, sources and protocols are never
pooled. Failed calls with usable timing remain in statistics; unavailable values
are counted explicitly. Reciprocal plots use 1000/aggregate_ms, not mean Hz.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SCOPES = {'wrapper_ms': ('wall_solve_time_s', 1000.), 'native_ms': ('native_solve_time_ms', 1.)}
STAGES = {'solve_started', 'solve_returned', 'solve_completed', 'solve_failed'}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def samples(entry, root):
    attempt = (root / entry.get('attempt', entry.get('path', ''))).resolve()
    if not attempt.is_relative_to(root):
        raise ValueError('Attempt path escapes collection')
    payload = attempt / 'f01-raw' if (attempt / 'f01-raw').is_dir() else attempt
    protocol, provenance, metrics = (read(payload / f'{name}.json') for name in ('protocol', 'provenance', 'metrics'))
    params = protocol.get('parameters', protocol)
    base = dict(attempt=str(attempt.relative_to(root)), machine=entry.get('machine', metrics.get('machine')),
                N=params.get('N', entry.get('N')), SQP=params.get('SQP', entry.get('SQP')),
                PCG=params.get('PCG', entry.get('PCG')), variant=params.get('variant'),
                source_commit=entry.get('source_commit') or provenance.get('source_commit'),
                source_identity=fingerprint(provenance.get('scientific_source_sha256', provenance.get('source_sha256', {}))),
                protocol_identity=fingerprint(protocol), cohort=provenance.get('cohort_id', entry.get('cohort_id')),
                timing_mode=metrics.get('timing_mode', params.get('timing_mode', entry.get('timing_mode'))),
                backend=metrics.get('execution_backend', params.get('backend', entry.get('execution_backend'))),
                evidence_kind=metrics.get('evidence_kind', provenance.get('evidence_kind', 'measurement')),
                execution_error=metrics.get('execution_error', entry.get('error')),
                attempt_status=metrics.get('outcome', entry.get('status', 'unknown')))
    grouped, archive, current_file = {}, None, None

    def timing(value, filename, multiplier):
        nonlocal archive, current_file
        if isinstance(value, dict) and 'npz_key' in value:
            target = (payload / filename).resolve()
            if not target.is_relative_to(payload):
                raise ValueError('Raw path escapes attempt')
            if current_file != target:
                if archive is not None:
                    archive.close()
                archive, current_file = np.load(target, allow_pickle=False), target
            array = archive[value['npz_key']]  # Load only this scalar, never plant/plan arrays.
            if array.shape != ():
                return None, 'invalid_shape'
            value = array.item()
        if value is None:
            return None, 'missing'
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            return None, 'invalid_type'
        value = float(value) * multiplier
        return value, 'nonfinite' if not np.isfinite(value) else 'nonpositive' if value <= 0 else 'finite'

    try:
        events = payload / 'events.jsonl'
        if events.is_file():
            with events.open() as stream:
                for line in stream:
                    if not line.endswith('\n'):  # Abruptly interrupted trailing metadata.
                        break
                    event = json.loads(line)
                    if event['event'] not in STAGES:
                        continue
                    record = event['record']
                    key = (record.get('episode_id'), record.get('slot'))
                    if key not in grouped:
                        grouped[key] = dict(base, episode_id=key[0], slot=key[1], category=record.get('category'),
                            excluded=record.get('excluded', False) or record.get('category') == 'process_cuda_warmup',
                            call_status='started_only', error=None,
                            **{field: None for field in SCOPES},
                            **{field + '_status': 'missing' for field in SCOPES})
                    row = grouped[key]
                    stage = event['event']
                    if stage == 'solve_failed':
                        row['call_status'] = 'failed'
                    elif row['call_status'] != 'failed':
                        row['call_status'] = {'solve_completed': 'completed', 'solve_returned': 'returned',
                                              'solve_started': 'started_only'}[stage]
                    row['error'] = record.get('error', record.get('telemetry_error', row['error']))
                    for scope, (field, multiplier) in SCOPES.items():
                        value, status = timing(record.get(field), event.get('raw_file', ''), multiplier)
                        if status != 'missing':
                            row[scope], row[scope + '_status'] = value, status
        if not grouped:
            # Setup errors are visible even when no solver could be called.
            grouped[None] = dict(base, episode_id=None, slot=None, category='no_solve', excluded=False,
                call_status='no_solve', error=base['execution_error'], **{field: None for field in SCOPES},
                **{field + '_status': 'missing' for field in SCOPES})
        return list(grouped.values()), attempt
    finally:
        if archive is not None:
            archive.close()


def summarize(rows):
    groups = {}
    for row in rows:
        for scope in SCOPES:
            groups.setdefault((row['attempt'], row['category'], scope), []).append(row)
    result = []
    for (_, category, scope), selected in groups.items():
        base = {k: selected[0][k] for k in ('attempt', 'machine', 'N', 'SQP', 'PCG', 'variant', 'source_commit',
                'source_identity', 'protocol_identity', 'cohort', 'timing_mode', 'backend', 'evidence_kind',
                'execution_error', 'attempt_status')}
        values = [row[scope] for row in selected if row[scope + '_status'] == 'finite']
        stats = dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                     p95_ms=float(np.percentile(values, 95))) if values else dict(mean_ms=None, median_ms=None, p95_ms=None)
        result.append(dict(base, category=category, scope=scope, excluded=selected[0]['excluded'],
            calls=sum(r['call_status'] != 'no_solve' for r in selected), failed=sum(r['call_status'] == 'failed' for r in selected), finite=len(values),
            unavailable=len(selected) - len(values), timing_status_counts=json.dumps(dict(Counter(r[scope + '_status'] for r in selected))),
            **stats, **{'reciprocal_' + k + '_Hz': 1000 / v if v is not None else None for k, v in stats.items()}))
    return result


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, summaries, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for index, attempt in enumerate(dict.fromkeys(r['attempt'] for r in rows)):
        selected = [r for r in rows if r['attempt'] == attempt and not r['excluded']]
        identity = next(r for r in rows if r['attempt'] == attempt)
        label = (f"{identity['machine']} N{identity['N']} SQP{identity['SQP']} PCG{identity['PCG']} "
                 f"mode {identity['timing_mode']} / {identity['backend']}\n"
                 f"{identity['evidence_kind']} source {identity['source_commit']} / protocol {identity['protocol_identity'][:10]}\n"
                 f"{attempt} / {identity['attempt_status']}\n"
                 f"Attempt error: {str(identity['execution_error'])[:200] if identity['execution_error'] else 'none'}")
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for column, scope in enumerate(SCOPES):
            axis = axes[0, column]
            all_values = [r[scope] for r in selected if r[scope + '_status'] == 'finite']
            edges = np.linspace(0., max(all_values) * 1.05 if all_values else 1., 21)
            for category in dict.fromkeys(r['category'] for r in selected):
                category_rows = [r for r in selected if r['category'] == category]
                values = [r[scope] for r in category_rows if r[scope + '_status'] == 'finite']
                axis.hist(values, bins=edges, histtype='step', label=f'{category}: {len(values)}/{len(category_rows)} usable')
            unavailable = sum(r[scope + '_status'] != 'finite' for r in selected)
            failed = sum(r['call_status'] == 'failed' for r in selected)
            axis.set(title=f'{scope}: {failed} failed calls; {unavailable} unavailable timings',
                     xlabel='Solve time (ms)', ylabel='Solve count')
            if selected:
                axis.legend(fontsize=8)
            reciprocal = [r for r in summaries if r['attempt'] == attempt and r['scope'] == scope and not r['excluded']]
            lower = axes[1, column]
            for offset, statistic in enumerate(('mean_ms', 'median_ms', 'p95_ms')):
                values = [r['reciprocal_' + statistic + '_Hz'] for r in reciprocal]
                lower.bar(np.arange(len(values)) + offset * .25, [np.nan if v is None else v for v in values],
                          width=.25, label='1000/' + statistic)
            lower.set_xticks(np.arange(len(reciprocal)) + .25, [r['category'] for r in reciprocal])
            lower.set(ylabel='Reciprocal aggregate solve time (Hz)', title='Reciprocal summary; not actuation rate')
            lower.legend(fontsize=8)
        fig.suptitle(label + '\nWarmup exported and excluded from plots; retries remain separate')
        fig.tight_layout()
        fig.savefig(output / f'attempt-{index:03d}.png', dpi=130)
        plt.close(fig)


def analyze(index_path, output, plots=True):
    index_path, output = Path(index_path).resolve(), Path(output).resolve()
    root = index_path.parent
    index = read(index_path)
    entries = index.get('attempts', []) if isinstance(index, dict) else index
    rows, errors = [], []
    for entry in entries:
        if entry.get('experiment') and 'f01ab' not in entry['experiment'].lower():
            continue
        attempt = (root / entry.get('attempt', entry.get('path', ''))).resolve()
        if output == attempt or output.is_relative_to(attempt):
            raise ValueError('Derived output must be outside raw attempts')
        try:
            loaded, attempt = samples(entry, root)
            rows.extend(loaded)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(attempt=entry.get('attempt', entry.get('path')), error=str(error)))
    output.mkdir(parents=True, exist_ok=True)
    summaries = summarize(rows)
    csv_write(output / 'all_samples.csv', rows)
    csv_write(output / 'cell_summaries.csv', summaries)
    (output / 'analysis_errors.json').write_text(json.dumps(errors, indent=2) + '\n')
    if plots and rows:
        plot(rows, summaries, output)
    return rows, summaries, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    _, _, errors = analyze(args.index, args.output, not args.no_plots)
    sys.exit(1 if errors else 0)
