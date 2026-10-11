#!/usr/bin/env python3
"""Saved F01c timing views, preserving attempts, failures and timing denominators.

Read a simple-runner collection index or historical pinned selection index.
Complete same-16 collection ratios are timing comparisons, not equal-quality
speedup claims. Existing correspondence evidence remains authoritative.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording import iter_events, sha, write_json
from analysis.f01ab import csv_write, read, fingerprint

SCOPES = ('native_ms', 'wrapper_ms')
IDENTITIES = ('machine', 'cohort', 'native_source_commit', 'collector_source_commit',
              'module_sha256', 'bank_manifest_sha256', 'timing_mode', 'execution_backend', 'evidence_kind', 'adapter_identity', 'scaling_protocol_identity', 'paired_protocol_identity', 'N', 'SQP', 'PCG')


def valid(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number))
            and np.isfinite(value) and value > 0)


def load_attempt(entry, index, root):
    path = entry.get('attempt', entry.get('path'))
    if path is None:
        raise ValueError('Missing selected attempt')
    attempt = (root / path).resolve()
    if not Path(path).is_absolute() and not attempt.is_relative_to(root):
        raise ValueError('Attempt path escapes collection')
    raw = attempt / 'f01c-raw' if (attempt / 'f01c-raw').is_dir() else attempt
    pin = entry.get('manifest_sha256')
    if pin:
        if sha(raw / 'SHA256SUMS.json') != pin:
            raise ValueError('Pinned attempt manifest differs')
        for relative, digest in read(raw / 'SHA256SUMS.json').items():
            target = (raw / relative).resolve()
            if not target.is_relative_to(raw) or sha(target) != digest:
                raise ValueError('Pinned raw bytes/path differ')
    protocol, provenance, metrics = (read(raw / (name + '.json')) for name in ('protocol', 'provenance', 'metrics'))
    params = protocol.get('parameters', entry.get('parameters', {}))
    metadata = {**index, **entry, **provenance, **protocol, **metrics, **params}
    metadata['module_sha256'] = metadata.get('module_sha256', metadata.get('module', {}).get('sha256'))
    metadata['collector_source_commit'] = metadata.get('collector_source_commit', metadata.get('original_collector_source_commit'))
    metadata['bank_manifest_sha256'] = metadata.get('bank_manifest_sha256', metadata.get('input_sha256', {}).get('SHA256SUMS.json'))
    metadata['adapter_identity'] = fingerprint({k: metadata.get(k) for k in ('adapter_sha256', 'scientific_source_sha256')}) if metadata.get('adapter_sha256') else 'historical-original-collector'
    # B and correspondence diagnostics legitimately differ in a same-pool pair.
    paired_protocol = {k: v for k, v in protocol.items() if k != 'parameters'}
    paired_protocol['parameters'] = {k: v for k, v in params.items() if k not in ('B', 'diagnostics')}
    metadata['paired_protocol_identity'] = fingerprint(paired_protocol)
    scaling_protocol = dict(paired_protocol, parameters={k: v for k, v in paired_protocol['parameters'].items() if k not in ('SQP', 'PCG')})
    metadata['scaling_protocol_identity'] = fingerprint(scaling_protocol)
    base = {key: metadata.get(key) for key in IDENTITIES}
    base['cohort'] = metadata.get('cohort_id', metadata.get('cohort', fingerprint({
        k: metadata.get(k) for k in ('native_source_commit', 'collector_source_commit', 'bank_manifest_sha256')})))
    base.update(attempt=str(attempt), B=params.get('B', metadata.get('B')),
                protocol_identity=fingerprint(protocol), manifest_sha256=pin,
                attempt_status=metrics.get('outcome', entry.get('status', 'unknown')),
                execution_error=metrics.get('execution_error', entry.get('error')))
    calls = {}
    def call(record):
        key = record['call_index']
        return calls.setdefault(key, dict(base, call_index=key, status='started', native_ms=None, wrapper_ms=None))
    if (raw / 'events.jsonl').exists():
        current = None
        for event in iter_events(raw):
            record, kind = event['record'], event['event']
            if 'call_index' in record and kind in ('solve_started', 'solve_returned', 'solve_completed', 'solve_failed', 'call_finished', 'execution_failed'):
                current = record['call_index']
                row = call(record)
                # Large plans/stats stay in raw; preserve call metadata and timings.
                for key in ('category', 'excluded', 'input_id', 'repeat', 'candidate_ids', 'B', 'error', 'error_type'):
                    if key in record:
                        row[key] = record[key]
                for field, source, scale in (('native_ms', 'native_solve_time_ms', 1.), ('wrapper_ms', 'wall_solve_time_s', 1000.)):
                    value = record.get(source, record.get(field))
                    if value is not None:
                        row[field] = float(value) * scale if np.ndim(value) == 0 else None
                if kind in ('solve_failed', 'execution_failed') or record.get('status') == 'failed':
                    row['status'] = 'failed'
                elif row['status'] != 'failed':
                    row['status'] = record.get('status', 'completed' if kind == 'solve_completed' else row['status'])
            elif kind == 'native_return' and current is not None:
                value = record.get('native_reply', {}).get('sqp_time_us')
                if value is not None and calls[current]['native_ms'] is None and np.ndim(value) == 0:
                    calls[current]['native_ms'] = float(value) / 1000.
    for final in metrics.get('calls', []):
        row = call(final)
        row.update(final)  # Collector rejection after native return is authoritative.
        row.update(base)
    rows = list(calls.values())
    if not rows:
        rows = [dict(base, call_index=None, category='no_solve', status='no_solve', native_ms=None, wrapper_ms=None)]
    for row in rows:
        row['excluded'] = row.get('excluded', row.get('category') != 'primary')
        for scope in SCOPES:
            value = row.get(scope)
            row[scope + '_status'] = 'finite' if valid(value) else 'missing' if value is None else 'invalid'
    collections = []
    for record in metrics.get('collections', []):
        selected = [r for r in rows if r.get('category') == 'primary' and
                    (r.get('input_id'), r.get('repeat')) == (record.get('input_id'), record.get('repeat'))]
        candidates = [candidate for r in selected for candidate in r.get('candidate_ids', [])]
        complete = (record.get('complete') is True and len(selected) == 16 // base['B'] and
                    all(r['status'] == 'completed' for r in selected) and sorted(candidates) == list(range(16)))
        collections.append(dict(base, **record, verified_same16=complete))
    return rows, collections, raw, metrics.get('missing_calls', [])


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['attempt'], row.get('category')), []).append(row)
    result = []
    for selected in groups.values():
        for scope in SCOPES:
            values = [r[scope] for r in selected if valid(r.get(scope))]
            stats = {name: None for name in ('mean_ms', 'median_ms', 'p95_ms')}
            if values:
                stats.update(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)))
            base = {k: selected[0].get(k) for k in (*IDENTITIES, 'B', 'attempt', 'attempt_status', 'execution_error', 'protocol_identity', 'category', 'excluded')}
            result.append(dict(base, scope=scope, calls=len(selected), failed=sum(r['status'] == 'failed' for r in selected),
                timed=len(values), unavailable=len(selected)-len(values), status_counts=dict(Counter(r['status'] for r in selected)),
                **stats, **{'reciprocal_' + key + '_Hz': 1000 / value if value else None for key, value in stats.items()}))
    return result


def pair_collections(collections):
    pairs = []
    for batch in collections:
        if batch['B'] != 16:
            continue
        refs = [r for r in collections if r['B'] == 1 and
                all(r.get(k) == batch.get(k) for k in (*IDENTITIES, 'input_id', 'repeat'))]
        identity_complete = all(batch.get(k) is not None for k in IDENTITIES if k != 'cohort')
        for scope in SCOPES:
            eligible = identity_complete and len(refs) == 1 and batch['verified_same16'] and refs[0]['verified_same16']
            scalar = refs[0] if len(refs) == 1 else None
            available = eligible and valid(batch.get(scope)) and valid(scalar.get(scope))
            pairs.append(dict(batch_attempt=batch['attempt'], serial_attempt=scalar['attempt'] if scalar else None,
                **{k: batch.get(k) for k in (*IDENTITIES, 'input_id', 'repeat')}, scope=scope,
                status='matched_complete_same16' if available else 'unavailable',
                reason=None if available else 'missing identity, ambiguous retry, incomplete collection or unavailable timing',
                serial_ms=scalar.get(scope) if scalar else None, batch_ms=batch.get(scope),
                serial_over_batch=scalar[scope]/batch[scope] if available else None,
                interpretation='solve-only time ratio; no equal-quality speedup or task-success claim'))
    return pairs


def plot(summaries, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for i, attempt in enumerate(dict.fromkeys(r['attempt'] for r in summaries)):
        selected = [r for r in summaries if r['attempt'] == attempt]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for scope in SCOPES:
            rows = [r for r in selected if r['scope'] == scope]
            labels = [f"{r['category']} ({r['timed']}/{r['calls']}, failed {r['failed']})" for r in rows]
            for axis, field in zip(axes, ('mean_ms', 'reciprocal_mean_ms_Hz')):
                axis.plot(labels, [np.nan if r[field] is None else r[field] for r in rows], 'o', label=scope)
                axis.tick_params(axis='x', labelrotation=30)
                axis.legend()
        axes[0].set_ylabel('Mean solve time (ms)')
        axes[1].set_ylabel('1000/mean solve time (Hz)')
        r = selected[0]
        fig.suptitle(f"{r['machine']} B{r['B']} N{r['N']} SQP{r['SQP']} PCG{r['PCG']} / {r['evidence_kind']}\n{attempt}\nAll categories shown; reciprocal is not actuation rate", fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f'attempt-{i:03d}.png', dpi=130)
        plt.close(fig)


def plot_scaling(summaries, output):
    """One point per primary attempt; never combine retries into a batch-size mean."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = {}
    keys = tuple(k for k in IDENTITIES if k not in ('SQP', 'PCG', 'paired_protocol_identity'))
    for row in summaries:
        if row['category'] == 'primary':
            groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    for i, rows in enumerate(groups.values()):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for col, scope in enumerate(SCOPES):
            selected = [r for r in rows if r['scope'] == scope]
            for budget in sorted({(r['SQP'], r['PCG']) for r in selected}):
                series = [r for r in selected if (r['SQP'], r['PCG']) == budget]
                for axis, field in zip(axes[:, col], ('mean_ms', 'reciprocal_mean_ms_Hz')):
                    axis.scatter([r['B'] for r in series], [np.nan if r[field] is None else r[field] for r in series],
                                 label=f'SQP{budget[0]}/PCG{budget[1]}')
                    for r in series:
                        if r[field] is not None:
                            axis.annotate(f"{r['timed']}/{r['calls']}; failed {r['failed']}", (r['B'], r[field]), fontsize=6)
                    axis.set_xticks((1, 2, 4, 8, 16)); axis.set_xlabel('Batch size B')
                    axis.legend(fontsize=7); axis.grid(alpha=.2)
            axes[0, col].set(title=scope, ylabel='Mean solve time (ms)')
            axes[1, col].set_ylabel('1000/mean solve time (Hz)')
        r = rows[0]
        fig.suptitle(f"F01c {r['machine']} N{r['N']} / {r['timing_mode']} / {r['execution_backend']} / {r['evidence_kind']}\n"
                     "Primary calls: one point per attempt; retries retained individually; no equal-quality claim", fontsize=9)
        fig.tight_layout()
        fig.savefig(output / f'cohort-{i:03d}-scaling.png', dpi=130)
        plt.close(fig)


def analyze(index_path, output, plots=True):
    index_path, output = Path(index_path).resolve(), Path(output).resolve()
    index = read(index_path)
    entries = index.get('attempts', []) if isinstance(index, dict) else index
    defaults = index if isinstance(index, dict) else {}
    rows, collections, missing, errors = [], [], [], []
    for entry in entries:
        if entry.get('experiment') and 'f01c' not in entry['experiment'].lower():
            continue
        path = entry.get('attempt', entry.get('path'))
        if path is not None and output.is_relative_to((index_path.parent / path).resolve()):
            raise ValueError('Derived output must be outside raw attempts')
        try:
            calls, groups, raw, absent = load_attempt(entry, defaults, index_path.parent)
            rows.extend(calls); collections.extend(groups)
            missing.extend(dict(attempt=str(raw), **r) for r in absent)
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(dict(attempt=path, error=str(error), entry=entry))
    output.mkdir(parents=True, exist_ok=True)
    summaries, pairs = summarize(rows), pair_collections(collections)
    for name, data in (('calls', rows), ('call_summaries', summaries), ('collections', collections), ('same16_pairs', pairs), ('missing_calls', missing)):
        csv_write(output / (name + '.csv'), data)
    write_json(output / 'analysis_errors.json', errors)
    write_json(output / 'provenance.json', dict(index_sha256=sha(index_path), analysis_sha256=sha(__file__),
        interpretation='Per-attempt timing; raw failures retained; known historical lane differences preclude equal-quality speedup claims'))
    if plots and summaries:
        plot(summaries, output)
        plot_scaling(summaries, output)
    return rows, summaries, collections, pairs, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    sys.exit(bool(analyze(args.index, args.output, not args.no_plots)[-1]))
