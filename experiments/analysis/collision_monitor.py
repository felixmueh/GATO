#!/usr/bin/env python3
"""Summarize saved collision-query/worker times without rerunning checks."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording import iter_events, sha, write_json


def table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(raw, output):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    if output.is_relative_to(raw):
        raise ValueError('Keep derived analysis outside retained raw data')
    output.mkdir(parents=True, exist_ok=False)
    attempts, calls, summaries, errors = [], [], [], []
    for path in sorted(raw.rglob('metrics.json')):
        try:
            metrics = json.loads(path.read_text())
            if metrics.get('campaign_id') != 'M02':
                continue
            attempt = path.parent
            receipt = json.loads((attempt / 'receipt.json').read_text()) if (attempt / 'receipt.json').exists() else {}
            provenance = json.loads((attempt / 'provenance.json').read_text()) if (attempt / 'provenance.json').exists() else {}
            identity = dict(attempt=str(attempt), machine=receipt.get('machine'),
                source=receipt.get('source_commit', provenance.get('source')),
                monitor_sha256=provenance.get('monitor_sha256'),
                protocol_sha256=provenance.get('protocol_sha256'),
                urdf_sha256=provenance.get('urdf_sha256'))
            attempts.append(dict(identity, outcome=metrics.get('outcome'),
                execution_error=metrics.get('execution_error'), metrics_sha256=sha(path)))
            if not (attempt / 'events.jsonl').exists():
                continue
            groups = defaultdict(list)
            for event in iter_events(attempt):
                kind, record = event['event'], event['record']
                if kind not in ('query', 'synchronous_worker'):
                    continue
                seconds = record.get('wall_s')
                row = dict(identity, scope=kind, query=record.get('query', 'runtime-margin-worker'),
                    state=record['state']['id'], repeat=record['repeat'], wall_s=seconds,
                    finite=seconds is not None and bool(np.isfinite(seconds)),
                    safety_rejected=bool(record.get('fault')) if kind == 'synchronous_worker' else None)
                calls.append(row)
                groups[(row['scope'], row['query'], row['state'])].append(row)
            for (scope, query, state), rows in groups.items():
                values = [r['wall_s'] * 1000 for r in rows if r['finite']]
                summaries.append(dict(identity, scope=scope, query=query, state=state,
                    calls=len(rows), finite_times=len(values),
                    median_ms=float(np.median(values)) if values else None,
                    p95_ms=float(np.percentile(values, 95)) if values else None,
                    safety_rejections=sum(r['safety_rejected'] is True for r in rows)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(path=str(path), error=str(error)))
    for name, rows in (('attempts', attempts), ('calls', calls), ('timings', summaries)):
        table(output / (name + '.csv'), rows)
    write_json(output / 'summary.json', dict(attempts=len(attempts), calls=len(calls), errors=errors,
        analysis_sha256=sha(__file__), raw=str(raw),
        scope='Per-attempt/per-state CPU timings. Exact-all-pair allocation and runtime first-violation margin are different workloads. Worker repetitions in both query cases use the same margin implementation. No pooling of retries, asynchronous frequency claim or safety certificate.'))
    return bool(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(analyze(args.raw, args.output))
