#!/usr/bin/env python3
"""Saved control-choice outcomes and per-attempt reaching traces; no solves."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording import sha, write_json


def analyze(raw, output):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    if output.is_relative_to(raw):
        raise ValueError('Keep derived analysis outside retained raw data')
    output.mkdir(parents=True, exist_ok=False)
    rows, errors = [], []
    for path in sorted(raw.rglob('metrics.json')):
        try:
            metrics = json.loads(path.read_text())
            if 'control_variant' not in metrics:
                continue
            attempt = path.parent
            provenance_path = attempt / 'provenance.json'
            provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
            row = dict(attempt=str(attempt), metrics_sha256=sha(path),
                machine=metrics.get('machine'), source_commit=provenance.get('source_commit'),
                module_sha256=provenance.get('module_sha256'),
                protocol=metrics.get('collection_protocol'),
                variant=metrics['control_variant'], task=metrics.get('base_task'),
                timing_mode=metrics.get('timing_mode'), backend=metrics.get('execution_backend'))
            for key in ('outcome', 'task_success', 'physical_pass', 'execution_error',
                        'observation_completed', 'first_rough_success_s', 'final_position_error_m',
                        'final_tool_axis_error_deg', 'final_full_SO3_error_deg', 'final_speed_L1_rad_s',
                        'tracking_RMS_m', 'tracking_maximum_m'):
                row[key] = metrics.get(key)
            rows.append(row)
            trace_path = attempt / 'executed_trace.csv'
            if not trace_path.exists():
                continue
            with trace_path.open() as stream:
                trace = list(csv.DictReader(stream))
            if not trace:
                continue
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            time = [float(r['time_s']) for r in trace]
            figure, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
            for axis, field, label in zip(axes,
                    ('position_error_m', 'tool_axis_error_deg', 'speed_L1_rad_s'),
                    ('Position error (m)', 'Tool-axis error (degrees)', 'Joint-speed L1 (rad/s)')):
                axis.plot(time, [float(r[field]) for r in trace])
                axis.set_ylabel(label)
                axis.grid(alpha=.2)
            axes[-1].set_xlabel('Plant time (s)')
            figure.suptitle(f"{row['variant']} / {row['task']} / {row['machine']}\n"
                f"{row['timing_mode']} / {row['backend']} / {row['source_commit']}\n"
                f"Task={row['task_success']}, physical={row['physical_pass']}, {row['outcome']}", fontsize=9)
            figure.tight_layout()
            figure.savefig(output / f"attempt-{len(rows):03d}.png", dpi=130)
            plt.close(figure)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(path=str(path), error=str(error)))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / 'outcomes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'summary.json', dict(attempts=len(rows), errors=errors,
        raw=str(raw), analysis_sha256=sha(__file__),
        scope='Every discovered control-choice attempt remains separate; no retry pooling, numerical-equivalence gate or extra optimizer calls. Rough success, observation completion and physical compliance are separate.'))
    return bool(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(analyze(args.raw, args.output))
