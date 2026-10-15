"""CPU-only export of the original collector's observed_task_trace event."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recording import iter_events


def export(raw, output):
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted(raw.glob('rtx5060/Gazebo24-N32-grid/*/attempt-*/metrics.json')):
        attempt = path.parent
        protocol = json.loads((attempt / 'protocol.json').read_text())
        metrics = json.loads(path.read_text())
        destination = output / attempt.name
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('protocol.json', 'metrics.json', 'provenance.json', 'receipt.json'):
            if (attempt / name).exists():
                shutil.copy2(attempt / name, destination / name)
        trace = next((e['record'] for e in iter_events(attempt)
                      if e['event'] == 'observed_task_trace'), None)
        if trace is not None:
            ref = np.asarray(protocol['reference_packed']).reshape(-1, 6)
            times = np.asarray(trace['time_s'])
            targets = np.column_stack([np.interp(times, np.arange(len(ref)) * protocol['reference_dt_s'], ref[:, j]) for j in range(3)])
            xyz = np.asarray(trace['tool_position'])
            rows = []
            for i, t in enumerate(times):
                row = dict(t=float(t), time_s=float(t), ref_x=targets[i, 0], ref_y=targets[i, 1], ref_z=targets[i, 2],
                           x=xyz[i, 0], y=xyz[i, 1], z=xyz[i, 2], error=float(np.linalg.norm(xyz[i] - targets[i])),
                           fk_x_m=xyz[i, 0], fk_y_m=xyz[i, 1], fk_z_m=xyz[i, 2])
                row.update({f'q{j}_rad': trace['q'][i, j] for j in range(7)})
                row.update({f'v{j}_rad_s': trace['qd'][i, j] for j in range(7)})
                rows.append(row)
            if rows:
                with (destination / 'observed_task_trace.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            np.save(destination / 'reference.npy', ref)
        summary.append(dict(experiment='Gazebo24-N32-grid', case='figure8-24s', attempt=attempt.name,
            status='execution_failed' if metrics.get('execution_error') else 'completed',
            N=32, dt_s=.03, SQP=5, PCG=120, period_s=24.,
            task_success=metrics.get('task_success'), physical_pass=metrics.get('physical_pass'),
            handback_confirmed=metrics.get('handback_confirmed'), execution_error=metrics.get('execution_error'),
            duration_s=metrics.get('executed_duration_s'), tracking_RMS_m=metrics.get('tracking_RMS_m'),
            tracking_maximum_m=metrics.get('tracking_maximum_m'), trace_file=f'{attempt.name}/observed_task_trace.csv' if trace is not None else None))
    with (output / 'case-summary.csv').open('w', newline='') as f:
        fields = list(summary[0]) if summary else ['experiment', 'case', 'status']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    report = ['# Gazebo 24 s / N32 grid counterpart', '',
        'Prepared period-only extension of collector90bdb825: N32,30ms,SQP5,PCG120 and the unchanged grid tracking preset/start state. Existing50Hz ROS joint-state configuration is retained. One planned attempt; no retries or tuning.', '',
        'Physical/torque violations remain reported with the original explicit Gazebo report-only policy; finite/stale/execution stops and controller handback remain required. The original passed grid canonical is reused and retained. No physical-robot run is authorized.', '',
        'Read attempt-001/observed_task_trace.csv for the observed Cartesian motion. Columns t,x,y,z match the saved September path convention; ref_x/ref_y/ref_z and error use reference interpolation at observed time. Boundary samples plus independently observed interior states are exactly the collector observed_task_trace event. Missing traces on failures remain missing; no synthetic motion is generated.', '',
        'The thesis-side figure_generators/evaluation/extract_figure_eight_paths.py is absent from this checkout, so its parser has not been executed here. Point it to the observed_task_trace.csv file above; the CSV includes both t/x/y/z and time_s/fk_* aliases.', '',
        '## Recorded outcomes', '', json.dumps(summary, indent=2), '',
        'Interpretation: this period-only pair can describe error changes with a slower reference. It does not isolate a causal plant or state-rate effect from all other ROS/model differences.']
    (output / 'REPORT.md').write_text('\n'.join(report) + '\n')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import textwrap
    lines = [part for line in report for part in (textwrap.wrap(line, width=95) or [''])]
    with PdfPages(output / 'REPORT.pdf') as pdf:
        for start in range(0, len(lines), 58):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(.07, .95, '\n'.join(lines[start:start + 58]), va='top', fontsize=8, family='monospace')
            pdf.savefig(fig); plt.close(fig)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(len(export(args.raw, args.output)), 'retained attempts')
