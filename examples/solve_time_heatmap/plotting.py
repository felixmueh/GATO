"""Render timing and outcome summaries without modifying raw experiment data."""
from collections import Counter
import csv
import json
import math
from pathlib import Path

from settings import CONTROL_DT, prediction_step


def normalize(manifest):
    """Accept the current schema and the two earlier experiment manifests."""
    config = dict(manifest['config'])
    if not isinstance(config.get('gpu_name'), str) or not config['gpu_name'].strip():
        raise ValueError('config.gpu_name is required')
    config.setdefault('protocol', 'wall-time' if 'wall_time' in config else 'reference')
    config.setdefault('max_sqp_iters', 1000 if config['protocol'] == 'wall-time' else 1)
    config.setdefault('repeats', 1)
    config.setdefault('sim_step', CONTROL_DT)
    config.setdefault('max_solve_ms', 1000 if config['protocol'] == 'wall-time' else 0)
    config.setdefault('horizon_time', None)
    config.setdefault('initialization', 'single')
    for axis in ('knots', 'batches'):
        if not config[axis] or len(set(config[axis])) != len(config[axis]):
            raise ValueError(f'{axis} must be nonempty and unique')
    if config['repeats'] < 1:
        raise ValueError('repeats must be positive')
    results, seen = [], set()
    for original in manifest['results']:
        row = dict(original)
        row.setdefault('repeat', 0)
        key = row['N'], row['batch_size'], row['repeat']
        if key in seen:
            raise ValueError(f'Duplicate repeat {key}')
        if (key[0] not in config['knots'] or key[1] not in config['batches']
                or not 0 <= key[2] < config['repeats']):
            raise ValueError(f'Repeat {key} lies outside the configured matrix')
        seen.add(key)
        if row['status'] == 'ok':
            row['status'] = 'completed'
        row.setdefault('avg_native_ms', row.get('avg_gpu_time_ms'))
        row.setdefault('tracking_mean_m', row.get('avg_goal_distance'))
        row.setdefault('samples', 0)
        row['prediction_dt_s'] = prediction_dt(config, row['N'])
        row['horizon_time_s'] = row['prediction_dt_s'] * (row['N'] - 1)
        initialization = row.get('initialization') or {}
        for field in ('status', 'call_count', 'total_native_ms', 'wall_time_s'):
            row[f'initialization_{field}'] = initialization.get(field)
        ms = row['avg_native_ms']
        row['reported_native_ms'] = ms
        row['native_time_valid'] = isinstance(ms, (int, float)) and math.isfinite(ms) and ms > 0
        if row['status'] == 'completed' and ms is not None and not row['native_time_valid']:
            raise ValueError(f'Invalid native mean in {key}: {ms}')
        if not row['native_time_valid']:
            # Failed updates can contain invalid timing, including JSON-cleaned
            # nulls. Preserve their reported value for inspection, not averaging.
            ms = row['avg_native_ms'] = None
        hz = 1000 / ms if ms is not None else None
        supplied = row.get('solver_frequency_hz')
        if ms is not None and supplied is not None and not math.isclose(supplied, hz, rel_tol=1e-9):
            raise ValueError(f'Frequency is not reciprocal mean solve time in {key}')
        row['solver_frequency_hz'] = hz
        if row['status'] == 'completed' and (ms is None or row['samples'] < 1):
            raise ValueError(f'Completed repeat {key} lacks measured timing')
        results.append(row)
    return config, results


def prediction_dt(config, n):
    return prediction_step(n, config.get('horizon_time'))


def horizon_label(config):
    duration = config.get('horizon_time')
    if duration is not None:
        return f'Fixed prediction duration: {duration:g} s; knot spacing varies with N'
    return 'Fixed prediction spacing: 10 ms; duration = (N − 1) × 10 ms'


def initialization_label(config):
    return ('Native-stop initialization excluded' if config.get('initialization') == 'native-stop'
            else 'One initial warm-up solve excluded')


def outcome_counts(rows):
    return ', '.join(f'{count} {status}' for status, count in sorted(Counter(r['status'] for r in rows).items()))


def aggregate(config, results):
    """Average completed repeat means equally; retain other outcomes explicitly."""
    rows = []
    for n in config['knots']:
        for b in config['batches']:
            group = [r for r in results if r['N'] == n and r['batch_size'] == b]
            good = [r for r in group if r['status'] == 'completed']
            statuses = {r['status'] for r in group}
            missing = config['repeats'] - len(group)
            if missing:
                statuses.add('missing')
            status = next(iter(statuses)) if len(statuses) == 1 else 'mixed'
            ms = sum(r['avg_native_ms'] for r in good) / len(good) if good else None
            tracking = [r['tracking_mean_m'] for r in good if r.get('tracking_mean_m') is not None]
            initialized = [r for r in group if r.get('initialization_status') is not None]
            rows.append(dict(
                N=n, batch_size=b, status=status,
                horizon_time_s=prediction_dt(config, n) * (n - 1),
                prediction_dt_s=prediction_dt(config, n),
                outcomes=outcome_counts(group) + (f', {missing} missing' if missing and group else f'{missing} missing' if missing else ''),
                completed_repeats=len(good), recorded_repeats=len(group), planned_repeats=config['repeats'],
                included_in_heatmap=bool(good), partial_repeats=bool(good) and len(good) < config['repeats'],
                mean_native_ms=ms, solver_frequency_hz=1000 / ms if ms is not None else None,
                samples=sum(r['samples'] for r in good), all_retained_samples=sum(r['samples'] for r in group),
                any_cap_count=sum(r.get('any_cap_count') or 0 for r in good),
                tracking_mean_m=sum(tracking) / len(tracking) if tracking else None,
                initialization_outcomes=outcome_counts([
                    {'status': r['initialization_status']} for r in initialized]),
                initialization_call_count=sum(r.get('initialization_call_count') or 0 for r in initialized) if initialized else None,
                initialization_total_native_ms=sum(r.get('initialization_total_native_ms') or 0 for r in initialized) if initialized else None,
                initialization_wall_time_s=sum(r.get('initialization_wall_time_s') or 0 for r in initialized) if initialized else None,
            ))
    return rows


def short_status(status):
    return {'unsupported_shared_memory': 'SMEM', 'slow_solve': 'SLOW',
            'initialization_incomplete': 'INIT CAP',
            'unstable_nonfinite': 'NONFINITE', 'nonfinite': 'NONFINITE',
            'running': 'RUNNING', 'missing': '—', 'pending': '—',
            'interrupted': 'PARTIAL', 'timeout': 'TIMEOUT',
            'reference_exhausted': 'REF END', 'mixed': 'MIXED'}.get(status, 'ERROR')


def numeric_label(value):
    return f'{value:.2f}' if value < 10 else f'{value:.1f}' if value < 100 else f'{value:,.0f}'


def draw(config, rows, output, *, frequency=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    import numpy as np

    ns, batches = config['knots'], config['batches']
    key = 'solver_frequency_hz' if frequency else 'mean_native_ms'
    z = np.array([r[key] if r['included_in_heatmap'] else np.nan for r in rows]).reshape(len(ns), len(batches))
    finite = z[np.isfinite(z)]
    lower = max(float(finite.min()) * .95, 1e-9) if finite.size else .1
    upper = max(float(finite.max()) * 1.05, lower * 1.1) if finite.size else 100
    cmap = plt.get_cmap('RdYlGn' if frequency else 'RdYlGn_r').copy()
    cmap.set_bad('#e4e4e4')
    with plt.rc_context({'font.family': 'serif', 'font.size': 12}):
        fig, ax = plt.subplots(figsize=(14, 8.5))
        im = ax.imshow(np.ma.masked_invalid(z), origin='lower', aspect='auto', cmap=cmap,
                       norm=LogNorm(lower, upper), interpolation='nearest')
        for index, row in enumerate(rows):
            i, j = divmod(index, len(batches))
            if np.isfinite(z[i, j]):
                rgba = cmap(im.norm(z[i, j]))
                color = 'black' if .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2] > .55 else 'white'
                marker = '*' if row['partial_repeats'] else ''
                if row['any_cap_count'] and config['max_sqp_iters'] != 1:
                    marker += '†'
                ax.text(j, i+.10, numeric_label(z[i, j]) + marker, ha='center', va='center',
                        fontsize=12, weight='bold', color=color)
                ax.text(j, i-.18, f'n={row["samples"]:,}', ha='center', va='center', fontsize=8, color=color)
            else:
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, hatch='///', edgecolor='#aaa', linewidth=0))
                label = short_status(row['status'])
                if row['all_retained_samples']:
                    label += f'\nn={row["all_retained_samples"]:,}'
                ax.text(j, i, label, ha='center', va='center', fontsize=9,
                        bbox={'facecolor': '#e4e4e4', 'edgecolor': 'none', 'pad': 1})
        ax.set_xticks(range(len(batches)), labels=batches)
        ax.set_yticks(range(len(ns)), labels=ns)
        ax.set_xlabel('Batch size', fontsize=18)
        ax.set_ylabel('Prediction knots (N)', fontsize=18)
        mode = '1 SQP iteration per update' if config['max_sqp_iters'] == 1 else f'Native stopping; up to {config["max_sqp_iters"]:,} SQP iterations'
        recorded = sum(row['recorded_repeats'] for row in rows)
        expected = len(rows) * config['repeats']
        if recorded < expected:
            mode += f' · Partial: {recorded}/{expected} repeat outcomes'
        ax.set_title(f'GATO / {config.get("plant_label", "Indy7")} — {config["gpu_name"]}\n{mode}\n{horizon_label(config)}', fontsize=16, pad=16)
        units = 'Solver frequency (Hz): 1000 / mean solve time (ms)' if frequency else 'Mean synchronized native solve time (ms)'
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03).set_label(units, fontsize=13)
        if config['protocol'] == 'wall-time':
            clock = f'{config["wall_time"]:g} s measured wall budget / repeat · {1000*config["sim_step"]:g} ms fixed simulation step'
        else:
            clock = f'{config["sim_time"]:g} s capped offline simulation / repeat'
        cutoff = config['max_solve_ms']
        notes = ('SLOW: a measured update exceeded ' + f'{cutoff:g} ms' if cutoff else 'Slow-solve cutoff disabled')
        cap = ' · †: an update reached the SQP cap' if config['max_sqp_iters'] != 1 else ' · One-iteration cap is intentional'
        init_note = (' · INIT CAP: initialization did not reach native stopping'
                     if config.get('initialization') == 'native-stop' else '')
        fig.text(.5, .050, clock + ' · ' + initialization_label(config) + '\n'
                 'n: measured solves (coverage varies) · SMEM: shared-memory limit · *: incomplete repeats' + cap + '\n'
                 + notes + init_note + '; partial means in repeat_summary.csv\n'
                 'Frequency is reciprocal full-batch solve time, not real-time control frequency. Native stopping is not a KKT certificate.',
                 ha='center', fontsize=9, va='center')
        fig.tight_layout(rect=(0, .115, 1, 1))
        stem = 'gato_solver_frequency_heatmap' if frequency else 'gato_solve_time_heatmap'
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(output / f'{stem}.{extension}', dpi=180)
        plt.close(fig)


def extent(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return '—'
    return f'{min(values):.5g}–{max(values):.5g}'


def report(config, results, rows):
    completed = [r for r in rows if r['included_in_heatmap']]
    total = len(rows) * config['repeats']
    lines = [f'# GATO / {config.get("plant_label", "Indy7")} — {config["gpu_name"]}', '',
             f'{len(results)}/{total} repeat outcomes recorded: {outcome_counts(results) or "none"}.', '',
             '“Completed” means the requested experiment budget was reached; it does not certify nonlinear convergence.', '',
             f'Completed-repeat cell means span **{extent([r["mean_native_ms"] for r in completed])} ms** and '
             f'**{extent([r["solver_frequency_hz"] for r in completed])} Hz**.', '',
             '[Solve-time grid](gato_solve_time_heatmap.png) · [Frequency grid](gato_solver_frequency_heatmap.png) · '
             '[Cell summary](summary.csv) · [Per-repeat measurements](repeat_summary.csv) · [Raw results](results.json)', '',
             '## Timing and protocol', '',
             'Native synchronized host timing measures one full-batch solve. Initialization, Python simulation, '
             'input preparation, and recording are excluded from solve-time samples. Frequency is '
             '`1000 / mean_native_ms`; it is neither batch-member throughput nor end-to-end control frequency. '
             'Each grid cell equally weights its completed repeat means, then takes the reciprocal for frequency. '
             'An asterisk marks cells with missing or unsuccessful repeats. Partial repeat means remain in the CSV.', '']
    if config['protocol'] == 'wall-time':
        lines += [f'Each repeat has a **{config["wall_time"]:g} s measured wall-time budget**, starting after initialization. '
                  f'Simulation advances **{1000*config["sim_step"]:g} ms per update**. Simulation and recording count toward '
                  'the wall budget; setup, initialization, and final output do not. Faster cells cover more simulated time. '
                  'The budget is checked between completed updates; an in-flight solve is retained in full.', '']
    else:
        lines += [f'Each repeat runs **{config["sim_time"]:g} s of simulated time**. Simulation advances by the previous '
                  'native solve time capped at the reference interval. This reproduces the capped offline reference protocol.', '']
    lines += [horizon_label(config) + '.', '',
              '| N | Prediction duration (s) | Knot spacing (ms) |',
              '|---:|---:|---:|']
    for n in config['knots']:
        dt = prediction_dt(config, n)
        lines.append(f'| {n} | {dt*(n-1):.6g} | {1000*dt:.6g} |')
    if config.get('horizon_time') is not None:
        lines += ['', config.get('cost_rule', 'Historical Indy7 running weights scaled relative to 10 ms, anchored at N32.')]
    lines += ['', '## Initialization', '']
    if config.get('initialization') == 'native-stop':
        limit = config.get('max_init_solves')
        bounded = f', with at most {limit} solver calls' if limit is not None else ''
        lines += ['The initial state and reference remain fixed while repeated solves warm-start the initial plan '
                  'until every batch member reports native stopping' + bounded + '. '
                  'Initialization cost is recorded separately and excluded from measured operation. '
                  '`INIT CAP` means initialization exhausted its call budget without all native stopping flags; '
                  'that repeat contributes no numeric heatmap value. Native stopping is not a nonlinear KKT certificate.', '']
    else:
        lines += ['Exactly one initial warm-up solve is excluded from measured operation. '
                  'This mode does not require initialization to reach native stopping.', '']
    initializations = [r for r in results if r.get('initialization_status') is not None]
    if initializations:
        lines += ['Initialization totals below cover all recorded repeats, including incomplete outcomes; '
                  'they are not solve-time samples. Per-cell totals are in `summary.csv` and per-repeat values '
                  'are in `repeat_summary.csv`.', '',
                  '| N | Batch | Repeat | Initialization outcome | Calls | Native total (ms) | Wall time (s) |',
                  '|---:|---:|---:|---|---:|---:|---:|']
        for row in initializations:
            costs = [str(row.get(f'initialization_{field}')) if row.get(f'initialization_{field}') is not None else '—'
                     for field in ('call_count', 'total_native_ms', 'wall_time_s')]
            lines.append(f'| {row["N"]} | {row["batch_size"]} | {row["repeat"]} | '
                         f'{row["initialization_status"]} | ' + ' | '.join(costs) + ' |')
        lines.append('')
    cutoff = config['max_solve_ms']
    tracking_reference = ('the current-time reference' if config.get('horizon_time') is not None or config.get('plant') == 'tiago_right'
                          else 'the next reference knot')
    lines += [(f'A repeat stops after its first measured solve strictly exceeding **{cutoff:g} ms**. '
               'Initialization never triggers this cutoff. The slow sample is retained; the cutoff is not a hard cancellation deadline. '
               'A slow repeat can have a mean below the cutoff.' if cutoff else 'The slow-solve cutoff is disabled.'), '',
              'Batch 0 controls the simulation; its unshifted plan supplies the next warm start. Timing and tracking '
              'summaries can cover unequal numbers of updates and unequal trajectory durations. Tracking is the '
              f'per-solve {config.get("tracking_frame", "joint-6 origin")} error against {tracking_reference}, with measured startup transients included.', '',
              '## SQP stopping', '']
    if config['max_sqp_iters'] == 1:
        lines += ['The solver performs at most **one SQP iteration per update**. Reaching that cap is intentional, '
                  'not a convergence failure. No convergence requirement is imposed on measured one-iteration updates.', '']
    else:
        lines += [f'The solver uses its unchanged native stopping rule with an SQP safety cap of **{config["max_sqp_iters"]:,}**. '
                  'Native stopping is the zero-inner-PCG-iteration flag; it is not a nonlinear KKT certificate. '
                  'A dagger marks completed grid cells with measured updates reaching the cap without native stopping.', '']
    recorded_flags = [r for r in results if r.get('all_native_stop_count') is not None]
    lines += [(f'Among repeats with recorded stopping flags, all batch members reported native stopping in '
               f'{sum(r["all_native_stop_count"] for r in recorded_flags):,} / '
               f'{sum(r["samples"] for r in recorded_flags):,} measured updates. '
               'Initialization is excluded; retained partial outcomes are included.' if recorded_flags else
               'These saved results do not contain native stopping counts.'), '',
              '## Cell outcomes', '',
              '| N | Batch | Repeat outcomes | Completed repeats | Mean solve time (ms) | Solve frequency (Hz) |',
              '|---:|---:|---|---:|---:|---:|']
    for row in rows:
        ms = f'{row["mean_native_ms"]:.5g}' if row['mean_native_ms'] is not None else '—'
        hz = f'{row["solver_frequency_hz"]:.5g}' if row['solver_frequency_hz'] is not None else '—'
        lines.append(f'| {row["N"]} | {row["batch_size"]} | {row["outcomes"]} | '
                     f'{row["completed_repeats"]}/{row["planned_repeats"]} | {ms} | {hz} |')
    return '\n'.join(lines) + '\n'


def plot(output):
    """Write figures, cell/repeat CSV summaries, and a compact Markdown report."""
    output = Path(output)
    manifest = json.loads((output / 'results.json').read_text())
    config, results = normalize(manifest)
    rows = aggregate(config, results)
    repeat_fields = ['N', 'batch_size', 'repeat', 'status', 'samples', 'avg_native_ms',
                     'reported_native_ms', 'native_time_valid',
                     'solver_frequency_hz', 'p95_native_ms', 'avg_sqp_iters', 'tracking_mean_m',
                     'all_native_stop_count', 'any_cap_count', 'simulation_end_s',
                     'measured_wall_elapsed_s', 'budget_overrun_s', 'slow_solve_count',
                     'horizon_time_s', 'prediction_dt_s', 'initialization_status',
                     'initialization_call_count', 'initialization_total_native_ms', 'initialization_wall_time_s']
    for filename, data, fields in [('summary.csv', rows, list(rows[0])),
                                   ('repeat_summary.csv', results, repeat_fields)]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)
    draw(config, rows, output)
    draw(config, rows, output, frequency=True)
    (output / 'REPORT.md').write_text(report(config, results, rows))
