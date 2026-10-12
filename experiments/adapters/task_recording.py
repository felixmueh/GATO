"""Serialization projection for future ordinary tasks; controller data is untouched.

Matched-input collectors continue to use RawRecorder directly. This contract
cannot replay individual solver calls: routine seeds and predicted horizons are
deliberately absent. Recording still consumes resources outside solve timers and
can indirectly change measured wall time and computation-paid feedback.
"""
import numpy as np

from recording import RawRecorder, same_value, sha, write_json


TASK_RECORDING_FORMAT = 'task-observations-v1'
NATIVE_DIAGNOSTIC_REPRESENTATION = 'finite-native-arrays-up-to-32-elements-as-lists-v1'
_HORIZONS = {'warm_start', 'implicit_warm_start', 'reference', 'output',
             'candidates', 'optimized', 'XU', 'XU_B', 'warm_seed', 'plans',
             'selected_plan', 'predicted_horizon', 'torques',
             'native_argument_arrays_at_return', 'native_argument_arrays_at_failure'}
_NATIVE_MATRICES = {'XU', 'c', 'r', 'lambda', 'q'}


def project_task_record(kind, record):
    """Keep observations/diagnostics, omit known solver I/O without mutation.

    Unknown fields are retained. Motion step events are never projected; native
    diagnostic dictionaries retain work, flags, merits and residual summaries.
    """
    if kind.startswith('step_') and kind != 'step_record':
        return record

    solver_event = kind.startswith(('solve', 'plan_', 'owner_','native_')) or kind == 'step_record'
    def project(value, native=False):
        if isinstance(value, dict):
            result = {key: project(item, native or key in
                    ('solver_stats', 'available_stats', 'native_reply', 'stats') or key.startswith(('native__', 'stats__')))
                    for key, item in value.items()
                    if not ((solver_event or native) and key in _HORIZONS)
                    and not (kind == 'planned_published_horizon' and key == 'torques')
                    and not (native and key in _NATIVE_MATRICES)
                    and not (key.startswith('lane_') and key.endswith(('_states', '_step_dt_s')))
                    and not (key.startswith('native__') and key.split('__')[-1] in _NATIVE_MATRICES)}
            quality = {}
            for key, item in value.items():
                if ((solver_event or native) and key in ('output', 'optimized', 'XU', 'XU_B')) or (native and key in _NATIVE_MATRICES) or (key.startswith('native__') and key.split('__')[-1] in _NATIVE_MATRICES):
                    array = np.asarray(item)
                    if array.dtype.kind in 'biufc':
                        finite = np.isfinite(array)
                        quality[key] = dict(shape=list(array.shape), dtype=str(array.dtype),
                            finite=bool(finite.all()), nonfinite_count=int(np.count_nonzero(~finite)),
                            finite_max_abs=float(np.max(np.abs(array[finite]))) if np.any(finite) else None)
            if quality:
                result['omitted_array_quality'] = dict(result.get('omitted_array_quality', {}), **quality)
            return result
        if isinstance(value, list):
            return [project(item, native) for item in value]
        if isinstance(value, tuple):
            return tuple(project(item, native) for item in value)
        if native and isinstance(value, np.ndarray) and 0 < value.size <= 32 and value.dtype.kind in 'biuf' and np.isfinite(value).all():
            # Tiny NPZ entries cost much more than their values. Lists preserve
            # dimensions and every finite numeric value, without per-call dtype.
            return value.tolist()
        return value

    return project(record)


class TaskRecorder(RawRecorder):
    """Reuse bounded chunks/references while trimming only serialized payloads."""
    def __init__(self, output, **kwargs):
        super().__init__(output, references=True, **kwargs)
        self.solve_count = self.step_count = 0
        write_json(self.output / 'recording.json', dict(
            recording_format=TASK_RECORDING_FORMAT, projection_sha256=sha(__file__), reference_window_events=16,
            native_diagnostic_representation=NATIVE_DIAGNOSTIC_REPRESENTATION,
            max_events=self.max_events, max_bytes=self.max_bytes,
            retained='Complete executed plant substeps/live observations; timing, native work/quality, warmups, failures and experiment observations',
            omitted='Routine warm starts, predicted horizons and native solver matrices; individual solver-call replay unavailable',
            durability='Completed bounded chunks retained; handled failures flush. Hard kill may lose current chunk (at most 64 events or configured byte bound plus one event).',
            timing_scope='Serialization outside solve timer; resource pressure can indirectly affect wall/native timing and mode B feedback.'))

    def __call__(self, kind, record):
        if kind == 'controller_stats':
            return self.write_controller_stats(record)
        event = super().__call__(kind, project_task_record(kind, record))
        if kind in ('solve_started', 'solve'):
            self.solve_count += 1
        elif kind == 'step_started':
            self.step_count += 1
        return event

    def controller_summary(self, stats):
        summary = dict(stats)
        references = dict(summary.get('history_references', {}))
        for key, count, events in (('raw_solves', self.solve_count, 'solve_* and plan_*'),
                                   ('raw_execution', self.step_count, 'step_*')):
            if key in summary:
                if len(summary[key]) == count:
                    summary.pop(key)
                    references[key] = dict(events=events, count=count)
                else:
                    # Partial/changed callbacks: preserve compact fallback history.
                    summary[key] = [project_task_record('solve' if key == 'raw_solves' else 'step_completed', r)
                                    for r in summary[key]]
        solves = [s for s in stats.get('raw_solves', [])
                  if s.get('category') != 'startup/cold' and s.get('status') == 'completed']
        for key in ('pcg_iters', 'pcg_times_us', 'kkt_converged', 'sqp_iters'):
            values = stats.get(key)
            if values is None or len(values) != len(solves) or len(stats.get('raw_solves', [])) != self.solve_count:
                continue
            expected = []
            for solve in solves:
                native = solve.get('solver_stats', {})
                if key == 'sqp_iters':
                    lanes = np.asarray(native.get(key, [])).reshape(-1)
                    lane = solve.get('best_trajectory_id', 0)
                    expected.append(int(lanes[lane]) if len(lanes) else 0)
                else:
                    expected.append(np.asarray(native.get(key, [])))
            if all(same_value(np.asarray(a), np.asarray(b)) for a, b in zip(values, expected)):
                summary.pop(key)
                references[key] = dict(events='solve_selected', exclude_category='startup/cold',
                    field='solver_stats.' + key, count=len(solves), selected_lane=key == 'sqp_iters')
        summary['history_references'] = references
        # Production finish_arrays already does this. Historical JSON inputs
        # carry the same feedback histories as lists; keep them in NPZ once,
        # instead of repeating numeric JSON in the event and summary envelope.
        for key, value in summary.items():
            if not isinstance(value, list):
                continue
            try:
                array = np.asarray(value)
            except (TypeError, ValueError):
                continue
            if array.dtype.kind in 'biufc':
                summary[key] = array
        return summary

    def write_controller_stats(self, stats):
        event = super().__call__('controller_stats', self.controller_summary(stats))
        self.flush()
        write_json(self.output / 'controller_stats.json', dict(event, recording_format=TASK_RECORDING_FORMAT))
        return event
