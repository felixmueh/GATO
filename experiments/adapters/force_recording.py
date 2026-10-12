"""Recording-only projection for future ordinary force tasks.

The E03 native bank remains full I/O: its paired ticks also supply E01's
matched-input/hypothesis correspondence. Executed plant steps are never thinned.
"""
from recording import RawRecorder, write_json
from adapters.task_recording import project_task_record, NATIVE_DIAGNOSTIC_REPRESENTATION


def project_force_record(kind, record):
    """Return a smaller payload without changing worker-owned values.

    Hypothesis predictions, errors, selections and native work/quality diagnostics
    remain measurements. Predicted TO horizons and duplicate request/seed copies
    are not ordinary task measurements. Failed events use the same projection,
    retaining their errors, diagnostics and all observed execution.
    """
    omitted = {'plans', 'selected_plan', 'warm_seed', 'reference',
               'inputs_submitted', 'prediction_request', 'to_request',
               'rng_state', 'rng_state_before', 'rng_state_after', 'applied_history'}
    if kind != 'force_bootstrap':
        omitted.add('construction')
    payload = {key: value for key, value in record.items() if key not in omitted}
    if isinstance(payload.get('inputs'), dict):
        payload['inputs'] = {key: value for key, value in payload['inputs'].items()
                             if key not in {'candidate_states', 'candidate_references',
                                            'candidate_warm_seeds', 'applied_history'}}
    return project_task_record(kind, payload)


class ForceRecorder(RawRecorder):
    """Use the existing chunk writer after projecting only saved payloads."""
    def __init__(self, output, *args, **kwargs):
        kwargs['references'] = True
        super().__init__(output, *args, **kwargs)
        write_json(self.output / 'recording.json', dict(
            recording_format='chunk-refs-v1', reference_window_events=16,
            measurement_contract='force-task-measurements-v1', full_solver_io=False,
            native_diagnostic_representation=NATIVE_DIAGNOSTIC_REPRESENTATION,
            max_events=self.max_events, max_bytes=self.max_bytes,
            retained='Every plant substep, timings/intervals, native work and quality flags, '
                     'force hypotheses/predictions/errors/selections, resources and failures',
            omitted='Routine predicted TO horizons, warm seeds and duplicate requests/replies; '
                    'configuration and RNG seed remain in protocol/provenance',
            timing_scope='Recording outside solve timers; changed resource pressure can indirectly '
                         'affect measured timings and computation-paid feedback'))

    def __call__(self, kind, record):
        return super().__call__(kind, project_force_record(kind, record))


class FullForceRecorder(RawRecorder):
    """Full I/O exemption for explicitly compared force-bank inputs/outputs."""
    def __init__(self, output, *args, **kwargs):
        kwargs['references'] = True
        super().__init__(output, *args, **kwargs)
        write_json(self.output / 'recording.json', dict(
            recording_format='chunk-refs-v1', reference_window_events=16,
            measurement_contract='force-comparison-full-io-v1', full_solver_io=True,
            reason='E03 native bank also supplies E01 matched-input/hypothesis correspondence, '
                   'or full I/O was explicitly requested for this case'))
