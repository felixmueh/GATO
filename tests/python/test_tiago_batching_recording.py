"""Compact recording exercises the existing BB loop with its CPU solver fixture."""
from pathlib import Path

import numpy as np
import pytest

from test_tiago_batching_mode_b import harness, P, folder

def test_compact_raw_keeps_clocks_arrays_and_one_final_history(harness, monkeypatch):
    args, settings, owners, clock = harness
    events, snapshots = [], []
    def record(kind, payload):
        # Match recorder snapshot semantics without testing the recorder itself.
        import copy
        events.append((kind, copy.deepcopy(payload)))
    original = P.np.savez_compressed
    def count(path, *values, **payload):
        if Path(path).name == 'executed.npz': snapshots.append(len(payload['states']))
        return original(path, *values, **payload)
    monkeypatch.setattr(P.np, 'savez_compressed', count)
    result = P.run_episode(args, -1, 1, .03, recorder=record, final_checkpoint_only=True)
    assert result['complete'] and result['simulated_seconds'] == pytest.approx(.1)
    assert owners[0].solver.dual == 1 and owners[0].solver.rho == 4
    assert snapshots == [len(np.load(folder(args) / 'executed.npz')['states'])] * 2
    assert not list(folder(args).glob('step_*'))
    returned = [payload for kind, payload in events if kind == 'solve_returned']
    assert len(returned) == 4
    assert np.isnan(returned[-1]['native__unfiltered'][0]) and np.isinf(returned[-1]['native__unfiltered'][1])
    assert returned[-1]['wall_seconds'] == pytest.approx(.04)
    feedback = [payload for kind, payload in events if kind == 'feedback_completed']
    assert feedback[1]['activation_time_s'] == pytest.approx(.0475)
    assert feedback[2]['warm_start_shift_s'] == pytest.approx(.0175)
    assert feedback[-1]['activation_status'] == 'discarded_task_terminal'
    assert sum(kind == 'step_started' for kind, _ in events) == sum(kind == 'step_completed' for kind, _ in events)


def test_compact_failure_retains_input_reply_and_final_executed_state(harness):
    args, settings, owners, clock = harness
    settings['fail'] = ('stats', 1)
    events = []
    with pytest.raises(RuntimeError, match='stats failure'):
        P.run_episode(args, -1, 1, .03, recorder=lambda kind, row: events.append((kind, row)),
                      final_checkpoint_only=True)
    assert sum(kind == 'solve_input' for kind, _ in events) == 2
    assert sum(kind == 'solve_returned' for kind, _ in events) == 2
    with np.load(folder(args) / 'executed.npz') as trace:
        assert trace['times'][-1] == pytest.approx(.03)
    assert (folder(args) / 'technical_failure.json').exists()
