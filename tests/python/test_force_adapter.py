"""Frozen CPU correspondence checks, independent of CUDA and native bindings."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.force_adapter.estimator import FrozenBatchForceEstimator
from experiments.force_adapter.orchestration import (
    AppliedSegment, TickInputs, WrenchConvention, run_tick,
)


CONVENTION = WrenchConvention("world", "tool origin", "stub world", "tool origin")


def fixture():
    batch = 16
    forces = np.zeros((batch, 6))
    forces[:, 0] = np.arange(batch)
    return forces, TickInputs(0.0, 0.5, np.array([2.0, 0.0]), np.array([5.5, 0.0]),
        (AppliedSegment(0.0, 0.25, np.array([1.0])),
         AppliedSegment(0.25, 0.5, np.array([3.0]))),
        np.column_stack([100 + np.arange(batch), np.zeros(batch)]),
        np.arange(batch * 3).reshape(batch, 3),
        np.arange(batch * 4).reshape(batch, 4))


class Estimator:
    def __init__(self, forces, calls):
        self.forces, self.calls = forces, calls
        self.generated = 0
        self.updates = []

    def generate_batch(self):
        self.generated += 1
        self.calls.append("generate")
        return self.forces

    def update_frozen(self, index, errors, candidates):
        self.calls.append("update")
        self.updates.append((index, errors.copy(), candidates.copy()))
        # Changing next-tick candidates must not alter this tick's TO force.
        self.forces[:] = -900


def execute(mode, *, predictor=None, optimizer=None, inputs=None):
    forces, default_inputs = fixture()
    calls, records, requests = [], [], []
    estimator = Estimator(forces, calls)

    def predict(request):
        calls.append("predict")
        requests.append(request)
        state = np.tile(request.inputs.previous_state, (16, 1))
        for segment in request.inputs.applied_history:
            state[:, 0] += ((segment.control[0] + request.hypothesis_wrenches[:, 0])
                            * (segment.end_s - segment.start_s))
        return state if predictor is None else predictor(request)

    def solve(request):
        calls.append("solve")
        requests.append(request)
        return request.warm_seeds + 10 if optimizer is None else optimizer(request)

    result = run_tick(mode, inputs or default_inputs, estimator=estimator,
                      predictor=predict, optimizer=solve, convention=CONVENTION,
                      transform_wrench=lambda state, force, convention: force,
                      observer=records.append)
    return result, calls, estimator, records, requests


def test_original_adapted_correspondence_and_exact_b1_work():
    original, ocalls, oe, _, _ = execute("original")
    adapted, acalls, ae, _, _ = execute("adapted")
    assert ocalls == ["generate", "solve", "predict", "update"]
    assert acalls == ["generate", "predict", "update", "solve"]
    assert original["plans"].shape == (16, 4)
    assert adapted["plans"].shape == (1, 4)
    assert original["selected_index"] == adapted["selected_index"] == 5
    for field in ("hypotheses", "predictions", "errors", "selected_force", "selected_plan"):
        np.testing.assert_array_equal(original[field], adapted[field])
    request = adapted["to_request"]
    assert request.candidate_indices == (5,)
    np.testing.assert_array_equal(request.states, [[105, 0]])
    np.testing.assert_array_equal(request.references, [[15, 16, 17]])
    np.testing.assert_array_equal(request.warm_seeds, [[20, 21, 22, 23]])
    np.testing.assert_array_equal(request.hypothesis_wrenches, [[5, 0, 0, 0, 0, 0]])
    for estimator, record in ((oe, original), (ae, adapted)):
        assert estimator.generated == len(estimator.updates) == 1
        np.testing.assert_array_equal(estimator.updates[0][2], record["hypotheses"])
    assert adapted["memory_bytes"] is None
    assert set(adapted["timings_s"]) == {
        "candidate_forward", "selection", "estimator_update", "to_backend", "total", "wrapper"}
    assert all(t >= 0 for t in adapted["timings_s"].values())


def test_all_applied_controls_and_earliest_tie():
    result, _, _, _, _ = execute("adapted")
    # Constant first-control replay would give 2.5 instead of 3.0 at candidate 0.
    assert result["predictions"][0, 0] == 3.0
    result, _, estimator, _, _ = execute("adapted", predictor=lambda req: np.zeros((16, 2)))
    assert result["selected_index"] == estimator.updates[0][0] == 0


@pytest.mark.parametrize("mode", ["original", "adapted"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_predictions_retained_without_selection(mode, bad_value):
    forces, inputs = fixture()
    records, calls = [], []
    estimator = Estimator(forces, calls)
    predictions = np.zeros((16, 2))
    predictions[3, 0] = bad_value
    with pytest.raises(ValueError, match="nonfinite"):
        run_tick(mode, inputs, estimator=estimator, predictor=lambda req: predictions,
                 optimizer=lambda req: np.zeros((len(req.candidate_indices), 4)),
                 convention=CONVENTION, transform_wrench=lambda s, f, c: f,
                 observer=records.append)
    record, = records
    assert record["status"] == "failed" and record["error"]["type"] == "ValueError"
    assert not np.isfinite(record["predictions"][3, 0])
    assert not np.isfinite(record["errors"][3])
    assert "selected_index" not in record and not estimator.updates
    assert ("plans" in record) == (mode == "original")


def test_backend_failure_retains_request_and_partial_timing():
    forces, inputs = fixture()
    records = []
    estimator = Estimator(forces, [])
    def fail(req):
        raise RuntimeError("stub backend unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        run_tick("adapted", inputs, estimator=estimator,
                 predictor=lambda req: np.zeros((16, 2)), optimizer=fail,
                 convention=CONVENTION, transform_wrench=lambda s, f, c: f,
                 observer=records.append)
    record, = records
    assert record["to_request"].candidate_indices == (0,)
    assert len(estimator.updates) == 1
    assert "to_backend" in record["timings_s"] and "plans" not in record


def test_nonfinite_plans_kept_in_raw_failure():
    forces, inputs = fixture()
    records = []
    with pytest.raises(ValueError, match="nonfinite optimizer"):
        run_tick("adapted", inputs, estimator=Estimator(forces, []),
                 predictor=lambda req: np.zeros((16, 2)),
                 optimizer=lambda req: np.full((1, 4), np.nan),
                 convention=CONVENTION, transform_wrench=lambda s, f, c: f,
                 observer=records.append)
    assert np.isnan(records[0]["plans"]).all()


def test_history_gap_rejected_and_observer_outside_total_timer(monkeypatch):
    import experiments.force_adapter.orchestration as module
    forces, inputs = fixture()
    clock, records = [0.0], []
    monkeypatch.setattr(module, "perf_counter", lambda: clock[0])
    def observer(record):
        clock[0] += 100
        records.append(record)
    bad = TickInputs(0, .5, inputs.previous_state, inputs.observed_state,
                     inputs.applied_history[:1], inputs.candidate_states,
                     inputs.candidate_references, inputs.candidate_warm_seeds)
    with pytest.raises(ValueError, match="entire"):
        run_tick("adapted", bad, estimator=Estimator(forces, []),
                 predictor=None, optimizer=None, convention=CONVENTION,
                 transform_wrench=None, observer=observer)
    assert records[0]["timings_s"]["total"] == 0 and clock[0] == 100


def test_existing_estimator_updates_frozen_candidates_and_restores_after_error(monkeypatch):
    from examples.force_estimator import ForceEstimator
    estimator = FrozenBatchForceEstimator(16)
    forces, _ = fixture()
    # Deliberately different from the estimator's ordinary generated hypotheses.
    expected = ForceEstimator(16)
    expected.generate_batch = lambda: forces.copy()
    monkeypatch.setattr(ForceEstimator, "_random_rotation_matrix", lambda self: np.eye(3))
    expected.update(5, np.arange(16), alpha=.6, beta=.5)
    estimator.update_frozen(5, np.arange(16), forces)
    for field in ("estimate", "momentum", "smoothed_estimate", "radius", "confidence"):
        np.testing.assert_array_equal(getattr(estimator, field), getattr(expected, field))
    assert estimator._tick_candidates is None
    assert not np.array_equal(estimator.generate_batch(), forces)
    def fail(self):
        raise RuntimeError("rotation failed")
    monkeypatch.setattr(ForceEstimator, "_random_rotation_matrix", fail)
    with pytest.raises(RuntimeError, match="rotation failed"):
        estimator.update_frozen(5, np.arange(16), forces)
    assert estimator._tick_candidates is None


def test_raw_observer_preserves_nonfinite_arrays_and_refuses_overwrite(tmp_path):
    import json
    from experiments.force_adapter.records import write_record
    destination = tmp_path / "tick"
    write_record(destination, {"raw": np.array([np.nan, np.inf, -np.inf]),
                               "error": {"type": "ValueError", "message": "nonfinite"}})
    metadata = json.loads((destination / "record.json").read_text())
    with np.load(destination / "arrays.npz", allow_pickle=False) as arrays:
        values = arrays[metadata["raw"]["npz_key"]]
        assert np.isnan(values[0]) and values[1] == np.inf and values[2] == -np.inf
    with pytest.raises(FileExistsError):
        write_record(destination, {})
