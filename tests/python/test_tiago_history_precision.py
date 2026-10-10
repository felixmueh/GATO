"""Retain subsecond controller history at realistic ROS timestamp magnitudes."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tiago_src"))

from gato_tiago.tiago_controller_process import (
    TiagoControllerOrchestrator,
    _load_state_history,
    summarize_state_history_frequency,
)
from gato_tiago.tiago_history_writer import (
    HistoryRecord,
    state_history_rows,
    write_history_outputs,
)


def _records(period_sec):
    return [
        HistoryRecord(
            source_seq=seq + 1,
            stamp_sec=1_790_000_000.125 + seq * period_sec,
            received_monotonic_sec=87_432.123456789 + seq * period_sec,
            controller_mode="RUNNING",
            q=np.arange(7, dtype=np.float64) * 0.001 + seq * 0.0001,
            qd=np.arange(7, dtype=np.float64) * 0.0001,
            positions_by_name={},
            velocities_by_name={},
        )
        for seq in range(12)
    ]


def _check_times_and_frequency(path, records):
    loaded = _load_state_history(path)
    original = state_history_rows(records)
    reloaded = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    np.testing.assert_array_equal(reloaded, original)
    assert len(loaded) == len(records)
    expected_stamps = original[:, 1]
    expected_receipts = original[:, 2]
    np.testing.assert_array_equal(np.diff(reloaded[:, 1]), np.diff(expected_stamps))
    np.testing.assert_array_equal(np.diff(reloaded[:, 2]), np.diff(expected_receipts))
    assert np.all(np.diff(reloaded[:, 1]) > 0.0)
    assert np.all(np.diff(reloaded[:, 2]) > 0.0)
    summary = summarize_state_history_frequency(loaded)
    assert summary["changed_transitions"] == len(records) - 1
    assert summary["elapsed_stamp_sec"] == expected_stamps[-1] - expected_stamps[0]
    assert summary["lower_bound_hz"] == (len(records) - 1) / (
        expected_stamps[-1] - expected_stamps[0]
    )


@pytest.mark.parametrize("period_sec", [0.001, 0.008, 0.02])
def test_worker_history_csv_preserves_epoch_updates_and_jsonl(tmp_path, period_sec):
    records = _records(period_sec)
    csv_path = tmp_path / "history.csv"
    jsonl_path = tmp_path / "history.jsonl"
    write_history_outputs(
        records=records,
        dropped_history_records=0,
        csv_path=csv_path,
        jsonl_path=jsonl_path,
    )

    _check_times_and_frequency(csv_path, records)
    json_records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert json_records == [record.to_json_dict() for record in records]


@pytest.mark.parametrize("period_sec", [0.001, 0.008, 0.02])
def test_orchestrator_export_preserves_loaded_history_frequency(tmp_path, period_sec):
    records = _records(period_sec)
    retained_path = tmp_path / "retained.csv"
    # A full-precision retained input isolates the separate export route.
    np.savetxt(retained_path, state_history_rows(records), delimiter=",", header="history")
    retained_bytes = retained_path.read_bytes()
    orchestrator = TiagoControllerOrchestrator.__new__(TiagoControllerOrchestrator)
    orchestrator._history_path = retained_path
    orchestrator._history_metadata_path = tmp_path / "metadata.json"
    exported_path = tmp_path / "exported.csv"

    summary = orchestrator.write_state_history_csv(exported_path)

    _check_times_and_frequency(exported_path, records)
    assert summary["lower_bound_hz"] == summarize_state_history_frequency(
        _load_state_history(exported_path)
    )["lower_bound_hz"]
    assert retained_path.read_bytes() == retained_bytes
