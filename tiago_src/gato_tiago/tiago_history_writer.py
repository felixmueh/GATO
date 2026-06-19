"""History buffering and file output for TIAGo controller runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_STATE_HISTORY_MODE_CODES = {
    "READY": 1.0,
    "RUNNING": 2.0,
    "RESTORED": 3.0,
    "ERROR_RESTORE": 4.0,
}


@dataclass(frozen=True)
class HistoryRecord:
    source_seq: int
    stamp_sec: float
    received_monotonic_sec: float
    controller_mode: str
    q: np.ndarray
    qd: np.ndarray
    positions_by_name: dict[str, float | None]
    velocities_by_name: dict[str, float | None]
    applied_tau: np.ndarray | None = None
    trajectory_id: int | None = None
    safety_status: str = "unchecked"
    safety_fault: str = ""
    safety_message: str = ""
    last_safety_checked_seq: int | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "source_seq": int(self.source_seq),
            "stamp_sec": float(self.stamp_sec),
            "received_monotonic_sec": float(self.received_monotonic_sec),
            "controller_mode": self.controller_mode,
            "q": [float(value) for value in np.asarray(self.q, dtype=np.float64)],
            "qd": [float(value) for value in np.asarray(self.qd, dtype=np.float64)],
            "positions_by_name": dict(self.positions_by_name),
            "velocities_by_name": dict(self.velocities_by_name),
            "applied_tau": (
                None
                if self.applied_tau is None
                else [
                    float(value)
                    for value in np.asarray(self.applied_tau, dtype=np.float64)
                ]
            ),
            "trajectory_id": (
                None if self.trajectory_id is None else int(self.trajectory_id)
            ),
            "safety_status": self.safety_status,
            "safety_fault": self.safety_fault,
            "safety_message": self.safety_message,
            "last_safety_checked_seq": (
                None
                if self.last_safety_checked_seq is None
                else int(self.last_safety_checked_seq)
            ),
        }


class TiagoHistoryBuffer:
    """Fixed-size controller-owned history buffer.

    Overflow drops the oldest records because history is diagnostic and must not
    slow or fault the live controller.
    """

    def __init__(self, max_records: int) -> None:
        if max_records <= 0:
            raise ValueError("history_max_records must be positive")
        self._records: deque[HistoryRecord] = deque(maxlen=int(max_records))
        self.dropped_history_records = 0

    def append(self, record: HistoryRecord) -> None:
        if len(self._records) == self._records.maxlen:
            self.dropped_history_records += 1
        self._records.append(record)

    def records(self) -> list[HistoryRecord]:
        return list(self._records)

    def metadata(self) -> dict[str, int]:
        return {
            "schema_version": 1,
            "record_count": len(self._records),
            "dropped_history_records": int(self.dropped_history_records),
        }


def make_history_record(
    state: Any,
    controller_mode: str,
    *,
    applied_tau: Sequence[float] | None = None,
    trajectory_id: int | None = None,
    safety_status: str = "unchecked",
    safety_fault: str = "",
    safety_message: str = "",
    last_safety_checked_seq: int | None = None,
) -> HistoryRecord:
    return HistoryRecord(
        source_seq=int(state.seq),
        stamp_sec=float(state.stamp_sec),
        received_monotonic_sec=float(state.received_monotonic_sec),
        controller_mode=str(controller_mode),
        q=np.asarray(state.q, dtype=np.float64).copy(),
        qd=np.asarray(state.qd, dtype=np.float64).copy(),
        positions_by_name=_json_float_mapping(_full_joint_positions(state)),
        velocities_by_name=_json_float_mapping(_full_joint_velocities(state)),
        applied_tau=(
            None
            if applied_tau is None
            else np.asarray(applied_tau, dtype=np.float64).copy()
        ),
        trajectory_id=None if trajectory_id is None else int(trajectory_id),
        safety_status=str(safety_status),
        safety_fault=str(safety_fault),
        safety_message=str(safety_message),
        last_safety_checked_seq=last_safety_checked_seq,
    )


def write_history_outputs(
    *,
    records: Sequence[HistoryRecord],
    dropped_history_records: int,
    csv_path: str | Path,
    jsonl_path: str | Path,
    metadata_path: str | Path | None = None,
) -> dict[str, int]:
    csv_path = Path(csv_path)
    jsonl_path = Path(jsonl_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    write_state_history_csv(records, csv_path)
    write_full_state_history_jsonl(records, jsonl_path)
    metadata = {
        "schema_version": 1,
        "record_count": len(records),
        "dropped_history_records": int(dropped_history_records),
    }
    if metadata_path is not None:
        path = Path(metadata_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def write_full_state_history_jsonl(
    records: Sequence[HistoryRecord],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")


def write_state_history_csv(
    records: Sequence[HistoryRecord],
    path: str | Path,
) -> None:
    rows = state_history_rows(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "source_seq,stamp_sec,received_monotonic_sec,controller_mode_code,"
        "q0,q1,q2,q3,q4,q5,q6,qd0,qd1,qd2,qd3,qd4,qd5,qd6"
    )
    np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.10g")


def state_history_rows(records: Sequence[HistoryRecord]) -> np.ndarray:
    rows = []
    for record in records:
        rows.append(
            [
                float(record.source_seq),
                float(record.stamp_sec),
                float(record.received_monotonic_sec),
                _STATE_HISTORY_MODE_CODES.get(record.controller_mode, 0.0),
                *[float(value) for value in np.asarray(record.q, dtype=np.float64)],
                *[float(value) for value in np.asarray(record.qd, dtype=np.float64)],
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def summarize_state_history_frequency(
    records: Sequence[HistoryRecord],
    *,
    change_atol: float = 0.0,
    running_only: bool = True,
) -> dict[str, float | int | bool | None]:
    samples = list(records)
    if running_only:
        samples = [sample for sample in samples if sample.controller_mode == "RUNNING"]
    if len(samples) < 2:
        return {
            "running_only": bool(running_only),
            "samples": len(samples),
            "changed_samples": 0,
            "changed_transitions": 0,
            "lower_bound_hz": None,
            "elapsed_stamp_sec": None,
            "elapsed_wall_receive_sec": None,
        }

    values = np.asarray(
        [
            np.concatenate([sample.q, sample.qd]).astype(np.float64)
            for sample in samples
        ],
        dtype=np.float64,
    )
    deltas = np.abs(np.diff(values, axis=0))
    changed = np.any(deltas > change_atol, axis=1)
    changed_transitions = np.flatnonzero(changed)
    if changed_transitions.size == 0:
        return {
            "running_only": bool(running_only),
            "samples": len(samples),
            "changed_samples": 0,
            "changed_transitions": 0,
            "lower_bound_hz": 0.0,
            "elapsed_stamp_sec": 0.0,
            "elapsed_wall_receive_sec": 0.0,
        }

    first = samples[int(changed_transitions[0])]
    last = samples[int(changed_transitions[-1] + 1)]
    changed_transition_count = int(changed_transitions.size)
    elapsed_stamp = float(last.stamp_sec - first.stamp_sec)
    elapsed_wall = float(last.received_monotonic_sec - first.received_monotonic_sec)
    if elapsed_stamp > 0.0:
        lower_bound_hz = float(changed_transition_count / elapsed_stamp)
    elif elapsed_wall > 0.0:
        lower_bound_hz = float(changed_transition_count / elapsed_wall)
    else:
        lower_bound_hz = None
    return {
        "running_only": bool(running_only),
        "samples": len(samples),
        "changed_samples": changed_transition_count + 1,
        "changed_transitions": changed_transition_count,
        "lower_bound_hz": lower_bound_hz,
        "elapsed_stamp_sec": elapsed_stamp,
        "elapsed_wall_receive_sec": elapsed_wall,
        "change_atol": float(change_atol),
    }


def _full_joint_positions(state: Any) -> dict[str, float]:
    positions = getattr(state, "joint_positions", None)
    if positions is None:
        raise RuntimeError("history requires full named joint positions")
    return {str(name): float(value) for name, value in positions.items()}


def _full_joint_velocities(state: Any) -> dict[str, float]:
    velocities = getattr(state, "joint_velocities", None)
    if velocities is None:
        raise RuntimeError("history requires full named joint velocities")
    return {str(name): float(value) for name, value in velocities.items()}


def _json_float_mapping(values: Mapping[str, float]) -> dict[str, float | None]:
    out = {}
    for name, value in values.items():
        numeric = float(value)
        out[str(name)] = numeric if np.isfinite(numeric) else None
    return out
