import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TIAGO_SRC = REPO_ROOT / "tiago_src"

if str(TIAGO_SRC) not in sys.path:
    sys.path.insert(0, str(TIAGO_SRC))

from gato_tiago.tiago_controller_process import elapsed_sim_time_from_stamp


def test_elapsed_sim_time_uses_initial_stamp_as_zero():
    assert elapsed_sim_time_from_stamp(100.0, 100.0) == pytest.approx(0.0)
    assert elapsed_sim_time_from_stamp(100.02, 100.0, 0.0) == pytest.approx(0.02)


def test_elapsed_sim_time_allows_repeated_robot_state_stamps():
    stamps = [100.0, 100.02, 100.02, 100.04]
    elapsed = []
    previous = None
    for stamp in stamps:
        current = elapsed_sim_time_from_stamp(stamp, stamps[0], previous)
        elapsed.append(current)
        previous = current

    assert elapsed == pytest.approx([0.0, 0.02, 0.02, 0.04])


def test_elapsed_sim_time_rejects_backward_robot_state_stamps():
    with pytest.raises(RuntimeError, match="moved backward"):
        elapsed_sim_time_from_stamp(100.01, 100.0, 0.02)
