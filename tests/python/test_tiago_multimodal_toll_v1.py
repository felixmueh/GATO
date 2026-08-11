from pathlib import Path

import numpy as np
import pytest

from gato_tiago import multimodal_toll as toll
from gato_tiago import multimodal_toll_oracle_schema as oracle
from gato_tiago import multimodal_toll_v0_runner as v0_runner
from gato_tiago import multimodal_toll_v0_worker as v0_worker
from gato_tiago import multimodal_toll_v1 as v1
from gato_tiago import multimodal_toll_v1_runner as v1_runner
from gato_tiago import multimodal_toll_v1_worker as v1_worker


def _inputs():
    return (
        np.asarray([-4.0, -2.5, -2.5, -2.5, -1.5, -2.0, -2.5], dtype=np.float64),
        np.asarray([1.0, 2.5, 2.7, 1.5, 3.0, 3.0, 2.7], dtype=np.float64),
        np.arange(1.0, 8.0, dtype=np.float64) * 10,
        np.asarray([-0.39, -1.73, -0.38, -2.35, 0.0, -1.21, 0.04], dtype=np.float64),
    )


def test_v1_direct_q_draw_has_exact_frozen_stream_and_interior_margins():
    lower, upper, effort, comfortable = _inputs()
    draws = v1.generate_v1_draws(lower, upper, effort, comfortable)
    rng = np.random.default_rng(v1.V1_RANDOM_SEED)
    expected_fk = rng.uniform(lower + 0.08, upper - 0.08, size=(32, 7))
    expected_q = rng.uniform(lower + 0.08, upper - 0.08, size=(32, 7))
    expected_qd = rng.uniform(-0.25, 0.25, size=(32, 7))
    expected_fractions = rng.uniform(-0.20, 0.20, size=(32, 7))
    np.testing.assert_array_equal(draws.fk_q_float64, expected_fk)
    np.testing.assert_array_equal(draws.dynamics_q_float64, expected_q)
    np.testing.assert_array_equal(draws.dynamics_qd_float64, expected_qd)
    np.testing.assert_array_equal(
        draws.dynamics_control_fractions_float64, expected_fractions
    )
    np.testing.assert_array_equal(
        draws.dynamics_u_float64, expected_fractions * effort[None, :]
    )
    assert np.all(draws.dynamics_q_float64 >= lower + 0.08)
    assert np.all(draws.dynamics_q_float64 <= upper - 0.08)
    assert draws.dynamics_x_float32.shape == (32, 14)
    assert draws.dynamics_x_float32.dtype == np.float32
    assert set(draws.array_hashes()) == set(draws.__dict__)
    repeated = v1.generate_v1_draws(lower, upper, effort, comfortable)
    assert repeated.array_hashes() == draws.array_hashes()


def test_only_second_draw_transform_changes_and_later_stream_bytes_are_exact():
    lower, upper, effort, comfortable = _inputs()
    v1_draws = v1.generate_v1_draws(lower, upper, effort, comfortable)
    old_stream = np.random.default_rng(v1.V1_RANDOM_SEED)
    old_fk = old_stream.uniform(lower + 0.08, upper - 0.08, size=(32, 7))
    old_q = comfortable[None, :] + old_stream.uniform(-0.05, 0.05, size=(32, 7))
    old_qd = old_stream.uniform(-0.25, 0.25, size=(32, 7))
    old_fractions = old_stream.uniform(-0.20, 0.20, size=(32, 7))
    np.testing.assert_array_equal(v1_draws.fk_q_float64, old_fk)
    assert not np.array_equal(v1_draws.dynamics_q_float64, old_q)
    np.testing.assert_array_equal(v1_draws.dynamics_qd_float64, old_qd)
    np.testing.assert_array_equal(
        v1_draws.dynamics_control_fractions_float64, old_fractions
    )
    np.testing.assert_array_equal(
        v1_draws.dynamics_u_float64, old_fractions * effort[None, :]
    )
    alternate = v1.generate_v1_draws(
        lower, upper, effort, comfortable + np.arange(7) * 0.01
    )
    for name in draws_random_fields():
        np.testing.assert_array_equal(getattr(v1_draws, name), getattr(alternate, name))


def draws_random_fields():
    return (
        "fk_q_float64",
        "dynamics_q_float64",
        "dynamics_qd_float64",
        "dynamics_control_fractions_float64",
        "dynamics_u_float64",
        "fk_q_float32",
        "dynamics_x_float32",
        "dynamics_u_float32",
    )


def test_v1_metadata_records_exact_diff_and_excludes_rejected_v0_artifacts():
    metadata = v1.frozen_v1_metadata()
    assert metadata["protocol_version"] == v1.V1_PROTOCOL_VERSION
    assert metadata["execution_authorized"] is False
    assert metadata["v0_status"] == "rejected_closed_pre_task_margin_gate"
    assert metadata["v0_artifacts_consumed"] is False
    assert metadata["v0_to_v1_exact_diff"]["only_changed_draw"] == "dynamics_q"
    assert metadata["v0_to_v1_exact_diff"]["rng_position"] == 2
    assert metadata["draw_order"] == [
        "fk_q_direct_interior",
        "dynamics_q_direct_interior",
        "dynamics_qd_uniform",
        "dynamics_control_fraction_uniform",
    ]
    assert all(
        len(value) == 64
        for value in metadata["v0_rejected_artifact_hashes_report_only"].values()
    )
    runner_metadata = v1_runner.describe_v1_runner()
    assert runner_metadata["v0_artifact_inputs"] == []
    assert runner_metadata["v0_array_inputs"] == []
    assert runner_metadata["fresh_artifact_namespace"] == v1.V1_OUTPUT_PATH


def test_v0_v1_and_shared_task_capabilities_are_all_blocked():
    assert v0_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v0_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert v1.V1_EXECUTION_AUTHORIZATION is None
    assert v1_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v1_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert oracle.TASK_CONSTRUCTION_AUTHORIZATION is None
    assert toll.OPTIMIZER_EXECUTION_AUTHORIZATION is None
    with pytest.raises(RuntimeError, match="blocked"):
        v1_runner.execute_v1_runner(v1.V1_OUTPUT_PATH)
    with pytest.raises(RuntimeError, match="blocked"):
        v1_worker.execute_worker("request", "output")


@pytest.mark.parametrize("field", ["lower", "upper", "effort", "comfortable"])
def test_v1_draw_inputs_are_exact_finite_float64_vectors(field):
    values = list(_inputs())
    index = {"lower": 0, "upper": 1, "effort": 2, "comfortable": 3}[field]
    values[index] = values[index].astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        v1.generate_v1_draws(*values)


def test_v1_static_sources_do_not_open_tasks_models_cuda_or_v0_artifacts():
    root = Path(__file__).resolve().parents[2]
    schema = (root / "tiago_src/gato_tiago/multimodal_toll_v1.py").read_text()
    runner = (root / "tiago_src/gato_tiago/multimodal_toll_v1_runner.py").read_text()
    worker = (root / "tiago_src/gato_tiago/multimodal_toll_v1_worker.py").read_text()
    combined = schema + runner + worker
    for forbidden in (
        "generate_task(",
        "load_model(",
        "import pinocchio",
        "import bsqp",
        "subprocess.run(",
        "sim_forward(",
        ".solve(",
        "np.load(",
    ):
        assert forbidden not in combined
