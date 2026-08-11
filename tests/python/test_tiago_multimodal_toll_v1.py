from pathlib import Path
import json

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


def test_exactly_three_v1_capabilities_enabled_and_all_other_paths_blocked():
    assert v0_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v0_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert v1.V1_EXECUTION_AUTHORIZATION is None
    assert v1_runner.RUNNER_EXECUTION_AUTHORIZATION is not None
    assert v1_worker.WORKER_EXECUTION_AUTHORIZATION is not None
    assert oracle.TASK_CONSTRUCTION_AUTHORIZATION is not None
    assert toll.OPTIMIZER_EXECUTION_AUTHORIZATION is None
    stage0_source = (
        Path(__file__).resolve().parents[2]
        / "tiago_examples/tiago_multimodal_toll_stage0.py"
    ).read_text()
    benchmark_source = (
        Path(__file__).resolve().parents[2]
        / "tiago_examples/tiago_multimodal_toll_benchmark.py"
    ).read_text()
    assert "STAGE0_EXECUTION_AUTHORIZATION = None" in stage0_source
    assert "BENCHMARK_EXECUTION_AUTHORIZATION = None" in benchmark_source
    assert v1_runner.AUTHORIZED_OUTPUT_PATH == Path(v1.V1_OUTPUT_PATH)
    assert v1_runner.AUTHORIZED_OUTPUT_PATH.parent == v1_worker.AUTHORIZED_RUN_ROOT


@pytest.mark.parametrize("field", ["lower", "upper", "effort", "comfortable"])
def test_v1_draw_inputs_are_exact_finite_float64_vectors(field):
    values = list(_inputs())
    index = {"lower": 0, "upper": 1, "effort": 2, "comfortable": 3}[field]
    values[index] = values[index].astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        v1.generate_v1_draws(*values)


def test_v1_static_sources_keep_runtime_behind_blocked_boundaries_and_exclude_v0_artifacts():
    root = Path(__file__).resolve().parents[2]
    schema = (root / "tiago_src/gato_tiago/multimodal_toll_v1.py").read_text()
    runner = (root / "tiago_src/gato_tiago/multimodal_toll_v1_runner.py").read_text()
    worker = (root / "tiago_src/gato_tiago/multimodal_toll_v1_worker.py").read_text()

    # The schema stays pure.  The full runner/worker intentionally contain the
    # future call-boundary implementation. Only their exact-path one-shot
    # capabilities are enabled in this authorization checkpoint.
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
        assert forbidden not in schema
    assert v1_runner.RUNNER_EXECUTION_AUTHORIZATION is not None
    assert v1_worker.WORKER_EXECUTION_AUTHORIZATION is not None
    assert "/tmp/tiago-tool-center-toll-v0-authorized-once" not in runner
    assert "/tmp/tiago-tool-center-toll-v0-authorized-once" not in worker


def test_v1_full_port_preserves_extension_and_task_ledgers_without_opening_rng(monkeypatch):
    assert v1_runner.FROZEN_EXTENSIONS == v0_runner.FROZEN_EXTENSIONS
    assert v1_runner.EXPECTED_TASK_IDENTITIES == v0_runner.EXPECTED_TASK_IDENTITIES
    assert v1_runner.AUTHORIZED_OUTPUT_PATH == Path(v1.V1_OUTPUT_PATH)
    assert v1_runner.RUNNER_EXECUTION_AUTHORIZATION is not None
    assert v1_worker.WORKER_EXECUTION_AUTHORIZATION is not None
    frozen = {seed for _, seed in v1_runner.EXPECTED_TASK_IDENTITIES}
    original = np.random.default_rng

    def guarded(seed=None):
        assert seed not in frozen
        return original(seed)

    monkeypatch.setattr(np.random, "default_rng", guarded)
    with pytest.raises(RuntimeError, match="blocked"):
        v1_runner.execute_v1_runner(v1.V1_OUTPUT_PATH, authorization=object())


def test_v1_runner_allows_only_exact_fresh_output_without_crossing_runtime(tmp_path, monkeypatch):
    authorized = (tmp_path / "authorized" / "v1.json").resolve()
    calls = []
    monkeypatch.setattr(v1_runner, "AUTHORIZED_OUTPUT_PATH", authorized)

    def fake_pipeline(output):
        output = Path(output)
        v1_runner._no_existing_artifacts(output)
        calls.append(output)
        v1_runner._publish_pre_model_checkpoint(
            output, {"static_test": True}, v1_runner.FROZEN_EXTENSIONS
        )
        return {"ok": True}

    monkeypatch.setattr(v1_runner, "_production_pipeline", fake_pipeline)
    with pytest.raises(RuntimeError, match="single authorized path"):
        v1_runner.execute_v1_runner(
            tmp_path / "replacement.json",
            authorization=v1_runner.RUNNER_EXECUTION_AUTHORIZATION,
        )
    assert calls == []
    assert v1_runner.execute_v1_runner(
        authorized, authorization=v1_runner.RUNNER_EXECUTION_AUTHORIZATION
    ) == {"ok": True}
    assert calls == [authorized]
    with pytest.raises(RuntimeError, match="resume, overwrite, or rerun"):
        v1_runner.execute_v1_runner(
            authorized, authorization=v1_runner.RUNNER_EXECUTION_AUTHORIZATION
        )
    assert calls == [authorized]


def test_v1_worker_allows_only_three_exact_pairs_and_refuses_overwrite(tmp_path, monkeypatch):
    root = (tmp_path / "authorized").resolve()
    monkeypatch.setattr(v1_worker, "AUTHORIZED_RUN_ROOT", root)
    calls = []

    def fake_run(request_path, output_path):
        request_path = Path(request_path)
        output_path = Path(output_path)
        calls.append((request_path, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}")
        return {"ok": True}

    monkeypatch.setattr(v1_worker, "_run_authorized_worker", fake_run)
    for module in v1_worker.SUPPORTED_MODULES:
        leaf = module.split(".")[-1]
        request = root / f"v1.{leaf}.request.json"
        output = root / f"v1.{leaf}.json"
        assert v1_worker.execute_worker(
            request,
            output,
            authorization=v1_worker.WORKER_EXECUTION_AUTHORIZATION,
        ) == {"ok": True}
    assert len(calls) == 3

    module = next(iter(v1_worker.SUPPORTED_MODULES))
    leaf = module.split(".")[-1]
    request = root / f"v1.{leaf}.request.json"
    output = root / f"v1.{leaf}.json"
    with pytest.raises(RuntimeError, match="overwrite or rerun"):
        v1_worker.execute_worker(
            request,
            output,
            authorization=v1_worker.WORKER_EXECUTION_AUTHORIZATION,
        )
    with pytest.raises(RuntimeError, match="outside the single authorized run"):
        v1_worker.execute_worker(
            request,
            root / "v1.wrong.json",
            authorization=v1_worker.WORKER_EXECUTION_AUTHORIZATION,
        )
    assert len(calls) == 3


def test_v1_independent_authenticity_rejects_any_draw_mutation():
    draws = v1.generate_v1_draws(*_inputs())
    assert v1._draws_are_authentic_v1(draws)[0]
    changed = draws.dynamics_q_float64.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    mutated = v1.V1Draws(
        **{
            **draws.__dict__,
            "dynamics_q_float64": changed,
        }
    )
    assert not v1._draws_are_authentic_v1(mutated)[0]


def test_v1_transaction_mock_is_synthetic_only_and_non_evidence(tmp_path):
    ledger = tuple(("synthetic", 93000 + index) for index in range(12))
    task_calls = []
    worker_calls = []

    def task_factory(identity):
        task_calls.append(identity)
        return {"identity": identity, "passes": True}, {"q": np.zeros(7)}

    specs = [{"module_name": f"synthetic.v1.module{index}"} for index in range(3)]

    def worker_launcher(spec, arrays):
        worker_calls.append(spec["module_name"])
        return {
            "identity": ("worker", spec["module_name"]),
            "passes": True,
            "command": ["python-static-mock", "-m", spec["module_name"]],
        }, {"captured_count": np.asarray([len(arrays)], dtype=np.int64)}

    output = tmp_path / "v1.json"
    result = v1_runner._run_pipeline(
        output,
        ledger=ledger,
        task_factory=task_factory,
        worker_specs=specs,
        worker_launcher=worker_launcher,
        base_arrays={"base": np.zeros(1)},
        provenance={"test_override": True, "v0_artifacts_consumed": False},
        test_override=True,
    )
    assert task_calls == list(ledger)
    assert worker_calls == [row["module_name"] for row in specs]
    assert result["all_v1_gates_pass"] is False
    assert result["test_override"] is True
    assert result["optimization_evidence"] is False
    assert result["sqp_optimization_calls"] == 0
    assert result["task_construction_calls"] == 12
    assert result["worker_subprocess_calls"] == 3
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["incomplete"] is False
    assert manifest["supersedes_partial_generation"] == 15


def test_v1_gen0_and_failed_task_are_immutable_and_fail_closed(tmp_path):
    output = tmp_path / "v1.json"
    ledger = tuple(("synthetic", 94000 + index) for index in range(12))
    calls = 0

    def fail_second(identity):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic V1 task failure")
        return {"identity": identity, "passes": True}, {"q": np.zeros(7)}

    with pytest.raises(ValueError, match="synthetic V1 task failure"):
        v1_runner._run_pipeline(
            output,
            ledger=ledger,
            task_factory=fail_second,
            worker_specs=(),
            worker_launcher=lambda *_: pytest.fail("worker called"),
            base_arrays={"base": np.zeros(1)},
            provenance={"test_override": True, "v0_artifacts_consumed": False},
            test_override=True,
        )
    pointer = json.loads((tmp_path / "v1.partial.latest.json").read_text())
    assert pointer["generation"] == 2
    checkpoint = json.loads(Path(pointer["json_path"]).read_text())
    assert checkpoint["incomplete"] is True
    assert checkpoint["all_v1_gates_pass"] is False
    assert checkpoint["task_construction_attempt_count"] == 2
    assert checkpoint["rows"][-1]["error_type"] == "ValueError"
    assert checkpoint["rows"][-1]["identity"] == list(ledger[1])
    assert not output.exists()
    with pytest.raises(RuntimeError, match="resume, overwrite, or rerun"):
        v1_runner._no_existing_artifacts(output)


def test_v1_source_provenance_is_versioned_and_v0_data_is_excluded():
    assert {
        "v1_schema",
        "v1_runner",
        "v1_worker",
        "v1_tests",
        "shared_preflight_gate_source",
    }.issubset(v1.REQUIRED_SOURCE_PATHS)
    assert not {
        "v0_runner",
        "v0_worker",
        "v0_tests",
        "v0_runner_tests",
    }.intersection(v1.REQUIRED_SOURCE_PATHS)
    runner_source = Path(v1_runner.__file__).read_text()
    worker_source = Path(v1_worker.__file__).read_text()
    assert "/tmp/tiago-tool-center-toll-v0-authorized-once" not in runner_source
    assert "/tmp/tiago-tool-center-toll-v0-authorized-once" not in worker_source
    assert "np.load(" not in Path(v1.__file__).read_text()
