import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from gato_tiago import multimodal_toll as toll
from gato_tiago import multimodal_toll_v4_model_preflight as schema
from gato_tiago import multimodal_toll_v4_model_preflight_runner as runner
from gato_tiago import multimodal_toll_v4_model_preflight_worker as worker
from gato_tiago import multimodal_toll_v4_oracle_schema as v4_oracle
from gato_tiago import multimodal_toll_v4_runner as v4_runner


def _smoke_raw(reference_size=10, delta=0.25):
    xu1 = np.zeros((1, 1337), dtype=np.float32)
    xu16 = np.zeros((16, 1337), dtype=np.float32)
    initial = np.asarray(
        np.r_[np.repeat(3.0, 8), np.repeat(3.0 + delta, 8)], dtype=np.float32
    )
    baseline = np.zeros(reference_size, dtype=np.float32)
    perturbed = baseline.copy()
    if reference_size == 10:
        perturbed[6:8] = 1
    return {
        "input_xu_b1": xu1,
        "output_xu_b1": xu1.copy(),
        "input_xu_b16": xu16,
        "output_xu_b16": xu16.copy(),
        "initial_merit_b1": np.asarray([3.0], dtype=np.float32),
        "final_merit_b1": np.asarray([3.0 + 1e-7], dtype=np.float32),
        "initial_merit_b16": initial,
        "final_merit_b16": (initial + np.float32(1e-7)).astype(np.float32),
        "sqp_iters_b1": np.zeros(1, dtype=np.int32),
        "sqp_iters_b16": np.zeros(16, dtype=np.int32),
        "pcg_iters_b1": np.empty((0, 1), dtype=np.int32),
        "pcg_iters_b16": np.empty((0, 16), dtype=np.int32),
        "ls_num_iters_b1": 0,
        "ls_num_iters_b16": 0,
        "baseline_reference": baseline,
        "perturbed_reference": perturbed,
    }


def _base():
    q0 = np.zeros((12, 14), dtype=np.float32)
    q8 = np.full((12, 7), 0.01, dtype=np.float64)
    one_x = np.zeros((24, 14), dtype=np.float32)
    one_u = np.zeros((24, 7), dtype=np.float32)
    dense_x = np.zeros((12, 65, 14), dtype=np.float64)
    lower = np.full(7, -2.0, dtype=np.float64)
    upper = np.full(7, 2.0, dtype=np.float64)
    velocity = np.full(7, 3.0, dtype=np.float64)
    effort = np.full(7, 100.0, dtype=np.float64)
    broad = schema.generate_broad_algebraic_rows(lower, upper, velocity, effort)
    return {
        **broad,
        "public_solver_x0_float32": q0,
        "public_reference_float32": np.zeros((12, 10), dtype=np.float32),
        "public_default_side_int8": np.ones(12, dtype=np.int8),
        "quarantined_q8_float64": q8,
        "quarantined_q8_float32": q8.astype(np.float32),
        "quarantined_q8_oracle_only_bool": np.ones(12, dtype=np.bool_),
        "quarantined_q8_initializer_eligible_bool": np.zeros(12, dtype=np.bool_),
        "quarantined_q8_solver_seed_eligible_bool": np.zeros(12, dtype=np.bool_),
        "pin_q0_tool_float64": np.zeros((12, 3), dtype=np.float64),
        "pin_q8_tool_float64": np.zeros((12, 3), dtype=np.float64),
        "one_step_x_float32": one_x,
        "one_step_u_float32": one_u,
        "pin_one_step_float64": one_x.astype(np.float64),
        "pin_broad_one_step_float64": broad["broad_x_float64"].copy(),
        "pin_broad_tool_float64": np.zeros((32, 3), dtype=np.float64),
        "dense_x0_float32": q0.copy(),
        "dense_u_float32": np.zeros((12, 7), dtype=np.float32),
        "pin_dense_states_float64": dense_x,
        "pin_dense_tool_positions_float64": np.zeros(
            (12, 65, 3), dtype=np.float64
        ),
        "model_lower_float64": lower,
        "model_upper_float64": upper,
        "model_velocity_float64": velocity,
        "model_effort_float64": effort,
        "public_handoff_field_names_unicode": np.asarray(
            [
                "public_solver_x0_float32",
                "public_reference_float32",
                "public_default_side_int8",
            ]
        ),
        "forbidden_call_counts_int64": np.zeros(3, dtype=np.int64),
        "v4_artifact_authentication_gate_bool": np.asarray(True, dtype=np.bool_),
    }


def _worker_arrays(base, variant="toll"):
    arrays = {name: np.zeros(1, dtype=np.float32) for name in worker.EXPECTED_WORKER_ARRAY_NAMES}
    arrays.update(
        {
            "captured_public_q0_float32": base["public_solver_x0_float32"][:, :7].copy(),
            "captured_quarantined_q8_float32": base["quarantined_q8_float64"].astype(np.float32),
            "cuda_q0_fk_b1_float32": base["pin_q0_tool_float64"].astype(np.float32),
            "cuda_q0_fk_b16_float32": base["pin_q0_tool_float64"].astype(np.float32),
            "cuda_q8_fk_b1_float32": base["pin_q8_tool_float64"].astype(np.float32),
            "cuda_q8_fk_b16_float32": base["pin_q8_tool_float64"].astype(np.float32),
            "captured_one_step_x_float32": base["one_step_x_float32"].copy(),
            "captured_one_step_u_float32": base["one_step_u_float32"].copy(),
            "cuda_one_step_b1_float32": base["pin_one_step_float64"].astype(np.float32),
            "cuda_one_step_b16_float32": base["pin_one_step_float64"].astype(np.float32),
            "captured_broad_x_float32": base["broad_x_float32"].copy(),
            "captured_broad_q_float32": base["broad_x_float32"][:, :7].copy(),
            "captured_broad_u_float32": base["broad_u_float32"].copy(),
            "cuda_broad_one_step_b1_float32": base[
                "pin_broad_one_step_float64"
            ].astype(np.float32),
            "cuda_broad_one_step_b16_float32": base[
                "pin_broad_one_step_float64"
            ].astype(np.float32),
            "cuda_broad_fk_b1_float32": base["pin_broad_tool_float64"].astype(
                np.float32
            ),
            "cuda_broad_fk_b16_float32": base["pin_broad_tool_float64"].astype(
                np.float32
            ),
            "captured_dense_x0_float32": base["dense_x0_float32"].copy(),
            "captured_dense_u_float32": base["dense_u_float32"].copy(),
            "cuda_dense_states_float32": base["pin_dense_states_float64"].astype(np.float32),
            "cuda_dense_tool_positions_float32": base["pin_dense_tool_positions_float64"].astype(np.float32),
            "dense_time_float64": np.arange(65, dtype=np.float64) * schema.ONE_STEP_DT / 64,
            "dense_controls_float32": np.zeros((12, 64, 7), dtype=np.float32),
        }
    )
    width = 10 if variant == "toll" else 6
    smoke = _smoke_raw(width)
    if variant != "toll":
        smoke["perturbed_reference"] = smoke["baseline_reference"].copy()
        changed = (0,) if variant == "plain_ref6" else (0, 3, 4)
        smoke["perturbed_reference"][list(changed)] = 1
    for name, value in smoke.items():
        if isinstance(value, np.ndarray):
            arrays[f"reference_smoke_{name}"] = value
    arrays["reference_smoke_ls_num_iters_b1"] = np.asarray(0, dtype=np.int32)
    arrays["reference_smoke_ls_num_iters_b16"] = np.asarray(0, dtype=np.int32)
    return arrays


def _rows_and_arrays(base):
    rows = []
    arrays = {}
    for spec in schema.FROZEN_EXTENSIONS:
        rows.append(
            {
                "module_name": spec["module_name"],
                "passes": True,
                "reference_smoke": {"all_reference_smoke_gates_pass": True},
                "sqp_optimization_calls": 0,
                "q8_scope": "tool_position_only",
                "q8_initializer_eligible": False,
                "q8_solver_seed_eligible": False,
            }
        )
        rows[-1]["reference_smoke"].update(
            {
                "wrong_width_rejected_before_native_kernel": True,
                "wrong_width_value_error_exact": True,
                "wrong_width_rejection_calls": 1,
                "diagnostic_zero_iteration_solve_calls": 2,
                "sqp_optimization_calls": 0,
                "optimization_evidence": False,
            }
        )
        arrays[spec["module_name"]] = _worker_arrays(base, spec["variant"])
    return rows, arrays


def test_static_tokens_are_closed_and_v4_artifact_is_hard_pinned():
    assert schema.MODEL_PREFLIGHT_EXECUTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert v4_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v4_oracle.TASK_CONSTRUCTION_AUTHORIZATION is None
    assert schema.EXPECTED_V4_ARRAY_COUNT == 603
    assert len(schema.V4_ARTIFACT_PATHS_AND_HASHES) == 4
    assert all(
        schema.is_sha256(digest)
        for _, digest in schema.V4_ARTIFACT_PATHS_AND_HASHES.values()
    )


def test_blocked_entrypoints_touch_no_files_or_subprocess(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "_run_authorized_worker", lambda *_: calls.append("worker"))
    monkeypatch.setattr(runner, "_production_pipeline", lambda *_args, **_kwargs: calls.append("runner"))
    with pytest.raises(RuntimeError, match="blocked"):
        worker.execute_worker(tmp_path / "request.json", tmp_path / "worker.json")
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_model_preflight(tmp_path / "preflight.json")
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_reference_smoke_retains_zero_iteration_native_width_behavior():
    assert worker.certify_reference_smoke(
        _smoke_raw(), variant="toll", reference_size=10
    )["all_reference_smoke_gates_pass"]
    for variant, fields in (("plain_ref6", (0,)), ("pillar_ref6", (0, 3, 4))):
        raw = _smoke_raw(6)
        raw["perturbed_reference"] = raw["baseline_reference"].copy()
        raw["perturbed_reference"][list(fields)] = 1
        assert worker.certify_reference_smoke(
            raw, variant=variant, reference_size=6
        )["all_reference_smoke_gates_pass"]


@pytest.mark.parametrize(
    "mutation",
    [
        "dtype",
        "nan",
        "pcg",
        "xu",
        "reference",
        "perturbed_repeat",
        "perturbed_final_repeat",
    ],
)
def test_reference_smoke_rejects_nonfinite_dtype_telemetry_and_behavior(mutation):
    raw = _smoke_raw()
    if mutation == "dtype":
        raw["input_xu_b1"] = raw["input_xu_b1"].astype(np.float64)
    elif mutation == "nan":
        raw["initial_merit_b1"][0] = np.nan
    elif mutation == "pcg":
        raw["pcg_iters_b16"] = np.zeros((1, 16), dtype=np.int32)
    elif mutation == "xu":
        raw["output_xu_b1"][0, 0] = 1
    elif mutation == "reference":
        raw["perturbed_reference"][:] = raw["baseline_reference"]
    elif mutation == "perturbed_repeat":
        raw["initial_merit_b16"][9] += np.float32(0.02)
    else:
        raw["final_merit_b16"][9] += np.float32(0.02)
    assert not worker.certify_reference_smoke(
        raw, variant="toll", reference_size=10
    )["all_reference_smoke_gates_pass"]


def test_model_certificate_passes_exact_synthetic_boundaries_and_reports_state_l2():
    base = _base()
    rows, arrays = _rows_and_arrays(base)
    result = runner.certify_model_preflight(base, rows, arrays)
    assert result["all_model_preflight_gates_pass"]
    assert all(
        row["max_dense_state_l2_report_only"] == 0
        for row in result["per_module"].values()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "q8_input",
        "q8_tool",
        "one_step",
        "broad_step",
        "broad_fk_b1",
        "broad_fk_b16",
        "broad_q_capture",
        "broad_code",
        "dense_tool",
        "dense_limit",
        "shape",
        "public_handoff",
        "quarantine",
        "forbidden_call",
        "extra_base",
        "missing_base",
    ],
)
def test_model_certificate_fails_adversarial_boundary_mutations(mutation):
    base = _base()
    rows, arrays = _rows_and_arrays(base)
    target = arrays[rows[0]["module_name"]]
    if mutation == "q8_input":
        target["captured_quarantined_q8_float32"][0, 0] += 1
    elif mutation == "q8_tool":
        target["cuda_q8_fk_b1_float32"][0, 0] += 0.01
    elif mutation == "one_step":
        target["cuda_one_step_b1_float32"][0, 0] += 0.01
    elif mutation == "broad_step":
        target["cuda_broad_one_step_b1_float32"][0, 0] += 0.01
    elif mutation == "broad_fk_b1":
        target["cuda_broad_fk_b1_float32"][0, 0] += 0.01
    elif mutation == "broad_fk_b16":
        target["cuda_broad_fk_b16_float32"][0, 0] += 0.01
    elif mutation == "broad_q_capture":
        target["captured_broad_q_float32"][0, 0] += 0.01
    elif mutation == "broad_code":
        base["broad_q_codes_int8"][0, 0] += 1
    elif mutation == "dense_tool":
        target["cuda_dense_tool_positions_float32"][0, 0, 0] += 0.01
    elif mutation == "dense_limit":
        target["cuda_dense_states_float32"][0, 0, 0] = 3
    elif mutation == "shape":
        target["cuda_dense_states_float32"] = target["cuda_dense_states_float32"][:, :-1]
    elif mutation == "public_handoff":
        base["public_handoff_field_names_unicode"] = np.asarray(
            ["public_solver_x0_float32", "quarantined_q8_float64"]
        )
    elif mutation == "quarantine":
        base["quarantined_q8_solver_seed_eligible_bool"][0] = True
    elif mutation == "forbidden_call":
        base["forbidden_call_counts_int64"][0] = 1
    elif mutation == "extra_base":
        base["fabricated_extra"] = np.zeros(1)
    else:
        del base["quarantined_q8_float32"]
    result = runner.certify_model_preflight(base, rows, arrays)
    assert not result["all_model_preflight_gates_pass"]


def test_split_fk_goal_and_dense_tolerances_are_independently_gated():
    base = _base()
    rows, arrays = _rows_and_arrays(base)
    target = arrays[rows[0]["module_name"]]
    target["cuda_q8_fk_b1_float32"][:, 0] = np.float32(2e-4)
    target["cuda_q8_fk_b16_float32"][:, 0] = np.float32(2e-4)
    result = runner.certify_model_preflight(base, rows, arrays)
    assert not result["per_module"][rows[0]["module_name"]][
        "q8_pin_cuda_tool_pass"
    ]
    assert result["per_module"][rows[0]["module_name"]][
        "q8_b1_public_goal_pass"
    ]
    base = _base()
    rows, arrays = _rows_and_arrays(base)
    target = arrays[rows[0]["module_name"]]
    target["cuda_q8_fk_b16_float32"][:, 0] = np.float32(6e-4)
    result = runner.certify_model_preflight(base, rows, arrays)
    report = result["per_module"][rows[0]["module_name"]]
    assert report["q8_b1_public_goal_pass"]
    assert not report["q8_b16_public_goal_pass"]


def test_broad_algebraic_grid_is_exact_in_limit_signed_and_rng_free(monkeypatch):
    calls = []
    monkeypatch.setattr(np.random, "default_rng", lambda *_: calls.append("rng"))
    lower = np.asarray([-2.0, -1.8, -2.2, -2.4, -2.6, -2.8, -3.0])
    upper = -lower
    velocity = np.arange(1, 8, dtype=np.float64)
    effort = 10 * velocity
    first = schema.generate_broad_algebraic_rows(lower, upper, velocity, effort)
    second = schema.generate_broad_algebraic_rows(lower, upper, velocity, effort)
    assert calls == []
    assert set(first) == {
        "broad_q_codes_int8",
        "broad_qd_codes_int8",
        "broad_u_codes_int8",
        "broad_x_float64",
        "broad_u_float64",
        "broad_x_float32",
        "broad_u_float32",
    }
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert np.unique(first["broad_x_float64"], axis=0).shape[0] == 32
    assert np.any(first["broad_x_float64"][:, 7:] < 0)
    assert np.any(first["broad_x_float64"][:, 7:] > 0)
    assert np.any(first["broad_u_float64"] < 0)
    assert np.any(first["broad_u_float64"] > 0)
    assert np.all(first["broad_x_float64"][:, :7] >= lower + 0.08)
    assert np.all(first["broad_x_float64"][:, :7] <= upper - 0.08)
    assert np.all(np.abs(first["broad_x_float64"][:, 7:]) <= 0.2 * velocity)
    assert np.all(np.abs(first["broad_u_float64"]) <= 0.2 * effort)


def test_q8_is_quarantined_and_public_initializer_taint_rejects_it():
    assert {"quarantined_q8", "oracle_endpoint_q", "construction_q8"}.issubset(
        toll.FORBIDDEN_INITIALIZER_FIELDS
    )
    with pytest.raises(ValueError, match="tainted"):
        toll.validate_initializer_inputs({"x0": np.zeros(14), "quarantined_q8": np.zeros(7)})
    source = inspect.getsource(worker._run_authorized_worker)
    before, after = source.split("# q8 is quarantined here:", 1)
    assert "quarantined_q8" in before
    q8_section, remainder = after.split("cuda_one_b1 =", 1)
    assert q8_section.count("tool_b1(quarantined_q8)") == 1
    assert q8_section.count("tool_b16(quarantined_q8)") == 1
    assert "quarantined_q8" not in remainder.split("arrays =", 1)[0]


def test_generation_zero_precedes_artifact_and_model_and_failure_is_retained(tmp_path):
    output = tmp_path / "preflight.json"
    order = []

    def artifact_loader():
        order.append("artifact")
        raise RuntimeError("injected artifact failure")

    with pytest.raises(RuntimeError, match="injected artifact"):
        runner._run_pipeline(
            output,
            provenance={"test_override": True},
            artifact_loader=artifact_loader,
            model_preflight_factory=lambda _: pytest.fail("model called"),
            worker_launcher=lambda *_: pytest.fail("worker called"),
            test_override=True,
        )
    assert order == ["artifact"]
    pointer = json.loads((tmp_path / "preflight.partial.latest.json").read_text())
    assert pointer["generation"] == 1
    retained = json.loads(Path(pointer["json_path"]).read_text())
    assert retained["stage"] == "artifact_failed"
    assert retained["rows"][-1]["error_type"] == "RuntimeError"
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_mock_transaction_uses_three_isolated_workers_and_never_passes_as_evidence(tmp_path):
    base = _base()
    rows, worker_arrays = _rows_and_arrays(base)
    calls = []

    def launcher(spec, _arrays):
        calls.append(spec["module_name"])
        index = calls.index(spec["module_name"])
        return rows[index], worker_arrays[spec["module_name"]]

    provenance = {
        "git_head_at_start": "a" * 40,
        "git_head_at_end": "a" * 40,
        "tracked_tree_clean_at_start": True,
        "tracked_tree_clean_at_end": True,
        "source_hashes": {
            key: {"path": path, "sha256": "b" * 64}
            for key, path in schema.REQUIRED_SOURCE_PATHS.items()
        },
        "task_rng_calls": 0,
        "task_construction_calls": 0,
        "predecessor_artifact_loads": 0,
    }
    output = tmp_path / "preflight.json"
    result = runner._run_pipeline(
        output,
        provenance=provenance,
        artifact_loader=lambda: (
            {
                key: value
                for key, value in base.items()
                if key.startswith("public_") or key.startswith("quarantined_")
            },
            {"all_v4_artifact_authentication_gates_pass": True},
        ),
        model_preflight_factory=lambda _: {
            key: value
            for key, value in base.items()
            if not key.startswith("public_") and not key.startswith("quarantined_")
        },
        worker_launcher=launcher,
        test_override=True,
    )
    assert calls == [row["module_name"] for row in schema.FROZEN_EXTENSIONS]
    assert result["worker_subprocess_calls"] == 3
    assert result["sqp_optimization_calls"] == 0
    assert result["optimization_evidence"] is False
    assert result["all_model_preflight_gates_pass"] is False
    assert (tmp_path / "preflight.partial.0000.json").is_file()
    assert (tmp_path / "preflight.partial.0005.json").is_file()


def test_static_sources_never_open_task_rng_or_regenerate_v4_tasks():
    sources = "\n".join(
        Path(module.__file__).read_text() for module in (schema, runner, worker)
    )
    assert "default_rng(" not in sources
    assert "generate_task(" not in sources
    assert "multimodal_toll_v3" not in sources
    assert "multimodal_toll_v2" not in sources
    production = inspect.getsource(runner._production_pipeline)
    assert production.index("artifact_loader=artifact_loader") > production.index(
        "provenance = _source_provenance"
    )
    assert "import pinocchio as pin" in inspect.getsource(
        runner._production_pipeline
    )


def test_exact_constant_acceleration_formula_and_dense_grid():
    x = np.arange(14, dtype=np.float64) / 10
    qdd = np.arange(7, dtype=np.float64) / 20
    result = runner.constant_acceleration_step(x, qdd, schema.ONE_STEP_DT)
    expected = np.r_[
        x[:7] + schema.ONE_STEP_DT * x[7:] + 0.5 * schema.ONE_STEP_DT**2 * qdd,
        x[7:] + schema.ONE_STEP_DT * qdd,
    ]
    assert np.array_equal(result, expected)
    worker_source = inspect.getsource(worker._run_authorized_worker)
    assert "ONE_STEP_DT / DENSE_SUBSTEPS" in worker_source


def test_repository_root_uses_actual_path_and_required_source_manifest_exists():
    root = Path(__file__).resolve().parents[2]
    assert runner.repository_root(Path(runner.__file__)) == root
    assert all((root / path).is_file() for path in schema.REQUIRED_SOURCE_PATHS.values())
    assert schema.REQUIRED_SOURCE_PATHS["python_bindings"] == "python/bindings.cu"
    assert (
        schema.REQUIRED_SOURCE_PATHS["tool_position_kernel"]
        == "gato/bsqp/kernels/tool_position.cuh"
    )
    assert schema.FROZEN_BUILD_COMMIT == "9bb1ceaf64787597f1ea7df5566f84d0635c4b7f"
    assert schema.FROZEN_CUDA_ARCH == "61-real"
    assert {
        path: runner.sha256_file(root / path)
        for path in schema.FROZEN_BUILD_SOURCE_HASHES
    } == schema.FROZEN_BUILD_SOURCE_HASHES
