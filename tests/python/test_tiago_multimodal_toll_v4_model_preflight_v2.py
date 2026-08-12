import hashlib
import inspect
import json
import copy
import re
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gato_tiago import multimodal_toll as toll
from gato_tiago import multimodal_toll_v4_model_preflight_v2 as schema
from gato_tiago import multimodal_toll_v4_model_preflight_v2_runner as runner
from gato_tiago import multimodal_toll_v4_model_preflight_v2_worker as worker
from gato_tiago import multimodal_toll_v4_model_preflight as v1_schema
from gato_tiago import multimodal_toll_v4_model_preflight_runner as v1_runner
from gato_tiago import multimodal_toll_v4_model_preflight_worker as v1_worker
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


def _valid_runner_provenance():
    sources = {
        name: {"path": path, "sha256": "b" * 64}
        for name, path in schema.REQUIRED_SOURCE_PATHS.items()
    }
    return {
        "git_head_at_start": "a" * 40,
        "git_head_at_end": "a" * 40,
        "tracked_tree_clean_at_start": True,
        "tracked_tree_clean_at_end": True,
        "source_hashes": copy.deepcopy(sources),
        "source_hashes_at_start": copy.deepcopy(sources),
        "source_hashes_at_end": copy.deepcopy(sources),
        "cwd": str(schema.AUTHORIZED_CWD),
        "orig_argv": list(schema.AUTHORIZED_ORIG_ARGV),
        "exact_command": " ".join(schema.AUTHORIZED_ORIG_ARGV),
        "task_rng_calls": 0,
        "task_construction_calls": 0,
        "predecessor_artifact_loads": 0,
        "accepted_portability_smoke_artifact_loads": 4,
        "accepted_v4_artifact_loads": 4,
        "rejected_v1_artifact_loads": 0,
        "rejected_v1_artifact_hashes_report_only": dict(
            schema.REJECTED_V1_ARTIFACT_HASHES_REPORT_ONLY
        ),
        "portability_smoke_authentication": {
            "all_portability_smoke_authentication_gates_pass": True
        },
        "extension_hashes": {
            row["module_name"]: row["extension_sha256"]
            for row in schema.FROZEN_EXTENSIONS
        },
        "extension_sizes": {
            row["module_name"]: row["extension_size_bytes"]
            for row in schema.FROZEN_EXTENSIONS
        },
        "frozen_build_commit": schema.FROZEN_BUILD_COMMIT,
        "frozen_build_source_hashes": dict(schema.FROZEN_BUILD_SOURCE_HASHES),
        "current_build_source_hashes": dict(schema.FROZEN_BUILD_SOURCE_HASHES),
        "configured_cuda_arch": schema.FROZEN_CUDA_ARCH,
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
                "identity": ["worker", spec["module_name"]],
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


def test_rejected_model_preflight_tokens_are_disabled_and_artifact_is_hard_pinned():
    assert schema.MODEL_PREFLIGHT_EXECUTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert v1_schema.MODEL_PREFLIGHT_EXECUTION_AUTHORIZATION is None
    assert v1_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v1_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert v4_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v4_oracle.TASK_CONSTRUCTION_AUTHORIZATION is None
    assert schema.EXPECTED_V4_ARRAY_COUNT == 603
    assert len(schema.V4_ARTIFACT_PATHS_AND_HASHES) == 4
    assert len(schema.PORTABILITY_SMOKE_PATHS_AND_HASHES) == 4
    assert all(
        schema.is_sha256(digest)
        for _, digest in schema.V4_ARTIFACT_PATHS_AND_HASHES.values()
    )
    assert all(
        schema.is_sha256(digest)
        for _, digest in schema.PORTABILITY_SMOKE_PATHS_AND_HASHES.values()
    )
    assert all(
        schema.is_sha256(digest)
        for digest in schema.REJECTED_V1_ARTIFACT_HASHES_REPORT_ONLY.values()
    )
    assert schema.AUTHORIZED_OUTPUT_PATH == Path(
        "/tmp/tiago-tool-center-toll-v4-model-preflight-v2-authorized-once/model.json"
    )
    assert schema.AUTHORIZED_CWD == Path("/workspace/GATO")
    assert list(schema.AUTHORIZED_ORIG_ARGV) == [
        "python",
        "-B",
        "-m",
        "gato_tiago.multimodal_toll_v4_model_preflight_v2_runner",
        "--execute",
        "--output",
        str(schema.AUTHORIZED_OUTPUT_PATH),
    ]
    assert [row["extension_sha256"] for row in schema.FROZEN_EXTENSIONS] == [
        "dbf7a016b644d75a36ebaddb6c25a456dec51814d5f794f5f2c5cfaa2b184ccb",
        "3a97a4682c949eba483ad7be39fc64416032abae0141e6f049af58405f8b8f9f",
        "988ef42c36dedd35227e54294ad1e2cbfb99f9c8f5da290aa8683b2e0a6de978",
    ]
    assert [row["extension_size_bytes"] for row in schema.FROZEN_EXTENSIONS] == [
        6686384,
        6764208,
        6678192,
    ]
    assert {
        name: digest
        for name, (_path, digest) in schema.PORTABILITY_SMOKE_PATHS_AND_HASHES.items()
    } == {
        "final_json": "1f50c8fe027cc0fdd790bf13446a6945bfe87388308ee2f4501dc2716951cb5e",
        "final_npz": "4b2cd6e13f14e43e8e37e67af03cfe9f850053075124528b11cd56a9f22f2760",
        "final_manifest": "0ecaee797c46d2ad8b50a3b2f8e0d4c936925fec1645076aa3e11df5cbdbaf03",
        "generation4_pointer": "5d2058149583a2dc6ad8cfebc31188d73c467c92aa9cfad8aef2611edda3772b",
    }


def test_wrong_authorizations_touch_no_files_or_subprocess(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "_run_authorized_worker", lambda *_: calls.append("worker"))
    monkeypatch.setattr(runner, "_production_pipeline", lambda *_args, **_kwargs: calls.append("runner"))
    with pytest.raises(RuntimeError, match="blocked"):
        worker.execute_worker(
            tmp_path / "request.json",
            tmp_path / "worker.json",
            authorization=object(),
        )
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute_model_preflight(
            tmp_path / "preflight.json",
            authorization=object(),
        )
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_all_public_execution_tokens_are_disabled_repo_wide():
    root = Path(__file__).resolve().parents[2]
    enabled = []
    pattern = re.compile(r"^([A-Z][A-Z0-9_]*AUTHORIZATION) = object\(\)$")
    for path in sorted((root / "tiago_src/gato_tiago").glob("*.py")):
        for line in path.read_text().splitlines():
            if pattern.fullmatch(line):
                enabled.append((path.name, line))
    assert enabled == []


def test_authorized_boundaries_are_exact_one_shot_paths_without_real_execution(
    tmp_path, monkeypatch
):
    runner_authorization = object()
    worker_authorization = object()
    monkeypatch.setattr(runner, "RUNNER_EXECUTION_AUTHORIZATION", runner_authorization)
    monkeypatch.setattr(worker, "WORKER_EXECUTION_AUTHORIZATION", worker_authorization)
    root = (tmp_path / "authorized").resolve()
    authorized = root / "model.json"
    runner_calls = []

    def fake_pipeline(output, *, token=None):
        assert token is runner._PRODUCTION_PIPELINE_TOKEN
        runner._no_existing_artifacts(output)
        runner_calls.append(Path(output))
        runner._checkpoint(output, 0, "mock", [], {}, {"test_override": True})
        return {"mock": True}

    monkeypatch.setattr(runner, "AUTHORIZED_OUTPUT_PATH", authorized)
    monkeypatch.setattr(runner, "_production_pipeline", fake_pipeline)
    with pytest.raises(RuntimeError, match="authorized path"):
        runner.execute_model_preflight(
            root / "replacement.json",
            authorization=runner_authorization,
        )
    assert runner_calls == []
    assert runner.execute_model_preflight(
        authorized, authorization=runner_authorization
    ) == {"mock": True}
    assert runner_calls == [authorized]
    with pytest.raises(RuntimeError, match="resume, overwrite, or rerun"):
        runner.execute_model_preflight(
            authorized, authorization=runner_authorization
        )

    monkeypatch.setattr(worker, "AUTHORIZED_RUN_ROOT", root)
    worker_calls = []

    def fake_worker(request, output):
        worker_calls.append((Path(request), Path(output)))
        return {"mock": True}

    monkeypatch.setattr(worker, "_run_authorized_worker", fake_worker)
    pairs = []
    for module in worker.SUPPORTED_MODULES:
        leaf = module.split(".")[-1]
        request = root / f"model.{leaf}.request.json"
        output = root / f"model.{leaf}.json"
        pairs.append((request, output))
        assert worker.execute_worker(
            request,
            output,
            authorization=worker_authorization,
        ) == {"mock": True}
    with pytest.raises(RuntimeError, match="single authorized run"):
        worker.execute_worker(
            root / "wrong.request.json",
            pairs[0][1],
            authorization=worker_authorization,
        )
    assert worker_calls == pairs
    pairs[0][1].parent.mkdir(parents=True, exist_ok=True)
    pairs[0][1].write_text("{}")
    with pytest.raises(RuntimeError, match="overwrite or rerun"):
        worker.execute_worker(
            *pairs[0],
            authorization=worker_authorization,
        )


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


def test_generation_zero_precedes_smoke_and_smoke_failure_is_retained(tmp_path):
    output = tmp_path / "preflight.json"
    order = []

    def smoke_loader():
        order.append("smoke")
        raise RuntimeError("injected smoke failure")

    with pytest.raises(RuntimeError, match="injected smoke"):
        runner._run_pipeline(
            output,
            provenance={"test_override": True},
            smoke_loader=smoke_loader,
            artifact_loader=lambda: pytest.fail("V4 artifact called"),
            model_preflight_factory=lambda _: pytest.fail("model called"),
            worker_launcher=lambda *_: pytest.fail("worker called"),
            test_override=True,
        )
    assert order == ["smoke"]
    pointer = json.loads((tmp_path / "preflight.partial.latest.json").read_text())
    assert pointer["generation"] == 1
    retained = json.loads(Path(pointer["json_path"]).read_text())
    assert retained["stage"] == "portability_smoke_failed"
    assert retained["completed_worker_count"] == 0
    assert retained["pending_worker_modules"] == [
        row["module_name"] for row in schema.FROZEN_EXTENSIONS
    ]
    assert retained["rows"][-1]["error_type"] == "RuntimeError"
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_v4_failure_after_smoke_is_retained_before_pin(tmp_path):
    output = tmp_path / "preflight.json"
    order = []

    def artifact_loader():
        order.append("v4")
        raise RuntimeError("injected V4 failure")

    with pytest.raises(RuntimeError, match="injected V4"):
        runner._run_pipeline(
            output,
            provenance={"test_override": True},
            smoke_loader=lambda: order.append("smoke")
            or {"all_portability_smoke_authentication_gates_pass": True},
            artifact_loader=artifact_loader,
            model_preflight_factory=lambda _: pytest.fail("model called"),
            worker_launcher=lambda *_: pytest.fail("worker called"),
            test_override=True,
        )
    assert order == ["smoke", "v4"]
    pointer = json.loads((tmp_path / "preflight.partial.latest.json").read_text())
    assert pointer["generation"] == 2
    retained = json.loads(Path(pointer["json_path"]).read_text())
    assert retained["stage"] == "artifact_failed"
    assert retained["completed_worker_count"] == 0
    assert retained["pending_worker_modules"] == [
        row["module_name"] for row in schema.FROZEN_EXTENSIONS
    ]


def test_model_failure_is_not_counted_as_a_worker_attempt(tmp_path):
    with pytest.raises(RuntimeError, match="injected model"):
        runner._run_pipeline(
            tmp_path / "preflight.json",
            provenance={"test_override": True},
            smoke_loader=lambda: {
                "all_portability_smoke_authentication_gates_pass": True
            },
            artifact_loader=lambda: (
                {},
                {"all_v4_artifact_authentication_gates_pass": True},
            ),
            model_preflight_factory=lambda _: (_ for _ in ()).throw(
                RuntimeError("injected model")
            ),
            worker_launcher=lambda *_: pytest.fail("worker called"),
            test_override=True,
        )
    pointer = json.loads(
        (tmp_path / "preflight.partial.latest.json").read_text()
    )
    retained = json.loads(Path(pointer["json_path"]).read_text())
    assert pointer["generation"] == 3
    assert retained["stage"] == "model_failed"
    assert retained["completed_worker_count"] == 0
    assert retained["pending_worker_modules"] == [
        row["module_name"] for row in schema.FROZEN_EXTENSIONS
    ]


@pytest.mark.parametrize("failed_prerequisite", ["smoke", "v4"])
def test_false_prerequisite_gate_makes_zero_pin_or_worker_calls(
    tmp_path, failed_prerequisite
):
    calls = []

    def smoke_loader():
        calls.append("smoke")
        return {
            "all_portability_smoke_authentication_gates_pass": (
                failed_prerequisite != "smoke"
            )
        }

    def artifact_loader():
        calls.append("v4")
        return {}, {
            "all_v4_artifact_authentication_gates_pass": failed_prerequisite != "v4"
        }

    with pytest.raises(RuntimeError, match="prerequisite"):
        runner._run_pipeline(
            tmp_path / "preflight.json",
            provenance={"test_override": True},
            smoke_loader=smoke_loader,
            artifact_loader=artifact_loader,
            model_preflight_factory=lambda _: calls.append("pin"),
            worker_launcher=lambda *_: calls.append("worker"),
            test_override=True,
        )
    assert calls == (["smoke"] if failed_prerequisite == "smoke" else ["smoke", "v4"])


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
        smoke_loader=lambda: {
            "all_portability_smoke_authentication_gates_pass": True
        },
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
    assert (tmp_path / "preflight.partial.0007.json").is_file()


def test_worker_failure_still_publishes_honest_generation7_before_abort(tmp_path):
    base = _base()
    rows, worker_arrays = _rows_and_arrays(base)
    attempted = []

    def launcher(spec, _arrays):
        attempted.append(spec["module_name"])
        if len(attempted) == 2:
            raise RuntimeError("injected worker failure")
        index = len(attempted) - 1
        return rows[index], worker_arrays[spec["module_name"]]

    provenance = {"git_head_at_end": None, "source_hashes_at_end": None}

    def finish(value):
        value["git_head_at_end"] = "a" * 40
        value["source_hashes_at_end"] = {"finished": True}
        value["finish_called"] = True

    with pytest.raises(RuntimeError, match="worker failure"):
        runner._run_pipeline(
            tmp_path / "preflight.json",
            provenance=provenance,
            smoke_loader=lambda: {
                "all_portability_smoke_authentication_gates_pass": True
            },
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
            finish_provenance=finish,
            test_override=False,
        )
    pointer = json.loads(
        (tmp_path / "preflight.partial.latest.json").read_text()
    )
    retained = json.loads(Path(pointer["json_path"]).read_text())
    assert pointer["generation"] == 7
    assert retained["stage"] == "end_provenance_captured"
    assert retained["completed_worker_count"] == 3
    assert retained["pending_worker_modules"] == []
    assert retained["provenance"]["finish_called"] is True
    assert retained["provenance"]["git_head_at_end"] == "a" * 40
    assert retained["provenance"]["source_hashes_at_end"] == {"finished": True}
    assert not (tmp_path / "preflight.json").exists()


def test_static_sources_never_open_task_rng_or_regenerate_v4_tasks():
    sources = "\n".join(
        Path(module.__file__).read_text() for module in (schema, runner, worker)
    )
    assert "default_rng(" not in sources
    assert "generate_task(" not in sources
    assert "multimodal_toll_v3" not in sources
    assert "multimodal_toll_v2" not in sources
    production = inspect.getsource(runner._production_pipeline)
    assert production.index("smoke_loader=smoke_loader") > production.index(
        "provenance = _source_provenance"
    )
    assert production.index("artifact_loader=artifact_loader") > production.index(
        "smoke_loader=smoke_loader"
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
    assert schema.FROZEN_BUILD_COMMIT == "154556ab7b45a5137a6d9c107c1e6e433b9cc057"
    assert schema.FROZEN_CUDA_ARCH == "61-real"
    assert {
        path: hashlib.sha256(
            subprocess.check_output(
                ["git", "show", f"{schema.FROZEN_BUILD_COMMIT}:{path}"], cwd=root
            )
        ).hexdigest()
        for path in schema.FROZEN_BUILD_SOURCE_HASHES
    } == schema.FROZEN_BUILD_SOURCE_HASHES
    provenance_source = inspect.getsource(runner._source_provenance)
    assert '"git_head_at_end": None' in provenance_source
    assert '"tracked_tree_clean_at_end": None' in provenance_source
    assert '"full_git_status_at_end": None' in provenance_source
    assert '"source_hashes_at_end": None' in provenance_source


def _synthetic_portability_smoke(tmp_path):
    arrays = {"q_float32": np.zeros(7, dtype=np.float32)}
    npz = tmp_path / "smoke.npz"
    np.savez_compressed(npz, **arrays)
    array_hashes = {name: schema.array_hash(value) for name, value in arrays.items()}
    summary = {
        "all_smoke_gates_pass": True,
        "incomplete": False,
        "certificate": {
            "all_smoke_gates_pass": True,
            "constructor_calls": 6,
            "fk_calls": 6,
            "cross_module_fk_bitwise_equal": True,
        },
        "optimization_evidence": False,
        "timing_evidence": False,
        "provenance": {
            "extension_hashes": {
                row["module_name"]: row["extension_sha256"]
                for row in schema.FROZEN_EXTENSIONS
            },
            "extension_sizes": {
                row["module_name"]: row["extension_size_bytes"]
                for row in schema.FROZEN_EXTENSIONS
            },
            "frozen_build_head": schema.FROZEN_BUILD_COMMIT,
            "frozen_cuda_arch": schema.FROZEN_CUDA_ARCH,
            "forbidden_call_counts": dict(
                schema.EXPECTED_SMOKE_FORBIDDEN_CALL_COUNTS
            ),
        },
        "rows": [{"passes": True} for _ in range(3)],
        "array_names": sorted(arrays),
        "array_hashes": array_hashes,
    }
    final_json = tmp_path / "smoke.json"
    final_json.write_text(json.dumps(summary))
    manifest = tmp_path / "smoke.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "all_smoke_gates_pass": True,
                "json_sha256": runner.sha256_file(final_json),
                "npz_sha256": runner.sha256_file(npz),
            }
        )
    )
    pointer = tmp_path / "smoke.partial.latest.json"
    pointer.write_text(json.dumps({"generation": 4}))
    paths = {
        "final_json": (final_json, runner.sha256_file(final_json)),
        "final_npz": (npz, runner.sha256_file(npz)),
        "final_manifest": (manifest, runner.sha256_file(manifest)),
        "generation4_pointer": (pointer, runner.sha256_file(pointer)),
    }
    return paths, summary, manifest


def test_smoke_authentication_is_hash_bound_semantic_and_precedes_pin(
    tmp_path, monkeypatch
):
    paths, summary, manifest = _synthetic_portability_smoke(tmp_path)
    monkeypatch.setattr(runner, "PORTABILITY_SMOKE_PATHS_AND_HASHES", paths)
    assert runner.authenticate_portability_smoke()[
        "all_portability_smoke_authentication_gates_pass"
    ]

    summary["all_smoke_gates_pass"] = False
    final_json = paths["final_json"][0]
    final_json.write_text(json.dumps(summary))
    manifest_value = json.loads(manifest.read_text())
    manifest_value["json_sha256"] = runner.sha256_file(final_json)
    manifest.write_text(json.dumps(manifest_value))
    paths = {
        **paths,
        "final_json": (final_json, runner.sha256_file(final_json)),
        "final_manifest": (manifest, runner.sha256_file(manifest)),
    }
    monkeypatch.setattr(runner, "PORTABILITY_SMOKE_PATHS_AND_HASHES", paths)
    with pytest.raises(RuntimeError, match="authenticity"):
        runner.authenticate_portability_smoke()


def test_worker_request_hard_gates_frozen_extension_size(tmp_path, monkeypatch):
    extension = tmp_path / "extension.so"
    extension.write_bytes(b"synthetic-extension")
    input_npz = tmp_path / "input.npz"
    input_npz.write_bytes(b"synthetic-input")
    module = "synthetic.module"
    monkeypatch.setattr(
        worker,
        "SUPPORTED_MODULES",
        {module: (6, "plain_ref6", extension.stat().st_size)},
    )
    request = {
        "module_name": module,
        "extension_path": str(extension),
        "extension_sha256": worker.sha256_file(extension),
        "extension_size_bytes": extension.stat().st_size,
        "input_npz": str(input_npz),
        "input_npz_sha256": worker.sha256_file(input_npz),
        "reference_size": 6,
        "protocol_version": worker.WORKER_PROTOCOL_VERSION,
    }
    assert worker.validate_worker_request(request)["module_name"] == module
    request["extension_size_bytes"] += 1
    with pytest.raises(ValueError, match="size"):
        worker.validate_worker_request(request)


def test_worker_module_identity_rejects_every_attr_hash_and_size_mutation():
    spec = dict(schema.FROZEN_EXTENSIONS[0])
    summary = {
        "extension_sha256": spec["extension_sha256"],
        "extension_size_bytes": spec["extension_size_bytes"],
        "KNOT_POINTS": 64,
        "REFERENCE_SIZE": spec["reference_size"],
        "TOOL_POSITION_FRAME": "arm_right_tool_joint_origin",
        "TOOL_POSITION_SIZE": 3,
        "reference_size": spec["reference_size"],
    }
    assert runner.certify_worker_module_identity(summary, spec)
    for key, value in (
        ("extension_sha256", "0" * 64),
        ("extension_size_bytes", spec["extension_size_bytes"] + 1),
        ("KNOT_POINTS", 63),
        ("REFERENCE_SIZE", 6),
        ("TOOL_POSITION_FRAME", "wrong"),
        ("TOOL_POSITION_SIZE", 4),
        ("reference_size", 6),
    ):
        mutated = dict(summary)
        mutated[key] = value
        assert not runner.certify_worker_module_identity(mutated, spec)


def test_runner_provenance_rejects_head_command_cwd_clean_and_source_mutations():
    provenance = _valid_runner_provenance()
    assert runner.certify_runner_provenance(provenance)
    mutations = []
    for key, value in (
        ("git_head_at_start", "not-a-head"),
        ("git_head_at_end", "c" * 40),
        ("tracked_tree_clean_at_end", False),
        ("cwd", "/tmp/wrong"),
        ("orig_argv", ["python", "wrong"]),
        ("exact_command", "python wrong"),
    ):
        mutated = copy.deepcopy(provenance)
        mutated[key] = value
        mutations.append(mutated)
    mutated = copy.deepcopy(provenance)
    mutated["source_hashes_at_end"]["preflight_runner"]["sha256"] = "c" * 64
    mutations.append(mutated)
    mutated = copy.deepcopy(provenance)
    mutated["source_hashes"]["preflight_runner"]["path"] = "wrong.py"
    mutated["source_hashes_at_start"] = copy.deepcopy(mutated["source_hashes"])
    mutated["source_hashes_at_end"] = copy.deepcopy(mutated["source_hashes"])
    mutations.append(mutated)
    for mutated in mutations:
        assert not runner.certify_runner_provenance(mutated)


def test_v2_sources_never_import_or_load_rejected_v1_namespace():
    sources = "\n".join(
        Path(module.__file__).read_text() for module in (schema, runner, worker)
    )
    assert "multimodal_toll_v4_model_preflight import" not in sources
    assert "/tmp/tiago-tool-center-toll-v4-model-preflight-authorized-once" not in sources
    assert "REJECTED_V1_ARTIFACT_HASHES_REPORT_ONLY" in sources
    assert "rejected_v1_artifact_loads" in sources


def test_v2_preserves_every_v1_scientific_constant_and_pure_certificate():
    constant_names = (
        "EXPECTED_NONWORKER_ARRAY_NAMES",
        "ONE_STEP_DT",
        "DENSE_SUBSTEPS",
        "FK_PIN_CUDA_TOLERANCE_M",
        "Q8_PUBLIC_GOAL_TOLERANCE_M",
        "DENSE_TOOL_POSITION_TOLERANCE_M",
        "ONE_STEP_STATE_L2_TOLERANCE",
        "ONE_STEP_STATE_MAXABS_TOLERANCE",
        "DENSE_STATE_L2_IS_REPORT_ONLY",
        "REFERENCE_SMOKE_MAX_SQP_ITERS",
        "EXPECTED_DIAGNOSTIC_SOLVE_CALLS_PER_MODULE",
        "EXPECTED_SQP_OPTIMIZATION_CALLS",
        "COLLISION_SCOPE",
        "BROAD_ROW_COUNT",
        "BROAD_Q_RESERVE_RAD",
        "BROAD_Q_SPAN_FRACTION",
        "BROAD_VELOCITY_FRACTION",
        "BROAD_CONTROL_FRACTION",
        "BROAD_CODE_MODULUS",
        "BROAD_CODE_HALF_RANGE",
    )
    for name in constant_names:
        assert getattr(schema, name) == getattr(v1_schema, name)
    assert worker.EXPECTED_WORKER_ARRAY_NAMES == v1_worker.EXPECTED_WORKER_ARRAY_NAMES
    base = _base()
    rows, worker_arrays = _rows_and_arrays(base)
    assert runner.certify_model_preflight(base, rows, worker_arrays) == (
        v1_runner.certify_model_preflight(base, rows, worker_arrays)
    )
    assert worker.certify_reference_smoke(
        _smoke_raw(), variant="toll", reference_size=10
    ) == v1_worker.certify_reference_smoke(
        _smoke_raw(), variant="toll", reference_size=10
    )
