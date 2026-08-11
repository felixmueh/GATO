import ast
import copy
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from gato_tiago import multimodal_toll as toll
from gato_tiago import multimodal_toll_v0 as v0


REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_TASK_IDENTITIES = tuple(
    [("synthetic_development", 91000 + index) for index in range(4)]
    + [("synthetic_heldout", 92000 + index) for index in range(8)]
)


def _limits():
    return (
        np.full(7, -2.0, dtype=np.float64),
        np.full(7, 2.0, dtype=np.float64),
        np.arange(1.0, 8.0, dtype=np.float64),
        np.zeros(7, dtype=np.float64),
    )


def _draws():
    return v0.generate_v0_draws(*_limits())


def _task_row(phase, seed):
    rng = np.random.default_rng(seed)
    q0_jitter = rng.uniform(-toll.Q0_JITTER_RAD, toll.Q0_JITTER_RAD, size=7)
    phi = float(rng.uniform(-toll.PLANAR_ANGLE_RAD, toll.PLANAR_ANGLE_RAD))
    offset_sign = int(rng.choice((-1, 1)))
    q0 = q0_jitter.copy()
    x0 = np.concatenate([q0, np.zeros(7)]).astype(np.float32)
    jacobian = np.column_stack([np.eye(3), np.zeros((3, 4))])
    requested_delta = toll.REQUESTED_TRAVEL_M * np.asarray(
        [np.cos(phi), np.sin(phi), 0.0]
    )
    dq = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + toll.DAMPING**2 * np.eye(3), requested_delta
    )
    q_goal64 = q0 + dq
    q_goal = q_goal64.astype(np.float32)
    start = np.asarray([-0.075, 0.0, 1.0], dtype=np.float64)
    goal = np.asarray([0.075, 0.0, 1.0], dtype=np.float64)
    direction = (goal[:2] - start[:2]) / np.linalg.norm(goal[:2] - start[:2])
    normal = np.asarray([-direction[1], direction[0]])
    cylinder = 0.5 * (start[:2] + goal[:2]) + offset_sign * 0.008 * normal
    default_side = -offset_sign
    toll_xy = cylinder + default_side * toll.TOLL_OFFSET_FROM_CYLINDER_M * normal
    reference = toll.TollReference(
        goal_xyz=tuple(goal),
        cylinder_xy=tuple(cylinder),
        physical_radius_m=toll.PHYSICAL_RADIUS_M,
        toll_xy=tuple(toll_xy),
        toll_sigma_m=toll.TOLL_SIGMA_M,
        clearance_margin_m=toll.CLEARANCE_MARGIN_M,
    ).as_float32()
    return {
        "phase": phase,
        "task_seed": seed,
        "solver_x0": x0,
        "solver_reference": reference,
        "q_goal_float32": q_goal,
        "q_goal_float64": q_goal64,
        "dq_float64": dq,
        "requested_delta_xyz_float64": requested_delta,
        "pin_position_jacobian_float64": jacobian,
        "q0_jitter_float64": q0_jitter,
        "phi_float64": phi,
        "offset_sign": offset_sign,
        "default_side": default_side,
        "pin_start_tool_xyz": start,
        "cuda_start_tool_xyz": start.astype(np.float32),
        "pin_q_goal_tool_xyz": goal,
        "cuda_q_goal_tool_xyz": goal.astype(np.float32),
        "solver_x0_sha256": v0._array_hash(x0),
        "solver_reference_sha256": v0._array_hash(reference),
        "q_goal_sha256": v0._array_hash(q_goal),
        "construction_draws_sha256": v0._array_hash(
            np.concatenate([q0_jitter, [phi, float(offset_sign)]])
        ),
        "retry_count": 0,
        "replacement_count": 0,
    }


def _task_rows():
    return [_task_row(phase, seed) for phase, seed in SYNTHETIC_TASK_IDENTITIES]


def _provenance():
    return {
        "tracked_tree_clean_at_start": True,
        "tracked_tree_clean_at_end": True,
        "git_head_at_start": "1" * 64,
        "git_head_at_end": "1" * 64,
        "model_sha256": "2" * 64,
        "model_path": str(toll.MODEL_PATH),
        "extension_sha256": "3" * 64,
        "exact_command": "python -B tiago_examples/tiago_multimodal_toll_stage0.py --execute",
        "full_git_status_at_start": ["?? felixnotes", "?? felixtools"],
        "full_git_status_at_end": ["?? felixnotes", "?? felixtools"],
        "python_version": "3.10-test",
        "numpy_version": "test",
        "pinocchio_version": "test",
        "retry_count": 0,
        "replacement_count": 0,
        "source_hashes": {
            label: {
                "path": path,
                "sha256": hashlib.sha256(label.encode()).hexdigest(),
            }
            for label, path in sorted(v0.REQUIRED_SOURCE_PATHS.items())
        },
    }


def _extension_manifest():
    manifest = copy.deepcopy(v0.EXPECTED_EXTENSION_MANIFEST)
    for index, (module_name, row) in enumerate(manifest.items()):
        row["extension_sha256"] = str(index + 3) * 64
        row["import_smoke_pass"] = True
        row["native_reference_shape_accepted"] = True
        row["wrong_reference_shape_rejected"] = True
        row["accepted_reference_width"] = row["reference_size"]
        row["rejected_reference_width"] = 6 if row["reference_size"] == 10 else 10
        row["b1_reference_cost_output_parity"] = True
        row["b16_reference_cost_output_parity"] = True
        row["b1_broadcast_fk_parity"] = True
        row["b16_broadcast_fk_parity"] = True
        row["b16_per_lane_fk_parity"] = True
        row["pin_fk_max_error_m"] = 5e-5
        row["cuda_reference_smoke_pass"] = True
        row["build_command"] = "./tools/build.sh --frozen"
        row["test_command"] = "python -B frozen_smoke.py"
        row["extension_path"] = f"build/{module_name}.so"
        row["source_commit"] = "1" * 64
    return manifest


def _certificate(**overrides):
    draws = _draws()
    pin_fk = np.arange(96, dtype=np.float64).reshape(32, 3) * 1e-3
    pin_qdd = np.zeros((32, 7), dtype=np.float64)
    pin_next = np.stack(
        [
            v0.constant_acceleration_step(
                np.concatenate(
                    [draws.dynamics_q_float64[index], draws.dynamics_qd_float64[index]]
                ),
                pin_qdd[index],
            )
            for index in range(32)
        ]
    )
    kwargs = {
        "draws": draws,
        "captured_fk_q_float32": draws.fk_q_float32.copy(),
        "captured_dynamics_x_float32": draws.dynamics_x_float32.copy(),
        "captured_dynamics_u_float32": draws.dynamics_u_float32.copy(),
        "pin_fk_positions": pin_fk,
        "cuda_fk_positions": pin_fk.astype(np.float32),
        "pin_qdd_float64": pin_qdd,
        "pin_next_states": pin_next,
        "cuda_next_states": pin_next.astype(np.float32),
        "task_rows": _task_rows(),
        "extension_manifest": _extension_manifest(),
        "provenance": _provenance(),
    }
    kwargs.update(overrides)
    frozen_ledger = v0.EXPECTED_TASK_IDENTITIES
    try:
        # Static tests exercise the exact production logic with an explicitly
        # synthetic ledger; frozen task RNG streams remain unopened.
        v0.EXPECTED_TASK_IDENTITIES = SYNTHETIC_TASK_IDENTITIES
        return v0.certify_v0(**kwargs)
    finally:
        v0.EXPECTED_TASK_IDENTITIES = frozen_ledger


def test_v0_draws_follow_exact_rng_order_and_preserve_float64_and_float32_bytes():
    lower, upper, effort, comfortable = _limits()
    draws = v0.generate_v0_draws(lower, upper, effort, comfortable)
    rng = np.random.default_rng(20260811)
    expected_fk = rng.uniform(lower + 0.08, upper - 0.08, size=(32, 7))
    expected_offsets = rng.uniform(-0.05, 0.05, size=(32, 7))
    expected_qd = rng.uniform(-0.25, 0.25, size=(32, 7))
    expected_fractions = rng.uniform(-0.20, 0.20, size=(32, 7))

    np.testing.assert_array_equal(draws.fk_q_float64, expected_fk)
    np.testing.assert_array_equal(draws.dynamics_q_offsets_float64, expected_offsets)
    np.testing.assert_array_equal(draws.dynamics_q_float64, comfortable + expected_offsets)
    np.testing.assert_array_equal(draws.dynamics_qd_float64, expected_qd)
    np.testing.assert_array_equal(draws.dynamics_control_fractions_float64, expected_fractions)
    np.testing.assert_array_equal(draws.dynamics_u_float64, expected_fractions * effort)
    np.testing.assert_array_equal(draws.fk_q_float32, expected_fk.astype(np.float32))
    assert draws.fk_q_float64.dtype == np.float64
    assert draws.fk_q_float32.dtype == np.float32
    assert set(draws.array_hashes()) == set(draws.__dict__)


def test_v0_draws_fail_instead_of_clipping_or_replacing_invalid_model_inputs():
    lower, upper, effort, comfortable = _limits()
    bad_comfortable = comfortable.copy()
    bad_comfortable[0] = upper[0] - 0.09
    with pytest.raises(ValueError, match="no clipping|violates frozen joint margins"):
        v0.generate_v0_draws(lower, upper, effort, bad_comfortable)
    with pytest.raises(ValueError, match="0.08-rad margin"):
        v0.generate_v0_draws(lower, lower + 0.15, effort, comfortable)
    with pytest.raises(ValueError, match="positive"):
        v0.generate_v0_draws(lower, upper, np.zeros(7), comfortable)


def test_v0_certificate_accepts_only_exact_inputs_models_tasks_and_manifests():
    result = _certificate()
    assert result["all_v0_gates_pass"]
    assert result["fk_sample_count"] == 32
    assert result["optimization_solve_calls"] == 0
    assert result["oracle_solve_calls"] == 0
    assert len(result["task_gates"]) == 12
    assert len(result["retained_array_hashes"]) == 8
    assert len(result["task_solver_byte_hashes"]) == 12
    assert result["task_feasibility_independent_of_toll_cost"]
    assert result["extension_and_ref6_manifest_pass"]
    assert result["collision_scope"] == "tool_center_only"
    assert all(
        row["exact_task_rng_draws"]
        and row["exact_dls_witness"]
        and row["exact_reconstructed_reference"]
        and row["direct_chord_intersects_physical_cylinder"]
        and row["direct_chord_intersects_inflated_keepout"]
        for row in result["task_gates"]
    )

    draws = _draws()
    fabricated = replace(draws, fk_q_float64=draws.fk_q_float64 + 0.01)
    assert not _certificate(draws=fabricated)["draws_independently_regenerated"]

    changed = draws.fk_q_float32.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(np.inf))
    assert not _certificate(captured_fk_q_float32=changed)["all_v0_gates_pass"]

    pin_fk = np.zeros((32, 3), dtype=np.float64)
    cuda_fk = np.zeros((32, 3), dtype=np.float32)
    cuda_fk[3, 1] = 1.01e-4
    assert not _certificate(pin_fk_positions=pin_fk, cuda_fk_positions=cuda_fk)[
        "all_fk_position_gates_pass"
    ]

    pin_next = np.zeros((32, 14), dtype=np.float64)
    cuda_next = np.zeros((32, 14), dtype=np.float32)
    cuda_next[7, 2] = 1.01e-3
    assert not _certificate(pin_next_states=pin_next, cuda_next_states=cuda_next)[
        "all_one_step_model_gates_pass"
    ]


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_task",
        "wrong_seed",
        "retry",
        "rng_draw",
        "dls",
        "reference",
        "q_goal",
        "frame",
        "chord",
        "hash",
    ),
)
def test_v0_task_preflight_is_fail_closed(mutation):
    rows = _task_rows()
    if mutation == "missing_task":
        rows.pop()
    elif mutation == "wrong_seed":
        rows[0]["task_seed"] = 999
    elif mutation == "retry":
        rows[0]["retry_count"] = 1
    elif mutation == "rng_draw":
        rows[0]["q0_jitter_float64"][0] += 1e-8
        rows[0]["construction_draws_sha256"] = v0._array_hash(
            np.concatenate(
                [
                    rows[0]["q0_jitter_float64"],
                    [rows[0]["phi_float64"], float(rows[0]["offset_sign"])],
                ]
            )
        )
    elif mutation == "dls":
        rows[0]["dq_float64"][0] += 1e-8
    elif mutation == "reference":
        rows[0]["solver_reference"][6] += np.float32(1e-4)
        rows[0]["solver_reference_sha256"] = v0._array_hash(
            rows[0]["solver_reference"]
        )
    elif mutation == "q_goal":
        rows[0]["cuda_q_goal_tool_xyz"][0] += np.float32(0.001)
    elif mutation == "frame":
        rows[0]["cuda_start_tool_xyz"][1] += np.float32(0.001)
    elif mutation == "chord":
        reference = rows[0]["solver_reference"].copy()
        reference[3:5] = (0.0, 0.2)
        rows[0]["solver_reference"] = reference
        rows[0]["solver_reference_sha256"] = v0._array_hash(reference)
    elif mutation == "hash":
        rows[0]["solver_x0_sha256"] = "0" * 64
    assert not _certificate(task_rows=rows)["all_v0_gates_pass"]


def test_v0_provenance_and_ref6_preservation_manifest_are_hard_gates():
    manifest = _extension_manifest()
    assert manifest["bsqpN64_tiago_right_multimodal_toll"]["reference_size"] == 10
    assert manifest["bsqpN64_tiago_right"]["reference_size"] == 6
    assert manifest["bsqpN64_tiago_right_multimodal"]["reference_size"] == 6
    assert all(
        row["tool_position_frame"] == "arm_right_tool_joint_origin"
        for row in manifest.values()
    )
    manifest["bsqpN64_tiago_right"]["reference_size"] = 10
    assert not _certificate(extension_manifest=manifest)["all_v0_gates_pass"]
    manifest = _extension_manifest()
    manifest["bsqpN64_tiago_right"]["extension_sha256"] = "bad"
    assert not _certificate(extension_manifest=manifest)["all_v0_gates_pass"]

    provenance = _provenance()
    provenance["tracked_tree_clean_at_end"] = False
    assert not _certificate(provenance=provenance)["all_v0_gates_pass"]
    provenance = _provenance()
    provenance["git_head_at_end"] = "9" * 64
    assert not _certificate(provenance=provenance)["all_v0_gates_pass"]
    provenance = _provenance()
    provenance["source_hashes"].pop("tool_position_kernel")
    assert not _certificate(provenance=provenance)["all_v0_gates_pass"]


def test_independent_pin_one_step_uses_one_aba_and_frozen_constant_acceleration_formula():
    calls = []

    def aba(q, qd, u):
        calls.append((q.copy(), qd.copy(), u.copy()))
        return np.arange(1.0, 8.0)

    x = np.arange(14, dtype=np.float64) * 0.01
    u = np.arange(7, dtype=np.float64)
    result = v0.pin_one_step_from_aba(aba, x, u)
    acceleration = np.arange(1.0, 8.0)
    expected = np.concatenate(
        [
            x[:7] + 0.0125 * x[7:] + 0.5 * 0.0125**2 * acceleration,
            x[7:] + 0.0125 * acceleration,
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert len(calls) == 1
    metadata = v0.frozen_v0_metadata()["independent_pin_one_step"]
    assert metadata["external_wrench"] == "zero"
    assert "pinocchio.aba" in metadata["acceleration"]


def _dense_certificate(**overrides):
    sample_count = v0.DENSE_SAMPLE_COUNT
    goal = np.asarray([0.10, 0.0, 1.0])
    reference = toll.TollReference(
        goal_xyz=tuple(goal),
        cylinder_xy=(0.0, 1.0),
        physical_radius_m=toll.PHYSICAL_RADIUS_M,
        toll_xy=(0.0, 1.055),
        toll_sigma_m=toll.TOLL_SIGMA_M,
        clearance_margin_m=toll.CLEARANCE_MARGIN_M,
    ).as_float32()
    pin_states = np.zeros((sample_count, 14), dtype=np.float64)
    pin_tool = np.tile(goal, (sample_count, 1)).astype(np.float64)
    kwargs = {
        "common_x0_float32": np.zeros(14, dtype=np.float32),
        "coarse_controls_float32": np.zeros((toll.KNOTS - 1, 7), dtype=np.float32),
        "dense_controls_float32": np.zeros((sample_count - 1, 7), dtype=np.float32),
        "dense_time_float64": np.arange(sample_count, dtype=np.float64)
        * (toll.DT / v0.DENSE_SUBSTEPS),
        "cuda_states_float32": pin_states.astype(np.float32),
        "pin_states_float64": pin_states,
        "cuda_tool_xyz_float32": pin_tool.astype(np.float32),
        "pin_tool_xyz_float64": pin_tool,
        "reference_float32": reference,
        "lower_float64": np.full(7, -2.0, dtype=np.float64),
        "upper_float64": np.full(7, 2.0, dtype=np.float64),
        "velocity_limit_float64": np.ones(7, dtype=np.float64),
        "effort_limit_float64": np.ones(7, dtype=np.float64),
    }
    kwargs.update(overrides)
    return v0.certify_dense_model_replay(**kwargs)


def test_dense_model_certificate_gates_each_model_and_keeps_state_l2_report_only():
    result = _dense_certificate()
    assert result["all_dense_model_gates_pass"]
    assert result["cuda"]["all_model_gates_pass"]
    assert result["pin"]["all_model_gates_pass"]
    assert result["state_l2_is_acceptance_gate"] is False
    assert result["dense_substeps_per_interval"] == 64
    assert result["collision_scope"] == "tool_center_only"
    assert len(result["retained_array_hashes"]) == 8

    pin_states = np.zeros((v0.DENSE_SAMPLE_COUNT, 14), dtype=np.float64)
    pin_states[1, 0] = 0.01
    report_only = _dense_certificate(pin_states_float64=pin_states)
    assert report_only["max_dense_state_l2_report_only"] == pytest.approx(0.01)
    assert report_only["all_dense_model_gates_pass"]

    pin_tool = np.tile(
        [0.10, 0.0, 1.0], (v0.DENSE_SAMPLE_COUNT, 1)
    ).astype(np.float64)
    pin_tool[1, 0] += 0.00101
    assert not _dense_certificate(pin_tool_xyz_float64=pin_tool)[
        "all_dense_model_gates_pass"
    ]
    controls = np.zeros((v0.DENSE_SAMPLE_COUNT - 1, 7), dtype=np.float32)
    controls[0, 0] = 1.01
    assert not _dense_certificate(dense_controls_float32=controls)[
        "all_dense_model_gates_pass"
    ]
    cuda_tool = np.tile(
        [0.10, 0.0, 1.0], (v0.DENSE_SAMPLE_COUNT, 1)
    ).astype(np.float32)
    cuda_tool[-1, 0] += np.float32(0.02)
    assert not _dense_certificate(cuda_tool_xyz_float32=cuda_tool)[
        "all_dense_model_gates_pass"
    ]


def test_execution_aborts_before_model_extension_task_or_draw_generation():
    calls = []
    with pytest.raises(RuntimeError, match="blocked"):
        v0.execute_v0(
            model_factory=lambda: calls.append("model"),
            extension_factory=lambda: calls.append("extension"),
        )
    assert calls == []
    assert v0.frozen_v0_metadata()["task_rows_status"] == "identity_only_not_instantiated"

    source = (REPO_ROOT / "tiago_src/gato_tiago/multimodal_toll_v0.py").read_text()
    imports = [
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    ]
    assert "gato_tiago.multimodal_toll_oracle_schema" not in imports
    assert "generate_task" not in source
    assert "solver.solve(" not in source


def test_tiago_tool_position_binding_is_read_only_batched_exact_frame_api():
    kernel = (REPO_ROOT / "gato/bsqp/kernels/tool_position.cuh").read_text()
    bindings = (REPO_ROOT / "python/bindings.cu").read_text()
    solver = (REPO_ROOT / "gato/bsqp/bsqp.cuh").read_text()

    assert "computeTiagoToolPose<T>" in kernel
    assert "tiagoJointSign<T>(joint)" in kernel
    assert "grid::EE_POS_DYNAMIC_SHARED_MEM_COUNT" in kernel
    assert "const T* d_q_batch" in kernel
    assert "void tool_position(T* d_positions, const T* d_q_batch)" in solver
    assert "This is read-only" in solver
    assert "CArray<T> q" in bindings
    assert 'q must have shape (NQ,) or (BatchSize, NQ)' in bindings
    assert '#if defined(PLANT_TIAGO_RIGHT)' in bindings
    assert '.def("tool_position"' in bindings
    assert 'm.attr("TOOL_POSITION_FRAME") = "arm_right_tool_joint_origin"' in bindings
    assert "const std::vector<py::ssize_t> shape" in bindings
    assert "static_cast<py::ssize_t>(BatchSize)" in bindings
    assert "static_cast<py::ssize_t>(3)" in bindings


def test_static_checkpoint_does_not_instantiate_frozen_tasks_or_touch_generated_grid():
    metadata = v0.frozen_v0_metadata()
    assert metadata["expected_task_identities"] == [
        list(row) for row in v0.EXPECTED_TASK_IDENTITIES
    ]
    assert metadata["execution_authorized"] is False
    changed = set(
        __import__("subprocess")
        .check_output(["git", "diff", "--name-only"], cwd=REPO_ROOT, text=True)
        .splitlines()
    )
    assert not any(path.endswith("_grid.cuh") for path in changed)


def test_static_certificate_never_opens_frozen_task_rng_or_oracle(monkeypatch):
    from gato_tiago import multimodal_toll_oracle_schema as oracle

    frozen = set(toll.DEVELOPMENT_TASK_SEEDS + toll.HELDOUT_TASK_SEEDS)
    observed_rng_seeds = []
    oracle_calls = []
    real_default_rng = np.random.default_rng

    def guarded_rng(seed=None, *args, **kwargs):
        assert seed not in frozen
        observed_rng_seeds.append(seed)
        return real_default_rng(seed, *args, **kwargs)

    def forbidden_generate_task(*args, **kwargs):
        oracle_calls.append((args, kwargs))
        raise AssertionError("static V0 certificate must not call generate_task")

    monkeypatch.setattr(np.random, "default_rng", guarded_rng)
    monkeypatch.setattr(oracle, "generate_task", forbidden_generate_task)
    assert _certificate()["all_v0_gates_pass"]
    assert frozen.isdisjoint(observed_rng_seeds)
    assert set(seed for seed in observed_rng_seeds if seed != v0.V0_RANDOM_SEED) == {
        seed for _, seed in SYNTHETIC_TASK_IDENTITIES
    }
    assert oracle_calls == []

    source = Path(__file__).read_text()
    tree = ast.parse(source)
    literal_rng_seeds = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "default_rng"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, int)
    }
    assert frozen.isdisjoint(literal_rng_seeds)
    assert not frozen.intersection(seed for _, seed in SYNTHETIC_TASK_IDENTITIES)
