import copy
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gato_tiago import multimodal_toll as toll
from gato_tiago import multimodal_toll_oracle_schema as toll_oracle
from bsqp.common import sample_reference
from bsqp.interface import validate_reference_shape
from bsqp.interface import BSQP
from gato_tiago.multimodal_pillar import MODEL_PATH, load_model, tool_position


REPO_ROOT = Path(__file__).resolve().parents[2]


def _reference():
    return toll.TollReference(
        goal_xyz=(0.16, 0.03, 1.05),
        cylinder_xy=(0.08, 0.01),
        physical_radius_m=toll.PHYSICAL_RADIUS_M,
        toll_xy=(0.08, -0.045),
        toll_sigma_m=toll.TOLL_SIGMA_M,
        clearance_margin_m=toll.CLEARANCE_MARGIN_M,
    )


def _fd_gradient(function, point, epsilon=1e-6):
    point = np.asarray(point, dtype=np.float64)
    result = np.empty(point.size)
    for index in range(point.size):
        plus = point.copy()
        minus = point.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        result[index] = (function(plus) - function(minus)) / (2 * epsilon)
    return result


def _semicircle(center, radius, mode, samples=101):
    if mode == "clockwise":
        angles = np.linspace(np.pi, 0.0, samples)
    else:
        angles = np.linspace(-np.pi, 0.0, samples)
    xy = np.asarray(center) + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return np.column_stack([xy, np.zeros(samples)])


def test_frozen_protocol_and_reference_bytes_are_exact():
    assert toll.KNOTS == 64
    assert toll.DT == 0.0125
    assert toll.HORIZON_S == pytest.approx(0.7875)
    assert toll.BUDGET == 16
    assert toll.REFERENCE_SIZE == 10
    assert toll.DEVELOPMENT_TASK_SEEDS == (12000, 12001, 12002, 12003)
    assert toll.HELDOUT_TASK_SEEDS == tuple(range(12100, 12108))
    assert toll.CENTRAL_TASK_SEED == 12104
    assert toll.MU == 20.0 and toll.RHO == 0.01

    row = _reference().as_float32()
    assert row.shape == (10,)
    assert row.dtype == np.float32
    assert np.array_equal(toll.TollReference.from_solver_bytes(row).as_float32(), row)
    batch = toll.reference_batch(_reference())
    assert batch.shape == (16, 640)
    assert batch.dtype == np.float32
    assert all(np.array_equal(candidate.reshape(64, 10)[0], row) for candidate in batch)

    with pytest.raises(ValueError, match=r"shape \(10,\)"):
        toll.validate_reference_row(np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError, match="dtype float32"):
        toll.validate_reference_row(row.astype(np.float64))
    for index, value in ((5, 0.031), (8, 0.021), (9, 0.006)):
        mutated = row.copy()
        mutated[index] = value
        with pytest.raises(ValueError, match="frozen values"):
            toll.validate_reference_row(mutated)


@pytest.mark.parametrize("terminal", [False, True])
def test_workspace_cost_gradient_matches_finite_difference_and_gn_is_psd(terminal):
    reference = _reference()
    point = np.asarray([0.105, -0.012, 1.01])
    result = toll.workspace_cost_gradient_gn(point, reference, terminal=terminal)
    finite = _fd_gradient(
        lambda value: toll.workspace_cost_gradient_gn(value, reference, terminal=terminal)["cost"],
        point,
    )
    np.testing.assert_allclose(result["gradient"], finite, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result["gauss_newton"], result["gauss_newton"].T)
    assert np.linalg.eigvalsh(result["gauss_newton"]).min() >= -1e-10
    if terminal:
        assert result["toll_cost"] == 0.0
    else:
        assert result["toll_cost"] > 0.0


def test_toll_is_cost_only_and_cylinder_uses_physical_radius_plus_margin():
    reference = _reference()
    at_required_margin = np.asarray(reference.cylinder_xy) + [reference.optimizer_radius_m, 0]
    result = toll.workspace_cost_gradient_gn(
        [*at_required_margin, reference.goal_xyz[2]], reference
    )
    assert result["cylinder_cost"] == pytest.approx(0.0, abs=1e-14)
    assert toll.physical_clearance(at_required_margin, reference) == pytest.approx(
        reference.clearance_margin_m
    )

    at_toll = np.asarray(reference.toll_xy)
    assert toll.workspace_cost_gradient_gn(
        [*at_toll, reference.goal_xyz[2]], reference
    )["toll_cost"] == pytest.approx(0.25)
    # Moving the toll cannot change physical collision clearance.
    mutated = copy.deepcopy(reference)
    object.__setattr__(mutated, "toll_xy", (9.0, -9.0))
    probe = np.asarray([0.12, 0.02])
    assert toll.physical_clearance(probe, reference) == toll.physical_clearance(probe, mutated)


def test_initializer_schema_is_obstacle_and_mode_neutral_and_antithetic():
    payload = {
        "robot_model": object(),
        "x0": np.zeros(14),
        "goal_xyz": np.ones(3),
        "limits": {},
        "knots": 64,
        "dt": 0.0125,
        "method_seed": 17,
    }
    assert toll.validate_initializer_inputs(payload)
    for forbidden in sorted(toll.FORBIDDEN_INITIALIZER_FIELDS):
        tainted = dict(payload)
        tainted[forbidden] = object()
        with pytest.raises(ValueError, match="tainted"):
            toll.validate_initializer_inputs(tainted)

    directions = toll.initializer_directions(17)
    assert directions.shape == (16, 7)
    assert np.array_equal(directions[0], np.zeros(7))
    for pair in range(7):
        np.testing.assert_array_equal(directions[1 + 2 * pair], -directions[2 + 2 * pair])
    paths, metadata = toll.planned_initializer_paths(np.zeros(7), np.ones(7) * 0.1, 17)
    assert paths.shape == (16, 64, 7)
    assert len({row.tobytes() for row in paths}) == 16
    assert np.all(paths[:, 0] == 0)
    assert metadata["rejection_attempts"] == 0
    assert metadata["feedback_iterations"] == 0
    assert metadata["candidate_zero"] == "obvious_default_partial_dls_nominal"
    assert "cold" not in metadata["direction_family"]

    jacobian = np.column_stack([np.eye(3), np.zeros((3, 4))])
    dq = toll.initializer_nominal_dq(jacobian, np.zeros(3), np.ones(3) * 0.1)
    expected = 0.1 / (1.0 + toll.DAMPING**2)
    np.testing.assert_allclose(dq[:3], expected)
    np.testing.assert_array_equal(dq[3:], 0.0)


def test_task_and_optimizer_execution_are_fail_closed_without_instantiating_seed():
    assert toll_oracle.TASK_CONSTRUCTION_AUTHORIZATION is None
    assert toll.OPTIMIZER_EXECUTION_AUTHORIZATION is None
    with pytest.raises(RuntimeError, match="blocked"):
        toll_oracle.generate_task(toll.DEVELOPMENT_TASK_SEEDS[0], object())
    for replacement in (11999, 12004, 12108, 99999):
        with pytest.raises(ValueError, match="frozen"):
            toll_oracle.generate_task(replacement, object())


def test_pairwise_topology_requires_all_keyed_same_and_cross_classes():
    center = np.asarray([0.0, 0.0])
    rows = []
    for candidate, (mode, radius) in enumerate(
        (("clockwise", 1.0), ("clockwise", 1.1), ("counterclockwise", 1.0), ("counterclockwise", 1.1))
    ):
        rows.append(
            {
                "candidate_index": candidate,
                "productive_certified": True,
                "tool_path": _semicircle(center, radius, mode),
                "normalized_controls": np.full((8, 7), candidate, dtype=float),
            }
        )
    certificate = toll.topology_certificate(rows, center)
    assert certificate["two_mode_two_support"]
    assert len(certificate["same_mode_pairs"]) == 2
    assert len(certificate["cross_mode_pairs"]) == 4
    assert all(row["nearest_integer"] == 0 for row in certificate["same_mode_pairs"])
    assert all(abs(row["nearest_integer"]) == 1 for row in certificate["cross_mode_pairs"])

    # An extra complete turn has the same open label, but must fail the keyed
    # same-mode class-zero requirement.
    angles = np.linspace(np.pi, -2 * np.pi, 201)
    rows[1]["tool_path"] = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(angles.size)])
    assert not toll.topology_certificate(rows, center)["two_mode_two_support"]


def test_topology_requires_distinct_solutions_and_exact_cuda_pin_keyed_agreement():
    center = np.zeros(2)
    rows = []
    for candidate, (mode, radius) in enumerate(
        (("clockwise", 1.0), ("clockwise", 1.1), ("counterclockwise", 1.0), ("counterclockwise", 1.1))
    ):
        path = _semicircle(center, radius, mode)
        rows.append(
            {
                "candidate_index": candidate,
                "productive_certified": True,
                "cuda_tool_path": path,
                "pin_tool_path": path.copy(),
                "normalized_controls": np.full((8, 7), candidate * 0.01),
            }
        )
    assert toll.model_topology_agreement(rows, center)["all_model_topology_gates_pass"]

    duplicate = copy.deepcopy(rows)
    duplicate[1]["cuda_tool_path"] = duplicate[0]["cuda_tool_path"].copy()
    duplicate[1]["pin_tool_path"] = duplicate[0]["pin_tool_path"].copy()
    duplicate[1]["normalized_controls"] = duplicate[0]["normalized_controls"].copy()
    assert not toll.model_topology_agreement(duplicate, center)["all_model_topology_gates_pass"]

    mismatch = copy.deepcopy(rows)
    angles = np.linspace(np.pi, -2 * np.pi, 201)
    mismatch[1]["pin_tool_path"] = np.column_stack(
        [np.cos(angles), np.sin(angles), np.zeros(angles.size)]
    )
    assert not toll.model_topology_agreement(mismatch, center)["all_model_topology_gates_pass"]

    bad_id = copy.deepcopy(rows)
    bad_id[0]["candidate_index"] = 16
    assert not toll.model_topology_agreement(bad_id, center)["all_model_topology_gates_pass"]
    duplicate_id = copy.deepcopy(rows)
    duplicate_id[1]["candidate_index"] = duplicate_id[0]["candidate_index"]
    assert not toll.model_topology_agreement(duplicate_id, center)["all_model_topology_gates_pass"]

    bad_shape = copy.deepcopy(rows)
    bad_shape[0]["cuda_tool_path"] = bad_shape[0]["cuda_tool_path"][:-1]
    assert not toll.model_topology_agreement(bad_shape, center)["all_model_topology_gates_pass"]
    bad_control_shape = copy.deepcopy(rows)
    bad_control_shape[0]["normalized_controls"] = bad_control_shape[0][
        "normalized_controls"
    ][:-1]
    assert not toll.model_topology_agreement(bad_control_shape, center)[
        "all_model_topology_gates_pass"
    ]

    # This near-duplicate is excluded from the distinct support count, but its
    # bad extra winding must still participate in every-pair topology and fail.
    dropped_bad = copy.deepcopy(rows)
    angles = np.linspace(np.pi, -2 * np.pi, 101)
    dropped_bad.append(
        {
            "candidate_index": 4,
            "productive_certified": True,
            "cuda_tool_path": np.column_stack(
                [np.cos(angles), np.sin(angles), np.zeros(angles.size)]
            ),
            "pin_tool_path": np.column_stack(
                [np.cos(angles), np.sin(angles), np.zeros(angles.size)]
            ),
            "normalized_controls": rows[0]["normalized_controls"].copy(),
        }
    )
    dropped_certificate = toll.model_topology_agreement(dropped_bad, center)
    assert dropped_certificate["cuda"]["classified_productive_counts"]["clockwise"] == 3
    assert len(dropped_certificate["cuda"]["same_mode_pairs"]) == 4
    assert not dropped_certificate["all_model_topology_gates_pass"]


def test_productive_certificate_is_external_strict_and_never_uses_toll_as_collision():
    metrics = {
        "finite": True,
        "cuda_seed_objective": 2.0,
        "cuda_final_objective": 1.5,
        "pin_seed_objective": 2.0,
        "pin_final_objective": 1.5,
        "seed_solver_merit": 3.0,
        "final_solver_merit": 2.0,
        "cuda_seed_terminal_error_m": 0.10,
        "cuda_final_terminal_error_m": 0.012,
        "pin_seed_terminal_error_m": 0.10,
        "pin_final_terminal_error_m": 0.018,
        "cuda_final_tool_speed_mps": 0.04,
        "pin_final_tool_speed_mps": 0.04,
        "cuda_physical_clearance_margin_m": 0.006,
        "pin_physical_clearance_margin_m": 0.006,
        "cuda_joint_ratio": 0.8,
        "pin_joint_ratio": 0.8,
        "cuda_velocity_ratio": 0.8,
        "pin_velocity_ratio": 0.8,
        "control_ratio": 0.8,
        "model_tool_error_m": 0.0005,
        "normalized_control_change": 0.002,
        "tool_path_rms_change_m": 0.006,
        "cuda_initial_collision_violation_m": 0.01,
        "cuda_final_collision_violation_m": 0.0,
        "pin_initial_collision_violation_m": 0.01,
        "pin_final_collision_violation_m": 0.0,
        "pcg_cap_hits": 0,
        "sqp_hit_max": False,
    }
    accepted = toll.certify_productive_replay(metrics)
    assert accepted["productive_certified"]
    assert not accepted["toll_used_as_feasibility"]
    mutations = {
        "cuda_final_terminal_error_m": 0.016,
        "pin_final_terminal_error_m": 0.021,
        "cuda_physical_clearance_margin_m": 0.004,
        "pin_physical_clearance_margin_m": 0.004,
        "control_ratio": 1.01,
        "model_tool_error_m": 0.0011,
        "normalized_control_change": 0.0009,
        "tool_path_rms_change_m": 0.0049,
        "pcg_cap_hits": 1,
        "sqp_hit_max": True,
        "final_solver_merit": 2.8,
    }
    for key, value in mutations.items():
        rejected = dict(metrics)
        rejected[key] = value
        assert not toll.certify_productive_replay(rejected)["productive_certified"], key
    for model in ("cuda", "pin"):
        rejected = dict(metrics)
        rejected[f"{model}_final_objective"] = rejected[f"{model}_seed_objective"]
        assert not toll.certify_productive_replay(rejected)["productive_certified"]
        rejected = dict(metrics)
        rejected[f"{model}_final_collision_violation_m"] = rejected[
            f"{model}_initial_collision_violation_m"
        ]
        assert not toll.certify_productive_replay(rejected)["productive_certified"]

    for key in (
        "cuda_seed_terminal_error_m",
        "pin_final_terminal_error_m",
        "cuda_joint_ratio",
        "pin_velocity_ratio",
        "normalized_control_change",
        "model_tool_error_m",
        "pcg_cap_hits",
        "cuda_initial_collision_violation_m",
    ):
        negative = dict(metrics)
        negative[key] = -1e-6
        assert not toll.certify_productive_replay(negative)["productive_certified"], key
        nonfinite = dict(metrics)
        nonfinite[key] = np.nan
        assert not toll.certify_productive_replay(nonfinite)["productive_certified"], key

    # A small-scale cost uses 10% of |seed|, not 10% of one.
    small = dict(metrics)
    small.update(
        cuda_seed_objective=0.05,
        cuda_final_objective=0.039,
        pin_seed_objective=0.05,
        pin_final_objective=0.039,
        seed_solver_merit=0.05,
        final_solver_merit=0.039,
    )
    assert toll.certify_productive_replay(small)["productive_certified"]


def test_general_reference_width_static_wiring_preserves_pose_width_and_old_plants():
    constants = (REPO_ROOT / "gato/constants.h").read_text()
    linalg = (REPO_ROOT / "gato/utils/linalg.cuh").read_text()
    setup = (REPO_ROOT / "gato/bsqp/kernels/setup_kkt.cuh").read_text()
    merit = (REPO_ROOT / "gato/bsqp/kernels/merit.cuh").read_text()
    bindings = (REPO_ROOT / "python/bindings.cu").read_text()
    tiago = (REPO_ROOT / "gato/dynamics/tiago_right/tiago_right_plant.cuh").read_text()
    indy = (REPO_ROOT / "gato/dynamics/indy7/indy7_plant.cuh").read_text()
    iiwa = (REPO_ROOT / "gato/dynamics/iiwa14/iiwa14_plant.cuh").read_text()

    assert "grid::REFERENCE_SIZE * KNOT_POINTS" in constants
    assert "knot_idx * grid::REFERENCE_SIZE" in linalg
    assert "2 * grid::REFERENCE_SIZE" in setup
    assert "block::copy<T, grid::REFERENCE_SIZE>" in merit
    assert 'm.attr("REFERENCE_SIZE") = grid::REFERENCE_SIZE' in bindings
    assert "ref_buf.ndim" in bindings and "reference input must have shape" in bindings
    assert "constexpr int REFERENCE_SIZE = 10" in tiago
    assert "constexpr int REFERENCE_SIZE = EE_POS_SIZE" in indy
    assert "constexpr int REFERENCE_SIZE = EE_POS_SIZE" in iiwa
    # FK pose workspace remains generated six-wide.
    assert "constexpr int EE_POS_SIZE = 6" in tiago
    assert "s_eePos_grad + 6 * grid::NUM_JOINTS" in tiago

    assert validate_reference_shape(np.zeros((2, 24)), 2, 4, 6, "indy7") == (2, 24)
    assert validate_reference_shape(np.zeros((2, 40)), 2, 4, 10, "tiago_toll") == (2, 40)
    with pytest.raises(ValueError, match=r"shape \(2, 40\)"):
        validate_reference_shape(np.zeros((2, 24)), 2, 4, 10, "tiago_toll")
    with pytest.raises(ValueError, match=r"shape \(2, 24\)"):
        validate_reference_shape(np.zeros((2, 40)), 2, 4, 6, "indy7")
    reference10 = np.arange(30, dtype=np.float64).reshape(3, 10)
    sampled10 = sample_reference(
        reference10, np.asarray([0.0, 1.0]), 1.0, reference_size=10
    )
    np.testing.assert_array_equal(sampled10, reference10[[0, 1]])


def test_all_tiago_variants_use_exact_tool_frame():
    model = load_model(MODEL_PATH)
    q = np.asarray([-0.39, -1.73, -0.38, -2.35, 0.0, -1.21, 0.04])
    expected = tool_position(model, model.createData(), q)
    for variant in ("tiago_right", "tiago_right_multimodal", "tiago_right_multimodal_toll"):
        solver = BSQP.__new__(BSQP)
        solver.model = model
        solver.data = model.createData()
        solver.plant_type = variant
        np.testing.assert_allclose(solver.ee_pos(q), expected, atol=1e-12)


def test_toll_plant_math_and_build_are_explicit_and_generated_files_untouched():
    plant = (REPO_ROOT / "gato/dynamics/tiago_right/tiago_right_plant.cuh").read_text()
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    assert "TIAGO_MULTIMODAL_TOLL" in plant
    assert "s_reference[5] + s_reference[9]" in plant
    assert "s_reference[6]" in plant and "s_reference[8]" in plant
    assert "return static_cast<T>(0.5);" in plant
    assert "blockIdx.x == KNOT_POINTS - 1" in plant
    assert plant.count("if (computeR)") >= 2
    assert "tiago_right_multimodal_toll" in cmake
    assert "TIAGO_MULTIMODAL_TOLL=1" in cmake
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--", "*grid.cuh"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert changed == []


def test_stage_clis_and_import_boundary_are_fail_closed(tmp_path):
    stage0 = (REPO_ROOT / "tiago_examples/tiago_multimodal_toll_stage0.py").read_text()
    benchmark = (REPO_ROOT / "tiago_examples/tiago_multimodal_toll_benchmark.py").read_text()
    module = (REPO_ROOT / "tiago_src/gato_tiago/multimodal_toll.py").read_text()
    oracle_module = (
        REPO_ROOT / "tiago_src/gato_tiago/multimodal_toll_oracle_schema.py"
    ).read_text()
    assert "STAGE0_EXECUTION_AUTHORIZATION = None" in stage0
    assert "BENCHMARK_EXECUTION_AUTHORIZATION = None" in benchmark
    assert "blocked pending verifier authorization" in stage0
    assert "blocked pending verifier authorization" in benchmark
    forbidden_import_fragments = ("stage0", "oracle", "tangent", "route_template")
    imports = [line for line in (module + "\n" + benchmark).splitlines() if line.startswith(("import ", "from "))]
    assert not any(fragment in line.lower() for fragment in forbidden_import_fragments for line in imports)
    assert "TASK_CONSTRUCTION_AUTHORIZATION = None" in oracle_module
    assert "q_goal" not in toll.TollTask.__dataclass_fields__
    assert "dq" not in toll.TollTask.__dataclass_fields__
    assert "construction_metadata" not in toll.TollTask.__dataclass_fields__
    with pytest.raises(ValueError, match="tainted"):
        toll.validate_initializer_inputs(
            {
                "robot_model": object(),
                "x0": np.zeros(14),
                "goal_xyz": np.ones(3),
                "limits": {},
                "knots": 64,
                "dt": 0.0125,
                "method_seed": 17,
                "construction_witness": object(),
            }
        )
    base_payload = {
        "robot_model": object(),
        "x0": np.zeros(14),
        "goal_xyz": np.ones(3),
        "limits": {},
        "knots": 64,
        "dt": 0.0125,
        "method_seed": 17,
    }
    for secret in ("default_side", "default_label"):
        with pytest.raises(ValueError, match="tainted"):
            toll.validate_initializer_inputs({**base_payload, secret: 1})
    assert toll.frozen_protocol_metadata()["collision_scope"] == "tool_center_only"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "tiago_src")
    for script in (
        REPO_ROOT / "tiago_examples/tiago_multimodal_toll_stage0.py",
        REPO_ROOT / "tiago_examples/tiago_multimodal_toll_benchmark.py",
    ):
        described = subprocess.run(
            ["python", "-B", str(script), "--describe"],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert '"task_instantiation_authorized": false' in described.stdout
        blocked = subprocess.run(
            ["python", "-B", str(script), "--execute", "--output", "/tmp/must-not-exist.json"],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert blocked.returncode != 0
        assert "blocked pending verifier authorization" in blocked.stderr
        assert not Path("/tmp/must-not-exist.json").exists()

    source = tmp_path / "source.py"
    source.write_bytes(b"frozen\n")
    first = toll.source_hashes({"schema": source})
    source.write_bytes(b"mutated\n")
    second = toll.source_hashes({"schema": source})
    assert first["schema"]["sha256"] != second["schema"]["sha256"]


def test_frozen_task_rng_order_and_no_retry_are_source_visible_without_seed_use():
    source = (
        REPO_ROOT / "tiago_src/gato_tiago/multimodal_toll_oracle_schema.py"
    ).read_text()
    jitter = source.index("rng.uniform(-Q0_JITTER_RAD, Q0_JITTER_RAD, size=7)")
    phi = source.index("rng.uniform(-PLANAR_ANGLE_RAD, PLANAR_ANGLE_RAD)")
    sign = source.index("rng.choice((-1, 1))")
    assert jitter < phi < sign
    generate_body = source[source.index("def generate_task"):]
    assert "while " not in generate_body
    assert "retry" not in generate_body.lower()
    assert "clip" not in generate_body
    assert "DAMPING**2" in source
    assert "np.linalg.norm(goal - start)" in source
    assert "np.linalg.norm(goal[:2] - start[:2])" in source
    assert "chord_intersects_physical_cylinder" in source
    assert "chord_intersects_optimizer_keepout" in source
