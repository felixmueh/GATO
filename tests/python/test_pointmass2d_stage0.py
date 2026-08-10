from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "pointmass_src"),
]

from gato_pointmass.pointmass2d import (
    BUDGET,
    CENTRAL_HELDOUT_TASK_SEED,
    DEVELOPMENT_METHOD_SEEDS,
    DEVELOPMENT_TASK_SEEDS,
    HELDOUT_METHOD_SEEDS,
    HELDOUT_TASK_SEEDS,
    PROTOCOL,
    affine_state_map,
    generate_obstacle_and_mode_neutral_seeds,
    generate_task,
    minimum_energy_controls,
    objective_control_derivatives,
    objective_cost,
    obstacle_residual,
    rollout_controls,
    sha256_array,
    solver_task_identity_hash,
    terminal_nullspace,
    terminal_target,
)
from pointmass_examples.pointmass2d_stage0_oracle import (
    MODE_SIDES,
    ORACLE_RESTART_MARGINS_M,
    TRUST_CONSTR_OPTIONS,
    Stage0Problem,
    task_aggregate,
)


def _finite_difference_gradient(function, point, epsilon=1e-6):
    gradient = np.empty_like(point)
    for index in range(len(point)):
        step = np.zeros_like(point)
        step[index] = epsilon
        gradient[index] = (function(point + step) - function(point - step)) / (
            2.0 * epsilon
        )
    return gradient


def test_frozen_protocol_and_seed_split_are_exact():
    assert PROTOCOL.knots == 32 and PROTOCOL.dt == 0.05
    assert PROTOCOL.position_limit == 0.60
    assert PROTOCOL.velocity_limit == 2.0
    assert PROTOCOL.acceleration_limit == 4.0
    assert PROTOCOL.running_position_weight == 0.1
    assert PROTOCOL.running_velocity_weight == 0.05
    assert PROTOCOL.running_control_weight == 0.01
    assert PROTOCOL.running_obstacle_weight == 100.0
    assert PROTOCOL.terminal_position_weight == 200.0
    assert PROTOCOL.terminal_velocity_weight == 20.0
    assert PROTOCOL.terminal_obstacle_weight == 100.0
    assert PROTOCOL.clearance_margin == 0.02
    assert PROTOCOL.seed_endpoint_fraction == 0.60
    assert PROTOCOL.seed_perturbation_fraction_of_acceleration_limit == 0.10
    assert PROTOCOL.dense_substeps == 16
    assert DEVELOPMENT_TASK_SEEDS == (7000, 7001, 7002)
    assert DEVELOPMENT_METHOD_SEEDS == (8000, 8001, 8002, 8003, 8004)
    assert HELDOUT_TASK_SEEDS == (7100, 7101, 7102, 7103, 7104)
    assert HELDOUT_METHOD_SEEDS == tuple(range(8100, 8120))
    assert CENTRAL_HELDOUT_TASK_SEED == 7102


def test_development_task_generator_is_deterministic_no_retry_and_bounded():
    task_identity_hashes = set()
    for task_seed in DEVELOPMENT_TASK_SEEDS:
        first = generate_task(task_seed)
        second = generate_task(task_seed)
        for key in (
            "direction",
            "normal",
            "start_position_m",
            "goal_position_m",
            "disk_center_m",
            "x0",
            "reference",
        ):
            np.testing.assert_array_equal(first[key], second[key])
        assert first["rejection_or_retry_count"] == 0
        assert first["x0"].dtype == np.float32
        assert first["reference"].dtype == np.float32
        assert first["x0"].shape == (4,)
        assert first["reference"].shape == (6,)
        np.testing.assert_array_equal(
            first["start_position_m"], first["x0"][:2].astype(np.float64)
        )
        np.testing.assert_array_equal(
            first["goal_position_m"], first["reference"][:2].astype(np.float64)
        )
        np.testing.assert_array_equal(
            first["disk_center_m"], first["reference"][2:4].astype(np.float64)
        )
        assert first["disk_radius_m"] == float(first["reference"][4])
        assert first["solver_x0_sha256"] == sha256_array(first["x0"])
        assert first["solver_reference_sha256"] == sha256_array(
            first["reference"]
        )
        assert first["solver_task_identity_sha256"] == solver_task_identity_hash(
            first["x0"], first["reference"]
        )
        task_identity_hashes.add(first["solver_task_identity_sha256"])
        mutated_reference = first["reference"].copy()
        mutated_reference.view(np.uint32)[0] ^= np.uint32(1)
        assert solver_task_identity_hash(first["x0"], mutated_reference) != first[
            "solver_task_identity_sha256"
        ]
        travel = first["goal_position_m"] - first["start_position_m"]
        expected_direction = travel / np.linalg.norm(travel)
        np.testing.assert_array_equal(first["direction"], expected_direction)
        np.testing.assert_array_equal(
            first["normal"], [-expected_direction[1], expected_direction[0]]
        )
        assert -np.pi <= first["theta_rad"] <= np.pi
        assert first["offset_sign"] in (-1.0, 1.0)
        assert 0.025 <= first["disk_offset_m"] <= 0.045
        assert 0.09 <= first["disk_radius_m"] <= 0.11
        assert np.max(np.abs(first["start_position_m"])) <= 0.35
        assert np.max(np.abs(first["goal_position_m"])) <= 0.35
        assert np.min(first["endpoint_physical_clearance_m"]) >= 0.02
        expected_worse = (
            "clockwise" if first["offset_sign"] > 0 else "counterclockwise"
        )
        assert first["predicted_worse_mode"] == expected_worse
    assert len(task_identity_hashes) == len(DEVELOPMENT_TASK_SEEDS)


def test_affine_dynamics_match_manual_and_batched_numpy_replay_exactly():
    task = generate_task(DEVELOPMENT_TASK_SEEDS[0])
    rng = np.random.default_rng(20260812)
    controls = rng.uniform(-0.3, 0.3, size=(BUDGET, 31, 2))
    batch = rollout_controls(task, controls)
    independent = np.asarray(
        [rollout_controls(task, controls[index]) for index in range(BUDGET)]
    )
    np.testing.assert_array_equal(batch, independent)
    manual = np.empty((32, 4), dtype=np.float64)
    manual[0] = task["x0"]
    for knot in range(31):
        manual[knot + 1, :2] = (
            manual[knot, :2]
            + PROTOCOL.dt * manual[knot, 2:]
            + 0.5 * PROTOCOL.dt**2 * controls[0, knot]
        )
        manual[knot + 1, 2:] = (
            manual[knot, 2:] + PROTOCOL.dt * controls[0, knot]
        )
    np.testing.assert_allclose(batch[0], manual, rtol=0.0, atol=2e-16)
    offset, mapping = affine_state_map(task)
    reconstructed = offset + np.einsum(
        "sij,j->si", mapping, controls[0].reshape(-1)
    )
    np.testing.assert_allclose(reconstructed, manual, rtol=0.0, atol=2e-16)


def test_objective_exact_gradient_hessian_and_normalized_hinge_derivatives():
    task = generate_task(DEVELOPMENT_TASK_SEEDS[0])
    controls = minimum_energy_controls(task, 1.0)
    point = controls.reshape(-1)
    value, gradient, hessian, _ = objective_control_derivatives(task, controls)
    assert np.isclose(value, objective_cost(task, controls))
    finite_gradient = _finite_difference_gradient(
        lambda candidate: objective_cost(task, candidate.reshape(31, 2)),
        point,
        epsilon=2e-6,
    )
    np.testing.assert_allclose(gradient, finite_gradient, rtol=2e-5, atol=2e-6)
    finite_hessian = np.column_stack(
        [
            (
                objective_control_derivatives(
                    task, (point + 2e-6 * np.eye(len(point))[index]).reshape(31, 2)
                )[1]
                - objective_control_derivatives(
                    task, (point - 2e-6 * np.eye(len(point))[index]).reshape(31, 2)
                )[1]
            )
            / (4e-6)
            for index in range(len(point))
        ]
    )
    np.testing.assert_allclose(hessian, finite_hessian, rtol=3e-5, atol=3e-6)
    center = task["disk_center_m"]
    residual, jacobian, residual_hessian = obstacle_residual(center, task)
    assert residual == 1.0
    finite_jacobian = _finite_difference_gradient(
        lambda position: obstacle_residual(position, task)[0],
        np.asarray(center, dtype=np.float64),
        epsilon=1e-6,
    )
    np.testing.assert_allclose(jacobian, finite_jacobian, atol=1e-8)
    finite_residual_hessian = np.column_stack(
        [
            (
                obstacle_residual(center + 1e-6 * np.eye(2)[axis], task)[1]
                - obstacle_residual(center - 1e-6 * np.eye(2)[axis], task)[1]
            )
            / 2e-6
            for axis in range(2)
        ]
    )
    np.testing.assert_allclose(residual_hessian, finite_residual_hessian, atol=1e-7)
    gn = PROTOCOL.running_obstacle_weight * np.outer(jacobian, jacobian)
    assert np.min(np.linalg.eigvalsh(gn)) >= -1e-12


def test_all_15_development_seed_families_are_exact_limited_and_obstacle_neutral():
    for task_seed in DEVELOPMENT_TASK_SEEDS:
        task = generate_task(task_seed)
        for method_seed in DEVELOPMENT_METHOD_SEEDS:
            states, controls, metadata = generate_obstacle_and_mode_neutral_seeds(
                task, method_seed
            )
            assert states.shape == (BUDGET, 32, 4)
            assert controls.shape == (BUDGET, 31, 2)
            assert metadata["initializer_family"] == "obstacle_and_mode_neutral"
            assert metadata["candidate_count"] == BUDGET
            assert metadata["candidate_labels"] == [
                "nominal_60_percent_endpoint",
                *[
                    label
                    for pair in range(7)
                    for label in (f"neutral_{pair}_plus", f"neutral_{pair}_minus")
                ],
                "neutral_7_independent",
            ]
            assert metadata["disk_or_mode_inputs"] == 0
            assert metadata["oracle_inputs_or_artifacts"] == 0
            assert metadata["collision_checks"] == 0
            assert metadata["retries_or_rejections"] == 0
            assert metadata["calibration_or_feedback_iterations"] == 0
            assert np.all(np.isfinite(states)) and np.all(np.isfinite(controls))
            np.testing.assert_array_equal(states, rollout_controls(task, controls))
            expected_terminal = terminal_target(task, 0.60)
            np.testing.assert_allclose(
                states[:, -1], np.tile(expected_terminal, (BUDGET, 1)), atol=2e-15
            )
            terminal_errors = np.linalg.norm(
                states[:, -1, :2] - task["goal_position_m"], axis=1
            )
            np.testing.assert_allclose(terminal_errors, 0.28, atol=2e-8)
            terminal_map, nullspace, _ = terminal_nullspace(task)
            nominal = controls[0]
            perturbations = controls - nominal
            np.testing.assert_allclose(
                perturbations.reshape(BUDGET, -1) @ terminal_map.T,
                0.0,
                atol=2e-15,
            )
            np.testing.assert_allclose(terminal_map @ nullspace, 0.0, atol=1e-15)
            assert np.max(np.abs(states[:, :, :2])) <= PROTOCOL.position_limit
            assert np.max(np.abs(states[:, :, 2:])) <= PROTOCOL.velocity_limit
            assert np.max(np.abs(controls)) <= PROTOCOL.acceleration_limit
            assert len({row.tobytes() for row in controls}) == BUDGET
            for pair in range(7):
                plus, minus = controls[1 + 2 * pair], controls[2 + 2 * pair]
                np.testing.assert_allclose(
                    plus - nominal, -(minus - nominal), atol=2e-16
                )
                assert np.isclose(np.max(np.abs(plus - nominal)), 0.4)
            assert np.isclose(np.max(np.abs(controls[-1] - nominal)), 0.4)

            changed_obstacle = deepcopy(task)
            changed_obstacle["disk_center_m"] = task["disk_center_m"] + [0.3, -0.2]
            changed_obstacle["disk_radius_m"] = 0.2
            changed_obstacle["required_distance_m"] = 0.22
            other_states, other_controls, _ = generate_obstacle_and_mode_neutral_seeds(
                changed_obstacle, method_seed
            )
            np.testing.assert_array_equal(controls, other_controls)
            np.testing.assert_array_equal(states, other_states)


def test_terminal_elimination_and_oracle_layout_are_exact_without_solving():
    task = generate_task(DEVELOPMENT_TASK_SEEDS[0])
    problem = Stage0Problem(task)
    terminal_map, nullspace, singular = terminal_nullspace(task)
    assert terminal_map.shape == (4, 62)
    assert nullspace.shape == (62, 58)
    assert len(singular) == 4 and np.min(singular) > 0.0
    np.testing.assert_allclose(terminal_map @ nullspace, 0.0, atol=1e-15)
    assert problem.variable_count == 58
    assert problem.original_linear.A.shape == (62 + 497 * 4, 58)
    assert problem.clearance(np.zeros(58)).shape == (497,)
    assert problem.clearance_jacobian(np.zeros(58)).shape == (497, 58)
    assert problem.clearance_hessian(np.zeros(58), np.ones(497)).shape == (58, 58)
    for mode_side in MODE_SIDES:
        acquired = problem._linear_constraint(mode_side)
        assert acquired.A.shape == (62 + 497 * 4 + 1, 58)
        for restart in range(len(ORACLE_RESTART_MARGINS_M)):
            z = problem.initial_z(mode_side, restart)
            controls = problem.controls(z)
            terminal = rollout_controls(task, controls)[-1]
            np.testing.assert_allclose(terminal, terminal_target(task, 1.0), atol=2e-15)


def test_independent_kkt_retains_full_dual_slack_and_residual_vectors():
    task = generate_task(DEVELOPMENT_TASK_SEEDS[0])
    problem = Stage0Problem(task)
    z = problem.initial_z(MODE_SIDES[0], 0)
    linear = problem.original_linear
    # A positive multiplier on the clearance lower bound violates SciPy's
    # lower-bound sign convention and must be visible independently.
    fake = SimpleNamespace(
        x=z,
        v=[np.zeros(len(linear.lb)), np.ones(len(problem.clearance(z)))],
    )
    kkt = problem.independent_kkt(fake, linear)
    assert kkt["linear_multiplier"].shape == linear.lb.shape
    assert kkt["clearance_multiplier"].shape == problem.clearance(z).shape
    assert kkt["linear_lower_slack"].shape == linear.lb.shape
    assert kkt["linear_upper_slack"].shape == linear.ub.shape
    assert kkt["clearance_lower_slack"].shape == problem.clearance(z).shape
    assert kkt["stationarity_vector"].shape == (problem.variable_count,)
    assert kkt["dual_sign_residual_vector"].shape == (
        len(linear.lb) + len(problem.clearance(z)),
    )
    assert kkt["complementarity_residual_vector"].shape == (
        2 * len(linear.lb) + len(problem.clearance(z)),
    )
    assert kkt["dual_sign_inf"] == 1.0
    assert np.all(np.isfinite(kkt["stationarity_vector"]))


def test_oracle_exact_derivatives_and_quarantine_boundary_are_static():
    task = generate_task(DEVELOPMENT_TASK_SEEDS[0])
    problem = Stage0Problem(task)
    z = problem.initial_z(MODE_SIDES[0], 0)
    finite_gradient = _finite_difference_gradient(problem.objective, z, epsilon=2e-6)
    np.testing.assert_allclose(
        problem.objective_jacobian(z), finite_gradient, rtol=2e-5, atol=2e-6
    )
    finite_clearance = np.column_stack(
        [
            (
                problem.clearance(z + 1e-6 * np.eye(len(z))[index])
                - problem.clearance(z - 1e-6 * np.eye(len(z))[index])
            )
            / 2e-6
            for index in range(len(z))
        ]
    )
    np.testing.assert_allclose(
        problem.clearance_jacobian(z), finite_clearance, rtol=1e-7, atol=1e-8
    )
    oracle_source = (
        REPO_ROOT / "pointmass_examples/pointmass2d_stage0_oracle.py"
    ).read_text()
    schema_source = (
        REPO_ROOT / "pointmass_src/gato_pointmass/pointmass2d.py"
    ).read_text()
    assert '"oracle_only": True' in oracle_source
    assert '"benchmark_seed_eligible": False' in oracle_source
    assert "--execute-stage0-oracle" in oracle_source
    assert "anchor_present_in_certifying_polish\": False" in oracle_source
    assert "hess=problem.objective_hessian" in oracle_source
    assert "hess=self.clearance_hessian" in oracle_source
    assert "pointmass2d_stage0_oracle" not in schema_source
    for required_provenance in (
        "git_status_porcelain_full",
        "python_version",
        "numpy_version",
        "scipy_version",
        "exact_command_argv",
        "protocol_sha256",
        "task_identity_sha256_by_seed",
        "source_hashes_at_start",
        "source_hashes_at_end",
        "source_hashes_stable_during_execution",
    ):
        assert required_provenance in oracle_source
    for retained_kkt_array in (
        "linear_multiplier",
        "clearance_multiplier",
        "linear_lower_slack",
        "linear_upper_slack",
        "clearance_lower_slack",
        "stationarity_vector",
        "dual_sign_residual_vector",
        "complementarity_residual_vector",
    ):
        assert retained_kkt_array in oracle_source
    assert TRUST_CONSTR_OPTIONS == {
        "maxiter": 2000,
        "gtol": 1e-10,
        "xtol": 1e-12,
        "barrier_tol": 1e-12,
        "initial_constr_penalty": 1.0,
        "initial_barrier_parameter": 0.1,
        "initial_barrier_tolerance": 0.1,
        "verbose": 0,
    }


def _mock_oracle_rows(clockwise_turns, counterclockwise_turns):
    rows = []
    for mode, turns in (
        ("clockwise", clockwise_turns),
        ("counterclockwise", counterclockwise_turns),
    ):
        for restart, turn in enumerate(turns):
            rows.append(
                {
                    "expected_mode": mode,
                    "restart_index": restart,
                    "passes": True,
                    "acquisition_converged": True,
                    "unanchored_polish_converged": True,
                    "unanchored_polish": {
                        "objective": 1.1 if mode == "clockwise" else 1.0,
                        "signed_turns": turn,
                        "mode": mode,
                    },
                }
            )
    return rows


def test_task_aggregate_rejects_pair_class_masking_and_failed_acquisition():
    task = generate_task(DEVELOPMENT_TASK_SEEDS[0])
    task["predicted_worse_mode"] = "clockwise"
    valid = _mock_oracle_rows([-0.5] * 5, [0.5] * 5)
    aggregate = task_aggregate(task, valid)
    assert aggregate["all_25_cross_mode_pairs_retained"]
    assert aggregate["cross_mode_class_one"]
    assert len(aggregate["cross_mode_pair_classes"]) == 25
    assert aggregate["same_mode_class_zero_with_integer_residual_at_most_0_10"]
    assert len(aggregate["same_mode_pair_classes"]) == 20
    assert aggregate["same_mode_pair_count_by_mode"] == {
        "clockwise": 10,
        "counterclockwise": 10,
    }

    # The mean difference is exactly one turn, but the (-.25,+.25) pair rounds
    # to class zero and must not be hidden by aggregate means.
    masked = _mock_oracle_rows(
        [-0.5625] * 4 + [-0.25], [0.5625] * 4 + [0.25]
    )
    assert np.isclose(
        np.mean([row["unanchored_polish"]["signed_turns"] for row in masked[5:]])
        - np.mean([row["unanchored_polish"]["signed_turns"] for row in masked[:5]]),
        1.0,
    )
    masked_aggregate = task_aggregate(task, masked)
    assert not masked_aggregate["cross_mode_class_one"]

    failed = deepcopy(valid)
    failed[3]["acquisition_converged"] = False
    failed[3]["passes"] = False
    failed_aggregate = task_aggregate(task, failed)
    assert not failed_aggregate["all_acquisitions_converged"]
    assert not failed_aggregate["all_restarts_pass"]

    # An extra full turn in one same-mode path must fail even when all labels and
    # cross-mode means could otherwise look plausible.
    extra_turn = _mock_oracle_rows([-0.5, -0.5, -0.5, -0.5, -1.5], [0.5] * 5)
    extra_aggregate = task_aggregate(task, extra_turn)
    assert not extra_aggregate[
        "same_mode_class_zero_with_integer_residual_at_most_0_10"
    ]
    assert not extra_aggregate["all_required_task_gates_pass"]

    # A cross-mode delta of .51 rounds to class one, but is not close enough to
    # the integer homotopy class under the predeclared .10 residual gate.
    fractional = _mock_oracle_rows([-0.25] * 5, [0.26] * 5)
    fractional_aggregate = task_aggregate(task, fractional)
    assert all(
        pair["absolute_class_one"]
        for pair in fractional_aggregate["cross_mode_pair_classes"].values()
    )
    assert not fractional_aggregate["cross_mode_class_one"]

    missing = valid[:-1]
    missing_aggregate = task_aggregate(task, missing)
    assert not missing_aggregate[
        "exact_five_restarts_per_mode_with_unique_identities"
    ]
    assert not missing_aggregate["all_required_task_gates_pass"]

    reversed_prediction_task = deepcopy(task)
    reversed_prediction_task["predicted_worse_mode"] = "counterclockwise"
    reversed_aggregate = task_aggregate(reversed_prediction_task, valid)
    assert not reversed_aggregate["predicted_worse_cost_strictly_greater"]
    assert not reversed_aggregate["all_required_task_gates_pass"]
