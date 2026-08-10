import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

from gato_tiago.iiwa_multimodal_sqp import (
    BUDGET,
    CENTRAL_FINAL_TASK_SEED,
    DEVELOPMENT_METHOD_SEEDS,
    DEVELOPMENT_PROTOCOL,
    DEVELOPMENT_TASK_SEEDS,
    FINAL_METHOD_SEEDS,
    FINAL_PROTOCOL,
    FINAL_TASK_SEEDS,
    all_rows_cap_free,
    batch_lane_repeatability,
    captured_solve_input_identity,
    candidate_directions,
    conditioning_candidate_labels,
    compare_serial_batch,
    cpu_dense_pin_seed_screen,
    discrete_acceleration_profile,
    discrete_rnea_candidate_labels,
    execute_after_seed_gate,
    external_replay_cost,
    generate_discrete_rnea_seeds,
    generate_planned_seeds,
    generate_instance,
    independent_pillar_cost,
    load_model,
    normalized_final_control_change,
    mode_certificates_agree,
    mode_support_certificate,
    pillar_residual_and_gradient,
    planned_seed_screen,
    productivity_gate,
    seed_kinematic_screen,
    solve_call_audit_passes,
    tool_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tiago_examples.iiwa_multimodal_target_only_ablation import (
    EXPECTED_CONTROLS_SHA256,
    EXPECTED_PLANNED_SEEDS_SHA256,
    exact_solver_parameter_ablation,
)
from tiago_examples.iiwa_multimodal_real_dev_probe import (
    seed_prerequisite_gate,
)


def test_protocol_and_seed_embargo_are_exact():
    assert DEVELOPMENT_PROTOCOL.knots == 16
    assert DEVELOPMENT_PROTOCOL.dt == 0.04
    assert np.isclose(
        (DEVELOPMENT_PROTOCOL.knots - 1) * DEVELOPMENT_PROTOCOL.dt, 0.60
    )
    assert DEVELOPMENT_PROTOCOL.joint_amplitude_rad == 0.01
    assert DEVELOPMENT_PROTOCOL.travel_m == 0.13
    assert DEVELOPMENT_PROTOCOL.goal_dz_m == 0.02
    assert DEVELOPMENT_PROTOCOL.pillar_radius_m == 0.025
    assert DEVELOPMENT_PROTOCOL.lateral_offset_m == 0.008
    assert DEVELOPMENT_PROTOCOL.q_cost == 2.0
    assert DEVELOPMENT_PROTOCOL.qd_cost == 0.01
    assert DEVELOPMENT_PROTOCOL.u_cost == 1e-4
    assert DEVELOPMENT_PROTOCOL.terminal_cost == 100.0
    assert DEVELOPMENT_PROTOCOL.pillar_cost == 800.0
    assert DEVELOPMENT_PROTOCOL.terminal_cuda_tolerance_m == 0.020
    assert DEVELOPMENT_PROTOCOL.terminal_pinocchio_tolerance_m == 0.022
    assert DEVELOPMENT_PROTOCOL.mu == 1000.0
    assert DEVELOPMENT_PROTOCOL.rho == 1.0
    assert DEVELOPMENT_PROTOCOL.max_sqp_iters == 60
    assert DEVELOPMENT_PROTOCOL.max_pcg_iters == 500
    assert DEVELOPMENT_PROTOCOL.pcg_tol == 1e-4
    assert DEVELOPMENT_TASK_SEEDS == (170, 171, 172)
    assert DEVELOPMENT_METHOD_SEEDS == (5000, 5001, 5002)
    assert FINAL_TASK_SEEDS == (180, 181, 182)
    assert FINAL_METHOD_SEEDS == tuple(range(6000, 6020))
    assert CENTRAL_FINAL_TASK_SEED == 181
    assert FINAL_PROTOCOL is None
    assert set(DEVELOPMENT_TASK_SEEDS).isdisjoint(FINAL_TASK_SEEDS)
    assert set(DEVELOPMENT_METHOD_SEEDS).isdisjoint(FINAL_METHOD_SEEDS)


def test_frozen_task_generator_is_exact_unique_and_no_retry():
    model = load_model()
    identities = []
    for seed in DEVELOPMENT_TASK_SEEDS:
        instance = generate_instance(seed, model=model)
        repeated = generate_instance(seed, model=model)
        np.testing.assert_array_equal(instance["q0"], repeated["q0"])
        np.testing.assert_array_equal(instance["reference"], repeated["reference"])
        metadata = instance["metadata"]
        assert len(metadata["q0_draw"]) == 7
        assert all(-0.10 <= value <= 0.10 for value in metadata["q0_draw"])
        assert -0.12 <= metadata["angle_jitter_draw_rad"] <= 0.12
        assert metadata["lateral_sign_draw"] in (-1.0, 1.0)
        assert metadata["minimum_endpoint_clearance_m"] >= 0.035
        assert metadata["rejection_or_retry_count"] == 0
        start = np.asarray(instance["start_position_m"])
        goal = np.asarray(instance["goal_position_m"])
        assert np.isclose(np.linalg.norm(goal[:2] - start[:2]), 0.13)
        assert np.isclose(goal[2] - start[2], 0.02)
        assert instance["pillar_radius_m"] == 0.025
        identities.append(instance["reference"].tobytes())
    assert len(set(identities)) == len(DEVELOPMENT_TASK_SEEDS)


def test_squared_hinge_residual_gradient_and_gn_are_correct():
    pillar = np.array([0.3, -0.1])
    radius = 0.025
    position = pillar + np.array([0.01, -0.005])
    residual, gradient = pillar_residual_and_gradient(position, pillar, radius)
    assert residual > 0.0
    epsilon = 1e-7
    finite_difference = np.asarray(
        [
            (
                pillar_residual_and_gradient(
                    position + epsilon * np.eye(2)[axis], pillar, radius
                )[0]
                - pillar_residual_and_gradient(
                    position - epsilon * np.eye(2)[axis], pillar, radius
                )[0]
            )
            / (2.0 * epsilon)
            for axis in range(2)
        ]
    )
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-7, atol=1e-7)
    jacobian = np.arange(14, dtype=np.float64).reshape(2, 7) / 10.0
    pillar_jacobian = gradient @ jacobian
    gn = 800.0 * np.outer(pillar_jacobian, pillar_jacobian)
    assert np.min(np.linalg.eigvalsh(gn)) >= -1e-8
    outside, outside_gradient = pillar_residual_and_gradient(
        pillar + np.array([0.03, 0.0]), pillar, radius
    )
    assert outside == 0.0
    np.testing.assert_array_equal(outside_gradient, 0.0)


def test_final_conditioning_seeds_are_deterministic_common_and_task_neutral():
    model = load_model()
    instance = generate_instance(DEVELOPMENT_TASK_SEEDS[0], model=model)
    first_seeds, first, metadata = generate_planned_seeds(
        model, instance["q0"], DEVELOPMENT_METHOD_SEEDS[0]
    )
    repeated_seeds, repeated, _ = generate_planned_seeds(
        model, instance["q0"], DEVELOPMENT_METHOD_SEEDS[0]
    )
    different_seeds, different, _ = generate_planned_seeds(
        model, instance["q0"], DEVELOPMENT_METHOD_SEEDS[1]
    )
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first_seeds, repeated_seeds)
    np.testing.assert_array_equal(first, different)
    assert not np.array_equal(first_seeds[1:], different_seeds[1:])
    directions = candidate_directions(model, DEVELOPMENT_METHOD_SEEDS[0])
    np.testing.assert_array_equal(directions[0], 0.0)
    for pair in range(7):
        np.testing.assert_array_equal(
            directions[1 + 2 * pair], -directions[2 + 2 * pair]
        )
    gravity = pin.rnea(
        model,
        model.createData(),
        instance["q0"].astype(np.float64),
        np.zeros(model.nv),
        np.zeros(model.nv),
    )
    np.testing.assert_allclose(
        first[0], np.tile(gravity, (first.shape[1], 1)), rtol=1e-6, atol=1e-6
    )
    assert np.max(np.abs(first) / model.effortLimit) <= 1.0
    for candidate in range(1, BUDGET):
        np.testing.assert_array_equal(first[0], first[candidate])
    assert metadata["inverse_dynamics_calls"] == 1
    assert metadata["expected_inverse_dynamics_calls"] == 1
    assert metadata["generation_latency_ms"] >= 0.0
    assert not metadata["rollout_used_to_populate_planned_states"]
    assert metadata["task_reference_or_obstacle_inputs"] == 0
    assert metadata["feedback_or_calibration_iterations"] == 0
    assert metadata["rejection_or_resampling_attempts"] == 0
    assert metadata["inverse_kinematics_calls"] == 0


def test_active_and_historical_initializer_labels_are_exact_and_disjoint():
    conditioning = conditioning_candidate_labels()
    discrete = discrete_rnea_candidate_labels()
    assert conditioning == [
        "stationary_gravity_hold",
        *[
            label
            for pair in range(7)
            for label in (
                f"conditioning_{pair}_plus",
                f"conditioning_{pair}_minus",
            )
        ],
        "conditioning_7_independent",
    ]
    assert discrete == [
        "stationary_inverse_dynamics_hold",
        *[
            label
            for pair in range(7)
            for label in (
                f"discrete_rnea_{pair}_plus",
                f"discrete_rnea_{pair}_minus",
            )
        ],
        "discrete_rnea_7_independent",
    ]
    assert len(conditioning) == len(discrete) == BUDGET
    assert set(conditioning).isdisjoint(discrete)

    model = load_model()
    instance = generate_instance(DEVELOPMENT_TASK_SEEDS[0], model=model)
    _, _, active_metadata = generate_planned_seeds(
        model, instance["q0"], DEVELOPMENT_METHOD_SEEDS[0]
    )
    _, _, historical_metadata = generate_discrete_rnea_seeds(
        model, instance["q0"], DEVELOPMENT_METHOD_SEEDS[0]
    )
    assert active_metadata["candidate_labels"] == conditioning
    assert historical_metadata["candidate_labels"] == discrete
    assert not any("discrete_rnea" in label for label in conditioning)
    assert not any("conditioning" in label for label in discrete)


def test_discrete_acceleration_profile_has_exact_frozen_projection_and_closure():
    discrete_protocol = replace(DEVELOPMENT_PROTOCOL, joint_amplitude_rad=0.04)
    displacement, velocity, acceleration, metadata = discrete_acceleration_profile(
        discrete_protocol
    )
    intervals = DEVELOPMENT_PROTOCOL.knots - 1
    indices = np.arange(intervals, dtype=np.float64)
    raw = np.cos(2.0 * np.pi * (indices + 0.5) / intervals)
    constraints = np.vstack([np.ones(intervals), intervals - indices - 0.5])
    projected = raw - constraints.T @ np.linalg.solve(
        constraints @ constraints.T, constraints @ raw
    )
    scale = acceleration[0] / projected[0]
    np.testing.assert_allclose(acceleration, scale * projected, atol=1e-12)
    np.testing.assert_allclose(constraints @ acceleration, 0.0, atol=1e-12)
    assert np.isclose(np.max(np.abs(displacement)), 0.04)
    assert abs(displacement[-1]) <= 1e-12
    assert abs(velocity[-1]) <= 1e-12
    assert metadata["interval_count"] == 15
    np.testing.assert_allclose(metadata["raw_profile"], raw)


def test_generator_makes_exactly_one_rnea_call_per_lane_interval(monkeypatch):
    model = load_model()
    instance = generate_instance(DEVELOPMENT_TASK_SEEDS[0], model=model)
    original = pin.rnea
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(pin, "rnea", counted)
    _, _, metadata = generate_discrete_rnea_seeds(
        model, instance["q0"], DEVELOPMENT_METHOD_SEEDS[0]
    )
    assert len(calls) == BUDGET * (DEVELOPMENT_PROTOCOL.knots - 1) == 240
    assert metadata["inverse_dynamics_calls"] == len(calls)


def test_final_conditioning_generator_makes_exactly_one_rnea_call(monkeypatch):
    model = load_model()
    instance = generate_instance(DEVELOPMENT_TASK_SEEDS[0], model=model)
    original = pin.rnea
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(pin, "rnea", counted)
    generate_planned_seeds(
        model, instance["q0"], DEVELOPMENT_METHOD_SEEDS[0]
    )
    assert len(calls) == 1


def test_productivity_requires_final_control_change_not_planned_state_change():
    model = load_model()
    instance = generate_instance(DEVELOPMENT_TASK_SEEDS[0], model=model)
    _, controls, _ = generate_planned_seeds(
        model, instance["q0"], DEVELOPMENT_METHOD_SEEDS[0]
    )
    assert normalized_final_control_change(
        controls[0], controls[0], model.effortLimit
    ) == 0.0
    changed = controls[0].copy()
    changed[:, 0] += 0.001 * model.effortLimit[0]
    assert normalized_final_control_change(
        changed, controls[0], model.effortLimit
    ) >= 1e-4


def test_all_development_planned_and_replayed_seeds_pass_frozen_gates():
    model = load_model()
    for task_seed in DEVELOPMENT_TASK_SEEDS:
        instance = generate_instance(task_seed, model=model)
        for method_seed in DEVELOPMENT_METHOD_SEEDS:
            seeds, controls, _ = generate_planned_seeds(
                model, instance["q0"], method_seed
            )
            planned = planned_seed_screen(model, instance["q0"], seeds)
            rows, aggregate = seed_kinematic_screen(model, instance, controls)
            dense_rows, dense_aggregate = cpu_dense_pin_seed_screen(
                model, instance, controls
            )
            assert len(rows) == BUDGET
            assert len(dense_rows) == BUDGET
            assert planned["passes"]
            assert planned["unique_planned_seed_count"] == BUDGET
            assert len(set(planned["planned_seed_sha256"])) == BUDGET
            assert planned["all_planned_values_finite"]
            assert planned["maximum_planned_velocity_limit_ratio"] <= 1.0
            assert planned["maximum_planned_control_limit_ratio"] <= 1.0
            assert planned["terminal_closure_within_1e_6"]
            assert planned["common_controls_all_candidates"]
            assert planned["cold_candidate_pinocchio_one_step_defect"] <= 1e-4
            assert 0.04 <= planned["maximum_planned_pinocchio_one_step_defect"] <= 0.08
            assert planned["predeclared_defect_profile_pass"]
            assert aggregate["all_candidates_finite"]
            assert aggregate["all_candidates_imperfect"]
            assert aggregate["all_candidate_controls_retained"]
            assert aggregate["all_candidates_within_limits"]
            assert aggregate["passes"]
            assert dense_aggregate["passes"]


def test_build_wiring_selects_an_isolated_non_generated_variant():
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    bindings = (REPO_ROOT / "python/bindings.cu").read_text()
    plant = (REPO_ROOT / "gato/dynamics/iiwa14/iiwa14_plant.cuh").read_text()
    assert 'plant STREQUAL "iiwa14_multimodal"' in cmake
    assert "IIWA_MULTIMODAL_PILLAR=1" in cmake
    assert "#define PLANT_SUFFIX iiwa14_multimodal" in bindings
    assert "#if defined(IIWA_MULTIMODAL_PILLAR)" in plant
    assert "pillarResidualAndWorkspaceGradient" in plant
    assert (
        "const T pillar_weight = blockIdx.x == KNOT_POINTS - 1 ? "
        "ee_orient_N_cost : ee_orient_cost;"
    ) in plant
    assert "s_qk[i] += pillar_weight * pillar_residual * pillar_jacobian;" in plant
    assert (
        "s_Qk[i * grid::NX + j] += pillar_weight * pillar_jacobian_i "
        "* pillar_jacobian_j;"
    ) in plant
    assert (
        "grid::NQ + grid::NU + 2 * grid::EE_POS_SIZE "
        "+ grid::EE_POS_DYNAMIC_SHARED_MEM_COUNT"
    ) in plant


def test_independent_pillar_report_distinguishes_integrated_and_peak_violation():
    model = load_model()
    q = np.zeros((DEVELOPMENT_PROTOCOL.knots, model.nq))
    tool_position = tool_path(model, q[:1])[0]
    states = np.hstack([q, np.zeros_like(q)])
    report = independent_pillar_cost(
        model, states, tool_position[:2] + np.array([0.005, 0.0]), 0.025, 800.0
    )
    assert report["weighted_cost"] > 0.0
    assert report["sum_positive_hinge"] >= report["maximum_positive_hinge"] > 0.0


def test_opt_in_diagnostics_are_frozen_and_do_not_run_by_default():
    source = (REPO_ROOT / "tiago_examples/iiwa_multimodal_diagnostics.py").read_text()
    assert 'choices=("seed-screen", "core-smoke")' in source
    assert '"sqps_or_optimization_calls": 0' in source
    assert "DEVELOPMENT_TASK_SEEDS[0]" in source
    assert "DEVELOPMENT_METHOD_SEEDS[0]" in source
    assert "if __name__ == \"__main__\":" in source
    assert "sum_positive_hinge" in source
    assert '"sqp_solve_calls": 3' in source
    assert '"benchmark_evidence": False' in source
    assert '"immutable_initial_collision": True' in source
    assert '"cpu_dense4_seed_replay_aggregate"' in source
    assert '"cuda_pinocchio_dense4_seed_replay_aggregate"' in source
    assert '"cuda_dense4_seed_rollout"' in source
    assert '"pinocchio_dense4_seed_rollout"' in source


def test_identical_input_batch_repeatability_rejects_any_hidden_lane_change():
    output = np.zeros((BUDGET, 5), dtype=np.float32)
    stats = {
        "initial_merit": np.full(BUDGET, 100.0),
        "final_merit": np.full(BUDGET, 10.0),
        "sqp_iterations": np.full(BUDGET, 2),
        "total_pcg_iterations": np.full(BUDGET, 12),
        "pcg_cap_hits": np.zeros(BUDGET, dtype=np.int64),
        "reached_requested_sqp_iteration_limit": np.ones(BUDGET, dtype=bool),
    }
    assert batch_lane_repeatability(output, stats)["passes"]
    changed_output = output.copy()
    changed_output[-1, -1] = 1.0
    assert not batch_lane_repeatability(changed_output, stats)["passes"]
    changed_stats = {key: value.copy() for key, value in stats.items()}
    changed_stats["total_pcg_iterations"][-1] += 1
    assert not batch_lane_repeatability(output, changed_stats)["passes"]
    changed_merit = {key: value.copy() for key, value in stats.items()}
    changed_merit["final_merit"][-1] *= 1.001
    assert not batch_lane_repeatability(output, changed_merit)["passes"]


def test_external_replay_cost_contains_every_frozen_nonnegative_term():
    model = load_model()
    knots = DEVELOPMENT_PROTOCOL.knots
    q = np.zeros((knots, model.nq))
    qd = np.full((knots, model.nv), 0.2)
    states = np.hstack([q, qd])
    controls = np.full((knots - 1, model.nv), 0.3)
    path = tool_path(model, q)
    goal = path[0] + np.array([0.01, -0.02, 0.03])
    pillar = path[0, :2] + np.array([0.005, 0.0])
    report = external_replay_cost(
        model, states, controls, goal, pillar, 0.025
    )
    goal_sq = np.sum((path - goal) ** 2, axis=1)
    residual = 1.0 - np.sum((path[:, :2] - pillar) ** 2, axis=1) / 0.025**2
    residual = np.maximum(residual, 0.0)
    expected = {
        "running_position": 0.5 * 2.0 * np.sum(goal_sq[:-1]),
        "terminal_position": 0.5 * 100.0 * goal_sq[-1],
        "velocity": 0.5 * 0.01 * np.sum(qd**2),
        "control": 0.5 * 1e-4 * np.sum(controls**2),
        "pillar": 0.5 * 800.0 * np.sum(residual**2),
    }
    for key, value in expected.items():
        assert np.isclose(report["components"][key], value)
    assert np.isclose(report["total"], sum(expected.values()))
    altered = replace(DEVELOPMENT_PROTOCOL, qd_cost=0.02)
    altered_report = external_replay_cost(
        model, states, controls, goal, pillar, 0.025, altered
    )
    assert np.isclose(
        altered_report["components"]["velocity"], 2.0 * expected["velocity"]
    )
    for key in ("running_position", "terminal_position", "control", "pillar"):
        assert np.isclose(altered_report["components"][key], expected[key])


def _topology_rows(model):
    def path(turns, radius):
        angle = np.linspace(0.0, 2.0 * np.pi * turns, 61)
        return np.column_stack(
            [radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)]
        )

    rows = []
    for index, (turns, radius) in enumerate(
        ((-0.5, 0.050), (-0.5, 0.052), (0.5, 0.050), (0.5, 0.052))
    ):
        controls = np.full(
            (DEVELOPMENT_PROTOCOL.knots - 1, model.nv),
            index * 0.001 * model.effortLimit,
        )
        rows.append(
            {
                "candidate_index": index,
                "productive_certified": True,
                "cuda_external_cost": 10.0 + index,
                "controls": controls,
                "path": path(turns, radius),
            }
        )
    return rows


def test_mode_certificate_rejects_neutral_pairwise_mismatch_and_duplicates():
    model = load_model()
    rows = _topology_rows(model)
    certificate = mode_support_certificate(
        rows, np.zeros(2), model.effortLimit, "path", "cuda_external_cost"
    )
    assert certificate["two_mode_support"]
    assert certificate["support"] == {"clockwise": 2, "counterclockwise": 2}

    neutral = copy.deepcopy(_topology_rows(model))
    angle = np.linspace(0.0, 2.0 * np.pi * 0.1, 61)
    neutral[3]["path"] = np.column_stack(
        [0.052 * np.cos(angle), 0.052 * np.sin(angle), np.zeros_like(angle)]
    )
    assert not mode_support_certificate(
        neutral, np.zeros(2), model.effortLimit, "path", "cuda_external_cost"
    )["two_mode_support"]

    mismatch = copy.deepcopy(_topology_rows(model))
    angle = np.linspace(0.0, 2.0 * np.pi * 0.35, 61)
    mismatch[3]["path"] = np.column_stack(
        [0.052 * np.cos(angle), 0.052 * np.sin(angle), np.zeros_like(angle)]
    )
    mismatch_certificate = mode_support_certificate(
        mismatch, np.zeros(2), model.effortLimit, "path", "cuda_external_cost"
    )
    assert mismatch_certificate["support"]["counterclockwise"] == 2
    assert not mismatch_certificate["two_mode_support"]
    assert not mode_certificates_agree(certificate, mismatch_certificate)

    duplicate = copy.deepcopy(_topology_rows(model))
    duplicate[1]["path"] = duplicate[0]["path"].copy()
    duplicate[1]["controls"] = duplicate[0]["controls"].copy()
    duplicate[3]["path"] = duplicate[2]["path"].copy()
    duplicate[3]["controls"] = duplicate[2]["controls"].copy()
    duplicate_certificate = mode_support_certificate(
        duplicate, np.zeros(2), model.effortLimit, "path", "cuda_external_cost"
    )
    assert duplicate_certificate["support"] == {
        "clockwise": 1,
        "counterclockwise": 1,
    }
    assert not duplicate_certificate["two_mode_support"]


def test_productivity_gate_rejects_every_independent_failure_class():
    evidence = {
        "seed_not_fully_feasible": True,
        "finite": True,
        "planned_trajectory_change": True,
        "final_control_change": True,
        "solver_merit_decrease": True,
        "cuda_external_cost_decrease": True,
        "pinocchio_external_cost_decrease": True,
        "terminal_reduction_and_final_tolerance": True,
        "dense_clearance": True,
        "limits": True,
        "dense_model_agreement": True,
        "cap_free": True,
    }
    assert productivity_gate(evidence)
    for key in evidence:
        failed = evidence.copy()
        failed[key] = False
        assert not productivity_gate(failed), key


def _comparison_rows():
    return [
        {
            "candidate_index": index,
            "productive_certified": True,
            "cuda_external_cost": 10.0,
            "pinocchio_external_cost": 10.0,
            "final_cuda": {"terminal_error_m": 0.01},
            "final_pinocchio": {"terminal_error_m": 0.01},
            "finite": True,
            "limits_pass": True,
            "pcg_cap_hits": 0,
            "max_sqp_reached": False,
        }
        for index in range(BUDGET)
    ]


def test_serial_batch_comparison_rejects_quality_cap_and_outcome_regressions():
    serial = _comparison_rows()
    assert compare_serial_batch(serial, copy.deepcopy(serial))[1]["passes"]
    assert all_rows_cap_free(serial)
    for mutation in ("cost", "terminal", "cap", "max_sqp", "outcome"):
        batch = copy.deepcopy(serial)
        if mutation == "cost":
            batch[-1]["cuda_external_cost"] = 10.2
        elif mutation == "terminal":
            batch[-1]["final_pinocchio"]["terminal_error_m"] = 0.012
        elif mutation == "cap":
            batch[-1]["pcg_cap_hits"] = 1
        elif mutation == "max_sqp":
            batch[-1]["max_sqp_reached"] = True
        else:
            batch[-1]["productive_certified"] = False
        assert not compare_serial_batch(serial, batch)[1]["passes"], mutation
        if mutation in ("cap", "max_sqp"):
            assert not all_rows_cap_free(batch)


def test_failed_seed_prerequisite_makes_zero_solve_calls():
    callbacks = []

    def callback():
        callbacks.append(True)
        return {
            "solved": True,
            "solve_call_audit": {
                "completed_b1_solve_once_calls": BUDGET,
                "completed_b16_solve_once_calls": 1,
            },
        }

    result, audit = execute_after_seed_gate(False, callback)
    assert result is None and sum(audit.values()) == 0 and callbacks == []
    assert not solve_call_audit_passes(audit)
    result, audit = execute_after_seed_gate(True, callback)
    assert result["solved"] and solve_call_audit_passes(audit)
    assert callbacks == [True]
    for wrong in (
        {
            "completed_b1_solve_once_calls": BUDGET - 1,
            "completed_b16_solve_once_calls": 1,
        },
        {
            "completed_b1_solve_once_calls": BUDGET,
            "completed_b16_solve_once_calls": 0,
        },
        {
            "completed_b1_solve_once_calls": BUDGET + 1,
            "completed_b16_solve_once_calls": 1,
        },
        {
            "completed_b1_solve_once_calls": BUDGET,
            "completed_b16_solve_once_calls": 2,
        },
    ):
        assert not solve_call_audit_passes(wrong)


def test_dense_seed_failure_makes_zero_real_probe_solve_calls():
    passed = {"passes": True}
    failed_dense = {"passes": False}
    for cpu_dense, model_dense in (
        (failed_dense, passed),
        (passed, failed_dense),
    ):
        gate = seed_prerequisite_gate(
            passed, passed, cpu_dense, passed, model_dense
        )
        calls = []

        def callback():
            calls.append(True)
            raise AssertionError("dense seed failure must stop before every solve")

        result, audit = execute_after_seed_gate(gate, callback)
        assert result is None
        assert calls == []
        assert audit == {
            "completed_b1_solve_once_calls": 0,
            "completed_b16_solve_once_calls": 0,
        }


def test_captured_solve_inputs_reject_signed_zero_bit_and_shape_mutations():
    widths = {"x0": 14, "reference": 96, "seeds": 329}
    batch = {
        key: np.zeros((BUDGET, width), dtype=np.float32)
        for key, width in widths.items()
    }
    serial = {
        key: value[:, None, :].copy() for key, value in batch.items()
    }
    assert captured_solve_input_identity(serial, batch)["passes"]
    for field in widths:
        signed_zero = {key: value.copy() for key, value in batch.items()}
        signed_zero[field][0, 0] = np.float32(-0.0)
        assert not captured_solve_input_identity(serial, signed_zero)["passes"], field

        one_bit = {key: value.copy() for key, value in batch.items()}
        bits = one_bit[field].view(np.uint32)
        bits[1, 1] ^= np.uint32(1)
        assert not captured_solve_input_identity(serial, one_bit)["passes"], field

    wrong_shape = {key: value.copy() for key, value in batch.items()}
    wrong_shape["reference"] = wrong_shape["reference"][:, :, None]
    assert not captured_solve_input_identity(serial, wrong_shape)["passes"]
    extra_serial = {key: value.copy() for key, value in serial.items()}
    extra_serial["seeds"] = np.concatenate(
        [extra_serial["seeds"], extra_serial["seeds"][:1]], axis=0
    )
    assert not captured_solve_input_identity(extra_serial, batch)["passes"]
    both_extra_serial = {key: value.copy() for key, value in serial.items()}
    both_extra_batch = {key: value.copy() for key, value in batch.items()}
    both_extra_serial["x0"] = both_extra_serial["x0"][:, :, :, None]
    both_extra_batch["x0"] = both_extra_batch["x0"][:, :, None]
    assert not captured_solve_input_identity(
        both_extra_serial, both_extra_batch
    )["passes"]
    float64_serial = {
        key: value.astype(np.float64) for key, value in serial.items()
    }
    float64_batch = {
        key: value.astype(np.float64) for key, value in batch.items()
    }
    assert not captured_solve_input_identity(
        float64_serial, float64_batch
    )["passes"]


def test_real_probe_is_fixed_to_one_development_case_and_no_timing_or_final():
    source = (
        REPO_ROOT / "tiago_examples/iiwa_multimodal_real_dev_probe.py"
    ).read_text()
    assert "DEVELOPMENT_TASK_SEEDS[0]" in source
    assert "DEVELOPMENT_METHOD_SEEDS[0]" in source
    assert '"warmups": 0' in source
    assert '"repeats": 1' in source
    assert '"timing_evidence": False' in source
    assert "FINAL_PROTOCOL is not None" in source
    assert "execute_after_seed_gate" in source
    diagnostics = (
        REPO_ROOT / "tiago_examples/iiwa_multimodal_diagnostics.py"
    ).read_text()
    assert '"executing_harness": sha256_file(harness_path)' in diagnostics


def test_target_only_ablation_changes_only_both_pillar_weights():
    ablation = exact_solver_parameter_ablation()
    assert ablation["passes"]
    assert set(ablation["changed_fields"]) == {
        "ee_orient_cost",
        "ee_orient_N_cost",
    }
    for key, value in ablation["baseline"].items():
        if key in ablation["changed_fields"]:
            assert value == 800.0 and ablation["ablation"][key] == 0.0
        else:
            assert ablation["ablation"][key] == value


def test_target_only_ablation_requires_exact_rejected_seed_tensors():
    model = load_model()
    instance = generate_instance(DEVELOPMENT_TASK_SEEDS[0], model=model)
    seeds, controls, _ = generate_discrete_rnea_seeds(
        model, instance["q0"], DEVELOPMENT_METHOD_SEEDS[0]
    )
    assert hashlib.sha256(seeds.tobytes()).hexdigest() == (
        EXPECTED_PLANNED_SEEDS_SHA256
    )
    assert hashlib.sha256(controls.tobytes()).hexdigest() == (
        EXPECTED_CONTROLS_SHA256
    )
    changed = seeds.copy()
    changed.view(np.uint32)[0, 0] ^= np.uint32(1)
    assert hashlib.sha256(changed.tobytes()).hexdigest() != (
        EXPECTED_PLANNED_SEEDS_SHA256
    )


def test_target_only_ablation_can_never_be_benchmark_or_multimodal_evidence():
    source = (
        REPO_ROOT / "tiago_examples/iiwa_multimodal_target_only_ablation.py"
    ).read_text()
    for exact in (
        '"benchmark_evidence": False',
        '"multimodal_evidence": False',
        '"all_gates_pass": False',
        '"known_seed_invalid_for_benchmark": True',
        '"mechanism_only_limit_bypass": True',
        '"topology_and_support": "not_applicable"',
        "FINAL_PROTOCOL is not None",
        "replace(DEVELOPMENT_PROTOCOL, pillar_cost=0.0)",
    ):
        assert exact in source
