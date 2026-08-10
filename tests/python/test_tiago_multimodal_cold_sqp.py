import copy

import numpy as np
import pinocchio as pin
import pytest

from gato_tiago.multimodal_cold_sqp import (
    BUDGET,
    DEVELOPMENT_METHOD_SEEDS,
    DEVELOPMENT_PROTOCOL,
    DEVELOPMENT_TASK_SEEDS,
    FINAL_PROTOCOL,
    FINAL_METHOD_SEEDS,
    FINAL_TASK_SEEDS,
    candidate_labels,
    candidate_directions,
    certify_mode_support,
    generate_cold_instance,
    generate_controls,
    model_pair_preflight,
    mode_certificates_agree,
    one_pass_rollout_and_pack,
    paired_mode_cost_summary,
    pairwise_closed_loop_winding,
    pinocchio_step,
    require_seed_screen_before_sqp,
    seed_rollout_screen,
)
from gato_tiago.multimodal_pillar import MODEL_PATH, load_model, unpack_trajectory


def semicircle(center, radius, mode, samples=80):
    angles = np.linspace(np.pi, 0.0, samples)
    if mode == "counterclockwise":
        angles = np.linspace(-np.pi, 0.0, samples)
    xy = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return np.column_stack([xy, np.zeros(samples)])


def test_development_and_final_task_and_method_seeds_are_disjoint():
    assert set(DEVELOPMENT_TASK_SEEDS).isdisjoint(FINAL_TASK_SEEDS)
    assert set(DEVELOPMENT_METHOD_SEEDS).isdisjoint(FINAL_METHOD_SEEDS)
    assert len(FINAL_TASK_SEEDS) == 3
    assert len(FINAL_METHOD_SEEDS) == 20
    assert FINAL_PROTOCOL is None
    assert DEVELOPMENT_PROTOCOL.difficulty.knots == 128
    assert DEVELOPMENT_PROTOCOL.difficulty.dt == 0.006
    assert DEVELOPMENT_PROTOCOL.joint_amplitude_rad == 0.03


def test_control_candidates_are_deterministic_bounded_and_antithetic():
    model = load_model(MODEL_PATH)
    q0 = np.zeros(model.nq)
    first, metadata = generate_controls(model, q0, method_seed=3000)
    repeated, _ = generate_controls(model, q0, method_seed=3000)
    different, _ = generate_controls(model, q0, method_seed=3001)
    gravity = pin.rnea(
        model,
        model.createData(),
        q0,
        np.zeros(model.nv),
        np.zeros(model.nv),
    )

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_allclose(
        first[0], np.tile(gravity, (first.shape[1], 1)), rtol=1e-6, atol=1e-6
    )
    assert not np.array_equal(first[1:], different[1:])
    assert first.shape == (BUDGET, DEVELOPMENT_PROTOCOL.difficulty.knots - 1, model.nv)
    assert np.max(np.abs(first) / model.effortLimit) <= 1.0
    directions = candidate_directions(model, 3000)
    np.testing.assert_array_equal(directions[0], 0.0)
    for pair in range(7):
        np.testing.assert_array_equal(directions[1 + 2 * pair], -directions[2 + 2 * pair])
    assert len({row.tobytes() for row in directions[1:]}) == BUDGET - 1
    assert metadata["inverse_dynamics_calls"] == BUDGET * (
        DEVELOPMENT_PROTOCOL.difficulty.knots - 1
    )
    assert metadata["task_reference_or_obstacle_inputs"] == 0
    assert metadata["rejection_or_resampling_attempts"] == 0
    assert metadata["feedback_or_calibration_iterations"] == 0
    assert metadata["candidate_labels"] == candidate_labels()


def test_development_task_generator_uses_exact_no_retry_geometry():
    model = load_model(MODEL_PATH)
    identities = []
    for task_seed in DEVELOPMENT_TASK_SEEDS:
        instance, metadata = generate_cold_instance(task_seed, model=model)
        repeated, repeated_metadata = generate_cold_instance(task_seed, model=model)
        assert instance == repeated
        assert metadata == repeated_metadata
        assert len(metadata["joint_jitter_draw_rad"]) == 7
        assert metadata["exact_xy_travel_m"] == 0.15
        assert metadata["exact_goal_dz_m"] == 0.0
        assert metadata["exact_pillar_fraction"] == 0.5
        assert abs(metadata["exact_lateral_offset_m"]) == 0.008
        assert metadata["exact_pillar_radius_m"] == 0.03
        assert metadata["minimum_endpoint_clearance_m"] >= 0.035
        assert metadata["rejection_or_retry_count"] == 0
        start = np.asarray(instance.start_position)
        goal = np.asarray(instance.goal_position)
        assert np.isclose(np.linalg.norm(goal[:2] - start[:2]), 0.15)
        assert goal[2] == start[2]
        identities.append((instance.start_q, instance.goal_position, instance.pillar_xy))
    assert len(set(identities)) == len(DEVELOPMENT_TASK_SEEDS)


def test_one_pass_seed_rollout_has_fixed_call_count_and_common_x0():
    model = load_model(MODEL_PATH)
    q0 = np.zeros(model.nq)
    x0 = np.hstack([q0, np.zeros(model.nv)]).astype(np.float32)
    controls, _ = generate_controls(model, q0, method_seed=3000)
    call_count = 0

    def fake_forward(states, inputs, dt):
        nonlocal call_count
        call_count += 1
        result = np.asarray(states).copy()
        result[:, model.nq :] += dt * inputs
        result[:, : model.nq] += dt * result[:, model.nq :]
        return result

    packed, metadata = one_pass_rollout_and_pack(fake_forward, x0, controls)

    assert call_count == DEVELOPMENT_PROTOCOL.difficulty.knots - 1
    assert metadata["feedback_or_calibration_iterations"] == 0
    assert metadata["collision_or_pillar_queries"] == 0
    for trajectory in packed:
        q, qd, unpacked_controls = unpack_trajectory(
            trajectory, model, DEVELOPMENT_PROTOCOL.difficulty.knots
        )
        np.testing.assert_array_equal(q[0], q0)
        np.testing.assert_array_equal(qd[0], 0.0)
        assert unpacked_controls.shape == controls.shape[1:]


def test_seed_screen_retains_every_imperfect_candidate_and_enforces_model_gate():
    model = load_model(MODEL_PATH)
    instance, _ = generate_cold_instance(DEVELOPMENT_TASK_SEEDS[0], model=model)
    q0 = np.asarray(instance.start_q)
    x0 = np.hstack([q0, np.zeros(model.nv)]).astype(np.float32)
    controls, _ = generate_controls(model, q0, method_seed=3000)

    def exact_forward(states, inputs, dt):
        states = np.asarray(states)
        inputs = np.asarray(inputs)
        if states.ndim == 1:
            states = states[None, :]
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        return np.asarray(
            [
                pinocchio_step(
                    model, model.createData(), states[index], inputs[index], dt
                )
                for index in range(len(states))
            ]
        )

    preflight = model_pair_preflight(
        exact_forward, model, q0, DEVELOPMENT_PROTOCOL.difficulty.dt
    )
    rows, aggregate, arrays = seed_rollout_screen(
        model,
        instance,
        DEVELOPMENT_PROTOCOL,
        x0,
        controls,
        exact_forward,
        preflight,
    )

    assert len(rows) == BUDGET
    assert aggregate["retained_candidate_count"] == BUDGET
    assert aggregate["dropped_or_retried_candidate_count"] == 0
    assert aggregate["unique_nonstationary_lanes"]
    assert aggregate["all_candidates_finite"]
    assert aggregate["all_candidates_imperfect"]
    assert aggregate["all_seed_only_diagnostic_gates_pass"]
    assert arrays["cuda_seed_rollout"].shape == arrays["pinocchio_seed_rollout"].shape

    bad_preflight = dict(preflight)
    bad_preflight["passes_1e-3_gate"] = False
    _, rejected, _ = seed_rollout_screen(
        model,
        instance,
        DEVELOPMENT_PROTOCOL,
        x0,
        controls,
        exact_forward,
        bad_preflight,
    )
    assert not rejected["all_seed_only_diagnostic_gates_pass"]


def test_failed_seed_screen_aborts_before_any_sqp_call():
    solve_calls = 0

    def forbidden_solve():
        nonlocal solve_calls
        solve_calls += 1

    with pytest.raises(RuntimeError, match="SQP disabled"):
        require_seed_screen_before_sqp(
            {"all_seed_only_diagnostic_gates_pass": False}
        )
        forbidden_solve()
    assert solve_calls == 0

    require_seed_screen_before_sqp(
        {"all_seed_only_diagnostic_gates_pass": True}
    )


def test_pairwise_closed_loop_winding_is_integer_for_opposite_modes():
    center = np.array([0.5, -0.2])
    clockwise = semicircle(center, 0.14, "clockwise")
    counterclockwise = semicircle(center, 0.14, "counterclockwise")
    same_mode_variant = semicircle(center, 0.16, "clockwise")

    cross = pairwise_closed_loop_winding(clockwise, counterclockwise, center)
    same = pairwise_closed_loop_winding(clockwise, same_mode_variant, center)

    assert cross["valid_integer"]
    assert abs(cross["nearest_integer"]) == 1
    assert same["valid_integer"]
    assert same["nearest_integer"] == 0


def test_model_pair_preflight_uses_fixed_benign_and_random_distribution():
    model = load_model(MODEL_PATH)
    q0 = np.zeros(model.nq)

    def exact_forward(states, controls, dt):
        states = np.asarray(states)
        controls = np.asarray(controls)
        if states.ndim == 1:
            states = np.tile(states, (BUDGET, 1))
        if controls.ndim == 1:
            controls = np.tile(controls, (BUDGET, 1))
        return np.asarray(
            [
                pinocchio_step(
                    model, model.createData(), states[index], controls[index], dt
                )
                for index in range(BUDGET)
            ]
        )

    result = model_pair_preflight(
        exact_forward, model, q0, DEVELOPMENT_PROTOCOL.difficulty.dt
    )

    assert result["passes_1e-3_gate"]
    assert result["maximum_l2_error"] <= 1e-6
    assert [row["name"] for row in result["benign"]] == ["zero", "gravity"]
    assert result["distribution"] == {
        "q_jitter_rad": [-0.05, 0.05],
        "qd_rad_s": [-0.25, 0.25],
        "control_effort_fraction": [-0.20, 0.20],
    }


def test_mode_support_requires_two_unique_candidates_and_pairwise_topology():
    center = np.array([0.5, -0.2])
    effort = np.ones(7)
    rows = []
    for mode_index, mode in enumerate(("clockwise", "counterclockwise")):
        for variant in range(2):
            controls = np.full((8, 7), 0.01 * (1 + mode_index * 2 + variant))
            rows.append(
                {
                    "candidate_index": len(rows),
                    "productive_certified": True,
                    "external_cost": 10.0 + mode_index + 0.1 * variant,
                    "controls": controls,
                    "cuda_tool_path": semicircle(
                        center, 0.14 + 0.01 * variant, mode
                    ),
                }
            )
    certificate = certify_mode_support(rows, center, effort)

    assert certificate["two_mode_support"]
    assert certificate["support"] == {"clockwise": 2, "counterclockwise": 2}
    assert certificate["all_cross_mode_windings_nonzero_integer"]
    assert certificate["all_same_mode_windings_zero_integer"]
    assert certificate["supported_candidate_indices_by_mode"] == {
        "clockwise": [0, 1],
        "counterclockwise": [2, 3],
    }
    assert mode_certificates_agree(certificate, copy.deepcopy(certificate))

    duplicate = dict(rows[2])
    duplicate["candidate_index"] = 99
    insufficient = certify_mode_support(rows[:3] + [duplicate], center, effort)
    assert insufficient["support"]["counterclockwise"] == 1
    assert not insufficient["two_mode_support"]


def test_mode_agreement_rejects_equal_counts_with_different_pairwise_class():
    center = np.array([0.5, -0.2])
    rows = []
    for mode_index, mode in enumerate(("clockwise", "counterclockwise")):
        for variant in range(2):
            rows.append(
                {
                    "candidate_index": len(rows),
                    "productive_certified": True,
                    "external_cost": 10.0 + mode_index + variant,
                    "controls": np.full((8, 7), 0.01 * (len(rows) + 1)),
                    "cuda_tool_path": semicircle(
                        center, 0.14 + 0.01 * variant, mode
                    ),
                }
            )
    certificate = certify_mode_support(rows, center, np.ones(7))
    mismatched_pair_class = copy.deepcopy(certificate)
    mismatched_pair_class["cross_mode_pairwise_closed_loop_winding"][0][
        "nearest_integer"
    ] *= -1

    assert mismatched_pair_class["support"] == certificate["support"]
    assert (
        mismatched_pair_class["all_cross_mode_windings_nonzero_integer"]
        == certificate["all_cross_mode_windings_nonzero_integer"]
    )
    assert not mode_certificates_agree(certificate, mismatched_pair_class)


def test_paired_mode_cost_summary_enforces_support_and_suboptimal_gap():
    supported = [
        {
            "method_seed": seed,
            "two_mode_support": True,
            "best_external_cost_by_mode": {
                "clockwise": 10.0 + 0.01 * index,
                "counterclockwise": 11.0 + 0.01 * index,
            },
        }
        for index, seed in enumerate(FINAL_METHOD_SEEDS)
    ]
    accepted = paired_mode_cost_summary(supported)

    assert accepted["paired_support_count"] == 20
    assert accepted["suboptimal_mode"] == "counterclockwise"
    assert accepted["paired_median_suboptimal_gap_fraction"] > 0.05
    assert accepted["suboptimal_mode_worse_fraction"] == 1.0
    assert accepted["passes"]

    for row in supported[14:]:
        row["two_mode_support"] = False
    rejected = paired_mode_cost_summary(supported)
    assert rejected["paired_support_count"] == 14
    assert not rejected["passes"]
