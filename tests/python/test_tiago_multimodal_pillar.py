import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from gato_tiago.multimodal_pillar import (
    DIFFICULTIES,
    candidate_specs,
    certify_results,
    generate_instance,
    load_model,
    pillar_residual_and_gradient,
    reference_batch,
    task_seed_path,
    winding_signature,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_artifact_json_conversion_rejects_nonstandard_nan_tokens():
    namespace = runpy.run_path(
        str(REPO_ROOT / "tiago_examples/tiago_multimodal_pillar.py")
    )
    converted = namespace["_jsonable"](
        {"python": float("nan"), "numpy": np.array([np.inf, -np.inf, 1.0])}
    )
    assert converted == {"python": None, "numpy": [None, None, 1.0]}
    encoded = json.dumps(converted, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_pillar_residual_gradient_matches_finite_difference_inside_keepout():
    point = np.array([0.51, -0.17])
    center = np.array([0.48, -0.20])
    radius = 0.12
    residual, gradient = pillar_residual_and_gradient(point, center, radius)
    eps = 1e-7
    finite_difference = np.empty(2)
    for axis in range(2):
        step = np.zeros(2)
        step[axis] = eps
        plus = pillar_residual_and_gradient(point + step, center, radius)[0]
        minus = pillar_residual_and_gradient(point - step, center, radius)[0]
        finite_difference[axis] = (plus - minus) / (2.0 * eps)

    assert residual > 0.0
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-7, atol=1e-7)


def test_pillar_residual_is_zero_with_zero_gradient_outside_keepout():
    residual, gradient = pillar_residual_and_gradient(
        [0.75, -0.2], [0.48, -0.2], 0.12
    )
    assert residual == 0.0
    np.testing.assert_array_equal(gradient, np.zeros(2))


def test_reference_is_identical_for_every_candidate():
    model = load_model()
    difficulty = DIFFICULTIES["easy"]
    instance = generate_instance(23, difficulty, model=model)
    reference = reference_batch(instance, difficulty.knots, batch_size=8)

    assert reference.shape == (8, difficulty.knots * 6)
    for row in reference[1:]:
        np.testing.assert_array_equal(row, reference[0])
    np.testing.assert_allclose(reference[0, :6], instance.reference_row())


def test_family_varies_start_posture_and_does_not_encode_side_as_seed_parity():
    model = load_model()
    difficulty = DIFFICULTIES["easy"]
    instances = [generate_instance(seed, difficulty, model=model) for seed in range(12)]
    starts = {tuple(np.round(instance.start_q, 5)) for instance in instances}
    offsets = []
    for instance in instances:
        start = np.asarray(instance.start_position)[:2]
        goal = np.asarray(instance.goal_position)[:2]
        direction = goal - start
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        line_mid = 0.5 * (start + goal)
        offsets.append(float((np.asarray(instance.pillar_xy) - line_mid) @ normal))

    assert len(starts) == len(instances)
    assert any(offset > 0.0 for offset in offsets[::2])
    assert any(offset < 0.0 for offset in offsets[::2])
    assert any(offset > 0.0 for offset in offsets[1::2])
    assert any(offset < 0.0 for offset in offsets[1::2])


def test_route_seeds_pass_on_opposite_sides_of_the_pillar():
    model = load_model()
    difficulty = DIFFICULTIES["medium"]
    instance = generate_instance(31, difficulty, model=model)
    clockwise = task_seed_path(instance, difficulty, route_side=1)
    counterclockwise = task_seed_path(instance, difficulty, route_side=-1)
    midpoint = difficulty.knots // 2
    start = np.asarray(instance.start_position)
    goal = np.asarray(instance.goal_position)
    direction = goal[:2] - start[:2]
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    center = np.asarray(instance.pillar_xy)

    assert (clockwise[midpoint, :2] - center) @ normal > instance.clearance_radius_m
    assert (counterclockwise[midpoint, :2] - center) @ normal < -instance.clearance_radius_m
    assert np.linalg.norm(
        np.asarray(instance.goal_position) - np.asarray(instance.start_position)
    ) > 0.25


def test_winding_signature_separates_the_two_semicircles():
    center = np.array([0.5, -0.2])
    radius = 0.15
    clockwise_angles = np.linspace(np.pi, 0.0, 80)
    counterclockwise_angles = np.linspace(-np.pi, 0.0, 80)

    def path(angles):
        xy = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        return np.column_stack([xy, np.zeros(angles.size)])

    clockwise_label, clockwise_angle = winding_signature(path(clockwise_angles), center)
    counterclockwise_label, counterclockwise_angle = winding_signature(
        path(counterclockwise_angles), center
    )

    assert clockwise_label == "clockwise"
    assert clockwise_angle < -3.0
    assert counterclockwise_label == "counterclockwise"
    assert counterclockwise_angle > 3.0


def test_certificate_requires_certifiable_support_in_both_modes():
    center = np.array([0.5, -0.2])
    radius = 0.1

    def result(mode, objective, variant):
        angle = np.linspace(np.pi, 0.0, 8)
        if mode == "counterclockwise":
            angle = np.linspace(-np.pi, 0.0, 8)
        path_radius = 0.14 + 0.01 * variant
        xy = center + path_radius * np.column_stack([np.cos(angle), np.sin(angle)])
        return {
            "mode": mode,
            "objective": objective,
            "nonnegative_task_motion_cost": objective,
            "certifiable": True,
            "tool_path": np.column_stack([xy, np.zeros(angle.size)]),
            "pillar_xy": center,
            "clearance_radius_m": radius,
        }

    insufficient = [
        result("clockwise", 10.0, 0),
        result("clockwise", 10.2, 1),
        result("counterclockwise", 11.0, 0),
    ]
    assert not certify_results(insufficient)["finite_budget_two_mode_coverage"]
    one_mode = certify_results(insufficient[:2])
    assert one_mode["best_known_mode"] == "clockwise"
    assert one_mode["best_known_solver_objective"] == 10.0

    certified = insufficient + [result("counterclockwise", 11.3, 1)]
    certificate = certify_results(certified)
    assert certificate["finite_budget_two_mode_coverage"]
    assert certificate["ranked_finite_budget_benchmark"]
    assert certificate["certifiable_support"] == {
        "clockwise": 2,
        "counterclockwise": 2,
    }
    assert certificate["best_known_solver_objective"] == 10.0
    assert certificate["best_known_mode"] == "clockwise"
    assert certificate["mode_objective_gap_absolute"] == 1.0
    assert certificate["mode_objective_gap_fraction"] == pytest.approx(1.0 / 10.5)
    assert certificate["cross_mode_midpoint_min_clearance_m"] < 0.0

    duplicated = certified[:3] + [dict(certified[2])]
    duplicate_certificate = certify_results(duplicated)
    assert duplicate_certificate["raw_certifiable_candidates"]["counterclockwise"] == 2
    assert duplicate_certificate["certifiable_support"]["counterclockwise"] == 1
    assert not duplicate_certificate["finite_budget_two_mode_coverage"]


def test_candidate_budgets_use_only_supported_batch_sizes():
    for budget in (1, 2, 4, 8):
        assert len(candidate_specs(budget)) == budget
