import inspect
from pathlib import Path
import re

import numpy as np
import pytest

from gato_tiago import circular_portfolio as schema
from gato_tiago import circular_portfolio_constructor as constructor
from gato_tiago import circular_portfolio_runner as runner
from gato_tiago import circular_portfolio_worker as worker


def _geometry():
    d = 0.14
    start = np.array([-d / 2, 0.0, 0.5])
    goal = np.array([d / 2, 0.0, 0.51])
    return schema.construct_geometry(start, goal, 1)


def test_exact_12_task_192_profile_ledger_and_only_campaign_tokens_enabled():
    assert schema.TASK_IDENTITIES == tuple(
        [("development", seed) for seed in range(12600, 12604)]
        + [("heldout", seed) for seed in range(12700, 12708)]
    )
    assert len(schema.EXPECTED_LEDGER) == 12 * 16 == 192
    assert schema.EXPECTED_LEDGER[0] == ("development", 12600, "short", 0)
    assert schema.EXPECTED_LEDGER[8] == ("development", 12600, "long", 0)
    assert schema.EXPECTED_LEDGER[-1] == ("heldout", 12707, "long", 7)
    assert schema.CONSTRUCTOR_EXECUTION_AUTHORIZATION is None
    assert schema.OPTIMIZER_EXECUTION_AUTHORIZATION is None
    assert constructor.RUNNER_EXECUTION_AUTHORIZATION is not None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is not None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is not None
    root = Path(__file__).resolve().parents[2]
    pattern = re.compile(r"^[A-Z][A-Z0-9_]*AUTHORIZATION = object\(\)$")
    enabled = [
        (path.relative_to(root).as_posix(), line)
        for path in (root / "tiago_src/gato_tiago").glob("*.py")
        for line in path.read_text().splitlines()
        if pattern.fullmatch(line)
    ]
    assert set(enabled) == {
        ("tiago_src/gato_tiago/circular_portfolio_constructor.py",
         "RUNNER_EXECUTION_AUTHORIZATION = object()"),
        ("tiago_src/gato_tiago/circular_portfolio_runner.py",
         "RUNNER_EXECUTION_AUTHORIZATION = object()"),
        ("tiago_src/gato_tiago/circular_portfolio_worker.py",
         "WORKER_EXECUTION_AUTHORIZATION = object()"),
    }


def test_exact_c2_profiles_are_distinct_monotone_and_zero_endpoint_derivatives():
    assert schema.PROFILE_BETA_NUMERATOR == (0, -3, -1, -1, 1, 1, 3, 1)
    assert schema.PROFILE_BETA_DENOMINATOR == (1, 8, 4, 8, 8, 4, 8, 16)
    tau = np.linspace(0.0, 1.0, 10001)
    rows = []
    for index in range(8):
        progress = schema.profile_progress(tau, index)
        rows.append(progress)
        assert progress[0] == 0.0 and progress[-1] == 1.0
        assert np.all(np.diff(progress) > 0.0)
        assert np.min(schema.profile_derivative_multiplier(
            schema.quintic_progress(tau), index
        )) >= 5.0 / 8.0
        h = 1e-5
        near = schema.profile_progress(np.array([0, h, 2*h, 1-2*h, 1-h, 1]), index)
        assert abs((near[1]-near[0])/h) < 1e-7
        assert abs((near[-1]-near[-2])/h) < 1e-7
    assert len({row.tobytes() for row in rows}) == 8


def test_geometry_has_complementary_160_200_arcs_and_frozen_chord_pillar_math():
    geometry = _geometry()
    assert schema.certify_geometry(geometry)["passes"]
    d = np.linalg.norm(geometry.goal_xyz[:2] - geometry.start_xyz[:2])
    assert geometry.circle_radius_m == pytest.approx(d / (2*np.sin(4*np.pi/9)))
    assert geometry.chord_offset_m == pytest.approx(d / (2*np.tan(4*np.pi/9)))
    assert np.allclose(
        geometry.pillar_xy,
        geometry.midpoint_xy - geometry.chord_offset_m * geometry.short_unit_xy,
    )
    for route, angle in (("short", 8*np.pi/9), ("long", 10*np.pi/9)):
        path = schema.circular_reference(geometry, route, 0)
        assert path.shape == (96, 3)
        assert np.allclose(path[0], geometry.start_xyz, atol=1e-12)
        assert np.allclose(path[-1], geometry.goal_xyz, atol=1e-12)
        radii = np.linalg.norm(path[:, :2] - geometry.pillar_xy, axis=1)
        assert np.max(np.abs(radii - geometry.circle_radius_m)) < 1e-12
        assert angle == pytest.approx(schema.SHORT_ANGLE_RAD if route == "short" else schema.LONG_ANGLE_RAD)


def test_geometry_fails_physical_chord_and_toll_length_mutations():
    geometry = _geometry()
    bad = schema.CircularGeometry(
        geometry.start_xyz, geometry.goal_xyz, geometry.midpoint_xy,
        geometry.chord_unit_xy, geometry.normal_xy, geometry.short_unit_xy,
        geometry.pillar_xy, geometry.circle_radius_m, 0.031,
        geometry.toll_length_m, geometry.orientation_sign,
    )
    assert not schema.certify_geometry(bad)["passes"]
    bad = schema.CircularGeometry(
        geometry.start_xyz, geometry.goal_xyz, geometry.midpoint_xy,
        geometry.chord_unit_xy, geometry.normal_xy, geometry.short_unit_xy,
        geometry.pillar_xy, geometry.circle_radius_m,
        geometry.chord_offset_m, -1e-3, geometry.orientation_sign,
    )
    assert not schema.certify_geometry(bad)["passes"]


def test_ref10_smoothstep_toll_is_c1_saturated_psd_and_terminal_off():
    geometry = _geometry()
    reference = schema.reference_from_geometry(geometry).as_float32()
    assert reference.shape == (10,) and reference.dtype == np.float32
    center = reference[3:5].astype(float)
    short = reference[6:8].astype(float)
    ell = float(reference[8])
    offset = float(reference[5] + reference[9] - 2*reference[8])
    midpoint = center + offset * short
    for z, expected in ((-1, 0), (0, 0), (0.5, 0.5), (1, 1), (2, 1)):
        residual, gradient, gn = schema.smoothstep_toll_residual_gradient(
            midpoint + z*ell*short, reference
        )
        assert residual == pytest.approx(expected)
        assert np.min(np.linalg.eigvalsh(gn)) >= -1e-12
        if z in (-1, 0, 1, 2): assert np.array_equal(gradient, np.zeros(2))
    probe = midpoint + .37*ell*short
    residual, gradient, _gn = schema.smoothstep_toll_residual_gradient(probe, reference)
    direction = np.array([.4, -.7]); step = 1e-7
    plus = schema.smoothstep_toll_residual_gradient(probe+step*direction, reference)[0]
    minus = schema.smoothstep_toll_residual_gradient(probe-step*direction, reference)[0]
    assert (plus-minus)/(2*step) == pytest.approx(gradient@direction, rel=1e-7, abs=1e-9)
    eps = 1e-7
    for z in (eps, 1-eps):
        _r, gradient, _gn = schema.smoothstep_toll_residual_gradient(
            midpoint + z*ell*short, reference
        )
        assert np.linalg.norm(gradient) < 1e-3
    cost, gradient, gn = schema.toll_cost_gradient_gn(midpoint+.5*ell*short, reference, terminal=True)
    assert cost == 0 and np.array_equal(gradient, np.zeros(2)) and np.array_equal(gn, np.zeros((2,2)))


def test_reference_rejects_wrong_width_dtype_radius_margin_and_unit():
    row = schema.reference_from_geometry(_geometry()).as_float32()
    for bad in (row[:-1], row.astype(np.float64)):
        with pytest.raises(ValueError): schema.validate_reference_row(bad)
    for index, value in ((5, .031), (9, .006), (6, 2.0), (8, -1.0)):
        bad = row.copy(); bad[index] = value
        with pytest.raises(ValueError): schema.validate_reference_row(bad)


def test_recurrence_dimensions_endpoint_maps_and_exact_proxy_indexing():
    assert schema.SHOOTING_VARIABLES == 665
    assert schema.ENDPOINT_EQUALITIES == 14
    assert schema.SHOOTING_INEQUALITIES == 4114
    qdd = np.arange(665, dtype=float).reshape(95, 7) * 1e-6
    q, qd = constructor.integrate_acceleration(np.zeros(7), np.zeros(7), qdd)
    dq, dv = constructor.recurrence_sensitivities()
    direction = np.cos(np.arange(665)).reshape(95, 7) * 1e-5
    qp, vp = constructor.integrate_acceleration(np.zeros(7), np.zeros(7), qdd+direction)
    assert np.allclose(qp-q, np.einsum("kij,j->ki", dq, direction.ravel()), atol=1e-14)
    assert np.allclose(vp-qd, np.einsum("kij,j->ki", dv, direction.ravel()), atol=1e-14)
    calls = []
    def kinematics(value):
        calls.append(value.copy()); return value[:3], np.c_[np.eye(3), np.zeros((3,4))]
    target = np.zeros((96, 3)); goal = np.ones(7)*.01
    history = constructor.proxy_joint_path(np.zeros(7), goal, target, kinematics)
    proxy = history["proxy_q_float64"]
    assert np.array_equal(proxy[0], np.zeros(7)) and np.array_equal(proxy[-1], goal)
    assert len(calls) == 94 * 8
    assert constructor.certify_proxy_construction(
        history, np.zeros(7), goal, target,
        lambda value: (value[:3], np.c_[np.eye(3), np.zeros((3,4))]),
    )["passes"]
    bad = {name: value.copy() for name, value in history.items()}
    bad["proxy_q_before_float64"][0, 0, 0] += 1e-6
    assert not constructor.certify_proxy_construction(
        bad, np.zeros(7), goal, target,
        lambda value: (value[:3], np.c_[np.eye(3), np.zeros((3,4))]),
    )["passes"]


def test_inequality_shape_and_mutations_are_fail_closed():
    q = np.zeros((96,7)); qd = np.zeros((96,7)); u = np.zeros((95,7))
    p = np.zeros((96,3)); p[:,0] = .05
    values = constructor.shooting_inequalities(
        q, qd, u, p, -np.ones(7), np.ones(7), np.ones(7), np.ones(7), np.zeros(2)
    )
    assert values.shape == (4114,) and np.min(values) >= 0
    p[:,0] = .039
    assert np.min(constructor.shooting_inequalities(
        q, qd, u, p, -np.ones(7), np.ones(7), np.ones(7), np.ones(7), np.zeros(2)
    )) < 0


def test_exact_shooting_first_derivatives_match_directional_finite_differences():
    rng = np.random.default_rng(44)
    point = rng.normal(scale=1e-3, size=665)
    direction = np.sin(np.arange(665, dtype=float))
    q0 = np.zeros(7); effort = np.ones(7); pillar = np.array([.2, .2])
    reference = np.zeros((96,3))
    J = np.c_[np.eye(3), np.zeros((3,4))]
    def kinematics(q): return np.asarray(q[:3]), J
    def rnea(q, qd, qdd): return np.asarray(qdd)
    def rnea_derivatives(q, qd, qdd):
        return np.zeros((7,7)), np.zeros((7,7)), np.eye(7)
    def evaluate(flat):
        qdd = flat.reshape(95,7)
        q, qd = constructor.integrate_acceleration(q0, np.zeros(7), qdd)
        u = constructor.reconstruct_controls(q, qd, qdd, rnea)
        p, dp, du = constructor.position_and_control_sensitivities(
            q, qd, qdd, kinematics, rnea_derivatives
        )
        return qdd, q, qd, u, p, dp, du
    def objective(flat):
        qdd, q, qd, u, p, _dp, _du = evaluate(flat)
        return constructor.shooting_objective(q, qdd, u, p, reference, effort)
    def gradient(flat):
        qdd, _q, _qd, u, p, dp, du = evaluate(flat)
        return constructor.shooting_objective_gradient(
            qdd, u, p, reference, effort, dp, du
        )
    assert constructor.directional_derivative_gate(
        objective, gradient, point, direction
    )["passes"]
    qdd, q, qd, u, p, dp, du = evaluate(point)
    analytic = constructor.shooting_inequality_jacobian(q, qd, u, p, pillar, dp, du)
    def inequality(flat):
        _qdd, q, qd, u, p, _dp, _du = evaluate(flat)
        return constructor.shooting_inequalities(
            q, qd, u, p, -np.ones(7), np.ones(7), np.ones(7), effort, pillar
        )
    assert constructor.directional_derivative_gate(
        inequality, lambda _flat: analytic, point, direction
    )["passes"]
    endpoint_jacobian = constructor.shooting_endpoint_jacobian()
    assert endpoint_jacobian.shape == (14, 665)
    assert endpoint_jacobian[0, 0] == pytest.approx(
        schema.DT**2 * (schema.INTERVALS - 0.5)
    )
    assert endpoint_jacobian[7, 0] == pytest.approx(schema.DT)


def test_pin_upper_mass_is_symmetrized_and_gated_against_rnea_acceleration_fd():
    mass = np.diag(np.arange(1.0, 8.0))
    mass[0, 3] = mass[3, 0] = .25
    pin_upper_with_garbage = np.triu(mass) + np.tril(np.full((7, 7), 99.0), -1)
    def derivatives(_q, _v, _a):
        return np.zeros((7, 7)), np.zeros((7, 7)), pin_upper_with_garbage
    def rnea(_q, _v, acceleration):
        return mass @ acceleration
    detail = constructor.certify_rnea_acceleration_derivative(
        np.zeros(7), np.zeros(7), np.linspace(-.1, .1, 7), rnea, derivatives
    )
    assert detail["passes"] and np.array_equal(detail["mass_float64"], mass)
    q = np.zeros((schema.KNOTS, 7)); qd = q.copy(); qdd = np.zeros((schema.INTERVALS, 7))
    zero_detail = constructor.certify_rnea_acceleration_derivative(
        np.zeros(7), np.zeros(7), np.zeros(7), rnea, derivatives
    )
    retained = {
        "rnea_mass_float64": np.repeat(mass[None], schema.INTERVALS, axis=0),
        "rnea_mass_fd_float64": np.repeat(zero_detail["mass_fd_float64"][None], schema.INTERVALS, axis=0),
        "rnea_mass_maxabs_float64": np.asarray(zero_detail["maxabs"]),
    }
    assert constructor.certify_retained_mass_diagnostics(
        retained, q, qd, qdd, rnea, derivatives
    )["passes"]
    mutated = {name: value.copy() for name, value in retained.items()}
    mutated["rnea_mass_float64"][5, 0, 3] += 1e-5
    assert not constructor.certify_retained_mass_diagnostics(
        mutated, q, qd, qdd, rnea, derivatives
    )["passes"]
    bad_mass = mass.copy(); bad_mass[0, 3] += .01
    def bad_derivatives(_q, _v, _a):
        return np.zeros((7, 7)), np.zeros((7, 7)), np.triu(bad_mass)
    assert not constructor.certify_rnea_acceleration_derivative(
        np.zeros(7), np.zeros(7), np.zeros(7), rnea, bad_derivatives
    )["passes"]
    retained_one = {
        "dq_float64": np.zeros((7,7)), "dv_float64": np.zeros((7,7)),
        "mass_float64": mass, "mass_fd_float64": zero_detail["mass_fd_float64"],
    }
    assert constructor.certify_full_rnea_derivatives(
        retained_one, np.zeros(7), np.zeros(7), np.zeros(7), rnea, derivatives
    )["passes"]
    retained_one["dv_float64"] = retained_one["dv_float64"].copy()
    retained_one["dv_float64"][0,0] = 1e-6
    assert not constructor.certify_full_rnea_derivatives(
        retained_one, np.zeros(7), np.zeros(7), np.zeros(7), rnea, derivatives
    )["passes"]


def test_proxy_linear_kkt_is_fully_retained_regenerated_and_mutation_bound():
    proxy = np.zeros((schema.KNOTS, 7))
    retained = constructor.linear_proxy_initial_acceleration(
        np.zeros(7), np.zeros(7), proxy
    )
    matrix, condition = constructor.proxy_kkt_matrix()
    global_arrays = {"proxy_kkt_matrix_float64": matrix,
                     "proxy_kkt_condition_float64": condition}
    detail = constructor.certify_proxy_kkt(
        retained, np.zeros(7), np.zeros(7), proxy, global_arrays
    )
    assert detail["passes"]
    assert matrix.shape == (679, 679)
    bad = {name: value.copy() for name, value in retained.items()}
    bad["proxy_kkt_rhs_float64"][0] += 1e-8
    assert not constructor.certify_proxy_kkt(
        bad, np.zeros(7), np.zeros(7), proxy, global_arrays
    )["passes"]
    bad_global = {name: value.copy() for name, value in global_arrays.items()}
    bad_global["proxy_kkt_matrix_float64"][0,0] += 1e-10
    assert not constructor.certify_proxy_kkt(
        retained, np.zeros(7), np.zeros(7), proxy, bad_global
    )["passes"]


def test_sparse_derivative_diagnostic_binds_canonical_csr_order_and_dense_bytes():
    from scipy.sparse import csr_matrix
    dense = np.asarray([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    csr = csr_matrix(dense)
    retained = {
        "data_float64": csr.data.astype(np.float64),
        "indices_int32": csr.indices.astype(np.int32),
        "indptr_int32": csr.indptr.astype(np.int32),
        "shape_int64": np.asarray(csr.shape, np.int64),
    }
    assert constructor.certify_sparse_derivative_diagnostic(dense, csr, retained)["passes"]
    bad = {name: value.copy() for name, value in retained.items()}
    bad["indices_int32"][[0, 1]] = bad["indices_int32"][[1, 0]]
    assert not constructor.certify_sparse_derivative_diagnostic(dense, csr, bad)["passes"]


def test_full_three_probe_derivative_evidence_binds_q_v_objective_and_true_csr():
    from scipy.sparse import csr_matrix
    dense = np.zeros((schema.SHOOTING_INEQUALITIES, schema.SHOOTING_VARIABLES))
    rows = np.arange(schema.SHOOTING_INEQUALITIES)
    dense[rows, rows % schema.SHOOTING_VARIABLES] = 1.0
    csr = csr_matrix(dense)
    arrays = {
        name: np.zeros(shape, dtype)
        for name, (shape, dtype) in constructor.DERIVATIVE_ARRAY_SPECS.items()
    }
    arrays["derivative_probe_indices_int64"] = np.asarray(
        constructor.DERIVATIVE_PROBE_INDICES, np.int64
    )
    arrays["derivative_endpoint_jacobian_float64"] = constructor.shooting_endpoint_jacobian()
    arrays["derivative_probe0_inequality_dense_float64"] = dense.copy()
    evaluations = []
    for probe in range(3):
        point = np.full(schema.SHOOTING_VARIABLES, .001 * (probe + 1))
        direction = np.sin(np.arange(schema.SHOOTING_VARIABLES) + probe)
        arrays["derivative_objective_point_float64"][probe] = point
        arrays["derivative_objective_direction_float64"][probe] = direction
        arrays["derivative_objective_gradient_float64"][probe] = point
        arrays["derivative_rnea_mass_float64"][probe] = np.eye(7)
        arrays["derivative_rnea_mass_fd_float64"][probe] = np.eye(7)
        base = f"derivative_probe{probe}_inequality_csr_"
        arrays[base+"data_float64"] = csr.data.astype(np.float64)
        arrays[base+"indices_int32"] = csr.indices.astype(np.int32)
        arrays[base+"indptr_int32"] = csr.indptr.astype(np.int32)
        arrays[base+"shape_int64"] = np.asarray(csr.shape, np.int64)
        arrays[base+"nnz_int64"] = np.asarray(csr.nnz, np.int64)
        evaluations.append({
            "q": np.zeros(7), "qd": np.zeros(7), "qdd": np.zeros(7),
            "rnea": lambda _q, _v, a: np.asarray(a),
            "rnea_derivatives": lambda _q, _v, _a: (
                np.zeros((7,7)), np.zeros((7,7)), np.eye(7)),
            "objective": lambda value: .5 * float(value @ value),
            "objective_gradient": lambda value: np.asarray(value),
            "inequality": lambda value, matrix=dense: matrix @ value,
            "inequality_jacobian": lambda _value, matrix=dense: matrix,
        })
    assert constructor.certify_production_derivative_evidence(arrays, evaluations)["passes"]
    bad = {name: value.copy() for name, value in arrays.items()}
    bad["derivative_rnea_dv_float64"][1,0,0] = 1e-4
    assert not constructor.certify_production_derivative_evidence(bad, evaluations)["passes"]
    bad = {name: value.copy() for name, value in arrays.items()}
    bad["derivative_probe2_inequality_csr_data_float64"][0] += 1e-4
    assert not constructor.certify_production_derivative_evidence(bad, evaluations)["passes"]


def test_replay_recomputes_route_turn_circular_error_reference_and_jqd_speed():
    geometry = _geometry()
    canonical = np.tile(
        schema.reference_from_geometry(geometry).as_float32(), (schema.KNOTS, 1)
    ).ravel()
    limits = np.ones(7)
    for route in ("short", "long"):
        solver_reference = schema.circular_reference(geometry, route, 0)
        dense = constructor._dense_reference(solver_reference)
        state = np.zeros((schema.DENSE_SAMPLES, 14))
        controls = np.zeros((schema.INTERVALS, 7))
        row = {
            "state": state, "control": controls, "tool": dense,
            "joint_lower": -limits, "joint_upper": limits,
            "velocity_limit": limits, "effort_limit": limits,
            "pillar_xy": geometry.pillar_xy, "physical_radius_m": .03,
            "x0": np.zeros(14), "reference_float32": canonical,
        }
        assert constructor.certify_independent_replay(
            row, {name: value.copy() if isinstance(value, np.ndarray) else value
                  for name, value in row.items()},
            solver_reference, route, canonical,
            lambda _model, _q, qd: qd[:3],
        )["passes"]
        bad = {name: value.copy() if isinstance(value, np.ndarray) else value
               for name, value in row.items()}
        bad["tool"][100, 0] += .005
        assert not constructor.certify_independent_replay(
            row, bad, solver_reference, route, canonical,
            lambda _model, _q, qd: qd[:3],
        )["passes"]
        assert not constructor.certify_independent_replay(
            row, row, solver_reference, route, canonical.copy(),
            lambda _model, _q, _qd: np.ones(3),
        )["passes"]


def test_seed_reversal_uses_both_models_actual_paths_costs_and_saturation():
    geometry = _geometry()
    reference = np.tile(
        schema.reference_from_geometry(geometry).as_float32(), (schema.KNOTS, 1)
    ).ravel()
    limits = np.ones(7)
    controls = np.zeros((schema.INTERVALS, 7))
    def retained(route):
        path = constructor._dense_reference(
            schema.circular_reference(geometry, route, 0)
        )
        state = np.zeros((schema.DENSE_SAMPLES, 14))
        row = {
            "pin_path": path, "cuda_path": path.copy(),
            "pin_state": state, "cuda_state": state.copy(),
            "controls": controls, "reference_float32": reference,
            "joint_lower": -limits, "joint_upper": limits,
            "velocity_limit": limits, "effort_limit": limits,
        }
        for model in ("pin", "cuda"):
            base = constructor.reconstruct_portfolio_cost(
                state[::schema.DENSE_SUBSTEPS], controls,
                path[::schema.DENSE_SUBSTEPS], reference, -limits, limits,
                limits, limits, include_toll=False,
            )
            full = constructor.reconstruct_portfolio_cost(
                state[::schema.DENSE_SUBSTEPS], controls,
                path[::schema.DENSE_SUBSTEPS], reference, -limits, limits,
                limits, limits, include_toll=True,
            )
            row[f"{model}_base_cost"] = base["cost"]
            row[f"{model}_full_cost"] = full["cost"]
            row[f"{model}_toll_residual"] = full["toll_residual_float64"]
        return row
    short, long = retained("short"), retained("long")
    assert constructor.certify_seed_reversal(short, long)["passes"]
    bad = {name: item.copy() if isinstance(item, np.ndarray) else item
           for name, item in short.items()}
    bad["pin_path"][::schema.DENSE_SUBSTEPS, 0] += .1
    assert not constructor.certify_seed_reversal(bad, long)["passes"]
    bad = {name: item.copy() if isinstance(item, np.ndarray) else item
           for name, item in short.items()}
    bad["pin_full_cost"] += 1e-6
    assert not constructor.certify_seed_reversal(bad, long)["passes"]
    bad = {name: item.copy() if isinstance(item, np.ndarray) else item
           for name, item in long.items()}
    bad["reference_float32"][0] += 1e-3
    assert not constructor.certify_seed_reversal(short, bad)["passes"]


def test_independent_shooting_certificate_reconstructs_controls_positions_and_aba():
    q0 = np.zeros(7); q_goal = np.zeros(7); qdd = np.zeros((95,7))
    q, qd = constructor.integrate_acceleration(q0, np.zeros(7), qdd)
    def kinematics(value): return value[:3].copy(), np.c_[np.eye(3), np.zeros((3,4))]
    def rnea(q, qd, qdd): return qdd.copy()
    def aba(q, qd, u): return u.copy()
    controls = constructor.reconstruct_controls(q, qd, qdd, rnea)
    positions = np.stack([kinematics(value)[0] for value in q])
    inequalities = constructor.shooting_inequalities(
        q, qd, controls, positions, -np.ones(7), np.ones(7), np.ones(7),
        np.ones(7), np.array([1.0, 0.0]),
    )
    objective = constructor.shooting_objective(
        q, qdd, controls, positions, positions, np.ones(7)
    )
    result = constructor.ShootingResult(
        schema.EXPECTED_LEDGER[0], qdd, q, qd, controls, positions, objective,
        np.zeros(14), inequalities, True, 1, 2, 1.0,
    )
    assert constructor.certify_shooting_result(
        result, q0, q_goal, positions, -np.ones(7), np.ones(7), np.ones(7),
        np.ones(7), np.array([1.0,0.0]), kinematics, rnea, aba,
    )["passes"]
    bad = constructor.ShootingResult(
        result.identity, result.acceleration_float64, result.q_float64,
        result.qd_float64, result.controls_float64.copy(), result.positions_float64,
        result.objective, result.endpoint_residual_float64,
        result.inequalities_float64, True, 1, 2, 1.0,
    )
    bad.controls_float64[0,0] = 1.0
    assert not constructor.certify_shooting_result(
        bad, q0, q_goal, positions, -np.ones(7), np.ones(7), np.ones(7),
        np.ones(7), np.array([1.0,0.0]), kinematics, rnea, aba,
    )["passes"]


def test_b1_is_bitwise_b16_lane_zero_and_all_16_seeds_are_unique():
    x16 = np.zeros((16,14), np.float32)
    r16 = np.zeros((16,960), np.float32)
    z16 = np.zeros((16,2009), np.float32)
    z16[:,0] = np.arange(16)
    b16 = {"x0_float32":x16,"reference_float32":r16,"seed_xu_float32":z16}
    b1 = {name:value[:1].copy() for name,value in b16.items()}
    assert constructor.certify_b1_b16_binding(b1,b16)["passes"]
    b1["seed_xu_float32"][0,0] = 2
    assert not constructor.certify_b1_b16_binding(b1,b16)["passes"]
    bad = {name: value.astype(np.float64) for name, value in b1.items()}
    assert not constructor.certify_b1_b16_binding(bad, b16)["passes"]
    bad = {name: value.copy() for name, value in b16.items()}
    bad["reference_float32"][0, 0] = np.nan
    assert not constructor.certify_b1_b16_binding(b1, bad)["passes"]


def test_canonical_solver_seed_is_interleaved_at_exact_gato_offsets():
    q = np.arange(schema.KNOTS * 7, dtype=np.float32).reshape(schema.KNOTS, 7)
    qd = 1000 + q
    controls = 2000 + np.arange(schema.INTERVALS * 7, dtype=np.float32).reshape(schema.INTERVALS, 7)
    seed = constructor.pack_solver_seed(q, qd, controls)
    assert seed.shape == (2009,) and seed.dtype == np.float32
    for knot in (0, 1, 47, 94, 95):
        offset = knot * 21
        assert np.array_equal(seed[offset:offset+7], q[knot])
        assert np.array_equal(seed[offset+7:offset+14], qd[knot])
        if knot < 95:
            assert np.array_equal(seed[offset+14:offset+21], controls[knot])
    recovered = constructor.unpack_solver_seed(seed)
    assert all(np.array_equal(left, right) for left, right in zip(recovered, (q, qd, controls)))
    wrong = np.r_[q.ravel(), qd.ravel(), controls.ravel()].astype(np.float32)
    assert not np.array_equal(seed, wrong)


def test_cuda_worker_is_fail_closed_until_separate_n96_build_and_owns_rollout_counts(tmp_path):
    assert worker.FROZEN_EXTENSION_SHA256 == "b079410ade9e7de19ed3d4b7ed6f6ace27172cd442ccea0d5a46bb7277067a2f"
    assert worker.FROZEN_EXTENSION_SIZE_BYTES == 6_690_480
    assert worker.FROZEN_BUILD_HEAD == "2f1011da2a240fe8eae9ff25b2b3ee991c11c6c2"
    assert worker.FROZEN_CUDA_ARCH == "61-real"
    assert worker.EXPECTED_SIM_FORWARD_CALLS == 95 * 64
    assert worker.EXPECTED_TOOL_POSITION_CALLS == (schema.DENSE_SAMPLES + 15) // 16
    with pytest.raises(RuntimeError, match="blocked"):
        worker.execute_worker(tmp_path/"request.json", tmp_path/"output.json", authorization=object())
    source = inspect.getsource(worker.execute_worker)
    assert "module.BSQP_16_float" in source
    assert "solver.sim_forward" in source and "solver.tool_position" in source
    assert "sim_forward_calls += 1" in source
    assert "tool_position_calls += 1" in source
    assert "constructor_calls += 1" in source
    assert "output_path.exists() or npz_path.exists()" in source


def test_cuda_worker_lane_binding_rejects_identity_path_and_control_mutations():
    index = 9
    paths = worker.expected_worker_paths(index)
    x0 = np.arange(14, dtype=np.float32)
    controls = np.arange(95 * 7, dtype=np.float32).reshape(95, 7)
    request = {
        "identity": list(schema.EXPECTED_LEDGER[index]),
        "input_path": str(paths["input_path"]),
        "output_path": str(paths["json_path"]),
    }
    inputs = {"x0_float32": x0.copy(), "controls_float32": controls.copy()}
    outputs = {
        "captured_x0_float32": x0.copy(),
        "captured_controls_float32": controls.copy(),
    }
    assert worker.certify_lane_binding(
        index, paths["request_path"], request, inputs, outputs, x0, controls
    )
    mutations = []
    bad = dict(request); bad["identity"] = list(schema.EXPECTED_LEDGER[index + 1]); mutations.append((bad, inputs, outputs, paths["request_path"]))
    bad = dict(request); bad["input_path"] += ".swapped"; mutations.append((bad, inputs, outputs, paths["request_path"]))
    bad_inputs = {name: value.copy() for name, value in inputs.items()}; bad_inputs["controls_float32"][0, 0] += 1; mutations.append((request, bad_inputs, outputs, paths["request_path"]))
    bad_outputs = {name: value.copy() for name, value in outputs.items()}; bad_outputs["captured_x0_float32"][0] += 1; mutations.append((request, inputs, bad_outputs, paths["request_path"]))
    mutations.append((request, inputs, outputs, paths["request_path"].with_name("p1.worker.010.request.json")))
    assert all(not worker.certify_lane_binding(
        index, request_path, bad_request, bad_inputs, bad_outputs, x0, controls
    ) for bad_request, bad_inputs, bad_outputs, request_path in mutations)


def test_cuda_worker_certificate_rejects_retained_pin_forgery_without_exact_worker_files():
    request_arrays = {
        "x0_float32": np.zeros(14, np.float32),
        "controls_float32": np.zeros((95,7), np.float32),
    }
    arrays = {
        "captured_x0_float32": request_arrays["x0_float32"],
        "captured_controls_float32": request_arrays["controls_float32"],
        "cuda_dense_state_float32": np.zeros((schema.DENSE_SAMPLES,14), np.float32),
        "cuda_dense_tool_float32": np.zeros((schema.DENSE_SAMPLES,3), np.float32),
    }
    hashes = {name: __import__("hashlib").sha256(
        f"{value.dtype.str}|{value.shape}|".encode()+value.tobytes()).hexdigest()
        for name, value in arrays.items()}
    summary = {
        "protocol": worker.WORKER_PROTOCOL, "identity": list(schema.EXPECTED_LEDGER[0]),
        "request_path": "/tmp/request", "request_sha256": "0"*64,
        "input_path": "/tmp/input", "input_sha256": "1"*64,
        "module_name": worker.MODULE_NAME, "module_path": "/tmp/module.so",
        "extension_sha256": None, "extension_size_bytes": None,
        "KNOT_POINTS":96, "REFERENCE_SIZE":10,
        "TOOL_POSITION_FRAME":"arm_right_tool_joint_origin", "TOOL_POSITION_SIZE":3,
        "constructor_calls":1, "sim_forward_calls":worker.EXPECTED_SIM_FORWARD_CALLS,
        "tool_position_calls":worker.EXPECTED_TOOL_POSITION_CALLS,
        "solve_calls":0, "sqp_calls":0, "array_names":sorted(arrays),
        "array_hashes":hashes, "npz_path":"/tmp/out.npz", "npz_sha256":"2"*64,
        "certificate":{"forged":True},
    }
    assert not worker.certify_worker_output(
        summary, arrays, {"identity": summary["identity"]}, request_arrays
    )["passes"]


def test_exact_profile_retention_schema_and_all16_all12_aggregation():
    arrays = {
        name: np.zeros(shape, dtype)
        for name, (shape, dtype) in constructor.PROFILE_ARRAY_SPECS.items()
    }
    assert constructor.exact_profile_array_schema(arrays)
    bad = dict(arrays); bad.pop(next(iter(bad)))
    assert not constructor.exact_profile_array_schema(bad)
    bad = dict(arrays); bad["q_float64"] = bad["q_float64"].astype(np.float32)
    assert not constructor.exact_profile_array_schema(bad)
    rows = [
        {"route":route,"profile":profile,"attempt_count":1,"retained":True,
         "passes":True,"filtered":False,"replacement":None}
        for route in ("short","long") for profile in range(8)
    ]
    assert constructor.aggregate_task_profiles(
        rows, {"passes":True}, {"passes":True}
    )["passes"]
    assert not constructor.aggregate_task_profiles(
        rows[:15], {"passes":True}, {"passes":True}
    )["passes"]
    tasks = [
        {"identity":identity,"passes":True,"profile_count":16}
        for identity in schema.TASK_IDENTITIES
    ]
    assert constructor.aggregate_campaign(tasks)["passes"]
    tasks[-1]["profile_count"] = 15
    assert not constructor.aggregate_campaign(tasks)["passes"]


def test_backend_and_watchdog_are_exact_one_attempt_no_fallback():
    declaration = constructor.backend_declaration()
    assert declaration["backend"] == "scipy.optimize.minimize:trust-constr"
    assert declaration["options"] == schema.BACKEND_OPTIONS
    assert "curvature" not in declaration["options"]
    assert declaration["hess_argument"] == "scipy.optimize.BFGS()"
    assert declaration["attempts_per_profile"] == 1
    assert declaration["fallback"] is None
    assert set(declaration["threads"].values()) == {"1"}
    assert not constructor.watchdog(100.0, 1)["runtime_watchdog_rejected"]
    rejected = constructor.watchdog(113.0, 1)
    assert rejected["runtime_watchdog_rejected"] and rejected["permanent_rejection"]
    assert rejected["projected_s"] == pytest.approx(113*192)


def test_public_handoff_rejects_every_route_constructor_secret():
    identity = schema.TASK_IDENTITIES[0]
    accepted_reference = np.r_[np.zeros(3), np.zeros(7)].astype(np.float32)
    reference = np.zeros(960, np.float32)
    public = {
        "identity": identity, "default_side": 1,
        "x0_float32": np.zeros(14, np.float32),
        "reference_float32": reference,
        "seed_xu_float32": np.zeros(2009, np.float32),
    }
    canonical = {
        "identity": identity, "default_side": 1,
        "x0_float32": np.zeros(14, np.float32),
        "reference_float32": accepted_reference,
    }
    assert schema.validate_public_handoff(public, canonical)
    for name in schema.FORBIDDEN_PUBLIC_FIELDS:
        with pytest.raises(ValueError, match="constructor-only"):
            schema.validate_public_handoff({**public, name: None}, canonical)
    for name, value in (
        ("x0_float32", np.zeros(14, np.float64)),
        ("reference_float32", np.zeros(959, np.float32)),
        ("seed_xu_float32", np.full(2009, np.nan, np.float32)),
    ):
        with pytest.raises(ValueError):
            schema.validate_public_handoff({**public, name: value}, canonical)


def test_exact_accepted_prerequisite_authentication_is_owned_not_summary_supplied():
    source = inspect.getsource(runner.authenticate_cpu_prerequisite)
    for required in (
        "PREREQUISITE_ARTIFACT_PINS", "sha256_file(path)",
        "recertify_retained_prerequisite", "return_payload=True",
    ):
        assert required in source
    assert all(len(row["sha256"]) == 64 for row in runner.PREREQUISITE_ARTIFACT_PINS.values())
    assert not hasattr(schema, "certify_prerequisite_binding")


def test_owned_prerequisite_auth_rejects_refreshed_summary_hash_lie(monkeypatch):
    pins = {key: dict(value) for key, value in runner.PREREQUISITE_ARTIFACT_PINS.items()}
    monkeypatch.setattr(runner, "PREREQUISITE_ARTIFACT_PINS", pins)
    monkeypatch.setattr(Path, "is_file", lambda _self: True)
    monkeypatch.setattr(runner, "sha256_file", lambda path: next(
        row["sha256"] for row in pins.values() if Path(row["path"]) == Path(path)
    ))
    monkeypatch.setattr(
        "gato_tiago.circular_portfolio_prerequisite_runner.recertify_retained_prerequisite",
        lambda _path, **_kwargs: {"passes": False},
    )
    with pytest.raises(RuntimeError, match="recertification failed"):
        runner.authenticate_cpu_prerequisite()


def _provenance(final=False):
    hashes = {name: "a" * 64 for name in runner.SOURCE_PATHS}
    return {
        "protocol": schema.PROTOCOL_VERSION, "cwd": runner.AUTHORIZED_CWD,
        "orig_argv": list(runner.AUTHORIZED_ORIG_ARGV),
        "exact_command": " ".join(runner.AUTHORIZED_ORIG_ARGV),
        "git_head_at_start": "b" * 40,
        "git_head_at_end": "b" * 40 if final else None,
        "tracked_clean_at_start": True,
        "tracked_clean_at_end": True if final else None,
        "source_hashes_at_start": hashes,
        "source_hashes_at_end": hashes if final else None,
        "task_artifact_pins": dict(schema.TASK_ARTIFACT_PINS),
        "model_artifact_pins": dict(schema.MODEL_ARTIFACT_PINS),
        "cpu_prerequisite_artifact_pins": runner.PREREQUISITE_ARTIFACT_PINS,
        "build_commit": runner.FROZEN_BUILD_COMMIT,
        "cuda_arch": runner.FROZEN_CUDA_ARCH,
        "extension": runner.FROZEN_EXTENSION,
        "thread_environment": runner.REQUIRED_THREAD_ENVIRONMENT,
        "runtime_versions": {name: "1" for name in ("python", "numpy", "scipy", "pinocchio", "cuda")},
    }


def _checkpoint(generation):
    completed = max(0, min(192, generation - 3))
    stage = ("gen0" if generation == 0 else "prerequisite_authenticated" if generation == 1
             else "extension_authenticated" if generation == 2
             else "production_model_authenticated" if generation == 3
             else "profile_completed" if generation <= 195 else "honest_end_provenance")
    return {
        "protocol": schema.PROTOCOL_VERSION, "generation": generation,
        "stage": stage, "incomplete": True,
        "completed_identities": [list(row) for row in schema.EXPECTED_LEDGER[:completed]],
        "pending_identities": [list(row) for row in schema.EXPECTED_LEDGER[completed:]],
        "counters": runner._expected_counters(generation),
        "provenance": _provenance(final=generation == 196),
        "evidence_flags": {"oracle_evidence": False, "benchmark_evidence": False},
        "watchdog": constructor.watchdog(0.0, completed),
        "profile_certificate": ({"identity": list(schema.EXPECTED_LEDGER[generation - 4]),
                                  "gates": {"constructor": True, "worker": True,
                                            "replay": True}, "passes": True}
                                if 4 <= generation <= 195 else None),
        "array_names": list(runner._checkpoint_array_names(generation)),
        "array_hashes": {}, "npz_path": "/tmp/checkpoint.npz",
        "npz_sha256": "c" * 64,
        "task_array_hash_digest_hex": None,
        "model_array_hash_digest_hex": None,
        "geometry_digest_hex": None,
    }


def _checkpoint_arrays(generation):
    arrays = {}
    for name in runner._checkpoint_array_names(generation):
        if name == "generation_int64": arrays[name] = np.asarray(generation, np.int64)
        elif name == "completed_count_int64":
            arrays[name] = np.asarray(max(0, min(192, generation - 3)), np.int64)
        elif name == "incomplete_bool": arrays[name] = np.asarray(True, np.bool_)
        elif name == "task_authentication_bool": arrays[name] = np.asarray(generation >= 1, np.bool_)
        elif name == "model_authentication_bool": arrays[name] = np.asarray(generation >= 1, np.bool_)
        elif name == "geometry_authentication_bool": arrays[name] = np.asarray(generation >= 1, np.bool_)
        elif name == "cross_bind_bool": arrays[name] = np.asarray(True, np.bool_)
        elif name.endswith("digest_uint8"): arrays[name] = np.zeros(32, np.uint8)
        else:
            profile_name = next(key for key in constructor.PROFILE_ARRAY_SPECS if name.endswith("_" + key))
            shape, dtype = constructor.PROFILE_ARRAY_SPECS[profile_name]
            arrays[name] = np.zeros(shape, dtype)
    return arrays


def _bound_checkpoint(generation):
    document = _checkpoint(generation); arrays = _checkpoint_arrays(generation)
    document["array_hashes"] = {name: runner.array_hash(value) for name, value in arrays.items()}
    document["task_array_hash_digest_hex"] = (
        arrays["task_array_hash_digest_uint8"].tobytes().hex() if generation >= 1 else None
    )
    document["model_array_hash_digest_hex"] = (
        arrays["model_array_hash_digest_uint8"].tobytes().hex() if generation >= 1 else None
    )
    return document, arrays


def _owned_checkpoint(generation, geometry_digest="e"*64):
    import hashlib, json
    detail = {"task": {"passes": True, "hashes": ["a"]},
              "model": {"passes": True, "hashes": ["b"]},
              "cross_bind": {"passes": True}}
    document, arrays = _bound_checkpoint(generation)
    task_digest = hashlib.sha256(json.dumps(detail["task"], sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    model_digest = hashlib.sha256(json.dumps({"model": detail["model"],
        "cross_bind": detail["cross_bind"]}, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if generation >= 1:
        arrays["task_array_hash_digest_uint8"] = np.frombuffer(bytes.fromhex(task_digest), np.uint8).copy()
        document["task_array_hash_digest_hex"] = task_digest
    if generation >= 1:
        arrays["model_array_hash_digest_uint8"] = np.frombuffer(bytes.fromhex(model_digest), np.uint8).copy()
        document["model_array_hash_digest_hex"] = model_digest
    document["geometry_digest_hex"] = geometry_digest if generation >= 1 else None
    document["array_hashes"] = {name: runner.array_hash(value) for name, value in arrays.items()}
    return document, arrays, detail, geometry_digest


def test_transaction_checkpoint_provenance_manifest_and_no_overwrite(tmp_path):
    for generation in (0, 1, 2, 3, 4):
        row, arrays = _bound_checkpoint(generation)
        assert runner.certify_checkpoint(row, generation)
        assert runner.certify_checkpoint_arrays(row, arrays)
        bad = {**row, "unexpected": True}
        assert not runner.certify_checkpoint(bad, generation)
    for generation in (195, 196):
        row = _checkpoint(generation)
        row["array_hashes"] = {name: "d" * 64 for name in row["array_names"]}
        assert runner.certify_checkpoint(row, generation)
    bad, _arrays = _bound_checkpoint(3); bad["provenance"] = dict(bad["provenance"])
    bad["provenance"]["task_artifact_pins"] = {**schema.TASK_ARTIFACT_PINS, "v4.json": "0"*64}
    assert not runner.certify_checkpoint(bad, 3)
    bad, arrays = _bound_checkpoint(4)
    bad["counters"] = dict(bad["counters"]); bad["counters"]["optimizer_calls"] = 0
    assert not runner.certify_checkpoint(bad, 4)
    bad, arrays = _bound_checkpoint(4)
    bad["watchdog"] = dict(bad["watchdog"]); bad["watchdog"]["projected_s"] = 999.0
    assert not runner.certify_checkpoint(bad, 4)
    bad, arrays = _bound_checkpoint(4); arrays = dict(arrays)
    key = next(name for name in arrays if name.startswith("profile_"))
    arrays[key] = arrays[key].copy(); arrays[key].flat[0] = 1
    assert not runner.certify_checkpoint_arrays(bad, arrays)
    output = tmp_path / "p1.json"
    output.write_text("occupied")
    with pytest.raises(FileExistsError): runner.refuse_existing_artifacts(output)
    assert not runner.recertify_retained_transaction(output)["passes"]
    assert len(runner.expected_final_array_names()) == len(set(runner.expected_final_array_names()))


def test_checkpoint_auth_bytes_bind_owned_prerequisite_and_geometry_digests():
    detail = {"task": {"passes": True, "hashes": ["a"]},
              "model": {"passes": True, "hashes": ["b"]},
              "cross_bind": {"passes": True}}
    geometry_digest = "e" * 64
    for generation in (1, 2, 3):
        document, arrays = _bound_checkpoint(generation)
        def digest(value):
            import hashlib, json
            return hashlib.sha256(json.dumps(value, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
        task_digest = digest(detail["task"])
        model_digest = digest({"model": detail["model"], "cross_bind": detail["cross_bind"]})
        if generation >= 1:
            arrays["task_array_hash_digest_uint8"] = np.frombuffer(bytes.fromhex(task_digest), np.uint8).copy()
            document["task_array_hash_digest_hex"] = task_digest
        if generation >= 1:
            arrays["model_array_hash_digest_uint8"] = np.frombuffer(bytes.fromhex(model_digest), np.uint8).copy()
            document["model_array_hash_digest_hex"] = model_digest
        document["geometry_digest_hex"] = geometry_digest if generation >= 1 else None
        document["array_hashes"] = {name: runner.array_hash(value) for name, value in arrays.items()}
        assert runner.certify_checkpoint_authentication(document, arrays, detail, geometry_digest)
        arrays[next(name for name in arrays if name.endswith("digest_uint8"))][0] ^= 1
        assert not runner.certify_checkpoint_authentication(document, arrays, detail, geometry_digest)


def test_transaction_publishes_immutable_checkpoints_and_never_false_final(tmp_path):
    output = tmp_path / "run" / "p1.json"
    first = runner.publish_checkpoint(output, _checkpoint(0), _checkpoint_arrays(0))
    second_doc, second_arrays, detail, geometry_digest = _owned_checkpoint(1)
    second = runner.publish_checkpoint(
        output, second_doc, second_arrays, prerequisite_detail=detail,
        geometry_digest_hex=geometry_digest,
    )
    assert all(path.exists() for path in (*first[:2], *second[:2]))
    latest = __import__("json").loads(first[2].read_text())
    assert latest["generation"] == 1 and latest["incomplete"] is True
    with pytest.raises(FileExistsError):
        runner.publish_checkpoint(
            output, second_doc, second_arrays, prerequisite_detail=detail,
            geometry_digest_hex=geometry_digest,
        )
    invalid_summary = {
        "protocol": schema.PROTOCOL_VERSION, "incomplete": False,
        "overall_pass": True, "canonical_profile_count": 192,
        "array_names": [], "array_hashes": {}, "npz_path": None,
        "npz_sha256": None, "provenance": _provenance(final=True),
        "checkpoint_count": 197, "watchdog": constructor.watchdog(1.0, 192),
        "prerequisite_authentication": {"passes": True},
    }
    with pytest.raises(ValueError, match="failed before publication"):
        runner.publish_final_documents(output, invalid_summary, {}, [first[:2]] * 197)
    assert not output.exists() and not output.with_suffix(".npz").exists()
    assert not output.with_name("p1.manifest.json").exists()


def test_uncompressed_storage_watchdog_rejects_before_candidate_write(tmp_path):
    assert runner.PROJECTED_FINAL_BYTES == 368_330_356
    assert runner.PROJECTED_PROFILE_BYTES == 1_569_841
    assert runner.PROJECTED_CHECKPOINT_CUMULATIVE_BYTES == 301_426_152
    assert runner.PROJECTED_CHECKPOINT_CUMULATIVE_BYTES == sum(
        runner.projected_checkpoint_array_bytes(generation)
        for generation in range(len(schema.EXPECTED_LEDGER) + 5)
    )
    assert runner.PROJECTED_NAMESPACE_BYTES == 669_756_508
    assert runner.PROJECTED_WORKER_SIDE_NPZ_ARRAY_BYTES == 80_436_480
    assert runner.PROJECTED_NAMESPACE_WITH_WORKER_SIDE_BYTES == 750_192_988
    assert runner.PROJECTED_FINAL_BYTES <= runner.FINAL_NPZ_BYTE_CAP
    assert runner.PROJECTED_PROFILE_BYTES <= runner.CHECKPOINT_NPZ_BYTE_CAP
    assert runner.PROJECTED_NAMESPACE_BYTES <= runner.CUMULATIVE_NPZ_BYTE_CAP
    small = {"a": np.zeros(16, np.float64)}
    report = runner.storage_report(small, [64] * 197, 256)
    assert report["passes"] and report["final_bytes"] == 128
    assert report["worker_side_npz_array_bytes"] == 256
    assert report["namespace_including_worker_side_npz_array_bytes"] == report["cumulative_bytes"] + 256
    assert report["candidate_peak_projected_bytes"] >= report["cumulative_bytes"]
    huge_checkpoint = {"a": np.zeros(runner.CHECKPOINT_NPZ_BYTE_CAP // 8 + 1)}
    with pytest.raises(ValueError, match="4 MiB"):
        runner.publish_checkpoint(tmp_path/"p1.json", _checkpoint(0), huge_checkpoint)
    oversized = {"a": np.zeros(runner.FINAL_NPZ_BYTE_CAP // 8 + 1)}
    assert not runner.storage_report(oversized, [1])["passes"]


def test_semantic_recert_rejects_correctly_shaped_zero_profile_not_pass_flags(monkeypatch):
    identity = schema.TASK_IDENTITIES[0]; seed = identity[1]
    x0 = np.zeros(14, np.float32); accepted_reference = np.zeros(10, np.float32)
    accepted_reference[0] = .14
    task_arrays = {
        f"task_{seed}_x0_float32": x0,
        f"task_{seed}_reference_float32": accepted_reference,
        f"task_{seed}_history_q_float64": np.zeros((9, 7), np.float64),
    }
    task_summary = {"rows": [{"public_task": {"default_side": 1}}]}
    model_arrays = {
        "model_lower_float64": -np.ones(7), "model_upper_float64": np.ones(7),
        "model_velocity_float64": np.ones(7), "model_effort_float64": np.ones(7),
    }
    def kinematics(q): return np.r_[np.asarray(q)[:2], 0.0], np.c_[np.eye(3), np.zeros((3, 4))]
    def rnea(_q, _v, a): return np.asarray(a)
    def derivatives(_q, _v, _a): return np.zeros((7,7)), np.zeros((7,7)), np.eye(7)
    def aba(_q, _v, u): return np.asarray(u)
    def velocity(_model, _q, qd): return np.asarray(qd)[:3]
    def dense(x, _u):
        return np.repeat(np.asarray(x)[None], schema.DENSE_SAMPLES, axis=0), np.zeros((schema.DENSE_SAMPLES,3))
    monkeypatch.setattr(runner, "_production_pin_context",
                        lambda: (object(), kinematics, rnea, derivatives, aba, velocity, dense))
    arrays = {
        "public_x0_float32": np.repeat(x0[None], 12, axis=0),
        "public_reference_float32": np.zeros((12, 960), np.float32),
        "public_default_side_int8": np.ones(12, np.int8),
        "quarantined_q8_float64": np.zeros((12,7)),
    }
    geometry = schema.construct_geometry(np.zeros(3), accepted_reference[:3], 1)
    arrays["public_reference_float32"][0] = np.tile(
        schema.reference_from_geometry(geometry).as_float32(), (schema.KNOTS, 1)
    ).ravel()
    ledger = schema.EXPECTED_LEDGER[0]
    prefix = f"profile_000_{ledger[1]}_{ledger[2]}_{ledger[3]}"
    arrays.update({f"{prefix}_{name}": np.zeros(shape, dtype)
                   for name, (shape, dtype) in constructor.PROFILE_ARRAY_SPECS.items()})
    matrix, condition = constructor.proxy_kkt_matrix()
    arrays["proxy_kkt_matrix_float64"] = matrix
    arrays["proxy_kkt_condition_float64"] = condition
    result = runner._production_semantic_recert(
        {"canonical_rows": [], "task_certificates": [], "campaign_certificate": {},
         "b1_b16_certificate": {}, "derivative_certificate": {},
         "cuda_worker_rows": [{} for _ in schema.EXPECTED_LEDGER]},
        arrays, ({"passes": True}, task_summary, task_arrays, {}, model_arrays),
    )
    assert not result["passes"] and result["failed_identity"] == list(ledger)


def test_runner_is_hard_blocked_before_files_or_private_calls(tmp_path):
    with pytest.raises(RuntimeError, match="blocked"):
        runner.execute(tmp_path/"p1.json", object())
    with pytest.raises(RuntimeError, match="blocked"):
        constructor.execute_constructor(authorization=object())
    assert list(tmp_path.iterdir()) == []
    transaction = runner.transaction_schema()
    assert len(transaction["pending_identities"]) == 192
    assert all(transaction[key] == 0 for key in runner.COUNTER_KEYS)
    assert not transaction["oracle_evidence"] and not transaction["benchmark_evidence"]


def test_production_gen0_precedes_the_single_cpu_prerequisite_load(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "require_thread_environment", lambda: calls.append(("environment", 0)))
    monkeypatch.setattr(runner, "_start_provenance", lambda _root: _provenance())
    monkeypatch.setattr(runner, "publish_checkpoint", lambda _output, document, _arrays, **_kw:
                        calls.append(("checkpoint", document["generation"])) or (Path("j"), Path("n"), Path("p")))
    def fail_authentication():
        calls.append(("prerequisite", 1))
        raise RuntimeError("synthetic prerequisite failure")
    monkeypatch.setattr(runner, "authenticate_cpu_prerequisite", fail_authentication)
    with pytest.raises(RuntimeError, match="synthetic prerequisite failure"):
        runner._production_pipeline(tmp_path / "p1.json", monotonic=lambda: 0.0)
    assert calls == [("environment", 0), ("checkpoint", 0), ("prerequisite", 1)]


def test_constructor_execution_delegates_exactly_once_when_capability_is_enabled(monkeypatch):
    token = object(); calls = []
    monkeypatch.setattr(constructor, "RUNNER_EXECUTION_AUTHORIZATION", token)
    monkeypatch.setattr(constructor, "run_production_constructor",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {"passes": True})
    assert constructor.execute_constructor("identity", authorization=token, marker=7) == {"passes": True}
    assert len(calls) == 1 and calls[0][0] == ("identity",)
    assert calls[0][1] == {"authorization": token, "marker": 7}


def test_permanent_failure_and_watchdog_records_are_immutable_non_evidence(tmp_path):
    provenance = _provenance()
    failed_output = tmp_path / "failure" / "p1.json"
    failed_output.parent.mkdir()
    failure = RuntimeError("synthetic profile failure")
    json_path, npz_path = runner.publish_permanent_rejection(
        failed_output, 0, provenance, 1.0, failure, completed=0,
        stage="profile_failed",
    )
    document = __import__("json").loads(json_path.read_text())
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    assert runner.certify_permanent_rejection(document, arrays)
    assert document["completed_identities"] == []
    assert len(document["pending_identities"]) == 192
    assert not document["oracle_evidence"] and not document["benchmark_evidence"]
    with pytest.raises(FileExistsError):
        runner.refuse_existing_artifacts(failed_output)

    watchdog_output = tmp_path / "watchdog" / "p1.json"
    watchdog_output.parent.mkdir()
    row_arrays = {"raw_status_int64": np.asarray(0, np.int64)}
    json_path, npz_path = runner.publish_permanent_rejection(
        watchdog_output, 0, provenance, 113.0,
        RuntimeError("campaign watchdog permanently rejected P1"), completed=1,
        arrays=row_arrays, stage="runtime_watchdog_rejected",
    )
    document = __import__("json").loads(json_path.read_text())
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    assert runner.certify_permanent_rejection(document, arrays)
    assert document["watchdog"]["runtime_watchdog_rejected"] is True
    assert document["completed_identities"] == [list(schema.EXPECTED_LEDGER[0])]
    assert len(document["pending_identities"]) == 191
    assert not watchdog_output.exists() and not watchdog_output.with_suffix(".npz").exists()


def test_production_loop_has_one_attempt_no_retry_and_fail_closed_subprocess_chain():
    source = inspect.getsource(runner._production_pipeline)
    assert source.count("result = execute_constructor(") == 1
    assert source.count("_run_cuda_worker(") == 1
    assert "for index, identity in enumerate(EXPECTED_LEDGER)" in source
    assert "except Exception as error:" in source
    assert source.count("publish_permanent_rejection(") == 2
    assert "completed=index, arrays=retained, stage=\"profile_failed\"" in source
    assert "completed=index+1, arrays=retained" in source
    assert "raise error" in source


def test_profile_clock_worker_timeout_and_single_thread_environment_are_hard_boundaries(monkeypatch):
    production = inspect.getsource(constructor.run_production_constructor)
    assert production.index("started=monotonic()") < production.index("proxy=proxy_joint_path")
    assert "guarded_kinematics" in production and "guarded_rnea_derivatives" in production
    for callback in ("objective", "gradient", "equality", "inequality", "inequality_jac"):
        body = production[production.index(f"def {callback}("):]
        assert "guard()" in body.split("\n    def ", 1)[0]
    worker_source = inspect.getsource(runner._run_cuda_worker)
    assert "remaining = float(campaign_deadline) - monotonic()" in worker_source
    assert "timeout=remaining" in worker_source
    assert "certify_worker_output" in worker_source and "certify_lane_binding" in worker_source
    monkeypatch.setattr(runner.os, "environ", {
        **runner.os.environ, **runner.REQUIRED_THREAD_ENVIRONMENT,
    })
    assert runner.require_thread_environment() == runner.REQUIRED_THREAD_ENVIRONMENT
    monkeypatch.setitem(runner.os.environ, "OMP_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="single-thread"):
        runner.require_thread_environment()


def test_final_stage_uses_one_semantic_recert_and_has_permanent_failure_boundary(tmp_path):
    source = inspect.getsource(runner._finalize_production)
    assert source.count("_production_semantic_recert(") == 1
    assert "storage_report(" in source
    assert source.index("_production_semantic_recert(") < source.index("summary[\"watchdog\"]")
    assert "owned_semantic=owned_semantic" in source
    assert "stage=\"final_certification_failed\"" in source
    provenance = _provenance(final=True)
    output = tmp_path / "final" / "p1.json"; output.parent.mkdir()
    json_path, npz_path = runner.publish_permanent_rejection(
        output, 191, provenance, 100.0, RuntimeError("injected final cert failure"),
        completed=192, stage="final_certification_failed",
    )
    document = __import__("json").loads(json_path.read_text())
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    assert runner.certify_permanent_rejection(document, arrays)
    assert document["completed_identities"] == [list(row) for row in schema.EXPECTED_LEDGER]
    assert document["pending_identities"] == [] and document["permanent_rejection"] is True
    assert not output.exists() and not output.with_suffix(".npz").exists()


def test_static_sources_do_not_execute_v4_artifacts_models_cuda_or_optimizers():
    sources = Path(schema.__file__).read_text()
    for forbidden in (
        "default_rng(", "generate_task(",
        "import importlib", "import subprocess", "import torch", "minimize(", "sim_forward(",
        ".solver.solve(",
    ):
        assert forbidden not in sources
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is not None
    assert constructor.RUNNER_EXECUTION_AUTHORIZATION is not None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is not None
    assert "return _production_pipeline" in inspect.getsource(runner.execute)
    production = inspect.getsource(constructor.run_production_constructor)
    assert "result=minimize(" in production and "hess=BFGS()" in production
    assert "callback=callback" in production and "PROFILE_WALL_LIMIT_S" in production
    assert "trust-constr" in sources
    assert "RUNNER_EXECUTION_AUTHORIZATION = object()" in Path(constructor.__file__).read_text()
    assert "import pinocchio as pin" in inspect.getsource(runner._production_pin_context)
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is not None


def test_static_paths_never_open_frozen_seed_rng(monkeypatch):
    calls = []
    monkeypatch.setattr(
        np.random, "default_rng", lambda seed=None: calls.append(seed) or object()
    )
    assert schema.frozen_metadata()["ledger_count"] == 192
    assert schema.certify_geometry(_geometry())["passes"]
    assert runner.transaction_schema()["task_rng_calls"] == 0
    assert calls == []


def test_cmake_binding_and_plant_source_have_isolated_symbol_and_terminal_guard():
    root = Path(__file__).resolve().parents[2]
    cmake = (root/"CMakeLists.txt").read_text()
    binding = (root/"python/bindings.cu").read_text()
    plant = (root/"gato/dynamics/tiago_right/tiago_right_plant.cuh").read_text()
    assert 'plant STREQUAL "tiago_right_circular_portfolio_toll"' in cmake
    assert "TIAGO_CIRCULAR_PORTFOLIO_TOLL=1" in cmake
    assert "PLANT_SUFFIX tiago_right_circular_portfolio_toll" in binding
    assert "3) * z * z" in plant and "2) * z * z * z" in plant
    assert "if (computeR)" in plant
    assert "blockIdx.x == KNOT_POINTS - 1" in plant
    assert not list(root.glob("**/grid.cuh")) == []


def test_existing_ref6_and_gaussian_ref10_variants_remain_source_distinct():
    root = Path(__file__).resolve().parents[2]
    cmake = (root/"CMakeLists.txt").read_text()
    binding = (root/"python/bindings.cu").read_text()
    plant = (root/"gato/dynamics/tiago_right/tiago_right_plant.cuh").read_text()
    for plant_name in (
        "tiago_right", "tiago_right_multimodal",
        "tiago_right_multimodal_toll", "tiago_right_circular_portfolio_toll",
    ):
        assert plant_name in cmake
    assert "#define PLANT_SUFFIX tiago_right_multimodal_toll" in binding
    assert "#define PLANT_SUFFIX tiago_right_multimodal" in binding
    assert "const T residual = exp(" in plant
    assert "TIAGO_CIRCULAR_PORTFOLIO_TOLL" in plant
    interface = (root/"python/bsqp/interface.py").read_text()
    assert 'self.plant_type.startswith("tiago_right")' in interface


def test_generated_grid_files_are_not_part_of_static_diff():
    root = Path(__file__).resolve().parents[2]
    import subprocess
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--", "**/grid.cuh"],
        cwd=root, check=True, text=True, capture_output=True,
    ).stdout
    assert changed == ""
