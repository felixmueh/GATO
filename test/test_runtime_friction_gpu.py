"""Opt-in runtime-width CUDA checks; run only in a coordinated GPU window.

GATO_TEST_FRICTION_GPU=1 pytest test/test_runtime_friction_gpu.py
The test builds one N8 friction-enabled module in pytest's temporary cache.
Set --basetemp to the owning checkout's _expr_data/ directory for retained data.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import gato

pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(os.environ.get("GATO_TEST_FRICTION_GPU") != "1",
                                 reason="explicit coordinated GPU opt-in required")]


def test_runtime_width_value_gradient_merit_and_instance_isolation(tmp_path, monkeypatch, urdfs):
    pin = pytest.importorskip("pinocchio")
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path / "cache"))
    # A fixed existing arm topology, with explicit nonzero passive coefficients.
    tree = ET.parse(urdfs["indy7"])
    for joint in tree.getroot().findall("joint"):
        if joint.get("type") in ("revolute", "prismatic", "continuous"):
            dynamics = joint.find("dynamics")
            if dynamics is None:
                dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("friction", "0.3")
            dynamics.set("damping", "0.15")
    urdf = tmp_path / "runtime_width.urdf"
    tree.write(urdf)
    name = "runtime_width_indy"
    gato.register_robot(name, urdf, N=[8], collision_res=None, joint_friction_smoothing=.03)
    params = dict(max_sqp_iters=1, q_cost=0., qd_cost=0., u_cost=0., N_cost=0.,
                  q_lim_cost=0., vel_lim_cost=0., ctrl_lim_cost=0., mu=1., linsys="pcg")
    a = gato.BSQP(urdf, batch_size=1, N=8, dt=.002, plant_type=name, **params)
    b = gato.BSQP(urdf, batch_size=1, N=8, dt=.002, plant_type=name,
                  joint_friction_smoothing=.12, **params)
    assert a.lib is b.lib
    model = pin.buildModelFromUrdf(str(urdf)); data = model.createData()
    nq = model.nq; nx = 2*nq
    q = np.linspace(.1, .4, nq)
    v = np.linspace(-.02, .02, nq)
    u = pin.rnea(model, data, q, v, np.zeros(nq)).copy()
    state = np.r_[q, v]
    step_b_before = b.sim_forward(state, u, b.dt).copy()
    step_a_before = a.sim_forward(state, u, a.dt).copy()
    assert np.max(np.abs(step_a_before-step_b_before)) > 1e-4

    def expected_step(x, widths):
        qn, vn = x[:nq], x[nq:]
        losses = .15*vn + .3*np.tanh(vn/widths)
        acc = pin.aba(model, data, qn, vn, u-losses).copy()
        return np.r_[qn + vn*a.dt + .5*acc*a.dt*a.dt, vn + acc*a.dt]

    for widths in (.03, np.linspace(.04, .09, nq), .12):
        a.set_joint_friction_smoothing(widths)
        for velocity in (np.zeros(nq), v, v*10):
            x = np.r_[q, velocity]
            expected = expected_step(x, widths)
            actual = np.asarray(a.sim_forward(x, u, a.dt))[0]
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=1e-3)
            xu = np.tile(np.r_[x, u], a.N)[:-nq][None]
            ref = np.zeros((1, a.N*6), dtype=np.float32)
            kkt = a.debug_setup_kkt(xu, x[None], ref)
            analytical = np.asarray(kkt["A"])[0].reshape(a.N, nx, nx)[0].T
            epsilon = 1e-5
            fd = np.column_stack([(expected_step(x+delta, widths)-expected_step(x-delta, widths))/(2*epsilon)
                                  for delta in np.eye(nx)*epsilon])
            np.testing.assert_allclose(analytical, fd, atol=2e-3, rtol=1e-2)
            # Zero costs leave initial merit equal to sum of dynamics defects.
            # Every knot in this held trajectory repeats the same transition.
            result = a.solve(x[None], ref, xu_warm=xu)
            expected_merit = (a.N-1)*np.sum(np.abs(x-expected))
            np.testing.assert_allclose(result.stats.initial_merit, expected_merit, atol=2e-3, rtol=1e-2)
        np.testing.assert_allclose(b.sim_forward(state, u, b.dt), step_b_before, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(a.sim_forward(state, u, a.dt), step_b_before, atol=1e-6, rtol=1e-6)
    # Native users get the same finite/positive checks, including invalid shape.
    for bad in (np.zeros(nq), np.full(nq, np.nan), np.ones(nq+1)):
        with pytest.raises((ValueError, RuntimeError)):
            a.solver.set_joint_friction_smoothing(bad.astype(np.float32))
    np.testing.assert_allclose(a.sim_forward(state, u, a.dt), step_b_before, atol=1e-6, rtol=1e-6)
