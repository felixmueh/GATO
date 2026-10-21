"""Opt-in self-building CUDA trace regression, including from a clean checkout.

GATO_TEST_ITERATION_TRACE_GPU=1 pytest test/test_iteration_trace_gpu.py \
  --basetemp=_expr_data/iteration-trace-gpu
Run only in a coordinated quiet GPU window. Build/outputs remain in basetemp.
"""
import os

import numpy as np
import pytest

import gato
from gato.common import initialize_warm_start
from conftest import arm_problem

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.self_building_gpu,
              pytest.mark.skipif(os.environ.get("GATO_TEST_ITERATION_TRACE_GPU") != "1",
                                 reason="explicit coordinated GPU opt-in required")]


def test_actual_outer_snapshots_final_output_and_stopping(tmp_path, monkeypatch, urdfs):
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path / "cache"))
    name, N, B = "iteration_trace_indy", 8, 2
    gato.register_robot(name, urdfs["indy7"], N=[N])

    def make(cap=3, *, admm=True, **kw):
        s = gato.BSQP(urdfs["indy7"], batch_size=B, N=N, dt=.01,
                      plant_type=name, max_sqp_iters=cap, linsys="bdsv", **kw)
        if admm:
            s.enable_limit_admm(rho=.01, iters=3)
        return s

    x, ref = arm_problem("indy7", N, B)
    s = make()
    seed = np.stack([initialize_warm_start(row, N, s.nx, s.nu) for row in x]).astype(np.float32)
    # Force a knot-0 replacement to verify that slot 0 captures the native seed.
    seed[:, :s.nx] += .1
    original = seed.copy()
    r = s.solve(x, ref, xu_warm=seed, capture_iterations=True)
    tr = r.iteration_trace
    expected_seed = seed.copy(); expected_seed[:, :s.nx] = x
    np.testing.assert_array_equal(tr.xu[0], expected_seed)
    np.testing.assert_array_equal(tr.xu[-1], r.xu)
    np.testing.assert_array_equal(seed, original)
    assert tr.iterations_run == 3 and tr.stop_reason == "max_sqp_iters"
    assert tr.xu.shape == (4, B, s.xu_size) and tr.xu.dtype == np.float32
    assert np.isfinite(tr.xu).all() and tr.line_search_ran.all()
    np.testing.assert_array_equal(tr.pcg_iters, r.stats.pcg_iters)
    assert np.max(np.abs(tr.xu[1] - tr.xu[0])) > 1e-6
    # Prefix solves are separate fresh solvers with a finite outer cap, never
    # repeated one-iteration calls on a stateful solver. Each prefix must end
    # at the captured boundary of the full native solve.
    for k in (1, 2, 3):
        plain = make(k).solve(x, ref, xu_warm=seed)
        assert plain.iteration_trace is None
        np.testing.assert_allclose(tr.xu[k], plain.xu, atol=2e-5, rtol=2e-5)

    no_stats = make(); no_stats.set_collect_stats(False)
    quiet = no_stats.solve(x, ref, xu_warm=seed, capture_iterations=True)
    assert quiet.stats.pcg_iters.shape == (0, B)
    np.testing.assert_array_equal(quiet.iteration_trace.pcg_iters, tr.pcg_iters)
    np.testing.assert_allclose(quiet.xu, r.xu, atol=2e-5, rtol=2e-5)

    # Existing practical shortcut remains active outside ADMM.
    shortcut = gato.BSQP(urdfs["indy7"], batch_size=B, N=N, dt=.01,
                         plant_type=name, max_sqp_iters=1000, linsys="pcg", pcg_tol=1e20)
    stopped = shortcut.solve(x, ref, capture_iterations=True).iteration_trace
    assert stopped.stop_reason == "zero_pcg" and stopped.iterations_run == 1
    assert not stopped.line_search_ran.any() and not stopped.pcg_iters.any()
    np.testing.assert_array_equal(stopped.xu[0], stopped.xu[1])

    # A budget still stops at the next outer boundary, after iteration 0.
    budget = make(1000).solve(x, ref, budget_s=1e-12, capture_iterations=True)
    assert budget.stats.budget_hit
    assert budget.iteration_trace.stop_reason == "budget"
    assert budget.iteration_trace.iterations_run == 1
    np.testing.assert_array_equal(budget.iteration_trace.xu[-1], budget.xu)
