"""Opt-in self-building CUDA trace regression, including from a clean checkout.

GATO_TEST_ITERATION_TRACE_GPU=1 pytest test/test_iteration_trace_gpu.py \
  --basetemp=_expr_data/iteration-trace-gpu
Run only in a coordinated quiet GPU window. Build/outputs remain in basetemp.
"""
import ctypes
import gc
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
    from gato import _cache
    ensure_module = _cache.ensure_module

    def ensure_with_half_cores(*args, **kwargs):
        kwargs["jobs"] = max(1, len(os.sched_getaffinity(0)) // 2)
        return ensure_module(*args, **kwargs)

    monkeypatch.setattr(_cache, "ensure_module", ensure_with_half_cores)
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

    np.savez_compressed(tmp_path / "trace-regression.npz", seed=expected_seed,
                        xu=tr.xu, pcg_iters=tr.pcg_iters,
                        line_search_ran=tr.line_search_ran, final=r.xu)

    # Exercise the existing zero-work shortcut deterministically outside ADMM.
    # A zero inner cap supplies zero PCG counts by construction; this is a
    # boundary/metadata fixture and makes no claim about stationarity.
    shortcut = gato.BSQP(urdfs["indy7"], batch_size=B, N=N, dt=.01,
                         plant_type=name, max_sqp_iters=1000, linsys="pcg",
                         max_pcg_iters=0)
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

    # The NumPy capsule owns snapshots independently of the result and solver.
    retained = tr.xu
    expected = retained.copy()
    del tr, r, s
    gc.collect()
    np.testing.assert_array_equal(retained, expected)

    # Exercise the real native device-pointer entry point without requiring an
    # optional tensor framework. Inputs use the CUDA runtime already linked by
    # the module; no new public allocator or dependency is added for inspection.
    cudart = ctypes.CDLL("libcudart.so.12")
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    pointers = []
    try:
        for values in (x, ref):
            values = np.ascontiguousarray(values, dtype=np.float32)
            pointer = ctypes.c_void_p()
            assert cudart.cudaMalloc(ctypes.byref(pointer), values.nbytes) == 0
            pointers.append(pointer)
            assert cudart.cudaMemcpy(pointer, values.ctypes.data, values.nbytes, 1) == 0  # host -> device
        device_solver = make()
        raw = device_solver.solver.solve_device(0, device_solver.dt,
                                                pointers[0].value, pointers[1].value,
                                                capture_iterations=True)
        device_xu = raw["XU"].numpy()
        device_trace = raw["iteration_trace"]
        cold_host = make().solve(x, ref, capture_iterations=True)
        np.testing.assert_array_equal(device_trace["xu"][-1], device_xu)
        np.testing.assert_array_equal(device_trace["xu"][0], cold_host.iteration_trace.xu[0])
        np.testing.assert_allclose(device_xu, cold_host.xu, atol=2e-5, rtol=2e-5)
        assert device_trace["iterations_run"] == 3
    finally:
        for pointer in pointers:
            assert cudart.cudaFree(pointer) == 0
