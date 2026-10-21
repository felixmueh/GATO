"""Host contract and native allocation bounds for explicit solve inspection."""
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from gato import BSQP, IterationTrace


def fake_solver():
    s = BSQP.__new__(BSQP)
    s.lib = SimpleNamespace(ITERATION_CAPTURE_AVAILABLE=True)
    s.batch_size, s.N, s.nx, s.nu, s.n_actuated = 1, 3, 2, 1, 1
    s.xu_size, s.dt, s.floating_base, s._time_budget_us = 8, .01, False, 0.
    s.calls = []

    def solve(seed, dt, x, goals, **kwargs):
        s.calls.append((seed.copy(), kwargs))
        raw = dict(XU=seed + 1, sqp_time_us=10, sqp_iters=[2], kkt_converged=[1],
                   final_merit=[1.], initial_merit=[2.], ls_num_iters=1,
                   pcg_iters=[], pcg_times_us=[], ls_min_merit=[], ls_step_size=[], budget_hit=False)
        if kwargs.get("capture_iterations"):
            raw["iteration_trace"] = dict(
                xu=np.stack([seed, seed + 1, seed + 1]),
                pcg_iters=np.array([[2], [0]], np.int32),
                line_search_ran=np.array([True, False]),
                iterations_run=2, stop_reason="zero_pcg")
        return raw

    s.solver = SimpleNamespace(solve=solve)
    return s


def test_capture_is_per_call_and_preserves_seed_and_binary_arrays():
    s = fake_solver()
    x, goals = np.array([[.2, .3]], np.float32), np.zeros((1, 18), np.float32)
    warm = np.arange(8, dtype=np.float32)[None]
    original = warm.copy()
    r = s.solve(x, goals, xu_warm=warm, capture_iterations=True)
    assert isinstance(r.iteration_trace, IterationTrace)
    trace = r.iteration_trace
    expected_seed = warm.copy(); expected_seed[:, :2] = x
    np.testing.assert_array_equal(trace.xu[0], expected_seed)
    np.testing.assert_array_equal(trace.xu[-1], r.xu)
    np.testing.assert_array_equal(warm, original)
    assert trace.xu.dtype == np.float32 and trace.pcg_iters.dtype == np.int32
    assert trace.line_search_ran.dtype == bool
    assert trace.iterations_run == 2 and trace.stop_reason == "zero_pcg"
    assert trace.pcg_iters.shape == (2, 1) and r.stats.pcg_iters.shape == (0, 1)
    # General stats can be absent while trace PCG rows remain aligned.
    assert r.stats.ls_num_iters == 1 < trace.iterations_run
    assert s.solve(x, goals).iteration_trace is None
    assert s.calls[-1][1] == {}  # compatible with the original native signature


def test_old_native_module_keeps_normal_solve_and_rejects_capture():
    s = fake_solver(); del s.lib.ITERATION_CAPTURE_AVAILABLE
    x, goals = np.zeros((1, 2)), np.zeros((1, 18))
    assert s.solve(x, goals).iteration_trace is None
    with pytest.raises(RuntimeError, match="rebuilt native module"):
        s.solve(x, goals, capture_iterations=True)
    assert len(s.calls) == 1


def test_device_path_forwards_capture_and_keeps_trace_on_host(monkeypatch):
    from gato import _arrays
    s = fake_solver()
    x, ref = object(), object()
    monkeypatch.setattr(_arrays, "kind", lambda a: "numpy")
    monkeypatch.setattr(_arrays, "is_cuda", lambda a: True)
    monkeypatch.setattr(_arrays, "device_f32", lambda a, count, name: (11 if a is x else 22, a))
    monkeypatch.setattr(_arrays, "from_device", lambda value, kind: value)
    device_calls = []

    def solve_device(*args, **kwargs):
        device_calls.append((args, kwargs))
        return s.solver.solve(np.zeros((1, 8), np.float32), s.dt, None, None, **kwargs)

    s.solver.solve_device = solve_device
    r = s.solve(x, ref, capture_iterations=True)
    assert device_calls == [((0, s.dt, 11, 22), {"capture_iterations": True})]
    assert isinstance(r.iteration_trace.xu, np.ndarray)
    np.testing.assert_array_equal(r.iteration_trace.xu[-1], r.xu)


def test_native_capture_capacity_is_checked_without_overflow(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "bounds.cpp"
    source.write_text('''#include "utils/iteration_trace.h"
#include <cassert>
#include <limits>
int main() {
  assert(iteration_trace_elements<float>(1, 128*21-7, 1000) == 1001ul*(128*21-7));
  const size_t cap = 256ul*1024*1024/sizeof(float);
  assert(iteration_trace_elements<float>(1, cap, 0) == cap);
  for (int variant = 0; variant < 4; ++variant) {
    bool threw = false;
    try {
      if (variant == 0) iteration_trace_elements<float>(1, cap, 1);
      if (variant == 1) iteration_trace_elements<float>(std::numeric_limits<size_t>::max(), 2, 1);
      if (variant == 2) iteration_trace_elements<float>(2, 2, std::numeric_limits<uint32_t>::max());
      if (variant == 3) iteration_trace_elements<float>(0, 1, 1);
    } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
  }
}''')
    binary = tmp_path / "bounds"
    include = Path(__file__).resolve().parents[1] / "gato"
    subprocess.run([compiler, "-std=c++17", "-I", str(include), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
