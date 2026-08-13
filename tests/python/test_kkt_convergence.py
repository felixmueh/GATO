from pathlib import Path

import numpy as np


SOURCE=(Path(__file__).resolve().parents[2]/"gato/bsqp/bsqp.cuh").read_text()
BINDINGS=(Path(__file__).resolve().parents[2]/"python/bindings.cu").read_text()
INTERFACE=(Path(__file__).resolve().parents[2]/"python/bsqp/interface.py").read_text()
LINE_SEARCH=(Path(__file__).resolve().parents[2]/"gato/bsqp/kernels/line_search.cuh").read_text()


def _converged(q,r,c,tolerance,pcg_status):
    arrays=(q,r,c)
    return bool(pcg_status==1 and all(np.isfinite(value).all() for value in arrays)
        and all(np.max(np.abs(value))<tolerance for value in arrays))


def test_kkt_convergence_requires_state_control_and_primal_residuals():
    tolerance=1e-3;small=np.array([2e-4,-9e-4]);large=np.array([1.1e-3])
    assert _converged(small,small,small,tolerance,1)
    assert not _converged(small,small,small,tolerance,0)
    assert not _converged(np.array([tolerance]),small,small,tolerance,1)
    assert not _converged(large,small,small,tolerance,1)
    assert not _converged(small,large,small,tolerance,1)
    assert not _converged(small,small,large,tolerance,1)
    assert not _converged(small,np.array([np.nan]),small,tolerance,1)


def test_async_residual_and_iteration_copies_are_synchronized_before_host_use():
    q_copy=SOURCE.index("cudaMemcpyAsync(h_q_batch_")
    r_copy=SOURCE.index("cudaMemcpyAsync(h_r_batch_",q_copy)
    c_copy=SOURCE.index("cudaMemcpyAsync(h_c_batch_",r_copy)
    iterations_copy=SOURCE.index("cudaMemcpyAsync(pcg_stats.num_iterations.data()",c_copy)
    synchronize=SOURCE.index("cudaStreamSynchronize(stream_)",iterations_copy)
    stats_copy=SOURCE.index("sqp_stats.pcg_stats.push_back(pcg_stats)",synchronize)
    host_read=SOURCE.index("const T* q_ptr = h_q_batch_",stats_copy)
    assert q_copy<r_copy<c_copy<iterations_copy<synchronize<stats_copy<host_read


def test_pcg_zero_iterations_is_not_used_as_sqp_convergence_evidence():
    for condition in ("q_max < kkt_tol_","r_max < kkt_tol_","c_max < kkt_tol_"):
        assert condition in SOURCE
    region=SOURCE[SOURCE.index("T q_max ="):SOURCE.index("if (num_solved",SOURCE.index("T q_max ="))]
    assert "num_iterations[b] == 0" not in region
    assert "pcg_stats.status[b] == PCG_CONVERGED" in region
    assert "std::isfinite(q_ptr[j])" in region and "std::isfinite(r_ptr[j])" in region


def test_line_search_copies_are_synchronized_before_stats_are_retained():
    merit=SOURCE.index("cudaMemcpyAsync(ls_stats.min_merit.data()")
    step=SOURCE.index("cudaMemcpyAsync(ls_stats.step_size.data()",merit)
    synchronize=SOURCE.index("cudaStreamSynchronize(stream_)",step)
    retain=SOURCE.index("sqp_stats.line_search_stats.push_back(ls_stats)",synchronize)
    assert merit<step<synchronize<retain


def test_line_search_rejects_nonfinite_merit_and_invalid_pcg_directions():
    assert "static_cast<T>(INFINITY)" in LINE_SEARCH
    assert "isfinite(min_merit)" in LINE_SEARCH
    assert "isfinite(incumbent_merit)" in LINE_SEARCH
    assert "pcg_status == PCG_CONVERGED" in LINE_SEARCH
    assert "pcg_status == PCG_MAX_ITERS" in LINE_SEARCH
    assert "pcg_status == PCG_BREAKDOWN" not in LINE_SEARCH


def test_lane_convergence_is_recomputed_and_never_copied_as_a_freeze_mask():
    loop=SOURCE[SOURCE.index("for (uint32_t i = 0;"):SOURCE.index("// Final merit")]
    assert "h_kkt_converged_batch_[b] =" in loop
    assert "if (!h_kkt_converged_batch_[b])" not in loop
    assert "cudaMemcpyAsync(d_kkt_converged_batch_, h_kkt_converged_batch_" not in loop


def test_binding_preserves_terminal_pcg_row_when_kkt_break_precedes_line_search():
    assert "const size_t num_pcg_iters = stats.pcg_stats.size();" in BINDINGS
    region=BINDINGS[BINDINGS.index("std::vector<float> pcg_times_us"):
        BINDINGS.index("std::vector<float> ls_min_merit")]
    assert "static_cast<py::ssize_t>(num_pcg_iters)" in region
    assert "sh_iters = { static_cast<py::ssize_t>(num_iters)" not in region
    assert 'pcg_status = _to_np(result.get("pcg_status"' in INTERFACE
    assert 'self.stats["pcg_status"] = pcg_status' in INTERFACE
    assert 'pcg_status.shape != pcg_iters.shape' in INTERFACE
