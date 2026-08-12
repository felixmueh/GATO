from pathlib import Path

import numpy as np


SOURCE=(Path(__file__).resolve().parents[2]/"gato/bsqp/kernels/pcg.cuh").read_text()


def _reference_pcg(matrix,preconditioner,rhs,x,iterations,abs_tol=1e-6):
    r=rhs-matrix@x;z=preconditioner@r;p=z.copy();rho=r@z
    if abs(rho)<abs_tol:return x.copy(),0
    rho_initial=abs(rho);completed=0
    for _ in range(iterations):
        work=matrix@p;alpha=rho/(p@work);x=x+alpha*p;r=r-alpha*work
        z=preconditioner@r;rho_new=r@z;completed+=1
        if abs(rho_new)<abs_tol+8e-4*rho_initial:break
        p=z+(rho_new/rho)*p;rho=rho_new
    return x,completed


def _reused_work_pcg(matrix,preconditioner,rhs,x,iterations,abs_tol=1e-6):
    # CPU transcription of the CUDA storage schedule: x is updated in its
    # external buffer and one work vector alternates between z and A*p.
    x=x.copy();r=rhs-matrix@x;work=preconditioner@r;p=work.copy();rho=r@work
    if abs(rho)<abs_tol:return x,0
    rho_initial=abs(rho);completed=0
    for _ in range(iterations):
        work=matrix@p;alpha=rho/(p@work);x+=alpha*p;r-=alpha*work
        work=preconditioner@r;rho_new=r@work;completed+=1
        if abs(rho_new)<abs_tol+8e-4*rho_initial:break
        p[:]=work+(rho_new/rho)*p;rho=rho_new
    return x,completed


def test_reused_work_vector_is_numerically_equivalent_to_reference_pcg():
    rng=np.random.default_rng(20260812);factor=rng.normal(size=(41,41))
    matrix=factor.T@factor+np.eye(41);preconditioner=np.diag(1/np.diag(matrix))
    rhs=rng.normal(size=41);x=rng.normal(size=41)
    actual,actual_iterations=_reused_work_pcg(matrix,preconditioner,rhs,x,18)
    expected,expected_iterations=_reference_pcg(matrix,preconditioner,rhs,x,18)
    assert actual_iterations==expected_iterations
    assert np.allclose(actual,expected,rtol=2e-13,atol=2e-13)


def test_zero_residual_takes_the_early_exit_without_changing_global_iterate():
    matrix=np.diag(np.linspace(1.,4.,19));preconditioner=np.diag(1/np.diag(matrix))
    x=np.linspace(-1.,1.,19);rhs=matrix@x
    actual,iterations=_reused_work_pcg(matrix,preconditioner,rhs,x,500)
    assert iterations==0 and np.array_equal(actual,x)


def test_one_step_exact_solution_and_nontrivial_path_cover_global_updates():
    matrix=np.eye(23)*3.;preconditioner=np.eye(23)/3.;rhs=np.linspace(-2.,2.,23)
    actual,iterations=_reused_work_pcg(matrix,preconditioner,rhs,np.zeros(23),500)
    assert iterations==1 and np.allclose(actual,rhs/3.,rtol=0,atol=2e-15)


def test_pcg_uses_three_vectors_exact_reduction_scratch_and_immediate_launch_check():
    assert "block::zeroSharedMemory<T, 3 * VEC_SIZE_PADDED>(s_mem);" in SOURCE
    assert "3 * VEC_SIZE_PADDED + block::WARP_SIZE" in SOURCE
    assert "5 * VEC_SIZE_PADDED" not in SOURCE
    assert "+ PCG_THREADS" not in SOURCE
    launch=SOURCE.index("<<<grid, thread_block, s_mem_size, stream>>>")
    check=SOURCE.index("gpuErrchk(cudaPeekAtLastError());",launch)
    assert check>launch


def test_n260_float_shared_memory_fits_pascal_default_block_limit():
    state_size=14;knots=260;vec_size_padded=(knots+2)*state_size
    old_bytes=4*(5*vec_size_padded+32+5+1024)
    new_bytes=4*(3*vec_size_padded+32)
    assert old_bytes==77604 and old_bytes>49152
    assert new_bytes==44144 and new_bytes<=49152
    # The 32-float tail is exact for 1024 threads / 32 threads per warp.
    assert 1024//32==32


def test_iterate_remains_global_and_work_vector_is_reused_without_final_copy():
    assert "d_x_vector[j] += s_alpha * s_p_vector[j];" in SOURCE
    assert "s_work_vector, d_A_matrix, s_p_vector" in SOURCE
    assert "s_work_vector, d_M_inv_matrix, s_r_vector" in SOURCE
    assert "block::copy<T, VEC_SIZE_PADDED>(s_x_vector" not in SOURCE
    assert "block::copy<T, VEC_SIZE_PADDED>(d_x_vector" not in SOURCE
    # Matrix products never alias an output vector with their input vector.
    assert "(s_work_vector, d_A_matrix, s_work_vector)" not in SOURCE
    assert "(s_work_vector, d_M_inv_matrix, s_work_vector)" not in SOURCE
    update=SOURCE.index("d_x_vector[j] += s_alpha * s_p_vector[j];")
    barrier=SOURCE.index("__syncthreads();",update)
    reuse=SOURCE.index("s_work_vector, d_M_inv_matrix, s_r_vector",barrier)
    assert update<barrier<reuse


def test_compact_pcg_has_distinct_artifact_suffix_with_identical_plant_flags():
    root=Path(__file__).resolve().parents[2]
    cmake=(root/"CMakeLists.txt").read_text();bindings=(root/"python/bindings.cu").read_text()
    case=cmake.split(
        'elseif(plant STREQUAL "tiago_right_constructed_route_portfolio_toll_pcg_compact")',1
    )[1].split("else()",1)[0]
    for definition in ("KNOT_POINTS=${knot}","PLANT_TIAGO_RIGHT=1",
            "TIAGO_CIRCULAR_PORTFOLIO_TOLL=1",
            "TIAGO_CONSTRUCTED_ROUTE_PORTFOLIO_TOLL=1",
            "TIAGO_CONSTRUCTED_ROUTE_PORTFOLIO_TOLL_PCG_COMPACT=1"):
        assert definition in case
    assert ("#elif defined(TIAGO_CONSTRUCTED_ROUTE_PORTFOLIO_TOLL_PCG_COMPACT)\n"
        "#define PLANT_SUFFIX tiago_right_constructed_route_portfolio_toll_pcg_compact") in bindings
