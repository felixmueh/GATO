from pathlib import Path

import numpy as np


SOURCE=(Path(__file__).resolve().parents[2]/"gato/bsqp/kernels/pcg.cuh").read_text()


def _reference_pcg(matrix,preconditioner,rhs,x,iterations,tolerance=8e-4,abs_tol=1e-6):
    r=rhs-matrix@x;z=preconditioner@r;p=z.copy();rho=r@z
    residual_initial=r@r
    if residual_initial<=abs_tol**2:return x.copy(),0
    completed=0
    for _ in range(iterations):
        work=matrix@p;alpha=rho/(p@work);x=x+alpha*p;r=r-alpha*work
        completed+=1
        if r@r<=abs_tol**2+tolerance**2*residual_initial:break
        z=preconditioner@r;rho_new=r@z
        p=z+(rho_new/rho)*p;rho=rho_new
    return x,completed


def _reused_work_pcg(matrix,preconditioner,rhs,x,iterations,tolerance=8e-4,abs_tol=1e-6):
    # CPU transcription of the CUDA storage schedule: x is updated in its
    # external buffer and one work vector alternates between z and A*p.
    x=x.copy();r=rhs-matrix@x;residual_initial=r@r
    if residual_initial<=abs_tol**2:return x,0
    work=preconditioner@r;p=work.copy();rho=r@work;completed=0
    for _ in range(iterations):
        work=matrix@p;alpha=rho/(p@work);x+=alpha*p;r-=alpha*work
        completed+=1
        if r@r<=abs_tol**2+tolerance**2*residual_initial:break
        work=preconditioner@r;rho_new=r@work
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


def test_convergence_uses_true_residual_not_preconditioned_energy():
    # The second mode is heavily attenuated by the preconditioner.  Its
    # preconditioned energy is tiny even though the linear-system residual is
    # still O(1), reproducing the solver's false-convergence mode without
    # depending on a particular horizon length.
    matrix=np.diag([1.,2.]);preconditioner=np.diag([1.,1e-12]);rhs=np.array([0.,1.])
    residual=rhs.copy();energy=abs(residual@preconditioner@residual)
    assert energy<1e-6+8e-4
    assert np.linalg.norm(residual)>8e-4*np.linalg.norm(rhs)
    assert "epsilon * epsilon * s_residual_norm_sq_init" in SOURCE
    assert "epsilon * s_rho_init" not in SOURCE


def test_recursive_residual_is_explicitly_verified_and_periodically_replaced():
    assert "PCG_RESIDUAL_REPLACEMENT_PERIOD = 32" in SOURCE
    assert "s_r_vector, d_A_matrix, d_x_vector" in SOURCE
    assert "s_work_vector, d_A_matrix, d_x_vector" in SOURCE
    assert "s_replace_residual = s_converged" in SOURCE
    verify=SOURCE.index("Verify/reseed from the explicit residual")
    convergence=SOURCE.index("if (s_converged) { break; }",verify)
    assert verify<convergence


def test_pcg_exit_status_is_explicit_and_only_true_residual_convergence_succeeds():
    for name,value in (("PCG_MAX_ITERS",0),("PCG_CONVERGED",1),
                       ("PCG_BREAKDOWN",2),("PCG_INVALID_TOLERANCE",3),
                       ("PCG_SKIPPED",4)):
        assert f"{name} = {value}" in SOURCE
    final=SOURCE.index("A recursive residual is never authoritative on exit")
    explicit=SOURCE.index("explicit_convergence",final)
    status=SOURCE.index("d_status_batch[solve_idx] = explicit_convergence",explicit)
    assert final<explicit<status
    assert "gpuErrchk(cudaPeekAtLastError());" in SOURCE
    assert "if (d_status_batch[solve_idx] == PCG_CONVERGED) return;" in SOURCE
    assert "d_lambda[index] = static_cast<T>(0);" in SOURCE
    assert "s_rho_new >= static_cast<T>(0)" in SOURCE


def test_nonfinite_initial_true_residual_is_breakdown_not_convergence():
    initial=SOURCE.index("s_residual_norm_sq_init = s_residual_norm_sq")
    finite=SOURCE.index("!isfinite(s_residual_norm_sq_init)",initial)
    status=SOURCE.index("d_status_batch[solve_idx] = PCG_BREAKDOWN",finite)
    early_return=SOURCE.index("if (s_breakdown) { return; }",status)
    final=SOURCE.index("const bool explicit_convergence",early_return)
    assert initial < finite < status < early_return < final
    assert "isfinite(s_residual_norm_sq_init)" in SOURCE[final:]


def test_pcg_uses_three_vectors_exact_reduction_scratch_and_immediate_launch_check():
    assert "block::zeroSharedMemory<T, 3 * VEC_SIZE_PADDED>(s_mem);" in SOURCE
    assert "3 * VEC_SIZE_PADDED + block::WARP_SIZE" in SOURCE
    assert "5 * VEC_SIZE_PADDED" not in SOURCE
    assert "+ PCG_THREADS" not in SOURCE
    launch=SOURCE.index("<<<grid, thread_block, s_mem_size, stream>>>")
    check=SOURCE.index("gpuErrchk(cudaPeekAtLastError());",launch)
    assert check>launch


def test_schur_regularization_covers_state_positions_and_all_controls():
    root=Path(__file__).resolve().parents[2]
    schur=(root/"gato/bsqp/kernels/schur_linsys.cuh").read_text()
    q=schur.index("block::addScaledIdentity<T, STATE_SIZE>(s_Q_k, rho_penalty);")
    q_next=schur.index(
        "block::addScaledIdentity<T, STATE_SIZE>(s_Q_kp1, rho_penalty);",q
    )
    control=schur.index(
        "block::addScaledIdentityFull<T, CONTROL_SIZE>(s_R_k, rho_penalty);",q_next
    )
    inversion=schur.index("block::invertMatrix<T>",control)
    assert q<q_next<control<inversion
    linalg=(root/"gato/utils/linalg.cuh").read_text()
    helper=linalg.split("void addScaledIdentityFull",1)[1].split("// C = A * B",1)[0]
    assert "diagonal < dim" in helper
    assert "A[diagonal * dim + diagonal] += alpha;" in helper
    assert "dim/2" not in helper


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
