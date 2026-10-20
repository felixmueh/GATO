#pragma once
#include "bsqp/kernels/pcg.cuh"

// Research-only extraction from pinned public GLASS, no private solver code.
// grid-rbd 0.5.1; GLASS 9e57178ca146fa1cecae7e27c22260603b993aa6.
// Source: src/base/pcg/solve.cuh
// Full source SHA256: f4b087771e28363ca6d1aad7f7ca5edb714fbf25e38b1a47bb365dda9983b131
// Transformations: rename/template the function; append Apply; replace its
// two preconditioner applications by apply(z,r,dead_Ap), and initialize p
// by copying z after the first application. All recurrence arithmetic,
// dot reductions, scalars, barriers outside these applications, stopping,
// iteration accounting and solution publication retain the public source.
// Pinv remains an argument for the original signature/layout; Apply owns
// its interpretation. The caller's native production path still uses GLASS.
namespace upstream_precond {
// glass.cuh includes the original solve.cuh inside namespace glass::block.
// Retain that lookup: bare glass:: vector operations may redispatch bodies.
using glass::block::flat_rank;
using glass::block::set_const;
using glass::block::copy;
using glass::block::bdmv;
using glass::block::axpby;
using glass::block::dot_fast;
using glass::block::axpy;

template <typename T, uint32_t state_size, uint32_t knot_points, typename Apply, bool TRAILING_SYNC = true>
__device__ void pcg_with_apply(T *x, T *S, T *Pinv, T *b, T *s_mem,
                    uint32_t max_iters, T rel_tol, T abs_tol, uint32_t *iters, Apply apply)
{
    constexpr uint32_t VEC = (knot_points + 2) * state_size;

    T *s_Ap  = s_mem;
    T *s_x   = s_Ap + VEC;
    T *s_r   = s_x  + VEC;
    T *s_z   = s_r  + VEC;
    T *s_p   = s_z  + VEC;
    T *s_scr = s_p  + VEC;            // warp-dot scratch: ceil(blockDim/32) elems
    __shared__ T s_rho, s_rho_new, s_alpha, s_beta, s_rho_init;

    uint32_t rank = flat_rank();

    set_const<T, 5 * VEC, /*TRAILING_SYNC=*/false>(static_cast<T>(0), s_mem);   // zero the 5 vectors incl. pads
    __syncthreads();

    copy<T, VEC, /*TRAILING_SYNC=*/false>(x, s_x);      // s_x = x (initial guess)
    __syncthreads();

    bdmv<T, knot_points, state_size, /*TRAILING_SYNC=*/false>(s_r, S, s_x);             // r = S x
    __syncthreads();
    axpby<T, VEC, /*TRAILING_SYNC=*/false>(static_cast<T>(1), b, static_cast<T>(-1), s_r, s_r); // r = b - S x
    __syncthreads();

    apply(s_z, s_r, s_Ap);
    __syncthreads();
    copy<T, VEC, /*TRAILING_SYNC=*/false>(s_z, s_p);     // p = z
    __syncthreads();

    dot_fast<T, VEC, /*TRAILING_SYNC=*/false>(s_r, s_z, &s_rho, s_scr); // rho = rᵀ z
    __syncthreads();

    T arho = (s_rho < static_cast<T>(0)) ? -s_rho : s_rho;
    if (arho < abs_tol) {
        if (rank == 0) *iters = 0;
        copy<T, VEC, /*TRAILING_SYNC=*/false>(s_x, x);
        if constexpr (TRAILING_SYNC) __syncthreads();
        return;
    }
    __syncthreads();  // scalar-bank reads above complete before rank 0 writes s_rho_init
    if (rank == 0) s_rho_init = arho;
    __syncthreads();

    uint32_t it = 0;
    for (uint32_t i = 0; i < max_iters; i++) {
        it = i + 1;

        bdmv<T, knot_points, state_size, /*TRAILING_SYNC=*/false>(s_Ap, S, s_p);        // Ap = S p
        __syncthreads();

        dot_fast<T, VEC, /*TRAILING_SYNC=*/false>(s_p, s_Ap, &s_alpha, s_scr); // pᵀ Ap
        __syncthreads();
        if (rank == 0) s_alpha = s_rho / s_alpha;                      // alpha
        __syncthreads();

        axpy<T, VEC, /*TRAILING_SYNC=*/false>(s_alpha, s_p, s_x);    // x += alpha p
        axpy<T, VEC, /*TRAILING_SYNC=*/false>(-s_alpha, s_Ap, s_r);  // r -= alpha Ap
        __syncthreads();

        apply(s_z, s_r, s_Ap);     // Ap is dead until the next matrix product
        __syncthreads();

        dot_fast<T, VEC, /*TRAILING_SYNC=*/false>(s_r, s_z, &s_rho_new, s_scr); // rho_new = rᵀ z
        __syncthreads();

        T arho_new = (s_rho_new < static_cast<T>(0)) ? -s_rho_new : s_rho_new;
        if (arho_new < abs_tol + rel_tol * s_rho_init) break;
        __syncthreads();  // scalar-bank reads above complete before rank 0 rewrites s_beta/s_rho

        if (rank == 0) { s_beta = s_rho_new / s_rho; s_rho = s_rho_new; }
        __syncthreads();

        axpby<T, VEC, /*TRAILING_SYNC=*/false>(static_cast<T>(1), s_z, s_beta, s_p, s_p); // p = z + beta p
        __syncthreads();
    }

    if (rank == 0) *iters = it;
    copy<T, VEC, /*TRAILING_SYNC=*/false>(s_x, x);                     // write solution back
    if constexpr (TRAILING_SYNC) __syncthreads();
}

}  // namespace upstream_precond
