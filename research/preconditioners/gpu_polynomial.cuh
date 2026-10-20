#pragma once
#include "pcg_with_apply.cuh"

namespace upstream_precond {

// Same [L|D|R] row-major indexing and accumulation order as pinned GLASS
// src/base/banded/bdmv.cuh (SHA256
// 141e51a9b1aeb022a411ebf7047768fa2003d70bb55a3cf4cf368e5ec09aae16).
// The single change is += instead of =. Input v is distinct from output z.
template <typename T, uint32_t state_size, uint32_t knot_points>
__device__ void add_bdmv(T* z, const T* matrix, const T* v)
{
    constexpr uint32_t BRL = 3 * state_size;
    uint32_t rank = glass::block::flat_rank();
    uint32_t size = glass::block::flat_size();
    for (uint32_t idx = rank; idx < knot_points * state_size; idx += size) {
        uint32_t br = idx / state_size;
        uint32_t row = idx % state_size;
        const T* blk = matrix + br * BRL * state_size;
        const T* vec = v + br * state_size;
        T sum = static_cast<T>(0);
        for (uint32_t col = 0; col < BRL; col++)
            sum += blk[row * BRL + col] * vec[col];
        z[(br + 1) * state_size + row] += sum;
    }
}

// Native signs are retained: z=P*r, then z+=P*(r-S*z). In the positive
// convention A=-S, M=-P this is the negative of a Richardson polynomial
// in A,M. Order 2 is not unconditionally definite; see accompanying notes.
template <typename T, uint32_t state_size, uint32_t knot_points, int Order>
struct PolynomialApply {
    static_assert(Order == 2 || Order == 3, "polynomial order must be 2 or 3");
    const T* S;
    const T* P;

    __device__ void operator()(T* z, const T* r, T* dead_Ap) const
    {
        constexpr uint32_t VEC = (knot_points + 2) * state_size;
        glass::block::bdmv<T, knot_points, state_size, false>(z, P, r);
        __syncthreads();
#pragma unroll
        for (int pass = 1; pass < Order; ++pass) {
            glass::block::bdmv<T, knot_points, state_size, false>(dead_Ap, S, z);
            __syncthreads();
            glass::block::axpby<T, VEC, false>(static_cast<T>(1), const_cast<T*>(r), static_cast<T>(-1), dead_Ap, dead_Ap);
            __syncthreads();
            add_bdmv<T, state_size, knot_points>(z, P, dead_Ap);
            __syncthreads();
        }
    }
};

template <typename T, int Order>
__global__ __launch_bounds__(PCG_THREADS) void solve_gpu_polynomial_kernel(
    uint32_t* __restrict__ iterations, T* __restrict__ x_batch,
    const T* __restrict__ S_batch, const T* __restrict__ P_batch,
    const T* __restrict__ b_batch, const T* __restrict__ epsilon_batch,
    uint32_t max_iters, int32_t* __restrict__ skip_batch)
{
    using namespace gato::constants;
    const uint32_t lane = blockIdx.x;
    if (skip_batch[lane]) {
        if (threadIdx.x == 0) iterations[lane] = 0;
        return;
    }
    extern __shared__ T s_mem[];
    const T* S = gato::get_offset_block_row_padded<T>(S_batch, lane, 0);
    const T* P = gato::get_offset_block_row_padded<T>(P_batch, lane, 0);
    T* x = gato::get_padded_vector<T>(x_batch, lane);
    const T* b = gato::get_padded_vector<T>(b_batch, lane);
    pcg_with_apply<T, STATE_SIZE, KNOT_POINTS>(x, const_cast<T*>(S), const_cast<T*>(P),
        const_cast<T*>(b), s_mem, max_iters, epsilon_batch[lane], static_cast<T>(1e-6),
        &iterations[lane], PolynomialApply<T, STATE_SIZE, KNOT_POINTS, Order>{S, P});
}

template <typename T, int Order>
__host__ void solve_gpu_polynomial(uint32_t batch_size, T* lambda, SchurSystem<T> schur,
    T* epsilon, uint32_t max_iters, int32_t* skip, uint32_t* iterations,
    cudaStream_t stream = nullptr)
{
    // Same scratch contract as native PCG; large horizons require opt-in.
    const size_t bytes = get_solve_pcg_batched_smem_size<T>();
    static size_t configured_bytes=0;
    opt_in_dynamic_smem(solve_gpu_polynomial_kernel<T,Order>,bytes,configured_bytes);
    solve_gpu_polynomial_kernel<T, Order><<<batch_size, PCG_THREADS, bytes, stream>>>(
        iterations, lambda, schur.d_S_batch, schur.d_P_inv_batch, schur.d_gamma_batch,
        epsilon, max_iters, skip);
    gpuErrchk(cudaGetLastError());
}

}  // namespace upstream_precond
