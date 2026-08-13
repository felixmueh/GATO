#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "utils/cuda.cuh"
#include "utils/linalg.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;

// Recursive Krylov residuals drift from b - A*x in float arithmetic on long,
// ill-conditioned trajectory systems.  Periodically replace the recurrence
// and always verify a prospective convergence against the explicit residual.
constexpr uint32_t PCG_RESIDUAL_REPLACEMENT_PERIOD = 32;
enum PCGStatus : int32_t {
        PCG_MAX_ITERS = 0,
        PCG_CONVERGED = 1,
        PCG_BREAKDOWN = 2,
        PCG_INVALID_TOLERANCE = 3,
        PCG_SKIPPED = 4,
};

template<typename T, uint32_t BatchSize>
__global__ __launch_bounds__(PCG_THREADS) void solvePCGBatchedKernel(uint32_t* __restrict__       d_iterations,
                                                                    T* __restrict__              d_x_batch,
                                                                    const T* __restrict__        d_A_batch,
                                                                    const T* __restrict__        d_M_inv_batch,
                                                                    const T* __restrict__        d_b_batch,
                                                                    const T* __restrict__        d_epsilon_batch,
                                                                    uint32_t                     max_pcg_iters,
                                                                    const int32_t* __restrict__  d_kkt_converged_batch,
                                                                    int32_t* __restrict__        d_status_batch)
{
        const uint32_t solve_idx = blockIdx.x;
        const T        epsilon = d_epsilon_batch[solve_idx];

        const T abs_tol = static_cast<T>(1e-6);

        // skip solve if rho_max_reached
        if (d_kkt_converged_batch[solve_idx]) {
                if (threadIdx.x == 0) {
                        d_iterations[solve_idx] = 0;
                        d_status_batch[solve_idx] = PCG_SKIPPED;
                }
                return;
        }

        // ----- Shared Memory -----
        // Three padded vectors: residual, search direction, and reusable work
        // storage (preconditioned residual / matrix-vector product). The PCG
        // iterate stays in its existing global-memory buffer. The trailing
        // WARP_SIZE values are the block::dot warp-reduction scratch.
        extern __shared__ T s_mem[];
        block::zeroSharedMemory<T, 3 * VEC_SIZE_PADDED>(s_mem);

        // vectors
        T* s_r_vector = s_mem;
        T* s_p_vector = s_r_vector + VEC_SIZE_PADDED;
        T* s_work_vector = s_p_vector + VEC_SIZE_PADDED;

        // scratch for dot product
        T* s_scratch = s_work_vector + VEC_SIZE_PADDED;

        // scalars
        __shared__ T   s_rho, s_rho_new, s_alpha, s_beta;
        __shared__ T   s_residual_norm_sq, s_residual_norm_sq_init;
        __shared__ int s_converged, s_replace_residual, s_breakdown;

        uint32_t iterations = 0;

        __syncthreads();
        if (threadIdx.x == 0) {
                s_breakdown = !isfinite(epsilon) || epsilon <= static_cast<T>(0);
                d_status_batch[solve_idx] = s_breakdown
                                                ? PCG_INVALID_TOLERANCE
                                                : PCG_MAX_ITERS;
        }
        __syncthreads();
        if (s_breakdown) {
                if (threadIdx.x == 0) { d_iterations[solve_idx] = 0; }
                return;
        }

        // get A, M_inv, b, x pointers for current batch
        const T* d_A_matrix = getOffsetBlockRowPadded<T, BatchSize>(d_A_batch, solve_idx, 0);
        const T* d_M_inv_matrix = getOffsetBlockRowPadded<T, BatchSize>(d_M_inv_batch, solve_idx, 0);

        // getOffsetStatePadded points to the start of data, we want to point to the start of padding
        const T* d_b_vector = getOffsetStatePadded<T, BatchSize>(d_b_batch, solve_idx, 0) - STATE_SIZE;  // TODO: consider using shared memory for b
        T* d_x_vector = getOffsetStatePadded<T, BatchSize>(d_x_batch, solve_idx, 0) - STATE_SIZE;

        // ----- Init PCG -----

        // r = b - A * x
        block::btdMatrixVectorProduct<T, KNOT_POINTS, STATE_SIZE>(s_r_vector, d_A_matrix, d_x_vector);
        __syncthreads();

        block::vecSub<T, VEC_SIZE_PADDED>(s_r_vector, d_b_vector, s_r_vector);
        __syncthreads();

        // Convergence is defined in the true linear-system residual norm.
        // r^T M^-1 r can be tiny while ||r|| remains large when the
        // preconditioner strongly attenuates a mode, and epsilon is a norm
        // tolerance rather than a squared-energy tolerance.
        block::dot<T>(&s_residual_norm_sq, s_r_vector, s_r_vector, s_scratch, VEC_SIZE_PADDED);
        __syncthreads();
        if (threadIdx.x == 0) {
                s_residual_norm_sq_init = s_residual_norm_sq;
                s_breakdown = !isfinite(s_residual_norm_sq_init);
                if (s_breakdown) {
                        d_iterations[solve_idx] = 0;
                        d_status_batch[solve_idx] = PCG_BREAKDOWN;
                }
        }
        __syncthreads();
        if (s_breakdown) { return; }
        if (s_residual_norm_sq <= abs_tol * abs_tol) {
                if (threadIdx.x == 0) {
                        d_iterations[solve_idx] = 0;
                        d_status_batch[solve_idx] = PCG_CONVERGED;
                }
                __syncthreads();
                return;
        }

        // z, p = M^-1 * r
        block::btdMatrixVectorProduct<T, KNOT_POINTS, STATE_SIZE>(s_work_vector, s_p_vector, d_M_inv_matrix, s_r_vector);
        __syncthreads();

        // rho = r^T * z
        block::dot<T>(&s_rho, s_r_vector, s_work_vector, s_scratch, VEC_SIZE_PADDED);
        __syncthreads();
        if (threadIdx.x == 0) {
                // S and its approximate inverse are both negative definite.
                s_breakdown = !isfinite(s_rho) || s_rho >= static_cast<T>(0);
        }
        __syncthreads();

        // ----- PCG Loop -----
        for (uint32_t i = 0; i < max_pcg_iters; i++) {
                if (s_breakdown) { break; }
                iterations++;

                // A_p = A * p
                block::btdMatrixVectorProduct<T, KNOT_POINTS, STATE_SIZE>(s_work_vector, d_A_matrix, s_p_vector);
                __syncthreads();

                // alpha = rho / (p^T * A_p)
                block::dot<T>(&s_alpha, s_p_vector, s_work_vector, s_scratch, VEC_SIZE_PADDED);
                __syncthreads();
                if (threadIdx.x == 0) {
                        const T denominator = s_alpha;
                        s_breakdown = !isfinite(denominator)
                                      || denominator >= static_cast<T>(0);
                        if (!s_breakdown) {
                                s_alpha = s_rho / denominator;
                                s_breakdown = !isfinite(s_alpha);
                        }
                }
                __syncthreads();
                if (s_breakdown) { break; }

                // x = x + alpha * p
                // r = r - alpha * A_p
#pragma unroll
                for (uint32_t j = threadIdx.x; j < VEC_SIZE_PADDED; j += blockDim.x) {
                        d_x_vector[j] += s_alpha * s_p_vector[j];
                        s_r_vector[j] -= s_alpha * s_work_vector[j];
                }
                __syncthreads();

                block::dot<T>(&s_residual_norm_sq, s_r_vector, s_r_vector, s_scratch, VEC_SIZE_PADDED);
                __syncthreads();
                if (threadIdx.x == 0) {
                        s_converged = s_residual_norm_sq
                                      <= abs_tol * abs_tol
                                         + epsilon * epsilon * s_residual_norm_sq_init;
                        s_replace_residual = s_converged
                                             || ((i + 1) % PCG_RESIDUAL_REPLACEMENT_PERIOD == 0);
                }
                __syncthreads();

                if (s_replace_residual) {
                        // Verify/reseed from the explicit residual.  A small
                        // recursive residual alone is not convergence evidence.
                        block::btdMatrixVectorProduct<T, KNOT_POINTS, STATE_SIZE>(s_work_vector, d_A_matrix, d_x_vector);
                        __syncthreads();
                        block::vecSub<T, VEC_SIZE_PADDED>(s_r_vector, d_b_vector, s_work_vector);
                        __syncthreads();
                        block::dot<T>(&s_residual_norm_sq, s_r_vector, s_r_vector, s_scratch, VEC_SIZE_PADDED);
                        __syncthreads();
                        if (threadIdx.x == 0) {
                                s_converged = isfinite(s_residual_norm_sq)
                                              && s_residual_norm_sq
                                                     <= abs_tol * abs_tol
                                                        + epsilon * epsilon * s_residual_norm_sq_init;
                        }
                        __syncthreads();
                        if (s_converged) { break; }
                }

                // z = M^-1 * r
                block::btdMatrixVectorProduct<T, KNOT_POINTS, STATE_SIZE>(s_work_vector, d_M_inv_matrix, s_r_vector);
                __syncthreads();

                // rho_new = r^T * z
                block::dot<T>(&s_rho_new, s_r_vector, s_work_vector, s_scratch, VEC_SIZE_PADDED);
                __syncthreads();

                // beta = rho_new / rho
                // rho = rho_new
                if (threadIdx.x == 0) {
                        s_breakdown = !isfinite(s_rho_new)
                                      || s_rho_new >= static_cast<T>(0)
                                      || !isfinite(s_rho)
                                      || s_rho >= static_cast<T>(0);
                        s_beta = s_replace_residual
                                     ? static_cast<T>(0)
                                     : s_rho_new / s_rho;
                        s_breakdown = s_breakdown || !isfinite(s_beta);
                        s_rho = s_rho_new;
                }
                __syncthreads();
                if (s_breakdown) { break; }

                // p = z + beta * p
#pragma unroll
                for (uint32_t j = threadIdx.x; j < VEC_SIZE_PADDED; j += blockDim.x) { s_p_vector[j] = s_work_vector[j] + s_beta * s_p_vector[j]; }
                __syncthreads();
        }
        // ----- End PCG -----

        // A recursive residual is never authoritative on exit.  Recompute the
        // exact residual for converged, exhausted, and breakdown paths.
        block::btdMatrixVectorProduct<T, KNOT_POINTS, STATE_SIZE>(
            s_work_vector, d_A_matrix, d_x_vector);
        __syncthreads();
        block::vecSub<T, VEC_SIZE_PADDED>(
            s_r_vector, d_b_vector, s_work_vector);
        __syncthreads();
        block::dot<T>(&s_residual_norm_sq, s_r_vector, s_r_vector,
                      s_scratch, VEC_SIZE_PADDED);
        __syncthreads();

        // save stats for current batch
        if (threadIdx.x == 0) {
                d_iterations[solve_idx] = iterations;
                const bool explicit_convergence = isfinite(s_residual_norm_sq_init)
                    && isfinite(s_residual_norm_sq)
                    && s_residual_norm_sq
                           <= abs_tol * abs_tol
                              + epsilon * epsilon * s_residual_norm_sq_init;
                d_status_batch[solve_idx] = explicit_convergence
                                                ? PCG_CONVERGED
                                                : (s_breakdown
                                                       ? PCG_BREAKDOWN
                                                       : PCG_MAX_ITERS);
        }

}

template<typename T>
__host__ size_t getSolvePCGBatchedSMemSize()
{
        size_t size = sizeof(T) * (3 * VEC_SIZE_PADDED + block::WARP_SIZE);
        return size;
}

template<typename T, uint32_t BatchSize>
__host__ void solvePCGBatched(T* d_lambda_batch, SchurSystem<T, BatchSize> schur, T* d_epsilon_batch, uint32_t max_pcg_iters, int32_t* d_kkt_converged_batch, uint32_t* d_iterations, int32_t* d_status_batch, cudaStream_t stream)
{
        dim3           grid(BatchSize);
        dim3           thread_block(PCG_THREADS);
        const uint32_t s_mem_size = getSolvePCGBatchedSMemSize<T>();

        solvePCGBatchedKernel<T, BatchSize>
            <<<grid, thread_block, s_mem_size, stream>>>(d_iterations, d_lambda_batch, schur.d_S_batch, schur.d_P_inv_batch, schur.d_gamma_batch, d_epsilon_batch, max_pcg_iters, d_kkt_converged_batch, d_status_batch);
        gpuErrchk(cudaPeekAtLastError());
}

template<typename T, uint32_t BatchSize>
__global__ void resetFailedPCGIteratesKernel(
    T* d_lambda_batch, const int32_t* d_status_batch)
{
        const uint32_t solve_idx = blockIdx.x;
        if (d_status_batch[solve_idx] == PCG_CONVERGED) return;
        T* d_lambda = d_lambda_batch + solve_idx * VEC_SIZE_PADDED;
        for (uint32_t index = threadIdx.x; index < VEC_SIZE_PADDED;
             index += blockDim.x) {
                d_lambda[index] = static_cast<T>(0);
        }
}

template<typename T, uint32_t BatchSize>
__host__ void resetFailedPCGIterates(
    T* d_lambda_batch, const int32_t* d_status_batch, cudaStream_t stream)
{
        resetFailedPCGIteratesKernel<T, BatchSize>
            <<<BatchSize, 256, 0, stream>>>(d_lambda_batch, d_status_batch);
        gpuErrchk(cudaPeekAtLastError());
}
