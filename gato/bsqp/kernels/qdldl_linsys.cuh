#pragma once
#include "bsqp/qdldl_host.h"
#include "constants.h"
#include "types.cuh"
#include "utils/cuda.cuh"

namespace gato::qdldl_linsys {
// Hybrid backend: GPU assembly/recovery, serial CPU direct factorization.
// No result is published until every lane passed the host residual check.
template<typename T, uint32_t BatchSize>
void solveQDLDLBatched(T* d_lambda, SchurSystem<T, BatchSize> schur,
                      const T* d_tolerance, uint32_t* d_iterations, cudaStream_t stream) {
        using namespace gato::constants;
        std::vector<T> matrix(B3D_MATRIX_SIZE_PADDED * BatchSize);
        std::vector<T> rhs(VEC_SIZE_PADDED * BatchSize);
        std::vector<T> tolerance(BatchSize);
        std::vector<T> lambda(VEC_SIZE_PADDED * BatchSize);
        // Internal existing stats slot: one direct solve, never PCG iterations.
        std::vector<uint32_t> direct_work(BatchSize, 1);
        gpuErrchk(cudaMemcpyAsync(matrix.data(), schur.d_S_batch, matrix.size()*sizeof(T), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaMemcpyAsync(rhs.data(), schur.d_gamma_batch, rhs.size()*sizeof(T), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaMemcpyAsync(tolerance.data(), d_tolerance, tolerance.size()*sizeof(T), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        for (uint32_t lane = 0; lane < BatchSize; ++lane) {
                auto result = HostSchurSolver<STATE_SIZE, KNOT_POINTS>::solve(
                    matrix.data()+lane*B3D_MATRIX_SIZE_PADDED,
                    rhs.data()+lane*VEC_SIZE_PADDED, tolerance[lane]);
                std::copy(result.begin(), result.end(), lambda.begin()+lane*VEC_SIZE_PADDED);
        }
        gpuErrchk(cudaMemcpyAsync(d_lambda, lambda.data(), lambda.size()*sizeof(T), cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(d_iterations, direct_work.data(), direct_work.size()*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        // Host storage outlives queued copies.
        gpuErrchk(cudaStreamSynchronize(stream));
}
} // namespace gato::qdldl_linsys
