#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "constants.h"
#include "qdldl.h"
#include "types.cuh"
#include "utils/cuda.cuh"
#include "utils/linalg.cuh"

using namespace gato;
using namespace gato::constants;
using namespace sqp;

namespace gato {
namespace qdldl_linsys {

constexpr uint32_t SCHUR_DIM = STATE_SIZE * KNOT_POINTS;

inline size_t schurUpperNnz()
{
        return static_cast<size_t>(KNOT_POINTS) * ((STATE_SIZE * (STATE_SIZE + 1)) / 2) + static_cast<size_t>(KNOT_POINTS - 1) * STATE_SIZE_SQ;
}

template<typename T>
inline T getBlockRowValue(const T* h_S, uint32_t block_row, uint32_t block_col, uint32_t row, uint32_t col)
{
        const T* block = h_S + block_row * BLOCK_ROW_SIZE;
        if (block_col + 1 == block_row) { return block[row * BLOCK_ROW_R_DIM + col]; }
        if (block_col == block_row) { return block[row * BLOCK_ROW_R_DIM + STATE_SIZE + col]; }
        if (block_col == block_row + 1) { return block[row * BLOCK_ROW_R_DIM + 2 * STATE_SIZE + col]; }
        return static_cast<T>(0);
}

template<typename T>
void fillUpperCscFromSchur(const T* h_S, std::vector<QDLDL_int>& Ap, std::vector<QDLDL_int>& Ai, std::vector<QDLDL_float>& Ax)
{
        const uint32_t n = SCHUR_DIM;
        Ap.assign(n + 1, 0);
        Ai.clear();
        Ax.clear();
        Ai.reserve(schurUpperNnz());
        Ax.reserve(schurUpperNnz());

        for (uint32_t block_col = 0; block_col < KNOT_POINTS; ++block_col) {
                for (uint32_t col = 0; col < STATE_SIZE; ++col) {
                        const uint32_t global_col = block_col * STATE_SIZE + col;
                        Ap[global_col] = static_cast<QDLDL_int>(Ai.size());

                        if (block_col > 0) {
                                const uint32_t block_row = block_col - 1;
                                for (uint32_t row = 0; row < STATE_SIZE; ++row) {
                                        Ai.push_back(static_cast<QDLDL_int>(block_row * STATE_SIZE + row));
                                        Ax.push_back(static_cast<QDLDL_float>(getBlockRowValue(h_S, block_row, block_col, row, col)));
                                }
                        }

                        for (uint32_t row = 0; row <= col; ++row) {
                                Ai.push_back(static_cast<QDLDL_int>(block_col * STATE_SIZE + row));
                                Ax.push_back(static_cast<QDLDL_float>(getBlockRowValue(h_S, block_col, block_col, row, col)));
                        }
                }
        }
        Ap[n] = static_cast<QDLDL_int>(Ai.size());
}

template<typename T, uint32_t BatchSize>
void solveQDLDLBatched(T* d_lambda_batch, SchurSystem<T, BatchSize> schur, uint32_t* d_iterations, cudaStream_t stream)
{
        const uint32_t n = SCHUR_DIM;
        const size_t   nnz = schurUpperNnz();

        std::vector<T> h_S(B3D_MATRIX_SIZE_PADDED * BatchSize);
        std::vector<T> h_gamma(VEC_SIZE_PADDED * BatchSize);
        std::vector<T> h_lambda(VEC_SIZE_PADDED * BatchSize, static_cast<T>(0));
        std::vector<uint32_t> h_iterations(BatchSize, 1);

        gpuErrchk(cudaMemcpyAsync(h_S.data(), schur.d_S_batch, h_S.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaMemcpyAsync(h_gamma.data(), schur.d_gamma_batch, h_gamma.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));

        std::vector<QDLDL_int> Ap;
        std::vector<QDLDL_int> Ai;
        std::vector<QDLDL_float> Ax;
        std::vector<QDLDL_float> x(n);

        std::vector<QDLDL_int> etree(n);
        std::vector<QDLDL_int> Lnz(n);
        std::vector<QDLDL_int> iwork(3 * n);
        std::vector<QDLDL_bool> bwork(n);
        std::vector<QDLDL_float> fwork(n);
        std::vector<QDLDL_int> Lp(n + 1);
        std::vector<QDLDL_float> D(n);
        std::vector<QDLDL_float> Dinv(n);

        for (uint32_t solve_idx = 0; solve_idx < BatchSize; ++solve_idx) {
                const T* h_S_solve = h_S.data() + solve_idx * B3D_MATRIX_SIZE_PADDED;
                fillUpperCscFromSchur(h_S_solve, Ap, Ai, Ax);
                if (Ai.size() != nnz) { throw std::runtime_error("unexpected QDLDL Schur sparsity size"); }

                const QDLDL_int sumLnz = QDLDL_etree(static_cast<QDLDL_int>(n), Ap.data(), Ai.data(), iwork.data(), Lnz.data(), etree.data());
                if (sumLnz < 0) {
                        h_iterations[solve_idx] = 0;
                        continue;
                }

                std::vector<QDLDL_int> Li(sumLnz);
                std::vector<QDLDL_float> Lx(sumLnz);
                const QDLDL_int positive_d = QDLDL_factor(static_cast<QDLDL_int>(n), Ap.data(), Ai.data(), Ax.data(), Lp.data(), Li.data(), Lx.data(), D.data(), Dinv.data(), Lnz.data(), etree.data(), bwork.data(), iwork.data(), fwork.data());
                if (positive_d < 0) {
                        h_iterations[solve_idx] = 0;
                        continue;
                }

                const T* h_gamma_solve = h_gamma.data() + solve_idx * VEC_SIZE_PADDED;
                for (uint32_t knot = 0; knot < KNOT_POINTS; ++knot) {
                        for (uint32_t row = 0; row < STATE_SIZE; ++row) {
                                x[knot * STATE_SIZE + row] = static_cast<QDLDL_float>(h_gamma_solve[(knot + 1) * STATE_SIZE + row]);
                        }
                }

                QDLDL_solve(static_cast<QDLDL_int>(n), Lp.data(), Li.data(), Lx.data(), Dinv.data(), x.data());

                T* h_lambda_solve = h_lambda.data() + solve_idx * VEC_SIZE_PADDED;
                for (uint32_t knot = 0; knot < KNOT_POINTS; ++knot) {
                        for (uint32_t row = 0; row < STATE_SIZE; ++row) {
                                h_lambda_solve[(knot + 1) * STATE_SIZE + row] = static_cast<T>(x[knot * STATE_SIZE + row]);
                        }
                }
        }

        gpuErrchk(cudaMemcpyAsync(d_lambda_batch, h_lambda.data(), h_lambda.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(d_iterations, h_iterations.data(), h_iterations.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
}

}  // namespace qdldl_linsys
}  // namespace gato
