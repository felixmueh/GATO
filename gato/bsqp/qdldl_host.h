#pragma once
// Selective host math from 123f11d5 (original dc62927e); no PCG status dependency.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "qdldl.h"

namespace gato::qdldl_linsys {
template<uint32_t StateSize, uint32_t KnotPoints>
struct HostSchurSolver {
static constexpr uint32_t StateSizeSq = StateSize * StateSize;
static constexpr uint32_t SCHUR_DIM = StateSize * KnotPoints;
static constexpr uint32_t VEC_SIZE_PADDED = (KnotPoints + 2) * StateSize;
static constexpr uint32_t BLOCK_ROW_R_DIM = 3 * StateSize;
static constexpr uint32_t BLOCK_ROW_SIZE = 3 * StateSize * StateSize;
static constexpr uint32_t B3D_MATRIX_SIZE_PADDED = BLOCK_ROW_SIZE * KnotPoints;
static size_t schurUpperNnz()
{
        return static_cast<size_t>(KnotPoints) * ((StateSize * (StateSize + 1)) / 2) + static_cast<size_t>(KnotPoints - 1) * StateSizeSq;
}

template<typename T>
static T getBlockRowValue(const T* h_S, uint32_t block_row, uint32_t block_col, uint32_t row, uint32_t col)
{
        const T* block = h_S + block_row * BLOCK_ROW_SIZE;
        if (block_col + 1 == block_row) { return block[row * BLOCK_ROW_R_DIM + col]; }
        if (block_col == block_row) { return block[row * BLOCK_ROW_R_DIM + StateSize + col]; }
        if (block_col == block_row + 1) { return block[row * BLOCK_ROW_R_DIM + 2 * StateSize + col]; }
        return static_cast<T>(0);
}

template<typename T>
static void fillUpperCscFromSchur(const T* h_S, std::vector<QDLDL_int>& Ap, std::vector<QDLDL_int>& Ai, std::vector<QDLDL_float>& Ax)
{
        const uint32_t n = SCHUR_DIM;
        Ap.assign(n + 1, 0);
        Ai.clear();
        Ax.clear();
        Ai.reserve(schurUpperNnz());
        Ax.reserve(schurUpperNnz());

        for (uint32_t block_col = 0; block_col < KnotPoints; ++block_col) {
                for (uint32_t col = 0; col < StateSize; ++col) {
                        const uint32_t global_col = block_col * StateSize + col;
                        Ap[global_col] = static_cast<QDLDL_int>(Ai.size());

                        if (block_col > 0) {
                                const uint32_t block_row = block_col - 1;
                                for (uint32_t row = 0; row < StateSize; ++row) {
                                        Ai.push_back(static_cast<QDLDL_int>(block_row * StateSize + row));
                                        Ax.push_back(static_cast<QDLDL_float>(getBlockRowValue(h_S, block_row, block_col, row, col)));
                                }
                        }

                        for (uint32_t row = 0; row <= col; ++row) {
                                Ai.push_back(static_cast<QDLDL_int>(block_col * StateSize + row));
                                Ax.push_back(static_cast<QDLDL_float>(getBlockRowValue(h_S, block_col, block_col, row, col)));
                        }
                }
        }
        Ap[n] = static_cast<QDLDL_int>(Ai.size());
}

template<typename T>
static std::vector<T> solve(const T* matrix, const T* rhs, T tolerance) {
        const uint32_t n = SCHUR_DIM;
        const size_t nnz = schurUpperNnz();
        // Preserve the historical per-lane direct algorithm. A rejected solve
        // throws before any device direction recovery or zero-work shortcut.
        std::vector<T> h_lambda(VEC_SIZE_PADDED, static_cast<T>(0));
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

        if (!std::isfinite(tolerance) || tolerance <= 0) {
                throw std::runtime_error("QDLDL invalid direct-solve tolerance");
        }
        const T* h_S_solve = matrix;
        fillUpperCscFromSchur(h_S_solve, Ap, Ai, Ax);
        if (Ai.size() != nnz) { throw std::runtime_error("unexpected QDLDL Schur sparsity size"); }

        const T* h_gamma_solve = rhs;
        if (!std::all_of(Ax.begin(), Ax.end(), [](auto v) { return std::isfinite(v); })
            || !std::all_of(h_gamma_solve + StateSize, h_gamma_solve + StateSize + n,
                            [](auto v) { return std::isfinite(v); })) throw std::runtime_error("QDLDL nonfinite Schur input");

        const QDLDL_int sumLnz = QDLDL_etree(static_cast<QDLDL_int>(n), Ap.data(), Ai.data(), iwork.data(), Lnz.data(), etree.data());
        if (sumLnz < 0) {
                throw std::runtime_error("QDLDL symbolic factorization rejected");
        }

        std::vector<QDLDL_int> Li(sumLnz);
        std::vector<QDLDL_float> Lx(sumLnz);
        const QDLDL_int positive_d = QDLDL_factor(static_cast<QDLDL_int>(n), Ap.data(), Ai.data(), Ax.data(), Lp.data(), Li.data(), Lx.data(), D.data(), Dinv.data(), Lnz.data(), etree.data(), bwork.data(), iwork.data(), fwork.data());
        if (positive_d < 0) {
                throw std::runtime_error("QDLDL numeric factorization rejected");
        }

        for (uint32_t knot = 0; knot < KnotPoints; ++knot) {
                for (uint32_t row = 0; row < StateSize; ++row) {
                        x[knot * StateSize + row] = static_cast<QDLDL_float>(h_gamma_solve[(knot + 1) * StateSize + row]);
                }
        }

        QDLDL_solve(static_cast<QDLDL_int>(n), Lp.data(), Li.data(), Lx.data(), Dinv.data(), x.data());

        if (!std::all_of(x.begin(), x.end(), [](auto v) { return std::isfinite(v); }))
                throw std::runtime_error("QDLDL nonfinite direct solution");
        T* h_lambda_solve = h_lambda.data();
        for (uint32_t knot = 0; knot < KnotPoints; ++knot) {
                for (uint32_t row = 0; row < StateSize; ++row) {
                        h_lambda_solve[(knot + 1) * StateSize + row] = static_cast<T>(x[knot * StateSize + row]);
                }
        }

        // Check the original full Schur rows after casting the solution
        // back to T. This also detects asymmetry/roundoff hidden by an
        // upper-triangle-only factorization. Accumulate in double on CPU.
        double residual_sq = 0, rhs_sq = 0;
        for (uint32_t knot = 0; knot < KnotPoints; ++knot) {
                for (uint32_t row = 0; row < StateSize; ++row) {
                        double ax = 0;
                        for (uint32_t col = 0; col < BLOCK_ROW_R_DIM; ++col) {
                                ax += static_cast<double>(h_S_solve[knot * BLOCK_ROW_SIZE + row * BLOCK_ROW_R_DIM + col])
                                      * h_lambda_solve[knot * StateSize + col];
                        }
                        const double rhs = h_gamma_solve[(knot + 1) * StateSize + row];
                        residual_sq += (rhs - ax) * (rhs - ax);
                        rhs_sq += rhs * rhs;
                }
        }
        const double threshold = std::max(1e-6, static_cast<double>(tolerance) * std::sqrt(rhs_sq));
        if (!std::isfinite(residual_sq) || std::sqrt(residual_sq) > threshold) {
                throw std::runtime_error("QDLDL full Schur residual rejected");
        }
        return h_lambda;
}
};
} // namespace gato::qdldl_linsys
