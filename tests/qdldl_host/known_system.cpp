#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bsqp/qdldl_host.h"

// Same dense off-diagonal/padded known-system fixture as the historical port,
// with TIAGo's 14-state block size, entirely on CPU.
using Host = gato::qdldl_linsys::HostSchurSolver<14, 8>;
constexpr uint32_t STATE_SIZE = 14, KNOT_POINTS = 8;
constexpr uint32_t BLOCK_ROW_R_DIM = Host::BLOCK_ROW_R_DIM;
constexpr uint32_t BLOCK_ROW_SIZE = Host::BLOCK_ROW_SIZE;
constexpr uint32_t VEC_SIZE_PADDED = Host::VEC_SIZE_PADDED;
constexpr uint32_t B3D_MATRIX_SIZE_PADDED = Host::B3D_MATRIX_SIZE_PADDED;

// A known solution with dense, nonsymmetric off-diagonal blocks catches
// transpose, column-major, batch-stride, and padded-vector mistakes.
int main()
{
        constexpr uint32_t batch = 6;
        constexpr uint32_t n = STATE_SIZE * KNOT_POINTS;
        std::vector<float> matrix(batch * B3D_MATRIX_SIZE_PADDED, 0);
        std::vector<float> rhs(batch * VEC_SIZE_PADDED, 0);
        std::vector<float> expected(VEC_SIZE_PADDED, 0);
        std::vector<float> tolerance(batch, 1e-4f);
        for (uint32_t i = 0; i < n; ++i) expected[STATE_SIZE + i] = std::sin(float(i + 1));
        for (uint32_t b = 0; b < batch; ++b) {
                float* a = matrix.data() + b * B3D_MATRIX_SIZE_PADDED;
                for (uint32_t k = 0; k < KNOT_POINTS; ++k) {
                        for (uint32_t row = 0; row < STATE_SIZE; ++row) {
                                for (uint32_t col = 0; col < STATE_SIZE; ++col) {
                                        a[k * BLOCK_ROW_SIZE + row * BLOCK_ROW_R_DIM + STATE_SIZE + col]
                                            = row == col ? -4.0f : 0.001f * (row + col + 1);
                                        if (k + 1 < KNOT_POINTS) {
                                                const float off = 0.0001f * (row + 2 * col + 1);
                                                a[k * BLOCK_ROW_SIZE + row * BLOCK_ROW_R_DIM + 2 * STATE_SIZE + col] = off;
                                                a[(k + 1) * BLOCK_ROW_SIZE + col * BLOCK_ROW_R_DIM + row] = off;
                                        }
                                }
                        }
                }
                for (uint32_t k = 0; k < KNOT_POINTS; ++k) {
                        for (uint32_t row = 0; row < STATE_SIZE; ++row) {
                                double sum = 0;
                                for (uint32_t col = 0; col < BLOCK_ROW_R_DIM; ++col)
                                        sum += double(a[k * BLOCK_ROW_SIZE + row * BLOCK_ROW_R_DIM + col]) * expected[k * STATE_SIZE + col];
                                rhs[b * VEC_SIZE_PADDED + (k + 1) * STATE_SIZE + row] = float(sum);
                        }
                }
        }
        // Lane 1 singular, lane 2 nonfinite, lane 3 invalid tolerance,
        // lane 4 exactly zero RHS, lane 5 asymmetric full rows: factorization
        // succeeds but the returned solution fails the actual Schur residual.
        std::fill(matrix.begin() + B3D_MATRIX_SIZE_PADDED, matrix.begin() + 2 * B3D_MATRIX_SIZE_PADDED, 0);
        matrix[2 * B3D_MATRIX_SIZE_PADDED + STATE_SIZE] = std::numeric_limits<float>::quiet_NaN();
        tolerance[3] = 0;
        std::fill(rhs.begin() + 4 * VEC_SIZE_PADDED, rhs.begin() + 5 * VEC_SIZE_PADDED, 0);
        matrix[5 * B3D_MATRIX_SIZE_PADDED + BLOCK_ROW_R_DIM + STATE_SIZE] += 1;

        std::vector<float> solution(rhs.size(), 0);
        for (uint32_t b = 0; b < batch; ++b) {
                bool rejected = false;
                try {
                        auto values = Host::solve(matrix.data() + b * B3D_MATRIX_SIZE_PADDED,
                                                  rhs.data() + b * VEC_SIZE_PADDED, tolerance[b]);
                        std::copy(values.begin(), values.end(), solution.begin() + b * VEC_SIZE_PADDED);
                } catch (const std::runtime_error&) { rejected = true; }
                if (rejected != (b == 1 || b == 2 || b == 3 || b == 5))
                        throw std::runtime_error("Unexpected acceptance of direct solve");
        }
        double max_error = 0, residual_sq = 0, rhs_sq = 0;
        for (uint32_t k = 0; k < KNOT_POINTS; ++k) {
                for (uint32_t row = 0; row < STATE_SIZE; ++row) {
                        const uint32_t i = (k + 1) * STATE_SIZE + row;
                        max_error = std::max(max_error, std::abs(double(solution[i]) - expected[i]));
                        double ax = 0;
                        for (uint32_t col = 0; col < BLOCK_ROW_R_DIM; ++col)
                                ax += double(matrix[k * BLOCK_ROW_SIZE + row * BLOCK_ROW_R_DIM + col]) * solution[k * STATE_SIZE + col];
                        residual_sq += (ax - rhs[i]) * (ax - rhs[i]);
                        rhs_sq += double(rhs[i]) * rhs[i];
                }
        }
        for (uint32_t b = 0; b < batch; ++b) {
                for (uint32_t j = 0; j < VEC_SIZE_PADDED; ++j) {
                        const float value = solution[b * VEC_SIZE_PADDED + j];
                        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite output escaped");
                        if ((j < STATE_SIZE || j >= VEC_SIZE_PADDED - STATE_SIZE || b > 0) && value != 0)
                                throw std::runtime_error("Padding, failed lane, or zero RHS was not zero");
                }
        }
        if (max_error > 1e-5 || std::sqrt(residual_sq / rhs_sq) > 1e-5)
                throw std::runtime_error("Known Schur solution/residual mismatch");
        std::cout << "QDLDL padded block layout/status test: max_solution_error=" << max_error
                  << " relative_residual=" << std::sqrt(residual_sq / rhs_sq) << '\n';
}
