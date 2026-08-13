#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "dynamics/indy7/indy7_plant.cuh"
#include "constants.h"
#include "types.cuh"
#include "bsqp/kernels/pcg.cuh"

using namespace gato::constants;

namespace {

struct Result {
        int32_t status;
        uint32_t iterations;
        std::vector<float> solution;
};

Result run_case(float matrix_sign, float preconditioner_sign,
                float tolerance, uint32_t max_iterations,
                float rhs_value)
{
        std::vector<float> matrix(B3D_MATRIX_SIZE_PADDED, 0.0f);
        std::vector<float> preconditioner(B3D_MATRIX_SIZE_PADDED, 0.0f);
        std::vector<float> rhs(VEC_SIZE_PADDED, 0.0f);
        std::vector<float> solution(VEC_SIZE_PADDED, 0.0f);

        for (uint32_t block = 0; block < KNOT_POINTS; ++block) {
                const uint32_t base = block * BLOCK_ROW_SIZE;
                for (uint32_t row = 0; row < STATE_SIZE; ++row) {
                        const uint32_t diagonal = base + row * BLOCK_ROW_R_DIM
                                                  + STATE_SIZE + row;
                        matrix[diagonal] = matrix_sign;
                        preconditioner[diagonal] = preconditioner_sign;
                        rhs[(block + 1) * STATE_SIZE + row] = rhs_value;
                }
        }

        float *d_matrix, *d_preconditioner, *d_rhs, *d_solution, *d_tolerance;
        uint32_t* d_iterations;
        int32_t *d_skip, *d_status;
        gpuErrchk(cudaMalloc(&d_matrix, matrix.size() * sizeof(float)));
        gpuErrchk(cudaMalloc(&d_preconditioner, preconditioner.size() * sizeof(float)));
        gpuErrchk(cudaMalloc(&d_rhs, rhs.size() * sizeof(float)));
        gpuErrchk(cudaMalloc(&d_solution, solution.size() * sizeof(float)));
        gpuErrchk(cudaMalloc(&d_tolerance, sizeof(float)));
        gpuErrchk(cudaMalloc(&d_iterations, sizeof(uint32_t)));
        gpuErrchk(cudaMalloc(&d_skip, sizeof(int32_t)));
        gpuErrchk(cudaMalloc(&d_status, sizeof(int32_t)));
        gpuErrchk(cudaMemcpy(d_matrix, matrix.data(), matrix.size() * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_preconditioner, preconditioner.data(), preconditioner.size() * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_rhs, rhs.data(), rhs.size() * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemset(d_solution, 0, solution.size() * sizeof(float)));
        gpuErrchk(cudaMemcpy(d_tolerance, &tolerance, sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemset(d_iterations, 0, sizeof(uint32_t)));
        gpuErrchk(cudaMemset(d_skip, 0, sizeof(int32_t)));
        gpuErrchk(cudaMemset(d_status, 0, sizeof(int32_t)));

        SchurSystem<float, 1> schur{d_matrix, d_preconditioner, d_rhs};
        solvePCGBatched<float, 1>(d_solution, schur, d_tolerance,
                                 max_iterations, d_skip, d_iterations,
                                 d_status, nullptr);
        gpuErrchk(cudaDeviceSynchronize());

        Result result;
        gpuErrchk(cudaMemcpy(&result.status, d_status, sizeof(int32_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&result.iterations, d_iterations, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(solution.data(), d_solution, solution.size() * sizeof(float), cudaMemcpyDeviceToHost));
        result.solution = std::move(solution);

        gpuErrchk(cudaFree(d_matrix));
        gpuErrchk(cudaFree(d_preconditioner));
        gpuErrchk(cudaFree(d_rhs));
        gpuErrchk(cudaFree(d_solution));
        gpuErrchk(cudaFree(d_tolerance));
        gpuErrchk(cudaFree(d_iterations));
        gpuErrchk(cudaFree(d_skip));
        gpuErrchk(cudaFree(d_status));
        return result;
}

bool require(bool condition, const char* message)
{
        if (!condition) std::cerr << message << '\n';
        return condition;
}

}  // namespace

int main()
{
        bool ok = true;

        const Result zero = run_case(-1.0f, -1.0f, 1e-4f, 20, 0.0f);
        ok &= require(zero.status == PCG_CONVERGED && zero.iterations == 0,
                      "zero residual was not immediate convergence");

        const Result solved = run_case(-1.0f, -1.0f, 1e-4f, 20, 1.0f);
        ok &= require(solved.status == PCG_CONVERGED && solved.iterations == 1,
                      "negative-definite identity system did not converge in one iteration");
        for (uint32_t block = 0; block < KNOT_POINTS; ++block) {
                for (uint32_t row = 0; row < STATE_SIZE; ++row) {
                        ok &= require(std::abs(solved.solution[(block + 1) * STATE_SIZE + row] + 1.0f) < 1e-6f,
                                      "converged solution failed explicit residual check");
                }
        }

        const Result attenuated = run_case(-1.0f, -1e-12f, 8e-4f, 20, 1.0f);
        ok &= require(attenuated.status == PCG_CONVERGED
                          && attenuated.iterations == 1,
                      "tiny preconditioned energy caused false convergence");

        const Result invalid = run_case(-1.0f, -1.0f, 0.0f, 20, 1.0f);
        ok &= require(invalid.status == PCG_INVALID_TOLERANCE && invalid.iterations == 0,
                      "zero tolerance was not rejected");
        const Result nan_tolerance = run_case(
            -1.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(), 20, 1.0f);
        ok &= require(nan_tolerance.status == PCG_INVALID_TOLERANCE
                          && nan_tolerance.iterations == 0,
                      "NaN tolerance was not rejected");

        const Result exhausted = run_case(-1.0f, -1.0f, 1e-4f, 0, 1.0f);
        ok &= require(exhausted.status == PCG_MAX_ITERS && exhausted.iterations == 0,
                      "zero iteration budget was not reported as exhaustion");

        const Result curvature = run_case(1.0f, -1.0f, 1e-4f, 20, 1.0f);
        ok &= require(curvature.status == PCG_BREAKDOWN,
                      "wrong curvature sign was not reported as breakdown");
        const Result preconditioner = run_case(-1.0f, 1.0f, 1e-4f, 20, 1.0f);
        ok &= require(preconditioner.status == PCG_BREAKDOWN,
                      "wrong preconditioner sign was not reported as breakdown");

        const Result nonfinite = run_case(-1.0f, -1.0f, 1e-4f, 20,
                                          std::numeric_limits<float>::infinity());
        ok &= require(nonfinite.status == PCG_BREAKDOWN && nonfinite.iterations == 0,
                      "nonfinite initial true residual was not reported as breakdown");
        const Result overflow = run_case(-1.0f, -1.0f, 1e-4f, 20, 1e20f);
        ok &= require(overflow.status == PCG_BREAKDOWN && overflow.iterations == 0,
                      "finite residual with overflowed norm was not breakdown");

        float* d_lambda;
        int32_t* d_status;
        std::vector<float> lambda(2 * VEC_SIZE_PADDED, 7.0f);
        const int32_t status[2] = {PCG_CONVERGED, PCG_BREAKDOWN};
        gpuErrchk(cudaMalloc(&d_lambda, lambda.size() * sizeof(float)));
        gpuErrchk(cudaMalloc(&d_status, sizeof(status)));
        gpuErrchk(cudaMemcpy(d_lambda, lambda.data(), lambda.size() * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_status, status, sizeof(status), cudaMemcpyHostToDevice));
        resetFailedPCGIterates<float, 2>(d_lambda, d_status, nullptr);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(lambda.data(), d_lambda, lambda.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (uint32_t index = 0; index < VEC_SIZE_PADDED; ++index) {
                ok &= require(lambda[index] == 7.0f,
                              "reset changed a converged lane");
                ok &= require(lambda[VEC_SIZE_PADDED + index] == 0.0f,
                              "reset retained a failed lane");
        }
        gpuErrchk(cudaFree(d_lambda));
        gpuErrchk(cudaFree(d_status));

        return ok ? 0 : 1;
}
