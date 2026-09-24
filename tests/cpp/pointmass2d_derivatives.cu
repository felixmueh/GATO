// Standalone CUDA finite-difference check of the analytic point-mass plant.
// CMake target/CTest name: pointmass2d_derivatives.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "settings.h"
#include "dynamics/integrator.cuh"

void check(cudaError_t error)
{
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        std::exit(4);
    }
}

template<bool Terminal>
__global__ void evaluate(double* inputs, double* refs, double* output)
{
    if (blockIdx.x != (Terminal ? 31 : 0)) return;
    __shared__ double x[10], ref[6], Q[16], q[4], R[4], r[2], temp[64];
    __shared__ double A[16], B[8], next[4];
    for (int i = threadIdx.x; i < 10; i += blockDim.x) x[i] = inputs[i];
    for (int i = threadIdx.x; i < 6; i += blockDim.x) ref[i] = refs[i];
    __syncthreads();
    double cost = gato::plant::trackingcost<double>(4, 2, 32, x, ref, temp,
        nullptr, .3, .2, .01, 10., 2., 3., 4., 5., 6.);
    gato::plant::trackingCostGradientAndHessian<double, !Terminal>(4, 2, x,
        ref, Q, q, R, r, temp, nullptr, Terminal ? 10. : .3, .2, .01, 10.,
        Terminal ? 3. : 2., 3., 4., 5., 6.);
    if (threadIdx.x == 0) {
        output[0] = cost;
        for (int i = 0; i < 4; ++i) output[1+i] = q[i];
        for (int i = 0; i < 2; ++i) output[5+i] = Terminal ? 0. : r[i];
        for (int i = 0; i < 16; ++i) output[7+i] = Q[i];
    }
    gato::plant::compute_linearized_dynamics<double>(x, A, B, next, temp, nullptr, .05);
    if (threadIdx.x == 0) {
        for (int i = 0; i < 16; ++i) output[23+i] = A[i];
        for (int i = 0; i < 8; ++i) output[39+i] = B[i];
        for (int i = 0; i < 4; ++i) output[47+i] = next[i];
    }
}

int main()
{
    double *x, *ref, *out;
    check(cudaMallocManaged(&x, 10*sizeof(double)));
    check(cudaMallocManaged(&ref, 6*sizeof(double)));
    check(cudaMallocManaged(&out, 51*sizeof(double)));
    const double samples[5][6] = {
        {.03, .02, .3, -.2, .5, -.4},
        {.8, -.7, 2.3, -2.2, 4.5, -4.2},
        {.3, .4, .1, .2, -.2, .3},
        {0., 0., 0., 0., 0., 0.},
        {.095, 0., .1, -.1, .2, -.2},
    };
    double max_error = 0.;
    for (bool terminal : {false, true}) {
        auto run = [&]() {
            if (terminal) evaluate<true><<<32, 128>>>(x, ref, out);
            else evaluate<false><<<1, 128>>>(x, ref, out);
            check(cudaDeviceSynchronize());
        };
        for (const auto& sample : samples) {
            for (int i = 0; i < 10; ++i) x[i] = i < 6 ? sample[i] : 0.;
            ref[0] = .4; ref[1] = -.1; ref[2] = 0.; ref[3] = 0.; ref[4] = .1; ref[5] = 0.;
            run();
            double gradient[6];
            for (int i = 0; i < 6; ++i) gradient[i] = out[1+i];
            // Positive 2x2 position GN block and positive velocity diagonal.
            if (out[7] <= 0 || out[12] <= 0 || out[17] <= 0 || out[22] <= 0
                || std::abs(out[8]-out[11]) > 1e-12
                || out[7]*out[12]-out[8]*out[11] < -1e-9) return 5;
            for (int i = 0; i < 16; ++i) {
                const double expected = (i/4 == i%4 ? 1. : 0.)
                    + ((i/4 == 2 && i%4 == 0) || (i/4 == 3 && i%4 == 1) ? .05 : 0.);
                if (std::abs(out[23+i]-expected) > 1e-12) return 2;
            }
            for (int i = 0; i < 8; ++i) {
                const double expected = (i/4 == i%4 ? .00125 : 0.)
                    + (i/4+2 == i%4 ? .05 : 0.);
                if (std::abs(out[39+i]-expected) > 1e-12) return 3;
            }
            for (int i = 0; i < 6; ++i) {
                const double old = x[i], epsilon = 1e-6;
                x[i] = old+epsilon; run(); const double plus = out[0];
                x[i] = old-epsilon; run(); const double minus = out[0];
                x[i] = old;
                max_error = std::max(max_error, std::abs((plus-minus)/(2*epsilon)-gradient[i]));
            }
        }
    }
    check(cudaFree(x)); check(cudaFree(ref)); check(cudaFree(out));
    std::printf("Running/terminal gradient max error %.9g; analytic A/B and PSD checks passed\n", max_error);
    return max_error < 1e-6 ? 0 : 1;
}
