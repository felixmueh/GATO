#pragma once
#include <cassert>

// Handwritten analytic plant. Acceleration controls and the standard
// trapezoidal integrator give exact constant-control double-integrator motion.
namespace grid {
constexpr int NUM_JOINTS = 2;
constexpr int NQ = 2, NX = 4, NU = 2;
constexpr int REFERENCE_SIZE = 6, EE_POS_SIZE = 6;
constexpr unsigned EE_POS_DYNAMIC_SHARED_MEM_COUNT = 0;
constexpr int SUGGESTED_THREADS = 128;
template<typename T> struct robotModel {};
}

#include "utils/linalg.cuh"

namespace gato::plant {
template<typename T> void* initializeDynamicsConstMem() { return nullptr; }
template<typename T> void freeDynamicsConstMem(void*) {}

template<typename T>
__device__ void forwardDynamics(T* acceleration, T*, T*, T* control, T*, void*, T* = nullptr)
{
    for (int i = threadIdx.x; i < 2; i += blockDim.x) acceleration[i] = control[i];
    __syncthreads();
}

template<typename T>
__device__ void forwardDynamicsAndGradient(T* derivative, T* acceleration,
    const T*, const T*, const T* control, T*, void*, T* = nullptr)
{
    for (int i = threadIdx.x; i < 12; i += blockDim.x)
        derivative[i] = (i == 8 || i == 11) ? T(1) : T(0);
    for (int i = threadIdx.x; i < 2; i += blockDim.x) acceleration[i] = control[i];
    __syncthreads();
}

__host__ __device__ constexpr unsigned forwardDynamics_TempMemSize_Shared() { return 4; }
__host__ __device__ constexpr unsigned forwardDynamicsAndGradient_TempMemSize_Shared() { return 4; }

template<typename T>
__device__ T overflow(T value, T limit)
{
    return value > limit ? value - limit : (value < -limit ? value + limit : T(0));
}

// Reference: goal_x, goal_y, obstacle_x, obstacle_y, effective_radius, unused.
template<typename T>
__device__ T obstacleResidual(const T* x, const T* reference, T* gradient)
{
    const T dx = x[0] - reference[2], dy = x[1] - reference[3];
    const T r = reference[4];
    const T inverse = r > T(0) ? T(1) / (r * r) : T(0);
    const T h = r > T(0) ? T(1) - (dx * dx + dy * dy) * inverse : T(0);
    gradient[0] = h > T(0) ? -T(2) * dx * inverse : T(0);
    gradient[1] = h > T(0) ? -T(2) * dy * inverse : T(0);
    return h > T(0) ? h : T(0);
}

template<typename T>
__device__ T trackingcost(uint32_t, uint32_t, uint32_t knots, T* x,
    T* reference, T* temp, const grid::robotModel<T>*, T q_cost, T qd_cost,
    T u_cost, T N_cost, T obstacle_cost, T terminal_obstacle_cost,
    T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost)
{
    if (threadIdx.x == 0) {
        const bool terminal = blockIdx.x == knots - 1;
        T cost = 0, gradient[2];
        for (int i = 0; i < 2; ++i) {
            const T e = x[i] - reference[i], v = x[i + 2];
            const T qp = overflow(x[i], T(.6)), vp = overflow(v, T(2));
            cost += T(.5) * ((terminal ? N_cost : q_cost) * e * e
                + qd_cost * v * v + q_lim_cost * qp * qp + vel_lim_cost * vp * vp);
            if (!terminal) {
                const T u = x[i + 4], up = overflow(u, T(4));
                cost += T(.5) * (u_cost * u * u + ctrl_lim_cost * up * up);
            }
        }
        const T h = obstacleResidual(x, reference, gradient);
        temp[0] = cost + T(.5) * (terminal ? terminal_obstacle_cost : obstacle_cost) * h * h;
    }
    __syncthreads();
    return temp[0];
}

__host__ inline unsigned trackingcost_TempMemCt_Shared(uint32_t, uint32_t, uint32_t) { return 1; }

template<typename T, bool computeR = true>
__device__ void trackingCostGradientAndHessian(uint32_t, uint32_t, T* x,
    T* reference, T* Q, T* q, T* R, T* r, T*, void*, T q_cost, T qd_cost,
    T u_cost, T, T obstacle_cost, T, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost)
{
    if (threadIdx.x == 0) {
        for (int i = 0; i < 16; ++i) Q[i] = T(0);
        if constexpr (computeR) for (int i = 0; i < 4; ++i) R[i] = T(0);
        T gradient[2];
        const T h = obstacleResidual(x, reference, gradient);
        for (int i = 0; i < 2; ++i) {
            const T qp = overflow(x[i], T(.6));
            const T vp = overflow(x[i + 2], T(2));
            q[i] = q_cost * (x[i] - reference[i]) + q_lim_cost * qp + obstacle_cost * h * gradient[i];
            q[i + 2] = qd_cost * x[i + 2] + vel_lim_cost * vp;
            Q[i * 4 + i] = q_cost + (qp != T(0) ? q_lim_cost : T(0));
            Q[(i + 2) * 4 + i + 2] = qd_cost + (vp != T(0) ? vel_lim_cost : T(0));
            for (int j = 0; j < 2; ++j) Q[j * 4 + i] += obstacle_cost * gradient[i] * gradient[j];
            if constexpr (computeR) {
                const T up = overflow(x[i + 4], T(4));
                r[i] = u_cost * x[i + 4] + ctrl_lim_cost * up;
                R[i * 2 + i] = u_cost + (up != T(0) ? ctrl_lim_cost : T(0));
            }
        }
    }
    __syncthreads();
}

template<typename T>
__device__ void trackingCostGradientAndHessian_lastblock(uint32_t nx, uint32_t nu,
    T* x, T* reference, T* Q, T* q, T* R, T* r, T* QN, T* qN, T* temp, void* model,
    T q_cost, T qd_cost, T u_cost, T N_cost, T obstacle_cost, T terminal_obstacle_cost,
    T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost)
{
    trackingCostGradientAndHessian<T>(nx, nu, x, reference, Q, q, R, r, temp, model,
        q_cost, qd_cost, u_cost, N_cost, obstacle_cost, terminal_obstacle_cost,
        q_lim_cost, vel_lim_cost, ctrl_lim_cost);
    trackingCostGradientAndHessian<T, false>(nx, nu, x + nx + nu, reference + 6,
        QN, qN, nullptr, nullptr, temp, model, N_cost, qd_cost, u_cost, N_cost,
        terminal_obstacle_cost, terminal_obstacle_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost);
}

__host__ __device__ constexpr unsigned trackingCostGradientAndHessian_TempMemSize_Shared() { return 1; }
} // namespace gato::plant
