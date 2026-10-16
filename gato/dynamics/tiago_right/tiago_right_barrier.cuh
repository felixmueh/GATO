#pragma once
#include <math.h>
#if defined(__CUDACC__)
#define GATO_TIAGO_BARRIER_HD __host__ __device__
#else
#define GATO_TIAGO_BARRIER_HD
#endif
namespace gato { namespace plant {
        // Continue -log(distance) quadratically below one percent of the
        // bound interval. The value, gradient and Hessian share one C2
        // extension, including outside a limit. This fixed engineering choice
        // scales with the variable's raw units; it is not a physical margin.
        template<class T>
        GATO_TIAGO_BARRIER_HD constexpr T jointBarrierRelaxationDistance(T q_min, T q_max)
        {
                return static_cast<T>(0.01) * (q_max - q_min);
        }

        template<class T>
        GATO_TIAGO_BARRIER_HD T relaxedLogBarrierValue(T distance, T delta)
        {
                if (distance >= delta) { return -log(distance); }
                const T t = (delta - distance) / delta;
                return -log(delta) + t + static_cast<T>(0.5) * t * t;
        }

        template<class T>
        GATO_TIAGO_BARRIER_HD T relaxedLogBarrierGradient(T distance, T delta)
        {
                if (distance >= delta) { return -static_cast<T>(1) / distance; }
                const T t = (delta - distance) / delta;
                return -(static_cast<T>(1) + t) / delta;
        }

        template<class T>
        GATO_TIAGO_BARRIER_HD T relaxedLogBarrierHessian(T distance, T delta)
        {
                if (distance >= delta) { return static_cast<T>(1) / (distance * distance); }
                return static_cast<T>(1) / (delta * delta);
        }

        template<class T>
        GATO_TIAGO_BARRIER_HD T jointBarrier(T q, T q_min, T q_max)
        {
                const T delta = jointBarrierRelaxationDistance(q_min, q_max);
                return relaxedLogBarrierValue(q - q_min, delta) + relaxedLogBarrierValue(q_max - q, delta);
        }

        template<class T>
        GATO_TIAGO_BARRIER_HD T jointBarrierGradient(T q, T q_min, T q_max)
        {
                const T delta = jointBarrierRelaxationDistance(q_min, q_max);
                return relaxedLogBarrierGradient(q - q_min, delta) - relaxedLogBarrierGradient(q_max - q, delta);
        }

        template<class T>
        GATO_TIAGO_BARRIER_HD T jointBarrierHessian(T q, T q_min, T q_max)
        {
                const T delta = jointBarrierRelaxationDistance(q_min, q_max);
                return relaxedLogBarrierHessian(q - q_min, delta) + relaxedLogBarrierHessian(q_max - q, delta);
        }


// Experiment-only smooth replacement for the joint-position penalty. Its
// interior is exactly flat. A quartic crosses the physical margin and joins
// the historical one-sided quadratic continuation at the RAW bound in C2.
// The opposite, distant log side is omitted, and values have a new zero.
template<class T> struct TiagoBarrierDerivatives { T value, gradient, hessian; };

template<class T>
GATO_TIAGO_BARRIER_HD TiagoBarrierDerivatives<T> tiagoFlatBarrierSide(T outward, T margin, T delta)
{
    if (outward <= static_cast<T>(0)) { return {0, 0, 0}; }
    const T G = (static_cast<T>(2) + margin / delta) / delta;
    const T H = static_cast<T>(1) / (delta * delta);
    const T a = (static_cast<T>(3) * G - H * margin) / (static_cast<T>(3) * margin * margin);
    const T b = (H * margin - static_cast<T>(2) * G) / (static_cast<T>(4) * margin * margin * margin);
    if (outward < margin) {
        const T s = outward;
        return {s*s*s*(a + b*s), s*s*(static_cast<T>(3)*a + static_cast<T>(4)*b*s),
                s*(static_cast<T>(6)*a + static_cast<T>(12)*b*s)};
    }
    const T excess = outward - margin;
    const T at_margin = margin*margin*margin*(a+b*margin);
    return {at_margin + G*excess + static_cast<T>(0.5)*H*excess*excess, G+H*excess, H};
}

template<class T>
GATO_TIAGO_BARRIER_HD TiagoBarrierDerivatives<T> tiagoFlatJointBarrier(T q, T lower, T upper, T margin)
{
    const T delta = jointBarrierRelaxationDistance(lower, upper);
    const auto left = tiagoFlatBarrierSide(lower-q, margin, delta);
    const auto right = tiagoFlatBarrierSide(q-upper, margin, delta);
    return {left.value+right.value, right.gradient-left.gradient, left.hessian+right.hessian};
}

// These same scalar dispatch functions are compiled by the focused CPU checks
// and called from the GPU plant. Velocity/effort call the original functions.
template<class T>
GATO_TIAGO_BARRIER_HD T tiagoJointShapeValue(T q, T lower, T upper, T margin, int mode, T reference)
{
    if (mode == 1) { return tiagoFlatJointBarrier(q, lower, upper, margin).value; }
    const T base = jointBarrier(q, lower, upper);
    if (mode == 2) {
        return base - jointBarrier(reference, lower, upper) - jointBarrierGradient(reference, lower, upper)*(q-reference);
    }
    return base;
}
template<class T>
GATO_TIAGO_BARRIER_HD T tiagoJointShapeGradient(T q, T lower, T upper, T margin, int mode, T reference)
{
    if (mode == 1) { return tiagoFlatJointBarrier(q, lower, upper, margin).gradient; }
    const T base = jointBarrierGradient(q, lower, upper);
    if (mode == 2) { return base - jointBarrierGradient(reference, lower, upper); }
    return base;
}
template<class T>
GATO_TIAGO_BARRIER_HD T tiagoJointShapeHessian(T q, T lower, T upper, T margin, int mode, T reference)
{
    if (mode == 1) { return tiagoFlatJointBarrier(q, lower, upper, margin).hessian; }
    return jointBarrierHessian(q, lower, upper);
}

} } // namespace gato::plant
#undef GATO_TIAGO_BARRIER_HD
