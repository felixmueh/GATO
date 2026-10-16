#pragma once

// Interior bounds for soft limit COSTS only. Raw robot limits, dynamics and
// physical validation remain in their existing sources. Defaults preserve the
// original cost intervals. Experiments must record any non-default build values.
#ifndef TIAGO_COST_JOINT_MARGIN_RAD
#define TIAGO_COST_JOINT_MARGIN_RAD 0.0
#endif
#ifndef TIAGO_COST_VELOCITY_LIMIT_FRACTION
#define TIAGO_COST_VELOCITY_LIMIT_FRACTION 1.0
#endif
#ifndef TIAGO_COST_CONTROL_LIMIT_FRACTION
#define TIAGO_COST_CONTROL_LIMIT_FRACTION 1.0
#endif

static_assert(TIAGO_COST_JOINT_MARGIN_RAD >= 0.0, "Joint cost margin must be nonnegative");
static_assert(TIAGO_COST_VELOCITY_LIMIT_FRACTION > 0.0 && TIAGO_COST_VELOCITY_LIMIT_FRACTION <= 1.0,
              "Velocity cost fraction must be in (0, 1]");
static_assert(TIAGO_COST_CONTROL_LIMIT_FRACTION > 0.0 && TIAGO_COST_CONTROL_LIMIT_FRACTION <= 1.0,
              "Control cost fraction must be in (0, 1]");

#if defined(__CUDACC__)
#define GATO_TIAGO_COST_HOST_DEVICE __host__ __device__
#else
#define GATO_TIAGO_COST_HOST_DEVICE
#endif

namespace gato {
namespace plant {

        template<class T>
        struct TiagoCostBounds {
                T lower;
                T upper;
        };

        // The configured absolute margin must leave each joint interval nonempty.
        // The existing C2 continuation uses 1% of this resulting cost interval.
        template<class T>
        GATO_TIAGO_COST_HOST_DEVICE constexpr TiagoCostBounds<T> tiagoJointCostBounds(T lower, T upper)
        {
                return {lower + static_cast<T>(TIAGO_COST_JOINT_MARGIN_RAD),
                        upper - static_cast<T>(TIAGO_COST_JOINT_MARGIN_RAD)};
        }

        // Current velocity and effort limits are symmetric about zero. Fractions
        // preserve that center, including the distinct 43/26 Nm physical limits.
        template<class T>
        GATO_TIAGO_COST_HOST_DEVICE constexpr TiagoCostBounds<T> tiagoVelocityCostBounds(T lower, T upper)
        {
                return {lower * static_cast<T>(TIAGO_COST_VELOCITY_LIMIT_FRACTION),
                        upper * static_cast<T>(TIAGO_COST_VELOCITY_LIMIT_FRACTION)};
        }

        template<class T>
        GATO_TIAGO_COST_HOST_DEVICE constexpr TiagoCostBounds<T> tiagoControlCostBounds(T lower, T upper)
        {
                return {lower * static_cast<T>(TIAGO_COST_CONTROL_LIMIT_FRACTION),
                        upper * static_cast<T>(TIAGO_COST_CONTROL_LIMIT_FRACTION)};
        }

}  // namespace plant
}  // namespace gato

#undef GATO_TIAGO_COST_HOST_DEVICE
