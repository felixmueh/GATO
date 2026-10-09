#pragma once

#include <cmath>
#include <cstddef>

namespace gato {

template<typename T>
bool finiteResidualBelowTolerance(const T* residual, std::size_t count, T tolerance)
{
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(residual[i]) || !(std::abs(residual[i]) < tolerance)) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool kktResidualsConverged(const T* state_stationarity, std::size_t state_count,
                          const T* control_stationarity, std::size_t control_count,
                          const T* primal_defects, std::size_t defect_count, T tolerance)
{
    return finiteResidualBelowTolerance(state_stationarity, state_count, tolerance)
        && finiteResidualBelowTolerance(control_stationarity, control_count, tolerance)
        && finiteResidualBelowTolerance(primal_defects, defect_count, tolerance);
}

}  // namespace gato
