#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "bsqp/kkt_convergence.h"

void require(bool result, const char* message)
{
    if (!result) throw std::runtime_error(message);
}

template<typename T>
void check()
{
    const T tolerance = static_cast<T>(.001);
    std::vector<T> q(448, 0), r(224, 0), c(448, 0);
    auto converged = [&] {
        return gato::kktResidualsConverged(q.data(), q.size(), r.data(), r.size(),
                                         c.data(), c.size(), tolerance);
    };
    require(converged(), "finite zero KKT point must be certified");
    for (auto* residual : {&q, &r, &c}) {
        std::fill(residual->begin(), residual->end(), tolerance / 2);
    }
    require(converged(), "finite residuals strictly inside the configured gate must be certified");
    for (auto* residual : {&q, &r, &c}) {
        for (T value : {tolerance, -tolerance, tolerance * 2, -tolerance * 2}) {
            const auto last = residual->size() - 1;
            (*residual)[last] = value;
            require(!converged(), "state/control/primal violation must exclude a certificate");
            (*residual)[last] = tolerance / 2;
        }
        for (std::size_t i = 0; i < residual->size(); ++i) {
            for (T value : {std::numeric_limits<T>::quiet_NaN(),
                            std::numeric_limits<T>::infinity(),
                           -std::numeric_limits<T>::infinity()}) {
                (*residual)[i] = value;
                require(!converged(), "a nonfinite value anywhere must exclude a certificate");
            }
            (*residual)[i] = tolerance / 2;
        }
    }
    require(converged(), "restored finite point must still be certifiable");
}

int main()
{
    check<float>();
    check<double>();
    std::cout << "finite state/control/primal certificate checks pass for float and double\n";
}
