#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "bsqp/qdldl_host.h"
namespace py = pybind11;
using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

template<unsigned N> py::array_t<float> solve_n(Array matrix, Array rhs, float tolerance) {
    constexpr unsigned X = 14;
    if (matrix.size() != N*X*3*X || rhs.size() != (N+2)*X)
        throw std::invalid_argument("Padded Schur shape does not match N");
    auto result = gato::qdldl_linsys::HostSchurSolver<X,N>::solve(
        matrix.data(), rhs.data(), tolerance);
    py::array_t<float> out((N+2)*X);
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}
PYBIND11_MODULE(cpu_qdldl,m) {
    m.attr("QDLDL_SOURCE")="df481005111f2864770a3eccb05ed4120727eff6";
    m.attr("PRECISION")="float32 QDLDL; float64 Pinocchio/NumPy assembly";
    m.def("solve",[](unsigned n, Array matrix, Array rhs, float tolerance) {
        switch(n) {
        case 16:return solve_n<16>(matrix,rhs,tolerance);
        case 32:return solve_n<32>(matrix,rhs,tolerance);
        case 64:return solve_n<64>(matrix,rhs,tolerance);
        case 128:return solve_n<128>(matrix,rhs,tolerance);
        case 256:return solve_n<256>(matrix,rhs,tolerance);
        case 512:return solve_n<512>(matrix,rhs,tolerance);
        default:throw std::invalid_argument("N must be 16,32,64,128,256,512");
        }
    });
}
