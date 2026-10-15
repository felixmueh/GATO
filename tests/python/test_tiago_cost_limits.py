"""CPU-only checks of interior cost bounds and the existing C2 helper math."""
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[2]
PLANT = ROOT / "gato/dynamics/tiago_right"
HEADER = PLANT / "tiago_right_cost_limits.cuh"


def compiler():
    found = shutil.which("g++") or shutil.which("c++")
    if found is None:
        pytest.fail("A host C++ compiler is required for the requested CPU policy checks")
    return found


def program():
    joints = [j for j in ET.parse(PLANT / "tiago_right_arm.urdf").getroot().findall("joint")
              if j.get("type") != "fixed"]
    assert len(joints) == 7
    rows = ["{" + ",".join(j.find("limit").get(k) for k in ("lower", "upper", "velocity", "effort")) + "}"
            for j in joints]
    source = (PLANT / "tiago_right_plant.cuh").read_text()
    start = source.index("        template<class T>\n        __host__ __device__ constexpr T jointBarrierRelaxationDistance")
    end = source.index("        template<typename T>\n        __device__ void forwardDynamics(", start)
    # Compile the actual unchanged source helper bodies on the host. This does
    # not emulate kernels or claim CUDA numerical equivalence.
    helpers = source[start:end]
    return r'''
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include "tiago_right_cost_limits.cuh"
#define __host__
#define __device__
namespace gato { namespace plant {
HELPER_BODIES
}}
using namespace gato::plant;
const double raw[7][4] = {RAW};
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> bool close(T a, T b, T tolerance) {
    return std::abs(a-b) <= tolerance * (T(1) + std::abs(a) + std::abs(b));
}
template<class T> int check() {
    int samples = 0;
    const T tolerance = std::is_same<T,float>::value ? T(.03) : T(.0002);
    for (const auto& joint : raw) {
        const T lows[] = {T(joint[0]), T(-joint[2]), T(-joint[3])};
        const T highs[] = {T(joint[1]), T(joint[2]), T(joint[3])};
        const TiagoCostBounds<T> bounds[] = {
            tiagoJointCostBounds(lows[0], highs[0]),
            tiagoVelocityCostBounds(lows[1], highs[1]),
            tiagoControlCostBounds(lows[2], highs[2])};
        for (int category = 0; category < 3; ++category) {
            const auto b = bounds[category];
            require(b.lower < b.upper, "Cost interval must remain nonempty");
            require(b.lower >= lows[category] && b.upper <= highs[category], "Cost bounds must lie inside raw bounds");
            const double margin = category == 0 ? TIAGO_COST_JOINT_MARGIN_RAD : 0.;
            const double fraction = category == 1 ? TIAGO_COST_VELOCITY_LIMIT_FRACTION :
                                    category == 2 ? TIAGO_COST_CONTROL_LIMIT_FRACTION : 1.;
            require(close(b.lower, T(double(lows[category])*fraction+margin), T(32)*std::numeric_limits<T>::epsilon()), "Lower threshold differs");
            require(close(b.upper, T(double(highs[category])*fraction-margin), T(32)*std::numeric_limits<T>::epsilon()), "Upper threshold differs");
            if (margin == 0. && fraction == 1.) {
                require(std::memcmp(&b.lower, &lows[category], sizeof(T)) == 0 &&
                        std::memcmp(&b.upper, &highs[category], sizeof(T)) == 0, "Default policy changes raw bound bits");
            }
            const T delta = jointBarrierRelaxationDistance(b.lower, b.upper);
            const T positions[] = {b.lower-delta, b.lower, b.lower+T(.5)*delta,
                                   b.lower+delta, (b.lower+b.upper)/T(2),
                                   b.upper-T(.5)*delta, b.upper+delta};
            const T h = delta * (std::is_same<T,float>::value ? T(.01) : T(.0001));
            for (T x : positions) {
                const T value = jointBarrier(x, b.lower, b.upper);
                const T gradient = jointBarrierGradient(x, b.lower, b.upper);
                const T hessian = jointBarrierHessian(x, b.lower, b.upper);
                const T fd_gradient = (jointBarrier(x+h,b.lower,b.upper)-jointBarrier(x-h,b.lower,b.upper))/(T(2)*h);
                const T fd_hessian = (jointBarrierGradient(x+h,b.lower,b.upper)-jointBarrierGradient(x-h,b.lower,b.upper))/(T(2)*h);
                require(std::isfinite(value) && std::isfinite(gradient) && std::isfinite(hessian), "Nonfinite continued cost helper");
                require(close(gradient,fd_gradient,tolerance), "Continued value/gradient disagree at shifted bounds");
                require(close(hessian,fd_hessian,tolerance), "Continued gradient/Hessian disagree at shifted bounds");
                ++samples;
            }
            require(jointBarrierGradient(b.lower-delta,b.lower,b.upper) < T(0), "Lower outside descent must point inward");
            require(jointBarrierGradient(b.upper+delta,b.lower,b.upper) > T(0), "Upper outside descent must point inward");
        }
    }
    return samples;
}
int main() {
    std::cout << "CPU policy/C2 samples float=" << check<float>() << " double=" << check<double>() << '\n';
}
'''.replace("HELPER_BODIES", helpers).replace("RAW", ",\n".join(rows))


@pytest.mark.parametrize("defines", [[], ["-DTIAGO_COST_JOINT_MARGIN_RAD=0.02",
    "-DTIAGO_COST_VELOCITY_LIMIT_FRACTION=0.90", "-DTIAGO_COST_CONTROL_LIMIT_FRACTION=0.90"]], ids=["default", "interior"])
def test_cost_bounds_and_continued_derivatives(tmp_path, defines):
    source, binary = tmp_path / "policy.cpp", tmp_path / "policy"
    source.write_text(program())
    subprocess.run([compiler(), "-std=c++17", "-O2", "-I", str(PLANT), *defines,
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "float=147 double=147" in result.stdout


@pytest.mark.parametrize("define", ["-DTIAGO_COST_JOINT_MARGIN_RAD=-0.02",
    "-DTIAGO_COST_VELOCITY_LIMIT_FRACTION=0", "-DTIAGO_COST_CONTROL_LIMIT_FRACTION=1.1"])
def test_invalid_cost_policy_rejected(tmp_path, define):
    source = tmp_path / "invalid.cpp"
    source.write_text('#include "tiago_right_cost_limits.cuh"\nint main() {}\n')
    result = subprocess.run([compiler(), "-std=c++17", "-I", str(PLANT), define,
                             "-fsyntax-only", str(source)], capture_output=True, text=True)
    assert result.returncode != 0 and "static assertion failed" in result.stderr
