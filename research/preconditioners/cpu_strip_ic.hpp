#pragma once
#include "cpu_strip_common.hpp"
#include <Eigen/IterativeLinearSolvers>

namespace upstream_cpu {
class IncompleteCholesky final : public Method {
 public:
  void setup(const Blocks& H) override {
    const Eigen::SparseMatrix<float> sparse = H.sparse();
    solver_.compute(sparse);
    if (solver_.info() != Eigen::Success || !solver_.matrixL().coeffs().allFinite() ||
        !solver_.scalingS().allFinite()) throw std::runtime_error("CPU incomplete Cholesky failed");
  }
  Vector apply(const Vector& r) const override { return finite(solver_.solve(r)); }
  // Eigen's default AMD/scaling/fill/shift strategy is retained. Its actual
  // adaptive shift is not exposed; this value is the configured initial shift.
  float configured_initial_shift() const override { return 1.0e-3f; }
  bool actual_shift_known() const override { return false; }
 private:
  Eigen::IncompleteCholesky<float> solver_;
};
}  // namespace upstream_cpu
