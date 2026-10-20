#pragma once
#include "cpu_strip_common.hpp"

namespace upstream_cpu {
class Schwarz final : public Method {
 public:
  void setup(const Blocks& H) override {
    local_.clear(); size_ = H.count * H.width;
    for (int start = 0; start < H.count; start += 3) {
      const int count = std::min(4, H.count - start), size = count * H.width;
      Matrix A = Matrix::Zero(size, size);
      for (int k = 0; k < count; ++k) {
        A.block(k * H.width, k * H.width, H.width, H.width) = H.diagonal[start + k];
        if (k) {
          A.block(k * H.width, (k - 1) * H.width, H.width, H.width) = H.lower[start + k - 1];
          A.block((k - 1) * H.width, k * H.width, H.width, H.width) = H.lower[start + k - 1].transpose();
        }
      }
      local_.emplace_back(); auto& local = local_.back();
      local.offset = start * H.width; local.size = size; factor(local.solver, A);
      if (start + count == H.count) break;
    }
  }
  Vector apply(const Vector& r) const override {
    Vector x = Vector::Zero(size_);
    for (const auto& local : local_)
      x.segment(local.offset, local.size) += local.solver.solve(r.segment(local.offset, local.size));
    return finite(std::move(x));
  }
 private:
  struct Local { int offset = 0, size = 0; Llt solver; };
  int size_ = 0; std::vector<Local> local_;
};
}  // namespace upstream_cpu
