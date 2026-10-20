#pragma once
#include "cpu_strip_common.hpp"

namespace upstream_cpu {
class BlockJacobi final : public Method {
 public:
  void setup(const Blocks& H) override {
    width_ = H.width; diagonal_.resize(H.count);
    for (int k = 0; k < H.count; ++k) factor(diagonal_[k], H.diagonal[k]);
  }
  Vector apply(const Vector& r) const override {
    Vector x(r.size());
    for (int k = 0; k < static_cast<int>(diagonal_.size()); ++k)
      x.segment(k * width_, width_) = diagonal_[k].solve(r.segment(k * width_, width_));
    return finite(std::move(x));
  }
 private:
  int width_ = 0; std::vector<Llt> diagonal_;
};
}  // namespace upstream_cpu
