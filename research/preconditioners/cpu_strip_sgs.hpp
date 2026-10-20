#pragma once
#include "cpu_strip_common.hpp"

namespace upstream_cpu {
class BlockSgs final : public Method {
 public:
  void setup(const Blocks& H) override {
    width_ = H.width; lower_ = H.lower; diagonal_.resize(H.count);
    for (int k = 0; k < H.count; ++k) factor(diagonal_[k], H.diagonal[k]);
  }
  Vector apply(const Vector& r) const override {
    const int count = static_cast<int>(diagonal_.size()); Vector y(r.size());
    for (int k = 0; k < count; ++k) {
      Vector local = r.segment(k * width_, width_);
      if (k) local.noalias() -= lower_[k - 1] * y.segment((k - 1) * width_, width_);
      y.segment(k * width_, width_) = diagonal_[k].solve(local);
    }
    Vector x = y;
    for (int k = count - 2; k >= 0; --k) {
      Vector coupling = lower_[k].transpose() * x.segment((k + 1) * width_, width_);
      x.segment(k * width_, width_) -= diagonal_[k].solve(coupling);
    }
    return finite(std::move(x));
  }
 private:
  int width_ = 0; std::vector<Llt> diagonal_; std::vector<Matrix> lower_;
};
}  // namespace upstream_cpu
