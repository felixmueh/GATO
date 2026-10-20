#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Portable CPU preconditioner mathematics. No robot dimensions or private
// solver code: setup consumes the public row-major [L|D|R] Schur strips.
namespace upstream_cpu {
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXf;
using Llt = Eigen::LLT<Matrix>;
inline void factor(Llt& solver, const Matrix& A) {
  solver.compute(A);
  if (solver.info() != Eigen::Success || !solver.matrixLLT().allFinite())
    throw std::runtime_error("CPU block LLT failed");
}
inline Vector finite(Vector value) {
  if (!value.allFinite()) throw std::runtime_error("Nonfinite CPU preconditioner output");
  return value;
}

struct Blocks {
  int count = 0, width = 0;
  double projection_relative = 0.0;
  std::vector<Matrix> diagonal, lower;
  void setup(const float* S, int knots, int block_size) {
    if (!S || knots < 1 || block_size < 1) throw std::invalid_argument("Invalid Schur strip dimensions");
    Eigen::setNbThreads(1); count = knots; width = block_size;
    diagonal.resize(count); lower.resize(count - 1);
    const auto element = [&](int k, int row, int slot, int col) {
      return S[(static_cast<size_t>(k) * width + row) * (3 * width) + slot * width + col];
    };
    double original_sq = 0.0, projection_sq = 0.0;
    // O(knots*width^2), no full dense matrix. Only factor setup sees the
    // symmetric projection H=-0.5*(S+S^T); GPU matvec always uses raw S.
    for (int k = 0; k < count; ++k) {
      diagonal[k].resize(width, width);
      if (k > 0) lower[k - 1].resize(width, width);
      for (int row = 0; row < width; ++row)
        for (int col = 0; col < width; ++col) {
          const float a = element(k, row, 1, col), b = element(k, col, 1, row);
          if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("Nonfinite CPU setup Schur input");
          diagonal[k](row, col) = -0.5f * (a + b);
          original_sq += double(a) * a; projection_sq += 0.25 * (double(a) - b) * (double(a) - b);
          if (k > 0) {
            const float l = element(k, row, 0, col), u = element(k - 1, col, 2, row);
            if (!std::isfinite(l) || !std::isfinite(u)) throw std::runtime_error("Nonfinite CPU setup Schur input");
            lower[k - 1](row, col) = -0.5f * (l + u);
            original_sq += double(l) * l + double(u) * u;
            projection_sq += 0.5 * (double(l) - u) * (double(l) - u);
          }
        }
    }
    projection_relative = std::sqrt(projection_sq / std::max(original_sq, 1.0e-30));
  }
  Eigen::SparseMatrix<float> sparse() const {
    std::vector<Eigen::Triplet<float>> entries;
    entries.reserve(static_cast<size_t>(3 * count - 2) * width * width);
    for (int k = 0; k < count; ++k)
      for (int row = 0; row < width; ++row)
        for (int col = 0; col < width; ++col) {
          if (diagonal[k](row, col) != 0.0f)
            entries.emplace_back(k * width + row, k * width + col, diagonal[k](row, col));
          if (k > 0 && lower[k - 1](row, col) != 0.0f) {
            const float value = lower[k - 1](row, col);
            entries.emplace_back(k * width + row, (k - 1) * width + col, value);
            entries.emplace_back((k - 1) * width + col, k * width + row, value);
          }
        }
    Eigen::SparseMatrix<float> result(count * width, count * width);
    result.setFromTriplets(entries.begin(), entries.end()); result.makeCompressed();
    return result;
  }
};

class Method {
 public:
  virtual ~Method() = default;
  virtual void setup(const Blocks& H) = 0;
  virtual Vector apply(const Vector& r) const = 0;  // positive H convention
  virtual float configured_initial_shift() const { return 0.0f; }
  virtual bool actual_shift_known() const { return true; }
};
}  // namespace upstream_cpu
