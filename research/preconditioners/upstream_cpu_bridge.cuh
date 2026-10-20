#pragma once
#include "upstream_timing.cuh"
#include "cpu_strip_methods.hpp"
#include <type_traits>

// Experimental host execution of the CURRENT PUBLIC GLASS PCG formula.
// Source: pinned GLASS9e57178 src/base/pcg/solve.cuh, SHA256
// f4b087771e28363ca6d1aad7f7ca5edb714fbf25e38b1a47bb365dda9983b131.
// Native signs, input seed, order of updates, rho stopping and passed budget
// follow that public source. Host Eigen reductions/updates, per-matvec launches
// and synchronization differ from fused GLASS and are charged explicitly.
// cpu_native_control applies the ACTUAL native P, including every off-diagonal,
// to distinguish backend effects from changing the preconditioner. No private
// recurrence, seed guard, residual target, fallback or solver change is used.
namespace upstream_bench {
namespace cpu_detail {
using upstream_cpu::Vector;
constexpr unsigned D = gato::constants::STATE_SIZE;
constexpr unsigned N = KNOT_POINTS;
constexpr unsigned V = gato::constants::VEC_SIZE_PADDED;
constexpr unsigned M = gato::constants::B3D_MATRIX_SIZE_PADDED;

struct Stage {
  double& total; Clock::time_point start = Clock::now();
  explicit Stage(double& value) : total(value) {}
  ~Stage() { total += elapsed(start); }
};
struct Control { float epsilon; int32_t flag; uint32_t iterations; };
struct Workspace {
  float *S = nullptr, *P = nullptr, *b = nullptr, *x = nullptr, *input = nullptr, *output = nullptr;
  Control* control = nullptr;
  float *device_input = nullptr, *device_output = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  Workspace() {
    try {
      for (auto** pointer : {&S, &P}) check(cudaMallocHost(pointer, M * sizeof(float)));
      for (auto** pointer : {&b, &x, &input, &output}) check(cudaMallocHost(pointer, V * sizeof(float)));
      check(cudaMallocHost(&control, sizeof(Control)));
      check(cudaMalloc(&device_input, V * sizeof(float))); check(cudaMalloc(&device_output, V * sizeof(float)));
      check(cudaMemset(device_output, 0, V * sizeof(float)));
      check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    } catch (...) { release(); throw; }
  }
  void release() {
    for (auto* pointer : {S, P, b, x, input, output}) if (pointer) cudaFreeHost(pointer);
    if (control) cudaFreeHost(control);
    if (device_input) cudaFree(device_input); if (device_output) cudaFree(device_output);
    if (start) cudaEventDestroy(start); if (stop) cudaEventDestroy(stop);
  }
  ~Workspace() { release(); }
};
inline Workspace& workspace() { static thread_local Workspace value; return value; }

template<class Copies>
void transfer(Record& record, Copies copies) {
  check(cudaDeviceSynchronize());  // Previous compute stays OUTSIDE copy subtraction.
  const auto start = Clock::now();
  auto account = [&] { const double us = elapsed(start); record.additional_transfer_wall_us += us; additional_transfer_wall_us += us; };
  try { copies(); check(cudaDeviceSynchronize()); } catch (...) { account(); throw; }
  account();
}

__global__ void raw_matvec(float* output, const float* matrix, const float* input) {
  glass::block::bdmv<float, N, D, false>(output, matrix, input);
}
inline Vector matvec(const Vector& input, const float* S, Workspace& work, Record& record) {
  { Stage stage(record.cpu_algebra_us); std::copy_n(input.data(), V, work.input); }
  transfer(record, [&] { check(cudaMemcpyAsync(work.device_input, work.input, V * sizeof(float), cudaMemcpyHostToDevice)); });
  check(cudaEventRecord(work.start));
  raw_matvec<<<1, sqp::PCG_THREADS>>>(work.device_output, S, work.device_input);
  check(cudaGetLastError()); check(cudaEventRecord(work.stop)); check(cudaEventSynchronize(work.stop));
  float ms = 0; check(cudaEventElapsedTime(&ms, work.start, work.stop)); record.matvec_gpu_us += 1000.0 * ms;
  transfer(record, [&] { check(cudaMemcpyAsync(work.output, work.device_output, V * sizeof(float), cudaMemcpyDeviceToHost)); });
  Stage stage(record.cpu_algebra_us);
  return upstream_cpu::finite(Eigen::Map<const Vector>(work.output, V));
}
inline Vector native_apply(const float* P, const Vector& r) {
  // Same [L|D|R] indexing and scalar column order as GLASS bdmv, on the CPU.
  Vector z = Vector::Zero(V);
  for (unsigned k = 0; k < N; ++k)
    for (unsigned row = 0; row < D; ++row) {
      float sum = 0.0f;
      for (unsigned col = 0; col < 3 * D; ++col)
        sum += P[(k * D + row) * (3 * D) + col] * r[k * D + col];
      z[(k + 1) * D + row] = sum;
    }
  return upstream_cpu::finite(std::move(z));
}
}  // namespace cpu_detail

inline void prepare_cpu_bridge() { Eigen::setNbThreads(1); (void)cpu_detail::workspace(); }

inline void solve_cpu_public(float* lambda, SchurSystem<float> schur, float* epsilon,
                             uint32_t max_iters, int32_t* flags, uint32_t* iterations,
                             Record& record) {
  using namespace cpu_detail;
  auto& work = workspace(); record.backend = "host_public_glass_formula_gpu_bdmv";
  transfer(record, [&] {
    check(cudaMemcpyAsync(work.S, schur.d_S_batch, M * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpyAsync(work.P, schur.d_P_inv_batch, M * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpyAsync(work.b, schur.d_gamma_batch, V * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpyAsync(work.x, lambda, V * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpyAsync(&work.control->epsilon, epsilon, sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpyAsync(&work.control->flag, flags, sizeof(int32_t), cudaMemcpyDeviceToHost));
  });
  auto publish = [&](uint32_t count, bool write_x) {
    work.control->iterations = count;
    transfer(record, [&] {
      if (write_x) check(cudaMemcpyAsync(lambda, work.x, V * sizeof(float), cudaMemcpyHostToDevice));
      check(cudaMemcpyAsync(iterations, &work.control->iterations, sizeof(uint32_t), cudaMemcpyHostToDevice));
    });
  };
  if (work.control->flag) { publish(0, false); return; }

  std::unique_ptr<upstream_cpu::Method> preconditioner;
  const bool native_control = method == "cpu_native_control" || method == "symmetric_stair" ||
                              method == "polynomial2" || method == "polynomial3";
  const int polynomial_order = method == "polynomial2" ? 2 : method == "polynomial3" ? 3 : 1;
  if (!native_control) {
    Stage setup(record.cpu_setup_us);
    upstream_cpu::Blocks H; H.setup(work.S, N, D);
    record.input_projection_relative = H.projection_relative;
    preconditioner = upstream_cpu::make(method); preconditioner->setup(H);
    record.configured_initial_shift = preconditioner->configured_initial_shift();
    record.actual_shift_known = preconditioner->actual_shift_known();
  }
  auto apply = [&](const Vector& r) {
    Stage stage(record.cpu_apply_us);
    if (native_control) {
      Vector z = native_apply(work.P, r);
      for (int pass = 1; pass < polynomial_order; ++pass) {
        const Vector residual = r - native_apply(work.S, z);
        z += native_apply(work.P, residual);
      }
      return upstream_cpu::finite(std::move(z));
    }
    Vector z = Vector::Zero(V);
    // Positive factor convention H=-S: native inverse action is -G*r.
    z.segment(D, N * D) = -preconditioner->apply(r.segment(D, N * D));
    return upstream_cpu::finite(std::move(z));
  };
  Vector x, b, r, z, p;
  { Stage stage(record.cpu_algebra_us); x = Eigen::Map<const Vector>(work.x, V); b = Eigen::Map<const Vector>(work.b, V); }
  const Vector Sx = matvec(x, schur.d_S_batch, work, record);
  { Stage stage(record.cpu_algebra_us); r = b - Sx; }
  z = apply(r);
  float rho, rho_init;
  { Stage stage(record.cpu_algebra_us); p = z; rho = r.dot(z); rho_init = std::abs(rho); }
  if (!std::isfinite(rho)) throw std::runtime_error("Nonfinite host public PCG initial rho");
  if (rho_init < 1.0e-6f) { publish(0, true); return; }
  uint32_t count = 0;
  for (uint32_t i = 0; i < max_iters; ++i) {
    count = i + 1;
    const Vector Ap = matvec(p, schur.d_S_batch, work, record);
    {
      Stage stage(record.cpu_algebra_us);
      const float alpha = rho / p.dot(Ap);
      if (!std::isfinite(alpha)) throw std::runtime_error("Nonfinite host public PCG alpha");
      x.noalias() += alpha * p; r.noalias() -= alpha * Ap;
    }
    z = apply(r);
    bool converged;
    {
      Stage stage(record.cpu_algebra_us);
      const float next = r.dot(z);
      if (!std::isfinite(next)) throw std::runtime_error("Nonfinite host public PCG rho");
      converged = std::abs(next) < 1.0e-6f + work.control->epsilon * rho_init;
      if (!converged) {
        const float beta = next / rho; rho = next;
        if (!std::isfinite(beta)) throw std::runtime_error("Nonfinite host public PCG beta");
        p = z + beta * p;
      }
    }
    if (converged) break;
  }
  { Stage stage(record.cpu_algebra_us); if (!x.allFinite()) throw std::runtime_error("Nonfinite host public PCG result"); std::copy_n(x.data(), V, work.x); }
  publish(count, true);
}
}  // namespace upstream_bench
