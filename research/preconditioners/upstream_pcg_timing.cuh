#pragma once
// The public native PCG header must already be included. This host-only hook
// never copies or replaces GLASS/native PCG recurrence, formation or stopping.
#include <memory>
#include <type_traits>
#include "upstream_timing.cuh"

namespace upstream_bench {
struct Buffers {
  float *S = nullptr, *P = nullptr, *gamma = nullptr, *lambda = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  Buffers() {
    try {
      check(cudaMallocHost(&S, gato::constants::B3D_MATRIX_SIZE_PADDED * sizeof(float)));
      check(cudaMallocHost(&P, gato::constants::B3D_MATRIX_SIZE_PADDED * sizeof(float)));
      check(cudaMallocHost(&gamma, gato::constants::VEC_SIZE_PADDED * sizeof(float)));
      check(cudaMallocHost(&lambda, gato::constants::VEC_SIZE_PADDED * sizeof(float)));
      check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    } catch (...) { release(); throw; }
  }
  void release() {
    if (S) cudaFreeHost(S); if (P) cudaFreeHost(P);
    if (gamma) cudaFreeHost(gamma); if (lambda) cudaFreeHost(lambda);
    if (start) cudaEventDestroy(start); if (stop) cudaEventDestroy(stop);
  }
  ~Buffers() { release(); }
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;
};
inline Buffers& buffers() { static thread_local Buffers result; return result; }
inline void prepare() { (void)buffers(); }

template<typename T>
void solve_measured(uint32_t batch_size, T* lambda, SchurSystem<T> schur, T* epsilon,
                    uint32_t cap, int32_t* flags, uint32_t* iterations) {
  static_assert(std::is_same<T, float>::value, "Public timing benchmark supports float only");
  if (method == "normal_raw") {
    solve_pcg_batched<T>(batch_size, lambda, schur, epsilon, cap, flags, iterations);
    return;
  }
  if (batch_size != 1) throw std::invalid_argument("Public timing benchmark currently supports B1 only");
  auto& h = buffers();
  // Await formation before measuring PCG. This wait remains inside the whole
  // reach wall boundary and is never subtracted as a transfer.
  check(cudaDeviceSynchronize());
  const auto begin = Clock::now(); records.emplace_back(); auto& record = records.back();
  try {
    if (method == "dummy") {
      const auto transfer_begin = Clock::now();
      constexpr size_t matrix_bytes = gato::constants::B3D_MATRIX_SIZE_PADDED * sizeof(float);
      constexpr size_t vector_bytes = gato::constants::VEC_SIZE_PADDED * sizeof(float);
      check(cudaMemcpyAsync(h.S, schur.d_S_batch, matrix_bytes, cudaMemcpyDeviceToHost));
      check(cudaMemcpyAsync(h.P, schur.d_P_inv_batch, matrix_bytes, cudaMemcpyDeviceToHost));
      check(cudaMemcpyAsync(h.gamma, schur.d_gamma_batch, vector_bytes, cudaMemcpyDeviceToHost));
      check(cudaMemcpyAsync(h.lambda, lambda, vector_bytes, cudaMemcpyDeviceToHost));
      check(cudaDeviceSynchronize());
      check(cudaMemcpyAsync(schur.d_S_batch, h.S, matrix_bytes, cudaMemcpyHostToDevice));
      check(cudaMemcpyAsync(schur.d_P_inv_batch, h.P, matrix_bytes, cudaMemcpyHostToDevice));
      check(cudaMemcpyAsync(schur.d_gamma_batch, h.gamma, vector_bytes, cudaMemcpyHostToDevice));
      check(cudaMemcpyAsync(lambda, h.lambda, vector_bytes, cudaMemcpyHostToDevice));
      check(cudaDeviceSynchronize());
      record.additional_transfer_wall_us = elapsed(transfer_begin);
      additional_transfer_wall_us += record.additional_transfer_wall_us;
    }
    check(cudaEventRecord(h.start));
    solve_pcg_batched<T>(batch_size, lambda, schur, epsilon, cap, flags, iterations);
    check(cudaGetLastError()); check(cudaEventRecord(h.stop)); check(cudaEventSynchronize(h.stop));
    float milliseconds = 0.0f; check(cudaEventElapsedTime(&milliseconds, h.start, h.stop));
    record.pcg_gpu_us = 1000.0 * milliseconds;
  } catch (...) {
    record.pcg_compute_wall_us = elapsed(begin) - record.additional_transfer_wall_us;
    throw;
  }
  record.pcg_compute_wall_us = elapsed(begin) - record.additional_transfer_wall_us;
}
}  // namespace upstream_bench
