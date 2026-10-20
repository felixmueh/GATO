#pragma once
// Experimental host-copy instrumentation. Include before any native plant,
// GLASS or BSQP headers so their host copy callsites can share these wrappers.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace upstream_bench {
using Clock = std::chrono::steady_clock;
inline std::string method = "normal_raw";
inline double native_copy_wall_us = 0.0, additional_transfer_wall_us = 0.0;
inline uint64_t native_copy_calls = 0, native_copy_bytes = 0;
inline bool uncovered_copy = false;
struct Record {
  double additional_transfer_wall_us = 0.0, pcg_compute_wall_us = 0.0, pcg_gpu_us = 0.0;
};
inline std::vector<Record> records;
inline double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
inline void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
inline bool measured_kind(cudaMemcpyKind kind) {
  if (kind == cudaMemcpyDefault && method != "normal_raw") uncovered_copy = true;
  return method != "normal_raw" && (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost);
}

inline cudaError_t copy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
  if (!measured_kind(kind)) return cudaMemcpy(dst, src, bytes, kind);
  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess) return status;
  const auto start = Clock::now();
  status = cudaMemcpy(dst, src, bytes, kind);
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  native_copy_wall_us += elapsed(start); ++native_copy_calls; native_copy_bytes += bytes;
  return status;
}
inline cudaError_t copy_async(void* dst, const void* src, size_t bytes,
                              cudaMemcpyKind kind, cudaStream_t stream = nullptr) {
  if (!measured_kind(kind)) return cudaMemcpyAsync(dst, src, bytes, kind, stream);
  auto status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) return status;
  const auto start = Clock::now();
  status = cudaMemcpyAsync(dst, src, bytes, kind, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
  native_copy_wall_us += elapsed(start); ++native_copy_calls; native_copy_bytes += bytes;
  return status;
}
inline cudaError_t copy_2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                           size_t width, size_t height, cudaMemcpyKind kind) {
  if (!measured_kind(kind)) return cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height, kind);
  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess) return status;
  const auto start = Clock::now();
  status = cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height, kind);
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  native_copy_wall_us += elapsed(start); ++native_copy_calls; native_copy_bytes += width * height;
  return status;
}
inline void set_method(const std::string& requested) {
  if (requested != "normal_raw" && requested != "normal" && requested != "dummy")
    throw std::invalid_argument("Unavailable public preconditioner benchmark method: " + requested);
  method = requested;
}
inline void reset_metrics() {
  check(cudaDeviceSynchronize());
  native_copy_wall_us = additional_transfer_wall_us = 0.0;
  native_copy_calls = native_copy_bytes = 0; uncovered_copy = false; records.clear();
}
}  // namespace upstream_bench
