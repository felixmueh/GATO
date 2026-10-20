#pragma once
// The public native PCG header must already be included. This host-only hook
// never copies or replaces GLASS/native PCG recurrence, formation or stopping.
#include <memory>
#include <type_traits>
#include "upstream_timing.cuh"
#ifdef PRECOND_CPU_BENCH
#include "upstream_cpu_bridge.cuh"
#endif
#ifdef PRECOND_GPU_POLYNOMIAL
#include "gpu_polynomial.cuh"
#endif
#ifdef PRECOND_BLOCK_LDLT
#include "block_ldlt_solver.cuh"
#ifdef PRECOND_MANAGED_S
#include "block_ldlt_shared.cuh"
#endif
#endif

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
struct ResourceInfo {size_t dynamic_bytes=0,static_bytes=0;int normal_limit=0,optin_limit=0;};
template<class Kernel>ResourceInfo configure_resource(Kernel kernel){
  ResourceInfo result;int device=0;cudaFuncAttributes attributes{};
  check(cudaGetDevice(&device));check(cudaFuncGetAttributes(&attributes,kernel));
  check(cudaDeviceGetAttribute(&result.normal_limit,cudaDevAttrMaxSharedMemoryPerBlock,device));
  check(cudaDeviceGetAttribute(&result.optin_limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
  result.dynamic_bytes=get_solve_pcg_batched_smem_size<float>();result.static_bytes=attributes.sharedSizeBytes;
  if(result.dynamic_bytes+result.static_bytes>static_cast<size_t>(result.optin_limit))
    throw std::runtime_error("Requested PCG scratch exceeds this device's opt-in shared-memory capacity; no kernel launched");
  if(result.dynamic_bytes+result.static_bytes>static_cast<size_t>(result.normal_limit))
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(result.dynamic_bytes)));
  return result;
}
inline ResourceInfo prepare_resources(){
#ifdef PRECOND_GPU_POLYNOMIAL
  if(method=="gpu_polynomial2")return configure_resource(upstream_precond::solve_gpu_polynomial_kernel<float,2>);
  if(method=="gpu_polynomial3")return configure_resource(upstream_precond::solve_gpu_polynomial_kernel<float,3>);
#endif
#ifdef PRECOND_BLOCK_LDLT
  if(method=="block_ldlt"||method=="block_ldlt_shared")return configure_resource(upstream_precond::block_ldlt_pcg_kernel<float>);
#endif
  // Host-PCG bdmv needs no dynamic carve; this query also documents native
  // baseline feasibility. The campaign always compares against that baseline.
  return configure_resource(solve_pcg_batched_kernel<float>);
}

inline void prepare() { (void)prepare_resources(); (void)buffers();
#ifdef PRECOND_CPU_BENCH
  if(upstream_cpu::available(method))prepare_cpu_bridge();
#endif
#ifdef PRECOND_BLOCK_LDLT
  if(method=="block_ldlt")upstream_precond::prepare_block_ldlt();
#ifdef PRECOND_MANAGED_S
  if(method=="block_ldlt_shared")upstream_precond::prepare_block_ldlt_shared();
#endif
#endif
}

template<typename T>
void solve_measured(uint32_t batch_size, T* lambda, SchurSystem<T> schur, T* epsilon,
                    uint32_t cap, int32_t* flags, uint32_t* iterations) {
  static_assert(std::is_same<T, float>::value, "Public timing benchmark supports float only");
  if(end_to_end_mode&&(method=="normal"||method=="normal_shared")){
    records.emplace_back();records.back().pcg_gpu_event_measured=false;
    solve_pcg_batched<T>(batch_size,lambda,schur,epsilon,cap,flags,iterations);return;
  }
  if (method == "normal_raw") {
    solve_pcg_batched<T>(batch_size, lambda, schur, epsilon, cap, flags, iterations);
    return;
  }
  if (batch_size != 1) throw std::invalid_argument("Public timing benchmark currently supports B1 only");
  auto& h = buffers();
  // Await formation before measuring PCG. This wait remains inside the whole
  // reach wall boundary and is never subtracted as a transfer.
  if(!end_to_end_mode)check(cudaDeviceSynchronize());
  const auto begin = Clock::now(); records.emplace_back(); auto& record = records.back();
  record.pcg_gpu_event_measured=!end_to_end_mode;
  try {
#ifdef PRECOND_CPU_BENCH
    if(upstream_cpu::available(method)){
      if(end_to_end_mode)throw std::invalid_argument("CPU bridge requires instrumented transfer-subtraction mode");
      solve_cpu_public(lambda,schur,epsilon,cap,flags,iterations,record);
      record.setup_wall_us=record.cpu_setup_us;record.cpu_apply_wall_us=record.cpu_apply_us;
      record.pcg_gpu_us=record.matvec_gpu_us;
      record.pcg_compute_wall_us=elapsed(begin)-record.additional_transfer_wall_us;
      return;
    }
#endif
#ifdef PRECOND_BLOCK_LDLT
    if(method=="block_ldlt"||method=="block_ldlt_shared"){
      upstream_precond::BlockLdltStats stats;
      auto preserve=[&]{record.additional_transfer_wall_us=stats.transfer_wall_us;
        additional_transfer_wall_us+=stats.transfer_wall_us;record.setup_wall_us=stats.cpu_setup_us;
        record.additional_copy_calls=stats.transfer_calls;record.additional_copy_bytes=stats.transfer_bytes;
        record.pcg_gpu_us=stats.pcg_gpu_us;
        record.pcg_compute_wall_us=elapsed(begin)-stats.transfer_wall_us;};
      try{
#ifdef PRECOND_MANAGED_S
        if(method=="block_ldlt_shared")upstream_precond::solve_block_ldlt_shared(batch_size,lambda,schur,epsilon,cap,flags,iterations,stats,record.host_factor_write_wall_us,!end_to_end_mode);
        else
#endif
        upstream_precond::solve_block_ldlt(batch_size,lambda,schur,epsilon,cap,flags,iterations,stats,!end_to_end_mode);
      }
      catch(...){preserve();throw;}
      preserve();return;
    }
#endif
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
#ifdef PRECOND_GPU_POLYNOMIAL
    if(method=="gpu_polynomial2")upstream_precond::solve_gpu_polynomial<T,2>(batch_size,lambda,schur,epsilon,cap,flags,iterations);
    else if(method=="gpu_polynomial3")upstream_precond::solve_gpu_polynomial<T,3>(batch_size,lambda,schur,epsilon,cap,flags,iterations);
    else
#endif
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
