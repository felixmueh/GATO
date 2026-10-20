#pragma once
#include <atomic>
#include <algorithm>
#include "block_ldlt_solver.cuh"

namespace upstream_precond {
struct SharedBlockLdltWorkspace {
  float *lower=nullptr,*inverse=nullptr;
  cudaEvent_t start=nullptr,stop=nullptr;
  BlockLdltFactors factors;
  SharedBlockLdltWorkspace(){
    upstream_bench::require_managed_memory();
    constexpr size_t bytes=KNOT_POINTS*gato::constants::STATE_SIZE_SQ*sizeof(float);
    try{
      ldlt_check(cudaMallocManaged(&lower,bytes,cudaMemAttachGlobal));
      ldlt_check(cudaMallocManaged(&inverse,bytes,cudaMemAttachGlobal));
      ldlt_check(cudaEventCreate(&start));ldlt_check(cudaEventCreate(&stop));
    }catch(...){release();throw;}
  }
  void release(){if(lower)cudaFree(lower);if(inverse)cudaFree(inverse);if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);}
  ~SharedBlockLdltWorkspace(){release();}
  SharedBlockLdltWorkspace(const SharedBlockLdltWorkspace&)=delete;
};
inline SharedBlockLdltWorkspace& shared_block_ldlt_workspace(){static thread_local SharedBlockLdltWorkspace work;return work;}
inline void prepare_block_ldlt_shared(){(void)shared_block_ldlt_workspace();}
inline bool last_shared_s_pointer_verified=false;

inline void solve_block_ldlt_shared(uint32_t batch_size,float* lambda,SchurSystem<float> schur,
                                   float* epsilon,uint32_t cap,int32_t* skip,uint32_t* iterations,
                                   BlockLdltStats& stats,double& host_write_wall_us,bool measure_gpu=true){
  using namespace gato::constants;
  if(batch_size!=1)throw std::invalid_argument("Managed block LDLT research candidate is B1 only");
  if(!upstream_bench::managed_s_compiled())throw std::runtime_error("Managed S build option is disabled");
  auto& work=shared_block_ldlt_workspace();
  // Tegra may report concurrentManagedAccess=0. Do not touch ANY managed
  // allocation from the CPU while GPU work is active, even a different buffer.
  ldlt_check(cudaDeviceSynchronize());
  cudaPointerAttributes attributes{};
  ldlt_check(cudaPointerGetAttributes(&attributes,schur.d_S_batch));
  if(attributes.type!=cudaMemoryTypeManaged)
    throw std::runtime_error("block_ldlt_shared requires actual managed S; no copy fallback");
  last_shared_s_pointer_verified=true;
  int32_t host_skip=0;
  auto begin=LdltClock::now();
  ++stats.transfer_calls;stats.transfer_bytes+=sizeof(int32_t);
  auto error=cudaMemcpy(&host_skip,skip,sizeof(int32_t),cudaMemcpyDeviceToHost);
  stats.transfer_wall_us+=ldlt_elapsed(begin);ldlt_check(error);
  if(!host_skip){
    // Direct CPU reads of the ORIGINAL const managed S. No CUDA S download,
    // sign conversion in shared storage, or mutation of S is introduced.
    begin=LdltClock::now();
    try{work.factors.setup(schur.d_S_batch,B3D_MATRIX_SIZE_PADDED,KNOT_POINTS,STATE_SIZE);}
    catch(...){stats.cpu_setup_us+=ldlt_elapsed(begin);throw;}
    stats.cpu_setup_us+=ldlt_elapsed(begin);
    begin=LdltClock::now();
    std::copy(work.factors.lower().begin(),work.factors.lower().end(),work.lower);
    std::copy(work.factors.diagonal_inverse().begin(),work.factors.diagonal_inverse().end(),work.inverse);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // Explicit CPU->GPU ownership handoff. Runtime/cache coherence overhead
    // stays in whole-task wall; no prefetch/migration costs are subtracted.
    ldlt_check(cudaDeviceSynchronize());
    host_write_wall_us+=ldlt_elapsed(begin);
  }
  const size_t bytes=get_solve_pcg_batched_smem_size<float>();
  static size_t configured_bytes=0;
  opt_in_dynamic_smem(block_ldlt_pcg_kernel<float>,bytes,configured_bytes);
  if(measure_gpu)ldlt_check(cudaEventRecord(work.start));
  block_ldlt_pcg_kernel<float><<<1,PCG_THREADS,bytes>>>(iterations,lambda,schur.d_S_batch,
      schur.d_P_inv_batch,schur.d_gamma_batch,epsilon,cap,skip,work.lower,work.inverse);
  ldlt_check(cudaGetLastError());
  if(measure_gpu){ldlt_check(cudaEventRecord(work.stop));ldlt_check(cudaEventSynchronize(work.stop));
    float ms=0;ldlt_check(cudaEventElapsedTime(&ms,work.start,work.stop));stats.pcg_gpu_us=1000.0*ms;}
}
} // namespace upstream_precond
