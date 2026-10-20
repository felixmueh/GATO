#pragma once
#include <chrono>
#include <memory>
#include "block_ldlt_factors.hpp"
#include "block_ldlt_apply.cuh"
#include "pcg_with_apply.cuh"

namespace upstream_precond {
struct BlockLdltStats {
  double transfer_wall_us=0,cpu_setup_us=0,pcg_gpu_us=0;
  uint64_t transfer_calls=0,transfer_bytes=0;
};
inline void ldlt_check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
using LdltClock=std::chrono::steady_clock;
inline double ldlt_elapsed(LdltClock::time_point start){return std::chrono::duration<double,std::micro>(LdltClock::now()-start).count();}
struct BlockLdltWorkspace {
  float* host_S=nullptr;
  float *device_lower=nullptr,*device_inverse=nullptr;
  cudaEvent_t start=nullptr,stop=nullptr;
  BlockLdltFactors factors;
  BlockLdltWorkspace(){
    using namespace gato::constants;
    try{
      ldlt_check(cudaMallocHost(&host_S,B3D_MATRIX_SIZE_PADDED*sizeof(float)));
      ldlt_check(cudaMalloc(&device_lower,KNOT_POINTS*STATE_SIZE_SQ*sizeof(float)));
      ldlt_check(cudaMalloc(&device_inverse,KNOT_POINTS*STATE_SIZE_SQ*sizeof(float)));
      ldlt_check(cudaEventCreate(&start));ldlt_check(cudaEventCreate(&stop));
    }catch(...){release();throw;}
  }
  void release(){if(host_S)cudaFreeHost(host_S);if(device_lower)cudaFree(device_lower);if(device_inverse)cudaFree(device_inverse);if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);}
  ~BlockLdltWorkspace(){release();}
  BlockLdltWorkspace(const BlockLdltWorkspace&)=delete;
};
inline BlockLdltWorkspace& block_ldlt_workspace(){static thread_local BlockLdltWorkspace work;return work;}
inline void prepare_block_ldlt(){(void)block_ldlt_workspace();}

template<typename T>
__global__ __launch_bounds__(PCG_THREADS) void block_ldlt_pcg_kernel(
    uint32_t* iterations,T* lambda,const T* S,const T* P,const T* gamma,
    const T* epsilon,uint32_t cap,int32_t* skip,const T* lower,const T* inverse){
  using namespace gato::constants;
  if(skip[0]){if(threadIdx.x==0)iterations[0]=0;return;}
  extern __shared__ T shared[];
  pcg_with_apply<T,STATE_SIZE,KNOT_POINTS>(lambda,const_cast<T*>(S),const_cast<T*>(P),
      const_cast<T*>(gamma),shared,cap,epsilon[0],static_cast<T>(1e-6),iterations,
      BlockLdltApply<T,STATE_SIZE,KNOT_POINTS>{lower,inverse});
}

// Called by the common public timing hook. Setup/validation are inside the
// whole-reach boundary. All transfer walls are measured separately for that
// boundary's subtraction; CPU factorization is never charged as a transfer.
inline void solve_block_ldlt(uint32_t batch_size,float* lambda,SchurSystem<float> schur,
                            float* epsilon,uint32_t cap,int32_t* skip,
                            uint32_t* iterations,BlockLdltStats& stats,bool measure_gpu=true){
  using namespace gato::constants;
  if(batch_size!=1)throw std::invalid_argument("Block LDLT research candidate is B1 only");
  auto& work=block_ldlt_workspace();
  ldlt_check(cudaDeviceSynchronize());
  auto begin=LdltClock::now();
  auto transfer= [&]{stats.transfer_wall_us+=ldlt_elapsed(begin);};
  // Preserve the original assembled S and every public native PCG input.
  int32_t host_skip=0;
  ++stats.transfer_calls;stats.transfer_bytes+=sizeof(int32_t);
  auto error=cudaMemcpy(&host_skip,skip,sizeof(int32_t),cudaMemcpyDeviceToHost);
  if(error==cudaSuccess&&!host_skip){++stats.transfer_calls;stats.transfer_bytes+=B3D_MATRIX_SIZE_PADDED*sizeof(float);error=cudaMemcpy(work.host_S,schur.d_S_batch,B3D_MATRIX_SIZE_PADDED*sizeof(float),cudaMemcpyDeviceToHost);}
  if(error==cudaSuccess)error=cudaDeviceSynchronize();transfer();ldlt_check(error);
  if(!host_skip){
    begin=LdltClock::now();
    try{work.factors.setup(work.host_S,B3D_MATRIX_SIZE_PADDED,KNOT_POINTS,STATE_SIZE);}
    catch(...){stats.cpu_setup_us+=ldlt_elapsed(begin);throw;}
    stats.cpu_setup_us+=ldlt_elapsed(begin);
    begin=LdltClock::now();
    ++stats.transfer_calls;stats.transfer_bytes+=KNOT_POINTS*STATE_SIZE_SQ*sizeof(float);
    error=cudaMemcpy(work.device_lower,work.factors.lower().data(),KNOT_POINTS*STATE_SIZE_SQ*sizeof(float),cudaMemcpyHostToDevice);
    if(error==cudaSuccess){++stats.transfer_calls;stats.transfer_bytes+=KNOT_POINTS*STATE_SIZE_SQ*sizeof(float);error=cudaMemcpy(work.device_inverse,work.factors.diagonal_inverse().data(),KNOT_POINTS*STATE_SIZE_SQ*sizeof(float),cudaMemcpyHostToDevice);}
    if(error==cudaSuccess)error=cudaDeviceSynchronize();transfer();ldlt_check(error);
  }
  const size_t bytes=get_solve_pcg_batched_smem_size<float>();
  static size_t configured_bytes=0;
  opt_in_dynamic_smem(block_ldlt_pcg_kernel<float>,bytes,configured_bytes);
  if(measure_gpu)ldlt_check(cudaEventRecord(work.start));
  block_ldlt_pcg_kernel<float><<<1,PCG_THREADS,bytes>>>(iterations,lambda,schur.d_S_batch,
      schur.d_P_inv_batch,schur.d_gamma_batch,epsilon,cap,skip,work.device_lower,work.device_inverse);
  ldlt_check(cudaGetLastError());
  if(measure_gpu){ldlt_check(cudaEventRecord(work.stop));ldlt_check(cudaEventSynchronize(work.stop));
    float milliseconds=0;ldlt_check(cudaEventElapsedTime(&milliseconds,work.start,work.stop));stats.pcg_gpu_us=1000.0*milliseconds;}
}
} // namespace upstream_precond
