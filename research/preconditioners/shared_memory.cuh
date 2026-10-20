#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace upstream_bench {
inline bool shared_method(){return method=="normal_shared"||method=="block_ldlt_shared";}
inline bool managed_s_compiled(){
#ifdef PRECOND_MANAGED_S
  return true;
#else
  return false;
#endif
}
inline void require_managed_memory(){
  int device=0,supported=0;
  check(cudaGetDevice(&device));
  check(cudaDeviceGetAttribute(&supported,cudaDevAttrManagedMemory,device));
  if(!supported)throw std::runtime_error("Requested explicit managed memory is unsupported; no fallback");
}
inline bool last_s_allocation_managed=false;
inline uint64_t s_allocation_count=0;
template<typename T>
cudaError_t allocate_schur_s(T** pointer,size_t bytes){
  if(shared_method()){
    if(!managed_s_compiled())throw std::runtime_error("Shared S requires PRECOND_MANAGED_S=ON");
    require_managed_memory();
    const auto status=cudaMallocManaged(reinterpret_cast<void**>(pointer),bytes,cudaMemAttachGlobal);
    if(status==cudaSuccess){last_s_allocation_managed=true;++s_allocation_count;}
    return status;
  }
  const auto status=cudaMalloc(reinterpret_cast<void**>(pointer),bytes);
  if(status==cudaSuccess){last_s_allocation_managed=false;++s_allocation_count;}
  return status;
}
} // namespace upstream_bench
