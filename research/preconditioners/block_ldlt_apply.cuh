#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace upstream_precond {
// H=-S=L D L^T. Native public PCG uses negative S and negative P_inv, so
// replace its preconditioner by -H^-1, retaining the native rho/alpha signs.
// Uploaded factors are row-major [N,D,D], independent of native plant sizes.
template<typename T,uint32_t D,uint32_t N>
struct BlockLdltApply {
  const T* lower;
  const T* diagonal_inverse;
  __device__ void operator()(T* z,const T* r,T* dead_Ap)const {
    constexpr uint32_t padded=(N+2)*D;
    for(uint32_t i=threadIdx.x;i<padded;i+=blockDim.x){z[i]=T(0);dead_Ap[i]=T(0);}
    __syncthreads();
    // Forward unit-lower solve; blocks depend serially on their predecessor,
    // while independent rows within one block are evaluated in parallel.
    for(uint32_t k=0;k<N;++k){
      for(uint32_t row=threadIdx.x;row<D;row+=blockDim.x){
        T value=r[(k+1)*D+row];
        if(k)for(uint32_t col=0;col<D;++col)
          value-=lower[(k*D+row)*D+col]*dead_Ap[k*D+col];
        dead_Ap[(k+1)*D+row]=value;
      }
      __syncthreads();
    }
    for(uint32_t index=threadIdx.x;index<N*D;index+=blockDim.x){
      const uint32_t k=index/D,row=index%D;
      T value=T(0);
      for(uint32_t col=0;col<D;++col)
        value+=diagonal_inverse[(k*D+row)*D+col]*dead_Ap[(k+1)*D+col];
      z[(k+1)*D+row]=value;
    }
    __syncthreads();
    for(int k=static_cast<int>(N)-2;k>=0;--k){
      for(uint32_t row=threadIdx.x;row<D;row+=blockDim.x){
        T value=z[(k+1)*D+row];
        for(uint32_t col=0;col<D;++col)
          value-=lower[((k+1)*D+col)*D+row]*z[(k+2)*D+col];
        z[(k+1)*D+row]=value;
      }
      __syncthreads();
    }
    for(uint32_t index=threadIdx.x;index<N*D;index+=blockDim.x)z[D+index]=-z[D+index];
    __syncthreads();
  }
};
} // namespace upstream_precond
