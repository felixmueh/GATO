#include "../block_ldlt_factors.hpp"
#include "../block_ldlt_apply.cuh"
#include <iostream>

namespace {
constexpr unsigned N=128,D=12,V=(N+2)*D;
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
__global__ void apply_test(float* output,const float* rhs,const float* lower,const float* inverse){
  extern __shared__ float shared[];
  float* result=shared;float* scratch=shared+V;
  upstream_precond::BlockLdltApply<float,D,N>{lower,inverse}(result,rhs,scratch);
  for(unsigned i=threadIdx.x;i<V;i+=blockDim.x)output[i]=result[i];
}
struct Buffer{float* ptr=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&ptr,n*sizeof(float)));}~Buffer(){cudaFree(ptr);}};
}
int main(){
  try{
    // Independently SPD: each row diagonal3 strictly exceeds absolute
    // off-diagonal row sum. Dense diagonal blocks and adjacent coupling.
    std::vector<float> S(3*N*D*D,0),expected(V,0),rhs(V,0),actual(V);
    for(unsigned i=0;i<N*D;++i)expected[D+i]=.2f+float(i%17)*.01f;
    for(unsigned k=0;k<N;++k)for(unsigned row=0;row<D;++row)
      for(int offset=-1;offset<=1;++offset){
        const int neighbor=int(k)+offset;if(neighbor<0||neighbor>=int(N))continue;
        for(unsigned col=0;col<D;++col){
          const float value=offset==0?(row==col?3.0f:.01f):-.025f;
          S[(k*D+row)*3*D+(offset+1)*D+col]=-value;
          rhs[(k+1)*D+row]+=value*expected[(neighbor+1)*D+col];
        }
      }
    upstream_precond::BlockLdltFactors factors;factors.setup(S.data(),S.size(),N,D);
    Buffer lower(N*D*D),inverse(N*D*D),device_rhs(V),output(V);
    check(cudaMemcpy(lower.ptr,factors.lower().data(),N*D*D*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemcpy(inverse.ptr,factors.diagonal_inverse().data(),N*D*D*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_rhs.ptr,rhs.data(),V*sizeof(float),cudaMemcpyHostToDevice));
    apply_test<<<1,1024,2*V*sizeof(float)>>>(output.ptr,device_rhs.ptr,lower.ptr,inverse.ptr);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(actual.data(),output.ptr,V*sizeof(float),cudaMemcpyDeviceToHost));
    float maximum=0;
    for(unsigned i=0;i<V;++i){if(!std::isfinite(actual[i]))throw std::runtime_error("Nonfinite GPU LDLT apply");maximum=std::max(maximum,std::abs(actual[i]+expected[i]));}
    if(maximum>3e-5f)throw std::runtime_error("GPU -H^-1 callback disagrees with independently constructed known solution");
    std::cout<<"N128/D12 GPU callback known-solution check passed; max abs error "<<maximum<<'\n';
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
