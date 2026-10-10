// Q02 test-only accessor. Calls unchanged normal-plant functions; no SQP/KKT.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <chrono>
#include <stdexcept>
#include "settings.h"
#include "dynamics/integrator.cuh"
namespace py = pybind11;
using namespace gato::plant;
constexpr int QDD=0, DJ=7, STEP=154, A=168, B=364, POSE=462, PJ=468;
constexpr int COST=510, G=511, Q=532, R=728, TCOST=777, TG=778, TQ=792, QDDGRAD=988, SIZE=995;
// 32 blocks preserve the production blockIdx.x knot convention. Lanes use y.
template<int Batch>
__global__ void observe(const float* in, const float* refs, float* out, void* model, float dt) {
    if (blockIdx.x != 0 && blockIdx.x != 30 && blockIdx.x != 31) return;
    const int lane = blockIdx.y;
    __shared__ float xu[35], ref[12], work[4096], qdd[7], dj[147];
    __shared__ float step[14], a[196], b[98], pose[6], pj[42], qgrid[7];
    __shared__ float q[196], g[14], r[49], rg[7], tq[196], tg[14];
    for (int i=threadIdx.x; i<35; i+=blockDim.x) xu[i]=in[lane*21+(i<21 ? i : i-21)];
    for (int i=threadIdx.x; i<12; i+=blockDim.x) ref[i]=refs[lane*6+i%6];
    __syncthreads();
    float* dst=out+lane*SIZE;
    if (blockIdx.x == 0) {
        forwardDynamicsAndGradient<float>(dj,qdd,xu,xu+7,xu+14,work,model);
        __syncthreads();
        for(int i=threadIdx.x;i<7;i+=blockDim.x) dst[QDDGRAD+i]=qdd[i];
        __syncthreads();
        forwardDynamics<float>(qdd,xu,xu+7,xu+14,work,model);
        __syncthreads();
        integrator_inner<float,2,false>(step,step+7,xu,xu+7,qdd,dt);
        integrator_gradient_inner<float,2>(a,b,dj,dt);
        __syncthreads();
        for(int i=threadIdx.x;i<7;i+=blockDim.x) qgrid[i]=xu[i]*tiagoJointSign<float>(i);
        __syncthreads();
        computeTiagoToolPoseAndGradient<float>(pose,pj,qgrid,static_cast<grid::robotModel<float>*>(model),work);
        __syncthreads();
        for(int i=threadIdx.x;i<42;i+=blockDim.x) pj[i]*=tiagoJointSign<float>(i/6);
        __syncthreads();
        for(int i=threadIdx.x;i<7;i+=blockDim.x) dst[QDD+i]=qdd[i];
        for(int i=threadIdx.x;i<147;i+=blockDim.x) dst[DJ+i]=dj[i];
        for(int i=threadIdx.x;i<14;i+=blockDim.x) dst[STEP+i]=step[i];
        for(int i=threadIdx.x;i<196;i+=blockDim.x) dst[A+i]=a[i];
        for(int i=threadIdx.x;i<98;i+=blockDim.x) dst[B+i]=b[i];
        for(int i=threadIdx.x;i<6;i+=blockDim.x) dst[POSE+i]=pose[i];
        for(int i=threadIdx.x;i<42;i+=blockDim.x) dst[PJ+i]=pj[i];
        __syncthreads();
        float cost=trackingcost<float>(14,7,32,xu,ref,work,static_cast<grid::robotModel<float>*>(model),30.f,.1f,.001f,300.f,3.f,30.f,.01f,.1f,.01f);
        if(threadIdx.x==0) dst[COST]=cost;
        __syncthreads();
        trackingCostGradientAndHessian<float>(14,7,xu,ref,q,g,r,rg,work,model,30.f,.1f,.001f,300.f,3.f,30.f,.01f,.1f,.01f);
        for(int i=threadIdx.x;i<14;i+=blockDim.x) dst[G+i]=g[i];
        for(int i=threadIdx.x;i<7;i+=blockDim.x) dst[G+14+i]=rg[i];
        for(int i=threadIdx.x;i<196;i+=blockDim.x) dst[Q+i]=q[i];
        for(int i=threadIdx.x;i<49;i+=blockDim.x) dst[R+i]=r[i];
    } else if(blockIdx.x==30) {
        trackingCostGradientAndHessian_lastblock<float>(14,7,xu,ref,q,g,r,rg,tq,tg,work,model,30.f,.1f,.001f,300.f,3.f,30.f,.01f,.1f,.01f);
        for(int i=threadIdx.x;i<14;i+=blockDim.x) dst[TG+i]=tg[i];
        for(int i=threadIdx.x;i<196;i+=blockDim.x) dst[TQ+i]=tq[i];
    } else {
        float cost=trackingcost<float>(14,7,32,xu,ref,work,static_cast<grid::robotModel<float>*>(model),30.f,.1f,.001f,300.f,3.f,30.f,.01f,.1f,.01f);
        if(threadIdx.x==0) dst[TCOST]=cost;
    }
}
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    float* p=nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(&p,bytes)); }
    ~Buffer(){ if(p) cudaFree(p); }
};
class Probe {
    void* model;
public:
    Probe():model(initializeDynamicsConstMem<float>()) {}
    ~Probe(){ freeDynamicsConstMem<float>(model); }
    py::dict evaluate(py::array_t<float,py::array::c_style|py::array::forcecast> input,
                      py::array_t<float,py::array::c_style|py::array::forcecast> refs) {
        if(input.ndim()!=2 || input.shape(1)!=21 || (input.shape(0)!=1 && input.shape(0)!=4))
            throw py::value_error("input must have shape (1,21) or (4,21)");
        int batch=input.shape(0);
        if(refs.ndim()!=2 || refs.shape(0)!=batch || refs.shape(1)!=6)
            throw py::value_error("reference must have shape (B,6)");
        auto start=std::chrono::steady_clock::now();
        Buffer di(input.nbytes()), dr(refs.nbytes()), dout(batch*SIZE*sizeof(float));
        check(cudaMemcpy(di.p,input.data(),input.nbytes(),cudaMemcpyHostToDevice));
        check(cudaMemcpy(dr.p,refs.data(),refs.nbytes(),cudaMemcpyHostToDevice));
        // Sentinel retained if a declared field is not written.
        check(cudaMemset(dout.p,0xff,batch*SIZE*sizeof(float)));
        cudaEvent_t begin,end;
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        check(cudaEventRecord(begin));
        if(batch==1) observe<1><<<dim3(32,1),128>>>(di.p,dr.p,dout.p,model,.03f);
        else observe<4><<<dim3(32,4),128>>>(di.p,dr.p,dout.p,model,.03f);
        check(cudaGetLastError()); check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
        float ms; check(cudaEventElapsedTime(&ms,begin,end));
        check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
        py::array_t<float> output({batch,SIZE});
        check(cudaMemcpy(output.mutable_data(),dout.p,batch*SIZE*sizeof(float),cudaMemcpyDeviceToHost));
        py::dict result; result["flat"]=output; result["kernel_ms"]=ms;
        result["wall_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return result;
    }
};
PYBIND11_MODULE(q02_probe,m) {
    py::class_<Probe>(m,"Probe").def(py::init<>()).def("evaluate",&Probe::evaluate);
    m.attr("scope")="test-only unchanged normal tiago_right functions; N32; float32; B1/B4; no solver";
}
