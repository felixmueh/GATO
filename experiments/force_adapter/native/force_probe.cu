// Bounded actual production TIAGo acceleration and analytic-gradient capture.
#include <cuda_runtime.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "types.cuh"

void check(cudaError_t value) {
    if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value));
}

template<class T>
__global__ void probe(const T* input, T* output, void* dynamics) {
    __shared__ T state[21];
    __shared__ T accel[7];
    __shared__ T gradient[147];
    extern __shared__ unsigned char temp_bytes[];
    T* temp = reinterpret_cast<T*>(temp_bytes);
    for (int i=threadIdx.x; i<21; i+=blockDim.x) state[i]=input[i];
    __syncthreads();
    gato::plant::forwardDynamics<T>(accel,state,state+7,state+14,temp,dynamics,const_cast<T*>(input+21));
    __syncthreads();
    for (int i=threadIdx.x; i<7; i+=blockDim.x) output[i]=accel[i];
    __syncthreads();
    gato::plant::forwardDynamicsAndGradient<T>(gradient,accel,state,state+7,state+14,temp,dynamics,const_cast<T*>(input+21));
    __syncthreads();
    for (int i=threadIdx.x; i<7; i+=blockDim.x) output[7+i]=accel[i];
    for (int i=threadIdx.x; i<147; i+=blockDim.x) output[14+i]=gradient[i];
}

template<class T>
void capture(const std::vector<double>& inputs, const char* output_path) {
    T* input=nullptr;T* output=nullptr;
    check(cudaMalloc(&input,27*sizeof(T)));
    check(cudaMalloc(&output,161*sizeof(T)));
    void* dynamics=gato::plant::initializeDynamicsConstMem<T>();
    std::ofstream file(output_path,std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open output");
    for (size_t offset=0; offset<inputs.size(); offset+=27) {
        std::vector<T> row(27),values(161);
        for (int i=0;i<27;++i) row[i]=static_cast<T>(inputs[offset+i]);
        check(cudaMemcpy(input,row.data(),27*sizeof(T),cudaMemcpyHostToDevice));
        check(cudaMemset(output,0xff,161*sizeof(T)));
        const size_t shared=sizeof(T)*gato::plant::forwardDynamicsAndGradient_TempMemSize_Shared();
        probe<T><<<1,128,shared>>>(input,output,dynamics);
        check(cudaGetLastError());check(cudaDeviceSynchronize());
        check(cudaMemcpy(values.data(),output,161*sizeof(T),cudaMemcpyDeviceToHost));
        file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));
    }
    gato::plant::freeDynamicsConstMem<T>(dynamics);
    check(cudaFree(input));check(cudaFree(output));
}

int main(int argc,char** argv) {
    try {
        if (argc!=4) throw std::runtime_error("input.f64 float-output.f32 double-output.f64 required");
        std::ifstream file(argv[1],std::ios::binary);
        if (!file) throw std::runtime_error("Cannot read input");
        file.seekg(0,std::ios::end);size_t bytes=file.tellg();file.seekg(0);
        if (!bytes || bytes%(27*sizeof(double))) throw std::runtime_error("Invalid input layout");
        std::vector<double> inputs(bytes/sizeof(double));
        file.read(reinterpret_cast<char*>(inputs.data()),bytes);
        if (!file) throw std::runtime_error("Cannot read input bytes");
        capture<float>(inputs,argv[2]);
        capture<double>(inputs,argv[3]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<std::endl;return 1;
    }
}
