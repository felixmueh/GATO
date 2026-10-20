// Safe resource probe: never launches a kernel or dereferences device pointers.
#define GATO_PLANT_HEADER "dynamics/plant.cuh"
#include "settings.h"
#include "bsqp/rowgroups.cuh"
#include <iostream>

int main() {
  int device=0, normal=0, optin=0;
  gpuErrchk(cudaGetDevice(&device));
  gpuErrchk(cudaDeviceGetAttribute(&normal,cudaDevAttrMaxSharedMemoryPerBlock,device));
  gpuErrchk(cudaDeviceGetAttribute(&optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
  cudaFuncAttributes before{},after{};
  gpuErrchk(cudaFuncGetAttributes(&before,gato::rows::row_group_telemetry_batched_kernel<float>));
  const auto bytes=gato::rows::get_row_group_telemetry_smem_size<float>();
  const bool fits=bytes+before.sharedSizeBytes<=static_cast<size_t>(optin);
  bool configured=false;
  try {gato::rows::configure_row_group_telemetry_smem<float>();configured=true;}
  catch(const std::runtime_error& e) {std::cout<<e.what()<<'\n';}
  if(configured!=fits)return 1;
  gpuErrchk(cudaFuncGetAttributes(&after,gato::rows::row_group_telemetry_batched_kernel<float>));
  if(configured && bytes>48*1024 && after.maxDynamicSharedSizeBytes<bytes)return 2;
  if(bytes<=48*1024 && before.maxDynamicSharedSizeBytes!=after.maxDynamicSharedSizeBytes)return 3;
  std::cout<<"N"<<KNOT_POINTS<<" telemetry dynamic="<<bytes<<" static="<<before.sharedSizeBytes
           <<" normal="<<normal<<" optin="<<optin<<" configured="<<configured
           <<" launch_performed=false\n";
  return 0;
}
