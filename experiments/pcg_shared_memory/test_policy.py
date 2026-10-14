"""CPU-only CUDA API stubs exercise launch setup, not solver arithmetic."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def test_shared_memory_policy(tmp_path):
    (tmp_path / 'cuda_runtime.h').write_text(r'''
#pragma once
#include <cstddef>
using cudaError_t=int;
constexpr int cudaSuccess=0;
constexpr int cudaDevAttrMaxSharedMemoryPerBlock=1, cudaDevAttrMaxSharedMemoryPerBlockOptin=2;
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize=3;
struct cudaFuncAttributes { size_t sharedSizeBytes=32; int maxDynamicSharedSizeBytes=49120; };
inline int calls=0, sets=0, optin=101376, configured=49120, failure=0;
inline size_t static_bytes=32;
inline const void* selected=nullptr;
inline const char* cudaGetErrorString(int) { return "stub API failure"; }
template<class K> int cudaFuncGetAttributes(cudaFuncAttributes* a,K k) {
 ++calls; selected=(const void*)k; if(failure) return failure;
 a->sharedSizeBytes=static_bytes; a->maxDynamicSharedSizeBytes=configured; return 0;
}
inline int cudaGetDevice(int* d) { ++calls; *d=0; return 0; }
inline int cudaDeviceGetAttribute(int* value,int kind,int) {
 ++calls; *value=kind==1?49152:optin; return 0;
}
template<class K> int cudaFuncSetAttribute(K k,int,int value) {
 ++calls; ++sets; selected=(const void*)k; configured=value; return 0;
}
''')
    source = tmp_path / 'check.cpp'
    source.write_text(r'''
#include "bsqp/pcg_launch_resources.cuh"
#include <cassert>
int main() {
 int marker=0; auto kernel=&marker;
 auto small=gato::experiment::configure_pcg_shared_memory(kernel,40644);
 assert(calls==0 && !small.optin_requested && small.supported);
 auto large=gato::experiment::configure_pcg_shared_memory(kernel,76484);
 assert(large.supported && large.optin_requested && large.optin_applied);
 assert(sets==1 && configured==76484 && selected==kernel);
 assert(large.effective_limit_bytes==76516);
 auto again=gato::experiment::configure_pcg_shared_memory(kernel,76484);
 assert(again.supported && !again.optin_applied && sets==1);
 optin=49152; configured=49120;
 auto rejected=gato::experiment::configure_pcg_shared_memory(kernel,76484,false);
 assert(!rejected.supported && rejected.optin_requested && !rejected.optin_applied && sets==1);
 bool raised=false;
 try { gato::experiment::configure_pcg_shared_memory(kernel,76484); }
 catch(const std::runtime_error&) { raised=true; }
 assert(raised);
 optin=76500; // Dynamic alone fits, but dynamic+compiled static does not.
 auto static_reject=gato::experiment::configure_pcg_shared_memory(kernel,76484,false);
 assert(!static_reject.supported && sets==1);
 failure=1;
 auto api_error=gato::experiment::configure_pcg_shared_memory(kernel,76484,false);
 assert(!api_error.supported && !api_error.error.empty());
}
''')
    binary = tmp_path / 'check'
    subprocess.run(['g++', '-std=c++17', '-I', str(tmp_path), '-I', str(ROOT / 'gato'),
                    str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
