#include "bsqp/bsqp.cuh"

#if defined(PLANT_TIAGO_RIGHT)
constexpr const char* RESOURCE_PLANT = "tiago_right";
#elif defined(PLANT_INDY7)
constexpr const char* RESOURCE_PLANT = "indy7";
#else
#error "Unsupported matrix resource plant"
#endif

int main() {
    int device;
    gpuErrchk(cudaGetDevice(&device));
    cudaDeviceProp p;
    gpuErrchk(cudaGetDeviceProperties(&p, device));
    std::cout << "{\"plant\":\"" << RESOURCE_PLANT << "\",\"N\":" << KNOT_POINTS
              << ",\"device\":\"" << p.name << "\",\"compute_capability\":\""
              << p.major << "." << p.minor << "\",\"shared_limit_bytes\":"
              << p.sharedMemPerBlock << ",\"global_memory_bytes\":" << p.totalGlobalMem
              << ",\"multiprocessors\":" << p.multiProcessorCount << ",\"kernels\":[";
    bool first = true;
    auto emit = [&](const char* name, auto kernel, size_t dynamic_bytes, int threads) {
        cudaFuncAttributes a;
        gpuErrchk(cudaFuncGetAttributes(&a, kernel));
        if (!first) std::cout << ",";
        first = false;
        std::cout << "{\"name\":\"" << name << "\",\"dynamic_bytes\":" << dynamic_bytes
                  << ",\"static_bytes\":" << a.sharedSizeBytes
                  << ",\"threads\":" << threads << ",\"max_threads\":" << a.maxThreadsPerBlock
                  << ",\"fits\":" << ((dynamic_bytes + a.sharedSizeBytes <= p.sharedMemPerBlock
                      && threads <= a.maxThreadsPerBlock) ? "true" : "false") << "}";
    };
    emit("setup_kkt", setupKKTSystemBatchedKernel<float, 1>, getSetupKKTSystemBatchedSMemSize<float>(), KKT_THREADS);
    emit("merit", computeMeritBatchedKernel<float, 1>, getComputeMeritBatchedSMemSize<float>(), grid::SUGGESTED_THREADS);
    emit("pcg", solvePCGBatchedKernel<float, 1>, getSolvePCGBatchedSMemSize<float>(), PCG_THREADS);
    emit("schur1", formSchurSystemBatchedKernel1<float, 1>, getFormSchurSystemBatched1SMemSize<float>(), SCHUR_THREADS);
    emit("schur2", formSchurSystemBatchedKernel2<float, 1>, getFormSchurSystemBatched2SMemSize<float>(), SCHUR_THREADS);
    emit("dz", computeDzBatchedKernel<float, 1>, getComputeDzBatchedSMemSize<float>(), DZ_THREADS);
    emit("line_search", lineSearchAndUpdateBatchedKernel<float, 1, NUM_ALPHAS>,
         sizeof(float) * NUM_ALPHAS + sizeof(uint32_t) * NUM_ALPHAS, LINE_SEARCH_THREADS);
    std::cout << "]";
    cudaFuncAttributes pcg;
    gpuErrchk(cudaFuncGetAttributes(&pcg, solvePCGBatchedKernel<float, 1>));
    if (getSolvePCGBatchedSMemSize<float>() + pcg.sharedSizeBytes > p.sharedMemPerBlock) {
        // The invalid configuration is rejected before pointers are dereferenced.
        solvePCGBatchedKernel<float, 1><<<1, PCG_THREADS, getSolvePCGBatchedSMemSize<float>()>>>(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, nullptr);
        cudaError_t error = cudaGetLastError();
        std::cout << ",\"pcg_launch_error\":\"" << cudaGetErrorString(error) << "\"";
        if (error != cudaErrorInvalidConfiguration && error != cudaErrorInvalidValue) return 2;
    }
    std::cout << "}\n";
}
