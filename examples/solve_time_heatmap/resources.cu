#include "bsqp/bsqp.cuh"
#include <iomanip>
#include <sstream>

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
    int driver_version;
    gpuErrchk(cudaDriverGetVersion(&driver_version));
    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) uuid << '-';
        uuid << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(p.uuid.bytes[i]));
    }
#if !defined(GATO_LINSYS_QDLDL)
    const auto pcg_setup = gato::experiment::configure_pcg_shared_memory(
        solvePCGBatchedKernel<float, 1>, getSolvePCGBatchedSMemSize<float>(), false);
#else
    // PCG is not used by this backend; do not configure or reject its SMEM.
    const gato::experiment::PCGSharedMemorySetup pcg_setup{};
#endif
#if defined(GATO_LINSYS_QDLDL)
    constexpr const char* backend = "QDLDL";
#else
    constexpr const char* backend = "PCG";
#endif
    std::cout << "{\"linsys_solver\":\"" << backend << "\",\"plant\":\"" << RESOURCE_PLANT << "\",\"N\":" << KNOT_POINTS
              << ",\"device\":\"" << p.name << "\",\"compute_capability\":\""
              << p.major << "." << p.minor << "\",\"shared_limit_bytes\":"
              << p.sharedMemPerBlock << ",\"global_memory_bytes\":" << p.totalGlobalMem
              << ",\"multiprocessors\":" << p.multiProcessorCount
              << ",\"gpu_uuid\":" << std::quoted(uuid.str())
              << ",\"cuda_driver_version\":" << driver_version
              << ",\"pcg_shared_memory_policy\":{\"applicable\":" << (std::string(backend) == "PCG" ? "true" : "false")
              << ",\"optin_requested\":" << (pcg_setup.optin_requested ? "true" : "false")
              << ",\"optin_applied\":" << (pcg_setup.optin_applied ? "true" : "false")
              << ",\"attributes_queried\":" << (pcg_setup.attributes_queried ? "true" : "false")
              << ",\"supported\":" << (pcg_setup.supported ? "true" : "false")
              << ",\"device_optin_limit_bytes\":" << pcg_setup.device_optin_limit_bytes
              << ",\"effective_limit_bytes\":" << pcg_setup.effective_limit_bytes
              << ",\"configuration_error\":" << std::quoted(pcg_setup.error)
              << "},\"kernels\":[";
    bool first = true;
    auto emit = [&](const char* name, auto kernel, size_t dynamic_bytes, int threads) {
        cudaFuncAttributes a;
        gpuErrchk(cudaFuncGetAttributes(&a, kernel));
        size_t effective_limit = p.sharedMemPerBlock;
        if (a.maxDynamicSharedSizeBytes > 0)
            effective_limit = std::min(effective_limit, a.sharedSizeBytes + a.maxDynamicSharedSizeBytes);
        if (std::string(name) == "pcg" && pcg_setup.optin_requested && pcg_setup.supported)
            effective_limit = std::min(pcg_setup.device_optin_limit_bytes,
                a.sharedSizeBytes + a.maxDynamicSharedSizeBytes);
        const bool setup_ok = std::string(name) != "pcg" || pcg_setup.supported;
        if (!first) std::cout << ",";
        first = false;
        std::cout << "{\"name\":\"" << name << "\",\"dynamic_bytes\":" << dynamic_bytes
                  << ",\"static_bytes\":" << a.sharedSizeBytes
                  << ",\"threads\":" << threads << ",\"max_threads\":" << a.maxThreadsPerBlock
                  << ",\"configured_max_dynamic_bytes\":" << a.maxDynamicSharedSizeBytes
                  << ",\"effective_shared_limit_bytes\":" << effective_limit
                  << ",\"fits\":" << ((setup_ok && dynamic_bytes + a.sharedSizeBytes <= effective_limit
                      && threads <= a.maxThreadsPerBlock) ? "true" : "false") << "}";
    };
    emit("setup_kkt", setupKKTSystemBatchedKernel<float, 1>, getSetupKKTSystemBatchedSMemSize<float>(), KKT_THREADS);
    emit("merit", computeMeritBatchedKernel<float, 1>, getComputeMeritBatchedSMemSize<float>(), grid::SUGGESTED_THREADS);
#if !defined(GATO_LINSYS_QDLDL)
    emit("pcg", solvePCGBatchedKernel<float, 1>, getSolvePCGBatchedSMemSize<float>(), PCG_THREADS);
#endif
    emit("schur1", formSchurSystemBatchedKernel1<float, 1>, getFormSchurSystemBatched1SMemSize<float>(), SCHUR_THREADS);
    emit("schur2", formSchurSystemBatchedKernel2<float, 1>, getFormSchurSystemBatched2SMemSize<float>(), SCHUR_THREADS);
    emit("dz", computeDzBatchedKernel<float, 1>, getComputeDzBatchedSMemSize<float>(), DZ_THREADS);
    emit("line_search", lineSearchAndUpdateBatchedKernel<float, 1, NUM_ALPHAS>,
         sizeof(float) * NUM_ALPHAS + sizeof(uint32_t) * NUM_ALPHAS, LINE_SEARCH_THREADS);
    std::cout << "]";
    // Resource-only: never launch a valid (or invalid) kernel with null operands.
    std::cout << ",\"pcg_launch_attempted\":false,\"kernel_launches\":0}\n";
}
