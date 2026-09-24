// Read-only CUDA resource query, optionally followed by one solver iteration.
// Each invocation runs in its own process so unsupported launches are recorded
// without taking down an entire experimental matrix.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "bsqp/bsqp.cuh"

template<unsigned B>
int probe(bool smoke, bool optin)
{
    int device = 0, driver = 0, runtime = 0;
    gpuErrchk(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    gpuErrchk(cudaGetDeviceProperties(&properties, device));
    gpuErrchk(cudaDriverGetVersion(&driver));
    gpuErrchk(cudaRuntimeGetVersion(&runtime));
    cudaFuncAttributes attributes{};
    const size_t dynamic_bytes = getSolvePCGBatchedSMemSize<float>();
    // This optional diagnostic changes only this process's kernel attribute,
    // not the production extension. Baseline runs leave it disabled.
    const cudaError_t optin_status = optin ? cudaFuncSetAttribute(
        solvePCGBatchedKernel<float, B>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(dynamic_bytes)) : cudaSuccess;
    const cudaError_t attribute_status = cudaFuncGetAttributes(
        &attributes, solvePCGBatchedKernel<float, B>);
    int active_blocks = 0;
    const cudaError_t occupancy_status = attribute_status == cudaSuccess
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, solvePCGBatchedKernel<float, B>, PCG_THREADS, dynamic_bytes)
        : attribute_status;
    printf("{\"kind\":\"resources\",\"knots\":%u,\"batch\":%u,"
           "\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
           "\"driver_api\":%d,\"runtime_api\":%d,\"sm_count\":%d,"
           "\"global_memory_bytes\":%llu,\"shared_default_per_block\":%zu,"
           "\"shared_optin_per_block\":%zu,\"shared_per_sm\":%zu,"
           "\"threads_per_block_limit\":%d,\"registers_per_block\":%d,"
           "\"pcg_threads\":%u,\"pcg_dynamic_shared_bytes\":%zu,"
           "\"pcg_static_shared_bytes\":%zu,\"pcg_registers_per_thread\":%d,"
           "\"pcg_local_bytes\":%zu,\"pcg_function_dynamic_limit\":%d,"
           "\"pcg_function_max_threads\":%d,\"optin_requested\":%s,\"optin_status\":%d,"
           "\"function_attribute_status\":%d,\"occupancy_status\":%d,"
           "\"pcg_active_blocks_per_sm\":%d}\n",
           unsigned(KNOT_POINTS), B, properties.name, properties.major, properties.minor,
           driver, runtime, properties.multiProcessorCount,
           static_cast<unsigned long long>(properties.totalGlobalMem),
           properties.sharedMemPerBlock, properties.sharedMemPerBlockOptin,
           properties.sharedMemPerMultiprocessor, properties.maxThreadsPerBlock,
           properties.regsPerBlock, PCG_THREADS, dynamic_bytes,
           attributes.sharedSizeBytes, attributes.numRegs, attributes.localSizeBytes,
           attributes.maxDynamicSharedSizeBytes, attributes.maxThreadsPerBlock,
           optin ? "true" : "false", int(optin_status), int(attribute_status),
           int(occupancy_status), active_blocks);
    fflush(stdout);
    // An occupancy query may itself return a resource error. Clear it before
    // the optional real launch, whose own outcome is independently recorded.
    cudaGetLastError();
    if (!smoke) return attribute_status == cudaSuccess ? 0 : 2;

    BSQP<float, B> solver(.03f, 1, 0.f, 100, 1e-4f, 1.f, 10.f,
                         2.f, .01f, .0001f, 200.f, 1.f, 1.f, 0.f, 0.f, 0.f, .01f);
    solver.set_rho_adaptation(false);
    const float start[7] = {-.39f, -1.73f, -.38f, -2.35f, 0.f, -1.21f, .04f};
    std::vector<float> xu(B * TRAJ_SIZE, 0.f), x0(B * STATE_SIZE, 0.f);
    std::vector<float> reference(B * REFERENCE_TRAJ_SIZE, 0.f), wrench(B * 6, 0.f);
    for (unsigned b = 0; b < B; ++b) {
        for (unsigned j = 0; j < 7; ++j) x0[b * STATE_SIZE + j] = start[j];
        for (unsigned k = 0; k < KNOT_POINTS; ++k) {
            for (unsigned j = 0; j < 7; ++j)
                xu[b * TRAJ_SIZE + k * STATE_S_CONTROL + j] = start[j];
            float* ref = reference.data() + b * REFERENCE_TRAJ_SIZE + k * grid::REFERENCE_SIZE;
            ref[0] = .45f; ref[1] = -.22f; ref[2] = .30f;
            ref[3] = .40f; ref[4] = -.23f; ref[5] = .03f;
        }
    }
    float *device_xu = nullptr, *device_x0 = nullptr, *device_reference = nullptr;
    gpuErrchk(cudaMalloc(&device_xu, xu.size() * sizeof(float)));
    gpuErrchk(cudaMalloc(&device_x0, x0.size() * sizeof(float)));
    gpuErrchk(cudaMalloc(&device_reference, reference.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(device_xu, xu.data(), xu.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_x0, x0.data(), x0.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_reference, reference.data(), reference.size() * sizeof(float), cudaMemcpyHostToDevice));
    solver.set_f_ext_batch(wrench.data());
    solver.reset_dual(); solver.reset_rho();
    ProblemInputs<float, B> input{};
    input.timestep = .03f;
    input.d_x_s_batch = device_x0;
    input.d_reference_traj_batch = device_reference;
    const auto result = solver.solve(device_xu, input);
    gpuErrchk(cudaDeviceSynchronize());
    printf("{\"kind\":\"launch_smoke\",\"knots\":%u,\"batch\":%u,"
           "\"completed\":true,\"solve_us\":%.9g,"
           "\"scope\":\"one iteration; launch support only, not convergence or benchmark\"}\n",
           unsigned(KNOT_POINTS), B, double(result.solve_time_us));
    gpuErrchk(cudaFree(device_xu)); gpuErrchk(cudaFree(device_x0));
    gpuErrchk(cudaFree(device_reference));
    return 0;
}

int main(int argc, char** argv)
{
    unsigned batch = 1; bool smoke = false, optin = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--smoke") == 0) smoke = true;
        else if (strcmp(argv[i], "--pcg-opt-in") == 0) optin = true;
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = unsigned(atoi(argv[++i]));
        else { fprintf(stderr, "Usage: resource probe [--batch 1|16] [--smoke] [--pcg-opt-in]\n"); return 2; }
    }
    if (batch == 1) return probe<1>(smoke, optin);
    if (batch == 16) return probe<16>(smoke, optin);
    fprintf(stderr, "Resource probe supports batches 1 and 16.\n");
    return 2;
}
