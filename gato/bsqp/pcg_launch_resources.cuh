#pragma once
// Testing-only launch setup. No solver arithmetic or ordinary-horizon changes.
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace gato { namespace experiment {

struct PCGSharedMemorySetup {
    bool optin_requested = false;
    bool optin_applied = false;
    bool attributes_queried = false;
    bool supported = true;
    size_t dynamic_bytes = 0;
    size_t static_bytes = 0;
    size_t default_limit_bytes = 48 * 1024;
    size_t device_optin_limit_bytes = 0;
    size_t effective_limit_bytes = 48 * 1024;
    std::string error;
};

inline void pcg_resource_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

template<typename Kernel>
inline PCGSharedMemorySetup configure_pcg_shared_memory(
    Kernel kernel, size_t dynamic_bytes, bool require_supported = true) {
    PCGSharedMemorySetup result;
    result.dynamic_bytes = dynamic_bytes;
    // PCG has five static scalar T values (float32 compiles to 32 bytes).
    // Reserve 128 bytes for padding. Current float32 N<=128 fits this fast path.
    // Do not add any CUDA queries/attribute calls to that ordinary path.
    constexpr size_t static_reserve = 128;
    if (dynamic_bytes <= 48 * 1024 - static_reserve) return result;
    try {
        int device = 0, default_limit = 0, optin_limit = 0;
        cudaFuncAttributes attributes{};
        pcg_resource_check(cudaFuncGetAttributes(&attributes, kernel), "PCG cudaFuncGetAttributes");
        result.attributes_queried = true;
        result.static_bytes = attributes.sharedSizeBytes;
        pcg_resource_check(cudaGetDevice(&device), "PCG cudaGetDevice");
        pcg_resource_check(cudaDeviceGetAttribute(&default_limit, cudaDevAttrMaxSharedMemoryPerBlock, device),
                           "PCG default shared-memory query");
        result.default_limit_bytes = default_limit;
        result.effective_limit_bytes = default_limit;
        if (dynamic_bytes + attributes.sharedSizeBytes <= static_cast<size_t>(default_limit)) return result;
        result.optin_requested = true;
        pcg_resource_check(cudaDeviceGetAttribute(&optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
                           "PCG opt-in shared-memory query");
        result.device_optin_limit_bytes = optin_limit;
        if (dynamic_bytes + attributes.sharedSizeBytes > static_cast<size_t>(optin_limit))
            throw std::runtime_error("PCG shared-memory unsupported: requested dynamic=" +
                std::to_string(dynamic_bytes) + " plus static=" + std::to_string(attributes.sharedSizeBytes) +
                " exceeds device opt-in total=" + std::to_string(optin_limit));
        if (attributes.maxDynamicSharedSizeBytes < static_cast<int>(dynamic_bytes)) {
            pcg_resource_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_bytes)), "PCG cudaFuncSetAttribute");
            result.optin_applied = true;
            pcg_resource_check(cudaFuncGetAttributes(&attributes, kernel), "PCG verify shared-memory attribute");
        }
        result.effective_limit_bytes = attributes.sharedSizeBytes + attributes.maxDynamicSharedSizeBytes;
        if (dynamic_bytes > static_cast<size_t>(attributes.maxDynamicSharedSizeBytes))
            throw std::runtime_error("PCG shared-memory opt-in attribute remains below requested dynamic bytes");
    } catch (const std::exception& error) {
        result.supported = false;
        result.error = error.what();
        if (require_supported) throw;
    }
    return result;
}

}} // namespace gato::experiment
