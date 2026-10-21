#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Diagnostic output only: seed at slot 0, then the current trajectory after
// each actual outer iteration (including an unchanged zero-PCG shortcut).
template<typename T>
struct IterationTrace {
        std::vector<T> xu;
        std::vector<int32_t> pcg_iters;  // iteration-major, then batch
        std::vector<uint8_t> line_search_ran;
        const char* stop_reason = "max_sqp_iters";
};

// Bound the optional device trajectory allocation before multiplication. A
// second, equally sized host array is materialized only for the actual output.
// The cap covers N128/B1/1000 float32 iterations (~10 MiB).
template<typename T>
inline size_t iteration_trace_elements(size_t batch, size_t trajectory, uint32_t max_iters)
{
        constexpr size_t cap_bytes = 256ull * 1024 * 1024;
        const size_t cap_elements = cap_bytes / sizeof(T);
        const uint64_t snapshots = static_cast<uint64_t>(max_iters) + 1;
        if (!batch || !trajectory || batch > cap_elements / trajectory ||
            snapshots > cap_elements / (batch * trajectory)) {
                throw std::invalid_argument("iteration capture exceeds the 256 MiB trajectory limit");
        }
        return static_cast<size_t>(snapshots) * batch * trajectory;
}
