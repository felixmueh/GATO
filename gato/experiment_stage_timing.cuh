#pragma once

// Host-owned opt-in events only. The caller supplies its existing stream and
// reads completed windows after its existing final synchronization/chrono end.
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct ExperimentStageWindow {
        const char* name;
        int iteration;  // -1: initial/final, otherwise zero-based SQP iteration
        bool measured = false;
        double elapsed_us = 0;
        const char* unavailable_reason = "not_reached";
};

class ExperimentStageTiming {
        struct Pair {
                cudaEvent_t start = nullptr, stop = nullptr;
                bool begun = false, ended = false;
        };
        std::vector<Pair> pairs_;
        uint32_t cap_ = 0;
        static void check(cudaError_t error) {
                if (error != cudaSuccess) throw std::runtime_error(
                    std::string("experiment stage event: ") + cudaGetErrorString(error));
        }
        void release() noexcept {
                for (auto& p : pairs_) {
                        if (p.start) cudaEventDestroy(p.start);
                        if (p.stop) cudaEventDestroy(p.stop);
                }
                pairs_.clear();
        }
      public:
        ExperimentStageTiming() = default;
        ExperimentStageTiming(const ExperimentStageTiming&) = delete;
        ExperimentStageTiming& operator=(const ExperimentStageTiming&) = delete;
        ~ExperimentStageTiming() { release(); }
        void configure(uint32_t cap) {
                if (cap > 10) throw std::invalid_argument("stage profile supports SQP caps 0..10");
                if (pairs_.size() == 7 * cap + 2) return;
                release(); cap_ = cap; pairs_.resize(7 * cap + 2);
                try {
                        for (auto& p : pairs_) {
                                check(cudaEventCreate(&p.start));
                                check(cudaEventCreate(&p.stop));
                        }
                } catch (...) { release(); throw; }
        }
        void reset() {
                for (auto& p : pairs_) { p.begun = false; p.ended = false; }
        }
        size_t initial() const { return 0; }
        size_t iteration(uint32_t i, uint32_t stage) const { return 1 + 7 * i + stage; }
        size_t final() const { return 1 + 7 * cap_; }
        void begin(size_t index, cudaStream_t stream) {
                auto& p = pairs_.at(index);
                if (p.begun) throw std::logic_error("stage event pair would be overwritten");
                check(cudaEventRecord(p.start, stream)); p.begun = true;
        }
        void end(size_t index, cudaStream_t stream) {
                auto& p = pairs_.at(index);
                if (!p.begun || p.ended) throw std::logic_error("unpaired stage end");
                check(cudaEventRecord(p.stop, stream)); p.ended = true;
        }
        std::vector<ExperimentStageWindow> completed_windows() const {
                static const char* names[] = {"setup_kkt", "schur_system", "pcg", "compute_dz",
                    "residual_count_D2H", "merit_candidates", "line_search_and_stats_publication"};
                std::vector<ExperimentStageWindow> rows;
                for (size_t index = 0; index < pairs_.size(); ++index) {
                        bool outer = index == initial() || index == final();
                        ExperimentStageWindow row{index == initial() ? "initial_reset_merit_snapshot" :
                            (index == final() ? "final_reset_and_merit" : names[(index - 1) % 7]),
                            outer ? -1 : static_cast<int>((index - 1) / 7)};
                        const auto& p = pairs_[index];
                        if (p.begun != p.ended) throw std::logic_error("incomplete recorded stage");
                        if (p.ended) {
                                float elapsed_ms = 0;
                                check(cudaEventElapsedTime(&elapsed_ms, p.start, p.stop));
                                row.measured = true; row.elapsed_us = elapsed_ms * 1000.0;
                                row.unavailable_reason = nullptr;
                        }
                        rows.push_back(row);
                }
                return rows;
        }
};
