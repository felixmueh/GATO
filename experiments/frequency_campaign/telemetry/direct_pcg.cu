// Experiment fixture only: five finite real-space SPD cases, one kernel each.
// Both builds execute identical fixture plumbing; only native telemetry varies.
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "types.cuh"
#include "bsqp/kernels/pcg.cuh"

namespace fs = std::filesystem;

#if defined(GATO_EXPERIMENT_TELEMETRY) && GATO_EXPERIMENT_TELEMETRY
constexpr bool telemetry_enabled = true;
constexpr uint32_t stats_words = 2;
#else
constexpr bool telemetry_enabled = false;
constexpr uint32_t stats_words = 1;
#endif

struct Fixture {
        const char* name;
        float b0, b1, second_diagonal;
        int32_t kkt_mask;
        uint32_t cap;
};
const std::array<Fixture, 5> fixtures = {{
    {"kkt-mask-skip", 1, 1, 1, 1, 0},
    {"zero-initial-rho-zero-cap", 0, 0, 1, 0, 0},
    {"nonzero-initial-rho-zero-cap", 1, 1, 1, 0, 0},
    {"tolerance-exactly-at-cap", 1, 1, 1, 0, 1},
    {"one-iteration-cap-nonconverged", 1, 1, 2, 0, 1},
}};

template<typename Value>
void save(const fs::path& name, const std::vector<Value>& values)
{
        std::ofstream stream(name, std::ios::binary);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(Value));
}

int main(int argc, char** argv)
{
        if (argc != 2) {
                std::cerr << "Usage: direct_pcg NEW_OUTPUT_DIRECTORY\n";
                return 2;
        }
        try {
                const fs::path output(argv[1]);
                // Preserve each attempted fixture before any device work.
                if (!fs::create_directory(output)) {
                        throw std::runtime_error("Output directory already exists or cannot be created");
                }
                std::ofstream config(output / "configuration.json");
                config.exceptions(std::ios::badbit | std::ios::failbit);
                config << "{\"N\":" << KNOT_POINTS << ",\"batch_size\":1,\"state_size\":" << STATE_SIZE
                       << ",\"vector_size_padded\":" << VEC_SIZE_PADDED
                       << ",\"stats_words\":" << stats_words
                       << ",\"telemetry_enabled\":" << (telemetry_enabled ? "true" : "false") << "}\n";
                config.close();

                cudaStream_t stream;
                gpuErrchk(cudaStreamCreate(&stream));
                float *d_A, *d_M, *d_b, *d_x, *d_epsilon;
                uint32_t* d_stats;
                int32_t* d_mask;
                gpuErrchk(cudaMalloc(&d_A, B3D_MATRIX_SIZE_PADDED * sizeof(float)));
                gpuErrchk(cudaMalloc(&d_M, B3D_MATRIX_SIZE_PADDED * sizeof(float)));
                gpuErrchk(cudaMalloc(&d_b, VEC_SIZE_PADDED * sizeof(float)));
                gpuErrchk(cudaMalloc(&d_x, VEC_SIZE_PADDED * sizeof(float)));
                gpuErrchk(cudaMalloc(&d_epsilon, sizeof(float)));
                gpuErrchk(cudaMalloc(&d_stats, stats_words * sizeof(uint32_t)));
                gpuErrchk(cudaMalloc(&d_mask, sizeof(int32_t)));

                for (const auto& fixture : fixtures) {
                        const fs::path folder = output / fixture.name;
                        if (!fs::create_directory(folder)) throw std::runtime_error("Fixture output already exists");
                        std::vector<float> A(B3D_MATRIX_SIZE_PADDED, 0), M(B3D_MATRIX_SIZE_PADDED, 0);
                        std::vector<float> b(VEC_SIZE_PADDED, 0), x(VEC_SIZE_PADDED, 0);
                        std::vector<uint32_t> stats(stats_words, 0);
                        // Native blocks are row-major, left/diagonal/right concatenated.
                        for (uint32_t knot = 0; knot < KNOT_POINTS; ++knot) {
                                for (uint32_t j = 0; j < STATE_SIZE; ++j) {
                                        const uint32_t index = knot * BLOCK_ROW_SIZE + j * BLOCK_ROW_R_DIM + STATE_SIZE + j;
                                        A[index] = M[index] = 1;
                                }
                        }
                        A[BLOCK_ROW_R_DIM + STATE_SIZE + 1] = fixture.second_diagonal;
                        b[STATE_SIZE] = fixture.b0;
                        b[STATE_SIZE + 1] = fixture.b1;
                        save(folder / "A.f32", A);
                        save(folder / "M_inv.f32", M);
                        save(folder / "b.f32", b);
                        save(folder / "initial_x.f32", x);
                        std::ofstream request(folder / "request.json");
                        request.exceptions(std::ios::badbit | std::ios::failbit);
                        request << "{\"status\":\"started\",\"max_pcg_iters\":" << fixture.cap
                                << ",\"epsilon\":0,\"kkt_mask\":" << fixture.kkt_mask << "}\n";
                        request.close();
                        const float epsilon = 0;
                        gpuErrchk(cudaMemcpyAsync(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
                        gpuErrchk(cudaMemcpyAsync(d_M, M.data(), M.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
                        gpuErrchk(cudaMemcpyAsync(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
                        gpuErrchk(cudaMemcpyAsync(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
                        gpuErrchk(cudaMemcpyAsync(d_stats, stats.data(), stats.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
                        gpuErrchk(cudaMemcpyAsync(d_epsilon, &epsilon, sizeof(float), cudaMemcpyHostToDevice, stream));
                        gpuErrchk(cudaMemcpyAsync(d_mask, &fixture.kkt_mask, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                        SchurSystem<float, 1> schur{d_A, d_M, d_b};
                        std::ofstream submission(folder / "kernel_submission.json");
                        submission.exceptions(std::ios::badbit | std::ios::failbit);
                        submission << "{\"status\":\"pending kernel launch; process death leaves completion unknown\"}\n";
                        submission.close();
                        solvePCGBatched<float, 1>(d_x, schur, d_epsilon, fixture.cap, d_mask, d_stats, stream);
                        gpuErrchk(cudaGetLastError());
                        gpuErrchk(cudaMemcpyAsync(x.data(), d_x, x.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
                        gpuErrchk(cudaMemcpyAsync(stats.data(), d_stats, stats.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
                        // Test harness publication, identical between both variants.
                        gpuErrchk(cudaStreamSynchronize(stream));
                        save(folder / "output_x.f32", x);
                        save(folder / "stats.u32", stats);
                        std::ofstream completion(folder / "completion.json");
                        completion.exceptions(std::ios::badbit | std::ios::failbit);
                        completion << "{\"status\":\"completed\",\"iterations\":" << stats[0]
                                   << ",\"stop_reason_code\":";
                        if (telemetry_enabled) completion << stats[1]; else completion << "null";
                        completion << ",\"breakdown_stopping_supported\":false,\"breakdown\":null}\n";
                        completion.close();
                }
                gpuErrchk(cudaFree(d_A)); gpuErrchk(cudaFree(d_M));
                gpuErrchk(cudaFree(d_b)); gpuErrchk(cudaFree(d_x));
                gpuErrchk(cudaFree(d_epsilon)); gpuErrchk(cudaFree(d_stats));
                gpuErrchk(cudaFree(d_mask)); gpuErrchk(cudaStreamDestroy(stream));
                return 0;
        } catch (const std::exception& error) {
                std::cerr << error.what() << '\n';
                return 1;
        }
}
