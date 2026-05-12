#pragma once

#include <iostream>
#include <cstdint>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include "settings.h"
#include "constants.h"
#include "types.cuh"
#include "kernels/setup_kkt.cuh"
#include "kernels/schur_linsys.cuh"
#include "kernels/pcg.cuh"
#if defined(GATO_LINSYS_QDLDL)
#include "kernels/qdldl_linsys.cuh"
#endif
#include "kernels/merit.cuh"
#include "kernels/line_search.cuh"
#include "kernels/sim.cuh"

using namespace sqp;

template<typename T, uint32_t BatchSize>
class BSQP {
      public:
        // Default constructor for Python interface flexibility
        BSQP()
            : dt_(0.01), max_sqp_iters_(5), kkt_tol_(0.0001), max_pcg_iters_(100), pcg_tol_(1e-5), solve_ratio_(1.0), mu_(10.0), 
              q_cost_(1.0), qd_cost_(1e-3), u_cost_(1e-6), N_cost_(50.0), q_lim_cost_(1e-3), vel_lim_cost_(0.0), ctrl_lim_cost_(0.0), 
              rho_(1e-3), adapt_rho_(true), debug_schur_dump_count_(0)
        {
                gpuErrchk(cudaStreamCreate(&stream_));
                allocateMemory();
                for (uint32_t i = 0; i < BatchSize; i++) {
                        h_drho_batch_init_[i] = static_cast<T>(1.0);
                        h_rho_penalty_batch_init_[i] = static_cast<T>(rho_);
                        h_mu_batch_init_[i] = static_cast<T>(mu_);
                        h_pcg_tol_batch_init_[i] = static_cast<T>(pcg_tol_);
                }
                gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_mu_batch_, h_mu_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_pcg_tol_batch_, h_pcg_tol_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaDeviceSynchronize());
        }

        BSQP(T dt, uint32_t max_sqp_iters, T kkt_tol, uint32_t max_pcg_iters, T pcg_tol, T solve_ratio, T mu, T q_cost, T qd_cost, T u_cost, T N_cost, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, T rho)
            : dt_(dt), max_sqp_iters_(max_sqp_iters), kkt_tol_(kkt_tol), max_pcg_iters_(max_pcg_iters), pcg_tol_(pcg_tol), solve_ratio_(solve_ratio), mu_(mu), q_cost_(q_cost), qd_cost_(qd_cost),
              u_cost_(u_cost), N_cost_(N_cost), q_lim_cost_(q_lim_cost), vel_lim_cost_(vel_lim_cost), ctrl_lim_cost_(ctrl_lim_cost), rho_(rho), adapt_rho_(true), debug_schur_dump_count_(0)
        {
                gpuErrchk(cudaStreamCreate(&stream_));
                allocateMemory();
                for (uint32_t i = 0; i < BatchSize; i++) {
                        h_drho_batch_init_[i] = static_cast<T>(1.0);
                        h_rho_penalty_batch_init_[i] = static_cast<T>(rho_);
                        h_mu_batch_init_[i] = static_cast<T>(mu_);
                        h_pcg_tol_batch_init_[i] = static_cast<T>(pcg_tol_);
                }
                gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_mu_batch_, h_mu_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_pcg_tol_batch_, h_pcg_tol_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaDeviceSynchronize());
        }

        ~BSQP() { freeMemory(); gpuErrchk(cudaStreamDestroy(stream_)); }

        void set_f_ext_batch(T* h_f_ext_batch) { gpuErrchk(cudaMemcpy(d_f_ext_batch_, h_f_ext_batch, 6 * BatchSize * sizeof(T), cudaMemcpyHostToDevice)); }

        // Hyperparameter setters (batched)
        void set_rho_penalty_batch(const T* h_rho_penalty_batch, bool set_as_reset_default = true)
        {
                if (set_as_reset_default) { memcpy(h_rho_penalty_batch_init_, h_rho_penalty_batch, BatchSize * sizeof(T)); }
                gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
        }

        void set_drho_batch(const T* h_drho_batch, bool set_as_reset_default = true)
        {
                if (set_as_reset_default) { memcpy(h_drho_batch_init_, h_drho_batch, BatchSize * sizeof(T)); }
                gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
        }

        void set_mu_batch(const T* h_mu_batch) { gpuErrchk(cudaMemcpy(d_mu_batch_, h_mu_batch, BatchSize * sizeof(T), cudaMemcpyHostToDevice)); }
        void set_pcg_tol_batch(const T* h_pcg_tol_batch) { gpuErrchk(cudaMemcpy(d_pcg_tol_batch_, h_pcg_tol_batch, BatchSize * sizeof(T), cudaMemcpyHostToDevice)); }

        void reset_dual() { gpuErrchk(cudaMemset(d_lambda_batch_, 0, VEC_SIZE_PADDED * BatchSize * sizeof(T))); }

        void reset_rho()
        {
                gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
        }

        void set_rho_adaptation(bool enabled) { adapt_rho_ = enabled; }

        void sim_forward(T* d_xkp1_batch, T* d_xk, T* d_uk, T dt) { simForwardBatched<T, BatchSize>(d_xkp1_batch, d_xk, d_uk, d_GRiD_mem_, d_f_ext_batch_, dt, stream_); }

        void copy_final_merit_to_host(T* h_out)
        {
                gpuErrchk(cudaMemcpy(h_out, d_merit_initial_batch_, BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
        }

        void copy_initial_merit0_to_host(T* h_out)
        {
                gpuErrchk(cudaMemcpy(h_out, d_merit_initial0_batch_, BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
        }

        SQPStats<T, BatchSize> solve(T* d_xu_traj_batch, ProblemInputs<T, BatchSize> inputs)
        {
                SQPStats<T, BatchSize>        sqp_stats;
                PCGStats<BatchSize>           pcg_stats;
                LineSearchStats<T, BatchSize> ls_stats;

                auto sqp_start_time = std::chrono::high_resolution_clock::now();

                // set d_dz_batch_ to zero
                gpuErrchk(cudaMemsetAsync(d_dz_batch_, 0, TRAJ_SIZE * BatchSize * sizeof(T), stream_));
                gpuErrchk(cudaMemsetAsync(d_pcg_iterations_, 0, sizeof(uint32_t) * BatchSize, stream_));
                gpuErrchk(cudaMemsetAsync(d_kkt_converged_batch_, 0, sizeof(int32_t) * BatchSize, stream_));

                computeMeritBatched<T, BatchSize, 1>(
                    d_merit_initial_batch_, d_dz_batch_, d_xu_traj_batch, d_f_ext_batch_, inputs, d_mu_batch_, d_GRiD_mem_, q_cost_, qd_cost_, u_cost_, N_cost_, q_lim_cost_, vel_lim_cost_, ctrl_lim_cost_, stream_);
                gpuErrchk(cudaMemcpyAsync(d_merit_initial0_batch_, d_merit_initial_batch_, BatchSize * sizeof(T), cudaMemcpyDeviceToDevice, stream_));

                // SQP Loop
                for (uint32_t i = 0; i < max_sqp_iters_; i++) {
                        setupKKTSystemBatched<T, BatchSize>(kkt_system_batch_, inputs, d_xu_traj_batch, d_f_ext_batch_, d_GRiD_mem_, q_cost_, qd_cost_, u_cost_, N_cost_, q_lim_cost_, vel_lim_cost_, ctrl_lim_cost_, stream_);
                        formSchurSystemBatched<T, BatchSize>(schur_system_batch_, kkt_system_batch_, d_rho_penalty_batch_, stream_);

                        std::vector<T> debug_lambda_before;
                        std::vector<T> debug_trace_rho;
                        std::vector<T> debug_trace_denom;
                        std::vector<T> debug_trace_alpha;
                        T*             d_debug_trace_rho = nullptr;
                        T*             d_debug_trace_denom = nullptr;
                        T*             d_debug_trace_alpha = nullptr;
                        uint32_t       debug_trace_stride = 0;
#if !defined(GATO_LINSYS_QDLDL)
                        const bool debug_wants_dump = debugSchurDumpEnabled();
                        if (debug_wants_dump && debugSchurShouldWatchIteration(i)) {
                                debug_lambda_before.resize(VEC_SIZE_PADDED * BatchSize);
                                gpuErrchk(cudaMemcpyAsync(debug_lambda_before.data(), d_lambda_batch_, debug_lambda_before.size() * sizeof(T), cudaMemcpyDeviceToHost, stream_));
                                debug_trace_stride = max_pcg_iters_ + 1;
                                const size_t trace_size = debug_trace_stride * BatchSize;
                                debug_trace_rho.resize(trace_size, static_cast<T>(0));
                                debug_trace_denom.resize(trace_size, static_cast<T>(0));
                                debug_trace_alpha.resize(trace_size, static_cast<T>(0));
                                gpuErrchk(cudaMalloc(&d_debug_trace_rho, trace_size * sizeof(T)));
                                gpuErrchk(cudaMalloc(&d_debug_trace_denom, trace_size * sizeof(T)));
                                gpuErrchk(cudaMalloc(&d_debug_trace_alpha, trace_size * sizeof(T)));
                                gpuErrchk(cudaMemsetAsync(d_debug_trace_rho, 0, trace_size * sizeof(T), stream_));
                                gpuErrchk(cudaMemsetAsync(d_debug_trace_denom, 0, trace_size * sizeof(T), stream_));
                                gpuErrchk(cudaMemsetAsync(d_debug_trace_alpha, 0, trace_size * sizeof(T), stream_));
                        }
#endif

                        // gpuErrchk(cudaEventRecord(pcg_start_event_));
#if defined(GATO_LINSYS_QDLDL)
                        auto linsys_start_time = std::chrono::high_resolution_clock::now();
                        gato::qdldl_linsys::solveQDLDLBatched<T, BatchSize>(d_lambda_batch_, schur_system_batch_, d_pcg_iterations_, stream_);
                        gpuErrchk(cudaStreamSynchronize(stream_));
                        auto linsys_end_time = std::chrono::high_resolution_clock::now();
#else
                        solvePCGBatched<T, BatchSize>(d_lambda_batch_,
                                                      schur_system_batch_,
                                                      d_pcg_tol_batch_,
                                                      max_pcg_iters_,
                                                      d_kkt_converged_batch_,
                                                      d_pcg_iterations_,
                                                      stream_,
                                                      d_debug_trace_rho,
                                                      d_debug_trace_denom,
                                                      d_debug_trace_alpha,
                                                      debug_trace_stride);
#endif
                        // gpuErrchk(cudaEventRecord(pcg_stop_event_));
                        // gpuErrchk(cudaEventSynchronize(pcg_stop_event_));

                        computeDzBatched<T, BatchSize>(d_dz_batch_, d_lambda_batch_, kkt_system_batch_, stream_);

                        // d_q_batch, d_r_batch contain the KKT residuals after computeDzBatched
                        gpuErrchk(cudaMemcpyAsync(h_q_batch_, kkt_system_batch_.d_q_batch, STATE_P_KNOTS * BatchSize * sizeof(T), cudaMemcpyDeviceToHost, stream_));
                        gpuErrchk(cudaMemcpyAsync(h_c_batch_, kkt_system_batch_.d_c_batch, STATE_P_KNOTS * BatchSize * sizeof(T), cudaMemcpyDeviceToHost, stream_));
                        // gpuErrchk(cudaMemcpy(h_r_batch_, kkt_system_batch_.d_r_batch, CONTROL_P_KNOTS * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));

                        gpuErrchk(cudaMemcpyAsync(pcg_stats.num_iterations.data(), d_pcg_iterations_, sizeof(uint32_t) * BatchSize, cudaMemcpyDeviceToHost, stream_));
#if defined(GATO_LINSYS_QDLDL)
                        pcg_stats.solve_time_us = std::chrono::duration_cast<std::chrono::microseconds>(linsys_end_time - linsys_start_time).count();
#else
                        pcg_stats.solve_time_us = 0;
                        if (debug_trace_stride > 0) {
                                const size_t trace_size = debug_trace_stride * BatchSize;
                                gpuErrchk(cudaMemcpyAsync(debug_trace_rho.data(), d_debug_trace_rho, trace_size * sizeof(T), cudaMemcpyDeviceToHost, stream_));
                                gpuErrchk(cudaMemcpyAsync(debug_trace_denom.data(), d_debug_trace_denom, trace_size * sizeof(T), cudaMemcpyDeviceToHost, stream_));
                                gpuErrchk(cudaMemcpyAsync(debug_trace_alpha.data(), d_debug_trace_alpha, trace_size * sizeof(T), cudaMemcpyDeviceToHost, stream_));
                        }
                        debugMaybeDumpSchur(i, debug_lambda_before, pcg_stats.num_iterations, debug_trace_rho, debug_trace_denom, debug_trace_alpha, debug_trace_stride);
                        if (d_debug_trace_rho != nullptr) { gpuErrchk(cudaFree(d_debug_trace_rho)); }
                        if (d_debug_trace_denom != nullptr) { gpuErrchk(cudaFree(d_debug_trace_denom)); }
                        if (d_debug_trace_alpha != nullptr) { gpuErrchk(cudaFree(d_debug_trace_alpha)); }
#endif
                        sqp_stats.pcg_stats.push_back(pcg_stats);

                        // KKT condition check on cpu is async with gpu
                        uint32_t num_solved = 0;
                        for (uint32_t b = 0; b < BatchSize; ++b) {
                                const T* q_ptr = h_q_batch_ + b * STATE_P_KNOTS;
                                const T* c_ptr = h_c_batch_ + b * STATE_P_KNOTS;

                                auto abs_cmp = [](T a, T b) { return std::abs(a) < std::abs(b); };

                                T q_max = std::abs(*std::max_element(q_ptr, q_ptr + STATE_P_KNOTS, abs_cmp));
                                T c_max = std::abs(*std::max_element(c_ptr, c_ptr + STATE_P_KNOTS, abs_cmp));

                                // within kkt exit tol or pcg exit tol (no steps taken)
                                if (pcg_stats.num_iterations[b] == 0) {   // || (q_max < kkt_tol_ && c_max < kkt_tol_)
                                        h_kkt_converged_batch_[b] = 1;
                                        h_sqp_iters_B_[b] += 1;
                                }

                                if (h_kkt_converged_batch_[b]) {
                                        num_solved++;
                                } else {
                                        h_sqp_iters_B_[b] += 1;
                                }
                        }

                        if (num_solved >= BatchSize * solve_ratio_) break;

                        gpuErrchk(cudaMemcpyAsync(d_kkt_converged_batch_, h_kkt_converged_batch_, BatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

                        computeMeritBatched<T, BatchSize, NUM_ALPHAS>(
                            d_merit_batch_, d_dz_batch_, d_xu_traj_batch, d_f_ext_batch_, inputs, d_mu_batch_, d_GRiD_mem_, q_cost_, qd_cost_, u_cost_, N_cost_, q_lim_cost_, vel_lim_cost_, ctrl_lim_cost_, stream_);
                        lineSearchAndUpdateBatched<T, BatchSize, NUM_ALPHAS>(
                            d_xu_traj_batch, d_dz_batch_, d_merit_batch_, d_merit_initial_batch_, d_step_size_batch_, d_rho_penalty_batch_, d_drho_batch_, adapt_rho_ ? 1 : 0, stream_);

                        gpuErrchk(cudaMemcpyAsync(ls_stats.min_merit.data(), d_merit_initial_batch_, BatchSize * sizeof(T), cudaMemcpyDeviceToHost, stream_));
                        gpuErrchk(cudaMemcpyAsync(ls_stats.step_size.data(), d_step_size_batch_, BatchSize * sizeof(T), cudaMemcpyDeviceToHost, stream_));
                        sqp_stats.line_search_stats.push_back(ls_stats);
                }

                // Final merit on updated trajectory for selection
                gpuErrchk(cudaMemsetAsync(d_dz_batch_, 0, TRAJ_SIZE * BatchSize * sizeof(T), stream_));
                computeMeritBatched<T, BatchSize, 1>(
                    d_merit_initial_batch_, d_dz_batch_, d_xu_traj_batch, d_f_ext_batch_, inputs, d_mu_batch_, d_GRiD_mem_, q_cost_, qd_cost_, u_cost_, N_cost_, q_lim_cost_, vel_lim_cost_, ctrl_lim_cost_, stream_);

                gpuErrchk(cudaStreamSynchronize(stream_));
                auto sqp_end_time = std::chrono::high_resolution_clock::now();
                gpuErrchk(cudaMemsetAsync(d_sqp_iters_B_, 0, BatchSize * sizeof(uint32_t), stream_));
                gpuErrchk(cudaMemsetAsync(d_all_kkt_converged_, 0, sizeof(int32_t), stream_));
                gpuErrchk(cudaMemsetAsync(d_kkt_converged_batch_, 0, BatchSize * sizeof(int32_t), stream_));
                gpuErrchk(cudaMemcpyAsync(d_drho_batch_, h_drho_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice, stream_));
                sqp_stats.solve_time_us = std::chrono::duration_cast<std::chrono::microseconds>(sqp_end_time - sqp_start_time).count();
                memcpy(sqp_stats.kkt_converged.data(), h_kkt_converged_batch_, BatchSize * sizeof(int32_t));
                memcpy(sqp_stats.sqp_iterations.data(), h_sqp_iters_B_, BatchSize * sizeof(uint32_t));
                memset(h_kkt_converged_batch_, 0, BatchSize * sizeof(int32_t));
                memset(h_sqp_iters_B_, 0, BatchSize * sizeof(uint32_t));

                return sqp_stats;
        }

      private:
        bool debugSchurDumpEnabled() const { return std::getenv("GATO_DEBUG_DUMP_SCHUR_DIR") != nullptr; }

        bool debugSchurShouldWatchIteration(uint32_t sqp_iter) const
        {
                const char* target_iter = std::getenv("GATO_DEBUG_DUMP_SQP_ITER");
                if (target_iter == nullptr) { return true; }
                return sqp_iter == static_cast<uint32_t>(std::strtoul(target_iter, nullptr, 10));
        }

        static void debugWriteBinary(const std::string& path, const void* data, size_t bytes)
        {
                std::ofstream f(path, std::ios::binary);
                f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        }

        void debugMaybeDumpSchur(uint32_t sqp_iter,
                                 const std::vector<T>& lambda_before,
                                 const std::vector<int>& pcg_iterations,
                                 const std::vector<T>& trace_rho,
                                 const std::vector<T>& trace_denom,
                                 const std::vector<T>& trace_alpha,
                                 uint32_t trace_stride)
        {
                const char* dump_dir_env = std::getenv("GATO_DEBUG_DUMP_SCHUR_DIR");
                if (dump_dir_env == nullptr || !debugSchurShouldWatchIteration(sqp_iter)) { return; }

                gpuErrchk(cudaStreamSynchronize(stream_));

                const bool first_cap_only = std::getenv("GATO_DEBUG_DUMP_ALL_SCHUR") == nullptr;
                bool hit_cap = false;
                for (uint32_t b = 0; b < BatchSize; ++b) {
                        if (pcg_iterations[b] >= static_cast<int>(max_pcg_iters_)) {
                                hit_cap = true;
                                break;
                        }
                }
                if (first_cap_only && (!hit_cap || debug_schur_dump_count_ > 0)) { return; }

                std::vector<T> h_S(B3D_MATRIX_SIZE_PADDED * BatchSize);
                std::vector<T> h_P_inv(B3D_MATRIX_SIZE_PADDED * BatchSize);
                std::vector<T> h_gamma(VEC_SIZE_PADDED * BatchSize);
                std::vector<T> h_lambda_after(VEC_SIZE_PADDED * BatchSize);

                gpuErrchk(cudaMemcpyAsync(h_S.data(), schur_system_batch_.d_S_batch, h_S.size() * sizeof(T), cudaMemcpyDeviceToHost, stream_));
                gpuErrchk(cudaMemcpyAsync(h_P_inv.data(), schur_system_batch_.d_P_inv_batch, h_P_inv.size() * sizeof(T), cudaMemcpyDeviceToHost, stream_));
                gpuErrchk(cudaMemcpyAsync(h_gamma.data(), schur_system_batch_.d_gamma_batch, h_gamma.size() * sizeof(T), cudaMemcpyDeviceToHost, stream_));
                gpuErrchk(cudaMemcpyAsync(h_lambda_after.data(), d_lambda_batch_, h_lambda_after.size() * sizeof(T), cudaMemcpyDeviceToHost, stream_));
                gpuErrchk(cudaStreamSynchronize(stream_));

                std::ostringstream prefix;
                prefix << dump_dir_env << "/schur_dump_" << debug_schur_dump_count_ << "_sqp" << sqp_iter;
                const std::string prefix_str = prefix.str();
                debugWriteBinary(prefix_str + "_S.bin", h_S.data(), h_S.size() * sizeof(T));
                debugWriteBinary(prefix_str + "_P_inv.bin", h_P_inv.data(), h_P_inv.size() * sizeof(T));
                debugWriteBinary(prefix_str + "_gamma.bin", h_gamma.data(), h_gamma.size() * sizeof(T));
                debugWriteBinary(prefix_str + "_lambda_after.bin", h_lambda_after.data(), h_lambda_after.size() * sizeof(T));
                if (!lambda_before.empty()) {
                        debugWriteBinary(prefix_str + "_lambda_before.bin", lambda_before.data(), lambda_before.size() * sizeof(T));
                }
                if (trace_stride > 0 && !trace_rho.empty()) {
                        debugWriteBinary(prefix_str + "_trace_rho.bin", trace_rho.data(), trace_rho.size() * sizeof(T));
                        debugWriteBinary(prefix_str + "_trace_denom.bin", trace_denom.data(), trace_denom.size() * sizeof(T));
                        debugWriteBinary(prefix_str + "_trace_alpha.bin", trace_alpha.data(), trace_alpha.size() * sizeof(T));
                }

                std::ofstream manifest(prefix_str + "_manifest.json");
                manifest << "{\n";
                manifest << "  \"sqp_iter\": " << sqp_iter << ",\n";
                manifest << "  \"dump_index\": " << debug_schur_dump_count_ << ",\n";
                manifest << "  \"batch_size\": " << BatchSize << ",\n";
                manifest << "  \"knot_points\": " << KNOT_POINTS << ",\n";
                manifest << "  \"state_size\": " << STATE_SIZE << ",\n";
                manifest << "  \"control_size\": " << CONTROL_SIZE << ",\n";
                manifest << "  \"vec_size_padded\": " << VEC_SIZE_PADDED << ",\n";
                manifest << "  \"block_row_r_dim\": " << BLOCK_ROW_R_DIM << ",\n";
                manifest << "  \"block_row_size\": " << BLOCK_ROW_SIZE << ",\n";
                manifest << "  \"b3d_matrix_size_padded\": " << B3D_MATRIX_SIZE_PADDED << ",\n";
                manifest << "  \"max_pcg_iters\": " << max_pcg_iters_ << ",\n";
                manifest << "  \"trace_stride\": " << trace_stride << ",\n";
                manifest << "  \"pcg_iterations\": [";
                for (uint32_t b = 0; b < BatchSize; ++b) {
                        if (b > 0) { manifest << ", "; }
                        manifest << pcg_iterations[b];
                }
                manifest << "],\n";
                manifest << "  \"files\": {\n";
                manifest << "    \"S\": \"" << prefix_str << "_S.bin\",\n";
                manifest << "    \"P_inv\": \"" << prefix_str << "_P_inv.bin\",\n";
                manifest << "    \"gamma\": \"" << prefix_str << "_gamma.bin\",\n";
                manifest << "    \"lambda_before\": \"" << prefix_str << "_lambda_before.bin\",\n";
                manifest << "    \"lambda_after\": \"" << prefix_str << "_lambda_after.bin\"";
                if (trace_stride > 0 && !trace_rho.empty()) {
                        manifest << ",\n";
                        manifest << "    \"trace_rho\": \"" << prefix_str << "_trace_rho.bin\",\n";
                        manifest << "    \"trace_denom\": \"" << prefix_str << "_trace_denom.bin\",\n";
                        manifest << "    \"trace_alpha\": \"" << prefix_str << "_trace_alpha.bin\"\n";
                } else {
                        manifest << "\n";
                }
                manifest << "  }\n";
                manifest << "}\n";

                ++debug_schur_dump_count_;
        }

        void allocateMemory()
        {
                size_t BT = BatchSize * sizeof(T);
                size_t BI = BatchSize * sizeof(uint32_t);

                d_GRiD_mem_ = gato::plant::initializeDynamicsConstMem<T>();

                gpuErrchk(cudaEventCreate(&pcg_start_event_));
                gpuErrchk(cudaEventCreate(&pcg_stop_event_));

                // Allocate KKT system memory
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_Q_batch, STATE_SQ_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_R_batch, CONTROL_SQ_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_q_batch, STATE_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_r_batch, CONTROL_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_A_batch, STATE_SQ_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_B_batch, STATE_P_CONTROL_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_c_batch, STATE_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&d_dz_batch_, TRAJ_SIZE * BT));
                gpuErrchk(cudaMalloc(&d_lambda_batch_, VEC_SIZE_PADDED * BT));
                gpuErrchk(cudaMemset(d_lambda_batch_, 0, VEC_SIZE_PADDED * BT));

                // Allocate Schur system memory
                gpuErrchk(cudaMalloc(&schur_system_batch_.d_S_batch, B3D_MATRIX_SIZE_PADDED * BT));
                gpuErrchk(cudaMalloc(&schur_system_batch_.d_P_inv_batch, B3D_MATRIX_SIZE_PADDED * BT));
                gpuErrchk(cudaMalloc(&schur_system_batch_.d_gamma_batch, VEC_SIZE_PADDED * BT));
                gpuErrchk(cudaMemset(schur_system_batch_.d_S_batch, 0, B3D_MATRIX_SIZE_PADDED * BT));
                gpuErrchk(cudaMemset(schur_system_batch_.d_P_inv_batch, 0, B3D_MATRIX_SIZE_PADDED * BT));
                gpuErrchk(cudaMemset(schur_system_batch_.d_gamma_batch, 0, VEC_SIZE_PADDED * BT));

                gpuErrchk(cudaMalloc(&d_merit_initial_batch_, BT));
                gpuErrchk(cudaMalloc(&d_merit_initial0_batch_, BT));
                gpuErrchk(cudaMalloc(&d_merit_batch_, NUM_ALPHAS * BT));

                gpuErrchk(cudaMalloc(&d_sqp_iters_B_, BI));
                gpuErrchk(cudaMalloc(&d_pcg_iterations_, BI));
                gpuErrchk(cudaMalloc(&d_step_size_batch_, BT));
                gpuErrchk(cudaMalloc(&d_all_kkt_converged_, sizeof(int32_t)));
                gpuErrchk(cudaMalloc(&d_kkt_converged_batch_, BI));
                gpuErrchk(cudaMalloc(&d_rho_penalty_batch_, BT));
                gpuErrchk(cudaMalloc(&d_drho_batch_, BT));

                gpuErrchk(cudaMalloc(&d_f_ext_batch_, 6 * BT));
                gpuErrchk(cudaMemset(d_f_ext_batch_, 0, 6 * BT));

                // Batched hyperparameters
                gpuErrchk(cudaMalloc(&d_mu_batch_, BT));
                gpuErrchk(cudaMalloc(&d_pcg_tol_batch_, BT));

                gpuErrchk(cudaMallocHost(&h_q_batch_, STATE_P_KNOTS * BT));
                gpuErrchk(cudaMallocHost(&h_r_batch_, CONTROL_P_KNOTS * BT));
                gpuErrchk(cudaMallocHost(&h_c_batch_, STATE_P_KNOTS * BT));
                gpuErrchk(cudaMallocHost(&h_kkt_converged_batch_, BI));
                gpuErrchk(cudaMallocHost(&h_sqp_iters_B_, BI));
                memset(h_kkt_converged_batch_, 0, BI);
                memset(h_sqp_iters_B_, 0, BI);
        }

        void freeMemory()
        {
                gato::plant::freeDynamicsConstMem<T>(d_GRiD_mem_);

                gpuErrchk(cudaEventDestroy(pcg_start_event_));
                gpuErrchk(cudaEventDestroy(pcg_stop_event_));

                gpuErrchk(cudaFree(kkt_system_batch_.d_Q_batch));
                gpuErrchk(cudaFree(kkt_system_batch_.d_R_batch));
                gpuErrchk(cudaFree(kkt_system_batch_.d_q_batch));
                gpuErrchk(cudaFree(kkt_system_batch_.d_r_batch));
                gpuErrchk(cudaFree(kkt_system_batch_.d_A_batch));
                gpuErrchk(cudaFree(kkt_system_batch_.d_B_batch));
                gpuErrchk(cudaFree(kkt_system_batch_.d_c_batch));

                gpuErrchk(cudaFree(schur_system_batch_.d_S_batch));
                gpuErrchk(cudaFree(schur_system_batch_.d_P_inv_batch));
                gpuErrchk(cudaFree(schur_system_batch_.d_gamma_batch));

                gpuErrchk(cudaFree(d_lambda_batch_));
                gpuErrchk(cudaFree(d_dz_batch_));
                gpuErrchk(cudaFree(d_kkt_converged_batch_));
                gpuErrchk(cudaFree(d_merit_initial_batch_));
                gpuErrchk(cudaFree(d_merit_initial0_batch_));
                gpuErrchk(cudaFree(d_merit_batch_));
                gpuErrchk(cudaFree(d_sqp_iters_B_));
                gpuErrchk(cudaFree(d_pcg_iterations_));
                gpuErrchk(cudaFree(d_step_size_batch_));
                gpuErrchk(cudaFree(d_all_kkt_converged_));
                gpuErrchk(cudaFree(d_f_ext_batch_));
                gpuErrchk(cudaFree(d_rho_penalty_batch_));
                gpuErrchk(cudaFree(d_drho_batch_));
                gpuErrchk(cudaFree(d_mu_batch_));
                gpuErrchk(cudaFree(d_pcg_tol_batch_));

                gpuErrchk(cudaFreeHost(h_q_batch_));
                gpuErrchk(cudaFreeHost(h_r_batch_));
                gpuErrchk(cudaFreeHost(h_c_batch_));
                gpuErrchk(cudaFreeHost(h_kkt_converged_batch_));
        }

        // Device memory
        void*                     d_GRiD_mem_;
        KKTSystem<T, BatchSize>   kkt_system_batch_;
        SchurSystem<T, BatchSize> schur_system_batch_;
        T*                        d_lambda_batch_;
        T*                        d_dz_batch_;
        // PCG
        uint32_t* d_pcg_iterations_;
        // Merit
        T* d_merit_initial_batch_;
        T* d_merit_initial0_batch_;
        T* d_merit_batch_;
        // Line search
        T*        d_step_size_batch_;
        int32_t*  d_all_kkt_converged_;
        int32_t*  d_kkt_converged_batch_;
        uint32_t* d_sqp_iters_B_;
        T*        d_f_ext_batch_;

        T* d_rho_penalty_batch_;
        T  h_rho_penalty_batch_init_[BatchSize];
        T  h_drho_batch_init_[BatchSize];
        T* d_drho_batch_;

        // Batched hyperparameters
        T* d_mu_batch_;
        T  h_mu_batch_init_[BatchSize];
        T* d_pcg_tol_batch_;
        T  h_pcg_tol_batch_init_[BatchSize];

        // Host-side buffers for KKT check
        T*          h_q_batch_;
        T*          h_r_batch_;
        T*          h_c_batch_;
        int32_t*    h_kkt_converged_batch_;
        cudaEvent_t    pcg_start_event_, pcg_stop_event_;
        cudaStream_t   stream_;
        float          pcg_time_us_;
        uint32_t*   h_sqp_iters_B_;
        T           dt_;
        uint32_t    max_sqp_iters_;
        T           kkt_tol_;
        uint32_t    max_pcg_iters_;
        T           pcg_tol_;
        T           solve_ratio_;
        T           mu_;
        T           q_cost_;
        T           qd_cost_;
        T           u_cost_;
        T           N_cost_;
        T           q_lim_cost_;
        T           vel_lim_cost_;
        T           ctrl_lim_cost_;
        T           rho_;
        bool        adapt_rho_;
        uint32_t    debug_schur_dump_count_;
};
