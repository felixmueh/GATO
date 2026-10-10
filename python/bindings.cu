#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "bsqp/bsqp.cuh"
#include "types.cuh"
#include "utils/cuda.cuh"

namespace py = pybind11;

template<typename T, uint32_t BatchSize>
class PyBSQP {
      public:
        PyBSQP() : solver_()
        {
                // printDeviceInfo();
                setL2PersistingAccess(1.0);
                gpuErrchk(cudaMalloc(&d_xu_traj_batch_, TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_x_s_batch_, STATE_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_reference_traj_batch_, REFERENCE_TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xkp1_batch_, STATE_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xk_, STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_uk_, CONTROL_SIZE * sizeof(T)));

                h_xkp1_batch_.resize(STATE_SIZE * BatchSize);
        }

        PyBSQP(const T        dt,
               const uint32_t max_sqp_iters,
               const T        kkt_tol,
               const uint32_t max_pcg_iters,
               const T        pcg_tol,
               const T        solve_ratio,
               const T        mu,
               const T        q_cost,
               const T        qd_cost,
               const T        u_cost,
               const T        N_cost,
               const T        ee_orient_cost,
               const T        ee_orient_N_cost,
               const T        q_lim_cost,
               const T        vel_lim_cost,
               const T        ctrl_lim_cost,
               const T        rho)
            : solver_(dt, max_sqp_iters, kkt_tol, max_pcg_iters, pcg_tol, solve_ratio, mu, q_cost, qd_cost, u_cost, N_cost, ee_orient_cost, ee_orient_N_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost, rho)
        {
                // printDeviceInfo();
                setL2PersistingAccess(1.0);
                // std::cout << "T : " << typeid(T).name() << std::endl;

                gpuErrchk(cudaMalloc(&d_xu_traj_batch_, TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_x_s_batch_, STATE_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_reference_traj_batch_, REFERENCE_TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xkp1_batch_, STATE_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xk_, STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_uk_, CONTROL_SIZE * sizeof(T)));

                h_xkp1_batch_.resize(STATE_SIZE * BatchSize);
        }

        ~PyBSQP()
        {
                gpuErrchk(cudaFree(d_xu_traj_batch_));
                gpuErrchk(cudaFree(d_x_s_batch_));
                gpuErrchk(cudaFree(d_reference_traj_batch_));
                gpuErrchk(cudaFree(d_xkp1_batch_));
                gpuErrchk(cudaFree(d_xk_));
                gpuErrchk(cudaFree(d_uk_));
        }

        py::dict solve(py::array_t<T> xu_traj_batch, T timestep, py::array_t<T> x_s_batch, py::array_t<T> reference_traj_batch)
        {
                py::buffer_info xu_buf = xu_traj_batch.request();
                py::buffer_info xs_buf = x_s_batch.request();
                py::buffer_info ref_buf = reference_traj_batch.request();

                gpuErrchk(cudaMemcpy(d_xu_traj_batch_, xu_buf.ptr, TRAJ_SIZE * BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_x_s_batch_, xs_buf.ptr, STATE_SIZE * BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_reference_traj_batch_, ref_buf.ptr, REFERENCE_TRAJ_SIZE * BatchSize * sizeof(T), cudaMemcpyHostToDevice));

                ProblemInputs<T, BatchSize> inputs;
                inputs.timestep = timestep;
                inputs.d_x_s_batch = d_x_s_batch_;
                inputs.d_reference_traj_batch = d_reference_traj_batch_;

                // Solve
                SQPStats<T, BatchSize> stats = solver_.solve(d_xu_traj_batch_, inputs);

                // Copy trajectory back to host
                std::vector<T> h_xu_traj(TRAJ_SIZE * BatchSize);
                gpuErrchk(cudaMemcpy(h_xu_traj.data(), d_xu_traj_batch_, TRAJ_SIZE * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));

                // Copy final merits (computed at end of solve) and initial (pre-iteration) merits back to host
                std::vector<T> h_final_merit(BatchSize);
                std::vector<T> h_initial_merit(BatchSize);
                solver_.copy_final_merit_to_host(h_final_merit.data());
                solver_.copy_initial_merit0_to_host(h_initial_merit.data());

                py::dict result;
                result["XU"] = py::array_t<T>({BatchSize, TRAJ_SIZE}, h_xu_traj.data());
                result["sqp_time_us"] = stats.solve_time_us;
                result["sqp_iters"] = py::array_t<int32_t>({BatchSize},                 // shape
                                                           {sizeof(int32_t)},           // stride
                                                           stats.sqp_iterations.data()  // data
                );
                result["kkt_converged"] = py::array_t<int32_t>({BatchSize},                // shape
                                                               {sizeof(int32_t)},          // stride
                                                               stats.kkt_converged.data()  // data
                );
                result["final_merit"] = py::array_t<T>({BatchSize}, {sizeof(T)}, h_final_merit.data());
                result["initial_merit"] = py::array_t<T>({BatchSize}, {sizeof(T)}, h_initial_merit.data());

                // Per-iteration stats: shape them as (iters, BatchSize)
                const size_t num_iters = stats.line_search_stats.size();
                const size_t num_pcg_iters = stats.pcg_stats.size();
                result["ls_num_iters"] = static_cast<int>(num_iters);

                std::vector<float> pcg_times_us;
                std::vector<int>   pcg_iters;
                pcg_times_us.reserve(num_pcg_iters);
                pcg_iters.reserve(num_pcg_iters * BatchSize);
                for (const auto& pcg_stat : stats.pcg_stats) {
                        pcg_times_us.push_back(pcg_stat.solve_time_us);
                        for (size_t i = 0; i < BatchSize; ++i) { pcg_iters.push_back(pcg_stat.num_iterations[i]); }
                }
                {
                        std::vector<py::ssize_t> sh_times = { static_cast<py::ssize_t>(num_pcg_iters) };
                        result["pcg_times_us"] = py::array_t<float>(sh_times, pcg_times_us.data());
                }
                {
                        std::vector<py::ssize_t> sh_iters = { static_cast<py::ssize_t>(num_pcg_iters), static_cast<py::ssize_t>(BatchSize) };
                        result["pcg_iters"] = py::array_t<int>(sh_iters, pcg_iters.data());
                }

#if defined(GATO_EXPERIMENT_TELEMETRY) && GATO_EXPERIMENT_TELEMETRY
                // Include the last PCG row even when SQP exits before line search.
                std::vector<uint32_t> pcg_stop_reason;
                pcg_stop_reason.reserve(num_pcg_iters * BatchSize);
                for (const auto& pcg_stat : stats.pcg_stats) {
                        pcg_stop_reason.insert(pcg_stop_reason.end(), pcg_stat.stop_reason.begin(), pcg_stat.stop_reason.end());
                }
                py::dict telemetry;
                telemetry["schema_version"] = 1;
                telemetry["pcg_stop_reason_codes"] = py::array_t<uint32_t>(
                    {static_cast<py::ssize_t>(num_pcg_iters), static_cast<py::ssize_t>(BatchSize)}, pcg_stop_reason.data());
                telemetry["sqp_stop_reason_code"] = static_cast<uint32_t>(stats.stop_reason);
                telemetry["breakdown_stopping_supported"] = false;
                telemetry["breakdown"] = py::none();
                telemetry["breakdown_unavailable_reason"] = "No explicit breakdown stop exists in this solver; numerical anomalies are not instrumented.";
                result["experiment_telemetry"] = telemetry;
#endif

#if defined(GATO_EXPERIMENT_STAGE_TIMING) && GATO_EXPERIMENT_STAGE_TIMING
                py::dict profile;
                profile["schema_version"] = 1;
                profile["units"] = "stream_elapsed_us";
                profile["scope"] = "Existing-stream windows may include idle gaps from host submission/control; not pure active kernel duration.";
                profile["cohort"] = "profile_only_overhead_present";
                profile["pcg_times_us_is_placeholder"] = true;
                profile["pcg_times_us_unavailable_reason"] = "Existing field is literal zero; use separate experimental PCG stream window.";
                py::list windows;
                for (const auto& window : stats.experiment_stage_windows) {
                        py::dict row;
                        row["stage"] = window.name;
                        row["sqp_iteration"] = window.iteration < 0 ? py::object(py::none()) : py::cast(window.iteration);
                        row["elapsed_us"] = window.measured ? py::cast(window.elapsed_us) : py::object(py::none());
                        row["unavailable_reason"] = window.measured ? py::object(py::none()) : py::cast(window.unavailable_reason);
                        windows.append(row);
                }
                profile["windows"] = windows;
                result["experiment_profile_stages"] = profile;
#endif

                std::vector<float> ls_min_merit;
                std::vector<float> ls_step_size;
                ls_min_merit.reserve(num_iters * BatchSize);
                ls_step_size.reserve(num_iters * BatchSize);
                for (const auto& line_search_stat : stats.line_search_stats) {
                        for (size_t i = 0; i < BatchSize; ++i) {
                                ls_min_merit.push_back(line_search_stat.min_merit[i]);
                                ls_step_size.push_back(line_search_stat.step_size[i]);
                        }
                }
                {
                        std::vector<py::ssize_t> sh = { static_cast<py::ssize_t>(num_iters), static_cast<py::ssize_t>(BatchSize) };
                        result["ls_min_merit"] = py::array_t<float>(sh, ls_min_merit.data());
                        result["ls_step_size"] = py::array_t<float>(sh, ls_step_size.data());
                }

                return py::dict(result);
        }

        void set_f_ext_batch(py::array_t<T> f_ext_batch)
        {
                py::buffer_info f_ext_buf = f_ext_batch.request();
#if defined(PLANT_TIAGO_RIGHT)
                const T* values = static_cast<const T*>(f_ext_buf.ptr);
                for (py::ssize_t i = 0; i < f_ext_buf.size; ++i) {
                        if (values[i] != static_cast<T>(0)) {
                                throw py::value_error("External wrench dynamics are not implemented for plant_type='tiago_right'");
                        }
                }
#endif
                solver_.set_f_ext_batch(static_cast<T*>(f_ext_buf.ptr));
        }

        void set_rho_penalty_batch(py::array_t<T> rho_batch, bool set_as_reset_default = true)
        {
                py::buffer_info buf = rho_batch.request();
                solver_.set_rho_penalty_batch(static_cast<T*>(buf.ptr), set_as_reset_default);
        }

        void set_drho_batch(py::array_t<T> drho_batch, bool set_as_reset_default = true)
        {
                py::buffer_info buf = drho_batch.request();
                solver_.set_drho_batch(static_cast<T*>(buf.ptr), set_as_reset_default);
        }

        void set_cost_weights(T q_cost, T qd_cost, T u_cost, T N_cost,
                              T ee_orient_cost, T ee_orient_N_cost,
                              T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost)
        {
                solver_.set_cost_weights(q_cost, qd_cost, u_cost, N_cost,
                                         ee_orient_cost, ee_orient_N_cost,
                                         q_lim_cost, vel_lim_cost, ctrl_lim_cost);
        }

        void set_mu_batch(py::array_t<T> mu_batch)
        {
                py::buffer_info buf = mu_batch.request();
                solver_.set_mu_batch(static_cast<T*>(buf.ptr));
        }

        void set_pcg_tol_batch(py::array_t<T> eps_batch)
        {
                py::buffer_info buf = eps_batch.request();
                solver_.set_pcg_tol_batch(static_cast<T*>(buf.ptr));
        }

        py::array_t<T> sim_forward(py::array_t<T> xk, py::array_t<T> uk, T dt)
        {
                py::buffer_info xk_buf = xk.request();
                py::buffer_info uk_buf = uk.request();

                gpuErrchk(cudaMemcpy(d_xk_, xk_buf.ptr, STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_uk_, uk_buf.ptr, CONTROL_SIZE * sizeof(T), cudaMemcpyHostToDevice));

                solver_.sim_forward(d_xkp1_batch_, d_xk_, d_uk_, dt);
                gpuErrchk(cudaDeviceSynchronize());

                gpuErrchk(cudaMemcpy(h_xkp1_batch_.data(), d_xkp1_batch_, STATE_SIZE * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));

                return py::array_t<T>({BatchSize, STATE_SIZE}, h_xkp1_batch_.data());
        }

        void set_max_sqp_iters(uint32_t value) { solver_.set_max_sqp_iters(value); }
        void reset_dual() { solver_.reset_dual(); }
        void reset_rho() { solver_.reset_rho(); }
        void set_rho_adaptation(bool enabled) { solver_.set_rho_adaptation(enabled); }

      private:
        BSQP<T, BatchSize> solver_;
        T*                 d_xu_traj_batch_;
        T*                 d_x_s_batch_;
        T*                 d_reference_traj_batch_;

        // for sim_forward
        T *            d_xkp1_batch_, *d_xk_, *d_uk_;
        std::vector<T> h_xkp1_batch_;
};


#if defined(TIAGO_MULTIMODAL_PILLAR)
#define PLANT_SUFFIX tiago_right_multimodal
#elif defined(PLANT_INDY7)
#define PLANT_SUFFIX indy7
#elif defined(PLANT_IIWA14)
#define PLANT_SUFFIX iiwa14
#elif defined(PLANT_TIAGO_RIGHT)
#define PLANT_SUFFIX tiago_right
#else
#error "Unknown plant configuration for bindings"
#endif

#define MODULE_NAME_HELPER(knot, plant) bsqpN##knot##_##plant
#define MODULE_NAME(knot, plant) MODULE_NAME_HELPER(knot, plant)

// Macro to register a PyBSQP class with the given precision type and batch size
#define REGISTER_BSQP_CLASS(Type, BatchSize)                                                                                                                                                     \
        py::class_<PyBSQP<Type, BatchSize>>(m, "BSQP_" #BatchSize "_" #Type)                                                                                                                     \
            .def(py::init<>())                                                                                                                                                                     \
            .def(py::init<const Type, const uint32_t, const Type, const uint32_t, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type>()) \
            .def("solve", &PyBSQP<Type, BatchSize>::solve)                                                                                                                                       \
            .def("set_max_sqp_iters", &PyBSQP<Type, BatchSize>::set_max_sqp_iters) \
            .def("reset_dual", &PyBSQP<Type, BatchSize>::reset_dual)                                                                                                                             \
            .def("set_f_ext_batch", &PyBSQP<Type, BatchSize>::set_f_ext_batch)                                                                                                                   \
            .def("set_rho_penalty_batch", &PyBSQP<Type, BatchSize>::set_rho_penalty_batch, py::arg("rho_batch"), py::arg("set_as_reset_default") = true)                                      \
            .def("set_drho_batch", &PyBSQP<Type, BatchSize>::set_drho_batch, py::arg("drho_batch"), py::arg("set_as_reset_default") = true)                                                     \
            .def("set_cost_weights", &PyBSQP<Type, BatchSize>::set_cost_weights) \
            .def("set_mu_batch", &PyBSQP<Type, BatchSize>::set_mu_batch)                                                                                                                          \
            .def("set_pcg_tol_batch", &PyBSQP<Type, BatchSize>::set_pcg_tol_batch)                                                                                                                \
            .def("sim_forward", &PyBSQP<Type, BatchSize>::sim_forward)                                                                                                                           \
            .def("reset_rho", &PyBSQP<Type, BatchSize>::reset_rho)                                                                                                                                \
            .def("set_rho_adaptation", &PyBSQP<Type, BatchSize>::set_rho_adaptation)

#if defined(GATO_EXPERIMENT_RESOURCE_QUERY) && GATO_EXPERIMENT_RESOURCE_QUERY
// Queries compiled kernel pointers only. No constructor, launch, opt-in
// attribute changes, buffer allocation, or synchronization belongs here.
template<typename T, uint32_t B>
py::dict experimentResources() {
        auto status = [](cudaError_t error) {
                py::dict d; d["code"] = static_cast<int>(error);
                d["message"] = cudaGetErrorString(error); return d;
        };
        py::dict result;
        result["schema_version"] = 1; result["N"] = KNOT_POINTS; result["batch_size"] = B;
        result["precision_bytes"] = sizeof(T);
        result["scope"] = "query-only actual-module resources; occupancy/fit are predictions, not observed launches";
        int device = -1, driver = 0, runtime = 0;
        cudaError_t device_error = cudaGetDevice(&device);
        result["device_status"] = status(device_error);
        cudaError_t driver_error = cudaDriverGetVersion(&driver), runtime_error = cudaRuntimeGetVersion(&runtime);
        result["driver_status"] = status(driver_error); result["runtime_status"] = status(runtime_error);
        result["driver_version"] = driver_error == cudaSuccess ? py::cast(driver) : py::object(py::none());
        result["runtime_version"] = runtime_error == cudaSuccess ? py::cast(runtime) : py::object(py::none());
        cudaDeviceProp p{};
        cudaError_t props_error = device_error == cudaSuccess ? cudaGetDeviceProperties(&p, device) : device_error;
        result["properties_status"] = status(props_error);
        py::dict props;
        if (props_error == cudaSuccess) {
                props["name"] = p.name; props["device_index"] = device;
                props["compute_major"] = p.major; props["compute_minor"] = p.minor;
                props["SM_count"] = p.multiProcessorCount; props["global_memory_bytes"] = p.totalGlobalMem;
                props["shared_bytes_per_block_default"] = p.sharedMemPerBlock;
                props["shared_bytes_per_block_optin"] = p.sharedMemPerBlockOptin;
                props["shared_bytes_per_SM"] = p.sharedMemPerMultiprocessor;
                props["max_threads_per_block"] = p.maxThreadsPerBlock;
                props["max_threads_per_SM"] = p.maxThreadsPerMultiProcessor;
                props["registers_per_block"] = p.regsPerBlock; props["registers_per_SM"] = p.regsPerMultiprocessor;
        }
        result["device"] = props_error == cudaSuccess ? py::object(props) : py::object(py::none());
        py::list kernels;
        auto emit = [&](const char* name, auto kernel, size_t dynamic_bytes, int threads) {
                py::dict row; row["kernel"] = name; row["dynamic_shared_bytes"] = dynamic_bytes;
                row["threads"] = threads;
                cudaFuncAttributes a{};
                cudaError_t attr_error = cudaFuncGetAttributes(&a, kernel);
                row["attributes_status"] = status(attr_error);
                py::dict attributes;
                if (attr_error == cudaSuccess) {
                        attributes["static_shared_bytes"] = a.sharedSizeBytes;
                        attributes["registers_per_thread"] = a.numRegs;
                        attributes["local_bytes_per_thread"] = a.localSizeBytes;
                        attributes["max_threads_per_block"] = a.maxThreadsPerBlock;
                        attributes["max_dynamic_shared_bytes"] = a.maxDynamicSharedSizeBytes;
                        attributes["binary_version"] = a.binaryVersion; attributes["ptx_version"] = a.ptxVersion;
                }
                row["attributes"] = attr_error == cudaSuccess ? py::object(attributes) : py::object(py::none());
                row["fits_default_limits_prediction"] = attr_error == cudaSuccess && props_error == cudaSuccess ?
                    py::cast(dynamic_bytes + a.sharedSizeBytes <= p.sharedMemPerBlock &&
                        threads <= a.maxThreadsPerBlock && threads <= p.maxThreadsPerBlock) : py::object(py::none());
                int blocks = 0;
                cudaError_t occupancy_error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, dynamic_bytes);
                row["occupancy_status"] = status(occupancy_error);
                row["theoretical_active_blocks_per_SM"] = occupancy_error == cudaSuccess ? py::cast(blocks) : py::object(py::none());
                row["observed_launch"] = py::none();
                row["observed_launch_unavailable_reason"] = "query-only; join separately recorded F01c launch outcomes";
                kernels.append(row);
        };
        emit("setup_kkt", setupKKTSystemBatchedKernel<T, B>, getSetupKKTSystemBatchedSMemSize<T>(), KKT_THREADS);
        emit("merit", computeMeritBatchedKernel<T, B>, getComputeMeritBatchedSMemSize<T>(), grid::SUGGESTED_THREADS);
        emit("pcg", solvePCGBatchedKernel<T, B>, getSolvePCGBatchedSMemSize<T>(), PCG_THREADS);
        emit("schur1", formSchurSystemBatchedKernel1<T, B>, getFormSchurSystemBatched1SMemSize<T>(), SCHUR_THREADS);
        emit("schur2", formSchurSystemBatchedKernel2<T, B>, getFormSchurSystemBatched2SMemSize<T>(), SCHUR_THREADS);
        emit("dz", computeDzBatchedKernel<T, B>, getComputeDzBatchedSMemSize<T>(), DZ_THREADS);
        emit("line_search", lineSearchAndUpdateBatchedKernel<T, B, NUM_ALPHAS>,
            sizeof(T) * NUM_ALPHAS + sizeof(uint32_t) * NUM_ALPHAS, LINE_SEARCH_THREADS);
        result["kernels"] = kernels;
        return result;
}
#endif

PYBIND11_MODULE(MODULE_NAME(KNOT_POINTS, PLANT_SUFFIX), m)
{
        m.attr("KNOT_POINTS") = KNOT_POINTS;  // to check num knots for current module
#if (defined(GATO_EXPERIMENT_STAGE_TIMING) && GATO_EXPERIMENT_STAGE_TIMING) || (defined(GATO_EXPERIMENT_RESOURCE_QUERY) && GATO_EXPERIMENT_RESOURCE_QUERY)
        py::dict flags;
#if defined(GATO_EXPERIMENT_STAGE_TIMING) && GATO_EXPERIMENT_STAGE_TIMING
        flags["GATO_EXPERIMENT_STAGE_TIMING"] = true;
#else
        flags["GATO_EXPERIMENT_STAGE_TIMING"] = false;
#endif
#if defined(GATO_EXPERIMENT_RESOURCE_QUERY) && GATO_EXPERIMENT_RESOURCE_QUERY
        flags["GATO_EXPERIMENT_RESOURCE_QUERY"] = true;
#else
        flags["GATO_EXPERIMENT_RESOURCE_QUERY"] = false;
#endif
#if defined(GATO_EXPERIMENT_TELEMETRY) && GATO_EXPERIMENT_TELEMETRY
        flags["GATO_EXPERIMENT_TELEMETRY"] = true;
#else
        flags["GATO_EXPERIMENT_TELEMETRY"] = false;
#endif
        m.attr("experiment_build_flags") = flags;
#endif
#if defined(GATO_EXPERIMENT_RESOURCE_QUERY) && GATO_EXPERIMENT_RESOURCE_QUERY
        m.def("experiment_resources", [](uint32_t batch_size) {
#ifdef USE_DOUBLES
                using ResourceType = double;
#else
                using ResourceType = float;
#endif
                if (batch_size == 1) return experimentResources<ResourceType, 1>();
                if (batch_size == 16) return experimentResources<ResourceType, 16>();
                throw py::value_error("resource study queries only batch sizes 1 and 16");
        });
#endif



#ifdef USE_DOUBLES
        REGISTER_BSQP_CLASS(double, 1);
        REGISTER_BSQP_CLASS(double, 2);
        REGISTER_BSQP_CLASS(double, 4);
        REGISTER_BSQP_CLASS(double, 8);
        REGISTER_BSQP_CLASS(double, 16);
        REGISTER_BSQP_CLASS(double, 32);
        REGISTER_BSQP_CLASS(double, 64);
        REGISTER_BSQP_CLASS(double, 128);
#else
        REGISTER_BSQP_CLASS(float, 1);
        REGISTER_BSQP_CLASS(float, 2);
        REGISTER_BSQP_CLASS(float, 4);
        REGISTER_BSQP_CLASS(float, 8);
        REGISTER_BSQP_CLASS(float, 16);
        REGISTER_BSQP_CLASS(float, 32);
        REGISTER_BSQP_CLASS(float, 64);
        REGISTER_BSQP_CLASS(float, 128);
        REGISTER_BSQP_CLASS(float, 256);
        REGISTER_BSQP_CLASS(float, 512);
        REGISTER_BSQP_CLASS(float, 1024);
#endif
}
