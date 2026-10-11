#pragma once

// Simulation-only TIAGo allocation; launches the existing batched kernel.
template<typename T>
constexpr size_t tiagoModelDeviceBytes()
{
        // init_XImats allocates856 scalars, topology helpers are nullptr,
        // and init_robotModel allocates the two-pointer model descriptor.
        return 856 * sizeof(T) + sizeof(grid::robotModel<T>);
}

template<typename T, uint32_t BatchSize>
class PySimulation {
      public:
        PySimulation()
        {
                gpuErrchk(cudaStreamCreate(&stream_));
                model_ = gato::plant::initializeDynamicsConstMem<T>();
                gpuErrchk(cudaMalloc(&d_state_, STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_control_, CONTROL_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_wrenches_, 6 * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_output_, STATE_SIZE * BatchSize * sizeof(T)));
        }

        ~PySimulation()
        {
                gpuErrchk(cudaFree(d_state_));
                gpuErrchk(cudaFree(d_control_));
                gpuErrchk(cudaFree(d_wrenches_));
                gpuErrchk(cudaFree(d_output_));
                gato::plant::freeDynamicsConstMem<T>(model_);
                gpuErrchk(cudaStreamDestroy(stream_));
        }

        py::array_t<T> sim_forward(
            py::array_t<T, py::array::c_style | py::array::forcecast> state,
            py::array_t<T, py::array::c_style | py::array::forcecast> control,
            py::array_t<T, py::array::c_style | py::array::forcecast> linear_first_wrenches, T dt)
        {
                auto x = state.request(), u = control.request(), f = linear_first_wrenches.request();
                if (x.size != STATE_SIZE || u.size != CONTROL_SIZE || f.ndim != 2 ||
                    f.shape[0] != BatchSize || f.shape[1] != 6 || !std::isfinite(dt) || dt <= 0) {
                        throw py::value_error("Simulation requires state(nx), control(nu), wrenches(B,6), positive finite dt");
                }
                const T* linear = static_cast<T*>(f.ptr);
                std::vector<T> angular(6 * BatchSize);
                for (uint32_t b = 0; b < BatchSize; ++b) {
                        for (uint32_t j = 0; j < 3; ++j) {
                                angular[b*6+j] = linear[b*6+j+3];
                                angular[b*6+j+3] = linear[b*6+j];
                        }
                }
                for (auto buffer : {std::pair<T*, size_t>(static_cast<T*>(x.ptr), STATE_SIZE),
                                    std::pair<T*, size_t>(static_cast<T*>(u.ptr), CONTROL_SIZE),
                                    std::pair<T*, size_t>(angular.data(), angular.size())}) {
                        for (size_t j = 0; j < buffer.second; ++j) {
                                if (!std::isfinite(buffer.first[j])) throw py::value_error("Finite simulation inputs required");
                        }
                }
                gpuErrchk(cudaMemcpyAsync(d_state_, x.ptr, STATE_SIZE*sizeof(T), cudaMemcpyHostToDevice, stream_));
                gpuErrchk(cudaMemcpyAsync(d_control_, u.ptr, CONTROL_SIZE*sizeof(T), cudaMemcpyHostToDevice, stream_));
                gpuErrchk(cudaMemcpyAsync(d_wrenches_, angular.data(), angular.size()*sizeof(T), cudaMemcpyHostToDevice, stream_));
                simForwardBatched<T, BatchSize>(d_output_, d_state_, d_control_, model_, d_wrenches_, dt, stream_);
                py::array_t<T> result({BatchSize, STATE_SIZE});
                gpuErrchk(cudaMemcpyAsync(result.mutable_data(), d_output_, STATE_SIZE*BatchSize*sizeof(T), cudaMemcpyDeviceToHost, stream_));
                gpuErrchk(cudaStreamSynchronize(stream_));
                return result;
        }

        py::dict workspace_bytes() const
        {
                py::dict result;
                const size_t simulation = (STATE_SIZE + CONTROL_SIZE + (STATE_SIZE+6)*BatchSize)*sizeof(T);
                result["device_bytes"] = tiagoModelDeviceBytes<T>() + simulation;
                result["pinned_host_bytes"] = 0;
                result["model_device_bytes"] = tiagoModelDeviceBytes<T>();
                result["simulation_device_bytes"] = simulation;
                result["scope"] = "requested persistent allocations; excludes CUDA context/allocator overhead, ordinary heap and temporary returned NumPy arrays";
                return result;
        }

      private:
        cudaStream_t stream_;
        void* model_;
        T *d_state_, *d_control_, *d_wrenches_, *d_output_;
};
