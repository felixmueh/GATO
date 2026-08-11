#pragma once

#include "settings.h"
#include "utils/cuda.cuh"

#if !defined(PLANT_TIAGO_RIGHT)
#error "The tool-position query currently has an audited Tiago implementation only"
#endif

namespace gato {
namespace kernels {

// Read-only batched FK query for the exact Tiago tool frame used by the cost.
// Host-facing joint coordinates are remapped to GRiD's generated convention in
// the same way as the Tiago tracking-cost kernels.
template<typename T, uint32_t BatchSize>
__global__ void tiagoToolPositionKernel(
    T* d_positions,
    const T* d_q_batch,
    const grid::robotModel<T>* d_robot_model)
{
        extern __shared__ unsigned char shared_bytes[];
        T* s_q_grid = reinterpret_cast<T*>(shared_bytes);
        T* s_pose = s_q_grid + grid::NUM_JOINTS;
        T* s_workspace = s_pose + grid::EE_POS_SIZE;

        const uint32_t lane = blockIdx.x;
        if (lane >= BatchSize) { return; }
        for (uint32_t joint = threadIdx.x; joint < grid::NUM_JOINTS; joint += blockDim.x) {
                s_q_grid[joint] = d_q_batch[lane * grid::NUM_JOINTS + joint]
                                  * gato::plant::tiagoJointSign<T>(joint);
        }
        __syncthreads();

        gato::plant::computeTiagoToolPose<T>(s_pose, s_q_grid, d_robot_model, s_workspace);
        for (uint32_t xyz = threadIdx.x; xyz < 3; xyz += blockDim.x) {
                d_positions[lane * 3 + xyz] = s_pose[xyz];
        }
}

template<typename T, uint32_t BatchSize>
void tiagoToolPositionBatched(
    T* d_positions,
    const T* d_q_batch,
    const void* d_dynamics_model,
    cudaStream_t stream)
{
        constexpr uint32_t threads = 128;
        constexpr size_t shared_count = grid::NUM_JOINTS + grid::EE_POS_SIZE
                                        + grid::EE_POS_DYNAMIC_SHARED_MEM_COUNT;
        tiagoToolPositionKernel<T, BatchSize><<<BatchSize, threads, shared_count * sizeof(T), stream>>>(
            d_positions,
            d_q_batch,
            static_cast<const grid::robotModel<T>*>(d_dynamics_model));
        gpuErrchk(cudaPeekAtLastError());
}

}  // namespace kernels
}  // namespace gato
