/**
 * This instance of grid.cuh is optimized for the urdf: rizon4s
 *
 * Notes:
 *   Interface is:
 *       __host__   robotModel<T> *d_robotModel = init_robotModel<T>()
 *       __host__   cudaStream_t streams = init_grid<T>()
 *       __host__   gridData<T> *hd_ata = init_gridData<T,NUM_TIMESTEPS>();    __host__   close_grid<T>(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T> *hd_data)
 *   
 *       __device__ inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity)
 *       __device__ inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ inverse_dynamics_kernel<T>(T *d_c, const T *d_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __global__ inverse_dynamics_kernel<T>(T *d_c, const T *d_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   inverse_dynamics<T,USE_QDD_FLAG=false,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ inverse_dynamics_vaf_device<T>(T *s_vaf, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity)
 *       __device__ inverse_dynamics_vaf_device<T>(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)
 *   
 *       __device__ direct_minv_device<T>(T *s_Minv, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ direct_minv_Kernel<T>(T *d_Minv, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   direct_minv<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ forward_dynamics_device<T>(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ forward_dynamics_kernel<T>(T *d_qdd, const T *d_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   forward_dynamics<T>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ inverse_dynamics_gradient_device<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *robotModel<T> *d_robotModel, const T gravity)
 *       __device__ inverse_dynamics_gradient_device<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ inverse_dynamics_gradient_kernel<T>(T *d_dc_du, const T *d_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __global__ inverse_dynamics_gradient_kernel<T>(T *d_dc_du, const T *d_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   inverse_dynamics_gradient<T,USE_QDD_FLAG=false,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ forward_dynamics_gradient_device<T>(T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)
 *       __device__ forward_dynamics_gradient_device<T>(T *s_df_du, const T *s_q, const T *s_qd, const T *s_qdd, const T *s_Minv, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ forward_dynamics_gradient_kernel<T>(T *d_df_du, const T *d_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __global__ forward_dynamics_gradient_kernel<T>(T *d_df_du, const T *d_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   forward_dynamics_gradient<T,USE_QDD_MINV_FLAG=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ end_effector_pose_inner<T>(T *s_eePos, const T *s_q, const T *s_Xhom, int *s_topology_helpers, T *s_temp)
 *       __device__ end_effector_pose_device<T>(T *s_eePos, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ end_effector_pose_kernel<T>(T *d_eePos, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   end_effector_pose<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ end_effector_pose_gradient_inner<T>(T *s_deePos, const T *s_q, const T *s_Xhom, const T *s_dXhom, int *s_topology_helpers, T *s_temp)
 *       __device__ end_effector_pose_gradient_device<T>(T *s_deePos, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ end_effector_pose_gradient_kernel<T>(T *d_deePos, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   end_effector_pose_gradient<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ end_effector_pose_gradient_hessian_inner<T>(T *s_deePos, const T *s_q, const T *s_Xhom, const T *s_dXhom, int *s_topology_helpers, T *s_temp)
 *       __device__ end_effector_pose_gradient_hessian_device<T>(T *s_deePos, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ end_effector_pose_gradient_hessian_kernel<T>(T *d_deePos, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   end_effector_pose_gradient_hessian<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *   Suggested Type T is float
 *   
 *   Additional helper functions and ALGORITHM_inner functions which take in __shared__ memory temp variables exist -- see function descriptions in the file
 *   
 *   By default device and kernels need to be launched with dynamic shared mem of size <FUNC_CODE>_DYNAMIC_SHARED_MEM_COUNT where <FUNC_CODE> = [ID, MINV, FD, ID_DU, FD_DU]
 *
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils/cuda.cuh"

// single kernel timing helper code
#define time_delta_us_timespec(start,end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))

#define XIMAT_SIZE 36

template <typename T, int M, int N>
__host__ __device__
void printMat(T *A, int lda){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){printf("%.4f ",A[i + lda*j]);}
        printf("\n");
    }
}

template <typename T, int M, int N>
__host__ __device__
void printMat(const T *A, int lda){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){printf("%.4f ",A[i + lda*j]);}
        printf("\n");
    }
}

/**
 * All functions are kept in this namespace
 *
 */
namespace grid {
    constexpr int NUM_JOINTS = 7;
    constexpr int NUM_VEL = 7;
    constexpr int NUM_EES = 1;
    constexpr int NQ = 7;
    constexpr int NX = 14;
    constexpr int NU = 7;
    constexpr int EE_POS_SIZE = 6;
    constexpr int NEE = 6;
    constexpr int ID_DYNAMIC_SHARED_MEM_COUNT = 546;
    constexpr int MINV_DYNAMIC_SHARED_MEM_COUNT = 1171;
    constexpr int FD_DYNAMIC_SHARED_MEM_COUNT = 1220;
    constexpr int ID_DU_DYNAMIC_SHARED_MEM_COUNT = 2226;
    constexpr int FD_DU_DYNAMIC_SHARED_MEM_COUNT = 2226;
    constexpr int ID_DU_MAX_SHARED_MEM_COUNT = 2471;
    constexpr int FD_DU_MAX_SHARED_MEM_COUNT = 2625;
    constexpr int EE_POS_DYNAMIC_SHARED_MEM_COUNT = 144;
    constexpr int DEE_POS_DYNAMIC_SHARED_MEM_COUNT = 672;
    constexpr int D2EE_POS_DYNAMIC_SHARED_MEM_COUNT = 2160;
    constexpr int IDSVA_SO_DYNAMIC_SHARED_MEM_COUNT = 0;
    constexpr int FDSVA_SO_DYNAMIC_SHARED_MEM_COUNT = 2562;
    constexpr int SUGGESTED_THREADS = 352;
    // Define custom structs
    template <typename T>
    struct robotModel {
        T *d_XImats;
        int *d_topology_helpers;
    };
    template <typename T>
    struct gridData {
        // GPU INPUTS
        T *d_q_qd_u;
        T *d_q_qd;
        T *d_q;
        // CPU INPUTS
        T *h_q_qd_u;
        T *h_q_qd;
        T *h_q;
        // GPU OUTPUTS
        T *d_c;
        T *d_Minv;
        T *d_qdd;
        T *d_M;
        T *d_dc_du;
        T *d_df_du;
        T *d_eePos;
        T *d_deePos;
        T *d_d2eePos;
        T *d_idsva_so;
        T *d_df2;
        // CPU OUTPUTS
        T *h_c;
        T *h_Minv;
        T *h_qdd;
        T *h_M;
        T *h_dc_du;
        T *h_df_du;
        T *h_eePos;
        T *h_deePos;
        T *h_d2eePos;
        T *h_idsva_so;
        T *h_df2;
    };
    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(const T *vec1, const T *vec2) {
        T result = 0;
        for(int i = 0; i < N; i++) {
            result += vec1[i*S1] * vec2[i*S2];
        }
        return result;
    }

    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(T *vec1, const T *vec2) {
        T result = 0;
        for(int i = 0; i < N; i++) {
            result += vec1[i*S1] * vec2[i*S2];
        }
        return result;
    }

    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(const T *vec1, T *vec2) {
        T result = 0;
        for(int i = 0; i < N; i++) {
            result += vec1[i*S1] * vec2[i*S2];
        }
        return result;
    }

    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(T *vec1, T *vec2) {
        T result = 0;
        for(int i = 0; i < N; i++) {
            result += vec1[i*S1] * vec2[i*S2];
        }
        return result;
    }

    /**
     * Generates the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx0(T *s_vecX, const T *s_vec) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = s_vec[2];
        s_vecX[2] = -s_vec[1];
        s_vecX[3] = static_cast<T>(0);
        s_vecX[4] = s_vec[5];
        s_vecX[5] = -s_vec[4];
    }

    /**
     * Adds the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx0_peq(T *s_vecX, const T *s_vec) {
        s_vecX[1] += s_vec[2];
        s_vecX[2] += -s_vec[1];
        s_vecX[4] += s_vec[5];
        s_vecX[5] += -s_vec[4];
    }

    /**
     * Generates the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx0_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = s_vec[2]*alpha;
        s_vecX[2] = -s_vec[1]*alpha;
        s_vecX[3] = static_cast<T>(0);
        s_vecX[4] = s_vec[5]*alpha;
        s_vecX[5] = -s_vec[4]*alpha;
    }

    /**
     * Adds the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx0_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[1] += s_vec[2]*alpha;
        s_vecX[2] += -s_vec[1]*alpha;
        s_vecX[4] += s_vec[5]*alpha;
        s_vecX[5] += -s_vec[4]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx1(T *s_vecX, const T *s_vec) {
        s_vecX[0] = -s_vec[2];
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = s_vec[0];
        s_vecX[3] = -s_vec[5];
        s_vecX[4] = static_cast<T>(0);
        s_vecX[5] = s_vec[3];
    }

    /**
     * Adds the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx1_peq(T *s_vecX, const T *s_vec) {
        s_vecX[0] += -s_vec[2];
        s_vecX[2] += s_vec[0];
        s_vecX[3] += -s_vec[5];
        s_vecX[5] += s_vec[3];
    }

    /**
     * Generates the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx1_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = -s_vec[2]*alpha;
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = s_vec[0]*alpha;
        s_vecX[3] = -s_vec[5]*alpha;
        s_vecX[4] = static_cast<T>(0);
        s_vecX[5] = s_vec[3]*alpha;
    }

    /**
     * Adds the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx1_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] += -s_vec[2]*alpha;
        s_vecX[2] += s_vec[0]*alpha;
        s_vecX[3] += -s_vec[5]*alpha;
        s_vecX[5] += s_vec[3]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx2(T *s_vecX, const T *s_vec) {
        s_vecX[0] = s_vec[1];
        s_vecX[1] = -s_vec[0];
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = s_vec[4];
        s_vecX[4] = -s_vec[3];
        s_vecX[5] = static_cast<T>(0);
    }

    /**
     * Adds the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx2_peq(T *s_vecX, const T *s_vec) {
        s_vecX[0] += s_vec[1];
        s_vecX[1] += -s_vec[0];
        s_vecX[3] += s_vec[4];
        s_vecX[4] += -s_vec[3];
    }

    /**
     * Generates the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx2_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = s_vec[1]*alpha;
        s_vecX[1] = -s_vec[0]*alpha;
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = s_vec[4]*alpha;
        s_vecX[4] = -s_vec[3]*alpha;
        s_vecX[5] = static_cast<T>(0);
    }

    /**
     * Adds the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx2_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] += s_vec[1]*alpha;
        s_vecX[1] += -s_vec[0]*alpha;
        s_vecX[3] += s_vec[4]*alpha;
        s_vecX[4] += -s_vec[3]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx3(T *s_vecX, const T *s_vec) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = static_cast<T>(0);
        s_vecX[4] = s_vec[2];
        s_vecX[5] = -s_vec[1];
    }

    /**
     * Adds the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx3_peq(T *s_vecX, const T *s_vec) {
        s_vecX[4] += s_vec[2];
        s_vecX[5] += -s_vec[1];
    }

    /**
     * Generates the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx3_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = static_cast<T>(0);
        s_vecX[4] = s_vec[2]*alpha;
        s_vecX[5] = -s_vec[1]*alpha;
    }

    /**
     * Adds the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx3_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[4] += s_vec[2]*alpha;
        s_vecX[5] += -s_vec[1]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx4(T *s_vecX, const T *s_vec) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = -s_vec[2];
        s_vecX[4] = static_cast<T>(0);
        s_vecX[5] = s_vec[0];
    }

    /**
     * Adds the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx4_peq(T *s_vecX, const T *s_vec) {
        s_vecX[3] += -s_vec[2];
        s_vecX[5] += s_vec[0];
    }

    /**
     * Generates the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx4_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = -s_vec[2]*alpha;
        s_vecX[4] = static_cast<T>(0);
        s_vecX[5] = s_vec[0]*alpha;
    }

    /**
     * Adds the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx4_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[3] += -s_vec[2]*alpha;
        s_vecX[5] += s_vec[0]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx5(T *s_vecX, const T *s_vec) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = s_vec[1];
        s_vecX[4] = -s_vec[0];
        s_vecX[5] = static_cast<T>(0);
    }

    /**
     * Adds the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx5_peq(T *s_vecX, const T *s_vec) {
        s_vecX[3] += s_vec[1];
        s_vecX[4] += -s_vec[0];
    }

    /**
     * Generates the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx5_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = s_vec[1]*alpha;
        s_vecX[4] = -s_vec[0]*alpha;
        s_vecX[5] = static_cast<T>(0);
    }

    /**
     * Adds the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx5_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[3] += s_vec[1]*alpha;
        s_vecX[4] += -s_vec[0]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mxX(T *s_vecX, const T *s_vec, const int S_ind) {
        switch(S_ind){
            case 0: mx0<T>(s_vecX, s_vec); break;
            case 1: mx1<T>(s_vecX, s_vec); break;
            case 2: mx2<T>(s_vecX, s_vec); break;
            case 3: mx3<T>(s_vecX, s_vec); break;
            case 4: mx4<T>(s_vecX, s_vec); break;
            case 5: mx5<T>(s_vecX, s_vec); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mxX_peq(T *s_vecX, const T *s_vec, const int S_ind) {
        switch(S_ind){
            case 0: mx0_peq<T>(s_vecX, s_vec); break;
            case 1: mx1_peq<T>(s_vecX, s_vec); break;
            case 2: mx2_peq<T>(s_vecX, s_vec); break;
            case 3: mx3_peq<T>(s_vecX, s_vec); break;
            case 4: mx4_peq<T>(s_vecX, s_vec); break;
            case 5: mx5_peq<T>(s_vecX, s_vec); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mxX_scaled(T *s_vecX, const T *s_vec, const T alpha, const int S_ind) {
        switch(S_ind){
            case 0: mx0_scaled<T>(s_vecX, s_vec, alpha); break;
            case 1: mx1_scaled<T>(s_vecX, s_vec, alpha); break;
            case 2: mx2_scaled<T>(s_vecX, s_vec, alpha); break;
            case 3: mx3_scaled<T>(s_vecX, s_vec, alpha); break;
            case 4: mx4_scaled<T>(s_vecX, s_vec, alpha); break;
            case 5: mx5_scaled<T>(s_vecX, s_vec, alpha); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mxX_peq_scaled(T *s_vecX, const T *s_vec, const T alpha, const int S_ind) {
        switch(S_ind){
            case 0: mx0_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 1: mx1_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 2: mx2_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 3: mx3_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 4: mx4_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 5: mx5_peq_scaled<T>(s_vecX, s_vec, alpha); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_matX is the destination matrix
     * @param s_vecX is the source vector
     */
    template <typename T>
    __device__
    void fx(T *s_matX, const T *s_vecX) {
        s_matX[6*0 + 0] = static_cast<T>(0);
        s_matX[6*0 + 1] = s_vecX[2];
        s_matX[6*0 + 2] = -s_vecX[1];
        s_matX[6*0 + 3] = static_cast<T>(0);
        s_matX[6*0 + 4] = static_cast<T>(0);
        s_matX[6*0 + 5] = static_cast<T>(0);
        s_matX[6*1 + 0] = -s_vecX[2];
        s_matX[6*1 + 1] = static_cast<T>(0);
        s_matX[6*1 + 2] = s_vecX[0];
        s_matX[6*1 + 3] = static_cast<T>(0);
        s_matX[6*1 + 4] = static_cast<T>(0);
        s_matX[6*1 + 5] = static_cast<T>(0);
        s_matX[6*2 + 0] = s_vecX[1];
        s_matX[6*2 + 1] = -s_vecX[0];
        s_matX[6*2 + 2] = static_cast<T>(0);
        s_matX[6*2 + 3] = static_cast<T>(0);
        s_matX[6*2 + 4] = static_cast<T>(0);
        s_matX[6*2 + 5] = static_cast<T>(0);
        s_matX[6*3 + 0] = static_cast<T>(0);
        s_matX[6*3 + 1] = s_vecX[5];
        s_matX[6*3 + 2] = -s_vecX[4];
        s_matX[6*3 + 3] = static_cast<T>(0);
        s_matX[6*3 + 4] = s_vecX[2];
        s_matX[6*3 + 5] = -s_vecX[1];
        s_matX[6*4 + 0] = -s_vecX[5];
        s_matX[6*4 + 1] = static_cast<T>(0);
        s_matX[6*4 + 2] = s_vecX[3];
        s_matX[6*4 + 3] = -s_vecX[2];
        s_matX[6*4 + 4] = static_cast<T>(0);
        s_matX[6*4 + 5] = s_vecX[0];
        s_matX[6*5 + 0] = s_vecX[4];
        s_matX[6*5 + 1] = -s_vecX[3];
        s_matX[6*5 + 2] = static_cast<T>(0);
        s_matX[6*5 + 3] = s_vecX[1];
        s_matX[6*5 + 4] = -s_vecX[0];
        s_matX[6*5 + 5] = static_cast<T>(0);
    }

    /**
     * Generates the motion vector cross product matrix for a pre-zeroed destination
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *   Assumes destination is zeroed
     *
     * @param s_matX is the destination matrix
     * @param s_vecX is the source vector
     */
    template <typename T>
    __device__
    void fx_zeroed(T *s_matX, const T *s_vecX) {
        s_matX[6*0 + 1] = s_vecX[2];
        s_matX[6*0 + 2] = -s_vecX[1];
        s_matX[6*1 + 0] = -s_vecX[2];
        s_matX[6*1 + 2] = s_vecX[0];
        s_matX[6*2 + 0] = s_vecX[1];
        s_matX[6*2 + 1] = -s_vecX[0];
        s_matX[6*3 + 1] = s_vecX[5];
        s_matX[6*3 + 2] = -s_vecX[4];
        s_matX[6*3 + 4] = s_vecX[2];
        s_matX[6*3 + 5] = -s_vecX[1];
        s_matX[6*4 + 0] = -s_vecX[5];
        s_matX[6*4 + 2] = s_vecX[3];
        s_matX[6*4 + 3] = -s_vecX[2];
        s_matX[6*4 + 5] = s_vecX[0];
        s_matX[6*5 + 0] = s_vecX[4];
        s_matX[6*5 + 1] = -s_vecX[3];
        s_matX[6*5 + 3] = s_vecX[1];
        s_matX[6*5 + 4] = -s_vecX[0];
    }

    /**
     * Generates the motion vector cross product matrix and multiples by the input vector
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_result is the result vector
     * @param s_fxVec is the fx vector
     * @param s_timesVec is the multipled vector
     */
    template <typename T>
    __device__
    void fx_times_v(T *s_result, const T *s_fxVec, const T *s_timesVec) {
        s_result[0] = -s_fxVec[2] * s_timesVec[1] + s_fxVec[1] * s_timesVec[2] - s_fxVec[5] * s_timesVec[4] + s_fxVec[4] * s_timesVec[5];
        s_result[1] =  s_fxVec[2] * s_timesVec[0] - s_fxVec[0] * s_timesVec[2] + s_fxVec[5] * s_timesVec[3] - s_fxVec[3] * s_timesVec[5];
        s_result[2] = -s_fxVec[1] * s_timesVec[0] + s_fxVec[0] * s_timesVec[1] - s_fxVec[4] * s_timesVec[3] + s_fxVec[3] * s_timesVec[4];
        s_result[3] =                                                          - s_fxVec[2] * s_timesVec[4] + s_fxVec[1] * s_timesVec[5];
        s_result[4] =                                                            s_fxVec[2] * s_timesVec[3] - s_fxVec[0] * s_timesVec[5];
        s_result[5] =                                                          - s_fxVec[1] * s_timesVec[3] + s_fxVec[0] * s_timesVec[4];
    }

    /**
     * Adds the motion vector cross product matrix multiplied by the input vector
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_result is the result vector
     * @param s_fxVec is the fx vector
     * @param s_timesVec is the multipled vector
     */
    template <typename T>
    __device__
    void fx_times_v_peq(T *s_result, const T *s_fxVec, const T *s_timesVec) {
        s_result[0] += -s_fxVec[2] * s_timesVec[1] + s_fxVec[1] * s_timesVec[2] - s_fxVec[5] * s_timesVec[4] + s_fxVec[4] * s_timesVec[5];
        s_result[1] +=  s_fxVec[2] * s_timesVec[0] - s_fxVec[0] * s_timesVec[2] + s_fxVec[5] * s_timesVec[3] - s_fxVec[3] * s_timesVec[5];
        s_result[2] += -s_fxVec[1] * s_timesVec[0] + s_fxVec[0] * s_timesVec[1] - s_fxVec[4] * s_timesVec[3] + s_fxVec[3] * s_timesVec[4];
        s_result[3] +=                                                          - s_fxVec[2] * s_timesVec[4] + s_fxVec[1] * s_timesVec[5];
        s_result[4] +=                                                            s_fxVec[2] * s_timesVec[3] - s_fxVec[0] * s_timesVec[5];
        s_result[5] +=                                                          - s_fxVec[1] * s_timesVec[3] + s_fxVec[0] * s_timesVec[4];
    }

    /**
     * Initializes the topology_helpers in GPU memory
     *
     * @return A pointer to the topology_helpers memory in the GPU
     */
    template <typename T>
    __host__
    int *init_topology_helpers() {
        int h_topology_helpers[] = {2,1,2,1,2,1,2}; // S_inds
        int *d_topology_helpers; gpuErrchk(cudaMalloc((void**)&d_topology_helpers,7*sizeof(int)));
        gpuErrchk(cudaMemcpy(d_topology_helpers,h_topology_helpers,7*sizeof(int),cudaMemcpyHostToDevice));
        return d_topology_helpers;
    }

    /**
     * Initializes the Xmats and Imats in GPU memory
     *
     * Notes:
     *   Memory order is X[0...N], I[0...N]
     *
     * @return A pointer to the XI memory in the GPU
     */
    template <typename T>
    __host__
    T* init_XImats() {
        T *h_XImats = (T *)malloc(504*sizeof(T));
        // X[0]
        h_XImats[0] = static_cast<T>(0);
        h_XImats[1] = static_cast<T>(0);
        h_XImats[2] = static_cast<T>(0);
        h_XImats[3] = static_cast<T>(0);
        h_XImats[4] = static_cast<T>(0);
        h_XImats[5] = static_cast<T>(0);
        h_XImats[6] = static_cast<T>(0);
        h_XImats[7] = static_cast<T>(0);
        h_XImats[8] = static_cast<T>(0);
        h_XImats[9] = static_cast<T>(0);
        h_XImats[10] = static_cast<T>(0);
        h_XImats[11] = static_cast<T>(0);
        h_XImats[12] = static_cast<T>(0);
        h_XImats[13] = static_cast<T>(0);
        h_XImats[14] = static_cast<T>(1.00000000000000);
        h_XImats[15] = static_cast<T>(0);
        h_XImats[16] = static_cast<T>(0);
        h_XImats[17] = static_cast<T>(0);
        h_XImats[18] = static_cast<T>(0);
        h_XImats[19] = static_cast<T>(0);
        h_XImats[20] = static_cast<T>(0);
        h_XImats[21] = static_cast<T>(0);
        h_XImats[22] = static_cast<T>(0);
        h_XImats[23] = static_cast<T>(0);
        h_XImats[24] = static_cast<T>(0);
        h_XImats[25] = static_cast<T>(0);
        h_XImats[26] = static_cast<T>(0);
        h_XImats[27] = static_cast<T>(0);
        h_XImats[28] = static_cast<T>(0);
        h_XImats[29] = static_cast<T>(0);
        h_XImats[30] = static_cast<T>(0);
        h_XImats[31] = static_cast<T>(0);
        h_XImats[32] = static_cast<T>(0);
        h_XImats[33] = static_cast<T>(0);
        h_XImats[34] = static_cast<T>(0);
        h_XImats[35] = static_cast<T>(1.00000000000000);
        // X[1]
        h_XImats[36] = static_cast<T>(0);
        h_XImats[37] = static_cast<T>(0);
        h_XImats[38] = static_cast<T>(0);
        h_XImats[39] = static_cast<T>(0);
        h_XImats[40] = static_cast<T>(-0.210000000000000);
        h_XImats[41] = static_cast<T>(0);
        h_XImats[42] = static_cast<T>(0);
        h_XImats[43] = static_cast<T>(1.00000000000000);
        h_XImats[44] = static_cast<T>(0);
        h_XImats[45] = static_cast<T>(0);
        h_XImats[46] = static_cast<T>(0);
        h_XImats[47] = static_cast<T>(0);
        h_XImats[48] = static_cast<T>(0);
        h_XImats[49] = static_cast<T>(0);
        h_XImats[50] = static_cast<T>(0);
        h_XImats[51] = static_cast<T>(0);
        h_XImats[52] = static_cast<T>(0);
        h_XImats[53] = static_cast<T>(0);
        h_XImats[54] = static_cast<T>(0);
        h_XImats[55] = static_cast<T>(0);
        h_XImats[56] = static_cast<T>(0);
        h_XImats[57] = static_cast<T>(0);
        h_XImats[58] = static_cast<T>(0);
        h_XImats[59] = static_cast<T>(0);
        h_XImats[60] = static_cast<T>(0);
        h_XImats[61] = static_cast<T>(0);
        h_XImats[62] = static_cast<T>(0);
        h_XImats[63] = static_cast<T>(0);
        h_XImats[64] = static_cast<T>(1.00000000000000);
        h_XImats[65] = static_cast<T>(0);
        h_XImats[66] = static_cast<T>(0);
        h_XImats[67] = static_cast<T>(0);
        h_XImats[68] = static_cast<T>(0);
        h_XImats[69] = static_cast<T>(0);
        h_XImats[70] = static_cast<T>(0);
        h_XImats[71] = static_cast<T>(0);
        // X[2]
        h_XImats[72] = static_cast<T>(0);
        h_XImats[73] = static_cast<T>(0);
        h_XImats[74] = static_cast<T>(0);
        h_XImats[75] = static_cast<T>(0);
        h_XImats[76] = static_cast<T>(0);
        h_XImats[77] = static_cast<T>(0.0350000000000000);
        h_XImats[78] = static_cast<T>(0);
        h_XImats[79] = static_cast<T>(0);
        h_XImats[80] = static_cast<T>(0);
        h_XImats[81] = static_cast<T>(0);
        h_XImats[82] = static_cast<T>(0);
        h_XImats[83] = static_cast<T>(0);
        h_XImats[84] = static_cast<T>(0);
        h_XImats[85] = static_cast<T>(0);
        h_XImats[86] = static_cast<T>(1.00000000000000);
        h_XImats[87] = static_cast<T>(0);
        h_XImats[88] = static_cast<T>(0);
        h_XImats[89] = static_cast<T>(0);
        h_XImats[90] = static_cast<T>(0);
        h_XImats[91] = static_cast<T>(0);
        h_XImats[92] = static_cast<T>(0);
        h_XImats[93] = static_cast<T>(0);
        h_XImats[94] = static_cast<T>(0);
        h_XImats[95] = static_cast<T>(0);
        h_XImats[96] = static_cast<T>(0);
        h_XImats[97] = static_cast<T>(0);
        h_XImats[98] = static_cast<T>(0);
        h_XImats[99] = static_cast<T>(0);
        h_XImats[100] = static_cast<T>(0);
        h_XImats[101] = static_cast<T>(0);
        h_XImats[102] = static_cast<T>(0);
        h_XImats[103] = static_cast<T>(0);
        h_XImats[104] = static_cast<T>(0);
        h_XImats[105] = static_cast<T>(0);
        h_XImats[106] = static_cast<T>(0);
        h_XImats[107] = static_cast<T>(1.00000000000000);
        // X[3]
        h_XImats[108] = static_cast<T>(0);
        h_XImats[109] = static_cast<T>(0);
        h_XImats[110] = static_cast<T>(0);
        h_XImats[111] = static_cast<T>(0);
        h_XImats[112] = static_cast<T>(0.190000000000000);
        h_XImats[113] = static_cast<T>(0);
        h_XImats[114] = static_cast<T>(0);
        h_XImats[115] = static_cast<T>(-1.00000000000000);
        h_XImats[116] = static_cast<T>(0);
        h_XImats[117] = static_cast<T>(0);
        h_XImats[118] = static_cast<T>(0);
        h_XImats[119] = static_cast<T>(0);
        h_XImats[120] = static_cast<T>(0);
        h_XImats[121] = static_cast<T>(0);
        h_XImats[122] = static_cast<T>(0);
        h_XImats[123] = static_cast<T>(0);
        h_XImats[124] = static_cast<T>(0.0200000000000000);
        h_XImats[125] = static_cast<T>(0);
        h_XImats[126] = static_cast<T>(0);
        h_XImats[127] = static_cast<T>(0);
        h_XImats[128] = static_cast<T>(0);
        h_XImats[129] = static_cast<T>(0);
        h_XImats[130] = static_cast<T>(0);
        h_XImats[131] = static_cast<T>(0);
        h_XImats[132] = static_cast<T>(0);
        h_XImats[133] = static_cast<T>(0);
        h_XImats[134] = static_cast<T>(0);
        h_XImats[135] = static_cast<T>(0);
        h_XImats[136] = static_cast<T>(-1.00000000000000);
        h_XImats[137] = static_cast<T>(0);
        h_XImats[138] = static_cast<T>(0);
        h_XImats[139] = static_cast<T>(0);
        h_XImats[140] = static_cast<T>(0);
        h_XImats[141] = static_cast<T>(0);
        h_XImats[142] = static_cast<T>(0);
        h_XImats[143] = static_cast<T>(0);
        // X[4]
        h_XImats[144] = static_cast<T>(0);
        h_XImats[145] = static_cast<T>(0);
        h_XImats[146] = static_cast<T>(0);
        h_XImats[147] = static_cast<T>(0);
        h_XImats[148] = static_cast<T>(0);
        h_XImats[149] = static_cast<T>(0.0250000000000000);
        h_XImats[150] = static_cast<T>(0);
        h_XImats[151] = static_cast<T>(0);
        h_XImats[152] = static_cast<T>(0);
        h_XImats[153] = static_cast<T>(0);
        h_XImats[154] = static_cast<T>(0);
        h_XImats[155] = static_cast<T>(0.0200000000000000);
        h_XImats[156] = static_cast<T>(0);
        h_XImats[157] = static_cast<T>(0);
        h_XImats[158] = static_cast<T>(1.00000000000000);
        h_XImats[159] = static_cast<T>(0);
        h_XImats[160] = static_cast<T>(0);
        h_XImats[161] = static_cast<T>(0);
        h_XImats[162] = static_cast<T>(0);
        h_XImats[163] = static_cast<T>(0);
        h_XImats[164] = static_cast<T>(0);
        h_XImats[165] = static_cast<T>(0);
        h_XImats[166] = static_cast<T>(0);
        h_XImats[167] = static_cast<T>(0);
        h_XImats[168] = static_cast<T>(0);
        h_XImats[169] = static_cast<T>(0);
        h_XImats[170] = static_cast<T>(0);
        h_XImats[171] = static_cast<T>(0);
        h_XImats[172] = static_cast<T>(0);
        h_XImats[173] = static_cast<T>(0);
        h_XImats[174] = static_cast<T>(0);
        h_XImats[175] = static_cast<T>(0);
        h_XImats[176] = static_cast<T>(0);
        h_XImats[177] = static_cast<T>(0);
        h_XImats[178] = static_cast<T>(0);
        h_XImats[179] = static_cast<T>(1.00000000000000);
        // X[5]
        h_XImats[180] = static_cast<T>(0);
        h_XImats[181] = static_cast<T>(0);
        h_XImats[182] = static_cast<T>(0);
        h_XImats[183] = static_cast<T>(0);
        h_XImats[184] = static_cast<T>(-0.190000000000000);
        h_XImats[185] = static_cast<T>(0);
        h_XImats[186] = static_cast<T>(0);
        h_XImats[187] = static_cast<T>(1.00000000000000);
        h_XImats[188] = static_cast<T>(0);
        h_XImats[189] = static_cast<T>(0);
        h_XImats[190] = static_cast<T>(0);
        h_XImats[191] = static_cast<T>(0);
        h_XImats[192] = static_cast<T>(0);
        h_XImats[193] = static_cast<T>(0);
        h_XImats[194] = static_cast<T>(0);
        h_XImats[195] = static_cast<T>(0);
        h_XImats[196] = static_cast<T>(0);
        h_XImats[197] = static_cast<T>(0);
        h_XImats[198] = static_cast<T>(0);
        h_XImats[199] = static_cast<T>(0);
        h_XImats[200] = static_cast<T>(0);
        h_XImats[201] = static_cast<T>(0);
        h_XImats[202] = static_cast<T>(0);
        h_XImats[203] = static_cast<T>(0);
        h_XImats[204] = static_cast<T>(0);
        h_XImats[205] = static_cast<T>(0);
        h_XImats[206] = static_cast<T>(0);
        h_XImats[207] = static_cast<T>(0);
        h_XImats[208] = static_cast<T>(1.00000000000000);
        h_XImats[209] = static_cast<T>(0);
        h_XImats[210] = static_cast<T>(0);
        h_XImats[211] = static_cast<T>(0);
        h_XImats[212] = static_cast<T>(0);
        h_XImats[213] = static_cast<T>(0);
        h_XImats[214] = static_cast<T>(0);
        h_XImats[215] = static_cast<T>(0);
        // X[6]
        h_XImats[216] = static_cast<T>(0);
        h_XImats[217] = static_cast<T>(0);
        h_XImats[218] = static_cast<T>(-1.00000000000000);
        h_XImats[219] = static_cast<T>(0);
        h_XImats[220] = static_cast<T>(0);
        h_XImats[221] = static_cast<T>(0);
        h_XImats[222] = static_cast<T>(0);
        h_XImats[223] = static_cast<T>(0);
        h_XImats[224] = static_cast<T>(0);
        h_XImats[225] = static_cast<T>(0);
        h_XImats[226] = static_cast<T>(0);
        h_XImats[227] = static_cast<T>(-0.110000000000000);
        h_XImats[228] = static_cast<T>(0);
        h_XImats[229] = static_cast<T>(0);
        h_XImats[230] = static_cast<T>(0);
        h_XImats[231] = static_cast<T>(0);
        h_XImats[232] = static_cast<T>(0);
        h_XImats[233] = static_cast<T>(0.0730000000000000);
        h_XImats[234] = static_cast<T>(0);
        h_XImats[235] = static_cast<T>(0);
        h_XImats[236] = static_cast<T>(0);
        h_XImats[237] = static_cast<T>(0);
        h_XImats[238] = static_cast<T>(0);
        h_XImats[239] = static_cast<T>(-1.00000000000000);
        h_XImats[240] = static_cast<T>(0);
        h_XImats[241] = static_cast<T>(0);
        h_XImats[242] = static_cast<T>(0);
        h_XImats[243] = static_cast<T>(0);
        h_XImats[244] = static_cast<T>(0);
        h_XImats[245] = static_cast<T>(0);
        h_XImats[246] = static_cast<T>(0);
        h_XImats[247] = static_cast<T>(0);
        h_XImats[248] = static_cast<T>(0);
        h_XImats[249] = static_cast<T>(0);
        h_XImats[250] = static_cast<T>(0);
        h_XImats[251] = static_cast<T>(0);
        // I[0]
        h_XImats[252] = static_cast<T>(0.11362);
        h_XImats[253] = static_cast<T>(0.0);
        h_XImats[254] = static_cast<T>(0.0);
        h_XImats[255] = static_cast<T>(0.0);
        h_XImats[256] = static_cast<T>(-0.555);
        h_XImats[257] = static_cast<T>(0.037000000000000005);
        h_XImats[258] = static_cast<T>(0.0);
        h_XImats[259] = static_cast<T>(0.11325);
        h_XImats[260] = static_cast<T>(-0.00555);
        h_XImats[261] = static_cast<T>(0.555);
        h_XImats[262] = static_cast<T>(0.0);
        h_XImats[263] = static_cast<T>(0.0);
        h_XImats[264] = static_cast<T>(0.0);
        h_XImats[265] = static_cast<T>(-0.00555);
        h_XImats[266] = static_cast<T>(0.01037);
        h_XImats[267] = static_cast<T>(-0.037000000000000005);
        h_XImats[268] = static_cast<T>(0.0);
        h_XImats[269] = static_cast<T>(0.0);
        h_XImats[270] = static_cast<T>(0.0);
        h_XImats[271] = static_cast<T>(0.555);
        h_XImats[272] = static_cast<T>(-0.037000000000000005);
        h_XImats[273] = static_cast<T>(3.7);
        h_XImats[274] = static_cast<T>(0.0);
        h_XImats[275] = static_cast<T>(0.0);
        h_XImats[276] = static_cast<T>(-0.555);
        h_XImats[277] = static_cast<T>(0.0);
        h_XImats[278] = static_cast<T>(0.0);
        h_XImats[279] = static_cast<T>(0.0);
        h_XImats[280] = static_cast<T>(3.7);
        h_XImats[281] = static_cast<T>(0.0);
        h_XImats[282] = static_cast<T>(0.037000000000000005);
        h_XImats[283] = static_cast<T>(0.0);
        h_XImats[284] = static_cast<T>(0.0);
        h_XImats[285] = static_cast<T>(0.0);
        h_XImats[286] = static_cast<T>(0.0);
        h_XImats[287] = static_cast<T>(3.7);
        // I[1]
        h_XImats[288] = static_cast<T>(0.051320000000000005);
        h_XImats[289] = static_cast<T>(0.0);
        h_XImats[290] = static_cast<T>(0.0);
        h_XImats[291] = static_cast<T>(0.0);
        h_XImats[292] = static_cast<T>(-0.27);
        h_XImats[293] = static_cast<T>(0.10800000000000001);
        h_XImats[294] = static_cast<T>(0.0);
        h_XImats[295] = static_cast<T>(0.047);
        h_XImats[296] = static_cast<T>(-0.010800000000000002);
        h_XImats[297] = static_cast<T>(0.27);
        h_XImats[298] = static_cast<T>(0.0);
        h_XImats[299] = static_cast<T>(0.0);
        h_XImats[300] = static_cast<T>(0.0);
        h_XImats[301] = static_cast<T>(-0.0108);
        h_XImats[302] = static_cast<T>(0.004320000000000001);
        h_XImats[303] = static_cast<T>(-0.10800000000000001);
        h_XImats[304] = static_cast<T>(0.0);
        h_XImats[305] = static_cast<T>(0.0);
        h_XImats[306] = static_cast<T>(0.0);
        h_XImats[307] = static_cast<T>(0.27);
        h_XImats[308] = static_cast<T>(-0.10800000000000001);
        h_XImats[309] = static_cast<T>(2.7);
        h_XImats[310] = static_cast<T>(0.0);
        h_XImats[311] = static_cast<T>(0.0);
        h_XImats[312] = static_cast<T>(-0.27);
        h_XImats[313] = static_cast<T>(0.0);
        h_XImats[314] = static_cast<T>(0.0);
        h_XImats[315] = static_cast<T>(0.0);
        h_XImats[316] = static_cast<T>(2.7);
        h_XImats[317] = static_cast<T>(0.0);
        h_XImats[318] = static_cast<T>(0.10800000000000001);
        h_XImats[319] = static_cast<T>(0.0);
        h_XImats[320] = static_cast<T>(0.0);
        h_XImats[321] = static_cast<T>(0.0);
        h_XImats[322] = static_cast<T>(0.0);
        h_XImats[323] = static_cast<T>(2.7);
        // I[2]
        h_XImats[324] = static_cast<T>(0.05056);
        h_XImats[325] = static_cast<T>(0.0);
        h_XImats[326] = static_cast<T>(0.0031200000000000004);
        h_XImats[327] = static_cast<T>(0.0);
        h_XImats[328] = static_cast<T>(-0.312);
        h_XImats[329] = static_cast<T>(0.0);
        h_XImats[330] = static_cast<T>(0.0);
        h_XImats[331] = static_cast<T>(0.0508);
        h_XImats[332] = static_cast<T>(0.0);
        h_XImats[333] = static_cast<T>(0.312);
        h_XImats[334] = static_cast<T>(0.0);
        h_XImats[335] = static_cast<T>(0.024);
        h_XImats[336] = static_cast<T>(0.00312);
        h_XImats[337] = static_cast<T>(0.0);
        h_XImats[338] = static_cast<T>(0.00024);
        h_XImats[339] = static_cast<T>(0.0);
        h_XImats[340] = static_cast<T>(-0.024);
        h_XImats[341] = static_cast<T>(0.0);
        h_XImats[342] = static_cast<T>(0.0);
        h_XImats[343] = static_cast<T>(0.312);
        h_XImats[344] = static_cast<T>(0.0);
        h_XImats[345] = static_cast<T>(2.4);
        h_XImats[346] = static_cast<T>(0.0);
        h_XImats[347] = static_cast<T>(0.0);
        h_XImats[348] = static_cast<T>(-0.312);
        h_XImats[349] = static_cast<T>(0.0);
        h_XImats[350] = static_cast<T>(-0.024);
        h_XImats[351] = static_cast<T>(0.0);
        h_XImats[352] = static_cast<T>(2.4);
        h_XImats[353] = static_cast<T>(0.0);
        h_XImats[354] = static_cast<T>(0.0);
        h_XImats[355] = static_cast<T>(0.024);
        h_XImats[356] = static_cast<T>(0.0);
        h_XImats[357] = static_cast<T>(0.0);
        h_XImats[358] = static_cast<T>(0.0);
        h_XImats[359] = static_cast<T>(2.4);
        // I[3]
        h_XImats[360] = static_cast<T>(0.04616);
        h_XImats[361] = static_cast<T>(0.0007199999999999999);
        h_XImats[362] = static_cast<T>(0.0024000000000000002);
        h_XImats[363] = static_cast<T>(0.0);
        h_XImats[364] = static_cast<T>(-0.24);
        h_XImats[365] = static_cast<T>(0.072);
        h_XImats[366] = static_cast<T>(0.0007199999999999999);
        h_XImats[367] = static_cast<T>(0.04424);
        h_XImats[368] = static_cast<T>(-0.0072);
        h_XImats[369] = static_cast<T>(0.24);
        h_XImats[370] = static_cast<T>(0.0);
        h_XImats[371] = static_cast<T>(0.024);
        h_XImats[372] = static_cast<T>(0.0024);
        h_XImats[373] = static_cast<T>(-0.0072);
        h_XImats[374] = static_cast<T>(0.0024);
        h_XImats[375] = static_cast<T>(-0.072);
        h_XImats[376] = static_cast<T>(-0.024);
        h_XImats[377] = static_cast<T>(0.0);
        h_XImats[378] = static_cast<T>(0.0);
        h_XImats[379] = static_cast<T>(0.24);
        h_XImats[380] = static_cast<T>(-0.072);
        h_XImats[381] = static_cast<T>(2.4);
        h_XImats[382] = static_cast<T>(0.0);
        h_XImats[383] = static_cast<T>(0.0);
        h_XImats[384] = static_cast<T>(-0.24);
        h_XImats[385] = static_cast<T>(0.0);
        h_XImats[386] = static_cast<T>(-0.024);
        h_XImats[387] = static_cast<T>(0.0);
        h_XImats[388] = static_cast<T>(2.4);
        h_XImats[389] = static_cast<T>(0.0);
        h_XImats[390] = static_cast<T>(0.072);
        h_XImats[391] = static_cast<T>(0.024);
        h_XImats[392] = static_cast<T>(0.0);
        h_XImats[393] = static_cast<T>(0.0);
        h_XImats[394] = static_cast<T>(0.0);
        h_XImats[395] = static_cast<T>(2.4);
        // I[4]
        h_XImats[396] = static_cast<T>(0.05056);
        h_XImats[397] = static_cast<T>(0.0);
        h_XImats[398] = static_cast<T>(0.0);
        h_XImats[399] = static_cast<T>(0.0);
        h_XImats[400] = static_cast<T>(-0.312);
        h_XImats[401] = static_cast<T>(0.0);
        h_XImats[402] = static_cast<T>(0.0);
        h_XImats[403] = static_cast<T>(0.05056);
        h_XImats[404] = static_cast<T>(0.0);
        h_XImats[405] = static_cast<T>(0.312);
        h_XImats[406] = static_cast<T>(0.0);
        h_XImats[407] = static_cast<T>(0.0);
        h_XImats[408] = static_cast<T>(0.0);
        h_XImats[409] = static_cast<T>(0.0);
        h_XImats[410] = static_cast<T>(0.0);
        h_XImats[411] = static_cast<T>(0.0);
        h_XImats[412] = static_cast<T>(0.0);
        h_XImats[413] = static_cast<T>(0.0);
        h_XImats[414] = static_cast<T>(0.0);
        h_XImats[415] = static_cast<T>(0.312);
        h_XImats[416] = static_cast<T>(0.0);
        h_XImats[417] = static_cast<T>(2.4);
        h_XImats[418] = static_cast<T>(0.0);
        h_XImats[419] = static_cast<T>(0.0);
        h_XImats[420] = static_cast<T>(-0.312);
        h_XImats[421] = static_cast<T>(0.0);
        h_XImats[422] = static_cast<T>(0.0);
        h_XImats[423] = static_cast<T>(0.0);
        h_XImats[424] = static_cast<T>(2.4);
        h_XImats[425] = static_cast<T>(0.0);
        h_XImats[426] = static_cast<T>(0.0);
        h_XImats[427] = static_cast<T>(0.0);
        h_XImats[428] = static_cast<T>(0.0);
        h_XImats[429] = static_cast<T>(0.0);
        h_XImats[430] = static_cast<T>(0.0);
        h_XImats[431] = static_cast<T>(2.4);
        // I[5]
        h_XImats[432] = static_cast<T>(0.028700000000000003);
        h_XImats[433] = static_cast<T>(-0.00264);
        h_XImats[434] = static_cast<T>(-0.0030800000000000007);
        h_XImats[435] = static_cast<T>(0.0);
        h_XImats[436] = static_cast<T>(-0.15400000000000003);
        h_XImats[437] = static_cast<T>(0.132);
        h_XImats[438] = static_cast<T>(-0.00264);
        h_XImats[439] = static_cast<T>(0.021660000000000006);
        h_XImats[440] = static_cast<T>(-0.009240000000000002);
        h_XImats[441] = static_cast<T>(0.15400000000000003);
        h_XImats[442] = static_cast<T>(0.0);
        h_XImats[443] = static_cast<T>(-0.044000000000000004);
        h_XImats[444] = static_cast<T>(-0.0030800000000000007);
        h_XImats[445] = static_cast<T>(-0.009240000000000002);
        h_XImats[446] = static_cast<T>(0.0088);
        h_XImats[447] = static_cast<T>(-0.132);
        h_XImats[448] = static_cast<T>(0.044000000000000004);
        h_XImats[449] = static_cast<T>(0.0);
        h_XImats[450] = static_cast<T>(0.0);
        h_XImats[451] = static_cast<T>(0.15400000000000003);
        h_XImats[452] = static_cast<T>(-0.132);
        h_XImats[453] = static_cast<T>(2.2);
        h_XImats[454] = static_cast<T>(0.0);
        h_XImats[455] = static_cast<T>(0.0);
        h_XImats[456] = static_cast<T>(-0.15400000000000003);
        h_XImats[457] = static_cast<T>(0.0);
        h_XImats[458] = static_cast<T>(0.044000000000000004);
        h_XImats[459] = static_cast<T>(0.0);
        h_XImats[460] = static_cast<T>(2.2);
        h_XImats[461] = static_cast<T>(0.0);
        h_XImats[462] = static_cast<T>(0.132);
        h_XImats[463] = static_cast<T>(-0.044000000000000004);
        h_XImats[464] = static_cast<T>(0.0);
        h_XImats[465] = static_cast<T>(0.0);
        h_XImats[466] = static_cast<T>(0.0);
        h_XImats[467] = static_cast<T>(2.2);
        // I[6]
        h_XImats[468] = static_cast<T>(0.0034999999999999996);
        h_XImats[469] = static_cast<T>(0.0);
        h_XImats[470] = static_cast<T>(0.0);
        h_XImats[471] = static_cast<T>(0.0);
        h_XImats[472] = static_cast<T>(-0.06999999999999999);
        h_XImats[473] = static_cast<T>(0.0);
        h_XImats[474] = static_cast<T>(0.0);
        h_XImats[475] = static_cast<T>(0.0034999999999999996);
        h_XImats[476] = static_cast<T>(0.0);
        h_XImats[477] = static_cast<T>(0.06999999999999999);
        h_XImats[478] = static_cast<T>(0.0);
        h_XImats[479] = static_cast<T>(0.0);
        h_XImats[480] = static_cast<T>(0.0);
        h_XImats[481] = static_cast<T>(0.0);
        h_XImats[482] = static_cast<T>(0.0);
        h_XImats[483] = static_cast<T>(0.0);
        h_XImats[484] = static_cast<T>(0.0);
        h_XImats[485] = static_cast<T>(0.0);
        h_XImats[486] = static_cast<T>(0.0);
        h_XImats[487] = static_cast<T>(0.06999999999999999);
        h_XImats[488] = static_cast<T>(0.0);
        h_XImats[489] = static_cast<T>(1.4);
        h_XImats[490] = static_cast<T>(0.0);
        h_XImats[491] = static_cast<T>(0.0);
        h_XImats[492] = static_cast<T>(-0.06999999999999999);
        h_XImats[493] = static_cast<T>(0.0);
        h_XImats[494] = static_cast<T>(0.0);
        h_XImats[495] = static_cast<T>(0.0);
        h_XImats[496] = static_cast<T>(1.4);
        h_XImats[497] = static_cast<T>(0.0);
        h_XImats[498] = static_cast<T>(0.0);
        h_XImats[499] = static_cast<T>(0.0);
        h_XImats[500] = static_cast<T>(0.0);
        h_XImats[501] = static_cast<T>(0.0);
        h_XImats[502] = static_cast<T>(0.0);
        h_XImats[503] = static_cast<T>(1.4);
        T *d_XImats; gpuErrchk(cudaMalloc((void**)&d_XImats,504*sizeof(T)));
        gpuErrchk(cudaMemcpy(d_XImats,h_XImats,504*sizeof(T),cudaMemcpyHostToDevice));
        free(h_XImats);
        return d_XImats;
    }

    /**
     * Initializes the robotModel helpers in GPU memory
     *
     * @return A pointer to the robotModel struct
     */
    template <typename T>
    __host__
    robotModel<T>* init_robotModel() {
        robotModel<T> h_robotModel;
        h_robotModel.d_XImats = init_XImats<T>();
        h_robotModel.d_topology_helpers = init_topology_helpers<T>();
        robotModel<T> *d_robotModel; gpuErrchk(cudaMalloc((void**)&d_robotModel,sizeof(robotModel<T>)));
        gpuErrchk(cudaMemcpy(d_robotModel,&h_robotModel,sizeof(robotModel<T>),cudaMemcpyHostToDevice));
        return d_robotModel;
    }

    template<typename T>
    __host__ void free_robotModel(robotModel<T>* d_robotModel)
    {
        gpuErrchk(cudaFree(d_robotModel));
    }
    /**
     * Allocated device and host memory for all computations
     *
     * @return A pointer to the gridData struct of pointers
     */
    template <typename T, int NUM_TIMESTEPS>
    __host__
    gridData<T> *init_gridData(){
        gridData<T> *hd_data = (gridData<T> *)malloc(sizeof(gridData<T>));// first the input variables on the GPU
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd_u, 3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_q_qd_u = (T *)malloc(3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_q_qd = (T *)malloc(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_q = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        // then the GPU outputs
        gpuErrchk(cudaMalloc((void**)&hd_data->d_c, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_Minv, NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_qdd, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_M, NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_dc_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_df_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_eePos, 6*NUM_EES*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_deePos, 6*NUM_EES*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_d2eePos, 6*NUM_EES*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_idsva_so, 4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_df2, 4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_c = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_Minv = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_M = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_qdd = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_dc_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_df_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_eePos = (T *)malloc(6*NUM_EES*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_deePos = (T *)malloc(6*NUM_EES*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_d2eePos = (T *)malloc(6*NUM_EES*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_idsva_so = (T *)malloc(4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_df2 = (T *)malloc(4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        return hd_data;
    }

    /**
     * Allocated device and host memory for all computations
     *
     * @param Max number of timesteps in the trajectory
     * @return A pointer to the gridData struct of pointers
     */
    template <typename T>
    __host__
    gridData<T> *init_gridData(int NUM_TIMESTEPS){
        gridData<T> *hd_data = (gridData<T> *)malloc(sizeof(gridData<T>));// first the input variables on the GPU
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd_u, 3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_q_qd_u = (T *)malloc(3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_q_qd = (T *)malloc(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_q = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        // then the GPU outputs
        gpuErrchk(cudaMalloc((void**)&hd_data->d_c, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_Minv, NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_qdd, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_M, NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_dc_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_df_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_eePos, 6*NUM_EES*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_deePos, 6*NUM_EES*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_d2eePos, 6*NUM_EES*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_idsva_so, 4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_df2, 4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_c = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_Minv = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_M = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_qdd = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_dc_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_df_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_eePos = (T *)malloc(6*NUM_EES*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_deePos = (T *)malloc(6*NUM_EES*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_d2eePos = (T *)malloc(6*NUM_EES*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_idsva_so = (T *)malloc(4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_df2 = (T *)malloc(4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        return hd_data;
    }

    /**
     * Updates the Xmats in (shared) GPU memory acording to the configuration
     *
     * @param s_XImats is the (shared) memory destination location for the XImats
     * @param s_q is the (shared) memory location of the current configuration
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)
     * @param s_temp is temporary (shared) memory used to compute sin and cos if needed of size: 14
     */
    template <typename T>
    __device__
    void load_update_XImats_helpers(T *s_XImats, const T *s_q, int *s_topology_helpers, const robotModel<T> *d_robotModel, T *s_temp) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 504; ind += blockDim.x*blockDim.y){
            s_XImats[ind] = d_robotModel->d_XImats[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_topology_helpers[ind] = d_robotModel->d_topology_helpers[ind];
        }
        for(int k = threadIdx.x + threadIdx.y*blockDim.x; k < 7; k += blockDim.x*blockDim.y){
            s_temp[k] = static_cast<T>(sin(s_q[k]));
            s_temp[k+7] = static_cast<T>(cos(s_q[k]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // X[0]
            s_XImats[0] = static_cast<T>(-s_temp[7]);
            s_XImats[1] = static_cast<T>(s_temp[0]);
            s_XImats[3] = static_cast<T>(0.155*s_temp[0]);
            s_XImats[4] = static_cast<T>(0.155*s_temp[7]);
            s_XImats[6] = static_cast<T>(-s_temp[0]);
            s_XImats[7] = static_cast<T>(-s_temp[7]);
            s_XImats[9] = static_cast<T>(-0.155*s_temp[7]);
            s_XImats[10] = static_cast<T>(0.155*s_temp[0]);
            // X[1]
            s_XImats[36] = static_cast<T>(s_temp[8]);
            s_XImats[38] = static_cast<T>(s_temp[1]);
            s_XImats[39] = static_cast<T>(-0.03*s_temp[1]);
            s_XImats[41] = static_cast<T>(0.03*s_temp[8]);
            s_XImats[45] = static_cast<T>(0.21*s_temp[8]);
            s_XImats[47] = static_cast<T>(0.21*s_temp[1]);
            s_XImats[48] = static_cast<T>(-s_temp[1]);
            s_XImats[50] = static_cast<T>(s_temp[8]);
            s_XImats[51] = static_cast<T>(-0.03*s_temp[8]);
            s_XImats[53] = static_cast<T>(-0.03*s_temp[1]);
            // X[2]
            s_XImats[72] = static_cast<T>(s_temp[9]);
            s_XImats[73] = static_cast<T>(-s_temp[2]);
            s_XImats[75] = static_cast<T>(-0.205*s_temp[2]);
            s_XImats[76] = static_cast<T>(-0.205*s_temp[9]);
            s_XImats[78] = static_cast<T>(s_temp[2]);
            s_XImats[79] = static_cast<T>(s_temp[9]);
            s_XImats[81] = static_cast<T>(0.205*s_temp[9]);
            s_XImats[82] = static_cast<T>(-0.205*s_temp[2]);
            s_XImats[87] = static_cast<T>(-0.035*s_temp[9]);
            s_XImats[88] = static_cast<T>(0.035*s_temp[2]);
            // X[3]
            s_XImats[108] = static_cast<T>(-s_temp[10]);
            s_XImats[110] = static_cast<T>(-s_temp[3]);
            s_XImats[111] = static_cast<T>(0.03*s_temp[3]);
            s_XImats[113] = static_cast<T>(-0.03*s_temp[10]);
            s_XImats[117] = static_cast<T>(-0.02*s_temp[3] - 0.19*s_temp[10]);
            s_XImats[119] = static_cast<T>(-0.19*s_temp[3] + 0.02*s_temp[10]);
            s_XImats[120] = static_cast<T>(-s_temp[3]);
            s_XImats[122] = static_cast<T>(s_temp[10]);
            s_XImats[123] = static_cast<T>(-0.03*s_temp[10]);
            s_XImats[125] = static_cast<T>(-0.03*s_temp[3]);
            // X[4]
            s_XImats[144] = static_cast<T>(-s_temp[11]);
            s_XImats[145] = static_cast<T>(s_temp[4]);
            s_XImats[147] = static_cast<T>(0.195*s_temp[4]);
            s_XImats[148] = static_cast<T>(0.195*s_temp[11]);
            s_XImats[150] = static_cast<T>(-s_temp[4]);
            s_XImats[151] = static_cast<T>(-s_temp[11]);
            s_XImats[153] = static_cast<T>(-0.195*s_temp[11]);
            s_XImats[154] = static_cast<T>(0.195*s_temp[4]);
            s_XImats[159] = static_cast<T>(0.02*s_temp[4] + 0.025*s_temp[11]);
            s_XImats[160] = static_cast<T>(-0.025*s_temp[4] + 0.02*s_temp[11]);
            // X[5]
            s_XImats[180] = static_cast<T>(s_temp[12]);
            s_XImats[182] = static_cast<T>(s_temp[5]);
            s_XImats[183] = static_cast<T>(-0.03*s_temp[5]);
            s_XImats[185] = static_cast<T>(0.03*s_temp[12]);
            s_XImats[189] = static_cast<T>(0.19*s_temp[12]);
            s_XImats[191] = static_cast<T>(0.19*s_temp[5]);
            s_XImats[192] = static_cast<T>(-s_temp[5]);
            s_XImats[194] = static_cast<T>(s_temp[12]);
            s_XImats[195] = static_cast<T>(-0.03*s_temp[12]);
            s_XImats[197] = static_cast<T>(-0.03*s_temp[5]);
            // X[6]
            s_XImats[219] = static_cast<T>(-0.11*s_temp[6] + 0.073*s_temp[13]);
            s_XImats[220] = static_cast<T>(-0.073*s_temp[6] - 0.11*s_temp[13]);
            s_XImats[222] = static_cast<T>(s_temp[6]);
            s_XImats[223] = static_cast<T>(s_temp[13]);
            s_XImats[225] = static_cast<T>(0.015*s_temp[13]);
            s_XImats[226] = static_cast<T>(-0.015*s_temp[6]);
            s_XImats[228] = static_cast<T>(s_temp[13]);
            s_XImats[229] = static_cast<T>(-s_temp[6]);
            s_XImats[231] = static_cast<T>(-0.015*s_temp[6]);
            s_XImats[232] = static_cast<T>(-0.015*s_temp[13]);
        }
        __syncthreads();
        for(int kcr = threadIdx.x + threadIdx.y*blockDim.x; kcr < 63; kcr += blockDim.x*blockDim.y){
            int k = kcr / 9; int cr = kcr % 9; int c = cr / 3; int r = cr % 3;
            int srcInd = k*36 + c*6 + r; int dstInd = srcInd + 21; // 3 more rows and cols
            s_XImats[dstInd] = s_XImats[srcInd];
        }
        __syncthreads();
    }

    /**
     * Updates the (d)XmatsHom in (shared) GPU memory acording to the configuration
     *
     * @param s_XmatsHom is the (shared) memory destination location for the XmatsHom
     * @param s_q is the (shared) memory location of the current configuration
     * @param d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)
     * @param s_temp is temporary (shared) memory used to compute sin and cos if needed of size: 14
     */
    template <typename T>
    __device__
    void load_update_XmatsHom_helpers(T *s_XmatsHom, const T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            s_XmatsHom[ind] = d_robotModel->d_XImats[ind+504];
        }
        for(int k = threadIdx.x + threadIdx.y*blockDim.x; k < 7; k += blockDim.x*blockDim.y){
            s_temp[k] = static_cast<T>(sin(s_q[k]));
            s_temp[k+7] = static_cast<T>(cos(s_q[k]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // X_hom[0]
            s_XmatsHom[0] = static_cast<T>(s_temp[7]);
            s_XmatsHom[1] = static_cast<T>(s_temp[0]);
            s_XmatsHom[4] = static_cast<T>(-s_temp[0]);
            s_XmatsHom[5] = static_cast<T>(s_temp[7]);
            // X_hom[1]
            s_XmatsHom[16] = static_cast<T>(s_temp[1]);
            s_XmatsHom[18] = static_cast<T>(s_temp[8]);
            s_XmatsHom[20] = static_cast<T>(s_temp[8]);
            s_XmatsHom[22] = static_cast<T>(-s_temp[1]);
            // X_hom[2]
            s_XmatsHom[33] = static_cast<T>(s_temp[9]);
            s_XmatsHom[34] = static_cast<T>(s_temp[2]);
            s_XmatsHom[37] = static_cast<T>(-s_temp[2]);
            s_XmatsHom[38] = static_cast<T>(s_temp[9]);
            // X_hom[3]
            s_XmatsHom[48] = static_cast<T>(s_temp[10]);
            s_XmatsHom[50] = static_cast<T>(s_temp[3]);
            s_XmatsHom[52] = static_cast<T>(-s_temp[3]);
            s_XmatsHom[54] = static_cast<T>(s_temp[10]);
            // X_hom[4]
            s_XmatsHom[64] = static_cast<T>(s_temp[11]);
            s_XmatsHom[66] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[68] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[70] = static_cast<T>(-s_temp[11]);
            // X_hom[5]
            s_XmatsHom[80] = static_cast<T>(s_temp[5]);
            s_XmatsHom[82] = static_cast<T>(s_temp[12]);
            s_XmatsHom[84] = static_cast<T>(s_temp[12]);
            s_XmatsHom[86] = static_cast<T>(-s_temp[5]);
            // X_hom[6]
            s_XmatsHom[97] = static_cast<T>(s_temp[13]);
            s_XmatsHom[98] = static_cast<T>(s_temp[6]);
            s_XmatsHom[101] = static_cast<T>(-s_temp[6]);
            s_XmatsHom[102] = static_cast<T>(s_temp[13]);
        }
        __syncthreads();
    }

    /**
     * Updates the (d)XmatsHom in (shared) GPU memory acording to the configuration
     *
     * @param s_XmatsHom is the (shared) memory destination location for the XmatsHom
     * @param s_dXmatsHom is the (shared) memory destination location for the dXmatsHom
     * @param s_q is the (shared) memory location of the current configuration
     * @param d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)
     * @param s_temp is temporary (shared) memory used to compute sin and cos if needed of size: 14
     */
    template <typename T>
    __device__
    void load_update_XmatsHom_helpers(T *s_XmatsHom, T *s_dXmatsHom, const T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            s_XmatsHom[ind] = d_robotModel->d_XImats[ind+504];
            s_dXmatsHom[ind] = d_robotModel->d_XImats[ind+616];
        }
        for(int k = threadIdx.x + threadIdx.y*blockDim.x; k < 7; k += blockDim.x*blockDim.y){
            s_temp[k] = static_cast<T>(sin(s_q[k]));
            s_temp[k+7] = static_cast<T>(cos(s_q[k]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // X_hom[0]
            s_XmatsHom[0] = static_cast<T>(s_temp[7]);
            s_XmatsHom[1] = static_cast<T>(s_temp[0]);
            s_XmatsHom[4] = static_cast<T>(-s_temp[0]);
            s_XmatsHom[5] = static_cast<T>(s_temp[7]);
            // X_hom[1]
            s_XmatsHom[16] = static_cast<T>(s_temp[1]);
            s_XmatsHom[18] = static_cast<T>(s_temp[8]);
            s_XmatsHom[20] = static_cast<T>(s_temp[8]);
            s_XmatsHom[22] = static_cast<T>(-s_temp[1]);
            // X_hom[2]
            s_XmatsHom[33] = static_cast<T>(s_temp[9]);
            s_XmatsHom[34] = static_cast<T>(s_temp[2]);
            s_XmatsHom[37] = static_cast<T>(-s_temp[2]);
            s_XmatsHom[38] = static_cast<T>(s_temp[9]);
            // X_hom[3]
            s_XmatsHom[48] = static_cast<T>(s_temp[10]);
            s_XmatsHom[50] = static_cast<T>(s_temp[3]);
            s_XmatsHom[52] = static_cast<T>(-s_temp[3]);
            s_XmatsHom[54] = static_cast<T>(s_temp[10]);
            // X_hom[4]
            s_XmatsHom[64] = static_cast<T>(s_temp[11]);
            s_XmatsHom[66] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[68] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[70] = static_cast<T>(-s_temp[11]);
            // X_hom[5]
            s_XmatsHom[80] = static_cast<T>(s_temp[5]);
            s_XmatsHom[82] = static_cast<T>(s_temp[12]);
            s_XmatsHom[84] = static_cast<T>(s_temp[12]);
            s_XmatsHom[86] = static_cast<T>(-s_temp[5]);
            // X_hom[6]
            s_XmatsHom[97] = static_cast<T>(s_temp[13]);
            s_XmatsHom[98] = static_cast<T>(s_temp[6]);
            s_XmatsHom[101] = static_cast<T>(-s_temp[6]);
            s_XmatsHom[102] = static_cast<T>(s_temp[13]);
            // dX_hom[0]
            s_dXmatsHom[0] = static_cast<T>(-s_temp[0]);
            s_dXmatsHom[1] = static_cast<T>(s_temp[7]);
            s_dXmatsHom[4] = static_cast<T>(-s_temp[7]);
            s_dXmatsHom[5] = static_cast<T>(-s_temp[0]);
            // dX_hom[1]
            s_dXmatsHom[16] = static_cast<T>(s_temp[8]);
            s_dXmatsHom[18] = static_cast<T>(-s_temp[1]);
            s_dXmatsHom[20] = static_cast<T>(-s_temp[1]);
            s_dXmatsHom[22] = static_cast<T>(-s_temp[8]);
            // dX_hom[2]
            s_dXmatsHom[33] = static_cast<T>(-s_temp[2]);
            s_dXmatsHom[34] = static_cast<T>(s_temp[9]);
            s_dXmatsHom[37] = static_cast<T>(-s_temp[9]);
            s_dXmatsHom[38] = static_cast<T>(-s_temp[2]);
            // dX_hom[3]
            s_dXmatsHom[48] = static_cast<T>(-s_temp[3]);
            s_dXmatsHom[50] = static_cast<T>(s_temp[10]);
            s_dXmatsHom[52] = static_cast<T>(-s_temp[10]);
            s_dXmatsHom[54] = static_cast<T>(-s_temp[3]);
            // dX_hom[4]
            s_dXmatsHom[64] = static_cast<T>(-s_temp[4]);
            s_dXmatsHom[66] = static_cast<T>(-s_temp[11]);
            s_dXmatsHom[68] = static_cast<T>(-s_temp[11]);
            s_dXmatsHom[70] = static_cast<T>(s_temp[4]);
            // dX_hom[5]
            s_dXmatsHom[80] = static_cast<T>(s_temp[12]);
            s_dXmatsHom[82] = static_cast<T>(-s_temp[5]);
            s_dXmatsHom[84] = static_cast<T>(-s_temp[5]);
            s_dXmatsHom[86] = static_cast<T>(-s_temp[12]);
            // dX_hom[6]
            s_dXmatsHom[97] = static_cast<T>(-s_temp[6]);
            s_dXmatsHom[98] = static_cast<T>(s_temp[13]);
            s_dXmatsHom[101] = static_cast<T>(-s_temp[13]);
            s_dXmatsHom[102] = static_cast<T>(-s_temp[6]);
        }
        __syncthreads();
    }

    /**
     * Updates the (d)XmatsHom in (shared) GPU memory acording to the configuration
     *
     * @param s_XmatsHom is the (shared) memory destination location for the XmatsHom
     * @param s_d2XmatsHom is the (shared) memory destination location for the d2XmatsHom
     * @param s_dXmatsHom is the (shared) memory destination location for the dXmatsHom
     * @param s_q is the (shared) memory location of the current configuration
     * @param d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)
     * @param s_temp is temporary (shared) memory used to compute sin and cos if needed of size: 14
     */
    template <typename T>
    __device__
    void load_update_XmatsHom_helpers(T *s_XmatsHom, T *s_dXmatsHom, T *s_d2XmatsHom, const T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            s_XmatsHom[ind] = d_robotModel->d_XImats[ind+504];
            s_dXmatsHom[ind] = d_robotModel->d_XImats[ind+616];
            s_d2XmatsHom[ind] = d_robotModel->d_XImats[ind+728];
        }
        for(int k = threadIdx.x + threadIdx.y*blockDim.x; k < 7; k += blockDim.x*blockDim.y){
            s_temp[k] = static_cast<T>(sin(s_q[k]));
            s_temp[k+7] = static_cast<T>(cos(s_q[k]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // X_hom[0]
            s_XmatsHom[0] = static_cast<T>(s_temp[7]);
            s_XmatsHom[1] = static_cast<T>(s_temp[0]);
            s_XmatsHom[4] = static_cast<T>(-s_temp[0]);
            s_XmatsHom[5] = static_cast<T>(s_temp[7]);
            // X_hom[1]
            s_XmatsHom[16] = static_cast<T>(s_temp[1]);
            s_XmatsHom[18] = static_cast<T>(s_temp[8]);
            s_XmatsHom[20] = static_cast<T>(s_temp[8]);
            s_XmatsHom[22] = static_cast<T>(-s_temp[1]);
            // X_hom[2]
            s_XmatsHom[33] = static_cast<T>(s_temp[9]);
            s_XmatsHom[34] = static_cast<T>(s_temp[2]);
            s_XmatsHom[37] = static_cast<T>(-s_temp[2]);
            s_XmatsHom[38] = static_cast<T>(s_temp[9]);
            // X_hom[3]
            s_XmatsHom[48] = static_cast<T>(s_temp[10]);
            s_XmatsHom[50] = static_cast<T>(s_temp[3]);
            s_XmatsHom[52] = static_cast<T>(-s_temp[3]);
            s_XmatsHom[54] = static_cast<T>(s_temp[10]);
            // X_hom[4]
            s_XmatsHom[64] = static_cast<T>(s_temp[11]);
            s_XmatsHom[66] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[68] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[70] = static_cast<T>(-s_temp[11]);
            // X_hom[5]
            s_XmatsHom[80] = static_cast<T>(s_temp[5]);
            s_XmatsHom[82] = static_cast<T>(s_temp[12]);
            s_XmatsHom[84] = static_cast<T>(s_temp[12]);
            s_XmatsHom[86] = static_cast<T>(-s_temp[5]);
            // X_hom[6]
            s_XmatsHom[97] = static_cast<T>(s_temp[13]);
            s_XmatsHom[98] = static_cast<T>(s_temp[6]);
            s_XmatsHom[101] = static_cast<T>(-s_temp[6]);
            s_XmatsHom[102] = static_cast<T>(s_temp[13]);
            // dX_hom[0]
            s_dXmatsHom[0] = static_cast<T>(-s_temp[0]);
            s_dXmatsHom[1] = static_cast<T>(s_temp[7]);
            s_dXmatsHom[4] = static_cast<T>(-s_temp[7]);
            s_dXmatsHom[5] = static_cast<T>(-s_temp[0]);
            // dX_hom[1]
            s_dXmatsHom[16] = static_cast<T>(s_temp[8]);
            s_dXmatsHom[18] = static_cast<T>(-s_temp[1]);
            s_dXmatsHom[20] = static_cast<T>(-s_temp[1]);
            s_dXmatsHom[22] = static_cast<T>(-s_temp[8]);
            // dX_hom[2]
            s_dXmatsHom[33] = static_cast<T>(-s_temp[2]);
            s_dXmatsHom[34] = static_cast<T>(s_temp[9]);
            s_dXmatsHom[37] = static_cast<T>(-s_temp[9]);
            s_dXmatsHom[38] = static_cast<T>(-s_temp[2]);
            // dX_hom[3]
            s_dXmatsHom[48] = static_cast<T>(-s_temp[3]);
            s_dXmatsHom[50] = static_cast<T>(s_temp[10]);
            s_dXmatsHom[52] = static_cast<T>(-s_temp[10]);
            s_dXmatsHom[54] = static_cast<T>(-s_temp[3]);
            // dX_hom[4]
            s_dXmatsHom[64] = static_cast<T>(-s_temp[4]);
            s_dXmatsHom[66] = static_cast<T>(-s_temp[11]);
            s_dXmatsHom[68] = static_cast<T>(-s_temp[11]);
            s_dXmatsHom[70] = static_cast<T>(s_temp[4]);
            // dX_hom[5]
            s_dXmatsHom[80] = static_cast<T>(s_temp[12]);
            s_dXmatsHom[82] = static_cast<T>(-s_temp[5]);
            s_dXmatsHom[84] = static_cast<T>(-s_temp[5]);
            s_dXmatsHom[86] = static_cast<T>(-s_temp[12]);
            // dX_hom[6]
            s_dXmatsHom[97] = static_cast<T>(-s_temp[6]);
            s_dXmatsHom[98] = static_cast<T>(s_temp[13]);
            s_dXmatsHom[101] = static_cast<T>(-s_temp[13]);
            s_dXmatsHom[102] = static_cast<T>(-s_temp[6]);
            // d2X_hom[0]
            s_d2XmatsHom[0] = static_cast<T>(-s_temp[7]);
            s_d2XmatsHom[1] = static_cast<T>(-s_temp[0]);
            s_d2XmatsHom[4] = static_cast<T>(s_temp[0]);
            s_d2XmatsHom[5] = static_cast<T>(-s_temp[7]);
            // d2X_hom[1]
            s_d2XmatsHom[16] = static_cast<T>(-s_temp[1]);
            s_d2XmatsHom[18] = static_cast<T>(-s_temp[8]);
            s_d2XmatsHom[20] = static_cast<T>(-s_temp[8]);
            s_d2XmatsHom[22] = static_cast<T>(s_temp[1]);
            // d2X_hom[2]
            s_d2XmatsHom[33] = static_cast<T>(-s_temp[9]);
            s_d2XmatsHom[34] = static_cast<T>(-s_temp[2]);
            s_d2XmatsHom[37] = static_cast<T>(s_temp[2]);
            s_d2XmatsHom[38] = static_cast<T>(-s_temp[9]);
            // d2X_hom[3]
            s_d2XmatsHom[48] = static_cast<T>(-s_temp[10]);
            s_d2XmatsHom[50] = static_cast<T>(-s_temp[3]);
            s_d2XmatsHom[52] = static_cast<T>(s_temp[3]);
            s_d2XmatsHom[54] = static_cast<T>(-s_temp[10]);
            // d2X_hom[4]
            s_d2XmatsHom[64] = static_cast<T>(-s_temp[11]);
            s_d2XmatsHom[66] = static_cast<T>(s_temp[4]);
            s_d2XmatsHom[68] = static_cast<T>(s_temp[4]);
            s_d2XmatsHom[70] = static_cast<T>(s_temp[11]);
            // d2X_hom[5]
            s_d2XmatsHom[80] = static_cast<T>(-s_temp[5]);
            s_d2XmatsHom[82] = static_cast<T>(-s_temp[12]);
            s_d2XmatsHom[84] = static_cast<T>(-s_temp[12]);
            s_d2XmatsHom[86] = static_cast<T>(s_temp[5]);
            // d2X_hom[6]
            s_d2XmatsHom[97] = static_cast<T>(-s_temp[13]);
            s_d2XmatsHom[98] = static_cast<T>(-s_temp[6]);
            s_d2XmatsHom[101] = static_cast<T>(s_temp[6]);
            s_d2XmatsHom[102] = static_cast<T>(-s_temp[13]);
        }
        __syncthreads();
    }

    /**
     * Computes the End Effector Position
     *
     * Notes:
     *   Assumes the Xhom matricies have already been updated for the given q
     *
     * @param s_eePos is a pointer to shared memory of size 6*NUM_EE where NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_Xhom is the pointer to the homogenous transformation matricies 
     * @param s_temp is a pointer to helper shared memory of size 32
     */
    template <typename T>
    __device__
    void end_effector_pose_inner(T *s_eePos, const T *s_q, const T *s_Xhom, T *s_temp) {
        //
        // For each branch in parallel chain up the transform
        // Keep chaining until reaching the root (starting from the leaves)
        //
        // Serial chain manipulator so optimize as parent is jid-1
        // First set to leaf transform
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            s_temp[ind] = s_Xhom[16*6 + ind];
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 1/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*5 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 2/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*4 + row], &s_temp[16 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 3/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*3 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 4/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*2 + row], &s_temp[16 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 5/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*1 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 6/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*0 + row], &s_temp[16 + 4*col]);
        }
        __syncthreads();
        //
        // Now extract the eePos from the Tansforms
        // TODO: ADD OFFSETS
        //
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 3; ind += blockDim.x*blockDim.y){
            // xyz is easy
            int xyzInd = ind % 3; int eeInd = ind / 3; T *s_Xmat_hom = &s_temp[0 + 16*eeInd];
            s_eePos[6*eeInd + xyzInd] = s_Xmat_hom[12 + xyzInd];
            // roll pitch yaw is a bit more difficult
            if(xyzInd > 0){continue;}
            s_eePos[6*eeInd + 3] = atan2(s_Xmat_hom[6],s_Xmat_hom[10]);
            s_eePos[6*eeInd + 4] = -atan2(s_Xmat_hom[2],sqrt(s_Xmat_hom[6]*s_Xmat_hom[6] + s_Xmat_hom[10]*s_Xmat_hom[10]));
            s_eePos[6*eeInd + 5] = atan2(s_Xmat_hom[1],s_Xmat_hom[0]);
        }
        __syncthreads();
    }

    /**
     * Computes the End Effector Position
     *
     * @param s_eePos is a pointer to shared memory of size 6*NUM_EE where NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_pose_device(T *s_eePos, const T *s_q, const robotModel<T> *d_robotModel) {
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_temp = &s_XHomTemp[112];
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_q, d_robotModel, s_temp);
        end_effector_pose_inner<T>(s_eePos, s_q, s_XmatsHom, s_temp);
    }

    /**
     * Computes the End Effector Position
     *
     * @param s_eePos is a pointer to shared memory of size 6*NUM_EE where NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_temp_in is the pointer to the temporary shared memory
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_pose_device(T *s_eePos, const T *s_q, T* s_temp_in, const robotModel<T> *d_robotModel) {
        T* s_XHomTemp = s_temp_in;
        T* s_XmatsHom = s_XHomTemp;
        T* s_temp = &s_XHomTemp[112];
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_q, d_robotModel, s_temp);
        end_effector_pose_inner<T>(s_eePos, s_q, s_XmatsHom, s_temp);
    }


    /**
     * Compute the End Effector Position
     *
     * @param d_eePos is the vector of end effector positions
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_pose_kernel_single_timing(T *d_eePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_eePos[6];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_temp = &s_XHomTemp[112];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_q[ind] = d_q[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_q, d_robotModel, s_temp);
            end_effector_pose_inner<T>(s_eePos, s_q, s_XmatsHom, s_temp);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            d_eePos[ind] = s_eePos[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the End Effector Position
     *
     * @param d_eePos is the vector of end effector positions
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_pose_kernel(T *d_eePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_eePos[6];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_temp = &s_XHomTemp[112];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_q, d_robotModel, s_temp);
            end_effector_pose_inner<T>(s_eePos, s_q, s_XmatsHom, s_temp);
            __syncthreads();
            // save down to global
            T *d_eePos_k = &d_eePos[k*6];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
                d_eePos_k[ind] = s_eePos[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the End Effector Pose
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_COMPRESSED_MEM) {end_effector_pose_kernel<T><<<block_dimms,thread_dimms,EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_kernel<T><<<block_dimms,thread_dimms,EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_eePos,hd_data->d_eePos,6*NUM_EES*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the End Effector Pose
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                              const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_COMPRESSED_MEM) {end_effector_pose_kernel_single_timing<T><<<block_dimms,thread_dimms,EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_kernel_single_timing<T><<<block_dimms,thread_dimms,EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_eePos,hd_data->d_eePos,6*NUM_EES*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call EEPOS %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the End Effector Pose
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                             const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_COMPRESSED_MEM) {end_effector_pose_kernel<T><<<block_dimms,thread_dimms,EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_kernel<T><<<block_dimms,thread_dimms,EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * Notes:
     *   Assumes the Xhom and dXhom matricies have already been updated for the given q
     *
     * @param s_deePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_Xhom is the pointer to the homogenous transformation matricies 
     * @param s_dXhom is the pointer to the gradient of the homogenous transformation matricies 
     * @param s_temp is a pointer to helper shared memory of size 448
     */
    template <typename T>
    __device__
    void end_effector_pose_gradient_inner(T *s_deePos, const T *s_q, const T *s_Xhom, const T *s_dXhom, T *s_temp) {
        //
        // For each branch/gradient in parallel chain up the transform
        // Keep chaining until reaching the root (starting from the leaves)
        //
        T *s_eeTemp = &s_temp[0]; T *s_deeTemp = &s_temp[224];
        // Serial chain manipulator so optimize as parent is jid-1
        // First set the leaf transforms for eePos and deePos
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int eeIndStart = 16*6;
            s_eeTemp[ind] = s_Xhom[eeIndStart + rc];
            s_deeTemp[ind] = (djid == 6) ? s_dXhom[eeIndStart + rc] : s_Xhom[eeIndStart + rc];
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 1/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            s_eeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom[16*5 + row], &s_eeTemp[0 + colInd]);
            const T *s_Xhom_dXhom = ((djid == 5) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*5 + row], &s_deeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 2/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            s_eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*4 + row], &s_eeTemp[112 + colInd]);
            const T *s_Xhom_dXhom = ((djid == 4) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*4 + row], &s_deeTemp[112 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 3/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            s_eeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom[16*3 + row], &s_eeTemp[0 + colInd]);
            const T *s_Xhom_dXhom = ((djid == 3) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*3 + row], &s_deeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 4/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            s_eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*2 + row], &s_eeTemp[112 + colInd]);
            const T *s_Xhom_dXhom = ((djid == 2) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*2 + row], &s_deeTemp[112 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 5/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            s_eeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom[16*1 + row], &s_eeTemp[0 + colInd]);
            const T *s_Xhom_dXhom = ((djid == 1) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*1 + row], &s_deeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 6/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            s_eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*0 + row], &s_eeTemp[112 + colInd]);
            const T *s_Xhom_dXhom = ((djid == 0) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*0 + row], &s_deeTemp[112 + colInd]);
        }
        __syncthreads();
        //
        // Now extract the eePos from the Tansforms
        // TODO: ADD OFFSETS
        //
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int outputInd = ind % 6; int deeInd = ind / 6;
            T *s_Xmat_hom = &s_eeTemp[0 + 16*deeInd]; T *s_dXmat_hom = &s_deeTemp[0 + 16*deeInd];
            // xyz is easy
            if (outputInd < 3){s_deePos[6*deeInd + outputInd] = s_dXmat_hom[12 + outputInd];}
            // roll pitch yaw is a bit more difficult
            // note: d/dz of arctan2(y(z),x(z)) = [-x'(z)y(z)+x(z)y'(z)]/[(x(z)^2 + y(z)^2)]
            // Also note that d/dz of sqrt(f(z)) = f'(z)/2sqrt(f(z))
            else {
                // simpler to recompute
                T sqrtTerm = sqrt(s_Xmat_hom[10]*s_Xmat_hom[10] + s_Xmat_hom[6]*s_Xmat_hom[6]);
                T dsqrtTerm = (s_Xmat_hom[10]*s_dXmat_hom[10] + s_Xmat_hom[6]*s_dXmat_hom[6])/sqrtTerm;
                // branch to get pointer locations
                T y; T x; T y_prime; T x_prime;
                     if (outputInd == 3){ y = s_Xmat_hom[6]; x = s_Xmat_hom[10]; y_prime = s_dXmat_hom[6]; x_prime = s_dXmat_hom[10]; }
                else if (outputInd == 4){ y = -s_Xmat_hom[2]; x = sqrtTerm; y_prime = -s_dXmat_hom[2]; x_prime = dsqrtTerm; }
                else              { y = s_Xmat_hom[1]; x = s_Xmat_hom[0]; y_prime = s_dXmat_hom[1]; x_prime = s_dXmat_hom[0]; }
                s_deePos[6*deeInd + outputInd] = (-x_prime*y + x*y_prime)/(x*x + y*y);
            }
        }
        __syncthreads();
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param s_deePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_temp_in is the pointer to the temporary shared memory
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_pose_gradient_device(T *s_deePos, const T *s_q, T* s_temp_in, const robotModel<T> *d_robotModel) {
        T* s_XHomTemp = s_temp_in; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_temp = &s_dXmatsHom[112];
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_q, d_robotModel, s_temp);
        end_effector_pose_gradient_inner<T>(s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_temp);
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param d_deePos is the vector of end effector positions gradients
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_pose_gradient_kernel_single_timing(T *d_deePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_deePos[42];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_temp = &s_dXmatsHom[112];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_q[ind] = d_q[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_q, d_robotModel, s_temp);
            end_effector_pose_gradient_inner<T>(s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_temp);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            d_deePos[ind] = s_deePos[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param d_deePos is the vector of end effector positions gradients
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_pose_gradient_kernel(T *d_deePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_deePos[42];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_temp = &s_dXmatsHom[112];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_q, d_robotModel, s_temp);
            end_effector_pose_gradient_inner<T>(s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_temp);
            __syncthreads();
            // save down to global
            T *d_deePos_k = &d_deePos[k*42];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                d_deePos_k[ind] = s_deePos[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_gradient(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_kernel<T><<<block_dimms,thread_dimms,DEE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_kernel<T><<<block_dimms,thread_dimms,DEE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_deePos,hd_data->d_deePos,6*NUM_EES*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_gradient_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                              const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,DEE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,DEE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_deePos,hd_data->d_deePos,6*NUM_EES*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call DEEPOS %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_gradient_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                             const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_kernel<T><<<block_dimms,thread_dimms,DEE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_kernel<T><<<block_dimms,thread_dimms,DEE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * Notes:
     *   Assumes the Xhom and dXhom matricies have already been updated for the given q
     *
     * @param s_d2eePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_deePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_Xhom is the pointer to the homogenous transformation matricies 
     * @param s_dXhom is the pointer to the 1st derivative of the homogenous transformation matricies 
     * @param s_d2Xhom is the pointer to the 2nd derivative of the homogenous transformation matricies 
     * @param s_temp is a pointer to helper shared memory of size 448
     */
    template <typename T>
    __device__
    void end_effector_pose_gradient_hessian_inner(T *s_d2eePos, T *s_deePos, const T *s_q, const T *s_Xhom, const T *s_dXhom, const T *s_d2Xhom, T *s_temp) {
        //
        // For each branch/gradient in parallel chain up the transform
        // Keep chaining until reaching the root (starting from the leaves)
        // Keep all gradients (and the value) as we need them for the hessian
        //
        T *s_eeTemp = &s_temp[0]; T *s_deeTemp = &s_temp[32];
        // Serial chain manipulator so optimize as parent is jid-1
        // First set the leaf transforms for eePos and deePos
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int eeIndStart = 16*6;
            if(djid == 0){s_eeTemp[ind] = s_Xhom[eeIndStart + rc];}
            s_deeTemp[ind] = (djid == 6) ? s_dXhom[eeIndStart + rc] : s_Xhom[eeIndStart + rc];
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 1/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            if(djid == 0){s_eeTemp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*5 + row], &s_eeTemp[0 + colInd]);}
            const T *s_Xhom_dXhom = ((djid == 5) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*5 + row], &s_deeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 2/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            if(djid == 0){s_eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*4 + row], &s_eeTemp[16 + colInd]);}
            const T *s_Xhom_dXhom = ((djid == 4) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*4 + row], &s_deeTemp[112 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 3/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            if(djid == 0){s_eeTemp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*3 + row], &s_eeTemp[0 + colInd]);}
            const T *s_Xhom_dXhom = ((djid == 3) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*3 + row], &s_deeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 4/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            if(djid == 0){s_eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*2 + row], &s_eeTemp[16 + colInd]);}
            const T *s_Xhom_dXhom = ((djid == 2) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*2 + row], &s_deeTemp[112 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 5/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            if(djid == 0){s_eeTemp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*1 + row], &s_eeTemp[0 + colInd]);}
            const T *s_Xhom_dXhom = ((djid == 1) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*1 + row], &s_deeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 6/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            if(djid == 0){s_eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*0 + row], &s_eeTemp[16 + colInd]);}
            const T *s_Xhom_dXhom = ((djid == 0) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*0 + row], &s_deeTemp[112 + colInd]);
        }
        __syncthreads();
        int eeTemp_Offset = 0;
        int deeTemp_Offset = 0;
        //
        // For each hessian term in parallel chain up the transform
        // Keep chaining until reaching the root (starting from the leaves)
        //
        T *s_d2eeTemp = &s_deeTemp[224];
        // set eeTemp and deeTemp to the right offsets
        s_eeTemp = &s_eeTemp[eeTemp_Offset];
        s_deeTemp = &s_deeTemp[deeTemp_Offset];
        // Serial chain manipulator so optimize as parent is jid-1
        // First set the leaf transforms
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int eeIndStart = 16*6;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 6)) ? s_d2Xhom : (((djid_i == 6) || (djid_j == 6)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind] = s_Xhom_dXhom_d2Xhom[eeIndStart + rc];
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 1/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 5)) ? s_d2Xhom : (((djid_i == 5) || (djid_j == 5)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind + 784] = dot_prod<T,4,4,1>(&s_Xhom_dXhom_d2Xhom[16*5 + row], &s_d2eeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 2/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 4)) ? s_d2Xhom : (((djid_i == 4) || (djid_j == 4)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom_d2Xhom[16*4 + row], &s_d2eeTemp[784 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 3/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 3)) ? s_d2Xhom : (((djid_i == 3) || (djid_j == 3)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind + 784] = dot_prod<T,4,4,1>(&s_Xhom_dXhom_d2Xhom[16*3 + row], &s_d2eeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 4/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 2)) ? s_d2Xhom : (((djid_i == 2) || (djid_j == 2)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom_d2Xhom[16*2 + row], &s_d2eeTemp[784 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 5/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 1)) ? s_d2Xhom : (((djid_i == 1) || (djid_j == 1)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind + 784] = dot_prod<T,4,4,1>(&s_Xhom_dXhom_d2Xhom[16*1 + row], &s_d2eeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 6/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 0)) ? s_d2Xhom : (((djid_i == 0) || (djid_j == 0)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom_d2Xhom[16*0 + row], &s_d2eeTemp[784 + colInd]);
        }
        __syncthreads();
        int d2eeTemp_Offset = 0;
        //
        // Finally Extract the eePos from the Tansforms
        // TODO: ADD OFFSETS
        //
        // set d2eeTemp to the right offsets
        s_d2eeTemp = &s_d2eeTemp[d2eeTemp_Offset];
        // For all n*n*num_ee in parallel
        for(int ind6 = threadIdx.x + threadIdx.y*blockDim.x; ind6 < 294; ind6 += blockDim.x*blockDim.y){
            int ind = ind6 / 6; int outputInd = ind6 % 6;
            int curr_ee = ind / 49; int djid_ij = ind % 49; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7;
            int ee_offset = 16 * curr_ee; int dee_offset = ee_offset * 7; int d2ee_offset = dee_offset * 7;
            T *s_Xmat_hom = &s_eeTemp[ee_offset]; T *s_d2Xmat_hom_ij = &s_d2eeTemp[d2ee_offset + 16 * djid_i * 7 + 16 * djid_j];
            T *s_dXmat_hom_i = &s_deeTemp[dee_offset + 16 * djid_i]; T *s_dXmat_hom_j = &s_deeTemp[dee_offset + 16 * djid_j];
            T *s_deePos_i = &s_deePos[6*djid_i]; int d2eePosOffset_ij = djid_i * 7 + djid_j; int d2eePosOffset_ind = 49;
            // Note: djid_j == 0 computes gradient too
            // xyz is easy
            if (outputInd < 3){
                if (djid_j == 0){s_deePos[outputInd + 6*djid_i] = s_dXmat_hom_i[12 + outputInd];}
                s_d2eePos[outputInd*d2eePosOffset_ind + d2eePosOffset_ij] = s_d2Xmat_hom_ij[12 + outputInd];
            }
            // roll pitch yaw is a bit more difficult
            // note: d/dz of arctan2(y(z),x(z)) = [-x'(z)y(z)+x(z)y'(z)]/[(x(z)^2 + y(z)^2)]
            //       d/dz of sqrt(f(z)) = f'(z)/2sqrt(f(z))
            //       d2/dz of arctan2(y(z),x(z)) is (bottom*dtop - top*dbottom) / (bottom*bottom) of:
            //          top = -x_prime_i*y + x*y_prime_i
            //          dtop = -x_prime_prime*y + x*y_prime_prime + (i != j)*(-x_prime_i*y_prime_j + x_prime_j*y_prime_i)
            //          bottom = x*x + y*y
            //          dbottom = 2*x*x_prime_j + 2*y*y_prime_j
            else {
                T pitchSqrtTerm = sqrt(s_Xmat_hom[10]*s_Xmat_hom[10] + s_Xmat_hom[6]*s_Xmat_hom[6]);
                T dpitchSqrtTerm_i = (s_Xmat_hom[10]*s_dXmat_hom_i[10] + s_Xmat_hom[6]*s_dXmat_hom_i[6])/pitchSqrtTerm;
                T dpitchSqrtTerm_j = (s_Xmat_hom[10]*s_dXmat_hom_j[10] + s_Xmat_hom[6]*s_dXmat_hom_j[6])/pitchSqrtTerm;
                T dpitchSqrtTerm_i_top_dj = s_dXmat_hom_j[10]*s_dXmat_hom_i[10] + s_Xmat_hom[10]*s_d2Xmat_hom_ij[10] + 
                                            s_dXmat_hom_j[6]*s_dXmat_hom_i[6] + s_Xmat_hom[6]*s_d2Xmat_hom_ij[6];
                T d2pitchSqrtTerm_ij = (pitchSqrtTerm*dpitchSqrtTerm_i_top_dj - dpitchSqrtTerm_i*dpitchSqrtTerm_j) / (pitchSqrtTerm*pitchSqrtTerm);
                // branch to get pointer locations
                T y; T x; T y_prime_i; T x_prime_i; T y_prime_j; T x_prime_j; T y_dprime_ij; T x_dprime_ij;
                     if (outputInd == 3){ y = s_Xmat_hom[6]; x = s_Xmat_hom[10]; y_prime_i = s_dXmat_hom_i[6]; x_prime_i = s_dXmat_hom_i[10]; y_prime_j = s_dXmat_hom_j[6]; x_prime_j = s_dXmat_hom_j[10]; y_dprime_ij = s_d2Xmat_hom_ij[6]; x_dprime_ij = s_d2Xmat_hom_ij[10]; }
                else if (outputInd == 4){ y = -s_Xmat_hom[2]; x = pitchSqrtTerm; y_prime_i = -s_dXmat_hom_i[2]; x_prime_i = dpitchSqrtTerm_i; y_prime_j = -s_dXmat_hom_j[2]; x_prime_j = dpitchSqrtTerm_j; y_dprime_ij = -s_d2Xmat_hom_ij[2]; x_dprime_ij = d2pitchSqrtTerm_ij; }
                else              { y = s_Xmat_hom[1]; x = s_Xmat_hom[0]; y_prime_i = s_dXmat_hom_i[1]; x_prime_i = s_dXmat_hom_i[0]; y_prime_j = s_dXmat_hom_j[1]; x_prime_j = s_dXmat_hom_j[0]; y_dprime_ij = s_d2Xmat_hom_ij[1]; x_dprime_ij = s_d2Xmat_hom_ij[0]; }
                T top = -x_prime_i*y + x*y_prime_i;     T bottom = x*x + y*y;
                T dtop = -x_dprime_ij*y + x*y_dprime_ij + (djid_i != djid_j)*(-x_prime_i*y_prime_j + x_prime_j*y_prime_i);
                T dbottom = 2*x*x_prime_j + 2*y*y_prime_j;
                if (djid_j == 0){s_deePos[outputInd + 6*djid_i] = top/bottom;}
                s_d2eePos[outputInd*d2eePosOffset_ind + d2eePosOffset_ij] = (bottom*dtop - top*dbottom) / (bottom*bottom);
            }
        }
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * @param s_d2eePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_deePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_pose_gradient_hessian_device(T *s_d2eePos, T *s_deePos, const T *s_q, const robotModel<T> *d_robotModel) {
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_d2XmatsHom = &s_dXmatsHom[112]; T *s_temp = &s_d2XmatsHom[112];
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_d2XmatsHom, s_q, d_robotModel, s_temp);
        end_effector_pose_gradient_hessian_inner<T>(s_d2eePos, s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_d2XmatsHom, s_temp);
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * @param d_d2eePos is the vector of end effector positions gradients
     * @param d_deePos is the vector of end effector positions gradients
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_pose_gradient_hessian_kernel_single_timing(T *d_d2eePos, T *d_deePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_d2eePos[294];
        __shared__ T s_deePos[42];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_d2XmatsHom = &s_dXmatsHom[112]; T *s_temp = &s_d2XmatsHom[112];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_q[ind] = d_q[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_d2XmatsHom, s_q, d_robotModel, s_temp);
            end_effector_pose_gradient_hessian_inner<T>(s_d2eePos, s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_d2XmatsHom, s_temp);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 294; ind += blockDim.x*blockDim.y){
            d_d2eePos[ind] = s_d2eePos[ind];
        }
        __syncthreads();
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            d_deePos[ind] = s_deePos[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * @param d_d2eePos is the vector of end effector positions gradients
     * @param d_deePos is the vector of end effector positions gradients
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_pose_gradient_hessian_kernel(T *d_d2eePos, T *d_deePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_d2eePos[294];
        __shared__ T s_deePos[42];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_d2XmatsHom = &s_dXmatsHom[112]; T *s_temp = &s_d2XmatsHom[112];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_d2XmatsHom, s_q, d_robotModel, s_temp);
            end_effector_pose_gradient_hessian_inner<T>(s_d2eePos, s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_d2XmatsHom, s_temp);
            __syncthreads();
            // save down to global
            T *d_d2eePos_k = &d_d2eePos[k*294];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 294; ind += blockDim.x*blockDim.y){
                d_d2eePos_k[ind] = s_d2eePos[ind];
            }
            __syncthreads();
            // save down to global
            T *d_deePos_k = &d_deePos[k*42];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                d_deePos_k[ind] = s_deePos[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_gradient_hessian(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_hessian_kernel<T><<<block_dimms,thread_dimms,D2EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_d2eePos,hd_data->d_deePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_hessian_kernel<T><<<block_dimms,thread_dimms,D2EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_d2eePos,hd_data->d_deePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_deePos,hd_data->d_deePos,6*NUM_EES*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(hd_data->h_d2eePos,hd_data->d_d2eePos,6*NUM_EES*NUM_JOINTS*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_gradient_hessian_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                              const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_hessian_kernel_single_timing<T><<<block_dimms,thread_dimms,D2EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_d2eePos,hd_data->d_deePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_hessian_kernel_single_timing<T><<<block_dimms,thread_dimms,D2EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_d2eePos,hd_data->d_deePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_deePos,hd_data->d_deePos,6*NUM_EES*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(hd_data->h_d2eePos,hd_data->d_d2eePos,6*NUM_EES*NUM_JOINTS*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call DEEPOS %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_gradient_hessian_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                             const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_hessian_kernel<T><<<block_dimms,thread_dimms,D2EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_d2eePos,hd_data->d_deePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_hessian_kernel<T><<<block_dimms,thread_dimms,D2EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_d2eePos,hd_data->d_deePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *
     * @param s_c is the vector of output torques
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 126
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is (optional vector of joint accelerations
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 42
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: joint1
        //     links are: link1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravityS[k]*qdd[k]
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0]; s_vaf[42 + jid6 + 2] += s_qdd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: joint2
        //     links are: link2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 1;
            T qd_qdd_val = (row == 1) * (vFlag * s_qd[1] + !vFlag * s_qdd[1]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*0]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx1_peq_scaled<T>(&s_vaf[48], &s_vaf[6], s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: joint3
        //     links are: link3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 2;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[2] + !vFlag * s_qdd[2]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*1]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[12], s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: joint4
        //     links are: link4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 3;
            T qd_qdd_val = (row == 1) * (vFlag * s_qd[3] + !vFlag * s_qdd[3]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*2]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx1_peq_scaled<T>(&s_vaf[60], &s_vaf[18], s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: joint5
        //     links are: link5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 4;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[4] + !vFlag * s_qdd[4]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*3]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[24], s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: joint6
        //     links are: link6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 5;
            T qd_qdd_val = (row == 1) * (vFlag * s_qd[5] + !vFlag * s_qdd[5]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*4]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx1_peq_scaled<T>(&s_vaf[72], &s_vaf[30], s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: joint7
        //     links are: link7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 6;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[6] + !vFlag * s_qdd[6]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*5]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[78], &s_vaf[36], s_qd[6]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 42 + jid6;
            T *dst = IaFlag ? &s_vaf[84] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[252 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[84 + jid6], &s_vaf[jid6], &s_temp[jid6]);
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: joint7
        //     links are: link7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: joint6
        //     links are: link6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: joint5
        //     links are: link5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: joint4
        //     links are: link4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: joint3
        //     links are: link3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: joint2
        //     links are: link2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row], &s_vaf[84 + 6*1]);
            int dstOffset = 84 + 6*0 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        //
        // s_c extracted in parallel (S*f)
        //
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            s_c[jid] = s_vaf[84 + 6*jid + s_topology_helpers[jid]];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   optimized for qdd = 0
     *
     * @param s_c is the vector of output torques
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 126
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 42
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity, T *d_f_ext) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: joint1
        //     links are: link1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: joint2
        //     links are: link2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 1;
            T qd_qdd_val = (row == 1) * (vFlag * s_qd[1]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*0]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx1_peq_scaled<T>(&s_vaf[48], &s_vaf[6], s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: joint3
        //     links are: link3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 2;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[2]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*1]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[12], s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: joint4
        //     links are: link4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 3;
            T qd_qdd_val = (row == 1) * (vFlag * s_qd[3]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*2]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx1_peq_scaled<T>(&s_vaf[60], &s_vaf[18], s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: joint5
        //     links are: link5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 4;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[4]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*3]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[24], s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: joint6
        //     links are: link6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 5;
            T qd_qdd_val = (row == 1) * (vFlag * s_qd[5]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*4]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx1_peq_scaled<T>(&s_vaf[72], &s_vaf[30], s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: joint7
        //     links are: link7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 6;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[6]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*5]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[78], &s_vaf[36], s_qd[6]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 42 + jid6;
            T *dst = IaFlag ? &s_vaf[84] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[252 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[84 + jid6], &s_vaf[jid6], &s_temp[jid6]);

            if (jid == 6 && d_f_ext != nullptr) {
                // Add external forces if provided
                for (int row = 0; row < 6; ++row) {
                    s_vaf[84 + jid6 + row] -= d_f_ext[row];
                }
            }
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: joint7
        //     links are: link7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: joint6
        //     links are: link6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: joint5
        //     links are: link5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: joint4
        //     links are: link4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: joint3
        //     links are: link3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: joint2
        //     links are: link2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row], &s_vaf[84 + 6*1]);
            int dstOffset = 84 + 6*0 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        //
        // s_c extracted in parallel (S*f)
        //
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            s_c[jid] = s_vaf[84 + 6*jid + s_topology_helpers[jid]];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   used to compute vaf as helper values
     *
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 126
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is (optional vector of joint accelerations
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 42
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner_vaf(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity, T *d_f_ext) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: joint1
        //     links are: link1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravityS[k]*qdd[k]
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0]; s_vaf[42 + jid6 + 2] += s_qdd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: joint2
        //     links are: link2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 1;
            T qd_qdd_val = (row == 1) * (vFlag * s_qd[1] + !vFlag * s_qdd[1]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*0]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx1_peq_scaled<T>(&s_vaf[48], &s_vaf[6], s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: joint3
        //     links are: link3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 2;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[2] + !vFlag * s_qdd[2]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*1]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[12], s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: joint4
        //     links are: link4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 3;
            T qd_qdd_val = (row == 1) * (vFlag * s_qd[3] + !vFlag * s_qdd[3]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*2]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx1_peq_scaled<T>(&s_vaf[60], &s_vaf[18], s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: joint5
        //     links are: link5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 4;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[4] + !vFlag * s_qdd[4]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*3]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[24], s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: joint6
        //     links are: link6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 5;
            T qd_qdd_val = (row == 1) * (vFlag * s_qd[5] + !vFlag * s_qdd[5]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*4]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx1_peq_scaled<T>(&s_vaf[72], &s_vaf[30], s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: joint7
        //     links are: link7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 6;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[6] + !vFlag * s_qdd[6]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*5]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[78], &s_vaf[36], s_qd[6]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 42 + jid6;
            T *dst = IaFlag ? &s_vaf[84] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[252 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[84 + jid6], &s_vaf[jid6], &s_temp[jid6]);

            if (jid == 6 && d_f_ext != nullptr) {
                // Add external forces if provided
                for (int row = 0; row < 6; ++row) {
                    s_vaf[84 + jid6 + row] -= d_f_ext[row];
                }
            }

        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: joint7
        //     links are: link7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: joint6
        //     links are: link6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: joint5
        //     links are: link5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: joint4
        //     links are: link4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: joint3
        //     links are: link3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: joint2
        //     links are: link2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row], &s_vaf[84 + 6*1]);
            int dstOffset = 84 + 6*0 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   used to compute vaf as helper values
     *   optimized for qdd = 0
     *
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 126
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 42
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner_vaf(T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: joint1
        //     links are: link1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: joint2
        //     links are: link2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 1;
            T qd_qdd_val = (row == 1) * (vFlag * s_qd[1]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*0]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx1_peq_scaled<T>(&s_vaf[48], &s_vaf[6], s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: joint3
        //     links are: link3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 2;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[2]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*1]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[12], s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: joint4
        //     links are: link4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 3;
            T qd_qdd_val = (row == 1) * (vFlag * s_qd[3]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*2]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx1_peq_scaled<T>(&s_vaf[60], &s_vaf[18], s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: joint5
        //     links are: link5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 4;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[4]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*3]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[24], s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: joint6
        //     links are: link6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 5;
            T qd_qdd_val = (row == 1) * (vFlag * s_qd[5]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*4]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx1_peq_scaled<T>(&s_vaf[72], &s_vaf[30], s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: joint7
        //     links are: link7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 6;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[6]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*5]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[78], &s_vaf[36], s_qd[6]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 42 + jid6;
            T *dst = IaFlag ? &s_vaf[84] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[252 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[84 + jid6], &s_vaf[jid6], &s_temp[jid6]);
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: joint7
        //     links are: link7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: joint6
        //     links are: link6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: joint5
        //     links are: link5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: joint4
        //     links are: link4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: joint3
        //     links are: link3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: joint2
        //     links are: link2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row], &s_vaf[84 + 6*1]);
            int dstOffset = 84 + 6*0 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param s_c is the vector of output torques
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_device(T *s_c,  const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[126];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param s_c is the vector of output torques
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_device(T *s_c,  const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[126];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   used to compute vaf as helper values
     *
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_vaf_device(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   used to compute vaf as helper values
     *   optimized for qdd = 0
     *
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_vaf_device(T *s_vaf, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param d_qdd is the vector of joint accelerations
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel_single_timing(T *d_c, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_qdd[7]; 
        __shared__ T s_c[7];
        __shared__ T s_vaf[126];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_qdd[ind] = d_qdd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            d_c[ind] = s_c[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param d_qdd is the vector of joint accelerations
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel(T *d_c, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_qdd[7]; 
        __shared__ T s_c[7];
        __shared__ T s_vaf[126];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            const T *d_qdd_k = &d_qdd[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_c_k = &d_c[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_c_k[ind] = s_c[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel_single_timing(T *d_c, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_c[7];
        __shared__ T s_vaf[126];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            d_c[ind] = s_c[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel(T *d_c, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_c[7];
        __shared__ T s_vaf[126];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_c_k = &d_c[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_c_k[ind] = s_c[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_c,hd_data->d_c,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_c,hd_data->d_c,NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call ID %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param s_Minv is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is a pointer to helper shared memory of size 667
     */
    template <typename T>
    __device__
    void direct_minv_inner(T *s_Minv, const T *s_q, T *s_XImats, int *s_topology_helpers, T *s_temp) {
        // T *s_F = &s_temp[0]; T *s_IA = &s_temp[294]; T *s_U = &s_temp[546]; T *s_Dinv = &s_temp[588]; T *s_Ia = &s_temp[595]; T *s_IaTemp = &s_temp[631];
        // Initialize IA = I
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 252; ind += blockDim.x*blockDim.y){
            s_temp[294 + ind] = s_XImats[252 + ind];
        }
        // Zero Minv and F
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
            if(ind < 294){s_temp[0 + ind] = static_cast<T>(0);}
            else{s_Minv[ind - 294] = static_cast<T>(0);}
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // backward pass updates where bfs_level is 6
        //     joints are: joint7
        //     links are: link7
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 36 + row] = s_temp[294 + 6*36 + 6*2 + row];
            if(row == 2){
                s_temp[588 + 6] = static_cast<T>(1)/s_temp[546 + 36 + 2];
                s_Minv[8 * 6] = s_temp[588 + 6];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            s_Minv[42 + 6] -= s_temp[588 + 6] * s_temp[0 + 42*6 + 36 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 42*6 + 36 + row] += s_temp[546 + 6*6 + row] * s_Minv[42 + 6];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[595 + ind] = s_temp[510 + ind] - (s_temp[582 + row] * s_temp[594] * s_temp[582 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_temp[0 + 42*6 + 6*6]; T *dst = &s_temp[0 + 42*5 + 6*6];
            // adjust for temp comps
            if (col >= 1) {
                col -= 1; src = &s_temp[595 + 6*col]; dst = &s_temp[631 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            s_temp[474 + 6*col + row] += dot_prod<T,6,6,1>(&s_temp[631 + row],&s_XImats[216 + 6*col]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 5
        //     joints are: joint6
        //     links are: link6
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 30 + row] = s_temp[294 + 6*30 + 6*1 + row];
            if(row == 1){
                s_temp[588 + 5] = static_cast<T>(1)/s_temp[546 + 30 + 1];
                s_Minv[8 * 5] = s_temp[588 + 5];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(5 + ind); int jid_subtreeN = 7*(5 + ind);
            s_Minv[jid_subtreeN + 5] -= s_temp[588 + 5] * s_temp[0 + 42*5 + jid_subtree6 + 1];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 42*5 + jid_subtree6 + row] += s_temp[546 + 6*5 + row] * s_Minv[jid_subtreeN + 5];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[595 + ind] = s_temp[474 + ind] - (s_temp[576 + row] * s_temp[593] * s_temp[576 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 48; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_temp[0 + 42*5 + 6*(5 + col)]; T *dst = &s_temp[0 + 42*4 + 6*(5 + col)];
            // adjust for temp comps
            if (col >= 2) {
                col -= 2; src = &s_temp[595 + 6*col]; dst = &s_temp[631 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            s_temp[438 + 6*col + row] += dot_prod<T,6,6,1>(&s_temp[631 + row],&s_XImats[180 + 6*col]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 4
        //     joints are: joint5
        //     links are: link5
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 24 + row] = s_temp[294 + 6*24 + 6*2 + row];
            if(row == 2){
                s_temp[588 + 4] = static_cast<T>(1)/s_temp[546 + 24 + 2];
                s_Minv[8 * 4] = s_temp[588 + 4];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 3; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(4 + ind); int jid_subtreeN = 7*(4 + ind);
            s_Minv[jid_subtreeN + 4] -= s_temp[588 + 4] * s_temp[0 + 42*4 + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 42*4 + jid_subtree6 + row] += s_temp[546 + 6*4 + row] * s_Minv[jid_subtreeN + 4];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[595 + ind] = s_temp[438 + ind] - (s_temp[570 + row] * s_temp[592] * s_temp[570 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 54; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_temp[0 + 42*4 + 6*(4 + col)]; T *dst = &s_temp[0 + 42*3 + 6*(4 + col)];
            // adjust for temp comps
            if (col >= 3) {
                col -= 3; src = &s_temp[595 + 6*col]; dst = &s_temp[631 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            s_temp[402 + 6*col + row] += dot_prod<T,6,6,1>(&s_temp[631 + row],&s_XImats[144 + 6*col]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 3
        //     joints are: joint4
        //     links are: link4
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 18 + row] = s_temp[294 + 6*18 + 6*1 + row];
            if(row == 1){
                s_temp[588 + 3] = static_cast<T>(1)/s_temp[546 + 18 + 1];
                s_Minv[8 * 3] = s_temp[588 + 3];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 4; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(3 + ind); int jid_subtreeN = 7*(3 + ind);
            s_Minv[jid_subtreeN + 3] -= s_temp[588 + 3] * s_temp[0 + 42*3 + jid_subtree6 + 1];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 42*3 + jid_subtree6 + row] += s_temp[546 + 6*3 + row] * s_Minv[jid_subtreeN + 3];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[595 + ind] = s_temp[402 + ind] - (s_temp[564 + row] * s_temp[591] * s_temp[564 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 60; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_temp[0 + 42*3 + 6*(3 + col)]; T *dst = &s_temp[0 + 42*2 + 6*(3 + col)];
            // adjust for temp comps
            if (col >= 4) {
                col -= 4; src = &s_temp[595 + 6*col]; dst = &s_temp[631 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            s_temp[366 + 6*col + row] += dot_prod<T,6,6,1>(&s_temp[631 + row],&s_XImats[108 + 6*col]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 2
        //     joints are: joint3
        //     links are: link3
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 12 + row] = s_temp[294 + 6*12 + 6*2 + row];
            if(row == 2){
                s_temp[588 + 2] = static_cast<T>(1)/s_temp[546 + 12 + 2];
                s_Minv[8 * 2] = s_temp[588 + 2];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 5; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(2 + ind); int jid_subtreeN = 7*(2 + ind);
            s_Minv[jid_subtreeN + 2] -= s_temp[588 + 2] * s_temp[0 + 42*2 + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 42*2 + jid_subtree6 + row] += s_temp[546 + 6*2 + row] * s_Minv[jid_subtreeN + 2];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[595 + ind] = s_temp[366 + ind] - (s_temp[558 + row] * s_temp[590] * s_temp[558 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 66; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_temp[0 + 42*2 + 6*(2 + col)]; T *dst = &s_temp[0 + 42*1 + 6*(2 + col)];
            // adjust for temp comps
            if (col >= 5) {
                col -= 5; src = &s_temp[595 + 6*col]; dst = &s_temp[631 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            s_temp[330 + 6*col + row] += dot_prod<T,6,6,1>(&s_temp[631 + row],&s_XImats[72 + 6*col]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 1
        //     joints are: joint2
        //     links are: link2
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 6 + row] = s_temp[294 + 6*6 + 6*1 + row];
            if(row == 1){
                s_temp[588 + 1] = static_cast<T>(1)/s_temp[546 + 6 + 1];
                s_Minv[8 * 1] = s_temp[588 + 1];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(1 + ind); int jid_subtreeN = 7*(1 + ind);
            s_Minv[jid_subtreeN + 1] -= s_temp[588 + 1] * s_temp[0 + 42*1 + jid_subtree6 + 1];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 42*1 + jid_subtree6 + row] += s_temp[546 + 6*1 + row] * s_Minv[jid_subtreeN + 1];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[595 + ind] = s_temp[330 + ind] - (s_temp[552 + row] * s_temp[589] * s_temp[552 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_temp[0 + 42*1 + 6*(1 + col)]; T *dst = &s_temp[0 + 42*0 + 6*(1 + col)];
            // adjust for temp comps
            if (col >= 6) {
                col -= 6; src = &s_temp[595 + 6*col]; dst = &s_temp[631 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            s_temp[294 + 6*col + row] += dot_prod<T,6,6,1>(&s_temp[631 + row],&s_XImats[36 + 6*col]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 0
        //     joints are: joint1
        //     links are: link1
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 0 + row] = s_temp[294 + 6*0 + 6*2 + row];
            if(row == 2){
                s_temp[588 + 0] = static_cast<T>(1)/s_temp[546 + 0 + 2];
                s_Minv[8 * 0] = s_temp[588 + 0];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(0 + ind); int jid_subtreeN = 7*(0 + ind);
            s_Minv[jid_subtreeN + 0] -= s_temp[588 + 0] * s_temp[0 + 42*0 + jid_subtree6 + 2];
        }
        __syncthreads();
        //
        // Forward Pass
        //   Note that due to the i: operation we need to go serially over all n
        //
        // forward pass for jid: 0
        // F[i,:,i:] = S * Minv[i,i:] as parent is base so rest is skipped
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[0 + ind] = (row == 2) * s_Minv[0 + 7 * col];
        }
        __syncthreads();
        // forward pass for jid: 1
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 6;
            s_temp[42 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[36 + row], &s_temp[0 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 1;
            T *s_Fcol = &s_temp[42 + 6*col_ind];
            s_Minv[7 * col_ind + 1] -= s_temp[589] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[552]);
            s_Fcol[1] += s_Minv[7 * col_ind + 1];
        }
        __syncthreads();
        // forward pass for jid: 2
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 30; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 12;
            s_temp[84 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[72 + row], &s_temp[42 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 5; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 2;
            T *s_Fcol = &s_temp[84 + 6*col_ind];
            s_Minv[7 * col_ind + 2] -= s_temp[590] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[558]);
            s_Fcol[2] += s_Minv[7 * col_ind + 2];
        }
        __syncthreads();
        // forward pass for jid: 3
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 18;
            s_temp[126 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[108 + row], &s_temp[84 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 4; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 3;
            T *s_Fcol = &s_temp[126 + 6*col_ind];
            s_Minv[7 * col_ind + 3] -= s_temp[591] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[564]);
            s_Fcol[1] += s_Minv[7 * col_ind + 3];
        }
        __syncthreads();
        // forward pass for jid: 4
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 18; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 24;
            s_temp[168 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[144 + row], &s_temp[126 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 3; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 4;
            T *s_Fcol = &s_temp[168 + 6*col_ind];
            s_Minv[7 * col_ind + 4] -= s_temp[592] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[570]);
            s_Fcol[2] += s_Minv[7 * col_ind + 4];
        }
        __syncthreads();
        // forward pass for jid: 5
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 30;
            s_temp[210 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[180 + row], &s_temp[168 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 5;
            T *s_Fcol = &s_temp[210 + 6*col_ind];
            s_Minv[7 * col_ind + 5] -= s_temp[593] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[576]);
            s_Fcol[1] += s_Minv[7 * col_ind + 5];
        }
        __syncthreads();
        // forward pass for jid: 6
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 36;
            s_temp[252 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[216 + row], &s_temp[210 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 6;
            T *s_Fcol = &s_temp[252 + 6*col_ind];
            s_Minv[7 * col_ind + 6] -= s_temp[594] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[582]);
        }
        __syncthreads();
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param s_Minv is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void direct_minv_device(T *s_Minv, const T *s_q, const robotModel<T> *d_robotModel){
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp);
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param d_Minv is a pointer to memory for the final result
     * @param d_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void direct_minv_kernel_single_timing(T *d_Minv, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS){
        __shared__ T s_q[7];
        __shared__ T s_Minv[49];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_q[ind] = d_q[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            d_Minv[ind] = s_Minv[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param d_Minv is a pointer to memory for the final result
     * @param d_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void direct_minv_kernel(T *d_Minv, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS){
        __shared__ T s_q[7];
        __shared__ T s_Minv[49];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp);
            __syncthreads();
            // save down to global
            T *d_Minv_k = &d_Minv[k*49];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                d_Minv_k[ind] = s_Minv[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void direct_minv(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                     const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_COMPRESSED_MEM) {direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_Minv,hd_data->d_Minv,NUM_JOINTS*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void direct_minv_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                   const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_COMPRESSED_MEM) {direct_minv_kernel_single_timing<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {direct_minv_kernel_single_timing<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_Minv,hd_data->d_Minv,NUM_JOINTS*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call Minv %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void direct_minv_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                  const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_COMPRESSED_MEM) {direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Finish the forward dynamics computation with qdd = Minv*(u-c)
     *
     * Notes:
     *   Assumes s_Minv and s_c are already computed
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_u is the vector of joint input torques
     * @param s_c is the bias vector
     * @param s_Minv is the inverse mass matrix
     */
    template <typename T>
    __device__
    void forward_dynamics_finish(T *s_qdd, const T *s_u, const T *s_c, const T *s_Minv) {
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 7; row += blockDim.x*blockDim.y){
            T val = static_cast<T>(0);
            for(int col = 0; col < 7; col++) {
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                val += s_Minv[index] * (s_u[col] - s_c[col]);
            }
            s_qdd[row] = val;
        }
    }

    /**
     * Computes forward dynamics
     *
     * Notes:
     *   Assumes s_XImats is updated already for the current s_q
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of joint input torques
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is the pointer to the shared memory needed of size: 716
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void forward_dynamics_inner(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity, T *d_f_ext) {
        direct_minv_inner<T>(s_temp, s_q, s_XImats, s_topology_helpers, &s_temp[49]);
        inverse_dynamics_inner<T>(&s_temp[49], &s_temp[56], s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[182], gravity, d_f_ext);
        forward_dynamics_finish<T>(s_qdd, s_u, &s_temp[49], s_temp);
    }

    /**
     * Computes forward dynamics
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of joint input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void forward_dynamics_device(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, gravity, nullptr);
    }

    /**
     * Computes forward dynamics
     *
     * @param d_qdd is a pointer to memory for the final result
     * @param d_q_qd_u is the vector of joint positions, velocities, and input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_kernel_single_timing(T *d_qdd, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[21]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
        __shared__ T s_qdd[7];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
            s_q_qd_u[ind] = d_q_qd_u[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, gravity, nullptr);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            d_qdd[ind] = s_qdd[ind];
        }
        __syncthreads();
    }

    /**
     * Computes forward dynamics
     *
     * @param d_qdd is a pointer to memory for the final result
     * @param d_q_qd_u is the vector of joint positions, velocities, and input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_kernel(T *d_qdd, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[21]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
        __shared__ T s_qdd[7];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, gravity, nullptr);
            __syncthreads();
            // save down to global
            T *d_qdd_k = &d_qdd[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_qdd_k[ind] = s_qdd[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T>
    __host__
    void forward_dynamics(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd_u = 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_u*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        forward_dynamics_kernel<T><<<block_dimms,thread_dimms,FD_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd_u,d_robotModel,gravity,num_timesteps);
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_qdd,hd_data->d_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T>
    __host__
    void forward_dynamics_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd_u = 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_u*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        forward_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,FD_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd_u,d_robotModel,gravity,num_timesteps);
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_qdd,hd_data->d_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call FD %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T>
    __host__
    void forward_dynamics_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd_u = 3*NUM_JOINTS;
        // then call the kernel
        forward_dynamics_kernel<T><<<block_dimms,thread_dimms,FD_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd_u,d_robotModel,gravity,num_timesteps);
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   Assumes s_XImats is updated already for the current s_q
     *
     * @param s_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_vaf are the helper intermediate variables computed by inverse_dynamics
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is a pointer to helper shared memory of size 66*NUM_JOINTS + 6*sparse_dv,da,df_col_needs = 1722
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_gradient_inner(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_vaf, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity) {
        //
        // dv and da need 28 cols per dq,dqd
        // df needs 49 cols per dq,dqd
        //    out of a possible 49 cols per dq,dqd
        // Gradients are stored compactly as dv_i/dq_[0...a], dv_i+1/dq_[0...b], etc
        //    where a and b are the needed number of columns
        //
        // Temp memory offsets are as follows:
        // T *s_dv_dq = &s_temp[0]; T *s_dv_dqd = &s_temp[168]; T *s_da_dq = &s_temp[336];
        // T *s_da_dqd = &s_temp[504]; T *s_df_dq = &s_temp[672]; T *s_df_dqd = &s_temp[966];
        // T *s_FxvI = &s_temp[1260]; T *s_MxXv = &s_temp[1512]; T *s_MxXa = &s_temp[1554];
        // T *s_Mxv = &s_temp[1596]; T *s_Mxf = &s_temp[1638]; T *s_Iv = &s_temp[1680];
        //
        // Initial Temp Comps
        //
        // First compute Imat*v and Xmat*v_parent, Xmat*a_parent (store in FxvI for now)
        // Note that if jid_parent == -1 then v_parent = 0 and a_parent = gravity
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 126; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int jid = col % 7; int jid6 = 6*jid;
            bool parentIsBase = (jid-1) == -1;
            bool comp1 = col < 7; bool comp3 = col >= 14;
            int XIOffset  =  comp1 * 252 + 6*jid6 + row; // rowCol of I (comp1) or X (comp 2 and 3)
            int vaOffset  = comp1 * jid6 + !comp1 * 6*(jid-1) + comp3 * 42; // v_i (comp1) or va_parent (comp 2 and 3)
            int dstOffset = comp1 * 1680 + !comp1 * 1260 + comp3 * 42 + jid6 + row; // rowCol of dst
            s_temp[dstOffset] = (parentIsBase && !comp1) ? comp3 * s_XImats[XIOffset + 30] * gravity : 
                                                           dot_prod<T,6,6,1>(&s_XImats[XIOffset],&s_vaf[vaOffset]);
        }
        __syncthreads();
        // Then compute Mx(Xv), Mx(Xa), Mx(v), Mx(f)
        for(int col = threadIdx.x + threadIdx.y*blockDim.x; col < 28; col += blockDim.x*blockDim.y){
            int jid = col / 4; int selector = col % 4; int jid6 = 6*jid;
            // branch to get pointer locations
            int dstOffset; const T * src;
                 if (selector == 0){ dstOffset = 1512; src = &s_temp[1260]; }
            else if (selector == 1){ dstOffset = 1554; src = &s_temp[1302]; }
            else if (selector == 2){ dstOffset = 1596; src = &s_vaf[0]; }
            else              { dstOffset = 1638; src = &s_vaf[84]; }
            mxX<T>(&s_temp[dstOffset + jid6], &src[jid6], s_topology_helpers[jid]);
        }
        __syncthreads();
        //
        // Forward Pass
        //
        // We start with dv/du noting that we only have values
        //    for ancestors and for the current index else 0
        // dv/du where bfs_level is 0
        //     joints are: joint1
        //     links are: link1
        // when parent is base dv_dq = 0, dv_dqd = S
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int dq_flag = (ind / 6) == 0;
            int du_offset = dq_flag ? 0 : 168;
            s_temp[du_offset + 6*0 + row] = (!dq_flag && row == 2) * static_cast<T>(1);
        }
        __syncthreads();
        // dv/du where bfs_level is 1
        //     joints are: joint2
        //     links are: link2
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 1; int col_jid = col_du % 1;
            int dq_flag = col < 1;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            s_temp[du_col_offset + 6*1 + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*1 + row],&s_temp[du_col_offset + 6*0]);
            // then add {Mx(Xv) or S for col ind}
            s_temp[du_col_offset + 6*1 + 6 + row] = 
                dq_flag * s_temp[1512 + 6*1 + row] + (!dq_flag && row == 1) * static_cast<T>(1);
        }
        __syncthreads();
        // dv/du where bfs_level is 2
        //     joints are: joint3
        //     links are: link3
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 2; int col_jid = col_du % 2;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            s_temp[du_col_offset + 6*3 + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*2 + row],&s_temp[du_col_offset + 6*1]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 1) {
                s_temp[du_col_offset + 6*3 + 6 + row] = 
                    dq_flag * s_temp[1512 + 6*2 + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 3
        //     joints are: joint4
        //     links are: link4
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 3; int col_jid = col_du % 3;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            s_temp[du_col_offset + 6*6 + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*3 + row],&s_temp[du_col_offset + 6*3]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 2) {
                s_temp[du_col_offset + 6*6 + 6 + row] = 
                    dq_flag * s_temp[1512 + 6*3 + row] + (!dq_flag && row == 1) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 4
        //     joints are: joint5
        //     links are: link5
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 48; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 4; int col_jid = col_du % 4;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            s_temp[du_col_offset + 6*10 + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*4 + row],&s_temp[du_col_offset + 6*6]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 3) {
                s_temp[du_col_offset + 6*10 + 6 + row] = 
                    dq_flag * s_temp[1512 + 6*4 + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 5
        //     joints are: joint6
        //     links are: link6
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 60; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 5; int col_jid = col_du % 5;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            s_temp[du_col_offset + 6*15 + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*5 + row],&s_temp[du_col_offset + 6*10]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 4) {
                s_temp[du_col_offset + 6*15 + 6 + row] = 
                    dq_flag * s_temp[1512 + 6*5 + row] + (!dq_flag && row == 1) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 6
        //     joints are: joint7
        //     links are: link7
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 6; int col_jid = col_du % 6;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            s_temp[du_col_offset + 6*21 + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*6 + row],&s_temp[du_col_offset + 6*15]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 5) {
                s_temp[du_col_offset + 6*21 + 6 + row] = 
                    dq_flag * s_temp[1512 + 6*6 + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // start da/du by setting = MxS(dv/du)*qd + {MxXa, Mxv} for all n in parallel
        // start with da/du = MxS(dv/du)*qd
        for(int col = threadIdx.x + threadIdx.y*blockDim.x; col < 56; col += blockDim.x*blockDim.y){
            int col_du = col % 28;
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du < 3 && col_du >= 1) * 1 + (col_du < 6 && col_du >= 3) * 2 + (col_du < 10 && col_du >= 6) * 3 + (col_du < 15 && col_du >= 10) * 4 + (col_du < 21 && col_du >= 15) * 5 + (col_du >= 21) * 6;
            mxX_scaled<T>(&s_temp[336 + 6*col], &s_temp[0 + 6*col], s_qd[jid], s_topology_helpers[jid]);
            // then add {MxXa, Mxv} to the appropriate column
            int dq_flag = col == col_du; int src_offset = dq_flag * 1554 + !dq_flag * 1596 + 6*jid;
            if(col_du == ((jid+1)*(jid+2)/2 - 1)){
                for(int row = 0; row < 6; row++){
                    s_temp[336 + 6*col + row] += s_temp[src_offset + row];
                }
            }
        }
        __syncthreads();
        // Finish da/du with parent updates noting that we only have values
        //    for ancestors and for the current index and nothing for bfs 0
        // da/du where bfs_level is 1
        //     joints are: joint2
        //     links are: link2
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 1;
            int dq_flag = col == col_du; int col_jid = col_du % 1;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            s_temp[du_col_offset + 6*1 + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*1 + row],&s_temp[du_col_offset + 6*0]);
        }
        __syncthreads();
        // da/du where bfs_level is 2
        //     joints are: joint3
        //     links are: link3
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 2;
            int dq_flag = col == col_du; int col_jid = col_du % 2;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            s_temp[du_col_offset + 6*3 + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*2 + row],&s_temp[du_col_offset + 6*1]);
        }
        __syncthreads();
        // da/du where bfs_level is 3
        //     joints are: joint4
        //     links are: link4
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 3;
            int dq_flag = col == col_du; int col_jid = col_du % 3;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            s_temp[du_col_offset + 6*6 + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*3 + row],&s_temp[du_col_offset + 6*3]);
        }
        __syncthreads();
        // da/du where bfs_level is 4
        //     joints are: joint5
        //     links are: link5
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 48; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 4;
            int dq_flag = col == col_du; int col_jid = col_du % 4;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            s_temp[du_col_offset + 6*10 + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*4 + row],&s_temp[du_col_offset + 6*6]);
        }
        __syncthreads();
        // da/du where bfs_level is 5
        //     joints are: joint6
        //     links are: link6
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 60; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 5;
            int dq_flag = col == col_du; int col_jid = col_du % 5;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            s_temp[du_col_offset + 6*15 + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*5 + row],&s_temp[du_col_offset + 6*10]);
        }
        __syncthreads();
        // da/du where bfs_level is 6
        //     joints are: joint7
        //     links are: link7
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 6;
            int dq_flag = col == col_du; int col_jid = col_du % 6;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            s_temp[du_col_offset + 6*21 + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*6 + row],&s_temp[du_col_offset + 6*15]);
        }
        __syncthreads();
        // Init df/du to 0
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 588; ind += blockDim.x*blockDim.y){
            s_temp[672 + ind] = static_cast<T>(0);
        }
        __syncthreads();
        // Start the df/du by setting = fx(dv/du)*Iv and also compute the temp = Fx(v)*I 
        //    aka do all of the Fx comps in parallel
        // note that while df has more cols than dva the dva cols are the first few df cols
        for(int col = threadIdx.x + threadIdx.y*blockDim.x; col < 98; col += blockDim.x*blockDim.y){
            int col_du = col % 28;
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du < 3 && col_du >= 1) * 1 + (col_du < 6 && col_du >= 3) * 2 + (col_du < 10 && col_du >= 6) * 3 + (col_du < 15 && col_du >= 10) * 4 + (col_du < 21 && col_du >= 15) * 5 + (col_du >= 21) * 6;
            // Compute Offsets and Pointers
            int dq_flag = col == col_du; int dva_to_df_adjust = 7*jid - jid*(jid+1)/2;
            int Offset_col_du_src = dq_flag * 0 + !dq_flag * 168 + 6*col_du;
            int Offset_col_du_dst = dq_flag * 672 + !dq_flag * 966 + 6*(col_du + dva_to_df_adjust);
            T *dst = &s_temp[Offset_col_du_dst]; const T *fx_src = &s_temp[Offset_col_du_src]; const T *mult_src = &s_temp[1680 + 6*jid];
            // Adjust pointers for temp comps (if applicable)
            if (col >= 56) {
                int comp = col - 56; int comp_col = comp % 6; // int jid = comp / 6;
                int jid6 = comp - comp_col; int jid36_col6 = 6*jid6 + 6*comp_col;
                dst = &s_temp[1260 + jid36_col6]; fx_src = &s_vaf[jid6]; mult_src = &s_XImats[252 + jid36_col6];
            }
            fx_times_v<T>(dst, fx_src, mult_src);
        }
        __syncthreads();
        // Then in parallel finish df/du += I*da/du + (Fx(v)I)*dv/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 336; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col6 = ind - row; int col_du = (col % 28);
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du < 3 && col_du >= 1) * 1 + (col_du < 6 && col_du >= 3) * 2 + (col_du < 10 && col_du >= 6) * 3 + (col_du < 15 && col_du >= 10) * 4 + (col_du < 21 && col_du >= 15) * 5 + (col_du >= 21) * 6;
            // Compute Offsets and Pointers
            int dva_to_df_adjust = 7*jid - jid*(jid+1)/2;
            if (col >= 28){dva_to_df_adjust += 21;}
            T *df_row_col = &s_temp[672 + 6*dva_to_df_adjust + ind];
            const T *dv_col = &s_temp[0 + col6]; const T *da_col = &s_temp[336 + col6];
            int jid36 = 36*jid; const T *I_row = &s_XImats[252 + jid36 + row]; const T *FxvI_row = &s_temp[1260 + jid36 + row];
            // Compute the values
            *df_row_col += dot_prod<T,6,6,1>(I_row,da_col) + dot_prod<T,6,6,1>(FxvI_row,dv_col);
        }
        // At the same time compute the last temp var: -X^T * mx(f)
        // use Mx(Xv) temp memory as those values are no longer needed
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int XTcol = ind % 6; int jid6 = ind - XTcol;
            s_temp[1512 + ind] = -dot_prod<T,6,1,1>(&s_XImats[6*(jid6 + XTcol)], &s_temp[1638 + jid6]);
        }
        __syncthreads();
        //
        // BACKWARD Pass
        //
        // df/du update where bfs_level is 6
        //     joints are: joint7
        //     links are: link7
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int dst_adjust = (col_du >= 6) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*35 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row],&s_temp[du_col_offset + 6*42])
                          + dq_flag * (col_du == 6) * s_temp[1512 + 6*6 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 5
        //     joints are: joint6
        //     links are: link6
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int dst_adjust = (col_du >= 5) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*28 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row],&s_temp[du_col_offset + 6*35])
                          + dq_flag * (col_du == 5) * s_temp[1512 + 6*5 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 4
        //     joints are: joint5
        //     links are: link5
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int dst_adjust = (col_du >= 4) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*21 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row],&s_temp[du_col_offset + 6*28])
                          + dq_flag * (col_du == 4) * s_temp[1512 + 6*4 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 3
        //     joints are: joint4
        //     links are: link4
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int dst_adjust = (col_du >= 3) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*14 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row],&s_temp[du_col_offset + 6*21])
                          + dq_flag * (col_du == 3) * s_temp[1512 + 6*3 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 2
        //     joints are: joint3
        //     links are: link3
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int dst_adjust = (col_du >= 2) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*7 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row],&s_temp[du_col_offset + 6*14])
                          + dq_flag * (col_du == 2) * s_temp[1512 + 6*2 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 1
        //     joints are: joint2
        //     links are: link2
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int dst_adjust = (col_du >= 1) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*0 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row],&s_temp[du_col_offset + 6*7])
                          + dq_flag * (col_du == 1) * s_temp[1512 + 6*1 + row];
            *dst += update_val;
        }
        __syncthreads();
        // Finally dc[i]/du = S[i]^T*df[i]/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            int jid = ind % 7; int jid_dq_qd = ind / 7; int jid_du = jid_dq_qd % 7; int dq_flag = jid_du == jid_dq_qd;
            int Offset_src = dq_flag * 672 + !dq_flag * 966 + 6 * 7 * jid + 6 * jid_du + s_topology_helpers[jid];
            int Offset_dst = !dq_flag * 49 + 7 * jid_du + jid;
            s_dc_du[Offset_dst] = s_temp[Offset_src];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * @param s_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_gradient_device(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[126];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param s_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_gradient_device(T *s_dc_du, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[126];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel_single_timing(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_qdd[7]; 
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_qdd[ind] = d_qdd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            d_dc_du[ind] = s_dc_du[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_qdd[7]; 
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            const T *d_qdd_k = &d_qdd[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_dc_du_k = &d_dc_du[k*98];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_dc_du_k[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel_single_timing(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            d_dc_du[ind] = s_dc_du[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_dc_du_k = &d_dc_du[k*98];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_dc_du_k[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_gradient(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                   const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_dc_du,hd_data->d_dc_du,NUM_JOINTS*2*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_gradient_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                                 const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_dc_du,hd_data->d_dc_du,NUM_JOINTS*2*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call ID_DU %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_gradient_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                                const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * Notes:
     *   Uses the fd/du = -Minv*id/du trick as described in Carpentier and Mansrud 'Analytical Derivatives of Rigid Body Dynamics Algorithms'
     *
     * @param s_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void forward_dynamics_gradient_device(T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[126];
        __shared__ T s_dc_du[98];
        __shared__ T s_Minv[49];
        __shared__ T s_qdd[7];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
        direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp);
        inverse_dynamics_inner<T>(s_temp, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[7], gravity);
        forward_dynamics_finish<T>(s_qdd, s_u, s_temp, s_Minv);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            int row = ind % 7; int dc_col_offset = ind - row;
            // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
            T val = static_cast<T>(0);
            for(int col = 0; col < 7; col++) {
                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
            }
            s_df_du[ind] = -val;
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * Notes:
     *   Uses the fd/du = -Minv*id/du trick as described in Carpentier and Mansrud 'Analytical Derivatives of Rigid Body Dynamics Algorithms'
     *
     * @param s_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param s_Minv is the mass matrix
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void forward_dynamics_gradient_device(T *s_df_du, const T *s_q, const T *s_qd, const T *s_qdd, const T *s_Minv, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[126];
        __shared__ T s_dc_du[98];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            int row = ind % 7; int dc_col_offset = ind - row;
            // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
            T val = static_cast<T>(0);
            for(int col = 0; col < 7; col++) {
                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
            }
            s_df_du[ind] = -val;
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param d_Minv is the mass matrix
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel_single_timing(T *d_df_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ T s_qdd[7];
        __shared__ T s_Minv[49];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_qdd[ind] = d_qdd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            s_Minv[ind] = d_Minv[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                int row = ind % 7; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 7; col++) {
                    int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                }
                s_temp[ind] = -val;
            }
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            d_df_du[ind] = s_temp[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param d_Minv is the mass matrix
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel(T *d_df_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ T s_qdd[7];
        __shared__ T s_Minv[49];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            const T *d_qdd_k = &d_qdd[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd_k[ind];
            }
            const T *d_Minv_k = &d_Minv[k*49];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                s_Minv[ind] = d_Minv_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                int row = ind % 7; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 7; col++) {
                    int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                }
                s_temp[ind] = -val;
            }
            // save down to global
            T *d_df_du_k = &d_df_du[k*98];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_df_du_k[ind] = s_temp[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions, velocities, and input torques
     * @param stride_q_qd_u is the stide between each q, qd, u
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel_single_timing(T *d_df_du, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[3*7]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ T s_qdd[7];
        __shared__ T s_Minv[49];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
            s_q_qd_u[ind] = d_q_qd_u[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp);
            inverse_dynamics_inner<T>(s_temp, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[7], gravity);
            forward_dynamics_finish<T>(s_qdd, s_u, s_temp, s_Minv);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                int row = ind % 7; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 7; col++) {
                    int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                }
                s_temp[ind] = -val;
            }
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            d_df_du[ind] = s_temp[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions, velocities, and input torques
     * @param stride_q_qd_u is the stide between each q, qd, u
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel(T *d_df_du, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[3*7]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ T s_qdd[7];
        __shared__ T s_Minv[49];
        __shared__ int s_topology_helpers[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp);
            inverse_dynamics_inner<T>(s_temp, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[7], gravity);
            forward_dynamics_finish<T>(s_qdd, s_u, s_temp, s_Minv);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                int row = ind % 7; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 7; col++) {
                    int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                }
                s_temp[ind] = -val;
            }
            // save down to global
            T *d_df_du_k = &d_df_du[k*98];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_df_du_k[ind] = s_temp[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_MINV_FLAG = false>
    __host__
    void forward_dynamics_gradient(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd= 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        if (USE_QDD_MINV_FLAG) {
            gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[1]));
            gpuErrchk(cudaMemcpyAsync(hd_data->d_Minv,hd_data->h_Minv,NUM_JOINTS*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[2]));
        }
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_QDD_MINV_FLAG) {forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_Minv, d_robotModel,gravity,num_timesteps);}
        else {forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_df_du,hd_data->d_df_du,NUM_JOINTS*2*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_MINV_FLAG = false>
    __host__
    void forward_dynamics_gradient_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd= 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        if (USE_QDD_MINV_FLAG) {
            gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[1]));
            gpuErrchk(cudaMemcpyAsync(hd_data->d_Minv,hd_data->h_Minv,NUM_JOINTS*NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[2]));
        }
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_QDD_MINV_FLAG) {forward_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_Minv, d_robotModel,gravity,num_timesteps);}
        else {forward_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_df_du,hd_data->d_df_du,NUM_JOINTS*2*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call FD_DU %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_MINV_FLAG = false>
    __host__
    void forward_dynamics_gradient_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd= 3*NUM_JOINTS;
        // then call the kernel
        if (USE_QDD_MINV_FLAG) {forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_Minv, d_robotModel,gravity,num_timesteps);}
        else {forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Sets shared mem needed for gradient kernels and initializes streams for host functions
     *
     * @return A pointer to the array of streams
     */
    template <typename T>
    __host__
    cudaStream_t *init_grid(){
        // set the max temp memory for the gradient kernels to account for large robots
        auto id_kern1 = static_cast<void (*)(T *, const T *, const int, const T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);
        auto id_kern2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);
        auto id_kern_timing1 = static_cast<void (*)(T *, const T *, const int, const T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);
        auto id_kern_timing2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);
        auto fd_kern1 = static_cast<void (*)(T *, const T *, const int, const T *, const T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);
        auto fd_kern2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);
        auto fd_kern_timing1 = static_cast<void (*)(T *, const T *, const int, const T *, const T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);
        auto fd_kern_timing2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);
        cudaFuncSetAttribute(id_kern1,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(id_kern2,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(id_kern_timing1,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(id_kern_timing2,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(fd_kern1,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(fd_kern2,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(fd_kern_timing1,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(fd_kern_timing2,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        gpuErrchk(cudaDeviceSynchronize());
        // allocate streams
        cudaStream_t *streams = (cudaStream_t *)malloc(3*sizeof(cudaStream_t));
        int priority, minPriority, maxPriority;
        gpuErrchk(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
        for(int i=0; i<3; i++){
            int adjusted_max = maxPriority - i; priority = adjusted_max > minPriority ? adjusted_max : minPriority;
            gpuErrchk(cudaStreamCreateWithPriority(&(streams[i]),cudaStreamNonBlocking,priority));
        }
        return streams;
    }

    /**
     * Frees the memory used by grid
     *
     * @param streams allocated by init_grid
     * @param robotModel allocated by init_robotModel
     * @param data allocated by init_gridData
     */
    template <typename T>
    __host__
    void close_grid(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T> *hd_data){
        gpuErrchk(cudaFree(d_robotModel));
        gpuErrchk(cudaFree(hd_data->d_q_qd_u)); gpuErrchk(cudaFree(hd_data->d_q_qd)); gpuErrchk(cudaFree(hd_data->d_q));
        gpuErrchk(cudaFree(hd_data->d_c)); gpuErrchk(cudaFree(hd_data->d_Minv)); gpuErrchk(cudaFree(hd_data->d_qdd));
        gpuErrchk(cudaFree(hd_data->d_dc_du)); gpuErrchk(cudaFree(hd_data->d_df_du));
        free(hd_data->h_q_qd_u); free(hd_data->h_q_qd); free(hd_data->h_q);
        free(hd_data->h_c); free(hd_data->h_Minv); free(hd_data->h_qdd);
        free(hd_data->h_dc_du); free(hd_data->h_df_du);
        for(int i=0; i<3; i++){gpuErrchk(cudaStreamDestroy(streams[i]));} free(streams);
    }

}
