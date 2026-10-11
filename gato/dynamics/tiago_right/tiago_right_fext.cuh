#pragma once

// Force-aware overloads of unchanged TIAGo GRiD RNEA bodies. The generated
// tiago_right_grid.cuh remains intact. Wrench is [Tx,Ty,Tz,Fx,Fy,Fz] in the
// final joint frame at that joint origin, constant/body-fixed within a solve.
// Existing IDSVA consumes the force-aware v/a/f buffer for the same contract.
namespace grid {

    template <typename T>
    __device__
    void inverse_dynamics_inner(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, T *s_temp, const T gravity, const T *d_f_ext) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: arm_right_1_joint
        //     links are: arm_right_1_link
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: arm_right_2_joint
        //     links are: arm_right_2_link
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 1;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[1]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*0]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[48], &s_vaf[6], s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: arm_right_3_joint
        //     links are: arm_right_3_link
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
        //     joints are: arm_right_4_joint
        //     links are: arm_right_4_link
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 3;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[3]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*2]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[60], &s_vaf[18], s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: arm_right_5_joint
        //     links are: arm_right_5_link
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
        //     joints are: arm_right_6_joint
        //     links are: arm_right_6_link
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 5;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[5]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*4]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[72], &s_vaf[30], s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: arm_right_7_joint
        //     links are: arm_right_7_link
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
        // Body-fixed wrench at arm_right_7_joint origin, angular-first.
        // Apply before the unchanged generated backward force propagation.
        for (int row = threadIdx.x + threadIdx.y * blockDim.x; row < 6;
             row += blockDim.x * blockDim.y) {
            if (d_f_ext != nullptr) s_vaf[84 + 36 + row] -= d_f_ext[row];
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: arm_right_7_joint
        //     links are: arm_right_7_link
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: arm_right_6_joint
        //     links are: arm_right_6_link
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: arm_right_5_joint
        //     links are: arm_right_5_link
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: arm_right_4_joint
        //     links are: arm_right_4_link
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: arm_right_3_joint
        //     links are: arm_right_3_link
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: arm_right_2_joint
        //     links are: arm_right_2_link
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
        for(int dof_id = threadIdx.x + threadIdx.y*blockDim.x; dof_id < 7; dof_id += blockDim.x*blockDim.y){
            s_c[dof_id] = s_vaf[84 + 6*dof_id + 2];
        }
        __syncthreads();
    }

    template <typename T>
    __device__
    void inverse_dynamics_inner_vaf(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, T *s_temp, const T gravity, const T *d_f_ext) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: arm_right_1_joint
        //     links are: arm_right_1_link
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravityS[k]*qdd[k]
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0]; s_vaf[42 + jid6 + 2] += s_qdd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: arm_right_2_joint
        //     links are: arm_right_2_link
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 1;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[1] + !vFlag * s_qdd[1]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*0]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[48], &s_vaf[6], s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: arm_right_3_joint
        //     links are: arm_right_3_link
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
        //     joints are: arm_right_4_joint
        //     links are: arm_right_4_link
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 3;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[3] + !vFlag * s_qdd[3]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*2]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[60], &s_vaf[18], s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: arm_right_5_joint
        //     links are: arm_right_5_link
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
        //     joints are: arm_right_6_joint
        //     links are: arm_right_6_link
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 5;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[5] + !vFlag * s_qdd[5]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*4]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[72], &s_vaf[30], s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: arm_right_7_joint
        //     links are: arm_right_7_link
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
        // Body-fixed wrench at arm_right_7_joint origin, angular-first.
        // Apply before the unchanged generated backward force propagation.
        for (int row = threadIdx.x + threadIdx.y * blockDim.x; row < 6;
             row += blockDim.x * blockDim.y) {
            if (d_f_ext != nullptr) s_vaf[84 + 36 + row] -= d_f_ext[row];
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: arm_right_7_joint
        //     links are: arm_right_7_link
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: arm_right_6_joint
        //     links are: arm_right_6_link
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: arm_right_5_joint
        //     links are: arm_right_5_link
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: arm_right_4_joint
        //     links are: arm_right_4_link
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: arm_right_3_joint
        //     links are: arm_right_3_link
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: arm_right_2_joint
        //     links are: arm_right_2_link
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row], &s_vaf[84 + 6*1]);
            int dstOffset = 84 + 6*0 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
    }

    template <typename T>
    __device__
    void forward_dynamics_inner(T *s_qdd, const T *s_q, const T *s_qd,
                               const T *s_u, T *s_XImats, T *s_temp,
                               const T gravity, const T *d_f_ext) {
        direct_minv_inner<T>(s_temp, s_q, s_XImats, &s_temp[49]);
        inverse_dynamics_inner<T>(&s_temp[49], &s_temp[56], s_q, s_qd,
                                 s_XImats, &s_temp[182], gravity, d_f_ext);
        forward_dynamics_finish<T>(s_qdd, s_u, &s_temp[49], s_temp);
    }
} // namespace grid
