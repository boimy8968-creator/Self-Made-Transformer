#include "../layers/bmm.h"
#include <cuda_runtime.h>
#include <stdexcept>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

// ==========================================
// 1. GPU Worker: BMM Forward (Normal)
// ==========================================
__global__ void bmm_forward_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int batch_stride_A, int batch_stride_B, int batch_stride_C) 
{
    int batch_idx = blockIdx.z;
    const float* batch_A = A + batch_idx * batch_stride_A;
    const float* batch_B = B + batch_idx * batch_stride_B;
    float* batch_C = C + batch_idx * batch_stride_C;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];

    float r_C[TM][TN] = {0.0f};
    float r_A[TM];
    float r_B[TN];

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / 16;
    int load_b_col = (tid % 16) * 4;

    int num_tiles = K / BK;

    for (int t = 0; t < num_tiles; ++t) {
        int global_a_row = by * BM + load_a_row;
        int global_a_col = t * BK + load_a_col;
        
        if (global_a_row < M && global_a_col < K) {
            float4 vec_A = reinterpret_cast<const float4*>(&batch_A[global_a_row * K + global_a_col])[0];
            s_A[load_a_row * BK + load_a_col + 0] = vec_A.x;
            s_A[load_a_row * BK + load_a_col + 1] = vec_A.y;
            s_A[load_a_row * BK + load_a_col + 2] = vec_A.z;
            s_A[load_a_row * BK + load_a_col + 3] = vec_A.w;
        } else {
            s_A[load_a_row * BK + load_a_col + 0] = 0.0f;
            s_A[load_a_row * BK + load_a_col + 1] = 0.0f;
            s_A[load_a_row * BK + load_a_col + 2] = 0.0f;
            s_A[load_a_row * BK + load_a_col + 3] = 0.0f;
        }

        int global_b_row = t * BK + load_b_row;
        int global_b_col = bx * BN + load_b_col;

        if (global_b_row < K && global_b_col < N) {
            float4 vec_B = reinterpret_cast<const float4*>(&batch_B[global_b_row * N + global_b_col])[0];
            s_B[load_b_row * BN + load_b_col + 0] = vec_B.x;
            s_B[load_b_row * BN + load_b_col + 1] = vec_B.y;
            s_B[load_b_row * BN + load_b_col + 2] = vec_B.z;
            s_B[load_b_row * BN + load_b_col + 3] = vec_B.w;
        } else {
            s_B[load_b_row * BN + load_b_col + 0] = 0.0f;
            s_B[load_b_row * BN + load_b_col + 1] = 0.0f;
            s_B[load_b_row * BN + load_b_col + 2] = 0.0f;
            s_B[load_b_row * BN + load_b_col + 3] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) r_A[i] = s_A[(ty * TM + i) * BK + k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) r_B[j] = s_B[k * BN + (tx * TN + j)];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    r_C[i][j] += r_A[i] * r_B[j];
                }
            }
        }
        __syncthreads();
    }

    int thread_row = by * BM + ty * TM;
    int thread_col = bx * BN + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int out_row = thread_row + i;
            int out_col = thread_col + j;
            if (out_row < M && out_col < N) {
                batch_C[out_row * N + out_col] = r_C[i][j];
            }
        }
    }
}


// ==========================================
// 2. GPU Worker: BMM Forward (Transposed B)
// ==========================================
__global__ void bmm_forward_transB_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int batch_stride_A, int batch_stride_B, int batch_stride_C) 
{
    // C = A * B^T
    // A: [M, K], B: [N, K]
    
    int batch_idx = blockIdx.z;
    const float* batch_A = A + batch_idx * batch_stride_A;
    const float* batch_B = B + batch_idx * batch_stride_B; // B is physically N x K
    float* batch_C = C + batch_idx * batch_stride_C;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN]; // We will transpose it into shared memory

    float r_C[TM][TN] = {0.0f};
    float r_A[TM];
    float r_B[TN];

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / 4; // Read rows of B (which are N dims)
    int load_b_col = (tid % 4) * 4;

    int num_tiles = K / BK;

    for (int t = 0; t < num_tiles; ++t) {
        // Load A
        int global_a_row = by * BM + load_a_row;
        int global_a_col = t * BK + load_a_col;
        
        if (global_a_row < M && global_a_col < K) {
            float4 vec_A = reinterpret_cast<const float4*>(&batch_A[global_a_row * K + global_a_col])[0];
            s_A[load_a_row * BK + load_a_col + 0] = vec_A.x;
            s_A[load_a_row * BK + load_a_col + 1] = vec_A.y;
            s_A[load_a_row * BK + load_a_col + 2] = vec_A.z;
            s_A[load_a_row * BK + load_a_col + 3] = vec_A.w;
        } else {
            s_A[load_a_row * BK + load_a_col + 0] = 0.0f;
            s_A[load_a_row * BK + load_a_col + 1] = 0.0f;
            s_A[load_a_row * BK + load_a_col + 2] = 0.0f;
            s_A[load_a_row * BK + load_a_col + 3] = 0.0f;
        }

        // Load B^T (Read B which is N x K, write to s_B which expects K x BN)
        int global_b_row = bx * BN + load_b_row; // Row in B corresponds to column in C (N dimension)
        int global_b_col = t * BK + load_b_col;  // Col in B corresponds to inner dimension K
        
        if (global_b_row < N && global_b_col < K) {
            float4 vec_B = reinterpret_cast<const float4*>(&batch_B[global_b_row * K + global_b_col])[0];
            // Write transposed into s_B
            s_B[(load_b_col + 0) * BN + load_b_row] = vec_B.x;
            s_B[(load_b_col + 1) * BN + load_b_row] = vec_B.y;
            s_B[(load_b_col + 2) * BN + load_b_row] = vec_B.z;
            s_B[(load_b_col + 3) * BN + load_b_row] = vec_B.w;
        } else {
            s_B[(load_b_col + 0) * BN + load_b_row] = 0.0f;
            s_B[(load_b_col + 1) * BN + load_b_row] = 0.0f;
            s_B[(load_b_col + 2) * BN + load_b_row] = 0.0f;
            s_B[(load_b_col + 3) * BN + load_b_row] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) r_A[i] = s_A[(ty * TM + i) * BK + k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) r_B[j] = s_B[k * BN + (tx * TN + j)];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    r_C[i][j] += r_A[i] * r_B[j];
                }
            }
        }
        __syncthreads();
    }

    int thread_row = by * BM + ty * TM;
    int thread_col = bx * BN + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int out_row = thread_row + i;
            int out_col = thread_col + j;
            if (out_row < M && out_col < N) {
                batch_C[out_row * N + out_col] = r_C[i][j];
            }
        }
    }
}

// Backward kernels for BMM (simplified naive for demonstration, would be tiled in prod)
__global__ void bmm_backward_A_kernel(float* dA, const float* dC, const float* B, int M, int N, int K, int b_stride_A, int b_stride_B, int b_stride_C, bool transB) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float dC_val = dC[batch_idx * b_stride_C + row * N + j];
            float B_val = transB ? B[batch_idx * b_stride_B + j * K + col] : B[batch_idx * b_stride_B + col * N + j];
            sum += dC_val * B_val;
        }
        atomicAdd(&dA[batch_idx * b_stride_A + row * K + col], sum);
    }
}

__global__ void bmm_backward_B_kernel(float* dB, const float* A, const float* dC, int M, int N, int K, int b_stride_A, int b_stride_B, int b_stride_C, bool transB) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int limit1 = transB ? N : K;
    int limit2 = transB ? K : N;
    
    if (row < limit1 && col < limit2) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            float A_val = A[batch_idx * b_stride_A + i * K + (transB ? col : row)];
            float dC_val = dC[batch_idx * b_stride_C + i * N + (transB ? row : col)];
            sum += A_val * dC_val;
        }
        atomicAdd(&dB[batch_idx * b_stride_B + row * limit2 + col], sum);
    }
}

// ==========================================
// 1. GPU Worker: BMM Forward (Scalar Fallback for Edge Cases)
// ==========================================
__global__ void bmm_forward_scalar_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int batch_stride_A, int batch_stride_B, int batch_stride_C, bool transB) 
{
    int batch_idx = blockIdx.z;
    const float* batch_A = A + batch_idx * batch_stride_A;
    const float* batch_B = B + batch_idx * batch_stride_B;
    float* batch_C = C + batch_idx * batch_stride_C;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = batch_A[row * K + k];
            float b_val = transB ? batch_B[col * K + k] : batch_B[k * N + col];
            sum += a_val * b_val;
        }
        batch_C[row * N + col] = sum;
    }
}

// ==========================================
// 3. Host Manager: Graph Builder
// ==========================================
std::shared_ptr<Tensor> bmm(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B, bool transB) {
    if (A->shape.size() < 2 || B->shape.size() < 2) {
        throw std::invalid_argument("BMM requires at least 2D tensors.");
    }
    
    int ndimA = A->shape.size();
    int ndimB = B->shape.size();
    
    int M = A->shape[ndimA - 2];
    int K = A->shape[ndimA - 1];
    
    int B_rows = B->shape[ndimB - 2];
    int B_cols = B->shape[ndimB - 1];
    
    int N = transB ? B_rows : B_cols;
    int K_check = transB ? B_cols : B_rows;
    
    if (K != K_check) {
        throw std::invalid_argument("Error: Inner matrix dimensions mismatch.");
    }

    int batches = 1;
    for (int i = 0; i < ndimA - 2; ++i) batches *= A->shape[i];

    std::vector<int> out_shape = A->shape;
    out_shape[ndimA - 2] = M;
    out_shape[ndimA - 1] = N;

    bool req_grad = A->requires_grad || B->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);

    int b_stride_A = (ndimA >= 3) ? A->strides[ndimA - 3] : M * K;
    int b_stride_B = (ndimB >= 3) ? B->strides[ndimB - 3] : 0;
    int b_stride_C = M * N; // Output is always allocated contiguously

    bool aligned = (K % 4 == 0) && (transB || (N % 4 == 0));

    if (aligned) {
        dim3 threads(16, 16);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM, batches);
        if (transB) {
            bmm_forward_transB_kernel<<<blocks, threads>>>(
                A->data.get(), B->data.get(), out->data.get(), M, N, K, b_stride_A, b_stride_B, b_stride_C);
        } else {
            bmm_forward_kernel<<<blocks, threads>>>(
                A->data.get(), B->data.get(), out->data.get(), M, N, K, b_stride_A, b_stride_B, b_stride_C);
        }
    } else {
        // Fallback to scalar kernel for non-aligned shapes (e.g. odd sequence lengths)
        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (M + 15) / 16, batches);
        bmm_forward_scalar_kernel<<<blocks, threads>>>(
            A->data.get(), B->data.get(), out->data.get(), M, N, K, b_stride_A, b_stride_B, b_stride_C, transB);
    }
    // cudaDeviceSynchronize();

    if (req_grad) {
        out->parents.push_back(A);
        out->parents.push_back(B);
        
        bool req_a = A->requires_grad;
        bool req_b = B->requires_grad;

        std::weak_ptr<Tensor> weak_out = out;
        out->backward_closure = [A, B, M, N, K, b_stride_A, b_stride_B, b_stride_C, batches, transB, req_a, req_b, weak_out]() {
            auto out = weak_out.lock();
            
            dim3 threadsBack(16, 16);
            if (req_a) {
                dim3 gridA((K + 15)/16, (M + 15)/16, batches);
                bmm_backward_A_kernel<<<gridA, threadsBack>>>(
                    A->grad.get(), out->grad.get(), B->data.get(), M, N, K, b_stride_A, b_stride_B, b_stride_C, transB);
            }
            if (req_b) {
                int limit1 = transB ? N : K;
                int limit2 = transB ? K : N;
                dim3 gridB((limit2 + 15)/16, (limit1 + 15)/16, batches);
                bmm_backward_B_kernel<<<gridB, threadsBack>>>(
                    B->grad.get(), A->data.get(), out->grad.get(), M, N, K, b_stride_A, b_stride_B, b_stride_C, transB);
            }
            // cudaDeviceSynchronize();
        };
    }
    return out;
}
