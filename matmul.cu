#include "matmul.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Hardware Geometry
#define BM 64  // Block Row Tile Size
#define BN 64  // Block Col Tile Size
#define BK 16  // Block Inner Dimension
#define TM 4   // Thread Row Tile Size (Registers)
#define TN 4   // Thread Col Tile Size (Registers)

// ==========================================
// 1. GPU Worker: Forward Vectorized GEMM
// ==========================================
__global__ void matmul_register_tiled_vectorized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * blockDim.x + tx;

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];

    float r_C[TM][TN] = { 0.0f };
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

        float4 vec_A = reinterpret_cast<const float4*>(&A[global_a_row * K + global_a_col])[0];
        s_A[load_a_row * BK + load_a_col + 0] = vec_A.x;
        s_A[load_a_row * BK + load_a_col + 1] = vec_A.y;
        s_A[load_a_row * BK + load_a_col + 2] = vec_A.z;
        s_A[load_a_row * BK + load_a_col + 3] = vec_A.w;

        int global_b_row = t * BK + load_b_row;
        int global_b_col = bx * BN + load_b_col;

        float4 vec_B = reinterpret_cast<const float4*>(&B[global_b_row * N + global_b_col])[0];
        s_B[load_b_row * BN + load_b_col + 0] = vec_B.x;
        s_B[load_b_row * BN + load_b_col + 1] = vec_B.y;
        s_B[load_b_row * BN + load_b_col + 2] = vec_B.z;
        s_B[load_b_row * BN + load_b_col + 3] = vec_B.w;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                r_A[i] = s_A[(ty * TM + i) * BK + k];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                r_B[j] = s_B[k * BN + (tx * TN + j)];
            }

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
                C[out_row * N + out_col] = r_C[i][j];
            }
        }
    }
}

// ==========================================
// 2. GPU Workers: Backward Gradient Accumulation
// ==========================================

// Compute dA += dC * B^T  (dA shape: M x K)
__global__ void matmul_backward_A_kernel(
    float* dA, const float* dC, const float* B,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0 to K-1

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += dC[row * N + j] * B[col * N + j];
        }
        dA[row * K + col] += sum;
    }
}

// Compute dB += A^T * dC  (dB shape: K x N)
__global__ void matmul_backward_B_kernel(
    float* dB, const float* A, const float* dC,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to K-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0 to N-1

    if (row < K && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += A[i * K + row] * dC[i * N + col];
        }
        dB[row * N + col] += sum;
    }
}

// ==========================================
// 3. Host Manager: Graph Builder
// ==========================================
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B) {
    if (A->shape.size() != 2 || B->shape.size() != 2) {
        throw std::invalid_argument("Error: matmul requires 2D Tensors.");
    }

    int M = A->shape[0];
    int K = A->shape[1];
    int B_rows = B->shape[0];
    int N = B->shape[1];

    if (K != B_rows) {
        throw std::invalid_argument("Error: Inner matrix dimensions mismatch (A cols != B rows).");
    }

    if (K % 4 != 0 || N % 4 != 0) {
        throw std::invalid_argument("Error: Vectorized float4 kernel requires K and N to be multiples of 4.");
    }

    bool out_requires_grad = A->requires_grad || B->requires_grad;

    // Allocate Output Tensor on VRAM
    auto out = std::make_shared<Tensor>(std::vector<int>{M, N}, out_requires_grad);

    // Launch Forward Matrix Multiplication
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 64 - 1) / 64, (M + 64 - 1) / 64); // Assuming BN and BM are 64

    matmul_register_tiled_vectorized << <numBlocks, threadsPerBlock >> > (
        A->data.get(),
        B->data.get(),
        out->data.get(),
        M, N, K
        );

    // cudaDeviceSynchronize();

    // ==========================================
    // 4. Build Autograd Graph
    // ==========================================
    if (out_requires_grad) {
        out->parents.push_back(A);
        out->parents.push_back(B);

        // Store Lambda Time Capsule for Backward Pass
        std::weak_ptr<Tensor> weak_out = out;
        out->backward_closure = [A, B, M, N, K, weak_out]() {
            auto out = weak_out.lock();
            
            dim3 block(16, 16);

            if (A->requires_grad) {
                // dA += dOut * B^T
                dim3 gridA((K + 15) / 16, (M + 15) / 16);
                matmul_backward_A_kernel << <gridA, block >> > (
                    A->grad.get(),
                    out->grad.get(),
                    B->data.get(),
                    M, N, K
                    );
            }

            if (B->requires_grad) {
                // dB += A^T * dOut
                dim3 gridB((N + 15) / 16, (K + 15) / 16);
                matmul_backward_B_kernel << <gridB, block >> > (
                    B->grad.get(),
                    A->data.get(),
                    out->grad.get(),
                    M, N, K
                    );
            }

            // cudaDeviceSynchronize();
            };
    }

    return out;
}
