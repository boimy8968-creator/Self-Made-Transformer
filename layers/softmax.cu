#include "../layers/softmax.h"
#include <cuda_runtime.h>
#include <float.h>
#include <stdexcept>
#include <math.h>

// ==========================================
// Warp & Block Reduction Primitives
// ==========================================
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float blockReduceMax(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__inline__ __device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// ==========================================
// 1. GPU Worker: Softmax Forward
// ==========================================
__global__ void softmax_forward_kernel(const float* in, float* out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    const float* in_row = in + row * cols;
    float* out_row = out + row * cols;

    __shared__ float shared_mem[32]; // For block reduction (max 1024 threads / 32 = 32 warps)

    // 1. Find Max of valid (unmasked) scores for numerical stability
    float max_val = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = in_row[i];
        if (val > -1e8f && !isnan(val)) {
            max_val = fmaxf(max_val, val);
        }
    }
    max_val = blockReduceMax(max_val, shared_mem);
    
    // Broadcast max to all threads
    __shared__ float s_max;
    if (tid == 0) {
        if (max_val <= -1e8f || isnan(max_val)) max_val = 0.0f;
        s_max = max_val;
    }
    __syncthreads();
    max_val = s_max;

    // 2. Sum of Exponentials (only over valid unmasked scores)
    float sum_exp = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = in_row[i];
        if (val > -1e8f && !isnan(val)) {
            float safe_score = val - max_val;
            sum_exp += expf(safe_score);
        }
    }
    sum_exp = blockReduceSum(sum_exp, shared_mem);
    
    // Broadcast sum to all threads
    __shared__ float s_sum;
    if (tid == 0) {
        if (sum_exp <= 0.0f || isnan(sum_exp)) sum_exp = 1.0f;
        s_sum = sum_exp;
    }
    __syncthreads();
    sum_exp = s_sum;

    // 3. Compute Probabilities
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = in_row[i];
        if (val > -1e8f && !isnan(val)) {
            float safe_score = val - max_val;
            float prob = expf(safe_score) / sum_exp;
            out_row[i] = prob;
        } else {
            out_row[i] = 0.0f;
        }
    }
}

// ==========================================
// 2. GPU Worker: Softmax Backward (Jacobian)
// ==========================================
__global__ void softmax_backward_kernel(float* d_in, const float* out, const float* d_out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    const float* out_row = out + row * cols;
    const float* d_out_row = d_out + row * cols;
    float* d_in_row = d_in + row * cols;

    __shared__ float shared_mem[32];

    // 1. Dot product: sum(P * dOut)
    float sum_val = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float p = out_row[i];
        if (p > 0.0f) {
            sum_val += p * d_out_row[i];
        }
    }
    sum_val = blockReduceSum(sum_val, shared_mem);
    
    __shared__ float s_sum;
    if (tid == 0) s_sum = sum_val;
    __syncthreads();
    sum_val = s_sum;

    // 2. Compute local gradients: P * (dOut - sum)
    for (int i = tid; i < cols; i += blockDim.x) {
        float p = out_row[i];
        if (p > 0.0f) {
            float grad = p * (d_out_row[i] - sum_val);
            atomicAdd(&d_in_row[i], grad);
        }
    }
}

// ==========================================
// 3. Host Manager: Graph Builder
// ==========================================
std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> input) {
    if (input->shape.size() < 1) {
        throw std::invalid_argument("Error: Softmax requires at least a 1D tensor.");
    }
    
    int cols = input->shape.back(); // Softmax across the last dimension
    int rows = input->total_size / cols;
    
    auto out = std::make_shared<Tensor>(input->shape, input->requires_grad);
    
    // One block per row. Configure threads per block based on col size.
    int threadsPerBlock = 256;
    if (cols > 256) threadsPerBlock = 512;
    if (cols > 512) threadsPerBlock = 1024; // Max hardware limit
    
    int blocks = rows;

    softmax_forward_kernel<<<blocks, threadsPerBlock>>>(
        input->data.get(),
        out->data.get(),
        rows,
        cols
    );
    
    // cudaDeviceSynchronize(); // Synchronize to ensure values are ready

    // ==========================================
    // 4. Build the Autograd Graph
    // ==========================================
    if (input->requires_grad) {
        out->parents.push_back(input);
        
        std::weak_ptr<Tensor> weak_out = out;
        out->backward_closure = [input, rows, cols, blocks, threadsPerBlock, weak_out]() {
            auto out = weak_out.lock();
            
            softmax_backward_kernel<<<blocks, threadsPerBlock>>>(
                input->grad.get(),
                out->data.get(),
                out->grad.get(),
                rows,
                cols
            );
            // cudaDeviceSynchronize();
        };
    }
    
    return out;
}
