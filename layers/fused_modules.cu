#include "../layers/fused_modules.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

// ----------------------------------------------------
// Fused Add + GELU
// ----------------------------------------------------
__device__ inline float gelu_math(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_add_gelu_fwd_kernel(const float* __restrict__ in, const float* __restrict__ bias, float* __restrict__ out, int N, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int bias_idx = idx % N; 
        float val = in[idx] + bias[bias_idx];
        out[idx] = gelu_math(val);
    }
}

std::shared_ptr<Tensor> fused_add_gelu(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> bias) {
    auto out = std::make_shared<Tensor>(input->shape, input->requires_grad || bias->requires_grad);
    int total_size = input->total_size;
    int N = bias->total_size;
    
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    fused_add_gelu_fwd_kernel<<<blocks, threads>>>(input->data.get(), bias->data.get(), out->data.get(), N, total_size);
    // cudaDeviceSynchronize();
    
    // Note: backward pass mapping omitted to keep focus strictly on forward latency reduction as instructed,
    // but would follow the identical closure time-capsule logic.
    return out;
}

// ----------------------------------------------------
// Fused Scale + Softmax
// ----------------------------------------------------
__inline__ __device__ float warpReduceMaxF(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float blockReduceMaxF(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceMaxF(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warpReduceMaxF(val);
    return val;
}

__inline__ __device__ float warpReduceSumF(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float blockReduceSumF(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSumF(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSumF(val);
    return val;
}

__global__ void fused_scale_softmax_fwd_kernel(const float* __restrict__ in, float* __restrict__ out, float scale, int rows, int cols) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* in_row = in + row * cols;
    float* out_row = out + row * cols;
    
    // 1. Max (Fused with Scale)
    float max_val = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = max(max_val, in_row[i] * scale);
    }
    max_val = blockReduceMaxF(max_val, shared_mem);
    
    // 2. Sum (Fused with Scale)
    float sum_val = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum_val += expf((in_row[i] * scale) - max_val);
    }
    sum_val = blockReduceSumF(sum_val, shared_mem);
    
    // 3. Write Output
    for (int i = tid; i < cols; i += blockDim.x) {
        out_row[i] = expf((in_row[i] * scale) - max_val) / sum_val;
    }
}

std::shared_ptr<Tensor> fused_scale_softmax(std::shared_ptr<Tensor> input, float scale_factor) {
    auto out = std::make_shared<Tensor>(input->shape, input->requires_grad);
    int cols = input->shape.back();
    int rows = input->total_size / cols;
    
    int threadsPerBlock = 256;
    if (cols > 256) threadsPerBlock = 512;
    if (cols > 512) threadsPerBlock = 1024;
    
    size_t shared_mem_size = (threadsPerBlock / 32) * sizeof(float);
    
    fused_scale_softmax_fwd_kernel<<<rows, threadsPerBlock, shared_mem_size>>>(
        input->data.get(), out->data.get(), scale_factor, rows, cols);
    // cudaDeviceSynchronize();
    return out;
}
