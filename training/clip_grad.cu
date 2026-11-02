#include "../training/clip_grad.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ void sum_squares_kernel(const float* __restrict__ grad, float* __restrict__ partial_sums, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    if (i < size) {
        float g = grad[i];
        sum = g * g;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void scale_gradients_kernel(float* __restrict__ grad, float scale, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad[i] *= scale;
    }
}

void clip_grad_norm(const std::vector<std::shared_ptr<Tensor>>& parameters, float max_norm) {
    float total_sq_norm = 0.0f;
    
    // Accumulate squared norms
    for (auto& p : parameters) {
        if (!p->requires_grad || !p->grad) continue;
        
        int size = p->total_size;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        
        float* d_partial;
        cudaMalloc(&d_partial, blocks * sizeof(float));
        
        sum_squares_kernel<<<blocks, threads, threads * sizeof(float)>>>(p->grad.get(), d_partial, size);
        
        std::vector<float> h_partial(blocks);
        cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_partial);
        
        for (float val : h_partial) {
            total_sq_norm += val;
        }
    }
    
    float total_norm = std::sqrt(total_sq_norm);
    
    if (total_norm > max_norm) {
        float scale_factor = max_norm / (total_norm + 1e-6f);
        for (auto& p : parameters) {
            if (!p->requires_grad || !p->grad) continue;
            int size = p->total_size;
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            scale_gradients_kernel<<<blocks, threads>>>(p->grad.get(), scale_factor, size);
        }
    }
}
