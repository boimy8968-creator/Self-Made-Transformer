#include "../inference/sampling.h"
#include <cuda_runtime.h>
#include <cfloat>

__global__ void temperature_scale_kernel(float* logits, float temperature, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        logits[idx] /= temperature;
    }
}

void temperature_scale(std::shared_ptr<Tensor> logits, float temperature) {
    int size = logits->total_size;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    temperature_scale_kernel<<<blocks, threads>>>(logits->data.get(), temperature, size);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------
// TOP-P SAMPLING
// ---------------------------------------------------------------------

// 1. Initialize ProbIndex array and apply softmax
__global__ void init_probs_kernel(const float* logits, ProbIndex* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].prob = logits[idx];
        data[idx].index = idx;
    }
    // Note: In a full pipeline, we would apply softmax here. 
    // Assuming logits are pre-softmaxed or we do a quick local softmax.
    // For exactness, let's assume `logits` passed here are already probabilities.
}

// 2. Bitonic Sort Step
__global__ void bitonic_sort_step(ProbIndex* data, int j, int k, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int ixj = i ^ j;

    if (ixj > i && i < n && ixj < n) {
        bool swap = false;
        if ((i & k) == 0) {
            if (data[i].prob < data[ixj].prob) swap = true;
        } else {
            if (data[i].prob > data[ixj].prob) swap = true;
        }
        
        if (swap) {
            ProbIndex temp = data[i];
            data[i] = data[ixj];
            data[ixj] = temp;
        }
    }
}

// 3. Block Prefix Sum and Masking
__global__ void block_prefix_sum_mask_kernel(ProbIndex* data, int n, float p) {
    extern __shared__ float s_probs[];
    float block_sum = 0.0f;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        s_probs[threadIdx.x] = data[i].prob;
        __syncthreads();
        
        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            float t = 0.0f;
            if (threadIdx.x >= offset) {
                t = s_probs[threadIdx.x - offset];
            }
            __syncthreads();
            if (threadIdx.x >= offset) {
                s_probs[threadIdx.x] += t;
            }
            __syncthreads();
        }
        
        float cdf = block_sum + s_probs[threadIdx.x];
        
        if (cdf > p && i > 0) {
            data[i].prob = 0.0f; // Mask out the tail
        }
        
        block_sum += s_probs[blockDim.x - 1];
        __syncthreads();
    }
}

// 4. Final Sampling
__global__ void sample_kernel(ProbIndex* data, int n, float rand_val, int* out_token) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) sum += data[i].prob;
        
        float target = rand_val * sum;
        float current = 0.0f;
        *out_token = data[0].index; // Fallback to highest probability
        
        for (int i = 0; i < n; ++i) {
            current += data[i].prob;
            if (current >= target && data[i].prob > 0.0f) {
                *out_token = data[i].index;
                break;
            }
        }
    }
}

int top_p_sample(std::shared_ptr<Tensor> probs, float p, float rand_val) {
    int n = probs->total_size;
    int next_pow2 = 1;
    while (next_pow2 < n) next_pow2 <<= 1;

    ProbIndex* d_data;
    cudaMalloc(&d_data, next_pow2 * sizeof(ProbIndex));
    cudaMemset(d_data, 0, next_pow2 * sizeof(ProbIndex));
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // 1. Initialize
    init_probs_kernel<<<blocks, threads>>>(probs->data.get(), d_data, n);
    cudaDeviceSynchronize();
    
    // 2. Bitonic Sort (Descending)
    int sort_blocks = (next_pow2 + threads - 1) / threads;
    for (int k = 2; k <= next_pow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<sort_blocks, threads>>>(d_data, j, k, next_pow2);
            cudaDeviceSynchronize();
        }
    }
    
    // 3. Block Prefix Sum and Masking
    // We use a single block loop to compute cumulative sum sequentially across the entire array
    block_prefix_sum_mask_kernel<<<1, 256, 256 * sizeof(float)>>>(d_data, n, p);
    cudaDeviceSynchronize();
    
    // 4. Sample Token
    int* d_out_token;
    cudaMalloc(&d_out_token, sizeof(int));
    sample_kernel<<<1, 1>>>(d_data, n, rand_val, d_out_token);
    cudaDeviceSynchronize();
    
    int h_out_token;
    cudaMemcpy(&h_out_token, d_out_token, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_out_token);
    
    return h_out_token;
}
