#include "../layers/layernorm.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <math.h>

__inline__ __device__ float warpReduceSumLN(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float blockReduceSumLN(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSumLN(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSumLN(val);
    return val;
}

__global__ void layernorm_forward_kernel(
    const float* in, const float* weight, const float* bias, float* out, 
    float* mean_cache, float* inv_std_cache,
    int rows, int cols, float eps) 
{
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    const float* in_row = in + row * cols;
    float* out_row = out + row * cols;

    __shared__ float shared_mem[32];

    // 1. Mean
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sum += in_row[i];
    sum = blockReduceSumLN(sum, shared_mem);
    
    __shared__ float s_mean;
    if (tid == 0) {
        s_mean = sum / cols;
        mean_cache[row] = s_mean; // Cache for backward
    }
    __syncthreads();
    float mean = s_mean;

    // 2. Variance
    float var_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = in_row[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = blockReduceSumLN(var_sum, shared_mem);
    
    __shared__ float s_inv_std;
    if (tid == 0) {
        float var = var_sum / cols;
        s_inv_std = 1.0f / sqrtf(fmaxf(var, 0.0f) + 1e-5f);
        inv_std_cache[row] = s_inv_std; // Cache for backward
    }
    __syncthreads();
    float inv_std = s_inv_std;

    // 3. Normalize & Scale/Shift
    for (int i = tid; i < cols; i += blockDim.x) {
        float normalized = (in_row[i] - mean) * inv_std;
        float w = weight ? weight[i] : 1.0f;
        float b = bias ? bias[i] : 0.0f;
        out_row[i] = normalized * w + b;
    }
}

__global__ void layernorm_backward_kernel(
    float* d_in, float* d_weight, float* d_bias, 
    const float* d_out, const float* in, const float* weight, 
    const float* mean_cache, const float* inv_std_cache,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    const float* in_row = in + row * cols;
    const float* d_out_row = d_out + row * cols;
    float* d_in_row = d_in + row * cols;

    float mean = mean_cache[row];
    float inv_std = inv_std_cache[row];
    if (isnan(mean)) mean = 0.0f;
    if (isnan(inv_std) || isinf(inv_std) || inv_std <= 0.0f) {
        inv_std = 1.0f / sqrtf(1e-5f);
    }

    __shared__ float shared_mem[32];

    // Compute sums for dx
    float dx_hat_sum = 0.0f;
    float dx_hat_x_sum = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        float dout_val = d_out_row[i];
        
        float dx_hat = dout_val * w;
        float x_hat = (in_row[i] - mean) * inv_std;
        
        dx_hat_sum += dx_hat;
        dx_hat_x_sum += dx_hat * x_hat;
        
        if (d_weight) atomicAdd(&d_weight[i], dout_val * x_hat);
        if (d_bias) atomicAdd(&d_bias[i], dout_val);
    }
    
    dx_hat_sum = blockReduceSumLN(dx_hat_sum, shared_mem);
    
    __shared__ float s_dx_hat_sum;
    if (tid == 0) s_dx_hat_sum = dx_hat_sum;
    __syncthreads();
    
    dx_hat_x_sum = blockReduceSumLN(dx_hat_x_sum, shared_mem);
    
    __shared__ float s_dx_hat_x_sum;
    if (tid == 0) s_dx_hat_x_sum = dx_hat_x_sum;
    __syncthreads();

    // Compute dx
    float inv_N = 1.0f / cols;
    for (int i = tid; i < cols; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        float dx_hat = d_out_row[i] * w;
        float x_hat = (in_row[i] - mean) * inv_std;
        
        float dx = inv_std * (dx_hat - inv_N * (s_dx_hat_sum + x_hat * s_dx_hat_x_sum));
        atomicAdd(&d_in_row[i], dx);
    }
}

std::shared_ptr<Tensor> layernorm(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias, float eps) {
    int cols = input->shape.back();
    int rows = input->total_size / cols;

    bool req_grad = input->requires_grad || (weight && weight->requires_grad) || (bias && bias->requires_grad);
    auto out = std::make_shared<Tensor>(input->shape, req_grad);

    // Alloc caches for backward pass
    float* mean_cache;
    float* inv_std_cache;
    cudaMallocAsync(&mean_cache, rows * sizeof(float), 0);
    cudaMallocAsync(&inv_std_cache, rows * sizeof(float), 0);

    int threads = 256;
    if (cols > 256) threads = 512;
    if (cols > 512) threads = 1024;
    int blocks = rows;

    const float* w_ptr = weight ? weight->data.get() : nullptr;
    const float* b_ptr = bias ? bias->data.get() : nullptr;

    layernorm_forward_kernel<<<blocks, threads>>>(
        input->data.get(), w_ptr, b_ptr, out->data.get(),
        mean_cache, inv_std_cache, rows, cols, eps
    );
    // cudaDeviceSynchronize();

    if (req_grad) {
        out->parents.push_back(input);
        if (weight) out->parents.push_back(weight);
        if (bias) out->parents.push_back(bias);
        
        bool r_in = input->requires_grad;
        float* din_ptr = r_in ? input->grad.get() : nullptr;
        float* dw_ptr = (weight && weight->requires_grad) ? weight->grad.get() : nullptr;
        float* db_ptr = (bias && bias->requires_grad) ? bias->grad.get() : nullptr;

        // Wrap caches in shared_ptr to automatically free them when the time capsule is destroyed
        std::shared_ptr<float> auto_mean(mean_cache, [](float* p){ cudaFreeAsync(p, 0); });
        std::shared_ptr<float> auto_inv_std(inv_std_cache, [](float* p){ cudaFreeAsync(p, 0); });

        std::weak_ptr<Tensor> weak_out = out;
        out->backward_closure = [din_ptr, dw_ptr, db_ptr, input, weight, rows, cols, blocks, threads, auto_mean, auto_inv_std, weak_out]() {
            auto out = weak_out.lock();
            
            const float* w_ptr_bwd = weight ? weight->data.get() : nullptr;
            layernorm_backward_kernel<<<blocks, threads>>>(
                din_ptr, dw_ptr, db_ptr,
                out->grad.get(), input->data.get(), w_ptr_bwd,
                auto_mean.get(), auto_inv_std.get(),
                rows, cols
            );
            // cudaDeviceSynchronize();
        };
    } else {
        cudaFreeAsync(mean_cache, 0);
        cudaFreeAsync(inv_std_cache, 0);
    }

    return out;
}

