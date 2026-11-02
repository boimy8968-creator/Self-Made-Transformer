#include "../training/loss.h"
#include <cuda_runtime.h>
#include <cfloat>

// Forward kernel computes the loss bypassing intermediate probabilities allocation
__global__ void ce_forward_kernel(
    const float* logits, const int* targets, float* loss, 
    int num_rows, int vocab_size) {
    
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int target = targets[row];
    const float* row_logits = logits + row * vocab_size;

    // 1. Find local max per thread
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_logits[i]);
    }
    
    // Block-wide max reduction
    __shared__ float smem_max[1024];
    smem_max[threadIdx.x] = thread_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_max[threadIdx.x] = fmaxf(smem_max[threadIdx.x], smem_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_logit = smem_max[0];
    if (isnan(max_logit)) max_logit = 0.0f;

    // 2. Compute local sum of exps per thread
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        thread_sum += expf(row_logits[i] - max_logit);
    }
    
    // Block-wide sum reduction
    __shared__ float smem_sum[1024];
    smem_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_sum[threadIdx.x] += smem_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum_exp = smem_sum[0];
    if (sum_exp <= 0.0f || isnan(sum_exp)) sum_exp = 1.0f;

    // 3. Compute Negative Log-Likelihood via Log-Sum-Exp
    if (threadIdx.x == 0) {
        float target_logit = (target >= 0 && target < vocab_size) ? row_logits[target] : (max_logit - 20.0f);
        // NLL = log(sum_exp) - (target_logit - max_logit)
        // Strictly prevents exp underflow to 0 and log(0) -> -inf
        float nll = logf(fmaxf(sum_exp, 1.0f)) - (target_logit - max_logit);
        atomicAdd(loss, nll / (float)num_rows);
    }
}

// Backward kernel recomputes softmax in-place to avoid saving massive probability tensor
__global__ void ce_backward_kernel(
    const float* logits, const int* targets, float* logits_grad, const float* loss_grad, 
    int num_rows, int vocab_size) {
    
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int target = targets[row];
    const float* row_logits = logits + row * vocab_size;
    float* row_grad = logits_grad + row * vocab_size;

    // 1. Find local max
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_logits[i]);
    }
    
    __shared__ float smem_max[1024];
    smem_max[threadIdx.x] = thread_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_max[threadIdx.x] = fmaxf(smem_max[threadIdx.x], smem_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_logit = smem_max[0];
    if (isnan(max_logit)) max_logit = 0.0f;

    // 2. Local sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        thread_sum += expf(row_logits[i] - max_logit);
    }
    
    __shared__ float smem_sum[1024];
    smem_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_sum[threadIdx.x] += smem_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum_exp = smem_sum[0];
    if (sum_exp <= 0.0f || isnan(sum_exp)) sum_exp = 1.0f;

    // Upstream gradient from loss
    float l_grad = loss_grad ? *loss_grad : 1.0f;

    // 3. Compute and accumulate gradients
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float p = expf(row_logits[i] - max_logit) / sum_exp;
        float target_val = (i == target) ? 1.0f : 0.0f;
        float grad = p - target_val;
        atomicAdd(&row_grad[i], (grad / (float)num_rows) * l_grad);
    }
}

std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> logits, const int* d_targets) {
    auto loss = std::make_shared<Tensor>(std::vector<int>{1}, logits->requires_grad);
    cudaMemsetAsync(loss->data.get(), 0, sizeof(float), 0);

    int num_rows = 1;
    for (size_t i = 0; i < logits->shape.size() - 1; ++i) {
        num_rows *= logits->shape[i];
    }
    int vocab_size = logits->shape.back();
    
    // Exactly 1024 threads to match power-of-2 reductions in kernels
    int threads = 1024;
    int blocks = num_rows;

    ce_forward_kernel<<<blocks, threads>>>(logits->data.get(), d_targets, loss->data.get(), num_rows, vocab_size);
    // cudaDeviceSynchronize();

    if (logits->requires_grad) {
        loss->parents.push_back(logits);

        // Closure captures raw pointers / smart pointers for delayed execution
        std::weak_ptr<Tensor> weak_loss = loss;
        loss->backward_closure = [logits, d_targets, num_rows, vocab_size, blocks, threads, weak_loss]() {
            auto loss = weak_loss.lock();
            
            ce_backward_kernel<<<blocks, threads>>>(
                logits->data.get(),
                d_targets,
                logits->grad.get(),
                loss->grad.get(),
                num_rows,
                vocab_size
            );
        };
    }
    return loss;
}

