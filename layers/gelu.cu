#include "../layers/gelu.h"
#include <cuda_runtime.h>
#include <math.h>

#define GELU_COEF_A 0.7978845608f  // sqrt(2/pi)
#define GELU_COEF_B 0.0356774081f  // sqrt(2/pi) * 0.044715

__device__ float gelu_fwd(float x) {
    float cube = x * x * x;
    return 0.5f * x * (1.0f + tanhf(GELU_COEF_A * x + GELU_COEF_B * cube));
}

__device__ float gelu_bwd(float x) {
    float x_sq = x * x;
    float cube = x_sq * x;
    float inner = GELU_COEF_A * x + GELU_COEF_B * cube;
    float tanh_inner = tanhf(inner);
    float sech_sq = 1.0f - tanh_inner * tanh_inner;
    
    float left = 0.5f * (1.0f + tanh_inner);
    float right = 0.5f * x * sech_sq * (GELU_COEF_A + 3.0f * GELU_COEF_B * x_sq);
    return left + right;
}

__global__ void gelu_forward_kernel(const float* in, float* out, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        out[idx] = gelu_fwd(in[idx]);
    }
}

__global__ void gelu_backward_kernel(float* grad_in, const float* in, const float* grad_out, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        atomicAdd(&grad_in[idx], grad_out[idx] * gelu_bwd(in[idx]));
    }
}

std::shared_ptr<Tensor> gelu(std::shared_ptr<Tensor> input) {
    auto out = std::make_shared<Tensor>(input->shape, input->requires_grad);
    
    int total_size = input->total_size;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    
    gelu_forward_kernel<<<blocks, threads>>>(input->data.get(), out->data.get(), total_size);
    // cudaDeviceSynchronize();

    if (input->requires_grad) {
        out->parents.push_back(input);
        std::weak_ptr<Tensor> weak_out = out;
        out->backward_closure = [input, total_size, blocks, threads, weak_out]() {
            auto out = weak_out.lock();
            
            gelu_backward_kernel<<<blocks, threads>>>(input->grad.get(), input->data.get(), out->grad.get(), total_size);
            // cudaDeviceSynchronize();
        };
    }
    return out;
}
