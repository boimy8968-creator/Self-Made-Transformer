#include "../layers/scale.h"
#include <cuda_runtime.h>

__global__ void scale_forward_kernel(const float* in, float* out, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] * scalar;
    }
}

__global__ void scale_backward_kernel(float* grad_in, const float* grad_out, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&grad_in[idx], grad_out[idx] * scalar);
    }
}

std::shared_ptr<Tensor> scale(std::shared_ptr<Tensor> input, float scalar) {
    auto out = std::make_shared<Tensor>(input->shape, input->requires_grad);
    int threads = 256;
    int blocks = (input->total_size + threads - 1) / threads;
    
    scale_forward_kernel<<<blocks, threads>>>(input->data.get(), out->data.get(), scalar, input->total_size);
    // cudaDeviceSynchronize();
    
    if (input->requires_grad) {
        out->parents.push_back(input);
        std::weak_ptr<Tensor> weak_out = out;
        out->backward_closure = [input, scalar, blocks, threads, weak_out]() {
            auto out = weak_out.lock();
            
            scale_backward_kernel<<<blocks, threads>>>(input->grad.get(), out->grad.get(), scalar, input->total_size);
            // cudaDeviceSynchronize();
        };
    }
    return out;
}
