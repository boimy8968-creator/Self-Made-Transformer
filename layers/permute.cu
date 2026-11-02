#include "../layers/permute.h"
#include <cuda_runtime.h>
#include <stdexcept>

#define MAX_DIMS 8

struct PermuteInfo {
    int ndim;
    int out_shape[MAX_DIMS];
    int in_strides[MAX_DIMS]; // mapped to out dims
    int out_strides[MAX_DIMS];
};

__device__ int get_in_index(int out_idx, const PermuteInfo& info) {
    int in_idx = 0;
    int temp = out_idx;
    for (int i = info.ndim - 1; i >= 0; --i) {
        int coord = temp % info.out_shape[i];
        in_idx += coord * info.in_strides[i];
        temp /= info.out_shape[i];
    }
    return in_idx;
}

__global__ void permute_forward_kernel(const float* in, float* out, PermuteInfo info, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int in_idx = get_in_index(idx, info);
        out[idx] = in[in_idx];
    }
}

__global__ void permute_backward_kernel(float* grad_in, const float* grad_out, PermuteInfo info, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int in_idx = get_in_index(idx, info);
        atomicAdd(&grad_in[in_idx], grad_out[idx]);
    }
}

std::shared_ptr<Tensor> permute(std::shared_ptr<Tensor> input, const std::vector<int>& dims) {
    if (dims.size() != input->shape.size()) {
        throw std::invalid_argument("Permute dimensions must match input dimensions.");
    }

    int ndim = dims.size();
    if (ndim > MAX_DIMS) throw std::invalid_argument("Max dimension exceeded");

    PermuteInfo info;
    info.ndim = ndim;

    std::vector<int> out_shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        out_shape[i] = input->shape[dims[i]];
        info.out_shape[i] = out_shape[i];
        info.in_strides[i] = input->strides[dims[i]];
    }

    auto out = std::make_shared<Tensor>(out_shape, input->requires_grad);
    
    for (int i = 0; i < ndim; ++i) {
        info.out_strides[i] = out->strides[i];
    }

    int total_size = out->total_size;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;

    permute_forward_kernel<<<blocks, threads>>>(input->data.get(), out->data.get(), info, total_size);
    // cudaDeviceSynchronize();

    if (input->requires_grad) {
        out->parents.push_back(input);
        std::weak_ptr<Tensor> weak_out = out;
        out->backward_closure = [input, info, total_size, blocks, threads, weak_out]() {
            auto out = weak_out.lock();
            
            permute_backward_kernel<<<blocks, threads>>>(input->grad.get(), out->grad.get(), info, total_size);
            // cudaDeviceSynchronize();
        };
    }

    return out;
}

// Memory contiguous reshaping
__global__ void copy_kernel(const float* in, float* out, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) out[idx] = in[idx];
}

std::shared_ptr<Tensor> view(std::shared_ptr<Tensor> input, const std::vector<int>& shape) {
    int expected_size = 1;
    for (int s : shape) expected_size *= s;
    if (expected_size != input->total_size) {
        throw std::invalid_argument("View shape must have same total size as input.");
    }

    // Since our system requires contiguous tensors for vectorized BMM, view performs a physical copy to ensure safety.
    auto out = std::make_shared<Tensor>(shape, input->requires_grad);
    
    int threads = 256;
    int blocks = (input->total_size + threads - 1) / threads;
    
    copy_kernel<<<blocks, threads>>>(input->data.get(), out->data.get(), input->total_size);
    // cudaDeviceSynchronize();

    if (input->requires_grad) {
        out->parents.push_back(input);
        std::weak_ptr<Tensor> weak_out = out;
        out->backward_closure = [input, blocks, threads, weak_out]() {
            auto out = weak_out.lock();
            
            // For view, backward is just adding gradients back linearly
            permute_backward_kernel<<<blocks, threads>>>(input->grad.get(), out->grad.get(), PermuteInfo{1, {input->total_size}, {1}, {1}}, input->total_size);
            // cudaDeviceSynchronize();
        };
    }

    return out;
}
