#include "../layers/add.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <algorithm>

#define MAX_DIMS 8

// Struct to pass broadcast dimensions entirely by value to the GPU
struct BroadcastInfo {
    int ndim;
    int shape[MAX_DIMS];
    int strides_a[MAX_DIMS];
    int strides_b[MAX_DIMS];
};

__device__ void get_indices(int idx, const BroadcastInfo& info, int& idx_a, int& idx_b) {
    idx_a = 0;
    idx_b = 0;
    int temp = idx;
    for (int i = info.ndim - 1; i >= 0; --i) {
        int coord = temp % info.shape[i];
        idx_a += coord * info.strides_a[i];
        idx_b += coord * info.strides_b[i];
        temp /= info.shape[i];
    }
}

// ==========================================
// 1. GPU Worker: Broadcast Add Forward
// ==========================================
__global__ void broadcast_add_forward_kernel(const float* A, const float* B, float* C, BroadcastInfo info, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int idx_a, idx_b;
        get_indices(idx, info, idx_a, idx_b);
        C[idx] = A[idx_a] + B[idx_b];
    }
}

// ==========================================
// 2. GPU Worker: Broadcast Add Backward
// ==========================================
__global__ void broadcast_add_backward_kernel(float* grad_a, float* grad_b, const float* grad_out, BroadcastInfo info, int total_size, bool req_a, bool req_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int idx_a, idx_b;
        get_indices(idx, info, idx_a, idx_b);
        float g = grad_out[idx];
        
        // Use atomicAdd because multiple threads might write to the same broadcasted gradient
        if (req_a) atomicAdd(&grad_a[idx_a], g);
        if (req_b) atomicAdd(&grad_b[idx_b], g);
    }
}

// ==========================================
// 3. Host Manager: Graph Builder
// ==========================================
std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B) {
    int ndim = std::max(A->shape.size(), B->shape.size());
    if (ndim > MAX_DIMS) {
        throw std::invalid_argument("Error: Broadcast Add supports up to 8 dimensions.");
    }

    std::vector<int> out_shape(ndim);
    BroadcastInfo info;
    info.ndim = ndim;

    int offset_a = ndim - A->shape.size();
    int offset_b = ndim - B->shape.size();

    int total_size = 1;
    for (int i = 0; i < ndim; ++i) {
        int dim_a = (i >= offset_a) ? A->shape[i - offset_a] : 1;
        int dim_b = (i >= offset_b) ? B->shape[i - offset_b] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::invalid_argument("Error: Tensor shapes are not broadcastable.");
        }
        
        out_shape[i] = std::max(dim_a, dim_b);
        info.shape[i] = out_shape[i];
        
        // If a dimension is broadcasted (size 1), its stride is treated as 0.
        info.strides_a[i] = (dim_a == 1) ? 0 : A->strides[i - offset_a];
        info.strides_b[i] = (dim_b == 1) ? 0 : B->strides[i - offset_b];
        
        total_size *= out_shape[i];
    }

    bool out_requires_grad = A->requires_grad || B->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, out_requires_grad);

    int threadsPerBlock = 256;
    int blocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    broadcast_add_forward_kernel<<<blocks, threadsPerBlock>>>(
        A->data.get(),
        B->data.get(),
        out->data.get(),
        info,
        total_size
    );

    // Minimize CPU-GPU sync. Only sync if absolutely required by external calling code.
    // For now we sync to ensure completion, but in an ultra-optimized loop this could be removed.
    // cudaDeviceSynchronize();

    if (out_requires_grad) {
        out->parents.push_back(A);
        out->parents.push_back(B);
        
        bool req_a = A->requires_grad;
        bool req_b = B->requires_grad;

        std::weak_ptr<Tensor> weak_out = out;
        out->backward_closure = [A, B, info, total_size, blocks, threadsPerBlock, req_a, req_b, weak_out]() {
            auto out = weak_out.lock();
            
            broadcast_add_backward_kernel<<<blocks, threadsPerBlock>>>(
                A->grad.get(),
                B->grad.get(),
                out->grad.get(),
                info,
                total_size,
                req_a,
                req_b
            );
            // cudaDeviceSynchronize();
        };
    }

    return out;
}
