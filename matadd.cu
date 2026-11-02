#include "matadd.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// ==========================================
// 1. GPU Worker: Forward Addition
// ==========================================
__global__ void matadd_forward_kernel(const float* A, const float* B, float* C, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        C[idx] = A[idx] + B[idx];
    }
}

// ==========================================
// 2. GPU Worker: Backward Gradient Accumulation
// ==========================================
// Gradients accumulate (+=) rather than overwrite (=)
__global__ void inplace_add_kernel(float* target, const float* source, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        target[idx] += source[idx];
    }
}

// ==========================================
// 3. Host Manager: Graph Builder
// ==========================================
std::shared_ptr<Tensor> matadd(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B) {
    // Edge Case 1: Dimensionality Check
    if (A->shape != B->shape) {
        throw std::invalid_argument("Error: MatAdd requires tensors of exactly the same shape.");
    }

    int total_size = A->total_size;

    // Edge Case 2: Determine if the output needs to track gradients
    bool out_requires_grad = A->requires_grad || B->requires_grad;

    // Allocate the Output Tensor (VRAM automatically provisioned)
    auto out = std::make_shared<Tensor>(A->shape, out_requires_grad);

    // Grid geometry for flat 1D execution
    int threadsPerBlock = 256;
    int blocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    // Execute the mathematical forward pass
    matadd_forward_kernel << <blocks, threadsPerBlock >> > (
        A->data.get(),
        B->data.get(),
        out->data.get(),
        total_size
        );

    // cudaDeviceSynchronize();

    // ==========================================
    // 4. Build the Autograd Graph
    // ==========================================
    if (out_requires_grad) {
        // Log the lineage
        out->parents.push_back(A);
        out->parents.push_back(B);

        // Inject the Lambda Time Capsule
        std::weak_ptr<Tensor> weak_out = out;
        out->backward_closure = [A, B, total_size, blocks, threadsPerBlock, weak_out]() {
            auto out = weak_out.lock();
            
            // When backward() is called, this lambda will execute on the GPU.

            if (A->requires_grad) {
                // dA += dOut
                inplace_add_kernel << <blocks, threadsPerBlock >> > (
                    A->grad.get(),
                    out->grad.get(),
                    total_size
                    );
            }
            if (B->requires_grad) {
                // dB += dOut
                inplace_add_kernel << <blocks, threadsPerBlock >> > (
                    B->grad.get(),
                    out->grad.get(),
                    total_size
                    );
            }
            // Ensure hardware finishes adding gradients before moving to the next node
            // cudaDeviceSynchronize();
            };
    }

    return out;
}
