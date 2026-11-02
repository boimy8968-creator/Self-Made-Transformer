#include "../core/autograd.h"
#include <cuda_runtime.h>
#include <unordered_set>
#include <vector>
#include <stdexcept>

// ==========================================
// 1. GPU Workers: MSE Forward and Backward
// ==========================================

// Forward: Calculate element-wise squared error
__global__ void mse_forward_kernel(const float* pred, const float* target, float* loss, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        float diff = pred[idx] - target[idx];
        loss[idx] = (diff * diff) / total_size;
    }
}

// Backward: Calculate the analytical derivative of MSE
// Derivative of ((p - t)^2) / N is -> 2 * (p - t) / N
__global__ void mse_backward_kernel(float* d_pred, const float* pred, const float* target, const float* d_loss, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        float grad = 2.0f * (pred[idx] - target[idx]) / total_size;

        // Multiply by upstream gradient (d_loss) and accumulate
        d_pred[idx] += grad * d_loss[idx];
    }
}

// Kernel to seed the initial gradient of the loss node to 1.0
__global__ void seed_gradient_kernel(float* grad, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        grad[idx] = 1.0f;
    }
}

// ==========================================
// 2. Cost Function: Mean Squared Error (MSE)
// ==========================================
std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> predictions, std::shared_ptr<Tensor> targets) {
    if (predictions->shape != targets->shape) {
        throw std::invalid_argument("MSE Error: Predictions and Targets must have the same shape.");
    }

    int total_size = predictions->total_size;
    bool req_grad = predictions->requires_grad;

    // Allocate output tensor
    auto loss = std::make_shared<Tensor>(predictions->shape, req_grad);

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;

    // Forward pass
    mse_forward_kernel << <blocks, threads >> > (
        predictions->data.get(),
        targets->data.get(),
        loss->data.get(),
        total_size
        );
    // cudaDeviceSynchronize();

    // Backward Time Capsule
    if (req_grad) {
        loss->parents.push_back(predictions);
        // We usually don't track gradients for targets (the ground truth data)

        std::weak_ptr<Tensor> weak_loss = loss;
        loss->backward_closure = [predictions, targets, total_size, blocks, threads, weak_loss]() {
            auto loss = weak_loss.lock();
            
            if (predictions->requires_grad) {
                mse_backward_kernel << <blocks, threads >> > (
                    predictions->grad.get(),
                    predictions->data.get(),
                    targets->data.get(),
                    loss->grad.get(),
                    total_size
                    );
                // cudaDeviceSynchronize();
            }
            };
    }

    return loss;
}

// ==========================================
// 3. CPU Orchestrator: Topological Sort
// ==========================================

// Helper function to recursively build the graph
void build_topo(std::shared_ptr<Tensor> node, std::unordered_set<Tensor*>& visited, std::vector<std::shared_ptr<Tensor>>& topo) {
    // If we haven't seen this node yet
    if (visited.find(node.get()) == visited.end()) {
        visited.insert(node.get());

        // Visit all parents first
        for (auto& parent : node->parents) {
            build_topo(parent, visited, topo);
        }

        // Push this node only AFTER all its parents are processed
        topo.push_back(node);
    }
}

// The trigger function
void backward(std::shared_ptr<Tensor> loss_node) {
    if (!loss_node->requires_grad) {
        throw std::runtime_error("Cannot call backward on a tensor that does not require gradients.");
    }

    // 1. Build the topological order
    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;
    build_topo(loss_node, visited, topo);

    // 2. Seed the initial gradient to 1.0 (The derivative of Loss with respect to Loss is 1)
    int threads = 256;
    int blocks = (loss_node->total_size + threads - 1) / threads;
    seed_gradient_kernel << <blocks, threads >> > (loss_node->grad.get(), loss_node->total_size);
    // cudaDeviceSynchronize();

    // 3. Traverse the list in REVERSE and execute the time capsules
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_closure) {
            (*it)->backward_closure();
            // Explicitly destroy the lambda and any captured references
            (*it)->backward_closure.ptr = nullptr;
        }
        // Explicitly break circular/graph references to parent tensors
        (*it)->parents.clear();
    }
}
