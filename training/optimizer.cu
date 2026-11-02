#include "../training/optimizer.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

// Kernel for in-place AdamW weight updates
__global__ void adamw_kernel(
    float* w, const float* g, float* m, float* v, 
    float lr, float beta1, float beta2, float eps, float weight_decay, int t, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float weight = w[idx];
    float grad = g[idx];

    // In-place decoupled weight decay
    weight = weight - lr * weight_decay * weight;

    // Moving averages of gradient and squared gradient
    float m_t = beta1 * m[idx] + (1.0f - beta1) * grad;
    float v_t = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

    m[idx] = m_t;
    v[idx] = v_t;

    // Bias correction
    float m_hat = m_t / (1.0f - powf(beta1, t));
    float v_hat = fmaxf(v_t, 0.0f) / (1.0f - powf(beta2, t));

    // Weight update with safe epsilon addition
    float weight_update = lr * m_hat / (sqrtf(fmaxf(v_hat, 0.0f)) + eps);
    w[idx] = weight - weight_update;
}

AdamW::AdamW(std::vector<std::shared_ptr<Tensor>> parameters, 
             float lr, float beta1, float beta2, float eps, float weight_decay)
    : params(parameters), lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay), t(0) {
    
    for (auto& p : params) {
        float *m_ptr = nullptr, *v_ptr = nullptr;
        
        cudaError_t err = cudaMalloc((void**)&m_ptr, p->total_size * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error("CUDA Malloc failed for AdamW m buffer: " + std::string(cudaGetErrorString(err)));
        
        err = cudaMalloc((void**)&v_ptr, p->total_size * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error("CUDA Malloc failed for AdamW v buffer: " + std::string(cudaGetErrorString(err)));
        
        cudaMemset(m_ptr, 0, p->total_size * sizeof(float));
        cudaMemset(v_ptr, 0, p->total_size * sizeof(float));
        
        m_buffers.push_back(m_ptr);
        v_buffers.push_back(v_ptr);
    }
}

AdamW::~AdamW() {
    for (float* ptr : m_buffers) {
        if (ptr) cudaFree(ptr);
    }
    for (float* ptr : v_buffers) {
        if (ptr) cudaFree(ptr);
    }
}

void AdamW::step() {
    t++;
    for (size_t i = 0; i < params.size(); ++i) {
        auto& p = params[i];
        if (!p->requires_grad || !p->grad) continue;

        int size = p->total_size;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        float current_wd = (p->shape.size() >= 2) ? weight_decay : 0.0f;
        
        adamw_kernel<<<blocks, threads>>>(
            p->data.get(), p->grad.get(),
            m_buffers[i], v_buffers[i],
            lr, beta1, beta2, eps, current_wd, t, size
        );
    }
    // cudaDeviceSynchronize();
}

void AdamW::zero_grad() {
    for (auto& p : params) {
        p->zero_grad();
    }
}
