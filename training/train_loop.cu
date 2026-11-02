#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>
#include "../core/tensor.h"
#include "../core/graph_runner.h"
#include "../layers/fused_modules.h"
#include "../layers/bmm.h"
#include "../core/autograd.h"

// AdamW State struct
struct AdamWState {
    std::shared_ptr<Tensor> param;
    float* m;
    float* v;
    
    AdamWState(std::shared_ptr<Tensor> p) : param(p) {
        cudaMallocAsync(&m, p->total_size * sizeof(float), 0);
        cudaMemsetAsync(m, 0, p->total_size * sizeof(float), 0);
        cudaMallocAsync(&v, p->total_size * sizeof(float), 0);
        cudaMemsetAsync(v, 0, p->total_size * sizeof(float), 0);
    }
    ~AdamWState() {
        cudaFreeAsync(m, 0);
        cudaFreeAsync(v, 0);
    }
};

__global__ void step_t_kernel(int* t) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *t += 1;
    }
}

__global__ void adamw_kernel(float* param, float* grad, float* m, float* v, float lr, float beta1, float beta2, float eps, float weight_decay, int total_size, int* t_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        float g = grad[idx];
        float p = param[idx];
        
        p = p - lr * weight_decay * p;
        
        float mt = beta1 * m[idx] + (1.0f - beta1) * g;
        float vt = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        m[idx] = mt;
        v[idx] = vt;
        
        int t = *t_ptr;
        float m_hat = mt / (1.0f - powf(beta1, t));
        float v_hat = vt / (1.0f - powf(beta2, t));
        
        p -= lr * m_hat / (sqrtf(v_hat) + eps);
        param[idx] = p;
    }
}

// Cross-Entropy Loss
__global__ void cross_entropy_forward_kernel(const float* logits, const float* targets, float* loss, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float max_val = -1e20f;
        for (int c = 0; c < num_classes; ++c) {
            max_val = max(max_val, logits[idx * num_classes + c]);
        }
        
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            sum_exp += exp(logits[idx * num_classes + c] - max_val);
        }
        
        float prob = exp(logits[idx * num_classes + target_class] - max_val) / sum_exp;
        loss[idx] = -logf(fmaxf(prob, 1e-7f));
    }
}

__global__ void cross_entropy_backward_kernel(float* d_logits, const float* logits, const float* targets, const float* d_loss, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float max_val = -1e20f;
        for (int c = 0; c < num_classes; ++c) {
            max_val = max(max_val, logits[idx * num_classes + c]);
        }
        
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            sum_exp += exp(logits[idx * num_classes + c] - max_val);
        }
        
        int target_class = (int)targets[idx];
        float dl = d_loss[idx] / batch_size;
        for (int c = 0; c < num_classes; ++c) {
            float prob = exp(logits[idx * num_classes + c] - max_val) / sum_exp;
            float grad = prob;
            if (c == target_class) {
                grad -= 1.0f;
            }
            atomicAdd(&d_logits[idx * num_classes + c], grad * dl); 
        }
    }
}

std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> targets) {
    int batch_size = targets->total_size;
    int num_classes = logits->shape.back();
    
    auto loss = std::make_shared<Tensor>(targets->shape, logits->requires_grad);
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    cross_entropy_forward_kernel<<<blocks, threads>>>(logits->data.get(), targets->data.get(), loss->data.get(), batch_size, num_classes);
    
    if (logits->requires_grad) {
        loss->parents.push_back(logits);
        std::weak_ptr<Tensor> weak_loss = loss;
        loss->backward_closure = [logits, targets, batch_size, num_classes, blocks, threads, weak_loss]() {
            auto loss = weak_loss.lock();
            
            cross_entropy_backward_kernel<<<blocks, threads>>>(logits->grad.get(), logits->data.get(), targets->data.get(), loss->grad.get(), batch_size, num_classes);
        };
    }
    return loss;
}

// A simple module combining fused modules
class FusedTransformerTest {
public:
    std::shared_ptr<Tensor> q_w, k_w, v_w, o_w;
    std::shared_ptr<Tensor> mlp_w, mlp_b;
    std::shared_ptr<Tensor> mlp2_w, mlp2_b;
    
    std::vector<std::shared_ptr<Tensor>> params;
    
    FusedTransformerTest(int dim) {
        q_w = std::make_shared<Tensor>(std::vector<int>{dim, dim}, true);
        k_w = std::make_shared<Tensor>(std::vector<int>{dim, dim}, true);
        v_w = std::make_shared<Tensor>(std::vector<int>{dim, dim}, true);
        o_w = std::make_shared<Tensor>(std::vector<int>{dim, dim}, true);
        
        mlp_w = std::make_shared<Tensor>(std::vector<int>{dim * 4, dim}, true);
        mlp_b = std::make_shared<Tensor>(std::vector<int>{dim * 4}, true);
        
        mlp2_w = std::make_shared<Tensor>(std::vector<int>{dim, dim * 4}, true);
        mlp2_b = std::make_shared<Tensor>(std::vector<int>{dim}, true);
        
        std::vector<float> init_w(dim * dim, 0.01f);
        q_w->fill(init_w);
        k_w->fill(init_w);
        v_w->fill(init_w);
        o_w->fill(init_w);
        
        std::vector<float> init_mlp_w(dim * 4 * dim, 0.01f);
        mlp_w->fill(init_mlp_w);
        mlp2_w->fill(init_mlp_w);
        
        std::vector<float> init_b(dim * 4, 0.0f);
        mlp_b->fill(init_b);
        std::vector<float> init_b2(dim, 0.0f);
        mlp2_b->fill(init_b2);
        
        params = {q_w, k_w, v_w, o_w, mlp_w, mlp_b, mlp2_w, mlp2_b};
    }
    
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) {
        // Dummy attention (bmm)
        auto q = bmm(x, q_w, true);
        auto k = bmm(x, k_w, true);
        auto v = bmm(x, v_w, true);
        
        auto scores = bmm(q, k, true);
        auto probs = fused_scale_softmax(scores, 1.0f / sqrtf((float)x->shape.back()));
        
        auto context = bmm(probs, v, false);
        auto o = bmm(context, o_w, true);
        
        // MLP with fused add gelu
        auto h1 = bmm(o, mlp_w, true);
        auto h1_gelu = fused_add_gelu(h1, mlp_b);
        
        auto h2 = bmm(h1_gelu, mlp2_w, true);
        auto h2_out = fused_add_gelu(h2, mlp2_b); 
        
        return h2_out;
    }
};

int main() {
    // Hardware Saturation: Batch = 16, Seq = 128, Dim = 256, Heads = 8.
    int B = 16;
    int S = 128;
    int D = 256;
    
    FusedTransformerTest model(D);
    
    auto x = std::make_shared<Tensor>(std::vector<int>{B, S, D}, true);
    std::vector<float> x_data(B * S * D, 0.1f);
    x->fill(x_data);
    
    auto target = std::make_shared<Tensor>(std::vector<int>{B, S}, false);
    std::vector<float> t_data(B * S, 0.0f);
    target->fill(t_data);
    
    std::vector<AdamWState> states;
    for (auto& p : model.params) {
        states.emplace_back(p);
    }
    
    int* d_t;
    cudaMallocAsync(&d_t, sizeof(int), 0);
    cudaMemsetAsync(d_t, 0, sizeof(int), 0);
    
    CUDAGraphRunner runner;
    
    std::cout << "Capturing graph..." << std::endl;
    try {
        runner.capture([&]() {
            // Forward
            auto out = model.forward(x);
            
            out->shape = {B * S, D};
            target->shape = {B * S};
            auto loss = cross_entropy_loss(out, target);
            
            // Backward
            backward(loss);
            
            // AdamW
            step_t_kernel<<<1, 1>>>(d_t);
            for (auto& st : states) {
                int threads = 256;
                int blocks = (st.param->total_size + threads - 1) / threads;
                adamw_kernel<<<blocks, threads>>>(
                    st.param->data.get(),
                    st.param->grad.get(),
                    st.m,
                    st.v,
                    1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f,
                    st.param->total_size,
                    d_t
                );
                cudaMemsetAsync(st.param->grad.get(), 0, st.param->total_size * sizeof(float), 0);
            }
            cudaMemsetAsync(x->grad.get(), 0, x->total_size * sizeof(float), 0);
        });
    } catch(const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Graph captured. Running 100 iterations..." << std::endl;
    
    // Warmup
    try {
        for (int i = 0; i < 10; ++i) {
            runner.execute();
        }
    } catch(const std::exception& e) {
        std::cerr << "Exception during execute: " << e.what() << std::endl;
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    int iters = 100;
    for (int i = 0; i < iters; ++i) {
        runner.execute();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    
    double avg_ms = diff.count() / iters;
    std::cout << "Average latency per step: " << avg_ms << " ms" << std::endl;
    
    return 0;
}

