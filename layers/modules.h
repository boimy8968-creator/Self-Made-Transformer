#pragma once
#include "../core/tensor.h"
#include "../layers/add.h"
#include "../layers/bmm.h"
#include "../layers/gelu.h"
#include "../layers/layernorm.h"
#include "../layers/softmax.h"
#include "../layers/permute.h"

#include "../layers/scale.h"
#include "../layers/causal_mask.h"
#include <cmath>
#include <random>
#include <iostream>

// High-level Modules for building Transformer architectures
class Linear {
public:
    std::shared_ptr<Tensor> weight; // [Out, In] -> We store it this way to do (X * W^T) for BMM compatibility
    std::shared_ptr<Tensor> bias;

    Linear(int in_features, int out_features, bool has_bias = true) {
        weight = std::make_shared<Tensor>(std::vector<int>{out_features, in_features}, true);
        if (has_bias) {
            bias = std::make_shared<Tensor>(std::vector<int>{out_features}, true);
        }
        std::vector<float> init_w(out_features * in_features);
        std::mt19937 gen(42);
        std::normal_distribution<float> d(0.0f, 0.02f);
        for(int i=0; i<init_w.size(); ++i) init_w[i] = d(gen);
        weight->fill(init_w);
        if (has_bias) {
            std::vector<float> init_b(out_features, 0.0f);
            bias->fill(init_b);
        }
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) {
        auto out = bmm(x, weight, true);
        if (bias) {
            out = add(out, bias);
        }
        return out;
    }
};

class LayerNorm {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    float eps;

    LayerNorm(int normalized_shape, float epsilon = 1e-5f) : eps(epsilon) {
        weight = std::make_shared<Tensor>(std::vector<int>{normalized_shape}, true);
        bias = std::make_shared<Tensor>(std::vector<int>{normalized_shape}, true);
        std::vector<float> w(normalized_shape, 1.0f);
        std::vector<float> b(normalized_shape, 0.0f);
        weight->fill(w);
        bias->fill(b);
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) {
        return layernorm(x, weight, bias, eps);
    }
};

class MultiHeadAttention {
public:
    int embed_dim;
    int num_heads;
    int head_dim;
    
    std::shared_ptr<Linear> q_proj;
    std::shared_ptr<Linear> k_proj;
    std::shared_ptr<Linear> v_proj;
    std::shared_ptr<Linear> out_proj;

    MultiHeadAttention(int embed_dim, int num_heads) : embed_dim(embed_dim), num_heads(num_heads) {
        head_dim = embed_dim / num_heads;
        q_proj = std::make_shared<Linear>(embed_dim, embed_dim, true);
        k_proj = std::make_shared<Linear>(embed_dim, embed_dim, true);
        v_proj = std::make_shared<Linear>(embed_dim, embed_dim, true);
        out_proj = std::make_shared<Linear>(embed_dim, embed_dim, true);
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) {
        int B = x->shape[0];
        int Seq = x->shape[1];
        
        auto q = q_proj->forward(x);
        auto k = k_proj->forward(x);
        auto v = v_proj->forward(x);
        
        q = view(q, {B, Seq, num_heads, head_dim});
        k = view(k, {B, Seq, num_heads, head_dim});
        v = view(v, {B, Seq, num_heads, head_dim});
        
        q = permute(q, {0, 2, 1, 3});
        k = permute(k, {0, 2, 1, 3});
        v = permute(v, {0, 2, 1, 3});
        
        auto scores = bmm(q, k, true); 
        scores = scale(scores, 1.0f / std::sqrt((float)head_dim));
                scores = apply_causal_mask(scores);
        static int calls = 0;
        if (calls++ % 100 == 0) {
            std::vector<float> h_s(scores->total_size);
            cudaMemcpy(h_s.data(), scores->data.get(), scores->total_size * sizeof(float), cudaMemcpyDeviceToHost);
            float max_s = -1e9f;
            for (float s : h_s) if (s > max_s && s > -1e8f) max_s = s;
            std::cout << "[Softmax Telemetry] Max Pre-Softmax Logit: " << max_s << std::endl;
        }
        auto probs = softmax(scores);
        
        auto context = bmm(probs, v, false);
        context = permute(context, {0, 2, 1, 3});
        context = view(context, {B, Seq, embed_dim});
        
        return out_proj->forward(context);
    }
};
