#include <random>
#include "../layers/embedding.h"
#include "../layers/add.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void embedding_fwd_kernel(const float* weight, const float* ids, float* out, int vocab_size, int embed_dim, int total_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_tokens) {
        int token_id = (int)ids[idx];
        if (token_id < 0 || token_id >= vocab_size) token_id = 1; // fallback to <unk>
        for (int i = 0; i < embed_dim; ++i) {
            out[idx * embed_dim + i] = weight[token_id * embed_dim + i];
        }
    }
}

__global__ void embedding_bwd_kernel(float* grad_weight, const float* ids, const float* grad_out, int vocab_size, int embed_dim, int total_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_tokens) {
        int token_id = (int)ids[idx];
        if (token_id < 0 || token_id >= vocab_size) token_id = 1;
        for (int i = 0; i < embed_dim; ++i) {
            atomicAdd(&grad_weight[token_id * embed_dim + i], grad_out[idx * embed_dim + i]);
        }
    }
}

__global__ void embedding_fwd_kernel_int(const float* weight, const int32_t* ids, float* out, int vocab_size, int embed_dim, int total_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_tokens) {
        int token_id = ids[idx];
        if (token_id < 0 || token_id >= vocab_size) token_id = 1;
        for (int i = 0; i < embed_dim; ++i) {
            out[idx * embed_dim + i] = weight[token_id * embed_dim + i];
        }
    }
}

__global__ void embedding_bwd_kernel_int(float* grad_weight, const int32_t* ids, const float* grad_out, int vocab_size, int embed_dim, int total_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_tokens) {
        int token_id = ids[idx];
        if (token_id < 0 || token_id >= vocab_size) token_id = 1;
        for (int i = 0; i < embed_dim; ++i) {
            atomicAdd(&grad_weight[token_id * embed_dim + i], grad_out[idx * embed_dim + i]);
        }
    }
}

__global__ void generate_pos_ids_kernel(int32_t* pos_ids, int seq_len, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        pos_ids[idx] = idx % seq_len;
    }
}

Embedding::Embedding(int v_size, int dim) : vocab_size(v_size), embed_dim(dim) {
    weight = std::make_shared<Tensor>(std::vector<int>{vocab_size, embed_dim}, true);
    std::vector<float> data(vocab_size * embed_dim);
    std::mt19937 gen(42);
    std::normal_distribution<float> d(0.0f, 0.02f);
    for(int i=0; i<data.size(); ++i) data[i] = d(gen);
    weight->fill(data);
}

std::shared_ptr<Tensor> Embedding::forward(std::shared_ptr<Tensor> ids) {
    int batch_seq = ids->total_size;
    std::vector<int> out_shape = ids->shape;
    out_shape.push_back(embed_dim);
    auto out = std::make_shared<Tensor>(out_shape, weight->requires_grad);
    int threads = 256;
    int blocks = (batch_seq + threads - 1) / threads;
    embedding_fwd_kernel<<<blocks, threads>>>(weight->data.get(), ids->data.get(), out->data.get(), vocab_size, embed_dim, batch_seq);
    if (weight->requires_grad) {
        out->parents.push_back(weight);
        auto w = weight;
        std::weak_ptr<Tensor> weak_out = out;
        out->backward_closure = [w, ids, vocab_size=this->vocab_size, embed_dim=this->embed_dim, batch_seq, blocks, threads, weak_out]() {
            auto out = weak_out.lock();
            embedding_bwd_kernel<<<blocks, threads>>>(w->grad.get(), ids->data.get(), out->grad.get(), vocab_size, embed_dim, batch_seq);
        };
    }
    return out;
}

std::shared_ptr<Tensor> Embedding::forward_int(std::shared_ptr<Tensor> ids) {
    int batch_seq = ids->total_size;
    std::vector<int> out_shape = ids->shape;
    out_shape.push_back(embed_dim);
    auto out = std::make_shared<Tensor>(out_shape, weight->requires_grad);
    int threads = 256;
    int blocks = (batch_seq + threads - 1) / threads;
    embedding_fwd_kernel_int<<<blocks, threads>>>(weight->data.get(), (const int32_t*)ids->data.get(), out->data.get(), vocab_size, embed_dim, batch_seq);
    if (weight->requires_grad) {
        out->parents.push_back(weight);
        auto w = weight;
        std::weak_ptr<Tensor> weak_out = out;
        out->backward_closure = [w, ids, vocab_size=this->vocab_size, embed_dim=this->embed_dim, batch_seq, blocks, threads, weak_out]() {
            auto out = weak_out.lock();
            embedding_bwd_kernel_int<<<blocks, threads>>>(w->grad.get(), (const int32_t*)ids->data.get(), out->grad.get(), vocab_size, embed_dim, batch_seq);
        };
    }
    return out;
}

LearnedPositionalEncoding::LearnedPositionalEncoding(int max_len, int dim) : pe_embed(max_len, dim) {}

std::shared_ptr<Tensor> LearnedPositionalEncoding::forward(std::shared_ptr<Tensor> x) {
    int B = x->shape[0];
    int Seq = x->shape[1];
    if (!cached_pos_ids || cached_pos_ids->shape[0] != B || cached_pos_ids->shape[1] != Seq) {
        cached_pos_ids = std::make_shared<Tensor>(std::vector<int>{B, Seq}, false);
        int total = B * Seq;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        generate_pos_ids_kernel<<<blocks, threads>>>((int32_t*)cached_pos_ids->data.get(), Seq, total);
    }
    auto pe = pe_embed.forward_int(cached_pos_ids);
    return add(x, pe);
}
