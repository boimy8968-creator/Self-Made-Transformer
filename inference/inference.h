#pragma once
#include "../core/tensor.h"
#include "../layers/modules.h"
#include "../data/tokenizer.h"
#include "../layers/encoder.h"
#include "../inference/kv_cache.h"
#include <memory>
#include <string>

class CausalSelfAttention {
public:
    std::shared_ptr<Linear> q_proj;
    std::shared_ptr<Linear> k_proj;
    std::shared_ptr<Linear> v_proj;
    std::shared_ptr<Linear> out_proj;
    std::shared_ptr<KVCache> cache;

    CausalSelfAttention(int embed_dim, int num_heads, int max_seq_len, int batch_size) {
        q_proj = std::make_shared<Linear>(embed_dim, embed_dim);
        k_proj = std::make_shared<Linear>(embed_dim, embed_dim);
        v_proj = std::make_shared<Linear>(embed_dim, embed_dim);
        out_proj = std::make_shared<Linear>(embed_dim, embed_dim);
        cache = std::make_shared<KVCache>(max_seq_len, batch_size, num_heads, embed_dim / num_heads);
    }
};

class AutoregressiveTransformerBlock {
public:
    std::shared_ptr<LayerNorm> ln1;
    std::shared_ptr<CausalSelfAttention> attn;
    std::shared_ptr<LayerNorm> ln2;
    std::shared_ptr<Linear> ffn1;
    std::shared_ptr<Linear> ffn2;

    AutoregressiveTransformerBlock(int embed_dim, int num_heads, int max_seq_len, int batch_size) {
        ln1 = std::make_shared<LayerNorm>(embed_dim);
        attn = std::make_shared<CausalSelfAttention>(embed_dim, num_heads, max_seq_len, batch_size);
        ln2 = std::make_shared<LayerNorm>(embed_dim);
        ffn1 = std::make_shared<Linear>(embed_dim, embed_dim * 4);
        ffn2 = std::make_shared<Linear>(embed_dim * 4, embed_dim);
    }
};

class AutoregressiveModel {
public:
    std::vector<std::shared_ptr<AutoregressiveTransformerBlock>> blocks;
    std::shared_ptr<Linear> lm_head;
    int embed_dim;
    int vocab_size;

    AutoregressiveModel(int num_layers, int embed_dim, int num_heads, int vocab_size, int max_seq_len);
    void reset_cache();
    std::shared_ptr<Tensor> forward_token(std::shared_ptr<Tensor> x);
};

void run_autoregressive_inference(
    TransformerEncoder& model, 
    Linear& lm_head,
    Tokenizer& tokenizer, 
    const std::string& prompt, 
    int max_new_tokens, 
    float temperature, 
    float top_p);

void greedy_generate(AutoregressiveModel& model, std::shared_ptr<Tensor> initial_token_embeds, int max_tokens, std::shared_ptr<Tensor> token_embeddings, Tokenizer& tokenizer);
void copy_weights_to_inference(TransformerEncoder& src, AutoregressiveModel& dst);
