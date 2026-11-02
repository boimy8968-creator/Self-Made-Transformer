#pragma once
#include "../layers/modules.h"
#include "../layers/embedding.h"
#include <vector>
#include <memory>

class TransformerEncoderBlock {
public:
    MultiHeadAttention mhsa;
    LayerNorm ln1;
    LayerNorm ln2;
    Linear mlp1;
    Linear mlp2;

    TransformerEncoderBlock(int embed_dim, int num_heads);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

class TransformerEncoder {
public:
    Embedding token_embed;
    LearnedPositionalEncoding pos_embed;
    std::vector<std::shared_ptr<TransformerEncoderBlock>> blocks;
    LayerNorm final_ln;

    TransformerEncoder(int vocab_size, int embed_dim, int num_heads, int num_blocks, int max_seq_len);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input_ids);
};
