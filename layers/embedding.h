#pragma once
#include "../core/tensor.h"
#include <memory>
#include <vector>

class Embedding {
public:
    std::shared_ptr<Tensor> weight;
    int vocab_size;
    int embed_dim;

    Embedding(int vocab_size, int embed_dim);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> ids);
    std::shared_ptr<Tensor> forward_int(std::shared_ptr<Tensor> ids);
};

// Helper for learned positional encoding
class LearnedPositionalEncoding {
public:
    Embedding pe_embed;
    std::shared_ptr<Tensor> cached_pos_ids;
    LearnedPositionalEncoding(int max_len, int dim);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};
