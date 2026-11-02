#pragma once
#include "../core/tensor.h"
#include <memory>

class KVCache {
public:
    int max_seq_len;
    int batch_size;
    int num_heads;
    int head_dim;
    int current_seq_len;

    std::shared_ptr<Tensor> k_cache; // Shape: [batch_size, num_heads, max_seq_len, head_dim]
    std::shared_ptr<Tensor> v_cache; // Shape: [batch_size, num_heads, max_seq_len, head_dim]

    KVCache(int max_seq_len, int batch_size, int num_heads, int head_dim);

    // Appends the newest token's K and V vectors to the cache
    void append(std::shared_ptr<Tensor> new_k, std::shared_ptr<Tensor> new_v);
    
    // Returns a sliced tensor view representing only the valid history
    std::shared_ptr<Tensor> get_k_view();
    std::shared_ptr<Tensor> get_v_view();

    void reset();
};
