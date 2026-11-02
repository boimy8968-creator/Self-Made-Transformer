#include "../inference/kv_cache.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Kernel to append a slice of K or V to the pre-allocated cache
__global__ void append_cache_kernel(
    float* cache, const float* new_item,
    int batch_size, int num_heads, int max_seq_len, int head_dim,
    int current_seq_len, int new_seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * new_seq_len * head_dim;
    
    if (idx >= total_elements) return;

    // Decode idx into (b, h, s, d) for new_item
    int d = idx % head_dim;
    int s = (idx / head_dim) % new_seq_len;
    int h = (idx / (head_dim * new_seq_len)) % num_heads;
    int b = idx / (head_dim * new_seq_len * num_heads);

    // Encode into cache index utilizing the max_seq_len stride
    int cache_s = current_seq_len + s;
    int cache_idx = ((b * num_heads + h) * max_seq_len + cache_s) * head_dim + d;

    cache[cache_idx] = new_item[idx];
}

KVCache::KVCache(int max_seq_len, int batch_size, int num_heads, int head_dim)
    : max_seq_len(max_seq_len), batch_size(batch_size), num_heads(num_heads), head_dim(head_dim), current_seq_len(0) 
{
    // Pre-allocate maximum sequence length VRAM buffers, avoiding dynamic cudaMallocs during generation
    k_cache = std::make_shared<Tensor>(std::vector<int>{batch_size, num_heads, max_seq_len, head_dim}, false);
    v_cache = std::make_shared<Tensor>(std::vector<int>{batch_size, num_heads, max_seq_len, head_dim}, false);
}

void KVCache::append(std::shared_ptr<Tensor> new_k, std::shared_ptr<Tensor> new_v) {
    int new_seq_len = new_k->shape[new_k->shape.size() - 2];
    
    if (current_seq_len + new_seq_len > max_seq_len) {
        throw std::runtime_error("KVCache exceeded max_seq_len");
    }

    int total_elements = batch_size * num_heads * new_seq_len * head_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    append_cache_kernel<<<blocks, threads>>>(
        k_cache->data.get(), new_k->data.get(),
        batch_size, num_heads, max_seq_len, head_dim,
        current_seq_len, new_seq_len
    );

    append_cache_kernel<<<blocks, threads>>>(
        v_cache->data.get(), new_v->data.get(),
        batch_size, num_heads, max_seq_len, head_dim,
        current_seq_len, new_seq_len
    );
    
    // Explicit sync to guarantee completion before attention uses it
    cudaDeviceSynchronize();
    current_seq_len += new_seq_len;
}

std::shared_ptr<Tensor> KVCache::get_k_view() {
    // Create an empty tensor (no VRAM allocation)
    auto view = std::make_shared<Tensor>(std::vector<int>{}, false);
    view->shape = {batch_size, num_heads, current_seq_len, head_dim};
    
    // Retain original physical strides to allow BMM to skip padded space
    view->strides = k_cache->strides;
    view->total_size = batch_size * num_heads * current_seq_len * head_dim;
    view->data = k_cache->data; // Shared pointer reference counting keeps it safe
    return view;
}

std::shared_ptr<Tensor> KVCache::get_v_view() {
    auto view = std::make_shared<Tensor>(std::vector<int>{}, false);
    view->shape = {batch_size, num_heads, current_seq_len, head_dim};
    view->strides = v_cache->strides;
    view->total_size = batch_size * num_heads * current_seq_len * head_dim;
    view->data = v_cache->data;
    return view;
}

void KVCache::reset() {
    current_seq_len = 0;
}
