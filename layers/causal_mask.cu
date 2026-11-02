#include "../layers/causal_mask.h"
#include <cuda_runtime.h>

__global__ void causal_mask_forward_kernel(float* data, int seq_len, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int col = idx % seq_len;
        int row = (idx / seq_len) % seq_len;
        if (col > row) {
            data[idx] = -1e9f;
        }
    }
}

std::shared_ptr<Tensor> apply_causal_mask(std::shared_ptr<Tensor> input) {
    int seq_len = input->shape.back(); 
    int threads = 256;
    int blocks = (input->total_size + threads - 1) / threads;
    causal_mask_forward_kernel<<<blocks, threads>>>(input->data.get(), seq_len, input->total_size);
    return input; // In-place modification
}
