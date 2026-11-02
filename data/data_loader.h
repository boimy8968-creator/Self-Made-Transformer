#pragma once
#include <string>
#include <vector>
#include <memory>
#include <windows.h>
#include <cuda_runtime.h>
#include "../core/tensor.h"

class BinaryDataLoader {
private:
    HANDLE hFile;
    HANDLE hMapping;
    int32_t* mapped_data;
    size_t file_size;
    size_t total_tokens;
    size_t current_offset;

    // Persistent GPU buffer for incoming integers before casting to float
    int32_t* d_batch_ints;
    int max_capacity;

public:
    // Initialize the memory mapped file and pin it
    BinaryDataLoader(const std::string& filename, int max_batch_size = 1024, int max_seq_len = 1024);
    ~BinaryDataLoader();

    // Streams next batch asynchronously into VRAM
    std::shared_ptr<Tensor> get_next_batch(int batch_size, int seq_len, cudaStream_t stream = 0);
    
    void reset();
};
