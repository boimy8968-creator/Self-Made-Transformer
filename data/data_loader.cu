#include "../data/data_loader.h"
#include <stdexcept>
#include <iostream>

// Kernel to convert the raw 32-bit integer IDs to the float format expected by the engine's Tensor
__global__ void cast_int_to_float_kernel(const int32_t* __restrict__ in, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (float)in[idx];
    }
}

BinaryDataLoader::BinaryDataLoader(const std::string& filename, int max_batch_size, int max_seq_len) 
    : current_offset(0) {
    
    max_capacity = max_batch_size * max_seq_len;
    
    // Allocate persistent GPU buffer to avoid malloc overhead during streaming
    cudaError_t err = cudaMalloc(&d_batch_ints, max_capacity * sizeof(int32_t));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU buffer for DataLoader.");
    }

    // 1. Open the file mapping via Windows API
    hFile = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open binary dataset file: " + filename);
    }

    // 2. Determine file bounds
    LARGE_INTEGER size;
    GetFileSizeEx(hFile, &size);
    file_size = size.QuadPart;
    total_tokens = file_size / sizeof(int32_t);

    // 3. Create standard file mapping object
    hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapping == NULL) {
        CloseHandle(hFile);
        throw std::runtime_error("Failed to create file mapping.");
    }

    // 4. Map the entire view into the application's virtual address space
    mapped_data = (int32_t*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (mapped_data == NULL) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        throw std::runtime_error("Failed to map view of file.");
    }

    // 5. Pin the virtual memory! This is the core optimization for zero-copy streaming.
    // It locks the memory pages in physical RAM, giving the CUDA DMA engine direct access.
    err = cudaHostRegister(mapped_data, file_size, cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        std::cerr << "Warning: cudaHostRegister failed (code " << err << "). Transfers will fall back to slow pageable memory." << std::endl;
    }
}

BinaryDataLoader::~BinaryDataLoader() {
    if (d_batch_ints) cudaFree(d_batch_ints);
    if (mapped_data) {
        cudaHostUnregister(mapped_data); // Always unregister pinned memory
        UnmapViewOfFile(mapped_data);
    }
    if (hMapping) CloseHandle(hMapping);
    if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
}

std::shared_ptr<Tensor> BinaryDataLoader::get_next_batch(int batch_size, int seq_len, cudaStream_t stream) {
    int total_elements = batch_size * seq_len;
    if (total_elements > max_capacity) {
        throw std::invalid_argument("Batch size exceeds DataLoader pre-allocated capacity.");
    }
    
    // Infinite loop over dataset
    if (current_offset + total_elements > total_tokens) {
        current_offset = 0; 
    }

    auto out = std::make_shared<Tensor>(std::vector<int>{batch_size, seq_len}, false);
    
    // 1. DMA Async Stream: The GPU hardware directly pulls the memory over the PCIe bus
    // bypassing the CPU completely because mapped_data is pinned.
    cudaMemcpyAsync(d_batch_ints, mapped_data + current_offset, total_elements * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    
    // 2. Cast Kernel: Execute asynchronously on the same stream
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    cast_int_to_float_kernel<<<blocks, threads, 0, stream>>>(d_batch_ints, out->data.get(), total_elements);
    
    current_offset += total_elements;
    return out;
}

void BinaryDataLoader::reset() {
    current_offset = 0;
}
