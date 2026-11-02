#include "../core/tensor.h"
#include <stdexcept>
#include <string>
#include <iostream>

using namespace std;

Tensor::Tensor(const vector<int>& target_shape, bool req_grad) {
    shape = target_shape;
    requires_grad = req_grad;

    if (shape.empty()) {
        total_size = 0;
        data = nullptr;
        grad = nullptr;
        return;
    }

    total_size = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw invalid_argument("Dimensions must be greater than zero.");
        }
        total_size *= dim;
    }

    // 1. Allocate VRAM for Mathematical Data
    float* raw_device_ptr = nullptr;
    cudaError_t err = cudaMallocAsync((void**)&raw_device_ptr, total_size * sizeof(float), cudaStreamPerThread);
    if (err != cudaSuccess) {
        throw runtime_error("CUDA Malloc failed for data: " + string(cudaGetErrorString(err)));
    }

    data = shared_ptr<float>(raw_device_ptr, [](float* ptr) {
        if (ptr != nullptr) cudaFreeAsync(ptr, cudaStreamPerThread);
        });
    cudaMemsetAsync(data.get(), 0, total_size * sizeof(float), cudaStreamPerThread);

    // 2. Allocate VRAM for Gradients (If requested)
    if (requires_grad) {
        float* raw_grad_ptr = nullptr;
        err = cudaMallocAsync((void**)&raw_grad_ptr, total_size * sizeof(float), cudaStreamPerThread);
        if (err != cudaSuccess) {
            throw runtime_error("CUDA Malloc failed for grad: " + string(cudaGetErrorString(err)));
        }

        grad = shared_ptr<float>(raw_grad_ptr, [](float* ptr) {
            if (ptr != nullptr) cudaFreeAsync(ptr, cudaStreamPerThread);
            });

        // Gradients start at exactly zero
        cudaMemset(grad.get(), 0, total_size * sizeof(float));
    }
    else {
        grad = nullptr;
    }

    // 3. Initialize an empty time capsule
    backward_closure = []() {};

    compute_strides();
}

void Tensor::compute_strides() {
    if (shape.empty()) return;
    strides.resize(shape.size());

    strides.back() = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

int Tensor::get_offset(const vector<int>& indices) const {
    if (indices.size() != shape.size()) {
        throw invalid_argument("Error: Number of indices does not match tensor dimensions.");
    }

    int offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw out_of_range("Error: Index out of bounds at dimension " + to_string(i));
        }
        offset += indices[i] * strides[i];
    }
    return offset;
}

void Tensor::fill(const vector<float>& values) {
    if (values.size() != total_size) {
        throw invalid_argument("Error: Number of values provided does not match tensor's total_size.");
    }
    cudaError_t err = cudaMemcpy(data.get(), values.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw runtime_error("CUDA Memcpy HostToDevice failed: " + string(cudaGetErrorString(err)));
    }
}

void Tensor::print() const {
    if (total_size == 0) {
        cout << "Empty Tensor" << endl;
        return;
    }

    vector<float> host_temp(total_size);
    cudaError_t err = cudaMemcpy(host_temp.data(), data.get(), total_size * sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        throw runtime_error("CUDA Memcpy DeviceToHost failed: " + string(cudaGetErrorString(err)));
    }

    int row_width = shape.back();

    cout << "Tensor Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    cout << "]" << endl;

    for (int i = 0; i < total_size; ++i) {
        cout << host_temp[i] << "\t";
        if ((i + 1) % row_width == 0) {
            cout << endl;
        }
    }
    cout << endl;
}

void Tensor::zero_grad() {
    if (requires_grad && grad != nullptr) {
        // Fast hardware reset of gradient memory
        cudaMemsetAsync(grad.get(), 0, total_size * sizeof(float), 0);
    }
}



float Tensor::grad_norm() {
    if (!requires_grad || !grad) return 0.0f;
    std::vector<float> h_grad(total_size);
    cudaMemcpy(h_grad.data(), grad.get(), total_size * sizeof(float), cudaMemcpyDeviceToHost);
    float sum_sq = 0.0f;
    for (float g : h_grad) sum_sq += g * g;
    return std::sqrt(sum_sq);
}

