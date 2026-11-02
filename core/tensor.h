#pragma once
#include <vector>
#include <memory>

#include <stdexcept>
#include <cuda_runtime.h>

using namespace std;

struct IClosure { virtual void operator()() = 0; virtual ~IClosure() = default; };

template<typename F> struct Closure : public IClosure { F f; Closure(F f) : f(f) {} void operator()() override { f(); } };

struct Tensor {
    // ==========================================
    // Core Tensor Data
    // ==========================================
    vector<int> shape;
    vector<int> strides;
    int total_size;
    shared_ptr<float> data; // VRAM Pointer for mathematical values

    // ==========================================
    // Autograd Graph Architecture
    // ==========================================
    bool requires_grad;
    shared_ptr<float> grad; // VRAM Pointer for gradients
    vector<shared_ptr<Tensor>> parents; // Keeps upstream tensors alive in RAM
    struct ClosurePtr { std::shared_ptr<IClosure> ptr; template<typename F> ClosurePtr& operator=(F f) { ptr = std::make_shared<Closure<F>>(f); return *this; } operator bool() const { return ptr != nullptr; } void operator()() { if (ptr) (*ptr)(); } }; ClosurePtr backward_closure; // The Lambda Time Capsule

    // ==========================================
    // Methods
    // ==========================================
    Tensor(const vector<int>& target_shape, bool requires_grad = false);

    void compute_strides();
    int get_offset(const vector<int>& indices) const;
    void fill(const vector<float>& values);
    void print() const;

    // Autograd Utility
    void zero_grad();
    float grad_norm();
};


