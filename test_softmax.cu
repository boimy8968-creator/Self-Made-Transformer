#include "tensor.h"
#include "softmax.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

void test_softmax() {
    std::cout << "Testing Softmax (Forward & Backward)..." << std::endl;

    // 1. Forward Test: Batch size 2, Dim 3
    auto X = std::make_shared<Tensor>(std::vector<int>{2, 3}, true);
    
    // Fill with values that might overflow if naive e^x is used
    X->fill({1000.0f, 1000.0f, 1000.0f, // row 0: all same, prob = 0.333
              1.0f, 2.0f, 3.0f});       // row 1: e^1, e^2, e^3

    auto P = softmax(X);
    // cudaDeviceSynchronize();

    std::vector<float> h_P(6);
    cudaMemcpy(h_P.data(), P->data.get(), 6 * sizeof(float), cudaMemcpyDeviceToHost);

    // Assert row 0 probabilities are roughly 1/3
    assert(std::abs(h_P[0] - 0.3333f) < 1e-3);
    assert(std::abs(h_P[1] - 0.3333f) < 1e-3);
    assert(std::abs(h_P[2] - 0.3333f) < 1e-3);

    // 2. Backward Test
    // Seed gradient for P
    std::vector<float> dOut(6, 0.0f);
    dOut[0] = 1.0f; // Only grad for P[0][0]
    cudaMemcpy(P->grad.get(), dOut.data(), 6 * sizeof(float), cudaMemcpyHostToDevice);

    if (P->backward_closure) {
        P->backward_closure();
    }

    std::vector<float> dIn(6);
    cudaMemcpy(dIn.data(), X->grad.get(), 6 * sizeof(float), cudaMemcpyDeviceToHost);

    // dIn[0][0] = P[0][0] * (1.0 - P[0][0]*1.0) = 0.333 * (1 - 0.333) = 0.222
    // dIn[0][1] = P[0][1] * (0.0 - P[0][0]*1.0) = 0.333 * (-0.333) = -0.111
    assert(std::abs(dIn[0] - 0.2222f) < 1e-3);
    assert(std::abs(dIn[1] - -0.1111f) < 1e-3);

    std::cout << "Softmax Test Passed!" << std::endl;
}

int main() {
    test_softmax();
    return 0;
}
