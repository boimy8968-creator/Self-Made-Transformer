#include "tensor.h"
#include "bmm.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

void test_bmm() {
    std::cout << "Testing Batched Matrix Multiplication (Forward & Backward)..." << std::endl;

    // Test transB = true (Q * K^T)
    // A: [2, 4, 8] (Batch 2, M=4, K=8)
    // B: [2, 4, 8] (Batch 2, N=4, K=8)
    // C: [2, 4, 4]
    
    auto A = std::make_shared<Tensor>(std::vector<int>{2, 4, 8}, true);
    auto B = std::make_shared<Tensor>(std::vector<int>{2, 4, 8}, true);

    std::vector<float> data_A(64, 1.0f);
    std::vector<float> data_B(64, 2.0f);
    A->fill(data_A);
    B->fill(data_B);

    auto C = bmm(A, B, true);
    // cudaDeviceSynchronize();

    std::vector<float> h_C(32);
    cudaMemcpy(h_C.data(), C->data.get(), 32 * sizeof(float), cudaMemcpyDeviceToHost);

    // Each element in C should be 8 * (1.0 * 2.0) = 16.0
    assert(h_C[0] == 16.0f);
    assert(h_C[31] == 16.0f);

    std::cout << "BMM Test Passed!" << std::endl;
}

int main() {
    test_bmm();
    return 0;
}
