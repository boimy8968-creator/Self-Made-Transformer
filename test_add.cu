#include "tensor.h"
#include "add.h"
#include "autograd.h"
#include <iostream>
#include <vector>
#include <cassert>

void test_broadcast_add() {
    std::cout << "Testing Broadcast Add (Forward & Backward)..." << std::endl;

    // 1. Forward Test: A [2, 3] + B [3] = C [2, 3]
    auto A = std::make_shared<Tensor>(std::vector<int>{2, 3}, true);
    auto B = std::make_shared<Tensor>(std::vector<int>{3}, true);

    A->fill({1, 2, 3, 4, 5, 6});
    B->fill({10, 20, 30});

    auto C = add(A, B);

    // Synchronize and print C
    // cudaDeviceSynchronize();
    std::cout << "A + B = C shape: [" << C->shape[0] << ", " << C->shape[1] << "]" << std::endl;
    // Expected C: {11, 22, 33, 14, 25, 36}

    // 2. Backward Test
    // Initialize C's gradient to 1.0 (seed)
    std::vector<float> grad_seed(6, 1.0f);
    cudaMemcpy(C->grad.get(), grad_seed.data(), 6 * sizeof(float), cudaMemcpyHostToDevice);
    
    if (C->backward_closure) {
        C->backward_closure();
    }

    // A grad should be 1.0 everywhere
    // B grad should be {2.0, 2.0, 2.0} (sum over batch dimension)
    
    std::vector<float> h_grad_b(3);
    cudaMemcpy(h_grad_b.data(), B->grad.get(), 3 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Grad B (Bias): " << h_grad_b[0] << ", " << h_grad_b[1] << ", " << h_grad_b[2] << std::endl;
    assert(h_grad_b[0] == 2.0f);
    assert(h_grad_b[1] == 2.0f);
    assert(h_grad_b[2] == 2.0f);

    std::cout << "Broadcast Add Test Passed!" << std::endl;
}

int main() {
    test_broadcast_add();
    return 0;
}
