#include "tensor.h"
#include "permute.h"
#include <iostream>
#include <vector>
#include <cassert>

void test_permute() {
    std::cout << "Testing Permute & View..." << std::endl;

    // Simulate splitting heads: [B=2, Seq=4, Dim=8] -> view -> [2, 4, 2, 4] (Heads=2, HeadDim=4)
    auto X = std::make_shared<Tensor>(std::vector<int>{2, 4, 8}, true);
    std::vector<float> data(64);
    for (int i = 0; i < 64; ++i) data[i] = i;
    X->fill(data);

    auto X_view = view(X, {2, 4, 2, 4});
    assert(X_view->shape.size() == 4);
    assert(X_view->shape[2] == 2);

    // Permute to [B=2, Heads=2, Seq=4, HeadDim=4]
    // Dims map: 0->0, 1->2, 2->1, 3->3
    auto X_permuted = permute(X_view, {0, 2, 1, 3});
    
    // cudaDeviceSynchronize();
    assert(X_permuted->shape[1] == 2);
    assert(X_permuted->shape[2] == 4);

    // Check memory alignment (Head 1, Seq 0 should be right after Head 0, Seq 0)
    std::vector<float> h_perm(64);
    cudaMemcpy(h_perm.data(), X_permuted->data.get(), 64 * sizeof(float), cudaMemcpyDeviceToHost);

    // X[0, 0, 1, 0] in permuted corresponds to X_view[0, 1, 0, 0] = original 8
    assert(h_perm[1 * (4*4) + 0] == 8.0f);

    std::cout << "Permute & View Test Passed!" << std::endl;
}

int main() {
    test_permute();
    return 0;
}
