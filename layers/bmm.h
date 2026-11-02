#pragma once
#include "../core/tensor.h"
#include <memory>

// Batched Matrix Multiplication
// If transB is true, it computes A * B^T. 
// A is expected to be [..., M, K]
// If transB is false, B is expected to be [..., K, N]
// If transB is true, B is expected to be [..., N, K]
std::shared_ptr<Tensor> bmm(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B, bool transB = false);
