#pragma once
#include "tensor.h"
#include <memory>

// Forward Pass & Autograd Graph Builder
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B);