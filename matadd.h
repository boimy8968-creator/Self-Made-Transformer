#pragma once
#include "tensor.h"
#include <memory>

// Forward Pass & Graph Builder
std::shared_ptr<Tensor> matadd(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B);