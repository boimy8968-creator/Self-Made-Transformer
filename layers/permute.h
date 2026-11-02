#pragma once
#include "../core/tensor.h"
#include <memory>
#include <vector>

// Permutes the dimensions of a tensor
std::shared_ptr<Tensor> permute(std::shared_ptr<Tensor> input, const std::vector<int>& dims);

// View reshapes a tensor without changing its physical memory (if possible, here we will allocate for safety to maintain contiguous property for BMM)
std::shared_ptr<Tensor> view(std::shared_ptr<Tensor> input, const std::vector<int>& shape);
