#pragma once
#include "../core/tensor.h"
#include <memory>

std::shared_ptr<Tensor> gelu(std::shared_ptr<Tensor> input);
