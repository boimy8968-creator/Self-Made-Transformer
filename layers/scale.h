#pragma once
#include "../core/tensor.h"
#include <memory>

std::shared_ptr<Tensor> scale(std::shared_ptr<Tensor> input, float scalar);
