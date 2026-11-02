#pragma once
#include "../core/tensor.h"
#include <memory>

std::shared_ptr<Tensor> apply_causal_mask(std::shared_ptr<Tensor> input);
