#pragma once
#include "../core/tensor.h"
#include <memory>

// LayerNorm computes mean and variance over the last dimension of input
std::shared_ptr<Tensor> layernorm(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias, float eps = 1e-5f);
