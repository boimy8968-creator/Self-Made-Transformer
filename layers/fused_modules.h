#pragma once
#include "../core/tensor.h"
#include <memory>

std::shared_ptr<Tensor> fused_add_gelu(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> bias);
std::shared_ptr<Tensor> fused_scale_softmax(std::shared_ptr<Tensor> input, float scale_factor);
