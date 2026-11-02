#pragma once
#include "../core/tensor.h"
#include <vector>
#include <memory>

void clip_grad_norm(const std::vector<std::shared_ptr<Tensor>>& parameters, float max_norm);
