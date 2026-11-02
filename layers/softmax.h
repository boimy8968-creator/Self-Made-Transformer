#pragma once
#include "../core/tensor.h"
#include <memory>

std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> input);
