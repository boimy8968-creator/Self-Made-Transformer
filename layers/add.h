#pragma once
#include "../core/tensor.h"
#include <memory>

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B);
