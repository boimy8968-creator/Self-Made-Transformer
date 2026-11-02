#pragma once
#include "../core/tensor.h"
#include <memory>

// The Orchestrator: Triggers the backward pass through the whole graph
void backward(std::shared_ptr<Tensor> loss_node);

// The Cost Function: Mean Squared Error
std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> predictions, std::shared_ptr<Tensor> targets);