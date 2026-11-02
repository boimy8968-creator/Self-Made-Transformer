#pragma once
#include "../core/tensor.h"
#include <memory>

struct ProbIndex {
    float prob;
    int index;
};

// Applies temperature scaling to the logits
void temperature_scale(std::shared_ptr<Tensor> logits, float temperature);

// Applies Top-p (Nucleus) sampling and returns the selected token ID
// Expects logits to be [vocab_size]
int top_p_sample(std::shared_ptr<Tensor> logits, float p, float rand_val);
