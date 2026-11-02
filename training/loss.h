#pragma once
#include "../core/tensor.h"
#include <memory>

// Fused Softmax + Cross-Entropy Loss
// targets should be a GPU pointer to integers of size [Batch * Seq]
// logits should have shape [Batch * Seq, VocabSize] or [Batch, Seq, VocabSize]
std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> logits, const int* d_targets);
