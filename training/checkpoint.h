#pragma once
#include <string>
#include "../layers/encoder.h"
#include "../training/optimizer.h"
#include "../data/tokenizer.h"

void save_checkpoint(const std::string& filepath, TransformerEncoder& model, AdamW& optimizer, Tokenizer& tokenizer);
void load_checkpoint(const std::string& filepath, TransformerEncoder& model, AdamW& optimizer, Tokenizer& tokenizer);
