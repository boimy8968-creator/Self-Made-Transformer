#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <memory>
#include <cuda_runtime.h>

#include "data/data_loader.h"
#include "layers/encoder.h"
#include "training/optimizer.h"
#include "training/clip_grad.h"
#include "core/graph_runner.h"
#include "training/checkpoint.h"
#include "data/tokenizer.h"
#include "training/loss.h"
#include "core/autograd.h"
#include "inference/inference.h"

// Collect parameters for AdamW (simulating PyTorch's model.parameters())
#include <unordered_set>

std::vector<std::shared_ptr<Tensor>> collect_params(TransformerEncoder& model) {
    std::vector<std::shared_ptr<Tensor>> params;
    std::unordered_set<void*> seen;

    auto add_param = [&](std::shared_ptr<Tensor> p) {
        if (p && seen.find(p->data.get()) == seen.end()) {
            seen.insert(p->data.get());
            params.push_back(p);
        }
    };

    add_param(model.token_embed.weight);
    add_param(model.pos_embed.pe_embed.weight);
    for (auto& block : model.blocks) {
        add_param(block->mhsa.q_proj->weight);
        add_param(block->mhsa.q_proj->bias);
        add_param(block->mhsa.k_proj->weight);
        add_param(block->mhsa.k_proj->bias);
        add_param(block->mhsa.v_proj->weight);
        add_param(block->mhsa.v_proj->bias);
        add_param(block->mhsa.out_proj->weight);
        add_param(block->mhsa.out_proj->bias);

        add_param(block->ln1.weight);
        add_param(block->ln1.bias);

        add_param(block->mlp1.weight);
        add_param(block->mlp1.bias);

        add_param(block->mlp2.weight);
        add_param(block->mlp2.bias);

        add_param(block->ln2.weight);
        add_param(block->ln2.bias);
    }
    add_param(model.final_ln.weight);
    add_param(model.final_ln.bias);
    return params;
}

__global__ void cast_and_shift_targets_kernel(const float* float_input, int* int_targets, int B, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * S) {
        // Simple shift: target for token i is token i+1.
        // For the last token, we can just predict 0 (padding/eos).
        int b = idx / S;
        int s = idx % S;
        if (s < S - 1) {
            int_targets[idx] = (int)float_input[idx + 1];
        } else {
            int_targets[idx] = 0; // Pad/EOS
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --train | --chat" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    // Hyperparameters
    int B = 16;
    int S = 128;
    int D = 256;
    int H = 8;
    int L = 2;
    Tokenizer tokenizer;
    
    // Always attempt to load vocabulary so that model architecture matches checkpoint/data
    if (tokenizer.load_vocab("data/vocab.txt")) {
        std::cout << "Loaded vocab.txt with " << tokenizer.vocab_size << " tokens." << std::endl;
    } else {
        std::cout << "Warning: Failed to load data/vocab.txt. Vocabulary is only special tokens." << std::endl;
    }

    int V = tokenizer.vocab_size; // Dynamically match the vocabulary
    if (V < 30000) { 
        // If it's a tiny test vocab, we might still want a reasonable embedding size, but let's just stick to V.
    }

    TransformerEncoder model(V, D, H, L, S);
    
    // LM Head with weight tying to token embeddings
    std::shared_ptr<Linear> lm_head = std::make_shared<Linear>(D, V, false);
    lm_head->weight = model.token_embed.weight;

    auto params = collect_params(model);
    AdamW optimizer(params, 5e-4f, 0.9f, 0.999f, 1e-8f, 0.1f);

    if (mode == "--train") {
        std::cout << "Initializing Training Mode..." << std::endl;
        
        // Dummy dataset file for BinaryDataLoader
        BinaryDataLoader loader("data/train_tokens.bin", B, S);
        int* d_targets;
        cudaMallocAsync(&d_targets, B * S * sizeof(int), cudaStreamPerThread);
        
        std::cout << "Commencing eager training loop..." << std::endl;
        
        int steps = 10000;
        
        for (int i = 1; i <= steps; ++i) {
            // Explicitly zero gradients before forward pass
            for (auto& p : params) {
                if (p->grad) {
                    cudaMemset(p->grad.get(), 0, p->total_size * sizeof(float));
                }
            }

            auto input_ids = loader.get_next_batch(B, S);

            // Target shift
            int threads = 256;
            int blocks = (B * S + threads - 1) / threads;
            cast_and_shift_targets_kernel<<<blocks, threads>>>(input_ids->data.get(), d_targets, B, S);

            // Forward
            auto encoded = model.forward(input_ids);
            encoded->shape = {B * S, D};
            auto logits = lm_head->forward(encoded);

            // Loss & Backward
            auto loss = cross_entropy_loss(logits, d_targets);

            // Forward NaN Validation Trap
            float h_loss_val = 0.0f;
            cudaMemcpy(&h_loss_val, loss->data.get(), sizeof(float), cudaMemcpyDeviceToHost);
            if (std::isnan(h_loss_val) || std::isinf(h_loss_val)) {
                printf("[FATAL] Forward NaN/Inf detected at step %d: loss = %f\n", i, h_loss_val);
                exit(1);
            }

            backward(loss);

            // Backward NaN Validation Trap
            float lm_grad_norm = lm_head->weight->grad_norm();
            float l0_q_grad_norm = model.blocks[0]->mhsa.q_proj->weight->grad_norm();
            if (std::isnan(lm_grad_norm) || std::isinf(lm_grad_norm) ||
                std::isnan(l0_q_grad_norm) || std::isinf(l0_q_grad_norm)) {
                printf("[FATAL] Backward NaN/Inf detected at step %d: LM Head grad = %f, L0 Q grad = %f\n", i, lm_grad_norm, l0_q_grad_norm);
                exit(1);
            }

            clip_grad_norm(params, 1.0f);
            if (i % 50 == 0) {
                std::cout << "[Grad Norms] LM Head: " << lm_grad_norm 
                          << " | Layer 0 Query: " << l0_q_grad_norm << std::endl;
                std::cout << "[Step " << i << "] Eager Cross-Entropy Loss: " << h_loss_val << std::endl;
            }

            // Step
            optimizer.step();
            if (i % 1000 == 0) {
                cudaDeviceSynchronize();
                save_checkpoint("model_ckpt.bin", model, optimizer, tokenizer);
                std::cout << "--> Checkpoint saved at step " << i << std::endl;
            }

            // Explicitly destroy the computational graph to free VRAM immediately
            loss.reset();
            logits.reset();
            encoded.reset();
            input_ids.reset();
        }
        
    } else if (mode == "--chat") {
        std::cout << "Initializing Chat Mode..." << std::endl;
        
        try {
            load_checkpoint("model_ckpt.bin", model, optimizer, tokenizer);
            std::cout << "Checkpoint loaded successfully." << std::endl;
        } catch (...) {
            std::cout << "No checkpoint found. Proceeding with random initialized weights." << std::endl;
        }

        // Inference loop
        std::string prompt;
        while (true) {
            std::cout << "\nUser: ";
            if (!std::getline(std::cin, prompt)) break;
            if (prompt == "exit" || prompt == "quit") break;

            // Debug print
            std::vector<int> encoded_prompt = tokenizer.encode(prompt);
            std::cout << "Debug [Encoded]: ";
            for (int id : encoded_prompt) std::cout << id << " ";
            std::cout << std::endl;

            std::cout << "Bot: ";
            run_autoregressive_inference(model, *lm_head, tokenizer, prompt, 32, 0.8f, 0.9f);
        }
    } else {
        std::cerr << "Invalid mode: " << mode << "\nUse --train or --chat" << std::endl;
        return 1;
    }

    return 0;
}
