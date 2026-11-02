#include "tokenizer.h"
#include "encoder.h"
#include "autograd.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>

void check_gradients(std::shared_ptr<Tensor> t, const std::string& name) {
    if (!t || !t->grad) {
        std::cerr << "No gradient for " << name << std::endl;
        assert(false);
    }
    std::vector<float> g(t->total_size);
    cudaMemcpy(g.data(), t->grad.get(), t->total_size * sizeof(float), cudaMemcpyDeviceToHost);
    bool has_nonzero = false;
    for (float v : g) {
        if (std::isnan(v) || std::isinf(v)) {
            std::cerr << "NaN/Inf in " << name << std::endl;
            assert(false);
        }
        if (v != 0.0f) has_nonzero = true;
    }
    if (!has_nonzero) {
        std::cerr << "Zero gradient in " << name << std::endl;
        assert(false);
    }
    std::cout << "[PASS] " << name << " gradients are finite and non-zero." << std::endl;
}

int main() {
    std::cout << "Starting Transformer Encoder Pipeline Test..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Tokenizer
    Tokenizer tokenizer;
    std::string text = "CUDA Transformer Engine running natively.";
    // Add words to vocab
    tokenizer.add_token("CUDA");
    tokenizer.add_token("Transformer");
    tokenizer.add_token("Engine");
    tokenizer.add_token("running");
    tokenizer.add_token("natively.");
    
    std::vector<int> tokens = tokenizer.encode(text);
    std::cout << "Tokenized string to " << tokens.size() << " tokens." << std::endl;
    
    // Pad to Seq=8 if needed
    int Seq = 8;
    while (tokens.size() < Seq) {
        tokens.push_back(tokenizer.token_to_id["<pad>"]);
    }
    if (tokens.size() > Seq) {
        tokens.resize(Seq);
    }
    
    int Batch = 1;
    int Dim = 64;
    int Heads = 4;
    int Blocks = 2;
    int VocabSize = tokenizer.vocab_size;
    int MaxSeq = 64;

    // Build Model
    TransformerEncoder encoder(VocabSize, Dim, Heads, Blocks, MaxSeq);

    // Prepare Input Tensor
    auto input_ids = std::make_shared<Tensor>(std::vector<int>{Batch, Seq}, false);
    std::vector<float> input_ids_f(Batch * Seq);
    for (int i = 0; i < Seq; ++i) {
        input_ids_f[i] = (float)tokens[i];
    }
    input_ids->fill(input_ids_f);

    // 2. Parity & Correctness Audit for Embeddings
    auto emb_out = encoder.token_embed.forward(input_ids);
    
    // Copy weights and embeddings back to CPU for verification
    std::vector<float> w_cpu(VocabSize * Dim);
    cudaMemcpy(w_cpu.data(), encoder.token_embed.weight->data.get(), VocabSize * Dim * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::vector<float> e_cpu(Batch * Seq * Dim);
    cudaMemcpy(e_cpu.data(), emb_out->data.get(), Batch * Seq * Dim * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool embed_match = true;
    for (int s = 0; s < Seq; ++s) {
        int token_id = tokens[s];
        for (int d = 0; d < Dim; ++d) {
            float expected = w_cpu[token_id * Dim + d];
            float actual = e_cpu[s * Dim + d];
            if (std::abs(expected - actual) > 1e-5f) {
                embed_match = false;
            }
        }
    }
    if (embed_match) {
        std::cout << "[PASS] Token embeddings perfectly align with weight matrix rows." << std::endl;
    } else {
        std::cerr << "[FAIL] Token embedding mismatch!" << std::endl;
        assert(false);
    }

    // 3. Full Forward Pass
    auto final_out = encoder.forward(input_ids);
    
    auto fwd_end = std::chrono::high_resolution_clock::now();
    std::cout << "Forward Pass Complete." << std::endl;

    // 4. Backward Pass (Dummy loss)
    auto target = std::make_shared<Tensor>(final_out->shape, false);
    std::vector<float> zeros(final_out->total_size, 0.0f);
    target->fill(zeros);
    auto loss = mse_loss(final_out, target);
    
    backward(loss);
    // cudaDeviceSynchronize();
    
    auto bwd_end = std::chrono::high_resolution_clock::now();
    std::cout << "Backward Pass Complete." << std::endl;

    std::chrono::duration<double, std::milli> fwd_time = fwd_end - start_time;
    std::chrono::duration<double, std::milli> bwd_time = bwd_end - fwd_end;

    std::cout << "Forward Latency: " << fwd_time.count() << " ms" << std::endl;
    std::cout << "Backward Latency: " << bwd_time.count() << " ms" << std::endl;

    // 5. Verify Gradients Flow
    check_gradients(encoder.token_embed.weight, "Embedding Table");
    check_gradients(encoder.pos_embed.pe_embed.weight, "Positional Encoding Table");
    
    for (int i = 0; i < Blocks; ++i) {
        std::string prefix = "Block " + std::to_string(i) + " ";
        check_gradients(encoder.blocks[i]->mhsa.q_proj->weight, prefix + "W_Q");
        check_gradients(encoder.blocks[i]->mhsa.k_proj->weight, prefix + "W_K");
        check_gradients(encoder.blocks[i]->mhsa.v_proj->weight, prefix + "W_V");
        check_gradients(encoder.blocks[i]->mhsa.out_proj->weight, prefix + "W_O");
        check_gradients(encoder.blocks[i]->ln1.weight, prefix + "LN1 Gamma");
        check_gradients(encoder.blocks[i]->ln2.weight, prefix + "LN2 Gamma");
        check_gradients(encoder.blocks[i]->mlp1.weight, prefix + "MLP1 W");
        check_gradients(encoder.blocks[i]->mlp2.weight, prefix + "MLP2 W");
    }
    
    check_gradients(encoder.final_ln.weight, "Final LN Gamma");

    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
