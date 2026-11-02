#include "../inference/inference.h"
#include "../inference/inference.h"
#include "../inference/sampling.h"
#include <iostream>
#include <curand.h>
#include <random>

void run_autoregressive_inference(
    TransformerEncoder& model, 
    Linear& lm_head,
    Tokenizer& tokenizer, 
    const std::string& prompt, 
    int max_new_tokens, 
    float temperature, 
    float top_p) 
{
    // 1. Encode initial prompt
    std::vector<int> tokens = tokenizer.encode(prompt);
    std::cout << prompt;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int step = 0; step < max_new_tokens; ++step) {
        int seq_len = tokens.size();
        
        // 2. Prepare Input Tensor
        auto input_ids = std::make_shared<Tensor>(std::vector<int>{1, seq_len}, false);
        std::vector<float> input_ids_f(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            input_ids_f[i] = (float)tokens[i];
        }
        input_ids->fill(input_ids_f);
        
        // 3. Forward Pass
        auto hidden_states = model.forward(input_ids);
        auto logits = lm_head.forward(hidden_states);
        
        // 4. Extract Logits for the Last Token
        // Logits shape: [1, seq_len, vocab_size]
        int vocab_size = logits->shape[2];
        auto last_token_logits = std::make_shared<Tensor>(std::vector<int>{vocab_size}, false);
        
        // Copy the last row of the logits to our 1D tensor
        cudaMemcpy(
            last_token_logits->data.get(), 
            logits->data.get() + (seq_len - 1) * vocab_size, 
            vocab_size * sizeof(float), 
            cudaMemcpyDeviceToDevice
        );
        
        // 5. Apply Temperature Scaling
        if (temperature != 1.0f && temperature > 0.0f) {
            temperature_scale(last_token_logits, temperature);
        }
        
        // 6. Convert to Probabilities
        auto probs = softmax(last_token_logits);
        
        // 7. Top-p (Nucleus) Sampling
        float rand_val = dis(gen);
        int next_token = top_p_sample(probs, top_p, rand_val);
        
        // 8. Append and Print
        tokens.push_back(next_token);
        if (next_token >= 4) {
            std::string decoded = tokenizer.id_to_token[next_token];
            std::cout << " " << decoded << std::flush;
        }
        
        if (next_token == tokenizer.token_to_id["<eos>"]) {
            break;
        }
    }
    std::cout << std::endl;
}


void copy_weights_to_inference(TransformerEncoder& src, AutoregressiveModel& dst) {
    for (size_t i = 0; i < src.blocks.size(); ++i) {
        dst.blocks[i]->ln1->weight = src.blocks[i]->ln1.weight;
        dst.blocks[i]->ln1->bias = src.blocks[i]->ln1.bias;
        dst.blocks[i]->attn->q_proj->weight = src.blocks[i]->mhsa.q_proj->weight;
        dst.blocks[i]->attn->q_proj->bias = src.blocks[i]->mhsa.q_proj->bias;
        dst.blocks[i]->attn->k_proj->weight = src.blocks[i]->mhsa.k_proj->weight;
        dst.blocks[i]->attn->k_proj->bias = src.blocks[i]->mhsa.k_proj->bias;
        dst.blocks[i]->attn->v_proj->weight = src.blocks[i]->mhsa.v_proj->weight;
        dst.blocks[i]->attn->v_proj->bias = src.blocks[i]->mhsa.v_proj->bias;
        dst.blocks[i]->attn->out_proj->weight = src.blocks[i]->mhsa.out_proj->weight;
        dst.blocks[i]->attn->out_proj->bias = src.blocks[i]->mhsa.out_proj->bias;
        dst.blocks[i]->ln2->weight = src.blocks[i]->ln2.weight;
        dst.blocks[i]->ln2->bias = src.blocks[i]->ln2.bias;
        dst.blocks[i]->ffn1->weight = src.blocks[i]->mlp1.weight;
        dst.blocks[i]->ffn1->bias = src.blocks[i]->mlp1.bias;
        dst.blocks[i]->ffn2->weight = src.blocks[i]->mlp2.weight;
        dst.blocks[i]->ffn2->bias = src.blocks[i]->mlp2.bias;
    }
    // We assume lm_head weight is tied to token_embed.weight
    dst.lm_head->weight = src.token_embed.weight;
}

