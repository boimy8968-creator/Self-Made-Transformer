#include "../training/checkpoint.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

void save_tensor(std::ofstream& out, std::shared_ptr<Tensor> t) {
    if (!t) return;
    std::vector<float> host_data(t->total_size);
    cudaMemcpy(host_data.data(), t->data.get(), t->total_size * sizeof(float), cudaMemcpyDeviceToHost);
    out.write(reinterpret_cast<const char*>(host_data.data()), t->total_size * sizeof(float));
}

void load_tensor(std::ifstream& in, std::shared_ptr<Tensor> t) {
    if (!t) return;
    std::vector<float> host_data(t->total_size);
    in.read(reinterpret_cast<char*>(host_data.data()), t->total_size * sizeof(float));
    cudaMemcpy(t->data.get(), host_data.data(), t->total_size * sizeof(float), cudaMemcpyHostToDevice);
}

void save_checkpoint(const std::string& filepath, TransformerEncoder& model, AdamW& optimizer, Tokenizer& tokenizer) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing checkpoint");
    }

    // 1. Tokenizer
    out.write(reinterpret_cast<const char*>(&tokenizer.vocab_size), sizeof(int));
    int map_size = tokenizer.token_to_id.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(int));
    for (const auto& pair : tokenizer.token_to_id) {
        int str_len = pair.first.size();
        out.write(reinterpret_cast<const char*>(&str_len), sizeof(int));
        out.write(pair.first.c_str(), str_len);
        out.write(reinterpret_cast<const char*>(&pair.second), sizeof(int));
    }

    // 2. TransformerEncoder Weights
    save_tensor(out, model.token_embed.weight);
    save_tensor(out, model.pos_embed.pe_embed.weight);
    for (auto& block : model.blocks) {
        save_tensor(out, block->mhsa.q_proj->weight);
        if (block->mhsa.q_proj->bias) save_tensor(out, block->mhsa.q_proj->bias);
        save_tensor(out, block->mhsa.k_proj->weight);
        if (block->mhsa.k_proj->bias) save_tensor(out, block->mhsa.k_proj->bias);
        save_tensor(out, block->mhsa.v_proj->weight);
        if (block->mhsa.v_proj->bias) save_tensor(out, block->mhsa.v_proj->bias);
        save_tensor(out, block->mhsa.out_proj->weight);
        if (block->mhsa.out_proj->bias) save_tensor(out, block->mhsa.out_proj->bias);

        save_tensor(out, block->ln1.weight);
        save_tensor(out, block->ln1.bias);

        save_tensor(out, block->mlp1.weight);
        if (block->mlp1.bias) save_tensor(out, block->mlp1.bias);

        save_tensor(out, block->mlp2.weight);
        if (block->mlp2.bias) save_tensor(out, block->mlp2.bias);

        save_tensor(out, block->ln2.weight);
        save_tensor(out, block->ln2.bias);
    }
    save_tensor(out, model.final_ln.weight);
    save_tensor(out, model.final_ln.bias);

    // 3. AdamW State
    out.write(reinterpret_cast<const char*>(&optimizer.t), sizeof(int));
    for (size_t i = 0; i < optimizer.params.size(); ++i) {
        int size = optimizer.params[i]->total_size;
        std::vector<float> m(size);
        std::vector<float> v(size);
        cudaMemcpy(m.data(), optimizer.m_buffers[i], size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(v.data(), optimizer.v_buffers[i], size * sizeof(float), cudaMemcpyDeviceToHost);
        out.write(reinterpret_cast<const char*>(m.data()), size * sizeof(float));
        out.write(reinterpret_cast<const char*>(v.data()), size * sizeof(float));
    }
    
    out.close();
}

void load_checkpoint(const std::string& filepath, TransformerEncoder& model, AdamW& optimizer, Tokenizer& tokenizer) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading checkpoint");
    }

    // 1. Tokenizer
    in.read(reinterpret_cast<char*>(&tokenizer.vocab_size), sizeof(int));
    int map_size = 0;
    in.read(reinterpret_cast<char*>(&map_size), sizeof(int));
    tokenizer.token_to_id.clear();
    tokenizer.id_to_token.clear();
    for (int i = 0; i < map_size; ++i) {
        int str_len = 0;
        in.read(reinterpret_cast<char*>(&str_len), sizeof(int));
        std::string s(str_len, ' ');
        in.read(&s[0], str_len);
        int id = 0;
        in.read(reinterpret_cast<char*>(&id), sizeof(int));
        tokenizer.token_to_id[s] = id;
        tokenizer.id_to_token[id] = s;
    }

    // 2. TransformerEncoder Weights
    load_tensor(in, model.token_embed.weight);
    load_tensor(in, model.pos_embed.pe_embed.weight);
    for (auto& block : model.blocks) {
        load_tensor(in, block->mhsa.q_proj->weight);
        if (block->mhsa.q_proj->bias) load_tensor(in, block->mhsa.q_proj->bias);
        load_tensor(in, block->mhsa.k_proj->weight);
        if (block->mhsa.k_proj->bias) load_tensor(in, block->mhsa.k_proj->bias);
        load_tensor(in, block->mhsa.v_proj->weight);
        if (block->mhsa.v_proj->bias) load_tensor(in, block->mhsa.v_proj->bias);
        load_tensor(in, block->mhsa.out_proj->weight);
        if (block->mhsa.out_proj->bias) load_tensor(in, block->mhsa.out_proj->bias);

        load_tensor(in, block->ln1.weight);
        load_tensor(in, block->ln1.bias);

        load_tensor(in, block->mlp1.weight);
        if (block->mlp1.bias) load_tensor(in, block->mlp1.bias);

        load_tensor(in, block->mlp2.weight);
        if (block->mlp2.bias) load_tensor(in, block->mlp2.bias);

        load_tensor(in, block->ln2.weight);
        load_tensor(in, block->ln2.bias);
    }
    load_tensor(in, model.final_ln.weight);
    load_tensor(in, model.final_ln.bias);

    // 3. AdamW State
    in.read(reinterpret_cast<char*>(&optimizer.t), sizeof(int));
    for (size_t i = 0; i < optimizer.params.size(); ++i) {
        int size = optimizer.params[i]->total_size;
        std::vector<float> m(size);
        std::vector<float> v(size);
        in.read(reinterpret_cast<char*>(m.data()), size * sizeof(float));
        in.read(reinterpret_cast<char*>(v.data()), size * sizeof(float));
        cudaMemcpy(optimizer.m_buffers[i], m.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(optimizer.v_buffers[i], v.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    }

    in.close();
}
