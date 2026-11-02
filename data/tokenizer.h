#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    int vocab_size;

    Tokenizer();
    void add_token(const std::string& token);
    bool load_vocab(const std::string& vocab_path);
    bool save_vocab(const std::string& vocab_path);
    std::vector<int> encode(const std::string& text, bool update_vocab = false);
    std::string decode(const std::vector<int>& ids);
};
