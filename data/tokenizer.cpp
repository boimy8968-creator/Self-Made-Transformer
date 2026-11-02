#include "../data/tokenizer.h"
#include <sstream>
#include <fstream>

Tokenizer::Tokenizer() {
    add_token("<pad>"); // 0
    add_token("<unk>"); // 1
    add_token("<bos>"); // 2
    add_token("<eos>"); // 3
}

void Tokenizer::add_token(const std::string& token) {
    if (token_to_id.find(token) == token_to_id.end()) {
        int id = token_to_id.size();
        token_to_id[token] = id;
        id_to_token[id] = token;
        vocab_size = token_to_id.size();
    }
}

bool Tokenizer::load_vocab(const std::string& vocab_path) {
    std::ifstream infile(vocab_path);
    if (!infile.is_open()) return false;
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty()) {
            add_token(line);
        }
    }
    return true;
}

bool Tokenizer::save_vocab(const std::string& vocab_path) {
    std::ofstream outfile(vocab_path);
    if (!outfile.is_open()) return false;
    // Save tokens in order of ID
    for (int i = 0; i < vocab_size; ++i) {
        if (id_to_token.find(i) != id_to_token.end()) {
            outfile << id_to_token[i] << "\n";
        }
    }
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool update_vocab) {
    std::vector<int> ids;
    ids.push_back(token_to_id["<bos>"]);
    
    std::stringstream ss(text);
    std::string word;
    while (ss >> word) {
        if (token_to_id.find(word) != token_to_id.end()) {
            ids.push_back(token_to_id[word]);
        } else {
            if (update_vocab) {
                add_token(word);
                ids.push_back(token_to_id[word]);
            } else {
                ids.push_back(token_to_id["<unk>"]);
            }
        }
    }
    ids.push_back(token_to_id["<eos>"]);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string text = "";
    for (int i = 0; i < ids.size(); ++i) {
        if (id_to_token.find(ids[i]) != id_to_token.end()) {
            if (ids[i] >= 4) { // Don't print special tokens
                text += id_to_token[ids[i]] + (i == ids.size() - 1 ? "" : " ");
            }
        }
    }
    return text;
}
