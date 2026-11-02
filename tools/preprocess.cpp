#include "../data/tokenizer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <algorithm>
#include <sstream>

// Buffer limit to prevent out-of-memory errors on massive datasets
const size_t CHUNK_SIZE_TOKENS = 1000000; 
const int MAX_VOCAB_SIZE = 8192;
const int SPECIAL_TOKENS_COUNT = 4; // <pad>, <unk>, <bos>, <eos>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.txt> <output.bin>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    // Pass 1: Frequency count
    std::cout << "Starting Pass 1: Word frequency counting..." << std::endl;
    std::ifstream infile(input_path);
    if (!infile.is_open()) {
        std::cerr << "Failed to open input file: " << input_path << std::endl;
        return 1;
    }

    std::unordered_map<std::string, uint64_t> word_freqs;
    std::string line;
    uint64_t lines_processed = 0;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string word;
        while (ss >> word) {
            word_freqs[word]++;
        }
        lines_processed++;
        if (lines_processed % 1000000 == 0) {
            std::cout << "  Processed " << lines_processed << " lines..." << std::endl;
        }
    }
    infile.close();

    std::cout << "Found " << word_freqs.size() << " unique words. Sorting..." << std::endl;
    std::vector<std::pair<std::string, uint64_t>> sorted_freqs(word_freqs.begin(), word_freqs.end());
    std::sort(sorted_freqs.begin(), sorted_freqs.end(), [](const auto& a, const auto& b) {
        return a.second > b.second; // Descending
    });

    Tokenizer tokenizer; // Internally adds 4 special tokens

    int words_to_add = std::min(MAX_VOCAB_SIZE - SPECIAL_TOKENS_COUNT, (int)sorted_freqs.size());
    for (int i = 0; i < words_to_add; ++i) {
        tokenizer.add_token(sorted_freqs[i].first);
    }

    std::cout << "Vocabulary built with " << tokenizer.vocab_size << " tokens." << std::endl;

    if (tokenizer.save_vocab("data/vocab.txt")) {
        std::cout << "Saved vocabulary to data/vocab.txt" << std::endl;
    } else {
        std::cerr << "Failed to save vocabulary." << std::endl;
    }

    // Pass 2: Encoding
    std::cout << "Starting Pass 2: Tokenization stream..." << std::endl;
    infile.open(input_path);
    if (!infile.is_open()) {
        std::cerr << "Failed to re-open input file: " << input_path << std::endl;
        return 1;
    }

    std::ofstream outfile(output_path, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return 1;
    }

    std::vector<int32_t> buffer;
    buffer.reserve(CHUNK_SIZE_TOKENS);
    uint64_t total_tokens = 0;

    while (std::getline(infile, line)) {
        if (line.empty()) continue;

        // Encode the string (update_vocab = false)
        std::vector<int> tokens = tokenizer.encode(line, false);
        
        for (int token_id : tokens) {
            buffer.push_back(static_cast<int32_t>(token_id));
            
            // Chunking: Flush to disk if buffer reaches capacity
            if (buffer.size() >= CHUNK_SIZE_TOKENS) {
                outfile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(int32_t));
                total_tokens += buffer.size();
                buffer.clear();
            }
        }
    }

    // Flush any remaining tokens in the buffer
    if (!buffer.empty()) {
        outfile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(int32_t));
        total_tokens += buffer.size();
        buffer.clear();
    }

    infile.close();
    outfile.close();

    std::cout << "Preprocessing complete!" << std::endl;
    std::cout << "Successfully wrote " << total_tokens << " tokens to " << output_path << std::endl;
    
    return 0;
}
