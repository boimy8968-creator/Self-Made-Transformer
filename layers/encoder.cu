#include "../layers/encoder.h"

TransformerEncoderBlock::TransformerEncoderBlock(int embed_dim, int num_heads) 
    : mhsa(embed_dim, num_heads), ln1(embed_dim), ln2(embed_dim),
      mlp1(embed_dim, embed_dim * 4), mlp2(embed_dim * 4, embed_dim) {}

std::shared_ptr<Tensor> TransformerEncoderBlock::forward(std::shared_ptr<Tensor> x) {
    // X1 = X + MHSA(LN1(X))
    auto ln1_out = ln1.forward(x);
    auto attn_out = mhsa.forward(ln1_out);
    auto x1 = add(x, attn_out);
    
    // X2 = X1 + MLP(LN2(X1))
    auto ln2_out = ln2.forward(x1);
    auto mlp_h = gelu(mlp1.forward(ln2_out));
    auto mlp_out = mlp2.forward(mlp_h);
    auto x2 = add(x1, mlp_out);
    
    return x2;
}

TransformerEncoder::TransformerEncoder(int vocab_size, int embed_dim, int num_heads, int num_blocks, int max_seq_len)
    : token_embed(vocab_size, embed_dim), pos_embed(max_seq_len, embed_dim), final_ln(embed_dim) {
    for (int i = 0; i < num_blocks; ++i) {
        blocks.push_back(std::make_shared<TransformerEncoderBlock>(embed_dim, num_heads));
    }
}

std::shared_ptr<Tensor> TransformerEncoder::forward(std::shared_ptr<Tensor> input_ids) {
    auto x = token_embed.forward(input_ids);
    x = pos_embed.forward(x);
    
    for (auto& block : blocks) {
        x = block->forward(x);
    }
    
    return final_ln.forward(x);
}
