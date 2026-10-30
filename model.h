#pragma once
#include "tensor.h"
#include "layers.h"
#include "attention.h"
#include <vector>
#include <string>

class SwiGLU {
public:
    Linear W1, W3, W2;
    size_t dim, hidden;

    SwiGLU() = default;
    SwiGLU(size_t dim, size_t hidden, float std = 0.02f);

    Tensor forward(const Tensor& x) const;
    Tensor backward(const Tensor& x, const Tensor& grad_y);
    void zero_grad();
};

class TransformerBlock {
public:
    RMSNorm norm1, norm2;
    MultiHeadAttention attn;
    SwiGLU ffn;
    size_t dim, n_head;

    TransformerBlock() = default;
    TransformerBlock(size_t dim, size_t n_head, float std = 0.02f);

    Tensor forward(const Tensor& x) const;
    Tensor backward(const Tensor& x, const Tensor& grad_y);
    void zero_grad();
};

class Transformer {
public:
    size_t vocab_size, dim, n_layer, n_head, max_seq_len;

    Tensor token_emb;
    Tensor grad_token_emb;
    std::vector<TransformerBlock> blocks;
    RMSNorm final_norm;
    Linear lm_head;

    Transformer() = default;
    Transformer(size_t vocab_size, size_t dim, size_t n_layer, size_t n_head,
                size_t max_seq_len, float std = 0.02f);

    Tensor forward(const Tensor& tokens) const;
    Tensor backward(const Tensor& tokens, const Tensor& grad_logits);
    void zero_grad();

    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);
    int sample_top_k(const Tensor& logits, int current_len, float temperature = 1.0f, int k = 40) const;
};
