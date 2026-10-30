#include "model.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <fstream>
#include <random>
#include <algorithm>
#include <iostream>

SwiGLU::SwiGLU(size_t d, size_t h, float std)
    : W1(d, h, std), W3(d, h, std), W2(h, d, std), dim(d), hidden(h) {}

Tensor SwiGLU::forward(const Tensor& x) const {
    assert(x.shape.size() == 3);
    size_t B = x.shape[0], T = x.shape[1], D = x.shape[2];
    Tensor x2d({B * T, D});
    std::memcpy(x2d.data.data(), x.data.data(), sizeof(float) * x.size);

    Tensor g = W1.forward(x2d);
    Tensor u = W3.forward(x2d);
    Tensor gu({B * T, hidden});
    for (size_t i = 0; i < gu.size; ++i)
        gu.data[i] = gelu_scalar(g.data[i]) * u.data[i];

    Tensor y2d = W2.forward(gu);
    Tensor y({B, T, D});
    std::memcpy(y.data.data(), y2d.data.data(), sizeof(float) * y2d.size);
    return y;
}

Tensor SwiGLU::backward(const Tensor& x, const Tensor& grad_y) {
    size_t B = x.shape[0], T = x.shape[1], D = x.shape[2];
    Tensor x2d({B * T, D});
    std::memcpy(x2d.data.data(), x.data.data(), sizeof(float) * x.size);
    Tensor g = W1.forward(x2d), u = W3.forward(x2d);
    Tensor gy2d({B * T, D});
    std::memcpy(gy2d.data.data(), grad_y.data.data(), sizeof(float) * grad_y.size);

    Tensor grad_out({B * T, hidden});
    for (size_t b = 0; b < B*T; ++b)
        for (size_t h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (size_t d = 0; d < D; ++d)
                acc += gy2d.data[b * D + d] * W2.W.data[h * D + d];
            grad_out.data[b * hidden + h] = acc;
        }

    Tensor gu({B*T, hidden});
    for (size_t i = 0; i < gu.size; ++i)
        gu.data[i] = gelu_scalar(g.data[i]) * u.data[i];

    for (size_t h = 0; h < hidden; ++h)
        for (size_t d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (size_t b = 0; b < B*T; ++b)
                acc += gu.data[b * hidden + h] * gy2d.data[b * D + d];
            W2.grad_W.data[h * D + d] += acc;
        }
    for (size_t d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (size_t b = 0; b < B*T; ++b) acc += gy2d.data[b * D + d];
        W2.grad_b.data[d] += acc;
    }

    Tensor gelu_g = gelu(g);
    Tensor grad_gelu = gelu_backward(g, grad_out);
    Tensor grad_g({B*T, hidden}), grad_u({B*T, hidden});
    for (size_t i = 0; i < grad_g.size; ++i) {
        grad_g.data[i] = grad_gelu.data[i] * u.data[i];
        grad_u.data[i] = grad_out.data[i] * gelu_g.data[i];
    }

    Tensor grad_x1 = W1.backward(x2d, grad_g);
    Tensor grad_x2 = W3.backward(x2d, grad_u);
    Tensor grad_x({B, T, D});
    for (size_t i = 0; i < grad_x.size; ++i)
        grad_x.data[i] = grad_x1.data[i] + grad_x2.data[i];
    return grad_x;
}

void SwiGLU::zero_grad() { W1.zero_grad(); W2.zero_grad(); W3.zero_grad(); }

TransformerBlock::TransformerBlock(size_t d, size_t h, float std)
    : norm1(d), norm2(d), attn(d, h, std), ffn(d, 4 * d, std), dim(d), n_head(h) {}

Tensor TransformerBlock::forward(const Tensor& x) const {
    Tensor h = norm1.forward(x);
    Tensor a = attn.forward(h);
    Tensor x1(x.shape);
    for (size_t i = 0; i < x.size; ++i) x1.data[i] = x.data[i] + a.data[i];
    Tensor h2 = norm2.forward(x1);
    Tensor f = ffn.forward(h2);
    Tensor x2(x.shape);
    for (size_t i = 0; i < x.size; ++i) x2.data[i] = x1.data[i] + f.data[i];
    return x2;
}

Tensor TransformerBlock::backward(const Tensor& x, const Tensor& grad_y) {
    Tensor h1 = norm1.forward(x), a = attn.forward(h1);
    Tensor x1(x.shape);
    for (size_t i = 0; i < x.size; ++i) x1.data[i] = x.data[i] + a.data[i];
    Tensor h2 = norm2.forward(x1);

    Tensor grad_h2 = ffn.backward(h2, grad_y);
    Tensor grad_x1_from_norm2 = norm2.backward(x1, grad_h2);

    Tensor grad_x1(x.shape);
    for (size_t i = 0; i < x.size; ++i) grad_x1.data[i] = grad_y.data[i] + grad_x1_from_norm2.data[i];

    Tensor grad_h1 = attn.backward(h1, grad_x1);
    Tensor grad_x_from_norm1 = norm1.backward(x, grad_h1);

    Tensor grad_x_final(x.shape);
    for (size_t i = 0; i < x.size; ++i)
        grad_x_final.data[i] = grad_x1.data[i] + grad_x_from_norm1.data[i];
    return grad_x_final;
}

void TransformerBlock::zero_grad() {
    norm1.zero_grad(); norm2.zero_grad(); attn.zero_grad(); ffn.zero_grad();
}

Transformer::Transformer(size_t vs, size_t d, size_t nl, size_t nh,
                         size_t msl, float std)
    : vocab_size(vs), dim(d), n_layer(nl), n_head(nh), max_seq_len(msl),
      token_emb({vs, d}), grad_token_emb({vs, d}),
      final_norm(d), lm_head(d, vs, std) {
    token_emb.random_normal(0.0f, std);
    grad_token_emb.zero();
    blocks.reserve(nl);
    for (size_t i = 0; i < nl; ++i)
        blocks.emplace_back(d, nh, std);
}

Tensor Transformer::forward(const Tensor& tokens) const {
    assert(tokens.shape.size() == 2);
    size_t B = tokens.shape[0], T = tokens.shape[1];
    assert(T <= max_seq_len);

    Tensor x({B, T, dim});
    for (size_t b = 0; b < B; ++b)
        for (size_t t = 0; t < T; ++t) {
            size_t id = (size_t)tokens.data[b * T + t];
            assert(id < vocab_size);
            std::memcpy(&x.data[(b * T + t) * dim], &token_emb.data[id * dim], sizeof(float) * dim);
        }

    for (const auto& blk : blocks) x = blk.forward(x);
    x = final_norm.forward(x);

    Tensor x2d({B * T, dim});
    std::memcpy(x2d.data.data(), x.data.data(), sizeof(float) * x.size);
    Tensor logits2d = lm_head.forward(x2d);
    Tensor logits({B, T, vocab_size});
    std::memcpy(logits.data.data(), logits2d.data.data(), sizeof(float) * logits2d.size);
    return logits;
}

Tensor Transformer::backward(const Tensor& tokens, const Tensor& grad_logits) {
    size_t B = tokens.shape[0], T = tokens.shape[1];
    Tensor x({B, T, dim});
    for (size_t b = 0; b < B; ++b)
        for (size_t t = 0; t < T; ++t) {
            size_t id = (size_t)tokens.data[b * T + t];
            std::memcpy(&x.data[(b * T + t) * dim], &token_emb.data[id * dim], sizeof(float) * dim);
        }

    std::vector<Tensor> block_inputs;
    block_inputs.push_back(x);
    for (const auto& blk : blocks) { x = blk.forward(x); block_inputs.push_back(x); }

    Tensor h_final = final_norm.forward(x);
    Tensor h_final_2d({B * T, dim}), grad_logits_2d({B * T, vocab_size});
    std::memcpy(h_final_2d.data.data(), h_final.data.data(), sizeof(float) * h_final.size);
    std::memcpy(grad_logits_2d.data.data(), grad_logits.data.data(), sizeof(float) * grad_logits.size);

    Tensor grad_h_final_2d = lm_head.backward(h_final_2d, grad_logits_2d);
    Tensor grad_h_final({B, T, dim});
    std::memcpy(grad_h_final.data.data(), grad_h_final_2d.data.data(), sizeof(float) * grad_h_final_2d.size);

    Tensor grad_x = final_norm.backward(x, grad_h_final);
    for (int i = (int)n_layer - 1; i >= 0; --i)
        grad_x = blocks[i].backward(block_inputs[i], grad_x);

    grad_token_emb.zero();
    for (size_t b = 0; b < B; ++b)
        for (size_t t = 0; t < T; ++t) {
            size_t id = (size_t)tokens.data[b * T + t];
            for (size_t d = 0; d < dim; ++d)
                grad_token_emb.data[id * dim + d] += grad_x.data[(b * T + t) * dim + d];
        }
    return grad_x;
}

void Transformer::zero_grad() {
    grad_token_emb.zero(); final_norm.zero_grad(); lm_head.zero_grad();
    for (auto& blk : blocks) blk.zero_grad();
}

void Transformer::save_weights(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) { std::cerr << "Ошибка сохранения!\n"; return; }
    auto save = [&](const Tensor& t) {
        size_t sz = t.size;
        out.write((char*)&sz, sizeof(size_t));
        out.write((char*)t.data.data(), sz * sizeof(float));
    };
    save(token_emb); save(final_norm.weight); save(lm_head.W); save(lm_head.b);
    for (const auto& blk : blocks) {
        save(blk.norm1.weight); save(blk.norm2.weight);
        save(blk.attn.Wq.W); save(blk.attn.Wq.b);
        save(blk.attn.Wk.W); save(blk.attn.Wk.b);
        save(blk.attn.Wv.W); save(blk.attn.Wv.b);
        save(blk.attn.Wo.W); save(blk.attn.Wo.b);
        save(blk.ffn.W1.W); save(blk.ffn.W1.b);
        save(blk.ffn.W2.W); save(blk.ffn.W2.b);
        save(blk.ffn.W3.W); save(blk.ffn.W3.b);
    }
    std::cout << "Веса сохранены в " << filename << "\n";
}

void Transformer::load_weights(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Файл весов не найден, начинаем с нуля.\n"; return; }
    auto load = [&](Tensor& t) {
        size_t sz;
        in.read((char*)&sz, sizeof(size_t));
        if (!in || sz != t.size) { std::cerr << "Ошибка размера тензора!\n"; exit(1); }
        in.read((char*)t.data.data(), sz * sizeof(float));
        if (!in) { std::cerr << "Ошибка чтения тензора!\n"; exit(1); }
    };
    load(token_emb); load(final_norm.weight); load(lm_head.W); load(lm_head.b);
    for (auto& blk : blocks) {
        load(blk.norm1.weight); load(blk.norm2.weight);
        load(blk.attn.Wq.W); load(blk.attn.Wq.b);
        load(blk.attn.Wk.W); load(blk.attn.Wk.b);
        load(blk.attn.Wv.W); load(blk.attn.Wv.b);
        load(blk.attn.Wo.W); load(blk.attn.Wo.b);
        load(blk.ffn.W1.W); load(blk.ffn.W1.b);
        load(blk.ffn.W2.W); load(blk.ffn.W2.b);
        load(blk.ffn.W3.W); load(blk.ffn.W3.b);
    }
    std::cout << "Веса загружены из " << filename << "\n";
}

int Transformer::sample_top_k(const Tensor& logits, int current_len, float temperature, int k) const {
    size_t V = vocab_size;
    size_t offset = (current_len - 1) * V;

    std::vector<std::pair<float, int>> probs(V);
    for (size_t v = 0; v < V; ++v) probs[v] = {logits.data[offset + v], (int)v};

    std::sort(probs.begin(), probs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    int top_k = std::min((int)V, k);
    float mx = probs[0].first;
    float sum = 0.0f;
    std::vector<float> weights(top_k);
    for (int i = 0; i < top_k; ++i) {
        weights[i] = std::exp((probs[i].first - mx) / temperature);
        sum += weights[i];
    }

    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    float acc = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        acc += weights[i];
        if (r <= acc) return probs[i].second;
    }
    return probs[0].second;
}
