#include "attention.h"
#include <cmath>
#include <cassert>
#include <cstring>
#include <vector>

MultiHeadAttention::MultiHeadAttention(size_t d, size_t h, float std)
    : n_head(h), head_dim(d / h), dim(d),
      Wq(d, d, std), Wk(d, d, std), Wv(d, d, std), Wo(d, d, std) {
    assert(d % h == 0);
}

static std::vector<float> rope_cache(size_t head_dim, size_t max_pos) {
    std::vector<float> cache(head_dim * max_pos * 2);  // cos, sin
    const float base = 10000.0f;
    for (size_t p = 0; p < max_pos; ++p) {
        for (size_t i = 0; i < head_dim / 2; ++i) {
            float theta = std::pow(base, -2.0f * (float)i / (float)head_dim);
            float angle = (float)p * theta;
            cache[(p * head_dim / 2 + i) * 2 + 0] = std::cos(angle);
            cache[(p * head_dim / 2 + i) * 2 + 1] = std::sin(angle);
        }
    }
    return cache;
}

void apply_rope(Tensor& x, size_t n_head, size_t start_pos) {
    assert(x.shape.size() == 3);
    size_t B = x.shape[0], T = x.shape[1], D = x.shape[2];
    size_t hd = D / n_head;
    assert(hd % 2 == 0);
    static std::vector<float> cache = rope_cache(hd, 4096);

    for (size_t b = 0; b < B; ++b)
        for (size_t t = 0; t < T; ++t) {
            size_t pos = start_pos + t;
            if (pos >= 4096) continue;
            for (size_t h = 0; h < n_head; ++h) {
                float* vec = &x.data[((b * T + t) * D) + h * hd];
                for (size_t i = 0; i < hd; i += 2) {
                    float c = cache[(pos * hd / 2 + i / 2) * 2 + 0];
                    float s = cache[(pos * hd / 2 + i / 2) * 2 + 1];
                    float v0 = vec[i], v1 = vec[i + 1];
                    vec[i]     = v0 * c - v1 * s;
                    vec[i + 1] = v0 * s + v1 * c;
                }
            }
        }
}

void apply_rope_backward(Tensor& grad_x, size_t n_head, size_t start_pos) {
    size_t B = grad_x.shape[0], T = grad_x.shape[1], D = grad_x.shape[2];
    size_t hd = D / n_head;
    static std::vector<float> cache = rope_cache(hd, 4096);
    for (size_t b = 0; b < B; ++b)
        for (size_t t = 0; t < T; ++t) {
            size_t pos = start_pos + t;
            if (pos >= 4096) continue;
            for (size_t h = 0; h < n_head; ++h) {
                float* vec = &grad_x.data[((b * T + t) * D) + h * hd];
                for (size_t i = 0; i < hd; i += 2) {
                    float c = cache[(pos * hd / 2 + i / 2) * 2 + 0];
                    float s = cache[(pos * hd / 2 + i / 2) * 2 + 1];
                    float v0 = vec[i], v1 = vec[i + 1];
                    vec[i]     =  v0 * c + v1 * s;
                    vec[i + 1] = -v0 * s + v1 * c;
                }
            }
        }
}

Tensor MultiHeadAttention::forward(const Tensor& x) const {
    assert(x.shape.size() == 3);
    size_t B = x.shape[0], T = x.shape[1], D = x.shape[2];
    assert(D == dim);

    Tensor x2d({B * T, D});
    std::memcpy(x2d.data.data(), x.data.data(), sizeof(float) * x.size);

    Tensor Q2 = Wq.forward(x2d), K2 = Wk.forward(x2d), V2 = Wv.forward(x2d);
    Tensor Q({B, T, D}), K({B, T, D}), V({B, T, D});
    std::memcpy(Q.data.data(), Q2.data.data(), sizeof(float) * Q2.size);
    std::memcpy(K.data.data(), K2.data.data(), sizeof(float) * K2.size);
    std::memcpy(V.data.data(), V2.data.data(), sizeof(float) * V2.size);

    apply_rope(Q, n_head);
    apply_rope(K, n_head);

    size_t H = n_head, hd = head_dim;
    Tensor Qh({B * H, T, hd}), Kh({B * H, T, hd}), Vh({B * H, T, hd});
    for (size_t b = 0; b < B; ++b)
        for (size_t t = 0; t < T; ++t)
            for (size_t h = 0; h < H; ++h) {
                size_t dst = ((b * H + h) * T + t) * hd;
                std::memcpy(&Qh.data[dst], &Q.data[(b * T + t) * D + h * hd], sizeof(float) * hd);
                std::memcpy(&Kh.data[dst], &K.data[(b * T + t) * D + h * hd], sizeof(float) * hd);
                std::memcpy(&Vh.data[dst], &V.data[(b * T + t) * D + h * hd], sizeof(float) * hd);
            }

    Tensor out({B, T, D});
    float scale = 1.0f / std::sqrt((float)hd);

    for (size_t bh = 0; bh < B * H; ++bh) {
        Tensor scores({T, T});
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < T; ++j) {
                float s = 0.0f;
                const float* qi = &Qh.data[(bh * T + i) * hd];
                const float* kj = &Kh.data[(bh * T + j) * hd];
                for (size_t d = 0; d < hd; ++d) s += qi[d] * kj[d];
                s *= scale;
                if (j > i) s = -1e30f;
                scores.data[i * T + j] = s;
            }
        Tensor probs = Tensor::softmax_rows(scores);
        for (size_t i = 0; i < T; ++i)
            for (size_t d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (size_t j = 0; j < T; ++j)
                    acc += probs.data[i * T + j] * Vh.data[(bh * T + j) * hd + d];
                size_t b = bh / H, h = bh % H;
                out.data[((b * T + i) * D) + h * hd + d] = acc;
            }
    }

    Tensor out2d({B * T, D});
    std::memcpy(out2d.data.data(), out.data.data(), sizeof(float) * out.size);
    Tensor y2d = Wo.forward(out2d);
    Tensor y({B, T, D});
    std::memcpy(y.data.data(), y2d.data.data(), sizeof(float) * y2d.size);
    return y;
}

Tensor MultiHeadAttention::backward(const Tensor& x, const Tensor& grad_y) {
    size_t B = x.shape[0], T = x.shape[1], D = x.shape[2];
    size_t H = n_head, hd = head_dim;

    Tensor x2d({B * T, D});
    std::memcpy(x2d.data.data(), x.data.data(), sizeof(float) * x.size);

    Tensor Q2 = Wq.forward(x2d), K2 = Wk.forward(x2d), V2 = Wv.forward(x2d);
    Tensor Q({B, T, D}), K({B, T, D}), V({B, T, D});
    std::memcpy(Q.data.data(), Q2.data.data(), sizeof(float) * Q2.size);
    std::memcpy(K.data.data(), K2.data.data(), sizeof(float) * K2.size);
    std::memcpy(V.data.data(), V2.data.data(), sizeof(float) * V2.size);
    apply_rope(Q, H); apply_rope(K, H);

    Tensor Qh({B * H, T, hd}), Kh({B * H, T, hd}), Vh({B * H, T, hd});
    for (size_t b = 0; b < B; ++b)
        for (size_t t = 0; t < T; ++t)
            for (size_t h = 0; h < H; ++h) {
                size_t dst = ((b * H + h) * T + t) * hd;
                std::memcpy(&Qh.data[dst], &Q.data[(b * T + t) * D + h * hd], sizeof(float) * hd);
                std::memcpy(&Kh.data[dst], &K.data[(b * T + t) * D + h * hd], sizeof(float) * hd);
                std::memcpy(&Vh.data[dst], &V.data[(b * T + t) * D + h * hd], sizeof(float) * hd);
            }

    Tensor out_attn({B, T, D});
    float scale = 1.0f / std::sqrt((float)hd);
    Tensor probs_all({B * H, T, T});

    for (size_t bh = 0; bh < B * H; ++bh) {
        Tensor scores({T, T});
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < T; ++j) {
                float s = 0.0f;
                for (size_t d = 0; d < hd; ++d)
                    s += Qh.data[(bh * T + i) * hd + d] * Kh.data[(bh * T + j) * hd + d];
                s *= scale;
                if (j > i) s = -1e30f;
                scores.data[i * T + j] = s;
            }
        Tensor probs = Tensor::softmax_rows(scores);
        std::memcpy(&probs_all.data[bh * T * T], probs.data.data(), sizeof(float) * T * T);
        for (size_t i = 0; i < T; ++i)
            for (size_t d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (size_t j = 0; j < T; ++j)
                    acc += probs.data[i * T + j] * Vh.data[(bh * T + j) * hd + d];
                size_t b = bh / H, h = bh % H;
                out_attn.data[((b * T + i) * D) + h * hd + d] = acc;
            }
    }

    Tensor out2d({B * T, D}), grad_out2d({B * T, D});
    std::memcpy(out2d.data.data(), out_attn.data.data(), sizeof(float) * out_attn.size);
    std::memcpy(grad_out2d.data.data(), grad_y.data.data(), sizeof(float) * grad_y.size);

    Tensor grad_attn2d = Wo.backward(out2d, grad_out2d);
    Tensor grad_attn({B, T, D});
    std::memcpy(grad_attn.data.data(), grad_attn2d.data.data(), sizeof(float) * grad_attn2d.size);

    Tensor grad_Qh({B * H, T, hd}), grad_Kh({B * H, T, hd}), grad_Vh({B * H, T, hd});
    Tensor grad_Q({B, T, D}), grad_K({B, T, D}), grad_V({B, T, D});

    for (size_t bh = 0; bh < B * H; ++bh) {
        Tensor grad_probs({T, T}), grad_scores({T, T});
        size_t b = bh / H, h = bh % H;

        for (size_t i = 0; i < T; ++i)
            for (size_t d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (size_t j = 0; j < T; ++j)
                    acc += probs_all.data[bh * T * T + i * T + j] * grad_attn.data[((b * T + j) * D) + h * hd + d];
                grad_Vh.data[(bh * T + i) * hd + d] = acc;
            }

        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < T; ++j) {
                float acc = 0.0f;
                for (size_t d = 0; d < hd; ++d)
                    acc += grad_attn.data[((b * T + i) * D) + h * hd + d] * Vh.data[(bh * T + j) * hd + d];
                grad_probs.data[i * T + j] = acc;
            }

        for (size_t i = 0; i < T; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < T; ++j)
                sum += grad_probs.data[i * T + j] * probs_all.data[bh * T * T + i * T + j];
            for (size_t j = 0; j < T; ++j) {
                float p = probs_all.data[bh * T * T + i * T + j];
                grad_scores.data[i * T + j] = p * (grad_probs.data[i * T + j] - sum);
            }
        }

        for (size_t i = 0; i < T; ++i)
            for (size_t d = 0; d < hd; ++d) {
                float accQ = 0.0f, accK = 0.0f;
                for (size_t j = 0; j < T; ++j) {
                    accQ += grad_scores.data[i * T + j] * Kh.data[(bh * T + j) * hd + d];
                    accK += grad_scores.data[j * T + i] * Qh.data[(bh * T + j) * hd + d];
                }
                grad_Qh.data[(bh * T + i) * hd + d] = accQ * scale;
                grad_Kh.data[(bh * T + i) * hd + d] = accK * scale;
            }
    }

    for (size_t b = 0; b < B; ++b)
        for (size_t t = 0; t < T; ++t)
            for (size_t h = 0; h < H; ++h) {
                size_t dst = ((b * H + h) * T + t) * hd;
                std::memcpy(&grad_Q.data[(b * T + t) * D + h * hd], &grad_Qh.data[dst], sizeof(float) * hd);
                std::memcpy(&grad_K.data[(b * T + t) * D + h * hd], &grad_Kh.data[dst], sizeof(float) * hd);
                std::memcpy(&grad_V.data[(b * T + t) * D + h * hd], &grad_Vh.data[dst], sizeof(float) * hd);
            }

    apply_rope_backward(grad_Q, H);
    apply_rope_backward(grad_K, H);

    Tensor grad_Q2d({B * T, D}), grad_K2d({B * T, D}), grad_V2d({B * T, D});
    std::memcpy(grad_Q2d.data.data(), grad_Q.data.data(), sizeof(float) * grad_Q.size);
    std::memcpy(grad_K2d.data.data(), grad_K.data.data(), sizeof(float) * grad_K.size);
    std::memcpy(grad_V2d.data.data(), grad_V.data.data(), sizeof(float) * grad_V.size);

    Tensor grad_x_q = Wq.backward(x2d, grad_Q2d);
    Tensor grad_x_k = Wk.backward(x2d, grad_K2d);
    Tensor grad_x_v = Wv.backward(x2d, grad_V2d);

    Tensor grad_x({B, T, D});
    for (size_t i = 0; i < grad_x.size; ++i)
        grad_x.data[i] = grad_x_q.data[i] + grad_x_k.data[i] + grad_x_v.data[i];
    return grad_x;
}

void MultiHeadAttention::zero_grad() {
    Wq.zero_grad(); Wk.zero_grad(); Wv.zero_grad(); Wo.zero_grad();
}
