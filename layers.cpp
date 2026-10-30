#include "layers.h"
#include <cmath>
#include <cassert>
#include <algorithm>

Linear::Linear(size_t in_f, size_t out_f, float std)
    : W({in_f, out_f}), b({out_f}),
      grad_W({in_f, out_f}), grad_b({out_f}),
      in_features(in_f), out_features(out_f) {
    W.random_normal(0.0f, std);
    b.zero();
    grad_W.zero();
    grad_b.zero();
}

Tensor Linear::forward(const Tensor& x) const {
    assert(x.shape.size() == 2);
    assert(x.shape[1] == in_features);
    Tensor y = Tensor::matmul(x, W);
    size_t B = y.shape[0], O = out_features;
    for (size_t i = 0; i < B; ++i)
        for (size_t j = 0; j < O; ++j)
            y.data[i * O + j] += b.data[j];
    return y;
}

Tensor Linear::backward(const Tensor& x, const Tensor& grad_y) {
    assert(x.shape.size() == 2);
    assert(grad_y.shape.size() == 2);
    size_t B = x.shape[0], IN = in_features, OUT = out_features;

    for (size_t i = 0; i < IN; ++i) {
        for (size_t o = 0; o < OUT; ++o) {
            float acc = 0.0f;
            for (size_t b = 0; b < B; ++b)
                acc += x.data[b * IN + i] * grad_y.data[b * OUT + o];
            grad_W.data[i * OUT + o] += acc;
        }
    }
    for (size_t b = 0; b < B; ++b)
        for (size_t o = 0; o < OUT; ++o)
            grad_b.data[o] += grad_y.data[b * OUT + o];

    Tensor grad_x({B, IN});
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < IN; ++i) {
            float acc = 0.0f;
            for (size_t o = 0; o < OUT; ++o)
                acc += grad_y.data[b * OUT + o] * W.data[i * OUT + o];
            grad_x.data[b * IN + i] = acc;
        }
    }
    return grad_x;
}

void Linear::zero_grad() { grad_W.zero(); grad_b.zero(); }

Tensor gelu(const Tensor& x) {
    Tensor out(x.shape);
    const float c = std::sqrt(2.0f / 3.14159265358979f);
    for (size_t i = 0; i < x.size; ++i) {
        float v = x.data[i];
        float inner = c * (v + 0.044715f * v * v * v);
        out.data[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
    return out;
}

float gelu_scalar(float v) {
    const float c = std::sqrt(2.0f / 3.14159265358979f);
    float inner = c * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + std::tanh(inner));
}

Tensor gelu_backward(const Tensor& x, const Tensor& grad_y) {
    assert(x.shape == grad_y.shape);
    Tensor grad_x(x.shape);
    const float c = std::sqrt(2.0f / 3.14159265358979f);
    for (size_t i = 0; i < x.size; ++i) {
        float v = x.data[i];
        float v3 = v * v * v;
        float inner = c * (v + 0.044715f * v3);
        float t = std::tanh(inner);
        float dtanh = (1.0f - t * t) * c * (1.0f + 3.0f * 0.044715f * v * v);
        float dgelu = 0.5f * (1.0f + t) + 0.5f * v * dtanh;
        grad_x.data[i] = grad_y.data[i] * dgelu;
    }
    return grad_x;
}

RMSNorm::RMSNorm(size_t dim) : weight({dim}), grad_weight({dim}) {
    weight.fill(1.0f);
    grad_weight.zero();
}

Tensor RMSNorm::forward(const Tensor& x) const {
    assert(!x.shape.empty());
    size_t D = x.shape.back();
    assert(D == weight.shape[0]);
    Tensor out(x.shape);
    size_t rows = x.size / D;
    for (size_t r = 0; r < rows; ++r) {
        const float* xrow = &x.data[r * D];
        float* orow = &out.data[r * D];
        float sum_sq = 0.0f;
        for (size_t j = 0; j < D; ++j) sum_sq += xrow[j] * xrow[j];
        float rms = std::sqrt(sum_sq / D + eps);
        for (size_t j = 0; j < D; ++j)
            orow[j] = (xrow[j] / rms) * weight.data[j];
    }
    return out;
}

Tensor RMSNorm::backward(const Tensor& x, const Tensor& grad_y) {
    assert(x.shape == grad_y.shape);
    size_t D = x.shape.back();
    size_t rows = x.size / D;
    Tensor grad_x(x.shape);
    for (size_t r = 0; r < rows; ++r) {
        const float* xr = &x.data[r * D];
        const float* gyr = &grad_y.data[r * D];
        float* gxr = &grad_x.data[r * D];
        float sum_sq = 0.0f;
        for (size_t j = 0; j < D; ++j) sum_sq += xr[j] * xr[j];
        float rms = std::sqrt(sum_sq / D + eps);
        float inv_rms = 1.0f / rms;
        for (size_t j = 0; j < D; ++j)
            grad_weight.data[j] += gyr[j] * (xr[j] * inv_rms);
        float dot = 0.0f;
        for (size_t j = 0; j < D; ++j) dot += weight.data[j] * gyr[j] * xr[j];
        float coef = dot / ((float)D * rms * rms * rms);
        for (size_t i = 0; i < D; ++i) {
            float term1 = weight.data[i] * gyr[i] * inv_rms;
            float term2 = xr[i] * coef;
            gxr[i] = term1 - term2;
        }
    }
    return grad_x;
}

void RMSNorm::zero_grad() { grad_weight.zero(); }
