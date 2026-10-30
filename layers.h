#pragma once
#include "tensor.h"

class Linear {
public:
    Tensor W, b, grad_W, grad_b;
    size_t in_features = 0, out_features = 0;

    Linear() = default;
    Linear(size_t in_f, size_t out_f, float std = 0.02f);

    Tensor forward(const Tensor& x) const;
    Tensor backward(const Tensor& x, const Tensor& grad_y);
    void zero_grad();
};

Tensor gelu(const Tensor& x);
float gelu_scalar(float v);
Tensor gelu_backward(const Tensor& x, const Tensor& grad_y);

class RMSNorm {
public:
    Tensor weight, grad_weight;
    float eps = 1e-6f;

    RMSNorm() = default;
    explicit RMSNorm(size_t dim);

    Tensor forward(const Tensor& x) const;
    Tensor backward(const Tensor& x, const Tensor& grad_y);
    void zero_grad();
};
