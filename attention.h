#pragma once
#include "tensor.h"
#include "layers.h"

class MultiHeadAttention {
public:
    size_t n_head, head_dim, dim;
    Linear Wq, Wk, Wv, Wo;

    MultiHeadAttention() = default;
    MultiHeadAttention(size_t dim, size_t n_head, float std = 0.02f);

    Tensor forward(const Tensor& x) const;
    Tensor backward(const Tensor& x, const Tensor& grad_y);
    void zero_grad();
};

void apply_rope(Tensor& x, size_t n_head, size_t start_pos = 0);
void apply_rope_backward(Tensor& grad_x, size_t n_head, size_t start_pos = 0);
