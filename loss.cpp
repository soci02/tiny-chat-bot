#include "loss.h"
#include <cmath>
#include <cassert>
#include <algorithm>

float cross_entropy(const Tensor& logits, const Tensor& targets, Tensor& grad_logits) {
    assert(logits.shape.size() == 3);
    size_t B = logits.shape[0], T = logits.shape[1], V = logits.shape[2];
    assert(targets.shape.size() == 2);
    assert(targets.shape[0] == B && targets.shape[1] == T);

    grad_logits = Tensor(logits.shape);
    float total_loss = 0.0f;
    float invN = 1.0f / (float)(B * T);

    for (size_t b = 0; b < B; ++b)
        for (size_t t = 0; t < T; ++t) {
            const float* row = &logits.data[(b * T + t) * V];
            size_t target = (size_t)targets.data[b * T + t];
            assert(target < V);

            float mx = -1e30f;
            for (size_t v = 0; v < V; ++v) mx = std::max(mx, row[v]);
            float sum = 0.0f;
            for (size_t v = 0; v < V; ++v) sum += std::exp(row[v] - mx);
            float lse = mx + std::log(sum);

            total_loss += (lse - row[target]);

            float* grow = &grad_logits.data[(b * T + t) * V];
            for (size_t v = 0; v < V; ++v)
                grow[v] = std::exp(row[v] - lse) * invN;
            grow[target] -= invN;
        }

    return total_loss * invN;
}
