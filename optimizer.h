#pragma once
#include "tensor.h"
#include <cmath>
#include <unordered_map>

class AdamW {
public:
    float lr, beta1, beta2, eps, weight_decay;
    float warmup_steps = 0.0f;
    int step = 0;

    std::unordered_map<Tensor*, Tensor> m_cache;
    std::unordered_map<Tensor*, Tensor> v_cache;

    AdamW(float lr = 1e-3f, float b1 = 0.9f, float b2 = 0.999f,
          float eps = 1e-8f, float wd = 0.01f, float warmup = 0.0f)
        : lr(lr), beta1(b1), beta2(b2), eps(eps), weight_decay(wd),
          warmup_steps(warmup), step(0) {}

    void step_optimizer(Tensor& param, Tensor& grad) {
        if (m_cache.find(&param) == m_cache.end()) {
            m_cache[&param] = Tensor(param.shape);
            v_cache[&param] = Tensor(param.shape);
        }
        Tensor& m = m_cache[&param];
        Tensor& v = v_cache[&param];

        step++;
        float cur_lr = lr;
        if (warmup_steps > 0 && step < warmup_steps)
            cur_lr = lr * (float)step / warmup_steps;

        float b1_corr = 1.0f - std::pow(beta1, (float)step);
        float b2_corr = 1.0f - std::pow(beta2, (float)step);

        for (size_t i = 0; i < param.size; ++i) {
            float g = grad.data[i];
            m.data[i] = beta1 * m.data[i] + (1.0f - beta1) * g;
            v.data[i] = beta2 * v.data[i] + (1.0f - beta2) * g * g;
            float m_hat = m.data[i] / b1_corr;
            float v_hat = v.data[i] / b2_corr;
            // Настоящий AdamW: decay отдельно
            param.data[i] -= cur_lr * (m_hat / (std::sqrt(v_hat) + eps)
                                       + weight_decay * param.data[i]);
        }
    }
};
