#pragma once
#include "tensor.h"

float cross_entropy(const Tensor& logits, const Tensor& targets, Tensor& grad_logits);
