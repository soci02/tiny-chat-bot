#include "tensor.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

Tensor::Tensor(const std::vector<size_t>& shape_) : shape(shape_) {
    compute_strides();
    size = 1;
    for (auto d : shape) size *= d;
    data.assign(size, 0.0f);
}

void Tensor::compute_strides() {
    strides.resize(shape.size());
    size_t s = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = s;
        s *= shape[i];
    }
}

float& Tensor::at(const std::vector<size_t>& idx) {
    assert(idx.size() == shape.size());
    size_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) { assert(idx[i] < shape[i]); off += idx[i] * strides[i]; }
    return data[off];
}

const float& Tensor::at(const std::vector<size_t>& idx) const {
    assert(idx.size() == shape.size());
    size_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) { assert(idx[i] < shape[i]); off += idx[i] * strides[i]; }
    return data[off];
}

void Tensor::fill(float v) { std::fill(data.begin(), data.end(), v); }
void Tensor::zero() { fill(0.0f); }

void Tensor::random_normal(float mean, float std) {
    static std::mt19937 rng(12345);
    std::normal_distribution<float> dist(mean, std);
    for (auto& x : data) x = dist(rng);
}

Tensor Tensor::add(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor out(shape);
    for (size_t i = 0; i < size; ++i) out.data[i] = data[i] + other.data[i];
    return out;
}

Tensor Tensor::mul(float scalar) const {
    Tensor out(shape);
    for (size_t i = 0; i < size; ++i) out.data[i] = data[i] * scalar;
    return out;
}

Tensor Tensor::mul(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor out(shape);
    for (size_t i = 0; i < size; ++i) out.data[i] = data[i] * other.data[i];
    return out;
}

Tensor Tensor::matmul(const Tensor& A, const Tensor& B) {
    assert(A.shape.size() == 2 && B.shape.size() == 2);
    size_t M = A.shape[0], K = A.shape[1], N = B.shape[1];
    assert(B.shape[0] == K);
    Tensor C({M, N});
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a = A.data[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                C.data[i * N + j] += a * B.data[k * N + j];
            }
        }
    }
    return C;
}

Tensor Tensor::softmax_rows(const Tensor& X) {
    assert(X.shape.size() == 2);
    size_t M = X.shape[0], N = X.shape[1];
    Tensor out(X.shape);
    for (size_t i = 0; i < M; ++i) {
        float mx = -1e30f;
        for (size_t j = 0; j < N; ++j) mx = std::max(mx, X.data[i * N + j]);
        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            float e = std::exp(X.data[i * N + j] - mx);
            out.data[i * N + j] = e;
            sum += e;
        }
        for (size_t j = 0; j < N; ++j) out.data[i * N + j] /= sum;
    }
    return out;
}

void Tensor::print_shape() const {
    std::cout << "Tensor[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

void Tensor::print_some(size_t n) const {
    print_shape();
    for (size_t i = 0; i < std::min(n, size); ++i)
        std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
    std::cout << "\n";
}
