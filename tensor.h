#pragma once
#include <vector>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

class Tensor {
public:
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::vector<float> data;
    size_t size = 0;

    Tensor() = default;
    explicit Tensor(const std::vector<size_t>& shape_);

    float& at(const std::vector<size_t>& idx);
    const float& at(const std::vector<size_t>& idx) const;
    float& at2(size_t i, size_t j) { return data[i * strides[0] + j]; }
    const float& at2(size_t i, size_t j) const { return data[i * strides[0] + j]; }

    void fill(float v);
    void random_normal(float mean = 0.0f, float std = 0.02f);
    void zero();
    size_t numel() const { return size; }

    Tensor add(const Tensor& other) const;
    Tensor mul(float scalar) const;
    Tensor mul(const Tensor& other) const;
    Tensor operator+(const Tensor& o) const { return add(o); }
    Tensor operator*(float s) const { return mul(s); }

    static Tensor matmul(const Tensor& A, const Tensor& B);
    static Tensor softmax_rows(const Tensor& X);

    void print_shape() const;
    void print_some(size_t n = 8) const;

private:
    void compute_strides();
};
