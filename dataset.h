#pragma once
#include <vector>
#include <random>
#include <cstddef>
#include <stdexcept>
#include <algorithm>

class Dataset {
public:
    std::vector<int> data;
    std::vector<size_t> starts;
    size_t block_size, batch_size;
    bool dialogue_mode = false;

    Dataset(const std::vector<int>& tokens, size_t block, size_t batch,
            const std::vector<size_t>& dialogue_starts = {})
        : data(tokens), block_size(block), batch_size(batch) {
        if (data.size() <= block_size + 1)
            throw std::runtime_error("Датасет слишком мал для этого block_size!");
        starts = dialogue_starts;
        dialogue_mode = !starts.empty();
    }

    void get_batch(std::vector<int>& inputs, std::vector<int>& targets, std::mt19937& rng) {
        inputs.resize(batch_size * block_size);
        targets.resize(batch_size * block_size);

        for (size_t b = 0; b < batch_size; ++b) {
            size_t start;
            if (dialogue_mode) {
                std::uniform_int_distribution<size_t> d(0, starts.size() - 1);
                size_t di = d(rng);
                size_t s = starts[di];
                size_t e = (di + 1 < starts.size()) ? starts[di + 1] : data.size();
                size_t max_start = (e > s + block_size + 1) ? (e - block_size - 1) : s;
                std::uniform_int_distribution<size_t> dd(s, max_start);
                start = dd(rng);
            } else {
                std::uniform_int_distribution<size_t> d(0, data.size() - block_size - 2);
                start = d(rng);
            }
            for (size_t t = 0; t < block_size; ++t) {
                size_t idx = start + t;
                if (idx + 1 >= data.size()) idx = data.size() - block_size - 1;
                inputs[b * block_size + t] = data[idx];
                targets[b * block_size + t] = data[idx + 1];
            }
        }
    }
};
