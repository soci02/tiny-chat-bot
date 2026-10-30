#include "tokenizer.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <iostream>

std::vector<int> BPETokenizer::bytes_to_ids(const std::string& s) {
    std::vector<int> ids(s.size());
    for (size_t i = 0; i < s.size(); ++i) ids[i] = (unsigned char)s[i];
    return ids;
}

void BPETokenizer::train(const std::string& text, int target_vocab_size) {
    vocab.clear();
    for (int i = 0; i < 256; ++i) vocab.push_back(std::string(1, (char)i));
    vocab.push_back("<pad>");
    vocab.push_back("<unk>");
    vocab.push_back("<user>");
    vocab.push_back("<bot>");
    vocab.push_back("<end>");

    std::vector<int> data = bytes_to_ids(text);

    // ===== ќ“Ћјƒ ј =====
    std::cout << "[BPE] text.size() = " << text.size() << " байт\n";
    std::cout << "[BPE] data.size() = " << data.size() << "\n";
    std::cout << "[BPE] первые 200 символов текста:\n["
              << text.substr(0, 200) << "]\n";
    // ===================

    int n_merges = target_vocab_size - (int)vocab.size();
    std::cout << "[BPE] target_vocab = " << target_vocab_size
              << ", n_merges = " << n_merges << "\n";

    if (n_merges <= 0) { rebuild_maps(); return; }

    merges.clear();
    for (int m = 0; m < n_merges; ++m) {
        std::unordered_map<uint64_t, int> counts;
        counts.reserve(data.size());
        for (size_t i = 0; i + 1 < data.size(); ++i) {
            uint64_t key = ((uint64_t)data[i] << 32) | (uint32_t)data[i+1];
            counts[key]++;
        }
        if (counts.empty()) {
            std::cout << "[BPE] counts пустой на merge " << m << "\n";
            break;
        }

        uint64_t best_key = 0; int best_cnt = 0;
        for (auto& kv : counts)
            if (kv.second > best_cnt) { best_cnt = kv.second; best_key = kv.first; }

        if (best_cnt < 2) {
            std::cout << "[BPE] best_cnt < 2 на merge " << m
                      << " (best_cnt=" << best_cnt << ") Ч стоп\n";
            break;
        }

        int left = (int)(best_key >> 32);
        int right = (int)(best_key & 0xFFFFFFFF);
        int new_id = (int)vocab.size();
        vocab.push_back(vocab[left] + vocab[right]);
        merges.push_back({left, right});

        std::vector<int> new_data;
        new_data.reserve(data.size());
        for (size_t i = 0; i < data.size(); ) {
            if (i + 1 < data.size() && data[i] == left && data[i+1] == right) {
                new_data.push_back(new_id);
                i += 2;
            } else {
                new_data.push_back(data[i]);
                i += 1;
            }
        }
        data.swap(new_data);
    }

    std::cout << "[BPE] merges сделано: " << merges.size() << "\n";
    std::cout << "[BPE] итоговый vocab = " << vocab.size() << "\n";

    rebuild_maps();
}

void BPETokenizer::rebuild_maps() {
    token_to_id.clear();
    for (int i = 0; i < (int)vocab.size(); ++i) token_to_id[vocab[i]] = i;
    vocab_size = (int)vocab.size();
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> ids = bytes_to_ids(text);
    for (size_t mi = 0; mi < merges.size(); ++mi) {
        int left = merges[mi].first, right = merges[mi].second;
        int new_id = 256 + 5 + (int)mi;
        std::vector<int> out;
        out.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ) {
            if (i + 1 < ids.size() && ids[i] == left && ids[i+1] == right) {
                out.push_back(new_id); i += 2;
            } else { out.push_back(ids[i]); i += 1; }
        }
        ids.swap(out);
    }
    return ids;
}

std::vector<int> BPETokenizer::encode_chat(const std::string& user, const std::string& bot) const {
    std::vector<int> out;
    out.push_back(USER_ID);
    auto u = encode(user);
    out.insert(out.end(), u.begin(), u.end());
    out.push_back(BOT_ID);
    auto b = encode(bot);
    out.insert(out.end(), b.begin(), b.end());
    out.push_back(END_ID);
    return out;
}

std::string BPETokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
    std::string out;
    for (int id : ids) {
        if (skip_special && id >= 256 && id <= 260) continue;
        if (id < 0 || id >= (int)vocab.size()) continue;
        out += vocab[id];
    }
    return out;
}

void BPETokenizer::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot save tokenizer");
    int nv = (int)vocab.size(), nm = (int)merges.size();
    out.write((char*)&nv, 4); out.write((char*)&nm, 4);
    for (auto& s : vocab) {
        int len = (int)s.size();
        out.write((char*)&len, 4); out.write(s.data(), len);
    }
    for (auto& m : merges) {
        out.write((char*)&m.first, 4);
        out.write((char*)&m.second, 4);
    }
}

void BPETokenizer::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot load tokenizer");
    int nv, nm;
    in.read((char*)&nv, 4); in.read((char*)&nm, 4);
    vocab.resize(nv);
    for (int i = 0; i < nv; ++i) {
        int len; in.read((char*)&len, 4);
        vocab[i].resize(len); in.read(&vocab[i][0], len);
    }
    merges.resize(nm);
    for (int i = 0; i < nm; ++i) {
        in.read((char*)&merges[i].first, 4);
        in.read((char*)&merges[i].second, 4);
    }
    rebuild_maps();
}
