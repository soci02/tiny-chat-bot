#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

class BPETokenizer {
public:
    std::vector<std::string> vocab;
    std::vector<std::pair<int,int>> merges;
    std::unordered_map<std::string, int> token_to_id;
    int vocab_size = 0;

    static constexpr int PAD_ID  = 0;
    static constexpr int UNK_ID  = 1;
    static constexpr int USER_ID = 2;
    static constexpr int BOT_ID  = 3;
    static constexpr int END_ID  = 4;

    void train(const std::string& text, int target_vocab_size);
    std::vector<int> encode(const std::string& text) const;
    std::vector<int> encode_chat(const std::string& user, const std::string& bot) const;
    std::string decode(const std::vector<int>& ids, bool skip_special = true) const;

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    void rebuild_maps();
    static std::vector<int> bytes_to_ids(const std::string& s);
};
