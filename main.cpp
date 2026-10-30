#include "tensor.h"
#include "layers.h"
#include "attention.h"
#include "model.h"
#include "loss.h"
#include "optimizer.h"
#include "tokenizer.h"
#include "dataset.h"
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

// ================== ГИПЕРПАРАМЕТРЫ (единое место) ==================
static const size_t BLOCK_SIZE   = 32;
static const size_t BATCH_SIZE   = 2;
static const size_t DIM          = 192;
static const size_t N_LAYER      = 5;
static const size_t N_HEAD       = 6;
static const int    VOCAB_TARGET = 2000;
static const size_t BPE_CHARS    = 500'000;
static const size_t MAX_CORPUS   = 10'000'000;  // весь corpus.txt (126 МБ)
static const int    ITERS        = 1'000;
// ===================================================================

// ---------- Обновление весов ----------
void update_model(Transformer& model, AdamW& opt) {
    opt.step_optimizer(model.token_emb, model.grad_token_emb);
    opt.step_optimizer(model.final_norm.weight, model.final_norm.grad_weight);
    opt.step_optimizer(model.lm_head.W, model.lm_head.grad_W);
    opt.step_optimizer(model.lm_head.b, model.lm_head.grad_b);
    for (auto& blk : model.blocks) {
        opt.step_optimizer(blk.norm1.weight, blk.norm1.grad_weight);
        opt.step_optimizer(blk.norm2.weight, blk.norm2.grad_weight);
        opt.step_optimizer(blk.attn.Wq.W, blk.attn.Wq.grad_W);
        opt.step_optimizer(blk.attn.Wq.b, blk.attn.Wq.grad_b);
        opt.step_optimizer(blk.attn.Wk.W, blk.attn.Wk.grad_W);
        opt.step_optimizer(blk.attn.Wk.b, blk.attn.Wk.grad_b);
        opt.step_optimizer(blk.attn.Wv.W, blk.attn.Wv.grad_W);
        opt.step_optimizer(blk.attn.Wv.b, blk.attn.Wv.grad_b);
        opt.step_optimizer(blk.attn.Wo.W, blk.attn.Wo.grad_W);
        opt.step_optimizer(blk.attn.Wo.b, blk.attn.Wo.grad_b);
        opt.step_optimizer(blk.ffn.W1.W, blk.ffn.W1.grad_W);
        opt.step_optimizer(blk.ffn.W1.b, blk.ffn.W1.grad_b);
        opt.step_optimizer(blk.ffn.W2.W, blk.ffn.W2.grad_W);
        opt.step_optimizer(blk.ffn.W2.b, blk.ffn.W2.grad_b);
        opt.step_optimizer(blk.ffn.W3.W, blk.ffn.W3.grad_W);
        opt.step_optimizer(blk.ffn.W3.b, blk.ffn.W3.grad_b);
    }
}

// ---------- Прогресс-бар ----------
void print_progress(int current, int total, float loss,
                    std::chrono::steady_clock::time_point start_time) {
    float percent = (float)current / total;
    int percent_int = (int)(percent * 100);
    int bar_width = 30;
    int filled = (int)(percent * bar_width);

    std::string bar = "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) bar += "=";
        else if (i == filled) bar += ">";
        else bar += ".";
    }
    bar += "]";

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

    std::string eta_str = "--:--";
    if (current > 0) {
        double ms_per_iter = (double)elapsed / current;
        int iters_left = total - current;
        long long ms_left = (long long)(ms_per_iter * iters_left);
        int mins = ms_left / 60000;
        int secs = (ms_left % 60000) / 1000;
        std::ostringstream oss;
        if (mins > 0) oss << mins << "м " << secs << "с";
        else oss << secs << "с";
        eta_str = oss.str();
    }

    std::cout << "\r" << bar << " "
              << std::setw(3) << percent_int << "% | "
              << "iter " << current << "/" << total << " | "
              << "loss: " << std::fixed << std::setprecision(4) << loss << " | "
              << "осталось: " << eta_str << "    " << std::flush;
}

void print_start_msg(const std::string& name, int total_iters) {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ " << std::left << std::setw(58) << name << " ║\n";
    std::cout << "║ Всего итераций: " << std::left << std::setw(42) << total_iters << " ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
}

// ---------- Шаг 0: обучить BPE на фрагменте из разных мест файла ----------
void train_tokenizer_only(const std::string& corpus_path, const std::string& out_path,
                          int vocab_target, size_t max_chars) {
    std::ifstream f(corpus_path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "нет файла " << corpus_path << "\n"; return; }

    f.seekg(0, std::ios::end);
    size_t file_size = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);

    std::cout << "[tokenizer] размер файла: " << file_size << " байт\n";

    std::string text;
    text.reserve(max_chars);
    const int NCHUNKS = 10;
    size_t chunk_size = max_chars / NCHUNKS;
    for (int i = 0; i < NCHUNKS && text.size() < max_chars; ++i) {
        size_t offset = (file_size / NCHUNKS) * i;
        if (offset >= file_size) break;
        f.seekg(offset, std::ios::beg);
        std::string chunk(chunk_size, '\0');
        f.read(&chunk[0], chunk_size);
        chunk.resize((size_t)f.gcount());
        text += chunk;
    }
    f.close();

    // ===== ОТЛАДКА =====
    std::cout << "[tokenizer] обучаем BPE на " << text.size() << " символах...\n";
    std::cout << "[tokenizer] первые 200 символов:\n["
              << text.substr(0, 200) << "]\n";
    // ===================

    BPETokenizer tok;
    tok.train(text, vocab_target);
    tok.save(out_path);
    std::cout << "[tokenizer] готово, vocab=" << tok.vocab_size << "\n";
}

// ---------- Шаг 1: Pretrain (потоковое чтение корпуса) ----------
void pretrain(const std::string& corpus_path, const std::string& tok_path,
              const std::string& out_prefix, int iterations) {
    BPETokenizer tok;
    try { tok.load(tok_path); } catch(...) { std::cerr << "Сначала запусти Шаг 0!\n"; return; }

    std::ifstream f(corpus_path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "нет " << corpus_path << "\n"; return; }

    std::vector<int> tokens;
    tokens.reserve(50'000'000);
    std::string chunk;
    chunk.reserve(1 << 20);
    size_t total = 0;
    std::cout << "[pretrain] токенизируем корпус (блоками по 1 МБ)...\n";
    while (f && total < MAX_CORPUS) {
        size_t to_read = std::min<size_t>(1 << 20, MAX_CORPUS - total);
        chunk.resize(to_read);
        f.read(&chunk[0], to_read);
        size_t got = (size_t)f.gcount();
        if (got == 0) break;
        chunk.resize(got);
        total += got;
        auto ids = tok.encode(chunk);
        tokens.insert(tokens.end(), ids.begin(), ids.end());
        std::cout << "  " << total / 1'000'000 << " МБ, токенов: " << tokens.size() << "\n";
    }
    f.close();

    std::cout << "[pretrain] символов: " << total
              << ", токенов: " << tokens.size() << "\n";

    Dataset ds(tokens, BLOCK_SIZE, BATCH_SIZE);
    Transformer model(tok.vocab_size, DIM, N_LAYER, N_HEAD, BLOCK_SIZE);
    model.load_weights(out_prefix + "_brain.bin");

    AdamW opt(1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 100.0f);
    std::mt19937 rng(1337);
    std::vector<int> inputs, targets;

    print_start_msg("PRETRAIN (Обучение на тексте)", iterations);
    auto start_time = std::chrono::steady_clock::now();
    float current_loss = 0.0f;
    std::ofstream log("pretrain_log.txt");

    for (int it = 0; it < iterations; ++it) {
        ds.get_batch(inputs, targets, rng);
        Tensor x({BATCH_SIZE, BLOCK_SIZE}), y({BATCH_SIZE, BLOCK_SIZE});
        for (size_t i = 0; i < x.size; ++i) {
            x.data[i] = (float)inputs[i];
            y.data[i] = (float)targets[i];
        }
        Tensor logits = model.forward(x);
        Tensor grad_logits;
        current_loss = cross_entropy(logits, y, grad_logits);
        model.zero_grad();
        model.backward(x, grad_logits);
        update_model(model, opt);

        if (log.is_open() && (it % 10 == 0 || it == iterations - 1)) {
            log << "iter " << it << " loss " << current_loss << "\n";
            log.flush();
        }
        if (it % 5 == 0 || it == iterations - 1)
            print_progress(it + 1, iterations, current_loss, start_time);
        if (it > 0 && it % 500 == 0) {
            std::cout << "\n[*] Сохранение (iter " << it << ")...\n";
            model.save_weights(out_prefix + "_brain.bin");
        }
    }
    std::cout << "\n\n[pretrain] ГОТОВО! Финальный loss: " << current_loss << "\n";
    model.save_weights(out_prefix + "_brain.bin");
}

// ---------- Шаг 2: Finetune (потоковое чтение диалогов, все 1M) ----------
void finetune(const std::string& dialogues_path, const std::string& tok_path,
              const std::string& pretrain_prefix, const std::string& out_path,
              int iterations) {
    BPETokenizer tok;
    try { tok.load(tok_path); } catch(...) { std::cerr << "Сначала запусти Шаг 0!\n"; return; }

    std::ifstream f(dialogues_path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "нет " << dialogues_path << "\n"; return; }

    std::vector<int> tokens;
    std::vector<size_t> starts;
    tokens.reserve(50'000'000);
    starts.reserve(1'100'000);

    std::string line;
    size_t line_no = 0, count = 0;
    std::cout << "[finetune] читаем диалоги...\n";
    while (std::getline(f, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() < 3) continue;
        size_t pos = line.find('|');
        if (pos == std::string::npos) continue;
        std::string q = line.substr(0, pos), a = line.substr(pos + 1);
        if (q.empty() || a.empty() || q == a) continue;
        starts.push_back(tokens.size());
        auto ids = tok.encode_chat(q, a);
        tokens.insert(tokens.end(), ids.begin(), ids.end());
        ++count;
        if (count % 100000 == 0)
            std::cout << "  " << count << " диалогов, токенов: " << tokens.size() << "\n";
    }
    f.close();
    std::cout << "[finetune] итог: токенов " << tokens.size()
              << ", диалогов " << starts.size() << "\n";
    if (tokens.size() < 100) { std::cerr << "Слишком мало!\n"; return; }

    Dataset ds(tokens, BLOCK_SIZE, BATCH_SIZE, starts);
    Transformer model(tok.vocab_size, DIM, N_LAYER, N_HEAD, BLOCK_SIZE);
    model.load_weights(pretrain_prefix + "_brain.bin");

    AdamW opt(3e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, 50.0f);
    std::mt19937 rng(42);
    std::vector<int> inputs, targets;

    print_start_msg("FINETUNE (Обучение на диалогах)", iterations);
    auto start_time = std::chrono::steady_clock::now();
    float current_loss = 0.0f;
    std::ofstream log("finetune_log.txt");

    for (int it = 0; it < iterations; ++it) {
        ds.get_batch(inputs, targets, rng);
        Tensor x({BATCH_SIZE, BLOCK_SIZE}), y({BATCH_SIZE, BLOCK_SIZE});
        for (size_t i = 0; i < x.size; ++i) {
            x.data[i] = (float)inputs[i];
            y.data[i] = (float)targets[i];
        }
        Tensor logits = model.forward(x);
        Tensor grad_logits;
        current_loss = cross_entropy(logits, y, grad_logits);
        model.zero_grad();
        model.backward(x, grad_logits);
        update_model(model, opt);

        if (log.is_open() && (it % 10 == 0 || it == iterations - 1)) {
            log << "iter " << it << " loss " << current_loss << "\n";
            log.flush();
        }
        if (it % 5 == 0 || it == iterations - 1)
            print_progress(it + 1, iterations, current_loss, start_time);
        if (it > 0 && it % 500 == 0) {
            std::cout << "\n[*] Сохранение (iter " << it << ")...\n";
            model.save_weights(out_path);
        }
    }
    std::cout << "\n\n[finetune] ГОТОВО! Финальный loss: " << current_loss << "\n";
    model.save_weights(out_path);
}

// ---------- Шаг 3: Чат ----------
void chat(const std::string& tok_path, const std::string& brain_path) {
    std::ofstream chat_log("chat_log.txt");
    BPETokenizer tok;
    try { tok.load(tok_path); } catch(...) { std::cerr << "Нет словаря!\n"; return; }

    Transformer model(tok.vocab_size, DIM, N_LAYER, N_HEAD, BLOCK_SIZE);
    model.load_weights(brain_path);

    std::vector<int> history;
    std::cout << "\n=== ЧАТ ===\n(напиши 'выход' для завершения)\n";
    while (true) {
        std::string user_input;
        std::cout << "\nВы: ";
        std::getline(std::cin, user_input);
        if (user_input == "выход" || user_input == "exit") break;

        if (chat_log.is_open()) chat_log << "пользователь: " << user_input << "\n";

        history.push_back(BPETokenizer::USER_ID);
        auto u = tok.encode(user_input);
        history.insert(history.end(), u.begin(), u.end());
        history.push_back(BPETokenizer::BOT_ID);
        size_t prompt_len = history.size();

        for (int i = 0; i < 40; ++i) {
            size_t start = (history.size() > BLOCK_SIZE) ? history.size() - BLOCK_SIZE : 0;
            size_t cur_len = history.size() - start;
            Tensor x_gen({1, cur_len});
            for (size_t j = 0; j < cur_len; ++j) x_gen.data[j] = (float)history[start + j];
            Tensor logits = model.forward(x_gen);
            int next_id = model.sample_top_k(logits, (int)cur_len, 0.8f, 20);
            if (next_id == BPETokenizer::END_ID) break;
            history.push_back(next_id);
        }
        std::vector<int> answer(history.begin() + prompt_len, history.end());
        std::cout << "Бот: " << tok.decode(answer) << "\n";
        history.push_back(BPETokenizer::END_ID);
        if (history.size() > 200) history.erase(history.begin(), history.begin() + (history.size() - 200));

        if (chat_log.is_open()){
            chat_log << "нейроСеть: " << tok.decode(answer) << "\n";
            chat_log.flush();
        }
    }
}

// ---------- main ----------
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode; GetConsoleMode(h, &mode);
    SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
    while(true){
    std::cout << "0) обучить BPE (словарь " << VOCAB_TARGET << ", " << BPE_CHARS/1000000 << " МБ)\n";
    std::cout << "1) pretrain (" << ITERS << " итераций, до " << MAX_CORPUS/1000000 << " МБ)\n";
    std::cout << "2) finetune (" << ITERS << " итераций)\n";
    std::cout << "3) чат\n";
    std::cout << "> ";
    int choice; std::cin >> choice; std::cin.ignore();

    try {
        if (choice == 0) {
            train_tokenizer_only("corpus.txt", "pretrain_tokenizer.bin", VOCAB_TARGET, BPE_CHARS);
        } else if (choice == 1) {
            pretrain("corpus.txt", "pretrain_tokenizer.bin", "pretrain", ITERS);
        } else if (choice == 2) {
            finetune("dialogues.txt", "pretrain_tokenizer.bin", "pretrain", "chat_brain.bin", ITERS);
        } else if (choice == 3) {
            chat("pretrain_tokenizer.bin", "chat_brain.bin");
        }
    } catch (const std::exception& e) {
        std::cerr << "ОШИБКА: " << e.what() << "\n";
    }

    std::cout << "\nНажмите Enter...\n";
    std::cin.get();
    std::cout << "\033[2J\033[H";
    }
    return 0;
}
