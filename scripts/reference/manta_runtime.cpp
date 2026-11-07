// Build against llama.cpp. Usage: manta_runtime MODEL PROMPT INPUT_IDS STOP_IDS MAX [FORCED_IDS_OR_DASH [MICROBATCH [EXTRA_BUFFERS [TRACE_JSONL]]]]
#include "llama.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

struct trace_state {
    std::ofstream out;
    int prompt_tokens;
    bool enabled = true;
};
static bool trace_tensor(ggml_tensor *t, bool ask, void *user) {
    auto &trace = *static_cast<trace_state *>(user);
    const std::string name = ggml_get_name(t);
    bool selected = false;
    for (const char *prefix : {"attn_norm-", "Qcur-", "Kcur-", "Vcur-", "kqv_out-", "attn_out-", "ffn_norm-", "ffn_swiglu-", "ffn_down-", "l_out-"}) {
        if (name.rfind(prefix, 0) == 0) selected = true;
    }
    if (ask) return trace.enabled && selected && t->type == GGML_TYPE_F32;
    if (!trace.enabled || !selected) return true;
    int token_dim = -1;
    for (int d = 0; d < 4; ++d) if (t->ne[d] == trace.prompt_tokens) token_dim = d;
    size_t width = 1;
    for (int d = 0; d < 4; ++d) if (d != token_dim) width *= t->ne[d];
    trace.out << "{\"name\":\"" << name << "\",\"shape\":[";
    for (int d = 0; d < 4; ++d) { if (d) trace.out << ','; trace.out << t->ne[d]; }
    trace.out << "],\"values\":[";
    for (size_t i = 0; i < width; ++i) {
        size_t rest = i, offset = 0;
        for (int d = 0; d < 4; ++d) {
            const size_t index = d == token_dim ? t->ne[d] - 1 : rest % t->ne[d];
            if (d != token_dim) rest /= t->ne[d];
            offset += index * t->nb[d];
        }
        float value;
        ggml_backend_tensor_get(t, &value, offset, sizeof(value));
        if (i) trace.out << ',';
        trace.out << std::setprecision(9) << value;
    }
    trace.out << "]}\n";
    return true;
}

static std::vector<llama_token> read_ids(const char *path) {
    std::ifstream in(path);
    std::vector<llama_token> ids;
    llama_token id;
    while (in >> id) ids.push_back(id);
    return ids;
}
static void print_ids(const std::vector<llama_token> &ids) {
    std::cout << '[';
    for (size_t i = 0; i < ids.size(); ++i) { if (i) std::cout << ','; std::cout << ids[i]; }
    std::cout << ']';
}
int main(int argc, char **argv) {
    if (argc < 6) return 2;
    std::ifstream in(argv[2]);
    std::string prompt((std::istreambuf_iterator<char>(in)), {});
    auto input = read_ids(argv[3]), stop = read_ids(argv[4]);
    const bool forced = argc > 6 && std::string(argv[6]) != "-";
    auto force = forced ? read_ids(argv[6]) : std::vector<llama_token>{};
    const int steps = forced ? int(force.size()) : std::stoi(argv[5]);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    if (argc > 8) mp.use_extra_bufts = std::stoi(argv[8]) != 0;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto *vocab = llama_model_get_vocab(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 8192; cp.n_batch = 8192; cp.n_ubatch = 512;
    if (argc > 7) cp.n_ubatch = std::stoi(argv[7]);
    cp.n_threads = 4; cp.n_threads_batch = 4;
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    trace_state trace;
    if (argc > 9) {
        trace.out.open(argv[9]);
        trace.prompt_tokens = int(input.size());
        cp.cb_eval = trace_tensor;
        cp.cb_eval_user_data = &trace;
    }
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    std::vector<llama_token> tokenized(prompt.size() + 32);
    int n = llama_tokenize(vocab, prompt.data(), int(prompt.size()), tokenized.data(), int(tokenized.size()), false, true);
    if (n < 0) return 5;
    tokenized.resize(n);
    const bool ids_only = argc > 9 && std::string(argv[9]) == "--ids-only";
    if (!ids_only && tokenized != input) { std::cerr << "Tokenization mismatch\n"; return 9; }
    if (llama_decode(ctx, llama_batch_get_one(input.data(), int(input.size())))) return 6;
    trace.enabled = false;
    std::cout << std::setprecision(9) << "{\"prompt_tokens\":";
    print_ids(input);
    std::cout << ",\"trace\":[";
    std::vector<llama_token> generated;
    for (int step = 0; step < steps; ++step) {
        auto *logits = llama_get_logits_ith(ctx, -1);
        if (!logits) return 7;
        std::vector<int> ranked(llama_vocab_n_tokens(vocab));
        std::iota(ranked.begin(), ranked.end(), 0);
        std::partial_sort(ranked.begin(), ranked.begin() + 10, ranked.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
        if (step) std::cout << ',';
        std::cout << "{\"step\":" << step << ",\"greedy\":" << ranked[0] << ",\"top\":[";
        for (int k = 0; k < 10; ++k) {
            if (k) std::cout << ',';
            std::cout << "{\"id\":" << ranked[k] << ",\"logit\":" << logits[ranked[k]] << '}';
        }
        std::cout << "]}";
        auto next = forced ? force[step] : ranked[0];
        if (!forced && std::find(stop.begin(), stop.end(), next) != stop.end()) break;
        generated.push_back(next);
        if (step + 1 < steps && llama_decode(ctx, llama_batch_get_one(&next, 1))) return 8;
    }
    std::cout << "],\"generated_tokens\":";
    print_ids(generated);
    std::cout << "}\n";
    llama_free(ctx); llama_model_free(model); llama_backend_free();
}
