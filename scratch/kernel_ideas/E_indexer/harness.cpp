// Family E harness: production dGPU indexer kernels (UNMODIFIED in-tree sources compiled with
// $KFLAGS_V41 --genco) launched exactly as crates/v4flash-kernels/src/{indexer,candidate_blocks}.rs
// launch them, on synthetic data at the production shapes, plus candidate kernels from cand_*.hip.
//
//   ./harness_gfx1201 <section> [dir] [opts]
//   section: score | select | gather | cand | small
//   opts:    short   (a handful of calls, for ATT)
//            n=...   comma list of n_comp values (default 65536,131072,235000)
//            b=...   comma list of batch sizes   (default 1,4)
//            rounds=N
#include "kbench.h"

#include <sys/stat.h>

#include <map>
#include <memory>

// ---------------------------------------------------------------- production constants
static const unsigned STRIDE = 369152;          // ATTN_MIXED_MAX_KEYS (v41): 368640 + 128 + 384
static const unsigned TOP_K = 512;              // INDEXER_TOP_K
static const unsigned N_HEAD = 32, HEAD_DIM = 128;  // indexer heads (V4.1), dim
static const unsigned KEY_ROW_BYTES = 80;       // E2M1_KEY_ROW_BYTES
static const unsigned MAIN_HEAD_DIM = 512;      // N_HEAD_DIM (main comp_kv row = 1 KB f16)
static const unsigned CB_SIZE = 8, CB_TOPK = 2048;  // candidate pool
static const unsigned SORT_N = 4096;

// ---------------------------------------------------------------- helper kernels
__global__ void k_fix_exponents(unsigned char* keys, size_t rows, uint32_t seed) {
    size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; r < rows; r += (size_t)gridDim.x * blockDim.x) {
        unsigned char* row = keys + r * KEY_ROW_BYTES;
        uint32_t x = (uint32_t)r * 2654435761u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
        for (int i = 0; i < 4; i++) row[64 + i] = (unsigned char)(signed char)((int)((x >> (8 * i)) % 5u) - 3);
        for (int i = 68; i < 80; i++) row[i] = 0;
    }
}
__global__ void k_flush_read(const uint4* p, size_t n16, unsigned* sink) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    for (; i < n16; i += (size_t)gridDim.x * blockDim.x) { uint4 v = p[i]; acc += v.x ^ v.w; }
    if (acc == 0x12345678u) *sink = acc;
}
// random selection: b x TOP_K indices in [0, n_rows), row-distinct-ish (stride hash)
__global__ void k_gen_selected(int* sel, unsigned b, unsigned n_rows, uint32_t seed) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * TOP_K) return;
    uint32_t x = i * 2654435761u ^ seed;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    sel[i] = (int)(x % n_rows);
}
// prefill-like selection: token t's picks are a window of near-consecutive rows around a
// per-token random centre (adjacent tokens share most rows) — the LOCAL regime.
__global__ void k_gen_selected_local(int* sel, unsigned b, unsigned n_rows, uint32_t seed) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * TOP_K) return;
    unsigned t = i / TOP_K, k = i % TOP_K;
    uint32_t x = (t / 8) * 2654435761u ^ seed;   // centre shared by 8 consecutive tokens
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    unsigned centre = x % n_rows;
    uint32_t y = i * 0x9E3779B9u ^ seed; y ^= y >> 13; y *= 0x85ebca6bu; y ^= y >> 16;
    unsigned off = (k * 3u + (y % 3u)) % (TOP_K * 3u);   // ~1536-row window, 1/3 density
    sel[i] = (int)((centre + off) % n_rows);
}
__global__ void k_quantize(float* s, size_t n, float q) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) s[i] = rintf(s[i] / q) * q;
}
__global__ void k_set_u32(unsigned* p, size_t n, unsigned v) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

struct Flush {
    uint4* buf = nullptr; unsigned* sink = nullptr; size_t n16;
    explicit Flush(size_t bytes) : n16(bytes / 16) { buf = kb::dalloc<uint4>(n16); sink = kb::dalloc<unsigned>(1); KB_CHECK(hipMemset(buf, 1, bytes)); }
    void operator()(hipStream_t s) { hipLaunchKernelGGL(k_flush_read, dim3(2048), dim3(256), 0, s, (const uint4*)buf, n16, sink); }
};

static bool exists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }
static std::vector<unsigned> parse_list(const std::string& s) {
    std::vector<unsigned> v; size_t i = 0;
    while (i < s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); v.push_back((unsigned)atoi(s.substr(i, j - i).c_str())); i = j + 1; }
    return v;
}
struct Opts {
    std::string section, dir = ".";
    bool short_mode = false;
    std::vector<unsigned> ns = {65536, 131072, 235000}, bs = {1, 4};
    int rounds = 40;
};
static Opts parse(int argc, char** argv) {
    Opts o;
    if (argc < 2) { fprintf(stderr, "usage: harness <section> [dir] [short] [n=..] [b=..] [rounds=N]\n"); exit(64); }
    o.section = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "short") o.short_mode = true;
        else if (a.rfind("n=", 0) == 0) o.ns = parse_list(a.substr(2));
        else if (a.rfind("b=", 0) == 0) o.bs = parse_list(a.substr(2));
        else if (a.rfind("rounds=", 0) == 0) o.rounds = atoi(a.c_str() + 7);
        else o.dir = a;
    }
    return o;
}
static unsigned cdiv(unsigned a, unsigned b) { return (a + b - 1) / b; }
static std::string arch = "gfx1201";

// A code object that may not exist (candidate not built yet).
struct OptModule {
    std::unique_ptr<kb::Module> m;
    explicit OptModule(const std::string& path) { if (exists(path)) m.reset(new kb::Module(path)); else fprintf(stderr, "[harness] (no %s)\n", path.c_str()); }
    bool ok() const { return (bool)m; }
    hipFunction_t fn(const char* n) const { return m->fn(n); }
};

// ================================================================ score
// Synthetic keys: b_max per-row stores of n_max rows (production decode: each row is a different
// stream → its own store; keys_base_per[r] = r * n_max). q, hw random. Cold-key regime.
static void section_score(const Opts& o) {
    kb::Module base(o.dir + "/base_indexer_score_wmma_" + arch + ".hsaco");
    hipFunction_t f_mw = base.fn("indexer_score_wmma_batched_mw_e2m1");
    OptModule cand(o.dir + "/cand_score_" + arch + ".hsaco");

    unsigned n_max = 0, b_max = 0;
    for (unsigned n : o.ns) n_max = std::max(n_max, n);
    for (unsigned b : o.bs) b_max = std::max(b_max, b);
    const size_t key_rows = (size_t)b_max * n_max;
    unsigned char* keys = kb::dalloc<unsigned char>(key_rows * KEY_ROW_BYTES);
    kb::fill_bytes(keys, key_rows * KEY_ROW_BYTES, 7);
    hipLaunchKernelGGL(k_fix_exponents, dim3(4096), dim3(256), 0, 0, keys, key_rows, 11u);
    float* q = kb::dalloc<float>((size_t)b_max * N_HEAD * HEAD_DIM);
    float* hw = kb::dalloc<float>((size_t)b_max * N_HEAD);
    kb::fill_f32(q, (size_t)b_max * N_HEAD * HEAD_DIM, 3, -1.f, 1.f);
    kb::fill_f32(hw, (size_t)b_max * N_HEAD, 5, 0.f, 0.05f);
    float* scores = kb::dalloc<float>((size_t)b_max * STRIDE);
    float* scores2 = kb::dalloc<float>((size_t)b_max * STRIDE);
    unsigned* n_idx_per = kb::dalloc<unsigned>(b_max);
    unsigned* keys_base = kb::dalloc<unsigned>(b_max);
    std::vector<unsigned> h_base(b_max);
    for (unsigned r = 0; r < b_max; r++) h_base[r] = r * n_max;
    KB_CHECK(hipMemcpy(keys_base, h_base.data(), b_max * 4, hipMemcpyHostToDevice));
    Flush flush(64u << 20);
    fprintf(stderr, "[score] keys %.1f MB (%u rows x %u stores), scores %.1f MB, flush 64 MB\n",
            key_rows * 80 / 1e6, n_max, b_max, b_max * STRIDE * 4 / 1e6);

    // candidates with the SAME signature as the production kernel (grid differs per name)
    struct Cand { std::string name; hipFunction_t f; unsigned cols_per_wg; unsigned block; };
    std::vector<Cand> cands;
    if (cand.ok()) {
        const char* names[] = {"score_mw_pf", "score_mw_coal", "score_mw_pf_coal", "score_mw_qreg", "score_mw_pf_qreg", nullptr};
        for (int i = 0; names[i]; i++) {
            hipFunction_t f;
            if (hipModuleGetFunction(&f, cand.m->m, names[i]) == hipSuccess) cands.push_back({names[i], f, 1024, 256});
        }
    }

    for (unsigned n : o.ns) {
        for (unsigned b : o.bs) {
            std::vector<unsigned> h_n(b_max, n);
            KB_CHECK(hipMemcpy(n_idx_per, h_n.data(), b_max * 4, hipMemcpyHostToDevice));
            const unsigned gx = cdiv(n, 1024);
            auto run_base = [&](hipStream_t s, float* out) {
                kb::launch(f_mw, dim3(gx, b), dim3(256), 0, s, out, (const float*)q, (const float*)hw,
                           (const unsigned char*)keys, (const unsigned*)n_idx_per, STRIDE, (const unsigned*)keys_base);
            };
            const double bytes = (double)b * n * KEY_ROW_BYTES + (double)b * gx * 1024 * 4;
            const double flops = (double)b * n * N_HEAD * HEAD_DIM * 2;
            // correctness of candidates vs the production kernel
            KB_CHECK(hipMemset(scores, 0, (size_t)b_max * STRIDE * 4));
            run_base(0, scores);
            KB_CHECK(hipDeviceSynchronize());
            auto ref = kb::d2h(scores, (size_t)b * STRIDE);
            for (auto& c : cands) {
                KB_CHECK(hipMemset(scores2, 0, (size_t)b_max * STRIDE * 4));
                kb::launch(c.f, dim3(cdiv(n, c.cols_per_wg), b), dim3(c.block), 0, 0, scores2, (const float*)q, (const float*)hw,
                           (const unsigned char*)keys, (const unsigned*)n_idx_per, STRIDE, (const unsigned*)keys_base);
                KB_CHECK(hipDeviceSynchronize());
                auto got = kb::d2h(scores2, (size_t)b * STRIDE);
                // compare the written region [0, gx*1024) of every row (rest untouched by both)
                kb::Cmp tot; tot.n = 0;
                for (unsigned r = 0; r < b; r++) {
                    kb::Cmp c1 = kb::compare_f32(ref.data() + (size_t)r * STRIDE, got.data() + (size_t)r * STRIDE, (size_t)gx * 1024);
                    tot.n += c1.n; tot.n_bit_diff += c1.n_bit_diff; tot.n_nonfinite += c1.n_nonfinite;
                    tot.max_abs = std::max(tot.max_abs, c1.max_abs); tot.max_rel = std::max(tot.max_rel, c1.max_rel);
                    tot.rmse += c1.rmse * c1.rmse; tot.ref_rms += c1.ref_rms * c1.ref_rms;
                }
                tot.rmse = std::sqrt(tot.rmse / b); tot.ref_rms = std::sqrt(tot.ref_rms / b);
                char tag[128]; snprintf(tag, sizeof tag, "%s n=%u b=%u", c.name.c_str(), n, b);
                kb::print_cmp(tag, tot);
            }
            if (o.short_mode) { for (int i = 0; i < 6; i++) { flush(0); run_base(0, scores); } KB_CHECK(hipDeviceSynchronize()); continue; }

            std::vector<kb::Variant> vs;
            vs.push_back({"base batched_mw_e2m1", [&](hipStream_t s) { run_base(s, scores); }, bytes, flops});
            for (auto& c : cands)
                vs.push_back({"cand " + c.name, [&, c](hipStream_t s) {
                    kb::launch(c.f, dim3(cdiv(n, c.cols_per_wg), b), dim3(c.block), 0, s, scores2, (const float*)q, (const float*)hw,
                               (const unsigned char*)keys, (const unsigned*)n_idx_per, STRIDE, (const unsigned*)keys_base); }, bytes, flops});
            char tag[160];
            kb::AbOpts ab; ab.rounds = o.rounds; ab.inner = 1; ab.graph = false; ab.between = [&](hipStream_t s) { flush(s); };
            snprintf(tag, sizeof tag, "score n=%u b=%u COLD keys (flush 64MB), direct inner=1", n, b); ab.tag = tag;
            kb::ab(vs, ab);
            kb::AbOpts ab2; ab2.rounds = o.rounds; ab2.inner = 5; ab2.graph = true;
            snprintf(tag, sizeof tag, "score n=%u b=%u WARM keys (bound), graph inner=5", n, b); ab2.tag = tag;
            kb::ab(vs, ab2);
        }
    }
}

// ================================================================ select (top-k)
// Scores produced by the production score kernel (realistic distribution), warm in L2.
// Baseline = the production launcher: select_ilp + chunk + regroup* + merge (early-outs).
struct TopkChain {
    hipFunction_t sel, chunk, regroup, merge;
    int* selected; unsigned* done; unsigned* scratch; size_t scratch_per_row;
};
static std::vector<unsigned> merge_levels(unsigned n) {
    std::vector<unsigned> lv;
    if (n <= SORT_N) return lv;
    const unsigned group_span = (SORT_N / TOP_K) * TOP_K;
    lv.push_back(cdiv(n, SORT_N) * TOP_K);
    while (lv.back() > SORT_N) { unsigned nx = cdiv(lv.back(), group_span) * TOP_K; if (nx >= lv.back()) break; lv.push_back(nx); }
    return lv;
}
static void launch_topk_chain(const TopkChain& c, hipStream_t s, const float* scores, const unsigned* n_idx_per,
                              unsigned n_max, unsigned b, bool select_only) {
    kb::launch(c.sel, dim3(b), dim3(1024), 0, s, c.selected, c.done, scores, n_idx_per, STRIDE, TOP_K);
    if (n_max <= SORT_N || select_only) return;
    const unsigned n_words = cdiv(STRIDE, 32);
    const unsigned n_chunks = cdiv(n_max, SORT_N), n_cand = n_chunks * TOP_K;
    unsigned* nullb = nullptr;
    if (n_cand <= SORT_N) {
        kb::launch(c.chunk, dim3(n_chunks, b), dim3(1024), 0, s, c.scratch, scores, n_idx_per, STRIDE, n_cand, TOP_K, (const unsigned*)c.done);
        kb::launch(c.merge, dim3(1, b), dim3(1024), 0, s, c.selected, nullb, (const unsigned*)c.scratch, scores, n_idx_per,
                   STRIDE, n_cand, n_words, TOP_K, n_cand, (const unsigned*)c.done);
        return;
    }
    auto lv = merge_levels(n_max);
    std::vector<unsigned*> ptrs; size_t off = 0;
    for (unsigned l : lv) { ptrs.push_back(c.scratch + (size_t)b * off); off += l; }
    const unsigned group_span = (SORT_N / TOP_K) * TOP_K;
    kb::launch(c.chunk, dim3(n_chunks, b), dim3(1024), 0, s, ptrs[0], scores, n_idx_per, STRIDE, n_cand, TOP_K, (const unsigned*)c.done);
    for (size_t i = 0; i + 1 < lv.size(); i++)
        kb::launch(c.regroup, dim3(cdiv(lv[i], group_span), b), dim3(1024), 0, s, ptrs[i + 1], (const unsigned*)ptrs[i], scores, n_idx_per,
                   STRIDE, lv[i], lv[i + 1], TOP_K, group_span, (const unsigned*)c.done);
    const unsigned last = lv.back();
    kb::launch(c.merge, dim3(1, b), dim3(1024), 0, s, c.selected, nullb, (const unsigned*)ptrs.back(), scores, n_idx_per,
               STRIDE, last, n_words, TOP_K, last, (const unsigned*)c.done);
}

static void section_select(const Opts& o) {
    kb::Module base(o.dir + "/base_indexer_topk_bitonic_" + arch + ".hsaco");
    kb::Module scoremod(o.dir + "/base_indexer_score_wmma_" + arch + ".hsaco");
    hipFunction_t f_mw = scoremod.fn("indexer_score_wmma_batched_mw_e2m1");
    OptModule cand(o.dir + "/cand_topk_" + arch + ".hsaco");
    unsigned n_max = 0, b_max = 0;
    for (unsigned n : o.ns) n_max = std::max(n_max, n);
    for (unsigned b : o.bs) b_max = std::max(b_max, b);
    // one key store shared by all rows (rows differ by q → different scores)
    unsigned char* keys = kb::dalloc<unsigned char>((size_t)n_max * KEY_ROW_BYTES);
    kb::fill_bytes(keys, (size_t)n_max * KEY_ROW_BYTES, 7);
    hipLaunchKernelGGL(k_fix_exponents, dim3(4096), dim3(256), 0, 0, keys, (size_t)n_max, 11u);
    float* q = kb::dalloc<float>((size_t)b_max * N_HEAD * HEAD_DIM);
    float* hw = kb::dalloc<float>((size_t)b_max * N_HEAD);
    kb::fill_f32(q, (size_t)b_max * N_HEAD * HEAD_DIM, 3, -1.f, 1.f);
    kb::fill_f32(hw, (size_t)b_max * N_HEAD, 5, 0.f, 0.05f);
    float* scores = kb::dalloc<float>((size_t)b_max * STRIDE);
    unsigned* n_idx_per = kb::dalloc<unsigned>(b_max);
    unsigned* keys_base = kb::dalloc<unsigned>(b_max);
    KB_CHECK(hipMemset(keys_base, 0, b_max * 4));

    TopkChain B{base.fn("indexer_topk_select_batched_ilp"), base.fn("indexer_topk_chunk_4096_batched"),
                base.fn("indexer_topk_regroup_4096_batched"), base.fn("indexer_topk_merge_4096_batched")};
    {
        auto lv = merge_levels(n_max); size_t per = 0; for (unsigned l : lv) per += l;
        B.scratch_per_row = std::max<size_t>(per, 1);
        B.selected = kb::dalloc<int>((size_t)b_max * TOP_K); B.done = kb::dalloc<unsigned>(b_max);
        B.scratch = kb::dalloc<unsigned>((size_t)b_max * B.scratch_per_row);
    }
    // candidates: same signature as select_ilp (selected, done, scores, n_idx_per, stride, top_k), grid (b) x block
    struct Cand { std::string name; hipFunction_t f; unsigned block; int* selected; unsigned* done; };
    std::vector<Cand> cands;
    if (cand.ok()) {
        const char* names[] = {"topk_select_regsort", "topk_select_radix", "topk_select_radix_regsort", "topk_select_v3", nullptr};
        for (int i = 0; names[i]; i++) {
            hipFunction_t f;
            if (hipModuleGetFunction(&f, cand.m->m, names[i]) == hipSuccess)
                cands.push_back({names[i], f, 1024, kb::dalloc<int>((size_t)b_max * TOP_K), kb::dalloc<unsigned>(b_max)});
        }
    }
    auto cmp_sel = [&](const char* tag, const int* a, const unsigned* da, const int* bsel, const unsigned* db, unsigned b) {
        auto ha = kb::d2h(a, (size_t)b * TOP_K), hb = kb::d2h(bsel, (size_t)b * TOP_K);
        auto fa = kb::d2h(da, b), fb = kb::d2h(db, b);
        size_t diff = 0, first = (size_t)-1;
        for (size_t i = 0; i < ha.size(); i++) if (ha[i] != hb[i]) { diff++; if (first == (size_t)-1) first = i; }
        size_t fdiff = 0; for (unsigned i = 0; i < b; i++) fdiff += fa[i] != fb[i];
        printf("SEL %-50s b=%u exact=%s diff=%zu first_diff=%ld done_diff=%zu done=[", tag, b, diff == 0 && fdiff == 0 ? "YES" : "no",
               diff, (long)first, fdiff);
        for (unsigned i = 0; i < b; i++) printf("%u", fa[i]); printf("]\n");
        printf("KBJSON {\"sel\":\"%s\",\"b\":%u,\"exact\":%d,\"diff\":%zu,\"done_diff\":%zu}\n", tag, b, (int)(diff == 0 && fdiff == 0), diff, fdiff);
    };

    for (unsigned n : o.ns) {
        for (unsigned b : o.bs) {
            std::vector<unsigned> h_n(b_max, n);
            KB_CHECK(hipMemcpy(n_idx_per, h_n.data(), b_max * 4, hipMemcpyHostToDevice));
            // realistic scores
            kb::launch(f_mw, dim3(cdiv(n, 1024), b), dim3(256), 0, 0, scores, (const float*)q, (const float*)hw,
                       (const unsigned char*)keys, (const unsigned*)n_idx_per, STRIDE, (const unsigned*)keys_base);
            KB_CHECK(hipDeviceSynchronize());
            // ---- correctness: realistic, tie-heavy (quantised), and -inf masked variants
            for (int variant = 0; variant < 3 && !cands.empty(); variant++) {
                if (variant == 1) { hipLaunchKernelGGL(k_quantize, dim3(cdiv(b * STRIDE, 256)), dim3(256), 0, 0, scores, (size_t)b * STRIDE, 0.25f); }
                if (variant == 2) {  // -inf for 3/4 of the 8-blocks (candidate mask), leaves ties from variant 1
                    std::vector<float> h = kb::d2h(scores, (size_t)b * STRIDE);
                    for (unsigned r = 0; r < b; r++) for (unsigned p = 0; p < n; p++) if (((p / 8) * 2654435761u >> 7) % 4 != 0) h[(size_t)r * STRIDE + p] = -INFINITY;
                    KB_CHECK(hipMemcpy(scores, h.data(), h.size() * 4, hipMemcpyHostToDevice));
                }
                KB_CHECK(hipDeviceSynchronize());
                launch_topk_chain(B, 0, scores, n_idx_per, n, b, false);
                KB_CHECK(hipDeviceSynchronize());
                for (auto& c : cands) {
                    KB_CHECK(hipMemset(c.selected, 0xEE, (size_t)b * TOP_K * 4));
                    kb::launch(c.f, dim3(b), dim3(c.block), 0, 0, c.selected, c.done, (const float*)scores, (const unsigned*)n_idx_per, STRIDE, TOP_K);
                    KB_CHECK(hipDeviceSynchronize());
                    char tag[128]; snprintf(tag, sizeof tag, "%s n=%u %s", c.name.c_str(), n, variant == 0 ? "real" : variant == 1 ? "ties" : "ties+mask");
                    cmp_sel(tag, B.selected, B.done, c.selected, c.done, b);
                }
                if (variant == 2) {  // restore realistic scores
                    kb::launch(f_mw, dim3(cdiv(n, 1024), b), dim3(256), 0, 0, scores, (const float*)q, (const float*)hw,
                               (const unsigned char*)keys, (const unsigned*)n_idx_per, STRIDE, (const unsigned*)keys_base);
                    KB_CHECK(hipDeviceSynchronize());
                }
            }
            if (o.short_mode) { for (int i = 0; i < 6; i++) launch_topk_chain(B, 0, scores, n_idx_per, n, b, true); for (auto& c : cands) for (int i = 0; i < 6; i++) kb::launch(c.f, dim3(b), dim3(c.block), 0, 0, c.selected, c.done, (const float*)scores, (const unsigned*)n_idx_per, STRIDE, TOP_K); KB_CHECK(hipDeviceSynchronize()); continue; }
            std::vector<kb::Variant> vs;
            vs.push_back({"base select_ilp + chain early-outs (prod)", [&](hipStream_t s) { launch_topk_chain(B, s, scores, n_idx_per, n, b, false); }, (double)b * n * 4});
            vs.push_back({"base select_ilp alone", [&](hipStream_t s) { launch_topk_chain(B, s, scores, n_idx_per, n, b, true); }, (double)b * n * 4});
            for (auto& c : cands)
                vs.push_back({"cand " + c.name + " alone", [&, c](hipStream_t s) {
                    kb::launch(c.f, dim3(b), dim3(c.block), 0, s, c.selected, c.done, (const float*)scores, (const unsigned*)n_idx_per, STRIDE, TOP_K); }, (double)b * n * 4});
            char tag[160];
            kb::AbOpts ab; ab.rounds = o.rounds; ab.inner = 5; ab.graph = true;
            snprintf(tag, sizeof tag, "topk n=%u b=%u WARM scores, graph inner=5", n, b); ab.tag = tag;
            kb::ab(vs, ab);
        }
    }
}

// ================================================================ gather
static void section_gather(const Opts& o) {
    kb::Module base(o.dir + "/base_indexer_gather_" + arch + ".hsaco");
    hipFunction_t f_g = base.fn("indexer_gather_batched");
    OptModule cand(o.dir + "/cand_gather_" + arch + ".hsaco");
    const unsigned store_rows = 49152;  // 48 MB f16 store (production: up to 235K rows = 235 MB, cold)
    unsigned b_max = 0; for (unsigned b : o.bs) b_max = std::max(b_max, b);
    b_max = std::max(b_max, 32u);
    uint16_t* comp = kb::dalloc<uint16_t>((size_t)store_rows * MAIN_HEAD_DIM);
    kb::fill_f16(comp, (size_t)store_rows * MAIN_HEAD_DIM, 9);
    uint16_t* dst = kb::dalloc<uint16_t>((size_t)b_max * TOP_K * MAIN_HEAD_DIM);
    uint16_t* dst2 = kb::dalloc<uint16_t>((size_t)b_max * TOP_K * MAIN_HEAD_DIM);
    int* sel = kb::dalloc<int>((size_t)b_max * TOP_K);
    int* base_per = kb::dalloc<int>(b_max);
    KB_CHECK(hipMemset(base_per, 0, b_max * 4));
    Flush flush(64u << 20);
    struct Cand { std::string name; hipFunction_t f; unsigned rows_per_wg; unsigned block; };
    std::vector<Cand> cands;
    if (cand.ok()) {
        struct { const char* n; unsigned rpw, blk; } tab[] = {{"gather_u4_r4", 4, 256}, {"gather_u4_r8", 8, 512}, {"gather_u4_r2", 2, 128}, {"gather_u4_r16", 16, 1024}, {"gather_u4_r1", 1, 64}, {"gather_u2_r4", 4, 512}, {"gather_u4_r4_2ipt", 8, 256}};
        for (auto& t : tab) { hipFunction_t f; if (hipModuleGetFunction(&f, cand.m->m, t.n) == hipSuccess) cands.push_back({t.n, f, t.rpw, t.blk}); }
    }
    std::vector<unsigned> bs = o.bs; bs.push_back(32);  // 32 = prefill-shaped (512 rows per lane, extrapolate x16)
    for (int regime = 0; regime < 2; regime++) {
        for (unsigned b : bs) {
            if (regime == 0) hipLaunchKernelGGL(k_gen_selected, dim3(cdiv(b * TOP_K, 256)), dim3(256), 0, 0, sel, b, store_rows, 13u + b);
            else hipLaunchKernelGGL(k_gen_selected_local, dim3(cdiv(b * TOP_K, 256)), dim3(256), 0, 0, sel, b, store_rows, 17u + b);
            KB_CHECK(hipDeviceSynchronize());
            auto run_base = [&](hipStream_t s, uint16_t* out) {
                kb::launch(f_g, dim3(TOP_K, b, 2), dim3(256), 0, s, out, (const uint16_t*)comp, (const int*)sel, TOP_K, MAIN_HEAD_DIM, (const int*)base_per);
            };
            const double bytes = 2.0 * b * TOP_K * MAIN_HEAD_DIM * 2;
            KB_CHECK(hipMemset(dst, 0, (size_t)b * TOP_K * MAIN_HEAD_DIM * 2));
            run_base(0, dst); KB_CHECK(hipDeviceSynchronize());
            auto ref = kb::d2h(dst, (size_t)b * TOP_K * MAIN_HEAD_DIM);
            for (auto& c : cands) {
                KB_CHECK(hipMemset(dst2, 0, (size_t)b * TOP_K * MAIN_HEAD_DIM * 2));
                kb::launch(c.f, dim3(cdiv(TOP_K, c.rows_per_wg), b), dim3(c.block), 0, 0, dst2, (const uint16_t*)comp, (const int*)sel, TOP_K, MAIN_HEAD_DIM, (const int*)base_per);
                KB_CHECK(hipDeviceSynchronize());
                auto got = kb::d2h(dst2, (size_t)b * TOP_K * MAIN_HEAD_DIM);
                size_t diff = 0; for (size_t i = 0; i < ref.size(); i++) diff += ref[i] != got[i];
                printf("CMP %-34s b=%u regime=%s bitexact=%s diff=%zu\n", c.name.c_str(), b, regime ? "local" : "random", diff == 0 ? "YES" : "no", diff);
                printf("KBJSON {\"cmp\":\"%s b=%u %s\",\"bitexact\":%d,\"bit_diff\":%zu}\n", c.name.c_str(), b, regime ? "local" : "random", (int)(diff == 0), diff);
            }
            if (o.short_mode) { for (int i = 0; i < 6; i++) { flush(0); run_base(0, dst); } KB_CHECK(hipDeviceSynchronize()); continue; }
            std::vector<kb::Variant> vs;
            vs.push_back({"base indexer_gather_batched (512,b,2)x256", [&](hipStream_t s) { run_base(s, dst); }, bytes});
            for (auto& c : cands)
                vs.push_back({"cand " + c.name, [&, c](hipStream_t s) {
                    kb::launch(c.f, dim3(cdiv(TOP_K, c.rows_per_wg), b), dim3(c.block), 0, s, dst2, (const uint16_t*)comp, (const int*)sel, TOP_K, MAIN_HEAD_DIM, (const int*)base_per); }, bytes});
            char tag[160];
            kb::AbOpts ab; ab.rounds = o.rounds; ab.inner = 1; ab.graph = false; ab.between = [&](hipStream_t s) { flush(s); };
            snprintf(tag, sizeof tag, "gather b=%u %s rows, COLD store (48K rows, flush), direct inner=1", b, regime ? "LOCAL" : "RANDOM"); ab.tag = tag;
            kb::ab(vs, ab);
            if (b <= 8) {
                kb::AbOpts ab2; ab2.rounds = o.rounds; ab2.inner = 5; ab2.graph = true;
                snprintf(tag, sizeof tag, "gather b=%u %s rows, WARM store (bound), graph inner=5", b, regime ? "LOCAL" : "RANDOM"); ab2.tag = tag;
                kb::ab(vs, ab2);
            }
        }
    }
}

// ================================================================ candidate pool
static void section_cand(const Opts& o) {
    kb::Module base(o.dir + "/base_candidate_blocks_" + arch + ".hsaco");
    kb::Module scoremod(o.dir + "/base_indexer_score_wmma_" + arch + ".hsaco");
    hipFunction_t f_max = base.fn("candidate_block_max"), f_thr = base.fn("candidate_threshold"), f_mask = base.fn("candidate_mask_apply");
    hipFunction_t f_mw = scoremod.fn("indexer_score_wmma_batched_mw_e2m1");
    OptModule cand(o.dir + "/cand_cand_" + arch + ".hsaco");
    unsigned n_max = 0, b_max = 0;
    for (unsigned n : o.ns) n_max = std::max(n_max, n);
    for (unsigned b : o.bs) b_max = std::max(b_max, b);
    unsigned char* keys = kb::dalloc<unsigned char>((size_t)n_max * KEY_ROW_BYTES);
    kb::fill_bytes(keys, (size_t)n_max * KEY_ROW_BYTES, 7);
    hipLaunchKernelGGL(k_fix_exponents, dim3(4096), dim3(256), 0, 0, keys, (size_t)n_max, 11u);
    float* q = kb::dalloc<float>((size_t)b_max * N_HEAD * HEAD_DIM);
    float* hw = kb::dalloc<float>((size_t)b_max * N_HEAD);
    kb::fill_f32(q, (size_t)b_max * N_HEAD * HEAD_DIM, 3, -1.f, 1.f);
    kb::fill_f32(hw, (size_t)b_max * N_HEAD, 5, 0.f, 0.05f);
    float* scores = kb::dalloc<float>((size_t)b_max * STRIDE);
    float* scores2 = kb::dalloc<float>((size_t)b_max * STRIDE);
    unsigned* n_idx_per = kb::dalloc<unsigned>(b_max);
    unsigned* keys_base = kb::dalloc<unsigned>(b_max);
    KB_CHECK(hipMemset(keys_base, 0, b_max * 4));
    const unsigned nb_stride = cdiv(STRIDE, CB_SIZE);
    float* bscore = kb::dalloc<float>((size_t)b_max * nb_stride);
    float* bscore2 = kb::dalloc<float>((size_t)b_max * nb_stride);
    unsigned* thr = kb::dalloc<unsigned>(b_max);
    unsigned* thr2 = kb::dalloc<unsigned>(b_max);
    struct Cand { std::string name; hipFunction_t f; };
    std::vector<Cand> cthr;
    if (cand.ok()) {
        const char* names[] = {"candidate_threshold_ilp", "candidate_threshold_v2", nullptr};
        for (int i = 0; names[i]; i++) { hipFunction_t f; if (hipModuleGetFunction(&f, cand.m->m, names[i]) == hipSuccess) cthr.push_back({names[i], f}); }
    }
    for (unsigned n : o.ns) {
        for (unsigned b : o.bs) {
            std::vector<unsigned> h_n(b_max, n);
            KB_CHECK(hipMemcpy(n_idx_per, h_n.data(), b_max * 4, hipMemcpyHostToDevice));
            kb::launch(f_mw, dim3(cdiv(n, 1024), b), dim3(256), 0, 0, scores, (const float*)q, (const float*)hw,
                       (const unsigned char*)keys, (const unsigned*)n_idx_per, STRIDE, (const unsigned*)keys_base);
            KB_CHECK(hipDeviceSynchronize());
            const unsigned nb_max = cdiv(n, CB_SIZE);
            auto run_max = [&](hipStream_t s, float* bs) { kb::launch(f_max, dim3(cdiv(nb_max, 256), b), dim3(256), 0, s, bs, (const float*)scores, (const unsigned*)n_idx_per, STRIDE, nb_stride, CB_SIZE); };
            auto run_thr = [&](hipStream_t s, hipFunction_t f, unsigned* t, const float* bs) { kb::launch(f, dim3(b), dim3(256), 0, s, t, bs, (const unsigned*)n_idx_per, nb_stride, CB_SIZE, CB_TOPK); };
            auto run_mask = [&](hipStream_t s, float* sc) { kb::launch(f_mask, dim3(cdiv(n, 256), b), dim3(256), 0, s, sc, (const float*)bscore, (const unsigned*)thr, (const unsigned*)n_idx_per, STRIDE, nb_stride, CB_SIZE); };
            run_max(0, bscore); run_thr(0, f_thr, thr, bscore); KB_CHECK(hipDeviceSynchronize());
            for (auto& c : cthr) {
                run_thr(0, c.f, thr2, bscore); KB_CHECK(hipDeviceSynchronize());
                auto a = kb::d2h(thr, b), g = kb::d2h(thr2, b);
                size_t diff = 0; for (unsigned i = 0; i < b; i++) diff += a[i] != g[i];
                printf("CMP %-34s n=%u b=%u bitexact=%s diff=%zu thr0=%08x/%08x\n", c.name.c_str(), n, b, diff == 0 ? "YES" : "no", diff, a[0], g[0]);
                printf("KBJSON {\"cmp\":\"%s n=%u b=%u\",\"bitexact\":%d,\"bit_diff\":%zu}\n", c.name.c_str(), n, b, (int)(diff == 0), diff);
            }
            KB_CHECK(hipMemcpy(scores2, scores, (size_t)b * STRIDE * 4, hipMemcpyDeviceToDevice));
            if (o.short_mode) { for (int i = 0; i < 6; i++) { run_max(0, bscore); run_thr(0, f_thr, thr, bscore); run_mask(0, scores2); } KB_CHECK(hipDeviceSynchronize()); continue; }
            std::vector<kb::Variant> vs;
            vs.push_back({"base L20 build: block_max + threshold", [&](hipStream_t s) { run_max(s, bscore); run_thr(s, f_thr, thr, bscore); }, (double)b * n * 4});
            vs.push_back({"base candidate_block_max", [&](hipStream_t s) { run_max(s, bscore2); }, (double)b * n * 4});
            vs.push_back({"base candidate_threshold", [&](hipStream_t s) { run_thr(s, f_thr, thr, bscore); }, (double)b * nb_max * 4 * 4});
            for (auto& c : cthr) vs.push_back({"cand " + c.name, [&, c](hipStream_t s) { run_thr(s, c.f, thr2, bscore); }, (double)b * nb_max * 4 * 4});
            vs.push_back({"base L24+ candidate_mask_apply", [&](hipStream_t s) { run_mask(s, scores2); }, (double)b * n * 4});
            char tag[160];
            kb::AbOpts ab; ab.rounds = o.rounds; ab.inner = 5; ab.graph = true;
            snprintf(tag, sizeof tag, "candidate pool n=%u b=%u WARM scores, graph inner=5", n, b); ab.tag = tag;
            kb::ab(vs, ab);
        }
    }
}

// ================================================================ small kernels (fp4 QAT, idx-q matvec, proj matvec, scale)
static void section_small(const Opts& o) {
    kb::Module qat(o.dir + "/base_indexer_qat_" + arch + ".hsaco");
    kb::Module mv(o.dir + "/base_f16_matvec_" + arch + ".hsaco");
    kb::Module vs_(o.dir + "/base_vec_scale_inplace_" + arch + ".hsaco");
    hipFunction_t f_fp4 = qat.fn("indexer_fp4"), f_mv = mv.fn("f16_matvec_batched"), f_sc = vs_.fn("vec_scale_inplace");
    unsigned b_max = 0; for (unsigned b : o.bs) b_max = std::max(b_max, b);
    const unsigned N_LORA_Q = 1280, N_EMBD = 5120;
    uint16_t* wq = kb::dalloc<uint16_t>((size_t)N_HEAD * HEAD_DIM * N_LORA_Q);   // 4096 x 1280 f16 = 10.5 MB
    uint16_t* wp = kb::dalloc<uint16_t>((size_t)N_HEAD * N_EMBD);                // 32 x 5120 f16 = 328 KB
    kb::fill_f16(wq, (size_t)N_HEAD * HEAD_DIM * N_LORA_Q, 21, -0.02f, 0.02f);
    kb::fill_f16(wp, (size_t)N_HEAD * N_EMBD, 23, -0.02f, 0.02f);
    float* qr = kb::dalloc<float>((size_t)b_max * N_LORA_Q);
    float* xn = kb::dalloc<float>((size_t)b_max * N_EMBD);
    float* iq = kb::dalloc<float>((size_t)b_max * N_HEAD * HEAD_DIM);
    float* hw = kb::dalloc<float>((size_t)b_max * N_HEAD);
    kb::fill_f32(qr, (size_t)b_max * N_LORA_Q, 25); kb::fill_f32(xn, (size_t)b_max * N_EMBD, 27); kb::fill_f32(iq, (size_t)b_max * N_HEAD * HEAD_DIM, 29);
    Flush flush(64u << 20);
    for (unsigned b : o.bs) {
        char tag[160];
        {
            std::vector<kb::Variant> vs;
            vs.push_back({"base f16_matvec_batched idx-q 4096x1280 (512,1,b)", [&](hipStream_t s) { kb::launch(f_mv, dim3(N_HEAD * HEAD_DIM / 8, 1, b), dim3(256), 0, s, iq, (const uint16_t*)wq, (const float*)qr, N_LORA_Q, N_HEAD * HEAD_DIM); }, (double)N_HEAD * HEAD_DIM * N_LORA_Q * 2});
            kb::AbOpts ab; ab.rounds = o.rounds; ab.inner = 1; ab.between = [&](hipStream_t s) { flush(s); };
            snprintf(tag, sizeof tag, "idx-q matvec b=%u COLD weight (10.5 MB), direct inner=1", b); ab.tag = tag;
            kb::ab(vs, ab);
        }
        {
            std::vector<kb::Variant> vs;
            vs.push_back({"base indexer_fp4 (32b)x128", [&](hipStream_t s) { kb::launch(f_fp4, dim3(N_HEAD * b), dim3(128), 0, s, iq, N_HEAD * b); }, (double)b * N_HEAD * HEAD_DIM * 8});
            vs.push_back({"base f16_matvec_batched proj 32x5120 (4,1,b)", [&](hipStream_t s) { kb::launch(f_mv, dim3(4, 1, b), dim3(256), 0, s, hw, (const uint16_t*)wp, (const float*)xn, N_EMBD, N_HEAD); }, (double)N_HEAD * N_EMBD * 2});
            vs.push_back({"base vec_scale_inplace (32b)", [&](hipStream_t s) { kb::launch(f_sc, dim3(cdiv(N_HEAD * b, 256)), dim3(256), 0, s, hw, 0.015625f, N_HEAD * b); }, (double)b * N_HEAD * 8});
            kb::AbOpts ab; ab.rounds = o.rounds; ab.inner = 10; ab.graph = true;
            snprintf(tag, sizeof tag, "small indexer kernels b=%u WARM, graph inner=10", b); ab.tag = tag;
            kb::ab(vs, ab);
        }
    }
}

int main(int argc, char** argv) {
    Opts o = parse(argc, argv);
    kb::init();
    if (o.section == "score") section_score(o);
    else if (o.section == "select") section_select(o);
    else if (o.section == "gather") section_gather(o);
    else if (o.section == "cand") section_cand(o);
    else if (o.section == "small") section_small(o);
    else { fprintf(stderr, "unknown section %s\n", o.section.c_str()); return 64; }
    KB_CHECK(hipDeviceSynchronize());
    return 0;
}
