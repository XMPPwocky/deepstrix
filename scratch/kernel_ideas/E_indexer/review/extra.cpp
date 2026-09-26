// Reviewer's extra-shape correctness harness for E_indexer/topk_select_hybrid_sort_vec4.
//   ./extra_gfx1201 [dir]
// Compares production `indexer_topk_select_batched_ilp` (+ the production chain launched exactly
// as crates/v4flash-kernels/src/indexer.rs::launch_batched_sel does) against the candidate
// `topk_select_v3_u8` (+ the same chain) on shapes the engineer did NOT test: ragged n per row
// (n = 0, 1, 511, 4096, 4097, 32767/32768/32769 = the U*4096 float4 chunk edges, odd n), b = 8
// (max decode rows), a give-up trap (a >4096 cluster of distinct values sitting only at
// un-sampled positions -> done = 0 -> the chain must produce the row), all-equal, -inf-dominated,
// and a NaN sprinkle (reported, production is undefined there too). Also checks the final
// selection against a CPU sort by the reference total order (score desc, idx asc).
#include "kbench.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

static const unsigned STRIDE = 369152;   // ATTN_MIXED_MAX_KEYS (v41)
static const unsigned TOP_K = 512;
static const unsigned SORT_N = 4096;
static const unsigned SAMPLE = 2048;
static unsigned cdiv(unsigned a, unsigned b) { return (a + b - 1) / b; }

struct Chain {
    hipFunction_t sel, chunk, regroup, merge;
    int* selected; unsigned* done; unsigned* scratch;
};
static std::vector<unsigned> merge_levels(unsigned n) {
    std::vector<unsigned> lv;
    if (n <= SORT_N) return lv;
    const unsigned group_span = (SORT_N / TOP_K) * TOP_K;
    lv.push_back(cdiv(n, SORT_N) * TOP_K);
    while (lv.back() > SORT_N) { unsigned nx = cdiv(lv.back(), group_span) * TOP_K; if (nx >= lv.back()) break; lv.push_back(nx); }
    return lv;
}
// = IndexerTopkBitonic::launch_batched_sel with done=Some, allowed_bits=None.
static void launch_chain(const Chain& c, const float* scores, const unsigned* n_idx_per, unsigned n_max, unsigned b, bool select_only) {
    kb::launch(c.sel, dim3(b), dim3(1024), 0, 0, c.selected, c.done, scores, n_idx_per, STRIDE, TOP_K);
    if (n_max <= SORT_N || select_only) return;
    const unsigned n_words = cdiv(STRIDE, 32);
    const unsigned n_chunks = cdiv(n_max, SORT_N), n_cand = n_chunks * TOP_K;
    unsigned* nullb = nullptr;
    if (n_cand <= SORT_N) {
        kb::launch(c.chunk, dim3(n_chunks, b), dim3(1024), 0, 0, c.scratch, scores, n_idx_per, STRIDE, n_cand, TOP_K, (const unsigned*)c.done);
        kb::launch(c.merge, dim3(1, b), dim3(1024), 0, 0, c.selected, nullb, (const unsigned*)c.scratch, scores, n_idx_per,
                   STRIDE, n_cand, n_words, TOP_K, n_cand, (const unsigned*)c.done);
        return;
    }
    auto lv = merge_levels(n_max);
    std::vector<unsigned*> ptrs; size_t off = 0;
    for (unsigned l : lv) { ptrs.push_back(c.scratch + (size_t)b * off); off += l; }
    const unsigned group_span = (SORT_N / TOP_K) * TOP_K;
    kb::launch(c.chunk, dim3(n_chunks, b), dim3(1024), 0, 0, ptrs[0], scores, n_idx_per, STRIDE, n_cand, TOP_K, (const unsigned*)c.done);
    for (size_t i = 0; i + 1 < lv.size(); i++)
        kb::launch(c.regroup, dim3(cdiv(lv[i], group_span), b), dim3(1024), 0, 0, ptrs[i + 1], (const unsigned*)ptrs[i], scores, n_idx_per,
                   STRIDE, lv[i], lv[i + 1], TOP_K, group_span, (const unsigned*)c.done);
    const unsigned last = lv.back();
    kb::launch(c.merge, dim3(1, b), dim3(1024), 0, 0, c.selected, nullb, (const unsigned*)ptrs.back(), scores, n_idx_per,
               STRIDE, last, n_words, TOP_K, last, (const unsigned*)c.done);
}

struct Lcg { uint64_t s; explicit Lcg(uint64_t x) : s(x) {} uint32_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(s >> 33); }
             float uni(float lo, float hi) { return lo + (hi - lo) * (next() / 4294967296.0f); } };

// kinds: 0 random, 1 ties (quantised 1/16), 2 all-equal, 3 -inf mask 3/4 of 8-blocks + ties,
//        4 give-up trap, 5 -inf dominated (100 finite), 6 NaN sprinkle (report only)
static void gen_row(float* row, unsigned n, int kind, Lcg& rng) {
    for (unsigned i = 0; i < n; i++) row[i] = rng.uni(-1.f, 1.f);
    if (kind == 1 || kind == 3) for (unsigned i = 0; i < n; i++) row[i] = std::round(row[i] * 16.f) / 16.f;
    if (kind == 2) for (unsigned i = 0; i < n; i++) row[i] = 0.125f;
    if (kind == 3) for (unsigned i = 0; i < n; i++) if ((((i / 8) * 2654435761u) >> 7) % 4 != 0) row[i] = -INFINITY;
    if (kind == 4 && n > SAMPLE * 4) {
        // low values everywhere, 3.0 at the first 100 sampled positions, a cluster of ~4200
        // DISTINCT values in (2.0, 2.1) placed only at positions no sample touches.
        for (unsigned i = 0; i < n; i++) row[i] = rng.uni(0.f, 1.f);
        for (unsigned j = 0; j < 100; j++) row[(unsigned)(((unsigned long long)j * n) / SAMPLE)] = 3.0f;
        unsigned placed = 0;
        for (unsigned j = 0; j < SAMPLE && placed < 4200; j++) {
            const unsigned s0 = (unsigned)(((unsigned long long)j * n) / SAMPLE);
            const unsigned s1 = (unsigned)(((unsigned long long)(j + 1) * n) / SAMPLE);
            for (unsigned p = s0 + 1; p + 1 < s1 && p < n && placed < 4200; p++) {
                if (p - s0 > 3) break;
                row[p] = 2.0f + 0.1f * ((placed + 1) / 4300.f); placed++;
            }
        }
    }
    if (kind == 5) { for (unsigned i = 0; i < n; i++) row[i] = -INFINITY; for (unsigned k = 0; k < 100 && k < n; k++) row[(rng.next() % n)] = rng.uni(0.f, 1.f); }
    if (kind == 6) { for (unsigned k = 0; k < 8 && k < n; k++) row[(rng.next() % n)] = NAN; }
}

// CPU reference of the final selection: indices sorted by (score desc, idx asc), first top_k, -1 past n.
static std::vector<int> cpu_ref(const float* row, unsigned n) {
    std::vector<int> idx(n); for (unsigned i = 0; i < n; i++) idx[i] = (int)i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
    std::vector<int> out(TOP_K, -1);
    for (unsigned i = 0; i < TOP_K && i < n; i++) out[i] = idx[i];
    return out;
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    kb::init();
    kb::Module base(dir + "/base_indexer_topk_bitonic_gfx1201.hsaco");
    kb::Module cand(dir + "/cand_topk_gfx1201.hsaco");
    const unsigned B_MAX = 8;
    float* d_scores = kb::dalloc<float>((size_t)B_MAX * STRIDE);
    unsigned* d_n = kb::dalloc<unsigned>(B_MAX);
    Chain A{base.fn("indexer_topk_select_batched_ilp"), base.fn("indexer_topk_chunk_4096_batched"),
            base.fn("indexer_topk_regroup_4096_batched"), base.fn("indexer_topk_merge_4096_batched")};
    Chain C = A; C.sel = cand.fn("topk_select_v3_u8");
    size_t per = 0; for (unsigned l : merge_levels(STRIDE)) per += l;
    for (Chain* c : {&A, &C}) { c->selected = kb::dalloc<int>((size_t)B_MAX * TOP_K); c->done = kb::dalloc<unsigned>(B_MAX); c->scratch = kb::dalloc<unsigned>((size_t)B_MAX * per); }

    struct Case { const char* name; std::vector<unsigned> ns; int kind; };
    std::vector<Case> cases = {
        {"ragged b=8 random", {235000, 234999, 4096, 4097, 511, 1, 32768, 32769}, 0},
        {"ragged b=8 ties", {235000, 234999, 4096, 4097, 511, 1, 32768, 32769}, 1},
        {"ragged b=8 ties+mask", {235000, 234999, 4096, 4097, 511, 1, 32768, 32769}, 3},
        {"edges b=8 random", {32767, 32771, 36863, 36865, 8191, 8193, 4095, 100003}, 0},
        {"n=0 row + b=3", {0, 65537, 7}, 0},
        {"all-equal b=4", {235000, 65536, 4097, 513}, 2},
        {"-inf dominated b=4", {235000, 131072, 65536, 5000}, 5},
        {"give-up trap b=4", {235000, 131072, 65536, 235000}, 4},
        {"odd b=1 100003", {100003}, 0},
        {"odd b=1 100003 ties", {100003}, 1},
        {"b=8 235000 random", {235000, 235000, 235000, 235000, 235000, 235000, 235000, 235000}, 0},
        {"NaN sprinkle b=4 (report only)", {235000, 65536, 4097, 511}, 6},
    };
    Lcg rng(0x5EED2026u);
    int fails = 0;
    for (auto& cs : cases) {
        const unsigned b = (unsigned)cs.ns.size();
        unsigned n_max = 0; for (unsigned n : cs.ns) n_max = std::max(n_max, n);
        std::vector<float> h((size_t)b * STRIDE, -INFINITY);
        for (unsigned r = 0; r < b; r++) gen_row(&h[(size_t)r * STRIDE], cs.ns[r], cs.kind, rng);
        KB_CHECK(hipMemcpy(d_scores, h.data(), h.size() * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_n, cs.ns.data(), b * 4, hipMemcpyHostToDevice));
        // 1. select ALONE: done[] and selected[] of done rows must agree
        for (Chain* c : {&A, &C}) { KB_CHECK(hipMemset(c->selected, 0xEE, (size_t)b * TOP_K * 4)); KB_CHECK(hipMemset(c->done, 0xEE, b * 4)); }
        launch_chain(A, d_scores, d_n, n_max, b, true);
        launch_chain(C, d_scores, d_n, n_max, b, true);
        KB_CHECK(hipDeviceSynchronize());
        auto selA = kb::d2h(A.selected, (size_t)b * TOP_K), selC = kb::d2h(C.selected, (size_t)b * TOP_K);
        auto doneA = kb::d2h(A.done, b), doneC = kb::d2h(C.done, b);
        size_t done_diff = 0, sel_diff_alone = 0; unsigned giveups = 0;
        for (unsigned r = 0; r < b; r++) {
            done_diff += doneA[r] != doneC[r];
            giveups += doneA[r] == 0u;
            if (doneA[r] == 1u && doneC[r] == 1u)
                for (unsigned i = 0; i < TOP_K; i++) sel_diff_alone += selA[(size_t)r * TOP_K + i] != selC[(size_t)r * TOP_K + i];
        }
        // 2. select + production chain (early-outs on done): final selection must agree, and match the CPU order
        for (Chain* c : {&A, &C}) { KB_CHECK(hipMemset(c->selected, 0xEE, (size_t)b * TOP_K * 4)); }
        launch_chain(A, d_scores, d_n, n_max, b, false);
        launch_chain(C, d_scores, d_n, n_max, b, false);
        KB_CHECK(hipDeviceSynchronize());
        selA = kb::d2h(A.selected, (size_t)b * TOP_K); selC = kb::d2h(C.selected, (size_t)b * TOP_K);
        size_t sel_diff_chain = 0, cpu_diff = 0;
        for (unsigned r = 0; r < b; r++) {
            for (unsigned i = 0; i < TOP_K; i++) sel_diff_chain += selA[(size_t)r * TOP_K + i] != selC[(size_t)r * TOP_K + i];
            if (cs.kind != 6) {
                auto ref = cpu_ref(&h[(size_t)r * STRIDE], cs.ns[r]);
                for (unsigned i = 0; i < TOP_K; i++) cpu_diff += ref[i] != selA[(size_t)r * TOP_K + i];
            }
        }
        const bool report_only = cs.kind == 6;
        const bool ok = done_diff == 0 && sel_diff_alone == 0 && sel_diff_chain == 0 && cpu_diff == 0;
        if (!ok && !report_only) fails++;
        printf("CASE %-34s b=%u n_max=%-7u giveup_rows=%u  done_diff=%zu sel_diff_alone=%zu sel_diff_chain=%zu base_vs_cpu_diff=%zu  -> %s\n",
               cs.name, b, n_max, giveups, done_diff, sel_diff_alone, sel_diff_chain, cpu_diff, ok ? "PASS" : (report_only ? "DIFF (report only)" : "FAIL"));
        printf("KBJSON {\"case\":\"%s\",\"b\":%u,\"giveups\":%u,\"done_diff\":%zu,\"sel_diff_alone\":%zu,\"sel_diff_chain\":%zu,\"cpu_diff\":%zu,\"ok\":%d}\n",
               cs.name, b, giveups, done_diff, sel_diff_alone, sel_diff_chain, cpu_diff, (int)ok);
    }
    printf("EXTRA_RESULT fails=%d\n", fails);
    KB_CHECK(hipDeviceSynchronize());
    return fails ? 1 : 0;
}
