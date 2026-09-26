// Reviewer harness for E_indexer/candidate_threshold_ilp.
//   ./review_harness_gfx1201 [dir] [rounds=N] [shapes=n:b,n:b,...] [nolarge]
// Launches the PRODUCTION candidate_threshold exactly as S/candidate_blocks.rs does
// (grid (b), block 256, args (threshold, block_score, n_per, nb_stride, 8, 2048)) and the
// candidate with block 1024 (its launch knob), on synthesized block_score rows of several
// distributions, with per-row DIFFERENT n_per (production lanes hold different streams),
// against a CPU reference (topk-th largest order-preserving key), and times both in the
// engineer's regime (graph inner=5) AND the production launch regime (direct, inner=1).
#include "kbench.h"
#include <algorithm>
#include <random>
#include <memory>

static const unsigned STRIDE = 369152;          // ATTN_MIXED_MAX_KEYS
static const unsigned CB_SIZE = 8, CB_TOPK = 2048;

static unsigned cdiv(unsigned a, unsigned b) { return (a + b - 1) / b; }
static unsigned cb_key(float f) { unsigned u; memcpy(&u, &f, 4); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }

// CPU reference: threshold = key of the min(topk, n_blocks)-th largest block key, or 0 if n_blocks <= topk.
static unsigned ref_threshold(const float* bs, unsigned n) {
    unsigned nb = cdiv(n, CB_SIZE);
    if (nb <= CB_TOPK) return 0u;
    std::vector<unsigned> k(nb);
    for (unsigned i = 0; i < nb; i++) k[i] = cb_key(bs[i]);
    std::nth_element(k.begin(), k.begin() + (CB_TOPK - 1), k.end(), std::greater<unsigned>());
    return k[CB_TOPK - 1];
}

// Fill one row of block scores of n positions in distribution `dist`, pinned like candidate_block_max.
static void fill_row(float* bs, unsigned n, int dist, std::mt19937& rng) {
    unsigned nb = cdiv(n, CB_SIZE);
    std::normal_distribution<float> N(0.f, 4.f);
    std::uniform_int_distribution<int> U(0, 3);
    for (unsigned i = 0; i < nb; i++) {
        float v;
        switch (dist) {
            case 0: v = N(rng); break;                                   // score-like
            case 1: v = rintf(N(rng) * 2.f) * 0.5f; break;               // tie-heavy (quantised)
            case 2: v = 1.25f; break;                                    // all-equal
            case 3: v = (i < nb / 2) ? -INFINITY : N(rng); break;        // half unreachable (-inf)
            case 4: v = (float)U(rng) - 1.5f; break;                     // 4 distinct values only
            case 5: v = -N(rng) * 1e30f; break;                          // huge magnitudes, negative-heavy
            default: v = N(rng);
        }
        bs[i] = v;
    }
    if (n > 0) bs[(n - 1) / CB_SIZE] = INFINITY;   // newest block pinned
}

int main(int argc, char** argv) {
    std::string dir = ".";
    int rounds = 50;
    bool large = true, extra = false;
    std::vector<std::pair<unsigned, unsigned>> shapes = {
        {16384, 1}, {16385, 1}, {16392, 4}, {17000, 8}, {65536, 1}, {65536, 4}, {65536, 8},
        {100003, 1}, {100003, 4}, {131072, 8}, {235000, 1}, {235000, 4}, {235000, 8}, {368640, 8}};
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a.rfind("rounds=", 0) == 0) rounds = atoi(a.c_str() + 7);
        else if (a == "nolarge") large = false;
        else if (a == "extra") extra = true;
        else if (a.rfind("shapes=", 0) == 0) {
            shapes.clear();
            std::string s = a.substr(7); size_t p = 0;
            while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
                std::string t = s.substr(p, q - p); size_t c = t.find(':');
                shapes.push_back({(unsigned)atoi(t.substr(0, c).c_str()), (unsigned)atoi(t.substr(c + 1).c_str())}); p = q + 1; }
        } else dir = a;
    }
    if (large) { shapes.push_back({65536, 512}); shapes.push_back({235000, 512}); shapes.push_back({235000, 1024}); }
    kb::init();
    kb::Module base(dir + "/base_candidate_blocks_gfx1201.hsaco");
    kb::Module cand(dir + "/cand_cand_gfx1201.hsaco");
    hipFunction_t f_base = base.fn("candidate_threshold");
    hipFunction_t f_cand = cand.fn("candidate_threshold_ilp");
    struct Ex { std::string name; hipFunction_t f; unsigned blk; };
    std::vector<Ex> exs;
    std::unique_ptr<kb::Module> probe;
    if (extra) {
        exs.push_back({"eng ilp512 (512)", cand.fn("candidate_threshold_ilp512"), 512});
        probe.reset(new kb::Module(dir + "/cand_review_gfx1201.hsaco"));
        exs.push_back({"probe ilp256 U8 (256)", probe->fn("candidate_threshold_ilp256"), 256});
        exs.push_back({"probe ilp256 U4 (256)", probe->fn("candidate_threshold_ilp256_u4"), 256});
    }

    unsigned b_max = 1;
    for (auto& s : shapes) b_max = std::max(b_max, s.second);
    const unsigned nb_stride = cdiv(STRIDE, CB_SIZE);   // 46144, as production
    float* d_bs = kb::dalloc<float>((size_t)b_max * nb_stride);
    unsigned* d_n = kb::dalloc<unsigned>(b_max);
    unsigned* d_t0 = kb::dalloc<unsigned>(b_max);
    unsigned* d_t1 = kb::dalloc<unsigned>(b_max);
    std::vector<float> h_bs((size_t)b_max * nb_stride);
    std::vector<unsigned> h_n(b_max);
    std::mt19937 rng(12345);

    size_t fails = 0, checks = 0;
    for (auto& sh : shapes) {
        const unsigned n = sh.first, b = sh.second;
        for (int dist = 0; dist < 6; dist++) {
            // per-row different n (rows of a lane are different streams): row r gets n - r*1237, floored
            for (unsigned r = 0; r < b; r++) {
                unsigned nr = (dist == 0 && r > 0) ? std::max(1u, n - (r * 1237u) % (n / 2 + 1)) : n;
                h_n[r] = nr;
                fill_row(&h_bs[(size_t)r * nb_stride], nr, dist, rng);
            }
            KB_CHECK(hipMemcpy(d_bs, h_bs.data(), (size_t)b * nb_stride * 4, hipMemcpyHostToDevice));
            KB_CHECK(hipMemcpy(d_n, h_n.data(), b * 4, hipMemcpyHostToDevice));
            KB_CHECK(hipMemset(d_t0, 0xEE, b * 4)); KB_CHECK(hipMemset(d_t1, 0xEE, b * 4));
            kb::launch(f_base, dim3(b), dim3(256), 0, 0, d_t0, (const float*)d_bs, (const unsigned*)d_n, nb_stride, CB_SIZE, CB_TOPK);
            kb::launch(f_cand, dim3(b), dim3(1024), 0, 0, d_t1, (const float*)d_bs, (const unsigned*)d_n, nb_stride, CB_SIZE, CB_TOPK);
            KB_CHECK(hipDeviceSynchronize());
            auto t0 = kb::d2h(d_t0, b), t1 = kb::d2h(d_t1, b);
            for (auto& e : exs) {
                KB_CHECK(hipMemset(d_t1, 0xEE, b * 4));
                kb::launch(e.f, dim3(b), dim3(e.blk), 0, 0, d_t1, (const float*)d_bs, (const unsigned*)d_n, nb_stride, CB_SIZE, CB_TOPK);
                KB_CHECK(hipDeviceSynchronize());
                auto te = kb::d2h(d_t1, b); size_t d = 0; for (unsigned r = 0; r < b; r++) d += te[r] != t0[r];
                printf("CMPX %s n=%u b=%u dist=%d diff=%zu %s\n", e.name.c_str(), n, b, dist, d, d ? "MISMATCH" : "OK");
                fails += d;
            }
            size_t dbc = 0, dbr = 0, dcr = 0;
            for (unsigned r = 0; r < b; r++) {
                unsigned ref = ref_threshold(&h_bs[(size_t)r * nb_stride], h_n[r]);
                dbc += t0[r] != t1[r]; dbr += t0[r] != ref; dcr += t1[r] != ref;
            }
            checks += b; fails += dbc;
            printf("CMP n=%u b=%u dist=%d  cand_vs_base_diff=%zu  base_vs_ref_diff=%zu  cand_vs_ref_diff=%zu  thr0=%08x/%08x  %s\n",
                   n, b, dist, dbc, dbr, dcr, t0[0], t1[0], (dbc == 0 && dcr == 0) ? "OK" : "MISMATCH");
        }
        // timing on dist 0 (score-like, per-row n) — re-fill dist 0
        for (unsigned r = 0; r < b; r++) { h_n[r] = n; fill_row(&h_bs[(size_t)r * nb_stride], n, 0, rng); }
        KB_CHECK(hipMemcpy(d_bs, h_bs.data(), (size_t)b * nb_stride * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_n, h_n.data(), b * 4, hipMemcpyHostToDevice));
        std::vector<kb::Variant> vs;
        vs.push_back({"base candidate_threshold (256)", [&](hipStream_t s) { kb::launch(f_base, dim3(b), dim3(256), 0, s, d_t0, (const float*)d_bs, (const unsigned*)d_n, nb_stride, CB_SIZE, CB_TOPK); }, (double)b * cdiv(n, CB_SIZE) * 16});
        vs.push_back({"cand candidate_threshold_ilp (1024)", [&](hipStream_t s) { kb::launch(f_cand, dim3(b), dim3(1024), 0, s, d_t1, (const float*)d_bs, (const unsigned*)d_n, nb_stride, CB_SIZE, CB_TOPK); }, (double)b * cdiv(n, CB_SIZE) * 16});
        for (auto& e : exs) vs.push_back({e.name, [&, e](hipStream_t s) { kb::launch(e.f, dim3(b), dim3(e.blk), 0, s, d_t1, (const float*)d_bs, (const unsigned*)d_n, nb_stride, CB_SIZE, CB_TOPK); }, (double)b * cdiv(n, CB_SIZE) * 16});
        char tag[160];
        { kb::AbOpts ab; ab.rounds = rounds; ab.inner = 5; ab.graph = true;
          snprintf(tag, sizeof tag, "review threshold n=%u b=%u graph inner=5", n, b); ab.tag = tag; kb::ab(vs, ab); }
        { kb::AbOpts ab; ab.rounds = rounds; ab.inner = 1; ab.graph = false;
          snprintf(tag, sizeof tag, "review threshold n=%u b=%u DIRECT inner=1", n, b); ab.tag = tag; kb::ab(vs, ab); }
    }
    printf("SUMMARY correctness: %zu row-checks, %zu cand-vs-base mismatches\n", checks, fails);
    return fails ? 1 : 0;
}
