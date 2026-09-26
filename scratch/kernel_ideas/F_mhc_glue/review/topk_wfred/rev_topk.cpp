// Reviewer harness for the F_mhc_glue/topk_wfred claim (router_topk_par vs router_topk_wfred).
// Independent of the engineer's harness.cpp: same production launch geometry (grid (b) x 512,
// production argument set), more shapes / input classes for correctness, graph AND direct timing.
//   ./rev_topk corr          correctness sweep (all configs / input classes) -> summary lines
//   ./rev_topk time          timing: prior b=1/4/8 graph+direct, plain b=1 graph, plain b=512 direct
#include "../../../_infra/kbench.h"
#include <cstring>
#include <memory>
#include <string>

static const float EW_SCALE = 1.5f, W_EPS = 6.103515625e-5f;

struct Cfg {
    const char* name;
    unsigned b, n_expert, n_used, n_alt, n_protect, prior_dry;
    bool prior, orig, range, alt_w;
};

struct Out {
    int *sel, *orig, *alts; float *w, *range, *alt_w;
    Out(unsigned b, unsigned nu, unsigned na) {
        sel = kb::dalloc<int>((size_t)b * nu); orig = kb::dalloc<int>((size_t)b * nu);
        alts = kb::dalloc<int>((size_t)b * (na ? na : 1)); w = kb::dalloc<float>((size_t)b * nu);
        range = kb::dalloc<float>(b); alt_w = kb::dalloc<float>((size_t)b * (na ? na : 1));
    }
    void poison(unsigned b, unsigned nu, unsigned na) {
        KB_CHECK(hipMemset(sel, 0xEE, (size_t)b * nu * 4)); KB_CHECK(hipMemset(orig, 0xEE, (size_t)b * nu * 4));
        KB_CHECK(hipMemset(alts, 0xEE, (size_t)b * (na ? na : 1) * 4)); KB_CHECK(hipMemset(w, 0xEE, (size_t)b * nu * 4));
        KB_CHECK(hipMemset(range, 0xEE, (size_t)b * 4)); KB_CHECK(hipMemset(alt_w, 0xEE, (size_t)b * (na ? na : 1) * 4));
    }
};

static void launch_topk(hipFunction_t f, hipStream_t s, const Cfg& c, Out& o, const float* logits, const float* bias,
                        const float* prior) {
    void* nul = nullptr;
    kb::launch(f, dim3(c.b), dim3(512), 0, s, (void*)o.sel, (void*)o.w, (const void*)logits, (const void*)bias, c.n_expert,
               c.n_used, EW_SCALE, W_EPS, c.n_alt ? (void*)o.alts : nul, c.n_alt, (c.n_alt && c.alt_w) ? (void*)o.alt_w : nul,
               c.prior ? (const void*)prior : (const void*)nul, c.n_protect, c.prior_dry, c.orig ? (void*)o.orig : nul,
               c.range ? (void*)o.range : nul);
}

static void fill_inputs(int cls, unsigned b, unsigned ne, float* logits, float* bias, float* prior, uint32_t seed) {
    std::vector<float> L((size_t)b * ne), B(ne), P(ne);
    auto rnd = [&](size_t i, uint32_t s) {
        uint32_t x = (uint32_t)i * 2654435761u ^ s;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        return (x >> 8) * (1.0f / 16777216.0f);
    };
    for (size_t i = 0; i < B.size(); ++i) B[i] = -0.1f + 0.2f * rnd(i, seed + 7);
    for (size_t i = 0; i < P.size(); ++i) P[i] = ((i * 2654435761u >> 7) & 1) ? 0.05f : 0.f;
    switch (cls) {
    case 0: case 1:  // random logits U(-3,3)
        for (size_t i = 0; i < L.size(); ++i) L[i] = -3.f + 6.f * rnd(i, seed + cls);
        break;
    case 2:  // tie-heavy: 7-level grid, zero bias (prior stays half 0.05)
        for (size_t i = 0; i < L.size(); ++i) L[i] = (float)((int)((i * 2654435761u >> 9) % 7) - 3);
        std::fill(B.begin(), B.end(), 0.f);
        break;
    case 3:  // ALL EQUAL: every score identical, no bias, no prior -> pure index tie-break
        std::fill(L.begin(), L.end(), 0.25f); std::fill(B.begin(), B.end(), 0.f); std::fill(P.begin(), P.end(), 0.f);
        break;
    case 4:  // all logits -INF (p = 0): selection purely by bias (+prior)
        std::fill(L.begin(), L.end(), -INFINITY);
        break;
    case 5:  // extreme logits through every softplus branch, incl. exact zeros of p
        { const float vals[8] = {-1000.f, -100.f, -25.f, -10.f, 0.f, 10.f, 25.f, 100.f};
          for (size_t i = 0; i < L.size(); ++i) L[i] = vals[(i * 2654435761u >> 11) & 7]; }
        break;
    case 6:  // -0.0 everywhere it can be injected: logits -0.0, bias -0.0 on p==0 experts, prior -0.0
        for (size_t i = 0; i < L.size(); ++i) L[i] = ((i * 2654435761u >> 9) % 5 == 0) ? -0.0f : (((i * 2654435761u >> 9) % 5 == 1) ? -1000.f : -3.f + 6.f * rnd(i, seed));
        for (size_t i = 0; i < B.size(); ++i) B[i] = ((i * 2654435761u >> 9) % 5 == 1) ? -0.0f : B[i];
        for (size_t i = 0; i < P.size(); ++i) P[i] = P[i] == 0.f ? -0.0f : P[i];
        break;
    case 7:  // large bias range so bias-driven ranking != prob ranking; big prior boosts too
        for (size_t i = 0; i < L.size(); ++i) L[i] = -3.f + 6.f * rnd(i, seed + 3);
        for (size_t i = 0; i < B.size(); ++i) B[i] = -2.f + 4.f * rnd(i, seed + 9);
        for (size_t i = 0; i < P.size(); ++i) P[i] = P[i] > 0.f ? 1.5f : 0.f;
        break;
    case 8:  // near-tie: values that differ only in the last ulp, plus exact duplicates across rows
        for (size_t i = 0; i < L.size(); ++i) { float v = 1.0f + (float)((i * 2654435761u >> 9) % 3) * 1.1920929e-07f; L[i] = v; }
        std::fill(B.begin(), B.end(), 0.f);
        break;
    }
    KB_CHECK(hipMemcpy(logits, L.data(), L.size() * 4, hipMemcpyHostToDevice));
    KB_CHECK(hipMemcpy(bias, B.data(), B.size() * 4, hipMemcpyHostToDevice));
    KB_CHECK(hipMemcpy(prior, P.data(), P.size() * 4, hipMemcpyHostToDevice));
}

static size_t bitdiff(const std::vector<float>& a, const std::vector<float>& b) {
    size_t d = 0; for (size_t i = 0; i < a.size(); ++i) d += memcmp(&a[i], &b[i], 4) != 0; return d;
}
static size_t intdiff(const std::vector<int>& a, const std::vector<int>& b) {
    size_t d = 0; for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i]; return d;
}

int main(int argc, char** argv) {
    kb::init();
    const std::string dir = getenv("REV_DIR") ? getenv("REV_DIR") : ".";
    kb::Module mb(dir + "/base_router_topk_par_gfx1201.hsaco"), mc(dir + "/cand_topk2_gfx1201.hsaco");
    hipFunction_t fb = mb.fn("router_topk_par"), fc = mc.fn("router_topk_wfred");
    std::string mode = argc > 1 ? argv[1] : "corr";

    if (mode == "corr") {
        //            name                    b   ne   nu na np dry  prior  orig  range alt_w
        const Cfg cfgs[] = {
            {"prod b=1",                      1, 384, 6, 0, 2, 0,  true,  true, true, false},
            {"prod b=3 (odd)",                3, 384, 6, 0, 2, 0,  true,  true, true, false},
            {"prod b=4",                      4, 384, 6, 0, 2, 0,  true,  true, true, false},
            {"prod b=5 (odd)",                5, 384, 6, 0, 2, 0,  true,  true, true, false},
            {"prod b=8 (max decode)",         8, 384, 6, 0, 2, 0,  true,  true, true, false},
            {"prod b=512 (prefill w/ prior)", 512, 384, 6, 0, 2, 0, true, true, true, false},
            {"plain b=1 (prefill form)",      1, 384, 6, 0, 0, 0,  false, false, false, false},
            {"plain b=7",                     7, 384, 6, 0, 0, 0,  false, false, false, false},
            {"plain b=1024",                  1024, 384, 6, 0, 0, 0, false, false, false, false},
            {"plain + orig + range b=4",      4, 384, 6, 0, 0, 0,  false, true, true, false},
            {"plain + alts=4 + alt_w b=4",    4, 384, 6, 4, 0, 0,  false, true, true, true},
            {"prior + alts=2 + alt_w b=4",    4, 384, 6, 2, 2, 0,  true,  true, true, true},
            {"prior + alts=2 no alt_w b=3",   3, 384, 6, 2, 2, 0,  true,  true, true, false},
            {"prior dry b=4",                 4, 384, 6, 0, 2, 1,  true,  true, true, false},
            {"prior dry + alts=1 b=2",        2, 384, 6, 1, 2, 1,  true,  true, true, true},
            {"prior n_protect=0 b=4",         4, 384, 6, 0, 0, 0,  true,  true, true, false},
            {"prior n_protect=6 b=4",         4, 384, 6, 0, 6, 0,  true,  true, true, false},
            {"prior n_protect=9>n_used b=2",  2, 384, 6, 0, 9, 0,  true,  true, true, false},
            {"prior no orig/range b=4",       4, 384, 6, 0, 2, 0,  true,  false, false, false},
            {"prior n_used=8 b=2",            2, 384, 8, 0, 2, 0,  true,  true, true, false},
            {"prior n_used=1 b=2",            2, 384, 1, 0, 2, 0,  true,  true, true, false},
            {"prior n_used=8 alts=4 b=2",     2, 384, 8, 4, 2, 0,  true,  true, true, true},
            {"prior n_expert=512 (full) b=2", 2, 512, 6, 0, 2, 0,  true,  true, true, false},
            {"prior n_expert=256 b=2",        2, 256, 6, 0, 2, 0,  true,  true, true, false},
            {"prior n_expert=100 (odd) b=3",  3, 100, 6, 0, 2, 0,  true,  true, true, false},
            {"prior n_expert=33 b=2",         2, 33, 6, 0, 2, 0,   true,  true, true, false},
            {"plain n_expert=7 n_used=6 alts=1 b=2", 2, 7, 6, 1, 0, 0, false, true, true, true},
        };
        const char* cls_name[] = {"rand0", "rand1", "ties7", "allequal", "allneginf", "extreme", "negzero", "bigbias", "ulp-ties"};
        size_t total = 0, bad = 0;
        for (const Cfg& c : cfgs) {
            float* logits = kb::dalloc<float>((size_t)c.b * c.n_expert);
            float* bias = kb::dalloc<float>(c.n_expert);
            float* prior = kb::dalloc<float>(c.n_expert);
            Out ob(c.b, c.n_used, c.n_alt), oc(c.b, c.n_used, c.n_alt);
            for (int cls = 0; cls < 9; ++cls) {
                fill_inputs(cls, c.b, c.n_expert, logits, bias, prior, 1000 + cls);
                ob.poison(c.b, c.n_used, c.n_alt); oc.poison(c.b, c.n_used, c.n_alt);
                launch_topk(fb, 0, c, ob, logits, bias, prior);
                launch_topk(fc, 0, c, oc, logits, bias, prior);
                KB_CHECK(hipDeviceSynchronize());
                size_t nsel = (size_t)c.b * c.n_used, nalt = (size_t)c.b * (c.n_alt ? c.n_alt : 1);
                size_t d_sel = intdiff(kb::d2h(ob.sel, nsel), kb::d2h(oc.sel, nsel));
                size_t d_w = bitdiff(kb::d2h(ob.w, nsel), kb::d2h(oc.w, nsel));
                size_t d_orig = intdiff(kb::d2h(ob.orig, nsel), kb::d2h(oc.orig, nsel));      // poison-equal when unused
                size_t d_rng = bitdiff(kb::d2h(ob.range, c.b), kb::d2h(oc.range, c.b));
                size_t d_alts = intdiff(kb::d2h(ob.alts, nalt), kb::d2h(oc.alts, nalt));
                size_t d_altw = bitdiff(kb::d2h(ob.alt_w, nalt), kb::d2h(oc.alt_w, nalt));
                size_t d = d_sel + d_w + d_orig + d_rng + d_alts + d_altw;
                total++; bad += d != 0;
                auto s0 = kb::d2h(ob.sel, nsel); auto w0 = kb::d2h(ob.w, nsel);
                printf("CORR %-36s %-10s %s sel=%zu w=%zu orig=%zu range=%zu alts=%zu alt_w=%zu | base sel[0..]=%d,%d,%d,%d w0=%.6g\n",
                       c.name, cls_name[cls], d == 0 ? "OK  " : "DIFF", d_sel, d_w, d_orig, d_rng, d_alts, d_altw,
                       s0[0], nsel > 1 ? s0[1] : -9, nsel > 2 ? s0[2] : -9, nsel > 3 ? s0[3] : -9, w0[0]);
                if (d_sel) {
                    auto s1 = kb::d2h(oc.sel, nsel);
                    for (size_t j = 0; j < nsel && j < 64; ++j) if (s0[j] != s1[j]) printf("   sel[%zu]: base %d cand %d\n", j, s0[j], s1[j]);
                }
            }
            KB_CHECK(hipFree(logits)); KB_CHECK(hipFree(bias)); KB_CHECK(hipFree(prior));
        }
        printf("CORR SUMMARY: %zu configs x classes, %zu with any difference\n", total, bad);
        return bad ? 1 : 0;
    }

    if (mode == "time") {
        struct T { unsigned b; bool prior; bool graph; };
        const T ts[] = {{1, true, true}, {1, true, false}, {4, true, true}, {4, true, false}, {8, true, true}, {8, true, false},
                        {1, false, true}, {1, false, false}, {512, false, false}};
        for (const T& t : ts) {
            Cfg c = {"", t.b, 384, 6, 0, t.prior ? 2u : 0u, 0, t.prior, t.prior, t.prior, false};
            float* logits = kb::dalloc<float>((size_t)c.b * c.n_expert);
            float* bias = kb::dalloc<float>(c.n_expert);
            float* prior = kb::dalloc<float>(c.n_expert);
            fill_inputs(0, c.b, c.n_expert, logits, bias, prior, 23);
            Out ob(c.b, c.n_used, 0), oc(c.b, c.n_used, 0);
            std::vector<kb::Variant> vs = {
                {std::string("router_topk_par ") + (t.prior ? "prior" : "plain"), [&](hipStream_t s) { launch_topk(fb, s, c, ob, logits, bias, prior); }, 0},
                {std::string("router_topk_wfred ") + (t.prior ? "prior" : "plain"), [&](hipStream_t s) { launch_topk(fc, s, c, oc, logits, bias, prior); }, 0},
            };
            char tag[128];
            snprintf(tag, sizeof tag, "REVIEW router_topk b=%u %s %s warm", t.b, t.prior ? "prior" : "plain", t.graph ? "graph" : "direct");
            kb::AbOpts o; o.graph = t.graph; o.inner = 20; o.rounds = 40; o.tag = tag;
            kb::ab(vs, o);
            KB_CHECK(hipFree(logits)); KB_CHECK(hipFree(bias)); KB_CHECK(hipFree(prior));
        }
        return 0;
    }
    fprintf(stderr, "usage: rev_topk corr|time\n");
    return 2;
}
