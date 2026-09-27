// xover_f: crossover of the F_mhc_glue knobs over the batch size, old vs new symbol from the SAME
// in-tree code object (rms_norm.hip / f16_matvec.hip / router_topk_par.hip built with $KFLAGS_V41),
// launched as the Rust wrappers do. One process, every b; graph mode.
//   ./xover_f <section> b1 b2 ...     sections: rms router topk
#include "kbench.h"
#include <cstring>
#include <string>

static const unsigned NE = 5120, NEXP = 384, NUSED = 6;
static const float RMS_EPS = 1.0e-6f, EW_SCALE = 1.5f, W_EPS = 6.103515625e-5f;
static int ROUNDS = 40;

static kb::AbOpts opts(const std::string& tag, int inner) {
    kb::AbOpts o; o.graph = true; o.inner = inner; o.rounds = ROUNDS; o.warm_ms = 60;
    static std::string keep; keep = tag; o.tag = keep.c_str();
    return o;
}

static void sec_rms(const std::vector<unsigned>& bs) {
    kb::Module m("rms_norm_gfx1201.hsaco");
    hipFunction_t f_old = m.fn("rms_norm_weighted_batched"), f_new = m.fn("rms_norm_weighted_batched_fast");
    hipFunction_t f_old1 = m.fn("rms_norm_weighted");
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    float* x = kb::dalloc<float>((size_t)bmax * NE); kb::fill_f32(x, (size_t)bmax * NE, 1, -2.f, 2.f);
    float* w = kb::dalloc<float>(NE); kb::fill_f32(w, NE, 2, 0.2f, 1.8f);
    float* o0 = kb::dalloc<float>((size_t)bmax * NE); float* o1 = kb::dalloc<float>((size_t)bmax * NE);
    for (unsigned b : bs) {
        std::vector<kb::Variant> vs;
        for (unsigned n : {5120u, 1280u, 512u}) {
            kb::launch(f_old, dim3(b), dim3(256), 0, 0, o0, (const float*)x, (const float*)w, n, RMS_EPS);
            kb::launch(f_new, dim3(b), dim3(256), 0, 0, o1, (const float*)x, (const float*)w, n, RMS_EPS);
            KB_CHECK(hipDeviceSynchronize());
            auto a = kb::d2h(o0, (size_t)b * n), c = kb::d2h(o1, (size_t)b * n);
            size_t d = 0; for (size_t i = 0; i < a.size(); ++i) d += memcmp(&a[i], &c[i], 4) != 0;
            printf("CMP rms n=%u b=%u bit_diff=%zu\n", n, b, d);
            vs.push_back({"old n=" + std::to_string(n), [=](hipStream_t s) { kb::launch(f_old, dim3(b), dim3(256), 0, s, o0, (const float*)x, (const float*)w, n, RMS_EPS); }});
            vs.push_back({"fast n=" + std::to_string(n), [=](hipStream_t s) { kb::launch(f_new, dim3(b), dim3(256), 0, s, o1, (const float*)x, (const float*)w, n, RMS_EPS); }});
        }
        if (b == 1) {
            vs.push_back({"old head rms_norm_weighted (1)", [=](hipStream_t s) { kb::launch(f_old1, dim3(1), dim3(256), 0, s, o0, (const float*)x, (const float*)w, NE, RMS_EPS); }});
            vs.push_back({"fast head (1)", [=](hipStream_t s) { kb::launch(f_new, dim3(1), dim3(256), 0, s, o1, (const float*)x, (const float*)w, NE, RMS_EPS); }});
        }
        kb::ab(vs, opts("rms b=" + std::to_string(b) + " warm graph", 10));
    }
}

static void sec_router(const std::vector<unsigned>& bs) {
    kb::Module m("f16_matvec_gfx1201.hsaco");
    hipFunction_t f_old = m.fn("f16_matvec_batched"), f_new = m.fn("f16_matvec_batched_h20");
    const size_t wel = (size_t)NEXP * NE;
    const int copies = 21;  // 21 x 3.9 MB = 83 MB > MALL: every call reads its copy cold
    std::vector<uint16_t*> W;
    for (int c = 0; c < copies; ++c) { W.push_back(kb::dalloc<uint16_t>(wel)); kb::fill_f16(W.back(), wel, 10 + c, -0.05f, 0.05f); }
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    float* x = kb::dalloc<float>((size_t)bmax * NE); kb::fill_f32(x, (size_t)bmax * NE, 3);
    float* o0 = kb::dalloc<float>((size_t)bmax * NEXP); float* o1 = kb::dalloc<float>((size_t)bmax * NEXP);
    for (unsigned b : bs) {
        kb::launch(f_old, dim3(NEXP / 8, 1, b), dim3(256), 0, 0, o0, (const uint16_t*)W[0], (const float*)x, NE, NEXP);
        kb::launch(f_new, dim3(NEXP / 8, 1, b), dim3(256), 0, 0, o1, (const uint16_t*)W[0], (const float*)x, NE, NEXP);
        KB_CHECK(hipDeviceSynchronize());
        auto a = kb::d2h(o0, (size_t)b * NEXP), c = kb::d2h(o1, (size_t)b * NEXP);
        size_t d = 0; for (size_t i = 0; i < a.size(); ++i) d += memcmp(&a[i], &c[i], 4) != 0;
        printf("CMP router b=%u bit_diff=%zu\n", b, d);
        int* r0 = new int(0); int* r1 = new int(0);
        std::vector<kb::Variant> vs = {
            {"old f16_matvec_batched (48,1,b)", [=](hipStream_t s) { kb::launch(f_old, dim3(NEXP / 8, 1, b), dim3(256), 0, s, o0, (const uint16_t*)W[(*r0)++ % copies], (const float*)x, NE, NEXP); }},
            {"h20", [=](hipStream_t s) { kb::launch(f_new, dim3(NEXP / 8, 1, b), dim3(256), 0, s, o1, (const uint16_t*)W[(*r1)++ % copies], (const float*)x, NE, NEXP); }},
        };
        kb::ab(vs, opts("router 384x5120 b=" + std::to_string(b) + " COLD W (21 copies) graph", copies));
    }
}

static void sec_topk(const std::vector<unsigned>& bs, bool use_prior) {
    kb::Module m("router_topk_par_gfx1201.hsaco");
    hipFunction_t f_old = m.fn("router_topk_par"), f_new = m.fn("router_topk_wfred");
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    float* logits = kb::dalloc<float>((size_t)bmax * NEXP); kb::fill_f32(logits, (size_t)bmax * NEXP, 23, -3.f, 3.f);
    float* bias = kb::dalloc<float>(NEXP); kb::fill_f32(bias, NEXP, 24, -0.1f, 0.1f);
    float* prior = kb::dalloc<float>(NEXP); kb::fill_f32(prior, NEXP, 25, 0.f, 1.f);
    struct O { int* sel; float* ew; int* orig; float* range; };
    auto mk = [&]() { return O{kb::dalloc<int>((size_t)bmax * NUSED), kb::dalloc<float>((size_t)bmax * NUSED), kb::dalloc<int>((size_t)bmax * NUSED), kb::dalloc<float>(bmax)}; };
    O a = mk(), c = mk();
    auto L = [=](hipFunction_t f, hipStream_t s, unsigned b, O o) {
        void* nul = nullptr;
        kb::launch(f, dim3(b), dim3(512), 0, s, o.sel, o.ew, (const float*)logits, (const float*)bias, NEXP, NUSED,
                   EW_SCALE, W_EPS, nul, 0u, nul, use_prior ? (const void*)prior : nul, use_prior ? 2u : 0u, 0u,
                   use_prior ? (void*)o.orig : nul, use_prior ? (void*)o.range : nul);
    };
    const std::string cfg = use_prior ? "prior" : "plain";
    for (unsigned b : bs) {
        L(f_old, 0, b, a); L(f_new, 0, b, c); KB_CHECK(hipDeviceSynchronize());
        auto s0 = kb::d2h(a.sel, (size_t)b * NUSED), s1 = kb::d2h(c.sel, (size_t)b * NUSED);
        auto w0 = kb::d2h(a.ew, (size_t)b * NUSED), w1 = kb::d2h(c.ew, (size_t)b * NUSED);
        size_t d = 0; for (size_t i = 0; i < s0.size(); ++i) d += (s0[i] != s1[i]) + (memcmp(&w0[i], &w1[i], 4) != 0);
        printf("CMP topk b=%u diff=%zu\n", b, d);
        std::vector<kb::Variant> vs = {
            {"old router_topk_par " + cfg + " (b)x512", [=](hipStream_t s) { L(f_old, s, b, a); }},
            {"wfred " + cfg + " (b)x512", [=](hipStream_t s) { L(f_new, s, b, c); }},
        };
        kb::ab(vs, opts("topk " + cfg + " b=" + std::to_string(b) + " warm graph", 10));
    }
}

int main(int argc, char** argv) {
    kb::init();
    if (getenv("XO_ROUNDS")) ROUNDS = atoi(getenv("XO_ROUNDS"));
    std::string sec = argc > 1 ? argv[1] : "rms";
    std::vector<unsigned> bs;
    for (int i = 2; i < argc; ++i) bs.push_back((unsigned)atoi(argv[i]));
    if (sec == "rms") sec_rms(bs);
    else if (sec == "router") sec_router(bs);
    else if (sec == "topk") sec_topk(bs, true);
    else if (sec == "topkp") sec_topk(bs, false);
    else { fprintf(stderr, "unknown section\n"); return 64; }
    return 0;
}
