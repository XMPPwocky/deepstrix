// F_mhc_glue harness — production kernels (unmodified in-tree .hip -> .hsaco) launched exactly as
// the Rust wrappers do, at production shapes, plus candidates.
//   ./harness_gfx1201 decode  <b> [direct]       decode glue, graph mode (default) or direct launch
//   ./harness_gfx1201 prefill <B> [flush]        prefill mHC chain, direct launch, warm or 40 MB flush
//   ./harness_gfx1201 topk    <b> [prior]        router_topk_par baseline vs candidates (+ correctness)
//   ./harness_gfx1201 gemm    <B> [flush]        prefill pre-mix GEMM baseline vs candidates (+ correctness)
//   ./harness_gfx1201 prof    <what> <b>         short run for rocprofv3 (what = mhc|gemm|topk|chain)
#include "kbench.h"
#include <hip/hip_fp16.h>
#include <memory>

static const unsigned NE = 5120, NHC = 4, HCD = 20480, MIX = 24, NEXP = 384, NUSED = 6;
static const float RMS_EPS = 1.0e-20f, SK_EPS = 1.0e-6f, EW_SCALE = 1.5f, W_EPS = 6.103515625e-5f;
static const unsigned SK_ITERS = 20;

struct Mods {
    kb::Module mhc_fast, gemm, rms_nw, sink, hcw, rms, hcpost, f16mv, topk, vadd, arena, narrow;
    Mods(const std::string& d)
        : mhc_fast(d + "/base_mhc_fast_gfx1201.hsaco"), gemm(d + "/base_f16_gemm_wmma_gfx1201.hsaco"),
          rms_nw(d + "/base_rms_norm_no_weight_gfx1201.hsaco"), sink(d + "/base_hc_sinkhorn_par_gfx1201.hsaco"),
          hcw(d + "/base_hc_weighted_sum_gfx1201.hsaco"), rms(d + "/base_rms_norm_gfx1201.hsaco"),
          hcpost(d + "/base_hc_post_gfx1201.hsaco"), f16mv(d + "/base_f16_matvec_gfx1201.hsaco"),
          topk(d + "/base_router_topk_par_gfx1201.hsaco"), vadd(d + "/base_vec_add_gfx1201.hsaco"),
          arena(d + "/base_mhc_arena_gfx1201.hsaco"), narrow(d + "/base_f16_matvec_narrow_gfx1201.hsaco") {}
};

static bool file_exists(const std::string& p) { FILE* f = fopen(p.c_str(), "rb"); if (f) fclose(f); return f != nullptr; }

// ------------------------------------------------------------------ buffers at [B, ...]
struct Bufs {
    unsigned B;
    float *residual, *flat, *mix, *split, *carry, *cur, *norm, *norm_w, *scale, *base, *inv_rows;
    unsigned* counters;
    void* w_mix;  // [24, HCD] f16
    // router
    void* w_router;  // [384, 5120] f16
    float *x_ne, *logits, *bias, *prior, *ew, *alt_w, *range;
    int *sel, *alts, *orig;
    // hc_post / vec_add
    float *attn_out, *after_attn;
    Bufs(unsigned b, bool prefill) : B(b) {
        residual = kb::dalloc<float>((size_t)B * HCD);
        flat = kb::dalloc<float>((size_t)B * HCD);
        mix = kb::dalloc<float>((size_t)B * MIX);
        split = kb::dalloc<float>((size_t)B * MIX);
        carry = kb::dalloc<float>((size_t)B * MIX);
        cur = kb::dalloc<float>((size_t)B * NE);
        norm = kb::dalloc<float>((size_t)B * NE);
        norm_w = kb::dalloc<float>(NE);
        scale = kb::dalloc<float>(3);
        base = kb::dalloc<float>(MIX);
        inv_rows = kb::dalloc<float>(B);
        counters = kb::dalloc<unsigned>(B);
        w_mix = kb::dalloc<uint16_t>((size_t)MIX * HCD);
        kb::fill_f32(residual, (size_t)B * HCD, 11, -1.f, 1.f);
        kb::fill_f32(carry, (size_t)B * MIX, 12, 0.3f, 1.2f);
        kb::fill_f32(norm_w, NE, 13, 0.5f, 1.5f);
        kb::fill_f32(base, MIX, 14, -0.5f, 0.5f);
        float sc[3] = {1.0f, 1.0f, 1.0f};
        KB_CHECK(hipMemcpy(scale, sc, 12, hipMemcpyHostToDevice));
        KB_CHECK(hipMemset(counters, 0, B * 4));
        kb::fill_f16(w_mix, (size_t)MIX * HCD, 15, -0.02f, 0.02f);
        KB_CHECK(hipMemset(flat, 0, (size_t)B * HCD * 4));
        KB_CHECK(hipMemset(mix, 0, (size_t)B * MIX * 4));
        KB_CHECK(hipMemset(split, 0, (size_t)B * MIX * 4));
        // router side (decode-only buffers; skipped at prefill B to stay under the VRAM cap)
        w_router = nullptr; x_ne = logits = bias = prior = ew = alt_w = range = nullptr;
        sel = alts = orig = nullptr; attn_out = after_attn = nullptr;
        if (prefill) return;
        w_router = kb::dalloc<uint16_t>((size_t)NEXP * NE);
        kb::fill_f16(w_router, (size_t)NEXP * NE, 21, -0.05f, 0.05f);
        x_ne = kb::dalloc<float>((size_t)B * NE);
        kb::fill_f32(x_ne, (size_t)B * NE, 22, -1.f, 1.f);
        logits = kb::dalloc<float>((size_t)B * NEXP);
        kb::fill_f32(logits, (size_t)B * NEXP, 23, -3.f, 3.f);
        bias = kb::dalloc<float>(NEXP);
        kb::fill_f32(bias, NEXP, 24, -0.1f, 0.1f);
        prior = kb::dalloc<float>(NEXP);
        {  // ~half the experts "held": boost 0.05, else 0
            std::vector<float> p(NEXP);
            for (unsigned i = 0; i < NEXP; ++i) p[i] = ((i * 2654435761u >> 7) & 1) ? 0.05f : 0.f;
            KB_CHECK(hipMemcpy(prior, p.data(), NEXP * 4, hipMemcpyHostToDevice));
        }
        ew = kb::dalloc<float>((size_t)B * NUSED);
        alt_w = kb::dalloc<float>((size_t)B * 4);
        range = kb::dalloc<float>(B);
        sel = kb::dalloc<int>((size_t)B * NUSED);
        alts = kb::dalloc<int>((size_t)B * 4);
        orig = kb::dalloc<int>((size_t)B * NUSED);
        attn_out = kb::dalloc<float>((size_t)B * NE);
        kb::fill_f32(attn_out, (size_t)B * NE, 31, -1.f, 1.f);
        after_attn = kb::dalloc<float>((size_t)B * HCD);
    }
};

// ------------------------------------------------------------------ production launches
struct Prod {
    Mods& m;
    hipFunction_t f_mhc, f_gemm, f_rmsnw, f_sink, f_hcw, f_hcw1, f_rmsw, f_rmsw1, f_hcpost, f_mv, f_topk, f_vadd,
        f_ksplit, f_narrow;
    Prod(Mods& mm) : m(mm) {
        f_mhc = m.mhc_fast.fn("mhc_fast_batched");
        f_gemm = m.gemm.fn("f16_gemm_wmma_lds_tiled");
        f_rmsnw = m.rms_nw.fn("rms_norm_no_weight_batched");
        f_sink = m.sink.fn("hc_sinkhorn_par_batched");
        f_hcw = m.hcw.fn("hc_weighted_sum_batched");
        f_hcw1 = m.hcw.fn("hc_weighted_sum");
        f_rmsw = m.rms.fn("rms_norm_weighted_batched");
        f_rmsw1 = m.rms.fn("rms_norm_weighted");
        f_hcpost = m.hcpost.fn("hc_post_from_split_batched");
        f_mv = m.f16mv.fn("f16_matvec_batched");
        f_topk = m.topk.fn("router_topk_par");
        f_vadd = m.vadd.fn("vec_add_inplace");
        f_ksplit = m.arena.fn("mhc_mix_ksplit_batched");
        f_narrow = m.narrow.fn("f16_matvec_narrow_batched");
    }
    // mhc_fast_batched exactly as MhcArena::launch_fast builds it.
    void mhc_fast(hipStream_t s, Bufs& b, unsigned n_mix, unsigned do_collapse, unsigned write_carry, unsigned mode,
                  const float* x, const float* cx, float* cur_out, float* norm_out) {
        unsigned rms_wg = (n_mix > 0 && mode == 0) ? 1u : 0u;
        void* nul = nullptr;
        kb::launch(f_mhc, dim3(n_mix + do_collapse + rms_wg, 1, b.B), dim3(256), 0, s,
                   n_mix ? (void*)b.split : nul, n_mix ? (void*)b.mix : nul, n_mix ? (void*)b.counters : nul,
                   n_mix ? (void*)b.inv_rows : nul, n_mix ? b.w_mix : nul, n_mix ? (const void*)x : nul,
                   n_mix ? (void*)b.scale : nul, n_mix ? (void*)b.base : nul, (void*)b.carry,
                   do_collapse ? (const void*)cx : nul, do_collapse ? (void*)cur_out : nul,
                   do_collapse ? (void*)norm_out : nul, do_collapse ? (void*)b.norm_w : nul,
                   HCD, NE, n_mix, do_collapse, write_carry, mode, RMS_EPS, NHC, SK_ITERS, SK_EPS);
    }
    void gemm(hipStream_t s, float* out, const void* w, const float* x, unsigned K, unsigned M, unsigned B) {
        kb::launch(f_gemm, dim3((M + 63) / 64, (B + 63) / 64), dim3(128), 0, s, out, w, x, K, M, B);
    }
    void rms_nw(hipStream_t s, float* out, const float* x, unsigned B) {
        kb::launch(f_rmsnw, dim3(B, 1), dim3(256), 0, s, out, x, 1u, HCD, RMS_EPS);
    }
    void sinkhorn(hipStream_t s, float* out, const float* mix, Bufs& b, unsigned B) {
        kb::launch(f_sink, dim3(B), dim3(16), 0, s, out, mix, b.scale, b.base, NHC, SK_ITERS, SK_EPS);
    }
    void hcw(hipStream_t s, float* out, const float* x, const float* w, unsigned B) {
        kb::launch(f_hcw, dim3(NE / 256, 1, B), dim3(256), 0, s, out, x, w, NE, NHC, MIX);
    }
    void rmsw(hipStream_t s, float* out, const float* x, const float* w, unsigned B) {
        kb::launch(f_rmsw, dim3(B), dim3(256), 0, s, out, x, w, NE, RMS_EPS);
    }
    void hcpost(hipStream_t s, float* out, const float* blk, const float* res, const float* split, unsigned B) {
        kb::launch(f_hcpost, dim3(NE / 256, NHC, B), dim3(256), 0, s, out, blk, res, split, NHC, NE, NHC);
    }
    void router_mv(hipStream_t s, float* out, const void* w, const float* x, unsigned B) {
        kb::launch(f_mv, dim3(NEXP / 8, 1, B), dim3(256), 0, s, out, w, x, NE, NEXP);
    }
    void topk(hipStream_t s, Bufs& b, unsigned B, bool prior, int* sel, float* ew, int* orig, float* range) {
        void* nul = nullptr;
        kb::launch(f_topk, dim3(B), dim3(512), 0, s, sel, ew, (const float*)b.logits, (const float*)b.bias, NEXP, NUSED,
                   EW_SCALE, W_EPS, nul, 0u, nul, prior ? (const void*)b.prior : nul, prior ? 2u : 0u, 0u,
                   prior ? (void*)orig : nul, prior ? (void*)range : nul);
    }
    void vadd(hipStream_t s, float* out, const float* rhs, unsigned n) {
        kb::launch(f_vadd, dim3((n + 255) / 256), dim3(256), 0, s, out, rhs, n);
    }
    void chain(hipStream_t s, Bufs& b, unsigned B) {  // the prefill mHC sub-block as forward_prefill runs it
        rms_nw(s, b.flat, b.residual, B);
        gemm(s, b.mix, b.w_mix, b.flat, HCD, MIX, B);
        sinkhorn(s, b.split, b.mix, b, B);
        hcw(s, b.cur, b.residual, b.carry, B);
        KB_CHECK(hipMemcpyAsync(b.carry, b.split, (size_t)B * MIX * 4, hipMemcpyDeviceToDevice, s));
        rmsw(s, b.norm, b.cur, b.norm_w, B);
    }
};

static kb::AbOpts opts(const char* tag, bool graph, int inner, int rounds) {
    kb::AbOpts o;
    o.graph = graph; o.inner = inner; o.rounds = rounds; o.tag = tag;
    return o;
}

// ================================================================== modes
static int mode_decode(Mods& m, unsigned b, bool direct) {
    Bufs bf(b, false);
    Prod p(m);
    // one-time correctness-ish sanity: run the chain once so buffers hold sane values
    p.mhc_fast(0, bf, MIX, 1, 1, 0, bf.residual, bf.residual, bf.cur, bf.norm);
    KB_CHECK(hipDeviceSynchronize());
    std::vector<kb::Variant> vs = {
        {"mhc_fast pre_attn (26,1,b) mode0", [&](hipStream_t s) { p.mhc_fast(s, bf, MIX, 1, 1, 0, bf.residual, bf.residual, bf.cur, bf.norm); }, (double)b * (HCD * 4.0 * 2 + NE * 4.0 * 2) + MIX * HCD * 2.0},
        {"mhc_fast pre_ffn collapse (1,1,b)", [&](hipStream_t s) { p.mhc_fast(s, bf, 0, 1, 0, 0, nullptr, bf.residual, bf.cur, bf.norm); }, (double)b * (HCD * 4.0 + NE * 4.0 * 2)},
        {"mhc_fast late mix (24,1,b) mode1", [&](hipStream_t s) { p.mhc_fast(s, bf, MIX, 0, 1, 1, bf.residual, nullptr, nullptr, nullptr); }, (double)b * HCD * 4.0 + MIX * HCD * 2.0},
        {"router f16_matvec_batched (48,1,b)", [&](hipStream_t s) { p.router_mv(s, bf.logits, bf.w_router, bf.x_ne, b); }, NEXP * NE * 2.0 + (double)b * NE * 4.0},
        {"router_topk_par plain (b)x512", [&](hipStream_t s) { p.topk(s, bf, b, false, bf.sel, bf.ew, bf.orig, bf.range); }, 0},
        {"router_topk_par prior (b)x512", [&](hipStream_t s) { p.topk(s, bf, b, true, bf.sel, bf.ew, bf.orig, bf.range); }, 0},
        {"hc_post_from_split (20,4,b)", [&](hipStream_t s) { p.hcpost(s, bf.after_attn, bf.attn_out, bf.residual, bf.split, b); }, (double)b * (HCD * 4.0 * 2 + NE * 4.0)},
        {"vec_add_inplace 5120b", [&](hipStream_t s) { p.vadd(s, bf.cur, bf.attn_out, NE * b); }, (double)b * NE * 12.0},
        {"rms_norm_weighted_batched (b) 5120", [&](hipStream_t s) { p.rmsw(s, bf.norm, bf.cur, bf.norm_w, b); }, (double)b * NE * 8.0},
        {"rms_norm_weighted_batched (b) 1280 q_a", [&](hipStream_t s) { kb::launch(p.f_rmsw, dim3(b), dim3(256), 0, s, bf.norm, (const float*)bf.cur, (const float*)bf.norm_w, 1280u, RMS_EPS); }, (double)b * 1280 * 8.0},
        {"rms_norm_weighted_batched (b) 512 kv", [&](hipStream_t s) { kb::launch(p.f_rmsw, dim3(b), dim3(256), 0, s, bf.norm, (const float*)bf.cur, (const float*)bf.norm_w, 512u, RMS_EPS); }, (double)b * 512 * 8.0},
        {"head hc_weighted_sum 1 row (20)", [&](hipStream_t s) { kb::launch(p.f_hcw1, dim3(NE / 256), dim3(256), 0, s, bf.cur, (const float*)bf.residual, (const float*)bf.carry, NE, NHC); }, HCD * 4.0 + NE * 4.0},
        {"head rms_norm_weighted 1 row (1)", [&](hipStream_t s) { kb::launch(p.f_rmsw1, dim3(1), dim3(256), 0, s, bf.norm, (const float*)bf.cur, (const float*)bf.norm_w, NE, RMS_EPS); }, NE * 12.0},
    };
    char tag[128];
    snprintf(tag, sizeof tag, "decode glue b=%u warm %s", b, direct ? "direct" : "graph");
    kb::ab(vs, opts(tag, !direct, 20, 40));
    return 0;
}

static int mode_prefill(Mods& m, unsigned B, bool flush) {
    Bufs bf(B, true);
    Prod p(m);
    kb::Flusher* fl = flush ? new kb::Flusher(40u << 20) : nullptr;
    p.chain(0, bf, B);
    KB_CHECK(hipDeviceSynchronize());
    const double xb = (double)B * HCD * 4.0;
    std::vector<kb::Variant> vs = {
        {"rms_norm_no_weight_batched (B,1)", [&](hipStream_t s) { p.rms_nw(s, bf.flat, bf.residual, B); }, 2 * xb},
        {"f16_gemm_wmma_lds_tiled M=24 (1,B/64)", [&](hipStream_t s) { p.gemm(s, bf.mix, bf.w_mix, bf.flat, HCD, MIX, B); }, xb + MIX * HCD * 2.0},
        {"hc_sinkhorn_par_batched (B)x16", [&](hipStream_t s) { p.sinkhorn(s, bf.split, bf.mix, bf, B); }, (double)B * MIX * 8.0},
        {"hc_weighted_sum_batched (20,1,B)", [&](hipStream_t s) { p.hcw(s, bf.cur, bf.residual, bf.carry, B); }, xb + (double)B * NE * 4.0},
        {"rms_norm_weighted_batched (B) 5120", [&](hipStream_t s) { p.rmsw(s, bf.norm, bf.cur, bf.norm_w, B); }, (double)B * NE * 8.0},
        {"CHAIN rms_nw+gemm+sink+hcw+copy+rmsw", [&](hipStream_t s) { p.chain(s, bf, B); }, 4 * xb + (double)B * NE * 12.0},
        {"mhc_fast collapse-only (1,1,B) [hcw+rmsw fused, prod kernel]", [&](hipStream_t s) { p.mhc_fast(s, bf, 0, 1, 0, 0, nullptr, bf.residual, bf.cur, bf.norm); }, xb + (double)B * NE * 8.0},
    };
    char tag[128];
    snprintf(tag, sizeof tag, "prefill mHC chain B=%u direct %s", B, flush ? "flush40MB" : "warm");
    kb::AbOpts o = opts(tag, false, 3, 30);
    if (fl) o.between = [&](hipStream_t s) { (*fl)(s); };
    kb::ab(vs, o);
    // correctness of the production collapse-only mhc_fast vs hcw+rmsw at this B (documented bit-identical)
    {
        float* cur2 = bf.flat;                    // flat is free after the timing; reuse it as scratch
        float* norm2 = bf.flat + (size_t)B * NE;
        p.hcw(0, bf.cur, bf.residual, bf.carry, B);
        p.rmsw(0, bf.norm, bf.cur, bf.norm_w, B);
        p.mhc_fast(0, bf, 0, 1, 0, 0, nullptr, bf.residual, cur2, norm2);
        KB_CHECK(hipDeviceSynchronize());
        auto a = kb::d2h(bf.norm, (size_t)B * NE), c = kb::d2h(norm2, (size_t)B * NE);
        kb::print_cmp("mhc_fast collapse vs hcw+rmsw (norm)", kb::compare_f32(a.data(), c.data(), a.size()));
    }
    return 0;
}

static int mode_prof(Mods& m, const std::string& what, unsigned b) {
    Bufs bf(b, true);
    Prod p(m);
    hipStream_t s;
    KB_CHECK(hipStreamCreate(&s));
    kb::warm(200, s);
    const std::string dir = getenv("HARNESS_DIR") ? getenv("HARNESS_DIR") : ".";
    hipFunction_t f_gemmc = nullptr, f_mvc = nullptr, f_topkc = nullptr;
    std::unique_ptr<kb::Module> cg, cmv, ctk;
    if (what == "gemmc") { cg.reset(new kb::Module(dir + "/cand_gemm_gfx1201.hsaco")); f_gemmc = cg->fn("f16_gemm_narrow_n16_bk128_pf2"); }
    if (what == "mvc") { cmv.reset(new kb::Module(dir + "/cand_router_mv_gfx1201.hsaco")); f_mvc = cmv->fn("f16_matvec_batched_h20"); }
    if (what == "topkc") { ctk.reset(new kb::Module(dir + "/cand_topk2_gfx1201.hsaco")); f_topkc = ctk->fn("router_topk_wfred"); }
    Bufs* bd = (what == "mvc" || what == "mv" || what == "topkc" || what == "topk") ? new Bufs(b, false) : nullptr;
    for (int i = 0; i < 8; ++i) {
        if (what == "gemmc") {
            kb::launch(f_gemmc, dim3(1, (b + 15) / 16), dim3(256), 0, s, bf.mix, bf.w_mix, (const float*)bf.flat, HCD, MIX, b);
        } else if (what == "mvc") {
            kb::launch(f_mvc, dim3(NEXP / 8, 1, b), dim3(256), 0, s, bd->logits, bd->w_router, (const float*)bd->x_ne, NE, NEXP);
        } else if (what == "mv") {
            p.router_mv(s, bd->logits, bd->w_router, bd->x_ne, b);
        } else if (what == "topkc") {
            void* nul = nullptr;
            kb::launch(f_topkc, dim3(b), dim3(512), 0, s, bd->sel, bd->ew, (const float*)bd->logits, (const float*)bd->bias, NEXP, NUSED,
                       EW_SCALE, W_EPS, nul, 0u, nul, (const void*)bd->prior, 2u, 0u, (void*)bd->orig, (void*)bd->range);
        } else if (what == "mhc") {
            p.mhc_fast(s, bf, MIX, 1, 1, 0, bf.residual, bf.residual, bf.cur, bf.norm);
        } else if (what == "mhc_late") {
            p.mhc_fast(s, bf, MIX, 0, 1, 1, bf.residual, nullptr, nullptr, nullptr);
        } else if (what == "gemm") {
            p.gemm(s, bf.mix, bf.w_mix, bf.flat, HCD, MIX, b);
        } else if (what == "topk") {
            p.topk(s, *bd, b, true, bd->sel, bd->ew, bd->orig, bd->range);
        } else if (what == "chain") {
            p.chain(s, bf, b);
        } else if (what == "rmsnw") {
            p.rms_nw(s, bf.flat, bf.residual, b);
        }
    }
    KB_CHECK(hipStreamSynchronize(s));
    printf("prof %s b=%u done\n", what.c_str(), b);
    return 0;
}

// ------------------------------------------------------------------ candidates: top-k
// cand_topk.hip provides router_topk_wave(...) with the SAME signature as router_topk_par.
static int mode_topk(Mods& m, const std::string& dir, unsigned b, bool prior) {
    Bufs bf(b, false);
    Prod p(m);
    std::vector<kb::Variant> vs = {
        {prior ? "router_topk_par prior (b)x512" : "router_topk_par plain (b)x512",
         [&](hipStream_t s) { p.topk(s, bf, b, prior, bf.sel, bf.ew, bf.orig, bf.range); }, 0},
    };
    std::vector<std::unique_ptr<kb::Module>> cm;
    std::vector<hipFunction_t> cf;
    std::vector<std::string> names = {"router_topk_wfred", "router_topk_wave", "router_topk_wave_b2"};
    std::vector<std::string> files = {"cand_topk2_gfx1201.hsaco", "cand_topk_gfx1201.hsaco", "cand_topk_gfx1201.hsaco"};
    std::vector<dim3> blocks = {dim3(512), dim3(512), dim3(32)};
    std::vector<int*> sel2; std::vector<float*> ew2; std::vector<int*> orig2; std::vector<float*> range2;
    for (size_t i = 0; i < names.size(); ++i) {
        if (!file_exists(dir + "/" + files[i])) continue;
        cm.emplace_back(new kb::Module(dir + "/" + files[i]));
        hipFunction_t f;
        if (hipModuleGetFunction(&f, cm.back()->m, names[i].c_str()) != hipSuccess) { cm.pop_back(); continue; }
        cf.push_back(f);
        int* sl = kb::dalloc<int>((size_t)b * NUSED); float* ew = kb::dalloc<float>((size_t)b * NUSED);
        int* og = kb::dalloc<int>((size_t)b * NUSED); float* rg = kb::dalloc<float>(b);
        sel2.push_back(sl); ew2.push_back(ew); orig2.push_back(og); range2.push_back(rg);
        dim3 blk = blocks[i];
        size_t idx = cf.size() - 1;
        vs.push_back({names[i] + (prior ? " prior" : " plain"), [&, f, blk, idx](hipStream_t s) {
            void* nul = nullptr;
            kb::launch(f, dim3(b), blk, 0, s, sel2[idx], ew2[idx], (const float*)bf.logits, (const float*)bf.bias, NEXP, NUSED,
                       EW_SCALE, W_EPS, nul, 0u, nul, prior ? (const void*)bf.prior : nul, prior ? 2u : 0u, 0u,
                       prior ? (void*)orig2[idx] : nul, prior ? (void*)range2[idx] : nul);
        }, 0});
    }
    // ---- correctness on several logit seeds (incl. ties: quantized logits)
    for (int seed = 0; seed < 4; ++seed) {
        if (seed < 3) kb::fill_f32(bf.logits, (size_t)b * NEXP, 100 + seed, -3.f, 3.f);
        else {  // ties: logits on a coarse grid so equal scores appear
            std::vector<float> h((size_t)b * NEXP);
            for (size_t i = 0; i < h.size(); ++i) h[i] = (float)((int)((i * 2654435761u >> 9) % 7) - 3);
            KB_CHECK(hipMemcpy(bf.logits, h.data(), h.size() * 4, hipMemcpyHostToDevice));
        }
        if (seed == 3) {  // ties also need equal bias/prior
            KB_CHECK(hipMemset(bf.bias, 0, NEXP * 4));
        }
        for (size_t v = 0; v < vs.size(); ++v) vs[v].run(0);
        KB_CHECK(hipDeviceSynchronize());
        auto s0 = kb::d2h(bf.sel, (size_t)b * NUSED); auto w0 = kb::d2h(bf.ew, (size_t)b * NUSED);
        auto o0 = kb::d2h(bf.orig, (size_t)b * NUSED); auto r0 = kb::d2h(bf.range, b);
        for (size_t i = 0; i < cf.size(); ++i) {
            auto s1 = kb::d2h(sel2[i], (size_t)b * NUSED); auto w1 = kb::d2h(ew2[i], (size_t)b * NUSED);
            auto o1 = kb::d2h(orig2[i], (size_t)b * NUSED); auto r1 = kb::d2h(range2[i], b);
            size_t sd = 0, od = 0;
            for (size_t j = 0; j < s0.size(); ++j) { sd += s0[j] != s1[j]; if (prior) od += o0[j] != o1[j]; }
            char tag[96];
            snprintf(tag, sizeof tag, "%s seed%d sel", vs[i + 1].name.c_str(), seed);
            printf("CMP %-44s sel_diff=%zu orig_diff=%zu (of %zu)\n", tag, sd, od, s0.size());
            snprintf(tag, sizeof tag, "%s seed%d weights", vs[i + 1].name.c_str(), seed);
            kb::print_cmp(tag, kb::compare_f32(w0.data(), w1.data(), w0.size()));
            if (prior) {
                snprintf(tag, sizeof tag, "%s seed%d range", vs[i + 1].name.c_str(), seed);
                kb::print_cmp(tag, kb::compare_f32(r0.data(), r1.data(), b));
            }
        }
    }
    kb::fill_f32(bf.logits, (size_t)b * NEXP, 23, -3.f, 3.f);
    kb::fill_f32(bf.bias, NEXP, 24, -0.1f, 0.1f);
    char tag[128];
    snprintf(tag, sizeof tag, "router_topk b=%u %s graph warm", b, prior ? "prior" : "plain");
    kb::ab(vs, opts(tag, b <= 16, 20, 40));
    return 0;
}

// ------------------------------------------------------------------ candidates: prefill GEMM
// cand_gemm.hip provides kernels with the production signature (out, w, x, K, M, B) and a grid rule.
struct GemmCand { const char* name; const char* sym; unsigned block; unsigned m_tile; unsigned n_tile; };
static const GemmCand GEMM_CANDS[] = {  // cand_gemm.hip: grid (1, ceil(B/NT)) x 256
    {"cand narrow n16 bk64 pf4", "f16_gemm_narrow_n16_bk64_pf4", 256, 32, 16},
    {"cand narrow n16 bk128 pf3", "f16_gemm_narrow_n16_bk128_pf3", 256, 32, 16},
    {"cand narrow n16 bk128 pf2", "f16_gemm_narrow_n16_bk128_pf2", 256, 32, 16},
    {"cand narrow n16 bk64 pf6", "f16_gemm_narrow_n16_bk64_pf6", 256, 32, 16},
    {"cand narrow n32 bk64 pf4", "f16_gemm_narrow_n32_bk64_pf4", 256, 32, 32},
    {"cand narrow n32 bk128 pf2", "f16_gemm_narrow_n32_bk128_pf2", 256, 32, 32},
};
static int mode_gemm(Mods& m, const std::string& dir, unsigned B, bool flush) {
    Bufs bf(B, true);
    Prod p(m);
    kb::Flusher* fl = flush ? new kb::Flusher(40u << 20) : nullptr;
    p.rms_nw(0, bf.flat, bf.residual, B);
    KB_CHECK(hipDeviceSynchronize());
    const double xb = (double)B * HCD * 4.0;
    std::vector<kb::Variant> vs = {
        {"f16_gemm_wmma_lds_tiled M=24 (1,B/64)", [&](hipStream_t s) { p.gemm(s, bf.mix, bf.w_mix, bf.flat, HCD, MIX, B); }, xb + MIX * HCD * 2.0},
    };
    std::unique_ptr<kb::Module> cm;
    std::vector<float*> outs;
    std::vector<std::string> cnames;
    if (file_exists(dir + "/cand_gemm_gfx1201.hsaco")) {
        cm.reset(new kb::Module(dir + "/cand_gemm_gfx1201.hsaco"));
        for (const GemmCand& c : GEMM_CANDS) {
            hipFunction_t f;
            if (hipModuleGetFunction(&f, cm->m, c.sym) != hipSuccess) continue;
            float* o = kb::dalloc<float>((size_t)B * MIX);
            KB_CHECK(hipMemset(o, 0, (size_t)B * MIX * 4));
            outs.push_back(o);
            cnames.push_back(c.name);
            unsigned nt = c.n_tile, blk = c.block;
            vs.push_back({c.name, [&, f, o, nt, blk](hipStream_t s) {
                kb::launch(f, dim3(1, (B + nt - 1) / nt), dim3(blk), 0, s, o, bf.w_mix, (const float*)bf.flat, HCD, MIX, B);
            }, xb + MIX * HCD * 2.0});
        }
    }
    // correctness vs production at this B
    for (auto& v : vs) v.run(0);
    KB_CHECK(hipDeviceSynchronize());
    auto ref = kb::d2h(bf.mix, (size_t)B * MIX);
    for (size_t i = 0; i < outs.size(); ++i) {
        auto got = kb::d2h(outs[i], (size_t)B * MIX);
        char tag[96]; snprintf(tag, sizeof tag, "%s B=%u", cnames[i].c_str(), B);
        kb::print_cmp(tag, kb::compare_f32(ref.data(), got.data(), ref.size()));
    }
    char tag[128];
    snprintf(tag, sizeof tag, "prefill pre-mix GEMM K=20480 M=24 B=%u direct %s", B, flush ? "flush40MB" : "warm");
    kb::AbOpts o = opts(tag, false, 3, 30);
    if (fl) o.between = [&](hipStream_t s) { (*fl)(s); };
    kb::ab(vs, o);
    return 0;
}

// ------------------------------------------------------------------ candidates: rms_norm_weighted
// cand_rms.hip provides rms_norm_weighted_batched_fast (same signature/grid as production).
static int mode_rms(Mods& m, const std::string& dir, unsigned b, bool direct) {
    Bufs bf(b, false);
    Prod p(m);
    if (!file_exists(dir + "/cand_rms_gfx1201.hsaco")) { fprintf(stderr, "no cand_rms hsaco\n"); return 1; }
    kb::Module cm(dir + "/cand_rms_gfx1201.hsaco");
    hipFunction_t fc = cm.fn("rms_norm_weighted_batched_fast");
    float* out2 = kb::dalloc<float>((size_t)b * NE);
    const unsigned ns[3] = {5120u, 1280u, 512u};
    // correctness: every production n, 3 seeds, plus a tail n (production never uses it: generic path)
    for (int seed = 0; seed < 3; ++seed) {
        kb::fill_f32(bf.cur, (size_t)b * NE, 200 + seed, -2.f, 2.f);
        kb::fill_f32(bf.norm_w, NE, 300 + seed, 0.2f, 1.8f);
        for (unsigned n : {5120u, 1280u, 512u, 4864u}) {
            kb::launch(p.f_rmsw, dim3(b), dim3(256), 0, 0, bf.norm, (const float*)bf.cur, (const float*)bf.norm_w, n, RMS_EPS);
            kb::launch(fc, dim3(b), dim3(256), 0, 0, out2, (const float*)bf.cur, (const float*)bf.norm_w, n, RMS_EPS);
            KB_CHECK(hipDeviceSynchronize());
            auto a = kb::d2h(bf.norm, (size_t)b * n), c = kb::d2h(out2, (size_t)b * n);
            char tag[96]; snprintf(tag, sizeof tag, "rms_fast vs prod n=%u b=%u seed%d", n, b, seed);
            kb::print_cmp(tag, kb::compare_f32(a.data(), c.data(), a.size()));
        }
        // head prep: production rms_norm_weighted (1 row) vs fast at grid 1
        kb::launch(p.f_rmsw1, dim3(1), dim3(256), 0, 0, bf.norm, (const float*)bf.cur, (const float*)bf.norm_w, NE, RMS_EPS);
        kb::launch(fc, dim3(1), dim3(256), 0, 0, out2, (const float*)bf.cur, (const float*)bf.norm_w, NE, RMS_EPS);
        KB_CHECK(hipDeviceSynchronize());
        auto a = kb::d2h(bf.norm, (size_t)NE), c = kb::d2h(out2, (size_t)NE);
        char tag[96]; snprintf(tag, sizeof tag, "rms_fast(1) vs rms_norm_weighted seed%d", seed);
        kb::print_cmp(tag, kb::compare_f32(a.data(), c.data(), a.size()));
    }
    std::vector<kb::Variant> vs;
    for (unsigned n : ns) {
        char nm[80];
        snprintf(nm, sizeof nm, "rms_norm_weighted_batched (b) n=%u", n);
        vs.push_back({nm, [&, n](hipStream_t s) { kb::launch(p.f_rmsw, dim3(b), dim3(256), 0, s, bf.norm, (const float*)bf.cur, (const float*)bf.norm_w, n, RMS_EPS); }, (double)b * n * 12.0});
        snprintf(nm, sizeof nm, "cand rms_fast (b) n=%u", n);
        vs.push_back({nm, [&, n](hipStream_t s) { kb::launch(fc, dim3(b), dim3(256), 0, s, out2, (const float*)bf.cur, (const float*)bf.norm_w, n, RMS_EPS); }, (double)b * n * 12.0});
    }
    vs.push_back({"head rms_norm_weighted 1 row (1)", [&](hipStream_t s) { kb::launch(p.f_rmsw1, dim3(1), dim3(256), 0, s, bf.norm, (const float*)bf.cur, (const float*)bf.norm_w, NE, RMS_EPS); }, NE * 12.0});
    vs.push_back({"cand rms_fast 1 row (1)", [&](hipStream_t s) { kb::launch(fc, dim3(1), dim3(256), 0, s, out2, (const float*)bf.cur, (const float*)bf.norm_w, NE, RMS_EPS); }, NE * 12.0});
    char tag[128];
    snprintf(tag, sizeof tag, "rms_norm_weighted b=%u warm %s", b, direct ? "direct" : "graph");
    kb::ab(vs, opts(tag, !direct, 20, 40));
    return 0;
}

// ------------------------------------------------------------------ candidates: router matvec
// cand_router_mv.hip provides f16_matvec_batched_h{20,40,80} (same signature/grid as production).
static int mode_mv(Mods& m, const std::string& dir, unsigned b, bool direct, bool cold) {
    Bufs bf(b, false);
    Prod p(m);
    if (!file_exists(dir + "/cand_router_mv_gfx1201.hsaco")) { fprintf(stderr, "no cand_router_mv hsaco\n"); return 1; }
    kb::Module cm(dir + "/cand_router_mv_gfx1201.hsaco");
    const char* syms[3] = {"f16_matvec_batched_h20", "f16_matvec_batched_h40", "f16_matvec_batched_h80"};
    hipFunction_t fc[3];
    float* outs[3];
    for (int i = 0; i < 3; ++i) { fc[i] = cm.fn(syms[i]); outs[i] = kb::dalloc<float>((size_t)b * NEXP); }
    // correctness: 3 seeds at k = 5120 (production) and one non-multiple k (generic path)
    for (int seed = 0; seed < 3; ++seed) {
        kb::fill_f32(bf.x_ne, (size_t)b * NE, 400 + seed, -1.f, 1.f);
        kb::fill_f16(bf.w_router, (size_t)NEXP * NE, 500 + seed, -0.05f, 0.05f);
        for (unsigned k : {5120u, 4992u}) {
            kb::launch(p.f_mv, dim3(NEXP / 8, 1, b), dim3(256), 0, 0, bf.logits, bf.w_router, (const float*)bf.x_ne, k, NEXP);
            for (int i = 0; i < 3; ++i)
                kb::launch(fc[i], dim3(NEXP / 8, 1, b), dim3(256), 0, 0, outs[i], bf.w_router, (const float*)bf.x_ne, k, NEXP);
            KB_CHECK(hipDeviceSynchronize());
            auto a = kb::d2h(bf.logits, (size_t)b * NEXP);
            for (int i = 0; i < 3; ++i) {
                auto c = kb::d2h(outs[i], (size_t)b * NEXP);
                char tag[96]; snprintf(tag, sizeof tag, "%s k=%u b=%u seed%d", syms[i], k, b, seed);
                kb::print_cmp(tag, kb::compare_f32(a.data(), c.data(), a.size()));
            }
        }
    }
    // COLD W regime (production: the layer's router W is streamed once per lane-layer, ~300 ms
    // apart, with ~178 MB of dense weights in between => not in L2/MALL): rotate over NCOPY copies
    // of W (NCOPY x 3.9 MB = 125 MB > 64 MB MALL + 8 MB L2), one copy per captured call.
    const unsigned NCOPY = cold ? 32u : 1u;
    std::vector<void*> wcopies(NCOPY, bf.w_router);
    for (unsigned c = 1; c < NCOPY; ++c) {
        wcopies[c] = kb::dalloc<uint16_t>((size_t)NEXP * NE);
        KB_CHECK(hipMemcpy(wcopies[c], bf.w_router, (size_t)NEXP * NE * 2, hipMemcpyDeviceToDevice));
    }
    unsigned rot = 0;
    std::vector<kb::Variant> vs = {
        {"router f16_matvec_batched (48,1,b)", [&](hipStream_t s) { p.router_mv(s, bf.logits, wcopies[(rot++) % NCOPY], bf.x_ne, b); }, NEXP * NE * 2.0 + (double)b * NE * 4.0},
    };
    for (int i = 0; i < 3; ++i) {
        hipFunction_t f = fc[i]; float* o = outs[i];
        vs.push_back({std::string("cand ") + syms[i], [&, f, o](hipStream_t s) {
            kb::launch(f, dim3(NEXP / 8, 1, b), dim3(256), 0, s, o, wcopies[(rot++) % NCOPY], (const float*)bf.x_ne, NE, NEXP);
        }, NEXP * NE * 2.0 + (double)b * NE * 4.0});
    }
    char tag[128];
    snprintf(tag, sizeof tag, "router matvec 384x5120 b=%u %s %s", b, cold ? "coldW(32 copies)" : "warm", direct ? "direct" : "graph");
    kb::ab(vs, opts(tag, !direct, 20, 40));
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: harness <mode> <b> [opt]\n"); return 64; }
    std::string mode = argv[1];
    std::string dir = ".";
    if (const char* d = getenv("HARNESS_DIR")) dir = d;
    kb::init();
    Mods m(dir);
    if (mode == "decode") return mode_decode(m, atoi(argv[2]), argc > 3 && std::string(argv[3]) == "direct");
    if (mode == "prefill") return mode_prefill(m, atoi(argv[2]), argc > 3 && std::string(argv[3]) == "flush");
    if (mode == "topk") return mode_topk(m, dir, atoi(argv[2]), argc > 3 && std::string(argv[3]) == "prior");
    if (mode == "gemm") return mode_gemm(m, dir, atoi(argv[2]), argc > 3 && std::string(argv[3]) == "flush");
    if (mode == "rms") return mode_rms(m, dir, atoi(argv[2]), argc > 3 && std::string(argv[3]) == "direct");
    if (mode == "mv") {
        bool direct = false, cold = false;
        for (int i = 3; i < argc; ++i) { if (std::string(argv[i]) == "direct") direct = true; if (std::string(argv[i]) == "cold") cold = true; }
        return mode_mv(m, dir, atoi(argv[2]), direct, cold);
    }
    if (mode == "prof") return mode_prof(m, argv[2], argc > 3 ? atoi(argv[3]) : 1);
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 64;
}
