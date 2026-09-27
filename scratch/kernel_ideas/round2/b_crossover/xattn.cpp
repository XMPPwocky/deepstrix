// D_attention harness v2: production decode attention pair (attention_dec score + smwsum _ldsv_f16s)
// at production decode shapes, plus candidate kernels from cand2_attn_<arch>.hsaco (attempt 2) and
// cand_attn_<arch>.hsaco (attempt 1, optional).
//
//   ./attn_harness2 <arch> <dir> decode [b...]   graph-mode A/B at decode shapes (default b = 1 2 4 5 8)
//   ./attn_harness2 <arch> <dir> check           correctness sweep over shapes / tails / dense comp store
//   ./attn_harness2 <arch> <dir> prof <b> [cand] a few direct launches of base + one candidate (ATT)
//   ./attn_harness2 <arch> <dir> f16rt           prove f16_roundtrip is a no-op on the f16 cache contents
//
// Candidate kinds (looked up by symbol; missing symbols are skipped):
//   "score"  : same signature as attention_dec_score_htiled_wmma_f16s; keys per WG + block size
//   "fused"  : (out, sinks, q, raw_kv, comp_kv, n_raw_per, n_raw_offset_per, n_comp_per, n_head, kq_scale,
//               comp_kv_batch_stride, comp_base_per)  grid (4, b) x 512
//   "flash"  : part1 (part_o, part_ml, q, raw_kv, comp_kv, nr, noff, nc, sinks, n_head, kq_scale,
//               comp_kv_batch_stride, comp_base_per, max_splits) grid (4, S, b) x 512
//              + combine (out, part_o, part_ml, sinks, nr, nc, n_head, kps, max_splits) grid (64, b) x 512
#include "kbench.h"

#include <cmath>
#include <cstring>

static hipFunction_t opt_fn(hipModule_t m, const char* name) {
    hipFunction_t f = nullptr;
    if (!m) return nullptr;
    if (hipModuleGetFunction(&f, m, name) != hipSuccess) return nullptr;
    return f;
}
static hipModule_t opt_mod(const std::string& p) {
    hipModule_t m = nullptr;
    if (hipModuleLoad(&m, p.c_str()) != hipSuccess) { fprintf(stderr, "[no module %s]\n", p.c_str()); return nullptr; }
    return m;
}

struct Cand {
    std::string name, kind;
    hipFunction_t f = nullptr, f2 = nullptr;
    unsigned kpw = 256;   // score: keys per WG; flash: keys per split
    unsigned block = 512;
};

struct Shape {
    std::vector<int> nr, nc;   // per row
    bool dense;                // comp store: dense shared (comp_base_per set, stride 0) or gathered (stride 512)
    const char* tag;
};

int main(int argc, char** argv) {
    const std::string arch = argc > 1 ? argv[1] : "gfx1201";
    const std::string dir = argc > 2 ? argv[2] : ".";
    const std::string mode = argc > 3 ? argv[3] : "decode";
    kb::init();
    kb::Module base_mixed(dir + "/base_attention_mixed_" + arch + ".hsaco");
    kb::Module base_dec(dir + "/base_attention_dec_" + arch + ".hsaco");
    hipModule_t cand1 = opt_mod(dir + "/cand_attn_" + arch + ".hsaco");
    hipModule_t cand2 = opt_mod(dir + "/cand2_attn_" + arch + ".hsaco");
    hipModule_t cand3 = opt_mod(dir + "/cand3_attn_" + arch + ".hsaco");
    hipModule_t cand4 = opt_mod(dir + "/cand4_attn_" + arch + ".hsaco");
    hipModule_t cand5 = opt_mod(dir + "/cand5_attn_" + arch + ".hsaco");
    hipModule_t cand6 = opt_mod(dir + "/cand6_attn_" + arch + ".hsaco");
    hipModule_t cand7 = opt_mod(dir + "/cand7_attn_" + arch + ".hsaco");
    hipFunction_t f_dec_score = base_dec.fn("attention_dec_score_htiled_wmma_f16s");
    hipFunction_t f_mix_score = base_mixed.fn("attention_mixed_score_batched_htiled_wmma_f16s");
    hipFunction_t f_smwsum = base_mixed.fn("attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_f16s");

    // ------------------------------------------------------------------ f16rt no-op proof
    if (mode == "f16rt") {
        kb::Module m_fp4(dir + "/base_fp4_kv_quant_" + arch + ".hsaco");
        kb::Module m_rt(dir + "/base_f16_roundtrip_" + arch + ".hsaco");
        kb::Module m_app(dir + "/base_kv_cache_append_" + arch + ".hsaco");
        hipFunction_t f_fp8 = m_fp4.fn("fp8_act_quant_inplace");
        hipFunction_t f_rt = m_rt.fn("f16_roundtrip");
        hipFunction_t f_app = m_app.fn("kv_cache_append_batched");
        const unsigned B = 512, W = 512;
        float* x = kb::dalloc<float>((size_t)B * W);
        float* y = kb::dalloc<float>((size_t)B * W);
        uint16_t* ca = kb::dalloc<uint16_t>((size_t)B * W);
        uint16_t* cb = kb::dalloc<uint16_t>((size_t)B * W);
        size_t total_diff = 0, total_rt_changed = 0;
        const float mags[] = {1e-6f, 1e-4f, 3e-3f, 0.02f, 0.3f, 1.f, 30.f, 3000.f, 60000.f, 1e6f};
        for (int mi = 0; mi < 10; ++mi) {
            std::vector<float> h((size_t)B * W);
            kb::fill_f32(x, h.size(), 100 + mi, -1.f, 1.f);
            h = kb::d2h(x, h.size());
            for (size_t i = 0; i < h.size(); ++i) {
                // rows of mixed magnitude: row r scaled by mags[mi] * 2^(-(r%24)) so some 32-blocks are tiny
                const unsigned r = i / W;
                h[i] *= mags[mi] * std::ldexp(1.f, -(int)(r % 24));
                if ((i % 97) == 0) h[i] = 0.f;
            }
            KB_CHECK(hipMemcpy(x, h.data(), h.size() * 4, hipMemcpyHostToDevice));
            kb::launch(f_fp8, dim3(B), dim3(W), 0, 0, x, B, W);
            KB_CHECK(hipMemcpy(y, x, h.size() * 4, hipMemcpyDeviceToDevice));
            kb::launch(f_rt, dim3((B * W + 255) / 256), dim3(256), 0, 0, y, B * W);
            const int* null_i = nullptr;
            kb::launch(f_app, dim3(B), dim3(W), 0, 0, (_Float16*)ca, (const float*)x, 0u, W, null_i);
            kb::launch(f_app, dim3(B), dim3(W), 0, 0, (_Float16*)cb, (const float*)y, 0u, W, null_i);
            KB_CHECK(hipDeviceSynchronize());
            auto a = kb::d2h(ca, h.size()), c = kb::d2h(cb, h.size());
            auto xs = kb::d2h(x, h.size()), ys = kb::d2h(y, h.size());
            size_t nd = 0, nrt = 0;
            for (size_t i = 0; i < a.size(); ++i) { nd += a[i] != c[i]; nrt += memcmp(&xs[i], &ys[i], 4) != 0; }
            printf("f16rt mag=%g: fp8 rows where roundtrip CHANGED f32 value: %zu of %zu; f16 cache bits differ (with vs without f16rt): %zu\n",
                   mags[mi], nrt, a.size(), nd);
            total_diff += nd; total_rt_changed += nrt;
        }
        printf("KBJSON {\"cmp\":\"f16_roundtrip no-op on kv_cache_append output\",\"bitexact\":%d,\"bit_diff\":%zu,\"f32_changed_by_rt\":%zu}\n",
               (int)(total_diff == 0), total_diff, total_rt_changed);
        // timing of the small kv-chain kernels at decode b = 4 (graph mode, warm): the cost of the no-op
        for (unsigned bb : {1u, 4u, 8u}) {
            const int* null_i = nullptr;
            std::vector<kb::Variant> vs = {
                {"kv chain: fp8_act_quant + f16_roundtrip + kv_cache_append", [&](hipStream_t s) {
                    kb::launch(f_fp8, dim3(bb), dim3(W), 0, s, x, bb, W);
                    kb::launch(f_rt, dim3((bb * W + 255) / 256), dim3(256), 0, s, x, bb * W);
                    kb::launch(f_app, dim3(bb), dim3(W), 0, s, (_Float16*)ca, (const float*)x, 0u, W, null_i); }},
                {"kv chain without f16_roundtrip", [&](hipStream_t s) {
                    kb::launch(f_fp8, dim3(bb), dim3(W), 0, s, x, bb, W);
                    kb::launch(f_app, dim3(bb), dim3(W), 0, s, (_Float16*)ca, (const float*)x, 0u, W, null_i); }},
                {"f16_roundtrip alone", [&](hipStream_t s) { kb::launch(f_rt, dim3((bb * W + 255) / 256), dim3(256), 0, s, x, bb * W); }},
                {"fp8_act_quant_inplace alone", [&](hipStream_t s) { kb::launch(f_fp8, dim3(bb), dim3(W), 0, s, x, bb, W); }},
                {"kv_cache_append_batched alone", [&](hipStream_t s) { kb::launch(f_app, dim3(bb), dim3(W), 0, s, (_Float16*)ca, (const float*)x, 0u, W, null_i); }},
            };
            kb::AbOpts o;
            o.graph = true; o.inner = 10; o.rounds = 60;
            char tag[96]; snprintf(tag, sizeof tag, "kv chain b=%u graph warm", bb);
            o.tag = tag;
            kb::ab(vs, o);
        }
        return 0;
    }

    // ------------------------------------------------------------------ candidates
    std::vector<Cand> cands;
    auto add = [&](hipModule_t m, const char* sym, const char* kind, unsigned kpw = 256, unsigned blk = 512, const char* sym2 = nullptr) {
        Cand c; c.name = sym; c.kind = kind; c.f = opt_fn(m, sym); c.kpw = kpw; c.block = blk;
        if (sym2) c.f2 = opt_fn(m, sym2);
        if (c.f && (!sym2 || c.f2)) cands.push_back(c);
    };
    // attempt 2
    add(cand2, "attention_dec_score_blk256", "score", 128, 256);
    add(cand2, "attention_dec_score_blk128", "score", 64, 128);
    add(cand2, "attention_dec_score_blk64", "score", 32, 64);
    add(cand2, "attention_dec_fused_direct_g16", "fused");
    add(cand2, "attention_dec_fused_direct_g32", "fused");
    add(cand2, "attention_dec_fused_direct_g32d4", "fused");
    add(cand2, "attention_dec_flash_part1_k32", "flash", 32, 512, "attention_dec_flash_combine");
    add(cand2, "attention_dec_flash_part1_k64", "flash", 64, 512, "attention_dec_flash_combine");
    add(cand2, "attention_dec_flash_part1_k128", "flash", 128, 512, "attention_dec_flash_combine");
    // attempt 2b: register-lean flash
    add(cand3, "attention_dec_flash2_part1_k32", "flash", 32, 512, "attention_dec_flash2_combine");
    add(cand3, "attention_dec_flash2_part1_k64", "flash", 64, 512, "attention_dec_flash2_combine");
    add(cand3, "attention_dec_flash2_part1_k128", "flash", 128, 512, "attention_dec_flash2_combine");
    add(cand3, "attention_dec_flash2_part1_k64g32", "flash", 64, 512, "attention_dec_flash2_combine");
    // attempt 2c: q-in-LDS fused (bit-exact) + flash3 (16-warp score phase)
    add(cand4, "attention_dec_fused_qlds_pf2r2", "fused");
    add(cand4, "attention_dec_fused_qlds_pf2r4", "fused");
    add(cand4, "attention_dec_fused_qlds_pf1r4", "fused");
    add(cand4, "attention_dec_fused_qlds_pf1r8", "fused");
    add(cand4, "attention_dec_flash3_part1_k64", "flash", 64, 512, "attention_dec_flash3_combine");
    add(cand4, "attention_dec_flash3_part1_k128", "flash", 128, 512, "attention_dec_flash3_combine");
    // attempt 2f: software-pipelined transposed ring (bit-exact)
    add(cand7, "attention_dec_fused_vt_qreg_sp_d8", "fused");
    add(cand7, "attention_dec_fused_vt_qreg_sp_d4", "fused");
    // attempt 2e: transposed warp-private V ring (bit-exact)
    add(cand6, "attention_dec_fused_vt_d8", "fused");
    add(cand6, "attention_dec_fused_vt_qreg_d8", "fused");
    add(cand6, "attention_dec_fused_vt_d4", "fused");
    // attempt 2d: warp-private V ring (bit-exact)
    add(cand5, "attention_dec_fused_wpriv_d8", "fused");
    add(cand5, "attention_dec_fused_wpriv_s2_d8", "fused");
    add(cand5, "attention_dec_fused_wpriv_d4", "fused");
    // attempt 1 (reference points)
    add(cand1, "attention_dec_fused_f16s", "fused");
    add(cand1, "attention_dec_smwsum_pf2", "smwsum");
    // CANDS=name1,name2 (env) or `prof <b> <name>` restricts the candidate set
    std::string only = (mode == "prof" && argc > 5) ? argv[5] : "";
    if (only.empty() && getenv("CANDS")) only = getenv("CANDS");
    if (!only.empty()) {
        std::vector<Cand> keep;
        for (auto& c : cands) if (("," + only + ",").find("," + c.name + ",") != std::string::npos) keep.push_back(c);
        cands = keep;
    }
    for (auto& c : cands) fprintf(stderr, "[cand] %s (%s)\n", c.name.c_str(), c.kind.c_str());

    const unsigned NH = 64, HD = 512, STRIDE = 3072, MKW = (82176 + 31) / 32;
    const float kq_scale = 1.0f / std::sqrt((float)HD);

    // ---------------------------------------------------------------- buffers (max b = 8, n_total <= 640)
    const unsigned bmax = 16, raw_slots = 256, ncmax = 512, ntmax = 128 + ncmax;
    const unsigned stride = STRIDE;
    const unsigned max_splits = 20;
    float* q = kb::dalloc<float>((size_t)bmax * NH * HD);
    uint16_t* raw_kv = kb::dalloc<uint16_t>((size_t)bmax * raw_slots * HD);
    uint16_t* comp_kv = kb::dalloc<uint16_t>((size_t)bmax * ncmax * HD + (size_t)bmax * 640 * HD);  // gathered [b,512,512] or dense with per-row base
    uint16_t* scores = kb::dalloc<uint16_t>((size_t)bmax * NH * stride);
    uint16_t* scores_ref = kb::dalloc<uint16_t>((size_t)bmax * NH * stride);
    uint16_t* scores2 = kb::dalloc<uint16_t>((size_t)bmax * NH * stride);
    float* out = kb::dalloc<float>((size_t)bmax * NH * HD);
    float* out2 = kb::dalloc<float>((size_t)bmax * NH * HD);
    float* sinks = kb::dalloc<float>(NH);
    int* d_nr = kb::dalloc<int>(bmax);
    int* d_off = kb::dalloc<int>(bmax);
    int* d_nc = kb::dalloc<int>(bmax);
    int* d_cb = kb::dalloc<int>(bmax);
    float* part_o = kb::dalloc<float>((size_t)bmax * max_splits * NH * HD);
    float* part_ml = kb::dalloc<float>((size_t)bmax * max_splits * NH * 2);
    const size_t comp_elems = (size_t)bmax * ncmax * HD + (size_t)bmax * 640 * HD;
    kb::fill_f32(q, (size_t)bmax * NH * HD, 11, -0.5f, 0.5f);
    kb::fill_f16(raw_kv, (size_t)bmax * raw_slots * HD, 12, -1.f, 1.f);
    kb::fill_f16(comp_kv, comp_elems, 13, -1.f, 1.f);
    kb::fill_f32(sinks, NH, 14, -2.f, 2.f);

    // current shape state
    unsigned cur_comp_stride = ncmax;
    bool cur_dense = false;
    unsigned cur_ntmax = ntmax;
    auto set_shape = [&](const Shape& s) {
        const unsigned b = s.nr.size();
        std::vector<int> off(b), cb(b);
        unsigned ntm = 0;
        for (unsigned r = 0; r < b; ++r) {
            off[r] = (int)(r * raw_slots + 7 * (r & 1));   // window rows anywhere in the slot space
            cb[r] = (int)(bmax * ncmax + r * 640);         // dense: per-row base row in the shared store
            ntm = std::max(ntm, (unsigned)(s.nr[r] + s.nc[r]));
        }
        KB_CHECK(hipMemcpy(d_nr, s.nr.data(), b * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_off, off.data(), b * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_nc, s.nc.data(), b * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_cb, cb.data(), b * 4, hipMemcpyHostToDevice));
        cur_dense = s.dense;
        cur_comp_stride = s.dense ? 0u : ncmax;
        cur_ntmax = ntm;
    };
    const int* null_i = nullptr;
    const unsigned* null_u = nullptr;
    auto cbp = [&]() -> const int* { return cur_dense ? d_cb : null_i; };

    auto launch_dec_score = [&](hipFunction_t f, hipStream_t s, unsigned b, uint16_t* sc, unsigned keys_per_wg, unsigned blk) {
        kb::launch(f, dim3((cur_ntmax + keys_per_wg - 1) / keys_per_wg, NH / 16, b), dim3(blk), 0, s,
                   (_Float16*)sc, (const float*)q, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, null_u, MKW, NH, stride, kq_scale,
                   cur_comp_stride, cbp());
    };
    auto launch_mix_score = [&](hipStream_t s, unsigned b, uint16_t* sc) {
        kb::launch(f_mix_score, dim3((cur_ntmax + 255) / 256, 1, b), dim3(512), 0, s,
                   (_Float16*)sc, (const float*)q, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, null_u, MKW, NH, HD, stride, kq_scale,
                   cur_comp_stride, cbp());
    };
    auto launch_smwsum = [&](hipFunction_t f, hipStream_t s, unsigned b, uint16_t* sc, float* o) {
        kb::launch(f, dim3(NH / 16, b, 1), dim3(512), 0, s,
                   o, (_Float16*)sc, (const float*)sinks, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, NH, HD, stride, cur_comp_stride, cbp());
    };
    auto launch_fused = [&](hipFunction_t f, hipStream_t s, unsigned b, float* o) {
        kb::launch(f, dim3(NH / 16, b, 1), dim3(512), 0, s,
                   o, (const float*)sinks, (const float*)q, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, NH, kq_scale, cur_comp_stride, cbp());
    };
    auto launch_flash = [&](const Cand& c, hipStream_t s, unsigned b, float* o) {
        const unsigned splits = (cur_ntmax + c.kpw - 1) / c.kpw;
        kb::launch(c.f, dim3(NH / 16, splits, b), dim3(c.block), 0, s,
                   part_o, part_ml, (const float*)q, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, (const float*)sinks, NH, kq_scale,
                   cur_comp_stride, cbp(), max_splits);
        kb::launch(c.f2, dim3(NH, b, 1), dim3(512), 0, s,
                   o, (const float*)part_o, (const float*)part_ml, (const float*)sinks, (const int*)d_nr,
                   (const int*)d_nc, NH, c.kpw, max_splits);
    };
    auto run_cand = [&](const Cand& c, hipStream_t s, unsigned b) {
        if (c.kind == "score") launch_dec_score(c.f, s, b, scores2, c.kpw, c.block);
        else if (c.kind == "smwsum") { launch_dec_score(f_dec_score, s, b, scores2, 256, 512); launch_smwsum(c.f, s, b, scores2, out2); }
        else if (c.kind == "fused") launch_fused(c.f, s, b, out2);
        else if (c.kind == "flash") launch_flash(c, s, b, out2);
    };

    auto cmp_scores = [&](const char* tag, unsigned b) {
        auto a = kb::d2h(scores_ref, (size_t)b * NH * stride), c = kb::d2h(scores2, (size_t)b * NH * stride);
        size_t nd = 0;
        for (size_t i = 0; i < a.size(); ++i) nd += a[i] != c[i];
        printf("CMP %-40s n=%zu bitexact=%s bit_diff=%zu\n", tag, a.size(), nd ? "no" : "YES", nd);
        printf("KBJSON {\"cmp\":\"%s\",\"bitexact\":%d,\"bit_diff\":%zu}\n", tag, (int)(nd == 0), nd);
    };
    auto cmp_out = [&](const char* tag, unsigned b) {
        auto a = kb::d2h(out, (size_t)b * NH * HD), c = kb::d2h(out2, (size_t)b * NH * HD);
        kb::print_cmp(tag, kb::compare_f32(a.data(), c.data(), a.size()));
    };

    // CPU f64 reference of row 0 (no intermediate rounding) — to place production's and the flash
    // kernels' errors on the same scale.
    auto cpu_ref_row0 = [&](const Shape& sh, std::vector<double>& ref) {
        const unsigned nr = sh.nr[0], nc = sh.nc[0];
        auto hq = kb::d2h(q, (size_t)NH * HD);
        auto hraw = kb::d2h(raw_kv, (size_t)bmax * raw_slots * HD);
        auto hcomp = kb::d2h(comp_kv, comp_elems);
        auto hs = kb::d2h(sinks, NH);
        const unsigned off = 0;  // row 0: off = 0
        const size_t cbase = sh.dense ? (size_t)bmax * ncmax : 0;
        ref.assign((size_t)NH * HD, 0.0);
        std::vector<double> sc(nr + nc);
        for (unsigned h = 0; h < NH; ++h) {
            double m = hs[h];
            for (unsigned k = 0; k < nr + nc; ++k) {
                const uint16_t* row = (k < nr) ? &hraw[(size_t)(off + k) * HD] : &hcomp[(cbase + (k - nr)) * HD];
                double d = 0;
                for (unsigned i = 0; i < HD; ++i) d += (double)hq[(size_t)h * HD + i] * (double)kb::h2f(row[i]);
                sc[k] = d * kq_scale;
                m = std::max(m, sc[k]);
            }
            double l = std::exp(hs[h] - m);
            for (unsigned k = 0; k < nr + nc; ++k) {
                const double w = std::exp(sc[k] - m);
                l += w;
                const uint16_t* row = (k < nr) ? &hraw[(size_t)(off + k) * HD] : &hcomp[(cbase + (k - nr)) * HD];
                for (unsigned i = 0; i < HD; ++i) ref[(size_t)h * HD + i] += w * (double)kb::h2f(row[i]);
            }
            for (unsigned i = 0; i < HD; ++i) ref[(size_t)h * HD + i] /= l;
        }
    };
    auto err_vs_ref = [&](const char* tag, const float* dev, const std::vector<double>& ref) {
        auto g = kb::d2h(dev, (size_t)NH * HD);
        double num = 0, den = 0, maxabs = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double e = g[i] - ref[i];
            num += e * e; den += ref[i] * ref[i]; maxabs = std::max(maxabs, std::fabs(e));
        }
        printf("REF %-40s row0 vs f64 CPU: rel_rmse=%.3e max_abs=%.3e\n", tag, std::sqrt(num / den), maxabs);
        printf("KBJSON {\"ref\":\"%s\",\"rel_rmse\":%.4e,\"max_abs\":%.4e}\n", tag, std::sqrt(num / den), maxabs);
    };

    // correctness block for one shape: production pair -> scores_ref / out; then every candidate
    auto check_shape = [&](const Shape& sh, bool with_cpu_ref) {
        const unsigned b = sh.nr.size();
        set_shape(sh);
        printf("\n######## %s  b=%u n_total_max=%u comp=%s stride=%u\n", sh.tag, b, cur_ntmax, sh.dense ? "dense(base_per)" : "gathered(512)", stride);
        KB_CHECK(hipMemset(scores, 0, (size_t)bmax * NH * stride * 2));
        KB_CHECK(hipMemset(scores2, 0, (size_t)bmax * NH * stride * 2));
        launch_mix_score(0, b, scores2);
        launch_dec_score(f_dec_score, 0, b, scores, 256, 512);
        KB_CHECK(hipDeviceSynchronize());
        KB_CHECK(hipMemcpy(scores_ref, scores, (size_t)bmax * NH * stride * 2, hipMemcpyDeviceToDevice));
        cmp_scores("dec_score vs mixed_score (prod twin)", b);
        launch_smwsum(f_smwsum, 0, b, scores, out);
        KB_CHECK(hipDeviceSynchronize());
        std::vector<double> ref;
        if (with_cpu_ref) { cpu_ref_row0(sh, ref); err_vs_ref("production pair", out, ref); }
        for (auto& c : cands) {
            KB_CHECK(hipMemset(scores2, 0, (size_t)bmax * NH * stride * 2));
            KB_CHECK(hipMemset(out2, 0, (size_t)bmax * NH * HD * 4));
            run_cand(c, 0, b);
            KB_CHECK(hipDeviceSynchronize());
            if (c.kind == "score") cmp_scores((c.name + " scores").c_str(), b);
            else {
                cmp_out((c.name + " out").c_str(), b);
                if (with_cpu_ref) err_vs_ref(c.name.c_str(), out2, ref);
            }
        }
    };

    auto mk = [](std::vector<int> nr, std::vector<int> nc, bool dense, const char* tag) { Shape s; s.nr = nr; s.nc = nc; s.dense = dense; s.tag = tag; return s; };
    auto uni = [](unsigned b, int nr, int nc) { return std::make_pair(std::vector<int>(b, nr), std::vector<int>(b, nc)); };

    if (mode == "check") {
        std::vector<Shape> shapes = {
            mk(std::vector<int>(1, 128), std::vector<int>(1, 512), false, "b=1 128+512 gathered (production main)"),
            mk(std::vector<int>(4, 128), std::vector<int>(4, 512), false, "b=4 128+512 gathered"),
            mk({128, 128, 100, 17},   {512, 500, 512, 33},  false, "b=4 mixed tails gathered"),
            mk({128, 5, 128, 128, 1}, {0, 0, 300, 511, 1},  false, "b=5 n_comp 0 / tiny rows gathered"),
            mk({128, 128, 128},       {512, 300, 16},        true,  "b=3 dense comp store (comp_base_per)"),
            mk({128, 128, 64, 128, 128, 128, 128, 128}, {512, 512, 512, 129, 512, 512, 512, 512}, false, "b=8 gathered"),
            mk({128, 0},              {512, 0},              false, "b=2 with an EMPTY row (n_total=0)"),
        };
        for (auto& sh : shapes) check_shape(sh, sh.nr.size() == 1);
        return 0;
    }

    if (mode == "prof") {
        unsigned b = argc > 4 ? atoi(argv[4]) : 1;
        auto p = uni(b, 128, 512);
        set_shape(mk(p.first, p.second, false, "prof"));
        for (int i = 0; i < 6; ++i) {
            launch_dec_score(f_dec_score, 0, b, scores, 256, 512);
            launch_smwsum(f_smwsum, 0, b, scores, out);
            for (auto& c : cands) run_cand(c, 0, b);
            KB_CHECK(hipDeviceSynchronize());
        }
        printf("prof done b=%u\n", b);
        return 0;
    }

    // ---------------------------------------------------------------- decode timing
    std::vector<unsigned> bs;
    for (int i = 4; i < argc; ++i) bs.push_back(atoi(argv[i]));
    if (bs.empty()) bs = {1, 2, 4, 5, 8};
    for (unsigned b : bs) {
        auto p = uni(b, 128, 512);
        char tg[96]; snprintf(tg, sizeof tg, "decode b=%u 128+512 gathered", b);
        Shape sh = mk(p.first, p.second, false, tg);
        check_shape(sh, false);
        const unsigned n_total = 640;
        const double bytes_pair = (double)b * (2.0 * n_total * HD * 2 + NH * HD * 4 + 3.0 * NH * n_total * 2 + NH * HD * 4);
        const double flops_pair = (double)b * 2.0 * 2.0 * NH * n_total * HD;
        std::vector<kb::Variant> vs = {
            {"base pair: dec_score + smwsum", [&](hipStream_t s) { launch_dec_score(f_dec_score, s, b, scores, 256, 512); launch_smwsum(f_smwsum, s, b, scores, out); }, bytes_pair, flops_pair},
            {"base dec_score alone", [&](hipStream_t s) { launch_dec_score(f_dec_score, s, b, scores, 256, 512); }},
            {"base smwsum alone", [&](hipStream_t s) { launch_smwsum(f_smwsum, s, b, scores, out); }},
        };
        for (auto& c : cands) {
            if (c.kind == "score")
                vs.push_back({c.name + " alone", [&, c](hipStream_t s) { launch_dec_score(c.f, s, b, scores2, c.kpw, c.block); }});
            else if (c.kind == "smwsum")
                vs.push_back({"pair: dec_score + " + c.name, [&, c](hipStream_t s) { run_cand(c, s, b); }, bytes_pair, flops_pair});
            else if (c.kind == "fused")
                vs.push_back({c.name, [&, c](hipStream_t s) { run_cand(c, s, b); }, bytes_pair, flops_pair});
            else if (c.kind == "flash")
                vs.push_back({c.name + " (+combine)", [&, c](hipStream_t s) { run_cand(c, s, b); }, bytes_pair, flops_pair});
        }
        kb::AbOpts o;
        o.graph = true; o.inner = 10; o.rounds = 60;
        char tag[128]; snprintf(tag, sizeof tag, "decode b=%u n_total=%u graph warm", b, n_total);
        o.tag = tag;
        kb::ab(vs, o);
    }
    return 0;
}
