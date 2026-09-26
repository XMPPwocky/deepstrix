// Reviewer harness for D_attention/flash_splitk (flash3 k64/k128, flash2 k128 vs the production pair).
//   ./review_harness <arch> <dir> check2         correctness on EXTRA shapes x input regimes (uniform / peaked /
//                                               very peaked / sink-dominated), every row vs an f64 CPU reference
//   ./review_harness <arch> <dir> combine [b..]  timing: base pair vs flash3 part1 alone vs part1+combine vs combine alone
// Launch contracts copied from attn_harness2.cpp (which match S/attention_dec.rs + S/attention.rs).
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
struct Cand { std::string name; hipFunction_t f = nullptr, f2 = nullptr; unsigned kps = 128; };
struct Shape { std::vector<int> nr, nc; bool dense; const char* tag; };

int main(int argc, char** argv) {
    const std::string arch = argc > 1 ? argv[1] : "gfx1201";
    const std::string dir = argc > 2 ? argv[2] : ".";
    const std::string mode = argc > 3 ? argv[3] : "check2";
    kb::init();
    kb::Module base_mixed(dir + "/base_attention_mixed_" + arch + ".hsaco");
    kb::Module base_dec(dir + "/base_attention_dec_" + arch + ".hsaco");
    hipModule_t cand3 = opt_mod(dir + "/cand3_attn_" + arch + ".hsaco");
    hipModule_t cand4 = opt_mod(dir + "/cand4_attn_" + arch + ".hsaco");
    hipFunction_t f_dec_score = base_dec.fn("attention_dec_score_htiled_wmma_f16s");
    hipFunction_t f_smwsum = base_mixed.fn("attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_f16s");
    std::vector<Cand> cands;
    auto add = [&](hipModule_t m, const char* s1, const char* s2, unsigned kps) {
        Cand c; c.name = s1; c.f = opt_fn(m, s1); c.f2 = opt_fn(m, s2); c.kps = kps;
        if (c.f && c.f2) cands.push_back(c); else fprintf(stderr, "[missing %s]\n", s1);
    };
    add(cand4, "attention_dec_flash3_part1_k128", "attention_dec_flash3_combine", 128);
    add(cand4, "attention_dec_flash3_part1_k64", "attention_dec_flash3_combine", 64);
    add(cand3, "attention_dec_flash2_part1_k128", "attention_dec_flash2_combine", 128);

    const unsigned NH = 64, HD = 512, STRIDE = 3072, MKW = (82176 + 31) / 32;
    const float kq_scale = 1.0f / std::sqrt((float)HD);
    const unsigned bmax = 8, raw_slots = 256, ncmax = 512;
    const unsigned max_splits = 20;
    float* q = kb::dalloc<float>((size_t)bmax * NH * HD);
    uint16_t* raw_kv = kb::dalloc<uint16_t>((size_t)bmax * raw_slots * HD);
    const size_t comp_elems = (size_t)bmax * ncmax * HD + (size_t)bmax * 640 * HD;
    uint16_t* comp_kv = kb::dalloc<uint16_t>(comp_elems);
    uint16_t* scores = kb::dalloc<uint16_t>((size_t)bmax * NH * STRIDE);
    float* out = kb::dalloc<float>((size_t)bmax * NH * HD);
    float* out2 = kb::dalloc<float>((size_t)bmax * NH * HD);
    float* sinks = kb::dalloc<float>(NH);
    int* d_nr = kb::dalloc<int>(bmax);
    int* d_off = kb::dalloc<int>(bmax);
    int* d_nc = kb::dalloc<int>(bmax);
    int* d_cb = kb::dalloc<int>(bmax);
    float* part_o = kb::dalloc<float>((size_t)bmax * max_splits * NH * HD);
    float* part_ml = kb::dalloc<float>((size_t)bmax * max_splits * NH * 2);

    std::vector<float> hq((size_t)bmax * NH * HD), hs(NH);
    std::vector<uint16_t> hraw((size_t)bmax * raw_slots * HD), hcomp(comp_elems);
    auto fill_inputs = [&](float qscale, float sink_lo, float sink_hi, uint32_t seed) {
        kb::fill_f32(q, hq.size(), seed + 11, -0.5f * qscale, 0.5f * qscale);
        kb::fill_f16(raw_kv, hraw.size(), seed + 12, -1.f, 1.f);
        kb::fill_f16(comp_kv, comp_elems, seed + 13, -1.f, 1.f);
        kb::fill_f32(sinks, NH, seed + 14, sink_lo, sink_hi);
        KB_CHECK(hipDeviceSynchronize());
        hq = kb::d2h(q, hq.size()); hraw = kb::d2h(raw_kv, hraw.size()); hcomp = kb::d2h(comp_kv, comp_elems); hs = kb::d2h(sinks, NH);
    };

    unsigned cur_comp_stride = ncmax; bool cur_dense = false; unsigned cur_ntmax = 640;
    std::vector<int> h_off(bmax), h_cb(bmax);
    auto set_shape = [&](const Shape& s) {
        const unsigned b = s.nr.size();
        unsigned ntm = 0;
        for (unsigned r = 0; r < b; ++r) {
            h_off[r] = (int)(r * raw_slots + 7 * (r & 1));
            h_cb[r] = (int)(bmax * ncmax + r * 640);
            ntm = std::max(ntm, (unsigned)(s.nr[r] + s.nc[r]));
        }
        KB_CHECK(hipMemcpy(d_nr, s.nr.data(), b * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_off, h_off.data(), b * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_nc, s.nc.data(), b * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_cb, h_cb.data(), b * 4, hipMemcpyHostToDevice));
        cur_dense = s.dense; cur_comp_stride = s.dense ? 0u : ncmax; cur_ntmax = ntm;
    };
    const int* null_i = nullptr; const unsigned* null_u = nullptr;
    auto cbp = [&]() -> const int* { return cur_dense ? d_cb : null_i; };
    auto launch_pair = [&](hipStream_t s, unsigned b, float* o) {
        kb::launch(f_dec_score, dim3((cur_ntmax + 255) / 256, NH / 16, b), dim3(512), 0, s,
                   (_Float16*)scores, (const float*)q, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, null_u, MKW, NH, STRIDE, kq_scale, cur_comp_stride, cbp());
        kb::launch(f_smwsum, dim3(NH / 16, b, 1), dim3(512), 0, s,
                   o, (_Float16*)scores, (const float*)sinks, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, NH, HD, STRIDE, cur_comp_stride, cbp());
    };
    auto launch_part1 = [&](const Cand& c, hipStream_t s, unsigned b) {
        const unsigned splits = (cur_ntmax + c.kps - 1) / c.kps;
        kb::launch(c.f, dim3(NH / 16, splits, b), dim3(512), 0, s,
                   part_o, part_ml, (const float*)q, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, (const float*)sinks, NH, kq_scale,
                   cur_comp_stride, cbp(), max_splits);
    };
    auto launch_combine = [&](const Cand& c, hipStream_t s, unsigned b, float* o) {
        kb::launch(c.f2, dim3(NH, b, 1), dim3(512), 0, s,
                   o, (const float*)part_o, (const float*)part_ml, (const float*)sinks, (const int*)d_nr,
                   (const int*)d_nc, NH, c.kps, max_splits);
    };

    // f64 CPU reference for ALL rows of the current shape
    auto cpu_ref = [&](const Shape& sh, std::vector<double>& ref) {
        const unsigned b = sh.nr.size();
        ref.assign((size_t)b * NH * HD, 0.0);
        for (unsigned r = 0; r < b; ++r) {
            const unsigned nr = sh.nr[r], nc = sh.nc[r], off = h_off[r];
            const size_t cbase = sh.dense ? (size_t)h_cb[r] : (size_t)r * ncmax;
            std::vector<double> sc(nr + nc);
            for (unsigned h = 0; h < NH; ++h) {
                const float* qh = &hq[((size_t)r * NH + h) * HD];
                double m = hs[h];
                for (unsigned k = 0; k < nr + nc; ++k) {
                    const uint16_t* row = (k < nr) ? &hraw[(size_t)(off + k) * HD] : &hcomp[(cbase + (k - nr)) * HD];
                    double d = 0;
                    for (unsigned i = 0; i < HD; ++i) d += (double)qh[i] * (double)kb::h2f(row[i]);
                    sc[k] = d * kq_scale; m = std::max(m, sc[k]);
                }
                double l = std::exp(hs[h] - m);
                double* o = &ref[((size_t)r * NH + h) * HD];
                for (unsigned k = 0; k < nr + nc; ++k) {
                    const double w = std::exp(sc[k] - m); l += w;
                    const uint16_t* row = (k < nr) ? &hraw[(size_t)(off + k) * HD] : &hcomp[(cbase + (k - nr)) * HD];
                    for (unsigned i = 0; i < HD; ++i) o[i] += w * (double)kb::h2f(row[i]);
                }
                for (unsigned i = 0; i < HD; ++i) o[i] /= l;
            }
        }
    };
    auto err_vs_ref = [&](const char* tag, const float* dev, const std::vector<double>& ref) {
        auto g = kb::d2h(dev, ref.size());
        double num = 0, den = 0, maxabs = 0; size_t nonfinite = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (!std::isfinite(g[i])) { nonfinite++; continue; }
            const double e = g[i] - ref[i]; num += e * e; den += ref[i] * ref[i]; maxabs = std::max(maxabs, std::fabs(e));
        }
        printf("REF %-36s all rows vs f64: rel_rmse=%.3e max_abs=%.3e nonfinite=%zu\n", tag, std::sqrt(num / std::max(den, 1e-300)), maxabs, nonfinite);
        printf("KBJSON {\"ref\":\"%s\",\"rel_rmse\":%.4e,\"max_abs\":%.4e,\"nonfinite\":%zu}\n", tag, std::sqrt(num / std::max(den, 1e-300)), maxabs, nonfinite);
    };
    auto mk = [](std::vector<int> nr, std::vector<int> nc, bool dense, const char* tag) { Shape s; s.nr = nr; s.nc = nc; s.dense = dense; s.tag = tag; return s; };

    if (mode == "check2") {
        std::vector<Shape> shapes = {
            mk({128}, {512}, false, "b=1 128+512 (production main)"),
            mk({128}, {1},   false, "b=1 128+1 tail split (nk=1)"),
            mk({100}, {37},  false, "b=1 100+37 raw/comp straddle inside a split"),
            mk({17},  {0},   false, "b=1 17+0 (n_total < 32)"),
            mk({128}, {511}, false, "b=1 128+511 odd n_total"),
            mk({0},   {64},  false, "b=1 0+64 no window"),
            mk({128, 128, 64, 128, 1, 128, 128}, {512, 300, 512, 0, 1, 17, 255}, false, "b=7 mixed tails gathered"),
            mk(std::vector<int>(6, 128), std::vector<int>(6, 512), false, "b=6 128+512"),
            mk(std::vector<int>(8, 128), {512, 511, 1, 0, 33, 64, 129, 512}, false, "b=8 (max production b) tails"),
            mk({128, 128, 100, 128}, {512, 40, 500, 3}, true, "b=4 dense comp store tails"),
        };
        struct Regime { const char* name; float qscale, slo, shi; };
        std::vector<Regime> regimes = {
            {"uniform (engineer: q +-0.5, score std ~0.17)", 1.f, -2.f, 2.f},
            {"peaked (q x40, score std ~7)", 40.f, -2.f, 2.f},
            {"very peaked (q x120, score std ~20; f16 weight underflow)", 120.f, -2.f, 2.f},
            {"sink-dominated (q x40, sinks +20..+30)", 40.f, 20.f, 30.f},
        };
        for (auto& rg : regimes) {
            printf("\n================ regime: %s\n", rg.name);
            fill_inputs(rg.qscale, rg.slo, rg.shi, 1000);
            for (auto& sh : shapes) {
                const unsigned b = sh.nr.size();
                set_shape(sh);
                printf("\n######## %s  b=%u n_total_max=%u comp=%s\n", sh.tag, b, cur_ntmax, sh.dense ? "dense(base_per)" : "gathered(512)");
                KB_CHECK(hipMemset(scores, 0, (size_t)bmax * NH * STRIDE * 2));
                KB_CHECK(hipMemset(out, 0, (size_t)bmax * NH * HD * 4));
                launch_pair(0, b, out);
                KB_CHECK(hipDeviceSynchronize());
                std::vector<double> ref; cpu_ref(sh, ref);
                err_vs_ref("production pair", out, ref);
                auto hout = kb::d2h(out, (size_t)b * NH * HD);
                for (auto& c : cands) {
                    KB_CHECK(hipMemset(out2, 0, (size_t)bmax * NH * HD * 4));
                    KB_CHECK(hipMemset(part_o, 0x7f, (size_t)bmax * max_splits * NH * HD * 4));   // poison: unwritten partials -> NaN
                    KB_CHECK(hipMemset(part_ml, 0x7f, (size_t)bmax * max_splits * NH * 2 * 4));
                    launch_part1(c, 0, b);
                    launch_combine(c, 0, b, out2);
                    KB_CHECK(hipDeviceSynchronize());
                    auto h2 = kb::d2h(out2, (size_t)b * NH * HD);
                    kb::print_cmp((c.name + " vs prod").c_str(), kb::compare_f32(hout.data(), h2.data(), h2.size()));
                    err_vs_ref(c.name.c_str(), out2, ref);
                }
            }
        }
        return 0;
    }

    if (mode == "combine") {
        fill_inputs(1.f, -2.f, 2.f, 1000);
        std::vector<unsigned> bs;
        for (int i = 4; i < argc; ++i) bs.push_back(atoi(argv[i]));
        if (bs.empty()) bs = {1, 4};
        for (unsigned b : bs) {
            Shape sh = mk(std::vector<int>(b, 128), std::vector<int>(b, 512), false, "timing");
            set_shape(sh);
            std::vector<kb::Variant> vs = {
                {"base pair: dec_score + smwsum", [&](hipStream_t s) { launch_pair(s, b, out); }},
            };
            for (auto& c : cands) {
                vs.push_back({c.name + " part1 + combine", [&, c](hipStream_t s) { launch_part1(c, s, b); launch_combine(c, s, b, out2); }});
                vs.push_back({c.name + " part1 alone", [&, c](hipStream_t s) { launch_part1(c, s, b); }});
                vs.push_back({c.name + " combine alone", [&, c](hipStream_t s) { launch_combine(c, s, b, out2); }});
            }
            kb::AbOpts o; o.graph = true; o.inner = 10; o.rounds = 60;
            char tag[128]; snprintf(tag, sizeof tag, "review combine split b=%u n_total=640 graph warm", b);
            o.tag = tag;
            kb::ab(vs, o);
        }
        return 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
