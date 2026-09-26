// D_attention harness: production decode attention pair (attention_dec score + smwsum _ldsv_f16s)
// at production decode shapes, plus candidate kernels loaded from cand_attn_<arch>.hsaco.
//
//   ./attn_harness <arch> <dir> decode  [b...]      graph-mode A/B at decode shapes (default b = 1 2 4 5 8)
//   ./attn_harness <arch> <dir> prof    <b>         a handful of direct launches (for ATT / trace)
//   ./attn_harness <arch> <dir> prefill <b> <ncomp> prefill regime (dense shared comp store, flushed)
//
// Candidate kinds (looked up by symbol; missing symbols are skipped):
//   "score"  : same signature as attention_dec_score_htiled_wmma_f16s
//   "smwsum" : same signature as attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_f16s
//   "fused"  : (out, sinks, q, raw_kv, comp_kv, n_raw_per, n_raw_offset_per, n_comp_per,
//               n_head, kq_scale, comp_kv_batch_stride, comp_base_per)  grid (4, b) x 512
//   "split"  : flash-decoding split-K: part 1 (partial o/m/l per split) + part 2 (combine)
#include "kbench.h"

#include <cmath>
#include <map>

static hipFunction_t opt_fn(hipModule_t m, const char* name) {
    hipFunction_t f = nullptr;
    if (!m) return nullptr;
    if (hipModuleGetFunction(&f, m, name) != hipSuccess) return nullptr;
    return f;
}

struct Cand {
    std::string name, kind;
    hipFunction_t f = nullptr, f2 = nullptr;
    unsigned grid_x = 0;   // score: keys per WG (0 = 256); split: keys per split
    unsigned block = 512;
};

int main(int argc, char** argv) {
    const std::string arch = argc > 1 ? argv[1] : "gfx1201";
    const std::string dir = argc > 2 ? argv[2] : ".";
    const std::string mode = argc > 3 ? argv[3] : "decode";
    kb::init();
    kb::Module base_mixed(dir + "/base_attention_mixed_" + arch + ".hsaco");
    kb::Module base_dec(dir + "/base_attention_dec_" + arch + ".hsaco");
    hipModule_t cand_m = nullptr;
    {
        std::string p = dir + "/cand_attn_" + arch + ".hsaco";
        if (hipModuleLoad(&cand_m, p.c_str()) != hipSuccess) { cand_m = nullptr; fprintf(stderr, "[no candidate module %s]\n", p.c_str()); }
    }
    hipFunction_t f_dec_score = base_dec.fn("attention_dec_score_htiled_wmma_f16s");
    hipFunction_t f_mix_score = base_mixed.fn("attention_mixed_score_batched_htiled_wmma_f16s");
    hipFunction_t f_smwsum = base_mixed.fn("attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_f16s");

    // Candidate table (symbol -> kind). Missing symbols are silently skipped.
    std::vector<Cand> cands;
    auto add = [&](const char* sym, const char* kind, unsigned gx = 0, unsigned blk = 512, const char* sym2 = nullptr) {
        Cand c; c.name = sym; c.kind = kind; c.f = opt_fn(cand_m, sym); c.grid_x = gx; c.block = blk;
        if (sym2) c.f2 = opt_fn(cand_m, sym2);
        if (c.f && (!sym2 || c.f2)) cands.push_back(c);
    };
    add("attention_dec_score_g8", "score");
    add("attention_dec_score_g32", "score");
    add("attention_dec_score_blk256", "score", 128, 256);
    add("attention_dec_smwsum_pf2", "smwsum");
    add("attention_dec_smwsum_pf4", "smwsum");
    add("attention_dec_smwsum_regpf", "smwsum");
    add("attention_dec_fused_f16s", "fused");
    add("attention_dec_fused_pf_f16s", "fused");
    add("attention_dec_split_part1", "split", 128, 512, "attention_dec_split_combine");
    add("attention_dec_split64_part1", "split", 64, 512, "attention_dec_split_combine");
    for (auto& c : cands) fprintf(stderr, "[cand] %s (%s)\n", c.name.c_str(), c.kind.c_str());

    const unsigned NH = 64, HD = 512, STRIDE = 3072, MKW = (82176 + 31) / 32;
    const float kq_scale = 1.0f / std::sqrt((float)HD);

    // ---------------------------------------------------------------- buffers (max b = 8)
    unsigned bmax = 8, n_raw = 128, raw_slots = 256, n_comp = 512;
    unsigned comp_stride = n_comp;  // gathered active_comp_kv: [b, 512, 512]
    unsigned n_total = n_raw + n_comp;
    if (mode == "prefill") {
        bmax = argc > 4 ? atoi(argv[4]) : 32;
        n_comp = argc > 5 ? atoi(argv[5]) : 16384;
        comp_stride = 0;  // dense shared store
        n_total = n_raw + n_comp;
    }
    const unsigned stride = std::max(STRIDE, n_total);
    float* q = kb::dalloc<float>((size_t)bmax * NH * HD);
    uint16_t* raw_kv = kb::dalloc<uint16_t>((size_t)bmax * raw_slots * HD);
    uint16_t* comp_kv = kb::dalloc<uint16_t>((size_t)std::max(bmax * n_comp, n_comp) * HD);
    uint16_t* scores = kb::dalloc<uint16_t>((size_t)bmax * NH * stride);
    uint16_t* scores2 = kb::dalloc<uint16_t>((size_t)bmax * NH * stride);
    float* out = kb::dalloc<float>((size_t)bmax * NH * HD);
    float* out2 = kb::dalloc<float>((size_t)bmax * NH * HD);
    float* sinks = kb::dalloc<float>(NH);
    int* d_nr = kb::dalloc<int>(bmax);
    int* d_off = kb::dalloc<int>(bmax);
    int* d_nc = kb::dalloc<int>(bmax);
    // split-K scratch: partial o [b, splits, 64, 512] f32 + m,l [b, splits, 64]
    const unsigned max_splits = 16;
    float* part_o = kb::dalloc<float>((size_t)bmax * max_splits * NH * HD);
    float* part_ml = kb::dalloc<float>((size_t)bmax * max_splits * NH * 2);
    kb::fill_f32(q, (size_t)bmax * NH * HD, 11, -0.5f, 0.5f);
    kb::fill_f16(raw_kv, (size_t)bmax * raw_slots * HD, 12, -1.f, 1.f);
    kb::fill_f16(comp_kv, (size_t)std::max(bmax * n_comp, n_comp) * HD, 13, -1.f, 1.f);
    kb::fill_f32(sinks, NH, 14, -2.f, 2.f);
    {
        std::vector<int> nr(bmax, (int)n_raw), off(bmax), nc(bmax, (int)n_comp);
        for (unsigned r = 0; r < bmax; ++r) off[r] = (int)(r * raw_slots);
        KB_CHECK(hipMemcpy(d_nr, nr.data(), bmax * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_off, off.data(), bmax * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_nc, nc.data(), bmax * 4, hipMemcpyHostToDevice));
    }
    const int* null_i = nullptr;
    const unsigned* null_u = nullptr;

    auto launch_dec_score = [&](hipFunction_t f, hipStream_t s, unsigned b, uint16_t* sc, unsigned keys_per_wg, unsigned blk) {
        kb::launch(f, dim3((n_total + keys_per_wg - 1) / keys_per_wg, NH / 16, b), dim3(blk), 0, s,
                   (_Float16*)sc, (const float*)q, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, null_u, MKW, NH, stride, kq_scale,
                   comp_stride, null_i);
    };
    auto launch_mix_score = [&](hipStream_t s, unsigned b, uint16_t* sc) {
        kb::launch(f_mix_score, dim3((n_total + 255) / 256, 1, b), dim3(512), 0, s,
                   (_Float16*)sc, (const float*)q, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, null_u, MKW, NH, HD, stride, kq_scale,
                   comp_stride, null_i);
    };
    auto launch_smwsum = [&](hipFunction_t f, hipStream_t s, unsigned b, uint16_t* sc, float* o) {
        kb::launch(f, dim3(NH / 16, b, 1), dim3(512), 0, s,
                   o, (_Float16*)sc, (const float*)sinks, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, NH, HD, stride, comp_stride, null_i);
    };
    auto launch_fused = [&](hipFunction_t f, hipStream_t s, unsigned b, float* o) {
        kb::launch(f, dim3(NH / 16, b, 1), dim3(512), 0, s,
                   o, (const float*)sinks, (const float*)q, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, NH, kq_scale, comp_stride, null_i);
    };
    auto launch_split = [&](const Cand& c, hipStream_t s, unsigned b, float* o) {
        const unsigned splits = (n_total + c.grid_x - 1) / c.grid_x;
        kb::launch(c.f, dim3(NH / 16, splits, b), dim3(c.block), 0, s,
                   part_o, part_ml, (const float*)q, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, NH, kq_scale, comp_stride, null_i,
                   c.grid_x, max_splits);
        kb::launch(c.f2, dim3(NH, b, 1), dim3(256), 0, s,
                   o, (const float*)part_o, (const float*)part_ml, (const float*)sinks, (const int*)d_nr,
                   (const int*)d_nc, NH, c.grid_x, max_splits);
    };

    auto cmp_scores = [&](const char* tag, unsigned b) {
        auto a = kb::d2h(scores, (size_t)b * NH * stride), c = kb::d2h(scores2, (size_t)b * NH * stride);
        size_t nd = 0;
        for (size_t i = 0; i < a.size(); ++i) nd += a[i] != c[i];
        printf("CMP %-34s n=%zu bitexact=%s bit_diff=%zu\n", tag, a.size(), nd ? "no" : "YES", nd);
        printf("KBJSON {\"cmp\":\"%s\",\"bitexact\":%d,\"bit_diff\":%zu}\n", tag, (int)(nd == 0), nd);
    };
    auto cmp_out = [&](const char* tag, unsigned b) {
        auto a = kb::d2h(out, (size_t)b * NH * HD), c = kb::d2h(out2, (size_t)b * NH * HD);
        kb::print_cmp(tag, kb::compare_f32(a.data(), c.data(), a.size()));
    };

    if (mode == "prof") {
        unsigned b = argc > 4 ? atoi(argv[4]) : 1;
        for (int i = 0; i < 6; ++i) {
            launch_dec_score(f_dec_score, 0, b, scores, 256, 512);
            launch_smwsum(f_smwsum, 0, b, scores, out);
            for (auto& c : cands) {
                if (c.kind == "smwsum") launch_smwsum(c.f, 0, b, scores, out2);
                if (c.kind == "fused") launch_fused(c.f, 0, b, out2);
                if (c.kind == "split") launch_split(c, 0, b, out2);
                if (c.kind == "score") launch_dec_score(c.f, 0, b, scores2, c.grid_x ? c.grid_x : 256, c.block);
            }
            KB_CHECK(hipDeviceSynchronize());
        }
        printf("prof done b=%u\n", b);
        return 0;
    }

    if (mode == "prefill") {
        const unsigned b = bmax;
        const double kv_bytes = (double)n_total * HD * 2, sc_bytes = (double)b * NH * n_total * 2;
        // correctness of the mixed-score kernel vs itself is moot; check smwsum output stability only
        kb::Flusher flush(96u << 20);
        std::vector<kb::Variant> vs = {
            {"base mixed score (prefill)", [&](hipStream_t s) { launch_mix_score(s, b, scores); },
             kv_bytes + sc_bytes + (double)b * NH * HD * 4, 2.0 * b * NH * n_total * HD},
            {"base smwsum ldsv_f16s (prefill)", [&](hipStream_t s) { launch_smwsum(f_smwsum, s, b, scores, out); },
             (double)b * kv_bytes + 2 * sc_bytes + (double)b * NH * HD * 4, 2.0 * b * NH * n_total * HD},
        };
        kb::AbOpts o;
        o.graph = false; o.inner = 2; o.rounds = 15; o.between = std::ref(flush);
        char tag[128]; snprintf(tag, sizeof tag, "prefill b=%u n_total=%u stride=%u flushed", b, n_total, stride);
        o.tag = tag;
        kb::ab(vs, o);
        return 0;
    }

    // ---------------------------------------------------------------- decode
    std::vector<unsigned> bs;
    for (int i = 4; i < argc; ++i) bs.push_back(atoi(argv[i]));
    if (bs.empty()) bs = {1, 2, 4, 5, 8};
    for (unsigned b : bs) {
        printf("\n######## decode b=%u n_raw=%u n_comp=%u (gathered, stride %u) n_total=%u scores stride %u\n",
               b, n_raw, n_comp, comp_stride, n_total, stride);
        // --- correctness: production pair = dec score + smwsum
        KB_CHECK(hipMemset(scores, 0, (size_t)bmax * NH * stride * 2));
        KB_CHECK(hipMemset(scores2, 0, (size_t)bmax * NH * stride * 2));
        launch_mix_score(0, b, scores2);
        launch_dec_score(f_dec_score, 0, b, scores, 256, 512);
        KB_CHECK(hipDeviceSynchronize());
        cmp_scores("dec_score vs mixed_score (prod twin)", b);
        launch_smwsum(f_smwsum, 0, b, scores, out);
        KB_CHECK(hipDeviceSynchronize());
        auto ref_out = kb::d2h(out, (size_t)b * NH * HD);
        for (auto& c : cands) {
            KB_CHECK(hipMemset(scores2, 0, (size_t)bmax * NH * stride * 2));
            KB_CHECK(hipMemset(out2, 0, (size_t)bmax * NH * HD * 4));
            if (c.kind == "score") {
                launch_dec_score(c.f, 0, b, scores2, c.grid_x ? c.grid_x : 256, c.block);
                KB_CHECK(hipDeviceSynchronize());
                cmp_scores((c.name + " scores").c_str(), b);
            } else if (c.kind == "smwsum") {
                launch_dec_score(f_dec_score, 0, b, scores2, 256, 512);
                launch_smwsum(c.f, 0, b, scores2, out2);
                KB_CHECK(hipDeviceSynchronize());
                cmp_scores((c.name + " weights").c_str(), b);
                cmp_out((c.name + " out").c_str(), b);
            } else if (c.kind == "fused") {
                launch_fused(c.f, 0, b, out2);
                KB_CHECK(hipDeviceSynchronize());
                cmp_out((c.name + " out").c_str(), b);
            } else if (c.kind == "split") {
                launch_split(c, 0, b, out2);
                KB_CHECK(hipDeviceSynchronize());
                cmp_out((c.name + " out").c_str(), b);
            }
        }
        // --- timing, graph mode, warm (decode operands are small and warm)
        const double bytes_pair = (double)b * (2.0 * n_total * HD * 2 + NH * HD * 4 + 3.0 * NH * n_total * 2 + NH * HD * 4);
        std::vector<kb::Variant> vs = {
            {"base pair: dec_score + smwsum", [&](hipStream_t s) { launch_dec_score(f_dec_score, s, b, scores, 256, 512); launch_smwsum(f_smwsum, s, b, scores, out); }, bytes_pair},
            {"base dec_score alone", [&](hipStream_t s) { launch_dec_score(f_dec_score, s, b, scores, 256, 512); }},
            {"base smwsum alone", [&](hipStream_t s) { launch_smwsum(f_smwsum, s, b, scores, out); }},
        };
        for (auto& c : cands) {
            if (c.kind == "score")
                vs.push_back({c.name + " alone", [&, c](hipStream_t s) { launch_dec_score(c.f, s, b, scores2, c.grid_x ? c.grid_x : 256, c.block); }});
            else if (c.kind == "smwsum")
                vs.push_back({"pair: dec_score + " + c.name, [&, c](hipStream_t s) { launch_dec_score(f_dec_score, s, b, scores2, 256, 512); launch_smwsum(c.f, s, b, scores2, out2); }, bytes_pair});
            else if (c.kind == "fused")
                vs.push_back({c.name, [&, c](hipStream_t s) { launch_fused(c.f, s, b, out2); }, bytes_pair});
            else if (c.kind == "split")
                vs.push_back({c.name + " (2 launches)", [&, c](hipStream_t s) { launch_split(c, s, b, out2); }, bytes_pair});
        }
        kb::AbOpts o;
        o.graph = true; o.inner = 10; o.rounds = 60;
        char tag[128]; snprintf(tag, sizeof tag, "decode b=%u n_total=%u graph warm", b, n_total);
        o.tag = tag;
        kb::ab(vs, o);
    }
    return 0;
}
