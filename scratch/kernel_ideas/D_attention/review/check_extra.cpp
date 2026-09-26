// Reviewer extra-shape correctness check for D_attention/score_blk128 (and blk256, blk64):
// shapes the engineer did NOT test: b = 16 (max production b for attn_dec_score_for), odd row
// counts, small n_total_max (grid.x differs between 64- and 256-key blocks), no raw window,
// a non-null comp_allowed_bits mask (production passes null at decode, but the kernel has the path),
// dense comp store at b = 16, and a row whose n_total is not a multiple of 16.
// Reference = production attention_dec_score_htiled_wmma_f16s (grid (ceil(n/256), 4, b) x 512),
// cross-checked against attention_mixed_score_batched_htiled_wmma_f16s (the production twin).
//   ./check_extra <arch> <dir>
#include "kbench.h"
#include <cmath>
#include <cstring>

struct Cand { const char* name; hipFunction_t f; unsigned kpw, blk; };
struct Shape { std::vector<int> nr, nc; bool dense; bool mask; const char* tag; };

int main(int argc, char** argv) {
    const std::string arch = argc > 1 ? argv[1] : "gfx1201";
    const std::string dir = argc > 2 ? argv[2] : ".";
    kb::init();
    kb::Module base_mixed(dir + "/base_attention_mixed_" + arch + ".hsaco");
    kb::Module base_dec(dir + "/base_attention_dec_" + arch + ".hsaco");
    kb::Module cand2(dir + "/cand2_attn_" + arch + ".hsaco");
    hipFunction_t f_dec = base_dec.fn("attention_dec_score_htiled_wmma_f16s");
    hipFunction_t f_mix = base_mixed.fn("attention_mixed_score_batched_htiled_wmma_f16s");
    std::vector<Cand> cands = {
        {"attention_dec_score_blk256", cand2.fn("attention_dec_score_blk256"), 128, 256},
        {"attention_dec_score_blk128", cand2.fn("attention_dec_score_blk128"), 64, 128},
        {"attention_dec_score_blk64", cand2.fn("attention_dec_score_blk64"), 32, 64},
    };
    const unsigned NH = 64, HD = 512, STRIDE = 3072, MKW = (82176 + 31) / 32;
    const float kq_scale = 1.0f / std::sqrt((float)HD);
    const unsigned bmax = 16, raw_slots = 256, ncmax = 512;
    float* q = kb::dalloc<float>((size_t)bmax * NH * HD);
    uint16_t* raw_kv = kb::dalloc<uint16_t>((size_t)bmax * raw_slots * HD);
    const size_t comp_elems = (size_t)bmax * ncmax * HD + (size_t)bmax * 640 * HD;
    uint16_t* comp_kv = kb::dalloc<uint16_t>(comp_elems);
    uint16_t* s_ref = kb::dalloc<uint16_t>((size_t)bmax * NH * STRIDE);
    uint16_t* s_mix = kb::dalloc<uint16_t>((size_t)bmax * NH * STRIDE);
    uint16_t* s_c = kb::dalloc<uint16_t>((size_t)bmax * NH * STRIDE);
    unsigned* d_mask = kb::dalloc<unsigned>((size_t)bmax * MKW);
    int* d_nr = kb::dalloc<int>(bmax);
    int* d_off = kb::dalloc<int>(bmax);
    int* d_nc = kb::dalloc<int>(bmax);
    int* d_cb = kb::dalloc<int>(bmax);
    kb::fill_f32(q, (size_t)bmax * NH * HD, 211, -0.5f, 0.5f);
    kb::fill_f16(raw_kv, (size_t)bmax * raw_slots * HD, 212, -1.f, 1.f);
    kb::fill_f16(comp_kv, comp_elems, 213, -1.f, 1.f);
    kb::fill_bytes(d_mask, (size_t)bmax * MKW * 4, 214);
    // make some q rows large so a few scores saturate / round differently in f16 (non-benign data)
    {
        auto hq = kb::d2h(q, (size_t)bmax * NH * HD);
        for (size_t i = 0; i < hq.size(); ++i) if ((i / HD) % 5 == 0) hq[i] *= 40.f;
        KB_CHECK(hipMemcpy(q, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
    }
    const size_t sbytes = (size_t)bmax * NH * STRIDE * 2;
    size_t total_bad = 0;
    auto run_shape = [&](const Shape& sh) {
        const unsigned b = sh.nr.size();
        std::vector<int> off(b), cb(b);
        unsigned ntm = 0;
        for (unsigned r = 0; r < b; ++r) {
            off[r] = (int)(r * raw_slots + 7 * (r & 1));
            cb[r] = (int)(bmax * ncmax + r * 640);
            ntm = std::max(ntm, (unsigned)(sh.nr[r] + sh.nc[r]));
        }
        KB_CHECK(hipMemcpy(d_nr, sh.nr.data(), b * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_off, off.data(), b * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_nc, sh.nc.data(), b * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_cb, cb.data(), b * 4, hipMemcpyHostToDevice));
        const unsigned cstride = sh.dense ? 0u : ncmax;
        const int* cbp = sh.dense ? d_cb : nullptr;
        const unsigned* mp = sh.mask ? d_mask : nullptr;
        printf("\n######## %s  b=%u n_total_max=%u comp=%s mask=%s\n", sh.tag, b, ntm,
               sh.dense ? "dense(base_per)" : "gathered(512)", sh.mask ? "yes" : "null");
        if (ntm == 0) { printf("  (n_total_max = 0: production launcher returns early; skipped)\n"); return; }
        auto launch_score = [&](hipFunction_t f, uint16_t* sc, unsigned kpw, unsigned blk) {
            kb::launch(f, dim3((ntm + kpw - 1) / kpw, NH / 16, b), dim3(blk), 0, 0,
                       (_Float16*)sc, (const float*)q, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                       (const int*)d_nr, (const int*)d_off, (const int*)d_nc, mp, MKW, NH, STRIDE, kq_scale,
                       cstride, cbp);
        };
        // poison all score buffers with a sentinel (not 0) so a skipped store is visible
        KB_CHECK(hipMemset(s_ref, 0x7C, sbytes));   // 0x7C7C = f16 NaN pattern
        KB_CHECK(hipMemset(s_mix, 0x7C, sbytes));
        launch_score(f_dec, s_ref, 256, 512);
        kb::launch(f_mix, dim3((ntm + 255) / 256, 1, b), dim3(512), 0, 0,
                   (_Float16*)s_mix, (const float*)q, (const _Float16*)raw_kv, (const _Float16*)comp_kv,
                   (const int*)d_nr, (const int*)d_off, (const int*)d_nc, mp, MKW, NH, HD, STRIDE, kq_scale,
                   cstride, cbp);
        KB_CHECK(hipDeviceSynchronize());
        auto ref = kb::d2h(s_ref, (size_t)b * NH * STRIDE);
        auto mix = kb::d2h(s_mix, (size_t)b * NH * STRIDE);
        size_t nd = 0, written = 0, ninf = 0;
        for (size_t i = 0; i < ref.size(); ++i) { nd += ref[i] != mix[i]; written += ref[i] != 0x7C7C; ninf += ref[i] == 0xFC00; }
        size_t expect = 0;
        for (unsigned r = 0; r < b; ++r) expect += (size_t)NH * (sh.nr[r] + sh.nc[r]);
        printf("CMP %-42s bitexact=%s bit_diff=%zu written=%zu expected=%zu neg_inf=%zu\n",
               "dec_score vs mixed_score (prod twin)", nd ? "no" : "YES", nd, written, expect, ninf);
        if (written != expect) { printf("  !! production twin wrote %zu != %zu expected slots\n", written, expect); }
        for (auto& c : cands) {
            KB_CHECK(hipMemset(s_c, 0x7C, sbytes));
            launch_score(c.f, s_c, c.kpw, c.blk);
            KB_CHECK(hipDeviceSynchronize());
            auto got = kb::d2h(s_c, (size_t)b * NH * STRIDE);
            size_t d = 0, w = 0; size_t first = (size_t)-1;
            for (size_t i = 0; i < ref.size(); ++i) { if (ref[i] != got[i]) { d++; if (first == (size_t)-1) first = i; } w += got[i] != 0x7C7C; }
            printf("CMP %-42s bitexact=%s bit_diff=%zu written=%zu (ref %zu)", c.name, d ? "no" : "YES", d, w, written);
            if (d) printf(" first_diff_idx=%zu (b=%zu h=%zu key=%zu ref=%04x got=%04x)", first, first / (NH * STRIDE),
                          (first / STRIDE) % NH, first % STRIDE, ref[first], got[first]);
            printf("\n");
            printf("KBJSON {\"cmp\":\"%s %s\",\"bitexact\":%d,\"bit_diff\":%zu,\"written_ok\":%d}\n", c.name, sh.tag,
                   (int)(d == 0), d, (int)(w == written));
            total_bad += d + (w != written);
        }
    };
    auto mk = [](std::vector<int> nr, std::vector<int> nc, bool dense, bool mask, const char* tag) {
        Shape s; s.nr = nr; s.nc = nc; s.dense = dense; s.mask = mask; s.tag = tag; return s; };
    std::vector<Shape> shapes = {
        mk(std::vector<int>(16, 128), std::vector<int>(16, 512), false, false, "b=16 128+512 gathered (max production b)"),
        mk(std::vector<int>(16, 128), std::vector<int>(16, 512), true, false, "b=16 128+512 dense store"),
        mk({128}, {1}, false, false, "b=1 128+1 (n_total_max=129: 3 blk128 WGs vs 1 base WG)"),
        mk({128, 128, 17, 128, 64, 128, 3}, {33, 0, 512, 100, 7, 511, 0}, false, false, "b=7 odd rows, ragged tails"),
        mk(std::vector<int>(13, 128), std::vector<int>(13, 33), false, false, "b=13 128+33 (n_total_max=161)"),
        mk({128, 128, 128}, {512, 512, 512}, false, true, "b=3 128+512 with comp_allowed_bits mask"),
        mk({128, 100, 5}, {512, 300, 44}, true, true, "b=3 dense store + mask, ragged"),
        mk({0}, {512}, false, false, "b=1 no raw window (0+512)"),
        mk({128, 128}, {64, 64}, false, false, "b=2 128+64 (n_total=192 = 3x64, not a multiple of 256)"),
        mk({128, 0, 128, 128, 1, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
           {512, 0, 511, 65, 0, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 129}, false, false,
           "b=16 with an empty row and ragged tails"),
        mk({128}, {17}, false, true, "b=1 128+17 masked (n_total=145, 16-key tile partially past n_total)"),
        mk({9}, {0}, false, false, "b=1 9 keys only (single partial tile)"),
    };
    for (auto& sh : shapes) run_shape(sh);
    printf("\nTOTAL_BAD=%zu\n", total_bad);
    printf("KBJSON {\"cmp\":\"reviewer extra shapes all candidates\",\"bitexact\":%d,\"bit_diff\":%zu}\n", (int)(total_bad == 0), total_bad);
    return total_bad ? 1 : 0;
}
