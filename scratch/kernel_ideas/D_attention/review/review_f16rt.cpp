// Reviewer harness for D_attention/drop_f16_roundtrip.
// Production chain (FP:4024 -> 4038 -> 4187/4204): fp8_act_quant_inplace(kv_normed) ; f16_roundtrip(kv_normed) ;
// kv_cache_append_batched(cache, kv_normed, n_raw_before, 512, slot_per|null).
// Claim: the f16 cache is bit-identical with and without the f16_roundtrip launch.
// Extra coverage vs the engineer's test: odd b (1,3,5,7,8), the arena slot_per path, non-zero n_raw_before,
// b=512 prefill, and adversarial inputs: exact f16 rounding ties, f16 subnormals, overflow (>65504),
// -0, +-inf, NaN, and raw (NOT fp8-quantized) values so the no-op proof does not lean on the FP8 window.
//   ./review_f16rt <arch> <dir>
#include "kbench.h"
#include <cmath>
#include <cstring>
#include <random>

static float bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    const std::string arch = argc > 1 ? argv[1] : "gfx1201";
    const std::string dir = argc > 2 ? argv[2] : ".";
    kb::init();
    kb::Module m_fp4(dir + "/base_fp4_kv_quant_" + arch + ".hsaco");
    kb::Module m_rt(dir + "/base_f16_roundtrip_" + arch + ".hsaco");
    kb::Module m_app(dir + "/base_kv_cache_append_" + arch + ".hsaco");
    hipFunction_t f_fp8 = m_fp4.fn("fp8_act_quant_inplace");
    hipFunction_t f_rt = m_rt.fn("f16_roundtrip");
    hipFunction_t f_app = m_app.fn("kv_cache_append_batched");
    const unsigned W = 512, BMAX = 512, CACHE_ROWS = 128 + 512 + 64;
    float* x = kb::dalloc<float>((size_t)BMAX * W);
    float* y = kb::dalloc<float>((size_t)BMAX * W);
    uint16_t* ca = kb::dalloc<uint16_t>((size_t)CACHE_ROWS * W);
    uint16_t* cb = kb::dalloc<uint16_t>((size_t)CACHE_ROWS * W);
    int* slots = kb::dalloc<int>(BMAX);
    std::mt19937 rng(7);
    size_t grand_diff = 0, grand_rt = 0, cases = 0;

    // adversarial value generators
    auto gen = [&](int kind, size_t i, unsigned r) -> float {
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        switch (kind) {
        case 0: return u(rng) * std::ldexp(1.f, -(int)(r % 24));                        // mixed magnitude
        case 1: { // exact f16 rounding ties: f16 has 10 mantissa bits; a tie is bit 12 set, bits 0..11 clear in f32
            uint32_t e = 113 + (rng() % 28);  // biased exponents in the f16 normal range
            uint32_t man = ((rng() & 0x3ff) << 13) | (1u << 12);
            uint32_t s = (rng() & 1) << 31;
            return bits2f(s | (e << 23) | man); }
        case 2: { // f16 subnormal range (|x| < 2^-14), including f32 values below the f16 subnormal floor 2^-24
            return u(rng) * std::ldexp(1.f, -14 - (int)(rng() % 14)); }
        case 3: { // huge: around and past f16 max 65504
            return u(rng) * 70000.f; }
        case 4: { // specials
            switch (rng() % 6) { case 0: return -0.f; case 1: return INFINITY; case 2: return -INFINITY;
                                 case 3: return NAN; case 4: return 65520.f; default: return -65520.f; } }
        default: return u(rng);
        }
    };
    struct Case { unsigned b; bool slot_path; unsigned n_raw_before; int kind; bool fp8; const char* tag; };
    std::vector<Case> cs = {
        {1, false, 0, 0, true, "b=1 mixed-mag fp8"},
        {3, false, 5, 0, true, "b=3 odd, n_raw_before=5 fp8"},
        {5, false, 127, 0, true, "b=5 n_raw_before=127 fp8"},
        {7, false, 0, 0, true, "b=7 fp8"},
        {8, true, 0, 0, true, "b=8 arena slot_per fp8"},
        {4, true, 0, 0, true, "b=4 arena slot_per fp8"},
        {512, false, 128, 0, true, "b=512 prefill n_raw_before=128 fp8"},
        {8, false, 0, 1, false, "b=8 f16 rounding TIES, raw (no fp8)"},
        {8, false, 0, 1, true, "b=8 f16 rounding ties, fp8"},
        {8, false, 0, 2, false, "b=8 f16 subnormal range, raw"},
        {8, false, 0, 2, true, "b=8 f16 subnormal range, fp8"},
        {8, false, 0, 3, false, "b=8 |x| up to 70000 (f16 overflow), raw"},
        {8, false, 0, 3, true, "b=8 |x| up to 70000, fp8"},
        {8, false, 0, 4, false, "b=8 specials -0/inf/nan/65520, raw"},
        {512, true, 0, 1, false, "b=512 ties raw slot_per"},
    };
    for (auto& c : cs) {
        std::vector<float> h((size_t)c.b * W);
        for (size_t i = 0; i < h.size(); ++i) h[i] = gen(c.kind, i, (unsigned)(i / W));
        KB_CHECK(hipMemcpy(x, h.data(), h.size() * 4, hipMemcpyHostToDevice));
        if (c.fp8) kb::launch(f_fp8, dim3(c.b), dim3(W), 0, 0, x, c.b, W);
        KB_CHECK(hipMemcpy(y, x, h.size() * 4, hipMemcpyDeviceToDevice));
        kb::launch(f_rt, dim3((c.b * W + 255) / 256), dim3(256), 0, 0, y, c.b * W);
        const int* sp = nullptr;
        if (c.slot_path) {
            std::vector<int> hs(c.b);
            for (unsigned i = 0; i < c.b; ++i) hs[i] = (int)((i * 37 + 11) % CACHE_ROWS);  // scattered slots
            KB_CHECK(hipMemcpy(slots, hs.data(), c.b * 4, hipMemcpyHostToDevice));
            sp = slots;
        }
        KB_CHECK(hipMemset(ca, 0x5a, (size_t)CACHE_ROWS * W * 2));
        KB_CHECK(hipMemset(cb, 0x5a, (size_t)CACHE_ROWS * W * 2));
        kb::launch(f_app, dim3(c.b), dim3(W), 0, 0, (_Float16*)ca, (const float*)x, c.n_raw_before, W, sp);
        kb::launch(f_app, dim3(c.b), dim3(W), 0, 0, (_Float16*)cb, (const float*)y, c.n_raw_before, W, sp);
        KB_CHECK(hipDeviceSynchronize());
        auto a = kb::d2h(ca, (size_t)CACHE_ROWS * W), bb = kb::d2h(cb, (size_t)CACHE_ROWS * W);
        auto xs = kb::d2h(x, h.size()), ys = kb::d2h(y, h.size());
        size_t nd = 0, nrt = 0, nnan = 0;
        for (size_t i = 0; i < a.size(); ++i) nd += a[i] != bb[i];
        for (size_t i = 0; i < h.size(); ++i) { nrt += memcmp(&xs[i], &ys[i], 4) != 0; nnan += std::isnan(xs[i]); }
        printf("%-48s rows=%u  f32 changed by roundtrip: %zu / %zu (nan in %zu)   f16 cache bits differ: %zu\n",
               c.tag, c.b, nrt, h.size(), nnan, nd);
        grand_diff += nd; grand_rt += nrt; ++cases;
    }
    printf("KBJSON {\"cmp\":\"review f16_roundtrip no-op (cache bits)\",\"cases\":%zu,\"bitexact\":%d,\"bit_diff\":%zu,\"f32_changed_by_rt\":%zu}\n",
           cases, (int)(grand_diff == 0), grand_diff, grand_rt);
    return grand_diff == 0 ? 0 : 1;
}
