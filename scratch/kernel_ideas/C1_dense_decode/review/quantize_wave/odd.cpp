// Reviewer's correctness harness for q8_0_quantize_f32_wave (claim C1_dense_decode/quantize_wave).
// Production kernel (base_q8_gfx1201.hsaco, unmodified in-tree source) launched exactly as the Rust
// wrapper does: grid=(blocks) x block=32.  Candidate launched as the engineer's harness does:
// grid=ceil(blocks/8) x 256.  Compares xq (bytewise) and xscale (bitwise), checks the candidate does
// not write past `blocks` (sentinel guard), and drives adversarial inputs the engineer's random
// [-3,3] fill cannot: all-zero blocks, all-equal, single huge value, denormals, exact .5 ties,
// +-inf, and an all-NaN block (expected to differ: documented).
#include "kbench.h"
#include <cmath>
#include <cstring>
#include <string>

static hipFunction_t g_base, g_cand;
static int g_fail = 0;

// CPU model of the production kernel's arithmetic (serial fmaxf from 0, d=amax/127, id=1/d, lrintf).
static void cpu_ref(const std::vector<float>& x, unsigned blocks, std::vector<int8_t>& xq, std::vector<float>& xs) {
    xq.assign((size_t)blocks * 32, 0); xs.assign(blocks, 0.f);
    for (unsigned b = 0; b < blocks; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[(size_t)b * 32 + i]));
        float d = amax / 127.0f, id = d != 0.f ? 1.0f / d : 0.f;
        xs[b] = d;
        for (int i = 0; i < 32; ++i) {
            int q = (int)lrintf(x[(size_t)b * 32 + i] * id);
            if (q > 127) q = 127; if (q < -128) q = -128;
            xq[(size_t)b * 32 + i] = (int8_t)q;
        }
    }
}

static void run_case(const char* name, const std::vector<float>& hx, unsigned blocks, bool expect_exact = true) {
    const size_t n = (size_t)blocks * 32;
    const size_t GUARD = 4096;  // bytes / floats of sentinel after the real outputs
    float* x = kb::dalloc<float>(n + 64);
    KB_CHECK(hipMemcpy(x, hx.data(), n * 4, hipMemcpyHostToDevice));
    int8_t* xq_b = kb::dalloc<int8_t>(n + GUARD); int8_t* xq_c = kb::dalloc<int8_t>(n + GUARD);
    float* xs_b = kb::dalloc<float>(blocks + GUARD); float* xs_c = kb::dalloc<float>(blocks + GUARD);
    KB_CHECK(hipMemset(xq_b, 0x5A, n + GUARD)); KB_CHECK(hipMemset(xq_c, 0x5A, n + GUARD));
    KB_CHECK(hipMemset(xs_b, 0x5A, (blocks + GUARD) * 4)); KB_CHECK(hipMemset(xs_c, 0x5A, (blocks + GUARD) * 4));
    KB_CHECK(hipDeviceSynchronize());
    kb::launch(g_base, dim3(blocks), dim3(32), 0, 0, xq_b, xs_b, (const float*)x, (uint32_t)blocks);
    kb::launch(g_cand, dim3((blocks + 7) / 8), dim3(256), 0, 0, xq_c, xs_c, (const float*)x, (uint32_t)blocks);
    KB_CHECK(hipDeviceSynchronize());
    auto qb = kb::d2h(xq_b, n + GUARD), qc = kb::d2h(xq_c, n + GUARD);
    auto sb = kb::d2h(xs_b, blocks + GUARD), sc = kb::d2h(xs_c, blocks + GUARD);
    size_t qd = 0, sd = 0, guard_q = 0, guard_s = 0;
    for (size_t i = 0; i < n; ++i) qd += qb[i] != qc[i];
    for (size_t i = 0; i < blocks; ++i) sd += memcmp(&sb[i], &sc[i], 4) != 0;
    for (size_t i = n; i < n + GUARD; ++i) guard_q += (uint8_t)qc[i] != 0x5A;
    for (size_t i = blocks; i < blocks + GUARD; ++i) { uint32_t u; memcpy(&u, &sc[i], 4); guard_s += u != 0x5A5A5A5Au; }
    // base vs CPU model (sanity that the baseline is what we think it is)
    std::vector<int8_t> rq; std::vector<float> rs; cpu_ref(hx, blocks, rq, rs);
    size_t bq = 0, bs = 0;
    for (size_t i = 0; i < n; ++i) bq += qb[i] != rq[i];
    for (size_t i = 0; i < blocks; ++i) bs += memcmp(&sb[i], &rs[i], 4) != 0;
    bool ok = (qd == 0 && sd == 0 && guard_q == 0 && guard_s == 0);
    printf("CASE %-34s blocks=%6u WGs(cand)=%5u  xq_diff=%zu xscale_bitdiff=%zu guard_q=%zu guard_s=%zu | base-vs-cpu xq=%zu xs=%zu  -> %s%s\n",
           name, blocks, (blocks + 7) / 8, qd, sd, guard_q, guard_s, bq, bs,
           ok ? "BITEXACT" : "DIFF", (!ok && !expect_exact) ? " (expected)" : "");
    if (!ok && expect_exact) g_fail++;
    if (!ok) {  // print first differing block
        for (size_t i = 0; i < blocks; ++i) {
            bool bd = memcmp(&sb[i], &sc[i], 4) != 0; bool qdd = false;
            for (int j = 0; j < 32; ++j) qdd |= qb[i * 32 + j] != qc[i * 32 + j];
            if (bd || qdd) { printf("   first diff block %zu: xs base=%g cand=%g; x[0..3]=%g %g %g %g\n", i, sb[i], sc[i], hx[i*32], hx[i*32+1], hx[i*32+2], hx[i*32+3]); break; }
        }
    }
    KB_CHECK(hipFree(x)); KB_CHECK(hipFree(xq_b)); KB_CHECK(hipFree(xq_c)); KB_CHECK(hipFree(xs_b)); KB_CHECK(hipFree(xs_c));
}

static std::vector<float> rnd(size_t n, uint32_t seed, float lo, float hi) {
    std::vector<float> v(n); uint32_t s = seed * 2654435761u + 1;
    for (auto& f : v) { s = s * 1664525u + 1013904223u; f = lo + (hi - lo) * ((s >> 8) * (1.0f / 16777216.0f)); }
    return v;
}

int main() {
    kb::init();
    const char* dir = getenv("C1_DIR") ? getenv("C1_DIR") : ".";
    kb::Module base(std::string(dir) + "/base_q8_gfx1201.hsaco"), cand(std::string(dir) + "/cand_bpack_gfx1201.hsaco");
    g_base = base.fn("q8_0_quantize_f32"); g_cand = cand.fn("q8_0_quantize_f32_wave");

    // 1. Odd shapes / tails (blocks not a multiple of 8, single block, 1 WG) with random data.
    unsigned shapes[] = {1, 7, 9, 41, 160 /*K=5120 b=1*/, 480 /*b=3*/, 800 /*K=5120 b=5 (max prod b/lane)*/,
                         1280 /*K=5120 b=8*/, 40 /*K=1280*/, 72 * 5 /*K=2304 b=5*/, 256 * 5 /*K=8192 b=5*/,
                         1024 * 5 /*K=32768 b=5*/, 1024 * 4 /*K=32768 b=4 (pathology grid)*/, 1024 * 8, 2048 * 3 + 5};
    for (unsigned bl : shapes) {
        auto x = rnd((size_t)bl * 32, bl, -3.f, 3.f);
        run_case(("random[-3,3] blocks=" + std::to_string(bl)).c_str(), x, bl);
    }
    // 2. Adversarial values at K=5120 b=5 (800 blocks): special blocks scattered in.
    {
        unsigned bl = 800; auto x = rnd((size_t)bl * 32, 99, -1000.f, 1000.f);
        auto blk = [&](unsigned b) { return x.data() + (size_t)b * 32; };
        for (int i = 0; i < 32; ++i) blk(0)[i] = 0.f;                             // all zero -> d=0, id=0
        for (int i = 0; i < 32; ++i) blk(1)[i] = 2.5f;                            // all equal
        for (int i = 0; i < 32; ++i) blk(2)[i] = -7.f;                            // all equal negative
        for (int i = 0; i < 32; ++i) blk(3)[i] = (i == 17) ? 1e30f : 1e-3f;       // one huge value
        for (int i = 0; i < 32; ++i) blk(4)[i] = 1e-40f * (i + 1);                // denormals only
        for (int i = 0; i < 32; ++i) blk(5)[i] = (i & 1) ? 2.5f : -0.5f;          // amax=2.5 -> ties at .5 granularity
        for (int i = 0; i < 32; ++i) blk(6)[i] = (float)(i - 16) * 0.5f; blk(6)[31] = 127.f;  // d=1, id=1: x*id exact .5 ties
        for (int i = 0; i < 32; ++i) blk(7)[i] = (i == 3) ? -127.f : (float)(i - 16) * 0.5f;  // negative amax
        for (int i = 0; i < 32; ++i) blk(8)[i] = (i == 9) ? 3.4e38f : 1.f;        // near FLT_MAX
        for (int i = 0; i < 32; ++i) blk(9)[i] = (i == 0) ? -0.0f : 0.0f;         // signed zero
        for (int i = 0; i < 32; ++i) blk(10)[i] = (i == 5) ? 1.f / 0.f : 2.f;     // +inf in the block
        for (int i = 0; i < 32; ++i) blk(11)[i] = (i == 5) ? -1.f / 0.f : 2.f;    // -inf
        for (int i = 0; i < 32; ++i) blk(12)[i] = (i == 20) ? __builtin_nanf("") : 2.f;  // one NaN (prod ignores it via fmaxf(0,NaN)=0)
        for (int i = 0; i < 32; ++i) blk(13)[i] = 127.f * ((i % 2) ? 1.f : -1.f); // +-127 exactly -> q=+-127
        for (int i = 0; i < 32; ++i) blk(14)[i] = (i == 31) ? 100.f : 100.f * (1.f - 1.f / 254.f) * ((i % 2) ? 1.f : -1.f);  // rounding at 126.5 boundary
        run_case("adversarial blocks (K=5120 b=5)", x, bl);
    }
    // 3. All-NaN block: the serial fmaxf(0,NaN)=0 in production vs the tree in the candidate. Documented, not counted.
    {
        unsigned bl = 16; auto x = rnd((size_t)bl * 32, 5, -3.f, 3.f);
        for (int i = 0; i < 32; ++i) x[(size_t)3 * 32 + i] = __builtin_nanf("");
        run_case("all-NaN block (informational)", x, bl, false);
    }
    // 4. Values needing the clamp: cannot exceed 127 by construction (x*id <= 127*(1+eps)); check the largest-ratio case
    {
        unsigned bl = 64; std::vector<float> x((size_t)bl * 32);
        for (unsigned b = 0; b < bl; ++b) for (int i = 0; i < 32; ++i) x[(size_t)b * 32 + i] = (i == 0) ? (1.f + b * 0.37f) : -(1.f + b * 0.37f) * 0.999999f;
        run_case("clamp boundary (x ~ amax)", x, bl);
    }
    printf("ODD SUMMARY: %s (%d failing cases)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
