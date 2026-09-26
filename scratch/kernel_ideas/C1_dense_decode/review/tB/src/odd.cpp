// Reviewer odd-shape correctness check: candidate tB<b> / grouped tB<b> vs production bpack kernel
// on identical random inputs (incl. int8 extremes), plus a CPU reference of the production math.
// Shapes the engineer did NOT test: row tails (M % 8 != 0 -> production grid = ceil(M/8), row
// guard), block counts with a lane-loop tail (K/32 % 32 != 0), b in {3,6,7,9,10}, and adversarial
// activations (all 127 / all -128 with large scales). Output buffers carry a sentinel past M*b.
#include "kbench.h"
#include <string>
static uint32_t g_rng = 0xdeadbeefu;
static inline uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
static inline uint16_t f2h(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static std::vector<uint8_t> make_q8_rows(unsigned M, unsigned K, int mode) {
    const unsigned blocks = K / 32; const size_t rowb = (size_t)blocks * 34;
    std::vector<uint8_t> h(rowb * M);
    for (unsigned r = 0; r < M; ++r) {
        uint8_t* row = h.data() + r * rowb; uint16_t* sc = (uint16_t*)row;
        for (unsigned b = 0; b < blocks; ++b) sc[b] = f2h(mode ? 8.0f : ((rnd() >> 8) * (1.0f / 16777216.0f) - 0.5f) * 0.05f);
        uint8_t* q = row + blocks * 2;
        for (unsigned i = 0; i < blocks * 32; ++i) q[i] = mode ? (uint8_t)0x80 : (uint8_t)(rnd() & 0xff);
    }
    return h;
}
static void make_acts(unsigned K, unsigned b, int mode, std::vector<int8_t>& xq, std::vector<float>& xs) {
    xq.resize((size_t)K * b); xs.resize((size_t)K / 32 * b);
    for (auto& v : xq) v = mode ? (int8_t)127 : (int8_t)(rnd() & 0xff);
    for (auto& v : xs) v = mode ? 100.0f : ((rnd() >> 8) * (1.0f / 16777216.0f)) * 0.1f + 1e-3f;
}
static void cpu_ref(const std::vector<uint8_t>& w, unsigned M, unsigned K, const std::vector<int8_t>& xq,
                    const std::vector<float>& xs, unsigned b, std::vector<float>& out) {
    const unsigned blocks = K / 32; out.assign((size_t)M * b, 0.f);
    for (unsigned r = 0; r < M; ++r) {
        const uint8_t* row = w.data() + (size_t)r * blocks * 34; const uint16_t* sc = (const uint16_t*)row;
        const int8_t* q = (const int8_t*)(row + blocks * 2);
        for (unsigned bb = 0; bb < b; ++bb) {
            float lane[32] = {0};
            for (unsigned bl = 0; bl < blocks; ++bl) {
                int32_t dot = 0;
                for (unsigned i = 0; i < 32; ++i) dot += (int32_t)q[bl * 32 + i] * (int32_t)xq[(size_t)bb * K + bl * 32 + i];
                lane[bl % 32] += kb::h2f(sc[bl]) * xs[(size_t)bb * blocks + bl] * (float)dot;
            }
            for (int off = 16; off > 0; off >>= 1) for (int l = 0; l < off; ++l) lane[l] += lane[l + off];
            out[(size_t)bb * M + r] = lane[0];
        }
    }
}
static int g_fail = 0;
static void check(const char* tag, const kb::Cmp& c, bool need_bitexact) {
    kb::print_cmp(tag, c);
    if (need_bitexact && c.n_bit_diff) { printf("FAIL %s\n", tag); g_fail = 1; }
    if (c.n_nonfinite) { printf("FAIL nonfinite %s\n", tag); g_fail = 1; }
}
static void tail_check(const char* tag, const std::vector<float>& hb, const std::vector<float>& hc, size_t n) {
    if (memcmp(hb.data() + n, hc.data() + n, 64 * 4) != 0) { printf("FAIL sentinel differs %s\n", tag); g_fail = 1; }
    for (int i = 0; i < 64; ++i) { uint32_t u; memcpy(&u, &hc[n + i], 4); if (u != 0x7f7f7f7fu) { printf("FAIL cand wrote past end %s\n", tag); g_fail = 1; break; } }
}
static void gemv_case(kb::Module& base, kb::Module& cand, unsigned M, unsigned K, unsigned b, int mode) {
    const unsigned blocks = K / 32; const size_t wbytes = (size_t)M * blocks * 34;
    auto hw = make_q8_rows(M, K, mode); std::vector<int8_t> hxq; std::vector<float> hxs; make_acts(K, b, mode, hxq, hxs);
    uint8_t* w = kb::dalloc<uint8_t>(wbytes); KB_CHECK(hipMemcpy(w, hw.data(), wbytes, hipMemcpyHostToDevice));
    int8_t* xq = kb::dalloc<int8_t>(hxq.size()); float* xs = kb::dalloc<float>(hxs.size());
    KB_CHECK(hipMemcpy(xq, hxq.data(), hxq.size(), hipMemcpyHostToDevice));
    KB_CHECK(hipMemcpy(xs, hxs.data(), hxs.size() * 4, hipMemcpyHostToDevice));
    const size_t n = (size_t)M * b, npad = n + 64;
    float* ob = kb::dalloc<float>(npad); float* oc = kb::dalloc<float>(npad);
    KB_CHECK(hipMemset(ob, 0x7f, npad * 4)); KB_CHECK(hipMemset(oc, 0x7f, npad * 4));
    const dim3 grid((M + 7) / 8), block(256);  // production: n_rows.div_ceil(8)
    kb::launch(base.fn("q8_0_gemv_bpack_warp8"), grid, block, 0, 0, ob, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, K, M, blocks, b);
    std::string nm = "q8_0_gemv_bpack_tB" + std::to_string(b);
    hipFunction_t f; bool have = hipModuleGetFunction(&f, cand.m, nm.c_str()) == hipSuccess;
    if (have) kb::launch(f, grid, block, 0, 0, oc, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, K, M, blocks, b);
    KB_CHECK(hipDeviceSynchronize());
    auto hb = kb::d2h(ob, npad), hc = kb::d2h(oc, npad);
    char tag[160];
    snprintf(tag, sizeof tag, "gemv M=%u K=%u b=%u mode=%d %s", M, K, b, mode, have ? nm.c_str() : "(NO CAND SYMBOL)");
    if (!have) { printf("MISSING %s\n", tag); g_fail |= 2; }
    else { check(tag, kb::compare_f32(hb.data(), hc.data(), n), true); tail_check(tag, hb, hc, n); }
    if ((size_t)M * K <= 40000000 && mode == 0) {
        std::vector<float> ref; cpu_ref(hw, M, K, hxq, hxs, b, ref);
        snprintf(tag, sizeof tag, "  base vs cpu_ref M=%u K=%u b=%u", M, K, b);
        check(tag, kb::compare_f32(ref.data(), hb.data(), n), false);
    }
    KB_CHECK(hipFree(w)); KB_CHECK(hipFree(xq)); KB_CHECK(hipFree(xs)); KB_CHECK(hipFree(ob)); KB_CHECK(hipFree(oc));
}
static void grouped_case(kb::Module& grp, kb::Module& cand, unsigned G, unsigned R, unsigned GD, unsigned b, int mode) {
    const unsigned bpg = GD / 32, M = G * R; const size_t wbytes = (size_t)M * bpg * 34;
    auto hw = make_q8_rows(M, GD, mode); std::vector<int8_t> hxq; std::vector<float> hxs; make_acts(G * GD, b, mode, hxq, hxs);
    uint8_t* w = kb::dalloc<uint8_t>(wbytes); KB_CHECK(hipMemcpy(w, hw.data(), wbytes, hipMemcpyHostToDevice));
    int8_t* xq = kb::dalloc<int8_t>(hxq.size()); float* xs = kb::dalloc<float>(hxs.size());
    KB_CHECK(hipMemcpy(xq, hxq.data(), hxq.size(), hipMemcpyHostToDevice));
    KB_CHECK(hipMemcpy(xs, hxs.data(), hxs.size() * 4, hipMemcpyHostToDevice));
    const size_t n = (size_t)M * b, npad = n + 64;
    float* ob = kb::dalloc<float>(npad); float* oc = kb::dalloc<float>(npad);
    KB_CHECK(hipMemset(ob, 0x7f, npad * 4)); KB_CHECK(hipMemset(oc, 0x7f, npad * 4));
    const dim3 grid((M + 7) / 8), block(256);
    kb::launch(grp.fn("q8_0_grouped_gemv_bpack"), grid, block, 0, 0, ob, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, GD, R, bpg, G, b);
    std::string nm = "q8_0_grouped_gemv_bpack_tB" + std::to_string(b);
    hipFunction_t f; bool have = hipModuleGetFunction(&f, cand.m, nm.c_str()) == hipSuccess;
    if (have) kb::launch(f, grid, block, 0, 0, oc, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, GD, R, bpg, G, b);
    KB_CHECK(hipDeviceSynchronize());
    auto hb = kb::d2h(ob, npad), hc = kb::d2h(oc, npad);
    char tag[160];
    snprintf(tag, sizeof tag, "grouped G=%u R=%u GD=%u b=%u mode=%d %s", G, R, GD, b, mode, have ? nm.c_str() : "(NO CAND SYMBOL)");
    if (!have) { printf("MISSING %s\n", tag); g_fail |= 2; }
    else { check(tag, kb::compare_f32(hb.data(), hc.data(), n), true); tail_check(tag, hb, hc, n); }
    KB_CHECK(hipFree(w)); KB_CHECK(hipFree(xq)); KB_CHECK(hipFree(xs)); KB_CHECK(hipFree(ob)); KB_CHECK(hipFree(oc));
}
int main(int argc, char** argv) {
    kb::init();
    std::string dir = argc > 1 ? argv[1] : ".";
    kb::Module base(dir + "/base_q8_gfx1201.hsaco"), grp(dir + "/base_grp_gfx1201.hsaco"), cand(dir + "/cand_bpack_gfx1201.hsaco");
    for (unsigned b : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u}) gemv_case(base, cand, 1001, 5120, b, 0);  // row tail (grid 126, 7 idle waves)
    for (unsigned b : {1u, 3u, 5u, 6u, 10u}) gemv_case(base, cand, 523, 2304, b, 0);    // 72 blocks: lanes 0-7 do 3, rest 2
    for (unsigned b : {1u, 4u, 5u, 8u}) gemv_case(base, cand, 33, 1280, b, 0);          // 40 blocks, 5 WGs
    for (unsigned b : {2u, 5u}) gemv_case(base, cand, 4096, 32, b, 0);                  // 1 block per row: lanes 1..31 idle
    for (unsigned b : {4u, 5u, 8u}) gemv_case(base, cand, 1280, 5120, b, 1);            // adversarial: q=-128, xq=127, big scales
    gemv_case(base, cand, 1280, 5120, 16, 0);   // production max b for bpack (GEMV_BPACK_MAX=16): expect NO tB16 symbol
    for (unsigned b : {1u, 2u, 3u, 4u, 5u, 6u, 8u}) grouped_case(grp, cand, 8, 1000, 4096, b, 0);  // rank 1000: WG rows cross groups, tail
    grouped_case(grp, cand, 3, 1024, 2048, 5, 0);   // 64 blocks per group (2 per lane)
    grouped_case(grp, cand, 8, 1024, 4096, 4, 1);   // adversarial
    grouped_case(grp, cand, 8, 1024, 4096, 5, 0);   // production shape
    printf("ODD DONE fail=%d (bit1 = mismatch/overrun, bit2 = missing symbol only)\n", g_fail);
    return g_fail & 1;
}
