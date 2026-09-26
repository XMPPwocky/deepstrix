// Review check for C1_dense_decode/quantize_grid_pad: production q8_0_quantize_f32 (baseline hsaco)
// launched with grid = blocks (production) vs blocks+1 / blocks+7 / blocks+64, SAME process,
// interleaved kb::ab (warm activations, graph mode as production decode; env GC_NOGRAPH=1 for plain).
// Correctness: outputs of the padded launches vs the unpadded one (bytes), outputs pre-filled with
// 0xAA so an unwritten element shows, plus a CPU reference of the kernel math.
// usage: ./gpad_check K b [K b ...]   (blocks = K/32*b, K % 32 == 0)
#include "kbench.h"
#include <cmath>
#include <cstring>

static void cpu_quant(const std::vector<float>& x, unsigned blocks, std::vector<int8_t>& xq, std::vector<float>& xs) {
    xq.resize((size_t)blocks * 32); xs.resize(blocks);
    for (unsigned b = 0; b < blocks; ++b) {
        float amax = 0.f;
        for (unsigned i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[(size_t)b * 32 + i]));
        float d = amax / 127.0f, id = d != 0.f ? 1.0f / d : 0.f;
        xs[b] = d;
        for (unsigned i = 0; i < 32; ++i) {
            int q = (int)lrintf(x[(size_t)b * 32 + i] * id);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            xq[(size_t)b * 32 + i] = (int8_t)q;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3 || (argc - 1) % 2) { fprintf(stderr, "usage: gpad_check K b [K b ...]\n"); return 64; }
    kb::init();
    const char* dir = getenv("C1_DIR") ? getenv("C1_DIR") : ".";
    kb::Module base(std::string(dir) + "/base_q8_gfx1201.hsaco");
    hipFunction_t f = base.fn("q8_0_quantize_f32");
    const bool graph = getenv("GC_NOGRAPH") == nullptr;
    const unsigned pads[4] = {0, 1, 7, 64};
    int rc = 0;
    for (int a = 1; a + 1 < argc; a += 2) {
        const unsigned K = atoi(argv[a]), b = atoi(argv[a + 1]);
        if (K % 32) { fprintf(stderr, "K %u not %%32\n", K); return 64; }
        const unsigned blocks = K / 32 * b;
        const size_t n = (size_t)K * b;
        // input: random [-3,3]; block 0 all zeros (d=0 path), block 1 has a 1e30 spike, block 2 tiny values
        std::vector<float> hx(n);
        uint32_t s = 0x1234567u ^ blocks;
        for (size_t i = 0; i < n; ++i) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; hx[i] = ((s >> 8) * (1.0f / 16777216.0f) - 0.5f) * 6.f; }
        for (unsigned i = 0; i < 32; ++i) hx[i] = 0.f;
        if (blocks > 1) hx[32 + 3] = 1e30f;
        if (blocks > 2) for (unsigned i = 0; i < 32; ++i) hx[64 + i] *= 1e-30f;
        float* x = kb::dalloc<float>(n);
        KB_CHECK(hipMemcpy(x, hx.data(), n * 4, hipMemcpyHostToDevice));
        int8_t* xq[4]; float* xs[4];
        for (int p = 0; p < 4; ++p) {
            xq[p] = kb::dalloc<int8_t>(n); xs[p] = kb::dalloc<float>(blocks);
            KB_CHECK(hipMemset(xq[p], 0xAA, n)); KB_CHECK(hipMemset(xs[p], 0xAA, blocks * 4));
        }
        // GC_MIX=1: a different kernel (kb_fill_f32 rewriting x, as the producer kernel does in
        // production) runs before every quantize, so consecutive quantize dispatches are not back-to-back.
        const bool mix = getenv("GC_MIX") != nullptr;
        auto mk = [&](int p) {
            int8_t* q = xq[p]; float* sc = xs[p]; unsigned g = blocks + pads[p]; size_t nn = n;
            return [=](hipStream_t st) {
                if (mix) hipLaunchKernelGGL(kb::kb_fill_f32, dim3(1024), dim3(256), 0, st, x, nn, 7u, -3.f, 3.f);
                kb::launch(f, dim3(g), dim3(32), 0, st, q, sc, (const float*)x, blocks);
            };
        };
        for (int p = 0; p < 4; ++p) { mk(p)(0); }
        KB_CHECK(hipDeviceSynchronize());
        auto q0 = kb::d2h(xq[0], n); auto s0 = kb::d2h(xs[0], blocks);
        std::vector<int8_t> cq; std::vector<float> cs; cpu_quant(hx, blocks, cq, cs);
        size_t dq_cpu = 0, ds_cpu = 0, s_aa = 0;
        for (size_t i = 0; i < n; ++i) dq_cpu += q0[i] != cq[i];
        for (size_t i = 0; i < blocks; ++i) { ds_cpu += memcmp(&s0[i], &cs[i], 4) != 0; uint32_t u; memcpy(&u, &s0[i], 4); s_aa += u == 0xAAAAAAAAu; }
        printf("SHAPE K=%u b=%u blocks=%u grid0=%u | base vs CPU ref: xq_diff=%zu xs_bitdiff=%zu xs_unwritten(0xAAAAAAAA)=%zu\n",
               K, b, blocks, blocks, dq_cpu, ds_cpu, s_aa);
        for (int p = 1; p < 4; ++p) {
            auto q = kb::d2h(xq[p], n); auto ss = kb::d2h(xs[p], blocks);
            size_t dq = 0, ds = 0;
            for (size_t i = 0; i < n; ++i) dq += q[i] != q0[i];
            for (size_t i = 0; i < blocks; ++i) ds += memcmp(&ss[i], &s0[i], 4) != 0;
            printf("CMP pad=%u grid=%u vs grid=%u: xq_bytediff=%zu xscale_bitdiff=%zu %s\n", pads[p], blocks + pads[p], blocks, dq, ds, (dq || ds) ? "MISMATCH" : "bitexact");
            if (dq || ds) rc = 1;
        }
        std::vector<kb::Variant> vs;
        for (int p = 0; p < 4; ++p) vs.push_back({std::string("q8_0_quantize_f32 grid=blocks+") + std::to_string(pads[p]), mk(p), (double)n * 5});
        kb::AbOpts o; o.graph = graph; o.inner = 10; o.rounds = getenv("GC_ROUNDS") ? atoi(getenv("GC_ROUNDS")) : 60;
        std::string tag = std::string(mix ? "MIX(fill x + quantize) " : "") + "quantize K=" + std::to_string(K) + " b=" + std::to_string(b) + " blocks=" + std::to_string(blocks) + " WARM";
        o.tag = tag.c_str();
        kb::ab(vs, o);
        KB_CHECK(hipFree(x)); for (int p = 0; p < 4; ++p) { KB_CHECK(hipFree(xq[p])); KB_CHECK(hipFree(xs[p])); }
    }
    printf(rc ? "RESULT: MISMATCH\n" : "RESULT: all padded launches bit-exact vs production grid\n");
    return rc;
}
