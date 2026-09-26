// gtime.cpp -- reviewer's extra timing probes for E_indexer/gather_b128 (production indexer_gather_batched
// vs gather_u4_r1 / gather_u4_r4), regimes the engineer's harness does not cover:
//   A. decode at the PRODUCTION-MAX store (235000 rows = 235 MB, TLB reach), b = 1,2,4,5,8 per lane,
//      random picks, cold store (64 MB flush), direct inner=1 and graph inner=1 (production is graph-captured).
//   B. prefill lane shape b=512 with picks of controllable locality: the engineer's generator
//      (1536-row window shared by 8 tokens, ~whole store touched -> DRAM-bound) versus windows shared by
//      ALL 512 tokens (4K / 16K / 64K rows -> mostly L2/MALL hits), cold flush before each call, and
//      the 4K window with NO flush (warm). Production's reuse-gather moves 512 MB in 689 us/call
//      (780 GB/s > 640 GB/s DRAM), so its picks are largely cache-hot, not the cold synthetic point.
#include "kbench.h"

#include <random>
#include <string>

static const unsigned HD = 512, TOP_K = 512;

__global__ void k_flush_read(const uint4* p, size_t n16, unsigned* sink) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    for (; i < n16; i += (size_t)gridDim.x * blockDim.x) { uint4 v = p[i]; acc += v.x ^ v.w; }
    if (acc == 0x12345678u) *sink = acc;
}
struct Flush {
    uint4* buf; unsigned* sink; size_t n16;
    explicit Flush(size_t bytes) : n16(bytes / 16) { buf = kb::dalloc<uint4>(n16); sink = kb::dalloc<unsigned>(1); KB_CHECK(hipMemset(buf, 1, bytes)); }
    void operator()(hipStream_t s) { hipLaunchKernelGGL(k_flush_read, dim3(2048), dim3(256), 0, s, (const uint4*)buf, n16, sink); }
};

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    int rounds = argc > 2 ? atoi(argv[2]) : 30;
    kb::Module base(dir + "/base_indexer_gather_gfx1201.hsaco");
    kb::Module cand(dir + "/cand_gather_gfx1201.hsaco");
    hipFunction_t f_base = base.fn("indexer_gather_batched");
    struct C { const char* n; unsigned rpw, blk; hipFunction_t f; };
    C cs[] = {{"gather_u4_r1", 1, 64, cand.fn("gather_u4_r1")}, {"gather_u4_r4", 4, 256, cand.fn("gather_u4_r4")}};

    const unsigned rows = 235000;  // production max n_comp at 368K ctx / ratio 4 (V41_MAX_CTX 368640 / 4 ~ 92K per stream; arena max_b n_comp)
    uint16_t* comp = kb::dalloc<uint16_t>((size_t)rows * HD);
    kb::fill_f16(comp, (size_t)rows * HD, 9);
    const unsigned B_MAX = 512;
    uint16_t* dst = kb::dalloc<uint16_t>((size_t)B_MAX * TOP_K * HD);
    uint16_t* dst2 = kb::dalloc<uint16_t>((size_t)B_MAX * TOP_K * HD);
    int* sel = kb::dalloc<int>((size_t)B_MAX * TOP_K);
    int* base_per = kb::dalloc<int>(B_MAX);
    KB_CHECK(hipMemset(base_per, 0, B_MAX * 4));
    KB_CHECK(hipDeviceSynchronize());
    Flush flush(64u << 20);
    std::mt19937 rng(777);

    auto variants = [&](unsigned b, const int* bp) {
        std::vector<kb::Variant> vs;
        const double bytes = 2.0 * b * TOP_K * HD * 2;
        vs.push_back({"base indexer_gather_batched (512,b,2)x256", [=](hipStream_t s) {
            kb::launch(f_base, dim3(TOP_K, b, 2), dim3(256), 0, s, dst, (const uint16_t*)comp, (const int*)sel, TOP_K, HD, bp); }, bytes});
        for (auto& c : cs)
            vs.push_back({std::string("cand ") + c.n, [=](hipStream_t s) {
                kb::launch(c.f, dim3((TOP_K + c.rpw - 1) / c.rpw, b), dim3(c.blk), 0, s, dst2, (const uint16_t*)comp, (const int*)sel, TOP_K, HD, bp); }, bytes});
        return vs;
    };

    // ---- A. decode, production-max store, random picks, cold
    for (unsigned b : {1u, 2u, 4u, 5u, 8u}) {
        std::vector<int> hsel((size_t)b * TOP_K);
        for (auto& x : hsel) x = (int)(rng() % rows);
        KB_CHECK(hipMemcpy(sel, hsel.data(), hsel.size() * 4, hipMemcpyHostToDevice));
        auto vs = variants(b, base_per);
        char tag[200];
        kb::AbOpts ab; ab.rounds = rounds; ab.inner = 1; ab.graph = false; ab.between = [&](hipStream_t s) { flush(s); };
        snprintf(tag, sizeof tag, "A decode b=%u RANDOM picks, 235K-row store COLD, direct inner=1", b); ab.tag = tag;
        kb::ab(vs, ab);
        kb::AbOpts ab3 = ab; ab3.graph = true;
        snprintf(tag, sizeof tag, "A decode b=%u RANDOM picks, 235K-row store COLD, GRAPH inner=1 (prod)", b); ab3.tag = tag;
        kb::ab(vs, ab3);
    }

    // ---- B. prefill lane b=512, locality sweep
    struct W { const char* name; unsigned window; bool per8; };
    W ws[] = {{"engineer-style 1536-row window per 8 tokens", 1536, true},
              {"4K-row window shared by all 512 tokens", 4096, false},
              {"16K-row window shared by all 512 tokens", 16384, false},
              {"64K-row window shared by all 512 tokens", 65536, false}};
    const unsigned b = 512;
    for (auto& w : ws) {
        std::vector<int> hsel((size_t)b * TOP_K);
        unsigned centre_all = rng() % rows;
        for (unsigned t = 0; t < b; t++) {
            unsigned centre = w.per8 ? (unsigned)((t / 8) * 2654435761u) % rows : centre_all;
            for (unsigned k = 0; k < TOP_K; k++) {
                unsigned off = w.per8 ? (k * 3u + rng() % 3u) % w.window : rng() % w.window;
                hsel[(size_t)t * TOP_K + k] = (int)((centre + off) % rows);
            }
        }
        KB_CHECK(hipMemcpy(sel, hsel.data(), hsel.size() * 4, hipMemcpyHostToDevice));
        auto vs = variants(b, nullptr);  // prefill: comp_base_per = null
        char tag[200];
        kb::AbOpts ab; ab.rounds = rounds; ab.inner = 1; ab.graph = false; ab.between = [&](hipStream_t s) { flush(s); };
        snprintf(tag, sizeof tag, "B prefill b=512 %s, COLD flush, direct", w.name); ab.tag = tag;
        kb::ab(vs, ab);
        if (w.window == 4096 || w.window == 65536) {
            kb::AbOpts ab2; ab2.rounds = rounds; ab2.inner = 1; ab2.graph = false;
            snprintf(tag, sizeof tag, "B prefill b=512 %s, WARM (no flush), direct", w.name); ab2.tag = tag;
            kb::ab(vs, ab2);
        }
    }
    return 0;
}
