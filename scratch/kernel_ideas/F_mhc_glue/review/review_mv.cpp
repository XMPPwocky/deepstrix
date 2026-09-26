// Reviewer harness for F_mhc_glue/router_mv_h20.
//   ./review_mv_gfx1201 cmp                 correctness on shapes the engineer did not test
//   ./review_mv_gfx1201 time <b> <ncopy>    production shape 384x5120, graph mode, W rotated over ncopy copies
#include <hip/hip_runtime.h>
#include "kbench.h"
#include <algorithm>
#include <string>
#include <vector>

static const unsigned NE = 5120, NEXP = 384;

struct Fns { hipFunction_t base, h20; };

static void run(hipFunction_t f, hipStream_t s, float* out, const void* w, const float* x, unsigned k, unsigned n_rows, unsigned b) {
    kb::launch(f, dim3((n_rows + 7) / 8, 1, b), dim3(256), 0, s, out, w, x, k, n_rows);
}

static int mode_cmp(Fns fn) {
    struct Shape { unsigned n_rows, k, b; const char* why; };
    std::vector<Shape> shapes = {
        {384, 5120, 1, "prod b=1"}, {384, 5120, 8, "prod b=8 (max decode)"}, {384, 5120, 5, "prod b=5"},
        {383, 5120, 4, "odd rows: last wave idle"}, {41, 5120, 3, "partial block"}, {1, 5120, 1, "1 row"},
        {384, 5152, 2, "tail k=5120+32 (generic path)"}, {384, 4992, 1, "k=4992 generic"},
        {384, 640, 4, "k=640: 1 chunk"}, {384, 1920, 4, "k=1920: 3 chunks (odd)"}, {384, 1280, 4, "k=1280: 2 chunks"},
        {4096, 1280, 8, "idx q 4096x1280 (shared symbol)"}, {32, 5120, 8, "proj 32x5120 (shared symbol)"},
        {512, 5120, 8, "compressor 512x5120 (shared symbol)"}, {128, 512, 8, "index-K 128x512 (generic)"},
    };
    size_t max_w = 0, max_x = 0, max_o = 0;
    for (auto& s : shapes) { max_w = std::max(max_w, (size_t)s.n_rows * s.k); max_x = std::max(max_x, (size_t)s.b * s.k); max_o = std::max(max_o, (size_t)s.b * s.n_rows); }
    void* w = kb::dalloc<uint16_t>(max_w);
    float* x = kb::dalloc<float>(max_x);
    float* o0 = kb::dalloc<float>(max_o);
    float* o1 = kb::dalloc<float>(max_o);
    int bad = 0, total = 0;
    for (auto& s : shapes) {
        for (int dist = 0; dist < 3; ++dist) {
            // dist 0: engineer's ranges; 1: wide-magnitude (rounding-sensitive); 2: unit ranges, heavy cancellation
            float xl = dist == 1 ? -50.f : -1.f, xh = -xl;
            float wl = dist == 0 ? -0.05f : -1.f, wh = -wl;
            kb::fill_f32(x, (size_t)s.b * s.k, 900 + dist, xl, xh);
            kb::fill_f16(w, (size_t)s.n_rows * s.k, 700 + dist, wl, wh);
            KB_CHECK(hipMemset(o0, 0xff, max_o * 4)); KB_CHECK(hipMemset(o1, 0xff, max_o * 4));  // NaN fill: catch unwritten outputs
            run(fn.base, 0, o0, w, x, s.k, s.n_rows, s.b);
            run(fn.h20, 0, o1, w, x, s.k, s.n_rows, s.b);
            KB_CHECK(hipDeviceSynchronize());
            auto a = kb::d2h(o0, (size_t)s.b * s.n_rows), c = kb::d2h(o1, (size_t)s.b * s.n_rows);
            char tag[128]; snprintf(tag, sizeof tag, "h20 n=%u k=%u b=%u d%d %s", s.n_rows, s.k, s.b, dist, s.why);
            auto r = kb::compare_f32(a.data(), c.data(), a.size());
            kb::print_cmp(tag, r);
            ++total; if (r.n_bit_diff || r.n_nonfinite) ++bad;
        }
    }
    printf("REVIEW_CMP total=%d bad=%d\n", total, bad);
    return bad ? 1 : 0;
}

static int mode_time(Fns fn, unsigned b, unsigned ncopy) {
    void* w0 = kb::dalloc<uint16_t>((size_t)NEXP * NE);
    kb::fill_f16(w0, (size_t)NEXP * NE, 21, -0.05f, 0.05f);
    float* x = kb::dalloc<float>((size_t)b * NE); kb::fill_f32(x, (size_t)b * NE, 22, -1.f, 1.f);
    float* o0 = kb::dalloc<float>((size_t)b * NEXP); float* o1 = kb::dalloc<float>((size_t)b * NEXP);
    std::vector<void*> wc(ncopy, w0);
    for (unsigned c = 1; c < ncopy; ++c) { wc[c] = kb::dalloc<uint16_t>((size_t)NEXP * NE); KB_CHECK(hipMemcpy(wc[c], w0, (size_t)NEXP * NE * 2, hipMemcpyDeviceToDevice)); }
    unsigned rot = 0;
    double bytes = NEXP * NE * 2.0 + (double)b * NE * 4.0;
    std::vector<kb::Variant> vs = {
        {"prod f16_matvec_batched (48,1,b)", [&](hipStream_t s) { run(fn.base, s, o0, wc[(rot++) % ncopy], x, NE, NEXP, b); }, bytes},
        {"cand f16_matvec_batched_h20", [&](hipStream_t s) { run(fn.h20, s, o1, wc[(rot++) % ncopy], x, NE, NEXP, b); }, bytes},
    };
    char tag[128]; snprintf(tag, sizeof tag, "REVIEW router mv 384x5120 b=%u ncopy=%u graph", b, ncopy);
    kb::AbOpts o; o.graph = true; o.inner = 20; o.rounds = 40; o.tag = tag;
    kb::ab(vs, o);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: review_mv cmp | time <b> <ncopy>\n"); return 64; }
    kb::init();
    std::string dir = "."; if (const char* d = getenv("REVIEW_DIR")) dir = d;
    kb::Module mb(dir + "/base_f16_matvec_gfx1201.hsaco"), mc(dir + "/cand_router_mv_gfx1201.hsaco");
    Fns fn{mb.fn("f16_matvec_batched"), mc.fn("f16_matvec_batched_h20")};
    std::string mode = argv[1];
    if (mode == "cmp") return mode_cmp(fn);
    if (mode == "time" && argc >= 4) return mode_time(fn, atoi(argv[2]), atoi(argv[3]));
    return 64;
}
