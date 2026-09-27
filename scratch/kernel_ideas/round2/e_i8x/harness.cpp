// (e) int8-activation WMMA GEMMs for the dGPU dense prefill sites vs the production f16x arm.
// Production chain = f32_to_f16_cast_2d (rows b, pitch cols+64) -> the production f16x tile
// (V41_F16X_DB_BN64 / V41_F16X_256 selectors at this b). Candidate chain = q8_0_quantize_f32_wave
// (+grid pad) -> i8x kernel. Cold weights (flush between timed blocks), direct launches (prefill).
// Numerics vs a double-accumulated reference of the dequantised weights times the f32 activations.
//   ./harness <shape> <b> [flushMB]      shape: qa kv qb woa wob shg shd
#include "kbench.h"
#include <cmath>
#include <cstring>
#include <string>

struct Shape { const char* name; unsigned M, K, G; const char* prod; unsigned prod_bn, prod_bm; };

__global__ void ref_gemm(double* out, const unsigned char* w, const float* x, unsigned K, unsigned M, unsigned G,
                         unsigned batch, unsigned ldx) {
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;  // g*M + m
    const unsigned n = blockIdx.y;
    if (col >= G * M || n >= batch) return;
    const unsigned g = col / M;
    const unsigned blocks = K / 32;
    const unsigned char* row = w + (size_t)col * blocks * 34;
    const _Float16* sc = (const _Float16*)row;
    const int8_t* q = (const int8_t*)(row + blocks * 2);
    const float* xr = x + (size_t)n * ldx + (size_t)g * K;
    double acc = 0.0;
    for (unsigned k = 0; k < K; ++k) acc += (double)((float)sc[k / 32] * (float)q[k]) * (double)xr[k];
    out[(size_t)n * G * M + col] = acc;
}
__global__ void fix_scales(unsigned char* w, size_t rows, unsigned blocks, uint32_t seed) {
    size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; r < rows; r += (size_t)gridDim.x * blockDim.x) {
        _Float16* sc = (_Float16*)(w + r * blocks * 34);
        for (unsigned b = 0; b < blocks; ++b) {
            uint32_t h = (uint32_t)(r * 131 + b) * 2654435761u ^ seed;
            h ^= h >> 15;
            sc[b] = (_Float16)(0.002f + 0.004f * ((h & 1023) / 1023.0f));
        }
    }
}

int main(int argc, char** argv) {
    kb::init();
    const std::string sname = argc > 1 ? argv[1] : "qb";
    const unsigned b = argc > 2 ? atoi(argv[2]) : 512;
    const unsigned flush_mb = argc > 3 ? atoi(argv[3]) : 64;
    const Shape shapes[] = {
        {"qa", 1280, 5120, 1, "q8_0_gemm_wmma_f16x_db_bn64", 64, 128}, {"kv", 512, 5120, 1, "q8_0_gemm_wmma_f16x_db_bn64", 64, 128},
        {"qb", 32768, 1280, 1, "q8_0_gemm_wmma_f16x_256x128", 128, 256}, {"woa", 1024, 4096, 8, "q8_0_gemm_wmma_f16x_256x128", 128, 256},
        {"wob", 5120, 8192, 1, "q8_0_gemm_wmma_f16x", 128, 128}, {"shg", 2304, 5120, 1, "q8_0_gemm_wmma_f16x", 128, 128},
        {"shd", 5120, 2304, 1, "q8_0_gemm_wmma_f16x", 128, 128}};
    const Shape* sp = nullptr;
    for (auto& s : shapes) if (sname == s.name) sp = &s;
    if (!sp) { fprintf(stderr, "unknown shape\n"); return 64; }
    Shape s = *sp;
    if ((s.name == std::string("qb") || s.name == std::string("woa")) && b < 192) { s.prod = "q8_0_gemm_wmma_f16x"; s.prod_bm = 128; }
    if (s.name == std::string("qa") && b > 512) { s.prod = "q8_0_gemm_wmma_f16x"; s.prod_bn = 128; }
    const unsigned ldx = s.G * s.K, pitch16 = ldx + 64, blocks = s.K / 32, ncols = s.G * s.M;
    kb::Module mw("q8_0_matvec_wmma_gfx1201.hsaco"), mq("q8_0_matvec_gfx1201.hsaco"), mk("q8_k_quantize_gfx1201.hsaco"),
        mc("cand_i8x_gfx1201.hsaco");
    hipFunction_t f_prod = mw.fn(s.prod), f_cast = mk.fn("f32_to_f16_cast_2d"), f_qw = mq.fn("q8_0_quantize_f32_wave");
    struct C { const char* sym; unsigned bn, bm; };
    const C cands[] = {{"i8x_db_ld", 128, 128}, {"i8x_128_ld", 128, 128}, {"i8x_256x128_ld", 128, 256},
                       {"i8x_256x128_db_ld", 128, 256}, {"i8x_bn64_ld", 64, 128}};
    const size_t wbytes = (size_t)ncols * blocks * 34;
    unsigned char* w = kb::dalloc<unsigned char>(wbytes);
    kb::fill_bytes(w, wbytes, 7);
    hipLaunchKernelGGL(fix_scales, dim3(1024), dim3(256), 0, 0, w, (size_t)ncols, blocks, 99u);
    float* x = kb::dalloc<float>((size_t)b * ldx);
    kb::fill_f32(x, (size_t)b * ldx, 11, -2.f, 2.f);
    uint16_t* x16 = kb::dalloc<uint16_t>((size_t)b * pitch16);
    int8_t* xq = kb::dalloc<int8_t>((size_t)b * ldx);
    float* xs = kb::dalloc<float>((size_t)b * ldx / 32);
    float* out0 = kb::dalloc<float>((size_t)b * ncols);
    float* out1 = out0;  // one output buffer (VRAM budget); outputs are copied to the host in between
    const unsigned RB = b < 32 ? b : 32;  // rows checked against the f64 reference
    KB_CHECK(hipDeviceSynchronize());
    fprintf(stderr, "[e] %s M=%u K=%u G=%u b=%u: weight %.1f MB, out %.1f MB, x %.1f MB, flush %u MB\n", s.name, s.M, s.K, s.G, b,
            wbytes / 1e6, (double)b * ncols * 4 / 1e6, (double)b * ldx * 4 / 1e6, flush_mb);
    auto cast = [&](hipStream_t st) {
        const unsigned threads = b * (ldx / 8);
        kb::launch(f_cast, dim3((threads + 255) / 256 + 1), dim3(256), 0, st, x16, (const float*)x, b, ldx, pitch16); };
    auto quant = [&](hipStream_t st) {
        const unsigned nb = b * ldx / 32;
        kb::launch(f_qw, dim3((nb + 7) / 8 + 1), dim3(256), 0, st, xq, xs, (const float*)x, nb); };
    auto prod = [&](hipStream_t st) {
        kb::launch(f_prod, dim3((b + s.prod_bn - 1) / s.prod_bn, s.M / s.prod_bm, s.G), dim3(256), 0, st,
                   out0, (const unsigned char*)w, (const uint16_t*)x16, s.K, s.M, s.G, b, blocks, pitch16); };
    std::vector<hipFunction_t> cf;
    for (auto& c : cands) cf.push_back(mc.fn(c.sym));
    auto cand = [&](hipStream_t st, int i) {
        const C& c = cands[i];
        kb::launch(cf[i], dim3((b + c.bn - 1) / c.bn, s.M / c.bm, s.G), dim3(256), 0, st,
                   out1, (const unsigned char*)w, (const int8_t*)xq, (const float*)xs, s.K, s.M, s.G, b, blocks, ldx); };
    // ---------------- numerics
    cast(0); prod(0); quant(0);
    KB_CHECK(hipDeviceSynchronize());
    auto hprod = kb::d2h(out0, (size_t)b * ncols);
    double* ref = kb::dalloc<double>((size_t)RB * ncols);
    hipLaunchKernelGGL(ref_gemm, dim3((ncols + 255) / 256, RB), dim3(256), 0, 0, ref, (const unsigned char*)w, (const float*)x, s.K, s.M, s.G, RB, ldx);
    KB_CHECK(hipDeviceSynchronize());
    auto href = kb::d2h(ref, (size_t)RB * ncols);
    KB_CHECK(hipFree(ref));
    std::vector<float> reff(href.size());
    for (size_t i = 0; i < href.size(); ++i) reff[i] = (float)href[i];
    const size_t nref = (size_t)RB * ncols;
    kb::print_cmp((std::string(s.name) + " prod f16x vs f64 ref").c_str(), kb::compare_f32(reff.data(), hprod.data(), nref));
    for (size_t i = 0; i < sizeof(cands) / sizeof(cands[0]); ++i) {
        if (s.M % cands[i].bm) continue;
        KB_CHECK(hipMemset(out1, 0xFF, (size_t)b * ncols * 4));
        cand(0, (int)i);
        KB_CHECK(hipDeviceSynchronize());
        auto h = kb::d2h(out1, (size_t)b * ncols);
        kb::print_cmp((std::string(s.name) + " " + cands[i].sym + " vs f64 ref").c_str(), kb::compare_f32(reff.data(), h.data(), nref));
        kb::print_cmp((std::string(s.name) + " " + cands[i].sym + " vs prod f16x").c_str(), kb::compare_f32(hprod.data(), h.data(), h.size()));
    }
    // ---------------- timing (direct launches, cold weights)
    kb::Flusher fl((size_t)flush_mb << 20);
    std::vector<kb::Variant> vs = {
        {std::string("prod: cast + ") + s.prod, [&](hipStream_t st) { cast(st); prod(st); }, (double)wbytes},
        {std::string("prod GEMM alone ") + s.prod, [&](hipStream_t st) { prod(st); }, (double)wbytes},
    };
    for (size_t i = 0; i < sizeof(cands) / sizeof(cands[0]); ++i) {
        if (s.M % cands[i].bm) continue;
        vs.push_back({std::string("cand: quant + ") + cands[i].sym, [&, i](hipStream_t st) { quant(st); cand(st, (int)i); }, (double)wbytes});
        vs.push_back({std::string("cand GEMM alone ") + cands[i].sym, [&, i](hipStream_t st) { cand(st, (int)i); }, (double)wbytes});
    }
    kb::AbOpts o; o.graph = false; o.inner = 1; o.rounds = getenv("E_ROUNDS") ? atoi(getenv("E_ROUNDS")) : 20; o.warm_ms = 100;
    o.between = [&](hipStream_t st) { fl(st); };
    std::string tag = std::string(s.name) + " M=" + std::to_string(s.M) + " K=" + std::to_string(s.K) + " G=" + std::to_string(s.G) +
                      " b=" + std::to_string(b) + " cold(flush " + std::to_string(flush_mb) + "MB) direct";
    o.tag = tag.c_str();
    kb::ab(vs, o);
    return 0;
}
