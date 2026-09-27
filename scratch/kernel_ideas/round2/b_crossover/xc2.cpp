// C2_dense_prefill harness: production dGPU dense GEMM kernels at production shapes.
//
//   ./harness gfx1201 <dir> <shape> [--b N] [--warm] [--rounds R] [--inner I] [--short]
//                                   [--cand k1,k2,...] [--corr] [--flushmb N]
//
// shapes: qa qb kv woa wob shg shd  (f16x, prefill b=512 default)
//         engram [--m M]            (lds_tiled, b=64 chunk; M defaults to 12800 = half, see NOTES)
//         cast_<cols>               (f32_to_f16_cast_2d rows=b)
//         qb64 kv64 wob64           (replay: q8_0_gemv_batched_warp8 grid.z=b, b=64 default)
//         woa64                     (replay: q8_0_grouped_gemv_batched grid.z=b)
// --cand names: candidate kernels in cand_gfx1201.hsaco; launch geometry per name in cand_launch().
// --corr: correctness of every candidate vs the baseline on identical inputs, incl. tail batches.
#include "kbench.h"
#include <map>

static std::string g_arch, g_dir;
static bool g_corr = false;

struct Shape {
    std::string name;
    enum Kind { F16X, LDS_TILED, GEMV_B, GGEMV_B, CAST } kind;
    unsigned M, K, G, b, ldx;   // for GGEMV_B: M=rank, K=group_dim, G=n_groups
};

static const unsigned PAD = 64;
static Shape shape_of(const std::string& n, unsigned b, unsigned mover) {
    auto p = [](unsigned d) { return d + PAD; };
    if (n == "qa") return {n, Shape::F16X, 1280, 5120, 1, b, p(5120)};
    if (n == "qb") return {n, Shape::F16X, 32768, 1280, 1, b, p(1280)};
    if (n == "kv") return {n, Shape::F16X, 512, 5120, 1, b, p(5120)};
    if (n == "woa") return {n, Shape::F16X, 1024, 4096, 8, b, p(32768)};
    if (n == "wob") return {n, Shape::F16X, 5120, 8192, 1, b, p(8192)};
    if (n == "shg") return {n, Shape::F16X, 2304, 5120, 1, b, p(5120)};
    if (n == "shd") return {n, Shape::F16X, 5120, 2304, 1, b, p(2304)};
    if (n == "engram") return {n, Shape::LDS_TILED, mover ? mover : 12800, 6144, 1, b, 0};
    if (n.rfind("cast_", 0) == 0) return {n, Shape::CAST, 0, (unsigned)atoi(n.c_str() + 5), 1, b, p((unsigned)atoi(n.c_str() + 5))};
    if (n == "qb64") return {n, Shape::GEMV_B, 32768, 1280, 1, b, p(1280)};
    if (n == "kv64") return {n, Shape::GEMV_B, 512, 5120, 1, b, p(5120)};
    if (n == "wob64") return {n, Shape::GEMV_B, 5120, 8192, 1, b, p(8192)};
    if (n == "woa64") return {n, Shape::GGEMV_B, 1024, 4096, 8, b, p(32768)};
    if (n == "qa64") return {n, Shape::GEMV_B, 1280, 5120, 1, b, p(5120)};
    if (n == "shg64") return {n, Shape::GEMV_B, 2304, 5120, 1, b, p(5120)};
    if (n == "shd64") return {n, Shape::GEMV_B, 5120, 2304, 1, b, p(2304)};
    fprintf(stderr, "unknown shape %s\n", n.c_str());
    exit(64);
}

// Device buffers for one shape (sized for the LARGEST batch the run will use).
struct Bufs {
    unsigned char* w = nullptr;   // [G*M, K/32*34]
    uint16_t* x16 = nullptr;      // [bmax, ldx]
    int8_t* xq = nullptr;         // [bmax, G*K]
    float* xs = nullptr;          // [bmax, G*K/32]
    float* xf = nullptr;          // [bmax, G*K] f32 (cast source)
    float* out = nullptr;         // [bmax, G*M]
    float* out2 = nullptr;        // candidate output
    size_t wbytes = 0, outn = 0;
};

static void fix_scales_f16(unsigned char* w, unsigned rows, unsigned blocks, uint32_t seed) {
    // Q8_0 row = [blocks f16 scales][blocks*32 int8]. Random bytes give NaN/inf/huge scales; set
    // every scale to a random f16 in [0.002, 0.02] on the host once.
    std::vector<uint16_t> sc((size_t)rows * blocks);
    uint32_t x = seed | 1u;
    for (auto& s : sc) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        float f = 0.002f + 0.018f * ((x >> 8) * (1.0f / 16777216.0f));
        _Float16 h = (_Float16)f;
        memcpy(&s, &h, 2);
    }
    size_t rowbytes = (size_t)blocks * 34;
    for (unsigned r = 0; r < rows; ++r)
        KB_CHECK(hipMemcpy(w + r * rowbytes, sc.data() + (size_t)r * blocks, blocks * 2, hipMemcpyHostToDevice));
}

static Bufs alloc_bufs(const Shape& s, unsigned bmax) {
    Bufs B;
    unsigned blocks = s.K / 32;
    if (s.kind == Shape::CAST) {
        B.xf = kb::dalloc<float>((size_t)bmax * s.K);
        B.x16 = kb::dalloc<uint16_t>((size_t)bmax * s.ldx);
        B.out = (float*)kb::dalloc<uint16_t>((size_t)bmax * s.ldx);   // second f16 output (candidate)
        kb::fill_f32(B.xf, (size_t)bmax * s.K, 7);
        return B;
    }
    B.wbytes = (size_t)s.G * s.M * blocks * 34;
    B.w = kb::dalloc<unsigned char>(B.wbytes);
    kb::fill_bytes(B.w, B.wbytes, 11);
    fix_scales_f16(B.w, s.G * s.M, blocks, 13);
    B.outn = (size_t)bmax * s.G * s.M;
    B.out = kb::dalloc<float>(B.outn);
    KB_CHECK(hipMemset(B.out, 0, B.outn * 4));
    if (g_corr) { B.out2 = kb::dalloc<float>(B.outn); KB_CHECK(hipMemset(B.out2, 0, B.outn * 4)); }
    else B.out2 = B.out;   // timing only: every variant writes the same buffer
    if (s.kind == Shape::F16X) {
        B.x16 = kb::dalloc<uint16_t>((size_t)bmax * s.ldx);
        kb::fill_f16(B.x16, (size_t)bmax * s.ldx, 17, -1.f, 1.f);
    } else {
        B.xq = kb::dalloc<int8_t>((size_t)bmax * s.G * s.K);
        B.xs = kb::dalloc<float>((size_t)bmax * s.G * blocks);
        kb::fill_bytes(B.xq, (size_t)bmax * s.G * s.K, 19);
        kb::fill_f32(B.xs, (size_t)bmax * s.G * blocks, 23, 0.002f, 0.02f);
        if (s.kind == Shape::GEMV_B || s.kind == Shape::GGEMV_B) {
            // f16 twin of the same activation (for the f16x-at-b=64 comparison): x16 = xq*xs
            B.x16 = kb::dalloc<uint16_t>((size_t)bmax * s.ldx);
            std::vector<int8_t> hq = kb::d2h(B.xq, (size_t)bmax * s.G * s.K);
            std::vector<float> hs = kb::d2h(B.xs, (size_t)bmax * s.G * blocks);
            std::vector<uint16_t> h16((size_t)bmax * s.ldx, 0);
            for (unsigned r = 0; r < bmax; ++r)
                for (unsigned c = 0; c < s.G * s.K; ++c) {
                    _Float16 h = (_Float16)((float)hq[(size_t)r * s.G * s.K + c] * hs[(size_t)r * s.G * blocks + c / 32]);
                    memcpy(&h16[(size_t)r * s.ldx + c], &h, 2);
                }
            KB_CHECK(hipMemcpy(B.x16, h16.data(), h16.size() * 2, hipMemcpyHostToDevice));
        }
    }
    return B;
}

static double bytes_of(const Shape& s, unsigned b) {
    unsigned blocks = s.K / 32;
    double w = (double)s.G * s.M * blocks * 34;
    switch (s.kind) {
        case Shape::F16X: return w + (double)b * s.G * s.K * 2 + (double)b * s.G * s.M * 4;
        case Shape::LDS_TILED: return w + (double)b * s.K + (double)b * blocks * 4 + (double)b * s.M * 4;
        case Shape::GEMV_B: case Shape::GGEMV_B: return w + (double)b * s.G * s.K + (double)b * s.G * s.M * 4;  // ONE weight read (roofline), not the b re-reads
        case Shape::CAST: return (double)b * s.K * 6;
    }
    return 0;
}
static double flops_of(const Shape& s, unsigned b) { return s.kind == Shape::CAST ? 0 : 2.0 * b * s.G * s.M * s.K; }

// ---- launches (exactly the Rust wrapper geometry) --------------------------------------------
static void launch_base(hipFunction_t f, const Shape& s, const Bufs& B, unsigned b, float* out, hipStream_t st) {
    unsigned blocks = s.K / 32;
    switch (s.kind) {
        case Shape::F16X:   // q8_0.rs:604: grid (ceil(b/128), m/128, n_groups) x 256
            kb::launch(f, dim3((b + 127) / 128, s.M / 128, s.G), dim3(256), 0, st,
                       out, (const unsigned char*)B.w, (const uint16_t*)B.x16, s.K, s.M, s.G, b, blocks, s.ldx);
            break;
        case Shape::LDS_TILED:  // q8_0.rs:644: grid (M/64, ceil(b/64)) x 128
            kb::launch(f, dim3(s.M / 64, (b + 63) / 64, 1), dim3(128), 0, st,
                       out, (const unsigned char*)B.w, (const int8_t*)B.xq, (const float*)B.xs, s.K, s.M, b, blocks);
            break;
        case Shape::GEMV_B:  // q8_0.rs:262: grid (ceil(M/8), 1, b) x 256
            kb::launch(f, dim3((s.M + 7) / 8, 1, b), dim3(256), 0, st,
                       out, (const unsigned char*)B.w, (const int8_t*)B.xq, (const float*)B.xs, s.K, s.M, blocks);
            break;
        case Shape::GGEMV_B:  // q8_0.rs:870: grid (ceil(G*rank/8), 1, b) x 256
            kb::launch(f, dim3((s.G * s.M + 7) / 8, 1, b), dim3(256), 0, st,
                       out, (const unsigned char*)B.w, (const int8_t*)B.xq, (const float*)B.xs, s.K, s.M, blocks, s.G);
            break;
        case Shape::CAST: {  // q8_k.rs:126: grid ceil(rows*cols/8 / 256) x 256
            size_t threads = (size_t)b * (s.K / 8);
            kb::launch(f, dim3((unsigned)((threads + 255) / 256)), dim3(256), 0, st,
                       (unsigned short*)out, (const float*)B.xf, b, s.K, s.ldx);
            break;
        }
    }
}

// Candidate launch table: name -> geometry. Keep signatures identical to the baseline where possible.
static bool cand_launch(const std::string& name_in, hipFunction_t f, const Shape& s, const Bufs& B, unsigned b, float* out, hipStream_t st) {
    std::string name = name_in.rfind("small:", 0) == 0 ? name_in.substr(6) : name_in;
    if (name.rfind("mod:", 0) == 0) name = name.substr(name.find(':', 4) + 1);
    unsigned blocks = s.K / 32;
    // f16x twins with the production 128x128 geometry and signature
    if (name == "q8_0_gemm_wmma_f16x_v2" || name == "q8_0_gemm_wmma_f16x_pf2" || name == "q8_0_gemm_wmma_f16x_lb" || name == "q8_0_gemm_wmma_f16x_pf2b"
        || name == "q8_0_gemm_wmma_f16x_pf3" || name == "q8_0_gemm_wmma_f16x_pf4" || name.rfind("q8_0_gemm_wmma_f16x_abl", 0) == 0 || name == "q8_0_gemm_wmma_f16x_pf2n"
        || name == "q8_0_gemm_wmma_f16x_db" || name == "q8_0_gemm_wmma_f16x_db_pf1" || name == "q8_0_gemm_wmma_f16x_db_pf3"
        || name == "q8_0_gemm_wmma_f16x_pf2_w0" || name == "q8_0_gemm_wmma_f16x_db_w0") {
        kb::launch(f, dim3((b + 127) / 128, s.M / 128, s.G), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const uint16_t*)B.x16, s.K, s.M, s.G, b, blocks, s.ldx);
        return true;
    }
    // f16x big tiles: grid (ceil(b/BN), M/256, G) x (waves*32)
    if (name == "q8_0_gemm_wmma_f16x_256x128" || name == "q8_0_gemm_wmma_f16x_256x128_pf2" || name == "q8_0_gemm_wmma_f16x_256x128_db") {
        if (s.M % 256) { fprintf(stderr, "M %% 256\n"); return false; }
        kb::launch(f, dim3((b + 127) / 128, s.M / 256, s.G), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const uint16_t*)B.x16, s.K, s.M, s.G, b, blocks, s.ldx);
        return true;
    }
    if (name == "q8_0_gemm_wmma_f16x_256x256" || name == "q8_0_gemm_wmma_f16x_256x256_pf2") {
        if (s.M % 256) { fprintf(stderr, "M %% 256\n"); return false; }
        kb::launch(f, dim3((b + 255) / 256, s.M / 256, s.G), dim3(512), 0, st,
                   out, (const unsigned char*)B.w, (const uint16_t*)B.x16, s.K, s.M, s.G, b, blocks, s.ldx);
        return true;
    }
    // f16x 64x128 tile (BM=64 weight rows, BN=128 tokens): grid (ceil(b/128), M/64, G) x 256
    if (name == "q8_0_gemm_wmma_f16x_bm64") {
        kb::launch(f, dim3((b + 127) / 128, s.M / 64, s.G), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const uint16_t*)B.x16, s.K, s.M, s.G, b, blocks, s.ldx);
        return true;
    }
    // f16x 128x64 tile (BM=128, BN=64 tokens): grid (ceil(b/64), M/128, G) x 256
    if (name == "q8_0_gemm_wmma_f16x_bn64" || name == "q8_0_gemm_wmma_f16x_db_bn64") {
        kb::launch(f, dim3((b + 63) / 64, s.M / 128, s.G), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const uint16_t*)B.x16, s.K, s.M, s.G, b, blocks, s.ldx);
        return true;
    }
    // f16x 64x64 tile: grid (ceil(b/64), M/64, G) x 128
    if (name == "q8_0_gemm_wmma_f16x_64x64") {
        kb::launch(f, dim3((b + 63) / 64, s.M / 64, s.G), dim3(128), 0, st,
                   out, (const unsigned char*)B.w, (const uint16_t*)B.x16, s.K, s.M, s.G, b, blocks, s.ldx);
        return true;
    }
    // replay: f16x at b=64 launched on the GEMV shapes (production geometry of f16x), x16 twin input
    if (name == "f16x_at_b" && (s.kind == Shape::GEMV_B || s.kind == Shape::GGEMV_B)) {
        kb::launch(f, dim3((b + 127) / 128, s.M / 128, s.G), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const uint16_t*)B.x16, s.K, s.M, s.G, b, blocks, s.ldx);
        return true;
    }
    if (name == "f16x_bn64_at_b" && (s.kind == Shape::GEMV_B || s.kind == Shape::GGEMV_B)) {
        kb::launch(f, dim3((b + 63) / 64, s.M / 128, s.G), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const uint16_t*)B.x16, s.K, s.M, s.G, b, blocks, s.ldx);
        return true;
    }
    // replay: b-packed dp4a GEMV, batch <= 64, grid (M/8, 1, 1) x 256, same signature as bpack_warp8
    if (name == "q8_0_gemv_bpack64_warp8" && s.kind == Shape::GEMV_B) {
        kb::launch(f, dim3((s.M + 7) / 8, 1, 1), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const int8_t*)B.xq, (const float*)B.xs, s.K, s.M, blocks, b);
        return true;
    }
    // replay: b-packed dp4a GEMV, 8 rows per wave-pair ... (variants share the bpack signature)
    if ((name == "q8_0_gemv_bpack64_lds" || name == "q8_0_gemv_bpack64_w4") && s.kind == Shape::GEMV_B) {
        kb::launch(f, dim3((s.M + 7) / 8, 1, 1), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const int8_t*)B.xq, (const float*)B.xs, s.K, s.M, blocks, b);
        return true;
    }
    // replay: production bpack16 on grid.z = ceil(b/16) chunks (weight re-read 4x at b=64)
    if (name == "q8_0_gemv_bpack_z16" && s.kind == Shape::GEMV_B) {
        kb::launch(f, dim3((s.M + 7) / 8, 1, (b + 15) / 16), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const int8_t*)B.xq, (const float*)B.xs, s.K, s.M, blocks, b);
        return true;
    }
    if (name == "q8_0_grouped_gemv_bpack_z16" && s.kind == Shape::GGEMV_B) {
        kb::launch(f, dim3((s.G * s.M + 7) / 8, 1, (b + 15) / 16), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const int8_t*)B.xq, (const float*)B.xs, s.K, s.M, blocks, s.G, b);
        return true;
    }
    if (name == "q8_0_grouped_gemv_bpack64" && s.kind == Shape::GGEMV_B) {
        kb::launch(f, dim3((s.G * s.M + 7) / 8, 1, 1), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const int8_t*)B.xq, (const float*)B.xs, s.K, s.M, blocks, s.G, b);
        return true;
    }
    // Engram: f16x on the lds_tiled shape needs f16 activations -> not comparable here; a lds_tiled twin:
    if (name == "q8_0_gemm_wmma_lds_tiled_v2" && s.kind == Shape::LDS_TILED) {
        kb::launch(f, dim3(s.M / 64, (b + 63) / 64, 1), dim3(128), 0, st,
                   out, (const unsigned char*)B.w, (const int8_t*)B.xq, (const float*)B.xs, s.K, s.M, b, blocks);
        return true;
    }
    if ((name == "q8_0_gemm_wmma_i8x" || name == "i8x_r1" || name == "q8_0_gemm_wmma_i8x_pf2" || name == "q8_0_gemm_wmma_i8x_pf3" || name == "q8_0_gemm_wmma_i8x_grd"
         || name == "q8_0_gemm_wmma_i8x_aon" || name == "q8_0_gemm_wmma_i8x_grd_aon" || name == "q8_0_gemm_wmma_i8x_db_aon"
         || name.rfind("r1c_i8x", 0) == 0 || name == "q8_0_gemm_wmma_i8x_w0" || name == "q8_0_gemm_wmma_i8x_db_w0"
         || name == "q8_0_gemm_wmma_i8x_db" || name == "q8_0_gemm_wmma_i8x_db_grd") && s.kind == Shape::LDS_TILED) {  // 128x128 tile, i8 activations dequantised at stage
        kb::launch(f, dim3((b + 127) / 128, s.M / 128, 1), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const int8_t*)B.xq, (const float*)B.xs, s.K, s.M, 1u, b, blocks);
        return true;
    }
    if (name == "q8_0_gemm_wmma_i8x_bn64" && s.kind == Shape::LDS_TILED) {  // 128x64 tile
        kb::launch(f, dim3((b + 63) / 64, s.M / 128, 1), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const int8_t*)B.xq, (const float*)B.xs, s.K, s.M, 1u, b, blocks);
        return true;
    }
    if (name == "f32_to_f16_cast_2d_v2" && s.kind == Shape::CAST) {
        size_t threads = (size_t)b * (s.K / 8);
        kb::launch(f, dim3((unsigned)((threads + 255) / 256)), dim3(256), 0, st,
                   (unsigned short*)out, (const float*)B.xf, b, s.K, s.ldx);
        return true;
    }
    return false;
}

static const char* base_kernel(const Shape& s) {
    switch (s.kind) {
        case Shape::F16X: return "q8_0_gemm_wmma_f16x";
        case Shape::LDS_TILED: return "q8_0_gemm_wmma_lds_tiled";
        case Shape::GEMV_B: return "q8_0_gemv_batched_warp8";
        case Shape::GGEMV_B: return "q8_0_grouped_gemv_batched";
        case Shape::CAST: return "f32_to_f16_cast_2d";
    }
    return "";
}
static std::string base_module(const Shape& s) {
    switch (s.kind) {
        case Shape::F16X: case Shape::LDS_TILED: return g_dir + "/base_q8_0_matvec_wmma_" + g_arch + ".hsaco";
        case Shape::GEMV_B: return g_dir + "/base_q8_0_matvec_" + g_arch + ".hsaco";
        case Shape::GGEMV_B: return g_dir + "/base_q8_0_grouped_matvec_" + g_arch + ".hsaco";
        case Shape::CAST: return g_dir + "/base_q8_k_quantize_" + g_arch + ".hsaco";
    }
    return "";
}

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> o; size_t p = 0;
    while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) o.push_back(s.substr(p, q - p)); p = q + 1; }
    return o;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: harness <arch> <dir> <shape> [opts]\n"); return 64; }
    g_arch = argv[1]; g_dir = argv[2];
    std::string shape = argv[3];
    unsigned b = 0, mover = 0, flushmb = 0;
    bool warm = false, shortmode = false, corr = false;
    int rounds = 60, inner = 0;
    std::vector<std::string> cands;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--b") b = atoi(argv[++i]);
        else if (a == "--m") mover = atoi(argv[++i]);
        else if (a == "--warm") warm = true;
        else if (a == "--short") shortmode = true;
        else if (a == "--corr") corr = g_corr = true;
        else if (a == "--rounds") rounds = atoi(argv[++i]);
        else if (a == "--inner") inner = atoi(argv[++i]);
        else if (a == "--flushmb") flushmb = atoi(argv[++i]);
        else if (a == "--cand") cands = split(argv[++i]);
        else { fprintf(stderr, "bad arg %s\n", a.c_str()); return 64; }
    }
    Shape s0 = shape_of(shape, 1, mover);
    if (b == 0) b = (s0.kind == Shape::GEMV_B || s0.kind == Shape::GGEMV_B || s0.kind == Shape::LDS_TILED) ? 64 : 512;
    Shape s = shape_of(shape, b, mover);
    hipDeviceProp_t prop = kb::init();
    fprintf(stderr, "[kb] LDS/CU %zu B, regs/block %d, L2 %d B\n", (size_t)prop.sharedMemPerMultiprocessor, prop.regsPerBlock, prop.l2CacheSize);

    kb::Module base(base_module(s));
    hipFunction_t fbase = base.fn(base_kernel(s));
    std::string candpath = g_dir + "/cand_" + g_arch + ".hsaco";
    kb::Module* cm = nullptr;
    std::vector<std::pair<std::string, hipFunction_t>> cfs;
    if (!cands.empty()) {
        cm = new kb::Module(candpath);
        for (auto& c : cands) {
            // "f16x_at_b" / "f16x_bn64_at_b" are pseudo-names: production f16x (base module) / candidate bn64 on the GEMV shape.
            if (c.rfind("mod:", 0) == 0) {   // mod:<basename>:<kernel> -> <dir>/<basename>_<arch>.hsaco
                size_t p = c.find(':', 4);
                static std::map<std::string, kb::Module*> mods;
                std::string base = c.substr(4, p - 4);
                if (!mods.count(base)) mods[base] = new kb::Module(g_dir + "/" + base + "_" + g_arch + ".hsaco");
                cfs.push_back({c, mods[base]->fn(c.c_str() + p + 1)});
            }
            else if (c.rfind("small:", 0) == 0) { static kb::Module sm(g_dir + "/cand_small_" + g_arch + ".hsaco"); cfs.push_back({c, sm.fn(c.c_str() + 6)}); }
            else if (c == "f16x_at_b") { static kb::Module wm(g_dir + "/base_q8_0_matvec_wmma_" + g_arch + ".hsaco"); cfs.push_back({c, wm.fn("q8_0_gemm_wmma_f16x") }); }
            else if (c == "f16x_bn64_at_b") cfs.push_back({c, cm->fn("q8_0_gemm_wmma_f16x_bn64")});
            else if (c == "i8x_r1") { static kb::Module r1(g_dir + "/cand_r1_" + g_arch + ".hsaco"); cfs.push_back({c, r1.fn("q8_0_gemm_wmma_i8x")}); }
            else cfs.push_back({c, cm->fn(c.c_str())});
        }
    }

    unsigned bmax = b;
    Bufs B = alloc_bufs(s, bmax);
    double ws_mb = (B.wbytes + B.outn * (g_corr ? 8.0 : 4.0) + (B.x16 ? (double)bmax * s.ldx * 2 : 0.0)
                    + (B.xq ? (double)bmax * s.G * s.K * 1.125 : 0.0) + (B.xf ? (double)bmax * s.K * 4 : 0.0)) / 1048576.0;
    fprintf(stderr, "[kb] shape %s b=%u M=%u K=%u G=%u ldx=%u working set ~%.0f MB\n", s.name.c_str(), b, s.M, s.K, s.G, s.ldx, ws_mb);

    // ---- correctness (candidates vs baseline, identical inputs; also tail batches) -----------
    if (corr && !cfs.empty()) {
        std::vector<unsigned> bs = {b};
        if (s.kind == Shape::F16X) for (unsigned t : {1u, 3u, 17u, 100u, 129u, 255u, 256u, 500u}) if (t < b) bs.push_back(t);
        if (s.kind == Shape::GEMV_B || s.kind == Shape::GGEMV_B) for (unsigned t : {1u, 5u, 17u, 33u, 63u}) if (t < b) bs.push_back(t);
        if (s.kind == Shape::LDS_TILED) for (unsigned t : {1u, 17u, 63u}) if (t < b) bs.push_back(t);
        for (unsigned bb : bs) {
            Shape sb = shape_of(shape, bb, mover);
            KB_CHECK(hipMemset(B.out, 0, B.outn * 4));
            launch_base(fbase, sb, B, bb, B.out, 0);
            KB_CHECK(hipDeviceSynchronize());
            size_t n = (size_t)bb * s.G * s.M;
            auto ref = kb::d2h(B.out, n);
            for (auto& [name, f] : cfs) {
                KB_CHECK(hipMemset(B.out2, 0, B.outn * 4));
                if (!cand_launch(name, f, sb, B, bb, B.out2, 0)) { fprintf(stderr, "no launch rule for %s on %s\n", name.c_str(), shape.c_str()); continue; }
                KB_CHECK(hipDeviceSynchronize());
                auto got = kb::d2h(B.out2, n);
                char tag[128]; snprintf(tag, sizeof tag, "%s b=%u %s", shape.c_str(), bb, name.c_str());
                kb::print_cmp(tag, kb::compare_f32(ref.data(), got.data(), n));
            }
        }
    }

    // ---- timing ----------------------------------------------------------------------------
    std::vector<kb::Variant> vs;
    vs.push_back({std::string("base ") + base_kernel(s), [&](hipStream_t st) { launch_base(fbase, s, B, b, B.out, st); }, bytes_of(s, b), flops_of(s, b)});
    for (auto& [name, f] : cfs) {
        std::string nm = name; hipFunction_t ff = f;
        vs.push_back({"cand " + nm, [&, nm, ff](hipStream_t st) { if (!cand_launch(nm, ff, s, B, b, B.out, st)) exit(3); }, bytes_of(s, b), flops_of(s, b)});
    }
    kb::AbOpts o;
    if (inner == 0) inner = warm ? 5 : 1;   // cold regime: ONE call per flushed block
    o.rounds = shortmode ? 2 : rounds;
    o.inner = shortmode ? 1 : inner;
    o.warm_ms = shortmode ? 50 : 300;
    kb::Flusher* fl = nullptr;
    if (!warm) {
        if (flushmb == 0) { double left = 400.0 - ws_mb; flushmb = left > 96 ? 96 : (left < 24 ? 24 : (unsigned)left); }   // hub DOWN: README allows <= 4 GB dGPU; flush stays 96 MB (> MALL 64 + L2 8)
        fprintf(stderr, "[kb] flush %u MB per timed block (total device ~%.0f MB)\n", flushmb, ws_mb + flushmb);
        fl = new kb::Flusher((size_t)flushmb << 20);
        o.between = [&](hipStream_t st) { (*fl)(st); };
    }
    char tag[256];
    snprintf(tag, sizeof tag, "%s b=%u M=%u K=%u G=%u %s%s", shape.c_str(), b, s.M, s.K, s.G, warm ? "WARM" : "flush", warm ? "" : (std::to_string(flushmb) + "MB").c_str());
    o.tag = tag;
    kb::ab(vs, o);
    KB_CHECK(hipDeviceSynchronize());
    return 0;
}
