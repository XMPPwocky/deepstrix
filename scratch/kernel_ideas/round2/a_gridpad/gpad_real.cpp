// gpad_real: production kernels (code objects built from the UNMODIFIED in-tree .hip with
// $KFLAGS_V41) launched with the production grid vs the same grid + 1 work-group in x (every
// kernel here already guards x: extra WGs exit / are treated as sentinels), interleaved kb::ab,
// graph mode. Outputs of padded vs exact launches compared byte-for-byte.
// usage: ./gpad_real <section> [b ...]   sections: gather idxq comp1 fp8 rope fp4 cast rmsb kvapp
#include "kbench.h"
#include <cstring>
#include <string>

static std::string D = ".";
static std::string ARCH = "gfx1201";
static int ROUNDS = 40;
static const unsigned TOP_K = 512, HEAD_DIM = 512;

static hipFunction_t fn(kb::Module& m, const char* n) { return m.fn(n); }

template <typename T>
static bool same(const T* a, const T* b, size_t n, const char* tag) {
    auto ha = kb::d2h(a, n), hb = kb::d2h(b, n);
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i) diff += memcmp(&ha[i], &hb[i], sizeof(T)) != 0;
    printf("CMP %s: %s (diff=%zu of %zu)\n", tag, diff ? "MISMATCH" : "bytes-identical", diff, n);
    return diff == 0;
}

static void run_ab(std::vector<kb::Variant>& vs, const std::string& tag, bool graph, int inner,
                   std::function<void(hipStream_t)> between = nullptr) {
    kb::AbOpts o; o.graph = graph; o.inner = inner; o.rounds = ROUNDS; o.warm_ms = 60; o.tag = tag.c_str();
    o.between = between;
    kb::ab(vs, o);
}

// ------------------------------------------------------------------ gather (comp KV top-k gather)
static void sec_gather(const std::vector<unsigned>& bs) {
    kb::Module m(D + "/base_indexer_gather_" + ARCH + ".hsaco");
    hipFunction_t f_old = fn(m, "indexer_gather_batched"), f_u4 = fn(m, "gather_u4_r1");
    const unsigned store_rows = 32768;  // 32 MB f16 store, flushed (64 MB) between timed blocks
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    uint16_t* comp = kb::dalloc<uint16_t>((size_t)store_rows * HEAD_DIM);
    kb::fill_f16(comp, (size_t)store_rows * HEAD_DIM, 9);
    uint16_t* d0 = kb::dalloc<uint16_t>((size_t)bmax * TOP_K * HEAD_DIM);
    uint16_t* d1 = kb::dalloc<uint16_t>((size_t)bmax * TOP_K * HEAD_DIM);
    int* sel = kb::dalloc<int>((size_t)bmax * TOP_K);
    int* basep = kb::dalloc<int>(bmax);
    KB_CHECK(hipMemset(basep, 0, bmax * 4));
    std::vector<int> hs((size_t)bmax * TOP_K);
    uint32_t s = 12345;
    for (auto& v : hs) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; v = (int)(s % store_rows); }
    for (unsigned b = 0; b < bmax; ++b) hs[(size_t)b * TOP_K + 7] = -1;  // a sentinel per row
    KB_CHECK(hipMemcpy(sel, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
    kb::Flusher fl(64u << 20);
    for (unsigned b : bs) {
        const size_t n = (size_t)b * TOP_K * HEAD_DIM;
        auto old = [=](hipStream_t st, uint16_t* d, unsigned pad) {
            kb::launch(f_old, dim3(TOP_K + pad, b, 2), dim3(256), 0, st, d, (const uint16_t*)comp, (const int*)sel, TOP_K, HEAD_DIM, (const int*)basep);
        };
        auto u4 = [=](hipStream_t st, uint16_t* d, unsigned pad) {
            kb::launch(f_u4, dim3(TOP_K + pad, b), dim3(HEAD_DIM * 2 / 16), 0, st, d, (const uint16_t*)comp, (const int*)sel, TOP_K, HEAD_DIM, (const int*)basep);
        };
        for (int k = 0; k < 2; ++k) {
            KB_CHECK(hipMemset(d0, 0x5a, n * 2)); KB_CHECK(hipMemset(d1, 0x5a, n * 2));
            if (k == 0) { old(0, d0, 0); old(0, d1, 1); } else { u4(0, d0, 0); u4(0, d1, 1); }
            KB_CHECK(hipDeviceSynchronize());
            same(d0, d1, n, (std::string(k ? "gather_u4_r1" : "indexer_gather_batched") + " pad1 vs exact b=" + std::to_string(b)).c_str());
        }
        old(0, d0, 0); KB_CHECK(hipDeviceSynchronize());
        u4(0, d1, 0); KB_CHECK(hipDeviceSynchronize());
        same(d0, d1, n, ("gather_u4_r1 vs indexer_gather_batched b=" + std::to_string(b)).c_str());
        std::vector<kb::Variant> vs = {
            {"old (512,b,2)x256 exact", [=](hipStream_t st) { old(st, d0, 0); }, 2.0 * n * 2},
            {"old (513,b,2)x256 +1", [=](hipStream_t st) { old(st, d0, 1); }, 2.0 * n * 2},
            {"u4 (512,b)x64 exact", [=](hipStream_t st) { u4(st, d0, 0); }, 2.0 * n * 2},
            {"u4 (513,b)x64 +1", [=](hipStream_t st) { u4(st, d0, 1); }, 2.0 * n * 2},
        };
        run_ab(vs, "gather b=" + std::to_string(b) + " COLD store (flush 64MB), graph inner=1", true, 1, [&](hipStream_t st) { fl(st); });
        run_ab(vs, "gather b=" + std::to_string(b) + " WARM store, graph inner=10", true, 10);
    }
}

// ------------------------------------------------------------------ f16_matvec_batched (idx-q / ratio-1 compressor)
static void sec_f16mv(const std::vector<unsigned>& bs, unsigned n_rows, unsigned k, const char* name,
                      const char* mod = "f16_matvec", const char* sym = "f16_matvec_batched", unsigned rows_per_wg = 8) {
    kb::Module m(D + "/base_" + mod + "_" + ARCH + ".hsaco");
    hipFunction_t f = fn(m, sym);
    const size_t wel = (size_t)n_rows * k;
    // rotate over enough weight copies to exceed the 64 MB MALL (cold weights, as production)
    const int copies = std::max(1, (int)((80ull << 20) / (wel * 2)) + 1);
    std::vector<uint16_t*> W;
    for (int c = 0; c < copies; ++c) { W.push_back(kb::dalloc<uint16_t>(wel)); kb::fill_f16(W.back(), wel, 3 + c, -0.05f, 0.05f); }
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    float* x = kb::dalloc<float>((size_t)bmax * k); kb::fill_f32(x, (size_t)bmax * k, 5);
    float* o0 = kb::dalloc<float>((size_t)bmax * n_rows);
    float* o1 = kb::dalloc<float>((size_t)bmax * n_rows);
    const unsigned gx = (n_rows + rows_per_wg - 1) / rows_per_wg;
    for (unsigned b : bs) {
        auto L = [=](hipStream_t st, float* o, const uint16_t* w, unsigned pad) {
            kb::launch(f, dim3(gx + pad, 1, b), dim3(256), 0, st, o, w, (const float*)x, k, n_rows);
        };
        L(0, o0, W[0], 0); L(0, o1, W[0], 1); KB_CHECK(hipDeviceSynchronize());
        same(o0, o1, (size_t)b * n_rows, (std::string(name) + " pad1 vs exact b=" + std::to_string(b)).c_str());
        int* rot = new int(0);
        auto rotw = [=]() { int c = (*rot)++ % (int)W.size(); return (const uint16_t*)W[c]; };
        std::vector<kb::Variant> vs = {
            {std::string("exact (") + std::to_string(gx) + ",1,b)", [=](hipStream_t st) { L(st, o0, rotw(), 0); }, (double)wel * 2},
            {std::string("+1 (") + std::to_string(gx + 1) + ",1,b)", [=](hipStream_t st) { L(st, o0, rotw(), 1); }, (double)wel * 2},
        };
        // graph capture bakes the weight pointer of each captured call: capture `inner`=copies calls
        // so every timed block streams every copy once (cold, as production).
        run_ab(vs, std::string(name) + " " + std::to_string(n_rows) + "x" + std::to_string(k) + " b=" + std::to_string(b) +
               " COLD W (" + std::to_string(copies) + " copies), graph inner=" + std::to_string(copies), true, copies);
        if (getenv("GP_WARM")) {
            std::vector<kb::Variant> vw = {
                {"exact warm", [=](hipStream_t st) { L(st, o0, W[0], 0); }},
                {"+1 warm", [=](hipStream_t st) { L(st, o0, W[0], 1); }},
            };
            run_ab(vw, std::string(name) + " b=" + std::to_string(b) + " WARM W, graph inner=10", true, 10);
        }
    }
}

// ------------------------------------------------------------------ vec_add_inplace / hc_post_from_split_batched
static void sec_vecadd(const std::vector<unsigned>& bs) {
    kb::Module m(D + "/base_vec_add_" + ARCH + ".hsaco");
    hipFunction_t f = fn(m, "vec_add_inplace");
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    float* o = kb::dalloc<float>((size_t)bmax * 5120); float* r = kb::dalloc<float>((size_t)bmax * 5120);
    kb::fill_f32(o, (size_t)bmax * 5120, 71); kb::fill_f32(r, (size_t)bmax * 5120, 72, -1e-3f, 1e-3f);
    for (unsigned b : bs) {
        const unsigned n = 5120 * b, g = (n + 255) / 256;
        std::vector<kb::Variant> vs = {
            {"exact", [=](hipStream_t st) { kb::launch(f, dim3(g), dim3(256), 0, st, o, (const float*)r, n); }},
            {"+1", [=](hipStream_t st) { kb::launch(f, dim3(g + 1), dim3(256), 0, st, o, (const float*)r, n); }},
        };
        run_ab(vs, "vec_add_inplace 5120 b=" + std::to_string(b) + " grid=" + std::to_string(g) + " warm, graph inner=10", true, 10);
    }
}
static void sec_hcpost(const std::vector<unsigned>& bs) {
    kb::Module m(D + "/base_hc_post_" + ARCH + ".hsaco");
    hipFunction_t f = fn(m, "hc_post_from_split_batched");
    const unsigned NE = 5120, NH = 4, NW = 24, SS = NW + NH + NH * NH;
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    float* out = kb::dalloc<float>((size_t)bmax * NH * NE); float* out1 = kb::dalloc<float>((size_t)bmax * NH * NE);
    float* bo = kb::dalloc<float>((size_t)bmax * NE); float* rh = kb::dalloc<float>((size_t)bmax * NH * NE);
    float* sp = kb::dalloc<float>((size_t)bmax * SS);
    kb::fill_f32(bo, (size_t)bmax * NE, 81); kb::fill_f32(rh, (size_t)bmax * NH * NE, 82); kb::fill_f32(sp, (size_t)bmax * SS, 83);
    for (unsigned b : bs) {
        auto L = [=](hipStream_t st, float* o, unsigned pad) {
            kb::launch(f, dim3(NE / 256 + pad, NH, b), dim3(256), 0, st, o, (const float*)bo, (const float*)rh, (const float*)sp, NW, NE, NH);
        };
        L(0, out, 0); L(0, out1, 1); KB_CHECK(hipDeviceSynchronize());
        same(out, out1, (size_t)b * NH * NE, ("hc_post pad1 vs exact b=" + std::to_string(b)).c_str());
        std::vector<kb::Variant> vs = {
            {"exact (20,4,b)", [=](hipStream_t st) { L(st, out, 0); }},
            {"+1 (21,4,b)", [=](hipStream_t st) { L(st, out, 1); }},
        };
        run_ab(vs, "hc_post_from_split_batched b=" + std::to_string(b) + " warm, graph inner=10", true, 10);
    }
}

// ------------------------------------------------------------------ fp8_act_quant_inplace (window KV), in place
static void sec_fp8(const std::vector<unsigned>& bs) {
    kb::Module m(D + "/base_fp4_kv_quant_" + ARCH + ".hsaco");
    hipFunction_t f = fn(m, "fp8_act_quant_inplace");
    const unsigned W = 512;
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    float* src = kb::dalloc<float>((size_t)bmax * W); kb::fill_f32(src, (size_t)bmax * W, 11, -3.f, 3.f);
    float* x0 = kb::dalloc<float>((size_t)bmax * W);
    float* x1 = kb::dalloc<float>((size_t)bmax * W);
    for (unsigned b : bs) {
        const size_t n = (size_t)b * W;
        KB_CHECK(hipMemcpy(x0, src, n * 4, hipMemcpyDeviceToDevice)); KB_CHECK(hipMemcpy(x1, src, n * 4, hipMemcpyDeviceToDevice));
        kb::launch(f, dim3(b), dim3(W), 0, 0, x0, b, W); kb::launch(f, dim3(b + 1), dim3(W), 0, 0, x1, b, W);
        KB_CHECK(hipDeviceSynchronize());
        same(x0, x1, n, ("fp8_act_quant_inplace pad1 vs exact b=" + std::to_string(b)).c_str());
        std::vector<kb::Variant> vs = {
            {"exact (b)x512", [=](hipStream_t st) { kb::launch(f, dim3(b), dim3(W), 0, st, x0, b, W); }},
            {"+1 (b+1)x512", [=](hipStream_t st) { kb::launch(f, dim3(b + 1), dim3(W), 0, st, x0, b, W); }},
        };
        run_ab(vs, "fp8_act_quant_inplace b=" + std::to_string(b) + " warm, graph inner=10", true, 10);
    }
}

// ------------------------------------------------------------------ rope_tail_batched (q: 64 heads; idx q: 32 heads)
static void sec_rope(const std::vector<unsigned>& bs) {
    kb::Module m(D + "/base_rope_tail_" + ARCH + ".hsaco");
    hipFunction_t f = fn(m, "rope_tail_batched");
    const unsigned HD_Q = 512, HD_I = 128, NROT = 64;
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    // x0 holds the largest timed shape; src/x1 (correctness) only up to 64 rows (VRAM budget)
    const unsigned bcmp = std::min(bmax, 64u);
    float* src = kb::dalloc<float>((size_t)bcmp * 64 * HD_Q); kb::fill_f32(src, (size_t)bcmp * 64 * HD_Q, 21);
    float* x0 = kb::dalloc<float>((size_t)bmax * 64 * HD_Q);
    float* x1 = kb::dalloc<float>((size_t)bcmp * 64 * HD_Q);
    int* pos = kb::dalloc<int>(bmax);
    std::vector<int> hp(bmax); for (unsigned i = 0; i < bmax; ++i) hp[i] = 1000 + 3 * i;
    KB_CHECK(hipMemcpy(pos, hp.data(), bmax * 4, hipMemcpyHostToDevice));
    const float theta = powf(10000.f, -2.0f / NROT);
    for (unsigned b : bs) {
        for (int site = 0; site < 2; ++site) {
            const unsigned nh = site ? 32 : 64, hd = site ? HD_I : HD_Q;
            const size_t n = (size_t)b * nh * hd;
            auto L = [=](hipStream_t st, float* x, unsigned pad) {
                kb::launch(f, dim3(nh + pad, 1, b), dim3(NROT / 2), 0, st, x, (const int*)pos, nh, hd, NROT, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0);
            };
            std::string nm = std::string(site ? "rope idx-q (32,1,b)" : "rope q (64,1,b)") + " b=" + std::to_string(b);
            if (b <= bcmp) {
                KB_CHECK(hipMemcpy(x0, src, n * 4, hipMemcpyDeviceToDevice)); KB_CHECK(hipMemcpy(x1, src, n * 4, hipMemcpyDeviceToDevice));
                L(0, x0, 0); L(0, x1, 1); KB_CHECK(hipDeviceSynchronize());
                same(x0, x1, n, (nm + " pad1 vs exact").c_str());
            }
            std::vector<kb::Variant> vs = {
                {"exact", [=](hipStream_t st) { L(st, x0, 0); }},
                {"+1 in x", [=](hipStream_t st) { L(st, x0, 1); }},
            };
            run_ab(vs, nm + " warm, graph inner=10", true, 10);
        }
    }
}

// ------------------------------------------------------------------ indexer_fp4 (32 heads x b rows of 128), in place
static void sec_fp4(const std::vector<unsigned>& bs) {
    kb::Module m(D + "/base_indexer_qat_" + ARCH + ".hsaco");
    hipFunction_t f = fn(m, "indexer_fp4");
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    float* src = kb::dalloc<float>((size_t)bmax * 32 * 128); kb::fill_f32(src, (size_t)bmax * 32 * 128, 31, -2.f, 2.f);
    float* x0 = kb::dalloc<float>((size_t)bmax * 32 * 128);
    float* x1 = kb::dalloc<float>((size_t)bmax * 32 * 128);
    for (unsigned b : bs) {
        const unsigned rows = 32 * b; const size_t n = (size_t)rows * 128;
        KB_CHECK(hipMemcpy(x0, src, n * 4, hipMemcpyDeviceToDevice)); KB_CHECK(hipMemcpy(x1, src, n * 4, hipMemcpyDeviceToDevice));
        kb::launch(f, dim3(rows), dim3(128), 0, 0, x0, rows); kb::launch(f, dim3(rows + 1), dim3(128), 0, 0, x1, rows);
        KB_CHECK(hipDeviceSynchronize());
        same(x0, x1, n, ("indexer_fp4 pad1 vs exact b=" + std::to_string(b)).c_str());
        std::vector<kb::Variant> vs = {
            {"exact (32b)x128", [=](hipStream_t st) { kb::launch(f, dim3(rows), dim3(128), 0, st, x0, rows); }},
            {"+1", [=](hipStream_t st) { kb::launch(f, dim3(rows + 1), dim3(128), 0, st, x0, rows); }},
        };
        run_ab(vs, "indexer_fp4 b=" + std::to_string(b) + " warm, graph inner=10", true, 10);
    }
}

// ------------------------------------------------------------------ f32_to_f16_cast_2d (f16x arm inputs)
static void sec_cast(const std::vector<unsigned>& bs) {
    kb::Module m(D + "/base_q8_k_quantize_" + ARCH + ".hsaco");
    hipFunction_t f = fn(m, "f32_to_f16_cast_2d");
    const unsigned cols_list[] = {32768, 8192, 5120, 1280};
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    size_t cap = 0; for (unsigned b : bs) for (unsigned c : cols_list) cap = std::max(cap, (size_t)b * c);
    cap = std::min(cap, (size_t)6 << 20);
    float* x = kb::dalloc<float>(cap); kb::fill_f32(x, cap, 41);
    uint16_t* o0 = kb::dalloc<uint16_t>(cap); uint16_t* o1 = kb::dalloc<uint16_t>(cap);
    for (unsigned b : bs) for (unsigned cols : cols_list) {
        const size_t n = (size_t)b * cols; if (n > cap) continue;
        const unsigned threads = b * (cols / 8), g = (threads + 255) / 256;
        const unsigned waves = g * 8;
        if (!getenv("GP_ALL") && waves % 2048 != 0 && (waves + 64) % 2048 > 64) { continue; }  // only grids in/near a 2048-wave multiple
        auto L = [=](hipStream_t st, uint16_t* o, unsigned pad) { kb::launch(f, dim3(g + pad), dim3(256), 0, st, o, (const float*)x, b, cols, cols); };
        L(0, o0, 0); L(0, o1, 1); KB_CHECK(hipDeviceSynchronize());
        std::string nm = "cast_2d b=" + std::to_string(b) + " cols=" + std::to_string(cols) + " grid=" + std::to_string(g);
        same(o0, o1, n, (nm + " pad1 vs exact").c_str());
        std::vector<kb::Variant> vs = {
            {"exact", [=](hipStream_t st) { L(st, o0, 0); }, 6.0 * n},
            {"+1", [=](hipStream_t st) { L(st, o0, 1); }, 6.0 * n},
        };
        run_ab(vs, nm + " warm, graph inner=10", true, 10);
    }
}

// ------------------------------------------------------------------ kernels WITHOUT an x guard: b vs b+1 REAL rows
static void sec_rmsb(const std::vector<unsigned>& bs) {
    kb::Module m(D + "/base_rms_norm_" + ARCH + ".hsaco");
    hipFunction_t f = fn(m, "rms_norm_weighted_batched_fast");
    const unsigned ns[] = {512, 1280, 5120};
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    float* x = kb::dalloc<float>((size_t)(bmax + 1) * 5120); kb::fill_f32(x, (size_t)(bmax + 1) * 5120, 51);
    float* w = kb::dalloc<float>(5120); kb::fill_f32(w, 5120, 52, 0.5f, 1.5f);
    float* o = kb::dalloc<float>((size_t)(bmax + 1) * 5120);
    for (unsigned b : bs) for (unsigned n : ns) {
        std::vector<kb::Variant> vs = {
            {"b rows (exact grid b)", [=](hipStream_t st) { kb::launch(f, dim3(b), dim3(256), 0, st, o, (const float*)x, (const float*)w, n, 1e-6f); }},
            {"b+1 rows (grid b+1)", [=](hipStream_t st) { kb::launch(f, dim3(b + 1), dim3(256), 0, st, o, (const float*)x, (const float*)w, n, 1e-6f); }},
        };
        run_ab(vs, "rms_norm_weighted_batched_fast n=" + std::to_string(n) + " b=" + std::to_string(b) + " warm, graph inner=10", true, 10);
    }
}
static void sec_kvapp(const std::vector<unsigned>& bs) {
    kb::Module m(D + "/base_kv_cache_append_" + ARCH + ".hsaco");
    hipFunction_t f = fn(m, "kv_cache_append_batched");
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    float* kv = kb::dalloc<float>((size_t)(bmax + 1) * HEAD_DIM); kb::fill_f32(kv, (size_t)(bmax + 1) * HEAD_DIM, 61);
    uint16_t* cache = kb::dalloc<uint16_t>((size_t)(bmax + 1) * HEAD_DIM);
    for (unsigned b : bs) {
        std::vector<kb::Variant> vs = {
            {"b rows", [=](hipStream_t st) { kb::launch(f, dim3(b), dim3(HEAD_DIM), 0, st, cache, (const float*)kv, 0u, HEAD_DIM, (const int*)nullptr); }},
            {"b+1 rows", [=](hipStream_t st) { kb::launch(f, dim3(b + 1), dim3(HEAD_DIM), 0, st, cache, (const float*)kv, 0u, HEAD_DIM, (const int*)nullptr); }},
        };
        run_ab(vs, "kv_cache_append_batched b=" + std::to_string(b) + " warm, graph inner=10", true, 10);
    }
}

// ------------------------------------------------------------------ q8_0_quantize_f32 (the C1 case), for the iGPU check
static void sec_quant(const std::vector<unsigned>& bs) {
    kb::Module m(D + "/base_q8_0_matvec_" + ARCH + ".hsaco");
    hipFunction_t f = fn(m, "q8_0_quantize_f32");
    const unsigned K = 32768;
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    float* x = kb::dalloc<float>((size_t)bmax * K); kb::fill_f32(x, (size_t)bmax * K, 91, -3.f, 3.f);
    int8_t* q0 = kb::dalloc<int8_t>((size_t)bmax * K); int8_t* q1 = kb::dalloc<int8_t>((size_t)bmax * K);
    float* s0 = kb::dalloc<float>((size_t)bmax * K / 32); float* s1 = kb::dalloc<float>((size_t)bmax * K / 32);
    for (unsigned b : bs) {
        const unsigned blocks = K / 32 * b;
        kb::launch(f, dim3(blocks), dim3(32), 0, 0, q0, s0, (const float*)x, blocks);
        kb::launch(f, dim3(blocks + 1), dim3(32), 0, 0, q1, s1, (const float*)x, blocks);
        KB_CHECK(hipDeviceSynchronize());
        same(q0, q1, (size_t)blocks * 32, ("quantize xq pad1 vs exact b=" + std::to_string(b)).c_str());
        std::vector<kb::Variant> vs = {
            {"exact grid=blocks", [=](hipStream_t st) { kb::launch(f, dim3(blocks), dim3(32), 0, st, q0, s0, (const float*)x, blocks); }},
            {"+1", [=](hipStream_t st) { kb::launch(f, dim3(blocks + 1), dim3(32), 0, st, q0, s0, (const float*)x, blocks); }},
        };
        run_ab(vs, "q8_0_quantize_f32 K=32768 b=" + std::to_string(b) + " blocks=" + std::to_string(blocks) + " " + ARCH + " warm, graph inner=10", true, 10);
    }
}

int main(int argc, char** argv) {
    kb::init();
    if (getenv("GP_DIR")) D = getenv("GP_DIR");
    if (getenv("GP_ARCH")) ARCH = getenv("GP_ARCH");
    if (getenv("GP_ROUNDS")) ROUNDS = atoi(getenv("GP_ROUNDS"));
    std::string sec = argc > 1 ? argv[1] : "gather";
    std::vector<unsigned> bs;
    for (int i = 2; i < argc; ++i) bs.push_back((unsigned)atoi(argv[i]));
    if (sec == "gather") sec_gather(bs.empty() ? std::vector<unsigned>{1, 2, 3, 4, 5, 6, 8} : bs);
    else if (sec == "idxq") sec_f16mv(bs.empty() ? std::vector<unsigned>{1, 2, 3, 4, 5, 8} : bs, 4096, 1280, "idx-q f16_matvec_batched");
    else if (sec == "comp1") sec_f16mv(bs.empty() ? std::vector<unsigned>{1, 2, 3, 4, 5, 8} : bs, 512, 5120, "ratio-1 compressor f16_matvec_batched");
    else if (sec == "proj") sec_f16mv(bs.empty() ? std::vector<unsigned>{16, 64} : bs, 32, 5120, "idx proj f16_matvec_batched");
    else if (sec == "router") sec_f16mv(bs.empty() ? std::vector<unsigned>{16, 32, 64} : bs, 384, 5120, "router f16_matvec_batched_h20", "f16_matvec", "f16_matvec_batched_h20");
    else if (sec == "narrow") sec_f16mv(bs.empty() ? std::vector<unsigned>{32, 64} : bs, 24, 20480, "mhc narrow f16_matvec_narrow_batched", "f16_matvec_narrow", "f16_matvec_narrow_batched", 1);
    else if (sec == "vecadd") sec_vecadd(bs.empty() ? std::vector<unsigned>{16, 64, 128} : bs);
    else if (sec == "hcpost") sec_hcpost(bs.empty() ? std::vector<unsigned>{4, 8, 16, 64} : bs);
    else if (sec == "quant") sec_quant(bs.empty() ? std::vector<unsigned>{1, 2, 3, 4, 8} : bs);
    else if (sec == "fp8") sec_fp8(bs.empty() ? std::vector<unsigned>{128, 256, 512} : bs);
    else if (sec == "rope") sec_rope(bs.empty() ? std::vector<unsigned>{16, 32, 64, 512} : bs);
    else if (sec == "fp4") sec_fp4(bs.empty() ? std::vector<unsigned>{16, 64, 512} : bs);
    else if (sec == "cast") sec_cast(bs.empty() ? std::vector<unsigned>{16, 32, 64, 128, 512} : bs);
    else if (sec == "rmsb") sec_rmsb(bs.empty() ? std::vector<unsigned>{64, 128, 256, 512} : bs);
    else if (sec == "kvapp") sec_kvapp(bs.empty() ? std::vector<unsigned>{128, 256, 512} : bs);
    else { fprintf(stderr, "unknown section\n"); return 64; }
    return 0;
}
