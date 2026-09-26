// C1_dense_decode harness: production q8_0 bpack GEMV (baseline .hsaco from the unmodified in-tree
// source) vs candidates, at the decode dense shapes, COLD weights (72 MB clean READ flush before
// every timed block), graph mode, one call per timed block (rotation over copies for small weights).
//
// Several modes in ONE process (one scheduler job), separated by ':' on the command line:
//   ./harness gemv <shape> <b> [ncopies] : gemv qb 4 : grouped 4 : shared 1 ...
// modes:
//   gemv <shape> <b> [ncopies]     shape in {qa,qb,kv,wob,gate,down,engram,head,engram_full,head_full}
//   grouped <b>                    wo_a 8 x (1024 x 4096)
//   quant <K> <b>                  q8_0_quantize_f32 at (K/32*b) blocks
//   swiglu <b>
//   lanes <shape> <b0> <b1>        two lanes: 2 launches (b0, b1) vs one launch (b0+b1)
//   shared <b>                     shared-expert chain gate/up/swiglu/quant/down vs fused candidates
// env: C1_WARM=1 (no flush, inner 10), C1_ROUNDS=N, C1_KU=1 (also time the ku2/ku4 twins),
//      C1_DIR (hsaco dir)
#include "kbench.h"

#include <map>

struct Shape { const char* name; unsigned M, K; };
static Shape shape_of(const std::string& s) {
    static std::map<std::string, Shape> m = {
        {"qa", {"q_a 1280x5120", 1280, 5120}},       {"qb", {"qb 32768x1280", 32768, 1280}},
        {"kv", {"kv 512x5120", 512, 5120}},          {"wob", {"wo_b 5120x8192", 5120, 8192}},
        {"gate", {"sh_gate 2304x5120", 2304, 5120}}, {"down", {"sh_down 5120x2304", 5120, 2304}},
        {"engram", {"engram_proxy 6400x6144", 6400, 6144}},
        {"head", {"head_proxy 6464x5120", 6464, 5120}},
        {"engram_full", {"engram_wkv 25600x6144", 25600, 6144}},
        {"head_full", {"head 129280x5120", 129280, 5120}},
    };
    auto it = m.find(s);
    if (it == m.end()) { fprintf(stderr, "unknown shape %s\n", s.c_str()); exit(64); }
    return it->second;
}

static uint32_t g_rng = 0x9e3779b9u;
static inline uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
static inline uint16_t f2h(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }

// Q8_0 M18 layout: per row [f16 scales (blocks*2 B) | int8 quants (blocks*32 B)], 34 B per block.
static std::vector<uint8_t> make_q8_rows(unsigned M, unsigned K) {
    const unsigned blocks = K / 32;
    const size_t rowb = (size_t)blocks * 34;
    std::vector<uint8_t> h(rowb * M);
    for (unsigned r = 0; r < M; ++r) {
        uint8_t* row = h.data() + r * rowb;
        uint16_t* sc = (uint16_t*)row;
        for (unsigned b = 0; b < blocks; ++b) sc[b] = f2h(((rnd() >> 8) * (1.0f / 16777216.0f) - 0.5f) * 0.05f);
        uint32_t* q = (uint32_t*)(row + blocks * 2);
        for (unsigned i = 0; i < blocks * 8; ++i) q[i] = rnd();
    }
    return h;
}
static void make_acts(unsigned K, unsigned b, std::vector<int8_t>& xq, std::vector<float>& xs) {
    xq.resize((size_t)K * b); xs.resize((size_t)K / 32 * b);
    for (auto& v : xq) v = (int8_t)(rnd() & 0xff);
    for (auto& v : xs) v = ((rnd() >> 8) * (1.0f / 16777216.0f)) * 0.1f + 1e-3f;
}
// CPU reference of the production math (per row: lane-strided partials, then a shfl_down tree).
static void cpu_ref(const std::vector<uint8_t>& w, unsigned M, unsigned K, const std::vector<int8_t>& xq,
                    const std::vector<float>& xs, unsigned b, std::vector<float>& out) {
    const unsigned blocks = K / 32;
    out.assign((size_t)M * b, 0.f);
    for (unsigned r = 0; r < M; ++r) {
        const uint8_t* row = w.data() + (size_t)r * blocks * 34;
        const uint16_t* sc = (const uint16_t*)row;
        const int8_t* q = (const int8_t*)(row + blocks * 2);
        for (unsigned bb = 0; bb < b; ++bb) {
            float lane[32] = {0};
            for (unsigned bl = 0; bl < blocks; ++bl) {
                int32_t dot = 0;
                for (unsigned i = 0; i < 32; ++i) dot += (int32_t)q[bl * 32 + i] * (int32_t)xq[(size_t)bb * K + bl * 32 + i];
                float ws = kb::h2f(sc[bl]);
                lane[bl % 32] += ws * xs[(size_t)bb * blocks + bl] * (float)dot;
            }
            for (int off = 16; off > 0; off >>= 1)
                for (int l = 0; l < off; ++l) lane[l] += lane[l + off];
            out[(size_t)bb * M + r] = lane[0];
        }
    }
}

static const size_t FLUSH_BYTES = 72u << 20;

// READ-based cache flush: streams `n` uint4 through L2+MALL as CLEAN lines (a memset flush leaves
// dirty lines whose write-back is then charged to the kernel under test).
__global__ void c1_read_flush(const uint4* __restrict__ p, size_t n, unsigned* __restrict__ sink) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) { uint4 v = p[i]; acc += v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678u) sink[0] = acc;  // practically never; keeps the loads alive
}
__global__ void c1_null_kernel(unsigned* p) { if (threadIdx.x == 0 && blockIdx.x == 0 && p == nullptr) p[0] = 1; }
struct ReadFlusher {
    uint4* buf = nullptr; size_t n = 0; unsigned* sink = nullptr;
    void init(size_t bytes) {
        if (buf) return;
        n = bytes / 16;
        KB_CHECK(hipMalloc(&buf, bytes)); KB_CHECK(hipMalloc(&sink, 64));
        KB_CHECK(hipMemset(buf, 1, bytes)); KB_CHECK(hipDeviceSynchronize());
    }
    void operator()(hipStream_t s) { hipLaunchKernelGGL(c1_read_flush, dim3(2048), dim3(256), 0, s, (const uint4*)buf, n, sink); }
};
static ReadFlusher g_flush;
static unsigned* g_nullp = nullptr;
static bool g_warm = false;
static int g_rounds = 60;
static void add_null(std::vector<kb::Variant>& vs) {
    if (!g_nullp) g_nullp = kb::dalloc<unsigned>(16);
    vs.push_back({"null 1-WG kernel (launch floor)", [](hipStream_t s) { hipLaunchKernelGGL(c1_null_kernel, dim3(1), dim3(64), 0, s, g_nullp); }, 0.0});
}
static void run_ab(std::vector<kb::Variant>& vs, const std::string& tag, int inner_cold) {
    g_flush.init(g_warm ? (1u << 20) : FLUSH_BYTES);
    kb::AbOpts o;
    o.graph = getenv("C1_NOGRAPH") == nullptr;  // production decode stages are graph-captured
    o.inner = g_warm ? 10 : inner_cold;
    o.rounds = g_rounds;
    if (!g_warm) o.between = [](hipStream_t s) { g_flush(s); };
    std::string t = tag + (g_warm ? " WARM(inner10,no flush)" : " cold(readflush72MB)") + " graph";
    o.tag = t.c_str();
    kb::ab(vs, o);
}

struct Ctx {
    std::string dir;
    kb::Module *base = nullptr, *grp = nullptr, *swi = nullptr, *cand = nullptr, *shared = nullptr;
};

static hipFunction_t try_fn(kb::Module* m, const char* name) {
    if (!m) return nullptr;
    hipFunction_t f;
    if (hipModuleGetFunction(&f, m->m, name) != hipSuccess) return nullptr;
    return f;
}
template <class T> static void dfree(T* p) { if (p) KB_CHECK(hipFree((void*)p)); }

static int run_gemv(Ctx& c, const std::string& sname, unsigned b, unsigned ncopies) {
    Shape sh = shape_of(sname);
    const unsigned M = sh.M, K = sh.K, blocks = K / 32;
    const size_t wbytes = (size_t)M * blocks * 34;
    auto hw = make_q8_rows(M, K);
    std::vector<int8_t> hxq; std::vector<float> hxs;
    make_acts(K, b, hxq, hxs);

    std::vector<uint8_t*> w(ncopies);
    for (unsigned i = 0; i < ncopies; ++i) {
        w[i] = kb::dalloc<uint8_t>(wbytes);
        KB_CHECK(hipMemcpy(w[i], hw.data(), wbytes, hipMemcpyHostToDevice));
    }
    int8_t* xq = kb::dalloc<int8_t>(hxq.size());
    float* xs = kb::dalloc<float>(hxs.size());
    KB_CHECK(hipMemcpy(xq, hxq.data(), hxq.size(), hipMemcpyHostToDevice));
    KB_CHECK(hipMemcpy(xs, hxs.data(), hxs.size() * 4, hipMemcpyHostToDevice));
    float* out_base = kb::dalloc<float>((size_t)M * b);
    float* out_c = kb::dalloc<float>((size_t)M * b);

    hipFunction_t f_base = c.base->fn("q8_0_gemv_bpack_warp8");
    const unsigned gpad = getenv("C1_GPAD") ? atoi(getenv("C1_GPAD")) : 0;  // extra idle WGs (row guard)
    const dim3 grid(M / 8 + gpad), block(256);
    if (gpad) printf("gemv grid pad: %u + %u WGs\n", M / 8, gpad);
    auto launch_base = [&](hipStream_t s, float* o, const uint8_t* ww) {
        kb::launch(f_base, grid, block, 0, s, o, ww, (const int8_t*)xq, (const float*)xs, K, M, blocks, b);
    };

    // ---- correctness: production vs CPU reference of the production math
    launch_base(0, out_base, w[0]);
    KB_CHECK(hipDeviceSynchronize());
    auto ob = kb::d2h(out_base, (size_t)M * b);
    if (M * (size_t)K <= 20000000) {
        std::vector<float> ref;
        cpu_ref(hw, M, K, hxq, hxs, b, ref);
        kb::print_cmp((sname + " base vs cpu_ref b=" + std::to_string(b)).c_str(), kb::compare_f32(ref.data(), ob.data(), (size_t)M * b));
    }

    std::vector<kb::Variant> vs;
    int copy_idx = 0;
    vs.push_back({"base q8_0_gemv_bpack_warp8", [&](hipStream_t s) { launch_base(s, out_base, w[(copy_idx++) % ncopies]); }, (double)wbytes});
    if (b == 1) {
        hipFunction_t f1 = c.base->fn("q8_0_gemv_warp8");
        vs.push_back({"base q8_0_gemv_warp8 (b=1 kernel)", [&, f1](hipStream_t s) {
            kb::launch(f1, grid, block, 0, s, out_c, (const uint8_t*)w[(copy_idx++) % ncopies], (const int8_t*)xq, (const float*)xs, K, M, blocks); }, (double)wbytes});
    }
    struct CandGeo { std::string name; unsigned waves; };
    std::string tb = "q8_0_gemv_bpack_tB" + std::to_string(b);
    std::vector<CandGeo> cnames = {{tb, 8}, {tb + "_w4", 4}, {tb + "_w16", 16}};
    if (getenv("C1_KU")) { cnames.push_back({tb + "_ku2", 8}); cnames.push_back({tb + "_ku4", 8}); }
    for (auto& cg : cnames) {
        hipFunction_t f = try_fn(c.cand, cg.name.c_str());
        if (!f) continue;
        if (M % cg.waves) continue;
        const dim3 g2(M / cg.waves + gpad), b2(32 * cg.waves);
        // correctness on identical inputs
        KB_CHECK(hipMemset(out_c, 0, (size_t)M * b * 4));
        kb::launch(f, g2, b2, 0, 0, out_c, (const uint8_t*)w[0], (const int8_t*)xq, (const float*)xs, K, M, blocks, b);
        KB_CHECK(hipDeviceSynchronize());
        auto oc = kb::d2h(out_c, (size_t)M * b);
        kb::print_cmp((sname + " " + cg.name + " vs base b=" + std::to_string(b)).c_str(), kb::compare_f32(ob.data(), oc.data(), (size_t)M * b));
        vs.push_back({"cand " + cg.name, [&, f, g2, b2](hipStream_t s) {
            kb::launch(f, g2, b2, 0, s, out_c, (const uint8_t*)w[(copy_idx++) % ncopies], (const int8_t*)xq, (const float*)xs, K, M, blocks, b); }, (double)wbytes});
    }
    add_null(vs);
    run_ab(vs, std::string("gemv ") + sh.name + " b=" + std::to_string(b) + " copies=" + std::to_string(ncopies), (int)ncopies);
    for (auto p : w) dfree(p);
    dfree(xq); dfree(xs); dfree(out_base); dfree(out_c);
    return 0;
}

static int run_lanes(Ctx& c, const std::string& sname, unsigned b0, unsigned b1) {
    // Two lanes with b0 and b1 rows: production = two cold launches (one per lane); candidate = ONE
    // launch over b0+b1 rows (weight read once). Timing: each variant cold; compare
    // med(base b0) + med(base b1) against med(merged).
    Shape sh = shape_of(sname);
    const unsigned M = sh.M, K = sh.K, blocks = K / 32, bt = b0 + b1;
    const size_t wbytes = (size_t)M * blocks * 34;
    auto hw = make_q8_rows(M, K);
    std::vector<int8_t> hxq; std::vector<float> hxs;
    make_acts(K, bt, hxq, hxs);
    uint8_t* w = kb::dalloc<uint8_t>(wbytes);
    KB_CHECK(hipMemcpy(w, hw.data(), wbytes, hipMemcpyHostToDevice));
    int8_t* xq = kb::dalloc<int8_t>(hxq.size());
    float* xs = kb::dalloc<float>(hxs.size());
    KB_CHECK(hipMemcpy(xq, hxq.data(), hxq.size(), hipMemcpyHostToDevice));
    KB_CHECK(hipMemcpy(xs, hxs.data(), hxs.size() * 4, hipMemcpyHostToDevice));
    float* out0 = kb::dalloc<float>((size_t)M * bt);
    float* out1 = kb::dalloc<float>((size_t)M * bt);
    hipFunction_t f_base = c.base->fn("q8_0_gemv_bpack_warp8");
    hipFunction_t f_m = try_fn(c.cand, ("q8_0_gemv_bpack_tB" + std::to_string(bt)).c_str());
    hipFunction_t f_m4 = try_fn(c.cand, ("q8_0_gemv_bpack_tB" + std::to_string(bt) + "_w4").c_str());
    const dim3 grid(M / 8), block(256);
    // correctness: merged output rows must equal the two per-lane outputs
    kb::launch(f_base, grid, block, 0, 0, out0, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, K, M, blocks, b0);
    kb::launch(f_base, grid, block, 0, 0, out0 + (size_t)b0 * M, (const uint8_t*)w, (const int8_t*)(xq + (size_t)b0 * K), (const float*)(xs + (size_t)b0 * blocks), K, M, blocks, b1);
    if (f_m) kb::launch(f_m, grid, block, 0, 0, out1, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, K, M, blocks, bt);
    KB_CHECK(hipDeviceSynchronize());
    auto o0 = kb::d2h(out0, (size_t)M * bt), o1 = kb::d2h(out1, (size_t)M * bt);
    if (f_m) kb::print_cmp((sname + " merged tB" + std::to_string(bt) + " vs 2 lanes").c_str(), kb::compare_f32(o0.data(), o1.data(), (size_t)M * bt));

    std::vector<kb::Variant> vs;
    vs.push_back({"base lane0 b=" + std::to_string(b0), [&](hipStream_t s) {
        kb::launch(f_base, grid, block, 0, s, out0, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, K, M, blocks, b0); }, (double)wbytes});
    vs.push_back({"base lane1 b=" + std::to_string(b1), [&](hipStream_t s) {
        kb::launch(f_base, grid, block, 0, s, out0 + (size_t)b0 * M, (const uint8_t*)w, (const int8_t*)(xq + (size_t)b0 * K), (const float*)(xs + (size_t)b0 * blocks), K, M, blocks, b1); }, (double)wbytes});
    vs.push_back({"base merged b=" + std::to_string(bt) + " (runtime loop)", [&](hipStream_t s) {
        kb::launch(f_base, grid, block, 0, s, out1, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, K, M, blocks, bt); }, (double)wbytes});
    if (f_m) vs.push_back({"cand merged tB" + std::to_string(bt), [&](hipStream_t s) {
        kb::launch(f_m, grid, block, 0, s, out1, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, K, M, blocks, bt); }, (double)wbytes});
    if (f_m4 && M % 4 == 0) vs.push_back({"cand merged tB" + std::to_string(bt) + "_w4", [&](hipStream_t s) {
        kb::launch(f_m4, dim3(M / 4), dim3(128), 0, s, out1, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, K, M, blocks, bt); }, (double)wbytes});
    add_null(vs);
    run_ab(vs, std::string("lanes ") + sh.name + " b0=" + std::to_string(b0) + " b1=" + std::to_string(b1), 1);
    dfree(w); dfree(xq); dfree(xs); dfree(out0); dfree(out1);
    return 0;
}

static int run_grouped(Ctx& c, unsigned b) {
    const unsigned G = 8, GD = 4096, R = 1024, bpg = GD / 32, M = G * R;
    const size_t wbytes = (size_t)M * bpg * 34;
    auto hw = make_q8_rows(M, GD);
    std::vector<int8_t> hxq; std::vector<float> hxs;
    make_acts(G * GD, b, hxq, hxs);
    uint8_t* w = kb::dalloc<uint8_t>(wbytes);
    KB_CHECK(hipMemcpy(w, hw.data(), wbytes, hipMemcpyHostToDevice));
    int8_t* xq = kb::dalloc<int8_t>(hxq.size());
    float* xs = kb::dalloc<float>(hxs.size());
    KB_CHECK(hipMemcpy(xq, hxq.data(), hxq.size(), hipMemcpyHostToDevice));
    KB_CHECK(hipMemcpy(xs, hxs.data(), hxs.size() * 4, hipMemcpyHostToDevice));
    float* out_b = kb::dalloc<float>((size_t)M * b);
    float* out_c = kb::dalloc<float>((size_t)M * b);
    hipFunction_t f_base = c.grp->fn("q8_0_grouped_gemv_bpack");
    const unsigned gpad = getenv("C1_GPAD") ? atoi(getenv("C1_GPAD")) : 0;
    const dim3 grid(M / 8 + gpad), block(256);
    if (gpad) printf("grouped grid pad: %u + %u WGs\n", M / 8, gpad);
    auto lb = [&](hipStream_t s) { kb::launch(f_base, grid, block, 0, s, out_b, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, GD, R, bpg, G, b); };
    lb(0);
    KB_CHECK(hipDeviceSynchronize());
    auto ob = kb::d2h(out_b, (size_t)M * b);
    std::vector<kb::Variant> vs;
    vs.push_back({"base q8_0_grouped_gemv_bpack", lb, (double)wbytes});
    std::string cn = "q8_0_grouped_gemv_bpack_tB" + std::to_string(b);
    for (std::string n : {cn}) {
        hipFunction_t f = try_fn(c.cand, n.c_str());
        if (!f) continue;
        KB_CHECK(hipMemset(out_c, 0, (size_t)M * b * 4));
        kb::launch(f, grid, block, 0, 0, out_c, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, GD, R, bpg, G, b);
        KB_CHECK(hipDeviceSynchronize());
        auto oc = kb::d2h(out_c, (size_t)M * b);
        kb::print_cmp(("wo_a " + n + " vs base b=" + std::to_string(b)).c_str(), kb::compare_f32(ob.data(), oc.data(), (size_t)M * b));
        vs.push_back({"cand " + n, [&, f](hipStream_t s) { kb::launch(f, grid, block, 0, s, out_c, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, GD, R, bpg, G, b); }, (double)wbytes});
    }
    add_null(vs);
    run_ab(vs, "grouped wo_a 8x(1024x4096) b=" + std::to_string(b), 1);
    dfree(w); dfree(xq); dfree(xs); dfree(out_b); dfree(out_c);
    return 0;
}

// quant K b [xoff_B qoff_B soff_B]: the optional byte offsets shift the x / xq / xscale pointers inside
// oversized allocations (aliasing test: the 65536- and 131072-element sizes are 4x slower).
static int run_quant(Ctx& c, unsigned K, unsigned b, size_t xoff = 0, size_t qoff = 0, size_t soff = 0) {
    const unsigned blocks = K / 32 * b;
    const size_t PAD = 1u << 20;
    float* x_base = kb::dalloc<float>((size_t)K * b + PAD / 4);
    float* x = (float*)((char*)x_base + xoff);
    kb::fill_f32(x, (size_t)K * b, 7, -3.f, 3.f);
    int8_t* xq_base = kb::dalloc<int8_t>((size_t)K * b + PAD);
    int8_t* xq = xq_base + qoff;
    float* xs_base = kb::dalloc<float>(blocks + PAD / 4);
    float* xs = (float*)((char*)xs_base + soff);
    int8_t* xq2_base = kb::dalloc<int8_t>((size_t)K * b + PAD);
    int8_t* xq2 = xq2_base + qoff;
    float* xs2_base = kb::dalloc<float>(blocks + PAD / 4);
    float* xs2 = (float*)((char*)xs2_base + soff);
    printf("quant ptrs: x=%p xq=%p xs=%p xq2=%p xs2=%p (offs %zu %zu %zu)\n", (void*)x, (void*)xq, (void*)xs, (void*)xq2, (void*)xs2, xoff, qoff, soff);
    hipFunction_t f = c.base->fn("q8_0_quantize_f32");
    const unsigned gpad = getenv("C1_GPAD") ? atoi(getenv("C1_GPAD")) : 0;  // extra (idle) WGs on the grid
    printf("quant grid: base %u+%u WGs x 32, wave %u+%u WGs x 256\n", blocks, gpad, (blocks + 7) / 8, gpad);
    auto lb = [&](hipStream_t s) { kb::launch(f, dim3(blocks + gpad), dim3(32), 0, s, xq, xs, (const float*)x, blocks); };
    lb(0); KB_CHECK(hipDeviceSynchronize());
    auto q_ref = kb::d2h(xq, (size_t)K * b); auto s_ref = kb::d2h(xs, blocks);
    std::vector<kb::Variant> vs;
    vs.push_back({"base q8_0_quantize_f32 (blocks x 32thr)", lb, (double)K * b * 5});
    hipFunction_t fc = try_fn(c.cand, "q8_0_quantize_f32_wave");
    if (fc) {
        // one wave (32 lanes) per block, 8 blocks per WG
        auto lc = [&](hipStream_t s) { kb::launch(fc, dim3((blocks + 7) / 8 + gpad), dim3(256), 0, s, xq2, xs2, (const float*)x, blocks); };
        lc(0); KB_CHECK(hipDeviceSynchronize());
        auto q2 = kb::d2h(xq2, (size_t)K * b); auto s2 = kb::d2h(xs2, blocks);
        size_t nd = 0; for (size_t i = 0; i < q2.size(); ++i) nd += q2[i] != q_ref[i];
        printf("CMP quant_wave xq K=%u b=%u int8_diff=%zu\n", K, b, nd);
        kb::print_cmp("quant_wave xscale vs base", kb::compare_f32(s_ref.data(), s2.data(), blocks));
        vs.push_back({"cand q8_0_quantize_f32_wave", lc, (double)K * b * 5});
    }
    add_null(vs);
    bool saved = g_warm; g_warm = true;
    run_ab(vs, "quantize K=" + std::to_string(K) + " b=" + std::to_string(b) + " offs=" + std::to_string(xoff) + "/" + std::to_string(qoff) + "/" + std::to_string(soff), 10);
    g_warm = saved;
    dfree(x_base); dfree(xq_base); dfree(xs_base); dfree(xq2_base); dfree(xs2_base);
    return 0;
}

static int run_swiglu(Ctx& c, unsigned b) {
    const unsigned n = 2304 * b;
    float* g = kb::dalloc<float>(n); float* u = kb::dalloc<float>(n); float* o_ = kb::dalloc<float>(n);
    kb::fill_f32(g, n, 3, -12.f, 12.f); kb::fill_f32(u, n, 4, -12.f, 12.f);
    hipFunction_t f = c.swi->fn("swiglu");
    std::vector<kb::Variant> vs;
    vs.push_back({"base swiglu", [&](hipStream_t s) { kb::launch(f, dim3((n + 255) / 256), dim3(256), 0, s, o_, (const float*)g, (const float*)u, n, 10.0f); }, 12.0 * n});
    add_null(vs);
    bool saved = g_warm; g_warm = true;
    run_ab(vs, "swiglu n=2304*" + std::to_string(b), 10);
    g_warm = saved;
    dfree(g); dfree(u); dfree(o_);
    return 0;
}

// Shared-expert chain at decode (FP:1847-1901, b<=8): gate, up, swiglu, quantize_mid, down.
static int run_shared(Ctx& c, unsigned b) {
    const unsigned K = 5120, NFF = 2304, BLK = K / 32, BLKD = NFF / 32, M_D = 5120;
    const size_t wb_gu = (size_t)NFF * BLK * 34, wb_d = (size_t)M_D * BLKD * 34;
    auto hg = make_q8_rows(NFF, K), hu = make_q8_rows(NFF, K), hd = make_q8_rows(M_D, NFF);
    std::vector<int8_t> hxq; std::vector<float> hxs;
    make_acts(K, b, hxq, hxs);
    // activations: make them look like a normed hidden state so gate/up land in a realistic range
    uint8_t *wg = kb::dalloc<uint8_t>(wb_gu), *wu = kb::dalloc<uint8_t>(wb_gu), *wd = kb::dalloc<uint8_t>(wb_d);
    KB_CHECK(hipMemcpy(wg, hg.data(), wb_gu, hipMemcpyHostToDevice));
    KB_CHECK(hipMemcpy(wu, hu.data(), wb_gu, hipMemcpyHostToDevice));
    KB_CHECK(hipMemcpy(wd, hd.data(), wb_d, hipMemcpyHostToDevice));
    int8_t* xq = kb::dalloc<int8_t>(hxq.size());
    float* xs = kb::dalloc<float>(hxs.size());
    KB_CHECK(hipMemcpy(xq, hxq.data(), hxq.size(), hipMemcpyHostToDevice));
    KB_CHECK(hipMemcpy(xs, hxs.data(), hxs.size() * 4, hipMemcpyHostToDevice));
    const size_t nmid = (size_t)NFF * b, nout = (size_t)M_D * b, nblk = (size_t)BLKD * b;
    float *gate_o = kb::dalloc<float>(nmid), *up_o = kb::dalloc<float>(nmid), *mid = kb::dalloc<float>(nmid), *mid_c = kb::dalloc<float>(nmid);
    int8_t *mxq = kb::dalloc<int8_t>(nmid), *mxq_c = kb::dalloc<int8_t>(nmid);
    float *mxs = kb::dalloc<float>(nblk), *mxs_c = kb::dalloc<float>(nblk);
    float *out = kb::dalloc<float>(nout), *out_c = kb::dalloc<float>(nout);
    const float clamp = 10.0f;

    hipFunction_t f_bp = c.base->fn("q8_0_gemv_bpack_warp8"), f_q = c.base->fn("q8_0_quantize_f32"), f_sw = c.swi->fn("swiglu");
    hipFunction_t f_qw = try_fn(c.cand, "q8_0_quantize_f32_wave");
    const dim3 g_gu(NFF / 8), g_d(M_D / 8), blk256(256);
    auto base_chain = [&](hipStream_t s) {
        kb::launch(f_bp, g_gu, blk256, 0, s, gate_o, (const uint8_t*)wg, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, b);
        kb::launch(f_bp, g_gu, blk256, 0, s, up_o, (const uint8_t*)wu, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, b);
        kb::launch(f_sw, dim3((unsigned)((nmid + 255) / 256)), blk256, 0, s, mid, (const float*)gate_o, (const float*)up_o, (unsigned)nmid, clamp);
        kb::launch(f_q, dim3((unsigned)nblk), dim3(32), 0, s, mxq, mxs, (const float*)mid, (unsigned)nblk);
        kb::launch(f_bp, g_d, blk256, 0, s, out, (const uint8_t*)wd, (const int8_t*)mxq, (const float*)mxs, NFF, M_D, BLKD, b);
    };
    base_chain(0); KB_CHECK(hipDeviceSynchronize());
    auto r_mid = kb::d2h(mid, nmid), r_mxs = kb::d2h(mxs, nblk), r_out = kb::d2h(out, nout);
    auto r_mxq = kb::d2h(mxq, nmid);
    {   // range sanity
        float mx = 0; for (float v : r_mid) mx = std::max(mx, fabsf(v));
        printf("shared b=%u: |mid| max %.4g (clamp %.1f)\n", b, mx, clamp);
    }
    auto check = [&](const std::string& n, bool has_mid) {
        KB_CHECK(hipDeviceSynchronize());
        if (has_mid) kb::print_cmp((n + " mid vs base").c_str(), kb::compare_f32(r_mid.data(), kb::d2h(mid_c, nmid).data(), nmid));
        auto q = kb::d2h(mxq_c, nmid); size_t nd = 0; for (size_t i = 0; i < nmid; ++i) nd += q[i] != r_mxq[i];
        printf("CMP %s mid_xq int8_diff=%zu\n", n.c_str(), nd);
        kb::print_cmp((n + " mid_xs vs base").c_str(), kb::compare_f32(r_mxs.data(), kb::d2h(mxs_c, nblk).data(), nblk));
        kb::print_cmp((n + " out vs base").c_str(), kb::compare_f32(r_out.data(), kb::d2h(out_c, nout).data(), nout));
        KB_CHECK(hipMemset(mid_c, 0, nmid * 4)); KB_CHECK(hipMemset(mxq_c, 0, nmid)); KB_CHECK(hipMemset(mxs_c, 0, nblk * 4)); KB_CHECK(hipMemset(out_c, 0, nout * 4));
    };

    std::vector<kb::Variant> vs;
    const double bytes = 2.0 * wb_gu + wb_d;
    vs.push_back({"base chain gate,up,swiglu,quant,down (5 nodes)", base_chain, bytes});
    // A: fused gate+up+swiglu (+ production quantize) + down  (3 nodes)
    std::string an = "shared_gateup_swiglu_tB" + std::to_string(b);
    if (hipFunction_t fa = try_fn(c.shared, an.c_str())) {
        auto chainA = [&, fa](hipStream_t s) {
            kb::launch(fa, g_gu, blk256, 0, s, mid_c, (const uint8_t*)wg, (const uint8_t*)wu, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, clamp);
            kb::launch(f_q, dim3((unsigned)nblk), dim3(32), 0, s, mxq_c, mxs_c, (const float*)mid_c, (unsigned)nblk);
            kb::launch(f_bp, g_d, blk256, 0, s, out_c, (const uint8_t*)wd, (const int8_t*)mxq_c, (const float*)mxs_c, NFF, M_D, BLKD, b);
        };
        chainA(0); check(an + "+quant+down", true);
        vs.push_back({"cand A: fused gate+up+swiglu, quant, down (3 nodes)", chainA, bytes});
        if (f_qw) {
            auto chainA2 = [&, fa](hipStream_t s) {
                kb::launch(fa, g_gu, blk256, 0, s, mid_c, (const uint8_t*)wg, (const uint8_t*)wu, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, clamp);
                kb::launch(f_qw, dim3((unsigned)((nblk + 7) / 8)), blk256, 0, s, mxq_c, mxs_c, (const float*)mid_c, (unsigned)nblk);
                kb::launch(f_bp, g_d, blk256, 0, s, out_c, (const uint8_t*)wd, (const int8_t*)mxq_c, (const float*)mxs_c, NFF, M_D, BLKD, b);
            };
            chainA2(0); check(an + "+quant_wave+down", true);
            vs.push_back({"cand A2: fused gate+up+swiglu, quant_wave, down (3 nodes)", chainA2, bytes});
        }
        // A-only timing (the gate+up+swiglu part in isolation vs its 3 production nodes)
        vs.push_back({"base gate,up,swiglu only (3 nodes)", [&](hipStream_t s) {
            kb::launch(f_bp, g_gu, blk256, 0, s, gate_o, (const uint8_t*)wg, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, b);
            kb::launch(f_bp, g_gu, blk256, 0, s, up_o, (const uint8_t*)wu, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, b);
            kb::launch(f_sw, dim3((unsigned)((nmid + 255) / 256)), blk256, 0, s, mid, (const float*)gate_o, (const float*)up_o, (unsigned)nmid, clamp); }, 2.0 * wb_gu});
        vs.push_back({"cand A only: fused gate+up+swiglu (1 node)", [&, fa](hipStream_t s) {
            kb::launch(fa, g_gu, blk256, 0, s, mid_c, (const uint8_t*)wg, (const uint8_t*)wu, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, clamp); }, 2.0 * wb_gu});
    }
    // B: fused gate+up+swiglu+quantize (WG owns 32 rows) + down  (2 nodes)
    for (unsigned rpw : {1u, 2u, 4u}) {
        std::string bn = "shared_gateup_swiglu_q8_tB" + std::to_string(b) + "_r" + std::to_string(rpw);
        hipFunction_t fb = try_fn(c.shared, bn.c_str());
        if (!fb) continue;
        const dim3 gB(NFF / 32), bB(32 * (32 / rpw));
        auto chainB = [&, fb, gB, bB](hipStream_t s) {
            kb::launch(fb, gB, bB, 0, s, mxq_c, mxs_c, (const uint8_t*)wg, (const uint8_t*)wu, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, clamp);
            kb::launch(f_bp, g_d, blk256, 0, s, out_c, (const uint8_t*)wd, (const int8_t*)mxq_c, (const float*)mxs_c, NFF, M_D, BLKD, b);
        };
        chainB(0); check(bn + "+down", false);
        vs.push_back({"cand B r" + std::to_string(rpw) + ": fused gate+up+swiglu+quant (WG " + std::to_string(32 * (32 / rpw)) + "thr), down (2 nodes)", chainB, bytes});
        vs.push_back({"cand B r" + std::to_string(rpw) + " only: fused gate+up+swiglu+quant (1 node)", [&, fb, gB, bB](hipStream_t s) {
            kb::launch(fb, gB, bB, 0, s, mxq_c, mxs_c, (const uint8_t*)wg, (const uint8_t*)wu, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, clamp); }, 2.0 * wb_gu});
    }
    // Attribution: (i) the 5-node chain with tB gate/up/down (fusion-free tB gain); (ii) B r1 + tB down;
    // (iii) A2 + tB down.
    hipFunction_t f_tb = try_fn(c.cand, ("q8_0_gemv_bpack_tB" + std::to_string(b)).c_str());
    hipFunction_t fb1 = try_fn(c.shared, ("shared_gateup_swiglu_q8_tB" + std::to_string(b) + "_r1").c_str());
    hipFunction_t fa = try_fn(c.shared, an.c_str());
    if (f_tb) {
        auto chain_tb = [&, f_tb](hipStream_t s) {
            kb::launch(f_tb, g_gu, blk256, 0, s, gate_o, (const uint8_t*)wg, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, b);
            kb::launch(f_tb, g_gu, blk256, 0, s, up_o, (const uint8_t*)wu, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, b);
            kb::launch(f_sw, dim3((unsigned)((nmid + 255) / 256)), blk256, 0, s, mid_c, (const float*)gate_o, (const float*)up_o, (unsigned)nmid, clamp);
            kb::launch(f_q, dim3((unsigned)nblk), dim3(32), 0, s, mxq_c, mxs_c, (const float*)mid_c, (unsigned)nblk);
            kb::launch(f_tb, g_d, blk256, 0, s, out_c, (const uint8_t*)wd, (const int8_t*)mxq_c, (const float*)mxs_c, NFF, M_D, BLKD, b);
        };
        chain_tb(0); check("tB chain (5 nodes)", true);
        vs.push_back({"cand tB chain: tB gate, tB up, swiglu, quant, tB down (5 nodes)", chain_tb, bytes});
        if (fb1) {
            const dim3 gB(NFF / 32), bB(1024);
            auto chainB1t = [&, fb1, f_tb, gB, bB](hipStream_t s) {
                kb::launch(fb1, gB, bB, 0, s, mxq_c, mxs_c, (const uint8_t*)wg, (const uint8_t*)wu, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, clamp);
                kb::launch(f_tb, g_d, blk256, 0, s, out_c, (const uint8_t*)wd, (const int8_t*)mxq_c, (const float*)mxs_c, NFF, M_D, BLKD, b);
            };
            chainB1t(0); check("B r1 + tB down", false);
            vs.push_back({"cand B r1 + tB down (2 nodes)", chainB1t, bytes});
        }
        if (fa && f_qw) {
            auto chainA2t = [&, fa, f_tb](hipStream_t s) {
                kb::launch(fa, g_gu, blk256, 0, s, mid_c, (const uint8_t*)wg, (const uint8_t*)wu, (const int8_t*)xq, (const float*)xs, K, NFF, BLK, clamp);
                kb::launch(f_qw, dim3((unsigned)((nblk + 7) / 8)), blk256, 0, s, mxq_c, mxs_c, (const float*)mid_c, (unsigned)nblk);
                kb::launch(f_tb, g_d, blk256, 0, s, out_c, (const uint8_t*)wd, (const int8_t*)mxq_c, (const float*)mxs_c, NFF, M_D, BLKD, b);
            };
            chainA2t(0); check("A2 + tB down", true);
            vs.push_back({"cand A2 + tB down (3 nodes)", chainA2t, bytes});
        }
    }
    add_null(vs);
    run_ab(vs, "shared-expert chain b=" + std::to_string(b) + " (gate/up 2304x5120, down 5120x2304)", 1);
    dfree(wg); dfree(wu); dfree(wd); dfree(xq); dfree(xs); dfree(gate_o); dfree(up_o); dfree(mid); dfree(mid_c);
    dfree(mxq); dfree(mxq_c); dfree(mxs); dfree(mxs_c); dfree(out); dfree(out_c);
    return 0;
}

static kb::Module* try_module(const std::string& p) {
    hipModule_t m;
    if (hipModuleLoad(&m, p.c_str()) != hipSuccess) { fprintf(stderr, "[harness] no module %s\n", p.c_str()); return nullptr; }
    hipModuleUnload(m);
    return new kb::Module(p);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: see header\n"); return 64; }
    kb::init();
    Ctx c;
    c.dir = getenv("C1_DIR") ? getenv("C1_DIR") : ".";
    kb::Module base(c.dir + "/base_q8_gfx1201.hsaco"), grp(c.dir + "/base_grp_gfx1201.hsaco"), swi(c.dir + "/base_swiglu_gfx1201.hsaco");
    c.base = &base; c.grp = &grp; c.swi = &swi;
    c.cand = try_module(c.dir + "/cand_bpack_gfx1201.hsaco");
    c.shared = try_module(c.dir + "/cand_shared_gfx1201.hsaco");
    g_warm = getenv("C1_WARM") != nullptr;
    if (getenv("C1_ROUNDS")) g_rounds = atoi(getenv("C1_ROUNDS"));

    // split argv into ':'-separated commands
    std::vector<std::vector<std::string>> cmds(1);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == ":") { cmds.emplace_back(); continue; }
        cmds.back().push_back(argv[i]);
    }
    int rc = 0;
    for (auto& a : cmds) {
        if (a.empty()) continue;
        const std::string& mode = a[0];
        printf("\n#### %s", mode.c_str()); for (size_t i = 1; i < a.size(); ++i) printf(" %s", a[i].c_str()); printf("\n");
        fflush(stdout);
        if (mode == "gemv" && a.size() >= 3) {
            unsigned nc = a.size() > 3 ? atoi(a[3].c_str()) : 0;
            if (nc == 0) { Shape sh = shape_of(a[1]); size_t wb = (size_t)sh.M * (sh.K / 32) * 34; nc = (unsigned)std::max<size_t>(1, std::min<size_t>(8, (36u << 20) / wb)); }
            rc |= run_gemv(c, a[1], atoi(a[2].c_str()), nc);
        } else if (mode == "grouped" && a.size() >= 2) rc |= run_grouped(c, atoi(a[1].c_str()));
        else if (mode == "quant" && a.size() >= 3) rc |= run_quant(c, atoi(a[1].c_str()), atoi(a[2].c_str()),
                                                                 a.size() > 3 ? strtoull(a[3].c_str(), 0, 0) : 0,
                                                                 a.size() > 4 ? strtoull(a[4].c_str(), 0, 0) : 0,
                                                                 a.size() > 5 ? strtoull(a[5].c_str(), 0, 0) : 0);
        else if (mode == "swiglu" && a.size() >= 2) rc |= run_swiglu(c, atoi(a[1].c_str()));
        else if (mode == "lanes" && a.size() >= 4) rc |= run_lanes(c, a[1], atoi(a[2].c_str()), atoi(a[3].c_str()));
        else if (mode == "shared" && a.size() >= 2) rc |= run_shared(c, atoi(a[1].c_str()));
        else { fprintf(stderr, "bad command: %s\n", mode.c_str()); rc = 64; }
        fflush(stdout);
    }
    return rc;
}
