// C1_dense_decode harness: production q8_0 bpack GEMV (baseline .hsaco from the unmodified in-tree
// source) vs candidates, at the decode dense shapes, COLD weights (80 MB memset flush before every
// timed block), graph mode, one call per timed block.
//
//   ./harness gemv <shape> <b> [ncopies]     shape in {qa,qb,kv,wob,gate,down,engram,head}
//   ./harness grouped <b>                    wo_a 8 x (1024 x 4096)
//   ./harness quant <K> <b>                  q8_0_quantize_f32 at (K/32*b) blocks
//   ./harness swiglu <b>
//   ./harness lanes <shape> <b0> <b1>        two lanes: 2 launches (b0, b1) vs one launch (b0+b1)
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
    uint4* buf; size_t n; unsigned* sink;
    explicit ReadFlusher(size_t bytes) : n(bytes / 16) { KB_CHECK(hipMalloc(&buf, bytes)); KB_CHECK(hipMalloc(&sink, 64)); KB_CHECK(hipMemset(buf, 1, bytes)); KB_CHECK(hipDeviceSynchronize()); }
    void operator()(hipStream_t s) { hipLaunchKernelGGL(c1_read_flush, dim3(2048), dim3(256), 0, s, (const uint4*)buf, n, sink); }
};

struct Ctx {
    std::string dir;
    kb::Module *base = nullptr, *grp = nullptr, *swi = nullptr, *cand = nullptr;
};

static hipFunction_t try_fn(kb::Module* m, const char* name) {
    if (!m) return nullptr;
    hipFunction_t f;
    if (hipModuleGetFunction(&f, m->m, name) != hipSuccess) return nullptr;
    return f;
}

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
    const dim3 grid(M / 8), block(256);
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
    std::vector<std::string> cnames = {"q8_0_gemv_bpack_tB" + std::to_string(b), "q8_0_gemv_bpack_tB" + std::to_string(b) + "_ku2",
                                       "q8_0_gemv_bpack_tB" + std::to_string(b) + "_ku4"};
    std::vector<hipFunction_t> cf;
    for (auto& n : cnames) {
        hipFunction_t f = try_fn(c.cand, n.c_str());
        if (!f) continue;
        // correctness on identical inputs
        KB_CHECK(hipMemset(out_c, 0, (size_t)M * b * 4));
        kb::launch(f, grid, block, 0, 0, out_c, (const uint8_t*)w[0], (const int8_t*)xq, (const float*)xs, K, M, blocks, b);
        KB_CHECK(hipDeviceSynchronize());
        auto oc = kb::d2h(out_c, (size_t)M * b);
        kb::print_cmp((sname + " " + n + " vs base b=" + std::to_string(b)).c_str(), kb::compare_f32(ob.data(), oc.data(), (size_t)M * b));
        cf.push_back(f);
        vs.push_back({"cand " + n, [&, f](hipStream_t s) {
            kb::launch(f, grid, block, 0, s, out_c, (const uint8_t*)w[(copy_idx++) % ncopies], (const int8_t*)xq, (const float*)xs, K, M, blocks, b); }, (double)wbytes});
    }

    const bool warm = getenv("C1_WARM") != nullptr;
    ReadFlusher flush(warm ? (1u << 20) : FLUSH_BYTES);
    unsigned* nullp = kb::dalloc<unsigned>(16);
    vs.push_back({"null 1-WG kernel (launch floor)", [&](hipStream_t s) { hipLaunchKernelGGL(c1_null_kernel, dim3(1), dim3(64), 0, s, nullp); }, 0.0});
    kb::AbOpts o;
    o.graph = true;
    o.inner = warm ? 10 : (int)ncopies;
    o.rounds = getenv("C1_ROUNDS") ? atoi(getenv("C1_ROUNDS")) : 60;
    if (!warm) o.between = [&](hipStream_t s) { flush(s); };
    std::string tag = std::string("gemv ") + sh.name + " b=" + std::to_string(b) + (warm ? " WARM(inner10,no flush)" : " cold(readflush72MB)") + " copies=" + std::to_string(ncopies) + " graph";
    o.tag = tag.c_str();
    kb::ab(vs, o);
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
    hipFunction_t f_m2 = try_fn(c.cand, ("q8_0_gemv_bpack_tB" + std::to_string(bt) + "_ku2").c_str());
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
    if (f_m2) vs.push_back({"cand merged tB" + std::to_string(bt) + "_ku2", [&](hipStream_t s) {
        kb::launch(f_m2, grid, block, 0, s, out1, (const uint8_t*)w, (const int8_t*)xq, (const float*)xs, K, M, blocks, bt); }, (double)wbytes});
    ReadFlusher flush(FLUSH_BYTES);
    kb::AbOpts o;
    o.graph = true; o.inner = 1; o.rounds = 60;
    o.between = [&](hipStream_t s) { flush(s); };
    std::string tag = std::string("lanes ") + sh.name + " b0=" + std::to_string(b0) + " b1=" + std::to_string(b1) + " cold graph";
    o.tag = tag.c_str();
    kb::ab(vs, o);
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
    const dim3 grid(M / 8), block(256);
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
    ReadFlusher flush(FLUSH_BYTES);
    unsigned* nullp = kb::dalloc<unsigned>(16);
    vs.push_back({"null 1-WG kernel (launch floor)", [&](hipStream_t s) { hipLaunchKernelGGL(c1_null_kernel, dim3(1), dim3(64), 0, s, nullp); }, 0.0});
    kb::AbOpts o;
    o.graph = true; o.inner = 1; o.rounds = 60;
    o.between = [&](hipStream_t s) { flush(s); };
    std::string tag = "grouped wo_a 8x(1024x4096) b=" + std::to_string(b) + " cold graph";
    o.tag = tag.c_str();
    kb::ab(vs, o);
    return 0;
}

static int run_quant(Ctx& c, unsigned K, unsigned b) {
    const unsigned blocks = K / 32 * b;
    float* x = kb::dalloc<float>((size_t)K * b);
    kb::fill_f32(x, (size_t)K * b, 7, -3.f, 3.f);
    int8_t* xq = kb::dalloc<int8_t>((size_t)K * b);
    float* xs = kb::dalloc<float>(blocks);
    int8_t* xq2 = kb::dalloc<int8_t>((size_t)K * b);
    float* xs2 = kb::dalloc<float>(blocks);
    hipFunction_t f = c.base->fn("q8_0_quantize_f32");
    auto lb = [&](hipStream_t s) { kb::launch(f, dim3(blocks), dim3(32), 0, s, xq, xs, (const float*)x, blocks); };
    lb(0); KB_CHECK(hipDeviceSynchronize());
    auto q_ref = kb::d2h(xq, (size_t)K * b); auto s_ref = kb::d2h(xs, blocks);
    std::vector<kb::Variant> vs;
    vs.push_back({"base q8_0_quantize_f32 (blocks x 32thr)", lb, (double)K * b * 5});
    hipFunction_t fc = try_fn(c.cand, "q8_0_quantize_f32_wave");
    if (fc) {
        // one wave (32 lanes) per block, 8 blocks per WG
        auto lc = [&](hipStream_t s) { kb::launch(fc, dim3((blocks + 7) / 8), dim3(256), 0, s, xq2, xs2, (const float*)x, blocks); };
        lc(0); KB_CHECK(hipDeviceSynchronize());
        auto q2 = kb::d2h(xq2, (size_t)K * b); auto s2 = kb::d2h(xs2, blocks);
        size_t nd = 0; for (size_t i = 0; i < q2.size(); ++i) nd += q2[i] != q_ref[i];
        printf("CMP quant_wave xq K=%u b=%u int8_diff=%zu\n", K, b, nd);
        kb::print_cmp("quant_wave xscale vs base", kb::compare_f32(s_ref.data(), s2.data(), blocks));
        vs.push_back({"cand q8_0_quantize_f32_wave", lc, (double)K * b * 5});
    }
    kb::AbOpts o;
    o.graph = true; o.inner = 10; o.rounds = 40;
    std::string tag = "quantize K=" + std::to_string(K) + " b=" + std::to_string(b) + " warm graph(10 nodes)";
    o.tag = tag.c_str();
    kb::ab(vs, o);
    return 0;
}

static int run_swiglu(Ctx& c, unsigned b) {
    const unsigned n = 2304 * b;
    float* g = kb::dalloc<float>(n); float* u = kb::dalloc<float>(n); float* o_ = kb::dalloc<float>(n);
    kb::fill_f32(g, n, 3, -12.f, 12.f); kb::fill_f32(u, n, 4, -12.f, 12.f);
    hipFunction_t f = c.swi->fn("swiglu");
    std::vector<kb::Variant> vs;
    vs.push_back({"base swiglu", [&](hipStream_t s) { kb::launch(f, dim3((n + 255) / 256), dim3(256), 0, s, o_, (const float*)g, (const float*)u, n, 10.0f); }, 12.0 * n});
    kb::AbOpts o;
    o.graph = true; o.inner = 10; o.rounds = 40;
    std::string tag = "swiglu n=2304*" + std::to_string(b) + " warm graph(10 nodes)";
    o.tag = tag.c_str();
    kb::ab(vs, o);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: see header\n"); return 64; }
    kb::init();
    Ctx c;
    c.dir = getenv("C1_DIR") ? getenv("C1_DIR") : ".";
    kb::Module base(c.dir + "/base_q8_gfx1201.hsaco"), grp(c.dir + "/base_grp_gfx1201.hsaco"), swi(c.dir + "/base_swiglu_gfx1201.hsaco");
    c.base = &base; c.grp = &grp; c.swi = &swi;
    kb::Module* cand = nullptr;
    {
        hipModule_t m;
        std::string p = c.dir + "/cand_bpack_gfx1201.hsaco";
        if (hipModuleLoad(&m, p.c_str()) == hipSuccess) { cand = new kb::Module(p); c.cand = cand; hipModuleUnload(m); }
        else fprintf(stderr, "[harness] no candidate module %s\n", p.c_str());
    }
    std::string mode = argv[1];
    if (mode == "gemv" && argc >= 4) {
        unsigned nc = argc > 4 ? atoi(argv[4]) : 0;
        if (nc == 0) { Shape sh = shape_of(argv[2]); size_t wb = (size_t)sh.M * (sh.K / 32) * 34; nc = (unsigned)std::max<size_t>(1, std::min<size_t>(8, (36u << 20) / wb)); }
        return run_gemv(c, argv[2], atoi(argv[3]), nc);
    }
    if (mode == "grouped" && argc >= 3) return run_grouped(c, atoi(argv[2]));
    if (mode == "quant" && argc >= 4) return run_quant(c, atoi(argv[2]), atoi(argv[3]));
    if (mode == "swiglu" && argc >= 3) return run_swiglu(c, atoi(argv[2]));
    if (mode == "lanes" && argc >= 5) return run_lanes(c, argv[2], atoi(argv[3]), atoi(argv[4]));
    fprintf(stderr, "bad args\n");
    return 64;
}
