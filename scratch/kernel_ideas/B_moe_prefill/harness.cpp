// B_moe_prefill harness: production MXFP4 prefill MoE chain on the iGPU at prefill batch.
//
//   ./harness <mode> [B=1024] [N_EXP=96] [seed=1] [rounds=25]
//   modes: base     time every production kernel of the chain (+ roofline bytes)
//          cmp      correctness of the candidates vs the production kernels (+ CPU spot check)
//          ab       interleaved A/B: production kwide / kwide2 vs the WMMA candidates
//          grid     kwide/kwide2 with the production upper-bound grid vs the exact grid
//          prof_X   a few calls of one kernel for ATT/PMC (X = kwide, kwide2, gu, down)
//
// Regime: see NOTES.md. All buffers are HIP device 0 (gpu_run.sh masks the dGPU).
#include "kbench.h"

#include <cstdint>
#include <random>

static const unsigned N_EMBD = 5120, N_FF = 2304, N_USED = 6, N_VIRT = 384, CHUNK = 32;
static const unsigned NB_GATE = 20, NB_DOWN = 9, SB_BYTES = 136, Q8K_BYTES = 292;
static const float CLAMP = 10.0f;
static const unsigned GBPE = N_FF * NB_GATE * SB_BYTES;    // 6,266,880
static const unsigned DBPE = N_EMBD * NB_DOWN * SB_BYTES;  // 6,266,880

__global__ void patch_scales(uint8_t* w, size_t n_sb, uint32_t seed, unsigned lo, unsigned span) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n_sb * 8; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        w[(i / 8) * SB_BYTES + 128 + (i % 8)] = (uint8_t)(lo + x % span);
    }
}

// ---- CPU reference (port of mxfp4_tables::cpu_dot_mxfp4_q8_k, layout v2)
static const int8_t KV[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static float e8m0(uint8_t e) {
    uint32_t bits = e < 2 ? (0x00200000u << e) : (((uint32_t)e - 1u) << 23);
    float f; memcpy(&f, &bits, 4); return f;
}
static float cpu_dot(unsigned n_sb, const uint8_t* w, const uint8_t* y) {
    float sumf = 0.f;
    for (unsigned s = 0; s < n_sb; ++s) {
        const uint8_t* ws = w + s * SB_BYTES;
        const uint8_t* ys = y + s * Q8K_BYTES;
        float yd; memcpy(&yd, ys, 4);
        const int8_t* q8 = (const int8_t*)(ys + 4);
        float acc = 0.f;
        for (unsigned b = 0; b < 8; ++b) {
            float sc = e8m0(ws[128 + b]);
            int sumi = 0;
            for (unsigned j = 0; j < 16; ++j) {
                uint8_t q = ws[b * 16 + j];
                sumi += KV[q & 15] * q8[b * 32 + j];
                sumi += KV[q >> 4] * q8[b * 32 + 16 + j];
            }
            acc += sc * (float)sumi;
        }
        sumf += yd * acc;
    }
    return sumf;
}

struct Chain {
    unsigned B, n_exp;
    uint8_t *gate, *up, *down, *xq, *midq;
    float *x, *mid, *mid2, *partials, *partials2, *out, *ew;
    int *sel, *remap, *group_count, *members, *work_items, *n_wi_dev;
    unsigned n_wi_bound, n_wi_exact, touched, members_total, max_per_expert;
    std::vector<int> sel_h;
};

static Chain setup(unsigned B, unsigned n_exp, uint32_t seed, bool refs) {
    Chain c;
    c.B = B; c.n_exp = n_exp; c.max_per_expert = B;
    c.gate = kb::dalloc<uint8_t>((size_t)n_exp * GBPE);
    c.up = kb::dalloc<uint8_t>((size_t)n_exp * GBPE);
    c.down = kb::dalloc<uint8_t>((size_t)n_exp * DBPE);
    kb::fill_bytes(c.gate, (size_t)n_exp * GBPE, seed * 11u + 1u);
    kb::fill_bytes(c.up, (size_t)n_exp * GBPE, seed * 11u + 2u);
    kb::fill_bytes(c.down, (size_t)n_exp * DBPE, seed * 11u + 3u);
    hipLaunchKernelGGL(patch_scales, dim3(2048), dim3(256), 0, 0, c.gate, (size_t)n_exp * GBPE / SB_BYTES, seed + 7u, 116u, 13u);
    hipLaunchKernelGGL(patch_scales, dim3(2048), dim3(256), 0, 0, c.up, (size_t)n_exp * GBPE / SB_BYTES, seed + 8u, 116u, 13u);
    hipLaunchKernelGGL(patch_scales, dim3(2048), dim3(256), 0, 0, c.down, (size_t)n_exp * DBPE / SB_BYTES, seed + 9u, 116u, 13u);
    KB_CHECK(hipDeviceSynchronize());
    c.x = kb::dalloc<float>((size_t)B * N_EMBD);
    kb::fill_f32(c.x, (size_t)B * N_EMBD, seed + 21u, -1.f, 1.f);
    c.xq = kb::dalloc<uint8_t>((size_t)B * NB_GATE * Q8K_BYTES);
    c.mid = kb::dalloc<float>((size_t)B * N_USED * N_FF);
    KB_CHECK(hipMemset(c.mid, 0, (size_t)B * N_USED * N_FF * 4));
    c.midq = kb::dalloc<uint8_t>((size_t)B * N_USED * NB_DOWN * Q8K_BYTES);
    c.partials = kb::dalloc<float>((size_t)B * N_USED * N_EMBD);
    KB_CHECK(hipMemset(c.partials, 0, (size_t)B * N_USED * N_EMBD * 4));
    c.mid2 = c.partials2 = nullptr;
    if (refs) {
        c.mid2 = kb::dalloc<float>((size_t)B * N_USED * N_FF);
        KB_CHECK(hipMemset(c.mid2, 0, (size_t)B * N_USED * N_FF * 4));
        c.partials2 = kb::dalloc<float>((size_t)B * N_USED * N_EMBD);
        KB_CHECK(hipMemset(c.partials2, 0, (size_t)B * N_USED * N_EMBD * 4));
    }
    c.out = kb::dalloc<float>((size_t)B * N_EMBD);
    c.ew = kb::dalloc<float>((size_t)B * N_USED);
    kb::fill_f32(c.ew, (size_t)B * N_USED, seed + 31u, 0.05f, 0.5f);
    // Routing: 6 distinct virtual experts per row over 384; virtual >= n_exp -> sentinel -1.
    std::mt19937 rng(seed * 7919u + 13u);
    c.sel_h.assign((size_t)B * N_USED, -1);
    std::vector<int> gcount(N_VIRT, 0);
    for (unsigned b = 0; b < B; ++b) {
        int picks[N_USED];
        for (unsigned s = 0; s < N_USED; ++s) {
            int v;
            bool dup;
            do {
                v = (int)(rng() % N_VIRT);
                dup = false;
                for (unsigned t = 0; t < s; ++t) dup |= (picks[t] == v);
            } while (dup);
            picks[s] = v;
            if ((unsigned)v < n_exp) { c.sel_h[b * N_USED + s] = v; gcount[v]++; }
        }
    }
    c.members_total = 0; c.touched = 0; c.n_wi_exact = 0;
    int gmax = 0;
    for (unsigned e = 0; e < n_exp; ++e) {
        c.members_total += gcount[e];
        if (gcount[e] > 0) { c.touched++; c.n_wi_exact += (gcount[e] + CHUNK - 1) / CHUNK; }
        gmax = std::max(gmax, gcount[e]);
    }
    c.sel = kb::dalloc<int>((size_t)B * N_USED);
    KB_CHECK(hipMemcpy(c.sel, c.sel_h.data(), (size_t)B * N_USED * 4, hipMemcpyHostToDevice));
    std::vector<int> remap_h(N_VIRT);
    for (unsigned e = 0; e < N_VIRT; ++e) remap_h[e] = -(int)(e + 1);   // every pick is an iGPU miss (mode 0)
    c.remap = kb::dalloc<int>(N_VIRT);
    KB_CHECK(hipMemcpy(c.remap, remap_h.data(), N_VIRT * 4, hipMemcpyHostToDevice));
    c.group_count = kb::dalloc<int>(N_VIRT);
    c.members = kb::dalloc<int>((size_t)N_VIRT * B);
    size_t wi_cap = N_VIRT + (size_t)B * N_USED;
    c.work_items = kb::dalloc<int>(wi_cap);
    c.n_wi_dev = kb::dalloc<int>(1);
    // dispatch::moe_wi_upper_bound(members = 6B, groups = 384, chunk 32, cap)
    {
        size_t m = (size_t)B * N_USED, g = std::min<size_t>(m, N_VIRT);
        c.n_wi_bound = (unsigned)std::min<size_t>(std::min<size_t>(m, g + (m + CHUNK - 1) / CHUNK), wi_cap);
    }
    fprintf(stderr, "[setup] B=%u n_exp=%u members=%u touched=%u max_group=%d n_wi_exact=%u n_wi_bound=%u  mean members/expert=%.1f\n",
            B, n_exp, c.members_total, c.touched, gmax, c.n_wi_exact, c.n_wi_bound, (double)c.members_total / n_exp);
    return c;
}

struct Fns {
    hipFunction_t kwide, kwide2, q8k, gb, wib, red;
    hipFunction_t gu_wmma = nullptr, down_wmma = nullptr;
};

// ---- production launches (exact wrapper geometry)
static void l_q8k_pre(const Fns& f, Chain& c, hipStream_t s) {
    kb::launch(f.q8k, dim3(NB_GATE * c.B), dim3(256), 0, s, (unsigned char*)c.xq, (const float*)c.x, (unsigned)(NB_GATE * c.B));
}
static void l_builders(const Fns& f, Chain& c, hipStream_t s) {
    KB_CHECK(hipMemsetAsync(c.group_count, 0, N_VIRT * 4, s));
    kb::launch(f.gb, dim3((c.B * N_USED + 511) / 512), dim3(512), 0, s, c.group_count, c.members, (const int*)c.sel,
               (const int*)c.remap, 0u, 6u, c.B, N_USED, N_VIRT, c.max_per_expert);
    KB_CHECK(hipMemsetAsync(c.n_wi_dev, 0, 4, s));
    kb::launch(f.wib, dim3((N_VIRT + 255) / 256), dim3(256), 0, s, c.work_items, c.n_wi_dev, (const int*)c.group_count,
               N_VIRT, CHUNK, (unsigned)(N_VIRT + c.B * N_USED));
}
static void l_gb_only(const Fns& f, Chain& c, hipStream_t s) {
    KB_CHECK(hipMemsetAsync(c.group_count, 0, N_VIRT * 4, s));
    kb::launch(f.gb, dim3((c.B * N_USED + 511) / 512), dim3(512), 0, s, c.group_count, c.members, (const int*)c.sel,
               (const int*)c.remap, 0u, 6u, c.B, N_USED, N_VIRT, c.max_per_expert);
}
static void l_wib_only(const Fns& f, Chain& c, hipStream_t s) {
    KB_CHECK(hipMemsetAsync(c.n_wi_dev, 0, 4, s));
    kb::launch(f.wib, dim3((N_VIRT + 255) / 256), dim3(256), 0, s, c.work_items, c.n_wi_dev, (const int*)c.group_count,
               N_VIRT, CHUNK, (unsigned)(N_VIRT + c.B * N_USED));
}
static void l_kwide(hipFunction_t fn, Chain& c, float* mid, unsigned grid_y, const int* n_wi_dev, hipStream_t s) {
    kb::launch(fn, dim3(N_FF / 8, grid_y), dim3(256), 0, s, mid, (const uint8_t*)c.gate, (const uint8_t*)c.up,
               (const uint8_t*)c.xq, (const float*)c.ew, (const int*)c.group_count, (const int*)c.members,
               (const int*)c.work_items, GBPE, GBPE, N_USED, c.max_per_expert, CHUNK, CLAMP, N_FF, NB_GATE, n_wi_dev);
}
static void l_gu_wmma(hipFunction_t fn, Chain& c, float* mid, unsigned grid_y, const int* n_wi_dev, hipStream_t s) {
    kb::launch(fn, dim3(N_FF / 128, grid_y), dim3(256), 0, s, mid, (const uint8_t*)c.gate, (const uint8_t*)c.up,
               (const uint8_t*)c.xq, (const float*)c.ew, (const int*)c.group_count, (const int*)c.members,
               (const int*)c.work_items, GBPE, GBPE, N_USED, c.max_per_expert, CHUNK, CLAMP, N_FF, NB_GATE, n_wi_dev);
}
static void l_q8k_mid(const Fns& f, Chain& c, const float* mid, hipStream_t s) {
    kb::launch(f.q8k, dim3(NB_DOWN * N_USED * c.B), dim3(256), 0, s, (unsigned char*)c.midq, mid, (unsigned)(NB_DOWN * N_USED * c.B));
}
static void l_kwide2(hipFunction_t fn, Chain& c, float* partials, unsigned grid_y, const int* n_wi_dev, hipStream_t s) {
    kb::launch(fn, dim3(N_EMBD / 16, grid_y), dim3(256), 0, s, partials, (const uint8_t*)c.down, (const uint8_t*)c.midq,
               (const int*)c.group_count, (const int*)c.members, (const int*)c.work_items, DBPE,
               (unsigned)(NB_DOWN * Q8K_BYTES), N_USED, c.max_per_expert, CHUNK, N_EMBD, NB_DOWN, n_wi_dev);
}
static void l_down_wmma(hipFunction_t fn, Chain& c, float* partials, unsigned grid_y, const int* n_wi_dev, hipStream_t s) {
    kb::launch(fn, dim3(N_EMBD / 128, grid_y), dim3(256), 0, s, partials, (const uint8_t*)c.down, (const uint8_t*)c.midq,
               (const int*)c.group_count, (const int*)c.members, (const int*)c.work_items, DBPE,
               (unsigned)(NB_DOWN * Q8K_BYTES), N_USED, c.max_per_expert, CHUNK, N_EMBD, NB_DOWN, n_wi_dev);
}
static void l_reduce(const Fns& f, Chain& c, const float* partials, hipStream_t s) {
    size_t total = (size_t)c.B * N_EMBD;
    kb::launch(f.red, dim3((unsigned)((total + 255) / 256)), dim3(256), 0, s, c.out, partials, (const int*)c.sel,
               (const int*)c.remap, 0u, 6u, N_USED, N_EMBD);
}

static void roofline_print(const Chain& c) {
    double gu_bytes = (double)c.touched * 2.0 * GBPE + (double)c.members_total * (NB_GATE * Q8K_BYTES + N_FF * 4.0);
    double d_bytes = (double)c.touched * DBPE + (double)c.members_total * (NB_DOWN * Q8K_BYTES + N_EMBD * 4.0);
    double gu_flops = 2.0 * 2.0 * (double)c.members_total * N_FF * N_EMBD;
    double d_flops = 2.0 * (double)c.members_total * N_FF * N_EMBD;
    printf("ROOFLINE gate+up: %.3f GB (weights %.3f GB) -> %.2f ms at 230 GB/s; %.1f GFLOP\n", gu_bytes / 1e9,
           c.touched * 2.0 * GBPE / 1e9, gu_bytes / 230e6, gu_flops / 1e9);
    printf("ROOFLINE down:    %.3f GB (weights %.3f GB) -> %.2f ms at 230 GB/s; %.1f GFLOP\n", d_bytes / 1e9,
           c.touched * (double)DBPE / 1e9, d_bytes / 230e6, d_flops / 1e9);
    printf("SCALE: x%.2f to a full 384-expert layer (this launch touches %u experts)\n", 384.0 / c.touched, c.touched);
}

static void spot_check(const Fns& f, Chain& c, uint32_t seed) {
    // Baseline chain once, then check a few outputs against the CPU reference.
    hipStream_t s = 0;
    l_q8k_pre(f, c, s); l_builders(f, c, s);
    l_kwide(f.kwide, c, c.mid, c.n_wi_bound, c.n_wi_dev, s);
    l_q8k_mid(f, c, c.mid, s);
    l_kwide2(f.kwide2, c, c.partials, c.n_wi_bound, c.n_wi_dev, s);
    l_reduce(f, c, c.partials, s);
    KB_CHECK(hipDeviceSynchronize());
    int n_wi_h = 0;
    KB_CHECK(hipMemcpy(&n_wi_h, c.n_wi_dev, 4, hipMemcpyDeviceToHost));
    printf("device n_work_items=%d (host expected %u, bound %u)\n", n_wi_h, c.n_wi_exact, c.n_wi_bound);
    std::mt19937 rng(seed + 99u);
    std::vector<uint8_t> wrow(NB_GATE * SB_BYTES), urow(NB_GATE * SB_BYTES), xrow(NB_GATE * Q8K_BYTES);
    std::vector<uint8_t> drow(NB_DOWN * SB_BYTES), mrow(NB_DOWN * Q8K_BYTES);
    std::vector<float> ew_h = kb::d2h(c.ew, (size_t)c.B * N_USED);
    double worst_gu = 0, worst_d = 0;
    int checked = 0;
    for (int t = 0; t < 400 && checked < 48; ++t) {
        unsigned b = rng() % c.B, slot = rng() % N_USED;
        int e = c.sel_h[b * N_USED + slot];
        if (e < 0) continue;
        unsigned r = rng() % N_FF;
        KB_CHECK(hipMemcpy(wrow.data(), c.gate + (size_t)e * GBPE + (size_t)r * NB_GATE * SB_BYTES, wrow.size(), hipMemcpyDeviceToHost));
        KB_CHECK(hipMemcpy(urow.data(), c.up + (size_t)e * GBPE + (size_t)r * NB_GATE * SB_BYTES, urow.size(), hipMemcpyDeviceToHost));
        KB_CHECK(hipMemcpy(xrow.data(), c.xq + (size_t)b * NB_GATE * Q8K_BYTES, xrow.size(), hipMemcpyDeviceToHost));
        float g = cpu_dot(NB_GATE, wrow.data(), xrow.data()), u = cpu_dot(NB_GATE, urow.data(), xrow.data());
        if (g > CLAMP) g = CLAMP; if (u > CLAMP) u = CLAMP; if (u < -CLAMP) u = -CLAMP;
        float ref = g / (1.f + expf(-g)) * u * ew_h[b * N_USED + slot];
        float got;
        KB_CHECK(hipMemcpy(&got, c.mid + ((size_t)b * N_USED + slot) * N_FF + r, 4, hipMemcpyDeviceToHost));
        worst_gu = std::max(worst_gu, (double)fabsf(got - ref) / std::max(1e-3f, fabsf(ref)));
        // down: partials[(b*6+slot)*5120 + r2] = dot(down[e][r2], midq[b*6+slot])
        unsigned r2 = rng() % N_EMBD;
        KB_CHECK(hipMemcpy(drow.data(), c.down + (size_t)e * DBPE + (size_t)r2 * NB_DOWN * SB_BYTES, drow.size(), hipMemcpyDeviceToHost));
        KB_CHECK(hipMemcpy(mrow.data(), c.midq + ((size_t)b * N_USED + slot) * NB_DOWN * Q8K_BYTES, mrow.size(), hipMemcpyDeviceToHost));
        float dref = cpu_dot(NB_DOWN, drow.data(), mrow.data()), dgot;
        KB_CHECK(hipMemcpy(&dgot, c.partials + ((size_t)b * N_USED + slot) * N_EMBD + r2, 4, hipMemcpyDeviceToHost));
        worst_d = std::max(worst_d, (double)fabsf(dgot - dref) / std::max(1e-3f, fabsf(dref)));
        checked++;
    }
    printf("SPOTCHECK baseline vs CPU reference: %d samples, worst rel gate/up=%.3e down=%.3e\n", checked, worst_gu, worst_d);
    // reduce: out[b] == sum over member slots of partials
    std::vector<float> out_h = kb::d2h(c.out, (size_t)c.B * N_EMBD);
    std::vector<float> part_h = kb::d2h(c.partials, (size_t)c.B * N_USED * N_EMBD);
    double worst_r = 0;
    for (int t = 0; t < 2000; ++t) {
        unsigned b = rng() % c.B, r = rng() % N_EMBD;
        float acc = 0.f;
        for (unsigned sl = 0; sl < N_USED; ++sl) acc += part_h[((size_t)b * N_USED + sl) * N_EMBD + r];
        worst_r = std::max(worst_r, (double)fabsf(acc - out_h[(size_t)b * N_EMBD + r]));
    }
    printf("SPOTCHECK reduce: worst abs %.3e\n", worst_r);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "base";
    const unsigned B = argc > 2 ? atoi(argv[2]) : 1024;
    const unsigned n_exp = argc > 3 ? atoi(argv[3]) : 96;
    const uint32_t seed = argc > 4 ? atoi(argv[4]) : 1;
    const int rounds = argc > 5 ? atoi(argv[5]) : 25;
    const std::string dir = ".";
    kb::init();
    kb::Module m_pair(dir + "/base_mxfp4_pair_matvec_gfx1151.hsaco"), m_mv(dir + "/base_mxfp4_matvec_gfx1151.hsaco"),
        m_q8k(dir + "/base_q8_k_quantize_gfx1151.hsaco"), m_q2k(dir + "/base_q2_k_accumulate_matvec_par_gfx1151.hsaco"),
        m_gb(dir + "/base_moe_group_builder_gfx1151.hsaco"), m_wib(dir + "/base_moe_work_items_builder_gfx1151.hsaco");
    Fns f;
    f.kwide = m_pair.fn("mxfp4_pair_matvec_fused_swiglu_kwide");
    f.kwide2 = m_mv.fn("mxfp4_matvec_par_by_expert_kwide2");
    f.q8k = m_q8k.fn("q8_k_quantize");
    f.gb = m_gb.fn("moe_group_builder_hetsplit");
    f.wib = m_wib.fn("moe_work_items_builder");
    f.red = m_q2k.fn("q2_k_reduce_partials_hetsplit");
    kb::Module* m_cand = nullptr;
    {
        FILE* t = fopen((dir + "/cand_wmma_gfx1151.hsaco").c_str(), "rb");
        if (t) {
            fclose(t);
            m_cand = new kb::Module(dir + "/cand_wmma_gfx1151.hsaco");
            f.gu_wmma = m_cand->fn("mxfp4_pair_matvec_fused_swiglu_wmma");
            f.down_wmma = m_cand->fn("mxfp4_matvec_par_by_expert_wmma");
        }
    }
    const bool need_refs = (mode == "cmp");
    Chain c = setup(B, n_exp, seed, need_refs);
    roofline_print(c);
    // Always run the production chain once so every buffer holds production-shaped data.
    l_q8k_pre(f, c, 0); l_builders(f, c, 0);
    l_kwide(f.kwide, c, c.mid, c.n_wi_bound, c.n_wi_dev, 0);
    l_q8k_mid(f, c, c.mid, 0);
    l_kwide2(f.kwide2, c, c.partials, c.n_wi_bound, c.n_wi_dev, 0);
    l_reduce(f, c, c.partials, 0);
    KB_CHECK(hipDeviceSynchronize());

    const double gu_bytes = (double)c.touched * 2.0 * GBPE + (double)c.members_total * (NB_GATE * Q8K_BYTES + N_FF * 4.0);
    const double d_bytes = (double)c.touched * DBPE + (double)c.members_total * (NB_DOWN * Q8K_BYTES + N_EMBD * 4.0);
    const double gu_flops = 4.0 * (double)c.members_total * N_FF * N_EMBD;
    const double d_flops = 2.0 * (double)c.members_total * N_FF * N_EMBD;
    char tag[256];

    if (mode == "base") {
        spot_check(f, c, seed);
        std::vector<kb::Variant> vs = {
            {"kwide gate+up (prod grid bound)", [&](hipStream_t s) { l_kwide(f.kwide, c, c.mid, c.n_wi_bound, c.n_wi_dev, s); }, gu_bytes, gu_flops},
            {"kwide2 down (prod grid bound)", [&](hipStream_t s) { l_kwide2(f.kwide2, c, c.partials, c.n_wi_bound, c.n_wi_dev, s); }, d_bytes, d_flops},
            {"q8_k_quantize pre (B x 20)", [&](hipStream_t s) { l_q8k_pre(f, c, s); }, (double)B * N_EMBD * 4.0 + (double)B * NB_GATE * Q8K_BYTES, 0},
            {"q8_k_quantize mid (B x 54)", [&](hipStream_t s) { l_q8k_mid(f, c, c.mid, s); }, (double)B * N_USED * N_FF * 4.0 + (double)B * N_USED * NB_DOWN * Q8K_BYTES, 0},
            {"group_builder_hetsplit (+memset)", [&](hipStream_t s) { l_gb_only(f, c, s); }, (double)B * N_USED * 4.0 * 3.0, 0},
            {"work_items_builder (+memset)", [&](hipStream_t s) { l_wib_only(f, c, s); }, N_VIRT * 12.0, 0},
            {"reduce_partials_hetsplit", [&](hipStream_t s) { l_reduce(f, c, c.partials, s); }, (double)B * N_USED * N_EMBD * 4.0 + (double)B * N_EMBD * 4.0, 0},
        };
        kb::AbOpts o; o.rounds = rounds; o.inner = 2;
        snprintf(tag, sizeof tag, "B=%u n_exp=%u members=%u touched=%u cold-weights direct-launch", B, n_exp, c.members_total, c.touched);
        o.tag = tag;
        kb::ab(vs, o);
    } else if (mode == "grid") {
        std::vector<kb::Variant> vs = {
            {"kwide gate+up grid bound", [&](hipStream_t s) { l_kwide(f.kwide, c, c.mid, c.n_wi_bound, c.n_wi_dev, s); }, gu_bytes, gu_flops},
            {"kwide gate+up grid exact", [&](hipStream_t s) { l_kwide(f.kwide, c, c.mid, c.n_wi_exact, (const int*)nullptr, s); }, gu_bytes, gu_flops},
            {"kwide2 down grid bound", [&](hipStream_t s) { l_kwide2(f.kwide2, c, c.partials, c.n_wi_bound, c.n_wi_dev, s); }, d_bytes, d_flops},
            {"kwide2 down grid exact", [&](hipStream_t s) { l_kwide2(f.kwide2, c, c.partials, c.n_wi_exact, (const int*)nullptr, s); }, d_bytes, d_flops},
        };
        kb::AbOpts o; o.rounds = rounds; o.inner = 2;
        snprintf(tag, sizeof tag, "grid bound vs exact B=%u n_exp=%u", B, n_exp);
        o.tag = tag;
        kb::ab(vs, o);
    } else if (mode == "cmp") {
        if (!f.gu_wmma) { fprintf(stderr, "no candidate module\n"); return 3; }
        spot_check(f, c, seed);
        // gate/up: candidate writes mid2 from the same xq/groups
        l_gu_wmma(f.gu_wmma, c, c.mid2, c.n_wi_bound, c.n_wi_dev, 0);
        KB_CHECK(hipDeviceSynchronize());
        {
            auto a = kb::d2h(c.mid, (size_t)B * N_USED * N_FF), b = kb::d2h(c.mid2, (size_t)B * N_USED * N_FF);
            snprintf(tag, sizeof tag, "gu_wmma vs kwide B=%u", B);
            kb::print_cmp(tag, kb::compare_f32(a.data(), b.data(), a.size()));
        }
        // down: candidate reads the SAME midq (quantized from the baseline mid) and writes partials2
        l_down_wmma(f.down_wmma, c, c.partials2, c.n_wi_bound, c.n_wi_dev, 0);
        KB_CHECK(hipDeviceSynchronize());
        {
            auto a = kb::d2h(c.partials, (size_t)B * N_USED * N_EMBD), b = kb::d2h(c.partials2, (size_t)B * N_USED * N_EMBD);
            snprintf(tag, sizeof tag, "down_wmma vs kwide2 B=%u", B);
            kb::print_cmp(tag, kb::compare_f32(a.data(), b.data(), a.size()));
        }
        // exact-grid launch must give the same result as the bound grid
        l_gu_wmma(f.gu_wmma, c, c.mid2, c.n_wi_exact, (const int*)nullptr, 0);
        l_down_wmma(f.down_wmma, c, c.partials2, c.n_wi_exact, (const int*)nullptr, 0);
        KB_CHECK(hipDeviceSynchronize());
        {
            auto a = kb::d2h(c.mid, (size_t)B * N_USED * N_FF), b = kb::d2h(c.mid2, (size_t)B * N_USED * N_FF);
            snprintf(tag, sizeof tag, "gu_wmma(exact grid) vs kwide B=%u", B);
            kb::print_cmp(tag, kb::compare_f32(a.data(), b.data(), a.size()));
        }
    } else if (mode == "ab") {
        if (!f.gu_wmma) { fprintf(stderr, "no candidate module\n"); return 3; }
        std::vector<kb::Variant> vs = {
            {"kwide gate+up (prod)", [&](hipStream_t s) { l_kwide(f.kwide, c, c.mid, c.n_wi_bound, c.n_wi_dev, s); }, gu_bytes, gu_flops},
            {"gu_wmma gate+up (cand)", [&](hipStream_t s) { l_gu_wmma(f.gu_wmma, c, c.mid, c.n_wi_bound, c.n_wi_dev, s); }, gu_bytes, gu_flops},
        };
        kb::AbOpts o; o.rounds = rounds; o.inner = 2;
        snprintf(tag, sizeof tag, "gate+up A/B B=%u n_exp=%u members=%u touched=%u", B, n_exp, c.members_total, c.touched);
        o.tag = tag;
        kb::ab(vs, o);
        std::vector<kb::Variant> vd = {
            {"kwide2 down (prod)", [&](hipStream_t s) { l_kwide2(f.kwide2, c, c.partials, c.n_wi_bound, c.n_wi_dev, s); }, d_bytes, d_flops},
            {"down_wmma (cand)", [&](hipStream_t s) { l_down_wmma(f.down_wmma, c, c.partials, c.n_wi_bound, c.n_wi_dev, s); }, d_bytes, d_flops},
        };
        snprintf(tag, sizeof tag, "down A/B B=%u n_exp=%u members=%u touched=%u", B, n_exp, c.members_total, c.touched);
        o.tag = tag;
        kb::ab(vd, o);
    } else if (mode.rfind("prof_", 0) == 0) {
        std::string k = mode.substr(5);
        kb::warm(200);
        for (int i = 0; i < 6; ++i) {
            if (k == "kwide") l_kwide(f.kwide, c, c.mid, c.n_wi_bound, c.n_wi_dev, 0);
            else if (k == "kwide2") l_kwide2(f.kwide2, c, c.partials, c.n_wi_bound, c.n_wi_dev, 0);
            else if (k == "gu") l_gu_wmma(f.gu_wmma, c, c.mid, c.n_wi_bound, c.n_wi_dev, 0);
            else if (k == "down") l_down_wmma(f.down_wmma, c, c.partials, c.n_wi_bound, c.n_wi_dev, 0);
            else if (k == "q8k") l_q8k_mid(f, c, c.mid, 0);
            KB_CHECK(hipDeviceSynchronize());
        }
        printf("prof %s done (6 launches)\n", k.c_str());
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 64;
    }
    return 0;
}
