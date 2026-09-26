// A_moe_smallb harness: the production box-2 batched MoE chain (kwide gate/up -> q8k -> kwide2 down ->
// reduce, with the two builders) at b = 1..8 rows, plus the b=1 hetsplit twins, on synthetic data at the
// exact production shapes. Baseline code objects are the UNMODIFIED in-tree .hip files (build.sh).
//
//   ./harness <arch> <dir> <mode> [key=val ...]
//   modes: chain      timed A/B of the production chain vs candidates at one (b, E, ppr)
//          twin       b=1 hetsplit chain vs the batched chain at b=1 (equal bytes)
//          parts      per-kernel breakdown of the production chain
//          prof       a few direct launches of the chain for rocprofv3 (ATT / pmc)
//   keys: b=4 E=4 ppr=3 P=16 gbound=384 rounds=60 inner=8 cand=<list>  (see parse below)
#include "kbench.h"

#include <cstdarg>
#include <map>

namespace {
constexpr unsigned N_EMBD = 5120, N_FF = 2304, NU = 6, N_EXPERT = 384, SENTINEL = 384;
constexpr unsigned NB_GATE = 20, NB_DOWN = 9, SUPER = 136, Q8K = 292, CHUNK = 32;
constexpr float CLAMP = 10.0f;
constexpr size_t GBPE = (size_t)N_FF * NB_GATE * SUPER;    // 6,266,880
constexpr size_t DBPE = (size_t)N_EMBD * NB_DOWN * SUPER;  // 6,266,880
constexpr size_t XQ_TOK = (size_t)NB_GATE * Q8K;           // 5840
constexpr size_t MIDQ_SLOT = (size_t)NB_DOWN * Q8K;        // 2628

std::map<std::string, std::string> kv;
unsigned geti(const char* k, unsigned d) { auto it = kv.find(k); return it == kv.end() ? d : (unsigned)atoi(it->second.c_str()); }
std::string gets(const char* k, const char* d) { auto it = kv.find(k); return it == kv.end() ? d : it->second; }

// E8M0 scale bytes -> 118..125 (2^-10 .. 2^-3) so the synthetic dot products land near the clamp scale.
__global__ void fix_scales(uint8_t* w, size_t n_super) {
    size_t s = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; s < n_super; s += (size_t)gridDim.x * blockDim.x) {
        uint8_t* sup = w + s * SUPER;
        for (unsigned k = 0; k < 8; ++k) sup[128 + k] = (uint8_t)(118 + ((sup[k] * 7u + k * 13u + (unsigned)s) & 7u));
    }
}

struct Q8KHost {  // one Q8_K row of nb blocks
    std::vector<uint8_t> bytes;
    Q8KHost(unsigned nb, uint32_t seed, float lo, float hi) : bytes(nb * Q8K) {
        uint32_t x = seed * 2654435761u + 12345u;
        auto rnd = [&]() { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; };
        for (unsigned b = 0; b < nb; ++b) {
            uint8_t* blk = bytes.data() + b * Q8K;
            float d = lo + (hi - lo) * ((rnd() >> 8) * (1.0f / 16777216.0f));
            memcpy(blk, &d, 4);
            int32_t bs[16] = {0};
            for (unsigned i = 0; i < 256; ++i) {
                int8_t q = (int8_t)(rnd() & 0xff);
                blk[4 + i] = (uint8_t)q;
                bs[i >> 4] += q;
            }
            for (unsigned g = 0; g < 16; ++g) { int16_t v = (int16_t)bs[g]; memcpy(blk + 260 + 2 * g, &v, 2); }
        }
    }
};

struct Sel {  // one call's selection: d_selected[b*6] with sentinels, pool slots as expert ids
    std::vector<int> sel;
    std::vector<int> slots;  // distinct pool slots used
};

// Each row picks `ppr` distinct slots from the call's E slots (slots rotate with the call index).
Sel make_sel(unsigned k, unsigned b, unsigned E, unsigned ppr, unsigned P, uint32_t seed) {
    Sel s;
    s.sel.assign(b * NU, (int)SENTINEL);
    for (unsigned i = 0; i < E; ++i) s.slots.push_back((int)((k * E + i) % P));
    uint32_t x = seed ^ (k * 0x9E3779B9u);
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; };
    for (unsigned r = 0; r < b; ++r) {
        std::vector<int> pool = s.slots;
        for (unsigned p = 0; p < ppr && !pool.empty(); ++p) {
            unsigned j = rnd() % pool.size();
            // random slot position among the 6 picks (still distinct per row)
            unsigned pos;
            do { pos = rnd() % NU; } while (s.sel[r * NU + pos] != (int)SENTINEL);
            s.sel[r * NU + pos] = pool[j];
            pool.erase(pool.begin() + j);
        }
    }
    return s;
}

// Host mirror of the two builders (used for the "skip builders" idea and the per-kernel breakdown).
struct Groups {
    std::vector<int> group_count, expert_members, work_items;
    int n_wi = 0;
};
Groups build_groups(const Sel& s, unsigned b, unsigned gbound, unsigned mpe, unsigned chunk) {
    Groups g;
    g.group_count.assign(gbound, 0);
    g.expert_members.assign((size_t)gbound * mpe, 0);
    for (unsigned i = 0; i < b * NU; ++i) {
        int e = s.sel[i];
        if (e == (int)SENTINEL) continue;
        unsigned pos = g.group_count[e]++;
        g.expert_members[(size_t)e * mpe + pos] = (int)(((i / NU) << 16) | (i % NU));
    }
    for (unsigned e = 0; e < gbound; ++e) {
        int n = g.group_count[e];
        for (int c = 0; c * (int)chunk < n; ++c) g.work_items.push_back((int)((e << 16) | (unsigned)(c * chunk)));
    }
    g.n_wi = (int)g.work_items.size();
    return g;
}

unsigned wi_bound(unsigned members, unsigned groups, unsigned chunk) {
    unsigned g = std::min(members, groups);
    return std::min(members, g + (members + chunk - 1) / chunk);
}
}  // namespace

int main(int argc, char** argv) {
    const std::string arch = argc > 1 ? argv[1] : "gfx1151";
    const std::string dir = argc > 2 ? argv[2] : ".";
    const std::string mode = argc > 3 ? argv[3] : "chain";
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        auto p = a.find('=');
        if (p != std::string::npos) kv[a.substr(0, p)] = a.substr(p + 1);
    }
    const unsigned b = geti("b", 4), E = geti("E", 4), ppr = geti("ppr", 3), P = geti("P", 16);
    const unsigned gbound = geti("gbound", 384), mpe = geti("mpe", 64);
    const unsigned rounds = geti("rounds", 60), inner = geti("inner", 8);
    const bool graph = geti("graph", 1) != 0;
    const bool exact_wi = geti("exact", 0) != 0;   // grid.y = exact n_wi instead of the production upper bound
    const std::string cands = gets("cand", "");
    kb::init();

    // ---- modules: production baseline (unmodified in-tree sources) + candidates
    kb::Module m_pair(dir + "/base_pair_" + arch + ".hsaco"), m_down(dir + "/base_down_" + arch + ".hsaco"),
        m_q2k(dir + "/base_q2k_" + arch + ".hsaco"), m_q8k(dir + "/base_q8k_" + arch + ".hsaco"),
        m_grp(dir + "/base_grp_" + arch + ".hsaco"), m_wi(dir + "/base_wi_" + arch + ".hsaco");
    hipFunction_t f_kwide = m_pair.fn("mxfp4_pair_matvec_fused_swiglu_kwide");
    hipFunction_t f_pair_het = m_pair.fn("mxfp4_pair_matvec_fused_swiglu_batch_hetsplit");
    hipFunction_t f_kwide2 = m_down.fn("mxfp4_matvec_par_by_expert_kwide2");
    hipFunction_t f_down_het = m_down.fn("mxfp4_matvec_par_batched_hetsplit");
    hipFunction_t f_reduce = m_q2k.fn("q2_k_reduce_partials_hetsplit");
    hipFunction_t f_q8k = m_q8k.fn("q8_k_quantize");
    hipFunction_t f_grp = m_grp.fn("moe_group_builder_hetsplit");
    hipFunction_t f_wi = m_wi.fn("moe_work_items_builder");

    // ---- buffers
    const size_t pool_b = (size_t)P * GBPE;
    uint8_t* gate = kb::dalloc<uint8_t>(pool_b);
    uint8_t* up = kb::dalloc<uint8_t>(pool_b);
    uint8_t* down = kb::dalloc<uint8_t>((size_t)P * DBPE);
    kb::fill_bytes(gate, pool_b, 11);
    kb::fill_bytes(up, pool_b, 22);
    kb::fill_bytes(down, (size_t)P * DBPE, 33);
    hipLaunchKernelGGL(fix_scales, dim3(2048), dim3(256), 0, 0, gate, pool_b / SUPER);
    hipLaunchKernelGGL(fix_scales, dim3(2048), dim3(256), 0, 0, up, pool_b / SUPER);
    hipLaunchKernelGGL(fix_scales, dim3(2048), dim3(256), 0, 0, down, (size_t)P * DBPE / SUPER);
    KB_CHECK(hipDeviceSynchronize());
    fprintf(stderr, "[h] pool: %u experts x 18.8 MB = %.0f MB on device\n", P, 3.0 * pool_b / 1e6);

    // activations: b Q8_K rows
    uint8_t* xq = kb::dalloc<uint8_t>(b * XQ_TOK);
    {
        std::vector<uint8_t> h;
        for (unsigned r = 0; r < b; ++r) { Q8KHost q(NB_GATE, 100 + r, 0.002f, 0.02f); h.insert(h.end(), q.bytes.begin(), q.bytes.end()); }
        KB_CHECK(hipMemcpy(xq, h.data(), h.size(), hipMemcpyHostToDevice));
    }
    float* ew = kb::dalloc<float>(b * NU);
    kb::fill_f32(ew, b * NU, 7, 0.05f, 0.4f);
    int* remap = kb::dalloc<int>(N_EXPERT + 1);
    {
        std::vector<int> h(N_EXPERT + 1, 0);
        for (unsigned i = 0; i < P; ++i) h[i] = -(int)(i + 1);
        h[SENTINEL] = 0;
        KB_CHECK(hipMemcpy(remap, h.data(), h.size() * 4, hipMemcpyHostToDevice));
    }
    // per-call selections + host-built group arrays
    const unsigned NSEL = inner;
    std::vector<Sel> sels;
    std::vector<int*> d_sel(NSEL), d_gc(NSEL), d_em(NSEL), d_wi(NSEL), d_nwi(NSEL);
    std::vector<int> n_wi_exact(NSEL);
    std::vector<int*> d_gc8(NSEL), d_em8(NSEL), d_wi8(NSEL), d_nwi8(NSEL);  // chunk-8 builds for cands
    std::vector<int> n_wi8(NSEL);
    for (unsigned k = 0; k < NSEL; ++k) {
        sels.push_back(make_sel(k, b, E, ppr, P, 0xC0FFEE));
        d_sel[k] = kb::dalloc<int>(b * NU);
        KB_CHECK(hipMemcpy(d_sel[k], sels[k].sel.data(), b * NU * 4, hipMemcpyHostToDevice));
        Groups g = build_groups(sels[k], b, gbound, mpe, CHUNK);
        n_wi_exact[k] = g.n_wi;
        d_gc[k] = kb::dalloc<int>(gbound);
        d_em[k] = kb::dalloc<int>((size_t)gbound * mpe);
        d_wi[k] = kb::dalloc<int>(gbound + b * NU);
        d_nwi[k] = kb::dalloc<int>(1);
        KB_CHECK(hipMemcpy(d_gc[k], g.group_count.data(), gbound * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_em[k], g.expert_members.data(), (size_t)gbound * mpe * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_wi[k], g.work_items.data(), g.work_items.size() * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_nwi[k], &g.n_wi, 4, hipMemcpyHostToDevice));
        Groups g8 = build_groups(sels[k], b, gbound, mpe, 8);
        n_wi8[k] = g8.n_wi;
        d_gc8[k] = kb::dalloc<int>(gbound);
        d_em8[k] = kb::dalloc<int>((size_t)gbound * mpe);
        d_wi8[k] = kb::dalloc<int>(gbound + b * NU);
        d_nwi8[k] = kb::dalloc<int>(1);
        KB_CHECK(hipMemcpy(d_gc8[k], g8.group_count.data(), gbound * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_em8[k], g8.expert_members.data(), (size_t)gbound * mpe * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_wi8[k], g8.work_items.data(), g8.work_items.size() * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(d_nwi8[k], &g8.n_wi, 4, hipMemcpyHostToDevice));
    }
    {
        unsigned members = 0;
        for (auto& s : sels) for (int e : s.sel) members += (e != (int)SENTINEL);
        fprintf(stderr, "[h] b=%u E=%u ppr=%u: members/call=%.1f distinct experts/call=%u n_wi exact=%d bound=%u\n",
                b, E, ppr, (double)members / NSEL, E, n_wi_exact[0], wi_bound(b * NU, gbound, CHUNK));
    }
    // scratch: builder outputs (production path), mid, midq, partials, out
    int* gc = kb::dalloc<int>(gbound);
    int* em = kb::dalloc<int>((size_t)gbound * mpe);
    int* wi = kb::dalloc<int>(gbound + b * NU);
    int* nwi = kb::dalloc<int>(1);
    float* mid = kb::dalloc<float>((size_t)b * NU * N_FF);
    uint8_t* midq = kb::dalloc<uint8_t>((size_t)b * NU * MIDQ_SLOT);
    float* partials = kb::dalloc<float>((size_t)b * NU * N_EMBD);
    float* out = kb::dalloc<float>((size_t)b * N_EMBD);
    float* mid2 = kb::dalloc<float>((size_t)b * NU * N_FF);   // candidate outputs for compare
    float* partials2 = kb::dalloc<float>((size_t)b * NU * N_EMBD);
    float* out2 = kb::dalloc<float>((size_t)b * N_EMBD);
    uint8_t* midq2 = kb::dalloc<uint8_t>((size_t)b * NU * MIDQ_SLOT);

    const unsigned nwi_bound = wi_bound(b * NU, gbound, CHUNK);
    const double bytes_call = (double)E * (2.0 * GBPE + DBPE);
    const double flops_call = [&] { double m = 0; for (auto& s : sels) for (int e : s.sel) m += (e != (int)SENTINEL); m /= NSEL;
                                    return m * (2.0 * 2 * N_FF * N_EMBD + 2.0 * N_EMBD * N_FF); }();

    // ---- production chain pieces (exact wrapper geometry / args)
    auto L_builders = [&](hipStream_t s, unsigned k, unsigned chunk) {
        KB_CHECK(hipMemsetAsync(gc, 0, gbound * 4, s));
        kb::launch(f_grp, dim3((b * NU + 511) / 512), dim3(512), 0, s, gc, em, (const int*)d_sel[k], (const int*)remap,
                   0u, (unsigned)NU, b, (unsigned)NU, gbound, mpe);
        KB_CHECK(hipMemsetAsync(nwi, 0, 4, s));
        kb::launch(f_wi, dim3((gbound + 255) / 256), dim3(256), 0, s, wi, nwi, (const int*)gc, gbound, chunk, gbound + b * NU);
    };
    auto L_kwide = [&](hipStream_t s, hipFunction_t f, float* midp, const int* gcp, const int* emp, const int* wip, unsigned ny,
                       const int* nwip, unsigned chunk) {
        kb::launch(f, dim3(N_FF / 8, ny), dim3(256), 0, s, midp, (const uint8_t*)gate, (const uint8_t*)up, (const uint8_t*)xq,
                   (const float*)ew, gcp, emp, wip, (unsigned)GBPE, (unsigned)GBPE, (unsigned)NU, mpe, chunk, CLAMP,
                   (unsigned)N_FF, (unsigned)NB_GATE, nwip);
    };
    auto L_q8k = [&](hipStream_t s, uint8_t* mq, const float* m) {
        kb::launch(f_q8k, dim3(NB_DOWN * NU * b), dim3(256), 0, s, mq, m, NB_DOWN * NU * b);
    };
    auto L_kwide2 = [&](hipStream_t s, hipFunction_t f, float* part, const uint8_t* mq, const int* gcp, const int* emp,
                        const int* wip, unsigned ny, const int* nwip, unsigned chunk) {
        kb::launch(f, dim3(N_EMBD / 16, ny), dim3(256), 0, s, part, (const uint8_t*)down, mq, gcp, emp, wip, (unsigned)DBPE,
                   (unsigned)MIDQ_SLOT, (unsigned)NU, mpe, chunk, (unsigned)N_EMBD, (unsigned)NB_DOWN, nwip);
    };
    auto L_reduce = [&](hipStream_t s, float* o, const float* part, unsigned k) {
        kb::launch(f_reduce, dim3((N_EMBD * b + 255) / 256), dim3(256), 0, s, o, part, (const int*)d_sel[k], (const int*)remap,
                   0u, (unsigned)NU, (unsigned)NU, (unsigned)N_EMBD);
    };
    // Full production chain for call k (box-2 batched_pass, first pass).
    auto chain_base = [&](hipStream_t s, unsigned k) {
        L_builders(s, k, CHUNK);
        L_kwide(s, f_kwide, mid, gc, em, wi, exact_wi ? n_wi_exact[k] : nwi_bound, exact_wi ? nullptr : nwi, CHUNK);
        L_q8k(s, midq, mid);
        KB_CHECK(hipMemsetAsync(partials, 0, (size_t)b * NU * N_EMBD * 4, s));
        L_kwide2(s, f_kwide2, partials, midq, gc, em, wi, exact_wi ? n_wi_exact[k] : nwi_bound, exact_wi ? nullptr : nwi, CHUNK);
        L_reduce(s, out, partials, k);
    };
    // b=1 hetsplit twin, one row t (box-2 decode branch).
    auto chain_twin_row = [&](hipStream_t s, unsigned k, unsigned t, float* midp, uint8_t* mq, float* o) {
        kb::launch(f_pair_het, dim3(N_FF / 8, NU), dim3(256), 0, s, midp + (size_t)t * NU * N_FF, (const uint8_t*)gate,
                   (const uint8_t*)up, (const uint8_t*)(xq + t * XQ_TOK), (const float*)(ew + t * NU),
                   (const int*)(d_sel[k] + t * NU), (const int*)remap, 0u, (unsigned)NU, (unsigned)GBPE, (unsigned)GBPE,
                   CLAMP, (unsigned)N_FF, (unsigned)NB_GATE);
        kb::launch(f_q8k, dim3(NB_DOWN * NU), dim3(256), 0, s, mq + (size_t)t * NU * MIDQ_SLOT, (const float*)(midp + (size_t)t * NU * N_FF),
                   NB_DOWN * NU);
        kb::launch(f_down_het, dim3(N_EMBD / 8), dim3(256), 0, s, o + (size_t)t * N_EMBD, (const uint8_t*)down,
                   (const uint8_t*)(mq + (size_t)t * NU * MIDQ_SLOT), (const int*)(d_sel[k] + t * NU), (const int*)remap, 0u,
                   (unsigned)NU, (unsigned)DBPE, (unsigned)MIDQ_SLOT, (unsigned)NU, (unsigned)N_EMBD, (unsigned)NB_DOWN);
    };

    // ---- candidates (loaded on demand)
    struct Cand { std::string name; hipFunction_t f_gu = nullptr, f_dn = nullptr; bool chunk8 = false; bool own_groups = false; };
    std::vector<Cand> cl;
    std::vector<kb::Module*> cmods;
    {
        std::string rest = cands;
        while (!rest.empty()) {
            auto p = rest.find(',');
            std::string c = rest.substr(0, p);
            rest = p == std::string::npos ? "" : rest.substr(p + 1);
            if (c.empty()) continue;
            // spec: <hsaco-name>[.flags]; hsaco-name "base" = production kernels; flags: c8 (chunk 8), nobuild
            Cand cd;
            cd.name = c;
            std::string file = c.substr(0, c.find('.'));
            if (file != "base") {
                auto* m = new kb::Module(dir + "/cand_" + file + "_" + arch + ".hsaco");
                cmods.push_back(m);
                // convention: candidate hsaco exports cand_gate_up and/or cand_down (see cand_*.hip)
                hipFunction_t f;
                if (hipModuleGetFunction(&f, m->m, "cand_gate_up") == hipSuccess) cd.f_gu = f;
                if (hipModuleGetFunction(&f, m->m, "cand_down") == hipSuccess) cd.f_dn = f;
            }
            cd.chunk8 = c.find("c8") != std::string::npos;
            cd.own_groups = c.find("nobuild") != std::string::npos;
            cl.push_back(cd);
        }
    }
    auto chain_cand = [&](hipStream_t s, unsigned k, const Cand& c, float* midp, uint8_t* mq, float* part, float* o) {
        const unsigned chunk = c.chunk8 ? 8 : CHUNK;
        const int *gcp, *emp, *wip, *nwip;
        unsigned ny;
        if (c.own_groups) {  // idea: host-known groups (no builders); exact grid
            gcp = c.chunk8 ? d_gc8[k] : d_gc[k]; emp = c.chunk8 ? d_em8[k] : d_em[k]; wip = c.chunk8 ? d_wi8[k] : d_wi[k];
            nwip = nullptr; ny = c.chunk8 ? n_wi8[k] : n_wi_exact[k];
        } else {
            L_builders(s, k, chunk);
            gcp = gc; emp = em; wip = wi; nwip = exact_wi ? nullptr : nwi;
            ny = exact_wi ? (c.chunk8 ? n_wi8[k] : n_wi_exact[k]) : wi_bound(b * NU, gbound, chunk);
        }
        L_kwide(s, c.f_gu ? c.f_gu : f_kwide, midp, gcp, emp, wip, ny, nwip, chunk);
        L_q8k(s, mq, midp);
        KB_CHECK(hipMemsetAsync(part, 0, (size_t)b * NU * N_EMBD * 4, s));
        L_kwide2(s, c.f_dn ? c.f_dn : f_kwide2, part, mq, gcp, emp, wip, ny, nwip, chunk);
        L_reduce(s, o, part, k);
    };

    auto zero_all = [&]() {
        KB_CHECK(hipMemset(mid, 0, (size_t)b * NU * N_FF * 4)); KB_CHECK(hipMemset(mid2, 0, (size_t)b * NU * N_FF * 4));
        KB_CHECK(hipMemset(partials, 0, (size_t)b * NU * N_EMBD * 4)); KB_CHECK(hipMemset(partials2, 0, (size_t)b * NU * N_EMBD * 4));
        KB_CHECK(hipMemset(out, 0, (size_t)b * N_EMBD * 4)); KB_CHECK(hipMemset(out2, 0, (size_t)b * N_EMBD * 4));
        KB_CHECK(hipMemset(midq, 0, (size_t)b * NU * MIDQ_SLOT)); KB_CHECK(hipMemset(midq2, 0, (size_t)b * NU * MIDQ_SLOT));
    };

    char tagbuf[256];
    snprintf(tagbuf, sizeof tagbuf, "b=%u E=%u ppr=%u P=%u gbound=%u %s cold-rotating", b, E, ppr, P, gbound,
             exact_wi ? "exact-grid" : "bound-grid");

    if (mode == "chain" || mode == "twin") {
        // ---- correctness: base chain vs twin (sanity of the harness) and vs each candidate
        zero_all();
        chain_base(0, 0);
        KB_CHECK(hipDeviceSynchronize());
        auto mid_ref = kb::d2h(mid, (size_t)b * NU * N_FF), part_ref = kb::d2h(partials, (size_t)b * NU * N_EMBD),
             out_ref = kb::d2h(out, (size_t)b * N_EMBD);
        {
            size_t nz = 0, nf = 0;
            for (float v : out_ref) { nz += v != 0.f; nf += !std::isfinite(v); }
            printf("base out: %zu/%zu nonzero, %zu nonfinite, out[0..3]=%g %g %g %g\n", nz, out_ref.size(), nf, out_ref[0], out_ref[1], out_ref[2], out_ref[3]);
        }
        for (unsigned t = 0; t < b; ++t) chain_twin_row(0, 0, t, mid2, midq2, out2);
        KB_CHECK(hipDeviceSynchronize());
        {
            auto m2 = kb::d2h(mid2, (size_t)b * NU * N_FF), o2 = kb::d2h(out2, (size_t)b * N_EMBD);
            kb::print_cmp("twin(hetsplit b=1 per row) mid vs base", kb::compare_f32(mid_ref.data(), m2.data(), m2.size()));
            kb::print_cmp("twin(hetsplit b=1 per row) out vs base", kb::compare_f32(out_ref.data(), o2.data(), o2.size()));
        }
        for (auto& c : cl) {
            zero_all();
            chain_cand(0, 0, c, mid2, midq2, partials2, out2);
            KB_CHECK(hipDeviceSynchronize());
            auto m2 = kb::d2h(mid2, (size_t)b * NU * N_FF), p2 = kb::d2h(partials2, (size_t)b * NU * N_EMBD), o2 = kb::d2h(out2, (size_t)b * N_EMBD);
            kb::print_cmp((c.name + " mid vs base").c_str(), kb::compare_f32(mid_ref.data(), m2.data(), m2.size()));
            kb::print_cmp((c.name + " partials vs base").c_str(), kb::compare_f32(part_ref.data(), p2.data(), p2.size()));
            kb::print_cmp((c.name + " out vs base").c_str(), kb::compare_f32(out_ref.data(), o2.data(), o2.size()));
            // tails: every call's selection (different member counts / slot positions)
            size_t worst_bits = 0; double worst_rel = 0;
            for (unsigned k = 1; k < NSEL; ++k) {
                zero_all();
                chain_base(0, k); chain_cand(0, k, c, mid2, midq2, partials2, out2);
                KB_CHECK(hipDeviceSynchronize());
                auto o1 = kb::d2h(out, (size_t)b * N_EMBD), o2b = kb::d2h(out2, (size_t)b * N_EMBD);
                auto cm = kb::compare_f32(o1.data(), o2b.data(), o1.size());
                worst_bits = std::max(worst_bits, cm.n_bit_diff); worst_rel = std::max(worst_rel, cm.rmse / std::max(1e-30, cm.ref_rms));
            }
            printf("CMP %s out over %u selections: worst bit_diff=%zu worst rel_rmse=%.3e\n", c.name.c_str(), NSEL, worst_bits, worst_rel);
        }
        // ---- timing
        std::vector<kb::Variant> vs;
        std::vector<unsigned> ctr(2 + cl.size(), 0);
        vs.push_back({"base chain (builders+kwide+q8k+kwide2+reduce)", [&](hipStream_t s) { chain_base(s, ctr[0]++ % NSEL); }, bytes_call, flops_call});
        if (mode == "twin") {
            vs.push_back({"twin: hetsplit b=1 chain x b rows", [&](hipStream_t s) { unsigned k = ctr[1]++ % NSEL; for (unsigned t = 0; t < b; ++t) chain_twin_row(s, k, t, mid2, midq2, out2); }, bytes_call * (b == 1 ? 1.0 : 1.0), flops_call});
        }
        for (size_t i = 0; i < cl.size(); ++i) {
            const Cand& c = cl[i];
            vs.push_back({"cand " + c.name, [&, i](hipStream_t s) { chain_cand(s, ctr[2 + i]++ % NSEL, c, mid2, midq2, partials2, out2); }, bytes_call, flops_call});
        }
        kb::AbOpts o;
        o.graph = graph; o.inner = inner; o.rounds = rounds; o.tag = tagbuf;
        kb::ab(vs, o);
        double roof = bytes_call / 214e9 * 1e6;
        printf("roofline: %.1f MB/call -> %.1f us at 214 GB/s (%.1f us at 231)\n", bytes_call / 1e6, roof, bytes_call / 231e9 * 1e6);
    } else if (mode == "parts") {
        // per-kernel breakdown: each piece alone (graph of `inner` calls), inputs from the host-built groups.
        std::vector<kb::Variant> vs;
        std::vector<unsigned> ctr(16, 0);
        vs.push_back({"builders (2 memset + 2 kernels)", [&](hipStream_t s) { L_builders(s, ctr[0]++ % NSEL, CHUNK); }, 0, 0});
        vs.push_back({"kwide gate/up (bound grid, devcount)", [&](hipStream_t s) { unsigned k = ctr[1]++ % NSEL; L_kwide(s, f_kwide, mid, d_gc[k], d_em[k], d_wi[k], nwi_bound, d_nwi[k], CHUNK); }, (double)E * 2.0 * GBPE, 0});
        vs.push_back({"kwide gate/up (exact grid)", [&](hipStream_t s) { unsigned k = ctr[2]++ % NSEL; L_kwide(s, f_kwide, mid, d_gc[k], d_em[k], d_wi[k], n_wi_exact[k], nullptr, CHUNK); }, (double)E * 2.0 * GBPE, 0});
        vs.push_back({"q8k (9*6*b blocks)", [&](hipStream_t s) { L_q8k(s, midq, mid); }, (double)b * NU * (N_FF * 4 + MIDQ_SLOT), 0});
        vs.push_back({"partials memset", [&](hipStream_t s) { KB_CHECK(hipMemsetAsync(partials, 0, (size_t)b * NU * N_EMBD * 4, s)); }, (double)b * NU * N_EMBD * 4, 0});
        vs.push_back({"kwide2 down (bound grid, devcount)", [&](hipStream_t s) { unsigned k = ctr[3]++ % NSEL; L_kwide2(s, f_kwide2, partials, midq, d_gc[k], d_em[k], d_wi[k], nwi_bound, d_nwi[k], CHUNK); }, (double)E * DBPE, 0});
        vs.push_back({"kwide2 down (exact grid)", [&](hipStream_t s) { unsigned k = ctr[4]++ % NSEL; L_kwide2(s, f_kwide2, partials, midq, d_gc[k], d_em[k], d_wi[k], n_wi_exact[k], nullptr, CHUNK); }, (double)E * DBPE, 0});
        vs.push_back({"reduce hetsplit", [&](hipStream_t s) { L_reduce(s, out, partials, ctr[5]++ % NSEL); }, (double)b * NU * N_EMBD * 4, 0});
        vs.push_back({"twin pair_hetsplit b=1 (E experts)", [&](hipStream_t s) { unsigned k = ctr[6]++ % NSEL;
            kb::launch(f_pair_het, dim3(N_FF / 8, NU), dim3(256), 0, s, mid2, (const uint8_t*)gate, (const uint8_t*)up, (const uint8_t*)xq,
                       (const float*)ew, (const int*)d_sel[k], (const int*)remap, 0u, (unsigned)NU, (unsigned)GBPE, (unsigned)GBPE, CLAMP, (unsigned)N_FF, (unsigned)NB_GATE); },
            (double)E * 2.0 * GBPE, 0});
        vs.push_back({"twin down_hetsplit b=1 (E experts)", [&](hipStream_t s) { unsigned k = ctr[7]++ % NSEL;
            kb::launch(f_down_het, dim3(N_EMBD / 8), dim3(256), 0, s, out2, (const uint8_t*)down, (const uint8_t*)midq, (const int*)d_sel[k], (const int*)remap, 0u,
                       (unsigned)NU, (unsigned)DBPE, (unsigned)MIDQ_SLOT, (unsigned)NU, (unsigned)N_EMBD, (unsigned)NB_DOWN); },
            (double)E * DBPE, 0});
        for (size_t i = 0; i < cl.size(); ++i) {
            const Cand& c = cl[i];
            if (c.f_gu) vs.push_back({"cand " + c.name + " gate/up", [&, i](hipStream_t s) { unsigned k = ctr[8 + i]++ % NSEL; const Cand& c = cl[i];
                L_kwide(s, c.f_gu, mid2, c.chunk8 ? d_gc8[k] : d_gc[k], c.chunk8 ? d_em8[k] : d_em[k], c.chunk8 ? d_wi8[k] : d_wi[k],
                        exact_wi ? (c.chunk8 ? n_wi8[k] : n_wi_exact[k]) : wi_bound(b * NU, gbound, c.chunk8 ? 8 : CHUNK), exact_wi ? nullptr : (c.chunk8 ? d_nwi8[k] : d_nwi[k]), c.chunk8 ? 8 : CHUNK); },
                (double)E * 2.0 * GBPE, 0});
            if (c.f_dn) vs.push_back({"cand " + c.name + " down", [&, i](hipStream_t s) { unsigned k = ctr[12 + i]++ % NSEL; const Cand& c = cl[i];
                L_kwide2(s, c.f_dn, partials2, midq, c.chunk8 ? d_gc8[k] : d_gc[k], c.chunk8 ? d_em8[k] : d_em[k], c.chunk8 ? d_wi8[k] : d_wi[k],
                         exact_wi ? (c.chunk8 ? n_wi8[k] : n_wi_exact[k]) : wi_bound(b * NU, gbound, c.chunk8 ? 8 : CHUNK), exact_wi ? nullptr : (c.chunk8 ? d_nwi8[k] : d_nwi[k]), c.chunk8 ? 8 : CHUNK); },
                (double)E * DBPE, 0});
        }
        // the sub-kernel that streams weights must see them cold: the rotation handles that since every
        // variant's calls rotate over the same pool; the small kernels are warm as in production.
        kb::AbOpts o;
        o.graph = graph; o.inner = inner; o.rounds = rounds; o.tag = tagbuf;
        kb::ab(vs, o);
        printf("roofline: gate/up %.1f MB -> %.1f us; down %.1f MB -> %.1f us at 214 GB/s\n", E * 2.0 * GBPE / 1e6, E * 2.0 * GBPE / 214e3,
               E * (double)DBPE / 1e6, E * (double)DBPE / 214e3);
    } else if (mode == "prof") {
        // direct launches for rocprofv3: `n` chain calls (ATT picks kernel iterations 3-4 by regex)
        const unsigned n = geti("n", 6);
        kb::warm(200);
        for (unsigned i = 0; i < n; ++i) {
            if (cl.empty()) chain_base(0, i % NSEL);
            else chain_cand(0, i % NSEL, cl[0], mid2, midq2, partials2, out2);
        }
        if (geti("twin", 0)) for (unsigned i = 0; i < n; ++i) chain_twin_row(0, i % NSEL, 0, mid2, midq2, out2);
        KB_CHECK(hipDeviceSynchronize());
        printf("prof: %u chain calls done\n", n);
    }
    return 0;
}
