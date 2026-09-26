#!/usr/bin/env python3
# Patch 2 (2026-09-26 resume): harness module fix; branch-free loads; PF3/PF4; ablations; i8x_pf2/3.
import sys
p='harness.cpp'; s=open(p).read()
old='''            if (c == "f16x_at_b") cfs.push_back({c, base.fn("q8_0_gemm_wmma_f16x") });'''
new='''            if (c == "f16x_at_b") { static kb::Module wm(g_dir + "/base_q8_0_matvec_wmma_" + g_arch + ".hsaco"); cfs.push_back({c, wm.fn("q8_0_gemm_wmma_f16x") }); }'''
assert old in s; s=s.replace(old,new)
old2='''    if (name == "q8_0_gemm_wmma_f16x_v2" || name == "q8_0_gemm_wmma_f16x_pf2" || name == "q8_0_gemm_wmma_f16x_lb" || name == "q8_0_gemm_wmma_f16x_pf2b") {'''
new2='''    if (name == "q8_0_gemm_wmma_f16x_v2" || name == "q8_0_gemm_wmma_f16x_pf2" || name == "q8_0_gemm_wmma_f16x_lb" || name == "q8_0_gemm_wmma_f16x_pf2b"
        || name == "q8_0_gemm_wmma_f16x_pf3" || name == "q8_0_gemm_wmma_f16x_pf4" || name.rfind("q8_0_gemm_wmma_f16x_abl", 0) == 0 || name == "q8_0_gemm_wmma_f16x_pf2n") {'''
assert old2 in s; s=s.replace(old2,new2)
old3='''    if (name == "q8_0_gemm_wmma_i8x" && s.kind == Shape::LDS_TILED) {'''
new3='''    if ((name == "q8_0_gemm_wmma_i8x" || name == "q8_0_gemm_wmma_i8x_pf2" || name == "q8_0_gemm_wmma_i8x_pf3") && s.kind == Shape::LDS_TILED) {'''
assert old3 in s; s=s.replace(old3,new3)
open(p,'w').write(s); print("harness ok")

p='cand.hip'; s=open(p).read()
old='''template <int BM, int BN, int WM, int WN, int PF, bool BI8, bool LB>
__device__ __forceinline__ void f16x_core('''
new='''// ABL (ablation, NOT correct, for pricing the parts): 0 = none; 1 = no global loads after k-outer 0
// (registers reused); 2 = no LDS traffic after k-outer 0 (fragments reused; the global loads are
// still consumed through a cheap xor so they are not dead); 3 = no WMMAs (fragments xor-folded).
template <int BM, int BN, int WM, int WN, int PF, bool BI8, bool LB, int ABL = 0>
__device__ __forceinline__ void f16x_core('''
assert old in s; s=s.replace(old,new)
old='''        a_on[i] = slot < A_SLOTS;'''
new='''        a_on[i] = (A_SLOTS % NT_ == 0) ? true : (slot < A_SLOTS);   // compile-time when it can be'''
assert old in s; s=s.replace(old,new)
old='''        const bool in = slot < B_SLOTS;
        if (BI8) { b_row[i] = in ? slot >> 1 : 0u; b_koff[i] = (slot & 1u) * 16u; }
        else     { b_row[i] = in ? slot >> 2 : 0u; b_koff[i] = (slot & 3u) * 8u; }
        b_on[i] = in && (n0 + b_row[i]) < batch;
        if (BI8) {
            b_src[i] = xq + (size_t)(n0 + b_row[i]) * (size_t)ldx + (size_t)g * K;
            bs_src[i] = xscale + (size_t)(n0 + b_row[i]) * (size_t)(ldx / 32u) + (size_t)g * blocks;
        } else {
            b_src[i] = x16 + (size_t)(n0 + b_row[i]) * (size_t)ldx + (size_t)g * K;
            bs_src[i] = nullptr;
        }'''
new='''        const bool in = (B_SLOTS % NT_ == 0) ? true : (slot < B_SLOTS);
        if (BI8) { b_row[i] = in ? slot >> 1 : 0u; b_koff[i] = (slot & 1u) * 16u; }
        else     { b_row[i] = in ? slot >> 2 : 0u; b_koff[i] = (slot & 3u) * 8u; }
        b_on[i] = in && (n0 + b_row[i]) < batch;
        // Branch-free loads: an out-of-range token reads the LAST valid row (always mapped) and the
        // register is zeroed by a select afterwards, so the waitcnt pass sees straight-line loads
        // and can count them (an execz-guarded load forced s_wait_loadcnt 0 at every stage).
        const uint32_t brow = (n0 + b_row[i] < batch) ? (n0 + b_row[i]) : (batch - 1u);
        if (BI8) {
            b_src[i] = xq + (size_t)brow * (size_t)ldx + (size_t)g * K;
            bs_src[i] = xscale + (size_t)brow * (size_t)(ldx / 32u) + (size_t)g * blocks;
        } else {
            b_src[i] = x16 + (size_t)brow * (size_t)ldx + (size_t)g * K;
            bs_src[i] = nullptr;
        }'''
assert old in s; s=s.replace(old,new)
old='''        for (int i = 0; i < B_PER; ++i) {
            vb[p][i] = make_uint4(0u, 0u, 0u, 0u);
            if (BI8) xs[p][i] = 0.f;
            if (b_on[i]) {
                if (BI8) {
                    __builtin_memcpy(&vb[p][i], (const int8_t*)b_src[i] + ko + b_koff[i], 16);
                    xs[p][i] = bs_src[i][ko >> 5];
                } else {
                    __builtin_memcpy(&vb[p][i], (const uint16_t*)b_src[i] + ko + b_koff[i], 16);
                }
            }
        }
    };'''
new='''        for (int i = 0; i < B_PER; ++i) {
            const uint32_t slot = tid + i * NT_;
            if ((B_SLOTS % NT_ != 0) && slot >= B_SLOTS) { vb[p][i] = make_uint4(0u, 0u, 0u, 0u); if (BI8) xs[p][i] = 0.f; continue; }
            uint4 t; float ts = 0.f;
            if (BI8) {
                __builtin_memcpy(&t, (const int8_t*)b_src[i] + ko + b_koff[i], 16);
                ts = bs_src[i][ko >> 5];
            } else {
                __builtin_memcpy(&t, (const uint16_t*)b_src[i] + ko + b_koff[i], 16);
            }
            vb[p][i].x = b_on[i] ? t.x : 0u; vb[p][i].y = b_on[i] ? t.y : 0u;
            vb[p][i].z = b_on[i] ? t.z : 0u; vb[p][i].w = b_on[i] ? t.w : 0u;
            if (BI8) xs[p][i] = b_on[i] ? ts : 0.f;
        }
    };'''
assert old in s; s=s.replace(old,new)
old='''    #pragma unroll
    for (int p = 0; p < PF; ++p) if ((uint32_t)p * BK < K) load_raw(p, p * BK);
    for (uint32_t k0 = 0; k0 < K; k0 += PF * BK) {
        #pragma unroll
        for (int p = 0; p < PF; ++p) {
            const uint32_t ko = k0 + p * BK;
            if (ko >= K) break;
            stage(p);
            sync();
            if (ko + PF * BK < K) load_raw(p, ko + PF * BK);
            compute();
            sync();
        }
    }
'''
new='''    uint32_t sink = 0u;   // ABL 2/3: keeps loads/fragments alive
    half8 keep_a[MT], keep_b[NTL];
    #pragma unroll
    for (int p = 0; p < PF; ++p) if ((uint32_t)p * BK < K) load_raw(p, p * BK);
    if (ABL == 2 || ABL == 3) {   // one real stage so the fragments are defined
        stage(0); __syncthreads();
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) __builtin_memcpy(&keep_a[mt], &A_tile[(wave_m0 + mt * 16u + (lane & 15u)) * LDS_STRIDE + (lane >> 4) * 8u], 16);
        #pragma unroll
        for (int nt = 0; nt < NTL; ++nt) __builtin_memcpy(&keep_b[nt], &B_tile[(wave_n0 + nt * 16u + (lane & 15u)) * LDS_STRIDE + (lane >> 4) * 8u], 16);
        __syncthreads();
    }
    for (uint32_t k0 = 0; k0 < K; k0 += PF * BK) {
        #pragma unroll
        for (int p = 0; p < PF; ++p) {
            const uint32_t ko = k0 + p * BK;
            if (ko >= K) break;
            if (ABL == 0 || ABL == 1) { stage(p); sync(); }
            if (ABL == 2) {   // consume the raw loads without LDS
                #pragma unroll
                for (int i = 0; i < A_PER; ++i) sink ^= qa[p][i].x ^ qa[p][i].w ^ (uint32_t)sbits[p][i];
                #pragma unroll
                for (int i = 0; i < B_PER; ++i) sink ^= vb[p][i].x ^ vb[p][i].w;
            }
            if (ABL == 3) {   // stage + sync, then fold the fragments into acc without WMMA
                stage(p); sync();
                #pragma unroll
                for (uint32_t k_inner = 0; k_inner < (uint32_t)BK; k_inner += 16u) {
                    const uint32_t kf = k_inner + (lane >> 4) * 8u;
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) { half8 a; __builtin_memcpy(&a, &A_tile[(wave_m0 + mt * 16u + (lane & 15u)) * LDS_STRIDE + kf], 16); acc[mt][0][0] += (float)a[0]; }
                    #pragma unroll
                    for (int nt = 0; nt < NTL; ++nt) { half8 b; __builtin_memcpy(&b, &B_tile[(wave_n0 + nt * 16u + (lane & 15u)) * LDS_STRIDE + kf], 16); acc[0][nt][1] += (float)b[0]; }
                }
            }
            if (ABL != 1 || ko == 0) { if (ko + PF * BK < K) load_raw(p, ko + PF * BK); }
            if (ABL == 0 || ABL == 1) compute();
            if (ABL == 2) {
                #pragma unroll
                for (int k_inner = 0; k_inner < BK; k_inner += 16)
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
                        #pragma unroll
                        for (int nt = 0; nt < NTL; ++nt)
                            acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(keep_a[mt], keep_b[nt], acc[mt][nt]);
            }
            if (ABL == 0 || ABL == 1 || ABL == 3) sync();
        }
    }
    if (ABL == 2 && sink == 0xFFFFFFFFu) acc[0][0][0] += 1.f;   // practically never true; keeps sink live
'''
assert old in s; s=s.replace(old,new)
old='''F16X_KERNEL(q8_0_gemm_wmma_f16x_64x64,  64,  64, 2, 2, 1, true)
'''
new='''F16X_KERNEL(q8_0_gemm_wmma_f16x_64x64,  64,  64, 2, 2, 1, true)
F16X_KERNEL(q8_0_gemm_wmma_f16x_pf3,   128, 128, 2, 4, 3, true)
F16X_KERNEL(q8_0_gemm_wmma_f16x_pf4,   128, 128, 2, 4, 4, true)
#define F16X_ABL(NAME, PF, ABL)                                                                       \\
extern "C" __global__ void __launch_bounds__(256) NAME(                                              \\
    float* __restrict__ out, const unsigned char* __restrict__ w, const uint16_t* __restrict__ x16,   \\
    uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks, uint32_t ldx)         \\
{ f16x_core<128, 128, 2, 4, PF, false, true, ABL>(out, w, x16, nullptr, nullptr, K, M, n_groups, batch, blocks, ldx); }
F16X_ABL(q8_0_gemm_wmma_f16x_abl_noglobal, 1, 1)
F16X_ABL(q8_0_gemm_wmma_f16x_abl_nolds,    2, 2)
F16X_ABL(q8_0_gemm_wmma_f16x_abl_nowmma,   2, 3)
'''
assert old in s; s=s.replace(old,new)
old='''extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_bn64('''
new='''extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_pf2(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
    f16x_core<128, 128, 2, 4, 2, true, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_pf3(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
    f16x_core<128, 128, 2, 4, 3, true, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_bn64('''
assert old in s; s=s.replace(old,new)
open(p,'w').write(s); print("cand ok")
