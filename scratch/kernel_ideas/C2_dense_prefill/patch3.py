#!/usr/bin/env python3
# Patch 3: BRF flag (guarded vs branch-free loads) and DB (double-buffered LDS, one barrier per
# k-outer, fragment loads issued before the next tile's stage so LDS reads overlap the VALU dequant).
p='cand.hip'; s=open(p).read()

old='''template <int BM, int BN, int WM, int WN, int PF, bool BI8, bool LB, int ABL = 0>
__device__ __forceinline__ void f16x_core('''
new='''// BRF: branch-free B loads (clamped row + select) vs execz-guarded loads (production style).
// DB: double-buffered LDS tiles; per k-outer: [frag ds_loads from buf cur] [stage next -> buf cur^1]
//     [reload regs] [WMMAs] [ONE barrier]. Register prefetch PF stages deep = PF iterations of
//     global latency hiding. LDS 2x.
template <int BM, int BN, int WM, int WN, int PF, bool BI8, bool LB, int ABL = 0, bool BRF = true, bool DB = false>
__device__ __forceinline__ void f16x_core('''
assert old in s; s=s.replace(old,new)

old='''    __shared__ __attribute__((aligned(16))) _Float16 A_tile[BM * LDS_STRIDE];
    __shared__ __attribute__((aligned(16))) _Float16 B_tile[BN * LDS_STRIDE];
'''
new='''    __shared__ __attribute__((aligned(16))) _Float16 A_tile[(DB ? 2 : 1) * BM * LDS_STRIDE];
    __shared__ __attribute__((aligned(16))) _Float16 B_tile[(DB ? 2 : 1) * BN * LDS_STRIDE];
'''
assert old in s; s=s.replace(old,new)

# load_raw: both styles
old='''        for (int i = 0; i < B_PER; ++i) {
            const uint32_t slot = tid + i * NT_;
            if ((B_SLOTS % NT_ != 0) && slot >= B_SLOTS) { vb[p][i] = make_uint4(0u, 0u, 0u, 0u); if (BI8) xs[p][i] = 0.f; continue; }
            uint4 t; float ts = 0.f;'''
new='''        for (int i = 0; i < B_PER; ++i) {
            const uint32_t slot = tid + i * NT_;
            if ((B_SLOTS % NT_ != 0) && slot >= B_SLOTS) { vb[p][i] = make_uint4(0u, 0u, 0u, 0u); if (BI8) xs[p][i] = 0.f; continue; }
            if (!BRF) {   // production style: guarded load
                vb[p][i] = make_uint4(0u, 0u, 0u, 0u);
                if (BI8) xs[p][i] = 0.f;
                if (b_on[i]) {
                    if (BI8) { __builtin_memcpy(&vb[p][i], (const int8_t*)b_src[i] + ko + b_koff[i], 16); xs[p][i] = bs_src[i][ko >> 5]; }
                    else     { __builtin_memcpy(&vb[p][i], (const uint16_t*)b_src[i] + ko + b_koff[i], 16); }
                }
                continue;
            }
            uint4 t; float ts = 0.f;'''
assert old in s; s=s.replace(old,new)

# stage(): take a buffer offset
old='''    auto stage = [&](int p) {
        #pragma unroll
        for (int i = 0; i < A_PER; ++i) if (a_on[i]) {'''
new='''    auto stage = [&](int p, uint32_t buf) {
        _Float16* At = A_tile + buf * (BM * LDS_STRIDE);
        _Float16* Bt = B_tile + buf * (BN * LDS_STRIDE);
        #pragma unroll
        for (int i = 0; i < A_PER; ++i) if (a_on[i]) {'''
assert old in s; s=s.replace(old,new)
s=s.replace('''            uint4* dst = (uint4*)(&A_tile[a_row[i] * LDS_STRIDE + a_koff[i]]);
            dst[0] = make_uint4(o[0], o[1], o[2], o[3]);''','''            uint4* dst = (uint4*)(&At[a_row[i] * LDS_STRIDE + a_koff[i]]);
            dst[0] = make_uint4(o[0], o[1], o[2], o[3]);''')
s=s.replace('''                uint4* dst = (uint4*)(&B_tile[b_row[i] * LDS_STRIDE + b_koff[i]]);
                __builtin_memcpy(&dst[0], &v[0], 16);''','''                uint4* dst = (uint4*)(&Bt[b_row[i] * LDS_STRIDE + b_koff[i]]);
                __builtin_memcpy(&dst[0], &v[0], 16);''')
s=s.replace('''                uint4* dst = (uint4*)(&B_tile[b_row[i] * LDS_STRIDE + b_koff[i]]);
                dst[0] = vb[p][i];''','''                uint4* dst = (uint4*)(&Bt[b_row[i] * LDS_STRIDE + b_koff[i]]);
                dst[0] = vb[p][i];''')
# existing call sites stage(p) -> stage(p, 0)
s=s.replace('stage(0); __syncthreads();','stage(0, 0); __syncthreads();')
s=s.replace('if (ABL == 0 || ABL == 1) { stage(p); sync(); }','if (ABL == 0 || ABL == 1) { stage(p, 0); sync(); }')
s=s.replace('                stage(p); sync();\n                #pragma unroll\n                for (uint32_t k_inner','                stage(p, 0); sync();\n                #pragma unroll\n                for (uint32_t k_inner')
assert s.count('stage(p)') == 0, s.count('stage(p)')

# DB main loop, inserted before the existing (non-DB) loop
old='''    uint32_t sink = 0u;   // ABL 2/3: keeps loads/fragments alive'''
new='''    if (DB) {
        #pragma unroll
        for (int p = 0; p < PF; ++p) if ((uint32_t)p * BK < K) load_raw(p, p * BK);
        stage(0, 0);
        if ((uint32_t)PF * BK < K) load_raw(0, PF * BK);
        sync();
        for (uint32_t k0 = 0; k0 < K; k0 += PF * BK) {
            #pragma unroll
            for (int p = 0; p < PF; ++p) {
                const uint32_t ko = k0 + p * BK;
                if (ko >= K) break;
                const uint32_t cur = (ko / BK) & 1u;
                const _Float16* At = A_tile + cur * (BM * LDS_STRIDE);
                const _Float16* Bt = B_tile + cur * (BN * LDS_STRIDE);
                half8 a_frags[2][MT], b_frags[2][NTL];
                #pragma unroll
                for (int ki = 0; ki < 2; ++ki) {
                    const uint32_t kf = ki * 16u + (lane >> 4) * 8u;
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) __builtin_memcpy(&a_frags[ki][mt], &At[(wave_m0 + mt * 16u + (lane & 15u)) * LDS_STRIDE + kf], 16);
                    #pragma unroll
                    for (int nt = 0; nt < NTL; ++nt) __builtin_memcpy(&b_frags[ki][nt], &Bt[(wave_n0 + nt * 16u + (lane & 15u)) * LDS_STRIDE + kf], 16);
                }
                const uint32_t kn = ko + BK;
                if (kn < K) {
                    stage((p + 1) % PF, cur ^ 1u);
                    if (kn + PF * BK < K) load_raw((p + 1) % PF, kn + PF * BK);
                }
                #pragma unroll
                for (int ki = 0; ki < 2; ++ki)
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
                        #pragma unroll
                        for (int nt = 0; nt < NTL; ++nt)
                            acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frags[ki][mt], b_frags[ki][nt], acc[mt][nt]);
                sync();
            }
        }
    } else {
    uint32_t sink = 0u;   // ABL 2/3: keeps loads/fragments alive'''
assert old in s; s=s.replace(old,new)
old='''    if (ABL == 2 && sink == 0xFFFFFFFFu) acc[0][0][0] += 1.f;   // practically never true; keeps sink live
'''
new='''    if (ABL == 2 && sink == 0xFFFFFFFFu) acc[0][0][0] += 1.f;   // practically never true; keeps sink live
    }
'''
assert old in s; s=s.replace(old,new)

# new kernels
old='''F16X_ABL(q8_0_gemm_wmma_f16x_abl_noglobal, 1, 1)'''
new='''#define F16X_DB(NAME, BM, BN, WM, WN, PF)                                                            \\
extern "C" __global__ void __launch_bounds__(WM * WN * 32) NAME(                                     \\
    float* __restrict__ out, const unsigned char* __restrict__ w, const uint16_t* __restrict__ x16,   \\
    uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks, uint32_t ldx)         \\
{ f16x_core<BM, BN, WM, WN, PF, false, true, 0, true, true>(out, w, x16, nullptr, nullptr, K, M, n_groups, batch, blocks, ldx); }
F16X_DB(q8_0_gemm_wmma_f16x_db,      128, 128, 2, 4, 2)
F16X_DB(q8_0_gemm_wmma_f16x_db_pf1,  128, 128, 2, 4, 1)
F16X_DB(q8_0_gemm_wmma_f16x_db_pf3,  128, 128, 2, 4, 3)
F16X_DB(q8_0_gemm_wmma_f16x_db_bn64, 128,  64, 4, 2, 2)
F16X_ABL(q8_0_gemm_wmma_f16x_abl_noglobal, 1, 1)'''
assert old in s; s=s.replace(old,new)
old='''extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_bn64('''
new='''extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_grd(   // r1 style: guarded loads
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
    f16x_core<128, 128, 2, 4, 1, true, true, 0, false, false>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_db(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
    f16x_core<128, 128, 2, 4, 2, true, true, 0, true, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_db_grd(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
    f16x_core<128, 128, 2, 4, 2, true, true, 0, false, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_bn64('''
assert old in s; s=s.replace(old,new)
open(p,'w').write(s); print("cand ok")

p='harness.cpp'; s=open(p).read()
old='''        || name == "q8_0_gemm_wmma_f16x_pf3" || name == "q8_0_gemm_wmma_f16x_pf4" || name.rfind("q8_0_gemm_wmma_f16x_abl", 0) == 0 || name == "q8_0_gemm_wmma_f16x_pf2n") {'''
new='''        || name == "q8_0_gemm_wmma_f16x_pf3" || name == "q8_0_gemm_wmma_f16x_pf4" || name.rfind("q8_0_gemm_wmma_f16x_abl", 0) == 0 || name == "q8_0_gemm_wmma_f16x_pf2n"
        || name == "q8_0_gemm_wmma_f16x_db" || name == "q8_0_gemm_wmma_f16x_db_pf1" || name == "q8_0_gemm_wmma_f16x_db_pf3") {'''
assert old in s; s=s.replace(old,new)
old='''    if (name == "q8_0_gemm_wmma_f16x_bn64") {'''
new='''    if (name == "q8_0_gemm_wmma_f16x_bn64" || name == "q8_0_gemm_wmma_f16x_db_bn64") {'''
assert old in s; s=s.replace(old,new)
old='''    if ((name == "q8_0_gemm_wmma_i8x" || name == "q8_0_gemm_wmma_i8x_pf2" || name == "q8_0_gemm_wmma_i8x_pf3") && s.kind == Shape::LDS_TILED) {'''
new='''    if ((name == "q8_0_gemm_wmma_i8x" || name == "q8_0_gemm_wmma_i8x_pf2" || name == "q8_0_gemm_wmma_i8x_pf3" || name == "q8_0_gemm_wmma_i8x_grd"
         || name == "q8_0_gemm_wmma_i8x_db" || name == "q8_0_gemm_wmma_i8x_db_grd") && s.kind == Shape::LDS_TILED) {'''
assert old in s; s=s.replace(old,new)
open(p,'w').write(s); print("harness ok")
