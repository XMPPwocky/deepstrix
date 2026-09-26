#!/usr/bin/env python3
# Patch 11: W0 flag = explicit `s_wait_loadcnt 0x0` right before the WMMA block (loads complete before
# the matrix work, as in the fast r1 binary). Tests the "VMEM returns interfere with WMMA" hypothesis.
p = 'cand.hip'; s = open(p).read()
old = '''template <int BM, int BN, int WM, int WN, int PF, bool BI8, bool LB, int ABL = 0, bool BRF = true, bool DB = false, bool AON = false>
__device__ __forceinline__ void f16x_core('''
new = '''template <int BM, int BN, int WM, int WN, int PF, bool BI8, bool LB, int ABL = 0, bool BRF = true, bool DB = false, bool AON = false, bool W0 = false>
__device__ __forceinline__ void f16x_core('''
assert old in s; s = s.replace(old, new)
# non-DB compute(): wait before the WMMAs
old = '''    auto compute = [&]() {
        #pragma unroll
        for (uint32_t k_inner = 0; k_inner < (uint32_t)BK; k_inner += 16u) {'''
new = '''    auto compute = [&]() {
        if (W0) asm volatile("s_wait_loadcnt 0x0" ::: "memory");
        #pragma unroll
        for (uint32_t k_inner = 0; k_inner < (uint32_t)BK; k_inner += 16u) {'''
assert old in s; s = s.replace(old, new)
# DB loop: wait before the WMMA block
old = '''                #pragma unroll
                for (int ki = 0; ki < 2; ++ki)
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
                        #pragma unroll
                        for (int nt = 0; nt < NTL; ++nt)
                            acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frags[ki][mt], b_frags[ki][nt], acc[mt][nt]);
                sync();'''
new = '''                if (W0) asm volatile("s_wait_loadcnt 0x0" ::: "memory");
                #pragma unroll
                for (int ki = 0; ki < 2; ++ki)
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
                        #pragma unroll
                        for (int nt = 0; nt < NTL; ++nt)
                            acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frags[ki][mt], b_frags[ki][nt], acc[mt][nt]);
                sync();'''
assert old in s; s = s.replace(old, new)
old = '''extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_aon(   // BRF + runtime a_on'''
new = '''extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_w0(   // loads complete before the WMMAs
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
    f16x_core<128, 128, 2, 4, 1, true, true, 0, true, false, false, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_db_w0(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
    f16x_core<128, 128, 2, 4, 2, true, true, 0, true, true, false, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_f16x_pf2_w0(
    float* __restrict__ out, const unsigned char* __restrict__ w, const uint16_t* __restrict__ x16,
    uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks, uint32_t ldx)
{
    f16x_core<128, 128, 2, 4, 2, false, true, 0, true, false, false, true>(out, w, x16, nullptr, nullptr, K, M, n_groups, batch, blocks, ldx);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_f16x_db_w0(
    float* __restrict__ out, const unsigned char* __restrict__ w, const uint16_t* __restrict__ x16,
    uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks, uint32_t ldx)
{
    f16x_core<128, 128, 2, 4, 2, false, true, 0, true, true, false, true>(out, w, x16, nullptr, nullptr, K, M, n_groups, batch, blocks, ldx);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_aon(   // BRF + runtime a_on'''
assert old in s; s = s.replace(old, new)
open(p, 'w').write(s); print('cand ok')
h = open('harness.cpp').read()
old = '''         || name.rfind("r1c_i8x", 0) == 0'''
new = '''         || name.rfind("r1c_i8x", 0) == 0 || name == "q8_0_gemm_wmma_i8x_w0" || name == "q8_0_gemm_wmma_i8x_db_w0"'''
assert old in h; h = h.replace(old, new)
old = '''        || name == "q8_0_gemm_wmma_f16x_db" || name == "q8_0_gemm_wmma_f16x_db_pf1" || name == "q8_0_gemm_wmma_f16x_db_pf3") {'''
new = '''        || name == "q8_0_gemm_wmma_f16x_db" || name == "q8_0_gemm_wmma_f16x_db_pf1" || name == "q8_0_gemm_wmma_f16x_db_pf3"
        || name == "q8_0_gemm_wmma_f16x_pf2_w0" || name == "q8_0_gemm_wmma_f16x_db_w0") {'''
assert old in h; h = h.replace(old, new)
open('harness.cpp', 'w').write(h); print('harness ok')
