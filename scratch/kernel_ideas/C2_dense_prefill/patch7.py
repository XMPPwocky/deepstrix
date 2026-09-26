#!/usr/bin/env python3
# Patch 7: AON flag (runtime a_on predicate as in the r1 code) to bisect the r1-vs-current i8x gap.
p='cand.hip'; s=open(p).read()
old='''template <int BM, int BN, int WM, int WN, int PF, bool BI8, bool LB, int ABL = 0, bool BRF = true, bool DB = false>
__device__ __forceinline__ void f16x_core('''
new='''template <int BM, int BN, int WM, int WN, int PF, bool BI8, bool LB, int ABL = 0, bool BRF = true, bool DB = false, bool AON = false>
__device__ __forceinline__ void f16x_core('''
assert old in s; s=s.replace(old,new)
old='''        a_on[i] = (A_SLOTS % NT_ == 0) ? true : (slot < A_SLOTS);   // compile-time when it can be'''
new='''        a_on[i] = (AON || A_SLOTS % NT_ != 0) ? (slot < A_SLOTS) : true;   // AON: runtime predicate (r1 style)'''
assert old in s; s=s.replace(old,new)
old='''extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_grd(   // r1 style: guarded loads'''
new='''extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_aon(   // BRF + runtime a_on
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
    f16x_core<128, 128, 2, 4, 1, true, true, 0, true, false, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_grd_aon(   // = r1 code path
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
    f16x_core<128, 128, 2, 4, 1, true, true, 0, false, false, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_db_aon(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
    f16x_core<128, 128, 2, 4, 2, true, true, 0, true, true, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_grd(   // r1 style: guarded loads'''
assert old in s; s=s.replace(old,new)
open(p,'w').write(s); print("cand ok")
p='harness.cpp'; s=open(p).read()
old='''    if ((name == "q8_0_gemm_wmma_i8x" || name == "i8x_r1" || name == "q8_0_gemm_wmma_i8x_pf2" || name == "q8_0_gemm_wmma_i8x_pf3" || name == "q8_0_gemm_wmma_i8x_grd"'''
new='''    if ((name == "q8_0_gemm_wmma_i8x" || name == "i8x_r1" || name == "q8_0_gemm_wmma_i8x_pf2" || name == "q8_0_gemm_wmma_i8x_pf3" || name == "q8_0_gemm_wmma_i8x_grd"
         || name == "q8_0_gemm_wmma_i8x_aon" || name == "q8_0_gemm_wmma_i8x_grd_aon" || name == "q8_0_gemm_wmma_i8x_db_aon"'''
assert old in s; s=s.replace(old,new)
open(p,'w').write(s); print("harness ok")
