#!/usr/bin/env python3
# Patch 8: bisect the r1 i8x gap ON THE r1 SOURCE. The f16x core of cand_r1.hip (commit aab5a67) is
# appended inside namespace r1c with two single-change flags:
#   UNC : B loads unconditional (clamped row) + select at stage (patch6 style)   [r1 = guarded loads]
#   AONC: a_on compile-time true                                                  [r1 = runtime]
# Kernels: r1c_i8x (= r1), r1c_i8x_unc, r1c_i8x_aonc, r1c_i8x_both.
import re
src = open('cand_r1.hip').read()
start = src.index('// LDS-only workgroup barrier')
end = src.index('// ---------------------------------------------------------------------------------------------\n// Replay twin')
core = src[start:end]
# drop the r1 kernel instantiations (keep template + helpers)
core = core[:core.index('#define F16X_KERNEL')]
core = core.replace('__device__ __forceinline__ void sync_lds()', '__device__ __forceinline__ void sync_lds_r1()')
core = core.replace('auto sync = [&]() { if (LB) sync_lds(); else __syncthreads(); };', 'auto sync = [&]() { if (LB) sync_lds_r1(); else __syncthreads(); };')
core = core.replace('template <int BM, int BN, int WM, int WN, int PF, bool BI8, bool LB>\n__device__ __forceinline__ void f16x_core(',
                    'template <int BM, int BN, int WM, int WN, int PF, bool BI8, bool LB, bool UNC, bool AONC>\n__device__ __forceinline__ void f16x_core_r1(')
assert 'f16x_core_r1(' in core
old = '        a_on[i] = slot < A_SLOTS;'
assert old in core
core = core.replace(old, '        a_on[i] = AONC ? true : (slot < A_SLOTS);')
old = '''        if (BI8) {
            b_src[i] = xq + (size_t)(n0 + b_row[i]) * (size_t)ldx + (size_t)g * K;
            bs_src[i] = xscale + (size_t)(n0 + b_row[i]) * (size_t)(ldx / 32u) + (size_t)g * blocks;
        } else {
            b_src[i] = x16 + (size_t)(n0 + b_row[i]) * (size_t)ldx + (size_t)g * K;
            bs_src[i] = nullptr;
        }'''
assert old in core
core = core.replace(old, '''        const uint32_t brow = (!UNC || (n0 + b_row[i]) < batch) ? (n0 + b_row[i]) : (batch - 1u);
        if (BI8) {
            b_src[i] = xq + (size_t)brow * (size_t)ldx + (size_t)g * K;
            bs_src[i] = xscale + (size_t)brow * (size_t)(ldx / 32u) + (size_t)g * blocks;
        } else {
            b_src[i] = x16 + (size_t)brow * (size_t)ldx + (size_t)g * K;
            bs_src[i] = nullptr;
        }''')
old = '''            vb[p][i] = make_uint4(0u, 0u, 0u, 0u);
            if (BI8) xs[p][i] = 0.f;
            if (b_on[i]) {'''
assert old in core
core = core.replace(old, '''            vb[p][i] = make_uint4(0u, 0u, 0u, 0u);
            if (BI8) xs[p][i] = 0.f;
            if (UNC || b_on[i]) {''')
old = '''            if (BI8) {
                // (f16)((float)q * xscale), the lds_tiled expression, 16 values -> 2 x b128
                int8_t q[16]; __builtin_memcpy(q, &vb[p][i], 16);
                const float sc = xs[p][i];'''
assert old in core
core = core.replace(old, '''            uint4 vraw = vb[p][i];
            if (UNC && !b_on[i]) vraw = make_uint4(0u, 0u, 0u, 0u);
            if (BI8) {
                // (f16)((float)q * xscale), the lds_tiled expression, 16 values -> 2 x b128
                int8_t q[16]; __builtin_memcpy(q, &vraw, 16);
                const float sc = (UNC && !b_on[i]) ? 0.f : xs[p][i];''')
old = '''                uint4* dst = (uint4*)(&B_tile[b_row[i] * LDS_STRIDE + b_koff[i]]);
                dst[0] = vb[p][i];'''
assert old in core
core = core.replace(old, '''                uint4* dst = (uint4*)(&B_tile[b_row[i] * LDS_STRIDE + b_koff[i]]);
                dst[0] = vraw;''')
block = '''
// =============================================================================================
// r1 core bisect (patch8): the first attempt's f16x_core verbatim + UNC / AONC single-change flags.
// =============================================================================================
namespace r1c {
''' + core + '''
}  // namespace r1c
#define R1C_I8X(NAME, UNC, AONC)                                                                      \\
extern "C" __global__ void __launch_bounds__(256) NAME(                                              \\
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,      \\
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks) \\
{ r1c::f16x_core_r1<128, 128, 2, 4, 1, true, true, UNC, AONC>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K); }
#if defined(__gfx1200__) || defined(__gfx1201__)
R1C_I8X(r1c_i8x,      false, false)
R1C_I8X(r1c_i8x_unc,  true,  false)
R1C_I8X(r1c_i8x_aonc, false, true)
R1C_I8X(r1c_i8x_both, true,  true)
#endif
'''
s = open('cand.hip').read()
if 'namespace r1c' not in s:
    s += block
open('cand.hip', 'w').write(s); print('cand ok')
h = open('harness.cpp').read()
old = '''         || name == "q8_0_gemm_wmma_i8x_aon" || name == "q8_0_gemm_wmma_i8x_grd_aon" || name == "q8_0_gemm_wmma_i8x_db_aon"'''
assert old in h
h = h.replace(old, old + '''
         || name.rfind("r1c_i8x", 0) == 0''')
open('harness.cpp', 'w').write(h); print('harness ok')
