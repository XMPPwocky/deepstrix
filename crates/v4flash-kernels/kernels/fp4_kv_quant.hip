// fp4_kv_quant — DeepSeek-V4.1 compressed-KV fake quantisation
// (kernel.py `fp4_act_quant(x, block_size=16, inplace=True,
// scale_dtype=float8_e4m3fn)`), applied to the post-RoPE 512-d latent
// before it enters the compressed KV cache (model.py `Attention._compress_kv`):
//
//   per 16-element block:  amax  = max(max|x|, 6 · 2^-9)
//                          scale = e4m3fn_nearest(amax / 6)        (RNE, as torch's cast)
//                          x     = e2m1_nearest(clamp(x / scale, ±6)) · scale
//
// The reference stores the dequantised values in its bf16 cache; every
// product is a ≤3-bit-mantissa E2M1 value times an E4M3 scale, exact in f16
// and bf16, so the engine's f16 comp_kv store holds exactly the reference's
// cache after this kernel (a packed 288 B/row store is a later memory item).
//
// Launch: grid (n_rows, 1, 1), block (512, 1, 1) — one thread per element,
// block amax via 16-lane shuffles (a 16-block is half a wave32).

#include <hip/hip_runtime.h>

#include "e2m1_key_common.inc"
#include "fp8_e4m3fn_common.inc"

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

extern "C" __global__ void fp4_kv_quant_inplace(
        float        * __restrict__ x,      // [n_rows, 512]
        unsigned int   n_rows,
        unsigned int   width) {             // 512
    const unsigned int row = blockIdx.x;
    const unsigned int j = threadIdx.x;
    if (row >= n_rows || j >= width) return;
    float* xr = x + (unsigned long long)row * width;
    const float v = xr[j];
    // amax over this lane's 16-element block (lanes 16k..16k+15 of the wave).
    float a = fabsf(v);
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 1));
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 2));
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 4));
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 8));
    a = fmaxf(a, 6.0f * 0.001953125f);                       // 6 · 2^-9
    const float scale = c_e4m3fn_values[e4m3fn_nearest_index(a / 6.0f)];
    float q = v / scale;
    q = fminf(fmaxf(q, -6.0f), 6.0f);
    xr[j] = e2m1fn_dequant(q) * scale;
}

// ---------------------------------------------------------------------------
// fp8_act_quant_inplace — DeepSeek-V4.1 window-KV fake quantisation
// (kernel.py `act_quant(kv, 32, "ue8m0", inplace=True)`, model.py
// `Attention._window_kv`): applied to the whole post-RoPE 512-d window row,
// RoPE tail included, before it enters the window cache:
//
//   per 32-element block:  amax  = max(max|x|, 1e-4)
//                          scale = 2^ceil(log2(amax / 448))        (E8M0)
//                          x     = e4m3fn_nearest(clamp(x / scale, ±448)) · scale
//
// Products are an E4M3 value times a power of two: exact in f16, so the
// engine's f16 window cache then holds exactly the reference's bf16 cache.
// Launch: grid (n_rows), block (width) — one thread per element, block amax
// via 32-lane shuffles (a 32-block is one wave32).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp8_ue8m0_scale(float amax_in) {
    const float amax = fmaxf(amax_in, 1.0e-4f);
    // 2^ceil(log2(amax/448)), exactly: frexpf gives amax/448 = m·2^e, m ∈ [0.5, 1).
    const float t = amax / 448.0f;
    int e;
    const float m = frexpf(t, &e);
    const int cexp = (m == 0.5f) ? e - 1 : e;
    return ldexpf(1.0f, cexp);
}

extern "C" __global__ void fp8_act_quant_inplace(
        float        * __restrict__ x,      // [n_rows, width]
        unsigned int   n_rows,
        unsigned int   width) {
    const unsigned int row = blockIdx.x;
    const unsigned int j = threadIdx.x;
    if (row >= n_rows || j >= width) return;
    float* xr = x + (unsigned long long)row * width;
    const float v = xr[j];
    float a = fabsf(v);
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 1));
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 2));
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 4));
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 8));
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 16));
    const float scale = fp8_ue8m0_scale(a);
    float q = v / scale;
    q = fminf(fmaxf(q, -448.0f), 448.0f);
    xr[j] = dsv4_e4m3fn_dequant(q) * scale;
}
