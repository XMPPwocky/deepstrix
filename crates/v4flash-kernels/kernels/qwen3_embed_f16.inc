// The embed forward's one f32 -> f16 conversion, shared by qwen3_embed.hip (every
// activation cast, `qe_cast_f16`) and qwen3_attn_g4.hip (the attention epilogue's
// direct x16 store), so the two are bit-identical by construction
// (docs/v41/EMBED_PHASE_DESIGN.md §19.7).
//
// Saturates to +-65504 instead of overflowing to inf (Qwen-family activations
// can be large); NaN stays NaN so it surfaces in the output check. RNE.
#pragma once
#include <hip/hip_fp16.h>

__device__ __forceinline__ unsigned short qe_f16(float v) {
    if (!isnan(v)) v = fminf(fmaxf(v, -65504.0f), 65504.0f);
    return __half_as_ushort(__float2half_rn(v));
}
