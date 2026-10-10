// hc_post — combine sublayer output (block_out) with the HC residual via
// the per-dst post[dst] scale and per-(dst,src) comb mixing matrix.
// Mirrors ds4 hc_post_one (ds4.c:4400):
//
//   for dst in 0..n_hc, d in 0..n_embd:
//     acc = block_out[d] * post[dst]
//     for src in 0..n_hc:
//       acc += comb[dst + src*n_hc] * residual_hc[src*n_embd + d]
//     out_hc[dst*n_embd + d] = acc
//
// One thread per output element. Grid (n_embd/block, n_hc).

#include <hip/hip_runtime.h>

extern "C" __global__ void hc_post(
        float        * __restrict__ out_hc,        // [n_hc, n_embd]
        const float  * __restrict__ block_out,     // [n_embd]
        const float  * __restrict__ residual_hc,   // [n_hc, n_embd]
        const float  * __restrict__ post,          // [n_hc]
        const float  * __restrict__ comb,          // [n_hc, n_hc] addressed [dst + src*n_hc]
        unsigned int   n_embd,
        unsigned int   n_hc) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const unsigned int dst = blockIdx.y;

    float acc = block_out[d] * post[dst];
    for (unsigned int src = 0; src < n_hc; src++) {
        acc += comb[dst + src * n_hc]
             * residual_hc[(unsigned long long)src * n_embd + d];
    }
    out_hc[(unsigned long long)dst * n_embd + d] = acc;
}

// M50 Phase 2: batched hc_post.
// out_hc[B, n_hc, n_embd], block_out[B, n_embd], residual_hc[B, n_hc, n_embd],
// post[B, n_hc], comb[B, n_hc, n_hc]. Grid (n_embd/256, n_hc, B).
extern "C" __global__ void hc_post_batched(
        float        * __restrict__ out_hc,
        const float  * __restrict__ block_out,
        const float  * __restrict__ residual_hc,
        const float  * __restrict__ post,
        const float  * __restrict__ comb,
        unsigned int   n_embd,
        unsigned int   n_hc) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const unsigned int dst = blockIdx.y;
    const unsigned int b = blockIdx.z;

    const float *bo = block_out + (size_t)b * (size_t)n_embd;
    const float *rh = residual_hc + (size_t)b * (size_t)n_hc * (size_t)n_embd;
    const float *pb = post + (size_t)b * (size_t)n_hc;
    const float *cb = comb + (size_t)b * (size_t)n_hc * (size_t)n_hc;
    float       *out = out_hc + (size_t)b * (size_t)n_hc * (size_t)n_embd;

    float acc = bo[d] * pb[dst];
    for (unsigned int src = 0; src < n_hc; src++) {
        acc += cb[dst + src * n_hc] * rh[(size_t)src * (size_t)n_embd + d];
    }
    out[(size_t)dst * (size_t)n_embd + d] = acc;
}

// M50 Phase 2: batched hc_post that reads `post` and `comb` directly from
// the packed split buffer (same layout as the single-token launch_from_split).
// split[B, n_w + n_hc + n_hc*n_hc] with stride = n_w + n_hc + n_hc*n_hc.
// Grid (n_embd/256, n_hc, B).
extern "C" __global__ void hc_post_from_split_batched(
        float        * __restrict__ out_hc,
        const float  * __restrict__ block_out,
        const float  * __restrict__ residual_hc,
        const float  * __restrict__ split,
        unsigned int   n_w,
        unsigned int   n_embd,
        unsigned int   n_hc) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const unsigned int dst = blockIdx.y;
    const unsigned int b = blockIdx.z;

    const size_t split_stride = (size_t)n_w + (size_t)n_hc + (size_t)n_hc * (size_t)n_hc;
    const float *sp = split + (size_t)b * split_stride;
    const float *pb = sp + (size_t)n_w;             // post[n_hc]
    const float *cb = pb + (size_t)n_hc;            // comb[n_hc, n_hc]

    const float *bo = block_out + (size_t)b * (size_t)n_embd;
    const float *rh = residual_hc + (size_t)b * (size_t)n_hc * (size_t)n_embd;
    float       *out = out_hc + (size_t)b * (size_t)n_hc * (size_t)n_embd;

    float acc = bo[d] * pb[dst];
    for (unsigned int src = 0; src < n_hc; src++) {
        acc += cb[dst + src * n_hc] * rh[(size_t)src * (size_t)n_embd + d];
    }
    out[(size_t)dst * (size_t)n_embd + d] = acc;
}

// hc_post_from_split_batched_add2 — dGPU bundle slice 1 (2026-10-10, `V41_DGPU_COMBINE3`): the
// post-MoE `ffn_moe += ffn_shared` vec_add folded in as well. The block value is
// `(block_out[d] + a1[d]) + a2[d]` -- vec_add_inplace(block_out += shared) and then the
// _add kernel's `bv += remote`, the same two f32 adds in the same order -- then the
// unchanged hc_post expression: out_hc BIT-IDENTICAL to vec_add + hc_post_from_split_batched_add.
// block_out is NOT updated. Grid (n_embd/256, n_hc, B), as hc_post_from_split_batched.
extern "C" __global__ void hc_post_from_split_batched_add2(
        float        * __restrict__ out_hc,
        const float  * __restrict__ block_out,
        const float  * __restrict__ a1,
        const float  * __restrict__ a2,
        const float  * __restrict__ residual_hc,
        const float  * __restrict__ split,
        unsigned int   n_w,
        unsigned int   n_embd,
        unsigned int   n_hc) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const unsigned int dst = blockIdx.y;
    const unsigned int b = blockIdx.z;

    const size_t split_stride = (size_t)n_w + (size_t)n_hc + (size_t)n_hc * (size_t)n_hc;
    const float *sp = split + (size_t)b * split_stride;
    const float *pb = sp + (size_t)n_w;             // post[n_hc]
    const float *cb = pb + (size_t)n_hc;            // comb[n_hc, n_hc]

    const float *bo = block_out + (size_t)b * (size_t)n_embd;
    const float *x1 = a1 + (size_t)b * (size_t)n_embd;
    const float *x2 = a2 + (size_t)b * (size_t)n_embd;
    const float *rh = residual_hc + (size_t)b * (size_t)n_hc * (size_t)n_embd;
    float       *out = out_hc + (size_t)b * (size_t)n_hc * (size_t)n_embd;

    float bv = bo[d];
    bv += x1[d];
    bv += x2[d];
    float acc = bv * pb[dst];
    for (unsigned int src = 0; src < n_hc; src++) {
        acc += cb[dst + src * n_hc] * rh[(size_t)src * (size_t)n_embd + d];
    }
    out[(size_t)dst * (size_t)n_embd + d] = acc;
}

// hc_post_from_split_batched_add — 2026-09-27 round 2 (d_fusion): the post-MoE
// `vec_add_inplace(block_out += addend)` (box 2's remote partial) folded into
// hc_post_from_split_batched: the block value is `block_out[d] + addend[d]`, the same
// single f32 add vec_add_inplace performs, then the unchanged hc_post expression, so
// out_hc is BIT-IDENTICAL to vec_add + hc_post (tests/decode_fusion_bitexact.rs).
// block_out is NOT updated (the n_hc WGs of a column all read it; nothing reads it
// after hc_post). Grid (n_embd/256, n_hc, B), as hc_post_from_split_batched.
extern "C" __global__ void hc_post_from_split_batched_add(
        float        * __restrict__ out_hc,
        const float  * __restrict__ block_out,
        const float  * __restrict__ addend,
        const float  * __restrict__ residual_hc,
        const float  * __restrict__ split,
        unsigned int   n_w,
        unsigned int   n_embd,
        unsigned int   n_hc) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const unsigned int dst = blockIdx.y;
    const unsigned int b = blockIdx.z;

    const size_t split_stride = (size_t)n_w + (size_t)n_hc + (size_t)n_hc * (size_t)n_hc;
    const float *sp = split + (size_t)b * split_stride;
    const float *pb = sp + (size_t)n_w;             // post[n_hc]
    const float *cb = pb + (size_t)n_hc;            // comb[n_hc, n_hc]

    const float *bo = block_out + (size_t)b * (size_t)n_embd;
    const float *ad = addend + (size_t)b * (size_t)n_embd;
    const float *rh = residual_hc + (size_t)b * (size_t)n_hc * (size_t)n_embd;
    float       *out = out_hc + (size_t)b * (size_t)n_hc * (size_t)n_embd;

    float bv = bo[d];
    bv += ad[d];
    float acc = bv * pb[dst];
    for (unsigned int src = 0; src < n_hc; src++) {
        acc += cb[dst + src * n_hc] * rh[(size_t)src * (size_t)n_embd + d];
    }
    out[(size_t)dst * (size_t)n_embd + d] = acc;
}
