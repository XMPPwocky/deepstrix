// vec_add — out[i] += rhs[i]. In-place accumulate. Used to combine
// ffn_moe + ffn_shared inside the forward orchestrator.

#include <hip/hip_runtime.h>

extern "C" __global__ void vec_add_inplace(
        float        * __restrict__ out,
        const float  * __restrict__ rhs,
        unsigned int   n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] += rhs[i];
}

// gather_f32 -- out[i] = src[idx[i]] for i < n (a negative idx yields -inf).
// The DSpark exit fetches its top-M candidate logits with it, so only the
// M (id, logit) pairs of each draft position cross to the host.
extern "C" __global__ void gather_f32(
        float        * __restrict__ out,
        const float  * __restrict__ src,
        const int    * __restrict__ idx,
        unsigned int   n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int j = idx[i];
    out[i] = j >= 0 ? src[j] : -INFINITY;
}

// logits_nucleus_cands -- what the host sampler needs for one logit row
// without the 517 KB row itself (multistream decode, `head_cands`): one block
// of LNC_THREADS per row. params[3r..3r+3] = inv_t, lo, band. With
// l = x * inv_t - max * inv_t (the host sampler's tempered log-weight):
//   z    = sum of expf(l) over l >= lo  (the host's survivors)
//   cand = every (x, id) with l >= -band, in any order
// Row r's output at out + r * (4 + 2 * cap) (u32 words):
//   [0] candidate count (> cap: the row is incomplete)
//   [1] max (f32 bits)   [2] z (f32 bits)   [3] 0
//   [4 + 2k], [5 + 2k] = candidate k: raw logit (f32 bits), vocab id
// NaN never survives (fmaxf drops it from the max, every comparison fails).
#define LNC_THREADS 1024u
extern "C" __global__ __launch_bounds__(1024) void logits_nucleus_cands(
        const float  * __restrict__ logits,
        unsigned int   nv,
        const float  * __restrict__ params,
        unsigned int * __restrict__ out,
        unsigned int   cap) {
    __shared__ float red[LNC_THREADS];
    __shared__ unsigned int count;
    const unsigned int tid = threadIdx.x;
    const unsigned int row = blockIdx.x;
    const float *x = logits + (size_t)row * nv;
    unsigned int *o = out + (size_t)row * (4u + 2u * cap);
    const float inv_t = params[3u * row];
    const float lo = params[3u * row + 1u];
    const float band = params[3u * row + 2u];

    float m = -INFINITY;
    for (unsigned int i = tid; i < nv; i += LNC_THREADS) m = fmaxf(m, x[i]);
    red[tid] = m;
    if (tid == 0) count = 0u;
    __syncthreads();
    for (unsigned int s = LNC_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads(); // every thread has read red[0] before it is reused
    const float gmax = m * inv_t;
    float z = 0.0f;
    for (unsigned int i = tid; i < nv; i += LNC_THREADS) {
        const float v = x[i];
        const float l = v * inv_t - gmax;
        if (l >= lo) z += expf(l);
        if (l >= -band) {
            const unsigned int k = atomicAdd(&count, 1u);
            if (k < cap) {
                o[4u + 2u * k] = __float_as_uint(v);
                o[5u + 2u * k] = i;
            }
        }
    }
    red[tid] = z;
    __syncthreads();
    for (unsigned int s = LNC_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        o[0] = count;
        o[1] = __float_as_uint(m);
        o[2] = __float_as_uint(red[0]);
        o[3] = 0u;
    }
}
