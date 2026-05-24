// q8_k_quantize — pack an F32 activation row into Q8_K blocks of 256 elements.
// Mirrors ds4_quantize_row_q8_K (ds4.c:1655):
//   amax = max|x[j]|; max = signed value at the amax position
//   iscale = -127 / max
//   qs[j] = clamp(lrintf(iscale * x[j]), -128, 127)
//   bsums[j] = sum of qs[j*16 .. j*16+15]  (16 partials per superblock)
//   d = 1 / iscale
//
// Output layout matches block_q8_K (292 bytes per block):
//   float d (4) + int8_t qs[256] (256) + int16_t bsums[16] (32)
//
// Grid: one workgroup per QK_K=256 block. Block: 256 threads.
// Each thread owns one element; reductions via shared memory.

#include <hip/hip_runtime.h>

#define QK_K 256
#define BLOCK_BYTES 292
#define D_OFFSET 0          // float
#define QS_OFFSET 4         // int8_t[256]
#define BSUMS_OFFSET 260    // int16_t[16]

extern "C" __global__ void q8_k_quantize(
        unsigned char * __restrict__ out,   // n_blocks * 292 bytes
        const float   * __restrict__ x,     // n_blocks * 256 floats
        unsigned int    n_blocks) {
    __shared__ float s_max[256];     // per-element signed value of element with largest |x|
    __shared__ float s_amax[256];    // per-element |x|
    __shared__ int   s_bsum[16];     // per-group i32 partial sums

    const unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned int tid = threadIdx.x;

    const float v = x[b * QK_K + tid];
    s_amax[tid] = fabsf(v);
    s_max[tid] = v;
    __syncthreads();

    // Reduce: find the index of the maximum |x|, capture its signed value.
    // Pairwise reduction over 256 → 1.
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float a = s_amax[tid];
            const float b2 = s_amax[tid + stride];
            if (b2 > a) {
                s_amax[tid] = b2;
                s_max[tid] = s_max[tid + stride];
            }
        }
        __syncthreads();
    }

    const float amax = s_amax[0];
    const float mxv = s_max[0];

    unsigned char *blk = out + (unsigned long long)b * BLOCK_BYTES;

    if (amax == 0.0f) {
        if (tid == 0) {
            *((float *)(blk + D_OFFSET)) = 0.0f;
        }
        ((int8_t *)(blk + QS_OFFSET))[tid] = 0;
        if (tid < 16) {
            ((int16_t *)(blk + BSUMS_OFFSET))[tid] = 0;
        }
        return;
    }

    const float iscale = -127.0f / mxv;
    int qv = (int)rintf(iscale * v);
    if (qv > 127) qv = 127;
    if (qv < -128) qv = -128;
    ((int8_t *)(blk + QS_OFFSET))[tid] = (int8_t)qv;

    // bsums: per-16-element groups. Use shared memory partials.
    const unsigned int g = tid >> 4;          // 0..15
    const unsigned int lane = tid & 15;
    if (lane == 0) s_bsum[g] = 0;
    __syncthreads();
    atomicAdd(&s_bsum[g], qv);
    __syncthreads();

    if (tid < 16) {
        ((int16_t *)(blk + BSUMS_OFFSET))[tid] = (int16_t)s_bsum[tid];
    }
    if (tid == 0) {
        *((float *)(blk + D_OFFSET)) = 1.0f / iscale;
    }
}
