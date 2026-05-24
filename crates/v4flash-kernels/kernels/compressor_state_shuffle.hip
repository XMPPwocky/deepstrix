// compressor_state_shuffle — for ratio==4 only. After pool fires, ds4
// slides the current generation (rows 4..7) down to "previous" (rows 0..3),
// then duplicates 0..3 back to 4..7. The duplicate-back step is overwritten
// by the next 4 tokens, so this kernel only does the slide-down copy.
//
//   rows 0..3 := rows 4..7
//
// One thread per (row, col), grid = (4, ceil(width/256), 1), block = (256, 1, 1).

#include <hip/hip_runtime.h>

extern "C" __global__ void compressor_state_shuffle_r4(
        float        * __restrict__ state_kv,
        float        * __restrict__ state_score,
        unsigned int   width) {
    const unsigned int r = blockIdx.x;       // 0..3 (source row 4+r → dest row r)
    const unsigned int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= width) return;
    const unsigned long long src = (unsigned long long)(4u + r) * width + j;
    const unsigned long long dst = (unsigned long long)r * width + j;
    state_kv[dst] = state_kv[src];
    state_score[dst] = state_score[src];
}
