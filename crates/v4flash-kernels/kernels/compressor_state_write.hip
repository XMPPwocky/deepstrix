// compressor_state_write — combined APE add + state row write. Mirrors
// the post-matvec / pre-pool logic in ds4's compressor_decode_one (ds4.c:6625-6630).
//
//   state_kv[row, j]    = kv_cur[j]
//   state_score[row, j] = sc_cur[j] + f32(ape[pos_mod, j])
//
// Note APE is row-major [comp_width, ratio] in the GGUF (dim[0]=comp_width
// is innermost), so the indexing `ape[pos_mod * comp_width + j]` reads the
// value at (pos_mod-th column, j-th row in the 2D notation) — matching
// ds4's `tensor_2d_value(model, ape, j, pos_mod)`.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void compressor_state_write(
        float        * __restrict__ state_kv,
        float        * __restrict__ state_score,
        const float  * __restrict__ kv_cur,
        const float  * __restrict__ sc_cur,
        const __half * __restrict__ ape,
        unsigned int   width,
        unsigned int   row,
        unsigned int   pos_mod) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width) return;
    const unsigned long long stride = (unsigned long long)row * width;
    state_kv[stride + j] = kv_cur[j];
    const float ape_val = __half2float(ape[(unsigned long long)pos_mod * width + j]);
    state_score[stride + j] = sc_cur[j] + ape_val;
}

// M50 batched: per-batch row + pos_mod arrays. State is shared (single
// state_kv/state_score per layer); each b writes to its own (row, pos_mod)
// slot. Caller must guarantee no two batches collide on the same row.
// kv_cur / sc_cur are [B, width].
extern "C" __global__ void compressor_state_write_batched(
        float        * __restrict__ state_kv,
        float        * __restrict__ state_score,
        const float  * __restrict__ kv_cur,        // [B, width]
        const float  * __restrict__ sc_cur,        // [B, width]
        const __half * __restrict__ ape,
        const int    * __restrict__ row_per_b,     // [B]
        const int    * __restrict__ pos_mod_per_b, // [B]
        unsigned int   width) {
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    if (j >= width) return;
    const int row     = row_per_b[b];
    const int pos_mod = pos_mod_per_b[b];
    const unsigned long long stride       = (unsigned long long)row * width;
    const unsigned long long batch_stride = (unsigned long long)b   * width;
    state_kv[stride + j] = kv_cur[batch_stride + j];
    const float ape_val = __half2float(ape[(unsigned long long)pos_mod * width + j]);
    state_score[stride + j] = sc_cur[batch_stride + j] + ape_val;
}
