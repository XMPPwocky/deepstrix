// Repack an expert's MXFP4 weights from the HF safetensors layout to the ggml
// block layout, on the GPU.
//
// WHY THIS EXISTS. The expert pager reads a miss straight off the SSD in HF
// layout and the matvec kernels want ggml layout, so something must permute.
// Doing it on the CPU measured **1.50 ms of the 11.5 ms miss** (13%) — it is a
// per-byte nibble shuffle over 6.3 MB, pure scalar work. The iGPU does the same
// shuffle at streaming bandwidth (~6.3 MB read + 6.3 MB write at ~200 GB/s,
// i.e. tens of microseconds). The H2D volume is unchanged: we upload the raw HF
// bytes instead of the repacked ones.
//
// The alternative — teaching every MXFP4 matvec to index the HF layout — would
// avoid the permutation entirely but touches the kernels whose correctness is
// most expensive to disturb. This keeps the on-device layout exactly as the
// matvecs already expect.
//
// LAYOUT
//   in  packed [out, half]   half = in/2, one byte = two 4-bit elements
//   in  scale  [out, nb]     nb = in/32, one E8M0 byte per 32-element block
//   out dst    [out, nb, 17] blk[0] = scale, blk[1..17] = 16 nibble bytes
//
// HF splits each 32-element block into a low half (bytes 0..8) and a high half
// (bytes 8..16); ggml wants the two interleaved pairwise. Mirrors the scalar
// loop this replaces in `hf_v41.rs::read_expert_raw`.

#include <hip/hip_runtime.h>

extern "C" __global__ void mxfp4_repack_hf_to_ggml(
        unsigned char       * __restrict__ dst,     // [out * nb * 17]
        const unsigned char * __restrict__ packed,  // [out * half]
        const unsigned char * __restrict__ scale,   // [out * nb]
        unsigned int          out_rows,
        unsigned int          nb) {                 // blocks per row
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = out_rows * nb;
    if (gid >= total) return;

    const unsigned int r = gid / nb;   // output row
    const unsigned int j = gid - r * nb; // block within the row
    const unsigned int half = nb * 16u;  // bytes per row in `packed`

    const unsigned char* pb = packed + (size_t)r * (size_t)half + (size_t)j * 16u;
    unsigned char*       blk = dst    + ((size_t)r * (size_t)nb + (size_t)j) * 17u;

    blk[0] = scale[(size_t)r * (size_t)nb + (size_t)j];
    #pragma unroll
    for (unsigned int i = 0; i < 8u; i++) {
        const unsigned char lo = pb[i];
        const unsigned char hi = pb[8u + i];
        blk[1u + 2u * i] = (unsigned char)((lo & 0x0Fu) | (hi << 4));
        blk[2u + 2u * i] = (unsigned char)((lo >> 4)    | (hi & 0xF0u));
    }
}
