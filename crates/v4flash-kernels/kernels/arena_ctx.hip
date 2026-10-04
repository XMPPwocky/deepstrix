// `arena_ctx_store`: write one ArenaCtx entry (passed BY VALUE as the kernel argument) into the
// fixed device slot the `_ind` kernels read (docs/v41/GRAPH_KEYS_DESIGN.md 2.2). One workgroup:
// stream-ordered on the stages' own HW queue, nothing for the host to keep alive.
#include <hip/hip_runtime.h>
#include "arena_ctx.inc"

extern "C" __global__ void arena_ctx_store(ArenaCtx entry, ArenaCtx* __restrict__ dst) {
    const uint32_t t = threadIdx.x;
    if (t < ARENA_CTX_SLOTS) dst->p[t] = entry.p[t];
    if (t == 0) dst->seq = entry.seq;
    if (t == 1) dst->log = entry.log;
}

// A do-nothing kernel: the empty-launch baseline of the Step 0 measurements.
extern "C" __global__ void arena_ctx_nop(uint32_t* __restrict__ sink, uint32_t v) {
    if (threadIdx.x == 0 && v == 0xffffffffu) sink[0] = v;
}
