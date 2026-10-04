// The arena stage-graph CONTEXT (docs/v41/GRAPH_KEYS_DESIGN.md 2.1): a pointer-only device
// struct holding every per-layer / per-lane operand a captured stage reads. The host writes the
// entry for (lane, layer) into ONE fixed device slot before the stage runs (`arena_ctx_store`,
// stream-ordered); `_ind` kernel twins read their operands from it, so one graph per (stage,
// rows) serves every layer and lane. Layout mirrors `het::arena_ctx::ArenaCtx` (Rust, #[repr(C)]).
#pragma once
#include <stdint.h>

#define ARENA_CTX_SLOTS 32

// The canary log (design 2.8): thread 0 of block 0 of every `_ind` launch with a non-null canary
// appends (seq << 16 | tag) at an atomic cursor; records past `cap` are counted, not written.
struct ArenaCanaryLog {
    uint32_t cursor;
    uint32_t cap;
    uint64_t rec[1];  // [cap]
};

struct ArenaCtx {
    const void* p[ARENA_CTX_SLOTS];
    uint64_t seq;          // stamped by the host per write (the canary, design 2.8)
    ArenaCanaryLog* log;   // where `_ind` launches log the seq they read
};
#define ARENA_CTX_WORDS (ARENA_CTX_SLOTS + 2)

// Operand i of an `_ind` twin (design 2.3): when bit i of `mask` is set, the operand's own
// pointer argument is the ADDRESS of its context slot (process-static, safe to bake into a
// graph) and the prologue dereferences it. Thread-uniform, so the loads stay scalar. Resolve every
// operand before any store.
#define ARENA_CTX_DEREF(mask, i, T, ptr)                                                   \
    do {                                                                                   \
        if (((mask) >> (i)) & 1u) {                                                        \
            (ptr) = *(T const*)(ptr);                                                      \
        }                                                                                  \
    } while (0)

// The canary (design 2.8), after the operands resolved: `canary` is the context slot's own address
// (null in production -- one uniform branch, compiled into every twin), read in the same prologue
// load sequence as the operands.
#define ARENA_CTX_CANARY(canary, tag)                                                      \
    do {                                                                                   \
        if ((canary) != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && \
            threadIdx.x == 0) {                                                            \
            const uint64_t seq_ = (canary)->seq;                                           \
            ArenaCanaryLog* const log_ = (canary)->log;                                    \
            if (log_ != nullptr) {                                                         \
                const uint32_t i_ = atomicAdd(&log_->cursor, 1u);                          \
                if (i_ < log_->cap) log_->rec[i_] = (seq_ << 16) | (uint64_t)((tag) & 0xffffu); \
            }                                                                              \
        }                                                                                  \
    } while (0)
