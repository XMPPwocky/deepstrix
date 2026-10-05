// The arena stage-graph CONTEXT (docs/v41/GRAPH_KEYS_DESIGN.md 2.1): a pointer-only device
// struct holding every per-layer / per-lane operand a captured stage reads. The host writes the
// entry for (lane, layer) into ONE fixed device slot before the stage runs (`arena_ctx_store`,
// stream-ordered); `_ind` kernel twins read their operands from it, so one graph per (stage,
// rows) serves every layer and lane. Layout mirrors `het::arena_ctx::ArenaCtx` (Rust, #[repr(C)]).
#pragma once
#include <stdint.h>

#define ARENA_CTX_SLOTS 32

// The canary log (design 2.8): thread 0 of block 0 of every `_ind` launch with a non-null canary
// appends {seq << 16 | tag, XOR of its resolved operand pointers} at an atomic cursor; records
// past `cap` are counted, not written.
struct ArenaCanaryLog {
    uint32_t cursor;
    uint32_t cap;
    uint64_t rec[];   // [2 * cap] (flexible array member)
};

struct ArenaCtx {
    const void* p[ARENA_CTX_SLOTS];
    uint64_t seq;          // stamped by the host per write (the canary, design 2.8)
    ArenaCanaryLog* log;   // where `_ind` launches log what they read
};
#define ARENA_CTX_WORDS (ARENA_CTX_SLOTS + 2)
// The layout `het::arena_ctx` (Rust, #[repr(C)]) and `arena_ctx_store`'s by-value argument assume.
static_assert(sizeof(ArenaCtx) == 8 * ARENA_CTX_WORDS, "ArenaCtx layout");
static_assert(__builtin_offsetof(ArenaCtx, seq) == 8 * ARENA_CTX_SLOTS, "ArenaCtx::seq offset");
static_assert(__builtin_offsetof(ArenaCtx, log) == 8 * ARENA_CTX_SLOTS + 8, "ArenaCtx::log offset");
static_assert(__builtin_offsetof(ArenaCanaryLog, rec) == 8, "ArenaCanaryLog::rec offset");

// Operand i of an `_ind` twin (design 2.3): when bit i of `mask` is set, the operand's own
// pointer argument is the ADDRESS of its context slot (process-static, safe to bake into a
// graph) and the prologue dereferences it. Thread-uniform, so the loads stay scalar. Resolve every
// operand before any store.
// A pointer read from memory has no known address space, so the body's loads and stores through
// it would compile to FLAT instructions (Step 0, 2026-10-04: every global_load of the tB / mhc
// bodies became flat_load, waits merged into s_wait_loadcnt_dscnt, gemv b=8 +8%). Every operand is
// device global memory: the slot is read AS an address_space(1) pointer, so both incoming values of
// the resolved pointer are casts from global and InferAddressSpaces gives the body the direct
// kernel's global_* instructions. (Neither a generic -> global -> generic cast round trip, folded
// away as a no-op pair, nor __builtin_assume(!is_shared / !is_private) survived to the backend.)
template <typename P> struct ArenaCtxGlobal;
template <typename U> struct ArenaCtxGlobal<U*> { typedef __attribute__((address_space(1))) U* type; };
#define ARENA_CTX_DEREF(mask, i, T, ptr)                                                   \
    do {                                                                                   \
        if (((mask) >> (i)) & 1u) {                                                        \
            (ptr) = (T)(*(typename ArenaCtxGlobal<T>::type const*)(ptr));                  \
        }                                                                                  \
    } while (0)

// The canary (design 2.8), in two halves.
// ARENA_CTX_CANARY_LOAD, right after the operands resolved: `canary` (the context slot's own address,
// a kernel argument; null in production -- one uniform branch compiled into every twin) is read with
// scalar loads in the same prologue sequence as the operand dereferences, before any store, along
// with `xorv` (the XOR of the resolved operand pointers: what this launch really computes on).
// ARENA_CTX_CANARY_STORE, after the body (so no store precedes the body's loads): thread 0 of
// block 0 appends {seq << 16 | tag, xorv}.
#define ARENA_CTX_CANARY_LOAD(canary, xorv)                                                \
    uint64_t arena_canary_seq_ = 0, arena_canary_x_ = 0;                                   \
    ArenaCanaryLog* arena_canary_log_ = nullptr;                                           \
    if ((canary) != nullptr) {                                                             \
        arena_canary_seq_ = (canary)->seq;                                                 \
        arena_canary_log_ = (canary)->log;                                                 \
        arena_canary_x_ = (xorv);                                                          \
    }

#define ARENA_CTX_CANARY_STORE(tag)                                                        \
    do {                                                                                   \
        if (arena_canary_log_ != nullptr && blockIdx.x == 0 && blockIdx.y == 0 &&          \
            blockIdx.z == 0 && threadIdx.x == 0) {                                         \
            const uint32_t i_ = atomicAdd(&arena_canary_log_->cursor, 1u);                 \
            if (i_ < arena_canary_log_->cap) {                                             \
                arena_canary_log_->rec[2 * i_] =                                           \
                    (arena_canary_seq_ << 16) | (uint64_t)((tag) & 0xffffu);               \
                arena_canary_log_->rec[2 * i_ + 1] = arena_canary_x_;                      \
            }                                                                              \
        }                                                                                  \
    } while (0)

// The XOR of resolved operand pointers ARENA_CTX_CANARY_LOAD records.
#define ARENA_CTX_U64(ptr) ((uint64_t)(uintptr_t)(ptr))

// A context slot holding two f32 VALUES (low word, high word) rather than a pointer: the per-layer
// rope arguments ride in the entry itself (three slots), so a rope twin reads them in the same
// dependent round as its pointer operands (a pointer to a per-layer struct would add a round).
// `slotp` is the slot's address (a kernel argument), read as global memory (scalar load).
#define ARENA_CTX_F32X2(slotp, lo, hi)                                                     \
    do {                                                                                   \
        const uint64_t v_ = *(const __attribute__((address_space(1))) uint64_t*)(slotp);   \
        (lo) = __uint_as_float((uint32_t)v_);                                              \
        (hi) = __uint_as_float((uint32_t)(v_ >> 32));                                      \
    } while (0)
