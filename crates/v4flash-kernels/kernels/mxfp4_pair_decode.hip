// mxfp4_pair_decode -- MXFP4 gate+up fused-SwiGLU for the DECODE regime
// (`V41_MOE_PAIR_DEC=1`, default off), written from scratch to replace
// `mxfp4_pair_matvec_fused_swiglu_kwide` (kernels/mxfp4_pair_matvec.hip) at
// rows <= 8. Same arguments (+ `n_tok`, `ticket`), the same work-item /
// device-count contract and outputs, BIT-IDENTICAL `mid` (tests/mxfp4_pair_decode.rs).
//
// ---------------------------------------------------------------------------
// THE REGIME. A decode lane-layer has b = 1..8 token rows; each row picks 6
// experts and box 1 computes ~half of them, so a launch covers D ~ 4..20 DISTINCT
// experts with 1..3 members each (a row picks an expert at most once, so members
// <= b). Every lane-layer touches different experts: the weights are COLD.
//
// ROOFLINE (all measured on this gfx1151 beside live production, quiet-period
// medians of 60 interleaved rounds, every call on fresh cold experts, kwide A/A
// spread <= 0.6%; tests/mxfp4_pair_decode.rs `bench_dec_decode_regime`, 2026-10-13):
//   bytes per call     D x 2 x 6,266,880 B (2304 rows x 20 super-blocks x 136 B,
//                      gate + up); activations / outputs < 0.1% of it.
//                      D = 4: 50.1 MB, D = 8: 100.3 MB, D = 16: 200.5 MB.
//   ceiling            contiguous streaming read (ceil_read_v4, 200 MB cold):
//                      237 GB/s. The SAME bytes read by an ideal streaming
//                      kernel in one launch (`mxfp4_pair_dec_probe_read`):
//                      225 / 440 / 870 us at D = 4 / 8 / 16 = 222-231 GB/s.
//                      That probe is the floor quoted below.
//   b  D  members  bytes    probe   kwide (% of probe)   this (% of probe)  speedup
//   1  4    4      50 MB    225 us  261 us (86%)         240 us (93%)       1.09x
//   2  8    8     100 MB    440     517    (85%)         465    (95%)       1.11x
//   4  8   12     100 MB    439     518    (85%)         467    (94%)       1.11x
//   4 16   16     201 MB    870    1016    (86%)         912    (95%)       1.11x
//   8 16   24     201 MB    869    1038    (84%)         918    (95%)       1.13x
//   4  4   12      50 MB    226     285    (79%)         248    (91%)       1.15x
//   8  8   24     100 MB    439     566    (78%)         477    (92%)       1.19x
//   8  4   24      50 MB    226     349    (65%)         264    (86%)       1.32x
// (production decode, estimated from the routing: 2 lanes x 4 rows, ~12 box-1 picks
// over ~8-12 experts per lane-layer -- the 1-1.5 members-per-expert rows above.)
// Where the remaining ~5% goes (sub-staged in the same bench):
//   * ~3%: DRAM efficiency of the access pattern, NOT the instructions. The
//     kernel with its arithmetic removed (`mxfp4_pair_dec_loadonly`) runs at
//     ~97% of the probe (899 vs 870 us at D = 16). The lane-level pattern is not
//     the cause (contiguous 544 B per step, 16 B-aligned loads, no scale-byte
//     loads, burst issue, a 9-deep ring, 20-80 work-groups: all within 1%); the
//     active DRAM footprint is: one row pair per wave was 1.5% slower than one
//     row per wave, and the probe itself drops to this kernel's speed (911 us)
//     when its grid reads all 32 matrices at once instead of ~2 at a time.
//   * ~1-2%: arithmetic not fully hidden. The instruction stream alone (no memory
//     traffic, `mxfp4_pair_dec_noload`) costs 17-27% of the memory time at 1-1.5
//     members per expert, ~50% at 3, 83% at 6 -- why the shared-expert shapes sit
//     lower. Per weight it is already minimal for bit identity: one v_perm LUT
//     decode per 4 weights (6 VALU), the int8 dot, kwide's per-16-weight f32 fma.
//   * ~5-10 us fixed per launch: the activation staging and the scalar
//     work-item chain before the first weight load, and the drain at the end.
//
// ---------------------------------------------------------------------------
// WHY kwide CANNOT GET THERE (it was written for prefill chunks of 32 members).
// Per super-block pair it issues one 16 B + one 1 B load per matrix per lane,
// waits for them (vmcnt(0)), decodes through a 16-byte LDS LUT (32 ds_load_u8),
// re-stages the members' activations from L2 and takes two work-group barriers:
// each wave has one super-block pair (544 B) in flight per round trip. Its grid
// (n_rows / 8, work-item bound) also launches a work-group per (8 rows, bound
// slot): at b = 8 the device-count bound is 48 slots for 4-16 real items, and
// the dead work-groups drain behind the live ones.
//
// THIS KERNEL.
//   * PERSISTENT grid: one 16-wave work-group per CU, whatever the work-item
//     count -- no dead work-groups for the device-count bound. The waves stride
//     over every (work item, row) unit of the launch: wave w takes unit w first,
//     then the next free unit from a self-resetting ticket counter (`ticket`,
//     one per work-item buffer; V41_MOE_PAIR_DEC_DYN=0 or nullptr = the static
//     stride w + W, w + 2W, ...: 2% slower at D = 16, it waits for the slowest
//     CU and leaves a one-unit tail).
//   * Activations staged in LDS ONCE per work-group (all n_tok token rows, q8 +
//     d, 41.6 KB at 8 rows), with the first weight loads already in flight.
//   * A barrier-free software pipeline: a ring of P = 5 single-step slots (one
//     step = one super-block pair of gate and up: 4 loads per lane); step t + 4
//     is issued before step t is computed, continuing into the next unit, so
//     every wave always has 4 steps in flight. (MLP is not the limiter: a 9-deep
//     ring and half the waves with it measured the same.)
//   * kwide's lane layout, one row per wave: lane (sbh, b32, half) owns 16
//     weights of block b32 of super-block 2*sbp + sbh. (Two rows per wave, 16
//     lanes per row and no duplicated 16 B loads, is `mxfp4_pair_dec_2row`:
//     bit-identical too, 1-3% slower -- twice the active footprint per wave.)
//   * Nibble decode in registers with v_perm_b32 (two 8-entry byte tables picked
//     per byte by bit 3): 6 VALU per 4 weights, no LDS LUT.
//   * The member loop is runtime (`mi < nm`, wave-uniform) over MAXM = 8
//     accumulator pairs; the step loop is two rolled bodies of 5 unrolled steps.
//     (Fully unrolled 10 steps x 8 member blocks was ~40 KB of code: units with
//     3+ members ran out of the instruction cache, 25% slower at 6 members.)
//
// Bit identity. kwide's output for (member, row) is: 32 lane partials
// p[sbh][b32][half], each the f32 sum over sbp = 0..9 IN ORDER of
// fma((float)sumi, yd * ds, p) with sumi the int32 dot of 16 weights x 16 q8
// (exact in any order), then warp_sum (shfl_down 16, 8, 4, 2, 1), then the
// clamp / SwiGLU / weight epilogue. This kernel keeps the same lanes, the same
// fma sequence per lane, the same shuffle tree and the same epilogue source; only
// where the loads come from and when they are issued changed. (The 2-row
// reference keeps p[..][0] and p[..][1] in one lane and reduces 8, 4, 2, 1 then
// adds the halves -- the same pairs in the same tree order.)
//
// Compiler notes (each measured): an empty `asm volatile` with a memory clobber
// that reads the accumulators ends every step, or the optimiser issues all steps'
// loads first and sinks the arithmetic (every step live: 256 VGPRs, spills);
// the per-unit row offsets go through an opaque asm, or loop strength reduction
// makes each per-step address its own 64-bit induction variable (spills).
//
// Contract: n_blocks == 20 (V4.1 gate/up); n_rows even; n_tok (= rows) <= 8 and
// every member's token row < n_tok; a work item with more than 8 members takes
// several passes over its rows (weights re-read; never at rows <= 8).

#include <hip/hip_runtime.h>
#include <stdint.h>

#define MXDEC_NB 20u                    // Q8_K super-blocks per row (5120 / 256)
#define MXDEC_NBP (MXDEC_NB / 2u)       // super-block pairs per row = pipeline steps per unit
#define MXDEC_SUPER 136u                // bytes per MXFP4 super-block (v2 layout)
#define MXDEC_Q8K 292u                  // bytes per Q8_K block
#define MXDEC_P 5u                      // ring slots: P - 1 steps in flight while one computes
#define MXDEC_MAXM 8u                   // members per pass held in registers (= max token rows)
#define MXDEC_MAXTOK 8u                 // token rows staged in LDS (the launcher's rows <= 8)
#define MXDEC_FULL_MASK 0xFFFFFFFFFFFFFFFFull

static_assert(MXDEC_NBP % MXDEC_P == 0, "a unit must fill the ring a whole number of times");

__device__ __forceinline__ int32_t mxdec_sudot4(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

// 2^(e-128), identical to e8m0_half_to_f32_pair (mxfp4_pair_matvec.hip).
__device__ __forceinline__ float mxdec_e8m0(uint32_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : ((e - 1u) << 23);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

// Four nibble indices (bits 0..3 of each byte of n; bits 4..7 ignored) -> four
// int8 kvalues_mxfp4 {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}. v_perm_b32
// selector bytes 0..7 pick a byte of {S0:S1}: the positive and the negative
// 8-entry tables are looked up with idx & 7, then each byte picks one of the two
// by bit 3 (selector i or i + 4). 6 VALU per 4 weights, no LDS.
__device__ __forceinline__ uint32_t mxdec_lut4(uint32_t n) {
    const uint32_t i7  = n & 0x07070707u;
    const uint32_t pos = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, i7);
    const uint32_t neg = __builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, i7);
    const uint32_t sel = ((n >> 1) & 0x04040404u) | 0x03020100u;
    return __builtin_amdgcn_perm(neg, pos, sel);
}

// kwide's warp_sum_f32_pair verbatim.
__device__ __forceinline__ float mxdec_warp_sum32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(MXDEC_FULL_MASK, v, offset);
    }
    return v;
}

// kwide's warp_sum_f32_pair minus its offset-16 and offset-1 steps (2-row reference).
__device__ __forceinline__ float mxdec_warp_sum16(float v) {
    v += __shfl_down_sync(MXDEC_FULL_MASK, v, 8);
    v += __shfl_down_sync(MXDEC_FULL_MASK, v, 4);
    v += __shfl_down_sync(MXDEC_FULL_MASK, v, 2);
    v += __shfl_down_sync(MXDEC_FULL_MASK, v, 1);
    return v;
}

// One pipeline step of one lane: one super-block pair x (gate, up) x (16 nibble
// bytes of block b32 of super-block 2*sbp + sbh, + its scale byte).
struct MxdecSlot {
    uint4    g;
    uint4    u;
    uint32_t gs;
    uint32_t us;
};

// Per-lane offsets inside the expert (32-bit; the expert bases are uniform, so
// every load is saddr + voffset + immediate): nibbles at row*2720 + sbh*136 +
// b32*16 (8 B aligned only when sbh = 1: the b128 load is unaligned-safe on AMD,
// as kwide's mxfp4_load16; 16 B-aligned loads measured no faster), scale byte at
// row*2720 + sbh*136 + 128 + b32. Step sbp adds sbp * 272.
__device__ __forceinline__ void mxdec_load(MxdecSlot& s, const uint8_t* gate_e, const uint8_t* up_e,
                                           unsigned int on, unsigned int os, unsigned int sbp) {
    const unsigned int o = sbp * (2u * MXDEC_SUPER);
    __builtin_memcpy(&s.g, gate_e + (size_t)(on + o), 16);
    __builtin_memcpy(&s.u, up_e   + (size_t)(on + o), 16);
    s.gs = gate_e[(size_t)(os + o)];
    s.us = up_e[(size_t)(os + o)];
}

// What a variant does with a step's data. 0 = the kernel; 1 = LOAD_ONLY (bench:
// fold the loaded bits into a sink, no arithmetic); 2 = NOLOAD (bench: no loads
// after the prologue, the arithmetic on the same registers every step).
#define MXDEC_RUN 0u
#define MXDEC_LOAD_ONLY 1u
#define MXDEC_NOLOAD 2u

// One step, one row per wave (kwide's lanes: sbh = lane >> 4, b32, half): this
// lane's 16 gate + 16 up weights (one nibble half of its block), one q8 b128 per
// member; kwide's `lane_pacc += (float)sumi * (yd * ds)` into ag / au.
template <unsigned int MAXM, unsigned int MODE>
__device__ __forceinline__ void mxdec_compute1(
        const MxdecSlot& s, unsigned int t, unsigned int nm, unsigned int half_sh,
        const uint32_t* const (&q8_m)[MAXM], const float* const (&yd_m)[MAXM],
        float (&ag)[MAXM], float (&au)[MAXM], uint32_t& sink) {
    if (MODE == MXDEC_LOAD_ONLY) {
        sink ^= s.g.x ^ s.g.y ^ s.g.z ^ s.g.w ^ s.gs ^ s.u.x ^ s.u.y ^ s.u.z ^ s.u.w ^ s.us;
        return;
    }
    const int32_t g0 = (int32_t)mxdec_lut4(s.g.x >> half_sh);
    const int32_t g1 = (int32_t)mxdec_lut4(s.g.y >> half_sh);
    const int32_t g2 = (int32_t)mxdec_lut4(s.g.z >> half_sh);
    const int32_t g3 = (int32_t)mxdec_lut4(s.g.w >> half_sh);
    const int32_t u0 = (int32_t)mxdec_lut4(s.u.x >> half_sh);
    const int32_t u1 = (int32_t)mxdec_lut4(s.u.y >> half_sh);
    const int32_t u2 = (int32_t)mxdec_lut4(s.u.z >> half_sh);
    const int32_t u3 = (int32_t)mxdec_lut4(s.u.w >> half_sh);
    const float gds = mxdec_e8m0(s.gs);
    const float uds = mxdec_e8m0(s.us);
    #pragma unroll
    for (unsigned int mi = 0; mi < MAXM; mi++) {
        if (mi < nm) {   // wave-uniform
            const uint4 q = *(const uint4*)(q8_m[mi] + 2u * t * 64u);
            const float yd = yd_m[mi][2u * t];
            int32_t sg = 0, su = 0;
            sg = mxdec_sudot4(g0, (int32_t)q.x, sg);
            sg = mxdec_sudot4(g1, (int32_t)q.y, sg);
            sg = mxdec_sudot4(g2, (int32_t)q.z, sg);
            sg = mxdec_sudot4(g3, (int32_t)q.w, sg);
            su = mxdec_sudot4(u0, (int32_t)q.x, su);
            su = mxdec_sudot4(u1, (int32_t)q.y, su);
            su = mxdec_sudot4(u2, (int32_t)q.z, su);
            su = mxdec_sudot4(u3, (int32_t)q.w, su);
            ag[mi] = __builtin_fmaf((float)sg, yd * gds, ag[mi]);
            au[mi] = __builtin_fmaf((float)su, yd * uds, au[mi]);
        }
    }
}

// One step, two rows per wave (the 2-row reference: 16 lanes per row, lane (sbh,
// b32) decodes BOTH nibble halves of its block, accumulators per half).
template <unsigned int MAXM, unsigned int MODE>
__device__ __forceinline__ void mxdec_compute2(
        const MxdecSlot& s, unsigned int t, unsigned int nm,
        const uint32_t* const (&q8_m)[MAXM], const float* const (&yd_m)[MAXM],
        float (&ag0)[MAXM], float (&ag1)[MAXM],
        float (&au0)[MAXM], float (&au1)[MAXM], uint32_t& sink) {
    if (MODE == MXDEC_LOAD_ONLY) {
        sink ^= s.g.x ^ s.g.y ^ s.g.z ^ s.g.w ^ s.gs ^ s.u.x ^ s.u.y ^ s.u.z ^ s.u.w ^ s.us;
        return;
    }
    const uint32_t gq[4] = {s.g.x, s.g.y, s.g.z, s.g.w};
    const uint32_t uq[4] = {s.u.x, s.u.y, s.u.z, s.u.w};
    int32_t gl[4], gh[4], ul[4], uh[4];
    #pragma unroll
    for (unsigned int i = 0; i < 4u; i++) {
        gl[i] = (int32_t)mxdec_lut4(gq[i]);
        gh[i] = (int32_t)mxdec_lut4(gq[i] >> 4);
        ul[i] = (int32_t)mxdec_lut4(uq[i]);
        uh[i] = (int32_t)mxdec_lut4(uq[i] >> 4);
    }
    const float gds = mxdec_e8m0(s.gs);
    const float uds = mxdec_e8m0(s.us);
    #pragma unroll
    for (unsigned int mi = 0; mi < MAXM; mi++) {
        if (mi < nm) {   // wave-uniform
            // q8 elements [32*b32, +32): the low 16 pair with the low nibbles
            // (block elements 0..15), the high 16 with the high ones.
            const uint4* a = (const uint4*)(q8_m[mi] + 2u * t * 64u);
            const uint4 a0 = a[0];
            const uint4 a1 = a[1];
            const float yd = yd_m[mi][2u * t];
            int32_t sg0 = 0, sg1 = 0, su0 = 0, su1 = 0;
            sg0 = mxdec_sudot4(gl[0], (int32_t)a0.x, sg0);
            sg0 = mxdec_sudot4(gl[1], (int32_t)a0.y, sg0);
            sg0 = mxdec_sudot4(gl[2], (int32_t)a0.z, sg0);
            sg0 = mxdec_sudot4(gl[3], (int32_t)a0.w, sg0);
            sg1 = mxdec_sudot4(gh[0], (int32_t)a1.x, sg1);
            sg1 = mxdec_sudot4(gh[1], (int32_t)a1.y, sg1);
            sg1 = mxdec_sudot4(gh[2], (int32_t)a1.z, sg1);
            sg1 = mxdec_sudot4(gh[3], (int32_t)a1.w, sg1);
            su0 = mxdec_sudot4(ul[0], (int32_t)a0.x, su0);
            su0 = mxdec_sudot4(ul[1], (int32_t)a0.y, su0);
            su0 = mxdec_sudot4(ul[2], (int32_t)a0.z, su0);
            su0 = mxdec_sudot4(ul[3], (int32_t)a0.w, su0);
            su1 = mxdec_sudot4(uh[0], (int32_t)a1.x, su1);
            su1 = mxdec_sudot4(uh[1], (int32_t)a1.y, su1);
            su1 = mxdec_sudot4(uh[2], (int32_t)a1.z, su1);
            su1 = mxdec_sudot4(uh[3], (int32_t)a1.w, su1);
            const float tg = yd * gds;
            const float tu = yd * uds;
            ag0[mi] = __builtin_fmaf((float)sg0, tg, ag0[mi]);
            ag1[mi] = __builtin_fmaf((float)sg1, tg, ag1[mi]);
            au0[mi] = __builtin_fmaf((float)su0, tu, au0[mi]);
            au1[mi] = __builtin_fmaf((float)su1, tu, au1[mi]);
        }
    }
}

// Stage the first n_tok token rows of xq into LDS: q8 as [tok][sb][64 dwords],
// d as [tok][sb]. Loads go out in batches of 16 independent dwords per thread
// before any store (a load-store loop waits on every load: one L2 round trip per
// dword, which at nx-many work-groups per item was the first version's
// bottleneck at b = 8).
#define MXDEC_STAGE_BATCH 16u
__device__ __forceinline__ void mxdec_stage(uint32_t* s_q8, float* s_yd, const uint8_t* xq, unsigned int n_tok) {
    const unsigned int tid = threadIdx.x;
    const unsigned int per_tok = MXDEC_NB * 65u;
    const unsigned int total = n_tok * per_tok;
    const unsigned int xq_dw = MXDEC_NB * (MXDEC_Q8K / 4u);   // dwords per token row
    for (unsigned int base = 0; base < total; base += MXDEC_STAGE_BATCH * blockDim.x) {
        uint32_t v[MXDEC_STAGE_BATCH];
        #pragma unroll
        for (unsigned int j = 0; j < MXDEC_STAGE_BATCH; j++) {
            const unsigned int i = base + j * blockDim.x + tid;
            if (i < total) {
                const unsigned int tok = i / per_tok;
                const unsigned int r   = i - tok * per_tok;   // sb * 65 + k
                const unsigned int sb  = r / 65u;
                const unsigned int k   = r - sb * 65u;        // 0 = d, 1..64 = q8
                v[j] = ((const uint32_t*)xq)[tok * xq_dw + sb * (MXDEC_Q8K / 4u) + k];
            }
        }
        #pragma unroll
        for (unsigned int j = 0; j < MXDEC_STAGE_BATCH; j++) {
            const unsigned int i = base + j * blockDim.x + tid;
            if (i < total) {
                const unsigned int tok = i / per_tok;
                const unsigned int r   = i - tok * per_tok;
                const unsigned int sb  = r / 65u;
                const unsigned int k   = r - sb * 65u;
                if (k == 0) s_yd[tok * MXDEC_NB + sb] = __uint_as_float(v[j]);
                else        s_q8[(tok * MXDEC_NB + sb) * 64u + (k - 1u)] = v[j];
            }
        }
    }
}

// Ticket counter of the dynamic unit order: [0] = units taken past the static
// first round, [1] = waves retired. Wave-uniform results.
__device__ __forceinline__ unsigned int mxdec_grab(unsigned int* ticket, bool lane0) {
    unsigned int k = 0;
    if (lane0) k = atomicAdd(&ticket[0], 1u);
    return __builtin_amdgcn_readfirstlane(k);
}
__device__ __forceinline__ void mxdec_retire(unsigned int* ticket, unsigned int n_waves, bool lane0) {
    if (ticket == nullptr || !lane0) return;
    // Every grab of this wave has returned (its value was used) before this.
    if (atomicAdd(&ticket[1], 1u) == n_waves - 1u) {
        atomicExch(&ticket[0], 0u);
        atomicExch(&ticket[1], 0u);
    }
}

// A unit of work: one row (ROW1) or row pair of one work item, one pass of <=
// MAXM of its members. Every field is wave-uniform (SGPRs).
struct MxdecUnit {
    unsigned int u;            // unit index = item * units_per_item + r
    unsigned int pass;         // first member of this pass
    unsigned int end;          // one past the item's last member
    const uint8_t* gate_e;
    const uint8_t* up_e;
    const int* em;             // the expert's member list
    unsigned int row0;         // first row of the unit
};

__device__ __forceinline__ MxdecUnit mxdec_resolve(
        unsigned int u, unsigned int units_per_item, unsigned int rows_per_unit,
        const int* work_items, const int* group_count, const int* expert_members,
        const uint8_t* gate_w_base, const uint8_t* up_w_base,
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int max_per_expert, unsigned int chunk_size) {
    MxdecUnit r;
    const unsigned int item = u / units_per_item;
    const unsigned int ru   = u - item * units_per_item;
    const int packed = work_items[item];
    const unsigned int e     = (unsigned int)(packed >> 16) & 0xffffu;
    const unsigned int start = (unsigned int)packed & 0xffffu;
    const unsigned int gn    = (unsigned int)max(group_count[e], 0);
    r.u = u;
    r.pass = start;
    r.end = min(gn, start + chunk_size);          // <= start: an empty item, nothing written
    r.gate_e = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe;
    r.up_e   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe;
    r.em = expert_members + (size_t)e * (size_t)max_per_expert;
    r.row0 = rows_per_unit * ru;
    return r;
}

// The kernel. ROW1: one row per wave (production); else two (the reference).
template <unsigned int MAXM, bool ROW1, unsigned int MODE>
__device__ __forceinline__ void mxdec_persistent(
        float        * __restrict__ mid,
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,
        const float  * __restrict__ expert_w,
        const int    * __restrict__ group_count,
        const int    * __restrict__ expert_members,
        const int    * __restrict__ work_items,
        unsigned int    n_work_items,
        unsigned int    gate_bpe,
        unsigned int    up_bpe,
        unsigned int    n_used,
        unsigned int    max_per_expert,
        unsigned int    chunk_size,
        float           clamp,
        unsigned int    n_rows,
        const int    * __restrict__ n_work_items_dev,
        unsigned int    n_tok,
        unsigned int * __restrict__ ticket,
        uint32_t     * __restrict__ sink_out) {
    __shared__ uint32_t s_q8[MXDEC_MAXTOK * MXDEC_NB * 64u];   // 40 KiB
    __shared__ float    s_yd[MXDEC_MAXTOK * MXDEC_NB];
    constexpr unsigned int ROWS_PER_UNIT = ROW1 ? 1u : 2u;
    constexpr unsigned int P = MXDEC_P;

    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const bool lane0 = (lane == 0u);
    // ROW1 lanes (kwide's): sbh = lane >> 4, b32 = (lane & 15) >> 1, half = lane & 1.
    // 2-row lanes: rsel = lane >> 4 (row of the pair), sbh = (lane >> 3) & 1, b32 = lane & 7.
    const unsigned int rsel = ROW1 ? 0u : lane >> 4u;
    const unsigned int sbh  = ROW1 ? lane >> 4u : (lane >> 3u) & 1u;
    const unsigned int b32  = ROW1 ? (lane & 15u) >> 1u : lane & 7u;
    const unsigned int half_sh = (lane & 1u) * 4u;   // ROW1 only
    const unsigned int row_bytes = MXDEC_NB * MXDEC_SUPER;
    const unsigned int lane_n = rsel * row_bytes + sbh * MXDEC_SUPER + b32 * 16u;
    const unsigned int lane_s = rsel * row_bytes + sbh * MXDEC_SUPER + 128u + b32;
    const unsigned int upi = n_rows / ROWS_PER_UNIT;            // units per work item
    const unsigned int waves = blockDim.x >> 5u;
    const unsigned int wave = __builtin_amdgcn_readfirstlane(tid >> 5u);
    unsigned int n_wi = n_work_items;
    if (n_work_items_dev != nullptr) n_wi = min(n_wi, (unsigned int)max(*n_work_items_dev, 0));
    const unsigned int total = n_wi * upi;
    const unsigned int stride = gridDim.x * waves;   // = every wave of the grid
    uint32_t sink = 0;

    MxdecUnit cur = mxdec_resolve(blockIdx.x * waves + wave, upi, ROWS_PER_UNIT, work_items, group_count,
                                  expert_members, gate_w_base, up_w_base, gate_bpe, up_bpe, max_per_expert,
                                  chunk_size);
    const bool has_work = cur.u < total;
    MxdecSlot ring[P];
    // The first unit's first steps go out BEFORE the activation staging, so the
    // staging's L2 round trips hide under the weights' DRAM latency.
    if (has_work) {
        #pragma unroll
        for (unsigned int t = 0; t + 1u < P; t++) {
            mxdec_load(ring[t], cur.gate_e, cur.up_e, cur.row0 * row_bytes + lane_n, cur.row0 * row_bytes + lane_s, t);
        }
    }
    mxdec_stage(s_q8, s_yd, xq, n_tok);
    __syncthreads();
    if (!has_work) {
        mxdec_retire(ticket, stride, lane0);
        return;
    }

    const uint32_t* q8_lane = ROW1 ? s_q8 + sbh * 64u + (lane & 15u) * 4u : s_q8 + sbh * 64u + b32 * 8u;
    const float*    yd_lane = s_yd + sbh;
    for (;;) {
        // The unit after this one: another pass over the same rows when the item
        // has more members than a pass holds, else the next unit (stride or
        // ticket). When there is none, the tail re-loads this unit (cache hits,
        // unused): unconditional loads keep the vmcnt accounting static.
        MxdecUnit nxt;
        bool more;
        if (cur.pass + MAXM < cur.end) {
            nxt = cur;
            nxt.pass = cur.pass + MAXM;
            more = true;
        } else {
            const unsigned int un = (ticket != nullptr) ? stride + mxdec_grab(ticket, lane0) : cur.u + stride;
            more = un < total;
            nxt = more ? mxdec_resolve(un, upi, ROWS_PER_UNIT, work_items, group_count, expert_members,
                                       gate_w_base, up_w_base, gate_bpe, up_bpe, max_per_expert, chunk_size)
                       : cur;
        }
        unsigned int on   = cur.row0 * row_bytes + lane_n;
        unsigned int os   = cur.row0 * row_bytes + lane_s;
        unsigned int on_n = nxt.row0 * row_bytes + lane_n;
        unsigned int os_n = nxt.row0 * row_bytes + lane_s;
        // Opaque to the optimiser: otherwise loop strength reduction turns each
        // per-step address into its own 64-bit induction variable (spills).
        asm volatile("" : "+v"(on), "+v"(os), "+v"(on_n), "+v"(os_n));

        const unsigned int nm = (cur.end > cur.pass) ? min(MAXM, cur.end - cur.pass) : 0u;
        const uint32_t* q8_m[MAXM];
        const float*    yd_m[MAXM];
        #pragma unroll
        for (unsigned int mi = 0; mi < MAXM; mi++) {
            const unsigned int tok = (mi < nm) ? (unsigned int)(cur.em[cur.pass + mi] >> 16) : 0u;
            q8_m[mi] = q8_lane + tok * (MXDEC_NB * 64u);
            yd_m[mi] = yd_lane + tok * MXDEC_NB;
        }
        float ag0[MAXM], ag1[MAXM], au0[MAXM], au1[MAXM];
        #pragma unroll
        for (unsigned int mi = 0; mi < MAXM; mi++) {
            ag0[mi] = 0.0f; ag1[mi] = 0.0f; au0[mi] = 0.0f; au1[mi] = 0.0f;
        }
        // The 10 steps as a ROLLED loop of 2 bodies of P = 5 unrolled steps: the
        // ring index (t % P) stays static and the hot code fits the instruction
        // cache.
        #pragma unroll 1
        for (unsigned int h = 0; h < MXDEC_NBP / P; h++) {
          #pragma unroll
          for (unsigned int k = 0; k < P; k++) {
            const unsigned int t  = h * P + k;
            const unsigned int tl = t + P - 1u;            // step to issue now
            const bool in_next = tl >= MXDEC_NBP;          // wave-uniform
            if (MODE != MXDEC_NOLOAD) {
                mxdec_load(ring[(k + P - 1u) % P],
                           in_next ? nxt.gate_e : cur.gate_e, in_next ? nxt.up_e : cur.up_e,
                           in_next ? on_n : on, in_next ? os_n : os, in_next ? tl - MXDEC_NBP : tl);
            } else if (k == 0) {
                // Compute-only probe: the ring's registers stay "fresh" to the
                // compiler without any memory traffic.
                asm volatile("" : "+v"(ring[0].g.x), "+v"(ring[1].g.x), "+v"(ring[2].g.x), "+v"(ring[3].g.x),
                                  "+v"(ring[4].u.x));
            }
            if constexpr (ROW1) mxdec_compute1<MAXM, MODE>(ring[k], t, nm, half_sh, q8_m, yd_m, ag0, au0, sink);
            else mxdec_compute2<MAXM, MODE>(ring[k], t, nm, q8_m, yd_m, ag0, ag1, au0, au1, sink);
            // Keep the ring a ring: the step's arithmetic finishes here (the asm
            // reads the accumulators) and no later load is hoisted above it
            // (memory clobber).
            if (MODE == MXDEC_LOAD_ONLY) {
                asm volatile("" : "+v"(sink) : : "memory");
            } else {
                #pragma unroll
                for (unsigned int mi = 0; mi < MAXM; mi++) {
                    if constexpr (ROW1) asm volatile("" : "+v"(ag0[mi]), "+v"(au0[mi]) : : "memory");
                    else asm volatile("" : "+v"(ag0[mi]), "+v"(ag1[mi]), "+v"(au0[mi]), "+v"(au1[mi]) : : "memory");
                }
            }
          }
        }
        if (MODE != MXDEC_LOAD_ONLY) {
            #pragma unroll
            for (unsigned int mi = 0; mi < MAXM; mi++) {
                if (mi < nm) {
                    float g_sum, u_sum;
                    bool writer;
                    if constexpr (ROW1) {
                        g_sum = mxdec_warp_sum32(ag0[mi]);
                        u_sum = mxdec_warp_sum32(au0[mi]);
                        writer = lane0;
                    } else {
                        // kwide's offset-1 step is the sum of the two halves.
                        const float g0 = mxdec_warp_sum16(ag0[mi]);
                        const float g1 = mxdec_warp_sum16(ag1[mi]);
                        const float u0 = mxdec_warp_sum16(au0[mi]);
                        const float u1 = mxdec_warp_sum16(au1[mi]);
                        g_sum = g0 + g1;
                        u_sum = u0 + u1;
                        writer = (lane & 15u) == 0u;
                    }
                    if (writer) {
                        // kwide's epilogue verbatim.
                        const int packed        = cur.em[cur.pass + mi];
                        const unsigned int b    = (unsigned int)(packed >> 16);
                        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
                        if (clamp > 1.0e-6f) {
                            if (g_sum > clamp)  g_sum = clamp;
                            if (u_sum > clamp)  u_sum = clamp;
                            if (u_sum < -clamp) u_sum = -clamp;
                        }
                        const float ew  = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
                        const float sig = 1.0f / (1.0f + expf(-g_sum));
                        const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot)
                                             * (size_t)n_rows + (size_t)(cur.row0 + rsel);
                        mid[out_off] = g_sum * sig * u_sum * ew;
                    }
                }
            }
        }
        if (!more) break;
        cur = nxt;
    }
    mxdec_retire(ticket, stride, lane0);
    if (MODE == MXDEC_LOAD_ONLY && sink == 0x9e3779b9u) sink_out[0] = sink;
}

#define MXDEC_ARGS                                                               \
        float        * __restrict__ mid,           /* [B, n_used, n_rows] */      \
        const uint8_t * __restrict__ gate_w_base,                                 \
        const uint8_t * __restrict__ up_w_base,                                   \
        const uint8_t * __restrict__ xq,           /* [B, 20 * 292] */           \
        const float  * __restrict__ expert_w,      /* [B, n_used] */             \
        const int    * __restrict__ group_count,   /* [n_expert] */              \
        const int    * __restrict__ expert_members,/* [n_expert * max_per_expert] */ \
        const int    * __restrict__ work_items,    /* [n_work_items] */          \
        unsigned int    n_work_items,              /* exact count, or the bound */ \
        unsigned int    gate_bpe,                                                 \
        unsigned int    up_bpe,                                                   \
        unsigned int    n_used,                                                   \
        unsigned int    max_per_expert,                                           \
        unsigned int    chunk_size,                                               \
        float           clamp,                                                    \
        unsigned int    n_rows,                                                   \
        const int    * __restrict__ n_work_items_dev, /* nullptr = n_work_items is exact */ \
        unsigned int    n_tok,                     /* token rows of xq (<= 8; members' rows < it) */ \
        unsigned int * __restrict__ ticket         /* [2] zeroed, self-resetting; nullptr = static order */

#define MXDEC_PASS mid, gate_w_base, up_w_base, xq, expert_w, group_count, expert_members, \
                   work_items, n_work_items, gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, \
                   clamp, n_rows, n_work_items_dev, n_tok, ticket

// The kernel: persistent grid (the launcher: one 16-wave work-group per CU).
extern "C" __global__ void __launch_bounds__(512) mxfp4_pair_matvec_fused_swiglu_dec(MXDEC_ARGS) {
    mxdec_persistent<MXDEC_MAXM, true, MXDEC_RUN>(MXDEC_PASS, nullptr);
}

// Reference (bench / test): two rows per wave, no duplicated 16 B loads;
// bit-identical, 1-3% slower in the decode regime (twice the active footprint).
extern "C" __global__ void __launch_bounds__(512) mxfp4_pair_dec_2row(MXDEC_ARGS) {
    mxdec_persistent<MXDEC_MAXM, false, MXDEC_RUN>(MXDEC_PASS, nullptr);
}

// Test only: two members per pass, so a work item of 3..8 members takes the
// multi-pass path (production's rows <= 8 never does with 8 per pass).
extern "C" __global__ void __launch_bounds__(512) mxfp4_pair_dec_m2_test(MXDEC_ARGS) {
    mxdec_persistent<2, true, MXDEC_RUN>(MXDEC_PASS, nullptr);
}
extern "C" __global__ void __launch_bounds__(512) mxfp4_pair_dec_2row_m2_test(MXDEC_ARGS) {
    mxdec_persistent<2, false, MXDEC_RUN>(MXDEC_PASS, nullptr);
}

// Sub-stages (bench only): the kernel's loads with no arithmetic, and its
// arithmetic with no loads (same grid, staging, pipeline and instruction order).
extern "C" __global__ void __launch_bounds__(512) mxfp4_pair_dec_loadonly(MXDEC_ARGS, uint32_t* __restrict__ sink) {
    mxdec_persistent<MXDEC_MAXM, true, MXDEC_LOAD_ONLY>(MXDEC_PASS, sink);
}
extern "C" __global__ void __launch_bounds__(512) mxfp4_pair_dec_noload(MXDEC_ARGS) {
    mxdec_persistent<MXDEC_MAXM, true, MXDEC_NOLOAD>(MXDEC_PASS, nullptr);
}

// ROOFLINE PROBE (bench only): read exactly the bytes a launch must move -- the
// gate and up matrices (bpe bytes each) of every work item's expert -- with the
// best streaming pattern (16 B per lane, 4 loads in flight per lane, grid-stride
// inside one matrix), one launch, grid (nx, n_work_items, 2). Its time is the
// floor for the call: same bytes, same addresses, no arithmetic.
extern "C" __global__ void __launch_bounds__(256)
mxfp4_pair_dec_probe_read(
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const int    * __restrict__ work_items,
        unsigned int    bpe,
        uint32_t     * __restrict__ sink) {
    const unsigned int e = (unsigned int)(work_items[blockIdx.y] >> 16) & 0xffffu;
    const uint4* p = (const uint4*)((blockIdx.z == 0 ? gate_w_base : up_w_base) + (uint64_t)e * (uint64_t)bpe);
    const unsigned int n = bpe / 16u;
    const unsigned int stride = gridDim.x * blockDim.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (; i + 3u * stride < n; i += 4u * stride) {
        const uint4 a = p[i];
        const uint4 b = p[i + stride];
        const uint4 c = p[i + 2u * stride];
        const uint4 d = p[i + 3u * stride];
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w;
        acc ^= c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n; i += stride) {
        const uint4 a = p[i];
        acc ^= a.x ^ a.y ^ a.z ^ a.w;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;
}
