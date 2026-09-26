# A_moe_smallb — routed MoE at small batch (decode), iGPU gfx1151

Owner notes, written as work proceeds. Everything measured here is via `_infra/gpu_run.sh --dev igpu`.

## Production facts (read from the tree at 361d4f9)

Shapes (S/config.rs): N_EMBD 5120, N_FF_EXP 2304, 384 experts, top-6. BLOCKS_Q8K_GATE_IN = 20,
BLOCKS_Q8K_DOWN_IN = 9, SWIGLU_CLAMP_EXP = 10.0, CHUNK_SIZE = 32 (RE:76, FP:8523).
gate_bpe = up_bpe = 2304 x 20 x 136 = 6,266,880 B; down_bpe = 5120 x 9 x 136 = 6,266,880 B.
One expert = 18.80 MB. Q8_K block = 292 B; xq per token = 20 x 292 = 5840 B; midq per slot = 9 x 292 = 2628 B.

Box-2 batched chain (`batched_pass`, RE:4389-4497), b >= 2 (REQ_FLAG_BATCHED), mode 0, cap 6, all in one
stream, NOT graph-captured:
 1. memset group_count[gbound]              (gbound = n_resident slots; harness uses 384)
 2. moe_group_builder_hetsplit  grid ceil(6b/512) x 512
 3. memset n_work_items[1]
 4. moe_work_items_builder      grid ceil(gbound/256) x 256, chunk 32
 5. mxfp4_pair_matvec_fused_swiglu_kwide  grid (288, n_wi_bound) x 256, n_wi_bound = min(6b, min(6b,gbound)+ceil(6b/32)) = 6b at b<=8
 6. q8_k_quantize               grid (9*6*b) x 256
 7. memset partials[b*6*5120] f32 (first pass only)
 8. mxfp4_matvec_par_by_expert_kwide2  grid (320, n_wi_bound) x 256
 9. q2_k_reduce_partials_hetsplit  grid ceil(5120b/256) x 256
Sentinel picks: d_selected = 384 (SENTINEL_EXPERT), remap[384] = 0 (dGPU "takes" it, rank < cap), so mode-0
kernels skip it. Owned picks: remap[id] = -(slot+1); the group id = pool slot.
Hub (box 1) per lane b=1..5 runs the same kernels 5-9 with its own remap (pager slots), graph-captured.

b=1 twin (box 2 decode branch, RE:4252-4275): pair_hetsplit (288, 6) x 256 -> q8k (54) x 256 ->
down_hetsplit (640) x 256.

## Regime

Weights are cold per token. Harness rotates the selection over a pool of P=16 expert copies (301 MB > 32 MB
MALL); consecutive reuse of a copy is >= 225 MB apart. Activations warm (as in production).

## Log

### Step 1 baseline (run1, graph mode, inner=8 rounds=40, P=16 rotating, E=4 experts = 75.2 MB/call, roof 351 us @214 GB/s)
Chain = production batched_pass (builders+kwide+q8k+memset+kwide2+reduce). twin = hetsplit b=1 chain per row.
    b   chain med/p10/p90 (us)    GB/s   %roof   twin x b rows
    1   446 / 338 / 505           169    79%     360 (b=1 twin = 209 GB/s, 98%)
    2   593 / 497 / 670           127    59%     743
    4   715 / 578 / 784           105    49%     1430
    8   872 / 706 / 941            86    40%     2880
Bytes are identical at every b; the chain costs ~+60 us per extra row.
Parts at b=4 E=4 (each alone, graph, rotating): builders 12.8 | kwide 343 (146 GB/s, 68%; exact grid 361 - no win
from an exact grid.y) | q8k 12.0 | partials memset 5.0 | kwide2 233 (108 GB/s, 50%; exact 235) | reduce 6.6 |
twin pair_hetsplit(E=4) 254 (198 GB/s) | twin down_hetsplit(E=4) 106 (236 GB/s).
=> gate/up is 1.35x the twin at equal bytes, down is 2.2x the twin. Small kernels total ~37 us (5%).
ISA: kwide 119 VGPR / 70 SGPR; kwide2 173 VGPR (8 waves/SIMD cap); pair_hetsplit 80 VGPR LDS 9360; down_hetsplit 101 VGPR.
kwide ISA has v_movrels/v_movreld (dynamic register indexing of lane_pacc_g/u[mi] in the runtime-n_members loop) and
32 ds_load_u8 (LDS LUT) per pair; kwide2: 128 ds_load_u8 + 96 v_dot4 per superblock, 16 global_load_b128 (q8 loads DID
merge to b128 despite the 4-B-aligned 292-B block stride).
Harness sanity: batched chain vs per-row hetsplit twin agree to rel_rmse 1e-7 (different summation order; not bit-exact, as
production's RE:2107 comment says).

### ATT (b=4 E=4, dispatch 3-4 of the chain, results/att_kwide*_b4.txt)
kwide gate/up: stall/latency 46.6%; 63% of all stall on ONE `s_waitcnt vmcnt(3)` (the pair's weight loads), 5% vmcnt(0),
~5% lgkmcnt (LDS LUT / staging). The loop is: issue 2x(2 uint2 + u8) weight loads -> stage xq to LDS (global_load_b32 +
vmcnt(0) + ds_store, serialised) -> barrier -> dot (v_movrels/movreld per member: dynamic VGPR indexing) -> barrier.
Only one pair-iteration (34 B/lane = 1.1 KB/wave) of DRAM loads is ever in flight per wave; the barriers stop the compiler
from hoisting the next pair's loads. => memory-level-parallelism bound, not bandwidth bound.
kwide2 down: stall/latency 78%; vmcnt(5) 32% + vmcnt(3) 32% + vmcnt(2) 6%: waits on weights AND on the per-member q8
loads (4 x global_load_b128 per member per superblock, issued inside the member loop, 8 waves/SIMD at 173 VGPRs).
Lane 0 runs a second superblock iteration (9 sb over 8 block-lanes) with 4/32 lanes active.

### Step 2 ideas (ranked by expected gain x confidence / effort)
1. gu_smallb (structural): small-b gate/up twin. kwide lane map + unpack-once, but NO LDS xq staging and NO barriers:
   each lane reads its member's 16 q8 bytes straight from global (L0/L2; the 8 warps of a WG read the same bytes), the
   member loop is fully unrolled over MB=8 with uniform predicates (no movrel), the 10 superblock pairs are fully
   unrolled so all 340 B/lane of weight loads can be in flight (hetsplit-like MLP). Same work-item interface -> drop-in.
   Bit-exact with kwide by construction (same lane->data map, same accumulation order). Expect: 343 -> ~250 us at E=4.
2. dn_smallb (structural): small-b down twin of kwide2, same lane map/accumulation order (bit-exact), MB=8 member
   accumulators, both superblock iterations' weight loads hoisted, member loop unrolled with predicates. Expect 233 -> ~120.
3. kwide_c8 (knob): production kwide compiled with MXFP4_KW_MAX_CHUNK=8 (LDS 16 KB -> 4 KB, 64 -> 16 accumulator VGPRs),
   chunk_size 8. Cheapest possible change; tests whether occupancy is what limits kwide.
4. nobuild: host-known groups (b <= 8): the host has d_selected, so group_count/expert_members/work_items can be built on
   the CPU and uploaded once with the request, dropping 2 memsets + 2 kernels (~13 us) and the grid.y bound. Measured
   as chain-without-builders. Integration: host code in remote_experts.rs (medium).
5. exact grid.y instead of the devcount upper bound: measured, no gain (343 vs 361 us; the empty WGs are free). DEAD.
6. dn_bal (structural, not bit-exact): 4 rows/warp, 8 lanes x 9 consecutive blocks per row (72 = 8x9 exact tiling of the
   2304-wide down row); no idle lanes, 153 B/lane of weight loads in flight. Changes the reduction order -> fidelity gate.
7. fuse q8_k_quantize into the gate/up epilogue: BLOCKED. A Q8_K block spans 256 consecutive mid values = 256 rows =
   32 different WGs of the gate/up grid; the amax needs a cross-WG reduction. Would need a 256-row-per-WG gate/up
   (impossible at 8 warps) or a device-wide sync. Not pursued.
8. down + reduce fusion via float atomics onto out: drops 1 memset + 1 kernel (~12 us) but makes the sum order
   non-deterministic run to run (atomics) -> fails the determinism rule for expert-cache changes. Not pursued; the
   deterministic alternative (one WG owns all 6 slots of a row) re-streams weights per row = the b=1 twin (2x at b=4).
9. register (v_perm) nibble LUT instead of the LDS LUT: ATT shows LDS waits at ~5%; low priority, try if VALU shows up.
10. q8 activation padded to 16-B-aligned blocks (272-B stride) so the q8 loads are aligned b128: the compiler already
    merges the 4-B-aligned loads into global_load_b128; skip unless the ISA of the new kernels shows otherwise.
