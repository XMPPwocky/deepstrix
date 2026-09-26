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

## RESUMED 2026-09-26 evening (hub DOWN, GPUs idle; per-device scheduler; iGPU ATT refused)

### Baseline RE-TAKEN with the hub idle (results/parts_idle_run1_b{1,2,4,8}.txt, chain_*_run1 "base chain" rows)
The 04:00 numbers above were contaminated by the live hub (kwide b=4 343 us then, 280 us now). Also a harness
accounting fix: at b=1 a row picks ppr=3 of the E=4 pool slots, so only 3 experts (56.4 MB) are read, not 4; the
twins (row 0's picks) read 3 experts at every b. The harness now counts distinct experts actually read (E_eff).
Corrected idle-hub baseline (E=4 pool, ppr=3, P=16 rotating, graph of 8 calls):
    kernel                        b=1 (3 experts)   b=2 (4)     b=4 (4)     b=8 (4)
    kwide gate/up, bound grid     197.6 = 190 GB/s  248.6=202   280.4=179   339.8=148 GB/s
    kwide gate/up, exact grid     197.3             249.0       278.0       315.5=159
    kwide2 down,   bound grid      99.6 = 189 GB/s  133.7=187   178.6=140   252.9= 99 GB/s
    kwide2 down,   exact grid      99.3             128.1=196   159.2=157   204.8=122
    builders 12.0 | q8k 4.1-7.6 | partials memset 3.4-4.5 | reduce 4.9-7.3 (small kernels 25-32 us = 5-6%)
    twins (3 experts): pair_hetsplit 201 us = 187 GB/s; down_hetsplit 90 us = 209 GB/s  (= the CLOSED figures)
    chain: b=2 400 us (188 GB/s, 88% of the 351-us roof) | b=4 470-483 (156-160 GB/s) | b=8 590-615 (122-128 GB/s)
=> at b=2 the batched chain is within 12% of roof; the small-b gap is at b=4-8: gate/up loses 25% and down 45%
   of its b=1 GB/s, and the 6b-bound grid's EMPTY WGs are NOT free when the GPU is idle: kwide2 b=8 253 vs 205 us
   exact (-19%), b=4 179 vs 159; kwide b=8 340 vs 315. (The 04:00 "exact grid = no gain" was under load.)

### Candidates measured (chain A/B, graph, ratios = med vs base chain; every one BIT-EXACT vs base at b=2,4,8 x 8 selections)
    id                     b=2     b=4     b=8    notes
    smallb (v1, 04:00)     1.55    1.63    1.73   LOSS. 192 VGPR + spills; predicated member branches split every pair
                                                  into a basic block -> loads issued+drained per pair. Dead.
    gu2_mb8/mb4 (D=5)     19-26x  17-24x   19x    pathological: compiler hoisted ALL weight loads to the entry (invariant
                                                  loads from const __restrict__), 700-960 spills. Pinning addresses via
                                                  asm volatile did not help (pinned pointers spilled instead). Dead form.
    gu2_mb4_d2 (rolled)    0.993   1.006    -     NEUTRAL. LDS-staged all-member activations + 2-pair prefetch ring,
                                                  no barriers in the pair loop, straight-line 4-member dot. 151 VGPR.
    gu2_mb8_d2             1.072   1.098   1.087  LOSS (183 VGPR, 41.6 KB LDS = 1 WG/CU; 2 of 8 members wasted at nm=6)
    gu2_mb8_d2.w16         1.053   1.060   1.012  16-warp WGs help but still a loss
    kwide_c8.c8            0.998   0.978   0.930  production kwide compiled with MXFP4_KW_MAX_CHUNK=8 (LDS 16->4 KB,
                                                  2x32->2x8 accumulators, 119->71 VGPR); chunk 8. WIN at b=8.
    base.c8                0.996   0.994   0.998  chunk 8 with the production build: nothing -> the c8 gain is occupancy
    dn2_r2 (run1)          0.954   0.932   0.917  member-outer down twin of kwide2, 2 rows/warp, 99-105 VGPR, no
    dn2_r2 (run2)          0.953   0.924   0.921  arrays/movrel; whole 2-row weight stripe in flight. WIN.
    dn2_r1.r1              0.983   1.016   1.068  same with 1 row/warp (640 x n_wi grid): loss at b>=4
    combo_c8dn2.c8 (run1)  0.947   0.909   0.852  kwide_c8 + dn2_r2 in one code object. Best so far.
    smallb.nobuild         (host groups + exact grid) 16-88 us faster than smallb -> exact grid matters, builders 12 us
Why the gate/up redesign does not pay: with the hub idle kwide gate/up already streams at 202 GB/s at b=2 and 179 at
b=4 (94%/84% of 214) - the 04:00 ATT (63% of stall on one vmcnt) was taken under production memory traffic. The
remaining gate/up loss at b=8 (148 GB/s) is mostly the bound grid (315 exact) and the 32-member accumulator arrays.

### Work-item loop ("wl": grid.y capped at WL, kernels loop w = blockIdx.y; w < n_wi; w += gridDim.y) — BIT-EXACT
cand_wl_gu.hip / cand_wl_dn.hip = the production kwide / kwide2 bodies verbatim + the loop (the early return
`blockIdx.y >= *n_work_items_dev` becomes the loop bound; nullptr = gridDim.y = exact grid). cand_dn_v2.hip loops too.
    chain ratio (wl=8)     b=2     b=4     b=8
    wl (uncapped, 6b grid) 1.030   1.037   1.026   the loop kernel alone with the bound grid: empties still pay (+ LUT stage)
    wl.wl                  1.018   0.970   0.905   production bodies, capped grid
    wl_dn2.wl              0.965   0.896   0.830   + member-outer down
    wl_c8dn2.wl.c8         0.929   0.857   0.778   + kwide at MXFP4_KW_MAX_CHUNK=8. BEST. (run2: 0.946 0.878 0.788)
    combo_c8dn2.c8         0.948   0.915   0.872   same kernels without the cap (bound grid) -> the cap is worth 2-10%
Per kernel (parts_wl_run1): b=8 gate/up 356 -> 307 us (163 GB/s), down 255 -> 159 us (158 GB/s);
                            b=4 gate/up 289 -> 273, down 184 -> 129 (195 GB/s).
WL sensitivity (results/chain_wl{4,6,16}_b8, chain_E12wl*_b8): wl=4/6/8/16 at b=8: 0.771/0.775/0.778/0.786 (flat);
E=12 ppr=6 (n_wi=12 > cap): wl=8 0.844, wl=16 0.846, uncapped 0.879 -> the cap wins even when WGs loop over 2 items.
Production n_wi is 3-6 (box 2) so WL=8 never loops there; on the hub (b<=5, up to 30 distinct) a WG does <= 4 items.
Repeat runs of the final set (separate processes; ratios b=2/4/8):
    wl_c8dn2.wl.c8  run1 0.929/0.857/0.778  run2 0.946/0.878/0.788
    wl_dn2.wl       run1 0.965/0.896/0.830  run2 0.984/0.916/0.833
    wl.wl           run1 1.018/0.970/0.905  run2 1.027/0.993/0.921
    dn2_r2          run1 0.954/0.932/0.917  run2 0.953/0.924/0.921  run3 0.963/0.937/0.932
    kwide_c8.c8     run1 0.998/0.978/0.930  run2 1.002/0.993/0.957
dn2 member loop `#pragma unroll 2` (wl_c8dn2u2): compiler refused to unroll (break inside the it-loop) -> identical code. Dead.

### Tails, regimes, last knobs (all BIT-EXACT vs the production chain over 8 selections each)
    wl_c8dn2.wl.c8 chain ratio:  b=1 0.964 | b=2 0.93 | b=3 0.901 | b=4 0.86-0.88 | b=5 0.838 | b=6 0.812 | b=7 0.792 | b=8 0.78-0.79
    E=6 ppr=6 (6 distinct experts, every pick real): b=2 0.915 | b=4 0.852 | b=8 0.811 (results/chain_E6ppr6_run1_b*.txt)
    gu3 (kwide + next-pair weight prefetch across the barriers): 0.939/0.876/0.786 vs wl_c8dn2 0.935/0.862/0.776 -> NEUTRAL
    wl_c8u8dn2 (MXFP4_KW_UNROLL=8 on top): 0.860/0.771 vs 0.861/0.778 -> NEUTRAL
    b=1 (hub lanes at 1 row use these kernels too): wl_c8dn2 0.964, dn2 0.966, kwide_c8 1.006, wl.wl 1.030 (the loop
    kernel with the 6-bound grid at b=1 has 2 empties/tile x LUT stage -> cap 8 > bound 6 = uncapped: use min(bound, 8)).

### FINAL per-kernel picture (idle hub, E=4, cold rotating, graph; us)
                     b=1        b=4                    b=8
    gate/up   base   197.6      280-289                340-356          roof 175.7 (b=1, 3 experts) / 234 (4 experts)
              best   ~198       273 (c8+cap, 183 GB/s) 307 (163 GB/s)
    down      base    99.6      179-184                253-255          roof  87.8 / 117.1
              best   ~ 96       129 (dn2+cap, 195 GB/s) 159 (158 GB/s)
    small kernels (builders 12-13, q8k 4-8, memset 3-5, reduce 5-8) = 25-32 us, unchanged.
    chain     base   316        477-486                612-640
              best   305        419-426                482-498
Gate/up is now 84%/70% of its 4-expert roof at b=4/8 and down 92%/75%. What is left at b=8: gate/up's per-pair
barrier + LDS-stage structure and the dynamic-indexed 8-member accumulators (movrel); down's 1/9 second-iteration
lane idling (block_lane 0 only) and 6 sequential warp reductions per warp. Neither is a free lunch: the LDS-staged
gate/up rewrite (gu2) and 1-row down (dn2_r1) both lost.

### Integration recipe (all three bit-exact; est. from the chain deltas x 40 layers)
1. wl cap (small): kernels K/mxfp4_pair_matvec.hip kwide + K/mxfp4_matvec.hip kwide2: replace the
   `if (n_work_items_dev && blockIdx.y >= *n_work_items_dev) return;` early-out by the loop in cand_wl_gu.hip /
   cand_wl_dn.hip (n_items = *n_work_items_dev or gridDim.y; `for (w = blockIdx.y; w < n_items; w += gridDim.y)`;
   kwide needs a __syncthreads at the loop top). Wrapper: grid.y = min(n_work_items, 8) at S/mxfp4_pair.rs:289
   (`grid: (n_rows / 8, n_work_items, 1)`) and S/mxfp4.rs:183 (`grid: (n_rows / 16, n_work_items, 1)`); the bound
   itself comes from dispatch.rs `moe_wi_upper_bound` (decode: M = 6b). Prefill (B=512, n_wi ~ 300+) must keep the
   full bound or a larger cap -- family B's call; the loop is a no-op when grid.y = bound.
2. kwide_c8 (small): a second symbol compiled with MXFP4_KW_MAX_CHUNK=8 (e.g. `..._kwide_c8`), chosen by the
   wrapper when b <= 8, with chunk_size = 8 passed to moe_work_items_builder (groups have <= b <= 8 members so
   chunking never splits). Prefill keeps the 32-chunk symbol.
3. dn2 (small-medium): new kernel mxfp4_matvec_par_by_expert_smallb = cand_dn_v2.hip (DN_ROWS=2, same interface
   and (320, n_wi) grid as kwide2, includes the loop), selected for b <= 8 in S/mxfp4.rs; n_blocks_in is compile-time 9
   (returns early otherwise -- the wrapper must assert N_FF_EXP == 2304).
est ms per step: b=4 (one lane, E=4): chain 483 -> 420 us = -63 us x 40 = 2.5 ms; b=8: 620 -> 488 = -132 x 40 = 5.3 ms;
E=6 regime: b=4 738 -> 629 = 4.4 ms, b=8 965 -> 783 = 7.3 ms. Box-2 batched pass is not graph-captured (direct
launches) - the per-kernel deltas carry over; the hub's lanes are graph-captured as measured here.

### Repro (from this dir; every GPU command goes through the scheduler)
    bash build.sh gfx1151 wl wl_dn2 wl_c8dn2 dn2_r2 kwide_c8 combo_c8dn2 gu2_mb4_d2 gu2_mb8_d2 gu3_c8dn2 wl_c8u8dn2
    bash ../_infra/gpu_run.sh --dev igpu --mb 450 --label A_moe_smallb/final -- bash run_chain.sh final_runN wl.wl,wl_dn2.wl,wl_c8dn2.wl.c8,dn2_r2,kwide_c8.c8 "2 4 8" wl=8
    bash ../_infra/gpu_run.sh --dev igpu --mb 450 --label A_moe_smallb/parts -- bash run_parts.sh idle_runN - "1 2 4 8"
    bash ../_infra/gpu_run.sh --dev igpu --mb 450 --label A_moe_smallb/parts-cand -- bash run_parts.sh wl_runN wl.wl,wl_c8dn2.wl.c8,combo_c8dn2.c8 "4 8" wl=8
    bash ../_infra/gpu_run.sh --dev igpu --mb 450 --label A_moe_smallb/sens -- bash run_wlsens.sh
    bash ../_infra/gpu_run.sh --dev igpu --mb 450 --label A_moe_smallb/tails -- bash run_chain.sh tails_runN wl_c8dn2.wl.c8,dn2_r2,kwide_c8.c8,wl.wl "1 3 5 6 7" wl=8
    bash ../_infra/gpu_run.sh --dev igpu --mb 450 --label A_moe_smallb/E6 -- bash run_chain.sh E6ppr6_runN wl_c8dn2.wl.c8,wl_dn2.wl,dn2_r2 "2 4 8" wl=8 E=6 ppr=6
    bash show.sh chain <tag> "<b list>"; bash show.sh parts <tag> "<b list>"; bash isa_all.sh <cand names>
Harness caveats vs production: synthetic weights/activations (E8M0 scales 118..125 so dots land near the clamp);
selections rotate over a 16-expert pool (301 MB) so weights are cold; gbound=384 stands in for the pool-slot count;
box 2 launches directly (not a graph) - kernel times are the same, per-launch overhead is not measured here.
