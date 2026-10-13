# decode-opt branch: knob table (2026-10-13)

Branch `decode-opt` = production source `worktree-dgpu-fusion-prod` (9db5eb7d) plus, cherry-picked in
this order: the iGPU MoE decode kernels (`moe-dec-kernels`, fast-forward), the dGPU chain map bench
(`wf-dgpu-fusion-map` 64bef290), fusion group G1 (router block), G2 (q_a + kv front; the one-launch
experiment 271f512e and its revert 994044db skipped as a pair), G3 (attention block), and the
integration commit (stage attribution under G1, the combined bit-exact check).

**Every new knob defaults OFF. With all of them unset the binary runs the production chain**: each
fused path is a separate branch taken only when its knob is on and its preconditions hold; the old
code is unchanged on the off side (the only always-on additions are a 4-byte ticket and a 1 KB zero
row allocated per `BatchDgpuShared`, and engine modules loaded at start).

NOT deployed. None of the fused paths has run inside the hub yet (no restarts were allowed): before
flipping any of them, run the layer-0 exactness gate / golden gate with the knob on (hub down, box 2
up) and prove each path ran: G2 has the counter `forward_prefill::QKV_FRONT_LAUNCHES`; G1 and G3 have
no counter, so look for their `k.router.g1` / `k.attn.g3` stages in a sub-stage profile.

## Knobs

| Knob | Default | Read | What it switches | Bit-exact test (ignored GPU test) | Measured gain |
|---|---|---|---|---|---|
| `V41_MOE_PAIR_DEC` | off (`1` = on) | startup (LazyLock) | iGPU (and box-2 expertd) MXFP4 gate+up+SwiGLU passes of rows 1..=8 run `mxfp4_pair_decode.hip` (persistent, barrier-free, 5-slot ring) instead of `kwide` | `mxfp4_pair_decode::dec_matches_kwide_bit_exact` (17 selections, 952 launches, whole sentinel-filled `mid`) | 93-95% of the same-bytes streaming probe vs kwide's 84-86%; b=4 D=8 518 -> 467 us, b=8 D=8 566 -> 477 us, b=1 D=4 261 -> 240 us (1.09-1.32x, gfx1151, cold, beside the hub) |
| `V41_MOE_PAIR_DEC_DYN` | on (`0` = static stride) | startup | the decode gate+up kernel's dynamic work-unit ticket | same test | static stride is 2% slower at D=16 |
| `V41_MOE_PAIR_DEC_WAVES`, `V41_MOE_PAIR_DEC_WGS` | 16 waves / 1 WG per CU | startup | bench geometry overrides of the decode gate+up kernel | same test | tuning only |
| `V41_MOE_DOWN_DN3` | off (`1` = dn3q, `static` = dn3l, `grid` = dn3_r4) | startup | iGPU (and box-2 expertd) MXFP4 down passes of rows 1..=8 run dn3 instead of dn2; checked BEFORE `V41_MOE_DOWN_DN2` | `mxfp4_down_dn3::dn3_bit_identical_to_dn2` (2560 launches) + `dn3_exact_edges` (3840 launches, 4.08M partials vs the CPU replica of dn2) | dn3q 98-99% of the same-bytes probe at b=1..8, D=4..16; b=4 D=4 149.7 -> 115.3 us, b=8 D=8 289.3 -> 222.1, b=8 D=16 484.3 -> 430.6 |
| `V41_DGPU_G1_FUSED` | off | **live** (per decode lane-layer) | G1: pre-ffn collapse + router (l) + k1 look-ahead router (l+1) + cache-prior upload + both top-ks + box-2 Q8_K rows + readback pack + M62 counts as ONE launch (`g1_router_block.hip`, ticketed roles); `selected_ready` still recorded after it. Only for the direct mHC-fast / mix-late / fp32-router shape, b <= 16, no alternatives / k2 / image rows / pin / dump / stats | `g1_router_block_bitexact` (round 2: 1016 checks over b=1..16 x 4 configs x 2 lanes x every variant, ties forced, + `g1_router_block_big_ties_bit_identical` 1760 checks with 6 / 33 / 40 / 384 tied experts, + `g1_q8k_ties_are_exercised`) | group wall (cold) b=1 66.1 -> 34.4 us, b=4 72.0 -> 39.6, b=8 88.7 -> 51.6, b=16 122.8 -> 73.1; 10 queue ops -> 2. **Round 2 (G1-optimal, default kernels = family "r")**: cold group wall vs round 1 b=1 33.6 -> 31.9 us, b=2 36.9 -> 34.3, b=3 40.0 -> 36.5, b=4 41.5 -> 38.0, b=8 52.9 -> 50.2, b=16 74.4 -> 72.2; kernel span at b=4 26.0 -> 21.4 us (64-67% of the 13.6-14.3 us kernel floor, was 51%) |
| `V41_DGPU_G1_VARIANT`, `_NPF`, `_TS`, `_DBG`, `_HSACO_GFX1201` | unset | startup | G1 bench-only overrides (variant table -- `4,8,0` etc. select the round-1 kernels for A/B, `4,108,808321602` a family-"r" one; prefetch WGs (round-1 only), in-kernel phase timestamps, ablations, code-object override). Never set in production (`_HSACO_*` loads an arbitrary code object) | - | - |
| `V41_DGPU_QKV_FRONT` | off | **live** (per lane-layer) | G2: input quantize + q_a + kv GEMVs + q_a rms/quant (launch A, last-WG ticket) and q_b + q rope + kv rms/rope/fp8 + window append (launch B): 8 launches -> 2. Only at rows 1..=8 on the dp4a arms with `V41_DEC_FUSE`, Q8_0 weights, no kv f16 round trip, DIRECT stage (no graph capture) | `qkv_front_bitexact` (112 cases: b=1..8, plain + YaRN, arena + contiguous, 2 lanes) | -18.1..-19.5 us per lane-layer cold (b=1..8); G2 part 80-96% of its ~19.6 us floor (b=4: 24.1 us = 81%) |
| `V41_DGPU_QKV_FRONT_GRID` | 64 | startup | G2 launch A work-group count (bench override) | same test | G=64 optimal or tied at b=4/8 |
| `V41_ATTN_G3` | off (`1` = on) | **live** since 10-13 (read once per decode lane-layer) | G3: attn_meta_fill + indexer_gather + fused decode attention + rope_inv_quant_q8 as ONE launch (`attn_dec_g3.hip`; selection read in place, V by transposed loads, inverse rope + Q8_0 heads in the epilogue, written to the R1 `g3_heads_*` views). Index-source layers keep a 1-destination meta fill. Only with graphs off, the dec_fused attention, the `V41_DEC_FUSE` dp4a wo_a arm, `V41_DEC_SKIP_DEAD`, b <= 16, n_comp <= 512 and n_raw + n_comp <= 640, an f16 main store when the selection is used | `attn_dec_g3_bitexact::attn_dec_g3_is_bit_identical_to_the_chain` (256 comparisons, b=1..16, both lanes) | chain wall cold b=1 44.6 -> 25.8 us, b=4 48.5 -> 26.1, b=8 59.9 -> 30.2, b=16 98.4 -> 51.0; 4 launches -> 1 (2 on index-source layers); ~40% of the 7.2 us byte+launch roof at b=4 (verdict: not yet optimal; bit-identical two-launch split is the next step) |
| `V41_ATTN_G3_KERNEL` | `g3` | **live** (read per G3 launch) | which kernel a G3 lane-layer runs: `g3` = round 1's `attn_dec_g3`; `w1` / `w2` / `w4` = the G3-optimal "W" kernels (`attn_dec_g3o.hip`: one WG per row, head tile and 1/n of the dims; pipelined loads, faster phase A and epilogue; n = 2 / 4 replicate the scores + softmax per dim part); `auto` = `attn_dec_g3o::auto_mode`: W4 at b = 1, W2 at b = 2..3, W1 from b = 4 when every row holds >= 384 keys, else `g3`. Only matters with `V41_ATTN_G3=1`. Stage `k.attn.g3w` for the W kernels | `attn_g3o::g3o_bit_identical_to_the_chain` (10080 comparisons: 7 kernels x b = 1..16 x 2 lanes x 3 rope sets x 5 key layouts x steady / ragged / edge rows, subnormal + signed-zero KV, with and without heads_out; negative control) | `auto` vs `g3`, paired p50 (100 rounds, cold / warm): long rows (selected 128 + 512, index-source) b=1 -3.3 / -3.3 us, b=2 -2.3 / -2.1, b=3 -1.3 / -1.0, b=4 -0.7 / -1.2, b=8 -1.0 / -1.0, b=16 -1.4 / -1.4; ratio-0 (128 keys) b=1 -1.7, b=2 -1.2, b=3 -1.1, b>=4 0; short dense (128 + 200) b=1 -2.3, b=2 -1.5, b=3 -1.3, b>=4 0 (A/A p25..p75 within +-0.9). See "G3 round 2" |

## Combined check (G1 + G2 + G3 together)

`bench_dgpu_chain_map::combined_g123_bit_identical_to_chain` (ignored GPU test, dGPU only, ~115 MB):
one reuse lane-layer replayed with the production launchers in production order, the all-off chain vs
`reuse_g123` (all three fused groups), and each group alone (`reuse_g1`, `reuse_g2`, `reuse_g3`); two
lanes interleaved on one stream (lane B at 9 - b rows), b = 1..8 (G2's range), two consecutive passes
per b. Every buffer the lane-layer leaves for its readers is compared whole (rows past b included).
Excluded on purpose: the dead hand-offs the fused paths no longer write (`q`, `kv_normed`, the device
attention metadata, the gather destination, f32 `heads`). Result on decode-opt: **128 comparisons, 0
failures**; negative control (one residual input x 1.001) changes 21 buffers.

Integration note: the map replay's `SelCount` counted the prior-adjusted `sel`; production
(`forward_prefill` `plain_sel`) counts `d_orig_sel` under the live cache prior, which is what G1
reproduces. The replay was corrected to production semantics (the first combined run caught it).

## Combined chain wall (warm weights, map harness, 120 paired rounds beside the live hub)

`BENCH_SECTION=chain BENCH_CLASSES=reuse#AA,reuse_g1,reuse_g2,reuse_g3,reuse_g123,route:reuse,route:reuse_g123`.
Paired p50 saving per reuse lane-layer vs the production chain (queued wall):

| b | A/A \|d\| p50 / p90 | G1 | G2 | G3 | G1+G2+G3 | wall prod -> combined | launches+copies | route path (to selected_ready) prod -> combined |
|---|---|---|---|---|---|---|---|---|
| 2 | 1.0 / 20.5 | 24.7 | 17.3 | 19.8 | **61.6** | 261.4 -> 199.9 us | 33 -> 17 | 214.2 -> 154.8 us |
| 4 | 1.9 / 24.9 | 24.9 | 19.8 | 21.0 | **64.2** | 303.3 -> 237.2 us | 33 -> 17 | 246.7 -> 185.2 us |
| 8 | 3.1 / 10.5 | 29.8 | 15.0 | 25.3 | **75.0** | 434.1 -> 359.2 us | 36 -> 20 | 345.8 -> 273.0 us |

The savings add (b=4: 24.9 + 19.8 + 21.0 = 65.7 vs 64.2 together). Host enqueue per lane-layer drops
58.3 -> 29.3 us at b=4. Upper-bound projection at b=4: ~64 us x 80 lane-layers = ~5.1 ms of dGPU
queue time per step on the 67.6 ms step (index-source / dense layers gain less from G3; weights here
are MALL-warm, the cold focused benches above are the better per-group guide). The real step gain
must be measured live (profiled per-step dGPU wall, not tok/s: the e2e noise floor is ~8%).

## Floors (from the round-1 verdicts; see each group's bench)

- G1: kernel floor ~13.6-14.3 us (7.86 MB gate stream at 593-624 GB/s + ~1 us top-6). Round 1: kernel
  span 26.5 us at b=4 (51%). Round 2 (family "r", the default): 21.4 us at b=4 (64-67%); see
  "G1 round 2" below for what was measured and what still stands between it and the floor.
  Warm chain (map harness, paired vs production, 120 rounds): saved per lane-layer b=2 24.5 -> 26.1
  us, b=4 24.9 -> 27.1, b=8 28.3 -> 29.8 (round 1 -> round 2; A/A |d| p50 0.8-3.2); route path b=4
  77.2 -> 80.0 us saved.
- G2: G2 part floor ~19.1-19.6 us (9.75 MB at 590 GB/s + 1 gap + ~1 us tail); 81% at b=4, 67% at b=8.
  Next: Q8_0 of `attn_input_norm` emitted by its producer (drops the per-WG re-quantize), earlier q_a tail.
- G3: byte+launch roof 7.2-7.6 us at b=4 (3.2 MB at ~600 GB/s + one dependent launch). Device time
  (wall minus the event pair) at b=4 cold: round 1 20.0 us (38%), `auto` (W1) 19.3 us (39%); at b=1
  round 1 18.1 us vs a 3.6 us roof (20%), `auto` (W4) 14.8 us (24%). The bit-identical split the
  round-1 verdict proposed was built and measured: it does not beat round 1 from b = 3 (see "G3
  round 2" for what bounds this group on gfx1201).
- MoE: dn3q 98-99% and pair_dec 93-95% of the same-bytes streaming probe on gfx1151.

## G1 round 2 (G1-optimal): what the hardware does, what was built, what remains

Measured on gfx1201 beside the hub, dGPU only, production kernarg env (tests/g1o_probe.rs with
scratch/g1opt/probe.hip; in-kernel phase trace of tests/bench_g1_router_block.rs, V41_DGPU_G1_TS=1):

- Gate stream: u32 loads at 605 GB/s (whole row in flight, or only 8 words per lane in flight: 603);
  h20's u16 lane-strided loads 533. 7.86 MB -> 13.0 us of kernel span at the roof.
- A CU's vector memory path is a FIFO: an L2-hot load behind a WG's outstanding gate loads waits for
  them (~10 us behind a whole-row prefetch). Scalar loads do not queue there (0.35 us vs 0.09 idle).
- Outstanding bytes slow every other access: store + fence 0.16 us at 10 words per lane in flight,
  2.4 us at 40, 3.2 us at 80; scalar loads 0.23 -> 1.17 us. The collapse runs under the stream, so a
  deep prefetch delays its flag one-for-one (unpaced: flag 5.4 / 6.6 / 8.0 us at 16 / 24 / 32 words).
- `__syncthreads` drains every outstanding load in WGP mode; a scoped s_load can return a stale
  scalar-cache line (polls must invalidate first); `if (z < b)` guards in the unrolled loop make the
  waitcnt pass drain the ring (each guarded LDS read waited alone).
- The dispatcher puts a 4th router WG on ~4 WGPs (one per SE); those waves finish 1.5-2.5 us later.
  A design with exactly 3 WGs per WGP (16-wave WGs, all roles folded) removed that spread but ended
  at the same time: after the flag the aggregate stream is ~440-500 GB/s, and the end is set by bytes.

Built (family "r", default; bit-identical): paced u32 gate ring re-dealt to h20's lanes by DPP +
v_perm, x staged per wave in consumption order through LDS slabs with LDS-only barriers, scalar flag
polls (one wave per WG, bounded, sticky error word), a collapse tree with LDS-only barriers and wave
shuffles, parallel selection tails using an exact threshold + rank selection (serial passes as the
fallback), and a lane-parallel prior compose.

Where the b=4 kernel span (21.4 us; floor 13.6-14.3) still goes, each measured:
1. The collapse gate: x exists only at ~5.5-6 us (collapse flag), and the post-flag stream runs at
   ~440-500 GB/s. Skipping the flag wait (x treated as ready at t=0, timing only) ends the router
   phase 2-3 us earlier. The structural fix is to produce the collapse (cur, ffn_input_norm, Q8_K) in
   the hc_post producer, so G1 starts with x ready.
2. x staging + phase barriers inside the stream: no-x / no-barrier ablations (wrong results, timing
   only) end 1.5 / 0.7 us earlier.
3. The selection tail: 2.1-2.4 us from the last logit to the kernel end (detection 0.2-0.4, scores
   0.3, threshold + rank 0.6-1.0, emit + pinned pack writes 0.7) against ~1 us.
4. `selected_ready` (~3 us of queue time) is unchanged: the done word the kernel already writes needs
   the host protocol switched (owner approval + a gate run).

## G3 round 2 (G3-optimal): what was built, what was measured, what bounds it

Everything below: gfx1201 beside the live hub, dGPU only, production kernarg env, one process,
interleaved random order, MALL flushed for "cold"; tests/attn_g3o.rs (bench_g3o_chain_wall,
g3o_trace, bench_g3o_floors, g3o_load_probe + g3o_latency_probe with scratch/g3o/probe.hip). Device
work = wall minus the wall of the same launch with an empty body (10.4-10.8 us, incl. the 7.5 us
event pair).

Built (all bit-identical to the chain, tests/attn_g3o.rs):
1. The split the round-1 verdict proposed: S work-groups (row, 2 head tiles, 16 key tiles; every
   warp one K tile, two chains) write f16 scores (+ V^T fragments made by a WMMA against the identity
   from the K fragments they already hold) to a scratch; V work-groups (row, head tile, half the
   dims) replicate the 16-head softmax and run the W.V chains + epilogue. As two launches
   (`attn_g3o_s` + `attn_g3o_v`) and as ONE launch with ticketed roles (`attn_g3o_t`: V waits only on
   lower tickets, bounded scalar polls, self-resetting counters; scores and V^T counted separately).
   Result: -1.9..-2.5 us at b = 1, -1.0..-1.5 at b = 2, +0.4..+1.6 at b = 3..4, +8..+21 at b >= 5.
   Kept as bench-only modes.
2. The "W" family (production, `V41_ATTN_G3_KERNEL`): round 1's one-WG-per-(row, head tile)
   structure with every load pipelined (the selection, then q, then K; the next K tile loaded into the
   registers the chain just consumed; the W.V transposed-load ring started before the softmax), a
   branch-free phase A, the Q8_0 scale once per lane, and 1, 2 or 4 WGs per (row, head tile)
   (the scores + softmax replicated per dim part). Numbers in the knob table.

Measured on the way (each with a timing build or a probe):
- Per-tile branches around loads made the waitcnt pass put `s_wait_loadcnt 0` after every load (the
  W.V ring and the rolling K prefetch were fully serialized): runtime loops with branch-free bodies
  (table entries past n_tiles are the zero row, weights zero-padded to the ring) fixed it.
- `global_load_tr_b128` moves ~2 TB/s aggregate vs ~4.8 for the K-fragment pattern (g3o_load_probe,
  L2-resident, 32 WGs x 16 warps); one WG per WGP reaches ~150 GB/s of fragment loads.
- Dependent-load latency unloaded: 138 ns (L2), 259 ns (DRAM-cold) (g3o_latency_probe). Under this
  kernel's load the K / V phases are bandwidth-bound, not latency-bound: an L2 prefetch of the later
  K tiles (vector or `s_prefetch_data`) changed nothing or slowed it (+0.5..+2 us).
- W1 ablations at b=4 (wrong output, timing only): with no K / V traffic, no softmax and no W.V
  chain the kernel still needs 6.0 us (the 3 x 32-deep score chains per warp, the selection / q round
  trips, barriers, epilogue); K + V traffic adds 5.0 us (1.3 MB through one WGP), the softmax 2.6
  us (16 heads of phase A VALU on one WGP), the W.V chains 3.9 us.
- Split: S launch alone 8.5 us at b=4 (DRAM-bound K stream + V^T writes), the V tail after it
  6.8 us (score + V^T loads 1.7 us of issue stall, 16-head softmax 2.2, chains 1.4, epilogue 0.6).
- Neither 32-warp WGs (1024 threads, <= 2 score tiles per warp), a deeper W.V ring (8), a
  row-major dispatch order, nor staging the Q8 bytes through LDS helped (+0..+2 us).

What stops the 7.2-7.6 us roof at b=4, precisely:
- Bit-identity fixes the dependency shape: each score is one 32-step WMMA chain, each output one
  40-step chain over the key tiles in order, each head's softmax one warp in lane-strided order.
  On one WGP that is >= 2.1 us of WMMA issue per phase (1312 WMMAs / 4 SIMDs x 6.5 ns) plus the
  16-head phase A (2.6 us measured).
- WMMA M = 16 heads: every K and V tile is read once per head tile, 5.2 MB per row through the CUs
  (4x the 1.3 MB of unique bytes). Sharing a tile across head tiles needs a WG holding q of 64 heads
  (64 KB) or the weights of 64 heads (84 KB): the 64 KB per-WG LDS limit rules both out.
- Spreading a row over more WGPs (the split) keeps the 4x traffic and adds the score / V^T scratch
  round trips and a replicated 16-head softmax per V work-group: measured slower than one WG from
  b = 3. Two-launch costs a 2.3 us dependent launch on top.
- So at b >= 4 the group is set by per-WGP throughput (1.3 MB per (row, head tile) at ~150 GB/s)
  plus the chains and the softmax: ~15 us of work vs ~10-11 us for a perfectly overlapped W1. The
  remaining idea not built: a dedicated softmax role spread over 4 WGs per head tile (4 heads each)
  feeding V work-groups that hold all 64 heads' weights in two LDS halves -- it removes the 4x V
  re-read and the replicated softmax but adds a second hand-off; estimated -2..-4 us at b=4 from the
  measurements above, unproven.
- Not touched (other groups): index-source layers still run the 1-destination `attn_meta_fill`
  (~2.4 us x 16 lane-layers per step) because the indexer kernels read `n_index_comp_per_b` from
  device memory.

Per step (2 lanes x 40 layers, long context, b = 4): `auto` saves ~0.07 ms (76 compressed
lane-layers x ~0.9 us); at b = 1 ~0.13 ms per lane. Below the e2e noise floor: confirm with the
per-step dGPU busy / stage profile (`k.attn.g3w` vs `k.attn.g3`), not tok/s.

## Deploy notes (when the owner decides)

- `V41_DGPU_G1_FUSED`, `V41_DGPU_QKV_FRONT`, `V41_ATTN_G3` and `V41_ATTN_G3_KERNEL` are live knobs:
  they can be A/B'd without a restart once a decode-opt hub is running (`V41_ATTN_G3_KERNEL=auto`
  only acts where `V41_ATTN_G3=1`). `V41_MOE_PAIR_DEC`, `V41_MOE_DOWN_DN3` are read once at start.
- The MoE knobs also act inside `deepstrix-expertd` (box 2 runs the same dispatch): a box-2 change
  means rebuilding expertd on box 2 and the two-box restart order.
- Stage attribution with the knobs on: G1's launch is timed as `dgpu.router` (+ `k.router.g1`); the
  empty `dgpu.router` / `dgpu.rb_pack` spans are not opened under G1, `dgpu.mhc_pre_ffn` is then
  near-empty (its work moved into the G1 launch). G2's work shows under `dgpu.q_chain`
  (`k.q_chain.qkv_front_a/_b`) and `dgpu.kv_chain` disappears. G3 shows under `dgpu.attn_compute`
  (`k.attn.g3`, or `k.attn.g3w` for the W kernels), and its epilogue moves the rope_inv_quant work
  out of `dgpu.output_proj`. Compare
  sums of stages (or dgpu_busy), not single stages.
