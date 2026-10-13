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
| `V41_DGPU_G1_FUSED` | off | **live** (per decode lane-layer) | G1: pre-ffn collapse + router (l) + k1 look-ahead router (l+1) + cache-prior upload + both top-ks + box-2 Q8_K rows + readback pack + M62 counts as ONE launch (`g1_router_block.hip`, ticketed roles); `selected_ready` still recorded after it. Only for the direct mHC-fast / mix-late / fp32-router shape, b <= 16, no alternatives / k2 / image rows / pin / dump / stats | `g1_router_block_bitexact` (680 checks, b=1..16, ties forced) | group wall (cold) b=1 66.1 -> 34.4 us, b=4 72.0 -> 39.6, b=8 88.7 -> 51.6, b=16 122.8 -> 73.1; 10 queue ops -> 2. Kernel at 51% of its 14.3 us kernel floor at b=4 (verdict: not yet optimal) |
| `V41_DGPU_G1_VARIANT`, `_NPF`, `_TS`, `_DBG`, `_HSACO_GFX1201` | unset | startup | G1 bench-only overrides (variant table, prefetch WGs, in-kernel phase timestamps, ablations, code-object override). Never set in production (`_HSACO_*` loads an arbitrary code object) | - | - |
| `V41_DGPU_QKV_FRONT` | off | **live** (per lane-layer) | G2: input quantize + q_a + kv GEMVs + q_a rms/quant (launch A, last-WG ticket) and q_b + q rope + kv rms/rope/fp8 + window append (launch B): 8 launches -> 2. Only at rows 1..=8 on the dp4a arms with `V41_DEC_FUSE`, Q8_0 weights, no kv f16 round trip, DIRECT stage (no graph capture) | `qkv_front_bitexact` (112 cases: b=1..8, plain + YaRN, arena + contiguous, 2 lanes) | -18.1..-19.5 us per lane-layer cold (b=1..8); G2 part 80-96% of its ~19.6 us floor (b=4: 24.1 us = 81%) |
| `V41_DGPU_QKV_FRONT_GRID` | 64 | startup | G2 launch A work-group count (bench override) | same test | G=64 optimal or tied at b=4/8 |
| `V41_ATTN_G3` | off (`1` = on) | startup (LazyLock) | G3: attn_meta_fill + indexer_gather + fused decode attention + rope_inv_quant_q8 as ONE launch (`attn_dec_g3.hip`; selection read in place, V by transposed loads, inverse rope + Q8_0 heads in the epilogue, written to the R1 `g3_heads_*` views). Index-source layers keep a 1-destination meta fill. Only with graphs off, the dec_fused attention, the `V41_DEC_FUSE` dp4a wo_a arm, `V41_DEC_SKIP_DEAD`, b <= 16, n_comp <= 512 and n_raw + n_comp <= 640, an f16 main store when the selection is used | `attn_dec_g3_bitexact::attn_dec_g3_is_bit_identical_to_the_chain` (256 comparisons, b=1..16, both lanes) | chain wall cold b=1 44.6 -> 25.8 us, b=4 48.5 -> 26.1, b=8 59.9 -> 30.2, b=16 98.4 -> 51.0; 4 launches -> 1 (2 on index-source layers); ~40% of the 7.2 us byte+launch roof at b=4 (verdict: not yet optimal; bit-identical two-launch split is the next step) |

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

- G1: kernel floor ~13.6-14.3 us (7.86 MB gate stream at 593-624 GB/s + ~1 us top-6); kernel span
  26.5 us at b=4 (51%). Next: collapse moved into the hc_post epilogue, whole-row register prefetch,
  b32 loads + permlane, parallel selection tail, host done-word instead of `selected_ready`.
- G2: G2 part floor ~19.1-19.6 us (9.75 MB at 590 GB/s + 1 gap + ~1 us tail); 81% at b=4, 67% at b=8.
  Next: Q8_0 of `attn_input_norm` emitted by its producer (drops the per-WG re-quantize), earlier q_a tail.
- G3: byte+launch roof 7.2-7.6 us, split-limited ~10 us; G3 ~18 us at b=4. Next: bit-identical
  two-launch split (scores over all WGPs, W.V by 128-dim quarters), V prefetch before softmax.
- MoE: dn3q 98-99% and pair_dec 93-95% of the same-bytes streaming probe on gfx1151.

## Deploy notes (when the owner decides)

- `V41_DGPU_G1_FUSED` and `V41_DGPU_QKV_FRONT` are live knobs: they can be A/B'd without a restart
  once a decode-opt hub is running. `V41_ATTN_G3`, `V41_MOE_PAIR_DEC`, `V41_MOE_DOWN_DN3` are read
  once at start.
- The MoE knobs also act inside `deepstrix-expertd` (box 2 runs the same dispatch): a box-2 change
  means rebuilding expertd on box 2 and the two-box restart order.
- Stage attribution with the knobs on: G1's launch is timed as `dgpu.router` (+ `k.router.g1`); the
  empty `dgpu.router` / `dgpu.rb_pack` spans are not opened under G1, `dgpu.mhc_pre_ffn` is then
  near-empty (its work moved into the G1 launch). G2's work shows under `dgpu.q_chain`
  (`k.q_chain.qkv_front_a/_b`) and `dgpu.kv_chain` disappears. G3 shows under `dgpu.attn_compute`
  (`k.attn.g3`), and its epilogue moves the rope_inv_quant work out of `dgpu.output_proj`. Compare
  sums of stages (or dgpu_busy), not single stages.
