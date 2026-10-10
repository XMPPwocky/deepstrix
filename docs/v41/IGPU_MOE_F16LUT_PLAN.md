# iGPU prefill MoE: what is left after the int8-WMMA arm (plan, 2026-10-10)

Status (updated same day): the f16 kernel is BUILT and measured in the B_moe_prefill harness on the iGPU,
beside the live hub (owner: "slot beside live traffic"; ~2.7 GB, seconds per run, no profiler).
**Results are at the end of this file.** No production code changed; not wired into dispatch.

CORRECTION: the first compile table below ("3 tiles 233 VGPR, free") was measured on a generated file whose
3-tile dispatch was dead (it only existed under `WM_NTMAX >= 4`, so NTMAX=3 silently ran the 1-tile path).
The real numbers are in the results section.

## Where we are

The int8-WMMA MXFP4 arm already ships (`mxfp4_moe_wmma.hip`, `V41_MOE_WMMA_GATEUP` / `_DOWN`, default ON,
sweep ledger #27/#28: gate+up 1.56x at B=1024, down 2.7x). Measured iGPU matrix peak is ~55-59 TOPS
(`project_igpu_moe_wmma_f16`; dot4 and WMMA share it). The shipped arm realises about 12 TOPS-equivalent
at B=1024 (gate+up 7.3 ms / 96 experts, down 3.4 ms), i.e. ~20% of peak; the prefill roofline needs ~35.

## Why it stops at ~20% (from the kernel + the ledger's PMC table)

1. Weight-stationary over <= 32 members. `WM_MAX_CHUNK = 32`, one work item per (expert, 32-member chunk).
   An expert with 64 members is read from DRAM twice, with 1024 members 32 times. At 18.8 MB per
   chunk of 32 members the DRAM roof is 32 x 70.8 MF / 18.8 MB x 230 GB/s = **27.7 TOPS** even with
   perfect compute; layer-major windows (N/64 members per expert on average) make this the typical case.
2. Per-(block, tile) float epilogue. `acc += (sc*yd) * (float)C` is ~24 VALU per 2 WMMA (64 clk) per tile per
   block, linear in tiles, and WMMA occupies the VALU (nothing overlaps; ISA 7.9, measured in the f16 work).
   Plus the nibble unpack: 84 VALU per 32-weight block per matrix.
   Cycle model, fraction of the WMMA-pipe peak: 2 tiles 48%, 3 tiles 55%, 4 tiles 58%, infinite 73%.

## Compile-only data (gfx1151, this tree, `cand_wide.hip` = `mxfp4_moe_wmma.hip` with `-DWM_NTMAX=n`)

| tiles (members/work item) | gate+up VGPR / LDS / scratch | down VGPR |
|---|---|---|
| 2 (32, shipped) | 230 / 10.1 KB / 0 | 133 |
| 3 (48) | 233 / 15.2 KB / 0 | 128 |
| 4 (64) | 256 / 20.2 KB / 500 B (spills) | 231 |

So 48 members per work item is free on registers; 64 spills slightly on gate+up. Wider chunks alone buy the
DRAM side (27.7 -> 41 TOPS roof at 48 members) but not the VALU side (55% of peak).

## Proposal: f16 WMMA with the E8M0 scale folded into the dequant LUT (not built)

Replace the int8 dot + per-block float epilogue with an **f16 x f16 -> f32 WMMA chain over the whole 256-K
super-block**, so there is no per-block epilogue at all:

* Weights as f16 `kv * 2^(e8-128)` (kv = {0,1,2,3,4,6,8,12} x sign). Every e2m1 magnitude has f16 low byte 0, so
  the pattern is one hi byte: `(exp<<2 | mant?2:0)`. Scaling by 2^s is `+4*s` on that byte. Build the 8-byte
  perm LUT per (row, block) with **2 VALU** (add `4*(s+14)*0x01010101`, mask the zero entry), then per 4 weights:
  and, perm (LUT select), and+shift-or (sign), 2 spread perms = ~6 VALU. Block cost ~56 VALU vs 84+epilogue.
* Activations as exact f16 `q` (int8 values are exact in f16); `yd` is applied once per super-block per tile
  (`acc += yd * t`, 8 fma per 8 blocks). No `-12*bsum` offset trick (signed weights). Staging converts int8 -> f16
  once per WG (shared by 8 waves; ~2% of loop cost), LDS x2 (528 B/member row).
* Cycle model, per block per matrix: 64*NT clk WMMA + ~58 VALU -> NT=2 69%, NT=3 77%, NT=4 82% of the WMMA peak
  (vs 48/55/58% now): **1.4-1.7x in the compute-bound regime**, on top of the member-chunk widening.
* Numerics: products q*w are exact in f32 (<= 11 x 4 significant bits), the sum order differs from the int path
  only by f32 re-association inside WMMA/accumulation: the same class as the shipped arm (rel_rmse ~5e-7),
  so it goes through the same combined golden/KLD gate. Not bit-exact with kwide.

### De-risk evidence (all CPU, done today)

* **Scale range is narrow.** Read-only scan of the HF checkpoint (shards 05 and 10 = layers 2 and 7, first 1 MB of
  each of 40/60 expert scale tensors, 14.7M + 22.1M scales): e8 in 119..123, 99.99% in 120..121 (2^-8, 2^-7).
  f16 range needs e8-128 in [-14, 9]; huge margin. (`scan_scales.py` is in the job dir, not committed; rerun on
  more layers before the real build. Layers 20+ and the DSpark/MTP experts are unscanned.)
* **LUT math is exact.** `f16lut_emul.py`: emulates v_perm_b32, builds the LUT for every e8 in 100..139 (valid
  subset), all 16 nibbles x 4 lanes, compares with `kv*2^(e8-128)`: 1536 values, 0 mismatches, lane order correct.

## Build plan -- REVISED 2026-10-10 (owner: "batch across subchunks first")

Why this order. Today a prefill unit is `B_MAX = 1024` rows in TWO pipeline lanes of 512 rows, and each lane's routed
MoE is its own call (~8 members/expert, `batch_scratch.rs:59`, `forward_prefill.rs` `pipelined_range_impl`).
Measured, same-process interleaved A/B, shipped int8 arm vs the f16 arm (medians; p10 in the RESULTS section):

| rows per MoE call | members/expert | gate+up shipped -> f16 | down shipped -> f16 |
|---|---|---|---|
| 512 (today) | 8 | 12.6 -> 8.7 ms | 4.6 -> 3.9 ms |
| 1536 | 24 | 8.1 -> 5.2 ms | 3.2 -> 2.6 ms |
| 3072 (3 tiles, chunk 48) | 48 | 11.2 -> 7.6 ms | ~tie (p10 4.0 -> 3.4) |

(slices of 96/48/48 experts; absolute ms are not comparable across rows, the hub is live.) At today's 512-row calls
the f16 arm saves ~15% (p10) to 26% (median) of MoE kernel time, which is ~0-5% end to end because the wall is box 2's
round trip (RTT 47 ms/layer-unit = wire 28 + box-2 run 18, plus 30 ms turnaround; hub MoE units are ~28% of wall
and mostly hidden). The f16 kernel only turns into tok/s once the MoE call is >= ~1.5-3K rows AND the round trip is
not the wall. Routed-expert compute alone at 3072 rows: ~1.3K tok/s shipped vs ~1.8K f16 on ONE iGPU (x2 with both
iGPUs sharing), against a ~430-600 tok/s e2e today.

Constraints found while scoping:
* **dGPU VRAM is 97% full AT REST** (16.55 of 17.1 GB used, checked 10-10) -- but prefill never runs concurrently with
  decode, so decode-only memory is idle during a prefill phase and can be LENT to it. Source: the 10-09 dGPU-bundle
  session's census + phase-swap report (subagent `ae6e21916f26c4a0e`, transcript
  `~/.claude/projects/-home-claude-code-deepstrix--claude-worktrees-dgpu-fusion/749c61d3-d78c-4aa5-be8f-4f524d26dbb7.jsonl`
  line 7666, 2026-10-09 19:11 UTC; its verdict on hot experts is line 7524; "dGPU phase-pool VRAM work" is parked in
  the 10-10 01:41 summary). Findings that matter here:
  * KvArena (decode KV, 3.33 + 0.26 GB): prefill only reserves rows and fills them at admission, so the UNRESERVED tail
    is idle in prefill: **~2.5-2.8 GB typical**, one contiguous run after `compact_stores`; up to 3.6 GB if live KV
    (~0.8 GB, ~0.14 s each way at 6 GB/s) is swapped to host. The lent tail must be returned before any `reserve` /
    `grow` / compaction / `fill_reserved` (job start, admission, parked `try_admit` retries; those run every tick today
    and would need phase gating). Phases: prefill burst 120 s, decode burst 30 s (`multistream.rs:1112-1146`).
  * Decode-only twins of uncaptured prefill scratch (`attn_scores` 176, `attn_active_comp_kv` 235, `r3` 96,
    `candidate_block_score` 165, comp snapshots 34, `indexer_scores` un-carved from r1 ~600 MB; ~1.2 GB total) can be
    swapped per phase without touching graphs; two dead buffers (`verify_scores`, `DgpuScratch.attn_scores`, 190 MB).
  * Precedents: `dgpu_loan.rs` (embed phase lends views over existing allocations, no device alloc/free);
    `lm_store` lazy alloc with halve-on-OOM (`forward_prefill.rs:1121-1165`). No general VRAM budget, no HIP VMM.
  * Consequence for this plan: attention-side lanes CAN grow in prefill (the two-lane dGPU set is ~885 MiB at 512-row
    lanes per `batch_scratch.rs`; it scales ~linearly, so 1024-row lanes are ~1.8 GB and 2048-row lanes ~3.5 GB), so both "bigger lanes" and MoE-side staging are
    open. (iGPU GTT has room too: ~20 GB, but system RAM is tight, ~6 GB available.)
* **Wire (measured 2026-10-10; this CORRECTS the RTT breakdown above).** Bigger requests do not cut wire BYTES (f16
  replies are 10 KB/row/layer whatever the request size); they cut per-request latency, hub turnaround and box-2
  kernel time. Facts: `thunderbolt0` is a `usb4` link trained at **2 lanes x 10 Gb/s = 20 Gb/s on BOTH ends** (sysfs
  `rx_speed`/`tx_speed`; owner says the cable should be Gen3 = 2 x 20; the kernel cannot see the cable; no typec class).
  iperf3 over the Thunderbolt addresses, 3 s each: **10.1 Gb/s (1.26 GB/s) box1->box2, 9.1 Gb/s (1.13 GB/s) reverse, 0
  retransmits; 4 parallel streams 9.7 Gb/s total** (no gain), i.e. ~half the Gen2 line rate, typical of the
  `thunderbolt-net` software path. Live decode traffic is ~10-50 MB/s with 0 rx misses/s; the cumulative 840K rx
  "missed" on box 1 (0.24% of packets; box 2: 51) happens in bursts under bulk load; the current hub connection has had
  1 retransmit in 94 GB. THE STALE PART: the "wire 28 ms = 10 out + 18 back" of the 10-01 time split is 3.0 MB / 10 ms
  and 5.24 MB / 18 ms = 293 MB/s = the **2.5 GbE** link of that day (the hub moved to Thunderbolt on 10-03). At the
  measured 1.1-1.3 GB/s the pure transfer is ~2.5 + 4.4 = ~7 ms per round trip, so the 78 ms unit-layer breakdown is
  OUT OF DATE (no post-move breakdown exists; pull evtrace for one). Consequences: a Gen3 link (if it lifts the
  software path ~2x as line-rate scaling suggests; unverified) saves only ~3-4 ms of the round trip, not ~14; the wire
  is a small part of the wall today; at 2K tok/s the bulk traffic is ~640 MB/s = ~50% of the measured 1.2 GB/s cap, so
  it only becomes binding above ~2.5K tok/s.
* **Latency.** Prefill units interleave with decode bursts (scheduler); a longer non-preemptible MoE batch delays
  decode streams. Keep the batch <= ~2048 rows (~1 unit-equivalent in time) and A/B decode p90 with it.

Steps (each gated; nothing GPU beyond the harness until a window is agreed):

1. **MoE-side batching across lanes/sub-chunks.** Design + code-site map: see "Cross-sub-chunk MoE batching: design"
   below. Knob `V41_PREFILL_MOE_ROWS` (default OFF/0 = today), target 2048. Gate: bit-compatible routing (same experts
   per row), f32-association-only differences, `lm_prefill_bitexact`-style test extended with a batched arm.
2. **f16 arm into dispatch** (`V41_MOE_WMMA_F16`, default OFF): chunk 48 into the work-item builder, Rust wrappers for
   `cand_f16.hip` (NTMAX=3), `build.rs` registration, box 2 gets it for free (same kernels). Bench with the harness at
   the batched row counts first.
3. Kernel overlap work (double-buffered activation staging; PMC one counter per pass, never ATT on the iGPU): the f16 arm
   is at ~25% of the ~57 TOPS peak with ALL global loads removed -> ~28%; target ~40%.
4. (GPU, server down) golden/KLD gate for the f16 arm (and, still owed, for f16 box-2 replies), then e2e prefill A/B at
   long context with layer-major windows; decode p90 check.
5. Rest of the E8M0 range scan (only layers 2 and 7 sampled).

## Risks / unknowns

* The model assumes WMMA and VALU fully serialize and DRAM at 230 GB/s; measured kernels sit at 73-79% (gate+up)
  and 82-90% (down) of that, so the realistic ceiling is ~35 TOPS, not 48.
* 2 mats x 3 tiles at 233 VGPR is already at the occupancy edge (earlier pipelining attempts spilled badly:
  ledger #74/#75/#76).
* Box 2's iGPU is identical, but it runs the same kernels through the same dispatch; the xq rows it receives
  already carry the data this needs (int8 q + yd).
* NPU / dGPU are not touched by this.

## RESULTS (2026-10-10, gfx1151 iGPU, harness `scratch/kernel_ideas/B_moe_prefill`, beside live hub traffic)

Files: `mid_f16.inc` (the kernel arm) + `cand_wide.hip` (shared scaffold) -> `cand_f16.hip` via
`mkf16.py` (`build_f16abl.sh` calls it and builds the timing-only ablations). `cand_f16.hip` is committed too.
Same entry names/args as the shipped arm, so the harness swaps it in with `KB_CAND`. Harness additions: `KB_CHUNK`
(candidate work-item chunk), `abx` mode (shipped arm vs two candidates interleaved in one process), `run_f16.sh`.

**Resources (hipcc gfx1151, no spills anywhere):**

| | gate+up VGPR / LDS | down VGPR |
|---|---|---|
| shipped int8 arm (2 tiles) | 230 / 10.1 KB | 133 |
| f16, 2 tiles (32 members) | 179 / 17.3 KB | 123 |
| f16, 3 tiles (48 members) | 224 / 25.9 KB | 159 |

Static mix per super-block (`isa_mix.py`): VALU per WMMA 51.2 -> 29.6 (gate+up), 49.1 -> 28.5 (down) at 2 tiles.

**Correctness** (`cmp`, vs production kwide/kwide2, real harness inputs): gate+up rel_rmse 3.9e-7, down 1.1e-7
at 2 tiles B=512, and the same at 3 tiles / chunk 48 / B=3072 (max_group 68, so chunking is exercised). The shipped
int8 arm is 5.4e-7 / 1.7e-7, so the f16 arm is at least as close. The baseline itself matches the CPU reference
to 4e-6 / 1.7e-6. Not bit-exact (f32 reassociation, same class as the shipped arm): still needs the golden/KLD gate.

**Speed, interleaved in one process** (`abx`, B=3072, 48 experts = 48.5 members/expert, 12 rounds; the hub is
live so absolute numbers move 9-16 ms between runs for the SAME kernel; only same-process ratios are usable):

| comparison (vs shipped int8 arm) | gate+up med (p10) | down med (p10) |
|---|---|---|
| f16, 2 tiles, chunk 32 | 9.42 vs 15.99 ms = **1.70x** (9.05 vs 12.82 = 1.42x) | 4.66 vs 5.39 = 1.16x (4.56 vs 5.02) |
| f16, 3 tiles, chunk 48 | 7.57 vs 11.22 ms = **1.48x** (7.28 vs 10.57 = 1.45x) | 5.57 vs 5.44 = tie (3.43 vs 4.02 = 1.17x) |
| (f16 3 tiles vs f16 2 tiles, from the two runs) | ~1.24x | ~1.0x |

So the scale-in-LUT f16 arm is a real **~1.4-1.7x on gate+up** over what ships, and ~1.15-1.2x on down (down was
already near its DRAM roof). Wider chunks add a further ~1.2x on gate+up. At B=1024 / 16.5 members per expert the
two arms tie on gate+up (memory/latency bound there), as predicted: this lever only matters at big windows.

**Ablation (f16, 2 tiles, gate+up, B=3072):** removing the activation global loads: 8.9 ms (vs 9.2); weight loads:
8.1 (p10 6.8); both: 8.7 (p10 6.8). So even with ALL global loads removed the kernel sits at ~6.8 ms = 16 TFLOP/s =
28% of the ~57 TOPS peak: the WMMA pipe is ~1/3 busy, and it is NOT instruction count (VALU/WMMA fell 42% for ~1.4x)
nor memory. Suspects, untested: two `__syncthreads` per super-block with stage->barrier->compute serialisation
(no overlap inside a WG), LDS fragment reads (4 ds_read_b128 per tile-block), only 4 independent accumulator chains
per wave. Next experiments: double-buffered staging (stage sb+1 while computing sb), PMC (`SQ_BUSY_CYCLES`,
`SQ_INSTS_LDS`, ONE counter per pass, iGPU PMC only via the flake-pinned rocprofiler, NEVER ATT on the iGPU).

**Not yet done:** driver integration (a `V41_MOE_WMMA_F16` knob in `het::dispatch`, chunk 48 into the work-item builder,
rust wrappers, `build.rs` registration, expected default OFF), the golden/KLD gate (needs the hub down), e2e prefill
A/B, box 2 (same binary), and a scan of the remaining layers' E8M0 range (only layers 2 and 7 sampled; the kernel
clamps to e8 in [114,137], a weight outside that would silently flush/saturate).

## Cross-sub-chunk MoE batching: design (read-only code scoping, 2026-10-10; line numbers in `crates/v4flash-kernels/src/het/`)

Scoped by a read-only agent from the code (nothing built or run); per-row byte sizes are its arithmetic, not measurements.

**Where the per-lane MoE happens.** `pipelined_range_impl` (`forward_prefill.rs:2486`) runs `forward_layer_pre_moe_v2` (4813 =
`pre_moe_chain` 4848 -> `pre_moe_route` 8972 -> `pre_moe_prep` 10123 -> `pre_moe_launch` 10668) once per LANE per layer,
staggered by half a layer (steady state: `post_A(L)`, `pre_A(L+1)`, `post_B(L)`, `pre_B(L+1)`; 2740-2872): the dGPU chain of
one lane overlaps the iGPU MoE + box-2 round trip of the other. The box-2 request is built at `pre_moe_route` (submit at 10044:
`b x 5840 B` xq, `b x 6` picks, `b x 6` weights, f16 reply by default, ticket stashed per lane, nothing waits there); the iGPU
MoE builds expert member lists (`moe_group_builder_hetsplit`, `(b<<16)|slot`, atomicAdd order nondeterministic), then
gate+up / down WMMA arms, then the peer-push back; `forward_layer_post_moe_v2` (11575) adds moe, then shared, then remote
(that ORDER must be kept for a bit-compatible batch), then `hc_post`.

**There is no hub-side lane merge.** The only merge is box 2's same-layer coalescing (`V41_B2_MERGE`, `REQ_FLAG_PARTNER`,
`remote_experts.rs:2213, 8435-8520`, waits `V41_B2_MERGE_WAIT_US`=400 us for a partner). `partner_follows` is true only in
arena decode (4271), false in `pipelined_range_impl`; and lane B's request is submitted a whole chain time after A's, so
the daemon never sees a partner. (Inferred from code order, not traced.) Even when it fires it shares only box 2's pass.

**Dependency check (verified in code).** Unit k+1's layer-L attention needs layer L-1 output of its own rows and layer L's
KV / compressor / SWA ring after unit k's layer-L attention and eviction; it needs NOT unit k's layer-L MoE output. So
attention for several units at layer L, then ONE MoE, is legal. Positional state (KV/ring, compressor, index keys, Engram
rows) is safe layer-outer. What spans the MoE per lane: `after_attn_hc` (80 KB/row), `split`+`hc_pre_carry` (~0.2 KB/row),
`ffn_shared` (20 KB/row), the residuals (`lm_store` already homes them), `indexer_sel_saved` (2 KB/row, only inside a group).
Not row-independent host state: box-2 ownership split, pager `ensure`, `b2_mirror` substitution, cache-prior `V41_SUB`
(time-ordered by request order); open question whether `lm_prefill_bitexact` runs with them off.

**Memory for an M-row batch.** iGPU (GTT): ~0.35-0.45 GB at M=2048, ~0.7-0.9 GB at 4096 (`d_mid_cat` 55 KB/row,
`q2k_partials` 123 KB/row, `d_midq_cat` 15.8, `d_mid16` 27.6, xq 5.8+5.8, `ffn_moe` 20). Box 2 (expertd executor): ~0.2 GB at
1024 rows, ~0.4 GB at 2048. Host: request xq 12 MB / reply 21 MB (f16) at M=2048.

**Designs.**
* **A. Macro-lanes**: attention stays per 512-row sub-unit; the MoE runs per K staged sub-units; keep TWO macro-lanes so the
  dGPU still overlaps (2 x 2048 = exactly one default 4096-row window). dGPU footprint ~unchanged (+82 MB `ffn_shared`
  stash +8 MB index store). New driver replacing the lane loop, per-unit stashes, one consolidated box-2 request, `si`/`bi`
  sized for M. ~600-900 lines, HIGH risk (host sync ordering, `drain_before_ensure`, pager, `V41_SUB`, PREFETCH words,
  `partner_follows`). Unit latency 2-4x.
* **B. Merge the two existing lanes' MoE (1024 rows)**: serialises chain A, chain B, MoE(A+B), loses the half-layer overlap;
  ~150-250 lines. Reaches only 1024.
  * **B-hybrid (few lines, untested)**: set `partner_follows` on lane A's request in `pipelined_range_impl` so box 2 merges
    A+B (1024 rows) via the existing path; needs lane B's route/submit to happen inside the 400 us wait, i.e. reorder
    route/submit before the iGPU leg. Box-2 side only.
* **B'. Bigger lanes carved from the lent KV-arena tail** (see the VRAM bullet above): 1024-row lanes +885 MiB (fits the
  lent 2.5-2.8 GB with margin), 2048-row lanes +2.6 GiB (+ `lm_store` min window 2 x chunk = 8192 rows ~672 MB: no margin).
  Mostly constants/sizes (`B_MAX`, `KV_CACHE_ROWS`, lane alloc, `chunk_rows`, expertd `--max-batch`, which currently
  REJECTS > `B_MAX`=1024 at `deepstrix-expertd/src/main.rs:147` and `remote_experts.rs:9966`) plus the lend/return hooks
  (lend after `compact_stores` at prefill job start; return before any reserve/grow/compaction/`fill_reserved`/parked
  `try_admit`; fallback = halve the lane as `lm_store` does). Keeps the pipeline overlap. Unit latency 2x/4x. The reference
  for bit-identity moves (chunked prefill at the same lane size). UNKNOWN: whether the arena tail can be freed and
  hipMalloc'd again, or only lent as views (`dgpu_loan.rs` precedent = views via `LoanAlloc::take`, would need an allocator
  threaded through `BatchDgpuScratch::alloc_rows_ctx`).

**Box 2.** Same MoE kernels. The WMMA work item is <= 32 members (`WM_MAX_CHUNK`), so at 64 members/expert an expert's weights
are streamed ~twice; the gain from bigger batches is mainly FILLING the 32-member tile (~8 of 32 used today), which is
why the f16 arm's 48-member chunk (step 2) multiplies the batching gain.

**Recommendation (staged, cheapest information first).**
1. **Measure sensitivity before building**: June's 256 -> 512 rows/call gave +9.4% e2e, and the wall now is the box-2 round
   trip. Cheapest probe = B' at lane 1024 behind a knob, on a quiet window; compare e2e prefill, TTFT and decode p90.
   It needs only ~0.9 GB of the lent tail, and a 1024-row lane request is within box 2's current cap (`--max-batch`
   <= `B_MAX` = 1024), though a 2048-row chunk (two such lanes) changes what `B_MAX` and `KV_CACHE_ROWS` mean -- audit
   the other `B_MAX` uses first.
2. If lane 1024 moves e2e: add the f16 arm with chunk 48 (steps 2 above) and consider A (macro-lanes, K=2 over 1024-row
   lanes = 2048-row MoE) for the remaining 2x. If it does not move e2e, the wall is the round trip / wire and the right
   next work is the link (Gen3), not the MoE.
3. B-hybrid is a cheap side experiment on box 2 only; do not rely on it.

**Unverified (do not assume):** why the merge never fires (code-order inference); whether `hc_post`/`mhc` launches take
store views (avoids ~4 D2D copies of ~40 MB per unit-layer); `lm_prefill_bitexact` with `V41_SUB` / cache-prior off;
whether the arena tail can be freed and reallocated; the wire width of `b` in the box-2 header; whether the WMMA kernels
are tile-composition-independent per row (reduces to f32 association only if so).
