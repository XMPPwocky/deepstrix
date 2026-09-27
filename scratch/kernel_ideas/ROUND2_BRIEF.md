# Round 2 — pattern-driven kernel wins (one agent, sequential)

Context: `LEDGER.md` (round 1: 81 ideas, 28 confirmed, all integrated on this branch and DEPLOYED
2026-09-27 00:38 UTC behind 24 `V41_*` knobs — read the family commit messages e282846, c997165,
3e5347b, 35f2515, 1c5972f, cba1b00 for what is now production). `INVENTORY.md` = kernel map.
`_infra/README.md` = rules and tools (scheduler, kbench.h, isa.sh, prof.sh). `INTEGRATION_BRIEF.md`
= conventions for wiring (knobs default ON, `=0` restores the old path, bit-exact tests).

**The hub is LIVE on this box** and box 2 is live: every GPU job through `_infra/gpu_run.sh` /
`gpu_submit.sh`, <= 130 MB dGPU / <= 600 MB iGPU per job, one at a time, no model loads, no
rocprofv3 ATT on the iGPU (dGPU ATT is fine). Timing is under production load: use `kb::ab`
interleaved ratios only (both variants see the same load), >= 5 separate runs, report p10 as well
as median, and treat < 8% as unresolved. Baseline = the PRODUCTION code object built from the
in-tree source with `$KFLAGS_V41` (now including round-1 kernels: the baseline for a decode
attention idea is `attention_dec_fused_vt_qreg_sp_d4`, for a GEMV it is the `tB` twin, etc.).

Work the items in order; for each: measure, and if it wins (bit-exact unless stated), WIRE it into
the tree behind a knob with a bit-exact test, and commit (one commit per item, format as the round-1
commits, `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`). Do not push. Do not deploy.
Losses/neutrals: record in `scratch/kernel_ideas/round2/NOTES.md` with numbers and move on.
Time-box each item; the list is ordered by expected value / effort.

## (a) Power-of-two dispatch-granularity scan  [decode + prefill, dGPU, bit-exact by construction]
Round-1 finding (C1 `quantize_grid_pad`, CONFIRMED 2.27x): a launch whose grid is exactly a power
of two >= ~1024 WGs pays ~10 us of extra dispatch time on gfx1201; `blocks + 1` (an idle WG, kernel
guard makes it a no-op) removes it. Enumerate every production launch on the dGPU with an exact
2^k grid at production shapes (start from INVENTORY.md grids: e.g. the qb bpack GEMV = 4096 WGs at
32768/8; quantize at K=32768 is already padded; check head 16160 (no), wo_b 640 (no), attention,
indexer gather (512, b, dim_blk), rope (n_head=64, 1, b) x 32 threads, kv_cache_append, hc_post
(20,4,b)…). For each candidate: measure grid vs grid+1 in the C1 harness style (graph mode, cold
weights where streamed), and if it wins wire the pad in the wrapper with the same "DO NOT REMOVE"
comment. Also test whether the effect exists on gfx1151 (one iGPU launch with a 2^k grid).

## (b) Crossover scan of the deployed knobs  [defensive; both GPUs]
For each round-1 kernel with a shape gate or none, measure candidate-vs-old at b = 1,2,3,4,5,6,8,
12,16 (decode/replay) and 64/128/512/1024 where the kernel runs at prefill, using the family
harnesses that already exist (`<family>/harness*`, `intree/` scripts). Any b where the new kernel
LOSES > 5% => add/adjust the wrapper gate (commit). Known holes to re-check: gather_b128 (b=2 gated),
grouped tB1 (gated), f16x_db_bn64 q_a at b=1024 (gated), topk_wfred b>16 (gated), kwide_c8 not
integrated. Report a table.

## (c) z16 chunking for the remaining grid.z=B kernels  [bit-exact]
Round-1 C2 `replay_bpack_z16` (CONFIRMED 2.6x): re-reading weights once per row (grid.z=B) -> once
per 16-row chunk. Apply the same transform to: the ratio-1 compressor `f16_matvec_batched` at L20+
(K/f16_matvec.hip:157, FP:4299; ~2 ms per 512 rows), the idx-q `f16_matvec_batched` 4096x1280 at
decode/replay (FP:5390) and the proj 32x5120 (FP:5440), the router `f16_matvec_batched` at replay
(FP:6451, b<=64). Measure at b = 2, 4, 8, 16, 64, 128. (D's family listed `compressor_ratio1_tiled`
untested — the pair_tiled kernel for ratio-2 layers shows the pattern.)

## (d) Decode launch-count fusions + hoisted head prep  [dGPU, bit-exact, latency-bound]
Per lane-layer at decode ~35 launches remain at ~2.5 us graph-node cost each plus fill/drain.
Candidates (verify each one's adjacency in FP first): (1) kv chain fp8_act_quant_inplace +
kv_cache_append_batched (+ rope_tail where adjacent) as one (b)x512 launch; (2) rms_norm_weighted
+ q8_0_quantize pairs (q_a input, kv input) as one launch (rms_fast already hoists loads — fuse
the quantize epilogue); (3) hc_post_from_split + vec_add_inplace; (4) router matvec_h20 +
router_topk_wfred + readback_pack. (5) The per-row head prep hc_weighted_sum (20)x256 +
rms_norm_weighted (1)x256 (~62 us/row, forward_head.rs:57/69): hoist loads (F `rms_fast` style)
and/or batch across rows bit-identically per row. Measure in graph mode as a chain (old N launches
vs new), warm activations. Fusion caveat from project memory: keep multi-WG shapes; never collapse
a multi-WG matvec into one WG per token (that regressed 24x before).

## (e) int8-activation WMMA GEMM for the dGPU dense prefill sites  [prefill, NOT bit-exact]
C2's `q8_0_gemm_wmma_i8x_db` (Engram, 1.87x, int8 activations + per-row xscale) generalised to the
`q8_0_gemm_wmma_f16x` sites at b=512: qb (32768x1280), wo_a (8 groups 1024x4096), wo_b, shared
gate/up/down. These currently run at ~50% of the 194 TF matrix roof with global/LDS/WMMA phases
serialising (C2 notes). Numerics change (f16 -> int8 activations: same class as the dp4a path
production already uses at b<=16 and as REPLAY_F16X), so: measure, wire behind `V41_DENSE_I8X`
DEFAULT OFF, report rel_rmse vs f16x and vs an f32 reference, and leave it for the gate.

## Return
Per item: what was measured (shapes, regime, runs), the numbers (median + p10 ratio), wired or not,
knob name, test, commit hash; the crossover table from (b); the updated knob list (round-1 + yours)
with old-path values; and the ranked "next" list of what you saw but did not do.
