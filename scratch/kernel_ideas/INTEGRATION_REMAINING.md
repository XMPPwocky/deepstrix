# Remaining integration — families F, E, A+B, C2 (one agent, sequential)

> **Status (docs audit 2026-10-04):** DONE — F 3e5347b, E 35f2515, A+B 1c5972f, C2 cba1b00 (2026-09-26/27), all 17 knobs on main, default ON. Round 2 later moved four gates (343896b: `V41_F16X_256` from 192 rows, `V41_GEMV_BPACK_Z16` only n_rows >= 2048 or b >= 48, `V41_IDX_GATHER_B128` from b = 3, `V41_CAND_THRESH_ILP` up to b = 32), so the gates written below are not the current ones; read the knob doc comments in code.

Follow `INTEGRATION_BRIEF.md` for conventions, build, tests, commit format. Families D and C1 are
already integrated on this branch (commits `e282846`, and C1's); four golden-gate commits were
cherry-picked (`fidelity_tap.rs` + small hooks) — do not touch them. Work family by family in the
order below; **commit after each family** (one commit per family) so a failing gate can be
bisected by knob and by commit. Build the kernels crate after each family; the full server+expertd
build once at the end. Every knob defaults ON with `=0` restoring the exact previous path.

## F — mHC / norms / router (dGPU), candidates in `scratch/kernel_ideas/F_mhc_glue/`
1. `rms_fast` — `rms_norm_weighted_batched_fast` templates for n in {512, 1280, 5120} (NPT = n/256 per
   thread, all loads up front), replacing `rms_norm_weighted_batched` at q_a / kv / comp norms and the
   per-row head `rms_norm_weighted` (forward_head.rs). Other n -> old kernel. Knob `V41_RMS_FAST`.
2. `router_mv_h20` — router logits matvec (f16_matvec_batched, 384x5120, grid (48,1,b)) with 20 heads
   per WG (`f16_matvec_batched_h20`), decode/replay path (V41_ROUTER_WMMA off = b<=64). Knob `V41_ROUTER_MV_H20`.
3. `topk_wfred` — `router_topk_wfred` replacing `router_topk_par` ONLY on the decode path (the cache-prior /
   n_protect config); the reviewer measured a 10% LOSS at b=512, so prefill keeps `router_topk_par`.
   Gate on b <= 16. Knob `V41_TOPK_WFRED`.
4. `gemm_narrow_m24` — `f16_gemm_narrow_n16_bk128_pf2` replacing `f16_gemm_wmma_lds_tiled` at the two
   prefill mHC pre-mix sites (M=24, K=20480; FP:3717 and FP:6342), b > 64 path. Bit-exact incl. odd B.
   Knob `V41_MHC_GEMM_NARROW`.
Skip: `collapse_prefill` (refuted as stated; 1.13x — only wire it if it is a 10-line change and bit-exact; otherwise skip).

## E — indexer (dGPU), candidates in `scratch/kernel_ideas/E_indexer/`
1. `score_qreg_hw` — decode score kernel twin of `indexer_score_wmma_batched_mw_e2m1` with Q fragments
   + head weights in registers (S/indexer.rs:403 site, arena rows). Knob `V41_IDX_SCORE_QREG`.
2. `topk_select_hybrid_sort_vec4` — `indexer_topk_select_batched_ilp` twin with wave-shuffle bitonic
   stages (S/indexer.rs:883/910). Knob `V41_IDX_TOPK_HYBRID`.
3. `candidate_threshold_ilp` — `candidate_threshold` twin, block 1024, 8 loads in flight (S/candidate_blocks.rs:140). Knob `V41_CAND_THRESH_ILP`.
4. `gather_b128` — `gather_u4_r1` (16 B per thread) replacing `indexer_gather_batched` **only when b >= 4**
   (reviewer: 1.5-1.8x SLOWER at b=2); b < 4 stays on the production kernel. Also the prefill (b=512)
   sites, where it is 2.7x. Knob `V41_IDX_GATHER_B128`.

## A + B — routed MoE on the iGPU, candidates in `scratch/kernel_ideas/A_moe_smallb/` and `B_moe_prefill/`
(Same MXFP4 kernel sources serve box 1 decode/prefill AND box 2's expertd: `remote_experts.rs` batched
chain RE:4389-4497 uses the same dispatch — wire the dispatch layer so both callers get it.)
1. A `dn2_member_outer` — `cand_dn_v2.hip` (DN_ROWS=2) down twin of `mxfp4_matvec_par_by_expert_kwide2`
   (same interface/grid/lane map) for **b <= 8** (decode arena rows and expertd b<=8); reviewer: 1.26x kernel,
   1.085x chain at b=4, bit-exact at b=1..8, holds with 1-member groups. Larger b -> production kwide2
   (or the WMMA down below when its gate applies). Knob `V41_MOE_DOWN_DN2`.
   Skip `wl_cap` (REFUTED) and `kwide_c8` (gain came from empty WGs the production regime does not have).
2. B `wmma_i8_gateup` + `wmma_i8_down` — `cand_wmma.hip`: int8-WMMA gate+up (`mxfp4_pair_matvec_fused_swiglu_wmma`)
   and down (`mxfp4_matvec_par_by_expert_wmma`) for LARGE batches: reviewer numbers 1.56x gate+up /
   2.8x down at B=1024, 1.18x / 1.95x at B=512, ~1.0x / 1.31x at B=128. Gate: use them when the
   batch rows >= 256 (gate+up) / >= 128 (down) — or by rows-per-expert if the engineer's notes give a
   cleaner criterion; state the rule in the knob doc. Applies to box-1 prefill (FP:8893 / 9059) and
   expertd's batched chain (RE:4438 / 4484). **NOT bit-exact** (f32 re-association only,
   rel_rmse ~1e-7): allowed, the combined golden gate covers it. Knobs `V41_MOE_WMMA_GATEUP`,
   `V41_MOE_WMMA_DOWN`. The reviewer noted the +-10 clamp and E8M0 scale handling — copy the candidate
   verbatim, do not "clean up". Tests: `mxfp4_pair_oracle`, `mxfp4_iq2s_oracle`, `mxfp4_wi_devcount`
   (whichever exist) with the knobs on and off; the B harness has a `cmp` mode against the production
   kernels — rerun it against hsacos from the in-tree files.

## C2 — dGPU dense prefill, candidates in `scratch/kernel_ideas/C2_dense_prefill/`
1. `engram_i8x_db` — `q8_0_gemm_wmma_i8x_db` replacing `q8_0_gemm_wmma_lds_tiled` for the Engram wkv
   GEMM (FP:3526 / q8_0.rs:614), drop-in signature. Knob `V41_ENGRAM_I8X`.
2. `engram_chunk128` — Engram chunk 64 -> 128 rows (config/scratch sizing at the FP:3508-3526 loop;
   check `ENGRAM_CHUNK` in config.rs and any scratch buffer sized by it). Reviewer: 1.89x on top of item 1.
   Knob `V41_ENGRAM_CHUNK128` (must also work with item 1 off — measure nothing, just keep the old kernel
   correct at 128 rows or fall back to 64 when item 1 is off).
3. `replay_bpack_z16` — `q8_0_gemv_bpack_z16` / `q8_0_grouped_gemv_bpack_z16` (production bpack16 body,
   grid.z = ceil(b/16)) replacing `q8_0_gemv_batched_warp8` / `q8_0_grouped_gemv_batched` when
   16 < b <= 64 (the replay regime; q8_0.rs:258-269, 869-876). Bit-exact. Knob `V41_GEMV_BPACK_Z16`.
4. `f16x_db_bn64` — `q8_0_gemm_wmma_f16x_db_bn64` for kv (always at b > 64) and q_a only when
   M <= 1280 && b <= 512 (reviewer: q_a loses 4% at b=1024). Knob `V41_F16X_DB_BN64`.
5. `f16x_256x128` — `q8_0_gemm_wmma_f16x_256x128` for qb and wo_a (precondition M % 256 == 0, true at
   both). Knob `V41_F16X_256`.
6. `replay_f16x_b64` — lower the dp4a/f16x threshold `V41_PREFILL_F32_MATVEC` from 64 to 16 for qb,
   wo_a, wo_b ONLY (kv stays on the dp4a path: f16x loses there). This is a threshold/default change,
   NOT bit-exact (f16 activations at replay, rel_rmse 2.9e-4 vs dp4a) — implement as its own knob
   `V41_REPLAY_F16X` (default ON) that overrides the per-site threshold, so the gate can flip it alone.
Skip: `f16x_pf2` and `i8x_r1_codegen` (unreviewed).

## Return
Per family: items + knobs, skipped items, tests run + results, non-bit-exact items and shape gates
the gate must watch, deviations from the ledger. Plus: the full list of ALL sweep knobs on the
branch (D's and C1's included — read their commit messages) with the value that restores the OLD
path for each, so the orchestrator can run the knobs-off baseline.
