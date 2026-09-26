# DeepSeek V4.1-Flash production GPU kernel inventory (commit 361d4f9)

Read-only trace (2026-09-26) of every kernel on the V4.1 production paths: multistream decode,
chunked CED prefill, CED replay, the expertd decode and batched branches, vision, DSpark.

## 0. Conventions, shapes and regimes

**Paths.** ROOT = this worktree. `K/` = ROOT/crates/v4flash-kernels/kernels/; `S/` =
ROOT/crates/v4flash-kernels/src/; `FP` = S/het/forward_prefill.rs; `RE` = S/het/remote_experts.rs;
`Dd/` = ROOT/docs/v41/; `MS` = ROOT/crates/deepstrix-server/src/multistream.rs.

**Devices.** dG = HIP0 gfx1201 (RX 9070 XT). iG1 = box-1 HIP1 gfx1151. iG2 = box-2 gfx1151
running `deepstrix-expertd` (same kernels, same source).

**Phases.** D = arena decode (`forward_step_arena*` → `pre_moe_chain`/`pre_moe_launch`,
`RowLayout::Arena`). P = prefill chunk: CED encoder L0-19 Exact + L20 KvSourceOnly (FP:638-643).
R = CED replay: last ≤128 rows through L20-39 (FP:740-742). X = expertd. V = vision. S = DSpark.

**Model shapes (S/config.rs).** n_embd 5120 (:12); HC_DIM 20480; mix 24 (:14-15). 64 heads × 512,
n_rot 64 (:19-20). q_lora 1280 (:25); Q_FLAT 32768. wo_a 8 groups × (4096→1024); OUT_LOW 8192 (:60).
Expert FFN 2304 (:71); shared expert 2304 (:67). 384 experts, top-6 (:75-76). Vocab 129280 (:80);
40 layers (:84). Compress ratios: L0-1 = 0, L2-19 = 2, L20-39 = 1 (:138). KV sources {2,8,14,20}
(:146). Index sources {2,8,14,20,24,28,32,36} (:148). Engram layers {1,14} (:150); Engram wkv
25600×6144 Q8_0 (:29-30). Indexer 32 heads × 128, top-512 (:114-116). Candidate pool 2048 blocks × 8,
built at L20 (:193-197). Window 128 (:126). No hash-router layers under v41 (:204-206).
Bytes: one MXFP4 expert = 12.53 MB gate+up + 6.27 MB down (Dd/KERNEL_PERF_REVIEW.md:24); non-routed
Q8_0 weights ≈178 MB/layer (:27).

**Row regimes.**
- **D:** B = 1-8 rows per step. Below 6 rows one lane; ≥6 rows two lanes (MS:868
  `V41_MS_PIPELINE_MIN_ROWS=6`; MS:878 `V41_MS_LANES=2`), so b per lane is 1-5 (typically 3-4).
  Each lane runs all 40 layers → 80 lane-layers per step at ≥6 rows (Dd/AUDIT_2026-09-22.md:24).
  Each stage is replayed as a HIP graph (FP:944-948); ~40 kernels per lane-layer (FP:754-757).
- **P:** 1024-row chunks (MS:116) split into 2 lanes of 512 (crates/deepstrix-server/src/engine_worker.rs:1091).
- **R:** ≤128 rows over 2 lanes, ~64 rows each.
- **X:** expertd `--decode-max-b 1`. Branch at RE:4163: b=1 → decode chain (RE:4252-4275);
  b≥2 → by-expert batched chain (RE:4276-4337, `batched_pass` RE:4389-4497). Both hub lanes of a
  layer are merged into one pass (`V41_B2_MERGE`, RE:1754-1766; RE:4960-4975). Hub sets
  `REQ_FLAG_BATCHED` on multi-row submits (RE:2084-2108), always asks f32 replies (FP:7856).
- Sampling in multistream is host-side (MS:1224-1273); device sampler kernels are DSpark-only.

**Env knob resolution (production defaults, read from code).**

| knob | resolved | code | effect |
|---|---|---|---|
| V41_MHC_FAST / _ARENA_FUSED / _FFN_LATE | on / on / on | FP:830-834, 808-812, 871-875 | decode mHC = `mhc_fast_batched` |
| V41_MS_MHC_SPLIT | off | FP:788-800 | — |
| V41_MHC_PRE_SCALED / V41_MHC_NARROW | b≤8 / b≤64 | FP:398-412 | prefill b=512 uses the WMMA chain |
| V41_ATTN_META_FILL / V41_ATTN_DEC_SCORE | on, b≤16 | FP:843-847, 858-862 | decode attention kernels |
| V41_TOPK_SELECT_ILP; INDEXER_TOPK_SELECT | on; on | S/indexer.rs:669-673; FP:5622 | `indexer_topk_select_batched_ilp` |
| INDEXER_SCORE_VARIANT | gemm | FP:5474 | arena rows forced onto `mw_e2m1_rows` (FP:5483-5500) |
| V41_INDEX_K (prod =1), V41_CANDIDATE_POOL (prod =1) | default off | FP:101-106; S/het/forward_layer.rs:146-151 | sparse indexer and candidate pool active |
| V41_SWA_MIXED | on | FP:279-282 | ratio-0 layers use the mixed WMMA pair |
| V41_ENGRAM_GEMV | off | FP:268-270 | prefill Engram uses LDS-tiled WMMA |
| V41_SMALL_B_DENSE_DP4A / _MAX | on, ≤8 | S/het/dispatch.rs:244-265 | decode dense = dp4a bpack |
| V41_GEMV_BPACK | on, b≤16 | S/q8_0.rs:40-50, 258-260 | `matvec_batched` → `q8_0_gemv_bpack_warp8` |
| V41_PREFILL_F32_MATVEC | b≤64 | FP:381-387 | qb/kv/wo_a/wo_b "dp4a" for b≤64, "f16x" above |
| V41_ROUTER_WMMA | b>64 | FP:6435-6439 | router f32 matvec at decode and replay |
| V41_RB_PACK | on, ≤16 rows | FP:361-379 | `readback_pack` |
| V41_MOE_WI_DEVCOUNT | on | FP:883-887 | kwide grids are upper bounds |
| PAIR_VARIANT; IGPU_MOE_WMMA; Q2K_VARIANT | kwide; n/a for MXFP4; kwide2 forced | dispatch.rs:323-365, 339-354; FP:9003-9010 | MXFP4 MoE = kwide + kwide2 |
| DEEPSTRIX_COMP_TILED / _GEMM / _GATHER | on / off / on | FP:4309, 4249-4251, 4395-4397 | compressor kernels |
| ATTN_FUSED, DEEPSTRIX_F32_SCORES | off, off | FP:5771; S/het/batch_scratch.rs:75-81 | f16-scores split attention |
| V41_PAGED_EXPERTS=1 | no dGPU hot experts | S/het/weights.rs:1451-1458 | dGPU never runs routed experts |
| DEEPSTRIX_SEL_STATS | on | S/het/engine.rs:1720-1725 | `expert_sel_count` every layer |
| V41_PAGER_GPU_REPACK / V41_B2_GPU_REPACK | on / on | S/het/expert_pager.rs:542; RE:2076-2080 | `mxfp4_repack_hf_to_ggml` per miss |
| V41_MXFP4_PAIR_WARPS | 8 | S/mxfp4_pair.rs:25-34 | decode pair grid |
| V41_VERIFY_DECODE_MOE / _ATTN | off | FP:9930-9942 | — |

V41_DSPARK is 0 in production (engine_worker.rs:982-988): DSpark kernels are secondary.

## 1. Kernel inventory by family

Row format: kernel `sym` @ .hip:line | dev·phase | wrapper ← call site | grid×block @ typical shape | knob | measured.

### A. Routed MoE, small-B (decode). iG1 and iG2; MXFP4 kernel source shared with family B.

| sym @ K/ | dev·phase | wrapper ← call site | grid×block @ shape | knob | measured |
|---|---|---|---|---|---|
| `mxfp4_pair_matvec_fused_swiglu_kwide` mxfp4_pair_matvec.hip:395 | iG1 D,P,R; iG2 X b≥2 | S/mxfp4_pair.rs:250 ← dispatch.rs:507 ← FP:8893; RE:4438 | (288, n_wi)×256. n_wi = min(6b, G+⌈6b/32⌉) (dispatch.rs:420-424). Gate+up 2×2304×5120 per expert | kwide default; devcount | no small-B kernel timing. iG2 batched chain srv p50 386/396/444/465/487/527 us/layer at B=1/2/4/5/6/8 with a 3-expert pool (RE:2095-2099) |
| `mxfp4_matvec_par_by_expert_kwide2` mxfp4_matvec.hip:219 | same | S/mxfp4.rs:147 ← FP:9059; RE:4484 | (320, n_wi)×256. Down 5120×2304 | MXFP4 = kwide2 only | — |
| `mxfp4_pair_matvec_fused_swiglu_batch_hetsplit` mxfp4_pair_matvec.hip:238 | iG2 X b=1 only; S drafter | S/mxfp4_pair.rs:113 ← dispatch.rs:113 ← RE:4266; het/mtp.rs:1308 | (288, 6)×256 | decode_max_b=1 (expertd main.rs:96,119) | 218 us p50 per 3 experts = 173 GB/s (Dd/ROADMAP_2026-09-18.md:145). 189/201 GB/s ≈ 93% of 214 GB/s with a rotating selection; **CLOSED** (K/mxfp4_pair_matvec.hip:61-72; ROADMAP:123-133) |
| `mxfp4_matvec_par_batched_hetsplit` mxfp4_matvec.hip:158 | iG2 X b=1; S | S/mxfp4.rs:75 ← dispatch.rs:179 ← RE:4271 | (640)×256 | same | 91 us p50 = 206 GB/s, 96% of 214 (ROADMAP:146) |
| `q8_k_quantize` q8_k_quantize.hip:23 | iG1 pre+mid; iG2 mid; dG (box-2 xq) | S/q8_k.rs:31 ← FP:8503, 8977; RE:4270, 4447; FP:6685 | (20b)×256 pre; (54b)×256 mid | — | 116 us at B=1024, 232 GB/s (Dd/PREFILL_100K_PROFILE.md:219) |
| `moe_group_builder_hetsplit` moe_group_builder.hip:73 | iG1 D,P,R; iG2 X b≥2 | S/moe_group_builder.rs:205 ← FP:8587; RE:4408 | (⌈6b/512⌉)×512 | — | ~14 tiny 20-40 us kernels per daemon request (ROADMAP:141) |
| `moe_work_items_builder` moe_work_items_builder.hip:26 | same | S/moe_group_builder.rs:72 ← FP:8805; RE:4414 | (⌈bound/256⌉)×256 | devcount removes the host readback (FP:8815-8835) | `lh.work_items_count` was 90.7 ms/step before devcount (FP:9584) |
| `q2_k_reduce_partials_hetsplit` q2_k_accumulate_matvec_par.hip:737 | same | S/q2_k.rs:473 ← FP:9137; RE:4509 | (⌈5120b/256⌉)×256 | — | plain twin: 641 us at B=1024, 229 GB/s (PREFILL_100K:220) |
| `mxfp4_repack_hf_to_ggml` mxfp4_repack.hip:35 | iG1 pager miss, iG2 miss (D and P) | S/mxfp4_repack.rs:40,123 ← het/expert_pager.rs:2532-2590; RE:3797 | one launch per 18.8 MB expert | GPU repack on | 0.04 ms/expert; miss 10.57 → 6.60 ms (Dd/TUNING.md:37) |

### B. Routed MoE, large-B (prefill). Same symbols as rows A1, A2 and A5-A8.

- **Shapes.** iG1 b = 512 per lane. iG2 512-1024 rows per pass (lanes merged, RE:4960-4975);
  two-pass hits-first when `V41_B2_HITS_FIRST=1` (RE:4174, default off, RE:2271-2277).
- **Measured at B=1024, all 384 experts** (Dd/PREFILL_100K_PROFILE.md:217-218): kwide 39.1 ms,
  124.6 GB/s; kwide2 31.8 ms, 80.1 GB/s = 54% and 35% of the ~230 GB/s achievable.
  **Caveat:** at 230 GB/s the kernels would need 94% / 88% of non-dual-issue fp32 peak
  (PREFILL_100K_PROFILE:226-236) — the bandwidth headroom may be illusory; a compute roofline is needed.
- **Box-2 per request at B=512:** gpu p50 22.1 ms (PREFILL_100K_PROFILE:146).
- **Known structural items** (Dd/KERNEL_PERF_REVIEW.md:42, 87-102): MXFP4 has no WMMA arm;
  kwide2 re-reads activations ~13× the weight bytes; experts are streamed once per lane.

### C1. dGPU dense projections, decode (weight streaming)

| sym @ K/ | dev·phase | wrapper ← call site | grid×block @ shape | knob | measured |
|---|---|---|---|---|---|
| `q8_0_gemv_bpack_warp8` q8_0_matvec.hip:215 | dG D, S | S/q8_0.rs:301 (via `matvec_batched` :206, bpack_ok :44) ← dispatch.rs:302 (q_a FP:3832; shared gate/up/down FP:1847/1856/1901), FP:3892 qb, 3984 kv, 6167 wo_b, 3524 Engram wkv; head S/het/forward_head.rs:94 | (n_rows/8)×256: q_a 160, qb 4096, kv 64, wo_b 640, shared 288/288/640, Engram 3200, head 16160 WGs; b=1-5 per lane | SMALL_B_DENSE ≤8; PREFILL_F32_MATVEC ≤64 (FP:3878, 6158) | at B=1: q_a 13.5 us vs WMMA 96; shared gate 15 vs 89; down 15 vs 66. At B=8: 24/32/31 us (dispatch.rs:245-252). Head: 703 MB in 1112 us ≈ 630 GB/s (K/q8_0_matvec.hip:203-205; forward_head.rs:17-19) |
| `q8_0_grouped_gemv_bpack` q8_0_grouped_matvec.hip:194 | dG D, S | S/q8_0.rs:760 (via :821) ← FP:6132 | (1024)×256, wo_a 8×(1024×4096) | Q8_GROUPED_VARIANT=dp4a when b≤64 (FP:6123-6124) | — |
| `q8_0_quantize_f32` q8_0_matvec.hip:150 | dG D (6-7 per lane-layer), P (Engram), S | S/q8_0.rs:110 / :74 ← FP:3811, 3979, 3882, 6126, 6161, 1828, 1882, 3507; forward_head.rs:82 | (K/32·b)×32 | — | — |
| `swiglu` swiglu.hip:17 | dG D,P,R | S/ffn.rs:50 ← FP:1868 | (⌈2304b/256⌉)×256 | — | — |

### C2. dGPU dense projections, prefill (WMMA GEMMs)

| sym @ K/ | dev·phase | wrapper ← call site | grid×block @ b=512 | knob | measured |
|---|---|---|---|---|---|
| `q8_0_gemm_wmma_f16x` q8_0_matvec_wmma.hip:574 | dG P; R (q_a and shared only) | S/q8_0.rs:568 ← dispatch.rs:305 (q_a, shared ×3), FP:3887 qb, 3990 kv, 6129 wo_a (n_groups=8), 6164 wo_b | (⌈b/128⌉, M/128, g)×256: q_a (4,10), qb (4,256), kv (4,4), wo_a (4,8,8), wo_b (4,40) | QB_WMMA / Q8_GROUPED / Q8_OUT = f16x when b>64 | 4.4× the lds_tiled kernel (dispatch.rs:223). Grouped: 52% of matrix peak (FP:6105-6106) |
| `q8_0_gemm_wmma_lds_tiled` q8_0_matvec_wmma.hip:193 | dG P (L1, L14) | S/q8_0.rs:614 ← FP:3526 | (400, 1)×128 per 64-row Engram chunk (8 chunks per lane) | V41_ENGRAM_GEMV off | 8.8 → 1.4 ms (FP:3508-3518; KERNEL_PERF_REVIEW:83-85) |
| `f32_to_f16_cast_2d` q8_k_quantize.hip:118 | dG P (consumed); **D (dead)** | S/q8_k.rs:116 ← FP:3799, 3860, 6093, 6147, 1833, 1887 | (⌈rows·cols/256⌉)×256 | — | — |
| `q8_0_gemv_batched_warp8` q8_0_matvec.hip:270; `q8_0_grouped_gemv_batched` q8_0_grouped_matvec.hip:145 | dG R (b≈64 exceeds the bpack cap of 16) | S/q8_0.rs:261, 869 ← FP:3892, 3984, 6132, 6167 | (n_rows/8, 1, b): each grid.z slice re-reads the weight | PREFILL_F32_MATVEC ≤64 | re-read hazard: KERNEL_PERF_REVIEW:41, 78-81 |
| `q8_0_gemv_warp8` q8_0_matvec.hip:64 | dG P (last-token head, once per request); S | S/q8_0.rs:148 ← dispatch.rs:208 ← forward_head.rs:216; het/mtp.rs:1590 | (16160)×256 | — | 1112 us/call (forward_head.rs:17-19) |

### D. dGPU attention and window/compressed KV production

| sym @ K/ | dev·phase | wrapper ← call site | grid×block @ shape | knob | measured |
|---|---|---|---|---|---|
| `attention_dec_score_htiled_wmma_f16s` attention_dec.hip:40 | dG D, S | S/attention_dec.rs:32 ← FP:5176 (L0-1), 5805 (compressed) | (⌈n_tot/256⌉, 4, b)×512; n_tot ≈ 128 + min(n_comp, 512) | DEC_SCORE on, b≤16 | 21.0 → 13.6/13.9/16.0 us at b=1/2/3, incl. a 4.6 us event pair (K/attention_dec.hip:21-26); −7.5 us per lane-layer (commit 98bdd94) |
| `attention_mixed_score_batched_htiled_wmma_f16s` attention_mixed.hip:868 | dG P,R | S/attention.rs:926/947 ← FP:5195, 5824 | (⌈n_tot/256⌉, 1, B)×512 | b>16 | 7.3× over f32 (FP:5308-5310). 100K prefill `k.attn.score` 2.80 s (PREFILL_100K:26) |
| `attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_f16s` attention_mixed.hip:2278 | dG D,P,R,S | S/attention.rs:1097/1117 ← FP:5216, 5889 | (4, b)×512 (16-head tile, attention.rs:381) | — | −23% vs `_ldsv`: 13.05 → 10.09 ms at 32K, B=256 (FP:5846-5849). Decode ~16 us, 40 serial staged V tiles (commit 98bdd94). 100K `k.attn.smwsum` 3.47 s (PREFILL_100K:25). **LDS/occupancy axis exhausted** (KERNEL_PERF_REVIEW:160-164; `_ldsv_db` occupancy collapse at K/attention_mixed.hip:1321-1330, 1535-1540) |
| `attn_meta_fill` attn_meta.hip:20 | dG D (b≤16) | S/attn_meta.rs:62 ← FP:5136 | (1)×32 | META_FILL on | −13..−20 us per lane-layer (commit 4eb4b69) |
| `kv_cache_append_batched` kv_cache_append.hip:53 | dG D,P,R | S/kv_cache_append.rs:125 ← FP:4184 (arena rows) / 4201 | (b)×512 | — | — |
| `fp8_act_quant_inplace` fp4_kv_quant.hip:72 | dG D,P,R | S/fp4_kv.rs:47 ← FP:4024 | (b)×512 | — | noise (KERNEL_PERF_REVIEW:177) |
| `f16_roundtrip` f16_roundtrip.hip:8 | dG D,P,R | S/compressor.rs:590 ← FP:4038 | (⌈512b/256⌉) | — | numerically a no-op after the V4.1 FP8 window (KERNEL_PERF_REVIEW:48) |
| `rope_tail_batched` rope_tail.hip:69 | dG D,P,R,S | S/rope.rs:194/209 (launch :256) ← FP:3929 (q), 4009 (kv), 4691 (index-K), 4721 (comp), 5414 (idx q), 6080 (inverse) | (n_head, 1, b)×32 | — | — |
| `f16_matvec_pair_batched_tiled` f16_matvec_pair.hip:180 | dG D,P (L2, 8, 14) | S/f16.rs:185 ← FP:4310 | (64, 1, ⌈b/8⌉)×256; wkv+wgate 2×512×5120 f16 | COMP_TILED on | ~1 ms/layer per 512 rows (KERNEL_PERF_REVIEW:168-171) |
| `f16_matvec_batched` (ratio-1 compressor) f16_matvec.hip:157 | dG D, P (L20) | S/f16.rs:468 ← FP:4299 | (64, 1, b)×256 | — | grid.z=B re-read, ~2 ms per 512 rows (KERNEL_PERF_REVIEW:49) |
| `compressor_state_write_batched` compressor_state_write.hip:36; `compressor_snapshot_gather` compressor_state_snapshot.hip:49 | dG D,P / P | S/compressor.rs:499 / :378 ← FP:4418, 4507 / 4470 | (·, b) / (·, rows, n_bnd) | COMP_GATHER on | — |
| `compressor_pool_batched` compressor_pool.hip:79 | dG D,P | S/compressor.rs:127 ← FP:4611, 4622 | (1, n_bnd)×512 | — | — |
| `fp4_kv_quant_inplace` fp4_kv_quant.hip:25; `comp_kv_append_batched` comp_kv_append.hip:25 | dG D,P | S/fp4_kv.rs:29 ← FP:4737; S/comp_kv_append.rs:83 ← FP:4752 | (n_bnd)×512; (1, n_bnd)×512 | — | — |

### E. dGPU CSA2 indexer and candidate pool

The indexer runs on the 8 index-source layers when n_comp > 512 (FP:5097-5118); the other 30
compressed layers only gather (S2 reuse).

| sym @ K/ | dev·phase | wrapper ← call site | grid×block @ shape | knob | measured |
|---|---|---|---|---|---|
| `f16_matvec_batched` (idx q, 4096×1280) f16_matvec.hip:157 / `f16_gemm_wmma_lds_tiled` f16_gemm_wmma.hip:31 | dG D,R / P | S/f16.rs:468 / :79 ← FP:5390 / 5400 | (512, 1, b)×256 / (64, B/64)×128 | PREFILL_F32_MATVEC | — |
| `indexer_fp4` indexer_qat.hip:83 | dG D,P,R | S/indexer.rs:104 ← FP:5433 | (32b)×128 | v41 path (no Hadamard) | — |
| `f16_matvec_batched` (proj 32×5120), `vec_scale_inplace` vec_scale_inplace.hip:12 | dG D,P,R | S/f16.rs:468 ← FP:5440; S/indexer.rs:1216 ← FP:5452 | (4, 1, b)×256 | — | — |
| `indexer_score_wmma_batched_mw_e2m1` indexer_score_wmma.hip:909 | dG D | S/indexer.rs:403 ← FP:5489 | (n_chunks, b)×256, per-row key bases | forced for arena | mw vs 1-wave: 32.6 → 7.45 ms at 96K (FP:5461-5464) |
| `indexer_score_wmma_gemm_e2m1` (+`f32_to_f16_cast` q8_k_quantize.hip:96) indexer_score_wmma.hip:1048 | dG P,R,S | S/indexer.rs:496 ← FP:5504 (cast FP:5502) | (n_splits, groups)×256 | variant=gemm | 68% of matrix peak vs 15% for mw (FP:5470-5471) |
| `candidate_block_max`, `candidate_threshold` candidate_blocks.hip:24/54 | dG D,R (L20) | S/candidate_blocks.rs:140 ← FP:5594 | (⌈nb/256⌉, b); (b) | CANDIDATE_POOL=1 | — |
| `candidate_mask_apply` candidate_blocks.hip:115 | dG D,R (L24/28/32/36) | S/candidate_blocks.rs:170 ← FP:5606 | (⌈n/256⌉, b)×256 | same | — |
| `indexer_topk_select_batched_ilp` indexer_topk_bitonic.hip:796 | dG D,P,R | S/indexer.rs:883/910 ← FP:5623 | (b)×1024 | ILP on | ~185 us/launch at n=235K before ILP. ILP saves 9.5/20.6/56.4 us at 65K/131K/235K. ~65 us remains, mostly in the two bitonic sorts (commit 3feaeb6) |
| `indexer_topk_{chunk,regroup,merge}_4096_batched`, `indexer_topk_bitonic_4096_batched` indexer_topk_bitonic.hip:304/489/355/242 | dG (rows the select leaves undecided) | S/indexer.rs:962-1060 | (n_chunks, b)×1024 etc. | — | see above |
| `indexer_gather_batched` indexer_gather.hip:43 | dG D,P,R (index sources + all reuse layers) | S/indexer.rs:1151 ← FP:5653, 5728 | (512, b, dim_blk) | — | 100K prefill: `prefill_indexer` 3.16 s + `_reuse` 4.05 s (PREFILL_100K:23-24; stage times) |
| index-K producer: `f16_matvec_batched`/`f16_gemm_wmma_lds_tiled` 128×512, `rms_norm_weighted_batched`, `rope_tail_batched`, `index_kv_append_e2m1_batched` index_kv_e2m1.hip:93 | dG D,P (KV sources) | FP:4659/4669, 4679, 4691; S/index_kv_e2m1.rs:123 ← FP:4703 | (1, n_bnd)×128 | V41_INDEX_K | — |

### F. dGPU mHC, norms, router, combine glue (latency-bound small ops)

| sym @ K/ | dev·phase | wrapper ← call site | grid×block | knob | measured |
|---|---|---|---|---|---|
| `mhc_fast_batched` mhc_fast.hip:199 | dG D (3 per lane-layer); R (pre-ffn); S | S/mhc_arena.rs:85 ← FP:3602 (pre-attn, (26,1,b)), 6278 (pre-ffn collapse, (1,1,b)), 6785 (late mix, (24,1,b)); ×256 | MHC_FAST | pre_attn 51.8 → 22.8 us, pre_ffn 24.7 → 15.9, mix_late 49.6 → 22.7; −65 us per lane-layer. The kernel itself is ~11 us of a ~23 us stage; the rest is the event pair plus graph launch (K/mhc_fast.hip:55-60; commit cd91dad) |
| prefill mHC chain: `rms_norm_no_weight_batched` rms_norm_no_weight.hip:57, `f16_gemm_wmma_lds_tiled` (M=24, K=20480), `hc_sinkhorn_par_batched` hc_sinkhorn_par.hip:131, `hc_weighted_sum_batched` hc_weighted_sum.hip:28, `rms_norm_weighted_batched` rms_norm.hip:52 | dG P (×2 per layer) | S/rms_norm.rs:363 ← FP:3644, 6319; S/f16.rs:79 ← FP:3717, 6342; S/head.rs:237 ← FP:3730, 6355; S/head.rs:140 ← FP:3749, 6371; S/rms_norm.rs:137 ← FP:3772, 6381 | GEMM grid (1, B/64)×128 = **8 WGs at B=512**; sinkhorn (B)×16 | MHC_NARROW off for b>64 | **478 us/call at both b=6 and b=512** (FP:303-309). 100K `dgpu.mhc_pre_attn` 6.94 s stage (PREFILL_100K:20) |
| R pre-attn: `f16_matvec_narrow_batched` f16_matvec_narrow.hip:309 | dG R | S/f16.rs:511 ← FP:3707 | (24, 1, b)×256 | narrow ≤64 | MALL re-read at large B (KERNEL_PERF_REVIEW:44) |
| `hc_post_from_split_batched` hc_post.hip:68 | dG D,P,R,S (×2 per layer) | S/head.rs:347 ← FP:6200, 9525 | (20, 4, b)×256 | — | bandwidth-bound, adequate (KERNEL_PERF_REVIEW:178) |
| `rms_norm_weighted_batched` (q_a/kv/comp norms) | dG D,P,R | ← FP:3848, 3996, 4632 | (b)×256 | — | — |
| router: `f16_matvec_batched` 384×5120 / `f16_gemm_wmma_lds_tiled` | dG D,R / P | ← FP:6451 (48, 1, b) / 6441 (6, B/64) | V41_ROUTER_WMMA b>64 | 3.9 MB, ~10 us (KERNEL_PERF_REVIEW:172-173) |
| `router_topk_par` router_topk_par.hip:66 | dG D,P,R | S/router_topk.rs:222 (get :322) ← FP:6539 | (b)×512 (ROUTER_MAX_EXPERTS=512, build.rs) | — | ~10-15 us: 6 argmax passes × 10 barriers (KERNEL_PERF_REVIEW:49) |
| `readback_pack` readback_pack.hip:17 | dG D | S/readback_pack.rs:90 ← FP:6743 | (⌈words/256⌉)×256 | RB_PACK on | replaced a copy batch that cost ~0.6 ms/step; shared expert 3.7 vs 9.0 ms/step (FP:6693-6704) |
| `expert_sel_count` expert_sel_count.hip:17 | dG D,P,R | S/expert_sel_count.rs:34 ← engine.rs:1743 ← FP:6755 | tiny | SEL_STATS on | — |
| `vec_add_inplace` vec_add.hip:6 | dG D,P,R (×2 per layer) | S/ffn.rs:135 ← FP:9311, 9484 | (⌈5120b/256⌉) | — | — |
| `engram_gate_add` engram_gate_add.hip:27 | dG D,P (L1, L14) | S/engram_gate.rs:32 ← FP:3529 | (4, n)×256 | — | 320 KB/token, noise (KERNEL_PERF_REVIEW:175-176) |
| head prep: `hc_weighted_sum` hc_weighted_sum.hip:8, `rms_norm_weighted` rms_norm.hip:12 | dG D (per row), P | S/head.rs:110, S/rms_norm.rs:52 ← forward_head.rs:57/69 (164/175) | (20)×256, (1)×256 | — | ~62 us/row combined; kept per-row for fidelity (forward_head.rs:22-27) |

### G. Secondary paths: vision (iG1) and DSpark (off in production)

**Vision tower** runs on the box-1 iGPU (engine_worker.rs:1179). Kernels in
crates/v4flash-vision/kernels/vit.hip, wrappers in crates/v4flash-vision/src/kernels.rs:
`vit_gemm` :162 (kernels.rs:61; WMMA on gfx115x), `vit_rmsnorm_f16` :254 (:132), `vit_rope_split`
:277 (:155), `vit_attention` :322 (:179), `vit_swiglu_f16` :424 (:202), `vit_unfold` :436 (:216).
Shape: 32 layers, dim 1024, 16 heads, inter 2816. At n=3108 patches: 446 ms against a 167 ms floor;
`vit_gemm` 27 TF (45% of peak); attention 54% of wall at 19% of its floor, VALU-issue bound; a WMMA
rewrite hits the P→V permute "Pareto wall" (v4flash-vision/src/tower.rs:40-58). 613 ms per image
(Dd/VISION_PORT.md:137). Owner has said vision perf is good enough — NOT in this sweep.

**DSpark drafter** (S/het/mtp.rs): `q8_0_gemv_bpack_warp8` ×5, `q8_0_grouped_gemv_bpack`,
`q8_0_quantize_f32`, `q8_0_gemv_warp8` (head), `rms_norm_weighted_batched`, `hc_weighted_sum(_batched)`,
`hc_sinkhorn_par_batched`, `hc_post_from_split_batched`, `kv_post_fused` K/fp8_e4m3fn.hip:119,
`attention_mixed_score` K/attention_mixed.hip:53 + `attention_mixed_softmax_wsum` :111,
`router_topk_par` (128 experts, top-3), `rope_tail_batched`, `swiglu`, `vec_add_inplace`, MoE via
A3/A4, `argmax_one` K/softmax_sample.hip:36. DSpark verify reuses the batched path at b ≈ 5-6.

## 2. Families: time shares and roofline status

**Caveat.** Stage timings include intra-stage idle and host stalls (PREFILL_100K:67-77;
Dd/ROUTED_MOE_IS_NOT_A_KERNEL.md:1-21). Decode figures are step-level event sums taken **before**
the 09-26 knobs (which saved ~4.8 + 1 + 0.5 + 0.3 ms/step at 8 rows: cd91dad, 4eb4b69, 98bdd94, 3feaeb6).

Reference points: **Decode (8 rows, 2 lanes, ~300-325 ms/step):** dgpu 83, igpu 121, box-2 service
245 = page 140 + compute 105 ms/step (Dd/PROFILING_AUDIT_2026-09-21.md:222-235); an earlier 2-lane
point: dGPU 91, iG1 72, box-2 compute 94 (MS:870-877). **Prefill at 100K:** 160 s wall, 606 tok/s.

| family | decode share | prefill share (100K) | roofline / exhausted flags |
|---|---|---|---|
| **A** MoE small-B (iG1+iG2) | iG2 compute ~105 ms/step (35% of wall); iG1 busy 121 ms/step (event sum incl. builders and peer push). Box-2 paging (140 ms/step) is I/O, not kernels | small | b=1 kernels at 93-96% of 214 GB/s: **CLOSED** (ROADMAP:123-146). But iG2 runs the **kwide chain for every b≥2 request** (decode_max_b=1) and small-B kwide/kwide2 efficiency is **unmeasured**: OPEN |
| **B** MoE large-B (iG1+iG2) | — | **iG2 GPU 82.9 s = critical path** (+22 s read/write overhead); iG1 true 43 s (PREFILL_100K:146-152, 81) | 54% / 35% of achievable bandwidth, but a possible compute ceiling at ~94% / 88% (PREFILL_100K:214-236). No MXFP4 WMMA arm (KERNEL_PERF_REVIEW:42) |
| **C1** dense decode (dG) | byte floor ≈25.5 ms/step (178 MB × 80 lane-layers at 640 GB/s = 22.3 + Engram 1.0 + head 2.2). Measured `q_chain` 11 ms/step at 4 rows = 275 us/layer vs 81 us byte floor (FP:788-795); shared expert 3.7 ms/step at 1 row (FP:6702-6704). **Likely the largest dG family, ~30-50% of dG busy (estimate)** | — | bpack GEMV ~630 GB/s on the head; dense Q8 GEMVs 85-99% of bandwidth (KERNEL_PERF_REVIEW:175). Stage overhead is the gap |
| **C2** dense prefill (dG) | — | within the ~54 s of dG stages that are not `ffn_combine` wait (93.6 s − 39.6 s; PREFILL_100K:16-28). dG is not the pole (:30-41) | f16x at 52% of matrix peak (FP:6105) |
| **D** attention + KV | per lane-layer score 13.6 + smwsum ~16 + meta + small ops ≈ 2.5-4 ms/step (estimate) | score 2.8 s + smwsum 3.5 s (PREFILL_100K:25-26) | smwsum LDS/occupancy variants exhausted (KERNEL_PERF_REVIEW:160-164); decode is latency-bound |
| **E** indexer | 8 index layers per lane: select ≤130 us + ~65 us sorts + score ≈ 2-5 ms/step at 130-235K (estimate) | 3.16 s + 4.05 s reuse (PREFILL_100K:23-24) | gemm score at 68% of matrix peak (FP:5470) |
| **F** mHC/glue | mHC ≈ 61 us × 80 ≈ 4.9 ms/step after FAST; router topk ~1 ms; 1020 event pairs/step (PROFILING_AUDIT:39-42) | `mhc_pre_attn` 6.9 s stage (PREFILL_100K:20) | decode mHC at the launch floor (~11 us kernel in a 23 us stage). Prefill M=24 GEMM is 1-WG-wide (§3.6) |

## 3. Observations for owners (code facts)

1. **Box-2 small-B mismatch (A).** The "closed at 93%" kernel serves only 1-row requests. Rows 2-8
   run `kwide`/`kwide2`, sized for B=512 (RE:4163, 4276-4283). Batched chain +20 us/token (RE:2095-2099).
2. **Dead work at decode (C1).** Four `f32_to_f16_cast_2d` per lane-layer (FP:3799, 3860, 6093,
   6147) feed only the f16x arms (dispatch.rs:301-306; FP:3884-3889, 6128, 6163), which decode never
   takes. A duplicate `q8_0_quantize_f32` of `attn_input_norm` (FP:3811 and 3979; comment FP:3801-3807).
3. **No-op kernels and copies (D).** `f16_roundtrip` (FP:4038) is a no-op after the V4.1 FP8 window
   (KERNEL_PERF_REVIEW:48). The `q_normed` D2D copy (FP:3914-3919) exists only for buffer flow.
4. **Decode smwsum is thin (D).** Grid (4, b): at b=4 only 16 WGs on the dGPU (attention.rs:1143-1149). ~16 us/call.
5. **Replay regime b≈64 (C2).** dp4a arms exceed the bpack cap of 16 and fall back to grid.z=B GEMVs
   that re-read weights 64× (q8_0.rs:258-269, 869-876). Replay is 12.4 s of 160 s at 100K (PREFILL_100K:11-12).
6. **Prefill mHC pre-mix (F).** `f16_gemm_wmma_lds_tiled` at M=24 launches (1, B/64) WGs and takes
   478 us per call regardless of B (FP:303-309). A K-split variant `mhc_mix_ksplit_batched` exists but
   is unwired (K/mhc_arena.hip:262-276).
7. **Two-lane decode doubles dense bytes (C1).** Each lane streams all dense weights and its own head
   (FP:2753-2762; MS:941-943). Two lanes lose at 2 rows (MS:863-867).
8. **Stale roofline bench.** `bench_v41_kernel_roofline` still targets the old B=1 chain.
9. **KERNEL_PERF_REVIEW (09-13) status.** Addressed: #1 (index-K/indexer fire), #2 (Engram GEMM), #4
   (SWA via mixed), #6 (measured, closed). Partial: #5 (WMMA for b>64 but 1-WG-wide). Still apply:
   #3 (no MXFP4 WMMA arm), #10 (ratio-1 compressor FP:4299; router_topk).

## 4. Existing benches (crates/v4flash-kernels/tests/) and correctness oracles

| family | benches | oracle / bit-exact tests |
|---|---|---|
| A | `bench_mxfp4_moe_b1` (B=1, rotating selection); `bench_v41_kernel_roofline` decode_igpu; `bench_expert_miss_cost`; crates/deepstrix-expertd/src/bin/bench.rs | `mxfp4_pair_oracle` (batch, hetsplit, kwide, chunked vs CPU); `mxfp4_iq2s_oracle`; `mxfp4_wi_devcount`; `remote_experts_loopback`; `mxfp4_repack_parity`; `v41_sparse_verify_remap_pairing`; `expert_sel_stats_oracle` |
| B | `bench_v41_kernel_roofline` prefill_igpu (BENCH_B=1024) | same as A |
| C1/C2 | `bench_small_b_dense`; `bench_qb_wmma_isolated`; `bench_v41_kernel_roofline` decode_dgpu/prefill_dgpu; `bench_mall_retention`; `bench_launch_overhead`; `bench_device_ceilings` | `q8_0_matvec`; `q8_0_matvec_batched` (bpack b≤16); `q8_0_grouped_matvec`; `q8_0_gemm_f16x`; `q8_0_matvec_wmma`; `shared_expert`; `head_to_logits`; `head_chain`; `q8_k_quantize` |
| D | `bench_decode_latency_breakdown`; `bench_decode_latency_ab`; `bench_attention_isolated`; `bench_prefill_attention_isolated`; `bench_wmma_wsum`; `bench_single_layer` | `attention_dec_bitexact`; `attn_meta_fill`; `attention_vs_reference`; `attention_compute_chain`; `attention_swa`; `attention_swa_visible_window`; `multistream_row_bases`; `compressor_end_to_end`; `compressor_pool`; `compressor_gather_ab`; `kv_rollback_compressor`; `kv_arena_compact`; `rope_tail` |
| E | `bench_decode_latency_breakdown` (indexer at 65K/131K/235K); `bench_decode_latency_ab`; `bench_indexer_b1_decode_shapes`; `bench_indexer_score_isolated`; `bench_indexer_topk_isolated` | `indexer_topk_select_oracle`; `v41_indexer_selection_oracle`; `candidate_blocks_oracle`; `indexer_score_gemm_oracle`; `indexer_score_mw_oracle`; `indexer_score_e2m1`; `e2m1_keys_format`; `indexer_pipeline`; `indexer_topk` |
| F | `bench_decode_latency_breakdown` (mhc, overhead); `bench_decode_latency_ab`; `bench_mhc_arena_v41`; `bench_mhc_arena_ab`; `bench_mhc_pre_isolated`; `bench_router_readback_ab`; `bench_event_overhead` | `mhc_arena_bitexact` (incl. `mhc_fast_is_bit_identical`); `mhc_chain`; `rms_norm`; `rms_norm_no_weight`; `router_learned`; `router_topk_alts`; `readback_pack`; `f16_matvec`; `f16_gemm_batched_small_b`; `hip_graph_smoke` |
| model-level gates | — | `v41_layer0_parity`; `multistream_step`; `forward_prompt_batch_matches_sequential`; `v41_oracle_dump_smoke`; `logit_dump_long` |

(These in-tree benches are for READING the production launch shapes; the sweep does not run cargo.)
