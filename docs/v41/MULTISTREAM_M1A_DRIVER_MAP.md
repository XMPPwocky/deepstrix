# Multi-stream M1a: map of the batched layer driver (2026-09-21)

Structure of `forward_layer_pre_moe_v2` / `post_moe_v2` and the two-lane driver,
classified stage by stage as activation-only (reusable for rows from different
sequences as-is) or per-sequence (needs the KvArena's per-row tables). Produced by
a code-search pass; every claim cites file:line. Companion of
MULTISTREAM_M1A_INVENTORY.md; the design it feeds is MULTISTREAM_DECODE_PLAN.md 3.7.

# BATCHED prefill layer driver — structural map

Paths (given once, then abbreviated):

- **FP** = `/home/claude-code/deepstrix/crates/v4flash-kernels/src/het/forward_prefill.rs`
- **BS** = `/home/claude-code/deepstrix/crates/v4flash-kernels/src/het/batch_scratch.rs`
- **ST** = `/home/claude-code/deepstrix/crates/v4flash-kernels/src/het/state.rs`
- **EN** = `/home/claude-code/deepstrix/crates/v4flash-kernels/src/het/engine.rs`
- **CFG** = `/home/claude-code/deepstrix/crates/v4flash-kernels/src/config.rs`
- **ATT** = `/home/claude-code/deepstrix/crates/v4flash-kernels/src/attention.rs`
- **EW** = `/home/claude-code/deepstrix/crates/deepstrix-server/src/engine_worker.rs`
- **RE** = `/home/claude-code/deepstrix/crates/v4flash-kernels/src/het/remote_experts.rs`

---

## 0. Entry points and what they carry per call

| fn | file:line | per-sequence inputs |
|---|---|---|
| `forward_prompt_batch_v2` | FP:418-508 | one `pos0`, one `tokens`, one `state: &mut HetModelState`; loops layers calling `state.with_kv_source(layer, ...)` → `forward_layer_batch_v2` (FP:490-505) |
| `forward_prompt_batch_v2_pipelined` | FP:539-570 | thin wrapper → `_range(.., 0..N_LAYER, CedMode::Exact, None)` FP:558-562 |
| `forward_prompt_batch_v2_pipelined_range` | FP:574-960 | splits rows `[0,b_a)`→lane A / `[b_a,b)`→lane B, `pos0_a = pos0`, `pos0_b = pos0 + b_a` (FP:731-732) |
| `forward_layer_batch_v2` | FP:1740-1769 | wrapper: `pre_moe_v2` + `post_moe_v2` on `self.sync_events.layers[layer]` (FP:1764-1767) |
| `forward_layer_pre_moe_v2` | FP:1787-6167 | `(bd,bi,sd,si, ls:&mut HetLayerState, dlw, ilw, pos0, tokens, vis, stats, sev, pager, ced)` |
| `forward_layer_post_moe_v2` | FP:6173-6380 | `(bd, b, sev, hot_active)` — **no `ls`, no `pos0`, no `tokens`** |

The whole driver assumes rows are **contiguous positions of one sequence**:

- `pos_per_b[i] = pos0 + i` uploaded once per chunk, FP:481-484 (single-lane), FP:761-768 (two-lane). Consumed by rope at FP:2240-2250 (q), FP:2316-2326 (kv), FP:3536-3546 (indexer q), FP:4130-4140 (rope_inverse).
- All causal-prefix math derives from the scalars `pos0`, `ls.n_raw`, `ls.raw_off` plus the **row index `i`** — never from a per-row position table (FP:2394-2410, FP:2619-2627, FP:2674-2681, FP:2992-2999).
- `chunk_visibility(pos0, b, spans)` FP:384-397 and `image_spans::lane_split` FP:701 are also `pos0 + i` indexed.

---

## 1. Stage table (call order inside `forward_layer_pre_moe_v2`)

| # | Stage (file:line) | Scratch / state args touched | `state.layers[l]` / `HetCompressorState` fields | per-row tables filled/consumed | host scalars from the single sequence | verdict |
|---|---|---|---|---|---|---|
| 0 | **MTP residual capture** FP:1812-1885 | `bd.mtp_src`, `bd.residual`, `bd.mtp_hc_mean`, `bd.mtp_captured*` | none | none (`skip = tokens.len()-n`, FP:1861) | `pos0` only to stamp `bd.mtp_captured_pos0 = pos0 + skip` (FP:1881) | activation-only (tag needs a per-row pos) |
| 0b | subtensor dump FP:1888-1907 | `bd.residual`, `sd.attn_input_norm` | none | none | `pos0` only in the filename (FP:1901-1902) | activation-only |
| 1a | **mHC layer-0 carry seed** FP:1934-1942 | `bd.hc_pre_carry` | none | none | none (`layer == 0` test) | activation-only |
| 1b | **Engram** FP:1943-1985 | `bd.engram_rows/_xq/_xscale/_kv`, `bd.residual`; staged by `stage_engram_lane` FP:1705-1727 / `stage_engram_rows_batch` FP:1730-1738 | none | consumes the lane slice `buf[off*ein .. (off+n)*ein]` (FP:1725) | rows are gathered by the CALLER from `(token, pos)`; driver only slices by lane offset | activation-only (rows must be gathered per row by the caller) |
| 1c | **mhc_pre_attn** FP:1929-2101: `rms_nw` 1981-1986 → f16 matvec (3 arms: pre-scaled per-row 2005-2031, narrow 2032-2041, WMMA 2042-2045) → `hc_sinkhorn` 2047-2060 → `hc_weighted` + carry copy 2062-2085 → `rms_w` 2086-2098 | `sd.flat, sd.mix, sd.mhc_inv_scalar, sd.mhc_rms_partials`, `bd.split`, `bd.hc_pre_carry`, `sd.attn_cur`, `sd.attn_input_norm` | none | none | none | **activation-only** |
| 2 | **Q chain** FP:2103-2265 (guarded `ced != KvSourceOnly`, FP:2107) — cast f16 2109-2126, `qa_matvec` 2128-2157, `rms_w` 2158-2168, cast qr 2169-2173, qb variant 2175-2216, per-head rms/passthrough 2217-2231, **rope** 2233-2250 | `sd.x16_n_embd, sd.xq_n_embd, sd.xscale_n_embd, sd.kq_attn_q8k, sd.qr, sd.qr_normed, sd.qr16, sd.qr_xq, sd.q, sd.q_normed`; `bd.pos_per_b` | none | **consumes `bd.pos_per_b`** (FP:2240) | `pos0` (via `pos_per_b`) | **touches per-sequence position** (rope) — already table-driven, only `pos_per_b` must become per-row |
| 3 | **KV chain** FP:2268-2345 — matvec/gemm 2271-2297, `rms_w` 2299-2309, **rope** 2311-2326, fp8/fp4 window quant 2328-2341, f16rt 2342-2346 | `sd.kv_raw, sd.kv_normed, sd.x16_n_embd, sd.xq_n_embd` ; `bd.pos_per_b` | none | consumes `bd.pos_per_b` (FP:2316) | `pos0` | **touches per-sequence position** (rope only; output is a per-row K/V row) |
| 4a | **Raw-window counts** FP:2349-2452 | `n_raw_after`, `n_raw_offset_after` (host Vecs, FP:2360, 2392) | **reads `ls.n_raw`** (FP:2391) | produces host `n_raw_per` / `n_raw_offset_per` | `n_raw_before = ls.n_raw`, `i`, `SWA_WINDOW` (FP:2394-2400) | **per-sequence KV state** |
| 4b | **KV append** FP:2453-2481 | `sd.kv_normed` → `ls.kv_cache` | **writes `ls.kv_cache` at `append_at = ls.raw_off + n_raw_before`** (FP:2463-2479); `KV_CACHE_ROWS` bound assert 2464-2469 | none (single contiguous batched append) | `ls.raw_off`, `ls.n_raw`, `b` | **per-sequence KV state** |
| 4c | **Main compressor projection** FP:2483-2580 | `sd.kv_cur`, `sd.sc_cur`, `sd.attn_input_norm`, `dlw.compressor` | gated by `own_compressor = ratio>0 && dlw.compressor.is_some() && ced != Replay` (FP:2491) | none | `ratio` only | activation-only per se |
| 4d | **row/pos_mod table upload** FP:2613-2636 | `sd.row_per_b`, `sd.pos_mod_per_b` | none | **fills `row_per_b`, `pos_mod_per_b`** | `pos = pos0 + i`; `pm = pos % ratio`; `row = if ratio==4 {4+pm} else {pm}` (FP:2617-2624) | **per-sequence position** |
| 4e | **Compressor state write — FAST gather** FP:2653-2737 | `sd.comp_pos_per_boundary`, `sd.comp_state_kv_snapshots`, `sd.comp_state_score_snapshots`, `sd.kv_cur/sc_cur`, `sd.pos_mod_per_b` | **`cs.state_kv`, `cs.state_score`, `cs.n_comp += n_bnd`** (FP:2678, 2729-2736) | fills `comp_pos_per_boundary`, `n_comp_after` | `fast_ok = pos0 % ratio == 0 && b % ratio == 0 && b >= ratio` (FP:2671-2672); `pos_per_boundary[k] = pos0 + k*ratio` (FP:2676); `n_comp_after[k] = n_comp_start + (k+1)/ratio` (FP:2681) | **per-sequence compressor state** |
| 4e′ | **Compressor state write — SERIAL segments** FP:2737-2814 | same + `sd.row_per_b` | `cs.state_kv/state_score` per segment (FP:2740-2761), `compressor_shuffle` on fire at ratio 4 (FP:2790-2796), `cs.n_comp += 1` (FP:2799) | `n_comp_after` with pre-fire/post-fire snapshot (FP:2802-2812) | `pos_mod_now = (pos0+i)%ratio`, `seg_len = min(ratio-pos_mod_now, b-i)`, `comp_fires = (pos0+seg_end)%ratio==0`, boundary pos `pos0+seg_end-ratio` (FP:2740-2798) | **per-sequence compressor state** |
| 4f | **Per-boundary batched chain** FP:2824-2977: `compressor_pool` 2825-2833 → `rms_w` 2834-2842 → upload `comp_pos_per_boundary` 2843-2849 → **index-K (S1a)** 2850-2941 → rope 2916-2929 → fp4kv/fp8+f16rt 2930-2960 → `comp_kv_append` / `comp_kv_fp8` 2952-2972 | `sd.comp_pooled_batched, sd.comp_rows_batched, sd.index_k_rows_batched, sd.index_k_normed_batched, sd.comp_pos_per_boundary` | **`cs.comp_kv` (F16/Fp8)** appended at `n_comp_start` (FP:2952-2971); **`cs.index_k`** appended at `n_comp_start` and **`cs.n_index_comp = n_comp_start + n_boundaries`** (FP:2903-2911) | consumes `comp_pos_per_boundary` | boundary positions, `n_comp_start` | **per-sequence compressor state** |
| 4g | **Reuse-layer positional n_comp** FP:2979-3006 | — | reads `ls.compressor` (lent by `with_kv_source`), errors if `cs.n_comp < (pos0+b)/ratio` (FP:2992-3000) | `n_comp_after[k] = (pos0+k+1)/ratio` (FP:3002-3004) | `pos0`, `ratio` | **per-sequence position** |
| 4h | **Positional cross-check / cap** FP:3007-3053 | — | — | rewrites `n_comp_after[k]` to `(pos0+k+1)/ratio` under `V41_COMP_POSITIONAL=1` (FP:3020-3043); hard error if any `> ATTN_MIXED_MAX_KEYS` (FP:3046-3053) | `pos0` | **per-sequence position** |
| 4i | **CSA indexer compressor** (V4-Flash `ratio==4` only) FP:3071-3245 | `sd.kv_cur/sc_cur` (slice-reused), `sd.row_per_b`, `sd.pos_mod_per_b`, `sd.comp_*_snapshots`, `sd.comp_rows_batched` | **`ls.indexer_compressor`: `ics.state_kv/state_score`, `ics.n_comp`, `ics.comp_kv`** (FP:3109-3167, 3230-3243) | consumes `row_per_b`/`pos_mod_per_b`; builds `pos_per_boundary_idx` | same segment formulas as 4e′ (FP:3110-3167) | **per-sequence compressor state** |
| 4j | early return for `CedMode::KvSourceOnly` FP:3247-3251 | — | — | — | — | — |
| 5 | **Per-row attention tables upload** FP:3254-3282 | `sd.n_raw_per`, `sd.n_raw_offset_per`, `sd.n_comp_per` | — | **fills all three** from the host Vecs | — | **per-sequence-derived; already per-row** |
| 6a | **SWA-via-mixed (ratio 0)** FP:3284-3340 | `sd.attn_scores`, `sd.heads`, `sd.q_normed`, `ls.kv_cache` | **reads `ls.kv_cache`** | consumes `n_raw_per`/`n_raw_offset_per`/`n_comp_per` | `n_total_max = max(n_raw_after)` (FP:3289); `scores_stride` FP:3291 | **reads per-sequence KV; per-row tables already carry it** |
| 6b | **SWA batched (ratio 0)** FP:3341-3369 | `sd.heads`, `sd.q_normed`, `ls.kv_cache`, `dlw.attn_sinks` | reads `ls.kv_cache` | consumes `nrp_view`/`nrop_view` | `max_n_kv = SWA_WINDOW` (text) or `max(n_raw_after)` (image), FP:3344-3349 | same |
| 6c | **Mixed / compressed path** FP:3370-4030 | | | | | |
| 6c-1 | dense comp buffer pick FP:3396-3420 | — | **`cs.comp_kv.dense_f16(max_comp, ..)`** (FP:3407) | `n_comp_after` max | — | per-sequence |
| 6c-2 | **indexer** FP:3459-3799: key source pick 3459-3464, `need_mask` 3473-3479, `n_index_comp_per_b` upload **from `n_comp_after`** 3500-3505, `matvec_q` 3507-3533, rope 3534-3547, QAT 3548-3562, `matvec_proj` 3563-3574, scale 3575-3583, score wmma 3584-3691, **candidate blocks** 3692-3725, **topk** 3726-3746, **gather** 3747-3784, sparse `n_comp_per` re-upload 3785-3793, `bd.indexer_saved_store` publish 3794-3799 | `sd.indexer_q/_q16/_head_weights/_scores/_topk_scratch/_topk_done`, `sd.candidate_block_score`, `sd.candidate_threshold`, `sd.n_index_comp_per_b`, `sd.attn_active_comp_kv`, **`bd.indexer_sel_saved`** (FP:3731), `bd.pos_per_b` | **`cs.index_k`** keys (FP:3459-3463, 3624-3650); **`cs.comp_kv`** gathered (FP:3753-3781) | **fills `n_index_comp_per_b` = `n_comp_after`**, `bd.indexer_sel_saved`; rewrites `sd.n_comp_per = min(n_comp, INDEXER_TOP_K)` | `is_index_source_layer(layer)` (FP:3463; `forward_layer.rs:86`), `n_idx_max = max(n_comp_after)` | **per-sequence compressed store + position (rope)** |
| 6c-3 | **S2 selection reuse** FP:3800-3840 | `bd.indexer_sel_saved` re-gathered; predicate at FP:3466-3472 uses `bd.indexer_saved_store == index_source_of(layer)` | reads `cs.comp_kv` | consumes `indexer_sel_saved`, re-uploads sparse `n_comp_per` | — | per-lane (stored in `bd`, not `sd`, precisely because lanes differ — BS:219-234) |
| 6c-4 | **score + smwsum / fused** FP:3842-3962 | `sd.attn_scores`, `sd.heads`, `ls.kv_cache`, `eff_comp_kv_buf` | reads `ls.kv_cache` + comp store | consumes `nrp/nrop/ncp` views | `eff_n_total_max` FP:3849-3856; `scores_stride = sd.attn_scores_stride(b, eff_n_total_max)` FP:3866 | **reads per-sequence KV; tables already per-row** |
| 6c-5 | `V41_VERIFY_DECODE_ATTN` per-row replay FP:3963-4029 | `sd.verify_scores/_inv/_partials` | reads `ls.kv_cache` | consumes `n_raw_after[j]`, `n_comp_after[j]`; **hardcodes `raw_off = 0`** (FP:3987) | per-row `nr`,`nc` | per-sequence |
| 7 | **Post-attention eviction** FP:4033-4116 | `sd.kv_ring_scratch` | **writes `ls.kv_cache` (compaction), sets `ls.n_raw`, `ls.raw_off`** (FP:4089-4115) | none | `n_raw_during_chunk = n_raw_before + b` (FP:2482); `speculative_append()` branch FP:4089-4091 | **per-sequence KV state** |
| 8 | **Stage 6 output projection** FP:4118-4240: `rope_inverse` 4127-4141, cast heads 4142-4148, grouped matvec 4149-4210, cast low 4211-4216, `matvec_out` 4217-4239 | `sd.heads, sd.heads16, sd.heads_xq/_xscale, sd.low, sd.low16, sd.low_xq/_xscale, sd.attn_out`; `bd.pos_per_b` | none | consumes `bd.pos_per_b` (FP:4130) | `pos0` via `pos_per_b` | **position-only** (rope_inverse); rest activation-only |
| 9 | **Stage 7 mhc_post_attn** FP:4242-4257 | `bd.after_attn_hc`, `sd.attn_out`, `bd.residual`, `bd.split` | none | none | none | **activation-only** |
| 10 | **Stage 8 mhc_pre_ffn** FP:4259-4333 | `sd.flat, sd.mix`, `bd.split`, `bd.hc_pre_carry`, `sd.ffn_cur`, `bd.ffn_input_norm` | none | none | none | **activation-only** |
| 11 | **Stage 9 router** FP:4337-4511: gate proj 4345-4400, `router_topk.launch_batched` 4402-4418, hash-router host path 4437-4470, image-run `bias_vl` topk 4471-4510 | `sd.router_logits`, `sd.router_logits_host`, `bd.d_selected`, `bd.d_ew` | none | writes `d_selected`/`d_ew` `[b × N_EXPERT_USED]` | `tokens[i]` (hash router FP:4455, image test FP:4441); `image_runs = image_spans::image_runs(tokens)` FP:4420 | **activation + token-id only** (no position, no KV) |
| 11b | `record_sel_stats` FP:4513; optional host stats FP:4516-4524 | `bd.d_selected` | none | reads `d_selected` | `b` | activation-only |
| 12 | **Stage 10 shared expert** `issue_shared_expert_prefill` FP:1111-1227, called FP:4531 or (deferred) FP:5214 | `sd.xq_n_embd/_xscale/_kq_ffn_q8k/x16_n_embd/gate_sh/up_sh/mid_sh/mid_sh16/mid_sh_xq/_xscale`, `bd.ffn_input_norm` → `bd.ffn_shared` | none | none | `pos0` only for dump filename FP:1220 | **activation-only** |
| 13 | **Stage 11 pager + split decisions** FP:4536-5203: `remote_owns_layer` 4563-4569, `split_cap` 4578-4583, `sparse_resid_layer` 4645-4656, `moe_group_bound` 4661-4667, union readback of `d_selected` 4703-4717, box1/box2 pick loop 4778-4838, **remote submit** 4930-5060, `pg.ensure*` 5061-5140, exclusion + audit 5141-5200 | `bd.d_selected`, `bd.d_ew`, `bd.ffn_input_norm`, `sd.remote_xq`, `bd.remote_ticket`, `bd.remote_ffn_moe_layer` | none | reads `d_selected`/`d_ew` over `b*6` | `b` (small-B catch-all thresholds FP:4832-4836), `layer` | **activation-only** |
| 14 | **peer push dGPU→iGPU** FP:5270-5311 | `bd.ffn_input_norm/d_selected/d_ew` → `bi.ffn_input_norm_recv/d_selected/d_ew`; `sev.selected_ready/selected_pushed` | none | none | none | **activation-only** |
| 15 | **M61 hot leg (dGPU)** FP:5313-5440 | `sd.hot.*`, `bd.hot_ffn_moe_dgpu` | none | builds `group_count`/`expert_members` from `d_selected` | `b` | **activation-only** |
| 16 | **iGPU routed MoE** FP:5445-6146: cast/q8k 5455-5475, group builder 5519-5648, work items 5649-5754, iq2 variants 5755-5872, q8k mid 5873-5881, q2k down 5882-6079, `verify_decode_moe` 6080-6146 | `bi.*`, `si.*`, `routed_src`, `moe_remap` | none | consumes `bi.d_selected`/`bi.d_ew` | `b`, `moe_group_bound` | **activation-only** |
| 17 | **peer push back + `moe_arrived`** FP:6148-6166 | `bi.ffn_moe` → `bd.ffn_moe_recv` | none | none | none | **activation-only** |
| 18 | **Stage 12 `forward_layer_post_moe_v2`** FP:6173-6380: `vec_add(shared)` 6190-6199, remote `wait` 6206-6288, `vec_add(remote)` 6323-6345, `vec_add(hot)` 6346-6364, `hc_post` 6365-6377 | `bd.ffn_moe_recv, bd.ffn_shared, bd.remote_ffn_moe, bd.hot_ffn_moe_dgpu, bd.after_attn_hc, bd.split, bd.residual_next` | **none** | none | none | **activation-only — takes no `ls`, no `pos0`, no `tokens`** |

**Summary of the verdict column.** Everything from Stage 6 output-projection onward (stages 7/8/9/10/11/12 = mhc_post, mhc_pre_ffn, router, shared expert, MoE local+remote, combine) is pure activation math over `b` independent rows, plus `tokens[i]` for the hash router and the image-token test. The per-sequence surface is exactly: rope positions (`bd.pos_per_b`), the raw KV window (`ls.kv_cache` / `ls.n_raw` / `ls.raw_off`), the compressor state (`cs.state_kv/state_score/n_comp/comp_kv/index_k/n_index_comp`), and the three derived per-row tables plus the boundary list.

---

## 2. How the per-row tables are computed today (one sequence)

### `n_raw_per` / `n_raw_offset_per` — FP:2349-2452

Host scalar `n_raw_before = ls.n_raw` (FP:2391). Text path (FP:2394-2410):

```
causal_end = n_raw_before + i + 1          // exclusive cache slot
n_raw_per[i]        = min(causal_end, SWA_WINDOW)
n_raw_offset_per[i] = causal_end.saturating_sub(SWA_WINDOW)
```

Image path (FP:2411-2431): `image_spans::raw_window(n_raw_before, i, left, right)` returns `(offset, count)` directly, with a bound check `offset + count <= n_raw_before + b` (FP:2418-2426). The doc block FP:2363-2384 states the invariant: rows `i` live at cache slot `n_raw_before + i`, the causal window is `[max(0, p_i-W+1), p_i]`, and the widened image window can see keys *ahead* of the row inside the span.

Uploaded to device at FP:3266-3277 (`sd.n_raw_per`, `sd.n_raw_offset_per`, `sd.n_comp_per`, all `copy_from_host_async` on `de.compute`).

### KV append and ring-wrap handling — FP:2453-2481, FP:4033-4116

- Append offset: `append_at = ls.raw_off + n_raw_before` (FP:2463), assert `(append_at + b) <= KV_CACHE_ROWS` (FP:2464-2469), single `kv_append.launch_batched` (FP:2470-2477). The long comment FP:2438-2462 records that ignoring `raw_off` produced lane-B divergence at `pos >= SWA_WINDOW`.
- `KV_CACHE_ROWS = SWA_WINDOW + B_MAX` (ST:20), i.e. the cache is oversized so a whole chunk appends without eviction.
- Post-attention (FP:4033-4116), three branches:
  - `speculative_append()` true → **no compaction, no `raw_off` reset**; only `ls.n_raw = n_raw_during_chunk` (FP:4089-4113). Reasoning at FP:4053-4088.
  - `n_raw_during_chunk > SWA_WINDOW` → two-hop copy of the last `SWA_WINDOW` rows through `sd.kv_ring_scratch` into slots `[0,W)`, then `ls.n_raw = SWA_WINDOW; ls.raw_off = 0` (FP:4091-4110).
  - else `ls.n_raw = n_raw_during_chunk; ls.raw_off = 0` (FP:4112-4115).

### `row_per_b` / `pos_mod_per_b` — FP:2613-2636

```
pos = pos0 + i
pm  = pos % ratio
row = if ratio == 4 { 4 + pm } else { pm }
```
(FP:2617-2624), uploaded to `sd.row_per_b` / `sd.pos_mod_per_b` at FP:2628-2634. The CSA indexer compressor reuses both buffers unchanged because the ratio-4 row formula matches (comment FP:3065-3068; consumption at FP:3126-3127).

### Compressor boundaries + `comp_pos_per_boundary` + `n_comp_per`

Two paths, selected at FP:2671-2672:

```
fast_ok = gather_enabled && pos0 % ratio == 0 && b % ratio == 0 && b >= ratio
```

**Fast gather** (FP:2673-2737): `n_bnd = b / ratio` (FP:2674); boundary positions `pos0 + k*ratio` for `k in 0..n_bnd` (FP:2675-2677); `cs.n_comp += n_bnd` (FP:2678); `n_comp_after[k] = n_comp_start + (k+1)/ratio` (FP:2679-2681); upload `comp_pos_per_boundary` FP:2682-2687; one `compressor_state_snapshot.launch_gather` FP:2688-2704; end-of-chunk `compressor_state_write.launch_batched` over the last `ratio` rows into rows `0..ratio-1` FP:2705-2736.

**Serial segment loop** (FP:2737-2814):
```
pos_mod_now = (pos0 + i) % ratio
seg_len     = min(ratio - pos_mod_now, b - i)
comp_fires  = (pos0 + seg_end) % ratio == 0
boundary position pushed = pos0 + seg_end - ratio
```
(FP:2740-2742, 2772, 2798). `compressor_shuffle` fires immediately at ratio 4 (FP:2790-2796). `n_comp_after` uses the pre-fire/post-fire snapshot rule at FP:2802-2812 (only the last position of a firing segment sees post-fire).

**Reuse layers** (`ratio > 0`, no own compressor) FP:2979-3006: `n_comp_after[k] = (pos0 + k + 1) / ratio` — with an explicit warning (FP:2985-2991) that it must **not** be derived from the live `cs.n_comp`, because in the two-lane driver the source layer has already appended the other lane's rows.

**Ratio-0 layers**: `n_comp_after` is all zeros (FP:3007-3011).

Positional audit / override at FP:3020-3043 (`V41_COMP_POSITIONAL`), hard error above `ATTN_MIXED_MAX_KEYS` at FP:3046-3053.

### `n_index_comp_per_b` and candidate pool — FP:3500-3505, 3692-3746

- `n_index_comp_per_b` is a straight `copy_from_host_async(&n_comp_after)` (FP:3500-3505) — V4.1 advances index-K in lockstep with `n_comp` (comment FP:3450-3455; the write is `cs.n_index_comp = n_comp_start + n_boundaries`, FP:2910).
- `need_mask` (FP:3473-3479): V4.1 → any `n_comp_after > INDEXER_TOP_K` (or `> 0` under `V41_INDEXER_FORCE`); V4-Flash → `ratio == 4 && indexer_compressor.is_some() && any n_comp_after > INDEXER_TOP_K`.
- Candidate pool (FP:3692-3725): at `CANDIDATE_SOURCE_LAYER` (=20, CFG:181) `candidate_blocks.launch_build` writes per-row `sd.candidate_block_score` / `sd.candidate_threshold` from `sd.indexer_scores` + `sd.n_index_comp_per_b`; at layers above it `launch_mask` re-reads them. `nb_stride = CandidateBlocks::n_blocks(ATTN_MIXED_MAX_KEYS)` (FP:3694-3696).
- Top-k writes **`bd.indexer_sel_saved`** (`[rows, INDEXER_TOP_K]`, per-lane — FP:3731 and BS:219-234), then gather into `sd.attn_active_comp_kv` (FP:3747-3784), then `sd.n_comp_per` is re-uploaded as `min(n_comp, INDEXER_TOP_K)` (FP:3785-3792).
- `bd.indexer_nsparse_saved` is allocated (BS:1104) and **never read or written** anywhere else — the only other references are its declaration BS:236 and the alloc.

---

## 3. Lane / shared partition and capacities

**Module doc** BS:1-31 states the split: `BatchDgpuScratch`/`BatchIgpuScratch` are per-lane (buffers that outlive one lane's pre-MoE or are touched by `de.xfer`/`ie.xfer`); `BatchDgpuShared`/`BatchIgpuShared` are one instance for both lanes.

**Why sharing is legal** BS:356-368: both lanes issue all pre-MoE dGPU work on `de.compute` in program order, and the host issues one lane's whole `forward_layer_pre_moe_v2` before the other's — stream order per layer is `post_A(L) pre_A(L+1) post_B(L) pre_B(L+1)` (BS:180-183, FP:781-788). Every shared buffer is first-written and last-read inside one lane's pre-MoE on `de.compute` (async H2D uploads included).

**Per-lane dGPU (BS:214-347)**: `residual`, `residual_next`, `split`, `hc_pre_carry`, `engram_*`, `mtp_*`, `after_attn_hc`, `ffn_input_norm`, `d_selected`, `d_ew`, `ffn_shared`, `ffn_moe_recv`, **`pos_per_b`**, `hot_ffn_moe_dgpu`, `remote_ffn_moe`/`_valid`/`_layer`, `remote_ticket`, **`indexer_sel_saved`/`indexer_nsparse_saved`/`indexer_saved_store`**, `mtp_lane_cut`. Rationale list BS:176-213.

**Shared dGPU (BS:403-651)** holds **all** the per-row tables: `row_per_b` BS:509, `pos_mod_per_b` BS:510, `n_raw_per` BS:513, `n_raw_offset_per` BS:520, `n_comp_per` BS:523, `n_index_comp_per_b` BS:582, `comp_pos_per_boundary` BS:614, plus `comp_state_*_snapshots` BS:597-598, `attn_scores` BS:536, `attn_active_comp_kv` BS:590, `indexer_*` BS:552-579, `kv_ring_scratch` BS:500, `kv_cur`/`sc_cur` BS:503-504, `remote_xq` BS:467.

**Per-lane iGPU (BS:782-816)**: `ffn_input_norm_recv`, `ffn_moe`, `d_selected`, `d_ew`, `group_count`, `n_work_items`, `n_staged_work_items`, `n_chunked_work_items` (the last four because `fill_zero` is a null-stream memset, BS:791-795).
**Shared iGPU (BS:818-867)**: `d_xq_q8k`, `d_mid_cat`, `d_midq_cat`, `d_x16`, `d_mid16`, `expert_members`, `work_items`, `staged_work_items`, `chunked_work_items`, `q2k_partials`.

**Capacities**
- `B_MAX = 1024` (BS:59). Production splits every `B_MAX` chunk across two lanes of `ceil(B_MAX/2)`, so lanes **and** the shared sets are allocated at `B_MAX.div_ceil(2)` = 512 rows; the single-lane `forward_prefill` needs `rows >= B_MAX` on all four (BS:22-29, FP:998-1000).
- `check_rows` refuses `rows > B_MAX` (BS:175-178); `check_scratch_rows` refuses `b > bd.rows || bi.rows || sd.rows || si.rows` at every entry (FP:188-206), called at FP:459, FP:720-721, FP:1918.
- `MTP_CAP_ROWS = 128` (BS:212); the capture asserts `tokens.len() <= bd.rows` and `n <= MTP_CAP_ROWS` (FP:1824-1840).
- `HEAD_BATCH_MAX = 16` (`scratch.rs:39`), used to size `head_xq_b`/`head_xscale_b`/`logits_b` (`scratch.rs:414-416`) and enforced in `forward_head_batch` (`forward_head.rs:41`).
- **attn scores stride**: chosen per launch by `BatchDgpuShared::attn_scores_stride(b, n_total_max)` (BS:1209-1217) → `attention::attn_scores_stride(capacity_keys, batch, N_HEAD, n_total_max)` (ATT:300-338): errors if `n_total_max > capacity_keys / (b*N_HEAD)`, returns the legacy `ATTN_SCORES_STRIDE` when it fits, else `n_total_max`. Capacity is `attn_scores.len()` doubled on the f16 path (BS:1198-1205); sizing formula `attn_scores_capacity_keys(rows, n_kv_max)` BS:1156-1196 (charges CED decoder layers only `SWA_WINDOW/2` rows). Score and smwsum must be handed the same value (FP:3859-3866, both used at FP:3878/3900/3945/3959).
- `ATTN_MIXED_MAX_KEYS` = 82176 (V4-Flash, ATT:69) / V4.1 `V41_MAX_CTX = 307_200` (ATT:73); `INDEXER_TOP_K = 512` (CFG:116); `SWA_WINDOW = 128` (CFG:126); `ATTN_SWA_BATCHED_MAX_KV = 512` (ATT:29, enforced FP:3350-3354).

---

## 4. `KvMark` / `mark_kv` / `rollback_kv` / `normalize_raw_windows`

- **`HetLayerState`** ST:414-431: `kv_cache`, `n_raw`, `raw_off`, `compressor`, `indexer_compressor`. `raw_off` = first valid row of the SWA window; decode appends monotonically at `raw_off + n_raw` (ST:417-430).
- **`mark_kv` / `try_mark_kv`** ST:245-265: snapshots `(n_raw, raw_off)` per layer, plus `CompMark { main, indexer }` per layer that owns a compressor — `CompStateMark::capture` (ST:625-633) copies `n_comp`, `n_index_comp`, and the whole `state_kv`/`state_score` accumulators to host.
- **`rollback_kv`** ST:288-345: length checks (ST:289-305); refuses if any layer's `raw_off` went **backwards** since a non-slid mark (ST:306-318) — a compaction physically moved the window and the mark no longer addresses it; refuses a slid mark whose append pointer is within `MTP_BLOCK` of `KV_CACHE_ROWS` (ST:319-332); then restores `n_raw`/`raw_off` per layer (ST:333-337) and the compressor stores (ST:338-345, `CompStateMark::restore` ST:635+). Doc caveat about compressor streaming state at ST:270-283.
- **`KvMark::advanced_by(keep, abs_pos)`** ST:513-621 (the accept path's *partial* rollback): raw window becomes `end = off + nr + keep`, `new_nr = min(end, SWA_WINDOW)`, `new_off = end - new_nr` (ST:534-550) — slide, no byte movement; `slid: true` (ST:620). Compressor: `n_comp = ((abs_pos + keep)/ratio).max(before).min(before + keep)` (ST:563-574) and `n_index_comp` moves in lockstep by the same delta, clamped `<= n_comp` and never backwards (ST:575-618).
- **`normalize_raw_windows`** EN:644-673: per layer, if `raw_off != 0 && n_raw != 0`, two-hop copy of `[raw_off, raw_off+n_raw)` through `dgpu_scratch.kv_wrap_scratch` down to `[0, n_raw)` and `raw_off = 0`; syncs `de.compute`. Doc EN:632-643: decode keeps a monotonic ring while the prefill-path verify assumes `[0, n_raw)`, and calling this before the verify's `mark_kv` keeps them in agreement (KL jumps ~0.0008 → ~4 nats otherwise).
- **`SpeculativeAppend`** FP:6545-6570: a process-global `AtomicBool` scope guard; `speculative_append()` FP:6551-6553 is read by the eviction block (FP:4089) and by `sparse_resid_layer` (FP:4646) and the pager's prefill-counting (FP:5117).

### The speculative-verify driver (EW) vs. the prefill driver

Common preamble (EW:3588-3663): `SpeculativeAppend::begin()` EW:3589; batch is `[next, drafts[..k]]`, i.e. **K+1 contiguous positions of ONE sequence** starting at `pos` (EW:3623-3627, rationale EW:3615-3622); `normalize_raw_windows` EW:3659; `mark = state.state.mark_kv()` EW:3660.

`V41_VERIFY_DECODE_PATH=1` (`verify_decode_path()`, EW:3683) runs a **different** driver (EW:3684-3801):

- It does **not** call `forward_prefill_pipelined` at all (EW:3824-3833 skips it when `decode_path_logits.is_some()`), so `mtp_src` is never captured — warned at EW:3843-3851.
- Layer-major over rows: `for layer in 0..N_LAYER { for j in 0..bsz { ... } }` (EW:3701-3785), each row copying `resid[j]` into the single-token `dgpu_scratch.residual` (EW:3703) and `carry[j]` into `hc_pre_carry` (EW:3705).
- **Per-row `publish_pos_slot`** (EW:3709-3721): `slot = state.layers[layer].raw_off + state.layers[layer].n_raw`, then `engine.publish_pos_slot(&mut dgpu_scratch, pos + j, slot)`. `publish_pos_slot` (EN:~614-631) writes `pos_dev` and `kv_slot_dev` with stream-ordered `write_value32` on `de.compute`. Comment EW:3710-3712: *"Layer-major breaks the lockstep the token-major path assumes, so publish rope pos + KV slot from THIS layer's counters, per row."*
- Per row it calls `state.state.with_kv_source(layer, |ls| engine.forward_layer_standalone_graphs_paged(dgs, igs, ls, dlw, ilw, pos+j, toks[j], pg))` (EW:3722-3740) — decode's **standalone captured-graph** path, per row, with its own remote submit per row (noted EW:3679-3682).
- Buffer parity: the residual swap is done **once per layer, after the row loop**, not per row (EW:3778-3784), because HIP graphs capture pointers — layer L must see the same physical buffer decode sees (A for even L, B for odd), and every row of layer L must see the same one (EW:3757-3762). Output read from `residual_next` (EW:3763).
- Head is run per row afterwards into the same `[B*N_VOCAB]` layout (EW:3788-3801).

Batched path (default) EW:3803-3833: arms `mtp_capture_rows = toks.len()` on **both** lanes (EW:3820-3821, rationale EW:3803-3819 — arming only lane A fed the drafter stale residuals for rows `[b_a, b)`), then one `forward_prefill_pipelined(...)` call with `last_only = false`.

Accept/commit EW:4083-4103: `keep = n + 1`, `partial = mark.advanced_by(keep, pos)`, `state.state.rollback_kv(&partial)`; the global batch row → lane mapping goes through `state.bd_a.mtp_lane_cut` (EW:4109-4114; written at FP:713 and at FP:665 for the `b < 2` single-lane fallback).

---

## 5. MoE stages: local (pager) vs remote (RemoteExpertClient), for B rows

Everything MoE consumes exactly three B-shaped things: `bd.ffn_input_norm` `[b, N_EMBD]`, `bd.d_selected` `[b, N_EXPERT_USED]`, `bd.d_ew` `[b, N_EXPERT_USED]`.

**Layer-level split decisions** (all scalar-per-layer, not per-row):
- `remote_owns_layer` / `remote_split_on` FP:4563-4570; `split_cap` FP:4578-4583 (`N_EXPERT_USED` when the remote split or pager is on, else `hot_prefill_cap()`); mutual-exclusion error FP:4584-4589.
- `sparse_resid_layer` FP:4645-4656 and `moe_group_bound` FP:4661-4667 (pool slots vs `N_EXPERT`).

**Local (pager)** FP:4668-5203:
- Union path (`pager_union_prefill()`, FP:4679): `de.compute.synchronize()` then reads back **`d_selected` over `b * N_EXPERT_USED`** into `sel_host` (FP:4711-4717). The pick loop FP:4778-4838 walks that flat array, dedups by expert id (`seen`), and assigns each id to box 1 (`ids`) or box 2 (`extra_remote`) by residency/partition rules — **it never looks at which row a pick came from**.
- `pg.ensure(layer, &ids)` (sparse LRU, FP:5108) / `pg.ensure_layer_dense` FP:5131 / `pg.ensure_layer_union` FP:5133 / `ids.clear()` under replay offload FP:5062.
- Exclusion: `pg.mark_remote_after_ensure` (sparse) or `pg.set_remote_exclusion` (dense window) FP:5158-5162; audit `verify_routing_exactly_once(layer, &sel_host_remote, pg.remap(), Some(&owns_eff))` FP:5173-5178 — "O(picks) via a bitmap, so the B*6 prefill batch is fine" (FP:5171).
- Weights view chosen at FP:5216-5268: `pg.routed` (absolute pool slots) when `sparse_resid_layer`, else `pg.routed_window(layer)`; remap always passed (FP:5266).
- iGPU dispatch FP:5445-6146: `moe_group_builder.launch_hetsplit(group_count, expert_members, d_selected, remap, mode=0, split_cap, b, n_used, moe_group_bound, max_per_expert)` FP:5537-5549; `max_per_expert = b` when the group space is widened (FP:5498-5502); work-item builders FP:5649-5754; iq2 variants FP:5755-5872; `q2k_down` FP:5882-6079.

**Remote (`RemoteExpertClient`)** FP:4930-5060:
- Re-binds the dGPU uncached (FP:4946-4952), quantises `bd.ffn_input_norm` with the **same** `de.q8k.launch(..., BLOCKS_Q8K_GATE_IN * b)` kernel the iGPU would use (FP:4955-4961), syncs, and copies `xq_host` (`b * BLOCKS_Q8K_GATE_IN * BLOCK_Q8_K_BYTES` bytes, FP:4941-4944, 4963-4964) and `ew_host` (`b * N_EXPERT_USED`, FP:4965-4966).
- `owns_eff` mask FP:5187-5190 → `sel_for_remote` = `sel_host_remote` with non-owned slots replaced by `NO_PICK` (-1) (FP:5195-5203; `NO_PICK` at RE:74), `ew_for_remote` zeroed in the same slots (FP:5029-5035).
- `submit_dispatch(unmasked, layer, b, &xq_host, sel, ew, /*resp_f32=*/true)` FP:5020-5040; signature RE:3932-3938 → `submit_unmasked` / `submit`; `submit_inner` RE:3958-3973 validates `xq.len() == b*XQ_BYTES_PER_TOKEN && sel.len() == b*nu && ew.len() == b*nu` and sets `REQ_FLAG_BATCHED` when `b > 1` (RE:3966-3970).
- Ticket stashed on the **lane** scratch (`bd.remote_ticket`, `bd.remote_ffn_moe_layer`, FP:5045-5046) and awaited in post-MoE (FP:6206-6288), where `partial.f32()` must be `b * N_EMBD` (FP:6272-6277) and is uploaded to `bd.remote_ffn_moe` then `vec_add`-ed at FP:6323-6345.

**Row-independence.** Nothing in the MoE path — pager union, exclusion remap, group builder, work items, iq2/q2k, hot leg, remote submit, or combine — reads `pos0`, `ls`, `n_raw_per`, `n_comp_per`, or any positional table. The only per-row inputs are `ffn_input_norm[r]`, `d_selected[r*6..]`, `d_ew[r*6..]`, and `tokens[r]` upstream in the router (FP:4441, 4455, 4420). `forward_layer_post_moe_v2` (FP:6173-6380) takes no `ls`, no `pos0` and no `tokens` at all.

---

## 6. Additional facts that bear on a multi-stream variant

- **Lane B's `pos0_b = pos0 + b_a`** (FP:732) and the lane-A-before-lane-B KV ordering are load-bearing today: lane A's kv_append at layer L is queued before lane B's attention at L on the same FIFO (doc FP:513-520), and the reuse-layer `n_comp_after` is computed positionally *because* the source layer has already appended the other lane's rows (FP:2985-2991).
- The comment at FP:679-692 records a measured result: forcing the whole chunk into lane A (`single_lane_max`) changed the answer — "something in the per-lane KV/shared-scratch path is not row-independent."
- `restore_compressor_lending()` FP:607 (ST:380-392) and the manual lend/return around the layer loop FP:857-861 / FP:941-944 park a KV-source layer's `HetCompressorState` on its reuse layer for the duration of a call; `with_kv_source` (ST:394-412) is the exception-safe form used by `forward_prompt_batch_v2` (FP:490).
- `CedMode` FP:1129-1148 gates whole stage groups: `KvSourceOnly` skips stages 2-3 (FP:2107, 2481), the window append, and returns at FP:3247-3251 before attention/MoE; `Replay` skips the compressor projection and store write (`own_compressor` FP:2491) and takes the positional `n_comp` formula.
- `COMPRESS_RATIOS` (CFG:131/138), `KV_SOURCE_LAYERS = [2,8,14,20]`, `INDEX_SOURCE_LAYERS = [2,8,14,20,24,28,32,36]`, `ENGRAM_LAYERS = [1,14]` (CFG:150-156), `CED_DECODER_START = 20` (CFG:183), `CANDIDATE_SOURCE_LAYER = 20` (CFG:189).
- `HetCompressorState` (ST:~145-181) holds `state_kv`/`state_score` (iGPU), `comp_kv` (dGPU, `CompKvStore::{F16, Fp8{rows,head}, E2m1}`), `n_comp`, `width`, `head_dim`, `index_k: Option<DeviceBuffer<u8>>`, `n_index_comp`; capacity is `max_n_comp = (n_kv_max + ratio - 1)/ratio` rows fixed at alloc (ST:206).


---

## 6. What step 3 changed (2026-09-20)

The `RowLayout` arm touches exactly the per-sequence rows of the stage table.
Line numbers are post-edit (`forward_prefill.rs`).

| stage | contiguous (unchanged) | arena |
|---|---|---|
| entry | — | validates tables vs `tokens.len()`, refuses `vis`/CED/MTP; binds `arena_comp_base`, `arena_keys_base`, `arena_state_base`, `arena_fire_state_idx`, `arena_fire_dst_row` for this layer's store (`store_index_of`) and the `pos_at(i)` closure |
| 4a raw counts | `causal_end = n_raw_before + i + 1` | `n_per = min(n_raw_per[i] + 1, W)`, `offset = slot_per[i] + 1 - n_per` |
| 4b append | `launch_batched(append_at)` | `launch_batched_rows(0, Some(slot_per))` after a buffer-length check |
| 4d row/pos_mod | `pos0 + i` | `pos_at(i)` |
| 4e/4e′ state write | fast gather / serial segments | one `launch_batched_rows(.., Some(state_base_per))` over all b rows; `n_comp_after[i] = n_comp_per[i] + fires`, checked `== (pos+1)/ratio` |
| 4f pool | snapshots | `launch_batched_rows(cs.state_kv, cs.state_score, .., Some(fire_state_idx))` |
| 4f index-K / comp appends | `n_comp_start` | `Some(fire_dst_row)`; `cs.n_index_comp` not touched |
| 4g reuse-layer counts | positional from `pos0` | `n_comp_per[i] + fires`, checked positional |
| 4h positional audit | `pos0 + k + 1` | `pos_at(k) + 1` |
| 6c-2 indexer score | gemm / mw / sw | `launch_batched_mw_e2m1_rows(.., Some(keys_base_per))` (V4.1 keys only) |
| 6c-2 / 6c-3 gathers | `launch_batched` | `launch_batched_rows(.., Some(comp_base_per))` |
| 6c-4 score + smwsum | `_f16s` | `_f16s_rows(.., if indexer_fired { None } else { Some(comp_base_per) })`; fused / f32-scores refused |
| 6c-5 verify replay | env-gated | refused |
| 7 eviction | three branches | no-op (`KvArena::advance` after the step) |

Everything else (Q/KV chains, mHC, router, shared expert, MoE local/remote,
combine, output projection) runs as before: the caller uploads `tables.pos_per`
into `bd.pos_per_b`, which is all the rope stages read.
