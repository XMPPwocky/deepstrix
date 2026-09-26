# D_attention — dGPU attention + window/compressed KV (gfx1201) — sweep notes 2026-09-26

Base: 361d4f9 (production). All GPU runs via `../_infra/gpu_run.sh --dev dgpu`. Nothing here is merged.

## Production launch facts (read from S/attention_dec.rs, S/attention.rs, FP:5120-5910)

Decode (arena rows, b = 1..5 per lane, graph-captured stage):
- `attn_meta_fill` grid (1)x32: n_raw_per / n_raw_offset_per / n_comp_per as kernel args.
- `attention_dec_score_htiled_wmma_f16s` grid (ceil(n_total_max/256), n_head/16=4, b) x 512.
  args: scores(f16, stride `max_keys`), q f32 [b,64,512], raw_kv f16 (window cache, per-row
  offset n_raw_offset_per), comp_kv f16 (= `attn_active_comp_kv` [b, 512, 512] gathered top-K when
  the indexer fired -> comp_kv_batch_stride = INDEXER_TOP_K = 512, comp_base_per = null; else the
  dense per-row store with comp_base_per non-null and stride 0), mask = null,
  max_keys_words = ceil(82176/32) = 2568, n_head 64, max_keys = scores_stride = 3072
  (ATTN_SCORES_STRIDE whenever n_total_max <= 3072), kq_scale = 1/sqrt(512).
- `attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_f16s` grid (4, b) x 512, same buffers +
  sinks[64], out f32 [b,64,512].
- n_raw = 128 (SWA_WINDOW) at any context past 128 tokens; n_comp = min(n_comp, 512) -> n_total <= 640
  at decode (no image rows at decode). Contexts 8K-235K all collapse to 640 keys.
- Per lane-layer: score ~13.6 us + smwsum ~16 us (incl. event pairs) x 80 lane-layers = ~2.4 ms/step.

Prefill (b = 512 per lane): `attention_mixed_score_batched_htiled_wmma_f16s` grid
(ceil(n_total/256), 1, B) x 512 (each warp = 16 keys, loops all 4 head tiles x 32 K-chunks),
smwsum grid (4, B) x 512. n_total = 128 + P/2 on ratio-2 encoder layers (up to ~50K keys at 100K),
scores streamed through DRAM (stride = n_total_max when > 3072).

Roofline, decode b = 1, 640 keys: K/V 640 KB (read twice: score + wsum, second pass L2-warm),
q 128 KB, scores 82 KB written + read + rewritten, out 128 KB. ~1.9 MB -> 3.0 us at 640 GB/s;
84 MFLOP WMMA -> ~2 us at 40 TF. Both kernels are far from any byte/flop roof: latency-bound.

## Log
(see below, appended as work proceeds)
