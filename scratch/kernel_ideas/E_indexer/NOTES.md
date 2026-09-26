# E_indexer — dGPU CSA2 sparse indexer + candidate pool (gfx1201), 2026-09-26

Running notes. Everything measured is under `results/`; every command is recorded here.

## Production facts (read from the tree at 361d4f9)

* Scores stride `n_idx_stride = ATTN_MIXED_MAX_KEYS = V41_MAX_CTX (368640) + 128 + 384 = 369152`
  (src/attention.rs:112). One row of scores = 1.48 MB f32.
* Decode arena: `n_idx_max = max_b n_comp`, per-row `keys_base_per` (FP:5376, 5499). The
  keys of different rows in a lane are DIFFERENT streams' stores → no key sharing across b.
* Keys: packed E2M1, **80 B per position** (128-dim, one row per position; the 32 heads are
  on the Q side). 235K keys = 18.8 MB per layer store, NOT 481 MB as the task text says
  (that multiplied by 32 heads). Fits the 256 MB budget; tested at the real 235K.
* Score kernel `indexer_score_wmma_batched_mw_e2m1`: grid (ceil(n_idx_max/1024), b) × 256,
  args (scores, q f32, hw f32, keys u8, n_idx_per, n_idx_stride, keys_base_per).
* Top-k: `indexer_topk_select_batched_ilp` grid (b) × 1024, args (selected, done, scores,
  n_idx_per, stride, top_k=512), then (n_idx_max > 4096) the chain launches
  `chunk_4096_batched` grid (n_chunks, b), [`regroup_4096_batched`], `merge_4096_batched`
  grid (1, b) — all early-out on done[b]=1 (S/indexer.rs:930-1060). allowed_bits = null.
* Gather `indexer_gather_batched`: grid (512, b, 2) × 256, head_dim 512 f16 → **one f16 (2 B)
  per thread**, args (dst, comp_kv, selected, top_k, head_dim=512, comp_base_per).
* Candidate pool: `candidate_block_max` grid (ceil(nb/256), b) × 256, cb_size 8;
  `candidate_threshold` grid (b) × 256, topk 2048 blocks; `candidate_mask_apply`
  grid (ceil(n_idx_max/256), b) × 256. L20 builds, L24/28/32/36 mask.
* Cache regime: keys are streamed once per (layer, lane) per step; between two uses ~GBs of
  other traffic → COLD (flush 80 MB before each timed block). Scores are written by the score
  kernel and read by select right after → warm in L2 (no flush). comp_kv main store
  (1 KB/row f16, 235 MB per layer at 235K) → cold; gather tested with a 64K-row store + flush.

## Roofline (dGPU 640 GB/s DRAM, ~194 TF f16 WMMA peak)

| kernel | bytes/call | flops/call | floor |
|---|---|---|---|
| score mw, b rows, n keys | b·n·80 B (keys, cold) + b·n·4 B scores | b·n·32·128·2 | 235K,b=1: 18.8+0.94 MB → 31 us; WMMA 1.93 GF → 10 us. b=4: 79 MB → 123 us |
| select, per row | n·4 B × (1 sample + 1-3 count + 1 compact [+ tie]) from L2 | — | L2-resident; latency-bound (1 WG/row) |
| gather, b rows | b·512·1 KB read (random rows, cold) + same write | — | b=4: 4 MB → 6.5 us (+ launch) |
| block_max / mask_apply | n·4 B read (+ n/8·4 B) | — | 235K: ~1 MB → 1.5 us |

## Log
