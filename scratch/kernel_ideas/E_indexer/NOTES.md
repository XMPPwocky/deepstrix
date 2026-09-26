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

### 2026-09-26 03:5x — first attempt (hub LIVE, cut off by the 04:01 box hang)
Baselines under `results/*_base1.txt` were measured with the production hub running: p10/p90
spread up to 2x (e.g. gather b=4 cold 65 us med / 17 us p10). Treated as indicative only; every
number below is re-measured with the hub DOWN (`*_base2.txt`).

ATT of the production score kernel (`results/att_score/`, n=235K b=4, two dispatches, summarised
with `python3 ~/scripts/att_top.py <one csv> --by stall --top 25`):
* stall/latency = 89%. Top stalls: `s_wait_dscnt` (LDS) ~45% summed, `ds_load_b128` from lds_q
  (the 16 A-fragment reads per n-tile) ~30%, `ds_store_b16` (the f32->f16 Q staging, one b16
  store per element) ~8%, `ds_bpermute` (shfl_xor) 3%, `s_wait_loadcnt` (the packed key loads)
  only 6%.
* So the mw score kernel is NOT bound by the key stream: it is LDS-bound — it re-reads the whole
  8 KB f16 Q from LDS for every 16-key n-tile (16 x b128 per lane per tile = 8 KB per wave per
  tile, vs 1.25 KB of keys per tile). Per CU per tile-round: 64 KB of LDS reads for 128 WMMAs.
  Holding the wave's 16 A-fragments in VGPRs (64 regs) removes all of it -> candidate `pf_qreg`.
* Q staging: 4096 f32 -> f16 with scalar b16 LDS stores per WG (16 per thread + wait) = the
  ds_store_b16 lines; vectorisable (float4 -> 4 halves -> one b64 store).

ATT of the production select kernel (`results/att_select/`, dispatch_11 = n=235K):
* stall/latency = 40%, idle 1.07M of 1.14M latency cycles: a single 1024-thread WG per row
  leaves the device idle; the kernel is latency/barrier-bound.
* Top stalls: `s_wait_loadcnt 0xf` (the U=16 strided count/compact passes) 15%, `s_wait_dscnt`
  (bitonic-sort LDS exchanges) ~22% summed, `s_wait_kmcnt` 5%, LDS stride64 ops (sample sort).
* Both bitonic sorts (2048-key sample, 4096-pair final) do every stage through LDS + a
  1024-thread barrier: 66 + 78 = 144 barriers per row. Stages with j < 32 pair lanes of the same
  wave -> can be done with `__shfl_xor` (no LDS, no barrier): barrier count 144 -> 15 + 28.

### 2026-09-26 19:1x — hub DOWN re-baseline (`results/*_base2.txt`, rounds=30) and first candidates
Commands: `bash run_score.sh base2 30` / `bash run_rest.sh base2 30`, each under
`gpu_submit.sh --dev dgpu --mb 156 --label E_indexer/...`. p10/p90 now within ~1-3% of median.

| kernel (prod launch) | shape | med us | roof us | % of roof |
|---|---|---|---|---|
| score mw_e2m1, cold keys | 65K b1 / b4 | 38.8 / 88.2 | 8.6 / 34 | 22 / 39 |
| | 131K b1 / b4 | 46.8 / 156.7 | 17 / 69 | 37 / 44 |
| | 235K b1 / b4 | 82.8 / 274.4 | 31 / 123 | 37 / 45 |
| select_ilp alone (warm L2 scores) | 65K / 131K / 235K, b1 | 59.6 / 66.2 / 83.0 | latency-bound 1 WG/row | — |
| select + prod chain early-outs | same | 65.9 / 71.4 / 89.8 (b4: 80 / 76 / 92) | +6-20 us of early-out launches | — |
| gather (512,b,2)x256, cold random | b1 / b4 / b32 | 11.4 / 17.9 / 104 | 3+launch / 6.5 / 52 | — / 36 / 50 |
| candidate_threshold (1 WG x 256) | 65K/131K/235K | 41.8 / 57.2 / 87.2 (b=1; b=4 same) | ~120 KB from L2: ~2 us | <5 |
| candidate_block_max | 235K b4 | 6.8 | 5.9 | 87 |
| candidate_mask_apply | 235K b4 | 11.1 | 5.9 | 53 |
| idx-q f16_matvec 4096x1280 cold | b1 / b4 | 31.6 / 58.6 | 16.4 | 52 / 28 |
| indexer_fp4 / proj matvec / vec_scale (warm, graph) | b4 | 3.8 / 20.7 / 3.3 | launch floor ~2.5 | — |

**Score candidates (cand_score.hip, `results/score_base2.txt`)** — all four bit-exact at every shape:
* `score_mw_pf` (prefetch only): 1.01-1.06x SLOWER. `score_mw_coal` (b128 tile through LDS): 1.06x slower.
* `score_mw_pf_qreg` (Q A-fragments in 64 VGPRs, no LDS reads in the loop): **0.52-0.58x** cold at
  every shape: 235K b4 274.4 -> 143.4 us (86% of the 123 us DRAM floor); 235K b1 82.8 -> 45.0;
  131K b4 156.7 -> 86.6; 65K b1 38.8 -> 26.3. `pf_coal` ties it (the coalesced load is not the lever).
  Confirms the ATT: the kernel was LDS-bound on re-reading Q per n-tile.

**Top-k candidates (cand_topk.hip, `results/select_cand1.txt`)** — selection identical to the
production chain on real / tie-heavy (quantised) / tie+mask scores at all shapes (SEL exact=YES):
* `topk_select_regsort` (hybrid shuffle/LDS bitonic sorts, barriers 144 -> 56): 0.59-0.70x.
* `topk_select_v3` (+ float4 count/compact): 0.58-0.67x. `v3_u8` (8 float4 in flight): **0.58-0.63x**:
  65K 63.4 -> 40.5; 131K 69.1 -> 44.9; 235K 85.8 -> 59.2 us (alone; b=4 identical).

**Gather candidates (cand_gather.hip, `results/gather_cand1.txt`)** — bit-exact (pure copy):
* one b128 per thread (`gather_u4_r1..r16`, any rows/WG): cold random b4 19.1 -> 12.4 us (0.65x),
  b1 11.5 -> 9.5-10, b32 110 -> 48 (random) / 89 -> 33 us (local, 1.02 TB/s effective).
* multi-rows-per-thread (IPT 2-8) were SLOWER cold (b4 22.6 / 14.6 us) but faster in the warm
  graph regime — the index-load branches serialised the loads; rewritten branch-free (next run).
* warm graph inner=5 numbers invert (base 12 us, u4 variants 20 us): back-to-back same-dst launches
  are not the production regime; added a "graph inner=1 + flush" mode = production decode.

### 19:3x — score v2 (cand_score2.hip) and candidate_threshold (cand_cand.hip): `results/score_cand2.txt`, `results/cand_cand2.txt`
Score, cold keys, ratio vs base (65K b1 / 65K b4 / 131K b1 / 131K b4 / 235K b1 / 235K b4):
* `score_mw_pf_qreg` (previous best)          0.69 / 0.58 / 0.58 / 0.55 / 0.54 / 0.52
* `s2_w8n8_qreg` (qreg, NO prefetch)          0.69 / 0.59 / 0.59 / 0.56 / 0.57 / 0.53  -> prefetch worth ~2-5% at large n
* **`s2_w8n8_pf_hw`** (pf_qreg + hw in VGPRs)  **0.63 / 0.52 / 0.54 / 0.52 / 0.51 / 0.51**  -> best everywhere
  (235K b4 275.7 -> 140.1 us = 88% of the 123 us DRAM floor; 235K b1 83.1 -> 42.3; 65K b1 32.9 -> 20.8)
* `+vs` (vectorised Q staging): +1-2% slower than pf_hw -> the staging stores were not the lever once LDS reads are gone.
* geometry (all pf_hw_vs): w16n4 ties w8n8 at b=4 but loses warm; w4n16/w8n16/w4n8 within 1-3%;
  w4n32 / w2n32 lose at b=1 (0.73-1.05: too few WGs). Production's 8 waves x 8 tiles is right.
* `s2_w4n8` shows CMP bit_diff=512 at n=235000: its 512-col WGs stamp -inf only to 235008 while the
  base (1024-col) stamps to 235520; the harness compares [0, gx*1024). Not a kernel bug (production
  select reads [0, n)), and not the winner anyway.
candidate_threshold: `candidate_threshold_ilp` (block 1024, 8 loads in flight, same 4x8-bit radix
select): thresholds identical; 65K 42.3 -> 30.4, 131K 57.4 -> 32.4, 235K 87.5 -> 43.9 us (b=4 same).
Still a single WG doing 4 serial passes; a parallel bin scan / 2 x 16-bit passes could halve it again.

## Idea list (ranked by expected gain x confidence / effort), status
1. **score: Q A-fragments (+hw) in VGPRs** — attacks the measured LDS bottleneck. DONE: 0.51-0.63x, bit-exact.
2. **select: hybrid shuffle/LDS bitonic sorts + float4 passes** — attacks barrier/latency. DONE: 0.58-0.63x, exact.
3. **gather: b128 per thread, rows per WG** — attacks the 2 B/thread wave storm. DONE: cold b4 0.65x; IPT variants re-run pending.
4. candidate_threshold: 1024 threads + ILP. DONE: 0.49-0.68x. Further: parallel bin scan, 16-bit passes (not done).
5. select: single count pass with several thresholds (avoid the 2nd/3rd O(n) pass at 235K) — NOT DONE.
6. select: replace the 4096-pair sort by radix-select of the 512th + sort of 512 (barriers 36 -> ~15) — NOT DONE.
7. STRUCTURAL: fuse score -> select for decode: score WGs keep per-1024-chunk (max, count>t) stats or a
   per-chunk top-16 so select's O(n) passes read 1/64 of the data; final merge in one WG. Saves the L2
   round trip (~1 MB/row, ~2-3 us of the remaining ~59 us select) and 1-2 O(n) passes (~10 us). NOT DONE:
   the win is bounded by what is left in select after 2; the sample/threshold logic must stay exact.
8. STRUCTURAL: multi-WG select (sample kernel + parallel count/compact + 1-WG final sort): 3-4 graph nodes;
   would take the O(n) passes from ~5-10 us to ~1-2 us each. NOT DONE (launch cost ~2.5 us/node eats most of it).
9. select chain: the prod chain adds 3-5 early-out launches (+6-20 us at 65K b4) after select; with the select
   never giving up on real scores these are pure launch cost -> a device-side conditional or dropping the chain
   for n where the sample never fails is a Rust-side change (not a kernel). NOT DONE.
10. candidate_mask_apply fused into select's count/compact (read block_score + threshold inline instead of a
    separate -inf pass): saves 6-20 us x 4 layers per lane-step. NOT DONE (changes select's inputs).
11. idx-q f16_matvec 4096x1280 at 52% (b1) / 28% (b4) of bandwidth — family C's kernel; noted, not touched.

### 19:4x — gather decode regime + b=512, confirmation runs (3 processes each, rounds=50)
`results/gather_cand2.txt` (branch-free IPT, new GRAPH inner=1 + flush mode = production decode),
`results/gather_b512.txt` (`env GATHER_B512=1 GATHER_CANDS=... bash run_rest.sh b512 30 gather`, --mb 720),
`results/{score,select,gather,cand}_confirm_run{1,2,3}.txt` (`bash run_confirm.sh confirm 3`, --mb 156),
summarised with `python3 summarize.py results/<sec>_confirm_run{1,2,3}.txt`. Zero correctness failures.

Confirmed medians-of-3 (min..max within 1-3% unless noted):
| kernel | shape/regime | base us | cand us | ratio | cand |
|---|---|---|---|---|---|
| score | 65K b1 / b4 cold | 32.8 / 84.8 | 20.8 / 45.1 | 0.64 / 0.53 | s2_w8n8_pf_hw |
| score | 131K b1 / b4 cold | 47.4 / 157.7 | 26.5 / 83.1 | 0.56 / 0.53 | |
| score | 235K b1 / b4 cold | 82.9 / 275.8 | 42.5 / 139.5 | 0.51 / 0.51 | (b4 = 88% of the 123 us DRAM floor) |
| select alone | 65K / 131K / 235K, b4 | 60.1 / 67.6 / 85.4 | 38.3 / 43.4 / 58.8 | 0.48-0.57 / 0.57 / 0.63 | topk_select_v3_u8 |
| gather graph+cold random | b1 / b4 | 17.0 / 25.0 | 15.7 / 18.6 | 0.92 / 0.74 | gather_u4_r1 |
| gather direct cold random | b4 / b32 | 19.7 / 110.2 | 13.4 / 50.4 | 0.68 / 0.46 | |
| gather direct cold LOCAL | b32 / b512 | 83.6 / 1417 | 31.1 / 522 | 0.37 / 0.37 | (b512 = prefill lane shape, 1.03 TB/s effective) |
| candidate_threshold | 65K / 131K / 235K | 41.9 / 57.3 / 87.4 | 30.0 / 32.4 / 43.8 | 0.68 / 0.55 / 0.49 | candidate_threshold_ilp |
Noise notes: gather b=4 LOCAL in the graph+cold mode is bimodal (17 vs 29 us medians flip between
runs for every variant incl. base) — not used for claims; the RANDOM regime (= decode's picks) is clean.
`gather_u4_r4_2ipt` is consistently slow at b=4 cold (22 us) but best at b=1 and b=512: not the pick.

ISA (isa_all.sh): base score 59 VGPR / cand pf_hw 139 VGPR (64 for Q frags + 16 hw + prefetch; no
spills, ~10 waves/SIMD, LDS 8.3 KB); select 65 -> 94 VGPR (4 vals + 4 idxs + 8 float4 in flight);
gather 4 -> 9; threshold 10 -> 29.

ATT of the select candidate (`results/att_topk_v3/`): the two traced dispatches decoded to an empty
CSV (header only) although the .att files exist — not chased; the production-select ATT above plus
the measured 0.58-0.63x is enough for the write-up.

## Per-step arithmetic (decode, b=4 per lane, 2 lanes => 16 index-layer calls per step; gather 38 layers x 2 = 76)
* score pf_hw: saves 39.8 / 74.6 / 136.3 us per call at 65K/131K/235K -> 0.64 / 1.19 / 2.18 ms per step.
* select v3_u8: saves 21.8 / 24.2 / 26.6 us per call -> 0.35 / 0.39 / 0.43 ms per step.
* gather u4_r1: saves 6.5 us per call (graph+cold, b4) x 76 -> 0.49 ms per step at any n.
* candidate_threshold_ilp: once per lane at L20: saves 11.9 / 25 / 43.6 us x 2 -> 0.02 / 0.05 / 0.09 ms per step.
* Family total: ~1.5 / 2.1 / 3.2 ms per step at 65K / 131K / 235K (of the ~2-5 ms/step the inventory attributes to E).
Prefill (100K, 160 s): the score GEMM path is untouched (0%). Reuse-gather stage 4.05 s = 5880 calls x
689 us; the b=512 local ratio 0.37 (synthetic locality is worse than production's, so use 0.4-0.45) ->
saves ~2.2-2.5 s, plus <= 0.65 s of the 8 index layers' gathers inside `prefill_indexer` -> ~1.8% of wall.
Select at prefill (512 WGs per call, 1568 calls, ~0.6x) ~0.2-0.3%.

## Integration notes
* score: `s2_w8n8_pf_hw` has the production signature and grid ((ceil(n/1024), b) x 256): drop-in body
  replacement of `indexer_score_wmma_batched_mw_e2m1` (K/indexer_score_wmma.hip:909). Bit-exact.
* select: `topk_select_v3_u8` has the production signature/grid ((b) x 1024): drop-in for
  `indexer_topk_select_batched_ilp` (K/indexer_topk_bitonic.hip:796). Identical selection incl. ties
  and the done[] fallback semantics. Scores row must be 16-B aligned (n_idx_stride*4 is; the buffer is).
* gather: `gather_u4_r1` needs the wrapper grid changed to (top_k, b) x 64 (S/indexer.rs:1151); the
  kernel takes head_dim at runtime but requires head_dim*2 % 16 == 0 (512 -> yes). Bit-exact copy.
* candidate_threshold_ilp: block 256 -> 1024 in S/candidate_blocks.rs:140. Identical threshold.
