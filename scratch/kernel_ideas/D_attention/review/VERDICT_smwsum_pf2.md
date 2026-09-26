# Review: D_attention/smwsum_pf2 (attention_dec_smwsum_pf2) — CONFIRMED

Reviewer runs (dGPU via scheduler, label review/D_attention, tickets in results/tickets.txt):
repro_run{1,2,3}.txt = engineer's exact repro (harness2 unchanged, CANDS=attention_dec_smwsum_pf2,
b = 1 2 3 4 5 8, graph of 10 calls, 60 interleaved rounds). rv_check_run1.txt / rv_decode_run1.txt =
reviewer harness (bmax 16, 7 extra shapes).

## Baseline fidelity
* base_attention_mixed / base_attention_dec hsaco rebuilt from the UNMODIFIED in-tree sources
  (git diff 361d4f9 -- kernels/ src/attention*.rs: empty) with $KFLAGS_V41 --genco: sha256 IDENTICAL
  to the engineer's objects. Candidate rebuilt from cand_attn.hip: pf2 ISA identical to the
  engineer's object (isa_diff2.sh; hash differs only by the embedded source path).
* Symbol, grid (4, b, 1) x 512, arg order/types and semantics match
  S/attention.rs:1119-1147 (`launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows`) and
  the dec score matches S/attention_dec.rs:32-78 (grid (ceil(n_tot/256), 4, b) x 512).
  `max_keys_words` differs (2568 vs 11536) but is unused: mask = null on both decode call sites.
* Shape = production decode (128 window + 512 gathered, stride 3072, comp_kv_batch_stride 512),
  graph-captured like production stages, warm operands (production reads V rows the score kernel
  just touched; a latency-bound kernel).
* Caveat (absolute only): harness smwsum = 32.5 us; commit 98bdd94 / the in-tree
  bench_decode_latency_breakdown (same shape) quote ~16 us. Unexplained 2x; does not affect the A/B
  ratio, does affect est_ms_per_token.

## Reproduction (medians of 3 separate runs, us per call)
  b:            1      2      3      4      5      8     16
  base pair   41.45  41.40  43.44  46.71  47.06  53.17  71.96
  pf2 pair    30.53  30.62  32.93  36.21  36.56  42.48  62.75
  ratio       0.737  0.740  0.758  0.775  0.777  0.799  0.872
Claimed b=4: 46.9 -> 36.3 (1.29x). Reviewer b=4: 46.71 -> 36.21 = 1.290x. Reproduced to <1%.

## Correctness
Bit-exact `out` on all 7 engineer shapes and on 6 reviewer shapes: b=7 odd rows/odd tails, b=1
127+497, b=1 1+1, b=2 dense store short row, b=16 gathered (max production b), b=16 dense mixed.
OVERFLOW PROBE n_total=641 (1 key beyond the kernel's DEC_MAX_KEYS=640): WRONG (32768 diffs, 512
non-finite) — the kernel has NO device-side guard. Production decode guarantees <= 640 (arena rows:
n_raw = min(n_raw+1, 128) FP:4095; n_comp capped at INDEXER_TOP_K on both indexer branches
FP:5133/5764; dense branch only when all rows <= 512), but attn_dec_score_for(b) is true for any
b <= 16, so a <= 16-row multimodal prefill chunk with an image raw window (up to 512, FP:4129)
could exceed it. Integration MUST gate on n_total_max <= 640 (hard error or fallback to prod).
Skipped work: the softmax weights are not written back to sd.attn_scores. No reader exists after
the smwsum launch (FP: last use 5892; forward_layer.rs uses a different kernel chain;
V41_VERIFY_DECODE_ATTN recomputes rather than reads). OK for decode.

## Verdict: CONFIRMED (1.29x at b=4, bit-exact, baseline faithful). Superseded in value by the
same engineer's fused_vt_qreg_sp_d4 (20 us pair) if that one confirms.
