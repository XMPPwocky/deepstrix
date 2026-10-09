# Can a trained head beat the gate-margin heuristic at predicting the PROTECTED box-2 pick one layer early?

Offline, CPU-only, 2026-10-06. Scripts: `prep.py` (features), `study.py` (all methods, 5-fold), `lrcoef.py`
(LR coefficients), `freq.py` (trace rank-1 frequencies). Logs `study_top103.log`, `study_ge272.log`; `results_*.json`.

## Short answer

**No head trained on the residual beats the margin heuristic at 1K tokens, and it is not close.** The only
winner is a 15-feature logistic regression on quantities the hub already has in the readback pack (predicted
ids + gate weights): **+1.5 pt precision at matched volume (0.913 vs 0.898 at 3 hints/step; +3 pt under the
ids>=272 rule, 0.950 vs 0.924)**, stable from 80 training tokens. Residual heads (89K-1.2M params) land at
0.67-0.75 precision at 3/step, *below* the untrained gate, and their learning curve says parity needs
~10^4-10^5 tokens of residual dumps, which the CPU oracle cannot produce.

## Setup

* **Data**: `~/.cache/deepstrix/v41/agentic/main` is the only labeled dump: 1,006 tokens x 40 layers, residual
  `[T, 4 hc copies, 5120]` f32 + `topk_ids`. The other agentic dumps (~1,100 tokens) have residuals but **no
  topk_ids**; stage dumps are 6 tokens. Rows = (l, t), l in 0..38, target Lt = l+1: 39,234 rows.
* **Predictor input (proxy)**: mean over copies of the residual after layer l, layer l+1's ffn_norm + gate
  (the 09-14 `route_probe` method). Caveat: the live hub runs `gate_{L+1}` on **layer L's** ffn-normed router
  input (pre-MoE_L; `forward_prefill.rs:8558`, `bd.ffn_input_norm`). Heuristic and heads share the proxy, so
  the comparison is fair; absolute numbers are a few points optimistic.
* **Ownership**: primary = per-layer complement of the top-103 by live pick count (`hot_2337.json`, as Step 0);
  secondary = ids >= 272.
* **Protected set**: true rank-1 at Lt, box-2-owned, weighted by `p_miss(Lt, e)` from the LRU-4000 replay
  (`lru4000_p1.json`). Mass **6.69 words/step** weighted (29.0 unweighted) per 80 row-layers (2 rows x 40
  layers); predicted-rank-1 box-2 words at margin >= 0: 6.72/step.
* **Volume**: a candidate costs `p_miss` expected non-resident words; precision = hit mass / emitted mass,
  recall = hit mass / protected mass at V = 1, 2, 3, 5, 8 words/step. Unweighted precision (`pu` in the logs)
  tracks it within 1-3 pt.
* **Held-out**: 5 contiguous 201-token blocks; trained methods fit on 4, scored on the 5th, pooled.
* **Margin** = `(w1 - w2) / sum(top-6 raw weights)` as design 11.1 (raw sqrtsoftplus; top-6 by raw + bias).

## Ceiling

Protected recall within the look-ahead's predicted top-N (weighted; early L1-13 / mid 14-26 / late 27-39):

| N | all | early | mid | late |
|---|---|---|---|---|
| 1 | 0.710 | 0.614 | 0.718 | 0.773 |
| 2 | 0.861 | 0.776 | 0.867 | 0.917 |
| 3 | 0.912 | 0.848 | 0.918 | 0.955 |
| 6 | 0.955 | 0.908 | 0.965 | 0.980 |
| 12 | 0.971 | 0.936 | 0.984 | 0.987 |

97% of protected picks are in the predicted top-12 and 71% are the argmax, so the game is ranking the
predicted top-2/3 by confidence; early layers are the hard bucket. Step 0's k=0 vs k=1 (0.72 vs 0.71) shows
the look-ahead itself is free; the missing 29% at N=1 is layer l+1's own attention + MoE, invisible from
layer l's residual.

## Methods at matched volume (top-103 ownership, pooled held-out; precision/recall)

| method | V=1 | V=2 | V=3 | V=5 | V=8 |
|---|---|---|---|---|---|
| margin (heuristic) | .959/.143 | .932/.279 | .898/.403 | .802/.600 | .707/.710 (cap) |
| pred. weight, top-3 | .929/.139 | .891/.267 | .852/.382 | .764/.571 | .621/.743 |
| trace rank-1 freq | .755/.113 | .722/.216 | .735/.330 | .727/.551 | .707/.710 |
| margin x freq | .913/.136 | .869/.260 | .826/.371 | .765/.572 | .707/.710 |
| **logreg on pack features (top-4)** | **.968/.145** | **.947/.283** | **.913/.410** | **.813/.608** | .646/**.773** |
| untrained head, margin-scored (sanity) | .958/.143 | .931/.279 | .898/.403 | .801/.599 | .624/.746 |
| calib: tau + freq prior + per-(layer,expert) bias (15K) | .916/.137 | .875/.262 | .711/.319 | .608/.455 | .557/.508 |
| low-rank 16 (89K) | .847/.127 | .796/.238 | .723/.324 | .577/.432 | .428/.512 |
| low-rank 64 (355K) | .885/.132 | .799/.239 | .731/.328 | .583/.436 | .431/.515 |
| low-rank 64, early-stopped, wd 0.1 | .837/.125 | .753/.225 | .672/.301 | .522/.390 | .387/.463 |
| MLP 64x256 (446K) | .737/.110 | .551/.165 | .428/.192 | .490/.366 | .393/.470 |
| low-rank 64 + per-layer LoRA r4 (1.2M) | .879/.131 | .815/.244 | .751/.337 | .603/.451 | .457/.547 |

Design margin buckets (rank-1 only): >= 0: 6.37 words/step, p .725, r .690; >= 0.1: 1.88, .934, .263;
>= 0.2: 0.62, .967, .090; >= 0.3: 0.25, .981, .036. The 0.1 bucket is the 2-hints/step operating point.

* **Trace frequency is useless here**: box-2 experts are rare by construction, so a box-2 expert's rank-1
  frequency is ~1/384 noise; as a gate ("predicted rank-1 AND freq > p") it only removes recall.
* **The scoring rule matters more than any head**: scoring the untrained gate by softmax probability instead
  of the margin drops V=3 precision from 0.898 to 0.706.
* **Every residual head lost on held-out data**, with or without early stopping. Their own top-1 protected
  recall was 0.46-0.50 vs 0.71 for the untrained gate: 384-way cross-entropy on 31K samples de-calibrates the
  gate faster than it learns anything about layer l+1. The learned frequency prior (`beta` > 0) hurt.
* **Logreg wins modestly and for free.** Standardized coefficients: selection-score gap to the predicted top-1
  +2.47, sel1-sel2 +0.70, log trace rank-1 freq +0.54, predicted weight +0.43, log any-rank freq -0.32,
  rank-1 indicator +0.28, top-8 entropy -0.24, layer terms ~0.1. Its V=8 recall (0.773 vs 0.710) comes from
  emitting predicted rank-2 candidates once rank-1 volume is exhausted.
* **Per bucket at V=3 (margin -> logreg)**: early p .847->.874 with less volume (0.74->0.54 words), mid
  .875->.886, late .944->.943 with more volume (1.31->1.50) and recall .458->.527: mostly a reallocation of
  hints from early to late layers.
* **ids>=272**: protected mass 18.96/step; margin V=3/5 = .924/.146, .897/.237; logreg .950/.150, .930/.245.

## Learning curve (held-out = last 201 tokens; train on prefixes of the first 805)

V=3 precision, margin 0.765 on this block. Logreg 0.763 / 0.763 / 0.764 / 0.765 at 80 / 201 / 402 / 804
tokens (flat). Low-rank 64: 0.26 / 0.41 / 0.49 / 0.53; MLP 0.19 / 0.36 / 0.39 / 0.36. The head gains +0.15,
+0.08, +0.04 per doubling: decelerating; log-linear extrapolation puts *parity* at 6+ more doublings (~50K
tokens), with nothing suggesting it then exceeds the heuristic.

## Cost of the best head

* Logreg: 15 multiply-adds per candidate on the host, from the pack's 6 ids + 6 `look_ew` weights (selection
  gaps = raw + the known per-expert bias) plus an optional 40x384 host table of trace frequencies. Zero device
  cost.
* For reference, low-rank 64 = 355K params = 710 KB f16 per lane-layer batch, ~1.2 us at 600 GB/s plus two
  launches on the chain-enqueue path; LoRA r4 adds 44 KB/layer. Not worth building.

## Dump campaign?

For a residual head, yes, and it is not cheap: the CPU oracle made the 1,006-token dump at ~6 min/layer per
200 tokens (~20 h per 1K tokens), so 50K tokens is weeks. The hub's hooks (`DEEPSTRIX_DUMP_RESIDUAL_DIR`,
engine.rs:1499, one file per layer overwritten per token; the golden-gate `residual_sink` in
`fidelity_tap.rs`) live on the serial `forward_token_impl` path only, not the multistream/DSpark path, so a
campaign needs new hot-path code. Not recommended.

For the logreg no residuals are needed. The useful, fidelity-neutral collection is a host-side trace line per
decode row with the look-ahead's predicted top-6 ids + weights next to the existing `V41_PICK_TRACE` picks
(~60 B/row; the current trace has 8.9M decode rows in 900 MB): millions of labeled live rows, and it would
recalibrate the margin thresholds on the hub's real (pre-MoE_L) input, which this dump cannot.

## Caveats

* The dump says margin >= 0 emits 6.7 words/step for 6.7 protected at 0.72 precision; live measured 43 hints
  vs 3.5 blocked at lone DSpark. That 6x gap is not router accuracy (0.93 of hints are real top-6 picks live,
  0.937 here); it is the residency mirror, DSpark tree rows sharing picks, and the p_miss model. A better
  predictor lifts only the router-side factor (0.72 -> ~0.91 at 3/step); the volume problem is elsewhere.
* One agentic conversation, held-out blocks from the same document. Logreg's +1.5 pt is small enough that a
  live dry counter (`hits_prot / nonres` at a logreg threshold vs the margin buckets) should confirm it before
  any wire change.
