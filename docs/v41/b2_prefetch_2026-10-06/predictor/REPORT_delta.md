# Router + delta heads, regularized toward the router

Follow-up to REPORT.md (owner direction: "regularize towards the router"). Same data (agentic/main, 1,006
tokens, 39,234 (l, t) rows), 5 contiguous-block folds, top-103 ownership, p_miss-weighted protected set
(6.69 words/step), matched volume at 1/2/3/5/8 words per 80 row-layers, margin-equivalent scoring (the design's
`(w1-w2)/sum6` on the head's raw-equivalent weights; rank-2..4 candidates below rank-1). Scripts
`prep2.py`, `study_delta.py` (sweeps), `study_delta_t.py` (delta variants with a learned CE temperature);
logs `study_delta_top103.log`, `study_delta_t_top103.log`; `results_delta*_top103.json`.

Every head is the untrained one-layer-early gate plus a delta and reproduces the margin heuristic exactly at
zero delta (sanity row: .958/.143, .931/.279, .898/.403, .802/.600 at V=1/2/3/5 = REPORT.md baseline).
Per-layer parameters, one Adam per layer, 12 epochs, batch 128 tokens, L2 penalty = lambda x sum of squares
of the delta per layer. Objectives: `full` = CE over 384 experts on the true rank-1 at l+1; `box2` = CE over
box-2-owned experts on rows whose rank-1 is box-2-owned (the protected-rank-1 objective).

## Results (pooled held-out; precision / recall vs the protected set)

Reference: margin heuristic .959/.143 | .932/.279 | .898/.403 | .802/.600 | .707/.710(cap; .624/.746 with
rank-2..4 below). Margin buckets: >=0.1 = 1.88 words/step at .934/.263; >=0.2 = 0.62 at .967/.090.

**(a) affine, 385 params/layer: `z = a_l*sel + b_{l,e}`** (lambda on `sum b^2 + (a-1)^2`)

| lambda | V=1 | V=2 | V=3 | V=5 | V=8 |
|---|---|---|---|---|---|
| 0 | .974/.146 | .949/.284 | .909/.408 | .792/.592 | .616/.736 |
| 1e-3 | .974/.146 | .957/.286 | .912/.409 | .797/.596 | .620/.742 |
| **1e-2** | **.977/.146** | **.959/.287** | **.923/.414** | .816/.610 | .645/.771 |
| 1e-1 | .972/.145 | .950/.284 | .918/.412 | **.820/.613** | **.653/.781** |
| 1 | .960/.144 | .935/.280 | .904/.405 | .809/.605 | .631/.755 |
| 10 | .957/.143 | .933/.279 | .899/.403 | .803/.600 | .625/.747 (= router) |

Optimum lambda 1e-2..1e-1, flat over those two decades (V3 .918-.923), and never below the router at any
lambda (unregularized lambda=0 still +1.1 pt at V3, -1 pt at V5). The `box2` objective is uniformly ~0.5-1 pt
worse (.911/.409 at V3, lambda 1e-2): it cannot learn that the rank-1 is often a box-1 expert.

**(b) low-rank delta on W_gate(l+1): `logit' = logit0 + xn A_l B_l`**, r = 4 (22K params/layer) and 16
(88K), lambda on `||A_l B_l||_F^2`. Two trainings: plain CE, and CE with a learned per-layer temperature
(objective only; needed because raw gate scores are 9-62, so a temperature-1 softmax is near-hard and CE
drives huge deltas on the rows it gets wrong).

| variant | lambda 0 | 1e-3 | 1e-2 | 1e-1 | 1 | 10 |
|---|---|---|---|---|---|---|
| r4, V3 prec (plain) | .409 | .424 | .424 | .482 | .635 | **.900** |
| r4, V3 prec (+temp) | .474 | .442 | .404 | .558 | .892 | **.905** |
| r16, V3 prec (plain) | .482 | .494 | .504 | .542 | .666 | .896 |
| r16, V3 prec (+temp) | .314 | .312 | .342 | .611 | .889 | **.905** |
| r16 box2 (+temp) | .119 | .182 | .190 | .411 | .764 | **.911** |

Best points in full: r4 lambda 10 +temp .961/.144 | .936/.280 | .905/.406 | .812/.607 | .632/.756;
r16 box2 lambda 10 +temp .963/.144 | .947/.283 | .911/.409 | .822/.615 | .645/.772. The optimum is at the
edge of the sweep (strongest pull toward the router) for every (b) variant: the delta is 22K-88K parameters
per layer against ~805 training tokens per layer, and only a penalty strong enough to keep it near zero
survives held-out. Lambda 1 is already at or below the router; lambda <= 0.1 is a collapse (0.3-0.6).
The optimum is therefore not flat: it is a cliff on one side and the router on the other.

**(c) (b) + learned per-layer input scale g_l (5120, init 1)**, r16, `full`: plain .505/.537/.666/.898 at
lambda 1e-2/1e-1/1/10; +temp .348/.604/.890/.906. Indistinguishable from (b) at the best lambda; the extra
5,120 parameters per layer learn nothing the data can support.

**(d) 15-feature logistic regression stacked on the heads' (z, raw')**: on affine lambda 1e-2:
.977/.146 | .962/.288 | .926/.415 | .826/.617 | .652/.780; on r16 lambda 10: .981/.147 | .964/.288 |
.924/.415 | .821/.614 | .657/.786 (+temp: .919/.412 at V3). Versus the LR on the plain router (REPORT.md:
.968/.145 | .947/.283 | .913/.410 | .813/.608 | .646/.773) the stacking adds ~1 pt; the LR and the affine
head are learning overlapping corrections (the LR's biggest feature is the selection-score gap, which the
affine bias reshapes).

## Does the best variant beat margin >= 0.1 at the same volume?

Yes, modestly. At 1.88 words/step the margin bucket gives precision .934, recall .263. Affine lambda 1e-2
interpolates to ~.961 precision at that volume (.977 at V=1, .959 at V=2), recall ~.27; LR-on-affine ~.964.
The r16 box2 +temp head gives ~.949. So +2.5-3 pt precision at the >= 0.1 operating point, +2.5 pt at
3 words/step, +1.5-2 pt at 5, and at 8 words/step +4 pt precision and +6 pt recall because the heads also
rank the predicted rank-2 candidates (the margin heuristic is capped at 6.7 words/step by rank-1 only).

## Per bucket at V=3 (early / mid / late; volume, precision, recall)

margin: 0.74 .849 .320 | 0.96 .873 .411 | 1.30 .943 .457.
affine lambda 1e-2: 0.68 .912 .317 | 0.96 .892 .420 | 1.36 .948 .479.
r16 lambda 10: 0.89 .814 .370 | 0.83 .917 .374 | 1.28 .941 .447.
The affine head's gain is mostly in early layers (+6 pt precision at the same recall) plus a little more
volume spent late; the low-rank delta moves hints into early layers and loses precision there.

## Learning curve (held-out = last 201 tokens; margin on this block: V2 .858, V3 .764/.629, V5 .548)

V3 precision at 80 / 201 / 402 / 804 training tokens:

* affine lambda 1e-2: .759 / .768 / .780 / .777 (V5 .565 / .571 / .583 / .579). Above the router from
  201 tokens, +1.3-1.6 pt at 402-804, flat between 402 and 804: not monotone, saturating at ~+1.5 pt.
* r4 lambda 10 +temp: .765 / .764 / .769 / .777 (V5 .548 / .548 / .558 / .561): monotone but tiny, +1.3 pt
  at 804, and the first three points are the router itself (the penalty wins until there is enough data).
* r16 lambda 10 +temp: .765 / .765 / .766 / .776; (c): .764 / .765 / .764 / .777. Same shape.
* r4 / r16 lambda 10 plain CE: .769 / .762 / .746 / .745 and .762 / .731 / .743 / .748: get WORSE with more
  data, because a fixed lambda is a weaker pull per step as the number of CE steps grows; a per-sample
  lambda schedule or the temperature is needed for the pull to hold.

Read: with the pull toward the router, the heads are safe (never worse than the router at the chosen lambda)
and improve by ~1.5 pt per 1K tokens of data; the slope from 402 to 804 is ~0 for affine and ~+0.8 pt for
the deltas. There is no sign of a large gain waiting behind more data on this document.

## Inference cost delta vs the plain gate

* (a) affine: a per-layer scalar on the selection scores plus a 384-float bias add per layer: zero (fold
  `b_{l,e}` into the look-ahead's router bias buffer, `a_l` into the margin threshold). Nothing new on the
  wire: the margin can be computed from the pack's 6 `look_ew` weights as today, except that the raw-equivalent
  weights are `raw + b/a`, so either the bias-corrected weights are packed (same 6 f32/row) or the host adds
  `b/a` from a 40x384 table before computing the margin.
* (b) low-rank r: one extra `[b, 5120] x [5120, r]` and `[b, r] x [r, 384]` matvec per lane-layer, or fold
  `W + A B` into the look-ahead gate weights once at load (then zero cost; 5120x384 f16 per layer is already
  resident). No extra readback.
* (c): fold `g_l` into the look-ahead's ffn_norm weights: zero.
* (d) stacked LR: 15 multiply-adds per candidate on the host from the pack's ids + weights plus two 40x384
  host tables (trace frequency, gate bias). Zero device cost.

What the readback pack does not carry today: the predicted top-4 ids with their raw weights (the pack carries
the top-6 look-ahead picks and 6 weights, so the rank-2..4 candidates ARE available); the per-expert gate
bias and trace-frequency tables (host-resident); and, for (b)/(c) unfolded, nothing (device side).

## Bottom line

Regularizing toward the router turns "every head loses" into "every head is at least the router", but the
ceiling of what 1K tokens buys is +2.5 pt precision at 2-3 words/step (affine, 385 params/layer, lambda
1e-2..1e-1, a bias table at inference), i.e. .96 instead of .934 at the margin >= 0.1 volume. The low-rank
delta on the gate weights adds nothing over the affine bias at this data size and needs lambda at the edge
of the sweep to be safe. The owner's deployable version is (a) with the pack-derived margin, confirmed by the
live dry counters before any wire change; the data to train it at scale is the look-ahead trace line
proposed in REPORT.md (predicted ids + weights beside the pick trace), not a residual dump.
