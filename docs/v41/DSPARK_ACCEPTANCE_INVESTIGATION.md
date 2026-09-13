# DSpark acceptance investigation (2026-09-13)

**Question.** `scripts/v41_oracle/dspark_accept.py` reported draft-position-1 acceptance of 0.44 on
the model's own continuation (0.25 on a synthetic transcript), and PLAN §7f concluded from it that
speculation buys nothing on this hardware. The DSpark paper (Cheng et al. 2026, arXiv 2607.05147,
cited by the V4.1 tech report §2.4.3) reports ~0.93 at position 1. Is our number wrong?

**Answer.** Yes — a harness bug. Corrected, the reference drafter accepts **0.93 ± 0.03** at
position 1 on the model's own tool-call turn (0.84 ± 0.04 if the post-end-of-turn junk the
generator kept decoding is included), with conditional acceptance **0.91 / 0.95 / 0.89 / 0.94** at
positions 2–5, i.e. **E[accepted tokens/step] = 1.93 / 2.77 / 3.57 / 4.94 at K = 1 / 2 / 3 / 5**.
That is the paper's Math-domain number and above its Chat-domain number (Qwen3-4B, Figure 2),
and consistent with the "~5 token acceptance length" DeepSeek quotes for V4 production at γ=5.

## 1. Root cause: `ffn_norm.weight` was never loaded into the drafter

`lazy.load_module(mod, ckpt, prefix, skip)` filters parameters with `name.startswith(s)` for
`s in skip`. Both DSpark loaders (`dspark_accept.py` and `oracle_generate.py --dspark`) passed
`skip=("ffn", "embed", "head")` — meant to skip the routed MoE (replaced by `LazyMoE`) and the
tied embedding/head. `"ffn_norm.weight".startswith("ffn")` is also true, so every stage's
`ffn_norm` stayed at `RMSNorm`'s init of 1.0. `load_module` reports missing tensors only among
the names it did not skip, so nothing was printed. The backbone was never affected: `oracle.py`
and `oracle_generate.py` load `block.ffn_norm` as its own sub-module.

The trained gains are far from 1:

| stage | `mtp.S.ffn_norm.weight` mean (std) | FFN input scale error |
|---|---|---|
| mtp.0 | 0.157 (0.004) | 6.4× |
| mtp.1 | 0.200 (0.005) | 5.0× |
| mtp.2 | 0.241 (0.010) | 4.1× |

So every drafter FFN — the sqrtsoftplus router (top-3 of 128 + bias), the routed experts and the
shared expert, all with SwiGLU clamps at ±10 — saw an input 4–6× too large in all three stages.
The drafter still produced plausible tokens (its attention path and the tied head were intact),
which is why the earlier "structured-token sanity check" passed and why the number looked like a
weak drafter rather than a broken one.

**Fix** (`dspark_accept.py`, `oracle_generate.py`): `skip=("ffn.", "embed.", "head.")` — module
boundaries, not prefixes. `dspark_accept.py` now also runs `audit_stage()` after loading: every
parameter of each stage is compared by value against its checkpoint tensor (or checked finite
for the fp8→bf16 dequant path), and every `mtp.S.*` checkpoint tensor that no parameter consumed
is listed. After the fix: **0 problems, 0 unconsumed tensors** in all three stages. The old path is
kept as a control (`--legacy-skip-ffn-norm`, or the in-process `legacy` experiment which forces
the gains back to 1.0 on the loaded model) and reproduces the old JSON bit-for-bit
(0.438 / 0.258 / 0.135 / 0.067 / 0.056; RS 0.376).

### What the fix does and does not change (legitimacy)

The fix changes **weights only**. At step *i* (the main model has just processed position *i*)
the drafter sees exactly what the reference `forward_spec` gives it, unchanged from before:

1. the three window rings, holding `kv_norm(wkv(main_x_p))` for positions *p ≤ i*, where
   `main_x_p = main_norm(main_proj(cat(mean over hc copies of the residual ENTERING layers
   37/38/39 at p)))` — every entry a function of tokens ≤ *p*;
2. the anchor token: the token the main model emitted at step *i* (for position *i+1*);
3. four noise tokens (`dspark_noise_token_id` = 128799, same in both configs);
4. the markov bias chain: anchor → draft 1, draft *k* → draft *k+1*; the confidence head.

The main model's greedy token for position *i+2+k* (`main_argmax[i+1+k]`) is used only to score
draft *k+1*. Teacher forcing on the model's own greedy continuation is the free-running
trajectory: the `legacy` control's cumulative position-1 hits at 10/20/30/40 steps are 4/9/17/23,
the in-process free run (`oracle_generate.py --dspark`, same bug) had 3/8/16/23 — the difference
is the one-step start offset. The `hs=+1` probe below, which *would* leak the next position's
hidden state, scores lower than the fixed run, so none of the corrected number comes from leakage.

## 2. Method

Harness: `scripts/v41_oracle/dspark_accept.py` (reference `model.py` DSparkBlock / forward_spec
unmodified; CPU kernel shim; MXFP4 experts streamed per block into a bf16 LRU, lossless). Dump:
`~/.cache/deepstrix/v41/agentic/gen2` = `oracle_generate.py` on the 256-token agentic prompt,
96 greedy tokens of the model's own tool-call turn, T=352, residual after every layer,
`main_argmax` and `head_input` at every position. Drafter window seeded from positions 0..255,
89 scored steps (positions 256..344). All experiments run in one process on one weight load;
per-step records are in `gen2/dspark_accept_<exp>.json`. Tables from
`/tmp/.../scratchpad/analyze.py`; "≤330" restricts to steps whose five draft targets lie at or
before the `<|end_of_sentence|>` the model emitted at position 330 (the generator does not stop
at EOS; positions 331+ are BOS + free-association at 3–7 nats of main-model entropy, text no
server would ever decode).

Metric: greedy top-1 match of draft *k* against the main model's greedy token for the same
position; "cond p_k" = P(draft *k* accepted | drafts 1..k−1 accepted); E[tok/step] for K drafts
= 1 + Σ_{k≤K} P(prefix ≤ k accepted). "RS" = rejection-sampling acceptance at T=1,
Σ_x min(p_main(x), q_draft(x)), with q including the markov bias, main distributions from the
dumped head input through the tied head (for k>1: the main's distribution given the greedy
prefix, i.e. given drafts 1..k−1 accepted).

## 3. Results (gen2, tool-call turn)

| experiment | steps | pos-1 greedy ± 1σ | cond p2 p3 p4 p5 | E[tok/step] K=1 2 3 4 5 | main in top-5 | RS pos1 | RS E[tok] K=1 2 3 5 |
|---|---|---|---|---|---|---|---|
| **legacy** (old harness, ffn_norm = 1.0) | 89 | 0.438 ± 0.053 | 0.54 0.43 0.33 0.33 | 1.44 1.67 1.78 1.81 1.82 | 0.66 | 0.38 | 1.38 1.57 1.65 1.70 |
| legacy, ≤330 | 69 | 0.493 ± 0.060 | 0.59 0.45 0.33 0.33 | 1.49 1.78 1.91 1.96 1.97 | 0.75 | 0.44 | 1.44 1.68 1.78 1.84 |
| **base** (fixed) | 89 | **0.843 ± 0.039** | 0.87 0.92 0.88 0.91 | **1.84 2.57 3.25 3.84 4.38** | 0.92 | 0.85 | 1.85 2.55 3.15 4.13 |
| **base, ≤330** | 69 | **0.928 ± 0.031** (0.932 ± 0.030 on 73) | **0.91 0.95 0.89 0.94** | **1.93 2.77 3.57 4.28 4.94** | 1.00 | 0.93 | 1.93 2.73 3.44 4.64 |
| noseed (ring left at zero) | 89 | 0.764 ± 0.045 | 0.79 0.72 0.64 0.68 | 1.76 2.37 2.81 3.09 3.28 | 0.87 | 0.72 | 1.72 2.23 2.57 2.89 |
| incseed (ring seeded via the decode path) | 89 | 0.843 ± 0.039 | 0.87 0.92 0.88 0.91 | 1.84 2.57 3.25 3.84 4.38 | 0.92 | 0.85 | 1.85 2.55 3.15 4.13 |
| nomarkov (markov bias zeroed) | 89 | 0.640 ± 0.051 | 0.58 0.55 0.56 0.50 | 1.64 2.01 2.21 2.33 2.38 | 0.80 | 0.63 | 1.63 1.98 2.17 2.30 |
| hs=+1 (hidden of position i+1: leaks) | 89 | 0.809 ± 0.042 | 0.93 0.82 0.89 0.82 | 1.81 2.56 3.18 3.73 4.18 | 0.96 | 0.79 | 1.79 2.48 3.04 3.91 |
| hs=−1 (hidden of position i−1) | 89 | 0.719 ± 0.048 | 0.80 0.76 0.90 0.89 | 1.72 2.29 2.73 3.12 3.47 | 0.87 | 0.67 | 1.67 2.20 2.60 3.26 |
| ts=+1 (anchor = tok[i+2]) | 88 | 0.000 | — | 1.00 … | 0.02 | 0.01 | 1.01 … |
| ts=−1 (anchor = tok[i]) | 88 | 0.045 ± 0.022 | — | 1.05 1.08 1.11 1.14 1.16 | 0.14 | 0.04 | 1.04 1.06 1.08 1.11 |
| legacy, start 255 (in-process alignment) | 90 | 0.433 ± 0.052 | 0.54 0.43 0.33 0.33 | 1.43 1.67 1.77 1.80 1.81 | 0.67 | 0.37 | 1.37 1.57 1.65 1.69 |

PROSE_ROW_PLACEHOLDER

Other diagnostics of the fixed run: confidence-head AUC at position 1 **0.94** (was 0.52) and
0.80–0.82 at positions 2–5 — the confidence head is calibrated once its input is right, which
makes the paper's confidence-scheduled (adaptive-K) verification viable. Mean main-model entropy
0.57 nats over the 89 steps; 0.24 nats at accepted positions. Position-1 acceptance by quarter
of the run: 0.95 / 0.91 / 0.91 / 0.59 (last quarter = post-EOS junk).

## 4. The six hypotheses

1. **Window-ring seeding — correct, never the cause.** The harness seeds every prefix position
   through the reference's own `start_pos == 0` path (slot = position mod 128, last 128
   positions, RoPE at the true position, fp8 fake-quantised) from the attention INPUT of layers
   37/38/39 (`layer_{l-1}_residual`). `--check-ring` compares that ring against one filled by
   stepping `DSparkAttention.forward` one position at a time (the decode path): stages 0 and 1
   bit-identical, stage 2 differs in one element by 2.4e-4 (one bf16 ulp, batched vs single-row
   accumulation). The `incseed` experiment (drafting after decode-path seeding) is identical to
   `base` on all 89 × 5 draft positions. `noseed` shows the ring is genuinely used (0.764 vs
   0.843 at position 1; 0.21 vs 0.56 at position 5).
2. **Markov conditioning at depth 1 — correct.** Reference `forward_head`: `output_ids[:, 0]` =
   anchor; the bias for draft 1 is `markov_head(anchor)`, then each draft biases the next. The
   harness passes the anchor through `forward_spec(input_ids, …)` unchanged. `nomarkov` costs
   20 points at position 1 (0.640), so the chain is live and conditioned on the right token.
3. **Position alignment — correct.** `main_argmax[j]` is the greedy token for position j+1
   (teacher-match ≡ greedy-acc on greedy text confirms it); draft k+1 for position i+2+k is
   scored against `main_argmax[i+1+k]`. Every ±1 shift is worse: the token slot by ±1 destroys
   the drafter (0.00 / 0.045), the hidden by ±1 loses 3–12 points. (0, 0) is the maximum.
4. **Metric — like-for-like.** The paper's Figure 2 is position-wise conditional acceptance under
   greedy rollouts on the target's own generations (Qwen3-4B: Math ≈ 0.93, Chat ≈ 0.80 at
   position 1; Eval at T=1.0 elsewhere); at position 1 conditional = unconditional. Ours is the
   same statistic. Rejection-sampling acceptance at T=1 equals greedy here (0.85 vs 0.84; 0.93 vs
   0.93 pre-EOS) because the main model is peaked on its own text; on the old, broken drafter RS
   was *lower* than greedy (0.38 vs 0.44) — a symptom, in hindsight, of a drafter whose
   distribution was wrong, not merely uncertain. The paper's per-position curve is for a
   different target model and drafter; DeepSeek publishes no per-position number for V4-Flash,
   only "60–85% faster than MTP-1" and "~5 token acceptance length" at γ=5 (V4-Pro, SGLang blog),
   which our 4.4–4.9 at K=5 matches.
5. **Weights/config — one bug (above), otherwise clean.** Post-fix audit: 0 parameter problems,
   0 unconsumed `mtp.*` tensors per stage (attn incl. `attn_sink`, both norms, all six mHC
   tensors, `main_proj`+`main_norm`, `norm`, `markov_head.{embed,head}`, `confidence_head`,
   gate `weight/bias/bias_vl`, shared experts). Stage order 0→1→2 by construction; the 128 /
   top-3 router comes from `args.get_moe_config(layer_id ≥ 40)` in both `Gate` and `LazyMoE`
   (a 384-row gate would fail the byte-size check on load).
6. **Noise slots — as the reference.** `forward_embed` (called unchanged) fills slots 1–4 with
   `dspark_noise_token_id` = 128799; `inference/config.json` and the HF `config.json` agree.

## 5. Why the four old numbers looked "patterned"

They were the same broken drafter on different text, plus one bookkeeping accident:

* 0.575 (free run, 40 steps) = the first 40 positions of this turn, which are DSML tool-call
  markup; the teacher-forced control on the same span is 23/40 = 0.575 as well.
* 0.44 (teacher-forced, 89 steps) adds the last 20 positions, which come after the model's own
  `<|end_of_sentence|>` at 330 — free association at 3–7 nats, where even the fixed drafter is
  at 0.59. Acceptance in this transcript falls with position (0.50 / 0.59 / 0.36 / 0.32 by
  quarter under the bug); the "back half 0.70–0.80" was quarters 2–3 of the free run.
* 0.40 (`gen2/dspark_accept.json`, 40 steps) was the last entry of a `--variants` sweep, which
  overwrote the JSON — not a baseline.
* 0.25 (synthetic transcript) is human-written text: off-distribution for a drafter trained on
  target-model generations, and irrelevant to decode speed — decode only ever drafts the model's
  own output; human/tool text is prefill.

So there was no "healing"; a state defect would have shown up in `incseed`/`noseed`, and did not.

## 6. What it means for the decode plan

* PLAN §7f / §7f.1 and DECODE_M8_PLAN §3.4 were built on p1 = 0.44 (and "0.6–0.7 if a better
  drafter"); the measured chain is p1 ≈ 0.85–0.93 with ≈ 0.9 conditional at every later
  position. E[tokens/verify step] is 2.6–2.8 at K=2, 3.3–3.6 at K=3, 4.4–4.9 at K=5 — not
  1.7 / 1.8 / 1.8. The right K is now a cost question (per-pass expert bytes at B = K+1), not an
  acceptance question; the confidence head (AUC 0.94) supports the paper's adaptive verification
  length.
* The tok/s tables in §7a.1/§7d/§7e and DECODE_M8_PLAN §3.4 need re-running with these chains;
  this document does not redo the cost model.

## 7. Caveats and what is still open

* One text (73 on-distribution steps; ±0.03) plus the prose run above. Agentic traffic with
  long tool results is unmeasured but, again, only the model's own tokens are drafted.
* Numerics: the reference drafter with native MXFP4 experts (what the engine will hold) and
  fp8 projections dequantised to bf16 (the engine holds Q8_0; a smaller perturbation than fp8
  itself). The engine feeds the drafter its own backbone hidden states, whose Q8_0 drift from
  the oracle is ~2e-2 (M1 parity), so a few points of acceptance may go to that — measure once
  the drafter is in the engine.
* Greedy text at T=0; at T=1 the RS acceptance on the same text is the same number. Sampled
  rollouts at T>0 will be somewhat harder for the drafter (the paper evaluates at T=1.0).
* The synthetic-transcript number (0.25) was never re-measured with the fix; it is not a
  decode-relevant quantity.

## 8. Files

* `scripts/v41_oracle/dspark_accept.py` — fixed loader, `audit_stage`, bf16 LRU expert cache
  (`--cache-gb`), `--check-ring`, `--experiments` (base / legacy / noseed / incseed / nomarkov /
  hs=k / ts=k / start=p), per-step records in the JSON, conditional acceptance and RS chains.
* `scripts/v41_oracle/oracle_generate.py` — same skip fix for `--dspark`.
* `docs/v41/PLAN.md` §7f rewritten; §7f.1 superseded.
* Data: `~/.cache/deepstrix/v41/agentic/gen2/dspark_accept_*.json`,
  `~/.cache/deepstrix/v41/agentic/prose/dspark_accept_*.json`.
