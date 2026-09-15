# DSpark: complete state after the correctness pass (2026-09-15)

## What now works (proven at the logit level, not greedy)

- **Verify fidelity is measured with KLD, not cosine.** KL(decode‖verify) over the
  row-0 softmax. Cosine on raw logits is scale-sensitive and ignores the softmax;
  KLD is the distribution-level truth. Instrument: `V41_DSPARK_XCHECK=1` emits
  `mean_kld_nats`.
- **The verify reproduces decode's distribution.** Decode chain, any context
  length: **KLD 0.0005 nats, argmax agree 0.99–1.00**. This required fixing three
  real bugs and one addressing mismatch (below).
- **Accept mode survives past the SWA window.** Previously died with "rollback
  refused … layer 0 wrapped" once the context passed 128 tokens.

## The four bugs fixed this pass

1. `read_f32` copied box 2's MoE output without syncing the compute stream —
   returned the PREVIOUS request's result. Decode asks f16 (synced); the verify
   asks f32 (didn't). This alone was the 40%-wrong-layer-partial corruption.
2. Box 2's batched MoE never zeroed its `partials` buffer — stale slots summed in.
   Only the batched branch accumulates, and REQ_FLAG_BATCHED is set exactly when
   b > 1, so this was the "b >= 2 rows per lane" trigger.
3. The verify (prefill path) addresses the raw window as `[0, n_raw)` while decode
   uses a monotonic ring `[raw_off, raw_off+n_raw)`. They agree only at
   `raw_off == 0`, so the verify was faithful short-context and KLD ~4 long. Fixed
   by `normalize_raw_windows` (compact to `[0,n_raw)` before the mark). Cleaner
   alternative not yet taken: offset the verify's slot + attention by `raw_off`
   (a no-op for real prefill, which is always at raw_off=0).
4. (earlier) The compressor store was never rolled back; the b<2 engram drop.

## The ONE remaining blocker for 20 tok/s

Box 2 has two MoE chains and the verify must pick one:

    chain              KLD vs decode   argmax agree   verify cost
    batched (fast)     0.276 nats      0.82           283 ms
    decode  (faithful) 0.0005 nats     0.99           407 ms

- The **faithful** chain gives correct output but is SLOW (per-token weight
  reads), so DSpark on it is a net LOSS: 407 ms / E≈1.8 ≈ 226 ms/token vs decode's
  67 ms.
- The **fast** chain is 283 ms but NOT faithful: 0.276 nats, argmax agree 0.82, so
  ~18% of accepted tokens are wrong → the leaked-token garbage seen in accept
  mode.

**20 tok/s needs the fast chain to be faithful.** The divergence is purely box 2's
batched by-expert down path (`launch_by_expert_kwide2` + `reduce_partials_hetsplit`)
vs decode's (`moe_down_batched_hetsplit`) — same Q8_K inputs, same clamp, same
gate/up quantization, so it is the down/reduce kernels. 0.276 nats with 82% argmax
agreement is a STRUCTURAL difference, not f32 rounding (which is ~1e-4).

### Two ways to close it
1. **Make the batched by-expert down numerically match decode's.** Diff the two
   down kernels: weight dtype handling, the point at which the routing weight `ew`
   is applied, and the partial-accumulation layout. This is kernel-numerics work.
2. **Pipeline B=1 decode-chain requests.** Send the verify as B separate b=1
   submits per layer (each takes box 2's faithful decode kernel) but pipelined so
   the RTT amortises. Faithful by construction; fast if the pipeline hides the
   per-request latency. This is the "batched decode entry point" and the larger
   refactor.

## Drafter (not a blocker; upside)

Matches the reference `noseed` config (E≈3.08 shadow, fresh ring). Prefill window
seeding is unimplemented and worth +1.1 E (oracle noseed 3.281 → base 4.382).

## Validation still owed — and it may REDEFINE the blocker

Compare the engine's per-position logits against the **CPU oracle's** logits (KL),
not just engine-verify vs engine-decode. Two reasons, the second decisive:

1. If BOTH engine paths drift from the true model, KL-vs-decode stays ~0 while both
   are wrong — I would never see it by comparing them to each other.
2. **The disagreement localises the error.** I have been treating the decode chain
   as ground truth and calling the batched chain's 0.276-nats divergence "the bug
   to fix". But the oracle is the actual reference. Computing KL(oracle‖decode) and
   KL(oracle‖batched) tells us WHICH chain is closer to truth:
     - if decode is closer, fix the batched down kernel (as assumed);
     - if BATCHED is closer, then the faithful-but-slow "decode chain" is the wrong
       one, and the whole faithful/fast tradeoff is inverted — the fast chain was
       right all along and the *garbage* came only from the two now-fixed races
       (read_f32 sync, partials zeroing), which the 0.276 was measured BEFORE
       being fully re-checked against truth.

So the oracle run is not just validation — it decides which kernel to change.
`oracle_generate.py` already computes full reference logits
(`head(norm(x), full_logits=True)`) and needs only a dump of them (currently just
argmax is saved); run it teacher-forced on a fixed prompt, dump the engine's
decode AND batched verify logits at the same positions, compute both KLs. Slow
(CPU, lazy streaming) but one-time and decisive.
