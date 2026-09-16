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

## Narrowing the 0.276-nat batched-chain divergence (2026-09-15, cont.)

Compared the two box-2 MoE chains kernel by kernel:

- **Gate/up is IDENTICAL.** Decode uses `mxfp4_pair_matvec_fused_swiglu_batch_hetsplit`
  (via the shared `mxfp4_pair_row`), batched uses
  `mxfp4_pair_matvec_fused_swiglu_kwide`. Both finalize with exactly:
      if clamp: g_sum clamped upper; u_sum clamped both sides
      sig = 1/(1+expf(-g_sum));  mid = g_sum * sig * u_sum * ew
  Same clamp asymmetry, same sigmoid, same ew multiply point. So gate/up is not
  the divergence (only matvec accumulation order differs, ~1e-5).
- **Q8_K(mid) is identical.** Blocks are over N_FF_EXP within a (token,expert)
  row in both layouts, so the per-block scales match.

Therefore the 0.276 nats is in the DOWN path: decode's
`mxfp4_matvec_par_batched_hetsplit` (fused down + per-expert hetsplit reduce)
vs batched's `mxfp4_matvec_par_by_expert_kwide2` (per-expert partials) +
`reduce_partials_hetsplit`. Either a real numerical difference in the by-expert
down kernel, or a batch-membership/state issue (a buffer other than `partials`
— d_mid_cat non-member slots, work_items, or expert_members — carrying stale or
mis-indexed data at B>1). f32 reduce-order alone is ~1e-5, not 0.276, so a
structural cause is likely.

Next concrete step: force the batched verify to run the DECODE down kernel while
keeping the batched gate/up, and see if KLD drops to ~0. That isolates
down-numerics from batch-state. Then either fix the by-expert down or route the
verify's down through the decode kernel.

Prefill-side validation (per the user): the verify never runs the decoder, so
logits aren't comparable there — use activation RMSE at the last ENCODER layer
against the oracle's `layer_19_residual.pt` instead.

## ROOT CAUSE of the 0.276-nat divergence — FULLY LOCALIZED (2026-09-15)

Decisive isolation, down held constant (mid zeroed so the swap is valid):

    decode gate/up + decode down   KLD 0.0005   (full decode chain)
    batched gate/up + decode down  KLD 0.277    (ONLY the gate/up kernel differs)

The gate/up kernel accounts for the ENTIRE divergence. The down (`by_expert_kwide2`
+ `reduce_partials_hetsplit`) and the swiglu finalization are both innocent —
proven identical earlier.

The two gate/up MXFP4 matvecs use different ARITHMETIC:
- **decode** `mxfp4_pair_row` → `dot_super_half_mxfp4`: dequants each MXFP4 weight
  through an int8 LUT and accumulates the dot product in FLOAT (`float acc`).
- **fast/batched** `mxfp4_pair_matvec_fused_swiglu_kwide`: integer dp4a
  (`sudot4_pair`, int32 accumulation of Q8_K×MXFP4 nibbles) then one float scale
  `sumi * (yd * gds)` at the end.

MXFP4 carries a per-32-block E8M0 scale, so an integer dp4a that sums across
blocks before applying scales cannot be bit-faithful to the per-block float
accumulation. That precision gap is the 0.276 nats (argmax agree 0.80). The
decode float-LUT path is the validated reference (the engine's decode was built
against the CPU oracle); the kwide dp4a is the fast prefill kernel, and its
small per-token error — invisible when prefill only consumes the last token —
becomes visible when a speculative verify consumes ALL per-token logits.

### The fix for fast + faithful (→ 20 tok/s)
Make box 2's `kwide` gate/up numerically match the float-LUT path: apply the
MXFP4 per-block E8M0 scale per block inside the dp4a accumulation (scale each
32-block's integer partial before summing across blocks), instead of one scale
at the end. Then the fast chain (283 ms) becomes faithful (KLD → ~0) and, at the
measured drafter E≈3.0, prices at ~48 ms/token ≈ 20 tok/s. This is the last
item; everything else (verify window addressing, accept-survives-eviction, the
two box-2 races) is fixed and proven.

Diagnostics left in tree, default-off: `V41_B2_DECODE_DOWN=1` (per-token decode
down over batched gate/up), `V41_DSPARK_XCHECK=1` + `mean_kld_nats`,
`V41_DUMP_FIRST_LOGITS`.

## Kernel-diff status: same block granularity, residual is subtler (2026-09-15)

Read the kwide gate/up in full: `mxfp4_unpack16` extracts the E8M0 block scale
`gds` per unpack and applies it per-block (`sumi_g * (yd*gds)`), using the same
`s_lut` and `yd` as decode's `dot_super_half_mxfp4`. So the 0.276 is NOT
block-scale granularity. Remaining candidates, all subtle: float accumulation
(decode) vs int32 dp4a-then-scale (kwide) ordering across the 256-block; a
rounding difference in `mxfp4_unpack16`'s LUT path; or the `half` gate/up pairing.

DEFINITIVE next experiment (standalone, no server): dump both gate/up `mid`
outputs for one expert on identical `(xq, weights)` — decode via
`mxfp4_pair_row`, batched via the kwide inner loop — and diff element-wise. That
pinpoints the exact diverging operation, which the source read alone could not.
Then fix the kwide op and re-run `V41_DSPARK_XCHECK` (expect KLD -> ~0), giving a
fast+faithful verify and ~20 tok/s at E≈3.

Oracle validation note (2026-09-15): a fresh KL(oracle||engine) attempt gave 21.7
nats — an ORACLE-HARNESS artifact, not an engine error (the oracle mis-processes
chat special tokens; engine argmax "We" is correct, oracle argmax " (" is not).
"Which chain is closer to the oracle" therefore rests on existing layer-parity
validation: decode ~ oracle (M1-M6 PASS); kwide is the lossy prefill kernel. Fix
kwide toward the float-LUT path.

## CORRECTED CONCLUSION (2026-09-16): no kernel is broken; DSpark's economics are the wall

The loopback test overturned the kernel-divergence diagnosis. Full corrected picture:

- **Box 2's batched vs decode MoE kernels are IDENTICAL** (loopback CHAIN-DIFF,
  same weights+inputs: max rel 1.5e-7). The gate/up-is-the-culprit and
  down-is-the-culprit diagnoses were both WRONG — artifacts of confounded
  server-level measurements (separate runs; box 2's global-pool LRU makes the
  f32 partial sum-order history-dependent).
- **The verify's server-level KLD decomposes** (clean single run, warm):
    catch-all ON   0.356 nats   argmax 0.87
    catch-all OFF  0.074 nats   argmax 0.93
  The 0.28 the catch-all adds is pure f32 REDUCE-ORDER (verify routes all experts
  to box 2; decode splits box1/box2 — same values, different grouping), and
  decode is itself non-deterministic in that order. The residual 0.074 is box 1's
  prefill-vs-decode forward. Neither is a lossy kernel.

**Measured DSpark accept, all fixes in, fast chain + catch-all, fresh 500-tok:**
    E 1.811,  190 ms/token = 5.2 tok/s
vs baseline decode 67 ms/token = 15 tok/s (warm, floor 0). DSpark is a ~3x net
LOSS, and it is NOT a correctness/kernel problem — it is economics:

- shadow E (drafter scored vs true decode) = 3.077, but accept E = 1.81 because
  the catch-all's reduce-order drops verify argmax agreement to 0.87, and in
  accept mode a wrong accept cascades (the faithful catch-all-off verify would
  raise E but costs box-1 paging, i.e. a slower verify).
- the verify step is ~190-345 ms (box 1's dense forward + drafter + RTT +
  normalize), and E×decode_token = 1.8×67 = 121 ms. The verify's fixed cost
  exceeds what the drafter can repay because DECODE IS ALREADY FAST (15 tok/s
  after the O_DIRECT + pool-floor work).

**Bottom line:** DSpark cannot beat non-speculative decode on this two-box setup,
not because of a bug but because decode is now cheap (67 ms) and a two-box verify
is not. The lever for >15 tok/s is decode residency (fit more experts in RAM: the
expert-format work), not speculation. Every correctness bug found this pass is
real and worth keeping (they were corrupting the verify and, via the shared
KvMark/partials/read_f32 paths, risked decode too), but they do not make
speculation pay here.
