# Decode plan — what to build, in order, and why everything else was cut
### 2026-09-15. Baseline: 4.94 tok/s (202 ms/token) after the host tuning.

Fifteen ideas were generated, five shortlisted, and an adversarial workflow
killed four of them. This is what survives, in build order, with the arithmetic
each step is accountable to.

## The hard cap, stated once

    token = 40 x [ box1_pre + max(box1_post_submit, box2_leg) ]

Box 1's non-waiting time is 202 - 110.6 = **91.4 ms/token and does not shrink**
when box 2 gets faster. Miss work may therefore not be credited past **~45 ms**
total; past that the pole moves to box 1 and further miss elimination pays ZERO.

Also: the marginal on-critical-path cost of a miss is **4.2 ms**, not the 6.6-7.7
ms the daemon reports. 20.4 x 6.8 = 136 ms does not fit inside the 110.6 ms
remote leg alongside box 2's 25.9 ms of zero-miss service, so 2.4-3.5 ms/miss is
already hidden behind box 2's own compute and is not ours to save.

## Step 1 — phase-aware global pool  (-30 ms, 202 -> ~172, ~5.8 tok/s)

Box 2 runs an independent LRU per layer (260 encoder / 68 decoder slots). One
pool at the SAME capacity halves misses: simulated 20.4 -> 10.4, and the measured
static version (164/164) gave 15.8-17.3 -> 9.65 with decode +22%.

The static version costs prefill -37% because box 1's pinned windows fall 21 ->
12. Phase-aware avoids that: prefill and decode never overlap within a request,
so the pool can evict globally during decode and stay inside each layer's region
during prefill. Slot CONTENTS survive the switch; only the bookkeeping changes.

  * 1a. DONE (`8f693aa`) — absolute slot addressing. `layer_views` hands the
    kernel the whole pool and `remap` holds absolute slots. Verified
    byte-identical (sha 81a75bb557e818e2). No kernel or wire change.
  * 1b. Cross-region eviction: shard-wide LRU keyed `(layer, expert)`; victim
    search global during decode, region-restricted during prefill (phase inferred
    from request `b`). Evicting another layer's slot clears that layer's remap
    entry and marks it dirty for re-upload.

Gate: total ms/token AND prefill measured back-to-back. Never quote remote_rtt --
it is an exposed wait and collapses if box 1 merely slows down.

## Step 2 — pipeline the serial misses  (-10 ms, -> ~162, ~6.2 tok/s)

`ensure_layer_inner` services misses one at a time against three shared staging
buffers, so two misses in the same layer cannot overlap. Per-miss staging plus
concurrent issue. Bills the same pool as step 1, so it must be re-baselined
after 1b lands (30 + 10 = 40 respects the 45 ms cap).

## Step 3 — DSpark end to end

The only remaining multiplier. Requires, in order:

  * 3a. A batched DECODE path — batched attention and head over B tokens with
    experts routed through T2 catch-all. `forward_prefill` is NOT it: measured
    775 ms at B=5 because it pages a per-layer union out of box 1's own disk, and
    it is unrollbackable (its post-chunk eviction relocates the KV window, which
    `rollback_kv` correctly refuses).
  * 3b. The drafter: `mtp.0/1/2` are in the checkpoint with trained `ffn_norm`
    gains. Each stage is attention + MoE + router — the bulk of the work.
  * 3c. Accept/reject loop. KV rollback is DONE and verified byte-identical
    (`c0e3c8e`), and REQUIRES `V41_T2_CATCHALL=2`.

E = 1.93/2.77/3.57/4.94 at K=1/2/3/5 from the Python oracle (engine-independent).
B=5 balances the two legs to within 5%.

## Cut, with the reason

| idea | claimed | realistic | why |
|---|---|---|---|
| Request concurrency | 101 ms | **0** | MEASURED: aggregate flat at 4.33/4.23/4.30 tok/s for 1/2/4 in flight. Requests serialize. |
| Host off critical path | 16.5 ms | 1 ms | `sel_sync` is a full dGPU drain billing the layer's dense chain, not sync overhead. Confirmed: it was UNCHANGED (574 us/layer) by disabling C3. |
| Expert sharding across boxes | 47 ms | 0 | The phase already costs max(box1, box2) and the split is already tuned to that maximum. |
| Miss-path copy deletion | 60 ms | 3 ms | All three suspects already shipped or measured backwards. Rescoped into step 2. |
| dGPU hot tier | — | 3-5 ms | 17.1 GB is a UTILISATION number; ~4.5 GB is actually free after KV and attention scratch = ~240 slots. Plumbing is refused on the paged path and static placement retains ~30% held-out. |
| Predictive prefetch | — | negative | Recall on MISSES 0.08-0.47; catching 47% costs 278 MB/token against 23.6 ms saved. |

## Honest ceiling

Steps 1+2 land ~**6.3 tok/s single-stream**. That is not 30. DSpark composed with
it reaches the mid-20s. The parked expert-format work (~2.5 bits, everything
resident, zero misses) remains the only measured route that actually clears 30 —
this plan is the best available without it.
