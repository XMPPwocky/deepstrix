# Laguna prefill: two-lane pipelined het — investigation & port plan

**Status: READ-ONLY investigation (2026-07-27). No files changed, no GPU touched.**

## TL;DR (the premise is already satisfied)

The task asked us to make Laguna's het prefill two-lane, on the belief that
`prefill_batched_het` is single-lane and mutually exclusive with the two-lane
`prefill_batched_pipelined`. **That belief is false for the committed HEAD
(`79c4f99`).** Laguna already has a fourth prefill function,
`prefill_batched_het_pipelined` (`crates/v4flash-kernels/src/laguna_het.rs:2329`),
which is **two-lane AND het simultaneously**, and it is the **default** whenever
het is enabled (`LAGUNA_PREFILL_HET=1` + `LAGUNA_HOT_EXPERTS_DGPU=<file>`).

The dispatch chain:

- `prefill_batched` (`laguna_het.rs:1995`) → if `prefill_het && self.hot_split` →
  `prefill_batched_het` (`:2247`).
- `prefill_batched_het` (`:2247`) → if `LAGUNA_PIPELINE != 0` (**default true**) →
  `prefill_batched_het_pipelined` (`:2329`). Only `LAGUNA_PIPELINE=0` falls back to
  the single-lane split.

So enabling het already gives you the two-lane cross-device pipeline *and* the
hot/cold expert split. The confusion is a **stale comment** in the `prefill_batched`
dispatcher (`laguna_het.rs:2001-2002`: *"Single-lane, env-gated"*) that describes
the pre-pipeline state; both het and non-het paths default to two lanes now
(added together in commit `e986ad4`).

**The real remaining gap vs ds4 is small and structural** (a dedicated per-device
transfer stream), and the pipelined-het path appears **unbenchmarked** in the docs
(the PLAN's prefill table is the plain all-iGPU pipeline). See §4/§5.

---

## 1. Does ds4 do pipelined het prefill? — YES, both at once

ds4's `forward_prompt_batch_v2_pipelined` (`het/forward_prefill.rs:201-364`,
chunked entry `forward_prefill_pipelined` `:506-701`) runs a hard-coded **2-lane**
pipeline (lanes A/B = first `ceil(b/2)` / rest) *and* splits hot experts (dGPU) vs
cold experts (iGPU) inside every MoE layer of both lanes. Verdict: it does **both
simultaneously**, not one-or-the-other.

Mechanism (cited):

- **Lanes / scratch**: lane A uses `bd_a/bi_a`, lane B `bd_b/bi_b`; lane split at
  `forward_prefill.rs:230-237`. 2 lanes, not configurable.
- **Deep pipeline schedule** (`:270-348`): steady state enqueues, per lane,
  `post(L)` then `pre(L+1)` before switching lanes:
  `… [wait moe_A(L)] post_A(L) pre_A(L+1) [wait moe_B(L)] post_B(L) pre_B(L+1) …`,
  so lane A's iGPU MoE overlaps lane B's dGPU work.
- **Streams: 4 total — 2 per device** (`engine.rs`): each device engine has a
  `compute` **and a dedicated `xfer`** stream. Hot-expert GEMM runs on
  `dgpu.compute` (`:2315-2398`: group-build → q8k → fused swiglu → q2k down →
  reduce → `hd.ffn_moe_dgpu`); the activation/`selected`/`d_ew` peer-push runs
  **concurrently** on `dgpu.xfer` (`:2267-2298`). Cold-expert GEMM runs on
  `igpu.compute` after `wait_event(selected_pushed)` (`:2403-2834`). Hot and cold
  GEMMs have **no event dependency** on each other → genuine cross-device overlap.
- **Events: 6 per layer × 2 lane-sets** (`sync_events` = lane A, `sync_events_t1`
  = lane B; `LayerSyncEvents` in `engine.rs:229-236`): `ain_ready/ain_pushed`,
  `selected_ready/selected_pushed`, `moe_done/moe_arrived`.
- **Combine** (`forward_layer_post_moe_v2` `:2849-2897`): `dgpu.compute` waits
  `moe_arrived`, then `vec_add(ffn_moe_recv += ffn_shared)` then, if hot,
  `vec_add(ffn_moe_recv += hd.ffn_moe_dgpu)`, then post-FFN MHC.
- **Peer-copy hazard**: `het/sync.rs:19-62` `peer_push_f32/i32` assert the stream's
  device == source buffer's device (source-device-stream rule). iGPU→dGPU MoE push
  uses `igpu.xfer`; dGPU→iGPU activation push uses `dgpu.xfer`.

---

## 2. What Laguna's `prefill_batched_het_pipelined` already has

Structurally it mirrors ds4's design:

- **2 lanes** with per-lane scratch `ps_a/hs_a`, `ps_b/hs_b`, each sized to
  `ceil(b_max/2)` (`laguna_het.rs:2350-2354`) — same peak memory as single-lane.
- **Same deep schedule** as the plain pipeline: warmup `pre` both lanes, then
  steady-state `combine(L)/attn(L+1)/moe_split(L+1)` per lane, cooldown combine
  (`:2412-2434`). Layer 0 dense runs fully on the dGPU.
- **Per-lane events** `pipe_fn_in_evt[2]`, `pipe_moe_evt[2]`
  (`laguna_het.rs:266-267`, created `:685-687`) → two independent dGPU↔iGPU
  handoffs in flight.
- **Het split per MoE layer** (`moe_batched_split` `:1827-1962`): router + hot
  experts + q8k + hot down-reduce on `dstream` (→ `hs.acc`); cold experts +
  shared expert on `istream` (→ `ps.moe_recv`); combine on the dGPU
  (`combine_batched_split` `:1967-1985`): `h = op + (cold+shexp) + hot`.
- **Peer-copy correctness**: reuses the *same* `het::sync::peer_push_f32/i32`
  (`laguna_het.rs:38`, calls at `:1592/1866/1869/1958`), so the source-device
  stream rule is already honored (dGPU pushes on `dstream`, iGPU on `istream`).

Parity claim in-code: routed sum reordered (hot+cold vs all-iGPU),
greedy-exact + in-tol, oracle token 22718.

---

## 3. Diff: Laguna pipelined-het vs ds4 pipelined-het

| Aspect | ds4 | Laguna | Gap? |
|---|---|---|---|
| Two lanes | yes (A / t1) | yes (`pipe_*_evt[2]`) | none |
| Het hot/cold split per layer, both lanes | yes | yes (`moe_batched_split`) | none |
| Combine (vec_add hot+cold+shared on dGPU) | yes | yes | none |
| Peer-copy source-stream rule | `het/sync.rs` | same helper reused | none |
| **Streams per device** | **2 (`compute` + dedicated `xfer`)** | **1 (`dstream`/`istream` multiplex compute + peer copy)** | **YES — the one real gap** |
| Configurable lane count | no (2 hard-coded) | no (2 hard-coded) | none (parity) |
| Benched across context | prefill is ds4's mature path | pipelined-het **not in the docs' perf table** | validation gap |

**The single structural difference**: ds4 issues peer copies on a *dedicated*
`xfer` stream so the hot-expert GEMM (or cold GEMM) begins immediately while the
activation/result copy drains concurrently. Laguna multiplexes compute and peer
copies onto one stream per device (`laguna_het.rs:252-253` comments say so
explicitly), so in `moe_batched_split` the `peer_push` of `fn_in`/`sel`/`ew`
(`:1863-1870`) sits on `dstream` **ahead of** the hot GEMM (`:1873-1908`) and
serializes it; likewise the iGPU→dGPU `ffn_out` push (`:1954-1959`) serializes on
`istream` ahead of the next lane's cold work.

---

## 4. Port plan

The headline feature (two-lane + het) needs **no port — it exists**. The work is
(a) remove the confusion, (b) optionally close the one structural gap, and
(c) validate/measure. Order by value:

### Step 0 — Fix the stale comment (trivial, do first)
`laguna_het.rs:1999-2006`: update the `prefill_batched` dispatcher comment. It
currently says het is *"Single-lane, env-gated; falls through to the pure-iGPU
pipeline otherwise."* It should say het defaults to the **two-lane pipelined
het** path (`prefill_batched_het_pipelined`) unless `LAGUNA_PIPELINE=0`. This one
edit is what would have prevented the whole premise. Zero risk.

### Step 1 — Validate the existing pipelined-het path (no code change)
Before optimizing, confirm it is correct and measure it, because the docs never
did:
- Run the generate-vs-oracle harness with `LAGUNA_PREFILL_HET=1` +
  `LAGUNA_HOT_EXPERTS_DGPU=<file>` and confirm greedy token 22718 / in-tol at a
  few context lengths.
- Back-to-back A/B (per `feedback_bench_ab_methodology`) **pipelined-het vs plain
  pipeline** at 4K / 16K / 32K / 64K / 100K, `B_MAX=512`, `PIPELINE` on. This
  tells you whether het prefill wins at all before touching streams (§5 predicts
  it wins at short/mid ctx, tapers to neutral at long ctx).

### Step 2 — (Optional) dedicated per-device xfer streams — the actual ds4 port
Only if Step 1 shows the peer copies are on the critical path. Add
`dxfer: Stream` (dGPU) and `ixfer: Stream` (iGPU) alongside `dstream`/`istream`,
and route the peer copies + their event records onto them:
- In `attn_batched`/`moe_batched_split`: `peer_push_*` on `dxfer`; but the copy
  reads buffers produced on `dstream`, so first `dxfer.wait_event(<compute-done>)`
  then `peer_push` then `pipe_fn_in_evt[lane].record(&dxfer)`. The iGPU cold path
  keeps waiting on `pipe_fn_in_evt[lane]` — now recorded on `dxfer` — so ordering
  is preserved while the hot GEMM (still on `dstream`) no longer waits behind the
  push.
- Symmetrically, the `ffn_out` iGPU→dGPU push moves to `ixfer` with
  `ixfer.wait_event(<cold-done>)` then `pipe_moe_evt[lane].record(&ixfer)`.
- This mirrors ds4's `compute`/`xfer` split exactly and keeps the source-device
  rule (each push still uses its own device's stream).

### Step 3 — (Optional) context-length / hot-cap gate
Add a threshold so long context reduces `prefill_hot_cap` or falls back to the
plain pipeline (see §5). Cheapest form: gate on `n_kv_total` at tile time, mirror
the existing attention `PREFILL_ATTN_WMMA_MIN_KV` gating style.

### Risks
- **Event ordering / deadlock (Step 2)**: the compute stream produces the buffer
  the xfer stream copies; you MUST insert a `record` on the compute stream and a
  `wait_event` on the xfer stream, or the copy races the producer. Adding streams
  without this is a data-race, not a hang. Keep the two per-lane events but you
  now need a third pair (compute→xfer handoff) or reuse a scratch event.
- **Scratch sizing**: unchanged — lanes already size to `ceil(b_max/2)`
  (`:2350-2354`); the `HotScratch` per lane (`hs_a/hs_b`) already exists. Don't
  regress this to `b_max`.
- **Occupancy**: `moe_batched_split` adds router+q8k+hot GEMM+reduce to `dstream`.
  On global-attn layers at long ctx the dGPU is already busy (§5) — the extra
  hot work can *drop* the pipeline below the plain pipeline. Measure per depth.
- **Parity**: reorder of the routed sum is already accepted (greedy-exact). Any
  new stream split must not change kernel arg order, so parity should hold; still
  re-run the oracle.

---

## 5. Sanity check — is there dGPU headroom at long context? Honest gain curve

**Mechanism of the win**: in the plain pipeline all routed experts are on the
iGPU and the wall ≈ Σ(iGPU MoE), because the dGPU attention leg is hidden under
it. Het moves the K hottest experts onto the dGPU, **shrinking the binding iGPU
leg** by roughly `K/TOPK` of the routed-expert work. That is pure win **only while
the dGPU leg (attention + hot MoE) stays below the shrunken iGPU leg.**

**Where the dGPU stands (measured facts in the repo):**
- `PLAN.md:18-28`: plain-pipeline prefill = 490/467/443/362/**298** tok/s at
  4K/16K/32K/64K/100K. The 12 dense **global O(L²)** attention layers are the wall
  at long ctx; the WMMA attn kernel is at ~5% of matrix peak, memory/LDS-bound,
  occupancy lever dead.
- `PLAN.md:468-469`: **"Prefill stays iGPU-MoE-compute bound across the whole
  ≤262K range; global-attn O(L²) on the dGPU stays hidden under it until it
  crosses over near ~250–360K (≈ native max)."** — for the *plain* pipeline.

**Implication for het:** at short/mid context the dGPU attention is tiny, so the
dGPU is largely idle during the iGPU-MoE window → the hot-expert GEMM is close to
free and directly cuts the binding iGPU leg. As context grows, the O(L²) global
attention consumes that headroom; **and because het adds hot MoE onto the dGPU, it
lowers the dGPU↔iGPU crossover** below the plain pipeline's ~250K. So:

| Context | dGPU headroom | Expected pipelined-het vs plain pipeline |
|---|---|---|
| ≤ ~16K | large | Best win — hot MoE nearly free, iGPU leg cut by ~K/TOPK |
| ~32–64K | shrinking | Positive but tapering |
| ~100K | small | Small/marginal; approaching neutral |
| ≳ 130–250K | ~none (attn binds) | Neutral → **negative**; hot MoE steals dGPU cycles that attention needs → fall back to plain pipeline or drop hot cap |

**Honest bottom line:** pipelined-het prefill should help most at short-to-mid
context (≤~64K) and converge toward the plain pipeline as context grows, going
negative once global attention saturates the dGPU. This matches the user's own
suspicion that "it only helps below some context length." The gain is real where
the iGPU strongly binds and the dGPU is idle — which is exactly the ≤~64K regime
this hardware spends most prefill time in. No number can be quoted yet because the
pipelined-het path is **unmeasured in the docs** (Step 1). Recommend: bench it,
keep it default at short ctx, and gate hot-cap down (or off) past the measured
crossover (Step 3) so it never loses at 100K+.
