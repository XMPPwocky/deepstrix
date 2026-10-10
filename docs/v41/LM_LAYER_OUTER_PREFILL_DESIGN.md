# Layer-outer prefill, static expert split, rolling prefetch: design (rev 1, 2026-10-10)

Status: DESIGN FOR REVIEW. No production code written for it. Everything here is derived from reading the code and from the
measurements in `IGPU_MOE_F16LUT_PLAN.md` (POST-MOVE BREAKDOWN, ROOT CAUSE, streaming arithmetic). Numbers marked (inferred)
are NOT measured. Owner direction (10-10): "rebuild around layer-outer"; for long prefills changing the expert->box allocation
is acceptable; static split + rolling prefetch preferred over cache tuning.

## 0. Why (measured, one paragraph each)

* **Wall.** Steady 4096-row window = 8.02 s (511 rows/s, 10-10 16:44 cold 192K job). The hub is not the wall (reply->next submit
  10.4 ms, wire 7 ms/round trip); box 2 is: it serves the two 512-row lanes serially (19.4 ms kernel floor each => 40 ms period
  floor) and parks ~30 ms/unit-layer on page-ins (mean 7 missing experts/request; run 19.6/26.1/47.5/79.6 ms at 0-2/3-6/7-12/13+ missing).
  Hub-only gaps (Engram/G0 prep 1.25 s + group boundaries 0.63 s) are 23% of the window while box 2 idles.
* **Paging is a partition problem.** `expert_pager::partition_box2` = box 1 owns each layer's hottest ~103-115 ids by DECODE
  pick mass, box 2 everything else (~270/layer). Prefill touches ~every expert, so box 2's set is ~5.2-5.8K vs ~4.6K usable slots
  => ~1,146 re-reads (~21 GB) per window. Pins/budget ruled out (hub BAND=4096 releases pins to ~136 on prefill entry;
  `prefill_budget` is a decode-victim allowance and long jobs use `prefill_budget_long`=unlimited).
* **Streaming is bandwidth, not latency.** Each box streaming its half of the encoder weights = 72 GB/pass: needs window time
  >= 72 GB / disk rate (8K rows at ~900 rows/s needs ~8 GB/s per box; 16K rows ~4 GB/s). So: resident (set fits) at 4K windows,
  streaming at >= 8-16K windows.
* **The loop order forces big residency.** Today's unit = (layer GROUP, sub-chunk): a group's up-to-6 layers are re-run per
  sub-chunk, so the group's whole expert set must stay resident across 4 sub-chunks, and the prefetch lookahead is a whole group.

## 1. Goals / non-goals

Goals (each independently shippable behind a default-OFF knob, one-flip rollback):
1. **P1 Static prefill split** (hub-only): a capacity/bandwidth-balanced expert->box assignment for ENCODER layers of LONG jobs.
2. **P2 Layer-outer window driver**: for a window of W rows, run layer L for ALL lane-units, then layer L+1, as one continuous
   2-deep pipeline; removes group structure, fill/drain per group, and makes input prep overlappable.
3. **P3 Rolling whole-layer prefetch + larger windows** (8-16K): data-independent, lookahead 2-4 layers, both boxes stream
   their own share; window store lent from the idle KV-arena tail.
4. **P4 MoE batching**: box-2 pair merge (existing PARTNER mechanism) first; hub-side merged MoE later.
Non-goals: changing decode, DSpark, the KV arena, the CED/replay semantics, or the on-disk weight format. No expert requant.

## 2. Invariants that must hold (the review should attack these)

I1. **Row math unchanged.** For a fixed expert->box partition, layer-outer output (KV bytes, logits) is BIT-IDENTICAL to chunked
    prefill (the existing GPU gate `tests/lm_prefill_bitexact.rs` is the oracle; today's group-major LM already passes it).
I2. **Per-layer unit order unchanged.** For every layer L, lane-units execute their attention/KV-append/eviction in increasing
    position order (A_0, B_0, A_1, B_1, ...). This is exactly the order chunked prefill and group-major LM produce at that layer.
I3. **Dependency legality.** pre(L,u) needs only post(L-1,u) (own rows) and layer L's KV/ring/compressor after pre(L,u-1). Never
    post(L,u-1). (Verified by reading `pipelined_range_impl`; the review should re-verify against `pre_moe_chain`.)
I4. **No cross-task scratch dependence.** Everything in lane scratch (`BatchDgpuScratch`/`BatchIgpuScratch`) and shared scratch
    (`BatchDgpuShared`/`BatchIgpuShared`) is produced and consumed within ONE task's pre..post, except the items in section 4.
I5. **Decode safety.** Lane scratch is shared with decode steps (`prefill_job_lm_unit` already syncs and copies to the store before
    returning). The new driver drains and stores before ANY return to the scheduler.
I6. **Checkpoints/snapshots only between windows** (`checkpoint_ok`), unchanged.
I7. **Default OFF = byte-for-byte today's behaviour** (new knobs unset => existing code path).

## 3. The layer-outer driver (P2)

### 3.1 Task list and pipeline schedule
A window of W rows is cut by the existing `lm_plan_window`/`plan_chunk` into sub-chunks `(s,e,b_a)`; each sub-chunk gives two
lane-units (A: rows [s, s+b_a), B: rows [s+b_a, e)). With n sub-chunks there are U = 2n lane-units in position order
u = 0..U-1 (A_0, B_0, A_1, ...). **The lane cut stays exactly `plan_chunk`'s** (image spans are cut around by `lane_split`; the
code asserts the range returns the same cut: "lane cut != planned").

Tasks are T[t] = (L, u) in layer-major order for a quantum of layers [a, b): t = (L-a)*U + u. Lane scratch for task t is
`lane(t mod 2)` (bd_a/bi_a or bd_b/bi_b), and per-layer sync events are `sync_events.layers[L]` for even lanes,
`sync_events_t1.layers[L]` for odd (as today). The schedule is the existing steady state generalised:

    warm:   pre(T[0]); pre(T[1])
    step t: post(T[t-2]); (store out T[t-2]); pre(T[t])        for t = 2 .. N-1      (N = U*(b-a))
    drain:  post(T[N-2]); post(T[N-1])  (+ store out)

Today's steady state IS this with U=2: `post_A(L); pre_A(L+1); post_B(L); pre_B(L+1)`.
Legality for any U >= 2: pre(t) is issued right after post(t-2); the only data pre(L,u) needs from earlier tasks is
post(L-1,u) = task t-U <= t-2. For U >= 2 this is satisfied (equality at U=2).

### 3.2 State relocation (the full cross-task inventory; reviewer: find what I missed)
| state | today | layer-outer |
|---|---|---|
| residual [rows x HC_DIM f32] and `hc_pre_carry` [rows x HC_MIX_DIM] | lane scratch, swapped residual<->residual_next between layers | **`lm_store`** (exists: 82 KB/row): task start = async D2D copy-in to `bd.residual`/`bd.hc_pre_carry` (layers > a); task end (after post and the residual swap) = async D2D copy-out. ~0.14 ms per 512-row task each way (42 MB read + 42 MB write at ~600 GB/s) |
| layer-0 seed | host `hcs` copied to `bd.residual` (`RangeSeed::Host`) | unchanged for layer 0 tasks; hc_pre_carry reset via `hc_pre_onehot_rows` as today |
| compressor store lending (KV-source -> reuse layers) | `state.layers[L+1].compressor = state.layers[src].compressor.take()` before post_A(L)/pre_A(L+1), returned after pre_B(L+1) | bracket **per layer's pre-span**: take before the first pre of layer L+1 (task index (L+1-a)*U), return after its last pre (post never touches compressor state); with the skewed pipeline pre(L+1,u) tasks interleave with post(L,u') so the bracket spans from first to last pre(L+1,*) |
| `indexer_sel_saved`, `indexer_nsparse_saved`, `indexer_saved_store` (lane scratch; produced at index-source layers 2, 8, 14, written for the next up-to-5 reuse layers) | per-lane scratch, valid because the group cut guarantees the next sub-chunk does not overwrite it before the group ends | **per-window-row store** `idx_sel_store [W x 512 i32]` (2 KB/row = 8 MB @4096) + nsparse: copy-out at the end of an index-source task, copy-in at the start of each reuse-layer task. Exact field list and sizes to be fixed from `BatchDgpuScratch` when implementing |
| KV ring / raw SWA / compressor caches / index keys | `state.layers[L]`, positional | unchanged (per layer, positional; unit order I2) |
| Engram rows | host, staged per (lane, layer) right before the chain (`stage_engram_lane`); window copy `LmWindow.engram` | unchanged staging; source = the window input ring (section 6) |
| MTP capture / `mtp_lane_cut` | decode-only | n/a |
| `split`, `after_attn_hc`, `ffn_shared`, `ffn_input_norm`, `remote_*`, tickets | lane scratch, pre->post of one task | unchanged (I4) |
| per-layer `sync_events` | per layer x 2 lane sets, reused each layer | reused across units of the same lane parity at the same layer; safe because post(t-2) precedes pre(t) |
| pager state, `b2_mirror` windows, prefetch queue | global, time-ordered | unchanged; the ORDER of requests to box 2 changes (all units of layer L together) |

### 3.3 Quantum and scheduler interface
The scheduler calls the LM driver once per *quantum* instead of once per (group, sub-chunk): `prefill_job_lm_pass(job, a, b)` runs
layers [a, b) over ALL lane-units of the open window and returns, after the drain+store, the rows completed (W when b is the end
of the window's last layer, else 0). Default quantum `V41_LM_QUANTUM_LAYERS` = 4 (inferred: ~1.5 s, drain+store ~50 ms => ~3%).
Quantum boundaries replace group boundaries as the points at which decode may interleave; `lm_mid_group()` ("finish the group
before letting decode in", `V41_MS_FINISH_GROUP`) becomes "quantum in progress" and is vacuous because a quantum is atomic.
Layer 20 (`KvSourceOnly`, no MoE) is a quantum of one layer: pre only, no post, no residual change.
**The quantum is a scheduling unit only; the pipeline is continuous inside it.**

### 3.4 What gets deleted/replaced
`lm_groups` (group cuts) and the per-group `pf_sent` machinery are replaced by the quantum/layer plan; `prefill_job_lm_unit` is
replaced by `prefill_job_lm_pass`. The group-major driver stays in the tree, selected when `V41_LM_LAYER_OUTER` is off (I7).

## 4. Static prefill split (P1) and why it is separable

`home2 = expert_pager::partition_box2(layer, e)` at the pick loop (`forward_prefill.rs` ~9610-9630). Add
`prefill_home2(layer, e)`: for encoder layers (layer <= `CED_DECODER_START`) of a LONG job (tokens > `V41_LONG_PREFILL_TOKENS`)
and `V41_PREFILL_SPLIT` set, a **deterministic hash split** with box-2 share s2 (`V41_PREFILL_B2_SHARE`, default 0.5, weight by
disk bandwidth x pool capacity when the drives change), identical for every unit/layer/window of the job; decoder layers (replay)
keep the decode rule. Box 1 pages its half from its own disk (`V41_B1_PAGE_MISSES=1`, `V41_PAGER_MISS_PAR=8`) and the
decode "holder" fallbacks stay decode-only. Hub-only change; box 2 serves what it is sent (`--paged`).
Resident fit (inferred): box 2 20 x 192 = 3.8K <= ~4.6K usable; box 1 3.8K of 5,425 (plus ~700 decode-hot displaced, re-warmed after).
Numerics: which box sums which expert changes => f32 association only (same class as any partition change); I1 is stated
"for a fixed partition"; the partition change itself is gated by golden/KLD.

## 5. Rolling prefetch (P3)

Plan, not prediction: for each layer L in the quantum, the hub knows both boxes' encoder shares (section 4) => the full list
of experts each box must have. At task (L,u), the hub issues prefetch for layer L+k:
* box 2: prefetch words `(layer<<16)|e` carried on the request frames (existing `push_prefetch_words`/`set_prefetch_take_cap`;
  take cap raised so ~192 words/layer are drained over the U requests of a layer: >= ~24 words/request). Box 2's existing
  `own_sweep_rank` eviction (pages of layers the pass has gone past first) is exactly the rolling policy.
* box 1: `pg.prefetch_hint(layer, e)` + the async fill machinery (`b1_prefetch`).
Lookahead k (layers): k = 1 + ceil(u_disk) jitter margin, ~2 at disk utilisation <= 0.5, 3-4 at 0.8 (inferred). Resident need
= (k+1) x 192 x 18.8 MB ~ 11-18 GB per box. Layer-outer is what makes k small; group-major needs a whole group (~43 GB).
Disk need: 72 GB/pass/box; feasible when window time >= 72 GB / rate => W >= ~8K at ~900 rows/s (~8 GB/s/box), ~16K at 2K rows/s.
Box-1 second SSD: owner will fit a 9100 Pro 1 TB (check the slot's PCIe generation; put the encoder-layer experts on a
PLAINTEXT partition: box 1 reads through dm-crypt today).
Larger windows: `lm_store` W x 82 KB (8K = 672 MB, 16K = 1.34 GB), `idx_sel_store` +16-32 MB. dGPU has ~0.5 GB free at rest, so P3
needs the phase lend of the idle KV-arena tail (2.5-2.8 GB typical; census 10-09 dgpu-fusion session) or, as a stopgap, a
smaller `V41_MS_CTX_ROWS`. Fallback = the existing halve-on-OOM in `lm_open_window`.
Fully resident mode (P1 only, 4K windows) needs no prefetch beyond the cold fill; P3 is for fills after decode bursts and big windows.

## 6. Input pipeline (kills the 23% hub-only gaps)

Layer 0/1 need, per sub-chunk, the embedding HCs and Engram rows (Engram gather is disk-latency-bound: 1.7 ms/24 cold rows).
Today they are prepared between units (G0 prep 1.25 s/window + 0.63 s boundaries). Inputs depend only on token ids, not on compute:
a background producer fills a **ring of per-sub-chunk inputs for window n+1 while window n computes** (host RAM: HC 80 KB/row
= 336 MB/4K window, so for W >= 8K produce HC on the device from token embeddings instead; Engram rows ~24 KB/row/table).
Scheduler API: `set_window_inputs` (whole window, lazy mode) replaces the per-sub-chunk `set_chunk_inputs`; `engram_ahead` already
exists as the producer and moves from "next sub-chunk" to "next window".

## 7. MoE batching (P4)

7a. **Box-2 pair merge.** Change the schedule from depth-2 singles to depth-4 PAIRS: `pre(2j); pre(2j+1); post(2j-2); post(2j-1)`
with `partner_follows` on the first of each pair so the two requests (same layer, consecutive units) coalesce in box 2's existing
merge path (`REQ_FLAG_PARTNER`, `V41_B2_MERGE`, 400 us wait) into ONE 1024-row pass: one read of each expert's weights serves both.
Needs two extra lane-scratch sets (+~290 MiB: dGPU 128 MiB + iGPU 16 MiB each; `sd`/`si` are first-written/last-read inside pre, so
they stay single). Box-2 `--max-batch` is already 1024. Inferred: box-2 floor per pair ~28 ms vs 2 x 19.4.
7b. **Hub-side merged MoE** (iGPU kernel sees 1024+ rows): `si`/`bi` resized, merged work-items; larger change, later. The f16 MoE
arm (`cand_f16.hip`, 1.4-1.7x gate+up at >= 48 members/expert, ~15-26% at 8) is independent and lands in dispatch whenever.

## 8. Phases, gates, estimates (all estimates inferred from the 10-10 breakdown; steady window 8.02 s = 511 rows/s today)

| phase | change | gate | rollback | estimate (steady window / rows/s) |
|---|---|---|---|---|
| P0 | f16 MoE arm into `het::dispatch` (`V41_MOE_WMMA_F16`) | oracle + golden/KLD | knob | -4..-5% |
| P1 | static prefill split (long jobs) | `verify_routing_exactly_once`, golden/KLD vs old partition, one long A/B with the same evtrace analysis; hub `ensure` stalls in window 1 | knob | ~4.5-5.4 s / 760-900 |
| P2 | layer-outer driver at W=4096 + input ring | `lm_prefill_bitexact` extended with a layer-outer arm (4 cases), tail-latency of decode interleave | knob | -1..-1.5 s more: ~3.3-3.8 s / 1.1-1.2K |
| P3 | rolling prefetch + W 8-16K + arena-tail lend | bitexact at W=8192; decode p90; arena reserve/admit races (the lend rules) | knobs | robustness, enables decode-optimal ownership again; ~same or better |
| P4a | pair merge on box 2 | bitexact (merge is a box-2 pass; f32 association only) | knob | box-2 floor -25%: another -10..-15% |
Target region after P0-P4a: ~1.3-1.8K rows/s (inferred); 2K needs the dGPU chain and the hub MoE to keep up (no data: device
timing is off; enable `V41_EVTRACE_DEV` for a short prefill-only window first).

## 9. Test plan
* CPU (no GPU): a pure task-schedule generator + simulator: for random U (2..16), quantum sizes, layers: assert every (L,u)
  executes once, I2 order per layer, pre(t) strictly after post(t-2), residual flow (each pre reads the output of its own
  post(L-1,u)), compressor lend brackets balanced and non-overlapping, drain leaves nothing in flight, quantum boundaries
  resume correctly (state machine: window open -> quanta -> close).
* GPU (server DOWN): `lm_prefill_bitexact` + a `layer_outer` arm: fresh / multi-window / restored / lazy+foreign. Compare KV
  bytes and logits to chunked prefill (same partition). Then a W=8192 case.
* GPU live-window A/B: evtrace join (the `pf2` scripts), steady-window rows/s, decode p90 across a prefill.
* Existing guards to keep green: `verify_routing_exactly_once`, the loopback tests for any box-2 change (P4a only needs flags).

## 10. Risks / open questions (reviewer: please be adversarial here)
R1. **Drain cost** at quantum ends and any hidden per-task host sync that serialises the pipeline (e.g. `drain_before_ensure`).
R2. **`indexer_*_saved` exact inventory** and whether reuse layers read them for rows other than the current task's.
R3. **Compressor lending bracket under the skewed schedule** (pre(L+1,*) interleaves with post(L,*)): confirm post never touches
    compressor state and that `compressor.take()` while a prior task's kernels are still queued on `dgpu.compute` is safe
    (the move is of the Rust Option, not device memory; kernels hold device pointers).
R4. **Image spans / lane cuts**: units must reproduce `plan_chunk` cuts; spans crossing lane or sub-chunk boundaries.
R5. **dGPU VRAM** for W >= 8K and the lend protocol with `reserve`/`grow`/compaction/`fill_reserved`/parked `try_admit`.
R6. **Hub pager capacity** under P1 (prefill set ~3.8K + decode hot ~2.3K > 5,425), and hub `ensure` becoming a synchronous
    stall on box 1's dm-crypt disk in window 1.
R7. **Scheduler**: `FINISH_GROUP`, burst timing, `checkpoint_ok`, admission of other prefill jobs, `lm_rows`/`lm_store` lifetime.
R8. **Numerics of the partition change** and of the f16 arm vs the golden gate (already owed for f16 box-2 replies).
R9. **Whether the hub's dGPU chain or iGPU MoE becomes the pole after P1** (unknown: no device timing in the trace).
R10. **Replay/decoder layers** keep the decode partition: confirm replay (128 rows through layers 20-39) is unaffected and that
    its expert residency (decode-hot) is not destroyed by the prefill split.
