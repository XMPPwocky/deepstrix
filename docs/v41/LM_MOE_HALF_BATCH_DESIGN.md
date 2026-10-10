# Layer-outer MoE half-batching: one box-2 request per half-window per layer (design rev 1, 2026-10-10)

Status: DESIGN, nothing built. Follows `LM_LAYER_OUTER_PREFILL_DESIGN.md` (layer-outer = P2a, in production since 10-10).
Code map: read-only scoping 10-10 (line numbers below are `crates/v4flash-kernels/src/het/forward_prefill.rs` = FP,
`batch_scratch.rs` = BS, `remote_experts.rs` = RE at main 2fd3a7cc).

## 1. Why (measured)

* Production prefill (layer-outer + frozen split): 4096-row window in 3.8-4.0 s = 160 lane-unit-layers (20 encoder layers x
  8 units of 512 rows) at ~24 ms each. Per unit: hub work ~10.4 ms (reply -> next submit, 10-10 16:44 trace), box-2 run p50
  ~17-18 ms (+ wire ~7 ms f16 / ~13 ms f32 replies). Box 2 is the wall.
* **Box-2 prefill requests are WEIGHT-READ bound** (13,814 hit-only requests, 10-10 21:38-22:00 UTC): p10 run time is ~3 ms +
  ~100 us per DISTINCT expert (90 experts 11.6 ms, 130 15.5 ms, 190 21.3 ms), flat in rows. ~100 us x 18.8 MB = ~190 GB/s,
  the iGPU's DRAM roofline. A window re-reads ~8 x 134 = ~1070 experts per layer for a ~245-expert box-2 share.
* Box 2's same-layer merge never fires for prefill (0 of 13,814): `partner_follows` is false in the layer-outer path
  (FP:9392) and a pair of 512-row requests would need `REQ_FLAG_PARTNER` + both queued inside 400 us.

## 2. Idea

Split a window's U units (U = 8 for 4096 rows) into two HALVES of U/2 units. At each layer, the hub runs each unit's
attention + router + Q8 quantisation + its own (box-1) MoE per unit exactly as today, but ACCUMULATES the units' box-2 rows
on the host and sends ONE box-2 request per half (2048 rows). Box 2 reads each of its experts once per half instead of once
per unit: ~230 distinct experts per 2048-row half -> ~26 ms (int8 arm) vs 4 x ~16 ms today (floor model; f16 arm on box 2
~20 ms). The two halves ping-pong so the hub's dGPU chain of one half overlaps box 2 + wire of the other.

Expected (inferred): per half-layer hub 4 x 10.4 = ~42 ms, box 2 ~26 ms, wire back 17.5 ms (f16 replies) / 35 ms (f32,
current prod), wire out ~10 ms (full duplex). Hub becomes the wall: window ~40 x 42 ms = 1.7 s + gaps ~0.3 s = ~2.0 s =
**~2K rows/s** (vs ~1.05K today). With f32 replies the wire (35 ms) is close behind the hub; revert to f16 replies
(`V41_PREFILL_F16_REPLIES` default on) -- the 10-10 null showed f16 replies are not the run-to-run variance source.

## 3. The deadlock and the stash

Today's depth-2 lane pipeline (`Post(t-2)` before `Pre(t)`, lane = t % 2) needs unit u's REPLY before unit u+2 can reuse
its lane. A half's request goes out only after its LAST unit's route, so unit 0's post would wait for unit 3's route, which
waits for unit 0's lane: DEADLOCK. Fix: free the lane at the end of each unit's Pre by STASHING everything that spans the MoE
(map Q1: after_attn_hc, split, hc_pre_carry, ffn_shared, ffn_moe_recv; picks/xq are consumed before post):

| buffer | B/row | stash location |
|---|---|---|
| after_attn_hc | 81,920 | the unit's OWN `lm_store` residual rows: the layer-input residual is dead once the chain has produced after_attn_hc (FP:8533); Post writes residual_next back to the same rows as today |
| ffn_shared | 20,480 | window stash `st_shared` [win_rows x N_EMBD] |
| ffn_moe_recv (box-1 MoE, landed) | 20,480 | window stash `st_moe` |
| split, hc_pre_carry | 96 + 96 | `st_split` + the existing `lm_store` carry rows |

New dGPU memory: ~41 KB/row x 4096 = ~168 MB per window (allocated with `lm_store` / `lm_idx` in `lm_open_window`, halved
on OOM with them). Copies: Pre stash ~41 KB/row D2D, Post un-stash ~123 KB/row (incl. after_attn_hc from the store) -- vs
today's 82 KB/row store traffic; ~0.7 GB/window extra D2D at ~600 GB/s ~ 1.2 ms/window. Negligible.

## 4. Ops (unit level)

`Pre'(L,u)`:
1. as today: residual/carry/pos/saved-indexer/Engram staging, `pre_moe_chain` (attention -> mhc_pre_ffn -> router ->
   Q8 xq -> shared expert on de.compute).
2. `pre_moe_route` with a HALF ACCUMULATOR: readback picks/xq (rb_stream sync, FP:9588-9618), ownership per pick
   (`home2_or_partition`, row-independent), box-2 rows (xq_host, sel, ew for the box-2 picks) APPENDED to the half's host
   buffers at the unit's row offset; NO submit unless u is the half's last unit -> then ONE `submit_dispatch` of
   half_rows (partner_follows = false). The ticket goes to the HALF (not `bd.remote_ticket`).
3. `pre_moe_prep` + `pre_moe_launch` unchanged (box-1 MoE on the iGPU per unit; `si` stays serial across lanes).
4. STASH on de.compute: wait `moe_arrived` (the iGPU push into bd.ffn_moe_recv), then copy after_attn_hc -> store residual
   rows, ffn_shared -> st_shared, ffn_moe_recv -> st_moe, split -> st_split, hc_pre_carry -> store carry rows.
   The lane is free when these are queued (stream order protects the next Pre on this lane).

`Post'(L,u)`:
1. un-stash into the lane: store residual rows -> bd.after_attn_hc, st_shared -> bd.ffn_shared, st_moe -> bd.ffn_moe_recv,
   st_split -> bd.split.
2. box-2 part: the half's reply (waited ONCE, by the half's first Post'; host buffer kept until its last Post') -> upload this
   unit's row slice into bd.remote_ffn_moe (widen if f16).
3. `forward_layer_post_moe_v2` combine with the SAME add order (moe, shared, remote, hot) and hc_post (FP:12025-12397).
   Needs a variant that takes the remote partial already uploaded (skip its own ticket wait) -- or the half ticket stored
   per unit with a "pre-joined" flag.
4. residual_next / carry -> store (as today).

## 5. Schedule

Tasks = (L, half). `lm_schedule::ops` already generalises to any task count: with H = 2 halves per layer the op list is
`Pre(L0,H0) Pre(L0,H1) Post(L0,H0) Pre(L1,H0) Post(L0,H1) Pre(L1,H1) ...`, where `Pre(L,H)` = `Pre'(L,u)` for the half's
units in order and `Post(L,H)` = `Post'(L,u)` in order. Lanes alternate per UNIT (t_unit % 2) inside a half (lanes are
free after each Pre'), keeping the dGPU-chain / iGPU-MoE overlap between consecutive units. Box 2 + wire of half H0 overlap
the dGPU chains of half H1. Invariants to re-check (design doc sec 9 of layer-outer): I2 units of a layer in increasing
order (yes: H0's units then H1's), residual version (Post'(L,u) before Pre'(L+1,u): yes, depth-2 over halves).
U odd / small windows: halves of ceil/floor; U < 4 -> fall back to today's per-unit path.

## 6. Box 2

* Request rows up to half_rows (2048, or 4096 for a single-half experiment): the daemon's `--max-batch` is capped at
  `B_MAX` = 1024 (deepstrix-expertd main.rs:95,147), the client rejects `b > info.max_batch` (RE:10011), executor scratch is
  sized by rows (RE:7201-7232, partials 122,880 B/row -> +250 MB at 2048 rows; box 2 has ~6 GB free RAM). Introduce
  `REMOTE_B_MAX` (4096) for the remote request path only; lanes keep `B_MAX`. Hub-side remote host buffers are sized from
  `info.max_batch` at connect (RE:9850, 9888-9889, 10062) -> automatically larger.
* Box-2 redeploy needed (two-box restart order) -- bundle the pending landing-rank fix (f314438d) and the f16 MoE arm.
* Box 2 computes the half's rows in one executor pass: same kernels (WMMA arm >= 256/128 rows either way).

## 7. Fidelity

Box-1 math per unit unchanged. Box-2 per-row partial: reduction over a row's picks in fixed slot order (RE:7438-7447); the
open question is whether the WMMA kernels are tile-composition-independent PER ROW (rows share 16-member tiles with other
rows of the same expert: does a row's dot product depend on its tile-mates? The WMMA f32 accumulation over K is per output
element, so expected yes, NOT VERIFIED). Gate A (`lm_prefill_bitexact`, box 2 attached) with the knob on vs chunked
prefill: if tile-independent -> bit-identical; else f32 association -> statistical under a FIXED partition (pinned
routing + a fixed split; the live null is ~0.3 nats).

## 8. Knob, rollout

`V41_LM_HALF_BATCH` (LIVE, default off; decided per window like layer-outer; requires layer-outer and box 2 with
max_batch >= half_rows, else falls back per window with a warn-once). Live A/B by knob flip after one two-box restart.

## 9. Open questions for review

1. The `post_moe_v2` split: cleanest way to join a pre-uploaded remote slice (FP:12080-12228) without duplicating it.
2. `pre_moe_route` side effects per submit: pin ledger notes, prefetch-word drains (RE:10107,10132), pick trace,
   `note_submitted` (RE:9980): fine once per half? (They are per request today.)
3. Pager `ensure` + `drain_before_ensure` (FP:10625-10697): unchanged per unit -- any interaction with the delayed box-2
   submit (box-2-owned experts are excluded from the box-1 union per unit)?
4. Host memory/time for the accumulator: 2048 x 5840 B xq (12 MB) + sel/ew per half; the reply 42 MB (f32) / 21 MB (f16)
   per half-layer: copy into a pinned buffer once.
5. Does anything read `bd.remote_ticket` / `ev_req` / `lh2` per lane between Pre and Post (evtrace hub_req, lane health)?
6. Is ~2K rows/s right? The hub's 10.4 ms/unit was measured under group-major (2 lanes, box-2 bound); under half-batching
   the hub's dGPU chain + iGPU MoE per unit become the critical path -- measure them first (`V41_EVTRACE_DEV` short window).
