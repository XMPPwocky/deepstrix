# Early enqueue for the ready-first decode driver: design

Status: **design only (2026-10-16).** Branch `early-enqueue`, cut from production `87816c34`. Work stopped
before any code, as the brief requires when the design turns out much bigger than it looks. The version
the brief describes (one dGPU stream per lane) is that case; see section 1. Section 3 defines the smallest
safe slice (S1: early enqueue on the one existing stream, with a per-lane flag). It is ready to build.
Nothing in the tree changed; knob-off behaviour is untouched by construction.

Code references are to `crates/v4flash-kernels/src/het/forward_prefill.rs` at `87816c34` unless noted.

## 0. Problem and goal

The production driver is `forward_step_arena_ready_first` (`V41_MS_STAGGER=2`). It enqueues a lane's
`post(l)` (`forward_layer_post_moe_v2`: wait `moe_arrived`, take box 2's reply, upload the partial,
combine, or defer the combine under G6) and its `chain(l+1)` (`pre_moe_chain`, about 94 us of host
enqueue per lane-layer, `lh.pre_moe`) only after the host has taken that lane's box-2 reply. The live
perfetto trace (200 steps, 2 lanes) shows two things. About 36% of chain starts are gated on box 2. After
a reply lands, the dGPU then stays idle a further p50 67 us (p10 46, p90 95) while the host takes the
reply and enqueues the post and chain. That is about 0.45 ms/step of pure host latency on the critical
path.

Goal: put lane X's `post(l)` and `chain(l+1)` on the GPU **before** its reply arrives. The GPU gates
them on a device-visible flag. The host sets the flag once the reply's bytes are in place. The chain
then starts within microseconds of the release, and the host's enqueue time leaves the critical path.

## 1. Verdict and scope

| slice | what | size | removes |
|---|---|---|---|
| **S1** (recommended next) | early enqueue on the ONE existing `de.compute` stream; per-lane flag in coherent pinned memory; the engine thread releases it | ~700-900 LOC incl. test; no new stream, no new VRAM | the 94 us chain enqueue and most of the 67 us post-reply idle on every lane-layer that is eligible (section 3.4) |
| S2 (follow-up) | S1, plus box 2's reader thread copies the partial into the lane's staging and releases the flag itself | +~250 LOC in `remote_experts.rs` | the wait while the engine thread is busy in another lane's route/prep/launch (pager `ensure`, Engram join) |
| S3 (separate project) | one dGPU stream per lane, so the lanes stop sharing one in-order queue | 2-3 kLOC, a second `BatchDgpuShared`, a full shared-resource audit, a new gate | also the head-of-line block between lanes (about 14.5% of chain starts idle about 350 us behind an iGPU leg), plus everything in S1 |

**Why S3 is "much bigger than it looks".** Section 2 has the inventory. In short:
1. Every dGPU launch in `pre_moe_chain`, `pre_moe_route`, `pre_moe_prep`, `pre_moe_launch` and
   `forward_layer_post_moe_v2` uses `self.dgpu.compute` (or `de.compute`). That is 485 references in those
   ~7,800 lines alone, plus helpers: `issue_shared_expert_prefill`, `launch_g1`, `launch_g4`,
   `launch_g6`, the `MhcArena`/attention dispatch, `stage_b` and graph contexts. A lane stream either
   needs that parameter threaded through all of them, or a "current stream" indirection inside
   `v4flash_hip::Stream`. The indirection is a global hack, and wrong for any other thread that touches
   `de.compute`.
2. `BatchDgpuShared` (`sd`) is ONE set for both lanes. The chain uses about 70 of its buffers (`sd.mix`,
   `sd.attn_input_norm`, `sd.q_normed`, the R1/R2/R3 lifetime-union arenas, `mhc_counters`,
   `qkv_front_ticket`, `look_sel*`, `verify_*` and more). Their documentation says they are "safe only in
   program order on ONE stream". Two concurrent lanes need a second `sd`. Production allocates `sd` at
   `dgpu_rows` with `attn_scores` sized from `--ctx` (about 629 MiB at 512 rows before the ctx term). A
   decode-lane-sized copy still costs tens to hundreds of MiB of dGPU VRAM that the prefill VRAM loan
   uses. That needs the owner's sign-off.
3. Lanes of one stream on an UNORDERED cut can hold sibling tree rows of ONE slot that do not cross
   (`StepRows::crosses` false). Today one stream serializes their KV, compressor and index-key writes. On
   two streams they would run concurrently. Before S3 can treat "unordered" as "independent", someone
   must prove that no read-modify-write is shared (compressor accumulators, comp rows), or order every
   same-slot pair.
4. Null-stream blocking calls (`hipMemcpy` sync, `hipMemset`, `hipFree`) synchronize with ALL blocking
   streams on the device. Per-lane streams therefore do not lift the deadlock rule of section 5. S3 needs
   it too.

S1 keeps every one of these exactly as today, because the stream stays one stream (section 6). It does not
fix the cross-lane head-of-line block on iGPU legs. That needs S3.

## 2. Shared-resource inventory (the brief's item 1)

How each resource is ordered today, under S1, and what S3 would need.

| resource | today (one `de.compute`) | S1 | S3 would need |
|---|---|---|---|
| `de.compute` stream | carries both lanes; program order = host enqueue order | unchanged; it also holds at most one unreleased `wait_value` per lane | one stream per lane, plus start/end joins |
| `sd` (`BatchDgpuShared`): chain scratch, R1/R2/R3 unions, `mhc_counters`, `mhc_inv_rows`, `qkv_front_ticket`, `indexer_topk_done`, `look_sel*`, `router_logits*`, `verify_*`, `kv_ring_scratch` | dead between phases, in stream order ("every sd/si buffer is dead by the time pre-MoE returns") | unchanged: the early batch is enqueued as a unit, and every interleave it creates is one the driver already creates (section 6) | a per-lane `sd` (VRAM), with every one-stream lifetime invariant re-proven |
| `bd` / `bi` / `dev` (per lane) | per lane | per lane, plus new per-lane fields `ee_pending` and the flag slot | per lane |
| `si` (`BatchIgpuShared`), `remap_dev`, `ie.compute` | iGPU, host-serialized route+prep+launch per lane (G5c) | unchanged (S1 never moves iGPU work) | unchanged |
| KV arena (raw KV, comp rows, compressor state, index keys) | per slot; ordered cuts by `may_enter`/`may_route` in stream order; `compact_for_step` at step start | unchanged: `may_enter` is still checked before any chain is enqueued, early or not (`deps_done` asserts kept) | a cross-stream event per (lane, layer) for ordered cuts; proof or ordering for same-slot unordered lanes; lane streams wait on the step-start event |
| `arena.state.with_kv_source` (host) | host, per call | unchanged | unchanged |
| pager, box-1 paging (`ensure`, `touch_resident`), hot set | host plus iGPU pool, in rest | unchanged (still in rest) | unchanged |
| b2 mirror / cache prior (`V41_SUB=3`) | `prior_pin[layer]` filled by the host at `chain(l)` enqueue; G1 reads it zero-copy, else an async H2D on `de.compute` | filled at enqueue, then **refreshed at release** (section 4.3), so it keeps today's snapshot point | same as S1 |
| `rb_pack`, `selected_ready` (per lane, per layer) | event recorded in the chain after the router | unchanged; S1's staging-reuse argument rests on it (4.4) | per lane stream |
| `remote_ffn_moe`, `remote_ffn_moe_pin`, `remote_upload_done` (per lane) | the host writes the pin after the reply, then an async H2D or the G6 zero-copy read; `done.synchronize()` before the next write | the host writes the pin at release, before the flag; the copy/G6 read is enqueued behind the flag; `done.synchronize()` is **replaced** by an assert (4.4); a host sync on `done` could deadlock | same as S1 |
| `de.xfer` (dGPU to iGPU pushes), `de.up` (G6 `early` copy), `de.hc` (`V41_MS_MHC_SPLIT`) | shared FIFO side streams, waiting on events from `de.compute` | `de.up` is not used under S1 (the copy goes on `de.compute` behind the flag, same bytes); MHC split must be off (asserted) | the same HOL coupling stays on these streams |
| graphs (stage_b ctx, legacy graphs) | production runs `V41_MS_GRAPHS=0`; stage_b ctx entries built, not read | must be off: S1 requires graphs off, asserted at step start; graphs-on keeps the old path | captures are stream-bound; graphs-off only |
| `sel_stats_decode` bank, `EventPool` stage timers | device atomics, per-call events | unchanged | atomics commute; the timeline assumes one track |
| step end (`de.compute.synchronize()`, then the head) | after the loop | asserted: every flag released before the sync | join both lane streams into `de.compute` |
| Engram rows (`LazyEngramRows::table`) | blocking join at the Engram layer's chain | early chain only if the table's gather `is_finished()`; otherwise post-only early (the host never blocks earlier than today) | same |

## 3. S1: early enqueue on the one stream

### 3.1 Why one stream works, and when it would not

Under S1, `de.compute` executes the lanes' early batches in **route (rest) order**. Today it executes them
in reply-arrival order. The two orders agree whenever lanes complete in the order they routed. That holds
for both legs except one case:
- The iGPU legs share one in-order `ie.compute`, and each lane's leg is enqueued in its rest, so
  `moe_arrived` fires in rest order.
- Box 2 answers in arrival order, **except** that a request which must page is PARKED (`REQ_FLAG_OOO`,
  `V41_B2_PARK`), and the request queued behind it is answered first.

So the only head-of-line cost S1 can add is a parked (paging) request. The other lane's early batch would
then sit behind the parked lane's flag. That is the coupling the park was built to remove (about 131
ms/step of box-2 queueing, 2026-09-23). Eligibility rules 3.4(2) and (3) exclude it:
- A lane-layer goes early only if its request is predicted NOT to page.
- No early wait is enqueued while another lane has a normal-path Post outstanding, because that Post
  would land behind the wait.

The residual is mirror surprises: the mirror said held, box 2 paged. They are counted (`ee_ooo`, 3.6).

### 3.2 The flag

- **Memory.** `ee_flags: PinnedBuffer<u32>`, one word per lane (`MAX_LANES`), allocated with
  `HIP_HOST_MALLOC_COHERENT` (fine-grained). The CPU writes it and the dGPU's command processor polls it.
  A non-coherent (coarse-grained) word may be cached on the GPU and never seen to change. If the GPU gate
  shows the coherent host word does not release, the fallback is `hipExtMallocWithFlags(...,
  hipMallocSignalMemory)`, which is what the HIP header documents for these operations. The wrapper is
  about 15 LOC in `v4flash-hip`.
- **Device wait.** `Stream::wait_value32_gte` (already in `v4flash-hip/src/stream.rs`; hipStreamWaitValue32,
  compared at execution time, so it may be enqueued before the host writes). On the iGPU it has been used
  since `03e0c757` (`moe_signal`). On the dGPU only `write_value32` is used today, and **nothing has ever
  had the dGPU wait on a CPU-written word: the GPU gate's first item** (section 8).
- **Support check.** At engine construction with the knob on, read
  `hipDeviceGetAttribute(hipDeviceAttributeCanUseStreamWaitValue)` on the dGPU. In HIP 7.2.3 the enum value
  is `10013` (`AmdSpecificBegin` 10000 + 13; check it against the header when adding the constant). If it
  is 0, return an error at startup. The knob never silently degrades.
- **Values.** Per step, per lane: the step start writes 0 (after the previous step's
  `de.compute.synchronize()`, so no wait is in flight). A wait for layer `l` is `>= l + 1`. A release of
  layer `l` stores `l + 1`. Values only grow within a step, never wrap, and the word reads as "last
  released layer", which helps debugging. An assert at release requires `stored < l + 1`. Poison stores
  `u32::MAX`, which releases every pending wait.
- **Host store ordering.** Write the pin (the partial bytes, the prior slot), then
  `fence(Release)`, then an atomic store to the flag word (an `AtomicU32` view of the pinned word). x86
  keeps stores in order, and the coherent mapping makes them visible to PCIe reads in that order. The
  wait-value packet completes before any later packet on the stream, so a kernel or copy after it reads
  the new bytes. The GPU gate verifies this by changing the data every iteration.

### 3.3 What is enqueued early, and the phases

At lane X's `rest(l)` (route, prep, launch, route probe: unchanged), if eligible (3.4):

```
post_enqueue(X, l):  wait_event(moe_arrived X,l)
                     [box-2 ticket]  wait_value32_gte(flag[X], l+1)
                                     G6 zc:      nothing (G6 reads the pin in chain l+1)
                                     otherwise:  H2D pin -> remote_ffn_moe on de.compute
                     the combine, or G6 defer (bd.g6_pending), exactly as post! decides today
chain(X, l+1)        if may_enter(X, l+1) and the Engram table is ready; else Ph::Chain(l+1) as today
```

Then the lane goes to the new phase `Ph::Release(l)`. Readiness is "reply ready" (`RemoteExpertClient::ready`),
or "immediately" when the lane-layer had no ticket. The action, `post_release(X, l)`:
1. `remote.wait(t)` (non-blocking in practice; the hint said ready).
2. Validate (`src.len() == rows`, kind, status, layer/b). Any failure is an `Err`, and the exit guard
   poisons (5.2).
3. Widen f16 to f32 if needed. Assert the staging-reuse invariant (4.4). Copy into
   `remote_ffn_moe_pin`.
4. Refresh `prior_pin[l+1]` (4.3).
5. Release `flag[X] = l+1`.
6. **After** the release: the reply telemetry the old post did (lookahead scoring, evtrace `HUB_REQ`,
   perfetto, `trace::phase`, `LH_REMOTE_WAIT`), `recycle`, `remote_late`.

Then `ph = Route(l+1)` if the chain was entered, `Chain(l+1)` if not, or `Done` after the last layer.

A lane-layer with **no ticket** (box-1-only run, or a layer box 2 does not own) gets no flag wait at all.
Its post is gated only by `moe_arrived`, a GPU event, so early enqueue is trivially valid. This is also
what makes S1 testable without box 2.

Depth is one layer ahead per lane, and it is structural, not counted. X's next early batch happens at
`rest(l+1)`. That needs `selected_ready(l+1)`, which is recorded in `chain(l+1)` behind the flag wait for
layer l. So a lane can never have two unreleased waits. This is asserted (`ee_pending` is `Some` at most
once per lane).

### 3.4 Eligibility (fast path) and the slow path

**Early** iff all of the following hold. Each "no" is counted by reason (3.6).
1. Static. `V41_MS_EARLY_ENQUEUE=1`, graphs off (`ms_graphs()`, one read per step), `!ms_mhc_split()`,
   `!REMOTE_DBG`, no sub-tensor dump armed on `l` or `l+1`, not `stats`. With a ticket, the async
   pinned partial path must be available: `remote_add_partial()`, `REMOTE_PARTIAL_ASYNC`, and the lane
   has `remote_ffn_moe_pin` and `remote_ffn_moe`.
2. Predicted no paging. With a ticket, every box-2 pick of the request is mirror-held at submit (the
   ticket's `held`/`n_held` against the request's distinct picks). Without pin mode there is no
   prediction, so the lane-layer is never early.
3. No other lane is in a normal-path Post. Its post would be enqueued behind this wait.
4. Not the test hold (`READY_FIRST_TEST_HOLD_LANE0` applies to releases, see 7).

The chain part additionally needs `l + 1 < N_LAYER`, `may_enter(X, l+1)` (ordered cuts), and the Engram
table for `l+1` gathered (`ScopedJoinHandle::is_finished`). Otherwise the batch is **post-only early**,
and the chain follows on the `Ph::Chain` path as today. Under G6 the deferred combine waits in `bd`
(per lane), exactly as it does across a `Ph::Chain` wait today.

**Not early** means today's code path, byte for byte (`Ph::Post`, `post!`, `chain!`).

**Host decisions after the reply** (the brief's item 2). They fall into three classes:
- Decided **before** the reply, so they can shape the enqueue: `g6_try`/`g6_mode`, `defer_shared`,
  `fuse_remote_add`, `shared_in_combine`, `hot_active`, `remote_ffn_moe_valid` (= ticket present and
  `remote_add_partial`), `REMOTE_PARTIAL_ASYNC`, the `mhc_ffn_split` wait. All are functions of knobs,
  `b`, the layer and per-lane state set in the lane's own `rest(l)`.
- Taken **after** the reply, but host-only, so they cannot change what was enqueued: f16 widening, size
  and status checks (errors), mirror updates inside `wait()` (`update_pinned`/`soft`/`nursery`,
  surprises), lookahead scoring, evtrace/perfetto/phase counters, the slack probe (a sleep before the
  release: it still delays the GPU, as intended), `recycle`.
- Taken after the reply AND able to shape the enqueue: **none in the arena path**. `miss_mask` has no
  box-1 fallback here (box 2 pages and computes; only the old single-token path at
  `forward_layer.rs:3051` reacts to it). So the release step is the "host does its checks, then
  releases" model the brief asks for, and it covers every lane-layer: there is no separate slow-release
  path.

### 3.5 Knob

`V41_MS_EARLY_ENQUEUE` is a static flag, default off, defined in `knobs.rs` (`Knob::flag(..).` with no
`.live()`) and listed in the deepstrix-server knob table and `docs/v41/DECODE_OPT_KNOBS.md`. It applies to
the ready-first driver only; the other drivers never read it. Knob off: the driver never enters
`Ph::Release`, never allocates or touches `ee_flags`, and `post!`/`chain!` run unchanged. For tests there is
a `#[doc(hidden)]` override `EARLY_ENQUEUE_TEST: AtomicU8` (0 = knob, 1 = force on, 2 = force off), the same
pattern as `READY_FIRST_TEST_UNORDERED`. It lets one process compare on and off.

### 3.6 Evidence counters (`ms.step`, deltas as in `87816c34`)

These come from a separate `early_counter_deltas()`, so they do not widen `fusion_counter_deltas` (which
sx-defer widens, see 9):
- `ee_n`: lane-layers with post and chain early. `ee_post`: post-only early. `ee_rel`: releases with a
  ticket. `ee_rel0`: no-ticket early lane-layers.
- `ee_no`: lane-layers on the normal path, split by reason: `static` / `page` (predicted paging) /
  `other_post` (3.4 rule 3) / `engram` / `hold`.
- `ee_ooo`: releases that happened while an earlier-enqueued wait of another lane was still unreleased.
  These are the head-of-line events, with `ee_ooo_us` (time until that wait released).
- `ee_poison`: flags poisoned on an error exit (want 0).
- `ee_lat_us`: sum of reply-landed (reader `t_recv`) to release, for the mean host latency left in S1.

## 4. Correctness details

### 4.1 Same kernels, same stream

S1 changes when the host enqueues `post(l)` and `chain(l+1)`, and adds one `wait_value` packet. It changes
no kernel, operand, row layout or stream. The per-lane enqueue sequence on `de.compute` is identical:
`chain(l)`, (rest), `post(l)`, `chain(l+1)`.

### 4.2 The interleaving is one the driver already produces

The early batch of X lands right after X's `rest(l)`. Relative to lane Y it can land:
- between Y's `chain(l')` and Y's `route(l')`. The doc comment of the driver names this as the only
  reordering ready-first allows ("another lane's post+chain landing between a lane's chain and its
  route"), and G5e/G5g/G5h gate it bit-exact today.
- or anywhere Y's own batch boundaries fall. That is also reachable today, whenever X's reply arrives
  within one poll of X's rest.

Ordered cuts: `may_enter`/`may_route` and the `deps_done` asserts are evaluated before any chain is
enqueued, early or not. So the later lane of a cut still enqueues its `chain(l)` after the earlier lane's.

### 4.3 The cache prior (`V41_SUB=3`)

`pre_moe_chain` builds `prior_pin[l+1]` from the b2 mirror and the pager at enqueue time. Today that is
right after reply(l) is consumed. Early enqueue would move it one reply earlier. The fill block becomes a
function (`fill_prior(bd, layer, pager) -> bool`). The release step calls it again for `l+1` before the
flag, so the values the kernel reads (G1 zero-copy or the async H2D, both at execution time, after the
wait) are computed at today's point.

The on/off decision (`prior_on`) is fixed at enqueue, because it shapes the launch. If the release-time
fill would decide differently (warm-up only: `delta(layer)` or `resident(layer, 0)` becoming known),
that is counted (`ee_prior_flip`). It is not an error: under SUB=3 the prior is timing-dependent today
too (G5g is bit-exact only with `V41_SUB` unset).

`n_protect`/`prior_dry` are live scalars read at enqueue. That is one reply earlier than today, and only a
live flip can tell the difference.

### 4.4 Staging reuse without a host sync

Today the post calls `remote_upload_done.synchronize()` before rewriting the pin. Under S1 that sync can
**deadlock**: the latest record of `done` may sit behind an unreleased wait. So it is replaced by an
invariant.

At the release of layer `l`, the previous read of the pin (layer `l-1`: the H2D in `post(l-1)`, or the G6
launch at the head of `chain(l)`) precedes `selected_ready(l)` on `de.compute`. A reply for `l` exists
only after route(l), which waited on that event. Assert `sync_events_lane(X).layers[l].selected_ready.query()? == true`
(cheap, non-blocking), with an error naming the invariant. At `l = 0` the previous step's final
`synchronize` covers it.

### 4.5 Exactness claim

With `V41_SUB` unset (the test configuration), knob on equals knob off bit for bit on every row:
- the same per-lane sequences (4.1);
- an interleave already gated (4.2);
- the same partial bytes: written before a Release store, read after the wait (gate item 1 verifies
  the visibility);
- no-ticket lane-layers have no flag at all.

## 5. Deadlocks and errors

### 5.1 The rule

While any flag is pending, the host must never block on `de.compute` work enqueued after the wait. Those
are the host's own releases, so blocking on them waits for itself. Sites:
- `de.compute.synchronize()`; `Device::synchronize` on the dGPU.
- Null-stream blocking `hipMemcpy`/`hipMemset`/`hipFree` on the dGPU. These sync with all blocking
  streams, and `de.compute` is blocking (`Stream::new`). Today on this path they are only in diagnostic
  branches: subtensor dumps, `V41_SPARSE_REMAP_SYNC`, `V41_PAGER_SYNC_AFTER_ENSURE`, `REMOTE_DBG`,
  `stats`, SUB=2's route-time `copy_from_host` (11179) and the hash router (V4.1 has `N_HASH_LAYERS = 0`).
- `Event::synchronize` on an event recorded on `de.compute` after a pending wait: `remote_upload_done`
  (removed, 4.4), and `selected_ready` entered on a failed hint.

### 5.2 Enforcement: asserts, not hope

- `v4flash-hip` gets a per-device `AtomicU32` "early waits pending". `Stream::synchronize` of a stream
  marked `deny_while_pending` (`de.compute`), the dGPU's `Device::synchronize`, and the blocking
  copy/memset/free on that device return `Err("blocking dGPU call while N early-enqueue waits are pending
  would deadlock")` instead of hanging. That is one relaxed load per blocking call (about 40 LOC).
- `Ph::Route` under S1 treats a failing `selected_ready.query()` as an error, not as "ready", the
  `unwrap_or(true)` it is today. A hint can no longer lead into a blocking wait.
- Exit guard (merged with sx-defer's wrapper, see 9). On ANY exit, every lane with `ee_pending` is
  poisoned (`flag = u32::MAX`, `ee_poison += 1`), so the queued work drains instead of wedging the stream.
  The step's result is already an `Err`. The GPU then runs the queued post/chain on stale staging, which is
  the same class of leftover as today's mid-step abort: the step is not accepted, and the rows past the
  arena counters are dead (G5f's invariant). A normal end with any flag pending is itself an error, after
  the poison.

## 6. Why S1 needs no per-lane `sd`

S1 never runs two lanes' kernels concurrently: `de.compute` stays one in-order queue. The `sd` lifetime
invariants are therefore exactly today's (4.2). The cost is the head-of-line exposure of 3.1, bounded by
the eligibility rules and measured by `ee_ooo`.

## 7. Test plan (`tests/multistream_step.rs`; build only, never run by the agent)

New arm group **G5i EARLY ENQUEUE**. It replays the ready-first schedules of G5e (unordered, 2 lanes), G5g
(`spec2` ordered, `spec2_hold`) and G5h (two streams' blocks in one step, all three cut kinds) on fresh
arenas with `EARLY_ENQUEUE_TEST = 1`. Gates:
- every kept row is bit-identical to the same schedule's knob-off arm;
- `ee_n > 0` and `ee_post > 0` (Engram layers and the last layer give post-only);
- with box 2 attached, `ee_rel > 0`; box-1-only, `ee_rel0 > 0`;
- `ee_poison == 0`, no deny-blocking error;
- `spec2_hold` under S1: the hold applies to the RELEASE (lane 0's release held until lane 1 has
  released). It must finish (no deadlock) and stay exact, with `ee_ooo > 0`, which proves the HOL path
  ran.
- **Negative control**: `EARLY_ENQUEUE_TEST_LATE_PIN` makes the release store the flag BEFORE copying
  the partial. With box 2 attached it must differ somewhere, which proves the flag gates real data. Not
  run box-1-only.

Host-only unit tests (these the agent may run): the eligibility predicate over a table of states, and
the flag protocol (per-step reset, monotone values, poison, the "at most one pending per lane" assert)
as pure functions.

## 8. What the GPU gate must run

The order matters. Exclusive GPUs, server down, live knob file neutralised.
1. **dGPU flag microtest** (new `tests/early_enqueue_flag.rs`, `#[ignore]`). Steps:
   - attribute 10013 on gfx1201;
   - a coherent pinned word, CPU-written after a random delay, releases a `wait_value32_gte` on a dGPU
     stream;
   - a watchdog thread poisons after 2 s, so a broken mechanism errors out instead of hanging the box;
   - data written to pinned staging before the flag is seen by an H2D copy and by a zero-copy kernel read
     enqueued after the wait, over 10k iterations with changing data;
   - release-to-kernel-start latency p50/p99 (expect about 5-20 us).

   If the coherent word fails, repeat with `hipMallocSignalMemory`.
2. `multistream_step` with G5i (box 2 attached; then box-1-only for the no-ticket path).
3. Live A/B per the bench methodology. Skip the first about 3 minutes (hot-set refresh). Score:
   - the perfetto "reply landed to chain start" distribution (today p50 67 us; target at most 20 us);
   - `ee_*` counters, `ee_ooo` rate and cost;
   - step time.

   The expected gain is below the 8% E2E noise floor, so do not judge it on tok/s.

## 9. Expected gain and risks

**Gain.** The ceiling is the measured host latency on box-2-gated chain starts, about 0.45 ms/step by the
trace's accounting: 36% of 86 chain starts, p50 67 us each, plus the chain enqueue it hides. S1 keeps the
time for the engine thread to notice the reply, plus the pin copy and the wait-value latency: an
estimated 10-25 us per start. S2 removes most of the "notice" part when the engine thread is busy
elsewhere. On a 60-120 ms step that is about 0.5-1%. Real, but only measurable with the counters and the
trace.

**Risks.**
- `hipStreamWaitValue32` is marked BETA in HIP 7.2.3. On the dGPU it may be implemented as a
  barrier-value packet or as a spinning blit kernel; either is functional, and gate 1 measures the
  latency.
- Coherent-memory visibility (gate 1).
- A deadlock through a blocking call nobody listed. Enforced by 5.2, so it surfaces as an error, not a
  hang.
- HOL on mirror surprises (`ee_ooo`; turn the knob off if the cost is high).
- The SUB=3 decision flip at warm-up (`ee_prior_flip`).
- More host work per lane-layer while waiting is fine; the release is about 10 us.

## 10. Merge with `sx-defer` (in review on `origin/sx-defer`)

Expected conflicts:
- **Driver shape.** sx-defer splits `forward_step_arena_ready_first` into a wrapper and `_body` (the
  wrapper drops stale `sx_pending` on every exit). S1 needs an exit guard too (poison). Merge them into
  ONE wrapper: drop the SX entries, then poison the flags.
- **`chain!` / `post!` macros and the `Ph::Post` arm.** Both branches edit them. sx-defer's trigger (a),
  "flush the other lanes' deferred shared expert right after a chain", has a semantic conflict with S1:
  after an EARLY chain, the flushed SX would land behind an unreleased flag and wait for this lane's
  reply. Rule for the merge: under S1, trigger (a) runs **before** the early batch's `wait_value` (flush
  the other lanes first, then enqueue the batch).
- **Trigger (b)** ("before own Post") under S1 fires at `rest(l)`, earlier than sx-defer intends. It is
  harmless, but it reduces SX's benefit on early lane-layers. The A/B should score SX with S1 on and off.
- **Trigger (d)** (`self.dgpu.compute.query()` drained) is never true while a flag is pending. That is
  fine: there is no conflict, it just fires less.
- **`fusion_counter_deltas`** becomes `[u64; 10]` in sx-defer. S1 uses its own `early_counter_deltas()`;
  only the `ms.step` `tracing::info!` line conflicts, textually.
- `knobs.rs`, `batch_scratch.rs` (new `BatchDgpuScratch` fields), `multistream_step.rs` (new arms at the
  end) and `DECODE_OPT_KNOBS.md` would conflict only textually (adjacent inserts).

## 11. S1 build plan (files)

| file | change |
|---|---|
| `crates/v4flash-hip/src/{sys,device,stream,buffer}.rs` | `hipDeviceAttributeCanUseStreamWaitValue` (10013); `Device::can_use_stream_wait_value()`; the deny-while-pending check on blocking calls (5.2); optional `hipExtMallocWithFlags` |
| `crates/v4flash-kernels/src/knobs.rs`, `crates/deepstrix-server/src/knobs.rs`, `docs/v41/DECODE_OPT_KNOBS.md` | `V41_MS_EARLY_ENQUEUE` (static, off) |
| `crates/v4flash-kernels/src/het/engine.rs` | `ee_flags` (coherent pinned, `MAX_LANES`), allocated and attribute-checked only with the knob on |
| `crates/v4flash-kernels/src/het/batch_scratch.rs` | `BatchDgpuScratch::ee_pending: Option<EePending { layer, ticket_seq, chain_entered }>` |
| `forward_prefill.rs` | split `forward_layer_post_moe_v2` into `post_enqueue` and `consume_reply`/`stage_partial` (the knob-off path calls them in today's order); `fill_prior` extracted; `Ph::Release`; eligibility; the exit guard; counters; `early_counter_deltas()` |
| `crates/v4flash-kernels/src/het/forward_prefill.rs` (`LazyEngramRows`) | `table_ready(i)` (`is_finished`, non-blocking) |
| `crates/deepstrix-server/src/multistream.rs` | `ee_*` fields on `ms.step` |
| `tests/multistream_step.rs`, `tests/early_enqueue_flag.rs` | section 7 and gate 1 |

Step 0 (optional, accounting-only, safe on the production path): count today's out-of-route-order replies
in the ready-first loop. That is a lane's Post becoming ready while an earlier-routed lane still waits,
together with that lane's remaining wait. This prices S1's HOL exposure on live traffic before S1 is
built.
