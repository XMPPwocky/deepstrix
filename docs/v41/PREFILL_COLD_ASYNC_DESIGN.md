# Prefill cold reads on box 1, off the hub thread (design, stage 2, 2026-10-12)

Status: DRAFT rev 1. Owner 10-12: "ttft fix" -> "let's write up stage 2". Hub only; box 2 untouched.

## 0. Why (measured 10-12, timing metadata only)

- A turn's TTFT is dominated by its FIRST prefill window: 82% of the hub's wait on box 2 over 34,560 prefill requests
  (10:12-20:57 UTC) was in each job's first window; short turns (<= 1,024 rows) spend 3.9 of 4.6 s there blocked on box 2
  reading cold experts (decode evicts ~87% of the next turn's prefill set between turns; owner: no partitioned pool).
- Box 1 can read as fast as box 2: the E100 (box 1's model drive since 20:52 UTC, no dm-crypt) does 4.42 GB/s with 8 x
  6 MiB O_DIRECT readers; box 2's measured prefill read rate is ~4.4 GB/s over two drives. Box 1's drive sits idle
  during first windows.
- Stage 0 (`V41_PREFILL_COLD_B1`, hub 8f75266a) and stage 1 (`V41_PREFILL_PIN_INFLIGHT`, hub 17e2de01) moved part of the
  cold reads to box 1 INSIDE `ensure`: the hub's box-2 wait fell from 82% to 6-13%, but first windows got SLOWER
  (4-unit p50 6.82 -> 9.87 s with pins, 13.78 s through dm-crypt). `ensure` reads synchronously on the hub thread
  (~70 ms for 16 experts); while it reads, the hub issues nothing (the other lane's attention, its next route, box 2's
  next request), so both the dGPU and box 2 starve. Box 2's reads overlap the hub's work; box 1's never do.
- Today: free-only (`FRAC 0`: cold picks box 1 already holds are computed there, ~19% of cold picks) + pins on.

## 1. Idea: issue early, join late

A cold pick given to box 1 is NOT part of the unit's immediate iGPU MoE. At route time (`pre_moe_route`) its read is
queued to background readers; the unit proceeds exactly as if box 2 had it (a third "leg"). In the unit's `Post`, after
`moe_arrived` and before the remote combine, the hub joins the reads, lands them, runs a SMALL iGPU MoE pass over just
those experts' picks, and adds the result onto `ffn_moe_recv` -- the same place and the same kind of f32 add as box 2's
partial. The reads have one lane-layer period (Pre(t) .. Post(t), the other lane's Pre in between) to complete.

## 2. Pieces

### 2.1 Route (`pre_moe_route`, `cold_route` block)
- Today's set-aside `cold_pend` keeps its rule; of it, `cold_b1_take(n, FRAC, MAX)` picks become LATE instead of
  `ids.push`. Late ids go into a per-lane `late: Vec<u32>` on the carry; they are in neither `ids` nor `extra_remote`.
- Exclusion: the main box-1 pass must skip them (they are not resident). The exclusion remap is built from
  `owns_eff = extra_remote` today; it becomes `owns_eff | late` for the remap and `verify_routing_exactly_once`, while the
  box-2 request mask stays `extra_remote` only (box 2 must not compute them).
- Reads: `pg.read_async(layer, &late) -> ticket` (2.3). Queued before the box-2 submit, so both disks start together.

### 2.2 Late pass (`forward_layer_post_moe_v2`, after `de.compute.wait_event(moe_arrived)`, before the remote add)
1. `pg.join(ticket)`: wait for the reads (normally done), claim slots (2.4), repack from the staging buffers on the
   repack stream (`repack_in_place`, as `ensure`), write the late ids' remap entries.
2. Remap upload: `sync_remap_async(layer, ie.compute)` (stream-ordered, staging ring; same mechanism `ensure` uses).
3. Late MoE on `ie.compute`: the batched gate/up + q8k + down kernels over the lane's rows with a MASKED selection
   (`late_sel`: the unit's picks with every non-late id set to -1, its weights kept) and the lane's OWN xq copy (2.5),
   writing a per-lane `bi.late_ffn` [rows x N_EMBD] f32 (zeroed rows without late picks).
4. Peer-push `bi.late_ffn` -> `bd.late_ffn_recv` on `ie.xfer` after a `late_done` event; `de.compute` waits
   `late_arrived` and `vec_add`s it onto `ffn_moe_recv` BEFORE the remote add (fixed order: local, late, remote).
5. `pg.pin_inflight(layer, late, ie.compute)` (stage 1) so the other lane cannot evict them under the late kernel.
No late picks -> nothing of this runs (no launch, no event).

### 2.3 Background reads (`ExpertPager::read_async`)
- A reader pool (threads = `V41_PREFILL_COLD_READERS`, default 8 = the E100's measured optimum) reading each expert's
  three tensors O_DIRECT into a PINNED staging slot (the same layout `ensure`'s miss path repacks from). Ticket = the
  list of (id, staging slot); `join` blocks on the ticket's completion.
- Staging pool: `V41_PREFILL_COLD_STAGES` experts (default 32 x ~18.8 MB = ~600 MB pinned host; two lanes x MAX 16).
  No free stage -> the pick is not taken late (stays box 2's): reads never wait for staging.
- Errors: a failed read -> the join recomputes that id via a synchronous `ensure` on the spot (correct, slow, counted).

### 2.4 Where late experts land (box 1's pool is one LRU with no owned-set protection)
- Victim filter for late claims: skip slots holding a hot-set OWNED id (`hot_set::box1_owns == Some(true)`) and pinned
  slots; prefer the LRU among the rest. Decode's owned set (115/layer) is never evicted by a prefill late read; the spare
  slots (~825 = 5,425 - 4,600) take them, oldest first (the previous turn's cold reads).
- Budget per layer per window: `V41_PREFILL_COLD_LATE_MAX_LAYER` (default 20 ~ the spare per layer); the route stops
  taking late ids for a layer once the window has spent it.

### 2.5 The lane's xq copy
`si.d_xq_q8k` is SHARED scratch the other lane's launch overwrites before this lane's Post. When a unit has late picks,
`pre_moe_launch` copies its xq rows (D2D on `ie.compute`, rows x 5,840 B = ~3 MB at 512 rows) into a per-lane
`bi.late_xq` right after the quantize. Unit without late picks: no copy.

## 3. Fidelity
The late experts' contributions are the same f32 products added in a different order (a separate pass + one more
`vec_add`), the class of change every split already makes (live null ~0.3 nats from partition movement). Gate:
`lm_prefill_bitexact`-style equality is NOT expected; KLD vs the off arm on fixed prompts (golden recipe) within the split
null, plus `verify_routing_exactly_once` (fails closed) with the late class.

## 4. Knobs (all live unless noted)
| knob | default | meaning |
|---|---|---|
| `V41_PREFILL_COLD_ASYNC` | off | late pass instead of `ensure` for the FRAC/MAX share (needs `V41_PREFILL_COLD_B1` and `V41_PREFILL_PIN_INFLIGHT`) |
| `V41_PREFILL_COLD_B1_FRAC` / `_MAX` | 0.5 / 16 when async | share / cap per request |
| `V41_PREFILL_COLD_LATE_MAX_LAYER` | 20 | late reads per layer per window |
| `V41_PREFILL_COLD_READERS` (static) | 8 | reader threads |
| `V41_PREFILL_COLD_STAGES` (static) | 32 | pinned staging experts |

## 5. Expected effect (inferred)
Per first-window request ~50 cold experts: box 2 reads them today at ~4.4 GB/s (~210 ms of disk per layer's requests,
mostly exposed). Late reads of half on box 1 in parallel cut box 2's reads ~2x and add ~0 hub-thread time (a join that
normally finds the reads done, a ~1-2 ms late pass on few experts, one push). Estimate: first-window blocked time
3.6 -> ~1.8-2.2 s; short turns 4.3 -> ~2.7-3.2 s TTFT. Bounded by: box 1's iGPU doing more MoE (late experts carry few
picks: cheap), the extra `vec_add` + push (~0.3 ms per lane-layer with late picks).

## 6. Measurement
`first_windows.py` (job tmp; per job first-window wall + hub box-2 wait by units) against the 10-12 baseline (2-unit
4.27 s, 4-unit 6.82 s) in alternating on/off blocks of real traffic; counters `cold_late` / `cold_late_join_wait_ms` /
`cold_late_fallback` on the "stream admitted" line; decode box-1 misses after a prefill (must not rise: owned set is
protected by 2.4).

## 7. Risks
| risk | mitigation |
|---|---|
| a late expert evicted under its late kernel | pinned (stage 1) right after its launch; claims skip pinned and owned |
| xq overwritten before the late pass | per-lane copy (2.5) |
| reads not done at join (hub waits) | counted (`join_wait_ms`); FRAC/MAX live; worst case = today's synchronous read |
| exactly-once violated | late mask in the exclusion and the verify; fails closed |
| pinned host RAM (~600 MB) | box 1 RAM is shared with the iGPU: static knob, sized once; 0 disables |

## 8. Slices
1. `read_async` / `join` + staging pool + host test (tickets, failure fallback, no-stage refusal).
2. Route late class + exclusion/verify mask + carry plumbing (late empty -> today's path exactly).
3. xq copy + late pass + combine; host test of the masked-selection builder.
4. GPU check on the dev path (server down or a scratch hub): one prompt, late on vs off, KLD within the split null.
5. Deploy default off; flip live; on/off blocks.
