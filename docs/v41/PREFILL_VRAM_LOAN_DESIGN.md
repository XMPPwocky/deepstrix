# Prefill dGPU VRAM loan: 1024-row prefill lanes from decode-idle memory (design rev 2, 2026-10-10)

Status: DESIGN, nothing built. Owner 10-10: "we need to build the dgpu vram loan system" + "don't even fall back -- instead,
swap out decode KV basically". Code map: read-only scoping 10-10 (paths under `crates/`, HEAD dfc868aa).

## 1. Why

* Under layer-outer + split the prefill wall is the HUB iGPU MoE (22.5 ms per 512-row unit-layer; Tier B 10-10) with box 2
  close behind (17-21 ms); BOTH are weight-read bound (~100-125 us per distinct expert, flat in rows). Doubling rows per
  MoE call halves expert reads on both boxes: est. window 4.0 s -> ~2.0-2.3 s (~1.8-2K rows/s). `V41_PREFILL_LANE_ROWS=1024`
  exists (e918f2a4) but does not fit the dGPU: 15.8 of 16.3 GB used at rest.
* What 1024-row lanes cost (ALLOC_TRACE, 512-row values x2): lanes `bd_a`/`bd_b` 2 x 288 MB -> 2 x 576; shared `sd` ~1.7 GB ->
  ~3.3 GB (`r1_arena` 1512 MB [indexer_scores], `attn_active_comp_kv` 537, `attn_scores` 403, `r3_arena` 220, `q_normed`
  134, ...); raw KV `kv_cache_rows()` +42 MB per `HetModelState` (2 states). A full 1024-row prefill set is ~4.4 GB.
* Decode uses the SAME `bd_a`/`bd_b`/`sd` objects but at most 64 rows (`multistream.rs:281`): ~2 GB of the resident set is
  only ever touched by prefill.
* The KvArena (decode compressed KV) is 3.59 GB: 4 stores (KV-source layers 2/8/14 ratio 2 = 650K rows, 20 ratio 1 = 1.3M
  rows), each `comp_kv` (1024 B/row: 665.6 MB x3 + 1331.2 MB) + `index_k` (80 B/row: 52 MB x3 + 104 MB)
  (`kv_arena.rs:485-541`). Live decode KV is typically ~0.8 GB of it.

## 2. Design

### 2.1 Resident decode set sized for decode
Allocate the resident `bd_a`/`bd_b`/`sd` (and `bd_c`) for `DECODE_ROWS_MAX` = 64 rows (the arena driver's cap) instead of the
prefill lane size. Frees ~2.0 GB permanently. Graph capture is unaffected: decode graphs bake these addresses (legacy keys)
or register `sd` statically (stage_b) -- both stable from startup.
CHECK: every non-prefill user of these objects at > 64 rows (CED replay of 128 rows at `prefill_job_finish`? DSpark
verify? vision? embed?) -- those move to the prefill set (2.3) or keep a size floor.

### 2.2 Prefill pool (permanent)
One `hipMalloc` at startup of the freed size (~2.0 GB + the ~0.5 GB headroom left = ~2.4 GB, sized from live free VRAM
minus a margin). Holds the buffers that must persist across passes or exceed any arena store: `r1_arena` (1.51 GB at 1024
rows), `lm_store`/`lm_idx` (~344 MB at 4096-row windows; today hipMalloc'd per window with halve-on-OOM), later the
half-batching stash (~170 MB).

### 2.3 Whole-arena loan per PREFILL PHASE (owner: no fallback, swap decode KV out)
* `ms.phase` Decode -> Prefill (`multistream.rs:1112-1146`): sync the dGPU; copy every LIVE region of every store
  (`CompRegion{base, n_comp}` per stream per store, comp_kv + index_k rows `[base, base+n_comp)`) D2H into a host staging
  area (pageable, sized to the live rows; ~0.8 GB typical -> ~0.15-0.3 s); mark the arena LENT.
* While LENT the 8 store buffers (4 comp_kv + 4 index_k) are donors: `LoanAlloc::over(views)` (`dgpu_loan.rs:439-455`)
  hands out views in a FIXED order to build the prefill set's remaining buffers: `bd_a'`/`bd_b'` at 1024 rows (~575 MB
  each), the rest of `sd'` (~1.8 GB without r1), the prefill job states' raw-KV growth if needed. Largest single buffer
  after `r1_arena` = `attn_active_comp_kv` 537 MB <= 665.6 MB store: fits first-fit.
* `ms.phase` Prefill -> Decode: sync; H2D the staged live rows back to the SAME offsets; mark RETURNED. Decode graphs see
  unchanged addresses and contents.
* Prefill passes call `prefill_job_chunk` / `_lm_pass` / `_finish` with the prefill set (`bd_a'`, `bi_a`, `bd_b'`, `bi_b`,
  `sd'`, `si`) instead of the resident set. The prefill set is NEVER `register_static`'d (no graph references it); lane
  state set at startup (`mtp_capture_rows`, `multistream.rs:347-349`) is copied onto it. The DSpark ring seed after
  `prefill_job_finish` reads the prefill set's captures (`multistream.rs:1441`) -- before the loan returns.
* iGPU side (`bi_a`/`bi_b`/`si`): allocate at 1024 rows permanently (GTT = system RAM; `si.q2k_partials` 122 KB/row ->
  +63 MB etc.; box 1 has ~8 GB free).

### 2.4 Scheduler rules while LENT (every arena-mutating site, map 10-10)
| site | while LENT |
|---|---|
| `try_admit` -> `grow` / `fill_reserved` (`:1469`, `:1487`, `:1499`) | DEFER: the prefilled job parks as "prefilled, awaiting admission"; admitted at the next Prefill->Decode switch AFTER the swap-in (it could not decode before that anyway) |
| `compact_stores` + `reserve` at admission (`:1026-1029`) | `reserve` is bookkeeping (free list) -> allowed; `compact_stores` moves data -> DEFER (or reserve without compaction; a job needing compaction waits for the phase switch) |
| `release` (cancel / failure / finish) | bookkeeping only -> allowed; the staged rows of a released stream are dropped at swap-in |
| decode steps (`grow`, `compact_for_step`, kernel writes, `accept`/`advance`) | impossible: no decode while LENT (phase) |
| `abort_all` -> `rebuild_free_lists` | allowed (bookkeeping); swap-in skips aborted streams |
| other threads | none touch arena memory (map Q1) |
Invariant (debug-checked): no device write into arena memory except by prefill-set kernels while LENT; a guard object ends
the loan (swap-in) on every exit path; a panic with the loan out aborts (as `LoanOut` does).

### 2.5 Knobs
`V41_PREFILL_LANE_ROWS` (static, existing) = the prefill set's rows; `V41_PREFILL_VRAM_LOAN=1` (static: changes allocation
at startup) -> 2.1 + 2.2 + 2.3. Off = today exactly.

## 3. Costs / risks
* Swap per Decode<->Prefill switch: live KV ~0.8 GB typical (D2H + H2D ~0.3-0.6 s per cycle of ~150 s); worst case the full
  3.59 GB (~1.2-2.4 s per cycle) with many long live streams. Optimisation later: swap only stores/ranges the prefill set's
  placement actually overlaps; keep a pinned staging buffer if RAM allows.
* Admission latency: a request whose prefill finishes mid-phase waits for the phase switch to start decoding (today it is
  admitted at once but still cannot decode until the decode phase -> TTFT unchanged; check the "prefill burst ends early
  when the queue is empty" path).
* Host RAM: staging up to the live KV (~0.8 GB typical, 3.6 GB worst) on a box with ~8 GB free.
* Fidelity: none from the loan itself (views, same kernels); 1024-row lanes change chunking -> Gate A at 1024 rows
  (chunked vs layer-outer at the same lane size) + a live A/B.

## 4. Open questions for review
1. 2.1: every > 64-row user of the resident set (CED replay 128 rows at finish -- prefill-side? DSpark verify rows; vision;
   embed loan donors = immutable WEIGHTS, unaffected?). Is 64 the right floor?
2. 2.3: is `HetModelState` (the prefill job's scratch state, 1017 MB) touched by anything that assumes resident lanes?
3. 2.4: the exact deferral points in `multistream.rs`; the "prefill burst ends early" path; parked `try_admit` retries.
4. Can the swap be avoided when live KV fits outside the placement (free runs after compaction cover the prefill set's
   arena part)? Worth it in rev 1, or later?
5. Host staging: pageable (simple, ~3 GB/s) vs pinned (RAM).


## 5. REV 2 (Fable round 1 = REVISE; findings and dispositions)

### 5.1 Loan lifetime: prefill ACTIVITY, not `ms.phase` (finding 1, BLOCKER: deadlock with no live streams)
Rev 1 deferred admission to the next Prefill->Decode switch; with no live streams the scheduler returns early
(`ms:1085-1088`) before the phase logic, the phase never leaves Prefill, the loan never returns, and the parked retry
(`ms:883-891`) defers again forever. REV 2 has NO deferral table:
* **Lend** right before a prefill unit of a LONG job (`remote_long_job()`; >= 2 windows) runs, if no loan is out.
* **Return** (swap-in if anything was swapped; device-wide sync first) at the TOP of every arena-mutating call --
  `try_admit` (-> `grow` / `fill_reserved`), `compact_stores`, `grow` from a decode step -- and before any decode step.
  A guard object (`ArenaLoan`) is the only way to obtain arena write access while lent; its `end()` is idempotent; a
  panic with the loan out aborts (as `LoanOut`).
* Back-to-back long prefills keep the loan out (no arena op between their units); a short request's admission ends it
  (one swap-in), the next long unit re-lends. Per-request cost: at most one return + one re-lend.

### 5.2 Compaction-first, swap only the overlap (alternative (e), owner's "swap decode KV out" made conditional)
* Placement is FIXED at startup: the prefill set's arena part sits at the END of each store buffer (`comp_kv` x4,
  `index_k` x4); the views (`DeviceBuffer::from_raw_parts`, `is_view`) and the prefill `BatchDgpuScratch` x2 +
  `BatchDgpuShared` are built ONCE and never `register_static`'d.
* **Lend** = `compact_stores` (allowed: the loan is not out yet) -> every store's live rows are `[0, next_base)`; for each
  store, the live rows overlapping its reserved end range `[cap - lent_rows, cap)` are D2H'd to host staging
  (`(slot, store, base, n)` recorded) -- typically NONE (free tail 2.5-2.8 GB, 10-09 census).
* While lent, the free lists must not hand out rows in the lent end ranges for data: `reserve` (bookkeeping) is allowed
  anywhere, but any DATA write (fill/grow/compaction) ends the loan first (5.1), so no overlap can be written while lent.
* **Return** = H2D the recorded rows still live with the SAME base (finding 6: a released slot's rows may be re-reserved).

### 5.3 Memory (findings 2, 3)
* Pool = the resident shrink ONLY (~2.0 GB), keeping the ~0.5 GB free that lazily instantiated graphs need.
* Itemised placement computed at startup from the buffer size lists (`BatchDgpuScratch` / `BatchDgpuShared` size
  functions at `V41_PREFILL_LANE_ROWS`), first-fit-decreasing into: pool (r1_arena 1512, lm_store/idx 344, then fill),
  then the 8 store end ranges, with a `Carver` per donor (`bs:1623-1636`; `LoanAlloc::over` hands out ONE buffer per
  donor). If it does not fit, the loan is DISABLED with a loud startup error (no silent fallback mid-run). The script
  version of the placement is checked before building.

### 5.4 Smaller (findings 4-7)
* `DECODE_ROWS_MAX` = the arena driver's `rows_cap` (`n_slots x ARENA_ROWS_PER_STREAM`), not a constant.
* CED replay at `prefill_job_finish`, the DSpark ring seed (`ms:1431`, `1441`) and `mtp_capture_rows` (`ms:347-349`) use
  the PREFILL set.
* Pre-swap sync is device-wide (`de.compute`, `de.xfer`, `de.hc`, `rb_stream`); raw `kv_cache` and the per-slot
  `state_kv`/`state_score` blocks (`ka:494-509`) are not donors (invariant: no prefill view overlaps them).
* Staging: pageable first (RAM-safe); pinned later if the swap shows up.

### 5.5 Staging
0. **Probe (no new code)**: `V41_PREFILL_LANE_ROWS=768` fits the arena donors alone? No -- it needs the prefill set
   SEPARATE from the resident set too; the probe is only possible as "resident lanes at 768" if 768-row resident lanes
   fit the card (+~1.1 GB): they do not (0.5 GB free). Skip; the loan is the enabler.
1. Resident shrink + pool (static knob), prefill set built over the pool + fixed store end ranges, loan guard with
   compaction-first and overlap swap; unit tests for the placement and the swap bookkeeping (CPU), a GPU test of
   lend/return round-trip bit-identity of live KV.
2. Gate A at 1024-row lanes (chunked vs layer-outer at the same lane size) + a live A/B.

## 6. Review round 2: APPROVE (2026-10-10)
Fable r2 verified the 5.1 chokepoints (`ms:1026` compact_stores, `ms:1487`/`1591` grow, `ms:1499` fill_reserved,
`ms:1153` decode_step entry; bookkeeping-only: reserve/release/rebuild_free_lists), compaction-at-lend safety (row tables
rebuilt per step, stage_b bakes only store bases) and the itemised 1024-row set: lane 573 MB x2, `sd'` 3,233 MB, total
4,379 MB; pool 2,048 (r1 1,513 + lm_store/idx 344 + candidate_block_score 189); typical post-compaction tails 2,788 MB
(live KV 0.8 GB) vs 2,772 needed outside the pool -> **16 MB slack** with the dead `verify_scores` excluded. Placement
script: job tmp `review/placement.py` (copy into the repo with the build). Minors folded into the build:
compact only if live rows overlap the lent ranges (`overlaps_lent()`); exclude `verify_scores` from the prefill set;
token/flag-enforced arena writes. Overlap swap is a REGULAR path (live KV > ~0.8 GB); largest buffers in comp_kv1's end
range; reclaiming the dead `verify_scores` (94 MB) + `DgpuScratch.attn_scores` (190 MB) would give ~300 MB slack.
