# Box 2: evict behind the pass in a job's LAST prefill unit (design, rev 2, 2026-10-11)

rev 2 = review r1 (REVISE): "last unit" decided inside the engine (2.1), the replay gain scoped to single-unit jobs
(3), phase splits (2.4), flags2 OR + daemon gate (2.1/2.2), Q1 answered by measurement (4).

Owner 10-11: "design this and loop". Not a partition (owner: "no partitioned pool"): no slot is reserved for
anyone; this only changes WHICH eligible page a prefill claim evicts, and only for requests of a job's last unit.

## 1. Problem (measured 10-11, box-2 evtrace + hub logs)

* **Replay is box-2 paging bound.** `prefill_job_replay` (128 rows x decoder layers 20..42, two 64-row lanes):
  3.2 s p50 / 4.8 s p90 over 325 jobs; ~4.9 s p50 job finish incl. admission. One traced replay (10:09:59 UTC,
  job 2590): every decoder layer's lane-0 request missed 17-55 experts (page 83-187 ms), ~760 misses (~14 GB)
  over layers 20-39, ~140 ms/layer vs ~20-40 ms of compute.
* **Cause: the prefill phase wipes decode's set.** Same job's phase (one window, then the replay): it loaded
  2,543 distinct encoder-layer experts (~120-160/layer, ~48 GB) and evicted 2,867 decoder-layer pages + 609
  pre-phase encoder (decode) pages, + 903 of its own pages loaded earlier in the phase; interleaved decode of
  other streams re-read 1,493 decoder-layer experts during the phase. Mode-evict (`V41_B2_MODE_EVICT=1`) takes
  decode-class victims first while the phase's budget lasts: `V41_B2_PREFILL_BUDGET` = 3500, and a LONG job's
  budget (`V41_B2_PREFILL_BUDGET_LONG`) is UNLIMITED (since 10-11 05:25 UTC every job > 4096 tokens is long).
* **What the phase actually needs.** Layer-outer runs one layer over all of a window's rows, then the next. In a
  job's LAST window (and in its single chunk, for a chunked-only job), layer L's experts are dead for this job once
  the pass leaves L: nothing before the replay reads an encoder layer again, and the replay reads none. So the
  last unit needs ~1-2 layers of slots at a time (~150-300 experts) recycled behind the pass, not decode's ~3.5K.
* **What it costs today:** the replay's ~0.8-2.5 s of paging, the first decode steps after admission (cold
  decoder layers), interleaved decode during the phase. The next turn's first window loses nothing: decode already
  evicts the encoder pages between turns (87% of a first window's reads, measured 10-11 04:38 UTC).
* Prior idea checked and dropped: restoring decoder layers by a ranking (displaced set: 55% max coverage at 14%
  precision, 44 GB; decode-count top-K = random: box 2 holds the flat tail by the hot split's design).

## 2. Mechanism

### 2.1 Hub: `REQ2_FLAG_LAST` (second flags word, bit 1; an older daemon ignores it)

Set on every PREFILL request (never with `REQ_FLAG_DECODE`) of a job's LAST unit:
* layer-major (layer-outer or group-major): every request of the window whose end is the job's end (`win_end ==
  tokens.len()`), all its passes incl. the KV-only pass;
* chunked-only / chunked tail: the chunk whose end is the job's end;
* the replay (`prefill_job_finish`).
Plumbing mirrors `REQ_FLAG_LONG_JOB` (`RemoteClient::last_unit` + `set_last_unit`, `HetEngine::remote_set_last_unit`,
carried across a redial like `decode_phase` / `long_job`), but the decision is made INSIDE THE ENGINE where the unit's
extent is exact -- NOT by the scheduler's prediction (review r1 #1: with a layer-outer window open and `lo_layer > 0`,
`next_chunk_range` returns an EMPTY range, so passes 2..k of the last window would be unflagged; and the prediction can
exceed a window `lm_open_window` halved on OOM, flagging a non-last window):
* layer-major: in `prefill_job_chunk` right after `lm_open_window`, `last = job.lm.as_ref().is_some_and(|w| w.end ==
  job.tokens.len())` -- evaluated for EVERY pass (incl. the KV-only pass) of the open window;
* chunked: the chunk whose `chunk_end == tokens.len()` (`is_last_chunk`);
* `prefill_job_finish` (the replay): unconditionally last.
Set before the unit's first request, cleared when the unit returns (every path) AND by
`remote_set_phase_busy_poll(true)` (the decode drivers' phase switch), so a decode request can never carry it.
On the wire: `flags2 |= REQ2_FLAG_LAST` (OR: `patch_request_flags2` writes the whole word; SOFT is decode-only and LAST
prefill-only, so they never coincide today, but the OR keeps it correct by construction).

### 2.2 Box 2: per-request "last" mode in the tiered search

`note_request_phase` also records `req_last = flags & REQ_FLAG_DECODE == 0 && flags2 & REQ2_FLAG_LAST != 0` (gated on
not-decode, as the SOFT reads gate on decode). Per request, like `req_prefill`: saved/restored around a frame served
inside a park (re:8790-8792) and fed to the parked pass's mode (`park_prefill`, re:7613); landings carried by a LAST
request rank with it (the landing's carrier is known where `prefill_landing` is computed). In a prefill-mode search (`me_tiered`) of a LAST request:
1. **Decode tiers.** A separate per-phase counter `last_budget_left` (reset at each phase start to
   `V41_B2_PREFILL_BUDGET_LAST`, live, file key `prefill_budget_last`; default `MAX` = OFF at ship): a decode-class
   candidate is tier 2 only while BOTH `budget_left > 0` and `last_budget_left > 0`, else tier 4. Taking a tier-2
   victim in a last request decrements both. So with `prefill_budget_last = B`, a phase's last unit takes at most B
   decode victims (beyond free / stale), then recycles its own pages (tier 3).
2. **Own pages (tier 3), evict-behind order** (`last_sweep_rank(owner, rank_layer)`, smaller first):
   * `(0, rank_layer - l)` same region, `l < rank_layer`: a layer this pass has PASSED -- dead for this job; the
     farthest-passed first (oldest);
   * `(1, 0)` the other region (encoder pages during the replay; decoder pages during the encoder sweep, i.e. a
     previous job's replay pages);
   * `(2, u32::MAX - l)` a layer still ahead (`l >= rank_layer`), highest first, the next one last.
   Replaces `own_sweep_rank` (long) and plain LRU (normal) for LAST requests only; non-last requests unchanged.
3. Everything else unchanged: free (0) and stale prefill (1) first; touched decode (5) last; pins, `want`,
   park pins, floors, nursery are excluded exactly as today (the search only re-orders ELIGIBLE candidates, so the
   reserve argument in the pinning block is untouched); prefetch landings rank at the pass's current layer
   (landing-rank fix) and follow the same rules when their carrier request is LAST.

Decode-mode searches (decode requests, incl. those interleaved during the phase) are unchanged.

### 2.4 Phase splits (review r1 #3)

Any interleaved decode STEP ends box 2's prefill phase (>= 16 decode-flagged requests = `ME_DECODE_STREAK`); the next
prefill request starts a fresh phase (fresh budgets, `long` re-raised by the next long request). So: (a)
`last_budget_left` is per phase: a last window may take up to B decode victims per pass that follows a decode step;
(b) in the new phase every page the window loaded earlier is tier 1 (stale) and goes FIRST -- for a last window those
are passed layers (wanted) plus any rolling-prefetched L+1/L+2 pages (re-read, small); for non-last windows this is
today's behaviour. Do not assume one phase per window.

### 2.3 Why not just lower `prefill_budget(_long)`

A multi-window job's NON-last windows revisit all 20 encoder layers every window: keeping its ~2.5-4.5K encoder
set across windows is what makes steady windows ~3-3.7 s (re-reading ~48 GB per window would be ~3x slower). A
global low budget would break that. The last window of such a job starts with that large own set resident; the
evict-behind order recycles it and needs no decode victims at all. For a single-window job (most agent turns)
first = last: it takes free + stale + B decode victims, then recycles its own passed layers.

## 3. Expected effect

* SCOPE (review r1 #2): the warm replay is for SINGLE-UNIT jobs only -- a chunked-only job (<= 2048 tokens) or a
  one-window layer-major job, where first unit = last unit. In a multi-window job the non-last windows keep today's
  rule by design (2.3) and the FIRST window already evicts decode's decoder-layer pages while loading its encoder set;
  the last window's restraint cannot bring them back, so that replay still pages (~760). Fixing that needs a
  decode-class floor = a partition, ruled out by the owner: out of scope. Multi-window jobs get only the in-phase
  decode relief of their last window and a cheaper last window. Today's mix (hub logs 10-11 03:45-10:20 UTC): 512
  prefill jobs, 383 chunked-only + 93 of 129 layer-major jobs single-window = **93% single-unit**.
* Single-unit jobs: decoder layers stay resident -> the replay's ~760 misses -> few; 3.2 s -> ~0.8-1.2 s
  (compute-bound); the first decode steps after admission start warm; interleaved decode during the phase stops
  re-reading.
* Last/single windows: fewer own-page re-reads (903 self-evictions in the traced phase came from the plain
  oldest-first order taking pages the window still needed); maybe slightly faster.
* Cost: the NEXT job's first window when it is back-to-back in the same phase and shares experts with this job's
  passed layers (they are recycled). Measured pair 09:59 UTC: different conversations, 2.9 misses/req anyway.

## 4. Measurement / A-B

Live knob on box 2: `prefill_budget_last` alternating `MAX` (off) / `B` (e.g. 256) every 15 min
(`ab_mirror.sh`-style, box-2 knob file). Score per arm: replay `elapsed_s` (hub log), replay misses (hub_req layers
>= 20 in replay spans), decode `step_ms` in the 10 s after each admission and overall, first-window and steady
window times (`ab_mirror_eval.py`), box-2 per-phase eviction counters (`took_decode`, `took_own`, new
`took_last_own`), split by single-unit vs multi-window jobs. B: review r1 #5 -- after a decode burst the pool holds the
previous prefill phase's pages as tier 1 (thousands of slots: the previous job's encoder set + replay pages), so a cold
single window usually needs no decode victim at all; start the A/B at B = 0 vs MAX, read `took_stale` / `took_own` /
`took_decode` per arm, then try 256 if B = 0 shows own-page re-reads inside the window.

## 5. Tests and gates

* Pool unit tests (host): a last request with budget_last = 0 takes free, stale, then own passed layers in
  farthest-passed order, never a decode page; with B > 0 at most B decode victims; non-last requests in the same
  phase unchanged; the replay (decoder layers, last) takes encoder own pages first; pinned / wanted / floors never
  taken (existing invariant tests rerun under last mode).
* `remote_experts_pin_loopback` + `remote_experts_nursery_loopback` (box-1 iGPU, as slice D).
* No numerics change (residency only): no Gate A / KLD needed. Old daemon + new hub and new daemon + old hub
  both work (the bit is ignored / never set).
* Deploy: hub + box 2 (two-box restart; box 2's pool restarts cold); ship with `prefill_budget_last = MAX`.

## 6. Open questions for review

1. (r1: answered by measurement, sec 4.) Group-major windows: `w.end == tokens.len()` holds for every group unit of the
   last window, same rule.
2. Back-to-back jobs in one phase: job A's last unit recycles its encoder pages; job B (same phase) is not last
   for its non-last windows -> normal budget again, but `budget_left` was not consumed by A's own-page recycling.
   Fine?
3. Interleaved decode requests between passes: phase hysteresis (`ME_DECODE_STREAK` 16) -- can a long decode burst
   end the phase mid-window, so the last window's remaining passes start a NEW phase (fresh budgets; stale = the
   window's own earlier pages = tier 1, evicted first)? Probably desirable here.
4. Should the replay's own claims also prefer encoder own pages over FREE slots? (No: free is free.)
