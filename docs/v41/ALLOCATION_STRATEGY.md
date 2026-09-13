# Expert pool allocation strategy for the agentic workload (draft 1, 2026-09-13)

## The workload (from the user, not inferred)

**Mode A — agentic steady state (the common case).** One medium prefill (system
prompt), then: decode → short prefill → decode → short prefill → … Needs fast
*turnaround*. Latency-bound, and **decode dominates the wall clock**: at today's
~1.4 tok/s a 200-token reply is ~143 s while a 500-token prefill is ~8 s. Decode
is >90% of an agentic turn.

**Mode B — occasional very long prefill.** 100s of Ktok up to max context, after
memory edits change the system prompt, during context compression, or on a cache
miss. Throughput-bound, decode irrelevant while it runs.

These want *opposite* allocations, and the mode is knowable at request time
(prompt length / cache-miss), which is the key lever.

## Measured facts this rests on

* decode per token: `pager_ensure` **92.7%**, of which raw SSD read 82.4%;
  `dgpu_busy` only 7.5% (80 ms of a 1062 ms token). 11.6 ms per miss, 85
  misses/token.
* prefill is window-sensitive: 8→6 windows cost 70.8→61.1 tok/s.
* every window is ~7.2 GB that the decode LRU does not get.
* box 1 pool can be ~76-80 GB (measured OK; "used" in `free` is mostly the TTM
  page pool, reclaimable — GPUs showed gtt_used=14 MB when idle).
* two-box split validated: prefill 93 tok/s, decode 54.7→23.2 s wall, output
  identical.
* residency curve (this morning): misses/token 2186 slots→67, 3072→47, 4096→33,
  6144→15.

## Proposal

### 1. Default to decode. Windows ~1-2, LRU gets everything else.
Mode A is decode-dominated, and short prefills do not amortise a pinned window —
a window pays off only when the same layer is revisited across many chunks, which
is a LONG-prefill property. Today's 8-window default optimises the rare mode at
the expense of the common one. At 76 GB with 2 windows the LRU gets ~3400 slots
(~64 GB) vs 468 slots (8.8 GB) at the 8-window/62 GB setting we have been
benchmarking.

### 2. Mode B should be layer-major, not window-pinned.
Prefill is chunk-major today, so every layer re-pages for every chunk: 5 chunks ×
20 layers × 7.2 GB = 866 GB for a 4768-token prompt, and ~14 TB at 100K. Process
ALL chunks through layer L before L+1 and each layer loads **once**: ~144 GB
dense, ~76 GB with union paging, **independent of prompt length**. That is the
user's "stream each layer's experts in and crank the batch size" instinct, and it
is correct.
Correctness holds: when layer L runs chunk N, chunks 0..N-1 already passed layer
L, so their KV is present. Extra state is all chunks' hidden states at one layer
boundary — 1 GB at 100K, trivial.
Layer-major also makes windows **irrelevant**: you want one streaming buffer, not
a pinned set. So Mode B does not compete with the decode LRU for pool — it can
stream through a small dedicated arena and leave the LRU alone.

### 3. Box 2's assignment should be chosen for DECODE, i.e. uniform `all:<lo>-383`.
Earlier instinct was a hybrid weighted to encoder layers (better prefill). Given
Mode A dominates, that is wrong: decode traverses all 40 layers and cannot be
batched or amortised, so box 2's RAM is worth most there. Prefill's loss
(93→70.8 tok/s under `all:250-383`) is acceptable because Mode B should go
layer-major, where the per-layer load is paid once and the assignment matters far
less.

### 4. Do not reconfigure the pool per request.
Windows/slots are fixed at startup and making them dynamic means reallocating
device memory mid-flight. Layer-major removes the need: one allocation serves
both modes.

## What this predicts
* Mode A: decode LRU ~3400 slots → ~40-45 misses/token (curve) vs 85 today →
  roughly 2x decode, on top of the 2.4x already measured. Agentic turn
  ~143 s → ~60 s for 200 tokens.
* Mode B: 100K prefill read cost ~76 GB instead of ~14 TB.
* Neither needs the other's config.

## Open questions for review
1. Is "decode dominates the agentic turn" robust, or do short prefills at 1-5 Ktok
   add up enough to matter?
2. Layer-major needs the two-lane pipelined driver restructured — is that worth it
   versus simply raising B_MAX so fewer chunks exist? (B_MAX 1024→4096 cuts chunks
   4x and needs no reordering, but attention scratch is O(B) and at 100K the
   scores buffer is already 3157 MiB.)
3. Should box 2 hold a *different* set for the two modes (it can be reassigned in
   ~33 s), or is one assignment right for both?
4. Is there a case for the dGPU's 16 GB as a third tier for the hottest experts,
   given decode is 93% paging and the dGPU is 92% idle during it?
