# Expert allocation rework — plan

**Status: PLANNED — v1 REWRITTEN 2026-09-16 after adversarial review.**

> **v1's central proposal ("box 1 = hot cache, box 2 = victim cache, one flat
> LRU, no static placement") is WITHDRAWN.** Three findings killed it, all from
> in-tree evidence that predates the plan:
>
> 1. **`WHY_THE_BIG_POOL_REGRESSED.md:35-60`** — box 1's iGPU costs **140 us per
>    expert per layer** (it shares the device with attention, norms and the head)
>    against box 2's **87 us** (MoE only). Serving a pick box 2 would HIT is
>    **-53 us, a LOSS**; only serving one box 2 would MISS pays (+6547 us). The
>    victim gate is not backwards -- it is box 1's only sign-positive policy. The
>    asymmetry is structural and no allocation policy fixes it.
> 2. **`expert_pager.rs:1188-1193`** — prefill sweeps layers 0..N every chunk,
>    which is **LRU's worst case: a cyclic scan whose working set exceeds
>    capacity hits 0%**. That is why the pinned windows exist. "Replace pinning
>    with a true LRU" would make prefill 0% by construction.
> 3. **KNOWN_BUGS #1 is RESOLVED and was never a pager bug** -- so the plan's
>    premise that geometry corrupts the model is gone. Under `T2_CATCHALL=2`,
>    WINDOWS=21 and WINDOWS=4 produce IDENTICAL output at 6.00 vs **14.21
>    tok/s**. The geometry knob is safe and already worth 2.4x under mode 2.
>
> Also corrected: box 2 is **phase-aware**, not globally flat -- its victim
> search is region-restricted on prefill-shaped requests
> (`remote_experts.rs:1835`), and `enable_paging` deliberately does NOT advertise
> all-true because a B=1024 prefill union (~203/layer) will not fit a catch-all
> region. v1 proposed importing a structure box 2 does not have.

**Status: PLANNED, not implemented.** Written 2026-09-16 from measurements in
this session. Supersedes the current three-structure box-1 pager.

## What we do today, and why it is wrong

Box 1 (96 GB) holds 4,454 slots / 83.7 GB / 29% of the 15,360 experts, cut into
THREE regions fixed at startup:

| region | slots | used by | policy |
|---|---|---|---|
| pinned dense windows w=0..19 | 20 x 128 = 2,560 | PREFILL, layers 0-19 | demand union per (layer, chunk), packed by arrival |
| shared dense window w=20 | 384 | PREFILL, layers 20-39 SHARE it | each layer evicts the previous |
| decode/verify LRU | 1,510 | decode + speculative verify | see below |

Box 2 (128 GB) holds 6,560 slots / 123.3 GB / 42.7%, seeded 260/layer (L0-19)
and 68/layer (L20-39) from a frequency-ranked placement file, then `--paged`
makes every layer `owned` and runs ONE shard-wide global LRU with no per-layer
floor (`V41_B2_POOL_FLOOR=0`). Box 2's allocation is already dynamic and
reaches a 96.9% hit rate.

Three defects, all measured:

1. **66% of box 1 is withheld from decode.** The 2,944 dense-window slots are a
   PREFILL structure; decode never reads them. Decode gets 1,510 slots out of
   4,454. Widening decode's share measurably halves box 2's server time
   (158.7 -> 83.9 ms), so the partition is costing real throughput. (KNOWN_BUGS #15)

2. **Box 1's decode LRU never evicts.** `budget = pg.lru_free_slots()` counts
   EMPTY slots, so box 1 admits a new expert only while virgin slots remain;
   once full, `budget == 0` forever and the residency is frozen at whatever
   arrived first. It is not an LRU in steady state. (KNOWN_BUGS #16)

3. **The roles are inverted.** Box 1 is the HUB: it has the dGPU, it runs
   attention and the shared expert, and computing a pick locally costs no
   network hop. Yet box 1 is configured as a VICTIM cache -- it only admits
   experts box 2 reported missing (`victim_cache()`, default ON) -- while box 2
   holds the large dynamic pool. The tiering is backwards.

## Target design

**One policy, two tiers, no static partition.**

- **Box 1 = HOT cache.** One pool over its whole capacity, no phase partition,
  true LRU (or LFU/CLOCK) with real eviction. Holds the hottest experts across
  all 40 layers regardless of phase. Prefill and decode share it; whichever
  phase touches an expert keeps it warm for the other.
- **Box 2 = VICTIM cache.** Takes what box 1 evicts and anything box 1 misses,
  paged from its own NVMe. It already behaves this way; the change is that it
  stops being the primary and stops needing a frequency-ranked seed.
- **No static placement anywhere.** `--experts` / the placement file become a
  warm-start seed at most, and ideally are dropped entirely: both pools
  converge from traffic.

Routing then follows residency, as it already does:
`box 1 holds it -> box 1 computes it; else -> box 2`, which is the
`max(t_box1, t_box2 + rtt)` balance with the local arm preferred.

## Why this should win

- Removes the 2,944-slot hole in decode's cache without shrinking prefill's
  working set -- prefill's union is a SUBSET of the hot set, not a disjoint one.
- Restores eviction, so a large pool stops being worse than a small one. The
  in-tree note that a 1,396-slot LRU was slower than a 25-slot one is a symptom
  of the freeze, not evidence against big caches.
- 28.3% of experts (4,346 / 81.7 GB) are currently resident NOWHERE and every
  touch is a disk read. Deduplicating the two pools (box 1 must not mirror box
  2's hot set -- that is what the victim gate exists to prevent) is what buys
  coverage.

## MEASURED: the working set is NOT the model (2026-09-16)

From the `expert_stats.json` sidecar -- real device-side routed-pick counters
(`harvest_sel_stats`), 14,418,780 decode picks over 60,102 decode tokens,
accumulated across many prompts and sessions:

| coverage of decode picks | top-N pairs | % of corpus | GB |
|---|---|---|---|
| 50% | 1,509 | 9.8% | 27.7 |
| 80% | 4,699 | 30.6% | 86.3 |
| 90% | 6,913 | 45.0% | 126.9 |
| 95% | 8,781 | 57.2% | 161.2 |
| 99% | 11,861 | 77.2% | 217.8 |

    top  4,454 (box 1 alone)         cover 78.51% -> miss 21.49%
    top  6,560 (box 2 alone)         cover 88.75% -> miss 11.25%
    top 11,014 (both, DEDUPLICATED)  cover 98.28% -> miss  1.72%

**Half of all decode picks come from 1,509 experts (27.7 GB).** The existing
combined capacity, deduplicated and holding the actual hot set, covers 98.3%.

**This retires the "some picks always come off disk" framing.** That claim is
true only at 1.72%, not the 7-12% miss rates measured today. Re-pricing:
1.72% x 240 picks/token = 4.1 misses x ~7.2 ms = **~30 ms/token of paging**,
against today's ~20 misses/token = ~145 ms. So the rework is worth
**~115 ms/token**, not the ~30 ms the capacity-wall writeup assumed -- that
writeup priced a global pool while holding the STATIC partition and the frozen
LRU fixed, and assumed working set == corpus.

With compute at ~26 ms/token that puts a token near 56-60 ms = **~17 tok/s,
within reach of the 20 tok/s goal with NO expert-format change.**

**Caveats.** (a) These counts accumulated across builds that included known
correctness bugs -- notably the live `V41_PAGER_WINDOWS` output-dependence
(KNOWN_BUGS #1) and historically the 77% submit-mask bug -- and a routing bug
distorts which experts appear hot. Re-measure on a clean build. (b) Perfect
dedup with perfect hot-set knowledge is the CEILING an LRU approximates, not
achieves. (c) A static frequency table does NOT transfer across domains
(measured: a prose-fit table covers only 35.2% of CODE picks) -- which is an
argument FOR a dynamic cache, not for a placement file.

## What this can and cannot buy — READ BEFORE BUILDING

The capacity wall is MEASURED and post-fix
(`docs/v41/DECODE_CAPACITY_WALL.md`, memory
`project_v41_decode_capacity_wall_2026-09-14`):

    40 x 384 x 19.25 MB = 288.8 GB of experts  vs  96 + 128 = 224 GB of RAM

The two boxes CANNOT hold the expert set at Q8_K. Some picks always come off
disk at ~9.5 ms and **no cache policy changes that**. The same writeup prices
the global-pool change at **~30 ms/token, "not a transformation"**, and puts
all-resident decode at 0.65 ms/layer x 40 = 26 ms/token = **25-33 tok/s**,
i.e. the 20 tok/s goal is reachable ONLY all-resident, which needs
12.6-13.3 MB/expert = 5.6-5.9 bits/wt (Q5_K fits, Q6_K does not).

**SUPERSEDED by the measurement above.** The wall's arithmetic assumed the
working set is the whole 15,360-pair corpus. It is not: the top 11,014 pairs
cover 98.28% of decode picks, so the existing hardware can hold the hot set and
the rework is worth ~115 ms/token -- plausibly the goal itself, without a format
change. Keep the wall's compute figure (0.65 ms/layer x 40 = 26 ms/token) and
its OOM constraint; discard its "no cache policy changes that" conclusion.

**Beware `project_v41_expert_placement_2026-09-13`**: it claims 14.8 tok/s with
ZERO misses and a "~7-8k pair working set that both boxes together hold". That
file is stamped INVALIDATED -- every number in it was taken while the submit
mask silently dropped 77% of routed experts, which would make any measured
working set far too SMALL. Do not size this design from it. The post-fix view
is that the steady working set is effectively the whole 15,360-pair corpus,
consistent with the measured cross-domain coverage collapse (a table fit on
prose covers only 35.2% of CODE picks).

## Constraints to respect

- Box 1 pool at 86 GB OOMs (`hipErrorOutOfMemory`); 78 GB is near the ceiling
  on a 93 GB box. Capacity gains must come from BETTER ALLOCATION, not more GB.
- Prefill at large B needs a wide per-layer union (up to 384 experts x 40
  layers); a naive single LRU must not thrash it. Sizing/admission has to
  account for a prefill chunk touching ~everything once.
- `V41_PAGER_WINDOWS` currently changes temperature-0 OUTPUT (KNOWN_BUGS #1).
  That bug lives in this same code and should be understood BEFORE or DURING
  the rework -- not inherited into the new policy.


---

# v2 — what actually survives

**Do NOT invert the tiers.** Keep box 1 exclusive/victim; the 140-vs-87 us
asymmetry is structural. Re-target at CAPACITY FOR THE VICTIM SET, not at making
box 1 the hot cache.

**Make it phase-aware, not flat.** Per-layer regions during prefill (preserving
the cyclic-scan hit rate the pinning buys), roaming during decode. Slot contents
survive the switch; only bookkeeping changes. This is the structure box 2 already
runs, and it captures the decode-capacity prize without breaking prefill.

**Keep the split deterministic.** `T2_CATCHALL=2` is a hard requirement of
`scripts/v41_determinism_gate.sh`, and mode 1 is history-dependent (documented
degenerate output). Any residency-driven routing needs an answer to the f32
association problem FIRST -- a fixed-order reduction, not a hope.

**Known consequence to price:** under mode 2 box 1 computes ZERO routed experts,
so `victim_cache()` and `lru_free_slots()` are only read on the mode-1 path.
Box 1's decode LRU is not merely frozen (#16) -- in the shipped config it is
never populated at all. Any plan to give box 1 decode capacity must first say
what box 1 is allowed to compute, deterministically.

## Explicit work items v1 budgeted zero effort for

1. **Group-id space = pool slots, not `N_EXPERT`.** `moe_group_builder.hip:118`
   drops any `g >= n_expert`, and `n_expert` is hardcoded 384 at four call sites
   (`forward_prefill.rs` hetsplit builder, plain builder, work-items split,
   work-items) with matching buffers (`group_count[N_EXPERT]`,
   `expert_members[N_EXPERT * b]`, `work_items_len`). Pool-wide slots would be
   SILENTLY DROPPED. Cost is bounded: `expert_members` 786 KB -> 9.1 MB. Note the
   `n_expert > 1024` rejection in `moe_group_builder.rs:110-113` (dead on V4.1).
2. **Per-layer `remap_dev` + dirty flags.** `ExpertPager` has ONE `remap_dev`
   valid only for the most-recently-ensured layer. A shared pool where layer L's
   page-in can evict layer L''s slot requires per-layer remaps -- box 2 already
   has exactly this (`ShardPool { remap_hosts, dirty }`).
3. **A pin/epoch set.** Slots claimed by an issued-but-incomplete dispatch must
   not be eviction candidates. Today safety rests on `ensure_layer_union`
   REFUSING to evict, on LRU recency, and on layer serialisation -- "real
   eviction" removes all three at once. Box 2's `want` guard is the minimum
   viable version.
4. **A stream-ordering contract for page-ins.** Today ordering is an accidental
   null-stream drain (`buffer.rs`), with the repack kernel on a separate stream.
5. **An O(1) LRU.** `touch()` is a linear scan of the deque. Dense-window slots
   never enter it today; under one pool every prefill hit (~15k/chunk) would cost
   an O(4454) scan.
6. **`ExpertPlan { view, remap }`** (KNOWN_BUGS #4) lands in the SAME change, not
   after -- collapsing two slot spaces into one is exactly when the pairing must
   become unrepresentable.

## Re-measure defect 1 properly

The 1510 -> 3686 slot table is non-monotonic (4.57 / 4.23 / 5.40) and its
supporting evidence was box 2's SERVER TIME, which `feedback_exposed_wait_is_not_work`
says never to quote. Re-run interleaved in ONE process, citing box 1
`decode_misses` and box 2 `pg.misses`, with a determinism sha check.
