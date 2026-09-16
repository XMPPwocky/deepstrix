# Expert allocation rework — plan

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

So: **this rework is a real but bounded win (~30 ms/token), not the path to
20 tok/s.** Its value is removing the three defects above -- 66% of box 1
withheld from decode, a decode cache that never evicts, and inverted tiering --
which are pure waste on top of the capacity wall. Do not oversell it, and do
not let it substitute for the format decision.

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
