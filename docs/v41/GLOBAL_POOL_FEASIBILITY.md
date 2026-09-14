# Box 2's per-layer regions -> one global pool: feasible, no kernel change
### 2026-09-14. The largest measured non-requant decode lever.

## The prize

Box 2 runs an INDEPENDENT LRU per layer, sized by ownership: 260 slots on encoder
layers 0-19, **68 on decoder layers 20-39** (17.7% of 384). Replaying the routing
trace through both structures at IDENTICAL total capacity (6,560 slots):

    per-layer 260/68 (today)   20.4 misses/token   <- measured 20.2, model validated
    one global pool            10.4 misses/token

**~10 misses/token x ~7.5 ms = ~75 ms/token.** Against today's 202 ms/token that
is 202 -> 127 ms = **~7.9 tok/s, +60%**, with no requantisation, no extra RAM and
no new hardware. `remote_rtt` is 110.6 ms/token (73% of instrumented decode time)
and is almost entirely these misses, so this attacks the dominant term directly.

## It needs no kernel change

`REMOTE_EXPERTS.md` and `DECODE_CAPACITY_WALL.md` both record this as blocked
because "the wire format hands the executor a contiguous `[base_slot,
base_slot+n)` range per layer, so a global pool changes the executor contract".
**That is only true of `layer_views`, host-side.** The kernel does:

    const int dense = remap[sel];
    const int e = (dense >= 0) ? sel : (-dense - 1);     // decode the slot
    mxfp4_pair_row(..., gate_w_base, ..., e, gate_bpe, ...);

i.e. it indexes whatever base pointer it is handed by the decoded slot, and the
sign convention (`< 0` = ours, `>= 0` = the other device) is independent of where
the slot lives. Pass the WHOLE pool buffer instead of a slice, and store ABSOLUTE
slots in the remap, and it works untouched. The `res_rank`/`dgpu_cap` counting
also only tests the sign, so it is unaffected.

## What actually changes (all in `ExpertShard`, box 2 only)

1. `layer_views` returns `r.gate.buffer` etc. whole, not `slice_view(base*bpe, ..)`.
2. `remap_host[e] = -(base_slot + local_slot) - 1` instead of `-(local_slot) - 1`.
3. `LayerPager` (per-layer `lru`, `slot_of`, `slot_key`) becomes one shard-wide
   pager keyed on `(layer, expert)` with `slot_key: Vec<Option<(u32, u32)>>`.

The existing eviction guard ("victim is never an id we need this same call")
generalises for free and gets *more* candidates, not fewer. No wire-format
change: the remap is device-side and uploaded per layer as it is today.

## The real blocker, which is NOT the executor contract

**Prefill.** Box 2 also serves prefill, which needs a whole layer's union
resident SIMULTANEOUSLY (~203 experts at B=1024). A global LRU lets one layer's
prefill sweep evict another's, which is exactly the thrash that the per-layer
regions prevent. The prefill window/stride rules (box 2 must own >= 384-128 = 256
per encoder layer, or prefill 500s) exist for this reason.

So a global pool is a DECODE structure that must not break prefill. Options, in
increasing order of effort:

  * **Phase-aware**: per-layer regions during prefill, global during decode. The
    slot contents survive the switch; only the bookkeeping changes. Cheapest, and
    decode and prefill never run concurrently.
  * **Reserved floors**:每 layer keeps a guaranteed minimum (say 128 on encoder
    layers) and the remainder is globally shared. Bounds prefill's worst case
    while giving decode most of the win.
  * **Two pools**: split the 6,560 slots into a per-layer prefill region and a
    global decode region. Simplest to reason about, worst capacity utilisation.

## Falsifier before building

The 10.4 figure is a simulation. Validate it cheaply first: extend
`scratchpad/geom.py` to replay the SAME trace through a global LRU at exactly
6,560 slots while ALSO scoring the prefill union constraint (does any layer's
B=1024 union ever fail to be co-resident?). If the global arm cannot hold a
prefill union, only the phase-aware variant is viable and the decode win must be
re-priced with the switch cost.

Second falsifier: the simulation assumes box 2's misses cost 7.5 ms each
regardless of which layer they are on. Confirm with box 2's own per-layer miss
counters that decoder-layer misses dominate, as the 68-slot geometry predicts.
